"""Tests for the Wave 4 orchestrator CLI (issue #14).

Two layers, matching the issue's acceptance criteria:

* :func:`music_video_maker.cli.run_pipeline` is exercised end-to-end against
  a fully mocked ComfyUI (``FakeComfyUISession``), a faked stable-ts model, a
  faked ffmpeg/ffprobe runner, and a recording backoff sleeper -- the "dry-run
  mode (mocked ComfyUI) exercises the full wiring" acceptance criterion. No
  real GPU, network, server, or sleep anywhere in this module.
* :func:`music_video_maker.cli.main` is exercised at the argv/exit-code layer
  with ``run_pipeline`` monkeypatched out, so CLI wiring (config-error path,
  ``--resume`` pass-through, exit codes) is tested independently of the full
  pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from music_video_maker import cli, contracts
from music_video_maker.assembly import DEFAULT_OUTPUT_FILENAME
from music_video_maker.contracts import ChunkFingerprint
from music_video_maker.workflow_graph import (
    CLASS_TYPE_H3_IMAGE_TO_VIDEO,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    find_nodes_by_class_type,
    find_one_node,
)
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import make_workflow_baseline, make_workflow_i2v, write_silent_wav
from tests.harness.ws import ScriptedWebSocket, build_hang_sequence, build_success_sequence

# --------------------------------------------------------------------------- #
# Local scaffolding (mirrors tests/test_wave3_integration.py's rig)
# --------------------------------------------------------------------------- #


class SequencedWSFactory:
    """Hand out a differently-scripted WebSocket per connection -- a whole run
    submits once per attempt, and the mock assigns each submission its own
    ``prompt_id``, so the scripts must differ per connection."""

    def __init__(self, sequences: list[list[Any]]) -> None:
        self._sequences = deque(sequences)

    def __call__(self, url: str, **kwargs: Any) -> ScriptedWebSocket:
        if not self._sequences:
            raise AssertionError("SequencedWSFactory: more connections than scripted sequences")
        ws = ScriptedWebSocket(self._sequences.popleft())
        ws.connect(url, **kwargs)
        return ws


class FakeFfmpegRunner:
    """Stands in for both continuity's ffprobe/ffmpeg frame extraction and
    Stage 5's concat/mux ffmpeg calls -- no real binary anywhere."""

    def __init__(self, frame_count: int = 124) -> None:
        self.frame_count = frame_count
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(args, 0, stdout=f"{self.frame_count}\n".encode())
        dest = Path(args[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "-vf" in args:
            dest.write_bytes(make_fake_png_bytes(1344, 768))
        else:
            dest.write_bytes(b"fake-mp4-bytes")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class FakeClock:
    """Deterministic monotonically-increasing clock -- never real time."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        self._t += 1.0
        return self._t


def _abundant_disk(_path: str):
    class _Usage:
        free = 500 * 1024**3

    return _Usage()


class FakeAlignModel:
    """Stand-in for a stable-ts model: exposes ``.align()`` returning a
    pre-built raw result, never touching torch or a real model file."""

    def __init__(self, raw_result: SimpleNamespace) -> None:
        self._raw_result = raw_result

    def align(self, audio: str, text: str, **kwargs: Any) -> SimpleNamespace:
        return self._raw_result


def _raw_segment(text: str, start: float, end: float) -> SimpleNamespace:
    words = text.split()
    span = (end - start) / len(words)
    word_objs = [
        SimpleNamespace(word=w, start=start + i * span, end=start + (i + 1) * span)
        for i, w in enumerate(words)
    ]
    return SimpleNamespace(text=text, start=start, end=end, words=word_objs)


def _raw_result(specs: list[tuple[str, float, float]]) -> SimpleNamespace:
    return SimpleNamespace(segments=[_raw_segment(t, s, e) for t, s, e in specs], language="en")


DEFAULT_SEGMENT_SPECS = [
    ("walking through the empty halls tonight", 0.0, 6.5),
    ("nobody is watching nobody cares", 8.0, 12.5),
    ("the lights flicker but i do not mind", 14.0, 19.0),
]
"""Three segments, each within the default 4-15s chunk window, well clear of
each other -- no pad/merge/split enforcement kicks in, keeping this module
focused on CLI wiring rather than re-testing Stage 2's slicing edge cases."""


def _ALLOW_ANY_SEED(_frame_path) -> bool:
    """The rig's fake seed frames are solid-colour PNGs with no face in them,
    so the real #47 gate would (correctly) refuse to chain from every one of
    them. These tests are about the chaining wiring, not the detector, so they
    substitute a permissive predicate -- the gate's own behaviour is pinned by
    the tests below and by tests/test_faces.py."""
    return True


class Rig:
    """Everything wired the way ``cli.main`` eventually will: a real config,
    real stager, real mutator, real execution client, real resilient runner,
    real continuity provider, real assembly -- only ComfyUI, ffmpeg, sleep,
    the stable-ts model, and the clock are faked."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        segment_specs: list[tuple[str, float, float]] | None = None,
        i2v_continuity: bool = False,
        instrumental_coverage: bool = False,
    ) -> None:
        self.tmp_path = tmp_path
        self.session = FakeComfyUISession()
        self.ffmpeg = FakeFfmpegRunner()
        self.sleeper = RecordingSleeper()
        self.clock = FakeClock()
        self.align_model = FakeAlignModel(_raw_result(segment_specs or DEFAULT_SEGMENT_SPECS))

        master_audio = write_silent_wav(tmp_path / "audio" / "master.wav", seconds=25.0)
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text(
            "Walking through the empty halls tonight\n"
            "Nobody is watching nobody cares\n"
            "The lights flicker but I do not mind\n"
        )
        cast_image = tmp_path / "cast" / "dianne_ref.png"
        cast_image.parent.mkdir(parents=True, exist_ok=True)
        cast_image.write_bytes(make_fake_png_bytes(1344, 768))

        cast = {
            "Dianne": contracts.CastMember(
                name="Dianne", role="Lead Vocalist, smiling constantly, oblivious", image=cast_image
            )
        }

        chunks_dir = tmp_path / "output" / "chunks"
        final_video_dir = tmp_path / "output" / "final"

        base_template_path = tmp_path / "workflow_api.json"
        base_template_path.write_text(json.dumps(make_workflow_baseline()))

        i2v_template_path = None
        if i2v_continuity:
            i2v_template_path = tmp_path / "workflow_i2v_api.json"
            i2v_template_path.write_text(json.dumps(make_workflow_i2v()))

        from music_video_maker.config import RunConfig

        self.config = RunConfig(
            master_audio=master_audio,
            lyrics_file=lyrics_file,
            global_style="Refestramus progressive rock music video, 35mm film",
            narrative_concept="Wandering through a surgery, kicking a life support plug out",
            cast=cast,
            default_lead_vocalist="Dianne",
            comfyui_url=self.session.base_url,
            workflow_template=base_template_path,
            chunks_dir=chunks_dir,
            final_video_dir=final_video_dir,
            hardware=contracts.HardwareProfile(name="RTX 4090 24GB (doris)", vram_gb=24.0),
            max_render_attempts=3,
            retry_backoff_seconds=1.0,
            min_free_disk_gb=1.0,
            run_state_file=chunks_dir / "run_state.json",
            i2v_continuity=i2v_continuity,
            i2v_workflow_template=i2v_template_path,
            # Default off in the rig so the chunk count stays pinned to the
            # three lyric segments and each test below can seed exactly one
            # WebSocket sequence per chunk. The shipping default (on) is
            # exercised end to end by its own test at the bottom of the file.
            instrumental_coverage=instrumental_coverage,
        )

    def run(
        self,
        ws_sequences: list[list[Any]],
        *,
        resume: bool = False,
        only_chunks: tuple[int, ...] | None = None,
        seed_face_gate: Any = _ALLOW_ANY_SEED,
    ) -> cli.RunReport:
        self.run_report = cli.run_pipeline(
            self.config,
            resume=resume,
            align_model=self.align_model,
            comfyui_session=self.session,
            ws_factory=SequencedWSFactory(ws_sequences),
            ffmpeg_runner=self.ffmpeg,
            sleeper=self.sleeper,
            disk_usage=_abundant_disk,
            clock=self.clock,
            only_chunks=only_chunks,
            seed_face_gate=seed_face_gate,
        )
        return self.run_report

    def seed_success(self, prompt_number: int, chunk_id: int) -> str:
        prompt_id = f"prompt-{prompt_number:04d}"
        self.session.seed_history_success(
            prompt_id, video_filename=f"mvm_chunk_{chunk_id:04d}_00001.mp4"
        )
        return prompt_id

    @property
    def submitted(self) -> list[dict]:
        return [s for s in self.session.submitted_prompts if not s.get("rejected")]


def _is_i2v(workflow: contracts.Workflow) -> bool:
    return bool(find_nodes_by_class_type(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO))


def _is_base(workflow: contracts.Workflow) -> bool:
    return bool(find_nodes_by_class_type(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO))


def _submitted_prompt(workflow: contracts.Workflow) -> str:
    """The prompt string as it reached ComfyUI -- read off the H3 node this
    graph actually carries, rather than from a node id, since which of the two
    templates a chunk took is the thing these tests are checking."""
    class_type = (
        CLASS_TYPE_H3_IMAGE_TO_VIDEO if _is_i2v(workflow) else CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    )
    _node_id, node = find_nodes_by_class_type(workflow, class_type)[0]
    return node["inputs"]["prompt"]


# --------------------------------------------------------------------------- #
# run_pipeline: the happy path
# --------------------------------------------------------------------------- #


def test_happy_path_renders_every_chunk_and_assembles(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    report = rig.run(sequences)

    assert report.dead_lettered == ()
    assert report.rendered == 3
    assert report.cached == 0
    assert report.total_chunks == 3
    assert report.output_video == rig.config.final_video_dir / DEFAULT_OUTPUT_FILENAME
    assert report.output_video.is_file()
    assert report.wall_seconds > 0


def test_happy_path_stages_every_chunk_exactly_once(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
    rig.run(sequences)

    # One cast-photo upload (cached across all 3 chunks) + 3 distinct audio uploads.
    image_uploads = [u for u in rig.session.uploads if u.original_filename == "dianne_ref.png"]
    audio_uploads = [u for u in rig.session.uploads if u.original_filename.startswith("chunk_")]
    assert len(image_uploads) == 1
    assert len(audio_uploads) == 3


def test_happy_path_every_chunk_renders_through_the_base_template_without_continuity(
    tmp_path: Path,
):
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
    rig.run(sequences)

    workflows = [s["workflow"] for s in rig.submitted]
    assert len(workflows) == 3
    assert all(_is_base(w) and not _is_i2v(w) for w in workflows)
    # No continuity frame extraction (ffprobe / "-vf" select) without continuity enabled --
    # Stage 5's concat/mux calls are the only ffmpeg invocations.
    assert not any(call[0] == "ffprobe" or "-vf" in call for call in rig.ffmpeg.calls)


# --------------------------------------------------------------------------- #
# run_pipeline: I2V continuity wired through
# --------------------------------------------------------------------------- #


def test_i2v_continuity_bridges_chunks_after_the_first(tmp_path: Path):
    """Issue #12's every-chunk bridging, reachable since #28 via
    i2v_chain_scope='all' -- the default scope is 'instrumental' because the
    bridged path cannot lip-sync (see the chain-scope tests below)."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    report = rig.run(sequences)

    assert report.dead_lettered == ()
    workflows = [s["workflow"] for s in rig.submitted]
    assert _is_base(workflows[0]) and not _is_i2v(workflows[0])
    assert all(_is_i2v(w) and not _is_base(w) for w in workflows[1:])


# --------------------------------------------------------------------------- #
# run_pipeline: dead-letters skip assembly, not the run
# --------------------------------------------------------------------------- #


def test_dead_lettered_chunk_skips_assembly_but_the_run_still_reports(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [
        build_success_sequence(rig.seed_success(1, 0)),
        *[build_hang_sequence(f"prompt-{n:04d}") for n in (2, 3, 4)],  # chunk 1: 3 attempts, hangs
        build_success_sequence(rig.seed_success(5, 2)),
    ]

    report = rig.run(sequences)

    assert report.dead_lettered == (1,)
    assert report.rendered == 2
    assert report.output_video is None
    assert not (rig.config.final_video_dir / DEFAULT_OUTPUT_FILENAME).exists()


# --------------------------------------------------------------------------- #
# run_pipeline: resume
# --------------------------------------------------------------------------- #


def test_resume_reuses_a_cached_chunk(tmp_path: Path):
    rig = Rig(tmp_path, segment_specs=DEFAULT_SEGMENT_SPECS[:2])

    # First pass only gets through chunk 0 (chunk 1 hangs out its retries).
    sequences = [
        build_success_sequence(rig.seed_success(1, 0)),
        *[build_hang_sequence(f"prompt-{n:04d}") for n in (2, 3, 4)],
    ]
    first = rig.run(sequences)
    assert first.dead_lettered == (1,)
    assert first.run_state.results[0].status is contracts.ChunkStatus.RENDERED

    # Second pass resumes: chunk 0 comes back CACHED, chunk 1 (previously
    # dead-lettered) is retried and succeeds this time.
    second = rig.run([build_success_sequence(rig.seed_success(5, 1))], resume=True)

    assert second.run_state.results[0].status is contracts.ChunkStatus.CACHED
    assert second.run_state.results[1].succeeded
    assert second.dead_lettered == ()
    assert second.output_video is not None

    workflows = [s["workflow"] for s in rig.submitted]
    assert len(workflows) == 5  # 1 + 3 hung attempts + 1 -- chunk 0 never resubmitted


def test_resume_after_a_lyrics_edit_re_renders_the_moved_chunk(tmp_path: Path):
    """Issue #34's acceptance criterion. Correcting the lyrics file moves the
    chunk timeline; the mp4s from the old timeline are all still on disk and
    still valid, and reusing them would assemble a silently desynced video."""
    rig = Rig(tmp_path, segment_specs=DEFAULT_SEGMENT_SPECS[:2])
    first = rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2)])
    assert first.rendered == 2

    # The second line is corrected -- new words, and it now lands 1.5s later.
    rig.config.lyrics_file.write_text(
        "Walking through the empty halls tonight\nNobody is watching nobody cares at all\n"
    )
    rig.align_model = FakeAlignModel(
        _raw_result(
            [
                DEFAULT_SEGMENT_SPECS[0],
                ("nobody is watching nobody cares at all", 9.5, 15.2),
            ]
        )
    )

    second = rig.run([build_success_sequence(rig.seed_success(3, 1))], resume=True)

    assert second.run_state.results[0].status is contracts.ChunkStatus.CACHED
    assert second.run_state.results[1].status is contracts.ChunkStatus.RENDERED
    assert len(rig.submitted) == 3  # the untouched chunk 0 was not resubmitted


def test_resume_with_an_unchanged_config_still_reuses_everything(tmp_path: Path):
    """The fast path must stay fast -- identity checking must not turn every
    resume into a full re-render."""
    rig = Rig(tmp_path, segment_specs=DEFAULT_SEGMENT_SPECS[:2])
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2)])

    second = rig.run([], resume=True)  # no WS sequences: any submission fails loudly

    assert second.cached == 2
    assert second.rendered == 0
    assert len(rig.submitted) == 2


def test_only_chunks_renders_just_those_chunks_and_skips_assembly(tmp_path: Path):
    """A validation slice renders a named handful and must NOT assemble.

    Concatenating a subset would produce a file that looks like the song and
    is not -- the exact silently-desynced artifact fingerprints exist to
    prevent. The slice's job is chunks to watch, not a video to ship."""
    rig = Rig(tmp_path)

    report = rig.run([build_success_sequence(rig.seed_success(1, 0))], only_chunks=(1,))

    assert len(rig.submitted) == 1
    assert report.rendered == 1
    assert report.output_video is None, "a partial render must not claim to be the song"
    assert set(report.run_state.results) == {1}


def test_only_chunks_still_slices_the_whole_song_so_spans_are_unmoved(tmp_path: Path):
    """The slice must render chunk N *exactly* as a full run would.

    Stages 1-2 run over the whole track and only the render list is narrowed.
    Slicing just the selected span would re-time it against a shorter timeline
    and validate a chunk the real run will never produce."""
    rig_full = Rig(tmp_path / "full")
    rig_full.run([build_success_sequence(rig_full.seed_success(n, n - 1)) for n in (1, 2, 3)])

    rig_slice = Rig(tmp_path / "slice")
    rig_slice.run([build_success_sequence(rig_slice.seed_success(1, 0))], only_chunks=(1,))

    full_fp = rig_full.run_report.run_state.results[1].fingerprint
    slice_fp = rig_slice.run_report.run_state.results[1].fingerprint
    assert (full_fp.start, full_fp.end) == (slice_fp.start, slice_fp.end)
    assert full_fp.prompt_hash == slice_fp.prompt_hash
    assert full_fp.template_hash == slice_fp.template_hash


def test_only_chunks_rejects_an_id_that_is_not_in_the_song(tmp_path: Path):
    """A typo'd chunk id must fail loudly rather than render nothing and
    report success -- an empty slice looks identical to a finished one."""
    rig = Rig(tmp_path)

    with pytest.raises(cli.PipelineError, match="99"):
        rig.run([], only_chunks=(99,))


def test_resume_after_a_template_edit_re_renders_rather_than_mixing_graphs(tmp_path: Path):
    """Issue #45: editing a workflow template changes the pixels, and nothing
    in the state file used to notice -- so a resumed run assembled a video from
    two different configurations. #44 made that concrete: removing `last_frame`
    from the I2V template altered every chained chunk, and forcing the
    re-render meant deleting results from run_state.json by hand.

    The edit here is the same *kind* as #44's: a structural change to the
    authored graph, with every per-chunk value untouched."""
    rig = Rig(tmp_path, segment_specs=DEFAULT_SEGMENT_SPECS[:2])
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2)])

    template = json.loads(rig.config.workflow_template.read_text())
    sampler_id, sampler = find_one_node(contracts.Workflow(template), "BasicScheduler")
    template[sampler_id]["inputs"]["steps"] = sampler["inputs"]["steps"] + 5
    rig.config.workflow_template.write_text(json.dumps(template))

    # ...and the escape hatch must not forgive it. A template edit is not a
    # typo fix in a shot line: conditioning tier, deliberately.
    rig.config = replace(rig.config, resume_ignore_prompt_changes=True)

    second = rig.run(
        [build_success_sequence(rig.seed_success(n, n - 1)) for n in (3, 4)], resume=True
    )

    assert second.cached == 0, "an edited template must not be resumed over"
    assert second.rendered == 2


def test_resume_after_a_cosmetic_template_reformat_still_reuses_everything(tmp_path: Path):
    """The other half of #45, and the reason the hash is of the parsed
    structure rather than the file bytes: reindenting or reordering a template's
    JSON changes nothing about the render, and must not cost a song's worth of
    GPU time."""
    rig = Rig(tmp_path, segment_specs=DEFAULT_SEGMENT_SPECS[:2])
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2)])

    template = json.loads(rig.config.workflow_template.read_text())
    reformatted = {k: template[k] for k in sorted(template, key=int, reverse=True)}
    rig.config.workflow_template.write_text(json.dumps(reformatted, indent=4, sort_keys=True))

    second = rig.run([], resume=True)  # no WS sequences: any submission fails loudly

    assert second.cached == 2
    assert second.rendered == 0


# --------------------------------------------------------------------------- #
# run_pipeline: custody seam is actually wired through
# --------------------------------------------------------------------------- #


def test_custody_preflight_and_teardown_both_run(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
    rig.run(sequences)

    assert any(r.method == "GET" and r.url.endswith("/system_stats") for r in rig.session.requests)
    assert rig.session.free_calls == [{"unload_models": True, "free_memory": True}]


def test_free_vram_is_re_read_before_every_chunk_not_just_at_run_start(tmp_path: Path):
    """Issue #23's wiring, asserted from the outside: the custody pre-flight
    reads /system_stats once, and the runner must read it again before each
    chunk it submits. A probe that is built but never handed to the runner
    would leave exactly one reading for the whole run -- which is the hole
    the 2026-08-07 wedge went through."""
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    report = rig.run(sequences)

    stats_reads = [
        r for r in rig.session.requests if r.method == "GET" and r.url.endswith("/system_stats")
    ]
    assert report.rendered == 3
    assert len(stats_reads) == 4  # one custody pre-flight + one per rendered chunk


def test_custody_teardown_runs_even_when_a_chunk_dead_letters(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [
        build_success_sequence(rig.seed_success(1, 0)),
        *[build_hang_sequence(f"prompt-{n:04d}") for n in (2, 3, 4)],
        build_success_sequence(rig.seed_success(5, 2)),
    ]
    rig.run(sequences)
    # The dead-lettering chunk's own recovery sequence also calls free() (issue #10) --
    # custody's teardown call is guaranteed to be the last one, after the run completes.
    assert rig.session.free_calls[-1] == {"unload_models": True, "free_memory": True}
    assert len(rig.session.free_calls) >= 1


# --------------------------------------------------------------------------- #
# run_pipeline: no chunks
# --------------------------------------------------------------------------- #


def test_no_aligned_segments_raises_pipeline_error(tmp_path: Path, caplog):
    rig = Rig(tmp_path)
    rig.align_model = FakeAlignModel(_raw_result([]))

    with pytest.raises(cli.PipelineError, match="nothing to render"):
        rig.run([])

    # Custody teardown still ran despite the failure.
    assert rig.session.free_calls == [{"unload_models": True, "free_memory": True}]


# --------------------------------------------------------------------------- #
# main(): argv / exit-code layer, run_pipeline monkeypatched out
# --------------------------------------------------------------------------- #


def _make_config_file(tmp_path: Path) -> Path:
    from tests.test_config import DEFAULT_CAST_TOML, DEFAULT_HARDWARE_TOML, _create_default_assets

    _create_default_assets(tmp_path)
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        f"""
master_audio = "{tmp_path / "audio" / "master.wav"}"
lyrics_file = "{tmp_path / "lyrics.txt"}"
global_style = "Refestramus progressive rock music video, 35mm film"
narrative_concept = "Wandering through a surgery, kicking a life support plug"
default_lead_vocalist = "Dianne"
comfyui_url = "http://doris:8188"
workflow_template = "{tmp_path / "workflow_api.json"}"
chunks_dir = "{tmp_path / "output" / "chunks"}"
final_video_dir = "{tmp_path / "output" / "final"}"
{DEFAULT_CAST_TOML.format(cast_dir=tmp_path / "cast")}
{DEFAULT_HARDWARE_TOML}
"""
    )
    return config_path


def _fake_report(*, dead_lettered: tuple[int, ...] = ()) -> cli.RunReport:
    results = {0: contracts.ChunkResult(chunk_id=0, status=contracts.ChunkStatus.RENDERED)}
    for cid in dead_lettered:
        results[cid] = contracts.ChunkResult(
            chunk_id=cid, status=contracts.ChunkStatus.DEAD_LETTERED
        )
    return cli.RunReport(
        run_state=contracts.RunState(run_id="test-run", results=results),
        total_chunks=len(results),
        wall_seconds=1.0,
        output_video=None if dead_lettered else Path("/tmp/final_video.mp4"),
    )


def test_main_returns_zero_on_full_success(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_run_pipeline(config, *, resume, **_kwargs):
        captured["resume"] = resume
        return _fake_report()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["resume"] is False


def test_main_passes_resume_flag_through(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_run_pipeline(config, *, resume, **_kwargs):
        captured["resume"] = resume
        return _fake_report()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    cli.main(["--config", str(config_path), "--resume"])

    assert captured["resume"] is True


def test_main_ignore_prompt_changes_flag_overrides_the_config(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_run_pipeline(config, *, resume, **_kwargs):
        captured["config"] = config
        return _fake_report()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["--config", str(config_path), "--resume"])
    assert captured["config"].resume_ignore_prompt_changes is False

    cli.main(["--config", str(config_path), "--resume", "--ignore-prompt-changes"])
    assert captured["config"].resume_ignore_prompt_changes is True


def test_main_strict_alignment_flag_overrides_the_config(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_run_pipeline(config, *, resume, **_kwargs):
        captured["config"] = config
        return _fake_report()

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    cli.main(["--config", str(config_path)])
    assert captured["config"].strict_alignment is False

    cli.main(["--config", str(config_path), "--strict-alignment"])
    assert captured["config"].strict_alignment is True


def test_run_pipeline_logs_an_alignment_quality_summary(tmp_path: Path, caplog):
    """Issue #35's first acceptance criterion, asserted where it has to hold:
    every run's log carries a quality summary, not just runs that go wrong."""
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    with caplog.at_level(logging.INFO):
        rig.run(sequences)

    assert any(
        "alignment quality" in r.getMessage().lower()
        for r in caplog.records
        if r.name.startswith("music_video_maker")
    )


def test_main_returns_partial_failure_code_when_chunks_dead_lettered(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    monkeypatch.setattr(
        cli, "run_pipeline", lambda config, *, resume, **_kw: _fake_report(dead_lettered=(3,))
    )

    exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == cli.EXIT_PARTIAL_FAILURE


def test_main_returns_error_code_on_missing_config_file(tmp_path: Path, caplog):
    missing = tmp_path / "does_not_exist.toml"
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(["--config", str(missing)])

    assert exit_code == cli.EXIT_ERROR
    assert any("Failed to load run config" in r.message for r in caplog.records)


def test_main_returns_error_code_when_pipeline_raises(tmp_path: Path, monkeypatch, caplog):
    config_path = _make_config_file(tmp_path)

    def fake_run_pipeline(config, *, resume, **_kwargs):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(["--config", str(config_path)])

    assert exit_code == cli.EXIT_ERROR
    assert any("Pipeline run failed" in r.message for r in caplog.records)


def test_main_never_calls_run_pipeline_when_config_load_fails(tmp_path: Path, monkeypatch):
    missing = tmp_path / "does_not_exist.toml"
    calls = []
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: calls.append(1))

    cli.main(["--config", str(missing)])

    assert calls == []


def test_main_requires_config_argument():
    with pytest.raises(SystemExit):
        cli.main([])


# --------------------------------------------------------------------------- #
# The GPU custody seam wired into the orchestrator (issue #19).
#
# custody.py's own tests cover the manager in isolation. These cover the seam
# that module cannot test: that run_pipeline actually routes through
# build_custody_manager, so the pre-flight brackets the whole render and the
# card is released even when the run explodes.
# --------------------------------------------------------------------------- #






def test_run_pipeline_preflight_precedes_every_comfyui_render_call(tmp_path: Path):
    """Ordering is the whole point of custody: a free-VRAM assertion made
    after the first chunk was submitted would be asserting it about a card
    this run had already started loading H3 onto."""
    rig = Rig(tmp_path)
    order: list[str] = []

    original_get = rig.session.get
    original_post = rig.session.post

    def recording_get(url, *args, **kwargs):
        if url.endswith("/system_stats"):
            order.append("comfyui:/system_stats")
        return original_get(url, *args, **kwargs)

    def recording_post(url, *args, **kwargs):
        if url.endswith("/prompt"):
            order.append("comfyui:/prompt")
        return original_post(url, *args, **kwargs)

    rig.session.get = recording_get  # type: ignore[method-assign]
    rig.session.post = recording_post  # type: ignore[method-assign]

    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
    rig.run(sequences)

    assert order[0] == "comfyui:/system_stats"
    assert "comfyui:/prompt" in order
    assert order.index("comfyui:/system_stats") < order.index("comfyui:/prompt")


def test_run_pipeline_frees_comfyui_vram_even_when_the_pipeline_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rig = Rig(tmp_path)

    def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("stage 1 exploded")

    monkeypatch.setattr(cli, "parse_lyrics", boom)

    with pytest.raises(RuntimeError):
        rig.run([])

    assert any(r.method == "POST" and r.url.endswith("/free") for r in rig.session.requests)






# --------------------------------------------------------------------------- #
# Wave 5: issue #20's quantized length reaching the server.
#
# The seam no single lane could test. #20's slicer puts a frame count on every
# AudioChunk; #8's mutator injects a frame count it is handed. Neither one can
# prove the number that arrives at ComfyUI is the number the audio was sliced
# to -- which is the entire point of #20, since any gap between them is drift
# that Stage 5 will never correct.
# --------------------------------------------------------------------------- #


def _submitted_length(submission: dict) -> int:
    workflow = submission["workflow"]
    for node in workflow.values():
        if node.get("class_type") in (
            CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
            CLASS_TYPE_H3_IMAGE_TO_VIDEO,
        ):
            return node["inputs"]["length"]
    raise AssertionError("no H3 node in the submitted workflow")


def _submitted_chunk_id(submission: dict) -> int:
    """Recover which chunk a submission is for from the SaveVideo prefix the
    mutator stamps (`mvm_chunk_0007`) -- never from list position."""
    for node in submission["workflow"].values():
        if node.get("class_type") == "SaveVideo":
            return int(str(node["inputs"]["filename_prefix"]).rsplit("_", 1)[-1])
    raise AssertionError("no SaveVideo node in the submitted workflow")


def _wav_seconds(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / fh.getframerate()


def test_each_chunk_is_rendered_at_the_length_its_audio_was_sliced_to(tmp_path: Path):
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    assert len(rig.submitted) == 3
    for submission in rig.submitted:
        chunk_id = _submitted_chunk_id(submission)
        length = _submitted_length(submission)

        # Compared against the audio bytes actually written to disk, not a
        # re-derivation of what slicing should have produced -- the claim
        # under test is that the rendered video and its own stem are the same
        # duration, and the stem is the file.
        stem = rig.config.chunks_dir / f"chunk_{chunk_id:03d}.wav"
        assert stem.is_file()
        assert contracts.H3_FRAME_GRID.frames_to_seconds(length) == pytest.approx(
            _wav_seconds(stem), abs=0.01
        )
        assert contracts.H3_FRAME_GRID.is_valid(length)
        assert 124 <= length <= 362


def test_no_chunk_is_rendered_at_the_templates_placeholder_length(tmp_path: Path):
    """The regression that would make all of #20 inert: with no injection,
    every chunk renders at the authored template's placeholder (124 frames /
    5.167s) no matter how long its audio actually is."""
    rig = Rig(
        tmp_path,
        segment_specs=[
            ("a line that runs on quite a bit longer than five seconds", 0.0, 11.0),
            ("another long line to keep it going here", 12.0, 22.0),
        ],
    )
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2)]

    rig.run(sequences)

    lengths = [_submitted_length(sub) for sub in rig.submitted]
    assert lengths and all(length != 124 for length in lengths)


# --------------------------------------------------------------------------- #
# Instrumental coverage: the shipping default, end to end
# --------------------------------------------------------------------------- #


def test_instrumental_coverage_default_renders_a_contiguous_timeline(tmp_path: Path):
    """With coverage on -- the shipping default -- the rendered chunk set
    tiles the whole track instead of only its voiced spans.

    This is the end-to-end guard on the desync that gap-skipping causes:
    Stage 5 lays chunks end to end with `-c:v copy`, so a chunk's offset in
    the final video is the sum of the durations before it. Asserting that
    equals the chunk's own slice offset into the master is asserting the
    final mux is in sync.
    """
    rig = Rig(
        tmp_path,
        segment_specs=[
            ("walking through the empty halls tonight", 4.0, 10.0),
            # 6s instrumental break here -- skipped entirely without coverage.
            ("nobody is watching nobody cares", 16.0, 22.0),
        ],
        instrumental_coverage=True,
    )
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3, 4)]

    report = rig.run(sequences)

    assert report.dead_lettered == ()
    assert report.output_video.is_file()

    # More chunks than there are lyric segments: the intro and the break got
    # their own renders.
    assert report.total_chunks > 2

    lengths = [_submitted_length(sub) for sub in rig.submitted]
    assert all(124 <= length <= 362 for length in lengths)


# --------------------------------------------------------------------------- #
# Render resolution injection
# --------------------------------------------------------------------------- #


def _submitted_dimensions(submission: dict) -> tuple[int, int]:
    _, node = find_one_node(submission["workflow"], CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    return node["inputs"]["width"], node["inputs"]["height"]


def test_render_dimensions_from_config_are_injected_into_every_chunk(tmp_path: Path):
    """Resolution is the single biggest lever on run time, so a config that
    asks for 864x480 must actually render at 864x480 -- a knob nothing reads
    would silently render the whole song at the template's own size."""
    rig = Rig(tmp_path)
    rig.config = replace(rig.config, render_width=864, render_height=480)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    assert rig.submitted
    for submitted in rig.submitted:
        assert _submitted_dimensions(submitted) == (864, 480)


def test_without_render_dimensions_the_template_size_is_left_alone(tmp_path: Path):
    rig = Rig(tmp_path)
    assert rig.config.render_width is None
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    _, template_node = find_one_node(
        make_workflow_baseline(), CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    )
    expected = (template_node["inputs"]["width"], template_node["inputs"]["height"])
    for submitted in rig.submitted:
        assert _submitted_dimensions(submitted) == expected


# --------------------------------------------------------------------------- #
# Issue #25: vocal-stem conditioning wired through run_pipeline
# --------------------------------------------------------------------------- #


def test_run_pipeline_conditions_on_the_vocal_stem_when_configured(tmp_path: Path):
    """With ``vocal_stem`` set, the audio staged to ComfyUI is the stem slice
    (``chunk_NNN_vocal.wav``), never the mix slice -- while Stage 5 still gets
    the pristine master (the stem is conditioning only)."""
    rig = Rig(tmp_path)
    stem = write_silent_wav(tmp_path / "audio" / "vocals.wav", seconds=25.0)
    rig.config = replace(rig.config, vocal_stem=stem)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    report = rig.run(sequences)

    assert report.dead_lettered == ()
    audio_uploads = [u for u in rig.session.uploads if u.original_filename.startswith("chunk_")]
    assert audio_uploads, "no chunk audio reached ComfyUI at all"
    assert all(u.original_filename.endswith("_vocal.wav") for u in audio_uploads), (
        f"mix slices were staged instead of stem slices: "
        f"{[u.original_filename for u in audio_uploads]}"
    )


def test_run_pipeline_without_a_stem_stages_the_mix_unchanged(tmp_path: Path):
    """The default path is byte-for-byte today's: mix slices staged."""
    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    audio_uploads = [u for u in rig.session.uploads if u.original_filename.startswith("chunk_")]
    assert audio_uploads
    assert not any(u.original_filename.endswith("_vocal.wav") for u in audio_uploads)


# --------------------------------------------------------------------------- #
# Issue #38: one config seed, two consumers -- both must read the same value
# --------------------------------------------------------------------------- #


def test_run_pipeline_threads_the_config_seed_to_both_the_graph_and_the_fingerprint(
    tmp_path: Path,
):
    """The seed reaches ComfyUI via the workflow mutation (continuity ->
    workflow_graph) and reaches run_state.json via ChunkFingerprint.of (cli).
    If either path stops reading config.noise_seed, the run records a seed the
    render did not use -- a lie in the very record issue #38 added. This test
    pins both ends of both paths in one run."""
    rig = Rig(tmp_path)
    rig.config = replace(rig.config, noise_seed=42)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    # Path 1: every submitted graph carries the config seed.
    assert rig.submitted, "no workflows reached ComfyUI"
    for submission in rig.submitted:
        workflow = submission["workflow"]
        noise_nodes = [
            node for node in workflow.values()
            if isinstance(node, dict) and node.get("class_type") == "RandomNoise"
        ]
        assert len(noise_nodes) == 1
        assert noise_nodes[0]["inputs"]["noise_seed"] == 42

    # Path 2: every persisted fingerprint records the same seed.
    state = json.loads(rig.config.run_state_file.read_text())
    fingerprints = [r.get("fingerprint") for r in state["results"].values()]
    assert fingerprints and all(fp is not None for fp in fingerprints)
    assert all(fp["noise_seed"] == 42 for fp in fingerprints)


# --------------------------------------------------------------------------- #
# Issue #27: the shot plan's length requests reach slicing
# --------------------------------------------------------------------------- #


def test_run_pipeline_passes_shot_lengths_from_the_plan_into_slicing(
    tmp_path: Path, monkeypatch
):
    """The plan must be loaded BEFORE slice_audio and its length requests
    handed over -- issue #27's whole mechanism is inert if the plan still
    loads after the chunks exist."""
    rig = Rig(tmp_path)
    plan_path = tmp_path / "shot_plan.toml"
    plan_path.write_text(
        '[[shot]]\n'
        'chunk_id = 0\n'
        'start = 0.0\n'
        'length_seconds = 10.0\n'
        'shot = "One continuous move down the hall"\n'
    )
    rig.config = replace(
        rig.config, shot_plan=plan_path, instrumental_shot_seconds=12.0
    )

    captured: dict = {}
    real_slice = cli.slice_audio

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_slice(*args, **kwargs)

    monkeypatch.setattr(cli, "slice_audio", spy)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    assert captured.get("instrumental_shot_seconds") == pytest.approx(12.0)
    requests = captured.get("shot_lengths")
    assert requests, "the plan's length request never reached slice_audio"
    assert requests[0].start == pytest.approx(0.0)
    assert requests[0].length_seconds == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Issue #28: chain scope wiring
# --------------------------------------------------------------------------- #


def test_default_chain_scope_keeps_voiced_chunks_on_the_audio_conditioned_path(
    tmp_path: Path,
):
    """i2v_continuity on, scope 'instrumental', an all-voiced song: every
    chunk must render through ref2va (audio-conditioned), because a chained
    chunk cannot lip-sync and every chunk here has a lyric."""
    rig = Rig(tmp_path, i2v_continuity=True)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert len(workflows) == 3
    assert all(_is_base(w) for w in workflows), (
        "a voiced chunk took the fl2va path under the default scope -- it will not lip-sync"
    )


def test_chain_scope_all_chains_and_records_the_provenance(tmp_path: Path):
    """Scope 'all': chunks 1 and 2 chain off their predecessors, and the
    recorded fingerprints carry what actually happened."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert _is_base(workflows[0]) and _is_i2v(workflows[1]) and _is_i2v(workflows[2])

    state = json.loads(rig.config.run_state_file.read_text())
    chained = {
        int(cid): r["fingerprint"]["chained_from"] for cid, r in state["results"].items()
    }
    assert chained == {0: None, 1: 0, 2: 1}


def test_a_chained_chunk_is_not_told_the_appearance_a_second_time(tmp_path: Path):
    """Issue #46: the chained path is shown no cast photo, so its identity
    comes entirely from the predecessor's final frame -- a frame that is
    already the appearance clause's own output. Restating the clause applies
    it to its own result, and a relative directive compounds once per chained
    chunk until the lead becomes a different person.

    End-to-end through the real provider, because the whole difficulty is that
    only the provider knows which path a chunk took."""
    rig = Rig(tmp_path, i2v_continuity=True)
    original = rig.config.cast["Dianne"]
    rig.config = replace(
        rig.config,
        i2v_chain_scope="all",
        cast={
            "Dianne": contracts.CastMember(
                name=original.name,
                role=original.role,
                image=original.image,
                appearance="looking a few years younger",
            )
        },
    )
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert _is_base(workflows[0]) and _is_i2v(workflows[1])

    base_prompt = _submitted_prompt(workflows[0])
    chained_prompt = _submitted_prompt(workflows[1])

    # The photo is attached on the base path, so the clause that says how to
    # read it belongs there and only there.
    assert "a few years younger" in base_prompt
    assert "a few years younger" not in chained_prompt
    # ...but the chained shot must still know who is on screen.
    assert "Dianne" in chained_prompt


def test_a_chained_chunk_fingerprints_the_prompt_it_actually_rendered_with(tmp_path: Path):
    """Issue #46 + #34: a chunk records the sentence it was really submitted
    with, not the one Stage 2b planned. Hashing the planned prompt would let a
    chained chunk claim a prompt it never used -- and on fallback to the base
    path, block or permit reuse on evidence from the wrong variant."""
    rig = Rig(tmp_path, i2v_continuity=True)
    original = rig.config.cast["Dianne"]
    rig.config = replace(
        rig.config,
        i2v_chain_scope="all",
        cast={
            "Dianne": contracts.CastMember(
                name=original.name,
                role=original.role,
                image=original.image,
                appearance="looking a few years younger",
            )
        },
    )
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    state = json.loads(rig.config.run_state_file.read_text())

    for chunk_id, workflow in enumerate(workflows):
        recorded = state["results"][str(chunk_id)]["fingerprint"]["prompt_hash"]
        assert recorded == ChunkFingerprint.hash_prompt(_submitted_prompt(workflow)), (
            f"chunk {chunk_id} recorded a prompt hash it did not render with"
        )

    # And the two variants really are different, or this proves nothing.
    assert state["results"]["0"]["fingerprint"]["prompt_hash"] != (
        state["results"]["1"]["fingerprint"]["prompt_hash"]
    )


def test_fingerprints_record_which_audio_conditioned_the_render(tmp_path: Path):
    """Issue #25: a stem run and a mix run of the same span are different
    content, and run_state.json must say which one each mp4 is."""
    rig = Rig(tmp_path)
    stem = write_silent_wav(tmp_path / "audio" / "vocals.wav", seconds=25.0)
    rig.config = replace(rig.config, vocal_stem=stem)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    state = json.loads(rig.config.run_state_file.read_text())
    sources = {r["fingerprint"]["conditioning_source"] for r in state["results"].values()}
    assert sources == {"stem:vocals.wav"}


def test_alignment_overrides_reach_the_render_timeline(tmp_path: Path):
    """Issue #42: an override in the config must change where the chunks
    fall -- otherwise the conditioning stem built against the corrected
    timeline desyncs against a render that silently used the raw one."""
    from music_video_maker.contracts import AlignmentOverride

    rig = Rig(tmp_path)
    # Move the middle segment (8.0-12.5 in DEFAULT_SEGMENT_SPECS) later.
    rig.config = replace(
        rig.config,
        alignment_overrides=(
            AlignmentOverride(
                segment_index=1, start=9.0, end=13.5,
                reason="test: pin the middle line half a second later",
            ),
        ),
    )
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    state = json.loads(rig.config.run_state_file.read_text())
    starts = sorted(r["fingerprint"]["start"] for r in state["results"].values())
    assert any(abs(s - 9.0) < 0.6 for s in starts), (
        f"no chunk starts near the overridden 9.0s: {starts}"
    )


def test_fingerprints_record_which_text_encoder_encoded_the_prompt(tmp_path: Path):
    """Issue #39: the encoder is the component the comprehension defects are
    blamed on, and the only one a template edit can swap without leaving a
    trace. A run that pins nothing still writes down what it used."""
    from tests.harness.factories import WEIGHT_TEXT_ENCODER

    rig = Rig(tmp_path)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    state = json.loads(rig.config.run_state_file.read_text())
    encoders = {r["fingerprint"]["text_encoder"] for r in state["results"].values()}
    assert encoders == {WEIGHT_TEXT_ENCODER}


def test_pinned_text_encoder_reaches_the_graph_and_the_fingerprint(tmp_path: Path):
    """One truth, two paths -- the same value ComfyUI is asked to load is the
    one run_state.json records, pinned here so an A/B cannot silently
    compare a chunk against itself."""
    from music_video_maker.workflow_graph import read_text_encoder

    int8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    rig = Rig(tmp_path)
    rig.config = replace(rig.config, text_encoder=int8)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences)

    submitted = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert {read_text_encoder(w) for w in submitted} == {int8}

    state = json.loads(rig.config.run_state_file.read_text())
    assert {r["fingerprint"]["text_encoder"] for r in state["results"].values()} == {int8}


def test_a_run_holds_a_host_sleep_assertion_for_its_duration(tmp_path: Path, monkeypatch):
    """Issue #43: the orchestrator supervises a render by blocking on a
    socket, which reads to macOS as an idle machine. The assertion must be
    held for the whole run, not left to whoever launched it."""
    spawned: list[list[str]] = []

    class _Proc:
        def terminate(self) -> None:
            pass

        def wait(self, timeout=None):  # noqa: ANN001
            return 0

    def fake_spawner(argv):
        spawned.append(list(argv))
        return _Proc()

    monkeypatch.setattr("music_video_maker.custody.sys.platform", "darwin")
    monkeypatch.setattr(
        "music_video_maker.custody._default_assertion_spawner", fake_spawner
    )

    rig = Rig(tmp_path)
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)])

    assert spawned, "the run never asserted sleep prevention"
    assert spawned[0][0] == "caffeinate"


def test_a_faceless_seed_frame_is_not_chained_from(tmp_path: Path):
    """Issue #47: the chained node has no ``ref_images``, so the seed frame is
    the whole identity conditioning. When the previous shot ends on the back of
    the performer's head there is nothing to build a likeness from and the
    model invents one -- which is how the lead visibly became a different woman
    mid-video. Such a chunk must fall back to the base reference path."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    rig.run(sequences, seed_face_gate=lambda _p: False)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert all(_is_base(w) for w in workflows), (
        "no chunk may chain when every candidate seed frame lacks a face"
    )
    state = json.loads(rig.config.run_state_file.read_text())
    assert {r["fingerprint"]["chained_from"] for r in state["results"].values()} == {None}


def test_the_seed_face_gate_decides_per_boundary_not_per_run(tmp_path: Path):
    """A faceless boundary must not disable chaining everywhere else -- the
    question is about one frame, and shots differ."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    # Reject only the frame bridging chunk 1 -> chunk 2.
    rig.run(sequences, seed_face_gate=lambda p: "seed_0001_into_0002" not in p.name)

    workflows = [contracts.Workflow(s["workflow"]) for s in rig.submitted]
    assert _is_base(workflows[0]), "chunk 0 has no predecessor"
    assert _is_i2v(workflows[1]), "chunk 1's seed passed the gate and must still chain"
    assert _is_base(workflows[2]), "chunk 2's seed was rejected and must fall back"

    state = json.loads(rig.config.run_state_file.read_text())
    chained = {int(c): r["fingerprint"]["chained_from"] for c, r in state["results"].items()}
    assert chained == {0: None, 1: 0, 2: None}


def test_a_raising_seed_face_gate_degrades_instead_of_killing_the_run(tmp_path: Path, caplog):
    """A detector that blows up must cost a slightly different shot, not hours
    of GPU custody. 'Cannot prove a face is present' is the same answer as
    'there is no face' -- both mean the seed is unverified.

    The log assertion is not decoration. ``_stage_seed_frame`` already wraps
    everything in a broad degradation handler, so the run survives a raising
    gate whether or not the gate handles its own errors -- mutation testing
    confirmed that narrowing the gate's own ``except`` changes no observable
    behaviour. What *is* observable is which failure gets reported: the outer
    handler says extraction or upload failed, which would send someone
    debugging ffmpeg when the real problem is the face model. Pinning the
    message keeps the diagnosis honest."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    def exploding_gate(_p):
        raise RuntimeError("model file is corrupt")

    with caplog.at_level(logging.ERROR):
        report = rig.run(sequences, seed_face_gate=exploding_gate)

    assert report.rendered == 3
    assert report.dead_lettered == ()
    assert all(_is_base(contracts.Workflow(s["workflow"])) for s in rig.submitted)
    assert any("seed-face gate raised" in r.message for r in caplog.records), (
        "the gate's own failure must be reported as such, not as a staging failure"
    )
    assert not any("failed to extract/stage a seed frame" in r.message for r in caplog.records)


def test_resume_reuses_a_chunk_whose_chain_was_refused_at_render_time(tmp_path: Path):
    """A degradation is not a config change (issues #28, #47).

    ``planned_chain_source`` predicts chaining from configuration alone, but
    whether a chunk *actually* chains depends on rendered pixels -- the
    predecessor may have failed, its frame may not have extracted, or the #47
    face gate may refuse the seed. Those chunks record ``chained_from = None``
    while the plan still says they should have chained.

    If the resume comparison treats that as a mismatch, every such chunk
    re-renders on every resume, forever, and a run can never converge. The
    stored value is what *happened*; the plan is only a prediction, and a
    prediction the render declined to follow is not evidence the config
    moved."""
    rig = Rig(tmp_path, i2v_continuity=True)
    rig.config = replace(rig.config, i2v_chain_scope="all")
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    # Chunk 2's seed is refused; chunk 1's is fine.
    gate = lambda p: "seed_0001_into_0002" not in p.name  # noqa: E731
    first = rig.run(sequences, seed_face_gate=gate)
    assert first.rendered == 3

    state = json.loads(rig.config.run_state_file.read_text())
    assert state["results"]["2"]["fingerprint"]["chained_from"] is None

    # No WS sequences: any resubmission fails loudly.
    second = rig.run([], resume=True, seed_face_gate=gate)

    assert second.cached == 3, "a gate-refused chunk must not re-render on every resume"
    assert second.rendered == 0


def test_only_chunks_preserves_the_results_of_chunks_it_did_not_render(tmp_path: Path):
    """A slice must not erase the run it is slicing into.

    ``--only-chunks`` wrote a fresh run_state.json containing just the sliced
    chunks, so a full run's 36 records became 2 and the next ``--resume`` found
    nothing to reuse and re-rendered the song. The mp4s were still on disk;
    only the evidence about them was gone -- which is precisely what the state
    file exists to hold."""
    rig = Rig(tmp_path)
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)])
    before = json.loads(rig.config.run_state_file.read_text())
    assert set(before["results"]) == {"0", "1", "2"}

    rig.run([build_success_sequence(rig.seed_success(4, 1))], only_chunks=(1,))

    after = json.loads(rig.config.run_state_file.read_text())
    assert set(after["results"]) == {"0", "1", "2"}, (
        "a slice must augment the state, not replace it"
    )


def test_only_chunks_re_renders_its_chunks_even_when_they_are_cached(tmp_path: Path):
    """Preserving the other chunks must not turn the slice into a no-op: the
    named chunks are being re-rendered precisely because something the
    fingerprint cannot see has changed -- a fixed gate, a new model, a hunch."""
    rig = Rig(tmp_path)
    rig.run([build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)])
    submitted_before = len(rig.submitted)

    rig.run([build_success_sequence(rig.seed_success(4, 1))], only_chunks=(1,))

    # One new submission -- the slice re-rendered chunk 1 rather than reusing
    # the cached copy whose fingerprint still matches. Counted in submissions
    # rather than report.rendered, because the preserved chunks keep the
    # RENDERED status they earned in the first run.
    assert len(rig.submitted) == submitted_before + 1


# --------------------------------------------------------------------------- #
# --prepare (issue #52): Stage 1-2 only, no GPU, no ComfyUI, no custody
# --------------------------------------------------------------------------- #


def test_prepare_shot_plan_writes_a_skeleton_touching_no_comfyui(tmp_path: Path):
    rig = Rig(tmp_path)
    out_path = tmp_path / "shot_plan.toml"

    result_path = cli.prepare_shot_plan(
        rig.config,
        out_path,
        source="run.toml",
        generated_at="2026-08-12",
        align_model=rig.align_model,
    )

    assert result_path == out_path
    text = out_path.read_text()
    # Three lyric segments in DEFAULT_SEGMENT_SPECS, instrumental_coverage off
    # in the rig -- exactly three blank entries.
    assert text.count("[[shot]]") == 3
    assert text.count('shot = ""') == 3
    assert rig.session.requests == []


def test_prepare_shot_plan_refuses_to_overwrite_without_force(tmp_path: Path):
    rig = Rig(tmp_path)
    out_path = tmp_path / "shot_plan.toml"
    out_path.write_text("# hand-authored\n")

    from music_video_maker.shot_plan import ShotPlanError

    with pytest.raises(ShotPlanError):
        cli.prepare_shot_plan(
            rig.config,
            out_path,
            source="run.toml",
            generated_at="2026-08-12",
            align_model=rig.align_model,
        )

    assert out_path.read_text() == "# hand-authored\n"


def test_prepare_shot_plan_force_overwrites(tmp_path: Path):
    rig = Rig(tmp_path)
    out_path = tmp_path / "shot_plan.toml"
    out_path.write_text("# hand-authored\n")

    cli.prepare_shot_plan(
        rig.config,
        out_path,
        source="run.toml",
        generated_at="2026-08-12",
        align_model=rig.align_model,
        force=True,
    )

    assert "[[shot]]" in out_path.read_text()


# --------------------------------------------------------------------------- #
# --prepare --from-plan (issue #52 follow-up, needed by #54 design section 5)
# --------------------------------------------------------------------------- #


def _prepare(rig: Rig, out_path: Path, **kwargs) -> str:
    cli.prepare_shot_plan(
        rig.config,
        out_path,
        source="run.toml",
        generated_at="2026-08-13",
        align_model=rig.align_model,
        force=True,
        **kwargs,
    )
    return out_path.read_text()


def _anchors(skeleton_text: str) -> list[tuple[int, float]]:
    """``(chunk_id, start)`` for every entry in a skeleton, via the real
    loader rather than a regex -- what the skeleton *means* is whatever
    ``load_shot_plan`` reads out of it."""
    payload = tomllib.loads(skeleton_text)
    return [(e["chunk_id"], e["start"]) for e in payload["shot"]]


def test_prepare_from_plan_re_anchors_the_skeleton_against_the_plans_lengths(tmp_path: Path):
    """Without ``--from-plan`` a skeleton describes a timeline no render with
    that plan will ever produce (issue #54 design section 5).

    ``prepare_shot_plan`` used to call ``slice_audio`` with no ``shot_lengths``
    while ``run_pipeline`` passes ``shot_length_requests(plan)``, so the moment
    a plan set one ``length_seconds`` the skeleton's anchors and the render's
    chunks described two different timelines -- and every chunk after the
    first length raised ``ShotPlanDriftError``.
    """
    rig = Rig(tmp_path, instrumental_coverage=True)
    plain = _prepare(rig, tmp_path / "plain.toml")
    plain_anchors = _anchors(plain)

    # Author a long take onto the second chunk of that skeleton.
    plan_path = tmp_path / "authored.toml"
    plan_path.write_text(
        "\n".join(
            f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "beat {cid}"'
            + ("\nlength_seconds = 12.0" if index == 1 else "")
            for index, (cid, start) in enumerate(plain_anchors)
        )
    )

    from_plan = _prepare(rig, tmp_path / "from_plan.toml", from_plan=plan_path)
    from_plan_anchors = _anchors(from_plan)

    assert from_plan_anchors != plain_anchors
    # The long take swallowed the chunks it now covers, so there are fewer of
    # them and the anchors after it have moved.
    assert len(from_plan_anchors) < len(plain_anchors)
    assert from_plan_anchors[1][1] == pytest.approx(plain_anchors[1][1], abs=1e-6)


def test_prepare_from_plan_round_trips_onto_its_own_anchors(tmp_path: Path):
    """The property the whole flow rests on: re-authoring the same lengths at
    the anchors ``--from-plan`` just emitted must give those anchors back.

    Without it, every authoring pass would move the timeline again and a
    generated plan could never converge on the chunks a render produces.
    """
    rig = Rig(tmp_path, instrumental_coverage=True)
    plain_anchors = _anchors(_prepare(rig, tmp_path / "plain.toml"))

    def author(anchors: list[tuple[int, float]], path: Path) -> Path:
        path.write_text(
            "\n".join(
                f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "beat {cid}"'
                + ("\nlength_seconds = 12.0" if index == 1 else "")
                for index, (cid, start) in enumerate(anchors)
            )
        )
        return path

    first = _anchors(
        _prepare(rig, tmp_path / "v1.toml", from_plan=author(plain_anchors, tmp_path / "p1.toml"))
    )
    second = _anchors(
        _prepare(rig, tmp_path / "v2.toml", from_plan=author(first, tmp_path / "p2.toml"))
    )

    assert second == first


def test_prepare_from_plan_with_no_lengths_matches_a_plain_prepare(tmp_path: Path):
    """The conservative default survives the new flag: a plan that expresses
    no editorial length yields no requests, so the skeleton is unchanged."""
    rig = Rig(tmp_path, instrumental_coverage=True)
    plain = _prepare(rig, tmp_path / "plain.toml")

    plan_path = tmp_path / "no_lengths.toml"
    plan_path.write_text(
        "\n".join(
            f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "beat {cid}"'
            for cid, start in _anchors(plain)
        )
    )

    assert _prepare(rig, tmp_path / "same.toml", from_plan=plan_path) == plain


def test_prepare_from_plan_refuses_a_plan_it_cannot_read(tmp_path: Path):
    """A missing or malformed source plan fails the whole prepare rather than
    silently falling back to a length-free skeleton -- the fallback would look
    exactly like success and drift at render time."""
    from music_video_maker.shot_plan import ShotPlanError

    rig = Rig(tmp_path, instrumental_coverage=True)
    out_path = tmp_path / "shot_plan.toml"

    with pytest.raises(ShotPlanError):
        _prepare(rig, out_path, from_plan=tmp_path / "nope.toml")

    assert not out_path.exists()


def test_main_prepare_passes_from_plan_through(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_prepare_shot_plan(config, output_path, **kwargs):
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr(cli, "prepare_shot_plan", fake_prepare_shot_plan)

    exit_code = cli.main(
        ["--config", str(config_path), "--prepare", "--from-plan", str(tmp_path / "old.toml")]
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["from_plan"] == tmp_path / "old.toml"


def test_main_prepare_defaults_from_plan_to_none(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_prepare_shot_plan(config, output_path, **kwargs):
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr(cli, "prepare_shot_plan", fake_prepare_shot_plan)
    cli.main(["--config", str(config_path), "--prepare"])

    assert captured["from_plan"] is None


def test_main_prepare_flag_calls_prepare_shot_plan_and_returns_success(
    tmp_path: Path, monkeypatch
):
    config_path = _make_config_file(tmp_path)
    captured = {}

    def fake_prepare_shot_plan(config, output_path, **kwargs):
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(cli, "prepare_shot_plan", fake_prepare_shot_plan)
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **k: pytest.fail("must not render"))

    exit_code = cli.main(["--config", str(config_path), "--prepare"])

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["output_path"] == config_path.parent / "shot_plan.toml"
    assert captured["kwargs"]["force"] is False


def test_main_prepare_respects_shot_plan_out_and_force(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    custom_out = tmp_path / "custom" / "plan.toml"
    captured = {}

    def fake_prepare_shot_plan(config, output_path, **kwargs):
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(cli, "prepare_shot_plan", fake_prepare_shot_plan)

    cli.main(
        ["--config", str(config_path), "--prepare", "--shot-plan-out", str(custom_out), "--force"]
    )

    assert captured["output_path"] == custom_out
    assert captured["kwargs"]["force"] is True


def test_main_prepare_returns_error_code_when_it_raises(tmp_path: Path, monkeypatch, caplog):
    config_path = _make_config_file(tmp_path)

    def fake_prepare_shot_plan(config, output_path, **kwargs):
        raise cli.PipelineError("no chunks")

    monkeypatch.setattr(cli, "prepare_shot_plan", fake_prepare_shot_plan)

    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(["--config", str(config_path), "--prepare"])

    assert exit_code == cli.EXIT_ERROR
    assert any("Failed to prepare shot plan" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# The alignment model reaches Stage 1. `align()` has always taken a
# `model_size`, but nothing passed it, so every run was pinned to "base" no
# matter what the config said. The gap was invisible because align() has a
# working default -- the run succeeds, it just aligns with the wrong model.
# --------------------------------------------------------------------------- #


def test_configured_alignment_model_size_reaches_align(tmp_path, monkeypatch):
    """A config key nothing reads is worse than no key: it reports a choice
    the run did not make. On "Deathless" this was the difference between 77.2s
    and 187.2s of detected vocal."""
    rig = Rig(tmp_path)
    rig.config = replace(rig.config, alignment_model_size="small")

    seen: dict[str, Any] = {}
    real_align = cli.align

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_align(*args, **kwargs)

    monkeypatch.setattr(cli, "align", spy)
    rig.run([rig.seed_success(1, 0), rig.seed_success(2, 1), rig.seed_success(3, 2)])

    assert seen["model_size"] == "small"


def test_alignment_model_size_default_is_passed_not_assumed(tmp_path, monkeypatch):
    """The default must travel the same path as an override, so the plumbing
    is exercised by every run rather than only by configs that set the key."""
    rig = Rig(tmp_path)

    seen: dict[str, Any] = {}
    real_align = cli.align

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_align(*args, **kwargs)

    monkeypatch.setattr(cli, "align", spy)
    rig.run([rig.seed_success(1, 0), rig.seed_success(2, 1), rig.seed_success(3, 2)])

    assert seen["model_size"] == "base"
