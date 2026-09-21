"""A spoken prologue, end to end through the mock ComfyUI (issue #66).

The same rig ``tests/test_cli.py`` uses -- ``FakeComfyUISession``, a faked
stable-ts model, a faked ffmpeg/ffprobe runner, a recording sleeper -- with a
``[[segment]]`` in front of the song. No GPU, no network, no server, no real
ffmpeg, no real time.

What is worth testing here is only what the *seam* introduces, because
everything else is the song's own path unchanged: two timelines render into
two chunks directories under two run states, the fingerprints say which
timeline each chunk belongs to, ``--resume`` cannot cross between them, and
Stage 5 joins them rather than concatenating a song with a prologue's clips
mixed in.
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from music_video_maker import cli, contracts
from music_video_maker.assembly import DEFAULT_OUTPUT_FILENAME
from music_video_maker.timelines import Segment
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import make_workflow_baseline, write_silent_wav
from tests.harness.ws import build_success_sequence
from tests.test_cli import (
    _ALLOW_ANY_SEED,
    FakeClock,
    RecordingSleeper,
    SequencedWSFactory,
    _abundant_disk,
    _raw_result,
)

SONG_SECONDS = 25.0
PROLOGUE_SECONDS = 12.0

SONG_SEGMENTS = [
    ("walking through the empty halls tonight", 0.0, 6.5),
    ("nobody is watching nobody cares", 8.0, 12.5),
    ("the lights flicker but i do not mind", 14.0, 19.0),
]
PROLOGUE_SEGMENTS = [("you were never going to find me here", 1.0, 6.0)]
"""Deliberately a different shape from the song's: a prologue that happened
to produce the same chunk count would let a mix-up pass unnoticed."""


class PerAudioAlignModel:
    """A stable-ts stand-in that answers differently per audio file.

    The single-result fake ``tests/test_cli.py`` uses cannot express two
    timelines: it would hand the prologue the song's own segments, and every
    assertion about the two being independent would be vacuous."""

    def __init__(self, by_name: dict[str, Any]):
        self._by_name = by_name
        self.aligned: list[str] = []

    def align(self, audio: str, text: str, **kwargs: Any):
        name = Path(audio).name
        self.aligned.append(name)
        assert name in self._by_name, f"nothing scripted for {name}"
        return self._by_name[name]


class PrologueFfmpeg:
    """ffmpeg/ffprobe for a two-timeline assembly, modelled rather than
    stubbed -- see ``tests/test_assembly_timelines.py``'s module docstring for
    why a constant-returning fake would make the seam check vacuous.

    Chunk mp4s all report ``chunk_seconds``; a concat sums its list; an
    ``apad`` reports its ``whole_dur``; a mux keeps its video input's length.
    """

    def __init__(self, luminance_level: int = 128):
        self.luminance_level = luminance_level
        self.durations: dict[Path, float] = {}
        self.calls: list[list[str]] = []

    def register(self, path: Path, seconds: float) -> None:
        self.durations[Path(path)] = seconds

    def _duration_of(self, path: Path) -> float:
        """A rendered chunk is exactly as long as the stem it was cut to
        (issue #20), so the mp4's duration is read off its own wav rather
        than invented -- which is what makes the overshoot in this rig the
        *real* overshoot Stage 2 produced, not a number chosen to pass."""
        if path in self.durations:
            return self.durations[path]
        if path.suffix == ".wav":
            with wave.open(str(path)) as handle:
                return handle.getnframes() / handle.getframerate()
        if path.suffix == ".mp4":
            match = re.search(r"chunk_(\d+)", path.name)
            assert match, f"cannot tell which chunk {path} is"
            stem = path.parent / f"chunk_{int(match.group(1)):03d}.wav"
            return self._duration_of(stem)
        raise AssertionError(f"no duration registered for {path}")

    def __call__(self, args: Any) -> subprocess.CompletedProcess:
        args = [str(a) for a in args]
        self.calls.append(args)
        if args[0] == "ffprobe":
            return self._probe(args)
        return self._run(args)

    def _probe(self, args) -> subprocess.CompletedProcess:
        if any("format=duration" in a for a in args):
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{self._duration_of(Path(args[-1]))}\n".encode()
            )
        if any("codec_name" in a for a in args):
            body = "\n".join(
                f"{k}={v}"
                for k, v in {
                    "codec_name": "h264",
                    "profile": "High",
                    "level": "30",
                    "width": "864",
                    "height": "480",
                    "pix_fmt": "yuv420p",
                    "field_order": "progressive",
                    "r_frame_rate": "24/1",
                    "time_base": "1/12288",
                }.items()
            )
            return subprocess.CompletedProcess(args, 0, stdout=body.encode())
        if any("sample_rate" in a for a in args):
            return subprocess.CompletedProcess(args, 0, stdout=b"sample_rate=44100\nchannels=2\n")
        return subprocess.CompletedProcess(args, 0, stdout=b"124\n")

    def _run(self, args) -> subprocess.CompletedProcess:
        if args[-1] == "-":  # the #77 luminance probe / #81 scene probe sink
            width, height = 32, 18
            if "-s" in args:
                width, height = (int(p) for p in args[args.index("-s") + 1].split("x"))
            if "-f" in args and args[args.index("-f") + 1] == "null":
                return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
            frame = bytes([self.luminance_level]) * (width * height)
            return subprocess.CompletedProcess(args, 0, stdout=frame, stderr=b"")

        destination = Path(args[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-mp4-bytes")

        if "-f" in args and args[args.index("-f") + 1] == "concat":
            listing = Path(args[args.index("-i") + 1])
            entries = [
                Path(line[len("file '") : -1])
                for line in listing.read_text().splitlines()
                if line.startswith("file '")
            ]
            self.durations[destination] = sum(self._duration_of(e) for e in entries)
        elif "-af" in args:
            whole = re.search(r"whole_dur=([\d.]+)", args[args.index("-af") + 1])
            assert whole is not None
            source = Path(args[args.index("-i") + 1])
            self.durations[destination] = max(float(whole.group(1)), self._duration_of(source))
        elif "-filter_complex" in args:
            inputs = [Path(args[i + 1]) for i, a in enumerate(args) if a == "-i"]
            self.durations[destination] = sum(self._duration_of(p) for p in inputs)
        elif "-map" in args:
            self.durations[destination] = self._duration_of(Path(args[args.index("-i") + 1]))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


class PrologueRig:
    """``tests/test_cli.py``'s ``Rig`` with a ``[[segment]]`` in front."""

    def __init__(self, tmp_path: Path, *, with_prologue: bool = True):
        self.tmp_path = tmp_path
        self.session = FakeComfyUISession()
        self.ffmpeg = PrologueFfmpeg()
        self.sleeper = RecordingSleeper()
        self.clock = FakeClock()

        master = write_silent_wav(tmp_path / "audio" / "master.wav", seconds=SONG_SECONDS)
        prologue_audio = write_silent_wav(
            tmp_path / "audio" / "prologue.wav", seconds=PROLOGUE_SECONDS
        )
        self.ffmpeg.register(master, SONG_SECONDS)
        self.ffmpeg.register(prologue_audio, PROLOGUE_SECONDS)

        lyrics = tmp_path / "lyrics.txt"
        lyrics.write_text(
            "Walking through the empty halls tonight\n"
            "Nobody is watching nobody cares\n"
            "The lights flicker but I do not mind\n"
        )
        script = tmp_path / "prologue.txt"
        script.write_text("You were never going to find me here\n")

        cast_image = tmp_path / "cast" / "dianne_ref.png"
        cast_image.parent.mkdir(parents=True, exist_ok=True)
        cast_image.write_bytes(make_fake_png_bytes(1344, 768))

        chunks_dir = tmp_path / "output" / "chunks"
        template = tmp_path / "workflow_api.json"
        template.write_text(json.dumps(make_workflow_baseline()))

        from music_video_maker.config import RunConfig

        self.align_model = PerAudioAlignModel(
            {
                "master.wav": _raw_result(SONG_SEGMENTS),
                "prologue.wav": _raw_result(PROLOGUE_SEGMENTS),
            }
        )
        self.config = RunConfig(
            master_audio=master,
            lyrics_file=lyrics,
            global_style="Refestramus progressive rock music video, 35mm film",
            narrative_concept="Wandering through a surgery",
            cast={
                "Dianne": contracts.CastMember(
                    name="Dianne", role="Lead Vocalist", image=cast_image
                )
            },
            default_lead_vocalist="Dianne",
            comfyui_url=self.session.base_url,
            workflow_template=template,
            chunks_dir=chunks_dir,
            final_video_dir=tmp_path / "output" / "final",
            hardware=contracts.HardwareProfile(name="RTX 4090 24GB (doris)", vram_gb=24.0),
            max_render_attempts=3,
            retry_backoff_seconds=1.0,
            min_free_disk_gb=1.0,
            run_state_file=chunks_dir / "run_state.json",
            instrumental_coverage=True,
            segments=(
                (
                    Segment(
                        name="prologue",
                        audio=prologue_audio,
                        script=script,
                        position="before",
                    ),
                )
                if with_prologue
                else ()
            ),
        )

    def run(self, *, chunk_count: int, resume: bool = False, **kwargs) -> cli.RunReport:
        sequences = []
        for n in range(1, chunk_count + 1):
            prompt_id = f"prompt-{n:04d}"
            self.session.seed_history_success(
                prompt_id, video_filename=f"mvm_chunk_{n:04d}_00001.mp4"
            )
            sequences.append(build_success_sequence(prompt_id))
        return cli.run_pipeline(
            self.config,
            resume=resume,
            align_model=self.align_model,
            comfyui_session=self.session,
            ws_factory=SequencedWSFactory(sequences),
            ffmpeg_runner=self.ffmpeg,
            sleeper=self.sleeper,
            disk_usage=_abundant_disk,
            clock=self.clock,
            seed_face_gate=_ALLOW_ANY_SEED,
            **kwargs,
        )


def _states(report: cli.RunReport) -> dict[str, contracts.RunState]:
    return dict(report.timeline_states)


# --------------------------------------------------------------------------- #
# Two timelines, one video
# --------------------------------------------------------------------------- #


def test_a_prologue_renders_as_its_own_timeline_and_both_are_assembled(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    report = rig.run(chunk_count=12)

    assert report.dead_lettered == ()
    states = _states(report)
    assert list(states) == ["prologue", "song"]
    assert states["prologue"].results, "the prologue rendered no chunks at all"
    assert states["song"].results
    assert report.total_chunks == sum(len(s.results) for s in states.values())
    assert report.output_video == rig.config.final_video_dir / DEFAULT_OUTPUT_FILENAME
    assert report.output_video.is_file()


def test_each_timeline_aligns_its_own_audio_against_its_own_text(tmp_path: Path):
    """The reframing the whole feature rests on: a prologue is the same
    Stage 1 over a second (text, audio) pair, not a special case inside the
    song's."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)
    assert rig.align_model.aligned == ["prologue.wav", "master.wav"]


def test_the_two_timelines_chunks_never_share_a_directory(tmp_path: Path):
    """Chunk ids are a separate id space per timeline, which is only safe
    because the files are separate: both timelines have a chunk 0."""
    rig = PrologueRig(tmp_path)
    report = rig.run(chunk_count=12)
    states = _states(report)

    assert 0 in states["prologue"].results
    assert 0 in states["song"].results
    prologue_dir = rig.config.chunks_dir / "prologue"
    assert (prologue_dir / "chunk_000.wav").is_file()
    assert (rig.config.chunks_dir / "chunk_000.wav").is_file()
    assert (prologue_dir / "run_state.json").is_file()
    assert rig.config.run_state_file.is_file()


def test_every_chunk_records_the_timeline_it_belongs_to(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    report = rig.run(chunk_count=12)
    states = _states(report)

    assert all(
        r.fingerprint is not None and r.fingerprint.timeline == "prologue"
        for r in states["prologue"].results.values()
    )
    assert all(
        r.fingerprint is not None and r.fingerprint.timeline is None
        for r in states["song"].results.values()
    )


def test_a_prologue_chunk_is_prompted_as_speech_not_as_a_sung_lyric(tmp_path: Path):
    """H3 lip-syncs whatever audio it is handed either way; what this avoids
    is telling the model a sung performance is happening over dialogue."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    prompts = [
        node["inputs"]["prompt"]
        for submission in rig.session.submitted_prompts
        if not submission.get("rejected")
        for node in submission["workflow"].values()
        if node.get("class_type") == "MiniMaxH3ReferenceToVideo"
    ]
    spoken = [p for p in prompts if "actively speaking the line" in p]
    sung = [p for p in prompts if "actively singing the lyric" in p]
    assert spoken, "no prologue chunk was prompted as speech"
    assert sung, "the song's own chunks stopped being prompted as sung"
    assert "you were never going to find me" in " ".join(spoken).lower()


def test_the_run_state_files_do_not_overwrite_each_other(tmp_path: Path):
    """``RunState.results`` is keyed by chunk id, so one shared file would
    have the prologue's chunk 0 silently replace the song's."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    song_state = json.loads(rig.config.run_state_file.read_text())
    prologue_state = json.loads(
        (rig.config.chunks_dir / "prologue" / "run_state.json").read_text()
    )
    assert song_state["run_id"] != prologue_state["run_id"]
    song_fp = song_state["results"]["0"]["fingerprint"]
    prologue_fp = prologue_state["results"]["0"]["fingerprint"]
    assert song_fp["timeline"] is None
    assert prologue_fp["timeline"] == "prologue"


# --------------------------------------------------------------------------- #
# The seam, from the pipeline's own end
# --------------------------------------------------------------------------- #


def test_stage_5_joins_the_timelines_rather_than_concatenating_the_chunks(tmp_path: Path):
    """Each timeline is concatenated and *measured* on its own before they
    are joined -- that per-timeline duration is what the seam arithmetic
    needs, and summing chunk spans instead is what issue #22 found nothing
    was checking."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    final_dir = rig.config.final_video_dir
    assert (final_dir / "_timeline_prologue.mp4").is_file()
    assert (final_dir / "_timeline_song.mp4").is_file()
    assert (final_dir / "concat_list_prologue.txt").is_file()
    assert (final_dir / "concat_list_song.txt").is_file()

    joined = (final_dir / "concat_list.txt").read_text()
    assert "_timeline_prologue.mp4" in joined
    assert "_timeline_song.mp4" in joined
    assert "chunk_" not in joined, "the final concat should join timelines, not chunks"


def test_each_timelines_audio_is_padded_and_measured_before_the_mux(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    pads = [c for c in rig.ffmpeg.calls if "-af" in c and "apad" in c[c.index("-af") + 1]]
    assert len(pads) == 2
    probed = {
        Path(c[-1])
        for c in rig.ffmpeg.calls
        if c[0] == "ffprobe" and any("format=duration" in a for a in c)
    }
    for pad in pads:
        assert Path(pad[-1]) in probed


def test_the_master_track_is_still_the_songs_audio_at_the_seam(tmp_path: Path):
    """The invariant does not lapse because there are two timelines: the
    song's own region of the finished file is still the master, and the only
    thing added anywhere is silence."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    pad_inputs = [
        Path(c[c.index("-i") + 1]) for c in rig.ffmpeg.calls if "-af" in c
    ]
    assert rig.config.master_audio in pad_inputs
    assert rig.config.segments[0].audio in pad_inputs
    concat_calls = [c for c in rig.ffmpeg.calls if "-f" in c and "concat" in c]
    assert all("-an" in c for c in concat_calls)


# --------------------------------------------------------------------------- #
# --resume across timelines
# --------------------------------------------------------------------------- #


def test_resume_reuses_each_timelines_own_chunks_and_renders_nothing_new(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    first = rig.run(chunk_count=12)
    assert first.rendered > 0

    second = rig.run(chunk_count=0, resume=True)
    assert second.rendered == 0
    assert second.cached == first.rendered
    assert second.dead_lettered == ()


def test_a_prologue_chunk_cannot_be_reused_as_the_songs_chunk(tmp_path: Path):
    """The discriminator doing its job: copy the prologue's state file over
    the song's, and every chunk in it is refused as belonging to somewhere
    else rather than silently assembled into the song."""
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)

    prologue_state = (rig.config.chunks_dir / "prologue" / "run_state.json").read_text()
    rig.config.run_state_file.write_text(prologue_state)

    from music_video_maker.resilience import load_run_state

    stolen = load_run_state(rig.config.run_state_file)
    song_fingerprints = {
        cid: result.fingerprint for cid, result in stolen.results.items()
    }
    assert song_fingerprints
    for fingerprint in song_fingerprints.values():
        assert fingerprint is not None
        expected = replace(fingerprint, timeline=None)
        assert expected.timeline_differences(fingerprint) == ("timeline",)


# --------------------------------------------------------------------------- #
# A slice names its timeline
# --------------------------------------------------------------------------- #


def test_a_slice_renders_only_the_named_timeline(tmp_path: Path):
    """A slice assembles nothing, so rendering the other timelines would be
    hours of GPU custody spent on an artefact the run deliberately does not
    produce."""
    rig = PrologueRig(tmp_path)
    report = rig.run(chunk_count=2, only_chunks=(0,), flag_timeline="prologue")

    assert list(_states(report)) == ["prologue"]
    assert report.output_video is None
    assert not (rig.config.chunks_dir / "chunk_000.wav").exists()


def test_a_slice_defaults_to_the_song_when_no_timeline_is_named(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    report = rig.run(chunk_count=2, only_chunks=(0,))
    assert list(_states(report)) == ["song"]


def test_an_unknown_timeline_name_is_refused_rather_than_falling_back(tmp_path: Path):
    """Rendering the wrong timeline's chunk 3 because of a typo is exactly
    the confusion the separate id spaces exist to make impossible."""
    rig = PrologueRig(tmp_path)
    with pytest.raises(cli.PipelineError, match="not a timeline in this run"):
        rig.run(chunk_count=0, only_chunks=(0,), flag_timeline="epilogue")


def test_reseed_names_ids_in_the_timeline_it_was_pointed_at(tmp_path: Path):
    rig = PrologueRig(tmp_path)
    rig.run(chunk_count=12)
    with pytest.raises(cli.PipelineError, match="timeline 'prologue'"):
        rig.run(chunk_count=0, reseed_chunk_ids=(999,), flag_timeline="prologue")


# --------------------------------------------------------------------------- #
# A config with no segments is unchanged
# --------------------------------------------------------------------------- #


def test_without_a_segment_nothing_about_the_run_changes(tmp_path: Path):
    """The single-timeline path: one chunks directory, one run state, the
    same two ffmpeg calls Stage 5 has always made, no per-timeline files."""
    rig = PrologueRig(tmp_path, with_prologue=False)
    report = rig.run(chunk_count=12)

    assert [name for name, _ in report.timeline_states] == ["song"]
    assert not (rig.config.chunks_dir / "prologue").exists()
    final_dir = rig.config.final_video_dir
    assert (final_dir / "_concat_intermediate.mp4").is_file()
    assert not list(final_dir.glob("_timeline_*.mp4"))
    assert not any("-af" in c for c in rig.ffmpeg.calls)
    # And no new probing: the music-video path still measures nothing.
    assert not any(
        c[0] == "ffprobe" and any("format=duration" in a for a in c)
        for c in rig.ffmpeg.calls
    )


# --------------------------------------------------------------------------- #
# --prepare reports both timelines
# --------------------------------------------------------------------------- #


def test_prepare_writes_a_skeleton_per_timeline(tmp_path: Path):
    """#52's whole point is that a plan's anchors are never transcribed by
    hand, and a segment's anchors come from its own alignment exactly like
    the song's."""
    rig = PrologueRig(tmp_path)
    out = tmp_path / "shot_plan.toml"
    returned = cli.prepare_shot_plan(
        rig.config,
        out,
        source="run.toml",
        generated_at="2026-09-21",
        align_model=rig.align_model,
    )
    assert returned == out
    assert out.is_file()
    prologue_plan = tmp_path / "shot_plan__prologue.toml"
    assert prologue_plan.is_file()
    assert "chunk_id" in prologue_plan.read_text()
    assert prologue_plan.read_text() != out.read_text()


def test_prepare_reports_the_predicted_seam_before_any_gpu_time(tmp_path: Path, caplog):
    rig = PrologueRig(tmp_path)
    with caplog.at_level("INFO"):
        cli.prepare_timelines(rig.config, align_model=rig.align_model)
    assert "predicted (Stage 2, no GPU)" in caplog.text
    assert "Timeline 'prologue'" in caplog.text
    assert "Timeline 'song'" in caplog.text


def test_prepare_on_a_segment_free_config_reports_no_seam(tmp_path: Path, caplog):
    rig = PrologueRig(tmp_path, with_prologue=False)
    with caplog.at_level("INFO"):
        prepared = cli.prepare_timelines(rig.config, align_model=rig.align_model)
    assert len(prepared) == 1
    assert "Seam total" not in caplog.text
