"""Cross-lane integration: resilience (#10) x I2V continuity (#12).

Wave 3 was built as two independent lanes. Each is well covered on its own,
and neither can test the chain that only exists where they meet: the
resilience state machine calls a ``WorkflowProvider`` before every attempt,
and the continuity provider decides the I2V-vs-base path by reading the
``RunState`` that state machine is still building. The invariant issue #12
states outright -- *if chunk N dead-letters, chunk N+1 must fall back to the
non-bridged path* -- lives entirely in that seam: #10 owns writing
``DEAD_LETTERED`` into the run state, #12 owns reacting to it, and a lane
holding only one half can assert neither end of it.

Getting this wrong is expensive and quiet. A chunk that bridges off a
predecessor which never rendered would seed itself from a stale or absent
frame and the run would complete "successfully" with a visibly broken cut,
discovered only after the GPU hours are spent.

Still fully offline: :class:`FakeComfyUISession` and
:class:`ScriptedWebSocket` stand in for ComfyUI, ffmpeg/ffprobe are injected
out, and backoff sleeps through a recording stub. No GPU, no network, no
server, no real time.
"""

from __future__ import annotations

import copy
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from music_video_maker import contracts
from music_video_maker.continuity import (
    DEFAULT_CAST_IMAGE_TITLE,
    DEFAULT_SEED_FRAME_TITLE,
    ContinuityWorkflowProvider,
)
from music_video_maker.execution import ComfyUIExecutionClient
from music_video_maker.resilience import ResilientRunner
from music_video_maker.staging import ComfyUIAssetStager
from music_video_maker.workflow_graph import (
    CLASS_TYPE_H3_IMAGE_TO_VIDEO,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    CLASS_TYPE_IMAGE_LOAD,
    find_nodes_by_class_type,
    find_titled_node,
)
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import (
    make_workflow_baseline,
    make_workflow_i2v,
    write_silent_wav,
)
from tests.harness.ws import (
    ScriptedWebSocket,
    build_hang_sequence,
    build_success_sequence,
)

PROBED_FRAME_COUNT = 124


# --------------------------------------------------------------------------- #
# Local scaffolding
# --------------------------------------------------------------------------- #


class SequencedWSFactory:
    """Hand out a differently-scripted WebSocket per ``connect``.

    ``make_ws_factory`` replays one fixed script for every connection, which
    is fine for a single ``execute()``. A whole run submits once per attempt
    and the mock assigns each submission its own ``prompt_id``, so the
    scripts have to differ per connection.
    """

    def __init__(self, sequences: list[list[Any]]) -> None:
        self._sequences = deque(sequences)
        self.connections: list[ScriptedWebSocket] = []

    def __call__(self, url: str, **kwargs: Any) -> ScriptedWebSocket:
        if not self._sequences:
            raise AssertionError("SequencedWSFactory: more connections than scripted sequences")
        ws = ScriptedWebSocket(self._sequences.popleft())
        ws.connect(url, **kwargs)
        self.connections.append(ws)
        return ws


class FakeFfmpegRunner:
    """Stand in for ffprobe/ffmpeg without either binary.

    ffprobe reports :data:`PROBED_FRAME_COUNT`; ffmpeg writes a real (fake)
    PNG at the destination path, because the next thing that happens to that
    file is a genuine Stage 3 upload that reads and validates it.
    """

    def __init__(self, frame_count: int = PROBED_FRAME_COUNT) -> None:
        self.frame_count = frame_count
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(args, 0, stdout=f"{self.frame_count}\n".encode())
        dest = Path(args[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(make_fake_png_bytes(1344, 768))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


class RecordingSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _abundant_disk(_path: str):
    class _Usage:
        free = 500 * 1024**3

    return _Usage()


class Wave3Rig:
    """Everything wired the way the Wave 4 CLI eventually will: real stager,
    real mutator, real execution client, real resilient runner, real
    continuity provider -- only ComfyUI, ffmpeg and sleep are faked."""

    def __init__(self, tmp_path: Path, *, chunk_count: int, continuity_enabled: bool = True):
        self.tmp_path = tmp_path
        self.chunk_count = chunk_count
        self.session = FakeComfyUISession()
        self.base_template = make_workflow_baseline()
        self.i2v_template = make_workflow_i2v()
        self.base_before = copy.deepcopy(self.base_template)
        self.i2v_before = copy.deepcopy(self.i2v_template)

        image = tmp_path / "dianne_ref.png"
        image.write_bytes(make_fake_png_bytes(1344, 768))

        stager = ComfyUIAssetStager(base_url=self.session.base_url, session=self.session)
        self.prompts: dict[int, contracts.ExpandedPrompt] = {}
        self.assets: dict[int, contracts.StagedAssets] = {}

        for chunk_id in range(chunk_count):
            audio = write_silent_wav(tmp_path / f"chunk_{chunk_id:04d}.wav", seconds=5.2)
            chunk = contracts.AudioChunk(
                chunk_id=chunk_id,
                audio_file=audio,
                start=chunk_id * 5.2,
                end=(chunk_id + 1) * 5.2,
                text=f"line {chunk_id}",
                characters=("Dianne",),
            )
            prompt = contracts.ExpandedPrompt(
                chunk_id=chunk_id,
                prompt=f"expanded prompt {chunk_id}",
                image_ref=image,
                characters=("Dianne",),
            )
            self.prompts[chunk_id] = prompt
            self.assets[chunk_id] = stager.stage_chunk(prompt, chunk)

        self.ffmpeg = FakeFfmpegRunner()
        self.provider = ContinuityWorkflowProvider(
            base_template=self.base_template,
            i2v_template=self.i2v_template,
            chunk_prompts=self.prompts,
            chunk_assets=self.assets,
            asset_stager=stager,
            frames_dir=tmp_path / "frames",
            continuity_enabled=continuity_enabled,
            subprocess_runner=self.ffmpeg,
            expected_length_frames=PROBED_FRAME_COUNT,
        )
        self.sleeper = RecordingSleeper()
        self.output_dir = tmp_path / "out"

    def run(self, ws_sequences: list[list[Any]], *, max_render_attempts: int = 3, resume=False):
        self.ws_factory = SequencedWSFactory(ws_sequences)
        client = ComfyUIExecutionClient(
            base_url=self.session.base_url,
            session=self.session,
            ws_factory=self.ws_factory,
            client_id="wave3-client",
        )
        runner = ResilientRunner(
            client,
            run_state_file=self.tmp_path / "run_state.json",
            max_render_attempts=max_render_attempts,
            retry_backoff_seconds=1.0,
            min_free_disk_gb=1.0,
            run_id="wave3-run",
            sleeper=self.sleeper,
            disk_usage=_abundant_disk,
        )
        return runner.render_run(
            list(range(self.chunk_count)), self.provider, self.output_dir, resume=resume
        )

    def seed_success(self, prompt_number: int, chunk_id: int) -> str:
        """Seed the history + /view bytes for the Nth submission of the run."""
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


# --------------------------------------------------------------------------- #
# The happy chain
# --------------------------------------------------------------------------- #


@pytest.fixture
def bridged_run(tmp_path: Path):
    rig = Wave3Rig(tmp_path, chunk_count=3)
    sequences = []
    for chunk_id in range(3):
        prompt_id = rig.seed_success(chunk_id + 1, chunk_id)
        sequences.append(build_success_sequence(prompt_id))
    run_state = rig.run(sequences)
    return rig, run_state


def test_first_chunk_takes_the_base_path_and_the_rest_bridge(bridged_run):
    """Chunk 0 has no predecessor to seed from, so it must render through the
    base reference path; every later chunk should be on the I2V path."""
    rig, run_state = bridged_run

    assert all(r.succeeded for r in run_state.results.values())
    assert run_state.dead_lettered == ()

    workflows = [s["workflow"] for s in rig.submitted]
    assert len(workflows) == 3
    assert _is_base(workflows[0]) and not _is_i2v(workflows[0])
    for workflow in workflows[1:]:
        assert _is_i2v(workflow) and not _is_base(workflow)


def test_each_bridged_chunk_is_seeded_from_its_own_predecessor(bridged_run):
    """The seam's whole point. The seed frame for chunk N must come from
    chunk N-1's video -- an off-by-one here reads as a continuity glitch in
    the finished cut and nothing else flags it."""
    rig, _ = bridged_run
    workflows = [s["workflow"] for s in rig.submitted]

    for chunk_id in (1, 2):
        _, seed_node = find_titled_node(
            workflows[chunk_id], CLASS_TYPE_IMAGE_LOAD, DEFAULT_SEED_FRAME_TITLE
        )
        assert seed_node["inputs"]["image"] == f"seed_{chunk_id - 1:04d}_into_{chunk_id:04d}.png"

        # ...and the cast photo is still the cast photo, not the seed frame.
        _, cast_node = find_titled_node(
            workflows[chunk_id], CLASS_TYPE_IMAGE_LOAD, DEFAULT_CAST_IMAGE_TITLE
        )
        assert cast_node["inputs"]["image"] == "dianne_ref.png"


def test_seed_frames_are_extracted_from_the_real_rendered_files(bridged_run):
    """The frame is pulled from the MP4 the execution client actually wrote to
    disk, not from a path guessed off the chunk id."""
    rig, run_state = bridged_run
    probed = [c for c in rig.ffmpeg.calls if c[0] == "ffprobe"]

    assert len(probed) == 2  # one per bridge, none for chunk 0
    for chunk_id, call in zip((0, 1), probed, strict=True):
        assert call[-1] == str(run_state.results[chunk_id].video_file)
        assert Path(call[-1]).is_file()


def test_neither_template_is_mutated_in_place_across_the_run(bridged_run):
    """Wave 2's lesson, now with two templates in play: the run reuses one
    loaded copy of each, so a mutation leaking into the template would put
    chunk 0's prompt and audio into every later chunk."""
    rig, _ = bridged_run
    assert rig.base_template == rig.base_before
    assert rig.i2v_template == rig.i2v_before


# --------------------------------------------------------------------------- #
# The invariant neither lane could test
# --------------------------------------------------------------------------- #


def test_chunk_after_a_dead_lettered_chunk_falls_back_to_the_base_path(tmp_path: Path):
    """Issue #12, verbatim: "if chunk N dead-letters, chunk N+1 must fall back
    to the non-bridged path".

    #10 owns writing DEAD_LETTERED into the RunState; #12 owns reading it.
    Only here are both halves present. Chunk 1 hangs on every attempt until
    the watchdog exhausts its retries; chunk 2 must then render through the
    base template rather than seeding itself off a video that never existed.
    """
    rig = Wave3Rig(tmp_path, chunk_count=3)

    sequences = [build_success_sequence(rig.seed_success(1, 0))]
    # Chunk 1: three attempts, each starts then goes silent (watchdog fires).
    sequences += [build_hang_sequence(f"prompt-{n:04d}") for n in (2, 3, 4)]
    sequences.append(build_success_sequence(rig.seed_success(5, 2)))

    run_state = rig.run(sequences, max_render_attempts=3)

    # The run survived the failure and finished every other chunk.
    assert run_state.dead_lettered == (1,)
    assert run_state.results[1].attempts == 3
    assert len(run_state.results[1].errors) == 3
    assert run_state.results[0].succeeded and run_state.results[2].succeeded

    workflows = [s["workflow"] for s in rig.submitted]
    assert len(workflows) == 5  # 1 + 3 attempts + 1

    # Chunk 2's submission -- the last one -- must be the non-bridged path.
    assert _is_base(workflows[-1]), "chunk 2 bridged off a chunk that never rendered"
    assert not _is_i2v(workflows[-1])

    # Recovery really ran between attempts, and no real sleeping happened.
    assert rig.session.interrupt_calls >= 3
    assert rig.session.free_calls
    assert rig.sleeper.delays == [1.0, 2.0]  # backoff only between retries


def test_dead_lettered_predecessor_stages_no_seed_frame(tmp_path: Path):
    """The fallback must be a genuine skip, not a bridge onto a stale frame:
    no extraction and no upload should happen for the failed predecessor."""
    rig = Wave3Rig(tmp_path, chunk_count=2)
    sequences = [build_hang_sequence(f"prompt-{n:04d}") for n in (1, 2)]
    sequences.append(build_success_sequence(rig.seed_success(3, 1)))

    run_state = rig.run(sequences, max_render_attempts=2)

    assert run_state.dead_lettered == (0,)
    assert rig.ffmpeg.calls == []
    assert not any(u.original_filename.startswith("seed_") for u in rig.session.uploads)


# --------------------------------------------------------------------------- #
# Resume across the seam
# --------------------------------------------------------------------------- #


def test_a_resumed_cached_chunk_still_seeds_the_next_one(tmp_path: Path):
    """On resume, chunk 0 comes back as CACHED rather than RENDERED. The
    provider keys off ``ChunkResult.succeeded``, which covers both -- if it
    checked ``status is RENDERED`` instead, every resumed run would silently
    lose its continuity bridge at the resume point.
    """
    # First pass gets through chunk 0 only (the run was interrupted after it).
    rig = Wave3Rig(tmp_path, chunk_count=1)
    first = rig.run([build_success_sequence(rig.seed_success(1, 0))])
    assert first.results[0].status is contracts.ChunkStatus.RENDERED

    # Second pass: a fresh rig (new process, new ComfyUI session) reading the
    # persisted run state. Chunk 0's video is still on disk, so only chunk 1
    # should render.
    resumed_rig = Wave3Rig(tmp_path, chunk_count=2)
    resumed = resumed_rig.run(
        [build_success_sequence(resumed_rig.seed_success(1, 1))], resume=True
    )

    assert resumed.results[0].status is contracts.ChunkStatus.CACHED
    assert resumed.results[1].succeeded

    # Exactly one submission this pass -- chunk 0 was not re-rendered ...
    workflows = [s["workflow"] for s in resumed_rig.submitted]
    assert len(workflows) == 1
    # ... and chunk 1 still bridged off the cached predecessor.
    assert _is_i2v(workflows[0])
    _, seed_node = find_titled_node(workflows[0], CLASS_TYPE_IMAGE_LOAD, DEFAULT_SEED_FRAME_TITLE)
    assert seed_node["inputs"]["image"] == "seed_0000_into_0001.png"


def test_continuity_disabled_keeps_every_chunk_on_the_base_path(tmp_path: Path):
    """The toggle has to hold all the way through the state machine, not just
    at the provider's front door."""
    rig = Wave3Rig(tmp_path, chunk_count=2, continuity_enabled=False)
    sequences = [
        build_success_sequence(rig.seed_success(1, 0)),
        build_success_sequence(rig.seed_success(2, 1)),
    ]

    run_state = rig.run(sequences)

    assert all(r.succeeded for r in run_state.results.values())
    assert all(_is_base(s["workflow"]) for s in rig.submitted)
    assert rig.ffmpeg.calls == []
