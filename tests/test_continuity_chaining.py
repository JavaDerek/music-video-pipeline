"""Issue #28: chaining consecutive shots via H3's first/last-frame (fl2va) path.

Issue #12 built the seed-and-feed machinery; this covers what #28 adds on top
of it, all of which is about *how far* the chain is allowed to run and about
being able to say afterwards what each chunk was actually chained to:

* **Re-anchoring** (``reanchor_interval``) -- issue #28's "error accumulation"
  caveat. A chain that never resets compounds drift across a whole song, so
  every Nth chunk deliberately renders through the base reference path.
* **Eligibility** (``chainable_chunk_ids``) -- ``MiniMaxH3ImageToVideo`` has no
  audio-conditioning input at all, so a chained chunk cannot lip-sync. Which
  chunks may pay that price is the orchestrator's call, not this module's.
* **Provenance** (``chain_sources``) -- what each chunk's first frame actually
  came from, so ``--resume`` and the run report can tell a chained chunk from
  an unchained one instead of assuming.

Fully offline: no ffmpeg/ffprobe binary, no network, no ComfyUI, no GPU. The
end-to-end section drives the real stager / execution client / resilient runner
against :class:`FakeComfyUISession` and :class:`ScriptedWebSocket`.
"""

from __future__ import annotations

import copy
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from music_video_maker import continuity, contracts
from music_video_maker.contracts import (
    ChunkResult,
    ChunkStatus,
    ExpandedPrompt,
    RunState,
    StagedAssets,
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
    load_workflow_template,
)
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import (
    make_workflow_baseline,
    make_workflow_i2v,
    write_silent_wav,
)
from tests.harness.ws import ScriptedWebSocket, build_success_sequence

PROBED_FRAME_COUNT = 124


# --------------------------------------------------------------------------- #
# Local doubles
# --------------------------------------------------------------------------- #


def _prompt(chunk_id: int) -> ExpandedPrompt:
    return ExpandedPrompt(
        chunk_id=chunk_id,
        prompt=f"expanded prompt {chunk_id}",
        image_ref=Path(f"cast/ref_{chunk_id}.png"),
        characters=("Dianne",),
    )


def _assets(chunk_id: int) -> StagedAssets:
    return StagedAssets(
        chunk_id=chunk_id,
        image_filename=f"staged_image_{chunk_id:04d}.png",
        audio_filename=f"staged_audio_{chunk_id:04d}.wav",
    )


class _FakeStager:
    def __init__(self) -> None:
        self.uploaded: list[Path] = []

    def upload_image(self, path: Path) -> str:
        self.uploaded.append(Path(path))
        return f"server_{Path(path).name}"

    def upload_audio(self, path: Path) -> str:  # pragma: no cover - never used here
        raise NotImplementedError


class _FakeRunner:
    """Scripted ffprobe/ffmpeg. Writes a real (fake) PNG so uploads can read it."""

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
        dest.write_bytes(make_fake_png_bytes(864, 480))
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


def _rendered(chunk_id: int, video: Path) -> ChunkResult:
    return ChunkResult(chunk_id=chunk_id, status=ChunkStatus.RENDERED, video_file=video)


def _state_with_rendered(tmp_path: Path, *chunk_ids: int) -> RunState:
    results = {}
    for chunk_id in chunk_ids:
        video = tmp_path / f"chunk_{chunk_id:04d}.mp4"
        video.write_bytes(b"x")
        results[chunk_id] = _rendered(chunk_id, video)
    return RunState(run_id="r1", results=results)


def _provider(tmp_path: Path, **kwargs: Any) -> continuity.ContinuityWorkflowProvider:
    chunk_ids = kwargs.pop("chunk_ids", range(8))
    return continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts={cid: _prompt(cid) for cid in chunk_ids},
        chunk_assets={cid: _assets(cid) for cid in chunk_ids},
        asset_stager=kwargs.pop("asset_stager", None) or _FakeStager(),
        frames_dir=tmp_path / "frames",
        continuity_enabled=kwargs.pop("continuity_enabled", True),
        subprocess_runner=kwargs.pop("subprocess_runner", None) or _FakeRunner(),
        **kwargs,
    )


def _is_chained(workflow: contracts.Workflow) -> bool:
    return bool(find_nodes_by_class_type(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO))


def _is_base(workflow: contracts.Workflow) -> bool:
    return bool(find_nodes_by_class_type(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO))


# --------------------------------------------------------------------------- #
# Pure chain policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("chunk_id", "interval", "expected"),
    [
        (0, None, False),
        (7, None, False),
        (0, 4, True),
        (1, 4, False),
        (3, 4, False),
        (4, 4, True),
        (8, 4, True),
        (3, 1, True),  # interval=1 -- every chunk re-anchors, chaining is off
        (3, 0, False),  # nonsensical interval must never silently chain-block
        (3, -2, False),
    ],
)
def test_is_reanchor_chunk(chunk_id, interval, expected):
    assert continuity.is_reanchor_chunk(chunk_id, interval) is expected


def test_planned_chain_source_is_none_when_continuity_is_disabled():
    assert continuity.planned_chain_source(5, continuity_enabled=False) is None


def test_planned_chain_source_is_none_for_the_first_chunk():
    assert continuity.planned_chain_source(0, continuity_enabled=True) is None


def test_planned_chain_source_is_the_immediate_predecessor():
    assert continuity.planned_chain_source(5, continuity_enabled=True) == 4


def test_planned_chain_source_respects_the_reanchor_interval():
    assert continuity.planned_chain_source(4, continuity_enabled=True, reanchor_interval=4) is None
    assert continuity.planned_chain_source(5, continuity_enabled=True, reanchor_interval=4) == 4


def test_planned_chain_source_respects_the_eligibility_set():
    assert (
        continuity.planned_chain_source(1, continuity_enabled=True, chainable_chunk_ids={2, 3})
        is None
    )
    assert (
        continuity.planned_chain_source(2, continuity_enabled=True, chainable_chunk_ids={2, 3}) == 1
    )


def test_planned_chain_source_treats_an_empty_eligibility_set_as_no_chaining():
    # Distinct from None: an explicitly empty set means "nothing is eligible",
    # which is exactly what an orchestrator computes for an all-voiced song.
    assert (
        continuity.planned_chain_source(3, continuity_enabled=True, chainable_chunk_ids=set())
        is None
    )


@pytest.mark.parametrize(
    ("stored", "rerendered", "expected"),
    [
        (None, {3}, False),
        (3, {3}, True),
        (3, {4}, False),
        (0, set(), False),
        (0, {0}, True),
    ],
)
def test_chain_reuse_blocked(stored, rerendered, expected):
    assert continuity.chain_reuse_blocked(stored, rerendered) is expected


# --------------------------------------------------------------------------- #
# Re-anchoring (issue #28's error-accumulation caveat)
# --------------------------------------------------------------------------- #


def test_reanchor_interval_forces_the_base_path_even_with_a_good_predecessor(tmp_path, caplog):
    state = _state_with_rendered(tmp_path, 0, 1)
    provider = _provider(tmp_path, reanchor_interval=2)

    with caplog.at_level("INFO"):
        workflow = provider(2, state)

    assert _is_base(workflow)
    assert not _is_chained(workflow)
    assert any("re-anchor" in r.message for r in caplog.records)


def test_chunks_between_reanchor_points_still_chain(tmp_path):
    state = _state_with_rendered(tmp_path, 0, 1, 2)
    provider = _provider(tmp_path, reanchor_interval=3)

    assert _is_chained(provider(1, state))
    assert _is_chained(provider(2, state))
    assert _is_base(provider(3, state))


def test_no_seed_frame_is_extracted_at_a_reanchor_point(tmp_path):
    state = _state_with_rendered(tmp_path, 0, 1)
    runner = _FakeRunner()
    provider = _provider(tmp_path, reanchor_interval=2, subprocess_runner=runner)

    provider(2, state)

    assert runner.calls == []  # not even an ffprobe -- the decision is made first


def test_reanchor_interval_below_one_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="reanchor_interval"):
        _provider(tmp_path, reanchor_interval=0)


# --------------------------------------------------------------------------- #
# Eligibility (no audio conditioning on the fl2va node -> no lip sync)
# --------------------------------------------------------------------------- #


def test_ineligible_chunk_takes_the_base_path(tmp_path, caplog):
    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path, chainable_chunk_ids={5})

    with caplog.at_level("INFO"):
        workflow = provider(1, state)

    assert _is_base(workflow)
    assert any("not eligible" in r.message for r in caplog.records)


def test_eligible_chunk_takes_the_chained_path(tmp_path):
    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path, chainable_chunk_ids={1})

    assert _is_chained(provider(1, state))


def test_unrestricted_chaining_warns_that_chained_chunks_cannot_lip_sync(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _provider(tmp_path)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("lip" in message.lower() for message in warnings)


def test_restricted_chaining_does_not_emit_the_lip_sync_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _provider(tmp_path, chainable_chunk_ids={1, 2})

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert not any("lip" in message.lower() for message in warnings)


def test_disabled_continuity_does_not_emit_the_lip_sync_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _provider(tmp_path, continuity_enabled=False)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert not any("lip" in message.lower() for message in warnings)


# --------------------------------------------------------------------------- #
# Seed-face recognition wiring (issue #49): the gate's second argument
# --------------------------------------------------------------------------- #


def test_the_gate_is_given_each_chunks_own_reference_photo(tmp_path):
    """Issue #49: ``build_seed_face_gate``'s returned callable takes an
    optional second argument -- the active cast member's reference photo --
    so it can tell a big, confident face from *her* big, confident face.
    ``ContinuityWorkflowProvider`` is the only thing that knows both the
    extracted seed frame AND which chunk's ``ExpandedPrompt`` (and therefore
    which photo) is active, so it is the one that has to thread it through.

    Critically, the photo passed for a chained chunk must be THAT chunk's own
    ``image_ref`` -- the active cast member for the shot the seed frame is
    about to become the first frame of -- never the predecessor's."""
    calls: list[tuple[Path, Path | None]] = []

    def recording_gate(frame_path: Path, reference_photo: Path | None = None) -> bool:
        calls.append((frame_path, reference_photo))
        return True

    state = _state_with_rendered(tmp_path, 0, 1)
    provider = _provider(tmp_path, seed_face_gate=recording_gate)

    provider(1, state)
    provider(2, state)

    assert [reference_photo for _frame_path, reference_photo in calls] == [
        Path("cast/ref_1.png"),
        Path("cast/ref_2.png"),
    ]


def test_a_gate_with_no_reference_photo_configured_still_gets_called(tmp_path):
    """A gate built without recognition support (issue #47's original shape,
    or issue #49's off-by-default wiring in ``cli._resolve_seed_face_gate``)
    still accepts a second argument -- this provider always passes one -- and
    the decision to actually use it lives in how the gate itself was built,
    not in whether this module threads the photo down."""

    def detection_only_gate(_frame_path: Path, _reference_photo: Path | None = None) -> bool:
        return False

    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path, seed_face_gate=detection_only_gate)

    workflow = provider(1, state)

    assert _is_base(workflow), "a gate that refuses must still fall back cleanly"


# --------------------------------------------------------------------------- #
# Provenance: what each chunk was actually chained to
# --------------------------------------------------------------------------- #


def test_chain_sources_records_the_predecessor_for_a_chained_chunk(tmp_path):
    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path)

    provider(1, state)

    assert provider.chain_sources == {1: 0}
    assert provider.chain_source(1) == 0


def test_chain_sources_records_none_for_an_unchained_chunk(tmp_path):
    state = _state_with_rendered(tmp_path, 0, 1)
    provider = _provider(tmp_path, reanchor_interval=2)

    provider(2, state)

    assert provider.chain_sources == {2: None}
    assert provider.chain_source(2) is None


def test_chain_sources_is_a_copy_and_cannot_be_mutated_from_outside(tmp_path):
    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path)
    provider(1, state)

    snapshot = provider.chain_sources
    snapshot[1] = 99

    assert provider.chain_sources == {1: 0}


def test_chain_source_of_an_undecided_chunk_is_none(tmp_path):
    provider = _provider(tmp_path)
    assert provider.chain_source(3) is None


def test_a_degraded_chunk_records_no_chain_source_and_does_not_cascade(tmp_path):
    """A dead-lettered chunk breaks the chain for exactly one successor.

    Issue #28 asks the design to state what chunk N+1 does when N has no
    output. It falls back to the unchained base path -- and, critically,
    chunk N+2 chains again off N+1, so a single failure costs one boundary
    rather than every boundary after it.
    """
    video1 = tmp_path / "chunk_0001.mp4"
    video1.write_bytes(b"y")
    state = RunState(
        run_id="r1",
        results={
            0: ChunkResult(chunk_id=0, status=ChunkStatus.DEAD_LETTERED),
            1: _rendered(1, video1),
        },
    )
    provider = _provider(tmp_path)

    assert _is_base(provider(1, state))
    assert _is_chained(provider(2, state))
    assert provider.chain_sources == {1: None, 2: 1}


def test_an_extraction_failure_records_no_chain_source(tmp_path):
    class _Boom(_FakeRunner):
        def __call__(self, args):
            args = list(args)
            self.calls.append(args)
            return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"probe boom")

    state = _state_with_rendered(tmp_path, 0)
    provider = _provider(tmp_path, subprocess_runner=_Boom())

    assert _is_base(provider(1, state))
    assert provider.chain_sources == {1: None}


@pytest.mark.parametrize("chunk_id", [0, 1, 2, 3, 4, 5])
def test_planned_chain_source_matches_what_the_provider_actually_does(tmp_path, chunk_id):
    """The predicted value a resumed run compares against must agree with the
    decision the provider makes when nothing degrades -- otherwise every
    resume would re-render chunks that are in fact identical."""
    state = _state_with_rendered(tmp_path, 0, 1, 2, 3, 4)
    provider = _provider(tmp_path, reanchor_interval=3, chainable_chunk_ids={1, 2, 4, 5})

    provider(chunk_id, state)

    assert provider.chain_source(chunk_id) == continuity.planned_chain_source(
        chunk_id,
        continuity_enabled=True,
        reanchor_interval=3,
        chainable_chunk_ids={1, 2, 4, 5},
    )


# --------------------------------------------------------------------------- #
# End to end, offline: a whole run through the real stager/client/runner
# --------------------------------------------------------------------------- #


class _SequencedWSFactory:
    def __init__(self, sequences: list[list[Any]]) -> None:
        self._sequences = deque(sequences)
        self.connections: list[ScriptedWebSocket] = []

    def __call__(self, url: str, **kwargs: Any) -> ScriptedWebSocket:
        if not self._sequences:
            raise AssertionError("more connections than scripted sequences")
        ws = ScriptedWebSocket(self._sequences.popleft())
        ws.connect(url, **kwargs)
        self.connections.append(ws)
        return ws


def _abundant_disk(_path: str):
    class _Usage:
        free = 500 * 1024**3

    return _Usage()


def _run_chained_song(tmp_path: Path, *, chunk_count: int, **provider_kwargs: Any):
    session = FakeComfyUISession()
    session.enable_auto_history()

    image = tmp_path / "dianne_ref.png"
    image.write_bytes(make_fake_png_bytes(864, 480))

    stager = ComfyUIAssetStager(base_url=session.base_url, session=session)
    prompts: dict[int, ExpandedPrompt] = {}
    assets: dict[int, StagedAssets] = {}
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
        prompts[chunk_id] = ExpandedPrompt(
            chunk_id=chunk_id,
            prompt=f"expanded prompt {chunk_id}",
            image_ref=image,
            characters=("Dianne",),
        )
        assets[chunk_id] = stager.stage_chunk(prompts[chunk_id], chunk)

    base_template = make_workflow_baseline()
    i2v_template = make_workflow_i2v()
    base_before = copy.deepcopy(base_template)
    i2v_before = copy.deepcopy(i2v_template)

    provider = continuity.ContinuityWorkflowProvider(
        base_template=base_template,
        i2v_template=i2v_template,
        chunk_prompts=prompts,
        chunk_assets=assets,
        asset_stager=stager,
        frames_dir=tmp_path / "frames",
        continuity_enabled=True,
        subprocess_runner=_FakeRunner(),
        expected_length_frames=PROBED_FRAME_COUNT,
        **provider_kwargs,
    )

    ws_factory = _SequencedWSFactory(
        [build_success_sequence(f"prompt-{n:04d}") for n in range(1, chunk_count + 1)]
    )
    client = ComfyUIExecutionClient(
        base_url=session.base_url,
        session=session,
        ws_factory=ws_factory,
        client_id="issue28-client",
    )
    runner = ResilientRunner(
        client,
        run_state_file=tmp_path / "run_state.json",
        max_render_attempts=1,
        retry_backoff_seconds=0.0,
        min_free_disk_gb=1.0,
        run_id="issue28-run",
        sleeper=lambda _d: None,
        disk_usage=_abundant_disk,
    )
    run_state = runner.render_run(
        list(range(chunk_count)), provider, tmp_path / "out", resume=False
    )
    assert base_template == base_before, "base template mutated in place"
    assert i2v_template == i2v_before, "i2v template mutated in place"
    return session, provider, run_state


def test_end_to_end_reanchored_run_alternates_between_the_two_templates(tmp_path):
    session, provider, run_state = _run_chained_song(tmp_path, chunk_count=4, reanchor_interval=2)

    assert all(r.succeeded for r in run_state.results.values())
    submitted = [s["workflow"] for s in session.submitted_prompts if not s.get("rejected")]
    assert [_is_chained(w) for w in submitted] == [False, True, False, True]
    assert provider.chain_sources == {0: None, 1: 0, 2: None, 3: 2}


def test_end_to_end_chained_chunk_wires_the_seed_frame_into_first_frame(tmp_path):
    session, _provider_obj, _state = _run_chained_song(tmp_path, chunk_count=2)

    submitted = [s["workflow"] for s in session.submitted_prompts if not s.get("rejected")]
    chained = submitted[1]
    seed_id, seed_node = find_titled_node(
        chained, CLASS_TYPE_IMAGE_LOAD, continuity.DEFAULT_SEED_FRAME_TITLE
    )
    cast_id, cast_node = find_titled_node(
        chained, CLASS_TYPE_IMAGE_LOAD, continuity.DEFAULT_CAST_IMAGE_TITLE
    )
    _, h3_node = next(
        (nid, n)
        for nid, n in chained.items()
        if n["class_type"] == CLASS_TYPE_H3_IMAGE_TO_VIDEO
    )

    # first_frame is continuity: the previous chunk's last frame. Nothing is
    # wired to last_frame (issue #44) -- see
    # test_a_chained_chunk_pins_no_final_frame below for why.
    assert h3_node["inputs"]["first_frame"] == [seed_id, 0]
    assert "last_frame" not in h3_node["inputs"]
    assert cast_id
    assert seed_node["inputs"]["image"] != cast_node["inputs"]["image"]

    uploaded = [u.server_filename for u in session.uploads]
    assert seed_node["inputs"]["image"] in uploaded


def test_the_committed_fl2va_template_chains_without_modification(tmp_path):
    """The chained path must work against the *real* ``workflow_i2v_api.json``,
    not only the fixture mirroring it.

    That file was validated by an actual render on doris (issue #18), so it is
    the thing that has to still work; a fixture agreeing with itself proves
    nothing about the template the run will actually load.
    """
    repo_root = Path(__file__).resolve().parents[1]
    provider = continuity.ContinuityWorkflowProvider(
        base_template=load_workflow_template(repo_root / "workflow_api.json"),
        i2v_template=load_workflow_template(repo_root / "workflow_i2v_api.json"),
        chunk_prompts={0: _prompt(0), 1: _prompt(1)},
        chunk_assets={0: _assets(0), 1: _assets(1)},
        asset_stager=_FakeStager(),
        frames_dir=tmp_path / "frames",
        continuity_enabled=True,
        subprocess_runner=_FakeRunner(),
        chainable_chunk_ids={1},
    )

    workflow = provider(1, _state_with_rendered(tmp_path, 0))

    seed_id, seed_node = find_titled_node(
        workflow, CLASS_TYPE_IMAGE_LOAD, continuity.DEFAULT_SEED_FRAME_TITLE
    )
    cast_id, _cast_node = find_titled_node(
        workflow, CLASS_TYPE_IMAGE_LOAD, continuity.DEFAULT_CAST_IMAGE_TITLE
    )
    _, h3_node = next(
        (nid, n)
        for nid, n in workflow.items()
        if n["class_type"] == CLASS_TYPE_H3_IMAGE_TO_VIDEO
    )
    assert h3_node["inputs"]["first_frame"] == [seed_id, 0]
    assert "last_frame" not in h3_node["inputs"]
    assert cast_id
    assert seed_node["inputs"]["image"] == "server_seed_0000_into_0001.png"
    assert provider.chain_source(1) == 0


def test_end_to_end_chained_workflow_has_no_audio_conditioning_input(tmp_path):
    """The fl2va node takes no ``audio_vae``/``ref_audios`` -- this is why a
    chained chunk cannot lip-sync, and why chaining has to be selectable."""
    session, _provider_obj, _state = _run_chained_song(tmp_path, chunk_count=2)

    submitted = [s["workflow"] for s in session.submitted_prompts if not s.get("rejected")]
    _, h3_node = next(
        (nid, n)
        for nid, n in submitted[1].items()
        if n["class_type"] == CLASS_TYPE_H3_IMAGE_TO_VIDEO
    )
    assert "audio_vae" not in h3_node["inputs"]
    assert not any(key.startswith("ref_audios") for key in h3_node["inputs"])


# --------------------------------------------------------------------------- #
# Issue #44: nothing may dictate a chained chunk's final frame
# --------------------------------------------------------------------------- #


def test_a_chained_chunk_pins_no_final_frame(tmp_path):
    """``last_frame`` was wired to the cast photo in the belief that it
    anchored *identity*. ComfyUI's own node says otherwise:
    ``MiniMaxH3ImageToVideo`` resolves it to a keyframe at
    ``frame_count - 1`` and re-injects it at every sampling step, so it does
    not influence the shot -- it *dictates its last frame*.

    On lucky-ones v7 that meant every chained chunk ended on a cover-cropped
    portrait of the lead: a face flashing at each cut, shots dragged into a
    frontal smiling close-up for their whole length, and a performer spun
    round to camera to hit the mandated pose. Worse, the next chained chunk is
    seeded from that final frame, so it *began* on the portrait too, and
    consecutive chained shots degenerated into portrait-to-portrait cuts.

    Identity on this path comes from the seed frame; accumulated drift is
    answered by re-anchoring through the reference template, where the cast
    photo is a genuine reference rather than a mandated frame.
    """
    workflow = load_workflow_template(Path("workflow_i2v_api.json"))
    _, h3_node = next(
        (nid, n)
        for nid, n in workflow.items()
        if n["class_type"] == CLASS_TYPE_H3_IMAGE_TO_VIDEO
    )

    assert "first_frame" in h3_node["inputs"], "the chained path still needs its seed frame"
    assert "last_frame" not in h3_node["inputs"], (
        "the committed I2V template pins a final frame again -- every chained chunk will end "
        "on whatever image that is (issue #44)"
    )


def test_the_cast_reference_node_survives_for_the_orchestrator_to_inject_into():
    """The cast ``LoadImage`` stays in the graph, titled as before: the
    mutator locates it by title on every I2V mutation, and removing it would
    make that lookup fall through to the seed-frame node and overwrite the
    predecessor's frame with a portrait -- the same defect, one layer down.
    With nothing consuming its output, ComfyUI simply never executes it."""
    workflow = load_workflow_template(Path("workflow_i2v_api.json"))

    node_id, _ = find_titled_node(
        workflow, CLASS_TYPE_IMAGE_LOAD, continuity.DEFAULT_CAST_IMAGE_TITLE
    )
    consumers = [
        nid
        for nid, node in workflow.items()
        for value in node.get("inputs", {}).values()
        if isinstance(value, list) and value and value[0] == node_id
    ]
    assert consumers == [], f"the cast photo is wired into {consumers} again (issue #44)"


def test_a_configured_lora_reaches_both_rendered_graphs(tmp_path):
    """Issue #62 end to end: the adapter has to be in the graph that is
    actually POSTed, on both paths. A pin that reached only the base path
    would give a video rendered by two different models -- the same failure
    mode `text_encoder` (#39) documents for the encoder."""
    from music_video_maker.workflow_graph import CLASS_TYPE_LORA_LOADER, find_one_node

    provider = _provider(
        tmp_path,
        continuity_enabled=True,
        lora="realism.safetensors",
        lora_strength=0.8,
    )
    state = _state_with_rendered(tmp_path, 0)

    for chunk_id, path in ((0, "base"), (1, "chained")):
        graph = provider(chunk_id, state)
        _, node = find_one_node(graph, CLASS_TYPE_LORA_LOADER)
        assert node["inputs"]["lora_name"] == "realism.safetensors", path
        assert node["inputs"]["strength_model"] == 0.8, path
