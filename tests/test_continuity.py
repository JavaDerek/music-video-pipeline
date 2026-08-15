"""Tests for issue #12: seed-and-feed I2V latent continuity across chunks.

Fully offline: no real ffmpeg/ffprobe binary, no network, no ComfyUI. The
subprocess seam is injected as a fake that records invocations and returns a
scripted ``subprocess.CompletedProcess``, mirroring the pattern in
``tests/test_assembly.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from music_video_maker import continuity
from music_video_maker.contracts import (
    ChunkResult,
    ChunkStatus,
    ExpandedPrompt,
    RunState,
    StagedAssets,
)
from music_video_maker.workflow_graph import (
    CLASS_TYPE_H3_IMAGE_TO_VIDEO,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    CLASS_TYPE_IMAGE_LOAD,
    find_one_node,
)
from tests.harness.factories import make_workflow_baseline, make_workflow_i2v

# --------------------------------------------------------------------------- #
# Local helpers (tests/harness is off-limits to add continuity-specific
# fakes to beyond the workflow factory already added there)
# --------------------------------------------------------------------------- #


def _prompt(chunk_id: int, text: str = "a prompt") -> ExpandedPrompt:
    return ExpandedPrompt(
        chunk_id=chunk_id,
        prompt=text,
        image_ref=Path(f"cast/ref_{chunk_id}.png"),
        characters=("Dianne",),
    )


def _assets(chunk_id: int) -> StagedAssets:
    return StagedAssets(
        chunk_id=chunk_id,
        image_filename=f"staged_image_{chunk_id:04d}.png",
        audio_filename=f"staged_audio_{chunk_id:04d}.wav",
    )


def _succeeded_result(chunk_id: int, video_file: Path) -> ChunkResult:
    return ChunkResult(chunk_id=chunk_id, status=ChunkStatus.RENDERED, video_file=video_file)


class _FakeAssetStager:
    """Minimal contracts.AssetStager implementation. Records uploads."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.uploaded: list[Path] = []

    def upload_image(self, path: Path) -> str:
        if self.fail:
            raise RuntimeError("simulated upload failure")
        self.uploaded.append(Path(path))
        return f"server_{Path(path).name}"

    def upload_audio(self, path: Path) -> str:
        raise NotImplementedError("not used by continuity")

    def uploaded_server_filename(self) -> str:
        assert self.uploaded, "no upload recorded"
        return f"server_{self.uploaded[-1].name}"


class _FakeRunner:
    """Scripted ffprobe/ffmpeg runner. Dispatches on argv[0]."""

    def __init__(
        self,
        *,
        frame_count: int = 124,
        probe_returncode: int = 0,
        extract_returncode: int = 0,
        probe_stdout: str | None = None,
    ):
        self.frame_count = frame_count
        self.probe_returncode = probe_returncode
        self.extract_returncode = extract_returncode
        self.probe_stdout = probe_stdout if probe_stdout is not None else str(frame_count)
        self.calls: list[list[str]] = []

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args,
                returncode=self.probe_returncode,
                stdout=self.probe_stdout,
                stderr="" if self.probe_returncode == 0 else "probe boom",
            )
        # write a placeholder file so callers that check existence succeed
        dest = Path(args[-1])
        if self.extract_returncode == 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake-png-bytes")
        return subprocess.CompletedProcess(
            args,
            returncode=self.extract_returncode,
            stdout="",
            stderr="" if self.extract_returncode == 0 else "extract boom",
        )


# --------------------------------------------------------------------------- #
# Frame-grid math
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0, 5),
        (1, 5),
        (5, 5),
        (6, 22),
        (20, 22),  # issue #12's own worked example
        (22, 22),
        (23, 39),
        (124, 124),
        (125, 141),
        (362, 362),
    ],
)
def test_quantize_length_frames_rounds_up_to_grid(requested, expected):
    assert continuity.quantize_length_frames(requested) == expected


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (5, True),
        (22, True),
        (124, True),
        (362, True),
        (4, False),
        (6, False),
        (21, False),
        (0, False),
    ],
)
def test_is_valid_length_frames(frames, expected):
    assert continuity.is_valid_length_frames(frames) is expected


def test_decode_group_pattern_sums_to_the_step_size():
    # The docstring's reconciliation is only true if this holds.
    assert sum(continuity.DECODE_GROUP_PATTERN) == continuity.LENGTH_STEP_FRAMES


def test_frames_to_seconds_uses_24fps_by_default():
    assert continuity.frames_to_seconds(124) == pytest.approx(124 / 24)


# --------------------------------------------------------------------------- #
# probe_frame_count / extract_last_frame
# --------------------------------------------------------------------------- #


def test_probe_frame_count_parses_ffprobe_stdout():
    runner = _FakeRunner(frame_count=124)
    assert continuity.probe_frame_count(Path("chunk_0.mp4"), runner) == 124
    assert runner.calls[0][0] == "ffprobe"


def test_probe_frame_count_nonzero_exit_raises_and_logs(caplog):
    runner = _FakeRunner(probe_returncode=1)
    with caplog.at_level("ERROR"), pytest.raises(continuity.FrameExtractionError):
        continuity.probe_frame_count(Path("chunk_0.mp4"), runner)
    assert any("ffprobe" in r.message for r in caplog.records)


def test_probe_frame_count_non_integer_output_raises_and_logs(caplog):
    runner = _FakeRunner(probe_stdout="not-a-number")
    with caplog.at_level("ERROR"), pytest.raises(continuity.FrameExtractionError):
        continuity.probe_frame_count(Path("chunk_0.mp4"), runner)
    assert any("non-integer" in r.message for r in caplog.records)


def test_extract_last_frame_extracts_actual_last_index(tmp_path):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"not a real video")
    dest = tmp_path / "seed.png"
    runner = _FakeRunner(frame_count=124)

    result = continuity.extract_last_frame(video, dest, runner=runner, expected_length_frames=124)

    assert result == dest
    assert dest.exists()
    ffmpeg_call = runner.calls[1]
    assert ffmpeg_call[0] == "ffmpeg"
    assert "select=eq(n\\,123)" in ffmpeg_call


def test_extract_last_frame_mismatch_logs_loud_warning_but_still_uses_actual_count(
    tmp_path, caplog
):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"not a real video")
    dest = tmp_path / "seed.png"
    # Requested/quantized length says 124, but the real file only has 100.
    runner = _FakeRunner(frame_count=100)

    with caplog.at_level("WARNING"):
        continuity.extract_last_frame(video, dest, runner=runner, expected_length_frames=124)

    assert any("MISMATCH" in r.message for r in caplog.records)
    ffmpeg_call = runner.calls[1]
    # Must extract index 99 (the ACTUAL last frame), never 123 (an off-by-N
    # guess derived from the expected/requested length).
    assert "select=eq(n\\,99)" in ffmpeg_call
    assert "select=eq(n\\,123)" not in ffmpeg_call


def test_extract_last_frame_ffprobe_failure_raises_before_any_ffmpeg_call(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    dest = tmp_path / "seed.png"
    runner = _FakeRunner(probe_returncode=1)

    with caplog.at_level("ERROR"), pytest.raises(continuity.FrameExtractionError):
        continuity.extract_last_frame(video, dest, runner=runner)
    assert len(runner.calls) == 1  # never reached the ffmpeg extraction call


def test_extract_last_frame_ffmpeg_failure_raises_and_logs(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    dest = tmp_path / "seed.png"
    runner = _FakeRunner(frame_count=124, extract_returncode=1)

    with caplog.at_level("ERROR"), pytest.raises(continuity.FrameExtractionError):
        continuity.extract_last_frame(video, dest, runner=runner)
    assert any("extraction failed" in r.message for r in caplog.records)


def test_extract_last_frame_zero_frames_raises_and_logs(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    dest = tmp_path / "seed.png"
    runner = _FakeRunner(frame_count=0)

    with caplog.at_level("ERROR"), pytest.raises(continuity.FrameExtractionError):
        continuity.extract_last_frame(video, dest, runner=runner)


# --------------------------------------------------------------------------- #
# ContinuityWorkflowProvider construction
# --------------------------------------------------------------------------- #


def test_provider_requires_i2v_template_when_continuity_enabled(tmp_path):
    with pytest.raises(ValueError, match="i2v_template"):
        continuity.ContinuityWorkflowProvider(
            base_template=make_workflow_baseline(),
            chunk_prompts={},
            chunk_assets={},
            asset_stager=_FakeAssetStager(),
            frames_dir=tmp_path,
            continuity_enabled=True,
            i2v_template=None,
        )


def _provider(
    tmp_path,
    *,
    continuity_enabled=True,
    runner=None,
    asset_stager=None,
    chunk_ids=(0, 1, 2, 3),
):
    prompts = {cid: _prompt(cid) for cid in chunk_ids}
    assets = {cid: _assets(cid) for cid in chunk_ids}
    return continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts=prompts,
        chunk_assets=assets,
        asset_stager=asset_stager or _FakeAssetStager(),
        frames_dir=tmp_path / "frames",
        continuity_enabled=continuity_enabled,
        subprocess_runner=runner or _FakeRunner(),
    )


# --------------------------------------------------------------------------- #
# ContinuityWorkflowProvider.__call__ -- toggle + fallback paths
# --------------------------------------------------------------------------- #


def test_provider_disabled_always_uses_base_template_even_with_a_successful_predecessor(tmp_path):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    provider = _provider(tmp_path, continuity_enabled=False)

    workflow = provider(1, state)

    _, node = find_one_node(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert node["inputs"]["prompt"] == _prompt(1).prompt


def test_provider_chunk_zero_has_no_predecessor_uses_base_template(tmp_path, caplog):
    state = RunState(run_id="r1", results={})
    provider = _provider(tmp_path)

    with caplog.at_level("INFO"):
        workflow = provider(0, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("no predecessor" in r.message for r in caplog.records)


def workflow_graph_has(workflow, class_type) -> bool:
    return any(n.get("class_type") == class_type for n in workflow.values())


def test_provider_predecessor_missing_from_results_uses_base_template(tmp_path, caplog):
    state = RunState(run_id="r1", results={})
    provider = _provider(tmp_path)

    with caplog.at_level("INFO"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("no recorded result" in r.message for r in caplog.records)


def test_provider_predecessor_dead_lettered_uses_base_template(tmp_path, caplog):
    dead = ChunkResult(chunk_id=0, status=ChunkStatus.DEAD_LETTERED)
    state = RunState(run_id="r1", results={0: dead})
    provider = _provider(tmp_path)

    with caplog.at_level("WARNING"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("did not succeed" in r.message for r in caplog.records)


def test_provider_predecessor_succeeded_but_no_video_file_uses_base_template(tmp_path, caplog):
    result = ChunkResult(chunk_id=0, status=ChunkStatus.RENDERED, video_file=None)
    state = RunState(run_id="r1", results={0: result})
    provider = _provider(tmp_path)

    with caplog.at_level("WARNING"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("no video_file" in r.message for r in caplog.records)


def test_provider_predecessor_video_missing_from_disk_uses_base_template(tmp_path, caplog):
    missing_video = tmp_path / "does_not_exist.mp4"
    state = RunState(run_id="r1", results={0: _succeeded_result(0, missing_video)})
    provider = _provider(tmp_path)

    with caplog.at_level("WARNING"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("missing from disk" in r.message for r in caplog.records)


def test_provider_extraction_failure_falls_back_to_base_template_and_logs(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    runner = _FakeRunner(probe_returncode=1)  # ffprobe fails
    provider = _provider(tmp_path, runner=runner)

    with caplog.at_level("ERROR"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("failed to extract/stage" in r.message for r in caplog.records)


def test_provider_upload_failure_falls_back_to_base_template_and_logs(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    provider = _provider(tmp_path, asset_stager=_FakeAssetStager(fail=True))

    with caplog.at_level("ERROR"):
        workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert any("failed to extract/stage" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# ContinuityWorkflowProvider.__call__ -- the happy bridged path
# --------------------------------------------------------------------------- #


def test_provider_bridges_successful_predecessor_onto_i2v_template(tmp_path):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    stager = _FakeAssetStager()
    provider = _provider(tmp_path, asset_stager=stager)

    workflow = provider(1, state)

    assert workflow_graph_has(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO)
    assert not workflow_graph_has(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)

    _, i2v_node = find_one_node(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO)
    assert i2v_node["inputs"]["prompt"] == _prompt(1).prompt

    _, seed_node = find_one_node(
        workflow, CLASS_TYPE_IMAGE_LOAD, title=continuity.DEFAULT_SEED_FRAME_TITLE
    )
    assert seed_node["inputs"]["image"] == stager.uploaded_server_filename()

    _, cast_node = find_one_node(
        workflow, CLASS_TYPE_IMAGE_LOAD, title=continuity.DEFAULT_CAST_IMAGE_TITLE
    )
    assert cast_node["inputs"]["image"] == _assets(1).image_filename


def test_provider_chunk_two_bridges_from_chunk_one_not_chunk_zero(tmp_path):
    # Two consecutive successful predecessors exist (0 and 1); chunk 2 must
    # bridge from chunk 1, never chunk 0 -- the index-space guard this issue
    # calls out explicitly.
    video0 = tmp_path / "chunk_0000.mp4"
    video0.write_bytes(b"x")
    video1 = tmp_path / "chunk_0001.mp4"
    video1.write_bytes(b"y")
    state = RunState(
        run_id="r1",
        results={0: _succeeded_result(0, video0), 1: _succeeded_result(1, video1)},
    )
    runner = _FakeRunner()
    provider = _provider(tmp_path, runner=runner)

    provider(2, state)

    # The extraction call's -i argument must reference chunk 1's video, not chunk 0's.
    ffmpeg_calls = [c for c in runner.calls if c[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 1
    assert str(video1) in ffmpeg_calls[0]
    assert str(video0) not in ffmpeg_calls[0]


# --------------------------------------------------------------------------- #
# Index-space guard: ContinuityIndexError
# --------------------------------------------------------------------------- #


def test_provider_raises_when_runstate_key_disagrees_with_chunkresult_chunk_id(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    # Corrupted RunState: stored under key 0, but the object claims chunk_id=99.
    corrupted = ChunkResult(chunk_id=99, status=ChunkStatus.RENDERED, video_file=video)
    state = RunState(run_id="r1", results={0: corrupted})
    provider = _provider(tmp_path)

    with (
        caplog.at_level("ERROR"),
        pytest.raises(continuity.ContinuityIndexError) as excinfo,
    ):
        provider(1, state)

    assert excinfo.value.chunk_id == 1
    assert excinfo.value.other_id == 0
    assert "RunState.results key does not match ChunkResult.chunk_id" in str(excinfo.value)


def test_provider_raises_when_chunk_prompts_registry_key_disagrees_with_prompt_chunk_id(tmp_path):
    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts={1: _prompt(chunk_id=99)},  # registry key 1, but ExpandedPrompt says 99
        chunk_assets={1: _assets(1)},
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path,
        continuity_enabled=False,
    )
    state = RunState(run_id="r1", results={})

    with pytest.raises(continuity.ContinuityIndexError) as excinfo:
        provider(1, state)

    assert excinfo.value.chunk_id == 1
    assert excinfo.value.other_id == 99


def test_provider_raises_when_chunk_assets_registry_key_disagrees_with_assets_chunk_id(tmp_path):
    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts={1: _prompt(1)},
        chunk_assets={1: _assets(99)},  # registry key 1, but StagedAssets says 99
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path,
        continuity_enabled=False,
    )
    state = RunState(run_id="r1", results={})

    with pytest.raises(continuity.ContinuityIndexError) as excinfo:
        provider(1, state)

    assert excinfo.value.chunk_id == 1
    assert excinfo.value.other_id == 99


def test_provider_missing_prompt_registration_raises_keyerror(tmp_path):
    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        chunk_prompts={},
        chunk_assets={1: _assets(1)},
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path,
        continuity_enabled=False,
    )
    state = RunState(run_id="r1", results={})

    with pytest.raises(KeyError):
        provider(1, state)


# --------------------------------------------------------------------------- #
# WorkflowProvider structural shape (cross-lane contract with #10)
# --------------------------------------------------------------------------- #


def test_provider_matches_workflow_provider_callable_shape(tmp_path):
    provider = _provider(tmp_path, continuity_enabled=False)
    state = RunState(run_id="r1", results={})
    # Must be callable as (chunk_id: int, state: RunState) -> Workflow.
    result = provider(0, state)
    assert isinstance(result, dict)


def test_expected_length_frames_is_threaded_through_to_extraction(tmp_path, caplog):
    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    runner = _FakeRunner(frame_count=100)  # mismatched vs expected 124
    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts={0: _prompt(0), 1: _prompt(1)},
        chunk_assets={0: _assets(0), 1: _assets(1)},
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path / "frames",
        continuity_enabled=True,
        subprocess_runner=runner,
        expected_length_frames=124,
    )

    with caplog.at_level("WARNING"):
        provider(1, state)

    assert any("MISMATCH" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Issue #39: the pinned text encoder reaches every render path
# --------------------------------------------------------------------------- #


def _clip_name(workflow) -> str:
    from music_video_maker.workflow_graph import read_text_encoder

    return read_text_encoder(workflow)


def test_pinned_text_encoder_reaches_both_render_paths(tmp_path: Path):
    """A chained chunk renders through the *other* authored template, which
    carries its own CLIPLoader. Pinning only the base path would give a
    video whose shots were encoded at two different precisions."""
    from music_video_maker.contracts import RunState

    video = tmp_path / "chunk_0000.mp4"
    video.write_bytes(b"x")
    state = RunState(run_id="r1", results={0: _succeeded_result(0, video)})
    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        i2v_template=make_workflow_i2v(),
        chunk_prompts={0: _prompt(0), 1: _prompt(1)},
        chunk_assets={0: _assets(0), 1: _assets(1)},
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path / "frames",
        continuity_enabled=True,
        subprocess_runner=_FakeRunner(),
        text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    )

    unchained = provider(0, RunState(run_id="r1", results={}))
    chained = provider(1, state)

    assert _clip_name(unchained) == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    assert _clip_name(chained) == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def test_unpinned_text_encoder_leaves_the_authored_templates_alone(tmp_path: Path):
    """Default behaviour is unchanged: the template names the encoder."""
    from music_video_maker.contracts import RunState
    from tests.harness.factories import WEIGHT_TEXT_ENCODER

    provider = continuity.ContinuityWorkflowProvider(
        base_template=make_workflow_baseline(),
        chunk_prompts={0: _prompt(0)},
        chunk_assets={0: _assets(0)},
        asset_stager=_FakeAssetStager(),
        frames_dir=tmp_path / "frames",
    )

    assert _clip_name(provider(0, RunState(run_id="r1", results={}))) == WEIGHT_TEXT_ENCODER
