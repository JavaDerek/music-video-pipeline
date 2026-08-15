"""Foundation smoke tests (issue #1): packaging, logging, contracts, CLI wiring."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import music_video_maker
from music_video_maker import cli, contracts
from music_video_maker.logging_setup import configure_logging


def test_package_exposes_version():
    assert music_video_maker.__version__


def test_configure_logging_writes_to_stderr(capsys):
    configure_logging(logging.INFO, force=True)
    logging.getLogger("mvm.test").info("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert captured.out == ""


def test_cli_parses_args_and_reports_a_missing_config_file(caplog):
    """The orchestrator itself (issue #14, Wave 4) is covered end-to-end in
    tests/test_cli.py; this just locks in that the console-script entrypoint
    still resolves and surfaces a config error as a nonzero exit."""
    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(["--config", "run.toml"])
    assert exit_code == cli.EXIT_ERROR
    assert "Failed to load run config" in caplog.text


def test_cli_requires_config():
    with pytest.raises(SystemExit):
        cli.main([])


# --- contracts ------------------------------------------------------------- #


def test_aligned_segment_duration():
    seg = contracts.AlignedSegment(index=0, text="the lucky ones", start=1.5, end=6.0)
    assert seg.duration == pytest.approx(4.5)


def test_alignment_result_voiced_duration():
    result = contracts.AlignmentResult(
        segments=(
            contracts.AlignedSegment(index=0, text="a", start=0.0, end=2.0),
            contracts.AlignedSegment(index=1, text="b", start=5.0, end=8.0),
        ),
        track_duration=60.0,
    )
    assert result.voiced_duration == pytest.approx(5.0)


def test_chunk_result_succeeded_covers_cached():
    rendered = contracts.ChunkResult(0, contracts.ChunkStatus.RENDERED, Path("a.mp4"))
    cached = contracts.ChunkResult(1, contracts.ChunkStatus.CACHED, Path("b.mp4"))
    dead = contracts.ChunkResult(2, contracts.ChunkStatus.DEAD_LETTERED)
    assert rendered.succeeded and cached.succeeded
    assert not dead.succeeded


def test_run_state_reports_dead_lettered_in_order():
    state = contracts.RunState(run_id="r1")
    state.results[2] = contracts.ChunkResult(2, contracts.ChunkStatus.DEAD_LETTERED)
    state.results[0] = contracts.ChunkResult(0, contracts.ChunkStatus.RENDERED)
    state.results[1] = contracts.ChunkResult(1, contracts.ChunkStatus.DEAD_LETTERED)
    assert state.dead_lettered == (1, 2)


def test_contracts_are_frozen():
    seg = contracts.AlignedSegment(index=0, text="x", start=0.0, end=1.0)
    with pytest.raises(AttributeError):
        seg.text = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ChunkFingerprint -- what a cached chunk must prove about itself (issue #34).
# --------------------------------------------------------------------------- #


def _chunk(chunk_id: int = 0, *, start: float = 4.0, end: float = 10.0, frames: int = 141):
    return contracts.AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id}.wav"),
        start=start,
        end=end,
        text="the lucky ones",
        characters=("Dianne",),
        frame_count=frames,
    )


def _prompt(chunk_id: int = 0, *, text: str = "a prompt", character: str = "Dianne"):
    return contracts.ExpandedPrompt(
        chunk_id=chunk_id,
        prompt=text,
        image_ref=Path("/cast/dianne.png"),
        characters=(character,) if character else (),
    )


def test_chunk_fingerprint_of_captures_span_frames_and_resolution():
    fp = contracts.ChunkFingerprint.of(
        _chunk(), _prompt(), render_width=864, render_height=480
    )
    assert (fp.start, fp.end) == (4.0, 10.0)
    assert fp.frame_count == 141
    assert (fp.render_width, fp.render_height) == (864, 480)
    assert fp.character == "Dianne"
    assert fp.image_ref == "/cast/dianne.png"


def test_chunk_fingerprint_of_hashes_the_prompt_rather_than_storing_it():
    fp = contracts.ChunkFingerprint.of(_chunk(), _prompt(text="a very long composed prompt"))
    assert fp.prompt_hash and "very long composed prompt" not in fp.prompt_hash
    assert fp.prompt_hash == contracts.ChunkFingerprint.hash_prompt("a very long composed prompt")


def test_chunk_fingerprint_identical_inputs_compare_equal():
    a = contracts.ChunkFingerprint.of(_chunk(), _prompt(), render_width=864, render_height=480)
    b = contracts.ChunkFingerprint.of(_chunk(), _prompt(), render_width=864, render_height=480)
    assert a == b
    assert a.timeline_differences(b) == ()
    assert a.content_differences(b) == ()


def test_chunk_fingerprint_a_moved_span_is_a_timeline_difference():
    before = contracts.ChunkFingerprint.of(_chunk(start=4.0, end=10.0), _prompt())
    after = contracts.ChunkFingerprint.of(_chunk(start=5.5, end=11.5), _prompt())
    assert after.timeline_differences(before) == ("start", "end")
    assert after.content_differences(before) == ()


def test_chunk_fingerprint_resolution_change_is_a_timeline_difference():
    """Not a desync but a corruption: Stage 5 concats with ``-c:v copy``, so
    clips of differing dimensions cannot be joined."""
    small = contracts.ChunkFingerprint.of(
        _chunk(), _prompt(), render_width=864, render_height=480
    )
    large = contracts.ChunkFingerprint.of(
        _chunk(), _prompt(), render_width=1344, render_height=768
    )
    assert large.timeline_differences(small) == ("render_width", "render_height")


def test_chunk_fingerprint_a_changed_prompt_is_a_content_difference_only():
    before = contracts.ChunkFingerprint.of(_chunk(), _prompt(text="wide shot"))
    after = contracts.ChunkFingerprint.of(_chunk(), _prompt(text="close up"))
    assert after.timeline_differences(before) == ()
    assert after.content_differences(before) == ("prompt_hash",)


def test_chunk_fingerprint_a_recast_character_is_a_content_difference():
    before = contracts.ChunkFingerprint.of(_chunk(), _prompt(character="Dianne"))
    after = contracts.ChunkFingerprint.of(_chunk(), _prompt(character="Rex"))
    assert after.content_differences(before) == ("character",)


def test_chunk_fingerprint_ignores_sub_millisecond_float_jitter():
    """Chunk boundaries are floats recomputed every run. A shift far below one
    frame (41.7 ms at 24 fps) is not a moved chunk and must not re-render it."""
    before = contracts.ChunkFingerprint.of(_chunk(start=4.0, end=10.0), _prompt())
    after = contracts.ChunkFingerprint.of(
        _chunk(start=4.0 + 1e-9, end=10.0 - 1e-9), _prompt()
    )
    assert after == before
    assert after.timeline_differences(before) == ()


# --------------------------------------------------------------------------- #
# The noise seed in the fingerprint (issue #38).
#
# The seed is the last input that determines the pixels and was the only one
# the fingerprint could not see. It belongs to the CONTENT tier, not the
# TIMELINE tier: a re-seeded chunk still covers exactly the span, frame count
# and resolution it was rendered for, so it plays in the right place and stays
# in sync -- it is simply a different take of that moment. That is precisely
# "right place, wrong content", and it must stay escapable, because bumping a
# seed to re-roll one bad chunk must not force 39 hours of GPU time.
# --------------------------------------------------------------------------- #


def test_noise_seed_is_a_content_field_never_a_timeline_field():
    assert "noise_seed" in contracts.ChunkFingerprint.CONTENT_FIELDS
    assert "noise_seed" not in contracts.ChunkFingerprint.TIMELINE_FIELDS


def test_chunk_fingerprint_of_records_the_noise_seed_it_was_rendered_with():
    fp = contracts.ChunkFingerprint.of(_chunk(), _prompt(), noise_seed=1234567)
    assert fp.noise_seed == 1234567


def test_chunk_fingerprint_a_changed_seed_is_a_content_difference_only():
    """Same span, same prompt, different take -- never a moved timeline."""
    before = contracts.ChunkFingerprint.of(_chunk(), _prompt(), noise_seed=0)
    after = contracts.ChunkFingerprint.of(_chunk(), _prompt(), noise_seed=42)
    assert after.timeline_differences(before) == ()
    assert after.content_differences(before) == ("noise_seed",)


def test_chunk_fingerprint_seed_defaults_to_unrecorded_not_to_zero():
    """``None`` means 'nobody wrote the seed down', which is a different claim
    from 'it was seed 0'. Defaulting to 0 would fabricate the very evidence
    issue #38 exists to stop assuming."""
    assert contracts.ChunkFingerprint.of(_chunk(), _prompt()).noise_seed is None


def test_chunk_fingerprint_an_unrecorded_seed_never_matches_a_recorded_one():
    """The compat case, at the contract level: a chunk from before seeds were
    written down cannot prove it used seed 0, so it must not compare equal to
    one that did."""
    stored = contracts.ChunkFingerprint.of(_chunk(), _prompt())
    expected = contracts.ChunkFingerprint.of(_chunk(), _prompt(), noise_seed=0)
    assert expected.content_differences(stored) == ("noise_seed",)
    assert expected.timeline_differences(stored) == ()


# --------------------------------------------------------------------------- #
# FrameGrid -- the H3 ``length`` quantization seam frozen for issue #20.
# Ground truth: docs/h3-node-schema.md (live /object_info dump).
# --------------------------------------------------------------------------- #


def test_frame_grid_matches_the_documented_h3_schema():
    grid = contracts.H3_FRAME_GRID
    assert (grid.fps, grid.base_frames, grid.step_frames) == (24, 5, 17)
    assert (grid.trained_min_frames, grid.trained_max_frames) == (124, 362)
    # Both trained bounds must themselves land on the grid, or clamping would
    # hand ComfyUI a value it rejects.
    assert grid.is_valid(grid.trained_min_frames)
    assert grid.is_valid(grid.trained_max_frames)


@pytest.mark.parametrize(
    "frames,valid",
    [(4, False), (5, True), (6, False), (22, True), (124, True), (125, False), (362, True)],
)
def test_frame_grid_is_valid_accepts_only_5_plus_17k(frames, valid):
    assert contracts.H3_FRAME_GRID.is_valid(frames) is valid


@pytest.mark.parametrize(
    "requested,expected",
    [(1, 5), (5, 5), (6, 22), (20, 22), (22, 22), (23, 39), (124, 124), (125, 141)],
)
def test_frame_grid_quantize_up_never_rounds_down(requested, expected):
    """Issue #12's '20-frame context decodes to 22', generalized."""
    assert contracts.H3_FRAME_GRID.quantize_up(requested) == expected


@pytest.mark.parametrize(
    "requested,expected",
    [
        (5, 5),
        (13, 5),  # 8 above 5, 9 below 22 -> nearer 5 (rounds DOWN)
        (14, 22),  # 9 above 5, 8 below 22 -> nearer 22 (rounds UP)
        (22, 22),
        (130, 124),
        (133, 141),
    ],
)
def test_frame_grid_quantize_nearest_rounds_both_ways(requested, expected):
    """Nearest, not up: snapping every chunk upward would systematically
    lengthen the assembled video against the pristine master track. The
    down-rounding case (13 -> 5) is what distinguishes this from
    :meth:`quantize_up`, which would give 22."""
    assert contracts.H3_FRAME_GRID.quantize_nearest(requested) == expected
    assert contracts.H3_FRAME_GRID.is_valid(contracts.H3_FRAME_GRID.quantize_nearest(requested))


@pytest.mark.parametrize(
    "frames,expected",
    [(5, 124), (123, 124), (124, 124), (200, 200), (362, 362), (400, 362)],
)
def test_frame_grid_clamps_into_the_trained_range(frames, expected):
    assert contracts.H3_FRAME_GRID.clamp_to_trained(frames) == expected


def test_frame_grid_seconds_and_frames_round_trip_at_the_trained_bounds():
    grid = contracts.H3_FRAME_GRID
    assert grid.frames_to_seconds(124) == pytest.approx(5.1667, abs=1e-4)
    assert grid.frames_to_seconds(362) == pytest.approx(15.0833, abs=1e-4)
    assert grid.seconds_to_frames(grid.frames_to_seconds(362)) == pytest.approx(362)


def test_hardware_profile_carries_the_h3_grid_by_default():
    profile = contracts.HardwareProfile(name="test", vram_gb=24.0)
    assert profile.frame_grid is contracts.H3_FRAME_GRID


def test_audio_chunk_frame_count_defaults_to_none():
    """None means 'predates quantization', not 'zero frames' -- a caller must
    be able to tell the difference (issue #20)."""
    chunk = contracts.AudioChunk(
        chunk_id=0, audio_file=Path("c.wav"), start=0.0, end=6.0, text="x"
    )
    assert chunk.frame_count is None


# --------------------------------------------------------------------------- #
# Degradation is not a config change (issues #28, #45, #47)
# --------------------------------------------------------------------------- #


def _plan(**kw):
    base = dict(
        start=0.0, end=5.0, chained_from=3,
        template_hash="i2v", fallback_template_hash="base",
    )
    base.update(kw)
    return contracts.ChunkFingerprint(**base)


def _stored(**kw):
    base = dict(start=0.0, end=5.0, chained_from=None, template_hash="base")
    base.update(kw)
    return contracts.ChunkFingerprint(**base)


def test_a_chunk_that_declined_to_chain_is_not_a_config_change():
    """The plan predicts a chain from config; the render can decline it (dead
    predecessor, unextractable frame, #47's face gate). Reading that as a
    mismatch re-renders the chunk on every resume forever, because the next
    run degrades identically -- the run never converges."""
    assert _plan().content_differences(_stored()) == ()
    assert _plan().conditioning_differences(_stored()) == ()


def test_the_exemption_does_not_excuse_an_edited_template():
    """The one substitution forgiven is i2v -> base. A stored graph matching
    neither of this run's templates is exactly what #45 exists to catch, and
    must still re-render even though the chain also degraded."""
    stranger = _stored(template_hash="some-other-graph")
    assert "template_hash" in _plan().conditioning_differences(stranger)


def test_the_exemption_is_one_directional():
    """Expecting no chain and finding one means the cached video was built from
    footage this run does not intend -- never forgiven."""
    plan = _plan(chained_from=None, template_hash="base")
    stored = _stored(chained_from=3, template_hash="i2v")
    assert "chained_from" in plan.content_differences(stored)


def test_a_chain_from_a_different_predecessor_is_still_a_mismatch():
    stored = _stored(chained_from=7, template_hash="i2v")
    assert "chained_from" in _plan().content_differences(stored)
