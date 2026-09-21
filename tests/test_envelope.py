"""The proven render envelope and the measured-ceiling provenance (#24, #98).

Two things are pinned here that a reading of the code alone would not settle:

* **Dominance is per axis.** A point with a *larger latent volume* at a
  shorter duration must not cover a longer chunk at a smaller frame. The
  committed table contains exactly that pair, so the bug is one wrong
  comparison away and the test below is what stops it.
* **"Deathless" v13 must still run.** 192 frames at 864x480 is what actually
  rendered, and a gate that refuses it would have refused the render that
  proved it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest

from music_video_maker import envelope
from music_video_maker.contracts import ChunkFingerprint, ChunkResult, ChunkStatus, RunState
from music_video_maker.envelope import (
    CALIBRATED_CEILING,
    DORIS_4090_PROVEN,
    EnvelopePoint,
    UnprovenEnvelopeError,
    check_render_envelope,
    covering_point,
    largest_proven_frames,
    largest_rendered_frame_count,
    measured_ceiling,
    unproven_chunks,
)
from music_video_maker.resilience import dump_run_state

DORIS = "RTX 4090 24GB (doris)"


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: int
    frame_count: int | None


# --------------------------------------------------------------------------- #
# Dominance
# --------------------------------------------------------------------------- #


def test_a_point_covers_only_what_it_is_at_least_as_large_as_on_every_axis():
    point = EnvelopePoint(frames=141, width=1344, height=768, source="test")
    assert point.covers(141, 1344, 768)
    assert point.covers(124, 864, 480)
    assert not point.covers(158, 1344, 768)  # longer
    assert not point.covers(141, 1440, 768)  # wider
    assert not point.covers(141, 1344, 800)  # taller


def test_a_bigger_latent_volume_at_a_shorter_duration_does_not_prove_a_longer_chunk():
    """The whole reason :meth:`EnvelopePoint.covers` is not a volume comparison.

    141 frames at 1344x768 is 145.5 Mpx of latent against 192 frames at
    864x480's 79.6 Mpx -- so a volume rule would call the 192-frame point
    already proven and this project's one unmeasured axis (temporal VAE decode
    memory, which scales non-linearly with *frame count*) would be silently
    interpolated across.
    """
    big_frame = EnvelopePoint(frames=141, width=1344, height=768, source="test")
    assert big_frame.latent_megapixels > 192 * 864 * 480 / 1_000_000
    assert covering_point((big_frame,), 192, 864, 480) is None


def test_deathless_v13s_own_frame_count_and_resolution_are_inside_the_committed_table():
    """192 frames at 864x480 rendered 15 times; a gate refusing it is wrong."""
    assert covering_point(DORIS_4090_PROVEN, 192, 864, 480) is not None
    assert covering_point(DORIS_4090_PROVEN, 124, 864, 480) is not None


def test_the_twelve_second_lever_is_outside_the_committed_table():
    """277 frames is what ``max_chunk_seconds = 12.0`` actually produces on
    H3's 5+17k grid, and it is the thing issue #98 exists to measure."""
    assert covering_point(DORIS_4090_PROVEN, 277, 864, 480) is None


def test_largest_proven_frames_reports_the_table_maximum_and_none_when_empty():
    assert largest_proven_frames(DORIS_4090_PROVEN) == 192
    assert largest_proven_frames(()) is None


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_an_unproven_chunk_is_refused_by_default():
    with pytest.raises(UnprovenEnvelopeError) as excinfo:
        check_render_envelope(
            [FakeChunk(0, 124), FakeChunk(1, 277)],
            hardware_name=DORIS,
            width=864,
            height=480,
        )
    assert "chunk 1 at 277 frames" in str(excinfo.value)


def test_acknowledging_it_proceeds_and_returns_the_misses(caplog):
    with caplog.at_level(logging.WARNING, logger="music_video_maker.envelope"):
        misses = check_render_envelope(
            [FakeChunk(0, 277)],
            hardware_name=DORIS,
            width=864,
            height=480,
            acknowledged=True,
        )
    assert [m.chunk_id for m in misses] == [0]
    assert "acknowledge_unproven_envelope is set" in caplog.text


def test_a_run_inside_the_envelope_passes_silently():
    assert (
        check_render_envelope(
            [FakeChunk(i, 192) for i in range(80)],
            hardware_name=DORIS,
            width=864,
            height=480,
        )
        == ()
    )


def test_an_unrecorded_hardware_profile_gets_no_gate_at_all(caplog):
    """Unmeasured is not forbidden. Refusing a stranger's card on the strength
    of measurements taken on doris would be inventing a measurement."""
    with caplog.at_level(logging.INFO, logger="music_video_maker.envelope"):
        assert (
            check_render_envelope(
                [FakeChunk(0, 362)],
                hardware_name="somebody else's H100",
                width=1920,
                height=1080,
            )
            == ()
        )
    assert "no proven frame-count/resolution measurements are recorded" in caplog.text


def test_an_unresolvable_resolution_is_not_judged(caplog):
    with caplog.at_level(logging.WARNING, logger="music_video_maker.envelope"):
        assert (
            check_render_envelope(
                [FakeChunk(0, 362)], hardware_name=DORIS, width=None, height=480
            )
            == ()
        )
    assert "could not be resolved" in caplog.text


def test_a_chunk_with_no_frame_count_is_skipped_rather_than_flagged():
    assert (
        unproven_chunks(
            [FakeChunk(0, None)], width=864, height=480, points=DORIS_4090_PROVEN
        )
        == ()
    )


def test_the_refusal_names_at_most_five_chunks_and_counts_the_rest():
    with pytest.raises(UnprovenEnvelopeError) as excinfo:
        check_render_envelope(
            [FakeChunk(i, 277) for i in range(9)],
            hardware_name=DORIS,
            width=864,
            height=480,
        )
    assert "(4 more)" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Evidence from a run_state.json (issue #98's "make the number evidence")
# --------------------------------------------------------------------------- #


def _write_run_state(path: Path, entries: dict[int, tuple[ChunkStatus, int | None]]) -> None:
    state = RunState(run_id="r1")
    for chunk_id, (status, frames) in entries.items():
        state.results[chunk_id] = ChunkResult(
            chunk_id=chunk_id,
            status=status,
            video_file=None,
            prompt_id=None,
            fingerprint=ChunkFingerprint(start=0.0, end=1.0, frame_count=frames),
        )
    path.write_text(json.dumps(dump_run_state(state)), encoding="utf-8")


def test_the_ceiling_comes_from_what_a_previous_run_actually_rendered(tmp_path):
    path = tmp_path / "run_state.json"
    _write_run_state(
        path,
        {
            0: (ChunkStatus.RENDERED, 192),
            1: (ChunkStatus.CACHED, 158),
            2: (ChunkStatus.RENDERED, 124),
        },
    )
    ceiling = measured_ceiling(path)
    assert ceiling.frames == 192
    assert "chunk 0" in ceiling.provenance
    assert str(path) in ceiling.provenance


def test_a_dead_lettered_chunk_is_not_evidence_that_its_frame_count_rendered(tmp_path):
    path = tmp_path / "run_state.json"
    _write_run_state(
        path, {0: (ChunkStatus.RENDERED, 158), 1: (ChunkStatus.DEAD_LETTERED, 362)}
    )
    assert largest_rendered_frame_count(path).frames == 158


def test_a_run_state_that_reaches_lower_does_not_lower_the_ceiling(tmp_path):
    """One run is not the history of the card."""
    path = tmp_path / "run_state.json"
    _write_run_state(path, {0: (ChunkStatus.RENDERED, 124)})
    assert measured_ceiling(path) is CALIBRATED_CEILING


@pytest.mark.parametrize("supplied", [None, "missing.json"])
def test_no_readable_run_state_falls_back_to_the_calibrated_constant(tmp_path, supplied):
    target = None if supplied is None else tmp_path / supplied
    assert largest_rendered_frame_count(target) is None
    assert measured_ceiling(target) is CALIBRATED_CEILING


def test_an_unreadable_run_state_is_not_evidence_of_absence(tmp_path, caplog):
    path = tmp_path / "run_state.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="music_video_maker.envelope"):
        assert largest_rendered_frame_count(path) is None
    assert "not measured" in caplog.text


def test_a_run_state_with_no_frame_counts_yields_nothing(tmp_path):
    path = tmp_path / "run_state.json"
    _write_run_state(path, {0: (ChunkStatus.RENDERED, None)})
    assert largest_rendered_frame_count(path) is None


def test_the_calibrated_ceiling_never_claims_to_be_a_live_reading():
    """Issue #98's actual defect was a *sentence*, not a number."""
    assert "calibrated constant" in CALIBRATED_CEILING.provenance
    assert CALIBRATED_CEILING.frames == 141
    assert "141 frames" in CALIBRATED_CEILING.describe()


def test_every_committed_point_records_where_it_was_measured():
    for point in envelope.DORIS_4090_PROVEN:
        assert point.source.strip(), point
        assert point.frames > 0 and point.width > 0 and point.height > 0
