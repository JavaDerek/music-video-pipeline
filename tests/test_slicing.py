"""Tests for Stage 2a temporal audio slicing (issue #4) and H3 frame-grid
quantization (issue #20).

WAV-only, stdlib-``wave``-synthesized fixtures via ``tests/harness/factories``
-- no ffmpeg required (pydub's wav path uses the stdlib ``wave`` module for
both read and write, never shelling out). Any genuinely ffmpeg-dependent test
must be marked ``@pytest.mark.integration``; there are none here.

A note on fixture reuse: several of ``tests/harness/factories``'s shared
alignment fixtures (``make_alignment_result_normal_song``,
``make_alignment_result_short_segments``) were built for the *pre-#20* 4-15s
window and deliberately include segments in the 4-5.17s band that is now
below H3's trained floor. Their behavior under the new, higher minimum is
still exercised below (merging is still correct, just merges more
aggressively), but tests that need a segment to land *cleanly* inside the
trained window without merging use small local alignments built directly
with ``make_aligned_segment``, since 5.17-15.08s is a narrower band than the
shared fixtures were designed around.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import pytest

import music_video_maker.slicing as slicing_module
from music_video_maker import hardware
from music_video_maker.contracts import AlignmentResult, AudioChunk, HardwareProfile
from music_video_maker.slicing import slice_audio
from tests.harness.factories import (
    make_aligned_segment,
    make_alignment_result_long_segment,
    make_alignment_result_normal_song,
    make_alignment_result_short_segments,
    make_alignment_result_with_gaps,
    write_silent_wav,
)

DEFAULT_HARDWARE = hardware.PROFILE_RTX_4090_24GB
"""The real doris profile: min/max already pinned exactly to H3's trained
frame range (issue #20), so using it here (rather than a bare
``HardwareProfile(...)`` with contracts' pre-#20 defaults) means these tests
exercise the window slice_audio actually ships with, with no clamp warnings
as noise. Tests that specifically want to exercise clamping build their own
out-of-range ``HardwareProfile`` -- see the "clamping" section below.
"""

GRID = DEFAULT_HARDWARE.frame_grid
_EPS = 1e-6


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _assert_timeline_sane(chunks: tuple[AudioChunk, ...]) -> None:
    """Chronological, non-overlapping, monotonically increasing chunk ids."""
    prev_end = None
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == i
        assert chunk.end > chunk.start
        if prev_end is not None:
            assert chunk.start >= prev_end - _EPS
        prev_end = chunk.end


def _assert_frame_grid_exact(chunks: tuple[AudioChunk, ...]) -> None:
    """Every chunk's frame_count is a valid H3 grid point in the trained
    range, and its duration matches that frame_count exactly (issue #20's
    core acceptance criterion: nothing left for a later stage to round)."""
    for chunk in chunks:
        assert chunk.frame_count is not None
        assert GRID.is_valid(chunk.frame_count)
        assert GRID.trained_min_frames <= chunk.frame_count <= GRID.trained_max_frames
        assert chunk.duration == pytest.approx(GRID.frames_to_seconds(chunk.frame_count), abs=1e-6)


# --------------------------------------------------------------------------- #
# Well-behaved input: one chunk per segment, no merge/pad/split needed
# --------------------------------------------------------------------------- #


def test_normal_song_produces_one_chunk_per_segment(tmp_path):
    # Under the raised #20 minimum, two of this fixture's four segments
    # (4.8s, 5.0s) are themselves below the 5.167s trained floor and get
    # padded -- so this now checks coverage/ordering rather than an exact
    # start/end match against the untouched segment timestamps.
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks_dir = tmp_path / "chunks"

    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, chunks_dir)

    assert len(chunks) == len(alignment.segments)
    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)
    for chunk, seg in zip(chunks, alignment.segments, strict=True):
        # Padding only ever extends forward from a segment's own start.
        assert chunk.start == pytest.approx(seg.start)
        # Content is never dropped: the chunk always covers at least the
        # original segment's span, though quantization may extend past it.
        assert chunk.end >= seg.end - _EPS
        assert chunk.source_segment_indices == (seg.index,)
        assert chunk.is_split_continuation is False
        assert chunk.character == seg.character
        assert chunk.audio_file.exists()
        assert chunk.audio_file.name == f"chunk_{chunk.chunk_id:03d}.wav"


def test_exported_wav_duration_matches_chunk_duration(tmp_path):
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    for chunk in chunks:
        exported_seconds = _wav_duration_seconds(chunk.audio_file)
        assert exported_seconds == pytest.approx(chunk.duration, abs=0.02)


# --------------------------------------------------------------------------- #
# Minimum-duration enforcement: padding and merging
# --------------------------------------------------------------------------- #


def test_short_segments_are_padded_or_merged_to_minimum(tmp_path):
    alignment = make_alignment_result_short_segments()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)
    # Every chunk should meet the (grid-exact, >= trained floor) minimum,
    # *except possibly the very last* if the track itself runs out of room.
    for chunk in chunks[:-1]:
        assert chunk.duration >= DEFAULT_HARDWARE.min_chunk_seconds - _EPS

    # No audio content should be lost or duplicated: every source segment
    # index from the original alignment must be accounted for exactly once.
    covered = sorted(i for chunk in chunks for i in chunk.source_segment_indices)
    assert covered == [seg.index for seg in alignment.segments]


def test_merge_combines_source_segment_indices_and_text(tmp_path):
    alignment = make_alignment_result_short_segments()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    merged = [c for c in chunks if len(c.source_segment_indices) > 1]
    assert merged, "expected at least one merge among the sub-5.17s segments"
    for chunk in merged:
        # text should reflect all merged segments' words, in order
        expected_words = " ".join(
            alignment.segments[i].text for i in chunk.source_segment_indices
        )
        assert chunk.text == expected_words


def test_minimum_enforcement_never_produces_overlap(tmp_path):
    alignment = make_alignment_result_short_segments()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")
    _assert_timeline_sane(chunks)


def test_padding_extends_into_instrumental_gap_not_next_vocal(tmp_path):
    # Three segments spaced so the new (higher) minimum still leaves two
    # distinct groups: [0, 6.5) needs no help; [10.0, 10.8) is short and the
    # only room to pad into is the 20s gap before the *next* group -- which
    # must never be pushed past that next group's own vocal start.
    segments = (
        make_aligned_segment(0, "steady on the first line", 0.0, 6.5, "Dianne"),
        make_aligned_segment(1, "oh", 10.0, 10.8, "Dianne"),
        make_aligned_segment(2, "the second verse begins here now", 30.0, 36.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=40.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)
    assert len(chunks) == 3
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.end <= later.start + _EPS
    # The short middle chunk's padding must never reach into segment 2's
    # true vocal start (30.0s) -- there's ample instrumental room (20s+) so
    # it should land comfortably within the gap.
    assert chunks[1].end < 30.0


# --------------------------------------------------------------------------- #
# Maximum-duration enforcement: splitting
# --------------------------------------------------------------------------- #


def test_long_segment_is_split_into_multiple_chunks(tmp_path):
    alignment = make_alignment_result_long_segment()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) > 1
    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)


def test_split_pieces_marked_continuation_except_first(tmp_path):
    alignment = make_alignment_result_long_segment()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert chunks[0].is_split_continuation is False
    for chunk in chunks[1:]:
        assert chunk.is_split_continuation is True


def test_split_pieces_share_source_segment_index_and_are_contiguous(tmp_path):
    alignment = make_alignment_result_long_segment()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    original = alignment.segments[0]
    for chunk in chunks:
        assert chunk.source_segment_indices == (original.index,)

    assert chunks[0].start == pytest.approx(original.start)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.end == pytest.approx(later.start)


def test_split_pieces_never_shrink_below_their_raw_share_and_cover_all_content(tmp_path):
    # issue #20: split pieces round UP (never nearest/down) because there is
    # no instrumental padding *within* a continuous vocal run to shrink
    # into. The whole original span must still be fully covered -- the last
    # piece's end may run slightly past the original segment's end (grid
    # rounding absorbed as trailing padding into the gap that follows), but
    # never short of it.
    alignment = make_alignment_result_long_segment()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    original = alignment.segments[0]
    assert chunks[-1].end >= original.end - _EPS
    assert chunks[-1].end <= alignment.track_duration + _EPS
    total = sum(c.duration for c in chunks)
    assert total >= original.duration - _EPS
    # The overshoot from grid-rounding every piece up is bounded: at most
    # one grid step (~0.708s) of slack per piece.
    assert total - original.duration < len(chunks) * (GRID.step_frames / GRID.fps) + _EPS


def test_each_individual_split_piece_never_shrinks_below_its_own_raw_share(tmp_path):
    # Stronger than the aggregate coverage check above: EVERY split piece,
    # not just the group as a whole, must round up from its own raw
    # geometric share (7.6s here) -- rounding an *interior* piece down would
    # silently reassign real audio to its neighbor with no record of it
    # (see _split_for_maximum's docstring for why quantize_up, not
    # quantize_nearest, is used for every split piece).
    segments = (
        make_aligned_segment(
            0, "a very long single vocal line spanning many words indeed yes", 0.0, 15.2, "Dianne"
        ),
    )
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 2
    raw_share = 15.2 / 2
    assert chunks[0].duration >= raw_share - _EPS
    assert chunks[0].frame_count == 192  # 8.0s -- rounds UP from 7.6s, not down to 175 (7.29s)


# --------------------------------------------------------------------------- #
# issue #70: interior split boundaries prefer the nearer segment edge over a
# purely geometric cut point, so a sung phrase isn't split mid-utterance.
#
# Before this fix, pass 2 (`_split_for_maximum`) divided an over-long group
# into geometrically equal pieces with no regard for where the group's own
# member segments started and ended. On "Deathless" that put 26 of 80 chunk
# boundaries strictly inside an aligned vocal segment -- cutting the phrase
# mid-word and rendering the two halves as independent shots, with no error
# anywhere: every chunk is individually valid, the timeline is contiguous,
# every fingerprint matches, and the defect exists only in the assembled
# video's lip-sync.
#
# The first three tests exercise `_split_for_maximum` directly (white-box,
# like the ChunkFrameMismatchError tests above) with hand-built `_Group`s so
# the exact percentage-through-the-segment a boundary lands at can be
# controlled precisely -- through the full `slice_audio` pipeline, grid
# quantization cascades (and pass 1's own merge geometry -- see the
# integration test at the end of this section) make hitting a specific
# percentage by construction impractical. A `_Group` built this way is still
# a fully legitimate input to `_split_for_maximum`, whose contract only cares
# about `start`/`end`/`members`, never how the group was assembled.
# --------------------------------------------------------------------------- #


def test_boundary_shallow_into_a_segment_snaps_to_its_start(tmp_path, caplog):
    """The 7:57 lag from the issue: a boundary landing just 15% into a
    segment snaps backward to that segment's start instead, landing the cut
    in the instrumental gap before it rather than mid-phrase.

    ``member1``'s start is deliberately grid-exact (226 frames = 9.41666s)
    so the snap lands with zero grid-rounding residue -- see
    ``_quantize_split_end``'s ceil-based rounding, which this sidesteps by
    construction so the "clean escape" case is unambiguous. The
    duration-window-refusal and floor-clamp-residue cases are covered by the
    dedicated tests below and by the end-to-end integration test.
    """
    member1_start = 226 / 24  # 9.41666...s, exactly grid frame 226
    member0 = make_aligned_segment(0, "everything before it", 0.0, 9.0, "Dianne")
    member1 = make_aligned_segment(
        1, "the war.", member1_start, member1_start + 35 / 9, "Dianne"
    )
    group = slicing_module._Group(members=(member0, member1), start=0.0, end=20.0)

    with caplog.at_level(logging.INFO):
        pieces = slicing_module._split_for_maximum(
            [group], DEFAULT_HARDWARE.min_chunk_seconds, DEFAULT_HARDWARE.max_chunk_seconds, GRID
        )

    assert len(pieces) == 2
    first, second = pieces
    # The default geometric split (10.0s, the group's midpoint) would have
    # landed 15% into member1; the snap instead lands exactly on its start.
    assert first.end == pytest.approx(member1_start, abs=1e-6)
    assert second.start == first.end
    assert "snapped to the start edge" in caplog.text
    assert "issue #70" in caplog.text
    # No mid-utterance warning: the snap actually escaped the segment.
    assert "cutting a sung phrase mid-utterance" not in caplog.text


def test_boundary_deep_into_a_segment_snaps_to_its_end(tmp_path, caplog):
    """The 92%-through fragment from the issue (chunk 38): a boundary
    landing 92% through a segment snaps forward to that segment's end
    instead, so the whole phrase stays in one chunk rather than handing the
    final 8% to the next one as an orphan fragment.

    ``member1``'s end is deliberately grid-exact (243 frames = 10.125s) for
    the same zero-residual reason as the shallow-cut test above.
    """
    member0 = make_aligned_segment(0, "everything before it", 0.0, 8.0, "Dianne")
    member1 = make_aligned_segment(
        1, "the parts they played in america's war.", 8.5625, 10.125, "Dianne"
    )
    group = slicing_module._Group(members=(member0, member1), start=0.0, end=20.0)

    with caplog.at_level(logging.INFO):
        pieces = slicing_module._split_for_maximum(
            [group], DEFAULT_HARDWARE.min_chunk_seconds, DEFAULT_HARDWARE.max_chunk_seconds, GRID
        )

    assert len(pieces) == 2
    first, second = pieces
    # The default geometric split (10.0s) would have landed 92% into
    # member1 (fragmenting the last 8%); the snap lands exactly on its end.
    assert first.end == pytest.approx(10.125, abs=1e-6)
    assert second.start == first.end
    assert "snapped to the end edge" in caplog.text
    assert "issue #70" in caplog.text
    assert "cutting a sung phrase mid-utterance" not in caplog.text


def test_snap_refused_when_it_would_violate_the_duration_window(tmp_path, caplog):
    """A snap that would satisfy the segment-edge preference but push the
    resulting chunk outside the duration window is refused -- the window is
    a harder constraint than the preference for a clean cut. The unsnapped,
    geometric boundary is used instead, and it still cuts mid-utterance here,
    so that gets logged at WARNING with the percentage through the segment.
    """
    member = make_aligned_segment(
        0, "the parts they played in america's war.", 2.0, 8.5, "Dianne"
    )
    group = slicing_module._Group(members=(member,), start=0.0, end=15.0)

    with caplog.at_level(logging.WARNING):
        pieces = slicing_module._split_for_maximum(
            [group], MAX8_HARDWARE.min_chunk_seconds, MAX8_HARDWARE.max_chunk_seconds, GRID
        )

    assert len(pieces) == 2
    first, second = pieces
    # The nearer edge (the segment's end, 8.5s) would quantize to ~8.708s,
    # outside the 5.167-8.0s window this profile enforces -- refused. The
    # unsnapped geometric split (7.5s raw) quantizes to exactly 8.0s instead,
    # which still lands inside the segment.
    assert first.end == pytest.approx(8.0, abs=1e-6)
    assert second.start == first.end
    assert "snapped to the" not in caplog.text
    assert "lands 92.3% through segment" in caplog.text
    assert "duration window" in caplog.text
    assert "issue #70" in caplog.text


def test_snap_refused_when_no_usable_segment_edge_exists(tmp_path, caplog):
    """A single continuous segment spanning the whole group has no interior
    edge to offer -- both candidate "edges" collapse onto the group's own
    start and end, which is not a real split point. The cut still lands
    inside the segment and is still logged, but no snap is even attempted.
    """
    segments = (
        make_aligned_segment(
            0,
            "we started this together side by side and we'll carry on despite the noise "
            "no matter how far we wander we will always find our way back home again",
            0.0,
            18.5,
            "Dianne",
        ),
    )
    group = slicing_module._Group(members=segments, start=0.0, end=18.5)

    with caplog.at_level(logging.WARNING):
        pieces = slicing_module._split_for_maximum(
            [group], DEFAULT_HARDWARE.min_chunk_seconds, DEFAULT_HARDWARE.max_chunk_seconds, GRID
        )

    assert len(pieces) == 2
    assert "snapped to the" not in caplog.text
    assert "lands 50.9% through segment" in caplog.text
    assert "issue #70" in caplog.text


def test_split_for_maximum_on_empty_groups_returns_no_pieces():
    """The zero-segment / wholly-instrumental case (issue #70's fourth named
    scenario): with no groups at all -- what `_merge_for_minimum` returns for
    a zero-segment alignment -- there is nothing to split and nothing to
    snap. `cover_instrumentals` is what tiles the resulting empty piece list
    across the whole track; see the "wholly unvoiced track" test in the
    instrumental-coverage section further down, which confirms that still
    works end-to-end after this change."""
    pieces = slicing_module._split_for_maximum(
        [], DEFAULT_HARDWARE.min_chunk_seconds, DEFAULT_HARDWARE.max_chunk_seconds, GRID
    )
    assert pieces == []


def test_snap_refused_when_edge_is_closer_to_cursor_than_the_trained_floor(tmp_path, caplog):
    """Regression test for a real bug found on "Deathless": the acceptance
    check compared the *clamped* duration against ``[eff_min, eff_max]``,
    which is vacuous whenever ``eff_min`` sits at (or near) H3's trained
    floor -- as it does on the real, production ``DEFAULT_HARDWARE``
    profile -- because ``clamp_to_trained`` forces the duration into that
    exact range regardless of how far it actually had to move the boundary.

    Measured live: on "Deathless" (57 aligned segments, real ``run.toml``),
    segment 7's start sat only 1.66s past the cursor. The old code logged
    "snapped to the start edge (106.630s)" and then, for the very same
    boundary, "lands 45.7% through segment index=7" -- because clamping
    had silently relocated the "snapped" boundary 3.5s past the edge it
    claimed to hit, right back inside the segment. All 6 of that run's
    snap attempts behaved identically: 6 INFOs, 7 WARNINGs (the 7th had no
    snap attempt at all), and not one boundary actually reached an edge.

    This reproduces that shape with round numbers: a merge-formed group
    whose only interior edge sits 1.66s past ``cursor`` -- structurally
    exactly what ``_merge_for_minimum`` always produces (the merge only
    happens because that edge is *closer* to the group's start than
    ``eff_min``, so it can never independently satisfy the padding-alone
    branch -- see this module's section docstring) -- with
    ``DEFAULT_HARDWARE``'s trained-floor-pinned ``eff_min``.
    """
    member = make_aligned_segment(
        1, "out of the heat and into the fire,", 1.66, 20.0, "Dianne"
    )
    group = slicing_module._Group(members=(member,), start=0.0, end=20.0)

    with caplog.at_level(logging.INFO):
        pieces = slicing_module._split_for_maximum(
            [group], DEFAULT_HARDWARE.min_chunk_seconds, DEFAULT_HARDWARE.max_chunk_seconds, GRID
        )

    assert len(pieces) == 2
    first = pieces[0]
    # The edge (1.66s) is far too close to cursor=0 for any grid point that
    # actually reaches it to clear the 5.167s trained floor, so the snap
    # must be refused outright -- the boundary used is the untouched,
    # unsnapped geometric default (raw target 10.0s, quantized up to
    # 10.125s), identical to what pre-#70 code would have produced.
    assert first.end == pytest.approx(243 / 24, abs=1e-6)
    # No accepted-snap claim, ever, for this boundary.
    assert "snapped to the" not in caplog.text
    # Exactly one honest report of the cut that could not be avoided.
    assert caplog.text.count("cutting a sung phrase mid-utterance") == 1
    assert "trained-range clamp" in caplog.text


def test_full_pipeline_never_falsely_claims_a_snap_that_did_not_happen(tmp_path, caplog):
    """End-to-end version of the regression above, through `slice_audio`
    rather than `_split_for_maximum` directly: a short segment forces a
    pass-1 merge, the merged group exceeds the max-duration bound, and pass
    2's only interior edge (the merged-in segment's own start, 2.3s past
    the group's start) is too close to the cursor for any snap to survive
    the trained-floor clamp -- so `slice_audio` must report the resulting
    mid-utterance cut honestly rather than claim a snap it didn't achieve.

    This is not a hypothetical: on the real "Deathless" alignment, every
    merge-formed group's interior edges are -- by construction of
    `_merge_for_minimum` (see this module's section docstring) -- within
    `eff_min` of that group's own start, which on the real, production
    hardware profile is pinned to H3's trained floor. So on real material,
    a pass-2 snap essentially never has room to succeed; what it *can* do
    is correctly warn about the cut, instead of silently rendering the
    wrong chunk with no error anywhere (the original defect) or, worse,
    lying about having fixed it.
    """
    segments = (
        make_aligned_segment(0, "oh yeah", 0.0, 2.0, "Dianne"),
        make_aligned_segment(
            1,
            "the parts they played in america's war and the memories that linger long after",
            2.3,
            17.0,
            "Dianne",
        ),
    )
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.INFO):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)

    # The honest outcome: no claimed snap, and exactly one warning about
    # the cut it genuinely could not avoid.
    assert "snapped to the" not in caplog.text
    assert caplog.text.count("cutting a sung phrase mid-utterance") == 1
    assert "issue #70" in caplog.text

    # No content lost regardless: every original segment's span is still
    # fully covered by some chunk's [start, end) window (same invariant
    # already checked elsewhere in this file for the padding/quantization
    # passes) -- the preference failing to activate must never regress the
    # pre-existing "never truncate vocal audio" guarantee.
    by_index = {seg.index: seg for seg in alignment.segments}
    covered_end = {}
    for chunk in chunks:
        for seg_index in chunk.source_segment_indices:
            covered_end[seg_index] = max(covered_end.get(seg_index, 0.0), chunk.end)
    for seg_index, seg in by_index.items():
        assert seg_index in covered_end
        assert covered_end[seg_index] >= seg.end - _EPS


# --------------------------------------------------------------------------- #
# issue #70, second mechanism: instrumental-coverage's own re-anchoring can
# ALSO land a chunk boundary inside a segment -- and measured on the real
# "Deathless" alignment, this is the *dominant* source (27 final mid-segment
# boundaries vs. pass 2's 7), not pass 2's max-duration split. This pass
# only reports it (at the boundary a render will actually use), the same
# "cut where it must and log" fallback pass 2 uses when no snap is safe --
# it does not attempt to avoid it; see `_log_final_boundary_segment_cuts`'s
# docstring for why extending the *avoidance* here is out of scope for a
# change with no render available to verify it against.
# --------------------------------------------------------------------------- #


def test_retiling_residue_landing_inside_a_segment_is_logged(caplog):
    """A gap narrower than H3's trained floor (124 frames, 5.167s) cannot
    host even one filler chunk (`_plan_frames_run` returns `()` for it), so
    `_cover_instrumentals` absorbs it entirely: the next piece is placed
    immediately after the previous one's end, with no filler at all, rather
    than at its own true (pre-retiling) start.

    Here that shifts the second piece 1.0s earlier than where passes 1-3
    placed it. Its duration is unchanged, so its end also moves 1.0s
    earlier -- from 14.167s (which fully covered the segment's true
    8.5-14.0s span) to 13.167s, which now falls *inside* that span. Pass 2
    never had a boundary anywhere near this piece (it was never part of a
    max-duration split), so only this pass can see the cut.
    """
    seg0 = make_aligned_segment(0, "first line here", 0.0, 7.0, "Dianne")
    seg1 = make_aligned_segment(1, "second line here", 8.5, 14.0, "Dianne")
    piece0_end = 175 / 24  # grid-exact, 7.291666...s
    piece1_true_start = piece0_end + 1.0  # true gap: 1.0s, below the trained floor
    piece0 = slicing_module._Piece(
        members=(seg0,), start=0.0, end=piece0_end, is_split_continuation=False, frame_count=175
    )
    piece1 = slicing_module._Piece(
        members=(seg1,),
        start=piece1_true_start,
        end=piece1_true_start + 141 / 24,
        is_split_continuation=False,
        frame_count=141,
    )

    with caplog.at_level(logging.WARNING):
        covered = slicing_module._cover_instrumentals(
            [piece0, piece1],
            (seg0, seg1),
            track_duration=20.0,
            eff_min=DEFAULT_HARDWARE.min_chunk_seconds,
            eff_max=DEFAULT_HARDWARE.max_chunk_seconds,
            grid=GRID,
        )

    # The gap was fully absorbed: piece1 starts exactly where piece0 ends,
    # 1.0s earlier than its true start.
    retiled_piece1 = covered[1]
    assert retiled_piece1.start == pytest.approx(piece0_end, abs=1e-6)
    assert retiled_piece1.end == pytest.approx(piece0_end + 141 / 24, abs=1e-6)

    assert "Final chunk boundary" in caplog.text
    assert "after instrumental-coverage retiling" in caplog.text
    assert "lands 84.8% through segment index=1" in caplog.text
    assert "issue #70" in caplog.text
    # Distinguishable from pass 2's own wording -- this is a different
    # mechanism reporting a different (the actual, final) position.
    assert "No segment-edge snap" not in caplog.text


def test_no_final_boundary_warning_when_retiling_introduces_no_drift(caplog):
    """Control case: a gap that exactly matches a valid grid duration (here,
    the trained floor itself, 124 frames = 5.167s) tiles as a single filler
    chunk with zero residue, so the following piece lands exactly where
    passes 1-3 put it -- comfortably covering its own segment, same shape
    as the drifting case above -- and nothing is logged."""
    seg0 = make_aligned_segment(0, "first line here", 0.0, 7.0, "Dianne")
    seg1 = make_aligned_segment(1, "second line here", 12.6, 18.2, "Dianne")
    piece0_end = 175 / 24  # 7.291666...s
    gap = 124 / 24  # 5.166666...s -- exactly the trained floor, zero-residue
    piece1_start = piece0_end + gap  # 12.458333...s
    piece0 = slicing_module._Piece(
        members=(seg0,), start=0.0, end=piece0_end, is_split_continuation=False, frame_count=175
    )
    piece1 = slicing_module._Piece(
        members=(seg1,),
        start=piece1_start,
        end=piece1_start + 141 / 24,
        is_split_continuation=False,
        frame_count=141,
    )

    with caplog.at_level(logging.WARNING):
        covered = slicing_module._cover_instrumentals(
            [piece0, piece1],
            (seg0, seg1),
            track_duration=20.0,
            eff_min=DEFAULT_HARDWARE.min_chunk_seconds,
            eff_max=DEFAULT_HARDWARE.max_chunk_seconds,
            grid=GRID,
        )

    # covered[0] is piece0 (no leading gap, since piece0.start == 0.0);
    # covered[1] is the single zero-residue filler chunk covering the gap;
    # covered[2] is piece1, followed by trailing filler out to track_duration.
    retiled_piece1 = covered[2]
    assert retiled_piece1.start == pytest.approx(piece1_start, abs=1e-6)
    assert "Final chunk boundary" not in caplog.text


# --------------------------------------------------------------------------- #
# Large instrumental gaps: padding must not manufacture bogus long chunks
# --------------------------------------------------------------------------- #


def test_segments_with_large_gaps_stay_chronological_and_bounded(tmp_path):
    alignment = make_alignment_result_with_gaps()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)

    # Character identity must survive slicing untouched.
    characters = [c.character for c in chunks]
    assert "Dianne" in characters
    assert "Marcus" in characters


# --------------------------------------------------------------------------- #
# Empty input and chunk id bookkeeping
# --------------------------------------------------------------------------- #


def test_empty_alignment_returns_no_chunks(tmp_path):
    alignment = AlignmentResult(segments=(), track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", 10.0)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")
    assert chunks == ()


def test_chunk_ids_are_sequential_from_zero(tmp_path):
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunks_dir_created_if_missing(tmp_path):
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not chunks_dir.exists()
    slice_audio(master, alignment, DEFAULT_HARDWARE, chunks_dir)
    assert chunks_dir.exists()


def test_logs_info_on_successful_slice(tmp_path, caplog):
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    with caplog.at_level(logging.INFO):
        slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")
    assert "chunk" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# issue #20: frame-grid quantization -- snapping up, snapping down, and
# absorbing the remainder into padding rather than vocal content.
# --------------------------------------------------------------------------- #


def test_frame_count_is_set_and_valid_for_every_chunk(tmp_path):
    for factory in (
        make_alignment_result_normal_song,
        make_alignment_result_short_segments,
        make_alignment_result_long_segment,
        make_alignment_result_with_gaps,
    ):
        alignment = factory()
        master = write_silent_wav(tmp_path / f"{factory.__name__}.wav", alignment.track_duration)
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / factory.__name__)
        assert chunks, f"{factory.__name__} produced no chunks"
        _assert_frame_grid_exact(chunks)


def test_duration_snaps_up_to_nearest_grid_point(tmp_path):
    # A single, unpadded 5.6s segment: 5.6s = 134.4 frames. The nearest grid
    # point above (141 frames = 5.875s) is closer than the one below (124
    # frames = 5.1667s), so this should round UP.
    segments = (make_aligned_segment(0, "a short steady line of singing", 0.0, 5.6, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.frame_count == 141
    assert chunk.duration == pytest.approx(141 / 24, abs=1e-6)
    assert chunk.duration > 5.6  # rounded up, not down
    # Vocal content is still fully covered.
    assert chunk.start <= segments[0].start + _EPS
    assert chunk.end >= segments[0].end - _EPS


def test_duration_snaps_down_absorbing_padding_not_vocal_content(tmp_path):
    # A short 2.0s vocal segment padded (by a hand-built profile whose
    # min_chunk_seconds is a non-grid 6.0s) up to exactly 6.0s = 144 frames.
    # The nearest grid point below (141 frames = 5.875s) is closer than the
    # one above (158 frames = 6.5833s), so this should round DOWN --
    # shrinking the *padding* (6.0s -> 5.875s) while the 2.0s of real vocal
    # content stays fully intact within it.
    padded_profile = HardwareProfile(
        name="test-padded-min", vram_gb=24.0, min_chunk_seconds=6.0, max_chunk_seconds=15.0
    )
    segments = (make_aligned_segment(0, "oh yeah", 0.0, 2.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=30.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, padded_profile, tmp_path / "chunks")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.frame_count == 141
    assert chunk.duration == pytest.approx(141 / 24, abs=1e-6)
    assert chunk.duration < 6.0  # snapped down from the raw 6.0s padded duration
    # The original vocal span must still be fully covered -- the shrink
    # only ever comes out of padding, never the lyric content itself.
    assert chunk.start <= segments[0].start + _EPS
    assert chunk.end >= segments[0].end - _EPS


def test_remainder_absorbed_into_padding_never_drops_vocal_content(tmp_path):
    # Same scenario as the snap-down case, generalized: across every chunk
    # in a mixed run, the emitted [start, end] must always be a superset of
    # every member segment's own [start, end] -- quantization may only ever
    # add time, never remove real lyric audio.
    alignment = make_alignment_result_short_segments()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    by_index = {seg.index: seg for seg in alignment.segments}
    for chunk in chunks:
        for seg_index in chunk.source_segment_indices:
            seg = by_index[seg_index]
            assert chunk.start <= seg.start + _EPS
            assert chunk.end >= seg.end - _EPS


# --------------------------------------------------------------------------- #
# issue #20: clamping requested min/max into H3's trained frame range
# --------------------------------------------------------------------------- #


def test_min_below_trained_floor_is_clamped_with_warning(tmp_path, caplog):
    profile = HardwareProfile(
        name="stale-profile", vram_gb=24.0, min_chunk_seconds=1.0, max_chunk_seconds=15.0
    )
    segments = (make_aligned_segment(0, "a short line", 0.0, 2.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, profile, tmp_path / "chunks")

    assert "1.000" in caplog.text
    assert "5.167" in caplog.text
    assert "stale-profile" in caplog.text
    # The clamp actually took effect: the chunk still meets the trained
    # floor, not the stale 1.0s request.
    assert chunks[0].duration >= 5.0


def test_max_above_trained_ceiling_is_clamped_with_warning(tmp_path, caplog):
    # Mirrors the pre-#20 PROFILE_48GB bug: a profile asking for more than
    # the model was trained on.
    profile = HardwareProfile(
        name="too-generous", vram_gb=48.0, min_chunk_seconds=5.2, max_chunk_seconds=24.0
    )
    alignment = make_alignment_result_long_segment()  # 18.5s single segment
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, profile, tmp_path / "chunks")

    assert "24.000" in caplog.text
    assert "15.083" in caplog.text
    assert "too-generous" in caplog.text
    # The clamp took effect: the 18.5s segment still got split (it would not
    # have, against the stale 24.0s ceiling).
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.frame_count <= 362


def test_unsatisfiable_window_after_clamping_raises(tmp_path, caplog):
    # min_chunk_seconds (20.0) survives clamping unchanged (it's already
    # above the trained floor), but max_chunk_seconds (25.0) gets clamped
    # down to ~15.08s -- leaving effective min > effective max, which is not
    # a window slice_audio can honor.
    profile = HardwareProfile(
        name="impossible", vram_gb=24.0, min_chunk_seconds=20.0, max_chunk_seconds=25.0
    )
    segments = (make_aligned_segment(0, "line", 0.0, 6.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=30.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="impossible"):
        slice_audio(master, alignment, profile, tmp_path / "chunks")
    assert "20.000" in caplog.text
    assert "15.083" in caplog.text


def test_quantization_re_snaps_when_both_padding_gaps_are_tight(tmp_path, caplog):
    # A single 5.2s segment sitting almost flush against both the track's
    # start (0.05s of backward room) and the track's end (0.05s of forward
    # room) -- 0.1s combined, well short of the ~0.675s of padding needed to
    # reach the nearest safe grid point (5.875s/141 frames). This exercises
    # the "insufficient room" re-snap fallback in ``_quantize_single_piece``:
    # duration and frame_count must still match exactly even though the
    # original grid target could not be fully realized.
    segments = (make_aligned_segment(0, "one line", 0.05, 5.25, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=5.30)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert "only" in caplog.text and "instrumental padding is available" in caplog.text
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.duration == pytest.approx(GRID.frames_to_seconds(chunk.frame_count), abs=1e-6)
    assert chunk.start <= segments[0].start + _EPS
    assert chunk.end >= segments[0].end - _EPS


def test_quantization_overlap_from_insufficient_padding_room_still_raises(tmp_path):
    # Same tight-padding scenario as above, but with a *second* segment
    # crowding in right after the first -- so the grid-quantization growth
    # this chunk needs would run straight into the next chunk's true start.
    # This must not be silently absorbed: the pre-existing timeline-drift
    # guard is the explicit backstop for a configuration this tight.
    segments = (
        make_aligned_segment(0, "one", 0.05, 5.25, "Dianne"),
        make_aligned_segment(1, "two", 5.30, 10.5, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=15.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with pytest.raises(ValueError, match="precedes previous chunk end"):
        slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")


# --------------------------------------------------------------------------- #
# issue #20: the ~30-chunk no-cumulative-drift acceptance test
# --------------------------------------------------------------------------- #


def _make_long_song_alignment(n_chunks: int = 30) -> AlignmentResult:
    """A synthetic ~n_chunks-segment song with durations that deliberately
    straddle grid points in both directions (some snap up, some snap down
    once padded) and small, varying instrumental gaps between lines --
    representative of a real forced-alignment timeline, unlike the
    hand-picked single-segment unit tests above."""
    segments = []
    cursor = 0.0
    for i in range(n_chunks):
        # Cycle durations from 5.3s to 9.8s in 0.5s steps -- deliberately
        # off-grid so quantization is exercised on every single chunk.
        duration = 5.3 + (i % 10) * 0.5
        start = cursor
        end = start + duration
        segments.append(
            make_aligned_segment(i, f"line number {i} of the song", start, end, "Dianne")
        )
        # Grid step is 17/24 ~= 0.708s and quantization always rounds UP for
        # an un-padded single segment (see slicing._quantize_single_piece --
        # "nearest" that would round below the true vocal span is always
        # overridden to round up instead, to avoid truncating lyric audio),
        # so the gap must comfortably exceed one grid step or the padding
        # this chunk needs cannot fit before the next segment's true start.
        gap = 1.0 + (i % 3) * 0.4  # 1.0s - 1.8s instrumental gap
        cursor = end + gap
    track_duration = cursor + 2.0
    return AlignmentResult(segments=tuple(segments), track_duration=track_duration)


def test_thirty_chunk_run_has_zero_accumulated_drift(tmp_path):
    """The defect issue #20 exists to fix: without per-chunk quantization,
    each chunk's rendered video duration would silently round to H3's frame
    grid at execution time while the audio stem stayed at the unquantized
    duration, and that per-chunk mismatch would accumulate across the song.
    With quantization done here, each chunk's audio is sliced to *exactly*
    its frame_count's implied duration, so there is nothing left to
    accumulate: the error is zero after 1 chunk, and it is still exactly
    zero after 30.
    """
    alignment = _make_long_song_alignment(30)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 30
    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)

    # The core drift assertion: per-chunk audio duration vs. what Stage 4a
    # will request as `length` (frame_count) match to within floating-point
    # precision for *every* chunk -- not just on average. Summed over 30
    # chunks the total mismatch is therefore exactly zero, not merely small.
    total_mismatch = sum(
        abs(chunk.duration - GRID.frames_to_seconds(chunk.frame_count)) for chunk in chunks
    )
    assert total_mismatch < 30 * 1e-6

    # Every original segment's vocal content is still fully covered and the
    # chunk sequence never drifts backwards relative to the original
    # alignment timeline -- each segment's true start is read directly off
    # the immutable input, never recomputed from a previous chunk's
    # (possibly rounded) end, so nothing here can compound.
    by_index = {seg.index: seg for seg in alignment.segments}
    for chunk in chunks:
        for seg_index in chunk.source_segment_indices:
            seg = by_index[seg_index]
            assert chunk.start <= seg.start + _EPS
            assert chunk.end >= seg.end - _EPS

    # And the exported wav files really do match the declared durations.
    for chunk in chunks:
        exported_seconds = _wav_duration_seconds(chunk.audio_file)
        assert exported_seconds == pytest.approx(chunk.duration, abs=0.02)


def test_thirty_chunk_run_matches_unquantized_baseline_within_one_grid_step_per_chunk(tmp_path):
    """Sanity check on the *magnitude* of what quantization changes: total
    rendered duration across the run should track the raw, unquantized
    total closely (bounded by a small constant per chunk), confirming
    quantization is a per-chunk rounding correction, not a systematic
    lengthening or shortening of the whole song."""
    alignment = _make_long_song_alignment(30)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    raw_total = sum(seg.duration for seg in alignment.segments)
    quantized_total = sum(chunk.duration for chunk in chunks)
    one_grid_step = GRID.step_frames / GRID.fps
    assert abs(quantized_total - raw_total) < len(chunks) * one_grid_step


# --------------------------------------------------------------------------- #
# issue #20's index-space guard: ChunkFrameMismatchError
# --------------------------------------------------------------------------- #
#
# A chunk's frame_count and its sliced duration are computed together in
# three different code paths (pass 2's split-quantize loop, pass 3's
# quantize-and-pad, and the untouched pass-through) before landing on the
# same AudioChunk -- exactly the kind of index-space seam this project has
# been bitten by before (AlignedSegment.index vs chunk_id confusion). These
# tests monkeypatch the pass-3 quantizer to simulate that seam breaking, and
# confirm slice_audio refuses to silently emit a mismatched chunk.


def test_chunk_frame_mismatch_error_raised_for_invalid_frame_count(tmp_path, monkeypatch):
    import music_video_maker.slicing as slicing_module

    def _broken_quantize(piece, grid, prev_end, next_boundary):
        # Simulate a bug that never assigns a frame_count.
        return slicing_module._Piece(
            members=piece.members,
            start=piece.start,
            end=piece.end,
            is_split_continuation=piece.is_split_continuation,
            frame_count=None,
        )

    monkeypatch.setattr(slicing_module, "_quantize_single_piece", _broken_quantize)

    segments = (make_aligned_segment(0, "line", 0.0, 6.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with pytest.raises(slicing_module.ChunkFrameMismatchError, match="not a valid H3 grid point"):
        slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")


def test_chunk_frame_mismatch_error_raised_when_duration_disagrees_with_frame_count(
    tmp_path, monkeypatch
):
    import music_video_maker.slicing as slicing_module

    def _broken_quantize(piece, grid, prev_end, next_boundary):
        # Valid frame_count, but deliberately mismatched against the actual
        # [start, end) span -- the exact bug class this guard exists for.
        return slicing_module._Piece(
            members=piece.members,
            start=piece.start,
            end=piece.end,
            is_split_continuation=piece.is_split_continuation,
            frame_count=362,  # 15.083s -- does not match this short piece
        )

    monkeypatch.setattr(slicing_module, "_quantize_single_piece", _broken_quantize)

    segments = (make_aligned_segment(0, "line", 0.0, 6.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with pytest.raises(slicing_module.ChunkFrameMismatchError, match="implies duration"):
        slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")


# --------------------------------------------------------------------------- #
# Misc pre-existing behaviors retained from #4
# --------------------------------------------------------------------------- #


def test_track_running_out_of_room_emits_short_final_chunk_with_warning(tmp_path, caplog):
    # The very last segment is too short to reach the minimum even after
    # padding all the way to the track's end -- #4's original "accept the
    # shortfall" behavior, unchanged by #20 beyond the raised threshold.
    segments = (make_aligned_segment(0, "last word", 9.0, 9.5, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=9.6)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert "cannot reach the" in caplog.text and "minimum" in caplog.text
    assert len(chunks) == 1
    assert chunks[0].end <= alignment.track_duration + _EPS

    # Issue #20 defect (a), on the one path that can still reach it. Pass 1
    # gives up on the minimum here (the track itself runs out of room), so
    # this chunk arrives at quantization *below* the trained floor -- the
    # only case where clamp_to_trained actually changes the answer. Without
    # it the chunk would render at 22 frames (0.92s), far outside H3's
    # trained 124-362 range, which is exactly the out-of-distribution
    # rendering #20 exists to prevent. The floor wins over the short raw
    # duration: the chunk is padded backwards into the preceding
    # instrumental time rather than rendered out of range.
    assert chunks[0].frame_count == GRID.trained_min_frames
    assert chunks[0].duration == pytest.approx(
        GRID.frames_to_seconds(GRID.trained_min_frames), abs=1e-6
    )


def test_merged_chunk_with_differing_characters_attributes_to_dominant_voice(tmp_path, caplog):
    """Issue #40: a merge attributes to whichever segment contributed the most
    voiced duration, not whichever came first. Dianne's segment (2.9s) outlasts
    Marcus's (2.0s), so the chunk must resolve to Dianne even though Marcus is
    first in timeline order -- the exact "first-wins misattributes a handover"
    bug reported in #40.
    """
    segments = (
        make_aligned_segment(0, "his line", 0.0, 2.0, "Marcus"),
        make_aligned_segment(1, "her line", 2.1, 5.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    assert chunks[0].character == "Dianne"
    # The warning still fires (issue #40 keeps it), and now names the winner
    # and the voiced-duration numbers that decided it -- not just the bare
    # fact that characters differed.
    assert "differing characters" in caplog.text
    assert "Dianne" in caplog.text
    assert "2.900" in caplog.text  # Dianne's own contributed voiced duration
    assert "4.900" in caplog.text  # total voiced duration across both members


def test_merged_chunk_one_word_segment_loses_to_six_word_segment(tmp_path, caplog):
    """Acceptance criterion from issue #40: a 1-word segment from A merged with
    a 6-word segment from B attributes to B. This is the real-world case from
    "The Lucky Ones" chunk 26 -- Jan's trailing "defenestrated." handed off to
    Dianne's "There was a time I felt alone", and the chunk should read as
    Dianne's, not Jan's.
    """
    segments = (
        make_aligned_segment(0, "defenestrated", 0.0, 0.3, "Jan"),
        make_aligned_segment(1, "there was a time i felt", 0.4, 5.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    assert chunks[0].character == "Dianne"
    assert chunks[0].characters == ("Dianne",)


def test_merged_chunk_attribution_uses_voiced_duration_not_word_count(tmp_path):
    """Issue #40 picks voiced duration, not word count, as the measure of
    "who dominates this chunk" -- a single held note can outlast several short
    words. Marcus has more words but far less voiced time than Dianne's long
    held note, so Dianne must still win.
    """
    segments = (
        make_aligned_segment(0, "one two three", 0.0, 1.0, "Marcus"),
        make_aligned_segment(1, "ohhhh", 1.1, 6.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    assert chunks[0].character == "Dianne"


def test_merged_chunk_tie_in_voiced_duration_keeps_first_segment(tmp_path):
    """Ties keep today's first-wins behaviour (issue #40's explicit carve-out)."""
    segments = (
        make_aligned_segment(0, "his line here", 0.0, 2.5, "Marcus"),
        make_aligned_segment(1, "her line here", 2.6, 5.1, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    assert chunks[0].character == "Marcus"


def test_merged_chunk_with_same_character_behaves_as_before(tmp_path, caplog):
    """Acceptance criterion from issue #40: a merge of two segments from the
    same singer is unaffected by the dominant-voice attribution logic -- no
    warning, and the (only) character resolves normally.
    """
    segments = (
        make_aligned_segment(0, "a", 0.0, 0.3, "Dianne"),
        make_aligned_segment(1, "b c d e f g", 0.4, 5.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=10.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    assert len(chunks) == 1
    assert chunks[0].character == "Dianne"
    assert "differing characters" not in caplog.text


# --------------------------------------------------------------------------- #
# Instrumental coverage (contiguous timeline tiling)
#
# Without this, slice_audio emits chunks only for *voiced* spans. On a real
# song that is well under half the runtime -- "The Lucky Ones" aligns to
# 134.9s of vocals across a 261.7s track. Stage 5 concatenates chunks
# back-to-back with `-c:v copy`, so every skipped instrumental span is
# squeezed out of the video while the muxed master audio keeps it: the video
# runs short and every chunk after the first gap plays against the wrong
# moment of the song.
#
# The fix is contiguous tiling. `cover_instrumentals=True` fills the unvoiced
# spans with filler chunks and re-anchors every chunk so that chunk N starts
# exactly where chunk N-1 ended. That makes a chunk's position in the
# concatenated video identical to its slice position in the master by
# construction, which is the only way the final mux can be in sync.
# --------------------------------------------------------------------------- #


def _assert_contiguous_from_zero(chunks: tuple[AudioChunk, ...]) -> None:
    """The sync invariant: video position == audio position, for every chunk.

    Stage 5 concatenates, so chunk N's offset in the final video is the sum of
    durations 0..N-1. Asserting that equals chunk N's own ``start`` (its slice
    offset into the master) is exactly asserting the final mux is in sync.
    """
    assert chunks[0].start == pytest.approx(0.0, abs=_EPS)
    running = 0.0
    for chunk in chunks:
        assert chunk.start == pytest.approx(running, abs=1e-6)
        running += chunk.duration
        assert chunk.end == pytest.approx(running, abs=1e-6)


def test_instrumental_coverage_tiles_the_whole_timeline_contiguously(tmp_path):
    alignment = make_alignment_result_with_gaps()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)

    # Coverage reaches the end of the track to within one chunk's worth of
    # grid rounding -- never 30s short the way gap-skipping was.
    covered = sum(c.duration for c in chunks)
    assert covered == pytest.approx(
        alignment.track_duration, abs=GRID.frames_to_seconds(GRID.step_frames)
    )


def test_instrumental_coverage_fills_a_long_gap_with_filler_chunks(tmp_path):
    alignment = make_alignment_result_with_gaps()  # 30s instrumental at 5.0-35.0
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    filler = [c for c in chunks if c.is_instrumental]
    assert filler, "the 30s instrumental gap must be covered by filler chunks"

    # Filler carries no lyric and no source segments -- Stage 2b keys its
    # instrumental clause off the empty text.
    for chunk in filler:
        assert chunk.text == ""
        assert chunk.source_segment_indices == ()

    # The 30s instrumental is spanned by filler rather than left as a hole.
    # It is wider than one max-length chunk, so it takes more than one.
    gap_filler = [c for c in filler if 5.0 <= c.start < 34.0]
    assert len(gap_filler) >= 2
    assert sum(c.duration for c in gap_filler) >= 25.0


def test_instrumental_coverage_covers_intro_and_outro(tmp_path):
    segments = (
        make_aligned_segment(0, "a line arrives late", 20.0, 26.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=60.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    _assert_contiguous_from_zero(chunks)
    # The 20s intro before the first lyric is covered rather than skipped.
    assert chunks[0].is_instrumental
    assert chunks[0].start == pytest.approx(0.0, abs=_EPS)
    # ...and the 34s outro after the last one.
    assert chunks[-1].is_instrumental
    assert chunks[-1].end > 55.0


def test_instrumental_coverage_never_emits_below_the_trained_minimum(tmp_path):
    """A gap far too short to be its own chunk must not become a runt chunk.

    H3 is only trained down to 124 frames; emitting a 0.4s filler to patch a
    0.4s hole would render out of distribution. Small holes get absorbed by
    the neighbouring chunk instead.
    """
    segments = (
        make_aligned_segment(0, "first line here", 0.0, 6.0, "Dianne"),
        # 0.4s hole -- far below the 5.167s trained floor.
        make_aligned_segment(1, "second line here", 6.4, 12.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    _assert_frame_grid_exact(chunks)  # enforces >= trained_min_frames
    _assert_contiguous_from_zero(chunks)


def test_instrumental_coverage_preserves_lyric_chunk_text_and_character(tmp_path):
    alignment = make_alignment_result_with_gaps()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    voiced = [c for c in chunks if not c.is_instrumental]
    assert voiced, "lyric chunks must survive gap filling"
    joined = " ".join(c.text for c in voiced)
    assert "count the days until the sun comes back" in joined
    assert "nothing but the echo of a drum" in joined
    assert {c.character for c in voiced} == {"Dianne", "Marcus"}


def test_instrumental_coverage_writes_a_wav_for_every_chunk(tmp_path):
    alignment = make_alignment_result_with_gaps()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks_dir = tmp_path / "chunks"

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, chunks_dir, cover_instrumentals=True
    )

    for chunk in chunks:
        assert chunk.audio_file.exists()
        assert _wav_duration_seconds(chunk.audio_file) == pytest.approx(chunk.duration, abs=0.01)


def test_instrumental_coverage_tiles_a_wholly_unvoiced_track(tmp_path):
    """A song with ZERO aligned segments (an empty lyrics_file, e.g. because
    no one is ever shown singing even though the audio has vocals) must still
    get a full, contiguous chunk timeline when cover_instrumentals is on --
    the whole point of "the chunk timeline must cover the whole track".

    Before this test, ``slice_audio`` bailed out with an early
    ``if not alignment.segments: return ()`` guard that fired regardless of
    ``cover_instrumentals``, so this case produced zero chunks and the run
    failed outright rather than rendering an all-instrumental video."""
    alignment = AlignmentResult(segments=(), track_duration=30.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    assert chunks, "zero segments + cover_instrumentals=True must still tile the track"
    _assert_timeline_sane(chunks)
    _assert_frame_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    assert all(c.is_instrumental for c in chunks)
    assert all(c.text == "" for c in chunks)
    covered = sum(c.duration for c in chunks)
    assert covered == pytest.approx(
        alignment.track_duration, abs=GRID.frames_to_seconds(GRID.step_frames)
    )


def test_instrumental_coverage_is_off_by_default(tmp_path):
    """The flag is opt-in: the default path stays gap-skipping, so nothing
    that already depends on segment-anchored chunk boundaries changes."""
    alignment = make_alignment_result_with_gaps()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    default_chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "a")
    covered_chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "b", cover_instrumentals=True
    )

    assert not any(c.is_instrumental for c in default_chunks)
    assert len(covered_chunks) > len(default_chunks)


# --------------------------------------------------------------------------- #
# Per-chunk lyric text: only the words actually audible in the chunk
#
# A lyric line frequently straddles a chunk boundary -- either because pass 2
# split a long segment, or because instrumental coverage retiled the timeline.
# Assigning each chunk the FULL text of every overlapping segment prompts H3
# to perform the whole line in BOTH chunks, so the performance drifts away
# from the audio.
#
# Observed on a real render: "There was a time ... so much better now!" ran
# 84.46-91.63s. Chunk 13 (82.04-89.33) held the first 4.9s of it and chunk 14
# (89.33-95.92) the last 2.3s, and both were prompted with the entire line.
# H3 sang all twelve words in each, so the mouth was still on "so much better
# now" at 1:35 -- six seconds after the audio had said it.
#
# Words carry their own timings (AlignedSegment.words), so a chunk can be
# given exactly the words whose timing lands inside it. Chunks tile
# contiguously, so keying on each word's midpoint puts every word in exactly
# one chunk -- no duplication, no dropped words.
# --------------------------------------------------------------------------- #


MAX8_HARDWARE = HardwareProfile(
    name="RTX 4090 24GB (doris), 8s cap", vram_gb=24.0,
    min_chunk_seconds=5.167, max_chunk_seconds=8.0,
)
"""The window the real run config uses -- DEFAULT_HARDWARE allows 15.083s, so
a 12s lyric line would fit in a single chunk there and never straddle a
boundary, which is the case these tests exist to cover."""


def _seg_with_words(index, text, start, end, character="Dianne"):
    """An AlignedSegment whose words are spread evenly across its span."""
    from music_video_maker.contracts import AlignedSegment, WordTiming

    parts = text.split()
    step = (end - start) / len(parts)
    words = tuple(
        WordTiming(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(parts)
    )
    return AlignedSegment(
        index=index,
        text=text,
        start=start,
        end=end,
        words=words,
        characters=(character,) if character else (),
    )


def test_a_line_straddling_a_boundary_is_split_between_the_two_chunks(tmp_path):
    """Neither chunk may claim the whole line, and between them they must
    account for every word exactly once."""
    segments = (
        _seg_with_words(0, "one two three four five six seven eight nine ten", 0.0, 12.0),
    )
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, MAX8_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    sung = [c for c in chunks if c.text.strip()]
    assert len(sung) >= 2, "a 12s line must span more than one chunk at an 8s max"

    # No chunk repeats the entire line.
    for chunk in sung:
        assert chunk.text.strip() != "one two three four five six seven eight nine ten"

    # Every word appears exactly once across the whole run.
    words = " ".join(c.text for c in chunks).split()
    assert words == ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def test_each_word_lands_in_the_chunk_that_actually_contains_it(tmp_path):
    segments = (_seg_with_words(0, "alpha bravo charlie delta echo foxtrot", 0.0, 12.0),)
    alignment = AlignmentResult(segments=segments, track_duration=20.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, MAX8_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    step = 12.0 / 6
    for i, word in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]):
        midpoint = (i + 0.5) * step
        owner = next(c for c in chunks if c.start <= midpoint < c.end)
        assert word in owner.text, (
            f"{word!r} (mid {midpoint:.2f}s) missing from chunk {owner.chunk_id}"
        )


def test_a_chunk_holding_no_words_reads_as_instrumental(tmp_path):
    """The tail of a chunk whose words all fall earlier carries no lyric, so
    it must be prompted as instrumental rather than repeating the line."""
    segments = (_seg_with_words(0, "short line here", 0.0, 2.0),)
    alignment = AlignmentResult(segments=segments, track_duration=30.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    later = [c for c in chunks if c.start >= 10.0]
    assert later, "the 30s track must produce chunks past the lyric"
    for chunk in later:
        assert chunk.text == ""
        assert chunk.is_instrumental


def test_segments_without_word_timings_still_get_their_full_text(tmp_path):
    """Word timings are optional on AlignedSegment. With none available there
    is nothing to slice by, so the whole segment text is the honest fallback."""
    segments = (make_aligned_segment(0, "no word timings here", 0.0, 6.0, "Dianne"),)
    alignment = AlignmentResult(segments=segments, track_duration=12.0)
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )

    assert any("no word timings here" in c.text for c in chunks)


# --------------------------------------------------------------------------- #
# Issue #33 level 3: counterpoint reaches the chunks
# --------------------------------------------------------------------------- #


def test_counterpoint_streams_fold_into_the_chunks_they_overlap(tmp_path):
    """A chunk over a [simultaneously] block carries BOTH simultaneous lyric
    texts and every voice audible in its span -- spine first, so the dominant
    voice keeps the primary slot (#40) and the counterpoint faces still stage
    their reference photos (#33 levels 1-2 multi-reference)."""
    from music_video_maker.alignment import ConcurrentSegment, CounterpointAlignmentResult

    segments = (
        make_aligned_segment(0, "there was a time when i hoped", 0.0, 6.5, "Dianne"),
        make_aligned_segment(1, "the second verse begins here now", 10.0, 16.0, "Dianne"),
    )
    concurrent = ConcurrentSegment(
        index=2,
        text="i know when it's people like you",
        start=0.5,
        end=6.0,
        words=(),
        characters=("Marcus",),
        spine_segment_indices=(0,),
        stream_index=1,
        block_index=0,
    )
    alignment = CounterpointAlignmentResult(
        segments=segments, track_duration=20.0, concurrent_segments=(concurrent,)
    )
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)

    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")

    over_block = [c for c in chunks if c.start < 6.0 and c.end > 0.5 and not c.is_instrumental]
    assert over_block, "no chunk covers the counterpoint span"
    for chunk in over_block:
        assert chunk.concurrent_texts == ("i know when it's people like you",)
        assert chunk.concurrent_characters == (("Marcus",),)
        assert chunk.characters[0] == "Dianne", "the spine voice must keep the primary slot"
        assert "Marcus" in chunk.characters, "the counterpoint face must stage its photo"
    # Chunks outside the block carry nothing extra.
    for chunk in chunks:
        if chunk.start >= 6.0 or chunk.end <= 0.5:
            assert chunk.concurrent_texts == ()


def test_plain_alignment_results_leave_chunks_unchanged(tmp_path):
    """A pre-#33 AlignmentResult has no concurrent segments; every chunk's
    counterpoint fields stay empty."""
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    chunks = slice_audio(master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks")
    assert all(c.concurrent_texts == () and c.concurrent_characters == () for c in chunks)
