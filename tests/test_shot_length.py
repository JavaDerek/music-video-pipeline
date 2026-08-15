"""Tests for editorial shot length (issue #27).

Today every shot is 5-8 seconds because chunk boundaries fall out of
vocal-segment timing plus a fixed ``max_chunk_seconds``: a scene change every
~6 s for the whole video, which is the single most machine-generated thing
about the output. Shot length should be an *editorial* choice the shot plan
makes, not a side effect of where the lyrics happen to fall.

The mechanism under test:

* ``[[shot]]`` entries gain an optional ``length_seconds``
  (:mod:`music_video_maker.shot_plan`), anchored by the entry's existing
  ``start`` -- a *song time*, which is stable across re-slicing in a way
  ``chunk_id`` is not.
* ``shot_length_requests()`` projects a plan into anchored
  :class:`~music_video_maker.shot_plan.ShotLength` requests.
* ``slice_audio(..., shot_lengths=...)`` honours them by *retiling* the
  contiguous timeline: the chunk at the anchor takes the requested (grid
  quantized, trained-range clamped) frame count and swallows the chunks it
  now covers; whatever is left over is re-tiled behind it.

Everything the rest of the pipeline depends on has to survive that:
contiguous coverage from 0 (video offset == audio offset), exact frame-grid
quantization, no invented timing, and no lyric word lost or duplicated.

Conservative by construction: with no requests, slicing is unchanged.
"""

from __future__ import annotations

import logging

import pytest

from music_video_maker import hardware
from music_video_maker.contracts import (
    AlignedSegment,
    AlignmentResult,
    AudioChunk,
    HardwareProfile,
    WordTiming,
)
from music_video_maker.shot_plan import (
    MEASURED_MAX_FRAMES,
    ShotLength,
    ShotPlanError,
    load_shot_plan,
    shot_length_requests,
)
from music_video_maker.slicing import slice_audio
from tests.harness.factories import make_aligned_segment, write_silent_wav

DEFAULT_HARDWARE = hardware.PROFILE_RTX_4090_24GB
GRID = DEFAULT_HARDWARE.frame_grid
_EPS = 1e-6

NARROW_HARDWARE = HardwareProfile(
    name="narrow-8s", vram_gb=24.0, min_chunk_seconds=5.167, max_chunk_seconds=8.0
)
"""The window the real run config uses: 5-8 s chunks, i.e. exactly the
"scene change every six seconds" cadence issue #27 exists to break. Using it
here means a long take has to be *asked for* to appear, which is the point."""


def _write_plan(tmp_path, body: str):
    path = tmp_path / "shot_plan.toml"
    path.write_text(body)
    return path


def _assert_grid_exact(chunks: tuple[AudioChunk, ...]) -> None:
    for chunk in chunks:
        assert chunk.frame_count is not None
        assert GRID.is_valid(chunk.frame_count)
        assert GRID.trained_min_frames <= chunk.frame_count <= GRID.trained_max_frames
        assert chunk.duration == pytest.approx(GRID.frames_to_seconds(chunk.frame_count), abs=1e-6)


def _assert_contiguous_from_zero(chunks: tuple[AudioChunk, ...]) -> None:
    """video offset == audio offset, for every chunk (the sync invariant)."""
    assert chunks[0].start == pytest.approx(0.0, abs=_EPS)
    running = 0.0
    for chunk in chunks:
        assert chunk.start == pytest.approx(running, abs=1e-6)
        running += chunk.duration
        assert chunk.end == pytest.approx(running, abs=1e-6)


def _gap_song() -> AlignmentResult:
    """One line, a 34 s instrumental, another line, a short outro."""
    segments = (
        make_aligned_segment(0, "the first line of the song", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "and the second line arrives", 40.0, 46.0, "Dianne"),
    )
    return AlignmentResult(segments=segments, track_duration=60.0)


def _slice(tmp_path, alignment, name, profile=NARROW_HARDWARE, **kwargs):
    master = write_silent_wav(tmp_path / f"{name}.wav", alignment.track_duration)
    return slice_audio(
        master, alignment, profile, tmp_path / name, cover_instrumentals=True, **kwargs
    )


# --------------------------------------------------------------------------- #
# shot_plan: expressing a length
# --------------------------------------------------------------------------- #


def test_length_seconds_is_parsed_onto_the_entry(tmp_path):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
shot = "She steps out of her front door"

[[shot]]
chunk_id = 1
start = 8.0
length_seconds = 15.0
shot = "One continuous walk-away down the whole length of the street"
""",
    )
    plan = load_shot_plan(path)
    assert plan[0].length_seconds is None
    assert plan[1].length_seconds == pytest.approx(15.0)


def test_length_seconds_must_be_a_positive_number(tmp_path):
    for bad in ("0.0", "-4.0", '"15"', "true"):
        path = _write_plan(
            tmp_path,
            f"""
[[shot]]
chunk_id = 0
start = 0.0
length_seconds = {bad}
shot = "a shot"
""",
        )
        with pytest.raises(ShotPlanError, match="length_seconds"):
            load_shot_plan(path)


def test_a_plan_with_no_lengths_yields_no_requests(tmp_path):
    """Backward compatibility: every plan authored before #27 is unchanged."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
shot = "a shot"
""",
    )
    assert shot_length_requests(load_shot_plan(path)) == ()
    assert shot_length_requests(None) == ()


def test_requests_are_anchored_by_start_and_sorted(tmp_path):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 7
start = 60.0
length_seconds = 15.0
shot = "the long one"

[[shot]]
chunk_id = 2
start = 12.5
length_seconds = 10.0
shot = "the medium one"

[[shot]]
chunk_id = 3
start = 20.0
shot = "no length asked for"
""",
    )
    requests = shot_length_requests(load_shot_plan(path))
    assert [r.start for r in requests] == [12.5, 60.0]
    assert [r.length_seconds for r in requests] == [10.0, 15.0]
    assert [r.source_chunk_id for r in requests] == [2, 7]


def test_two_entries_asking_for_a_length_at_the_same_start_is_an_error(tmp_path):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 30.0
length_seconds = 10.0
shot = "one"

[[shot]]
chunk_id = 5
start = 30.0
length_seconds = 15.0
shot = "two"
""",
    )
    with pytest.raises(ShotPlanError, match="30.0"):
        shot_length_requests(load_shot_plan(path))


def test_load_warns_that_a_long_take_is_unmeasured_on_this_card(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
length_seconds = 15.0
shot = "the long one"
""",
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)
    assert str(MEASURED_MAX_FRAMES) in caplog.text
    assert "unmeasured" in caplog.text.lower()


def test_load_warns_when_a_requested_length_exceeds_h3s_trained_ceiling(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
length_seconds = 22.0
shot = "longer than H3 was ever trained for"
""",
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)
    assert "15.083" in caplog.text


def test_load_is_quiet_for_a_length_inside_measured_territory(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
length_seconds = 5.5
shot = "an ordinary shot"
""",
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)
    assert "unmeasured" not in caplog.text.lower()


# --------------------------------------------------------------------------- #
# slicing: honouring a length
# --------------------------------------------------------------------------- #


def test_no_requests_leaves_the_timeline_byte_identical(tmp_path):
    alignment = _gap_song()
    before = _slice(tmp_path, alignment, "before")
    after = _slice(tmp_path, alignment, "after", shot_lengths=())

    assert [(c.start, c.end, c.frame_count, c.text) for c in before] == [
        (c.start, c.end, c.frame_count, c.text) for c in after
    ]


def test_a_requested_long_take_replaces_the_run_of_short_ones(tmp_path):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")

    # Anchor on a filler chunk well inside the 34 s instrumental.
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)
    long_take = _slice(
        tmp_path,
        alignment,
        "long",
        shot_lengths=(ShotLength(start=anchor, length_seconds=15.0, source_chunk_id=3),),
    )

    _assert_grid_exact(long_take)
    _assert_contiguous_from_zero(long_take)

    shot = next(c for c in long_take if c.start == pytest.approx(anchor, abs=1e-6))
    assert shot.frame_count == 362
    assert shot.duration == pytest.approx(362 / 24, abs=1e-6)
    # Fewer, longer shots -- the whole point of the issue.
    assert len(long_take) < len(default_chunks)


def test_the_long_take_does_not_break_whole_track_coverage(tmp_path):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)

    long_take = _slice(
        tmp_path,
        alignment,
        "long",
        shot_lengths=(ShotLength(start=anchor, length_seconds=15.0),),
    )

    covered = sum(c.duration for c in long_take)
    baseline = sum(c.duration for c in default_chunks)
    one_step = GRID.frames_to_seconds(GRID.step_frames)
    assert covered == pytest.approx(baseline, abs=one_step)
    assert covered == pytest.approx(alignment.track_duration, abs=2 * one_step)


def test_two_requests_both_land(tmp_path):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchors = [c.start for c in default_chunks if c.is_instrumental and 6.0 < c.start < 40.0]
    assert len(anchors) >= 3

    chunks = _slice(
        tmp_path,
        alignment,
        "two",
        shot_lengths=(
            ShotLength(start=anchors[0], length_seconds=10.0),
            ShotLength(start=anchors[-1], length_seconds=10.0),
        ),
    )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    starts = {round(c.start, 3): c.frame_count for c in chunks}
    assert starts[round(anchors[0], 3)] == 243  # 10.125 s
    # The second anchor moves by at most the grid residue the first left behind.
    matched = min(chunks, key=lambda c: abs(c.start - anchors[-1]))
    assert abs(matched.start - anchors[-1]) < 1.0
    assert matched.frame_count == 243


def test_a_request_above_the_trained_ceiling_is_clamped_with_a_warning(tmp_path, caplog):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "clamped",
            shot_lengths=(ShotLength(start=anchor, length_seconds=25.0),),
        )

    assert "15.083" in caplog.text
    shot = next(c for c in chunks if c.start == pytest.approx(anchor, abs=1e-6))
    assert shot.frame_count == GRID.trained_max_frames
    _assert_grid_exact(chunks)


def test_a_request_below_the_trained_floor_is_clamped_with_a_warning(tmp_path, caplog):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "floor",
            shot_lengths=(ShotLength(start=anchor, length_seconds=2.0),),
        )

    assert "5.167" in caplog.text
    shot = next(c for c in chunks if c.start == pytest.approx(anchor, abs=1e-6))
    assert shot.frame_count == GRID.trained_min_frames


def test_a_chunk_above_141_frames_logs_that_it_is_unmeasured_territory(tmp_path, caplog):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)

    with caplog.at_level(logging.WARNING):
        _slice(
            tmp_path,
            alignment,
            "unmeasured",
            shot_lengths=(ShotLength(start=anchor, length_seconds=15.0),),
        )

    assert "141" in caplog.text
    assert "unmeasured" in caplog.text.lower()
    # Named, not gated: the run still produced the long take.
    assert "refus" not in caplog.text.lower()


def test_long_chunks_are_reported_even_when_nobody_asked_for_them(tmp_path, caplog):
    """The unmeasured-territory report is about the *rendered* frame counts,
    not about requests. With ``max_chunk_seconds`` at H3's trained ceiling a
    long instrumental already tiles into chunks well past 141 frames, and a
    run about to spend hours on frame counts nobody has measured should say so
    before it starts rather than after."""
    alignment = _gap_song()

    with caplog.at_level(logging.WARNING):
        chunks = _slice(tmp_path, alignment, "wide", profile=DEFAULT_HARDWARE)

    assert any((c.frame_count or 0) > MEASURED_MAX_FRAMES for c in chunks)
    assert f"exceed {MEASURED_MAX_FRAMES} frames" in caplog.text
    assert "unmeasured" in caplog.text.lower()


def test_the_unmeasured_report_is_silent_when_every_chunk_is_measured(tmp_path, caplog):
    alignment = _gap_song()
    short = HardwareProfile(
        name="short", vram_gb=24.0, min_chunk_seconds=5.167, max_chunk_seconds=5.875
    )
    with caplog.at_level(logging.WARNING):
        chunks = _slice(tmp_path, alignment, "short", profile=short)

    assert all((c.frame_count or 0) <= MEASURED_MAX_FRAMES for c in chunks)
    assert f"exceed {MEASURED_MAX_FRAMES} frames" not in caplog.text


def test_an_unmatched_anchor_warns_and_changes_nothing(tmp_path, caplog):
    alignment = _gap_song()
    baseline = _slice(tmp_path, alignment, "baseline")

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "stray",
            shot_lengths=(ShotLength(start=10.0, length_seconds=15.0, source_chunk_id=99),),
        )

    assert "10.000" in caplog.text
    assert "99" in caplog.text
    assert [(c.start, c.frame_count) for c in chunks] == [
        (c.start, c.frame_count) for c in baseline
    ]


def test_a_request_swallowed_by_an_earlier_long_take_warns_and_is_skipped(tmp_path, caplog):
    """Two requests close enough to match the same boundary must not silently
    become last-one-wins: the second would overwrite the length the first was
    given, with nothing in the log to say the first direction was discarded."""
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.is_instrumental and c.start > 10.0)

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "swallowed",
            shot_lengths=(
                ShotLength(start=anchor, length_seconds=15.0, source_chunk_id=1),
                # Half a second later: inside the 15 s take above, with no
                # boundary of its own left to be given a length.
                ShotLength(start=anchor + 0.5, length_seconds=6.0, source_chunk_id=2),
            ),
        )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    assert "swallow" in caplog.text.lower()
    # The *first* request is the one that survives, not the last.
    shot = next(c for c in chunks if c.start == pytest.approx(anchor, abs=1e-6))
    assert shot.frame_count == GRID.trained_max_frames


def test_requests_are_ignored_with_a_warning_when_coverage_is_off(tmp_path, caplog):
    alignment = _gap_song()
    master = write_silent_wav(tmp_path / "m.wav", alignment.track_duration)

    with caplog.at_level(logging.WARNING):
        chunks = slice_audio(
            master,
            alignment,
            NARROW_HARDWARE,
            tmp_path / "nocover",
            cover_instrumentals=False,
            shot_lengths=(ShotLength(start=0.0, length_seconds=15.0),),
        )

    assert "instrumental_coverage" in caplog.text
    assert all((c.frame_count or 0) <= 192 for c in chunks)


# --------------------------------------------------------------------------- #
# Several requests in one plan: the timeline moves under them
# --------------------------------------------------------------------------- #

UNIFORM_HARDWARE = HardwareProfile(
    name="uniform-5.875s", vram_gb=24.0, min_chunk_seconds=5.875, max_chunk_seconds=5.875
)
"""A window exactly one grid point wide, so every chunk is 141 frames.

That makes the grid's coarseness visible: a 362-frame take cannot be carved
out of 141-frame chunks without a leftover too small to be its own chunk, so
honouring one request provably moves every boundary after it. A plan is
authored in one pass, against one chunk list, so the *second* request in that
plan is aimed at a timeline the first one has already moved -- which is the
case these two tests exist for.
"""


def _instrumental_song() -> AlignmentResult:
    """One short line, then two minutes of instrumental to direct."""
    segments = (
        make_aligned_segment(0, "one line and then the band plays", 0.0, 6.0, "Dianne"),
    )
    return AlignmentResult(segments=segments, track_duration=120.0)


def test_a_later_request_still_lands_after_an_earlier_one_moved_the_timeline(tmp_path):
    alignment = _instrumental_song()
    baseline = _slice(tmp_path, alignment, "baseline", profile=UNIFORM_HARDWARE)
    anchors = [c.start for c in baseline if c.is_instrumental]
    assert len(anchors) > 12

    chunks = _slice(
        tmp_path,
        alignment,
        "both",
        profile=UNIFORM_HARDWARE,
        shot_lengths=(
            ShotLength(start=anchors[2], length_seconds=15.0, source_chunk_id=3),
            ShotLength(start=anchors[10], length_seconds=15.0, source_chunk_id=11),
        ),
    )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    assert sum(1 for c in chunks if c.frame_count == GRID.trained_max_frames) == 2


def test_the_grid_residue_of_several_long_takes_does_not_lengthen_the_video(tmp_path):
    """Each merge leaves a few frames that do not land on the grid. Left to
    accumulate they run the covering past the end of the master (a silent tail)
    or short of it (an unrendered outro); the tail absorbs them instead."""
    alignment = _instrumental_song()
    baseline = _slice(tmp_path, alignment, "baseline", profile=UNIFORM_HARDWARE)
    anchors = [c.start for c in baseline if c.is_instrumental]

    chunks = _slice(
        tmp_path,
        alignment,
        "residue",
        profile=UNIFORM_HARDWARE,
        shot_lengths=(
            ShotLength(start=anchors[2], length_seconds=15.0),
            ShotLength(start=anchors[10], length_seconds=15.0),
        ),
    )

    one_step = GRID.frames_to_seconds(GRID.step_frames)
    assert sum(c.duration for c in chunks) == pytest.approx(
        sum(c.duration for c in baseline), abs=2 * one_step
    )


def test_lengths_re_anchored_onto_the_retiled_timeline_reproduce_it(tmp_path):
    """The ``--prepare --from-plan`` round trip: v1 anchors + the same lengths
    must re-cut to exactly v1 (issue #54 design section 5, step 4).

    Two anchor conventions reach ``slice_audio`` and the request cannot say
    which it is. A plan authored straight onto a fresh ``--prepare`` skeleton
    anchors against the *un-retiled* timeline -- the case
    ``test_a_later_request_still_lands_after_an_earlier_one_moved_the_timeline``
    covers. A plan re-anchored by ``--prepare --from-plan`` anchors against the
    *retiled* one, because those are the starts a render will actually produce
    and therefore the only ones that survive ``ShotPlanDriftError``.

    Reproduced before this was fixed, on the project's own measured 141-frame
    chunk size: three 15 s takes went in, and re-requesting them at their own
    starts gave back **two**, one of them 3.3 s from where it was authored.
    The accumulated shift correction was being applied to anchors that already
    carried it, so the third request missed every boundary by more than the
    1 s tolerance and was dropped with a warning nobody would connect to the
    plan they had just generated.
    """
    alignment = _instrumental_song()
    baseline = _slice(tmp_path, alignment, "baseline", profile=UNIFORM_HARDWARE)
    anchors = [c.start for c in baseline if c.is_instrumental]
    assert len(anchors) > 14

    v1 = _slice(
        tmp_path,
        alignment,
        "v1",
        profile=UNIFORM_HARDWARE,
        shot_lengths=tuple(
            ShotLength(start=anchors[i], length_seconds=15.0, source_chunk_id=i)
            for i in (2, 7, 12)
        ),
    )
    long_takes = [c for c in v1 if c.frame_count == GRID.trained_max_frames]
    assert len(long_takes) == 3

    v2 = _slice(
        tmp_path,
        alignment,
        "v2",
        profile=UNIFORM_HARDWARE,
        shot_lengths=tuple(
            ShotLength(start=c.start, length_seconds=15.0, source_chunk_id=c.chunk_id)
            for c in long_takes
        ),
    )

    assert [(c.start, c.frame_count) for c in v2] == [(c.start, c.frame_count) for c in v1]


def test_a_take_longer_than_the_remaining_track_is_shortened_not_invented(tmp_path, caplog):
    """A 15 s take asked for over the last 6 s of the song has to give way.

    Rendering the requested length would mean generating video past the end of
    the master -- a silent tail the mux cannot fill. The shot takes whatever
    the timeline actually has, and the log says so."""
    alignment = _instrumental_song()
    baseline = _slice(tmp_path, alignment, "baseline", profile=UNIFORM_HARDWARE)
    anchor = baseline[-1].start
    remaining = alignment.track_duration - anchor
    assert remaining < 15.0

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "overrun",
            profile=UNIFORM_HARDWARE,
            shot_lengths=(ShotLength(start=anchor, length_seconds=15.0, source_chunk_id=19),),
        )

    assert "of timeline remains" in caplog.text
    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    # The covering does not grow to accommodate a shot the track cannot hold.
    assert chunks[-1].end <= baseline[-1].end + _EPS


def test_a_long_take_at_the_end_of_the_track_is_not_rebalanced_away(tmp_path, caplog):
    """Absorbing the grid residue must never come out of the shot itself.

    A take anchored near the end leaves nothing after it to rebalance into, so
    the sweep has to decline and say so -- taking the frames back from the
    shot would silently undo the one thing the author asked for.
    """
    alignment = _instrumental_song()
    baseline = _slice(tmp_path, alignment, "baseline", profile=UNIFORM_HARDWARE)
    anchor = baseline[-3].start

    with caplog.at_level(logging.WARNING):
        chunks = _slice(
            tmp_path,
            alignment,
            "endtake",
            profile=UNIFORM_HARDWARE,
            shot_lengths=(ShotLength(start=anchor, length_seconds=15.0, source_chunk_id=17),),
        )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    assert chunks[-1].start == pytest.approx(anchor, abs=1e-6)
    assert chunks[-1].frame_count == GRID.trained_max_frames


# --------------------------------------------------------------------------- #
# The editorial payoff: a separate length for instrumental spans
# --------------------------------------------------------------------------- #


def test_instrumental_shot_seconds_gives_the_solo_longer_takes(tmp_path):
    alignment = _gap_song()
    default_chunks = _slice(tmp_path, alignment, "default")
    long_fill = _slice(tmp_path, alignment, "longfill", instrumental_shot_seconds=15.0)

    _assert_grid_exact(long_fill)
    _assert_contiguous_from_zero(long_fill)

    def in_gap(chunks):
        return [c for c in chunks if 6.0 < c.start < 40.0 and c.is_instrumental]

    assert len(in_gap(long_fill)) < len(in_gap(default_chunks))
    assert max(c.frame_count for c in in_gap(long_fill)) > 192
    # Sung chunks keep the short window they were given.
    for chunk in long_fill:
        if not chunk.is_instrumental:
            assert chunk.frame_count <= 192


def test_instrumental_shot_seconds_is_clamped_into_the_trained_range(tmp_path, caplog):
    alignment = _gap_song()
    with caplog.at_level(logging.WARNING):
        chunks = _slice(tmp_path, alignment, "clampfill", instrumental_shot_seconds=30.0)
    assert "15.083" in caplog.text
    _assert_grid_exact(chunks)


# --------------------------------------------------------------------------- #
# The whole loop, the way it is actually authored
# --------------------------------------------------------------------------- #


def test_plan_file_to_retiled_timeline(tmp_path):
    """The authoring loop end to end: slice once to see the chunks, write the
    plan against those starts, slice again with the plan's lengths.

    This is the seam worth an explicit test -- ``start`` is the anchor on both
    sides, and a plan is authored by copying the emitted chunk starts, so the
    two halves have to agree on what a start *is*.
    """
    alignment = _gap_song()
    first_pass = _slice(tmp_path, alignment, "pass1")
    solo = next(c for c in first_pass if c.is_instrumental and c.start > 10.0)

    plan_path = _write_plan(
        tmp_path,
        f"""
[[shot]]
chunk_id = {first_pass[0].chunk_id}
start = {first_pass[0].start}
shot = "She steps out of her front door into an ordinary morning"

[[shot]]
chunk_id = {solo.chunk_id}
start = {solo.start}
length_seconds = 15.0
shot = "One unbroken crane move down the whole length of the empty street"
""",
    )
    requests = shot_length_requests(load_shot_plan(plan_path))
    assert [r.source_chunk_id for r in requests] == [solo.chunk_id]

    second_pass = _slice(tmp_path, alignment, "pass2", shot_lengths=requests)

    _assert_grid_exact(second_pass)
    _assert_contiguous_from_zero(second_pass)
    directed = next(c for c in second_pass if c.start == pytest.approx(solo.start, abs=1e-6))
    assert directed.frame_count == GRID.trained_max_frames
    assert len(second_pass) < len(first_pass)


# --------------------------------------------------------------------------- #
# Nothing a long take does may cost lyric content
# --------------------------------------------------------------------------- #


def _seg_with_words(index, text, start, end, character="Dianne"):
    parts = text.split()
    step = (end - start) / len(parts)
    words = tuple(
        WordTiming(word=w, start=start + i * step, end=start + (i + 1) * step)
        for i, w in enumerate(parts)
    )
    return AlignedSegment(
        index=index, text=text, start=start, end=end, words=words, characters=(character,)
    )


def test_every_lyric_word_still_appears_exactly_once_after_a_long_take(tmp_path):
    words = "one two three four five six seven eight nine ten eleven twelve"
    segments = (
        _seg_with_words(0, words, 8.0, 20.0),
    )
    alignment = AlignmentResult(segments=segments, track_duration=45.0)

    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = default_chunks[0].start  # the intro filler

    chunks = _slice(
        tmp_path,
        alignment,
        "long",
        shot_lengths=(ShotLength(start=anchor, length_seconds=15.0),),
    )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    assert " ".join(c.text for c in chunks).split() == words.split()


def test_a_long_take_over_a_lyric_still_covers_its_audio_exactly(tmp_path):
    """The stem handed to H3 must still be exactly ``frame_count`` long, and
    the chunk must still sit where its audio does -- the #20 invariant that
    makes lip-sync work at all."""
    segments = (
        make_aligned_segment(0, "a line that runs a while", 6.0, 13.0, "Dianne"),
    )
    alignment = AlignmentResult(segments=segments, track_duration=40.0)
    default_chunks = _slice(tmp_path, alignment, "default")
    anchor = next(c.start for c in default_chunks if c.start >= 5.0)

    chunks = _slice(
        tmp_path,
        alignment,
        "long",
        shot_lengths=(ShotLength(start=anchor, length_seconds=15.0),),
    )

    _assert_grid_exact(chunks)
    _assert_contiguous_from_zero(chunks)
    for chunk in chunks:
        assert chunk.audio_file.exists()
