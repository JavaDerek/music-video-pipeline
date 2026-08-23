"""Issue #83: the world-state continuity axis, checked on the beat sheet.

Two lints, both warning tier, both scored by the architect on the only real
corpus of enumerated per-chunk tags this project has (`location`, 80 chunks x
2 plans) -- see ``check_conditions``'s own docstring for the numbers and for
what that surrogate does and does not measure.
"""

from __future__ import annotations

import pytest

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.conditions import (
    FLIP_FLOP,
    REGRESSION,
    ConditionFinding,
    ConditionSpan,
    check_conditions,
    conditions_from_beats,
)


def _spans(*rows: tuple[int, float, str, bool]) -> tuple[ConditionSpan, ...]:
    return tuple(
        ConditionSpan(ref=ref, at_t=at_t, conditions=value, is_consequence=is_c)
        for ref, at_t, value, is_c in rows
    )


# --------------------------------------------------------------------------- #
# Flip-flop: A -> B -> A where B lasts exactly one span
# --------------------------------------------------------------------------- #


def test_a_single_span_interruption_that_comes_straight_back_is_reported():
    """The viewer's own case, in the shape it was reported: 4:44 buildings
    explode, 4:48 they are climbing snow with no debris, 4:55 he is back on a
    non-snowy tower."""
    findings = check_conditions(
        _spans(
            (44, 282.6, "smoke and settling debris", False),
            (45, 288.0, "heavy falling snow", False),
            (46, 293.5, "smoke and settling debris", False),
        )
    )
    assert [f.kind for f in findings] == [FLIP_FLOP]
    assert findings[0].ref == 45
    assert "heavy falling snow" in findings[0].message
    assert "smoke and settling debris" in findings[0].message


def test_a_state_that_simply_changes_and_stays_changed_is_silent():
    findings = check_conditions(
        _spans(
            (1, 0.0, "clear pre-dawn light", False),
            (2, 8.0, "heavy falling snow", False),
            (3, 16.0, "heavy falling snow", False),
            (4, 24.0, "flat grey daylight", False),
        )
    )
    assert findings == ()


def test_an_interruption_longer_than_one_span_is_not_reported():
    """Deliberately narrower than "A -> B -> A" in general. Measured on the
    location axis of the real plans, the any-length form fires twice as often
    as the single-span form (6 against 3 across shot_plan_v6/v12), and the
    defect this exists for is the single-span one."""
    findings = check_conditions(
        _spans(
            (1, 0.0, "clear", False),
            (2, 8.0, "snow", False),
            (3, 16.0, "snow", False),
            (4, 24.0, "clear", False),
        )
    )
    assert findings == ()


def test_the_same_state_recurring_much_later_is_silent():
    findings = check_conditions(
        _spans(
            (1, 0.0, "clear", False),
            (2, 8.0, "clear", False),
            (3, 16.0, "snow", False),
            (4, 24.0, "snow", False),
            (5, 32.0, "snow", False),
            (6, 40.0, "clear", False),
        )
    )
    assert findings == ()


# --------------------------------------------------------------------------- #
# Regression: a state that arrived on a `consequence` beat is one-way
# --------------------------------------------------------------------------- #


def test_going_back_to_a_state_a_consequence_ended_is_reported():
    findings = check_conditions(
        _spans(
            (10, 0.0, "green valley under low cloud", False),
            (20, 100.0, "burnt ground and drifting ash", True),
            (30, 200.0, "burnt ground and drifting ash", False),
            (40, 300.0, "green valley under low cloud", False),
        )
    )
    assert [f.kind for f in findings] == [REGRESSION]
    assert findings[0].ref == 40
    assert findings[0].against == 20
    assert "green valley under low cloud" in findings[0].message


def test_a_further_change_after_a_consequence_is_not_a_regression():
    """Only a return to a state the consequence ENDED is one-way. The world
    is allowed to keep changing -- ash can settle into a flat grey stillness
    without anything un-happening."""
    findings = check_conditions(
        _spans(
            (10, 0.0, "green valley under low cloud", False),
            (20, 100.0, "burnt ground and drifting ash", True),
            (25, 150.0, "burnt ground and drifting ash", False),
            (30, 200.0, "flat grey stillness", False),
        )
    )
    assert findings == ()


def test_a_state_that_ended_on_an_ordinary_beat_may_come_back():
    """Weather is reversible unless the beat sheet says an event ended it.
    This is what stops the check firing on every recurrence."""
    findings = check_conditions(
        _spans(
            (10, 0.0, "heavy falling snow", False),
            (20, 100.0, "clear cold light", False),
            (25, 150.0, "clear cold light", False),
            (30, 200.0, "heavy falling snow", False),
        )
    )
    assert findings == ()


def test_a_deliberate_return_authored_on_its_own_consequence_still_reports():
    """The preamble tells the author to put a genuine return on a
    `consequence` beat of its own. That makes the return part of the story --
    but it does not un-freeze what an earlier consequence ended, so the
    finding still stands and a human decides. Warning tier exists for exactly
    this."""
    findings = check_conditions(
        _spans(
            (10, 0.0, "green valley", False),
            (20, 100.0, "burnt ground", True),
            (25, 150.0, "burnt ground", False),
            (30, 200.0, "green valley", True),
        )
    )
    assert [f.kind for f in findings] == [REGRESSION]


def test_both_lints_can_fire_on_one_sheet_and_are_reported_together():
    findings = check_conditions(
        _spans(
            (1, 0.0, "clear", False),
            (2, 8.0, "snow", False),
            (3, 16.0, "clear", False),
            (4, 24.0, "burnt ground", True),
            (5, 32.0, "clear", False),
        )
    )
    assert {f.kind for f in findings} == {FLIP_FLOP, REGRESSION}
    assert all(isinstance(f, ConditionFinding) for f in findings)


# --------------------------------------------------------------------------- #
# Degradation and edges
# --------------------------------------------------------------------------- #


def test_no_spans_and_untagged_spans_are_silent_not_an_error():
    assert check_conditions(()) == ()
    assert check_conditions(_spans((1, 0.0, "", False), (2, 8.0, "   ", False))) == ()


def test_untagged_spans_are_skipped_without_joining_the_runs_either_side():
    """An untagged span carries no claim about the world, so it must neither
    fire nor silently bridge two runs into one. Here the tagged spans are
    clear -> snow -> clear with a blank in the middle: the blank is dropped
    and what remains is a single-span interruption, which is a finding."""
    findings = check_conditions(
        _spans(
            (1, 0.0, "clear", False),
            (2, 8.0, "snow", False),
            (3, 16.0, "", False),
            (4, 24.0, "clear", False),
        )
    )
    assert [f.kind for f in findings] == [FLIP_FLOP]
    assert findings[0].ref == 2


def test_spans_are_ordered_by_story_time_not_by_the_order_given():
    findings = check_conditions(
        _spans(
            (46, 293.5, "smoke", False),
            (44, 282.6, "smoke", False),
            (45, 288.0, "snow", False),
        )
    )
    assert [f.ref for f in findings] == [45]


def test_a_single_tagged_span_is_silent():
    assert check_conditions(_spans((1, 0.0, "snow", False))) == ()


def test_two_spans_sharing_a_start_time_are_refused_loudly():
    """Story time is the axis everything here is ordered by. Two spans at the
    same t is a caller bug -- the chunk timeline cannot produce it -- and
    guessing an order would silently change which state 'came back'."""
    with pytest.raises(ValueError, match="same story time"):
        check_conditions(
            _spans((1, 8.0, "clear", False), (2, 8.0, "snow", False))
        )


def test_findings_carry_the_offending_spans_own_story_time():
    findings = check_conditions(
        _spans(
            (44, 282.6, "smoke", False),
            (45, 288.0, "snow", False),
            (46, 293.5, "smoke", False),
        )
    )
    assert findings[0].at_t == pytest.approx(288.0)


# --------------------------------------------------------------------------- #
# conditions_from_beats -- the one place that knows a chunk id is what goes
# in ConditionSpan.ref. check_conditions itself stays opaque to the caller's
# units (the same contract worldstate.LocatedSpan documents); this is the
# adapter between a beat sheet and that opaque interface.
# --------------------------------------------------------------------------- #


def _beat(chunk_id, start, end, *, role="transition", conditions=""):
    return Beat(
        chunk_id=chunk_id,
        start=start,
        end=end,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=1,
        location="the room",
        conditions=conditions,
    )


def test_conditions_from_beats_maps_chunk_id_start_and_conditions():
    beats = (
        _beat(44, 282.58, 288.0, conditions="smoke and settling debris"),
        _beat(45, 288.0, 293.5, conditions="heavy falling snow"),
    )

    spans = conditions_from_beats(beats)

    assert spans == (
        ConditionSpan(ref=44, at_t=282.58, conditions="smoke and settling debris"),
        ConditionSpan(ref=45, at_t=288.0, conditions="heavy falling snow"),
    )


def test_conditions_from_beats_marks_only_consequence_beats():
    beats = (
        _beat(1, 0.0, 6.0, role="plant", conditions="green valley"),
        _beat(2, 6.0, 12.0, role="consequence", conditions="burnt ground"),
        _beat(3, 12.0, 18.0, role="transition", conditions="burnt ground"),
    )

    spans = conditions_from_beats(beats)

    assert [s.is_consequence for s in spans] == [False, True, False]


def test_conditions_from_beats_sorts_by_start_time_not_input_order():
    beats = (
        _beat(2, 6.0, 12.0, conditions="snow"),
        _beat(1, 0.0, 6.0, conditions="clear"),
    )

    spans = conditions_from_beats(beats)

    assert [s.ref for s in spans] == [1, 2]
    assert [s.at_t for s in spans] == [0.0, 6.0]


def test_conditions_from_beats_of_an_empty_sheet_is_empty():
    assert conditions_from_beats(()) == ()


def test_conditions_from_beats_feeds_check_conditions_end_to_end():
    """The viewer's own reported case (issue #83), built through the real
    adapter rather than hand-built ConditionSpan objects."""
    beats = (
        _beat(44, 282.6, 288.0, conditions="smoke and settling debris"),
        _beat(45, 288.0, 293.5, conditions="heavy falling snow"),
        _beat(46, 293.5, 299.0, conditions="smoke and settling debris"),
    )

    findings = check_conditions(conditions_from_beats(beats))

    assert [f.kind for f in findings] == [FLIP_FLOP]
    assert findings[0].ref == 45
