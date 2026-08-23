"""The world-state continuity axis (issue #83), checked on the beat sheet.

**The third axis, and the open design question, settled.** ``setting`` (#32)
fixes *what world this is* and is composed unchanged into every prompt.
``location`` (#78) fixes *where inside it anyone is* and is authored per
beat. Neither carries *what the world looks like right now* -- weather,
light, season, and the persistent aftermath of events the video has already
shown. A viewer on the second full "Deathless" render: *"4:44 buildings
explode (fine), 4:48 Jan and Dianne are climbing snow with no debris, 4:55 he
is back on non-snowy tower"* -- three consecutive chunks, about fifteen
seconds, in which the world gains snow, loses an explosion that just
happened, and loses the snow again.

Issue #83 asks whether this is a second enumerated axis beside ``location``
or whether both are facets of one "world state at chunk N" record. **It is a
second axis, and the argument is measured, not aesthetic: the two have
opposite correctness shapes.** Run the regression check below over the
``location`` values of the real 80-chunk plans and it fires 10 times on
``shot_plan_v6.toml`` and 10 on ``shot_plan_v12.toml``, every one of them a
false positive -- because a character is *supposed* to move back and forth
between places, and the world is *not* supposed to move back and forth
between states. One record would have to carry both rules anyway, and a
combined vocabulary multiplies out (this song's 6 locations x 4 conditions is
24 tags a generator has to keep consistent) which is the enumeration
explosion #78 already warns about.

What is genuinely one record is the *machine underneath*:
:mod:`~music_video_maker.authoring.worldstate` already models an
irreversible fact through exactly one write choke point, and the regression
check here is that choke point rather than a second copy of it -- see
:func:`check_conditions`.

**Nothing here reads English.** Like ``worldstate.check_location_tags``, both
checks compare *structures*: a run of spans that is interrupted for exactly
one span and comes straight back, and a value written after an irreversible
fact ended it. There is no vocabulary of weather words anywhere in this
module, and there must never be -- inferring meaning from arbitrary English
has failed every time it was tried in this codebase.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.worldstate import (
    IrreversibleFactViolation,
    WorldState,
)

logger = logging.getLogger(__name__)

FLIP_FLOP = "flip_flop"
"""A state that holds, is interrupted for exactly ONE span, and comes
straight back."""

REGRESSION = "regression"
"""A state that a `consequence` beat ended, returning later. Debris does not
un-explode."""

_WORLD_ENTITY_ID = "world"
_FACT_KEY_PREFIX = "conditions:"
"""One fact key per condition VALUE, holding ``"current"`` or ``"ended"``.

Deliberately not a single ``"conditions"`` key holding the current value:
that would make :class:`~...worldstate.IrreversibleFactViolation` fire on
*any* change after a consequence, including a further consequence of it (ash
settling into a flat grey stillness), which is not a regression and is not a
defect. Keyed per value, the irreversible fact says exactly the true thing --
*this particular state is over* -- and only a return to it violates."""

_CURRENT = "current"
_ENDED = "ended"


@dataclass(frozen=True)
class ConditionSpan:
    """One stretch of story time the caller has tagged with a world state.

    ``ref`` is **opaque**: carried into a finding untouched and never read,
    parsed or compared here -- the same contract ``worldstate.LocatedSpan``
    and ``Event.causes`` document. In practice it is a ``chunk_id``, and this
    module deliberately does not know that. ``at_t`` stays bare story-time
    seconds, the axis that survives the caller re-cutting its own chunks (a
    ``length_seconds`` re-anchor moves every chunk id and no timestamp).

    ``is_consequence`` is the caller's own classification of whether the beat
    opening this span is a `consequence` -- the beat sheet's word, not a
    guess made here from the text.
    """

    ref: Any
    at_t: float
    conditions: str
    is_consequence: bool = False


def conditions_from_beats(beats: Sequence[Beat]) -> tuple[ConditionSpan, ...]:
    """The adapter from a beat sheet to :func:`check_conditions`'s opaque
    span interface -- one :class:`ConditionSpan` per beat, sorted by
    ``at_t``.

    ``check_conditions`` is deliberately written against ``ConditionSpan``
    rather than :class:`.beats.Beat` directly, the same contract
    :class:`~...worldstate.LocatedSpan` documents for
    :func:`~...worldstate.check_location_tags`: the check never learns the
    caller's units, so it stays a pure structural test of a sequence of
    (time, value, is_consequence) triples and cannot drift into guessing
    meaning from a beat's other fields. This module is the one place that
    *does* know a chunk id is what belongs in ``ref`` -- ``beats.py`` itself
    stays ignorant of ``ConditionSpan`` entirely, the same separation
    ``.beats`` keeps from ``.reanchor``.

    ``is_consequence`` is read straight off ``beat_role`` -- the beat
    sheet's own classification (issue #83's rule 3c(ii): a state that
    ARRIVES on a `consequence` beat is that consequence's one-way aftermath),
    never inferred from ``conditions`` text."""
    return tuple(
        sorted(
            (
                ConditionSpan(
                    ref=beat.chunk_id,
                    at_t=beat.start,
                    conditions=beat.conditions,
                    is_consequence=(beat.beat_role == "consequence"),
                )
                for beat in beats
            ),
            key=lambda span: span.at_t,
        )
    )


@dataclass(frozen=True)
class ConditionFinding:
    """One thing :func:`check_conditions` objected to.

    Carries **one hop of causality** -- ``against``, the ref of the span that
    makes this a contradiction -- for the same reason
    ``worldstate.Contradiction`` and ``StaleLocationTag`` do: the reviewer's
    only decision is "which of these two is wrong", and that is undecidable
    without knowing what the other one was.
    """

    kind: str
    ref: Any
    conditions: str
    message: str
    against: Any = None
    at_t: float = 0.0


@dataclass(frozen=True)
class _Run:
    value: str
    at_t: float
    refs: tuple[Any, ...]
    opens_on_consequence: bool


def _runs(spans: Sequence[ConditionSpan]) -> tuple[_Run, ...]:
    """Collapse consecutive same-value spans into runs, in story-time order.

    Blank tags are dropped rather than treated as a state of their own: an
    untagged span makes no claim about the world, so it must neither fire nor
    silently bridge the runs either side into one. Dropping is what leaves
    ``clear -> snow -> (blank) -> clear`` reading as the single-span
    interruption it is.
    """
    tagged = [s for s in spans if s.conditions and s.conditions.strip()]
    ordered = sorted(tagged, key=lambda s: s.at_t)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if earlier.at_t == later.at_t:
            raise ValueError(
                f"two condition spans share the same story time t={earlier.at_t} "
                f"({earlier.ref!r} and {later.ref!r}); story time is the axis this "
                "check orders by and guessing an order would silently change which "
                "state 'came back'"
            )

    runs: list[_Run] = []
    for span in ordered:
        value = span.conditions.strip()
        if runs and runs[-1].value == value:
            runs[-1] = _Run(
                value=value,
                at_t=runs[-1].at_t,
                refs=(*runs[-1].refs, span.ref),
                opens_on_consequence=runs[-1].opens_on_consequence,
            )
            continue
        runs.append(
            _Run(
                value=value,
                at_t=span.at_t,
                refs=(span.ref,),
                opens_on_consequence=bool(span.is_consequence),
            )
        )
    return tuple(runs)


def _flip_flops(runs: Sequence[_Run]) -> list[ConditionFinding]:
    """A -> B -> A where B lasts exactly one span.

    **Deliberately narrower than "A -> B -> A" in general**, and the width was
    chosen by measurement rather than taste -- see :func:`check_conditions`.
    """
    findings: list[ConditionFinding] = []
    for before, middle, after in zip(runs, runs[1:], runs[2:], strict=False):
        if before.value != after.value or len(middle.refs) != 1:
            continue
        findings.append(
            ConditionFinding(
                kind=FLIP_FLOP,
                ref=middle.refs[0],
                conditions=middle.value,
                against=before.refs[-1],
                at_t=middle.at_t,
                message=(
                    f"the world is {before.value!r}, becomes {middle.value!r} for this one "
                    f"shot, and is {after.value!r} again immediately afterwards (issue #83). "
                    "Weather, light and the aftermath of what has already happened are "
                    "continuity, not per-shot dressing: a viewer reads a state that arrives "
                    "and leaves inside fifteen seconds as a mistake, not as weather. Pick "
                    "one of the two and let it run across all three shots, or -- if the "
                    "world really does change here -- give the change its own beat so it "
                    "reads as something that happened."
                ),
            )
        )
    return findings


def _regressions(runs: Sequence[_Run]) -> list[ConditionFinding]:
    """A state a `consequence` beat ended, returning later.

    Built on :mod:`~music_video_maker.authoring.worldstate` rather than
    beside it: the rule "an irreversible fact is never contradicted" already
    holds for that module's whole log *by construction*, enforced at its one
    write choke point, so this function does not re-implement the rule -- it
    writes the beat sheet into the log and reports what the choke point
    refuses. A second implementation would drift from it within a month, the
    same reason ``authoring.plan.check_plan`` runs the render's own loaders
    rather than a copy of them.
    """
    if not runs:
        return []

    world = WorldState()
    world.create_entity(
        kind="world",
        name="the world",
        created_at_t=runs[0].at_t,
        entity_id=_WORLD_ENTITY_ID,
    )
    world.set_fact(
        entity_id=_WORLD_ENTITY_ID,
        key=f"{_FACT_KEY_PREFIX}{runs[0].value}",
        value=_CURRENT,
        valid_from_t=runs[0].at_t,
    )

    ended_by: dict[str, _Run] = {}
    findings: list[ConditionFinding] = []
    for previous, run in zip(runs, runs[1:], strict=False):
        world.set_fact(
            entity_id=_WORLD_ENTITY_ID,
            key=f"{_FACT_KEY_PREFIX}{previous.value}",
            value=_ENDED,
            valid_from_t=run.at_t,
            irreversible=run.opens_on_consequence,
        )
        if run.opens_on_consequence:
            ended_by[previous.value] = run
        try:
            world.set_fact(
                entity_id=_WORLD_ENTITY_ID,
                key=f"{_FACT_KEY_PREFIX}{run.value}",
                value=_CURRENT,
                valid_from_t=run.at_t,
            )
        except IrreversibleFactViolation:
            ender = ended_by.get(run.value)
            findings.append(
                ConditionFinding(
                    kind=REGRESSION,
                    ref=run.refs[0],
                    conditions=run.value,
                    against=ender.refs[0] if ender is not None else None,
                    at_t=run.at_t,
                    message=(
                        f"the world goes back to {run.value!r}, which a consequence beat "
                        f"ended at {ender.at_t:.3f}s"
                        + (f" (chunk {ender.refs[0]})" if ender is not None else "")
                        + " (issue #83). The aftermath of something the video has already "
                        "shown is one-way: debris does not un-explode, a burnt valley does "
                        "not green over, and a shot that puts the world back the way it was "
                        "reads as the render having forgotten. If the world genuinely "
                        "recovers here, that recovery is itself an event and belongs on a "
                        "beat of its own -- and a human still has to agree, which is why "
                        "this is a warning."
                    ),
                )
            )
    return findings


def check_conditions(spans: Sequence[ConditionSpan]) -> tuple[ConditionFinding, ...]:
    """Both world-state continuity checks over one beat sheet's tags.

    Warning tier, always: never raises on a finding, and a caller must never
    block a run on one. (It *does* raise :class:`ValueError` on two spans
    sharing a story time, which is a caller bug the chunk timeline cannot
    produce, not a finding about the sheet.)

    **Scoring, and what it does and does not measure.** No song has been
    authored with ``conditions`` yet, so there is no corpus of real values to
    score against -- and inventing one by reading weather out of finished
    shot lines would be exactly the guessing this module refuses to do. What
    does exist is the only other enumerated per-chunk tag this project has:
    ``location``, 80 chunks each on ``shot_plan_v6.toml`` and
    ``shot_plan_v12.toml``. Running both checks over *that* axis measures
    false-positive surface -- how often the SHAPE occurs in a real authored
    sequence of tags -- and measures nothing at all about detection.

    ===============================================  ======  ======
    check, run over the ``location`` axis            v6      v12
    ===============================================  ======  ======
    flip-flop, interruption of exactly one span      3       3
    flip-flop, interruption of any length            6       5
    regression, any earlier value after any
    consequence (the loose form, REJECTED)           11      10
    regression, only a value a consequence ended
    (the form that ships)                            0       0
    ===============================================  ======  ======

    Read the first and last rows together and they are the argument for two
    axes rather than one record: the loose regression form fires 10-11 times
    on locations and every one is correct authoring, while the shipped form
    fires zero. The single-span flip-flop occurs 3 times in 80 chunks and the
    any-length form twice as often, which is why the narrow one ships -- and
    all three of those 3 are legitimate *for a location* (the cut goes to the
    watch-post for one shot and comes back), which is precisely the asymmetry
    that makes conditions a different axis.

    Detection is demonstrated by the viewer-reported sequence in
    ``tests/test_authoring_conditions.py``, and confirming it on a render
    needs the GPU: author ``conditions`` on "Deathless", re-render chunks
    44-46 with ``--only-chunks 44,45,46``, and check that the snow no longer
    arrives and leaves inside fifteen seconds. Note that an automated pixel
    check cannot stand in for that here -- a white-and-desaturated pixel
    fraction over the v12 renders of chunks 43-47 swings from 0.0% to 12.7%
    *within chunk 44 alone*, because the blast's own white light is
    indistinguishable from snow to any such measure. The state has to be
    authored and checked as a tag; it cannot be recovered from the pixels
    afterwards.
    """
    runs = _runs(spans)
    findings = [*_flip_flops(runs), *_regressions(runs)]
    findings.sort(key=lambda f: (f.at_t, f.kind))
    for finding in findings:
        logger.info(
            "World-state continuity (%s) at t=%.3fs on %r: %s",
            finding.kind,
            finding.at_t,
            finding.ref,
            finding.conditions,
        )
    return tuple(findings)


__all__ = [
    "FLIP_FLOP",
    "REGRESSION",
    "ConditionFinding",
    "ConditionSpan",
    "check_conditions",
    "conditions_from_beats",
]
