"""Re-anchoring beats onto the timeline their own lengths produced
(issue #54 design section 5).

**The one genuinely hard part of the authoring layer**, and the design says so:
"the piece most likely to be subtly wrong". If Stage 2 asks for a 9-second
take, ``slice_audio`` re-tiles from that anchor and every chunk after it gets
a new id and a new start. A beat sheet authored against the pre-length
skeleton describes a timeline that no longer exists.

So the flow is:

1. ``--prepare`` -> ``chunks_v0``. Concept and beats author against it.
2. If the sheet emitted any ``length_seconds``: re-slice with those requests
   -> ``chunks_v1``.
3. **Re-anchor by song time, never by chunk id** -- each beat maps to the
   ``v1`` chunk containing its midpoint. Same rule ``ShotLength`` already
   follows, for the same reason: a time in the song is stable across a
   re-slice and a chunk number is not.
4. Freeze ``chunks_v1``. Everything downstream runs against it.

Getting step 3 wrong does not raise anything. It silently attaches shot 12's
direction to shot 9's audio, which is precisely the failure
``ShotPlanDriftError`` exists to make loud -- so this module is a pure
function of a beat list and a chunk list, with no I/O, no model and no
slicing anywhere near it, and it has its own test module built on hand-made
timelines.

Many-to-one is *expected*: that is what a long take is. The merge rules are
stated in :func:`reanchor_beats`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from music_video_maker.authoring.beats import (
    FOCUS_ACTION,
    Beat,
    BeatsResult,
    BeatsValidationError,
    beat_length_requests,
    generate_beats,
)
from music_video_maker.authoring.driver import ModelDriver
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk
from music_video_maker.shot_plan import ShotLength

logger = logging.getLogger(__name__)

_ROLE_PRECEDENCE: tuple[str, ...] = (
    "instrumental",
    "transition",
    "plant",
    "contact",
    "consequence",
)
"""Which role a merged beat takes: the most consequential of its members.

A long take that swallowed a plant and its consequence is, on screen, the
consequence -- that is what the audience sees change state. Demoting it to
"plant" would also lose the ``focus = "action"`` that a consequence must
carry, which is the whole reason ``focus`` is per-chunk state."""

MAX_REANCHOR_ROUNDS = 2
"""How many times :func:`plan_beats` will send an uncovered timeline back to
the model before giving up (design section 6's "bounded at 2 rounds, then
abort"). Each round is a whole model call; grinding this is how a review loop
turns into a slot machine."""


@dataclass(frozen=True)
class ReanchorResult:
    """What re-anchoring a beat sheet onto a new timeline produced."""

    beats: tuple[Beat, ...]
    unmapped_chunk_ids: tuple[int, ...]
    """Chunks of the new timeline no beat mapped into. **Reported, never
    filled in**: a blank direction falls back to the global
    ``narrative_concept``, which renders as something deliberate-looking that
    nobody authored (design section 5, step 3)."""

    merges: tuple[tuple[int, tuple[int, ...]], ...]
    """``(new chunk_id, the old chunk ids that merged into it)``, for every
    chunk that took more than one beat. Log/audit only."""


def _containing_chunk(chunks: Sequence[AudioChunk], moment: float) -> AudioChunk:
    """The chunk whose half-open span ``[start, end)`` holds ``moment``.

    Clamped to the first/last chunk rather than raising, because the retiled
    timeline can be a fraction shorter than the one the beats were authored
    against -- this project already records a 0.28 s outro tail deficit -- and
    losing an authored beat to a rounding error would be worse than attaching
    it to the chunk next door.
    """
    for chunk in chunks:
        if chunk.start <= moment < chunk.end:
            return chunk
    if moment < chunks[0].start:
        return chunks[0]
    return chunks[-1]


def reanchor_beats(
    beats: Sequence[Beat], chunks: Sequence[AudioChunk]
) -> ReanchorResult:
    """Map ``beats`` onto ``chunks`` by song time and merge what collides.

    ``beats`` carry the span they were authored against, so the timeline they
    came *from* is not an argument here -- it is already in the data, and
    passing it separately would create a second source of truth for the same
    fact.

    The merge rules, all four of which exist because losing them silently
    changes what renders:

    * **text** -- members joined in chunk order, so the long take still
      contains every beat it swallowed;
    * **role** -- the most consequential member wins (:data:`_ROLE_PRECEDENCE`);
    * **focus** -- ``action`` if any member had it, or was a consequence;
    * **length** -- the first member that requested one. A later member's
      request was swallowed by the earlier take and is warned about, the same
      event ``slicing`` already warns about from the other side;
    * **location** (issue #78) -- the first member's, the same "first member
      wins" rule as ``beat_group``, and warned about the same way when
      members disagree: a vanished location boundary is exactly the kind of
      thing this field exists to make loud rather than silently merge away.
    """
    if not chunks:
        raise ValueError("cannot re-anchor beats onto an empty chunk timeline")

    buckets: dict[int, list[Beat]] = {}
    for beat in sorted(beats, key=lambda b: b.start):
        target = _containing_chunk(chunks, (beat.start + beat.end) / 2.0)
        buckets.setdefault(target.chunk_id, []).append(beat)

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    merged: list[Beat] = []
    merges: list[tuple[int, tuple[int, ...]]] = []

    for chunk in chunks:
        members = buckets.get(chunk.chunk_id)
        if not members:
            continue
        merged.append(_merge(members, by_id[chunk.chunk_id]))
        if len(members) > 1:
            merges.append((chunk.chunk_id, tuple(m.chunk_id for m in members)))

    unmapped = tuple(chunk.chunk_id for chunk in chunks if chunk.chunk_id not in buckets)

    if merges:
        logger.info(
            "Re-anchored %d beat(s) onto %d chunk(s); %d long take(s) merged: %s",
            len(beats),
            len(chunks),
            len(merges),
            ", ".join(f"chunk {new} <- {list(old)}" for new, old in merges),
        )
    if unmapped:
        logger.warning(
            "Re-anchoring left %d chunk(s) with no beat: %s. Honouring the sheet's own "
            "length_seconds re-cut the timeline into spans no authored beat covers.",
            len(unmapped),
            list(unmapped),
        )

    return ReanchorResult(
        beats=tuple(merged), unmapped_chunk_ids=unmapped, merges=tuple(merges)
    )


def _merge(members: Sequence[Beat], chunk: AudioChunk) -> Beat:
    first = members[0]
    if len(members) == 1:
        return replace(first, chunk_id=chunk.chunk_id, start=chunk.start, end=chunk.end)

    groups = {member.beat_group for member in members}
    if len(groups) > 1:
        logger.warning(
            "Chunk %d merges beats from %d different beat group(s) (%s); keeping group %d. "
            "A vanished group boundary is how a three-beat gag becomes one compressed "
            "shot, which is the defect docs/shot-writing-guide.md exists to prevent.",
            chunk.chunk_id,
            len(groups),
            sorted(groups),
            first.beat_group,
        )

    with_length = [m for m in members if m.length_seconds is not None]
    if len(with_length) > 1:
        logger.warning(
            "Chunk %d merges %d beats that each asked for a length (%s); keeping "
            "chunk %d's %.3fs. Two takes cannot both be on screen -- the later request "
            "was swallowed by the earlier, longer one.",
            chunk.chunk_id,
            len(with_length),
            [m.chunk_id for m in with_length],
            with_length[0].chunk_id,
            with_length[0].length_seconds,
        )

    locations = {member.location for member in members}
    if len(locations) > 1:
        logger.warning(
            "Chunk %d merges beats naming %d different location(s) (%s); keeping %r "
            "(issue #78). A vanished location boundary silently attaches one place's "
            "beat to another's -- if this long take really does cross two places, "
            "consider a shorter length_seconds instead so the boundary survives.",
            chunk.chunk_id,
            len(locations),
            sorted(locations),
            first.location,
        )

    role = max(
        (m.beat_role for m in members),
        key=lambda r: _ROLE_PRECEDENCE.index(r) if r in _ROLE_PRECEDENCE else -1,
    )
    focus = (
        FOCUS_ACTION
        if any(m.focus == FOCUS_ACTION or m.beat_role == "consequence" for m in members)
        else first.focus
    )
    return Beat(
        chunk_id=chunk.chunk_id,
        start=chunk.start,
        end=chunk.end,
        beat="; ".join(m.beat for m in members),
        beat_role=role,
        beat_group=first.beat_group,
        location=first.location,
        focus=focus,
        length_seconds=with_length[0].length_seconds if with_length else None,
        merged_from=tuple(m.chunk_id for m in members),
    )


# --------------------------------------------------------------------------- #
# The whole of Stage 2: generate, re-cut, re-anchor, and check the coverage
# --------------------------------------------------------------------------- #

ResliceFn = Callable[[Sequence[ShotLength]], Sequence[AudioChunk]]
"""How :func:`plan_beats` re-cuts the timeline. Injected rather than called
directly so the loop is testable with no audio, no ffmpeg and no pydub --
the same seam ``assembly.SubprocessRunner`` and ``driver.SubprocessRunner``
already use for the same reason."""


@dataclass(frozen=True)
class BeatsPlan:
    """Stage 2's finished output: beats anchored on the timeline a render
    will actually produce."""

    beats: tuple[Beat, ...]
    chunks: tuple[AudioChunk, ...]
    """The frozen ``chunks_v1``. Photography and prose run against this, not
    against the skeleton the concept was written from."""

    result: BeatsResult
    merges: tuple[tuple[int, tuple[int, ...]], ...] = ()
    reanchored: bool = False
    """True when the sheet's own ``length_seconds`` re-cut the timeline. False
    means ``chunks`` is the untouched skeleton -- the conservative default,
    and what a sheet with no editorial lengths produces."""


def plan_beats(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    concept: Mapping[str, Any],
    driver: ModelDriver,
    *,
    reslice: ResliceFn,
    notes: str | None = None,
    max_rounds: int = MAX_REANCHOR_ROUNDS,
) -> BeatsPlan:
    """Generate a beat sheet, honour its lengths, and re-anchor onto the
    result -- retrying when the re-cut timeline leaves a chunk undirected.

    The retry is the point. A chunk with no beat is not a formatting problem
    the validator can see: it only exists *after* the sheet's own length
    requests have re-tiled the timeline, so it has to come back here, be
    described in song time (the ``v1`` chunk ids mean nothing to a model that
    was shown ``v0``), and go back to the model as work to redo.
    """
    feedback: str | None = None
    for round_number in range(1, max_rounds + 1):
        result = generate_beats(
            config, chunks, concept, driver, notes=notes, extra_instructions=feedback
        )

        requests = beat_length_requests(result.beats)
        if not requests:
            logger.info("Beat sheet requests no editorial shot lengths; timeline unchanged")
            return BeatsPlan(
                beats=result.beats, chunks=tuple(chunks), result=result, reanchored=False
            )

        recut = tuple(reslice(requests))
        reanchored = reanchor_beats(result.beats, recut)
        if not reanchored.unmapped_chunk_ids:
            return BeatsPlan(
                beats=reanchored.beats,
                chunks=recut,
                result=result,
                merges=reanchored.merges,
                reanchored=True,
            )

        gaps = ", ".join(
            f"{c.start:.1f}-{c.end:.1f}s"
            for c in recut
            if c.chunk_id in set(reanchored.unmapped_chunk_ids)
        )
        logger.warning(
            "Round %d/%d: honouring your length_seconds left %d span(s) undirected (%s)",
            round_number,
            max_rounds,
            len(reanchored.unmapped_chunk_ids),
            gaps,
        )
        feedback = (
            "## Your previous beat sheet left part of the song with no direction\n"
            "Honouring the length_seconds you asked for re-cut the chunk timeline, and "
            f"these spans of the song ended up with no beat covering them: {gaps}.\n"
            "Reply again with a sheet that covers the whole song after your own long "
            "takes are applied -- shorten or drop some length_seconds, or move the beats "
            "around them."
        )

    raise BeatsValidationError(
        f"after {max_rounds} round(s), honouring the beat sheet's own length_seconds still "
        "left part of the song with no direction. Ask for fewer or shorter long takes"
    )


__all__ = [
    "MAX_REANCHOR_ROUNDS",
    "BeatsPlan",
    "ReanchorResult",
    "ResliceFn",
    "plan_beats",
    "reanchor_beats",
]
