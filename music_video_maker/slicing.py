"""Stage 2a: temporal audio slicing with duration-window enforcement (issue #4)
and H3 frame-grid quantization (issue #20).

Slices the master track into per-segment audio stems that fit MiniMax H3's
temporal context window, off the Stage 1 forced-alignment timestamps
(``AlignmentResult``). Uses ``pydub.AudioSegment`` for non-destructive,
in-memory millisecond slicing -- the wav read/export paths pydub takes here
use only the stdlib ``wave`` module, never ffmpeg, as long as inputs/outputs
stay ``.wav``.

Three enforcement passes, applied in order:

1. **Minimum** (the *effective* min -- see ``_effective_bounds`` below): a
   segment shorter than the minimum is first padded with adjacent
   instrumental time up to the next vocal segment's start (or the track's
   end, for the last segment); if the available room is not enough to reach
   the minimum that way, the segment is merged with the following lyric line
   entirely (chained, if needed, until the minimum is met or segments run
   out).
2. **Maximum** (the *effective* max): a (possibly already-merged) group
   longer than the maximum is split into pieces, each landing on a valid H3
   frame-grid point via ``FrameGrid.quantize_up`` (never down -- see
   ``_split_for_maximum``'s docstring for why). 2nd+ pieces get
   ``is_split_continuation=True``.
3. **Frame-grid quantization** (issue #20): every chunk that pass 2 did
   *not* already grid-quantize (i.e. every un-split single-group chunk) gets
   its duration snapped to the nearest valid H3 ``length`` frame count via
   ``FrameGrid.quantize_nearest`` + ``clamp_to_trained``, with the rounding
   remainder absorbed into adjacent instrumental padding -- never by cutting
   or stretching vocal content. See ``_quantize_single_piece``.

Why quantization matters: MiniMax H3's ``length`` input is a frame count at
24 fps, quantized by ComfyUI itself to ``5 + 17k`` (``docs/h3-node-schema.md``,
ground truth from a live ``/object_info``). If Stage 2a hands Stage 4a an
arbitrary audio-slice duration, ComfyUI silently rounds the derived `length`
to the nearest grid point when the workflow executes -- up to +/-0.354s per
chunk -- while the audio stem stays at the original, unquantized duration.
Stage 5 concatenates the rendered chunks and mux the *pristine* master track
over the result, so that per-chunk mismatch is never corrected; it
accumulates across the whole song into progressive lip-sync drift. Doing the
quantization here instead, and re-slicing the audio to exactly the chosen
frame count, means the audio duration and the rendered video duration are
identical by construction -- there is nothing left for a later stage to
round.

A fifth, opt-in pass sits on top of pass 4 (issue #27): **editorial shot
length**. Passes 1-3 make every shot 5-8 s -- not because anyone chose that
but because it is what vocal-segment timing plus ``max_chunk_seconds``
produces -- and a scene change every six seconds for a whole video is the
most machine-generated thing about the output. ``shot_lengths`` lets the shot
plan ask for a specific length at a specific moment of the song, and
``instrumental_shot_seconds`` lets the unvoiced spans (where a long take pays
off most: a 30 s solo should be one continuous move, not four scene changes)
run longer than the sung ones. See ``_apply_shot_lengths``.

Longer shots are effectively free in wall clock: render time tracks total
latent volume, and the song's length fixes the total frame count, so a longer
shot means fewer shots, not more work. What is *not* free is VRAM --
see :data:`~music_video_maker.shot_plan.MEASURED_MAX_FRAMES`.

Timeline integrity is non-negotiable: every chunk's ``start``/``end`` stays
anchored to the original ``AlignedSegment`` timeline as closely as the frame
grid allows, ``source_segment_indices`` records exactly which original
segments fed each chunk, and chunks are always chronological and
non-overlapping. Stage 5's final mux depends on this -- a drifting boundary
would silently desync the whole video.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from music_video_maker.contracts import (
    AlignedSegment,
    AlignmentResult,
    AudioChunk,
    FrameGrid,
    HardwareProfile,
)
from music_video_maker.shot_plan import MEASURED_MAX_FRAMES, ShotLength

logger = logging.getLogger(__name__)

_EPS = 1e-6
"""Float tolerance for duration/boundary comparisons -- avoids false
positives from ordinary floating-point slop, not a real timing slack."""


class ChunkFrameMismatchError(ValueError):
    """Raised when a about-to-be-emitted chunk's ``frame_count`` doesn't
    correspond to its own sliced ``duration`` (issue #20's index-space
    guard).

    A duration and a frame count are computed together in three different
    places in this module (the pass-2 split-quantize loop, the pass-3
    single-piece quantize-and-pad, and the pass-through case for a chunk
    that needed neither) before landing on the same ``AudioChunk``. A bug
    that paired piece *i*'s duration with piece *j*'s frame count -- the
    same class of index-space mixup this project has been bitten by before
    with ``AlignedSegment.index`` / ``chunk_id`` confusion -- would silently
    hand Stage 4a a ``length`` that does not match its own audio slice,
    which is exactly the drift issue #20 exists to eliminate. Raised with
    both the chunk id and the mismatched values named, never swallowed.
    """


@dataclass(frozen=True)
class _Group:
    """A run of one or more original segments planned to become one chunk
    (pre-split), after minimum-duration enforcement."""

    members: tuple[AlignedSegment, ...]
    start: float
    end: float


@dataclass(frozen=True)
class _Piece:
    """One final chunk-to-be.

    ``frame_count`` is ``None`` until a piece has been through frame-grid
    quantization. Pass 2 (``_split_for_maximum``) sets it directly for split
    pieces (they are quantized inline, with no separate pass-3 step); it
    leaves it ``None`` for un-split single-group pieces, which pass 3
    (``_quantize_single_piece``) then fills in. By the time ``slice_audio``
    emits ``AudioChunk``s, every piece must have a non-``None`` value here --
    that invariant is exactly what ``ChunkFrameMismatchError`` guards.
    """

    members: tuple[AlignedSegment, ...]
    start: float
    end: float
    is_split_continuation: bool
    frame_count: int | None = None


# --------------------------------------------------------------------------- #
# Effective duration bounds (issue #20): derive from the grid, not just the
# profile's raw fields, and clamp loudly into the trained range.
# --------------------------------------------------------------------------- #


def _effective_bounds(hardware: HardwareProfile) -> tuple[float, float, FrameGrid]:
    """Resolve the duration window ``slice_audio`` actually enforces.

    ``hardware.min_chunk_seconds`` / ``max_chunk_seconds`` are a profile's
    *requested* window -- they may predate the trained-range discovery in
    issue #20, or a caller-built ``HardwareProfile`` (e.g. from
    ``RunConfig.hardware``) may simply ask for something the model was never
    trained on. This clamps into ``[trained_min_frames, trained_max_frames]``
    (expressed in seconds via the profile's own ``frame_grid``) and logs a
    warning naming both the requested and the clamped value whenever
    clamping actually changes anything -- a profile is never silently
    obeyed outside the trained range.
    """
    grid = hardware.frame_grid
    trained_min_s = grid.frames_to_seconds(grid.trained_min_frames)
    trained_max_s = grid.frames_to_seconds(grid.trained_max_frames)

    eff_min = max(hardware.min_chunk_seconds, trained_min_s)
    eff_max = min(hardware.max_chunk_seconds, trained_max_s)

    if hardware.min_chunk_seconds < trained_min_s - _EPS:
        logger.warning(
            "HardwareProfile %r requested min_chunk_seconds=%.3fs, below H3's trained floor "
            "of %.3fs (%d frames); clamping the effective minimum up to the trained floor.",
            hardware.name,
            hardware.min_chunk_seconds,
            trained_min_s,
            grid.trained_min_frames,
        )
    if hardware.max_chunk_seconds > trained_max_s + _EPS:
        logger.warning(
            "HardwareProfile %r requested max_chunk_seconds=%.3fs, above H3's trained ceiling "
            "of %.3fs (%d frames); more VRAM does not extend the trained range, so clamping "
            "the effective maximum down to the trained ceiling.",
            hardware.name,
            hardware.max_chunk_seconds,
            trained_max_s,
            grid.trained_max_frames,
        )

    if eff_min > eff_max + _EPS:
        logger.error(
            "HardwareProfile %r has an unsatisfiable chunk window after clamping to H3's "
            "trained range: effective min=%.3fs > effective max=%.3fs (requested min=%.3fs, "
            "max=%.3fs).",
            hardware.name,
            eff_min,
            eff_max,
            hardware.min_chunk_seconds,
            hardware.max_chunk_seconds,
        )
        raise ValueError(
            f"HardwareProfile {hardware.name!r}: effective min_chunk_seconds "
            f"{eff_min!r} exceeds effective max_chunk_seconds {eff_max!r} after "
            "clamping to the H3 trained range"
        )

    return eff_min, eff_max, grid


# --------------------------------------------------------------------------- #
# Pass 1: minimum-duration merge/pad
# --------------------------------------------------------------------------- #


def _merge_for_minimum(
    segments: tuple[AlignedSegment, ...], track_duration: float, min_chunk_seconds: float
) -> list[_Group]:
    groups: list[_Group] = []
    n = len(segments)
    i = 0
    while i < n:
        members = [segments[i]]
        start = segments[i].start
        end = segments[i].end
        j = i
        while (end - start) < min_chunk_seconds - _EPS:
            has_next = (j + 1) < n
            next_boundary = segments[j + 1].start if has_next else track_duration

            if next_boundary - start >= min_chunk_seconds - _EPS:
                # Padding alone (instrumental time up to the next vocal
                # segment, or the track's end) reaches the minimum.
                end = start + min_chunk_seconds
                break

            if not has_next:
                # Nothing left to merge with and the track itself runs out
                # of room -- pad as far as it allows and accept the shortfall.
                logger.warning(
                    "Segment index=%d (start=%.3f) cannot reach the %.3fs minimum even after "
                    "padding to the track end (%.3f); emitting a short final chunk.",
                    segments[i].index,
                    start,
                    min_chunk_seconds,
                    track_duration,
                )
                end = next_boundary
                break

            # Padding is insufficient -- merge the next segment wholesale
            # and re-check the minimum against the merged group.
            j += 1
            members.append(segments[j])
            end = segments[j].end

        groups.append(_Group(members=tuple(members), start=start, end=end))
        i = j + 1

    return groups


# --------------------------------------------------------------------------- #
# Pass 2: maximum-duration split, grid-quantized inline
# --------------------------------------------------------------------------- #


def _split_for_maximum(
    groups: list[_Group], max_chunk_seconds: float, grid: FrameGrid
) -> list[_Piece]:
    """Split any group longer than ``max_chunk_seconds`` into grid-quantized
    pieces.

    Each split piece's duration is snapped with ``FrameGrid.quantize_up``
    (never ``quantize_nearest``/down). This is deliberately different from
    pass 3's un-split single pieces: a split piece is carved out of
    continuous vocal audio with no instrumental padding on either side to
    absorb a shrink into, so rounding down would either truncate lyric
    content (on the last piece) or silently reassign more audio than
    intended to the *next* sibling piece with no record of it happening.
    Rounding up only ever grows a piece; the growth cascades forward through
    ``cursor`` (each next piece starts later than an even geometric split
    would have placed it) and, for the final piece, spills past the group's
    real end into whatever instrumental time follows -- exactly the same
    "absorb into adjacent padding" rule pass 1 and pass 3 use, just applied
    once at the tail instead of per piece. If there truly is not enough room
    before the next chunk, ``slice_audio``'s pre-existing timeline-drift
    guard raises rather than silently producing an overlap.
    """
    pieces: list[_Piece] = []
    for group in groups:
        duration = group.end - group.start
        if duration <= max_chunk_seconds + _EPS:
            pieces.append(
                _Piece(
                    members=group.members,
                    start=group.start,
                    end=group.end,
                    is_split_continuation=False,
                    frame_count=None,
                )
            )
            continue

        piece_count = math.ceil(duration / max_chunk_seconds)
        piece_duration = duration / piece_count
        logger.info(
            "Splitting %.3fs group (segments=%s) into %d grid-quantized pieces of ~%.3fs to "
            "respect the %.3fs max-duration bound.",
            duration,
            tuple(m.index for m in group.members),
            piece_count,
            piece_duration,
            max_chunk_seconds,
        )

        cursor = group.start
        for p in range(piece_count):
            is_last = p == piece_count - 1
            raw_end = group.end if is_last else cursor + piece_duration
            raw_duration = raw_end - cursor

            # ceil, not round: quantize_up must never imply a duration below
            # raw_duration. A plain round() can land exactly on a valid grid
            # frame that is a hair *short* of raw_duration whenever
            # raw_duration's true (unrounded) frame count has a fractional
            # part just above that grid point (e.g. 6.6s = 158.4 frames,
            # where 158 itself is already grid-valid but 0.4 frames short).
            requested_frames = math.ceil(grid.seconds_to_frames(raw_duration) - _EPS)
            frames = grid.clamp_to_trained(grid.quantize_up(requested_frames))
            piece_end = cursor + grid.frames_to_seconds(frames)

            pieces.append(
                _Piece(
                    members=group.members,
                    start=cursor,
                    end=piece_end,
                    is_split_continuation=(p > 0),
                    frame_count=frames,
                )
            )
            cursor = piece_end

    return pieces


# --------------------------------------------------------------------------- #
# Pass 3: frame-grid quantization for un-split single-group pieces
# --------------------------------------------------------------------------- #


def _quantize_single_piece(
    piece: _Piece, grid: FrameGrid, prev_end: float, next_boundary: float
) -> _Piece:
    """Snap ``piece`` (which pass 2 left un-split, ``frame_count is None``)
    onto the H3 frame grid, absorbing the rounding remainder into adjacent
    instrumental padding.

    The true, non-negotiable floor is the piece's *vocal* span --
    ``members[0].start`` to ``members[-1].end`` -- never the possibly
    already-padded ``piece.start``/``piece.end`` pass 1 produced (pass 1
    only ever extends the trailing edge to reach the minimum, so any slack
    between the vocal span and the raw piece boundaries is padding that is
    always safe to redistribute or remove).

    Rounding prefers ``quantize_nearest`` (snap up or down, whichever is
    closer -- issue #20 is explicit that this should not systematically
    lengthen every chunk). If nearest would round *below* the vocal span
    itself (only possible when the raw piece had little or no padding to
    begin with), that would cut lyric audio, which is forbidden -- so this
    falls back to ``quantize_up`` from the vocal span instead, guaranteeing
    the chunk always covers everything the alignment says should be there.

    The resulting slack (target duration minus vocal duration) is drawn from
    the trailing instrumental gap first (mirrors pass 1's own padding
    direction), then from the leading gap if the trailing gap runs out. If
    even both combined are not enough, whatever is available is used and a
    warning is logged -- ``slice_audio``'s pre-existing overlap guard is the
    backstop for a genuinely unsatisfiable configuration.
    """
    vocal_start = piece.members[0].start
    vocal_end = piece.members[-1].end
    vocal_duration = vocal_end - vocal_start
    raw_duration = piece.end - piece.start

    raw_requested_frames = round(grid.seconds_to_frames(raw_duration))
    nearest_frames = grid.clamp_to_trained(grid.quantize_nearest(raw_requested_frames))
    nearest_duration = grid.frames_to_seconds(nearest_frames)

    if nearest_duration + _EPS >= vocal_duration:
        target_frames, target_duration = nearest_frames, nearest_duration
    else:
        # ceil (not round): see the matching comment in _split_for_maximum --
        # target_frames must never imply a duration below vocal_duration.
        vocal_requested_frames = math.ceil(grid.seconds_to_frames(vocal_duration) - _EPS)
        target_frames = grid.clamp_to_trained(grid.quantize_up(vocal_requested_frames))
        target_duration = grid.frames_to_seconds(target_frames)
        logger.warning(
            "Chunk (segments=%s): nearest grid duration %.3fs would truncate %.3fs of vocal "
            "content; rounding up to %.3fs (%d frames) instead so no lyric audio is dropped.",
            tuple(m.index for m in piece.members),
            nearest_duration,
            vocal_duration,
            target_duration,
            target_frames,
        )

    slack = max(0.0, target_duration - vocal_duration)
    forward_room = max(0.0, next_boundary - vocal_end)
    trailing_pad = min(slack, forward_room)
    remaining = slack - trailing_pad

    backward_room = max(0.0, vocal_start - prev_end)
    leading_pad = min(remaining, backward_room)
    remaining -= leading_pad

    new_start = vocal_start - leading_pad
    new_end = vocal_end + trailing_pad

    if remaining > _EPS:
        # Both instrumental gaps combined still fall short of the target --
        # an unusually tight configuration (both neighbors closer than one
        # grid step, ~0.708s). Re-snap frame_count to whatever duration this
        # padding *can* actually reach so duration and frame_count never
        # disagree (see ChunkFrameMismatchError). quantize_up, not nearest,
        # so the re-snapped duration never drops back below the
        # vocal-covering span we already have in new_start/new_end.
        achieved_duration = new_end - new_start
        achieved_requested_frames = math.ceil(grid.seconds_to_frames(achieved_duration) - _EPS)
        target_frames = grid.clamp_to_trained(grid.quantize_up(achieved_requested_frames))
        target_duration = grid.frames_to_seconds(target_frames)
        # The re-snapped target may ask for a hair more than achieved_duration
        # (quantize_up can round the already-tight achieved_duration itself
        # up to the next grid point); grow the trailing edge to cover that --
        # even past forward_room if necessary. slice_audio's pre-existing
        # timeline-drift guard is the real backstop if this pushes into the
        # next chunk's territory; it is a louder, more specific failure than
        # silently emitting a duration that doesn't match frame_count.
        new_end = new_start + target_duration
        logger.warning(
            "Chunk (segments=%s): only %.3fs of instrumental padding is available (forward="
            "%.3fs, backward=%.3fs) to reach the original frame-grid target; re-snapped to "
            "%.3fs (%d frames) instead. All vocal content is still covered.",
            tuple(m.index for m in piece.members),
            trailing_pad + leading_pad,
            forward_room,
            backward_room,
            target_duration,
            target_frames,
        )

    return _Piece(
        members=piece.members,
        start=new_start,
        end=new_end,
        is_split_continuation=piece.is_split_continuation,
        frame_count=target_frames,
    )


# --------------------------------------------------------------------------- #
# Pass 4 (opt-in): instrumental coverage / contiguous timeline tiling
# --------------------------------------------------------------------------- #


def _grid_frames_at_or_below(seconds: float, grid: FrameGrid) -> int:
    """Largest grid-valid, trained-range frame count whose duration is <= ``seconds``."""
    raw = int(math.floor(grid.seconds_to_frames(seconds) + _EPS))
    if raw < grid.base_frames:
        return grid.trained_min_frames
    steps = (raw - grid.base_frames) // grid.step_frames
    return grid.clamp_to_trained(grid.base_frames + steps * grid.step_frames)


def _instrumental_max_frames(
    instrumental_shot_seconds: float | None,
    max_frames: int,
    min_frames: int,
    grid: FrameGrid,
) -> int:
    """The longest a *filler* chunk may run (issue #27).

    Defaults to the same ceiling the sung chunks get, which is today's
    behaviour. Given its own value it becomes the editorial lever the issue
    asks for: sung shots want to be short and to break near lyric boundaries,
    while a 30s solo currently becomes four scene changes when it should be
    one continuous move. One knob could never express both.

    Clamped into H3's trained range like every other duration here, loudly --
    the ceiling is a property of the model weights, not of the card.
    """
    if instrumental_shot_seconds is None:
        return max_frames

    requested = grid.quantize_nearest(int(round(grid.seconds_to_frames(instrumental_shot_seconds))))
    frames = grid.clamp_to_trained(requested)
    if frames != requested:
        logger.warning(
            "instrumental_shot_seconds=%.3f (%d frames) is outside H3's trained range of "
            "%.3fs-%.3fs (%d-%d frames); clamping instrumental shots to %.3fs (%d frames).",
            instrumental_shot_seconds,
            requested,
            grid.frames_to_seconds(grid.trained_min_frames),
            grid.frames_to_seconds(grid.trained_max_frames),
            grid.trained_min_frames,
            grid.trained_max_frames,
            grid.frames_to_seconds(frames),
            frames,
        )
    frames = max(frames, min_frames)
    if frames > MEASURED_MAX_FRAMES:
        logger.warning(
            "instrumental_shot_seconds asks for instrumental shots up to %d frames (%.3fs), "
            "above the %d frames ever rendered on this card -- unmeasured territory. Longer "
            "shots cost no extra wall clock (fewer chunks over the same total frames), but "
            "VRAM at this frame count has never been measured here.",
            frames,
            grid.frames_to_seconds(frames),
            MEASURED_MAX_FRAMES,
        )
    return frames


def _grid_frames_at_or_above(seconds: float, grid: FrameGrid) -> int:
    """Smallest grid-valid, trained-range frame count whose duration is >= ``seconds``."""
    raw = int(math.ceil(grid.seconds_to_frames(seconds) - _EPS))
    return grid.clamp_to_trained(grid.quantize_up(raw))


def _plan_frames_run(
    target_frames: int, min_frames: int, max_frames: int, grid: FrameGrid
) -> tuple[int, ...]:
    """Grid-valid frame counts tiling ``target_frames`` frames of timeline.

    Returns ``()`` when the span is too short to host even one chunk at the
    trained floor -- a 0.4s hole must never become a 0.4s render, because H3
    is only trained from 124 frames up. Such a hole is absorbed by a
    neighbouring chunk instead (callers re-anchor, so no audio is lost).

    Otherwise the span is divided into ``k`` roughly equal pieces, each
    snapped to the grid. ``k`` is the fewest chunks that can cover it without
    any single one exceeding ``max_frames``. The running remainder is
    re-divided at every step, so the rounding error of piece *i* is corrected
    by piece *i+1* rather than accumulating across the fill.
    """
    if target_frames < min_frames:
        return ()

    count = max(1, -(-target_frames // max_frames))  # ceil division

    frames: list[int] = []
    remaining = target_frames
    for i in range(count):
        slots_left = count - i
        ideal = remaining / slots_left
        chosen = grid.clamp_to_trained(grid.quantize_nearest(int(round(ideal))))
        chosen = max(min_frames, min(max_frames, chosen))
        frames.append(chosen)
        remaining -= chosen

    return tuple(frames)


def _plan_filler_frames(
    gap_seconds: float, min_frames: int, max_frames: int, grid: FrameGrid
) -> tuple[int, ...]:
    """Frame counts for the filler chunks covering a ``gap_seconds`` hole.
    Thin seconds-facing wrapper around :func:`_plan_frames_run`."""
    return _plan_frames_run(
        int(round(grid.seconds_to_frames(gap_seconds))), min_frames, max_frames, grid
    )


# --------------------------------------------------------------------------- #
# Pass 5 (opt-in): editorial shot length (issue #27)
# --------------------------------------------------------------------------- #

SHOT_LENGTH_ANCHOR_TOLERANCE_SECONDS = 1.0
"""How far a request's anchor may sit from a real chunk boundary and still be
considered a match.

Wider than ``shot_plan.START_TOLERANCE_SECONDS`` (0.25s) on purpose, and for a
different job. That one asks "was this plan authored against this alignment?",
where anything past float noise is drift. This one asks "which boundary did the
author mean?", and the honest answer has to survive the grid residue that
honouring an *earlier* request leaves behind: merging m grid-valid chunks
(each ``5 + 17k`` frames) into one cannot land on the grid exactly, so
boundaries after it move by a fraction of a second. A second is far wider than
that residue and far narrower than the ~5s chunks it has to tell apart, so the
nearest-boundary match cannot silently pick the wrong one."""


def _boundary_starts(frames: Sequence[int], grid: FrameGrid) -> list[float]:
    """Cumulative start time of each chunk in a contiguous tiling from 0.

    The tiling ``_cover_instrumentals`` produces starts at 0 and never has a
    hole, so a chunk's start *is* the sum of the durations before it -- which
    is exactly why video offset == audio offset for every chunk."""
    starts: list[float] = []
    running = 0.0
    for count in frames:
        starts.append(running)
        running += grid.frames_to_seconds(count)
    return starts


def _grid_frames_at_or_below_count(count: int, grid: FrameGrid) -> int:
    """Largest grid-valid, trained-range frame count that is <= ``count``."""
    if count < grid.base_frames:
        return grid.trained_min_frames
    steps = (count - grid.base_frames) // grid.step_frames
    return grid.clamp_to_trained(grid.base_frames + steps * grid.step_frames)


def _requested_frames(request: ShotLength, grid: FrameGrid) -> int:
    """The frame count a request resolves to: grid-quantized, trained-clamped.

    ``quantize_nearest`` rather than up, matching pass 3: a request is a
    target, not a floor, and systematically rounding every one of them up
    would lengthen the video against the master track. Clamping is loud --
    an author who asked for 22s must not discover by watching the output that
    they got 15.083s.
    """
    quantized = grid.quantize_nearest(int(round(grid.seconds_to_frames(request.length_seconds))))
    frames = grid.clamp_to_trained(quantized)
    if frames != quantized:
        logger.warning(
            "Shot length request at %.3fs (plan chunk_id=%s) asked for %.3fs (%d frames), "
            "outside H3's trained range of %.3fs-%.3fs (%d-%d frames); clamped to %.3fs "
            "(%d frames). The trained range is a property of the model weights -- neither "
            "more VRAM nor a smaller resolution extends it (issue #20).",
            request.start,
            request.source_chunk_id,
            request.length_seconds,
            quantized,
            grid.frames_to_seconds(grid.trained_min_frames),
            grid.frames_to_seconds(grid.trained_max_frames),
            grid.trained_min_frames,
            grid.trained_max_frames,
            grid.frames_to_seconds(frames),
            frames,
        )
    return frames


_RETILE_CANDIDATES = 4
"""How many run lengths to weigh when deciding what a shot replaces.

The first run that covers the requested length is the obvious choice but not
always the tidiest: because the grid's usable frame counts are a *gapped* set
once the trained floor is applied (with a 141-192 window, 141-192 frames tile
and 193-281 do not), a leftover can be untileable and get rounded up by a
couple of seconds. Weighing the next few runs usually finds one whose leftover
tiles exactly, which keeps the rest of the timeline still. Bounded so a long
take never goes hunting through the whole song for a perfect fit."""


def _plan_replacement(
    frames: Sequence[int],
    index: int,
    target: int,
    min_frames: int,
    max_frames: int,
    grid: FrameGrid,
) -> tuple[int, tuple[int, ...]] | None:
    """Which chunks a shot of ``target`` frames replaces, and how to retile
    the leftover.

    Returns ``(end, tail)`` -- the shot replaces ``frames[index:end]`` and is
    followed by ``tail`` -- or ``None`` when the timeline runs out before the
    requested length is covered. Prefers the run whose leftover tiles most
    exactly, so that boundaries after the shot move as little as possible.
    """
    best: tuple[int, tuple[int, ...]] | None = None
    best_error: int | None = None
    run_total = 0
    end = index
    weighed = 0

    while end < len(frames) and weighed < _RETILE_CANDIDATES:
        run_total += frames[end]
        end += 1
        if run_total < target:
            continue
        weighed += 1
        residual = run_total - target
        tail = _plan_frames_run(residual, min_frames, max_frames, grid)
        error = abs(sum(tail) - residual)
        if best_error is None or error < best_error:
            best, best_error = (end, tail), error
        if error == 0:
            break

    return best


def _match_anchor(
    starts: Sequence[float], requested: float, shift: float
) -> tuple[int, float]:
    """Which boundary a request means, and how far off it landed.

    **Two anchor conventions reach this function and the number itself cannot
    say which one it is**, so both are tried and the nearer boundary wins:

    * ``requested + shift`` -- authored against the *un-retiled* timeline, i.e.
      straight onto a fresh ``--prepare`` skeleton. All the anchors in such a
      plan were read off one chunk list in one pass, so each one has to be
      corrected by however far the requests ahead of it have already moved the
      timeline (see :func:`_apply_shot_lengths`'s ``shift``).
    * ``requested`` -- authored against the *retiled* timeline, which is what
      ``--prepare --from-plan`` emits (issue #54 design section 5). Those
      anchors are the starts a render will actually produce, and therefore the
      only ones that survive ``shot_plan``'s drift check, so this is the
      convention a generated plan is in.

    Correcting a v1 anchor by the shift it already carries double-counts it.
    That is not theoretical: with the 141-frame chunks this project measures
    at, three 15 s takes accumulate ~3.4 s of shift, and the third request
    then misses every boundary by more than
    :data:`SHOT_LENGTH_ANCHOR_TOLERANCE_SECONDS` and is dropped -- authored
    direction that reads as applied and is not, which is the exact failure
    this file's warnings exist to prevent.

    Strictly wider than matching one space alone: when nothing has moved yet
    the two candidates are the same number, and every anchor that matched
    before still matches now.
    """
    best_index, best_offset = 0, float("inf")
    for anchor in (requested + shift, requested):
        index = min(range(len(starts)), key=lambda k: abs(starts[k] - anchor))
        offset = abs(starts[index] - anchor)
        if offset < best_offset:
            best_index, best_offset = index, offset
    return best_index, best_offset


def _apply_shot_lengths(
    boundaries: list[tuple[float, int, bool]],
    shot_lengths: Sequence[ShotLength],
    min_frames: int,
    max_frames: int,
    grid: FrameGrid,
) -> list[tuple[float, int, bool]]:
    """Retile a contiguous timeline so each request's anchor gets its length.

    A request says "the shot starting at T runs for L seconds". Honouring it
    is a *retiling*, never an insertion: the chunk at T takes the requested
    frame count and swallows the chunks it now covers, and whatever is left
    over at the far end is re-tiled behind it by :func:`_plan_frames_run`.
    That is what keeps every invariant this module exists to protect:

    * **Coverage survives.** The replaced run and its replacement span the
      same stretch of the timeline, so the tiling still starts at 0, still has
      no holes, and still ends where it did. Nothing after the replaced run
      even moves, beyond the grid residue described below.
    * **No timing is invented.** Boundaries are only ever merged away or
      re-divided; every one of them still traces back to the alignment
      timestamps passes 1-3 anchored on.
    * **No lyric is lost.** ``_cover_instrumentals`` re-derives each chunk's
      text from whichever words' midpoints land in its *final* window, so
      words follow the boundaries rather than the other way round.

    The grid makes exact preservation impossible: m chunks of ``5 + 17k``
    frames total ``5m + 17K``, which is only itself grid-valid for m = 1, so
    merging leaves a residue of a few frames. It is bounded (well under one
    grid step per request, ~0.2s), and :func:`_rebalance_tail` sweeps the
    accumulated total back onto the original length at the end rather than
    letting it ride out to the final mux.

    Every request that cannot be honoured -- an anchor matching no boundary,
    or one swallowed by an earlier, longer take -- is warned about by name and
    skipped. Silence there would mean authored direction that reads as applied
    and is not, which is the failure mode this whole file is built against.
    """
    frames = [count for _, count, _ in boundaries]
    continuations = [flag for _, _, flag in boundaries]
    original_total = sum(frames)
    applied_until = -1.0  # end of the last honoured shot, in seconds
    last_applied_index = -1
    shift = 0.0
    """How far honouring the requests so far has moved every boundary after
    them, in seconds. Anchors authored against the *un-retiled* timeline --
    all of them at once, in one editing pass -- have to be matched against
    their own position plus this, or the second request in a plan is compared
    against a timeline the first one already moved and drops out as "matches
    no boundary" for a reason the author cannot see. Anchors re-authored
    against the *retiled* timeline already carry it. :func:`_match_anchor`
    tries both, because the request cannot say which it is."""

    for request in sorted(shot_lengths, key=lambda r: r.start):
        starts = _boundary_starts(frames, grid)
        index, offset = _match_anchor(starts, request.start, shift)

        if offset > SHOT_LENGTH_ANCHOR_TOLERANCE_SECONDS:
            logger.warning(
                "Shot length request at %.3fs (plan chunk_id=%s) matches no chunk boundary "
                "-- the nearest chunk starts at %.3fs, %.3fs away (tolerance %.3fs), so no "
                "chunk starts where this shot was authored to. Ignoring it: its length is "
                "unchanged. Re-author the request against the current chunk list; an "
                "earlier, longer shot may have moved this boundary.",
                request.start,
                request.source_chunk_id,
                starts[index],
                offset,
                SHOT_LENGTH_ANCHOR_TOLERANCE_SECONDS,
            )
            continue

        if starts[index] < applied_until - _EPS:
            logger.warning(
                "Shot length request at %.3fs (plan chunk_id=%s) falls inside an earlier, "
                "longer shot that now runs to %.3fs, so it has been swallowed -- there is "
                "no chunk boundary left at this moment to give a length to. Ignoring it. "
                "Two overlapping takes cannot both be on screen; shorten the earlier one "
                "or drop this request.",
                request.start,
                request.source_chunk_id,
                applied_until,
            )
            continue

        target = _requested_frames(request, grid)

        chosen = _plan_replacement(frames, index, target, min_frames, max_frames, grid)
        if chosen is None:
            run_total = sum(frames[index:])
            end = len(frames)
            capped = _grid_frames_at_or_below_count(run_total, grid)
            logger.warning(
                "Shot length request at %.3fs (plan chunk_id=%s) asks for %.3fs but only "
                "%.3fs of timeline remains; shortened to %.3fs (%d frames) rather than "
                "rendering past the end of the track.",
                request.start,
                request.source_chunk_id,
                grid.frames_to_seconds(target),
                grid.frames_to_seconds(run_total),
                grid.frames_to_seconds(capped),
                capped,
            )
            target = capped
            tail = ()
        else:
            end, tail = chosen
            run_total = sum(frames[index:end])

        residual = run_total - target
        if chosen is not None and residual and not tail:
            # End of the timeline with a sub-floor leftover and nothing to
            # merge it into: give the frames back to the shot itself where the
            # grid allows, and say so when a sliver of the outro is dropped.
            grown = _grid_frames_at_or_below_count(run_total, grid)
            logger.warning(
                "Shot length request at %.3fs (plan chunk_id=%s) leaves %.3fs at the end of "
                "the timeline, below the %.3fs trained floor and with nothing after it to "
                "merge into; the shot takes %.3fs (%d frames) and the remaining %.3fs is "
                "dropped from the tail.",
                request.start,
                request.source_chunk_id,
                grid.frames_to_seconds(residual),
                grid.frames_to_seconds(min_frames),
                grid.frames_to_seconds(grown),
                grown,
                grid.frames_to_seconds(run_total - grown),
            )
            target = grown

        logger.info(
            "Shot length: chunk at %.3fs (plan chunk_id=%s) set to %d frames (%.3fs), "
            "replacing %d chunk(s) worth %.3fs; %d chunk(s) retile the remaining %.3fs.",
            starts[index],
            request.source_chunk_id,
            target,
            grid.frames_to_seconds(target),
            end - index,
            grid.frames_to_seconds(run_total),
            len(tail),
            grid.frames_to_seconds(sum(tail)),
        )
        if target > MEASURED_MAX_FRAMES:
            logger.warning(
                "Chunk at %.3fs will render %d frames (%.3fs), above the %d frames ever "
                "rendered on this card -- unmeasured territory. It is inside H3's trained "
                "range and costs no extra wall clock, but temporal VAE decode memory scales "
                "non-linearly with frame count and an over-committed card here can wedge "
                "the host instead of raising CUDA OOM (issues #23, #24). Watch this one.",
                starts[index],
                target,
                grid.frames_to_seconds(target),
                MEASURED_MAX_FRAMES,
            )

        frames[index:end] = [target, *tail]
        continuations[index:end] = [continuations[index], *([False] * len(tail))]
        applied_until = starts[index] + grid.frames_to_seconds(target)
        last_applied_index = index
        shift += grid.frames_to_seconds(target + sum(tail) - run_total)

    _rebalance_tail(
        frames,
        continuations,
        original_total,
        min_frames,
        max_frames,
        grid,
        last_applied_index + 1,
    )

    starts = _boundary_starts(frames, grid)
    return [
        (start, count, flag)
        for start, count, flag in zip(starts, frames, continuations, strict=True)
    ]


def _rebalance_tail(
    frames: list[int],
    continuations: list[bool],
    original_total: int,
    min_frames: int,
    max_frames: int,
    grid: FrameGrid,
    protected_index: int,
) -> None:
    """Re-tile the end of the timeline so the total length is restored.

    Each honoured request leaves a few frames of grid residue (see
    :func:`_apply_shot_lengths`). Left alone those accumulate, and the whole
    covering ends up longer or shorter than the track -- either a silent tail
    rendered past the end of the master, or a sliver of the outro never
    rendered at all. Both are cheap to avoid: absorb the *whole* accumulated
    difference once, into the last chunks, where the audio is the outro rather
    than a lyric. Mutates in place; a no-op when nothing drifted.

    ``protected_index`` is the first chunk this may touch: never an honoured
    shot itself, or rebalancing would quietly take back the length the author
    asked for -- the one thing this pass exists to deliver.
    """
    delta = original_total - sum(frames)
    if not delta:
        return

    index = len(frames)
    pool = 0
    while index > protected_index and pool + delta < min_frames:
        index -= 1
        pool += frames[index]

    replacement = (
        _plan_frames_run(pool + delta, min_frames, max_frames, grid)
        if index >= protected_index
        else ()
    )
    if not replacement:
        logger.warning(
            "Shot lengths left the timeline %.3fs %s than it was and the tail is too short "
            "to absorb it; the covering now runs %.3fs. Sync is unaffected (every chunk is "
            "still sliced at its own offset) but the final chunk no longer lines up with "
            "the end of the master track.",
            abs(grid.frames_to_seconds(delta)),
            "shorter" if delta > 0 else "longer",
            grid.frames_to_seconds(sum(frames)),
        )
        return

    logger.debug(
        "Rebalanced the last %d chunk(s) into %d to absorb %.3fs of grid residue left by "
        "the shot lengths.",
        len(frames) - index,
        len(replacement),
        grid.frames_to_seconds(delta),
    )
    frames[index:] = list(replacement)
    continuations[index:] = [False] * len(replacement)


_MIN_VOICED_FRACTION = 0.25
"""How much of a chunk must be voiced before it counts as a lyric chunk.

Pass 1 pads a short segment with adjacent instrumental time, so a chunk can
overlap a lyric segment by a fraction of a second and be silent for the rest.
Prompting that as "the character is actively singing <line>" would put a
singing performance over ten-plus seconds of instrumental. Below this
fraction the chunk is prompted as instrumental instead -- H3 still receives
the real audio, so the few voiced frames it does contain still lip-sync.
"""


def _segments_overlapping(
    segments: tuple[AlignedSegment, ...], start: float, end: float
) -> tuple[AlignedSegment, ...]:
    """Original aligned segments whose voiced span intersects ``[start, end)``."""
    return tuple(s for s in segments if s.start < end - _EPS and s.end > start + _EPS)


def _text_within(members: tuple[AlignedSegment, ...], start: float, end: float) -> str:
    """The lyric words actually audible in ``[start, end)``.

    A lyric line often straddles a chunk boundary -- pass 2 splits long
    segments, and instrumental coverage retiles the timeline independently of
    where segments fall. Handing each chunk the *full* text of every
    overlapping segment prompts H3 to perform the whole line in **both**
    chunks, so the performance drifts away from the audio it is supposed to
    match.

    Seen on a real render: "There was a time ... so much better now!" ran
    84.46-91.63s, split across chunk 13 (82.04-89.33, holding the first 4.9s)
    and chunk 14 (89.33-95.92, holding the last 2.3s). Both were prompted with
    all twelve words, H3 sang all twelve in each, and the mouth was still on
    "so much better now" six seconds after the audio had finished saying it.

    A word belongs to the chunk containing its **midpoint**. Chunks tile
    contiguously and without overlap, so every word lands in exactly one
    chunk: none duplicated, none dropped. Segments with no word timings fall
    back to their full text -- there is nothing to slice by, and dropping them
    silently would be worse than a slightly wide attribution.
    """
    parts: list[str] = []
    for member in members:
        if not member.words:
            parts.append(member.text)
            continue
        kept = [
            w.word.strip()
            for w in member.words
            if start - _EPS <= (w.start + w.end) / 2.0 < end - _EPS
        ]
        if kept:
            parts.append(" ".join(kept))
    return " ".join(p for p in parts if p).strip()


def _voiced_seconds_within(
    segments: tuple[AlignedSegment, ...], start: float, end: float
) -> float:
    """Total voiced time inside ``[start, end)``, summed over overlaps."""
    return sum(
        max(0.0, min(s.end, end) - max(s.start, start))
        for s in _segments_overlapping(segments, start, end)
    )


def _cover_instrumentals(
    pieces: list[_Piece],
    segments: tuple[AlignedSegment, ...],
    track_duration: float,
    eff_min: float,
    eff_max: float,
    grid: FrameGrid,
    shot_lengths: Sequence[ShotLength] = (),
    instrumental_shot_seconds: float | None = None,
) -> list[_Piece]:
    """Retile ``pieces`` into a contiguous covering of ``[0, track_duration]``.

    Passes 1-3 anchor each chunk to its own voiced span and skip everything
    between them. That is fine in isolation but wrong for Stage 5: the concat
    demuxer lays chunks end to end, so an unrendered instrumental span is
    squeezed out of the video while the muxed master audio still contains it.
    Every chunk after the first gap then plays against the wrong moment of
    the song, and the video finishes short by the total instrumental time.

    This pass fixes that by construction. It keeps the durations passes 1-3
    chose (so the voiced-span planning, merging and splitting all still
    govern where boundaries fall), inserts filler chunks across the holes,
    and re-anchors every chunk so chunk N starts exactly where chunk N-1
    ended. Because each chunk is then sliced from the master at its own
    position in the concatenated video, video offset == audio offset for
    every chunk and the final mux cannot drift.

    Lyric text is re-derived *after* re-anchoring, from whichever original
    segments overlap each chunk's final window -- never carried over from the
    pre-anchoring piece. Absorbing a sub-minimum hole shifts later chunks
    slightly earlier against the master, and a chunk must be prompted with
    the lyric its audio actually contains, not the one it was planned around.
    """
    min_frames = _grid_frames_at_or_above(eff_min, grid)
    max_frames = _grid_frames_at_or_below(eff_max, grid)
    if max_frames < min_frames:
        max_frames = min_frames

    filler_max_frames = _instrumental_max_frames(
        instrumental_shot_seconds, max_frames, min_frames, grid
    )

    boundaries: list[tuple[float, int, bool]] = []  # (start, frame_count, is_split_continuation)
    running = 0.0

    def _emit_filler(gap: float) -> None:
        nonlocal running
        for frame_count in _plan_filler_frames(gap, min_frames, filler_max_frames, grid):
            boundaries.append((running, frame_count, False))
            running += grid.frames_to_seconds(frame_count)

    for piece in pieces:
        if piece.start - running > _EPS:
            _emit_filler(piece.start - running)
        assert piece.frame_count is not None  # pass 3 guarantees this
        boundaries.append((running, piece.frame_count, piece.is_split_continuation))
        running += grid.frames_to_seconds(piece.frame_count)

    if track_duration - running > _EPS:
        _emit_filler(track_duration - running)

    if shot_lengths:
        boundaries = _apply_shot_lengths(
            boundaries, shot_lengths, min_frames, max_frames, grid
        )

    covered: list[_Piece] = []
    for start, frame_count, is_continuation in boundaries:
        end = start + grid.frames_to_seconds(frame_count)
        members = _segments_overlapping(segments, start, end)
        if members:
            voiced = _voiced_seconds_within(segments, start, end)
            if voiced < _MIN_VOICED_FRACTION * (end - start):
                logger.debug(
                    "Chunk at %.3fs is only %.3fs voiced across %.3fs; prompting it as "
                    "instrumental rather than as a sung lyric.",
                    start,
                    voiced,
                    end - start,
                )
                members = ()
        covered.append(
            _Piece(
                members=members,
                start=start,
                end=end,
                is_split_continuation=is_continuation,
                frame_count=frame_count,
            )
        )

    voiced = sum(1 for p in covered if p.members)
    logger.info(
        "Instrumental coverage: %d chunk(s) tiling %.3fs of a %.3fs track contiguously "
        "(%d with lyrics, %d instrumental filler)",
        len(covered),
        covered[-1].end if covered else 0.0,
        track_duration,
        voiced,
        len(covered) - voiced,
    )
    return covered


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def slice_audio(
    audio_path: Path,
    alignment: AlignmentResult,
    hardware: HardwareProfile,
    chunks_dir: Path,
    cover_instrumentals: bool = False,
    *,
    shot_lengths: Sequence[ShotLength] = (),
    instrumental_shot_seconds: float | None = None,
) -> tuple[AudioChunk, ...]:
    """Slice ``audio_path`` per ``alignment`` into ``AudioChunk``s honoring
    ``hardware``'s min/max chunk-duration window (clamped into H3's trained
    frame range -- see ``_effective_bounds``) and H3's frame-quantization
    grid (issue #20).

    Exports ``chunk_{idx:03d}.wav`` into ``chunks_dir`` (created if missing)
    for every resulting chunk. Chunk ``start``/``end`` are always seconds
    from the beginning of the master track, matching the timeline convention
    in ``contracts.py``. Every emitted chunk's ``frame_count`` is set to the
    exact H3 ``length`` its audio was sliced to match, so Stage 4a can inject
    it directly instead of re-deriving (and possibly re-rounding) it from
    ``duration``.

    ``cover_instrumentals`` adds a fourth pass (``_cover_instrumentals``)
    that fills the unvoiced spans between lyric lines -- and the intro and
    outro -- with instrumental filler chunks, and re-anchors every chunk so
    the returned tuple tiles ``[0, track_duration]`` contiguously. Without
    it, only voiced spans are rendered and Stage 5's concat silently desyncs
    the result against the master track; see that function's docstring.

    ``shot_lengths`` (issue #27) are the shot plan's editorial shot-length
    requests -- see :class:`~music_video_maker.shot_plan.ShotLength` and
    ``shot_plan.shot_length_requests``. ``instrumental_shot_seconds`` gives
    the unvoiced spans a longer ceiling than the sung ones. Both default to
    "no opinion", which leaves the timeline byte-identical to what it was
    before #27: a long take has to be *asked for*, never inferred, because
    nothing longer than
    :data:`~music_video_maker.shot_plan.MEASURED_MAX_FRAMES` frames has ever
    been rendered on this card.

    Both need the contiguous timeline ``cover_instrumentals`` builds -- there
    is no coherent way to retile a covering that has holes in it -- so they
    are ignored (loudly) when it is off.
    """
    if not alignment.segments and not cover_instrumentals:
        # With cover_instrumentals on, zero segments is not "nothing to
        # slice" -- it is a wholly unvoiced track (an empty lyrics_file, or
        # every line stripped out deliberately), and _cover_instrumentals
        # below already tiles the whole [0, track_duration] span as filler
        # when there are no voiced pieces to anchor around. Bailing out here
        # unconditionally used to block that path outright.
        logger.warning("slice_audio called with no aligned segments; nothing to slice")
        return ()

    if (shot_lengths or instrumental_shot_seconds is not None) and not cover_instrumentals:
        logger.warning(
            "Ignoring %d editorial shot length request(s)%s: they retile a contiguous "
            "timeline, and instrumental_coverage is off, so this run only renders the "
            "voiced spans and has no timeline to retile. Turn instrumental_coverage on "
            "(it is the default, and Stage 5 silently desyncs without it).",
            len(shot_lengths),
            "" if instrumental_shot_seconds is None else " and instrumental_shot_seconds",
        )
        shot_lengths = ()
        instrumental_shot_seconds = None

    eff_min, eff_max, grid = _effective_bounds(hardware)

    segments = tuple(sorted(alignment.segments, key=lambda s: s.start))

    groups = _merge_for_minimum(segments, alignment.track_duration, eff_min)
    raw_pieces = _split_for_maximum(groups, eff_max, grid)

    # Pass 3: quantize whatever pass 2 left un-split, in order, so each
    # piece's padding room is computed against its *already-finalized*
    # neighbors (the previous piece's real, post-quantization end; the next
    # piece's original start, which pass 2 never moves for a piece that
    # wasn't itself split).
    pieces: list[_Piece] = []
    prev_end = 0.0
    for idx, piece in enumerate(raw_pieces):
        if piece.frame_count is not None:
            quantized = piece
        else:
            next_boundary = (
                raw_pieces[idx + 1].start if idx + 1 < len(raw_pieces) else alignment.track_duration
            )
            quantized = _quantize_single_piece(piece, grid, prev_end, next_boundary)
        pieces.append(quantized)
        prev_end = quantized.end

    if cover_instrumentals:
        pieces = _cover_instrumentals(
            pieces,
            segments,
            alignment.track_duration,
            eff_min,
            eff_max,
            grid,
            shot_lengths=shot_lengths,
            instrumental_shot_seconds=instrumental_shot_seconds,
        )

    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    master = AudioSegment.from_file(str(audio_path))

    chunks: list[AudioChunk] = []
    prev_end = 0.0
    for idx, piece in enumerate(pieces):
        if piece.start < prev_end - _EPS:
            logger.error(
                "Timeline drift detected building chunk %d: start=%.6f precedes previous "
                "chunk's end=%.6f. Refusing to emit a desynced chunk.",
                idx,
                piece.start,
                prev_end,
            )
            raise ValueError(
                f"chunk {idx} start {piece.start!r} precedes previous chunk end {prev_end!r}"
            )

        if piece.frame_count is None or not grid.is_valid(piece.frame_count):
            logger.error(
                "Chunk %d (segments=%s) has no valid frame_count (%r) at emission time.",
                idx,
                tuple(m.index for m in piece.members),
                piece.frame_count,
            )
            raise ChunkFrameMismatchError(
                f"chunk {idx}: frame_count {piece.frame_count!r} is not a valid H3 grid point "
                f"(source_segment_indices={tuple(m.index for m in piece.members)!r})"
            )

        piece_duration = piece.end - piece.start
        expected_duration = grid.frames_to_seconds(piece.frame_count)
        if abs(piece_duration - expected_duration) > 1e-3:
            logger.error(
                "Chunk %d (segments=%s): frame_count=%d implies %.6fs but the sliced duration "
                "is %.6fs -- these must match exactly or Stage 4a's rendered video will drift "
                "from its own audio stem.",
                idx,
                tuple(m.index for m in piece.members),
                piece.frame_count,
                expected_duration,
                piece_duration,
            )
            raise ChunkFrameMismatchError(
                f"chunk {idx}: frame_count {piece.frame_count!r} implies duration "
                f"{expected_duration!r} but sliced duration is {piece_duration!r} "
                f"(source_segment_indices={tuple(m.index for m in piece.members)!r})"
            )

        start_ms = round(piece.start * 1000)
        end_ms = round(piece.end * 1000)
        sliced = master[start_ms:end_ms]

        # A chunk may legitimately run past the end of the master: the outro
        # filler is grid-quantized, and the nearest grid point can land
        # beyond the final sample. pydub truncates such a slice silently,
        # which would hand Stage 4a an audio stem shorter than its own
        # frame_count implies -- exactly the audio/video length mismatch
        # issue #20 exists to eliminate. Pad the shortfall with silence so
        # the stem is always exactly frame_count frames long.
        shortfall_ms = (end_ms - start_ms) - len(sliced)
        if shortfall_ms > 0:
            logger.info(
                "Chunk %d runs %dms past the end of the master track; padding the stem with "
                "silence so its duration still matches frame_count=%d exactly.",
                idx,
                shortfall_ms,
                piece.frame_count,
            )
            sliced = sliced + AudioSegment.silent(
                duration=shortfall_ms, frame_rate=master.frame_rate
            ).set_channels(master.channels).set_sample_width(master.sample_width)

        out_path = chunks_dir / f"chunk_{idx:03d}.wav"
        sliced.export(str(out_path), format="wav")

        text = _text_within(piece.members, piece.start, piece.end)
        if not piece.members or not text:
            # Instrumental filler: no lyric, no source segments, and no
            # character of its own -- Stage 2b falls back to the default lead
            # vocalist and its empty-text instrumental clause.
            chunks.append(
                AudioChunk(
                    chunk_id=idx,
                    audio_file=out_path,
                    start=piece.start,
                    end=piece.end,
                    text="",
                    characters=(),
                    source_segment_indices=(),
                    is_split_continuation=piece.is_split_continuation,
                    frame_count=piece.frame_count,
                    is_instrumental=True,
                )
            )
            prev_end = piece.end
            continue

        distinct = {m.character for m in piece.members}
        if len(distinct) > 1:
            # Attribute to whichever member contributed the most voiced
            # duration, not whichever comes first (issue #40). A merge can now
            # be a genuine mid-chunk handover between singers (issue #33 made
            # per-line attribution real), and picking by position silently
            # gave one singer's face to audio that was mostly the other's --
            # confirmed by ear on "The Lucky Ones" chunk 26. Voiced duration
            # (``AlignedSegment.duration``, already computed by Stage 1) is
            # the measure, not word count: a held note with few words can
            # still dominate a chunk's screen time, which is what perception
            # tracks. Strict ``>`` means a later member must *exceed* the
            # current leader to take over, so an exact tie keeps the first
            # member -- today's behaviour, unchanged.
            dominant = piece.members[0]
            for member in piece.members[1:]:
                if member.duration > dominant.duration + _EPS:
                    dominant = member
            total_voiced = sum(m.duration for m in piece.members)
            logger.warning(
                "Chunk %d merges segments with differing characters %s; attributing to "
                "%r, which contributes %.3fs of the chunk's %.3fs total voiced duration.",
                idx,
                sorted(c for c in distinct if c is not None),
                dominant.character,
                dominant.duration,
                total_voiced,
            )
        else:
            dominant = piece.members[0]
        # Deliberately the dominant member's cast, not the union: who is
        # audible across a merge is a real question (issue #33 widened the
        # field so it *can* be answered), but answering it by union here
        # would silently put a second face on screen for every merged chunk.
        # Issue #40 fixed *which* member wins; taking the union is still a
        # deferred staging decision, not made here.
        characters = dominant.characters

        chunks.append(
            AudioChunk(
                chunk_id=idx,
                audio_file=out_path,
                start=piece.start,
                end=piece.end,
                text=text,
                characters=characters,
                source_segment_indices=tuple(m.index for m in piece.members),
                is_split_continuation=piece.is_split_continuation,
                frame_count=piece.frame_count,
            )
        )
        prev_end = piece.end

    logger.info(
        "Sliced %d chunk(s) from %d aligned segment(s) into %s (frame-grid quantized, "
        "trained range %d-%d frames)",
        len(chunks),
        len(segments),
        chunks_dir,
        grid.trained_min_frames,
        grid.trained_max_frames,
    )
    _log_unmeasured_chunks(chunks, grid)
    return _fold_counterpoint(tuple(chunks), alignment)


def _fold_counterpoint(
    chunks: tuple[AudioChunk, ...], alignment: AlignmentResult
) -> tuple[AudioChunk, ...]:
    """Give every chunk the counterpoint audible in its span (issue #33 L3).

    A :class:`~music_video_maker.alignment.CounterpointAlignmentResult`
    carries concurrent segments -- text sung at the same time as the spine,
    in different words. They are deliberately NOT in the spine's segment
    index space, so they never participate in boundary planning above; this
    is a pure annotation pass over the finished timeline. Per chunk, each
    overlapping stream contributes one joined text (its lines in written
    order) and its voices -- appended after the spine's, which keeps the
    primary slot (#40's dominant-voice rule), so multi-reference staging
    picks the counterpoint faces up with no special case.

    A plain :class:`AlignmentResult` (every pre-#33 caller) has no
    ``concurrent_in_span`` and passes through untouched.
    """
    concurrent_lookup = getattr(alignment, "concurrent_in_span", None)
    if concurrent_lookup is None:
        return chunks

    folded: list[AudioChunk] = []
    annotated = 0
    for chunk in chunks:
        overlapping = concurrent_lookup(chunk.start, chunk.end)
        if not overlapping or chunk.is_instrumental:
            folded.append(chunk)
            continue

        # Group per stream, preserving written order within each.
        streams: dict[tuple[int, int], list] = {}
        for segment in overlapping:
            streams.setdefault((segment.block_index, segment.stream_index), []).append(segment)

        texts: list[str] = []
        stream_characters: list[tuple[str, ...]] = []
        merged_characters = list(chunk.characters)
        for key in sorted(streams):
            members = streams[key]
            texts.append(" ".join(s.text for s in members))
            voices = members[0].characters
            stream_characters.append(voices)
            for voice in voices:
                if voice not in merged_characters:
                    merged_characters.append(voice)

        folded.append(
            dataclasses.replace(
                chunk,
                characters=tuple(merged_characters),
                concurrent_texts=tuple(texts),
                concurrent_characters=tuple(stream_characters),
            )
        )
        annotated += 1

    if annotated:
        logger.info(
            "Counterpoint: %d chunk(s) carry a second simultaneous lyric stream", annotated
        )
    return tuple(folded)


def _log_unmeasured_chunks(chunks: Sequence[AudioChunk], grid: FrameGrid) -> None:
    """Name, once per run, every chunk longer than anything ever rendered here.

    Reported rather than refused, deliberately: 362 frames is inside H3's
    trained range and the whole point of issue #27 is that a long take is
    available. But nothing above
    :data:`~music_video_maker.shot_plan.MEASURED_MAX_FRAMES` has been rendered
    on this 4090 at any resolution, and an over-committed card here does not
    raise CUDA OOM -- it goes silent mid-load and wedges the host past SIGKILL
    (issues #23, #24). A run that is about to spend hours on frame counts
    nobody has measured should say so before it starts, not after.

    Note this fires for filler chunks too: with ``max_chunk_seconds`` at H3's
    trained ceiling, a long instrumental already tiles into chunks well past
    141 frames without anyone asking for a long take.
    """
    over = [c for c in chunks if (c.frame_count or 0) > MEASURED_MAX_FRAMES]
    if not over:
        return
    longest = max(over, key=lambda c: c.frame_count or 0)
    logger.warning(
        "%d of %d chunk(s) exceed %d frames -- the longest anything ever rendered on this "
        "card. The longest is chunk %d at %d frames (%.3fs). This is inside H3's trained "
        "range and costs no extra wall clock, but VRAM behaviour above %d frames is "
        "unmeasured here and a card that runs out can wedge the host rather than raising "
        "CUDA OOM. Watch the first one.",
        len(over),
        len(chunks),
        MEASURED_MAX_FRAMES,
        longest.chunk_id,
        longest.frame_count,
        longest.duration,
        MEASURED_MAX_FRAMES,
    )
