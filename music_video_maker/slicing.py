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
   ``is_split_continuation=True``. An interior split point that would
   otherwise land inside one of the group's own aligned segments prefers
   snapping to that segment's nearer edge instead (issue #70) -- a purely
   geometric split point has no idea a sung phrase is still in progress
   there, and the two halves render as independent shots with no error
   anywhere. The snap is only ever a preference: if it would violate the
   duration window (or there is no usable edge to snap to at all -- a single
   continuous segment spanning the whole group has none), the geometric
   point is used exactly as before and the cut is logged at WARNING with how
   far through the segment it lands, so a 92%-through fragment is visible
   before hours of GPU time rather than after. See ``_snap_to_segment_edge``.
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

A fourth, opt-in pass (``_cover_instrumentals``, issue #21) re-anchors every
chunk into a contiguous ``[0, track_duration]`` covering -- see that
function's docstring. Measured on real material, its own grid-quantized
filler tiling is issue #70's *larger* source of mid-utterance boundary cuts,
bigger than pass 2's max-duration split: on "Deathless" it moved every one
of pass 2's 12 pre-retiling hits to a new position and introduced 15 more
that pass 2 never touched, for 27 in total. Unlike pass 2, this pass does
not attempt to *avoid* those cuts -- only report them, at the boundary
position a render will actually use, via
``_log_final_boundary_segment_cuts`` -- because doing so safely would mean
biasing filler-tiling choices without weakening this file's one
non-negotiable invariant (below): a change not to make without a render to
verify it against.

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

A sixth, always-on refinement runs on top of pass 4's final timeline (issue
#79): **leading vocal offset**. H3 starts the mouth at frame 0 of a chunk
regardless of where in that chunk the voice actually starts -- a chunk
prompted with a lyric that begins seconds into its own span is out of phase
for the whole chunk, not just the silent lead-in. ``_prefer_vocal_onset``
moves what boundaries it safely can, one boundary at a time, by transferring
whole grid steps from a chunk to its predecessor -- compensated so the pair's
combined duration, and every other boundary in the timeline, is untouched;
this is deliberately not the kind of filler-tiling change #70 was closed
without making. ``_log_leading_vocal_offset`` then reports whatever offset
survives, unconditionally, so a leftover offset is visible before GPU time
even where the refinement had no room to act.

A seventh, opt-in pass runs last of all and is the only one that changes what
a boundary *is* (issue #100): **boundary overrun**. Passes 2-6 all treat a
chunk's length and its boundary as one decision, because H3 renders a chunk at
exactly its own length and only ``5 + 17k`` lengths exist -- which is why #70
closed on the finding that quantizing a chunk *moves* its boundary, and why the
depth preference it asked for cost 4-6 grid steps against neighbours with none
to give. Another pipeline built on this one took the other side of the trade
and reported it in issue #100: ask the generator for the next valid length
*past* the boundary, cut the audio stem to that same length, and discard the
overrun when the video is assembled. The boundary becomes a content decision,
the length stays a VRAM decision, and the grid is paid in frames nobody
watches. ``boundary_overrun`` is off by default -- it re-cuts every chunk in
the song and has not been rendered here. See ``_overrun_timeline``.

An opt-in replacement for pass 4's *placement* sits between the tiling and
pass 6 (``phrase_aware_slicing``): **plan every boundary at once**. Passes
1-4 decide boundaries one at a time by grid arithmetic, and #70 measured
where that lands -- 24 of "Deathless"'s 79 boundaries inside a sung phrase,
and a local preference that could never pay for a move because the chunk
that must pay sits at the 124-frame floor. Choosing the whole tiling at once,
over every grid-valid sequence of chunk lengths, lets the slack a move needs
come from anywhere in the passage -- an instrumental gap seconds away --
instead of only from the two chunks either side of the boundary. Passes 6
and 7 then run on its result unchanged. See ``_plan_phrase_boundaries``.

A cross-cutting fix, not a pass at all, because it moves no boundary (issue
#92): a chunk can merge segments from two singers, and the pipeline picks a
dominant one to attribute the chunk to but used to hand them the *whole*
span's text -- including the other singer's words. Measured on "Deathless",
every one of the three affected chunks sits at a character-change gap of
0.000-1.510s, well inside H3's 124-frame trained floor (5.167s), so no legal
chunk boundary can separate the two singers there -- see
``_merge_for_minimum``'s docstring for the rejected "don't merge across
characters" alternative. What *can* change without moving anything is what
the chunk is prompted with: ``_prompted_members`` narrows a chunk's text and
its issue-#79 onset to whichever character ``_dominant_character_member``
finds contributes the most voiced duration *inside the chunk* (not the whole
segment, which may extend well outside it -- the old measure). Used at every
site that decides what a chunk is prompted with, so "what is this chunk
prompted with" has one answer everywhere, the same reasoning issue #79
factored ``_words_attributed_to`` out for. ``source_segment_indices`` is
deliberately left un-narrowed: it records what the *audio* contains, and the
audio genuinely contains every singer merged into the chunk.

Timeline integrity is non-negotiable: every chunk's ``start``/``end`` stays
anchored to the original ``AlignedSegment`` timeline as closely as the frame
grid allows, ``source_segment_indices`` records exactly which original
segments fed each chunk, and chunks are always chronological and
non-overlapping. Stage 5's final mux depends on this -- a drifting boundary
would silently desync the whole video.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from music_video_maker.contracts import (
    AlignedSegment,
    AlignmentResult,
    AudioChunk,
    FrameGrid,
    HardwareProfile,
    WordTiming,
)
from music_video_maker.envelope import CALIBRATED_CEILING, MeasuredCeiling
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
    render_frames: int | None = None
    """The grid-valid length this piece will be *rendered* at, when the
    issue-#100 overrun pass has decoupled that from ``frame_count`` (which is
    then the frames the piece *keeps*). ``None`` everywhere else, including
    every pre-#100 path, and that is what keeps ``frame_count`` grid-valid by
    default."""


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
    """Pass 1 (see the module docstring): merge/pad adjacent segments up to
    ``min_chunk_seconds``, freely across characters.

    Issue #92 proposed refusing to merge across a character change instead.
    Measured on "Deathless" and found impossible, not merely unimplemented:
    the three chunks that end up billed to one singer while carrying
    another's words (ids 35, 58, 73) come from character-change gaps of
    0.000s, 1.510s and 0.000s -- pass-1 groups ``[23,24]``, ``[38,39]``,
    ``[52,53]``. H3's trained floor is 124 frames = 5.167s, so *any* legal
    chunk covering one side of a change that close also covers the other; a
    refusal here would not even survive to the render, because
    ``_cover_instrumentals`` re-derives every chunk's members from span
    overlap afterward regardless of how this pass grouped them. The fix for
    what a merge like this produces lives downstream, in what the chunk is
    *prompted with* (``_prompted_members``, issue #92) -- not here, where
    there is no alternative timeline to fall back to.
    """
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


def _quantize_split_end(
    cursor: float, target_end: float, grid: FrameGrid
) -> tuple[int, float, bool]:
    """Grid-quantize a candidate split boundary the way pass 2 always has:
    ceil, never down, from ``cursor`` to ``target_end``.

    Factored out of :func:`_split_for_maximum`'s main loop so
    :func:`_snap_to_segment_edge` (issue #70) can trial a segment-edge
    candidate through exactly the same math the real split uses, rather than
    a second implementation that could quietly disagree with it. See the
    ceil-vs-round comment inline in :func:`_split_for_maximum` for why this
    always rounds up.

    Returns ``(frames, piece_end, was_clamped)``. ``was_clamped`` is True
    when ``clamp_to_trained`` actually changed the grid-quantized frame
    count -- i.e. the point that truly reaches ``target_end`` lies outside
    H3's trained range, so what got returned is *not* that point but the
    nearest one inside the trained range instead. The unsnapped default
    path ignores this; :func:`_snap_to_segment_edge` cannot -- see its
    docstring for why a clamped snap is not a snap at all.
    """
    raw_duration = target_end - cursor
    requested_frames = math.ceil(grid.seconds_to_frames(raw_duration) - _EPS)
    frames_unclamped = grid.quantize_up(requested_frames)
    frames = grid.clamp_to_trained(frames_unclamped)
    return frames, cursor + grid.frames_to_seconds(frames), frames != frames_unclamped


def _segment_containing(
    point: float, members: tuple[AlignedSegment, ...]
) -> AlignedSegment | None:
    """The member whose *interior* strictly contains ``point``, or ``None``.

    Touching an edge (within ``_EPS``) is a clean cut between two segments,
    not a split of one -- so a point exactly equal to ``start`` or ``end``
    does not count as "inside".
    """
    for member in members:
        if member.start + _EPS < point < member.end - _EPS:
            return member
    return None


def _percent_through(point: float, segment: AlignedSegment) -> float:
    """How far ``point`` sits into ``segment``, as a percentage -- e.g. a
    92%-through fragment is a boundary that lands just before the segment
    ends, cutting off only its last 8%."""
    span = segment.end - segment.start
    if span <= _EPS:
        return 0.0
    return 100.0 * (point - segment.start) / span


def _snap_to_segment_edge(
    default_end: float,
    cursor: float,
    group_end: float,
    members: tuple[AlignedSegment, ...],
    eff_min: float,
    eff_max: float,
    grid: FrameGrid,
) -> tuple[int, float, AlignedSegment, str] | None:
    """Try snapping a pass-2 split boundary to the nearer edge of whichever
    aligned segment ``default_end`` falls inside (issue #70).

    A purely geometric split point has no idea a sung phrase is still in
    progress there: the alignment already knows exactly where every segment
    starts and ends, and this makes that a *preferred* cut point without
    ever forcing it -- the caller keeps the original geometric boundary
    whenever this returns ``None``.

    This function only ever returns a snap that is *actually* clean -- a
    caller that gets a non-``None`` result never needs to double-check it,
    and never needs to log both an accepted snap and a mid-utterance warning
    about the very same boundary. Three independent reasons a snap is
    refused, all deliberate:

    * The nearer edge doesn't leave room for a real split (``cursor < edge <
      group_end`` fails) -- a single continuous segment spanning the whole
      group has no internal edge to offer at all; both of its "edges" are
      the group's own start and end.
    * The point that actually reaches the edge would need
      ``clamp_to_trained`` to relocate it (:func:`_quantize_split_end`'s
      ``was_clamped``) -- checked *before* comparing against
      ``[eff_min, eff_max]``, not after. Checking post-clamp is vacuous:
      clamping forces the duration into the trained range, which for a
      real profile *is* ``[eff_min, eff_max]``, so the check always passes
      regardless of how far the clamp actually moved the boundary. On
      "Deathless" this was measured directly: segment 7's start sits only
      1.66s past the cursor, clamp_to_trained pushed the "snap" out to the
      5.167s trained floor, and every one of that run's 6 attempted snaps
      landed back inside the very segment they claimed to avoid (45.7% and
      up) -- announced as clean while never once being clean.
    * Even unclamped, plain grid rounding (``quantize_up``, at most one
      grid step -- ~0.708s) can still leave the final boundary inside the
      same segment or push it into a different one close behind. This is
      checked directly against the *actual* quantized ``piece_end``, not
      inferred from the duration.

    Returns ``(frames, piece_end, segment, edge_name)`` for an accepted
    snap; ``piece_end`` is quantized through :func:`_quantize_split_end`,
    the *same* function the unsnapped path uses, so a caller that accepts
    this never disagrees with how the rest of this module rounds.
    """
    containing = _segment_containing(default_end, members)
    if containing is None:
        return None

    to_start = default_end - containing.start
    to_end = containing.end - default_end
    edge_name, candidate = (
        ("start", containing.start) if to_start <= to_end else ("end", containing.end)
    )

    if not (cursor + _EPS < candidate < group_end - _EPS):
        return None

    frames, piece_end, was_clamped = _quantize_split_end(cursor, candidate, grid)
    if was_clamped:
        return None

    duration = grid.frames_to_seconds(frames)
    if not (eff_min - _EPS <= duration <= eff_max + _EPS):
        return None

    if _segment_containing(piece_end, members) is not None:
        return None

    return frames, piece_end, containing, edge_name


def _split_for_maximum(
    groups: list[_Group], eff_min: float, max_chunk_seconds: float, grid: FrameGrid
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

    Every *interior* boundary (every piece but the last -- the last always
    ends at the group's own true end, which is already a real segment edge)
    prefers to land on the nearer edge of whichever segment it would
    otherwise cut through (issue #70), via :func:`_snap_to_segment_edge`.
    Whether snapped or not, the piece actually emitted is checked again
    after quantization -- grid rounding can itself push a boundary into a
    segment it didn't start in, or (rarely) back out of one a snap aimed
    at -- and a boundary that still lands inside a segment logs a WARNING
    naming the segment and how far through it the cut falls.
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
            default_end = group.end if is_last else cursor + piece_duration

            # ceil, not round: quantize_up must never imply a duration below
            # raw_duration. A plain round() can land exactly on a valid grid
            # frame that is a hair *short* of raw_duration whenever
            # raw_duration's true (unrounded) frame count has a fractional
            # part just above that grid point (e.g. 6.6s = 158.4 frames,
            # where 158 itself is already grid-valid but 0.4 frames short).
            frames, piece_end, _was_clamped = _quantize_split_end(cursor, default_end, grid)

            if not is_last:
                # `_snap_to_segment_edge` only ever returns a snap that is
                # provably clean (see its docstring), so this is an if/else,
                # not two independent checks: a boundary gets an "accepted
                # snap" INFO or a "still cuts mid-utterance" WARNING, never
                # both about the same boundary.
                snapped = _snap_to_segment_edge(
                    default_end, cursor, group.end, group.members, eff_min, max_chunk_seconds, grid
                )
                if snapped is not None:
                    frames, piece_end, segment, edge_name = snapped
                    logger.info(
                        "Split boundary near %.3fs snapped to the %s edge (%.3fs) of segment "
                        "index=%d (%r, %.3f-%.3fs), avoiding a mid-utterance cut (issue #70).",
                        default_end,
                        edge_name,
                        getattr(segment, edge_name),
                        segment.index,
                        segment.text,
                        segment.start,
                        segment.end,
                    )
                else:
                    landed = _segment_containing(piece_end, group.members)
                    if landed is not None:
                        pct = _percent_through(piece_end, landed)
                        logger.warning(
                            "Split boundary at %.3fs lands %.1f%% through segment index=%d "
                            "(%r, %.3f-%.3fs) -- cutting a sung phrase mid-utterance; the two "
                            "halves will render as independent shots with no error anywhere "
                            "else. No segment-edge snap reaches this segment's edge without "
                            "either falling outside the %.3fs-%.3fs duration window or being "
                            "relocated by H3's trained-range clamp (issue #70).",
                            piece_end,
                            pct,
                            landed.index,
                            landed.text,
                            landed.start,
                            landed.end,
                            eff_min,
                            max_chunk_seconds,
                        )

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
            "above the %d frames MEASURED_MAX_FRAMES is calibrated at -- a constant, not a "
            "live reading, and renders here have gone past it (issue #98). Longer shots cost "
            "no extra wall clock (fewer chunks over the same total frames), but VRAM at this "
            "frame count has never been measured here.",
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
) -> tuple[list[tuple[float, int, bool]], frozenset[int]]:
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

    Returns ``(boundaries, pinned_indices)``. ``pinned_indices`` names every
    final-list position that is an honoured request's own shot chunk (never
    the retiled tail behind it, which is not itself an authored length) --
    issue #79's :func:`_prefer_vocal_onset` must never nudge a boundary an
    author explicitly set, so it needs to know which ones those are. Stable
    across every later request in this same call: requests are applied in
    ascending start order and an honoured one's match can only ever fall at
    or after the previous one's ``applied_until``, so a later splice never
    touches a position an earlier request already pinned.
    """
    frames = [count for _, count, _ in boundaries]
    continuations = [flag for _, _, flag in boundaries]
    original_total = sum(frames)
    applied_until = -1.0  # end of the last honoured shot, in seconds
    last_applied_index = -1
    pinned: set[int] = set()
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
                "Chunk at %.3fs will render %d frames (%.3fs), above the %d frames "
                "MEASURED_MAX_FRAMES is calibrated at -- a constant, not a live reading, and "
                "renders here have gone past it (issue #98). It is inside H3's trained range "
                "and costs no extra wall clock, but temporal VAE decode memory scales "
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
        pinned.add(index)
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
    return (
        [
            (start, count, flag)
            for start, count, flag in zip(starts, frames, continuations, strict=True)
        ],
        frozenset(pinned),
    )


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


def _words_attributed_to(
    member: AlignedSegment, start: float, end: float
) -> list[WordTiming]:
    """Words of ``member`` whose MIDPOINT lands in ``[start, end)``.

    This is the word-midpoint attribution rule itself, factored out so
    :func:`_text_within` and :func:`_first_prompted_word_onset` (issue #79)
    share exactly one implementation of "which words belong to this chunk" --
    two independent copies of that rule is exactly the kind of drift
    CLAUDE.md warns a second implementation invites within a month.
    """
    return [w for w in member.words if start - _EPS <= (w.start + w.end) / 2.0 < end - _EPS]


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

    A word belongs to the chunk containing its **midpoint** (see
    :func:`_words_attributed_to`). Chunks tile contiguously and without
    overlap, so every word lands in exactly one chunk: none duplicated, none
    dropped. Segments with no word timings fall back to their full text --
    there is nothing to slice by, and dropping them silently would be worse
    than a slightly wide attribution.
    """
    parts: list[str] = []
    for member in members:
        if not member.words:
            parts.append(member.text)
            continue
        kept = [w.word.strip() for w in _words_attributed_to(member, start, end)]
        if kept:
            parts.append(" ".join(kept))
    return " ".join(p for p in parts if p).strip()


def _first_prompted_word_onset(
    members: tuple[AlignedSegment, ...], start: float, end: float
) -> float | None:
    """The earliest start time of a word this chunk is actually prompted to
    sing -- issue #79's "leading vocal offset", measured from the *same*
    attribution rule :func:`_text_within` uses (:func:`_words_attributed_to`),
    so this can never disagree with what a chunk is actually prompted with.

    A segment with no word timings falls back to its own start clamped into
    the window, mirroring :func:`_text_within`'s whole-segment fallback --
    there is nothing finer to attribute for it either.

    Returns ``None`` when nothing attributes to this window at all: an
    instrumental chunk has no leading vocal offset to report.
    """
    best: float | None = None
    for member in members:
        if not member.words:
            candidate = max(member.start, start)
        else:
            words = _words_attributed_to(member, start, end)
            if not words:
                continue
            candidate = min(w.start for w in words)
        if best is None or candidate < best:
            best = candidate
    return best


def _voiced_seconds_by_character(
    members: tuple[AlignedSegment, ...], start: float, end: float
) -> dict[str | None, float]:
    """In-chunk voiced seconds per character, summed over every member that
    carries it -- the same clipped-overlap measure :func:`_voiced_seconds_within`
    uses for the pooled total, just split by :attr:`AlignedSegment.character`
    (issue #92)."""
    totals: dict[str | None, float] = {}
    for member in members:
        overlap = max(0.0, min(member.end, end) - max(member.start, start))
        totals[member.character] = totals.get(member.character, 0.0) + overlap
    return totals


def _dominant_character_member(
    members: tuple[AlignedSegment, ...], start: float, end: float
) -> AlignedSegment:
    """The member whose CHARACTER contributes the most voiced duration
    actually inside ``[start, end)`` (issue #92).

    Replaces the old rule, which compared :attr:`AlignedSegment.duration` --
    the whole segment's length, most of which can lie outside the chunk that
    is actually rendered. Measured on "Deathless": both rules pick the same
    winner on all three of its cross-character chunks, so this changes no
    attribution there -- it changes what the log *reports*. Chunk 35's old
    line credited Jan with "9.030s of the chunk's 10.590s" (his segment's
    whole duration, most of it outside the chunk); inside the chunk it is
    4.017s of the chunk's 5.577s total. The chunk is what gets rendered;
    measure the chunk.

    Strict ``>`` means a later character must *exceed* the current leader's
    in-chunk total to take over, so an exact tie keeps the earliest
    character -- unchanged from issue #40's behaviour. Returns the FIRST
    member carrying the winning character, matching how the caller derives
    ``AudioChunk.characters`` from a single representative member.
    """
    totals = _voiced_seconds_by_character(members, start, end)
    dominant_character = members[0].character
    best = totals.get(dominant_character, 0.0)
    seen = {dominant_character}
    for member in members[1:]:
        character = member.character
        if character in seen:
            continue
        seen.add(character)
        total = totals.get(character, 0.0)
        if total > best + _EPS:
            best = total
            dominant_character = character
    for member in members:
        if member.character == dominant_character:
            return member
    return members[0]  # unreachable: dominant_character always comes from members


def _prompted_members(
    members: tuple[AlignedSegment, ...], start: float, end: float, *, quiet: bool = False
) -> tuple[AlignedSegment, ...]:
    """The members this chunk is actually prompted with (issue #92).

    Narrows ``members`` down to whichever character
    :func:`_dominant_character_member` picks, so a chunk that merges two
    singers' segments is prompted with only the one it is attributed to --
    never both, which is the disagreement between the attribution rule and
    the text rule issue #92 reports (naming both risks issue #82's morphing
    defect; see the module docstring).

    Used at every site that decides what a chunk is prompted with
    (:func:`slice_audio`'s ``text``/``characters``,
    :func:`_log_leading_vocal_offset`, :func:`_prefer_vocal_onset`), so those
    three can never disagree -- the same reasoning issue #79 factored
    :func:`_words_attributed_to` out for.

    Deliberately NOT used for :func:`_voiced_seconds_within`,
    :func:`_is_instrumental_span`, or the ``_MIN_VOICED_FRACTION`` demotion:
    those ask how much voice is in the *audio*, which genuinely contains
    every singer merged into the chunk, not just the one it is prompted
    with.

    Guarded against emptying the prompt: narrowing to the dominant
    character's own members can still leave :func:`_text_within` with
    nothing, when that character contributes voiced overlap but no word
    whose midpoint lands in the window. When that happens this falls back to
    the full ``members`` tuple and logs a warning, rather than silently
    prompting the chunk as instrumental. ``quiet`` takes the same fallback
    without the warning, for :func:`_plan_phrase_boundaries`, which asks this
    question of thousands of trial chunks that are never emitted.
    """
    if not members:
        return members
    dominant = _dominant_character_member(members, start, end)
    narrowed = tuple(m for m in members if m.character == dominant.character)
    if not _text_within(narrowed, start, end):
        if quiet:
            return members
        logger.warning(
            "Chunk span %.3f-%.3fs: narrowing its prompt to the dominant character %r left "
            "no prompted words (it contributes voiced overlap but no word whose midpoint "
            "lands in this window); falling back to the full %d-member text rather than "
            "silently emptying the prompt (issue #92).",
            start,
            end,
            dominant.character,
            len(members),
        )
        return members
    return narrowed


def _voiced_seconds_within(
    segments: tuple[AlignedSegment, ...], start: float, end: float
) -> float:
    """Total voiced time inside ``[start, end)``, summed over overlaps."""
    return sum(
        max(0.0, min(s.end, end) - max(s.start, start))
        for s in _segments_overlapping(segments, start, end)
    )


def _log_effective_frame_window(eff_min: float, eff_max: float, grid: FrameGrid) -> None:
    """Say, once per run, what ``max_chunk_seconds`` actually bought (#98).

    A duration is not a frame count. H3's ``length`` lives on a ``5 + 17k``
    grid, so the ceiling a run enforces is the largest grid point *at or
    below* ``max_chunk_seconds`` -- and nothing anywhere says which one that
    turned out to be. Two consequences, both of which cost real time:

    * ``max_chunk_seconds = 12.0`` is **277** frames (11.542 s), not the 288
      that ``12.0 * 24`` suggests. 288 is not on the grid at all, so no
      setting produces it.
    * A duration written one decimal short silently buys the previous grid
      point, with no warning, because the value asked for is perfectly legal:
      ``10.833`` gives **243** frames where ``10.834`` gives 260, and
      ``15.083`` gives **345** where H3's trained ceiling is 362.

    This is INFO rather than a warning: nothing here is wrong, it is simply
    the one number an operator planning a long-take run needs and could not
    read anywhere. See ``docs/runbook-288-frame-proof.md``.
    """
    min_frames = _grid_frames_at_or_above(eff_min, grid)
    max_frames = max(_grid_frames_at_or_below(eff_max, grid), min_frames)
    logger.info(
        "Chunk window for this run: %.3f-%.3fs resolves to %d-%d frames on H3's %d+%dk grid "
        "(the ceiling is the largest grid point at or below max_chunk_seconds, so %.3fs buys "
        "%d frames = %.3fs -- a duration written one decimal short silently buys the previous "
        "grid point; issue #98).",
        eff_min,
        eff_max,
        min_frames,
        max_frames,
        grid.base_frames,
        grid.step_frames,
        eff_max,
        max_frames,
        grid.frames_to_seconds(max_frames),
    )


def _log_untenable_segments(
    segments: tuple[AlignedSegment, ...], eff_max: float, grid: FrameGrid
) -> None:
    """Name, once per run, every aligned segment too long for *this run's*
    ceiling to hold whole (issue #70, reopened).

    A sung phrase longer than the effective maximum chunk duration cannot fit
    in any chunk, so it is cut mid-utterance on every render no matter where
    the boundaries fall. That is a different class of defect from a cut that
    could in principle be moved, and the two have different remedies -- which
    is why the message distinguishes them:

    * **Longer than ``eff_max`` but inside H3's trained range.** The remedy is
      a config line. Measured on "Deathless", whose ``max_chunk_seconds`` is
      **8.0** against a trained ceiling of **15.083s**: segments 24 (9.030s)
      and 29 (8.020s) are exactly this case, and they are the phrases cut by
      two of the three chunks issue #70's reopen was built on. Raising the
      ceiling to 12.0s stops both from being cut at all, and takes the whole
      song's mid-phrase cuts from 24 to 18 and its cuts deeper than 2.5s from
      6 to 3.
    * **Longer than the trained maximum itself.** No setting helps; the lever
      is a model with a longer trained context, and saying "raise
      ``max_chunk_seconds``" there would be advice that cannot be taken.

    Note what the first case implies and how easily it is misread: 8.0s looks
    like a hardware limit in a run config and is not one. Check a segment's
    duration against ``eff_max``, never against a remembered number.
    """
    trained_max_s = grid.frames_to_seconds(grid.trained_max_frames)
    for segment in segments:
        if segment.duration <= eff_max + _EPS:
            continue
        if segment.duration <= trained_max_s + _EPS:
            needed_frames = grid.quantize_up(
                math.ceil(grid.seconds_to_frames(segment.duration) - _EPS)
            )
            remedy = (
                f"it does fit inside H3's trained range (up to {trained_max_s:.3f}s), so "
                f"raising max_chunk_seconds to at least "
                f"{grid.frames_to_seconds(needed_frames):.3f}s ({needed_frames} frames) "
                "would let one chunk hold it -- but that is past the "
                f"{MEASURED_MAX_FRAMES} frames MEASURED_MAX_FRAMES is calibrated at, so "
                "prove the VRAM on an attended one-chunk slice first "
                "(docs/runbook-288-frame-proof.md)"
            )
        else:
            remedy = (
                f"it is longer than H3's own trained maximum of {trained_max_s:.3f}s "
                f"({grid.trained_max_frames} frames), so no max_chunk_seconds setting can "
                "hold it -- the only lever is a model with a longer trained context"
            )
        logger.warning(
            "Aligned segment index=%d (%r, %.3f-%.3fs, %.3fs) is longer than this run's "
            "effective maximum chunk duration of %.3fs and so cannot be held whole by any "
            "chunk: it will be cut mid-utterance on every render regardless of where the "
            "boundaries fall, and the two halves render as independent shots. %s "
            "(issue #70).",
            segment.index,
            segment.text,
            segment.start,
            segment.end,
            segment.duration,
            eff_max,
            remedy,
        )


def _log_final_boundary_segment_cuts(
    covered: Sequence[_Piece],
    segments: tuple[AlignedSegment, ...],
    *,
    grid: FrameGrid,
    min_frames: int,
    max_frames: int,
    filler_max_frames: int,
    boundary_overrun: bool = False,
) -> None:
    """Issue #70, second mechanism: even where pass 2's segment-edge
    preference had nothing to snap to -- or nothing to say at all, since
    most chunks here were never split -- instrumental-coverage's own
    re-anchoring can still land a chunk boundary inside a segment.

    ``_cover_instrumentals`` keeps every piece's *duration* exactly as
    passes 1-3 chose it, but repositions its *start* to wherever the
    cumulative filler tiling before it lands -- and filler tiling is
    grid-quantized (:func:`_plan_frames_run`), so each gap it fills can
    close a few tenths of a second short or long of the gap's true size
    (worse, up to several seconds, when a gap itself is shorter than H3's
    trained floor and a filler chunk still has to cover it -- the same
    :func:`FrameGrid.clamp_to_trained` effect :func:`_snap_to_segment_edge`
    has to refuse a pass-2 snap over). That residue has nothing to do with
    where any segment falls, so as it accumulates across a whole song it
    will, essentially by chance, occasionally land a boundary inside a
    segment that pass 1-3 never touched and pass 2 never split.

    Measured directly on "Deathless" (57 aligned segments, 80 chunks): pass
    2 sees 12 mid-segment boundaries pre-retiling; after this pass runs,
    every one of those 12 has moved to a different position (none survives
    unchanged) and 15 entirely new ones appear that pass 2 never had a
    boundary anywhere near -- 27 in total, this pass's own count, against
    pass 2's 7 (some of the 12 pre-retiling hits resolve after retiling by
    chance; that is not something this pass, or pass 2, causes on purpose).

    This function does **not** attempt to avoid that. Doing so safely would
    mean biasing filler-tiling choices -- which piece gets nudged, and by
    how much -- without weakening the contiguity guarantee above (video
    offset == audio offset, "by construction"), the load-bearing invariant
    this whole pass exists to provide, and that is not a change to make
    without a render to verify against; none is available while this is
    being written. It only makes every such cut visible, at the *actual*
    final position a render will use -- unlike pass 2's own warning, whose
    cited timestamp this pass can, and on real material typically does,
    move past.

    Issue #70's 2026-08-23 reopen asked for a *depth-threshold preference* on
    top of this: prefer a segment edge only when the cut would otherwise land
    near the middle of a phrase. It was built as a prototype -- the mirror of
    :func:`_prefer_vocal_onset`, a compensated grid-step transfer moving a
    boundary *earlier* to the phrase start -- and scored against the real
    80-chunk "Deathless" timeline, where it **fired 0 times at every threshold
    tried** (1.0s, 2.0s, 2.5s, 3.0s). Clearing a phrase head is
    all-or-nothing -- land short of the segment's own start and the cut is
    still mid-utterance, just somewhere else, which is precisely the mistake
    the first #70 snap made -- so the move costs 4-6 grid steps of 0.708s,
    while the chunk that has to pay sits at the 124-frame floor in 20 of the
    24 cases and is never more than one step above it in the 6 that matter.
    Depth also failed to separate the labels it was drawn from: the nearest
    unreported chunk (21, 2.517s into its phrase) and the nearest reported one
    (39, 2.578s) are 0.061s apart, and chunk 19 at 48.6% is *nearer the
    middle* than either complaint and was never mentioned in two independent
    viewings.

    **Issue #100 is the reason that cost exists, and the way out of it.** The
    4-6 steps are the price of the grid deciding the boundary; a run with
    ``boundary_overrun`` on pays the grid in discarded frames instead and
    moves these boundaries for nothing. What survives there is only what the
    duration window itself refuses -- see :func:`_overrun_timeline`.

    So what this reports is the number that decided it. Per surviving cut:
    depth in **seconds** as well as percent (a percentage cannot be compared
    across phrases -- 45% of a 9s phrase and 45% of a 1.5s one are 4.0s and
    0.7s of already-sung audio, and only the seconds are what a boundary move
    has to pay for), **which chunk starts there** (every chunk a viewer called
    defective for this mechanism is named by its own *start* boundary, never
    its end), and the **transfer budget** in both directions -- what moving it
    to each phrase edge would cost in grid steps against what the neighbouring
    chunk actually has. See :func:`_log_untenable_segments` for the lever that
    *does* move these numbers on real material, and it is the run's own
    ``max_chunk_seconds``, not a preference.
    """
    step = grid.step_frames
    step_seconds = grid.frames_to_seconds(step)
    boundary_count = max(0, len(covered) - 1)
    # Issue #100 changes what a surviving cut means and what it would cost to
    # move, so the message must not keep quoting a currency that is no longer
    # the one being spent: with the overrun paying for the grid, a move costs
    # no grid steps at all and a cut survives only because the duration
    # window itself had no room for it.
    budget_note = (
        "boundary_overrun is ON (issue #100), so moving this boundary costs no grid "
        "steps -- it survives because one of the two chunks would fall outside this "
        "run's own min/max duration window, which the step figures below stand in for"
        if boundary_overrun
        else "a move is only possible where the cost is within the budget"
    )
    cut_count = 0
    deepest: tuple[int, float, float, AlignedSegment] | None = None

    for idx, earlier in enumerate(covered[:-1]):
        later = covered[idx + 1]
        boundary = earlier.end
        landed = _segment_containing(boundary, segments)
        if landed is None:
            continue
        cut_count += 1
        pct = _percent_through(boundary, landed)
        depth = boundary - landed.start
        remaining = landed.end - boundary
        if deepest is None or depth > deepest[1]:
            deepest = (idx + 1, depth, pct, landed)

        # What a move would cost, and what is actually available to spend.
        # Moving the boundary EARLIER to the phrase start shrinks the
        # preceding chunk and grows this one; moving it LATER to the phrase
        # end does the reverse. A filler chunk's ceiling is its own, not a
        # lyric chunk's -- the same distinction _prefer_vocal_onset draws.
        earlier_frames = earlier.frame_count or 0
        later_frames = later.frame_count or 0
        ceiling_earlier = filler_max_frames if not earlier.members else max_frames
        ceiling_later = filler_max_frames if not later.members else max_frames
        back_cost = math.ceil((depth - _EPS) / step_seconds)
        back_budget = max(
            0,
            min(
                (earlier_frames - min_frames) // step,
                (ceiling_later - later_frames) // step,
            ),
        )
        forward_cost = math.ceil((remaining - _EPS) / step_seconds)
        forward_budget = max(
            0,
            min(
                (ceiling_earlier - earlier_frames) // step,
                (later_frames - min_frames) // step,
            ),
        )

        logger.warning(
            "Final chunk boundary at %.3fs (after instrumental-coverage retiling) lands "
            "%.1f%% through segment index=%d (%r, %.3f-%.3fs) -- %.3fs into a %.3fs phrase, "
            "and chunk %d starts here. Cutting a sung phrase mid-utterance; the two halves "
            "will render as independent shots with no error anywhere else. Moving it back "
            "to the phrase start costs %d grid step(s), budget %d; forward to the phrase "
            "end costs %d, budget %d -- %s. This position was not visible to pass 2's own "
            "segment-edge preference, which only ever sees pre-retiling boundaries "
            "(issue #70).",
            boundary,
            pct,
            landed.index,
            landed.text,
            landed.start,
            landed.end,
            depth,
            landed.duration,
            idx + 1,
            back_cost,
            back_budget,
            forward_cost,
            forward_budget,
            budget_note,
        )

    if deepest is None:
        logger.info(
            "Mid-phrase boundary cuts: none -- no boundary lands inside an aligned segment "
            "across %d boundary/boundaries (issue #70).",
            boundary_count,
        )
        return
    deep_idx, deep_depth, deep_pct, deep_segment = deepest
    logger.info(
        "Mid-phrase boundary cuts: %d of %d boundaries land inside an aligned segment; "
        "deepest is chunk %d at %.3fs (%.1f%%) into segment index=%d (issue #70).",
        cut_count,
        boundary_count,
        deep_idx,
        deep_depth,
        deep_pct,
        deep_segment.index,
    )


LEADING_VOCAL_OFFSET_WARN_SECONDS = 1.0
"""Above this many seconds between a voiced chunk's own start and the first
word it is actually prompted to sing, issue #79's leading-vocal-offset is
reported at WARNING rather than only folded into the INFO summary.

H3 starts the mouth at frame 0 of a chunk regardless of where in that chunk
the voice actually starts, so this many seconds of a chunk's start is always
out of phase with what is prompted.

The threshold's original justification ("none of the other 13 chunks sitting
between 0.5s and 1.0s were ever reported as audible") was **falsified** by a
later viewing and is retired; keep reading for the re-derivation that
replaces it. The constant itself is unchanged at 1.0s -- it is the reasoning
that needed fixing, not the number.

On a later, fully-rendered "Deathless" (v12), a viewer who described
*symptoms* rather than timestamps named two chunks with "no vocals at the
start but his mouth is moving, then perfectly in sync when the singing
starts": chunks 38 (+2.588s) and 41 (+1.607s) -- **ranks 2 and 3 of 41**
voiced chunks by leading offset. The two chunks they called perfect are 40
(+0.098s) and 43 (+0.013s), ranks 26 and 29, and they volunteered the
mechanism unprompted: "starts with singing -- perfect." This is a
**replication**: an earlier viewing of a *different* render had already named
the same two chunks (38 and 41) for the same symptom, by ear, before this
metric existed.

The counter-example is real and is **not** explained away: chunk 20 has the
largest offset in the song (+2.650s) and has never been reported, on any
render. A run-local scan once scored it 0.0% face presence, which looked like
the explanation and was wrong for a reason worth keeping: that CSV was a scan
of a week-older render wearing the current render's filename (issue #93; see
:mod:`music_video_maker.facescan`, which now stamps provenance). On the
render actually viewed the face is large, central and fully lit, and the
detector finds it in 11 of 12 sampled frames. The extreme upward head angle
may make lip motion hard to read, but that is a guess, not a measurement.
Any threshold derived from this data has to live with chunk 20 not fitting
it.

Post-refinement distribution on the 41 voiced chunks of "Deathless": 29
positive, 12 negative, 0 exactly zero; positive > 0.5s on 11, > 1.0s on 4
(chunks 20, 38, 41, 74), > 2.0s on 2 (20 at +2.650s, 38 at +2.588s); worst
negative -0.707s. See :func:`_log_leading_vocal_offset` for why only the
positive side is warned on.
"""


def _log_leading_vocal_offset(
    covered: Sequence[_Piece], segments: tuple[AlignedSegment, ...]
) -> None:
    """Issue #79: name every voiced chunk whose first prompted word starts
    measurably after the chunk's own start -- H3 starts the mouth at frame 0
    regardless, so a chunk like that is out of phase for however long the
    gap lasts.

    Runs on the *final* ``covered`` timeline (called from
    :func:`_cover_instrumentals` right beside
    :func:`_log_final_boundary_segment_cuts`), after
    :func:`_prefer_vocal_onset` has already moved whatever boundaries it
    could -- this reports whatever offset survives that pass, not the
    pre-refinement position. Uses :func:`_first_prompted_word_onset` against
    :func:`_prompted_members` (issue #92) rather than ``piece.members``
    directly, so on a chunk that merges two singers this reports the
    ATTRIBUTED singer's offset, not whichever member's word happens to start
    earliest -- the same narrowing :func:`slice_audio` uses for the chunk's
    own ``text``, so the two can never disagree about what is actually
    prompted.

    A chunk with no prompted words at all -- ``piece.members`` empty, which
    is also true of a chunk demoted below :data:`_MIN_VOICED_FRACTION` and
    prompted as instrumental -- has no leading vocal offset to report and is
    skipped.

    Issue #79 follow-up: both signs are counted and reported in the INFO
    summary. A negative offset -- the first prompted word began BEFORE the
    chunk did -- is the mirror defect: the mouth opens on a word whose audio
    has already partly gone by, so it runs *late* rather than early. The
    WARNING stays positive-side only: measured on "Deathless", 12 of 41
    voiced chunks are negative, the worst is -0.707s and none reaches -1.0s,
    so a negative-side warning at any threshold comparable to
    :data:`LEADING_VOCAL_OFFSET_WARN_SECONDS` would never fire on the only
    corpus there is to calibrate it against.
    """
    voiced_count = 0
    positive_count = 0
    negative_count = 0
    zero_count = 0
    over_count = 0
    worst_positive: tuple[int, float, _Piece] | None = None
    worst_negative: tuple[int, float, _Piece] | None = None

    for idx, piece in enumerate(covered):
        if not piece.members:
            continue
        prompted = _prompted_members(piece.members, piece.start, piece.end)
        onset = _first_prompted_word_onset(prompted, piece.start, piece.end)
        if onset is None:
            continue
        voiced_count += 1
        offset = onset - piece.start

        if offset > _EPS:
            positive_count += 1
            if worst_positive is None or offset > worst_positive[1]:
                worst_positive = (idx, offset, piece)
            if offset > LEADING_VOCAL_OFFSET_WARN_SECONDS:
                over_count += 1
                text = _text_within(prompted, piece.start, piece.end)
                logger.warning(
                    "Chunk %d (%.3f-%.3fs) is prompted to sing starting %.3fs into its own "
                    "span (first word onset %.3fs) -- H3 starts the mouth at frame 0 "
                    "regardless, so the chunk is out of phase for its first %.3fs: %r "
                    "(issue #79).",
                    idx,
                    piece.start,
                    piece.end,
                    offset,
                    onset,
                    offset,
                    text,
                )
        elif offset < -_EPS:
            negative_count += 1
            if worst_negative is None or offset < worst_negative[1]:
                worst_negative = (idx, offset, piece)
        else:
            zero_count += 1

    if not voiced_count:
        return
    if worst_positive is None and worst_negative is None:
        logger.info(
            "Leading vocal offset: %d voiced chunk(s), all start at their own first prompted "
            "word.",
            voiced_count,
        )
        return

    summary = (
        f"Leading vocal offset: {voiced_count} voiced chunk(s) ({positive_count} positive, "
        f"{negative_count} negative, {zero_count} exactly zero), {over_count} over the "
        f"{LEADING_VOCAL_OFFSET_WARN_SECONDS:.2f}s warning threshold"
    )
    if worst_positive is not None:
        idx, offset, piece = worst_positive
        summary += (
            f"; worst positive is chunk {idx} at +{offset:.3f}s "
            f"({piece.start:.3f}-{piece.end:.3f}s)"
        )
    if worst_negative is not None:
        idx, offset, piece = worst_negative
        summary += (
            f"; worst negative is chunk {idx} at {offset:.3f}s "
            f"({piece.start:.3f}-{piece.end:.3f}s)"
        )
    logger.info("%s.", summary)


def _is_instrumental_span(
    segments: tuple[AlignedSegment, ...], start: float, end: float
) -> bool:
    """True when a chunk covering ``[start, end)`` would be prompted as
    instrumental -- either nothing overlaps it at all, or the overlap is too
    thin to count. The latter is the same :data:`_MIN_VOICED_FRACTION` rule
    ``_cover_instrumentals`` applies when it builds the final ``covered``
    pieces (below), so this can never disagree with what a chunk in that
    position will actually be prompted with.

    Used by :func:`_prefer_vocal_onset` to choose which ceiling applies to a
    neighbouring chunk -- a filler chunk's is ``filler_max_frames``, a lyric
    chunk's is ``max_frames``.
    """
    members = _segments_overlapping(segments, start, end)
    if not members:
        return True
    voiced = _voiced_seconds_within(segments, start, end)
    return voiced < _MIN_VOICED_FRACTION * (end - start)


def _prefer_vocal_onset(
    boundaries: list[tuple[float, int, bool]],
    segments: tuple[AlignedSegment, ...],
    pinned_indices: frozenset[int],
    min_frames: int,
    max_frames: int,
    filler_max_frames: int,
    grid: FrameGrid,
) -> list[tuple[float, int, bool]]:
    """Prefer a voiced chunk's start at its own leading vocal onset (issue
    #79), by transferring whole grid steps from the chunk to its
    predecessor.

    H3 starts the mouth at frame 0 of a chunk regardless of where in that
    chunk the voice actually begins, so a chunk prompted with a lyric that
    starts seconds into its own span is out of phase for the whole chunk
    (measured on "Deathless": 4.07s on chunk 20, 3.30s on chunk 38). This is
    a *local, zero-cascade* boundary refinement, never a re-plan: moving
    boundary ``i`` later by ``k`` grid steps grows chunk ``i-1`` by exactly
    the frames it shrinks chunk ``i`` by, so the two chunks' combined
    duration -- and every OTHER boundary's start in the whole timeline -- is
    untouched (the combined duration of chunks ``i-1`` and ``i`` is
    invariant under the transfer, so chunk ``i-1``'s own start and chunk
    ``i``'s own end never move; only the shared boundary between them does).
    #70 was closed unfixed because biasing the filler-tiling maths itself
    risked exactly the kind of cascade this sidesteps entirely: this pass
    never touches the tiling, only redistributes frames across one
    already-decided boundary at a time.

    A boundary is left exactly where it was -- silently, since
    :func:`_log_leading_vocal_offset` already reports whatever is left --
    whenever any of these hold:

    * either neighbouring chunk's start was pinned by an honoured
      :class:`~music_video_maker.shot_plan.ShotLength` request (see
      ``pinned_indices``, from :func:`_apply_shot_lengths`) -- an author's
      explicit length must never be silently overridden;
    * chunk ``i`` is not itself going to be prompted with a lyric (nothing
      overlaps it, or the overlap is too thin -- :func:`_is_instrumental_span`
      to match), or its first prompted word already starts at its own start;
    * the preceding chunk has no headroom below its own ceiling
      (``max_frames`` for a lyric chunk, ``filler_max_frames`` for an
      instrumental one), or chunk ``i`` has no headroom above ``min_frames``;
    * every grid step count that would move the boundary closer to the
      onset, from the largest permitted down to one, still lands the new
      boundary inside an aligned segment (:func:`_segment_containing`) --
      trading a leading-offset defect for a mid-utterance cut is not a
      preference, it is the other bug.

    The move never overshoots the onset: the largest candidate ``k`` is
    bounded so ``k`` grid steps is never more than the offset itself, so the
    boundary only ever approaches the vocal onset and never passes it --
    clipping the start of the very phrase this exists to protect would just
    relocate issue #79's defect rather than fix it.

    Issue #79 follow-up: every DECLINED candidate whose offset exceeds
    :data:`LEADING_VOCAL_OFFSET_WARN_SECONDS` is logged at INFO, naming the
    blocking constraint, so an unfixed defect says why it is unfixed rather
    than going silent (threshold-gated so it does not spam a line for every
    one of ~29 chunks whose offset was never going to be reported anyway).

    Measured on "Deathless": all four surviving offsets over 1.0s are
    blocked by the *same* constraint -- the chunk itself sits at H3's
    124-frame trained floor and has no grid step to give back::

        chunk 20: +2.650s  frames prev=158 own=124 next=124   prev can grow 2, own can shrink 0
        chunk 38: +2.588s  frames prev=192 own=124 next=158   prev can grow 0, own can shrink 0
        chunk 41: +1.607s  frames prev=175 own=124 next=124   prev can grow 1, own can shrink 0
        chunk 74: +1.067s  frames prev=141 own=124 next=158   prev can grow 3, own can shrink 0

    An alternative was considered and rejected: a *slide* (grow the
    predecessor by ``k``, move the whole chunk later by ``k`` while keeping
    its own duration, shrink the successor by ``k``) would rescue exactly
    one of the four -- chunk 74, +1.067s -> +0.359s -- and none of the other
    three, because their successors are also at the floor. Not built: it
    moves two boundaries instead of one and changes the tail text of two
    chunks, for one chunk's gain on the only song with labels.
    """
    if len(boundaries) < 2:
        return boundaries

    frames = [count for _, count, _ in boundaries]
    continuations = [flag for _, _, flag in boundaries]
    starts = [start for start, _, _ in boundaries]
    step = grid.step_frames
    step_seconds = grid.frames_to_seconds(step)

    for i in range(1, len(frames)):
        start_i = starts[i]
        end_i = start_i + grid.frames_to_seconds(frames[i])
        members_i = _segments_overlapping(segments, start_i, end_i)
        if not members_i:
            continue
        voiced_i = _voiced_seconds_within(segments, start_i, end_i)
        if voiced_i < _MIN_VOICED_FRACTION * (end_i - start_i):
            continue  # will be prompted as instrumental -- no offset applies

        prompted_i = _prompted_members(members_i, start_i, end_i)  # issue #92
        onset = _first_prompted_word_onset(prompted_i, start_i, end_i)
        if onset is None:
            continue
        offset = onset - start_i
        if offset <= _EPS:
            continue

        loggable = offset > LEADING_VOCAL_OFFSET_WARN_SECONDS

        if (i - 1) in pinned_indices or i in pinned_indices:
            if loggable:
                logger.info(
                    "Leading vocal offset: chunk %d's %.3fs offset was not reduced -- "
                    "boundary %d is pinned by an honoured shot-length request (issue #79).",
                    i,
                    offset,
                    i,
                )
            continue

        start_prev = starts[i - 1]
        ceiling_prev = (
            filler_max_frames
            if _is_instrumental_span(segments, start_prev, start_i)
            else max_frames
        )

        max_k_offset = int(math.floor((offset + _EPS) / step_seconds))
        max_k_prev = (ceiling_prev - frames[i - 1]) // step
        max_k_i = (frames[i] - min_frames) // step
        max_k = min(max_k_offset, max_k_prev, max_k_i)
        if max_k < 1:
            if loggable:
                reasons = []
                if max_k_prev < 1:
                    reasons.append(
                        f"the preceding chunk is already at its {ceiling_prev}-frame ceiling"
                    )
                if max_k_i < 1:
                    reasons.append(
                        f"this chunk is already at the {min_frames}-frame trained floor and "
                        "cannot shrink"
                    )
                if not reasons:
                    reasons.append("the offset is smaller than one grid step")
                logger.info(
                    "Leading vocal offset: chunk %d's %.3fs offset was not reduced -- %s "
                    "(issue #79).",
                    i,
                    offset,
                    " and ".join(reasons),
                )
            continue

        accepted_k = None
        for k in range(max_k, 0, -1):
            candidate_start = start_i + k * step_seconds
            if _segment_containing(candidate_start, segments) is not None:
                continue
            accepted_k = k
            break
        if accepted_k is None:
            if loggable:
                logger.info(
                    "Leading vocal offset: chunk %d's %.3fs offset was not reduced -- every "
                    "candidate boundary step (1-%d grid step(s)) lands inside an aligned "
                    "segment (issue #79).",
                    i,
                    offset,
                    max_k,
                )
            continue

        k = accepted_k
        new_prev_frames = frames[i - 1] + k * step
        new_i_frames = frames[i] - k * step
        assert grid.is_valid(new_prev_frames) and grid.is_valid(new_i_frames)
        assert grid.trained_min_frames <= new_prev_frames <= grid.trained_max_frames
        assert grid.trained_min_frames <= new_i_frames <= grid.trained_max_frames

        new_start_i = start_i + k * step_seconds
        frames[i - 1] = new_prev_frames
        frames[i] = new_i_frames
        starts[i] = new_start_i

        logger.info(
            "Leading vocal offset: boundary %d moved later by %d frame(s) (%.3fs: %.3fs -> "
            "%.3fs), cutting chunk %d's leading vocal offset from %.3fs to %.3fs toward its "
            "onset at %.3fs (issue #79).",
            i,
            k * step,
            k * step_seconds,
            start_i,
            new_start_i,
            i,
            offset,
            onset - new_start_i,
            onset,
        )

    return list(zip(starts, frames, continuations, strict=True))


# --------------------------------------------------------------------------- #
# Opt-in: phrase-aware boundary planning (keep a sung phrase in one chunk)
# --------------------------------------------------------------------------- #

# Costs, in tiers, dearest first. The order is the specification; the spacing
# between tiers is wide enough that, on a real song (~10^2 boundaries, offsets
# of a few seconds), no pile of cheaper events outweighs one dearer one.
# Measured on "Deathless" (v16 alignment: segment 56 overridden to
# 486.55-495.0 s, window 124-192 frames), the outcome falls on one of two
# plateaus, and which one is a judgement the order below makes:
#
#   plan B (shipped): 5 fitting phrases cut, long phrases cut between words
#     wherever any tiling allows (segment 29 never does), 6 voiced chunks
#     opening >1 s before their first word (worst +2.30 s; default 7 / +2.65 s),
#     and segment 26 -- one of the four a viewer reported -- among the 5.
#   plan A: the same 5-cut minimum but all four reported phrases whole and #79
#     at 4 / +1.93 s -- bought by cutting segment 56 0.53 s *into* the held
#     word "deathless,". Reached by pricing a late second above a long
#     phrase's in-word cut (_LATE_ONSET_COST_PER_SECOND >= 1e10).
_PHRASE_CUT_COST = 1e12
"""A boundary inside a phrase the chunk window could have held whole."""
_LONG_IN_WORD_COST = 1e10
"""A phrase longer than the window must be cut; cutting it inside a *word*
instead of between two costs this. Above the late-onset tier on purpose: a
forced cut should fall between words wherever any tiling allows it, which is
the fallback this mode promises."""
_LATE_ONSET_COST_PER_SECOND = 1e8
"""Issue #79's defect itself, priced: every second a voiced chunk's first
prompted word starts past ``LEADING_VOCAL_OFFSET_WARN_SECONDS`` into the
chunk. Without this tier the plan trades #79 away for phrases -- on
"Deathless" chunks over 1 s went 7 -> 9 and the worst +2.65 -> +5.16 s, with
two of the reported phrases still cut. Swept 0 / 1e7 / 1e8 / 1e9 / 1e10 /
3e10 / 1e11 / 1e12: 1e7-1e9 is plan B, 1e10-1e11 plan A, and at 1e12 the plan
starts buying onset seconds with a sixth phrase cut. 1e8 is mid-plateau."""
_IN_WORD_COST = 1e7
"""Added to a cut inside a phrase that *fits* (already a phrase cut) when it
also lands inside a word. Below the late-onset tier on purpose: stable-ts
word timings abut, so "between two words" is a single instant the frame grid
must hit to within half a frame, and pricing it above a late second (1e9)
bought 5 -> 3 in-word cuts for #79 going back to 9 chunks over 1 s and a
worst offset of +5.16 s, with the same number of phrase cuts."""
_LONG_PHRASE_CUT_COST = 1e5
"""A boundary inside a phrase longer than the window. Some cut there is
forced; this makes each one beyond the forced minimum cost something."""
_LOST_LYRIC_COST = 1e4
"""A chunk holding sung words but voiced for under ``_MIN_VOICED_FRACTION``
of its length, so it is prompted as instrumental and its words are never
sung on screen."""
_FRAGMENT_COST_PER_SECOND = 100.0
"""Within a cut tier, prefer the cut that leaves the smaller fragment of the
phrase on one side."""
_ONSET_COST_PER_SECOND = 100.0
"""Issue #79's objective as a tie-break below its warning line: a voiced chunk
should open on its own first prompted word, because H3 starts the mouth at
frame 0 regardless."""
_NEAR_EDGE_COST = 10.0
"""A boundary within half a frame of a phrase edge. Strictly inside the
phrase -- the frame grid cannot land closer -- but by under 21 ms, which is
inside the aligner's own precision."""
_MOVED_BOUNDARY_COST = 5.0
"""A boundary the default tiling did not have. A tie-break only: among equally
good timelines, keep the one closest to what earlier renders and plans were
built against."""
_OVERSHOOT_COST_PER_FRAME = 1.0
"""Frames rendered past the track's own end, which ``-shortest`` discards."""


def _cut_detail(t: float, segment: AlignedSegment, half_frame: float) -> tuple[bool, str]:
    """Whether ``t`` falls inside one of ``segment``'s words, and a phrase
    naming where the cut falls, for the log. A boundary within half a frame of
    a word's edge is at that edge: the grid cannot land any closer. A segment
    with no word timings is one word, the same fallback ``_text_within`` uses.
    """
    words = segment.words
    if not words:
        return True, "inside a phrase with no word timings"
    for word in words:
        if word.start + half_frame < t < word.end - half_frame:
            return True, f"inside the word {word.word.strip()!r}"
    before = [w for w in words if w.end <= t + half_frame]
    after = [w for w in words if w.start >= t - half_frame]
    left = before[-1].word.strip() if before else ""
    right = after[0].word.strip() if after else ""
    return False, f"between {left!r} and {right!r}"


def _plan_phrase_boundaries(
    boundaries: list[tuple[float, int, bool]],
    segments: tuple[AlignedSegment, ...],
    track_duration: float,
    *,
    min_frames: int,
    max_frames: int,
    filler_max_frames: int,
    grid: FrameGrid,
) -> list[tuple[float, int, bool]]:
    """Re-plan the whole tiling so that no sung phrase is cut in two when the
    chunk window can hold it whole (opt-in, ``phrase_aware_slicing``).

    **Why the default lands where it does.** Passes 1-3 size each voiced
    chunk around its own segments and pass 4 lays them end to end with
    grid-quantized filler between, so a boundary's position is the running
    sum of every grid length before it. Nothing in that sum knows where the
    *next* phrase starts: on "Deathless" 15 of the 24 mid-phrase cuts are
    made by that accumulation alone (#70). And no local repair can fix it,
    which is what #70 measured from the other side: a boundary can only move
    by whole grid steps (every length is ``5 + 17k``, so every chunk moves a
    boundary by a multiple of 17 frames relative to its index), clearing a
    phrase head costs 4-6 of them, and the chunk that must pay sits at the
    124-frame floor.

    **What this does instead.** A shortest-path over integer frame positions
    from 0 to the track's end, where each edge is one chunk of a grid-valid,
    in-window length: ``min_frames``-``max_frames`` when the chunk overlaps
    a phrase, up to ``filler_max_frames`` when it does not. That is exactly
    the set of timelines pass 4 could ever have produced, searched whole, so
    the slack a move needs can come from an instrumental gap seconds away,
    and the phase between two boundaries can be changed by changing how many
    chunks lie between them (5 is coprime with 17). Contiguity -- video
    offset == audio offset for every chunk -- holds by construction, the same
    way it does for pass 4: every boundary is a frame position, and every
    chunk's length is the difference of two of them.

    Cost tiers, dearest first (see the ``_*_COST`` constants): a cut in a
    phrase that fits the window; a cut inside a word of a phrase too long for
    any chunk; each second of issue #79's leading vocal offset beyond its 1 s
    warning line; a cut inside a word of a phrase that fits; any cut in a
    phrase too long for any chunk; a chunk whose words are demoted to
    instrumental;
    then tie-breaks -- the smaller phrase fragment, the leading offset below
    the line, a boundary within half a frame of a phrase edge, a boundary the
    default did not have, frames past the track's end. So a phrase longer
    than the window is cut, but between words wherever any tiling allows it,
    and #79's objective is carried inside the plan rather than left to
    :func:`_prefer_vocal_onset` afterwards, which can only move one boundary
    by whole grid steps and so cannot repair what a global plan trades away.

    Measured on "Deathless" with the v16 alignment (57 segments, segment 56
    overridden to 486.55-495.0 s, window 124-192 frames): mid-phrase
    boundaries 25 -> 8, phrases that fit the window and are cut 21 -> 5,
    voiced chunks opening more than 1 s before their first word 7 -> 6 (worst
    +2.65 -> +2.30 s), 80 -> 75 chunks. Every one of the 5 is forced: a
    shortest path over every grid-valid tiling finds none with fewer, and the
    same search over *every integer* length (issue #100's edge set) still
    finds 2 -- two phrases closer together than the 5.167 s floor. Which 5
    is the tier order's call; the comment above the constants records the
    alternative.

    **What it cannot do.** The window is 5.167-8.0 s on "Deathless", and two
    long phrases separated by less than the floor cannot both be held whole
    by any tiling at all, grid or no grid. Every cut that survives is the
    minimum any tiling makes, and each is named at WARNING. Issue #100's
    overrun widens the edge set to every integer length and so may remove
    more; it composes by running afterwards on this result, unchanged.

    Returns the same ``(start, frames, is_continuation)`` shape pass 4 hands
    on, with ``is_continuation`` set where a chunk starts inside a phrase.
    """
    fps = grid.fps
    half_frame = 0.5 / fps
    longest = max(max_frames, filler_max_frames)
    lengths = [n for n in range(min_frames, longest + 1) if grid.is_valid(n)]
    fit_seconds = grid.frames_to_seconds(max_frames)

    default_edges = {0}
    position = 0
    for _start, frames, _cont in boundaries:
        position += frames
        default_edges.add(position)

    track_frames = track_duration * fps
    # The same tolerance pass 4's tail rule uses: under one frame of uncovered
    # track is not representable in any video.
    need = max(1, math.floor(track_frames - 1.0 + _EPS) + 1)
    horizon = need + longest

    starts = [s.start for s in segments]
    ends = [s.end for s in segments]

    def _overlapping(start: float, end: float) -> tuple[AlignedSegment, ...]:
        lo = bisect.bisect_right(ends, start + _EPS)
        hi = bisect.bisect_left(starts, end - _EPS)
        return _segments_overlapping(segments[lo:hi], start, end)

    def _containing(t: float) -> AlignedSegment | None:
        index = bisect.bisect_right(starts, t) - 1
        if index < 0:
            return None
        return _segment_containing(t, (segments[index],))

    def _boundary_cost(frame: int) -> float:
        cost = 0.0 if frame in default_edges else _MOVED_BOUNDARY_COST
        t = frame / fps
        segment = _containing(t)
        if segment is None:
            return cost
        fragment = min(t - segment.start, segment.end - t)
        if fragment <= half_frame:
            return cost + _NEAR_EDGE_COST
        in_word, _where = _cut_detail(t, segment, half_frame)
        fits = segment.duration <= fit_seconds + _EPS
        in_word_cost = _IN_WORD_COST if fits else _LONG_IN_WORD_COST
        return (
            cost
            + (_PHRASE_CUT_COST if fits else _LONG_PHRASE_CUT_COST)
            + (in_word_cost if in_word else 0.0)
            + _FRAGMENT_COST_PER_SECOND * fragment
        )

    def _chunk_cost(frame: int, frames: int) -> float | None:
        start, end = frame / fps, (frame + frames) / fps
        members = _overlapping(start, end)
        if not members:
            return 0.0 if frames <= filler_max_frames else None
        if frames > max_frames:
            return None
        voiced = _voiced_seconds_within(members, start, end)
        if voiced < _MIN_VOICED_FRACTION * (end - start):
            return _LOST_LYRIC_COST if _text_within(members, start, end) else 0.0
        # A trial chunk is not a chunk: issue #92's fallback warning is for
        # the timeline that renders, so it stays quiet here.
        prompted = _prompted_members(members, start, end, quiet=True)
        onset = _first_prompted_word_onset(prompted, start, end)
        if onset is None:
            return 0.0
        offset = max(0.0, onset - start)
        late = max(0.0, offset - LEADING_VOCAL_OFFSET_WARN_SECONDS)
        return _ONSET_COST_PER_SECOND * offset + _LATE_ONSET_COST_PER_SECOND * late

    boundary_cost = [0.0] + [_boundary_cost(f) for f in range(1, need)]
    best = [math.inf] * (horizon + 1)
    back = [-1] * (horizon + 1)
    best[0] = 0.0
    for frame in range(need):
        if best[frame] == math.inf:
            continue
        for frames in lengths:
            nxt = frame + frames
            chunk = _chunk_cost(frame, frames)
            if chunk is None:
                continue
            arrive = (
                boundary_cost[nxt]
                if nxt < need
                else _OVERSHOOT_COST_PER_FRAME * max(0.0, nxt - track_frames)
            )
            total = best[frame] + chunk + arrive
            if total < best[nxt]:
                best[nxt] = total
                back[nxt] = frame

    end_frame = min(range(need, horizon + 1), key=lambda f: best[f])
    edges = [end_frame]
    while edges[-1] > 0:
        edges.append(back[edges[-1]])
    edges.reverse()
    frames_list = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    planned_starts = _boundary_starts(frames_list, grid)
    planned = [
        (start, frames, _containing(start) is not None)
        for start, frames in zip(planned_starts, frames_list, strict=True)
    ]

    _log_phrase_plan(edges, segments, fit_seconds, half_frame, grid, len(boundaries))
    return planned


def _log_phrase_plan(
    edges: Sequence[int],
    segments: tuple[AlignedSegment, ...],
    fit_seconds: float,
    half_frame: float,
    grid: FrameGrid,
    default_count: int,
) -> None:
    """Say what the phrase-aware plan achieved, phrase by phrase: every cut it
    could not avoid at WARNING with where it falls, and one summary line."""
    fps = grid.fps
    interior = [frame / fps for frame in edges[1:-1]]
    fitting = [s for s in segments if s.duration <= fit_seconds + _EPS]
    whole = near = 0
    for segment in segments:
        inside = [t for t in interior if segment.start + _EPS < t < segment.end - _EPS]
        material = [
            t for t in inside if min(t - segment.start, segment.end - t) > half_frame
        ]
        fits = segment.duration <= fit_seconds + _EPS
        if not inside:
            whole += fits
            continue
        if not material:
            near += fits
            logger.info(
                "Phrase-aware slicing: segment index=%d (%r) is split within half a frame "
                "(%.0f ms) of its own edge -- the frame grid cannot land closer, and this is "
                "inside the aligner's own precision.",
                segment.index,
                segment.text,
                1000.0 * half_frame,
            )
            continue
        for t in material:
            _in_word, where = _cut_detail(t, segment, half_frame)
            if fits:
                logger.warning(
                    "Phrase-aware slicing: segment index=%d (%r, %.3f-%.3fs, %.3fs) could not "
                    "be held whole -- cut at %.3fs, %.3fs in, %s. It fits the %.3fs window on "
                    "its own, but no tiling of in-window chunks on H3's %d+%dk grid holds "
                    "every phrase around it whole; this is the fewest phrase cuts any tiling "
                    "makes. boundary_overrun (issue #100) is the lever that can move it.",
                    segment.index,
                    segment.text,
                    segment.start,
                    segment.end,
                    segment.duration,
                    t,
                    t - segment.start,
                    where,
                    fit_seconds,
                    grid.base_frames,
                    grid.step_frames,
                )
            else:
                logger.warning(
                    "Phrase-aware slicing: segment index=%d (%r, %.3f-%.3fs, %.3fs) is longer "
                    "than the %.3fs window, so no chunk can hold it; cut at %.3fs, %.3fs in, "
                    "%s (an inter-word gap wherever any tiling allows one).",
                    segment.index,
                    segment.text,
                    segment.start,
                    segment.end,
                    segment.duration,
                    fit_seconds,
                    t,
                    t - segment.start,
                    where,
                )
    logger.info(
        "Phrase-aware slicing: %d chunk(s) (the default tiling had %d); %d of %d phrases that "
        "fit the %.3fs window held whole%s; %d phrase(s) longer than the window.",
        len(edges) - 1,
        default_count,
        whole,
        len(fitting),
        fit_seconds,
        f" and {near} more split within half a frame of their own edge" if near else "",
        len(segments) - len(fitting),
    )


# --------------------------------------------------------------------------- #
# Pass 7 (opt-in): quantize the work, not the boundary (issue #100)
# --------------------------------------------------------------------------- #


def _overrun_timeline(
    boundaries: list[tuple[float, int, bool]],
    segments: tuple[AlignedSegment, ...],
    track_duration: float,
    *,
    min_frames: int,
    max_frames: int,
    filler_max_frames: int,
    grid: FrameGrid,
) -> list[tuple[int, int, int, bool]]:
    """Move boundaries onto segment edges and pay the grid in *overrun*
    instead of in boundary position (issue #100).

    Returns one ``(start_frame, kept_frames, render_frames, is_continuation)``
    per chunk, in integer frames from the start of the track.

    **The idea, which came from another pipeline built on this one.** Every
    boundary in this file is a compromise between where the content wants a
    cut and where the ``5 + 17k`` grid allows a *length*. #70 closed on the
    finding that those are the same choice: quantizing a chunk moves its
    boundary, so the depth-threshold preference the issue asked for cost 4-6
    grid steps of 0.708 s against neighbours sitting at the 124-frame floor
    with nothing to give, and fired 0 times at every threshold tried.

    They are only the same choice because this pipeline renders a chunk at
    exactly its own length. Ask H3 for the next valid length *past* the
    boundary and discard the tail, and the two decisions come apart: the
    boundary becomes a content decision and the length stays a VRAM one. The
    4-6 steps a move used to cost become **zero**, because the neighbour is
    no longer being asked to pay for the grid -- the overrun is.

    So: for every interior boundary that lands inside a sung phrase (the
    cuts :func:`_log_final_boundary_segment_cuts` has only been able to
    *report*), try the nearer edge of that phrase first and then the other,
    accepting whichever keeps both neighbouring chunks inside the run's own
    duration window. Then shorten the final boundary to the track's own end,
    which the grid-tiled timeline overshoots by up to one trained-floor
    chunk -- 44 frames on "Deathless", frames ``-shortest`` throws away today
    with nothing recording it (CLAUDE.md's ``-shortest`` invariant, #22).

    **Why this is arithmetic on integers and not on seconds.** Boundaries are
    held as absolute frame positions, so a chunk's kept length is the
    difference of its two neighbours' positions and the whole timeline
    telescopes: the frames before boundary *i* sum to exactly ``edges[i]``,
    whatever moved and by how much. Video offset == audio offset for every
    chunk by construction, which is this file's one non-negotiable invariant
    and the thing #70 declined to put at risk. A seconds-based version of
    this function would have each chunk round its own duration and let the
    residue accumulate -- see :meth:`FrameGrid.frames_between`.

    The final boundary is only ever moved **earlier**, never later. Growing
    the last chunk to reach the track end is the F26 defect (the conditioning
    audio outranks the prompt, so a stretched sung chunk mouths over the
    outro); ``_cover_instrumentals``' own tail rule already handles an
    undershoot before this runs, and it does it by appending instrumental
    filler rather than by stretching anything.

    A boundary this cannot move is left exactly where it was and reported by
    the existing logger, unchanged -- the window is still the run's
    ``min_chunk_seconds``/``max_chunk_seconds``, because a chunk that keeps
    fewer frames than H3's trained floor would be mostly overrun, and one
    that keeps more than the ceiling cannot be rendered at all.
    """
    fps = grid.fps
    edges: list[int] = [0]
    for _start, frames, _cont in boundaries:
        edges.append(edges[-1] + frames)

    # A chunk's ceiling depends on whether it is filler, and that is decided
    # by span overlap -- read off the pre-move spans, which is what
    # _cover_instrumentals' own `members` derivation will agree with except
    # in the rare case where a move changes a chunk's overlap. Getting that
    # case wrong costs a rejected move, never a bad one.
    def _ceiling(index: int) -> int:
        start = edges[index] / fps
        end = edges[index + 1] / fps
        return max_frames if _segments_overlapping(segments, start, end) else filler_max_frames

    moved = 0
    for i in range(1, len(edges) - 1):
        landed = _segment_containing(edges[i] / fps, segments)
        if landed is None:
            continue
        ceiling_before = _ceiling(i - 1)
        ceiling_after = _ceiling(i)
        # The phrase's START first, and its end only as a fallback -- never
        # whichever is nearer. Both clear the phrase, but they are not equally
        # good: moving back to the start opens the *later* chunk on the
        # phrase's first word, which is issue #79's own objective (H3 starts
        # the mouth at frame 0 regardless of where the voice does), while
        # moving forward to the end leaves that chunk beginning in a vocal
        # gap -- the "mouth moving with no vocal, then perfect once the
        # singing starts" defect a viewer named twice. Issue #100's own
        # pipeline makes the same choice for the same reason: "a shot can
        # open on its first word instead of in a vocal gap". The reason this
        # is now affordable at all is that moving back is the expensive
        # direction on the grid -- 4-6 steps against a neighbour at the
        # 124-frame floor -- and the overrun pays it instead of the neighbour.
        for edge_seconds in (landed.start, landed.end):
            # Round OUTWARD, never to the nearest frame. A phrase edge is
            # almost never on a frame boundary, and rounding 72.920s to the
            # nearest frame gives 72.9166s -- which is still 0.003s inside the
            # phrase. That is #70's own first mistake arriving by a new road:
            # its first segment-edge snap "logged six snaps that every one of
            # them silently landed back inside the segment it claimed to
            # avoid". Measured here before the fix: Deathless's 24 mid-phrase
            # cuts went to 23 while the deepest went from 86.0% to 99.9%
            # through its phrase -- a pass reporting success at clearing a
            # phrase it had merely moved to the far end of. floor for a start
            # edge and ceil for an end edge put the boundary at or outside the
            # phrase by construction.
            target = (
                math.floor(edge_seconds * fps)
                if edge_seconds == landed.start
                else math.ceil(edge_seconds * fps)
            )
            kept_before = target - edges[i - 1]
            kept_after = edges[i + 1] - target
            if not min_frames <= kept_before <= ceiling_before:
                continue
            if not min_frames <= kept_after <= ceiling_after:
                continue
            logger.info(
                "Issue #100: moving the boundary at %.3fs to %.3fs (segment index=%d's "
                "%s edge), which cost %d grid step(s) before this pass existed and now "
                "costs none -- chunks %d and %d keep %d and %d frames and are rendered "
                "at %d and %d.",
                edges[i] / fps,
                target / fps,
                landed.index,
                "start" if edge_seconds == landed.start else "end",
                math.ceil(abs(target - edges[i]) / grid.step_frames),
                i - 1,
                i,
                kept_before,
                kept_after,
                grid.cover_frames(kept_before),
                grid.cover_frames(kept_after),
            )
            edges[i] = target
            moved += 1
            break

    # The tail. _cover_instrumentals tiles the grid past the track's own end
    # (up to one trained-floor chunk), and a free boundary can simply stop
    # where the song does. At least one kept frame, because a chunk with none
    # is not a chunk.
    track_frames = round(track_duration * fps)
    trimmed_tail = 0
    if len(edges) >= 2 and edges[-1] > track_frames >= edges[-2] + 1:
        trimmed_tail = edges[-1] - track_frames
        edges[-1] = track_frames

    planned: list[tuple[int, int, int, bool]] = []
    for idx, (_start, _frames, is_continuation) in enumerate(boundaries):
        kept = edges[idx + 1] - edges[idx]
        planned.append((edges[idx], kept, grid.cover_frames(kept), is_continuation))

    kept_total = sum(kept for _s, kept, _r, _c in planned)
    overrun_total = sum(render - kept for _s, kept, render, _c in planned)
    logger.info(
        "Issue #100 boundary overrun: %d of %d boundar(ies) moved onto a phrase edge, "
        "%d tail frame(s) given back to the track's own end, and %d of %d chunk(s) now "
        "render past their own end. Cost: %d frames rendered for %d kept, %+.2f%% extra "
        "frames (longest overrun %d frames / %.3fs). The overrun is discarded in Stage 5 "
        "and recorded per chunk, never implied.",
        moved,
        max(0, len(edges) - 2),
        trimmed_tail,
        sum(1 for _s, kept, render, _c in planned if render > kept),
        len(planned),
        kept_total + overrun_total,
        kept_total,
        100.0 * overrun_total / kept_total if kept_total else 0.0,
        max((render - kept for _s, kept, render, _c in planned), default=0),
        grid.frames_to_seconds(max((render - kept for _s, kept, render, _c in planned), default=0)),
    )
    return planned


def _cover_instrumentals(
    pieces: list[_Piece],
    segments: tuple[AlignedSegment, ...],
    track_duration: float,
    eff_min: float,
    eff_max: float,
    grid: FrameGrid,
    shot_lengths: Sequence[ShotLength] = (),
    instrumental_shot_seconds: float | None = None,
    boundary_overrun: bool = False,
    phrase_aware_slicing: bool = False,
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

    This re-anchoring is also, measured on real material, the *dominant*
    source of issue #70's mid-utterance boundary cuts -- more so than pass
    2's own max-duration split, which is the only place this file currently
    *avoids* one (:func:`_snap_to_segment_edge`). See
    :func:`_log_final_boundary_segment_cuts` for why this pass only reports
    that, rather than also avoiding it.

    ``boundary_overrun`` (issue #100, opt-in and off by default) hands the
    finished tiling to :func:`_overrun_timeline`, which *does* avoid those
    cuts -- by decoupling where a chunk ends from what length it is rendered
    at, so a move no longer has to be paid for out of a neighbour's duration.
    Contiguity is preserved by the same construction either way; see that
    function's docstring.
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
    last_is_filler = False

    def _emit_filler(gap: float) -> None:
        nonlocal running, last_is_filler
        for frame_count in _plan_filler_frames(gap, min_frames, filler_max_frames, grid):
            boundaries.append((running, frame_count, False))
            running += grid.frames_to_seconds(frame_count)
            last_is_filler = True

    for piece in pieces:
        if piece.start - running > _EPS:
            _emit_filler(piece.start - running)
        assert piece.frame_count is not None  # pass 3 guarantees this
        boundaries.append((running, piece.frame_count, piece.is_split_continuation))
        running += grid.frames_to_seconds(piece.frame_count)
        last_is_filler = False

    if track_duration - running > _EPS:
        _emit_filler(track_duration - running)

    # Whatever tail is still uncovered here is shorter than one chunk: either a
    # whole outro under the trained floor (which emits nothing above) or the
    # grid-rounding residue of the filler that was emitted. Unlike a hole
    # between chunks there is no later chunk to re-anchor over it, so the
    # timeline ended short of the track and the mux's -shortest cut the song's
    # own outro out of the video.
    #
    # If the last chunk is instrumental filler, grow it by grid steps: stretching
    # instrumental time costs nothing and overshoots by under one step. If it is
    # a sung chunk, never grow it -- the conditioning audio outranks the prompt
    # (F26), so it would mouth over the outro -- and append a floor-length
    # instrumental tile run past the end instead; that also covers a filler
    # already at its maximum. The overshoot is frames -shortest discards, named
    # before any GPU time by the Stage 2 drift report. Less than one frame of
    # tail is not representable in any video, so it is not worth a render.
    tail_frames = grid.seconds_to_frames(track_duration - running)
    if tail_frames >= 1.0:
        last_start, last_frames, last_continuation = boundaries[-1] if boundaries else (0, 0, False)
        grown = grid.quantize_up(last_frames + int(math.ceil(tail_frames)))
        if last_is_filler and grown <= filler_max_frames:
            boundaries[-1] = (last_start, grown, last_continuation)
            running = last_start + grid.frames_to_seconds(grown)
        else:
            boundaries.append((running, min_frames, False))
            running += grid.frames_to_seconds(min_frames)

    if phrase_aware_slicing:
        # Replaces where pass 4 put the boundaries, never what a chunk is:
        # everything after this line (#79, #100, the members/prompting
        # derivation) runs on its result exactly as it runs on pass 4's.
        boundaries = _plan_phrase_boundaries(
            boundaries,
            segments,
            track_duration,
            min_frames=min_frames,
            max_frames=max_frames,
            filler_max_frames=filler_max_frames,
            grid=grid,
        )

    if shot_lengths:
        boundaries, pinned_indices = _apply_shot_lengths(
            boundaries, shot_lengths, min_frames, max_frames, grid
        )
    else:
        pinned_indices = frozenset()

    boundaries = _prefer_vocal_onset(
        boundaries, segments, pinned_indices, min_frames, max_frames, filler_max_frames, grid
    )

    # (start_seconds, kept_frames, render_frames_or_None, is_continuation).
    # Without issue #100 the third entry is None everywhere, which is what
    # keeps every pre-#100 timeline byte-identical: `kept` is then the chunk's
    # own grid-valid length and nothing is rendered past its end.
    planned: list[tuple[float, int, int | None, bool]]
    if boundary_overrun:
        planned = [
            (start_frame / grid.fps, kept, render if render != kept else None, is_continuation)
            for start_frame, kept, render, is_continuation in _overrun_timeline(
                boundaries,
                segments,
                track_duration,
                min_frames=min_frames,
                max_frames=max_frames,
                filler_max_frames=filler_max_frames,
                grid=grid,
            )
        ]
    else:
        planned = [(start, kept, None, cont) for start, kept, cont in boundaries]

    covered: list[_Piece] = []
    for start, frame_count, render_frames, is_continuation in planned:
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
                render_frames=render_frames,
            )
        )

    _log_final_boundary_segment_cuts(
        covered,
        segments,
        grid=grid,
        min_frames=min_frames,
        max_frames=max_frames,
        filler_max_frames=filler_max_frames,
        boundary_overrun=boundary_overrun,
    )
    _log_leading_vocal_offset(covered, segments)

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


def _log_suspect_segment_chunks(
    chunks_on_suspect_prompt: Sequence[tuple[int, list[int]]],
    merges_on_suspect_drop: Sequence[tuple[int, list[int]]],
    suspects: frozenset[int],
) -> None:
    """Map the alignment-quality report's doubted segments onto chunk ids
    (issues #96, #92).

    Nothing else in the pipeline does this: the report names *segments* and a
    render names *chunks*, and the translation between them is exactly the
    slicing pass. Two aggregated WARNING lines rather than one per chunk --
    #96 can fire on many segments at once on a badly-aligned track, and a
    per-chunk line would bury the merge case, which is the one a human has to
    adjudicate.

    Reports only. Slicing does not treat a doubted segment differently, and
    deliberately: whether a dropped phrase is a phantom or a real second
    singer, dropping it from the attributed singer's prompt is the same right
    answer (#92), and refusing or re-attributing on a WARNING-grade acoustic
    finding would let a false positive move a timeline.
    """
    if not suspects:
        return
    if chunks_on_suspect_prompt:
        logger.warning(
            "%d voiced chunk(s) are prompted with words from aligned segment(s) the "
            "alignment-quality report doubts there is a voice under: %s. The timeline, the "
            "attribution and the text are unchanged -- this is the segment-to-chunk mapping "
            "nothing else produces, so a #96 finding can be checked against what actually "
            "renders. Listen to each span before treating its lyric as sung (issue #96).",
            len(chunks_on_suspect_prompt),
            ", ".join(
                f"chunk {chunk_id} <- segment(s) {indices}"
                for chunk_id, indices in chunks_on_suspect_prompt
            ),
        )
    if merges_on_suspect_drop:
        logger.warning(
            "%d cross-character merge(s) drop words belonging to a DOUBTED segment: %s. On "
            "\"Deathless\" two of the three chunks issue #92's fix acted on were this case -- "
            "the \"other singer's words\" were phantoms nobody sings (#96), so the chunk was "
            "never a merge and the leading-offset cost recorded against it was measured "
            "against words that are not there. Check these before quoting a cost for them.",
            len(merges_on_suspect_drop),
            ", ".join(
                f"chunk {chunk_id} <- dropped segment(s) {indices}"
                for chunk_id, indices in merges_on_suspect_drop
            ),
        )


def slice_audio(
    audio_path: Path,
    alignment: AlignmentResult,
    hardware: HardwareProfile,
    chunks_dir: Path,
    cover_instrumentals: bool = False,
    *,
    boundary_overrun: bool = False,
    phrase_aware_slicing: bool = False,
    shot_lengths: Sequence[ShotLength] = (),
    instrumental_shot_seconds: float | None = None,
    instrumental_audio_gain_db: float | None = None,
    suspect_segment_indices: Collection[int] = (),
    measured_ceiling: MeasuredCeiling = CALIBRATED_CEILING,
    timeline: str | None = None,
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

    ``boundary_overrun`` (issue #100) stops the frame grid deciding where
    chunks are cut. Every chunk whose boundary the content wants elsewhere
    keeps the number of frames its own span contains -- which is no longer
    grid-valid -- and is **rendered** at the next valid length past it
    (``AudioChunk.render_frames``, also the length its stem is cut to), with
    the overrun discarded in Stage 5. See :func:`_overrun_timeline`.

    **Opt-in, and off by default, deliberately.** It changes where every
    boundary in the song falls, so it invalidates every chunk already on disk
    and every cached chunk a ``--resume`` would reuse, and it has never been
    rendered on this project's own material -- the cost it trades for is real
    (a few per cent of extra frames, and up to one grid step of the *next*
    phrase inside each chunk's conditioning audio) and the benefit is so far
    an argument, not a measurement. Nothing enters the default path here
    without a measurement on real pixels. It needs ``cover_instrumentals``
    for the same reason ``shot_lengths`` does, and because the kept-frame
    arithmetic is measured from the start of the track: there is no coherent
    way to do it over a timeline with holes in it.

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

    ``suspect_segment_indices`` (issues #96, #92) are aligned segments the
    alignment-quality report doubts the existence of a voice under -- build it
    with ``alignment_quality.suspect_segment_indices(report)``. Slicing has
    always read ``alignment.segments`` as ground truth; on "Deathless" two of
    the three chunks the #92 cross-character fix acted on were not merges at
    all, because the "other singer's words" it dropped were phantoms nobody
    sings. This parameter changes **nothing** about the timeline, the
    attribution or the text -- dropping a phantom's words from a prompt is
    right whether or not it is a phantom. What it changes is that the run log
    can say which chunks rest on a doubted segment, which is the re-check
    issue #92's own follow-up asked for. Empty by default, so every existing
    caller behaves byte-identically.
    ``measured_ceiling`` (issue #98) is the frame count the
    "nothing longer has rendered here" warning is allowed to name, plus where
    that number came from. It defaults to the calibrated constant; a caller
    with a ``run_state.json`` in reach should pass
    ``envelope.measured_ceiling(run_state_file)`` instead, which reads the
    longest chunk a previous run *actually rendered* and so makes the number
    evidence rather than a memory. It changes nothing but a log line.
    ``timeline`` (issue #66) stamps every emitted chunk with the timeline it
    belongs to -- ``None``, the default, is the song, so a run with no
    ``[[segment]]`` table produces chunks identical to the ones it always
    produced. It changes nothing about the slicing itself: a prologue is the
    same five stages run over a second (audio, text) pair, and its chunk
    ``start``/``end`` are seconds from the start of *its own* audio, never
    offset into the finished video. See
    :attr:`~music_video_maker.contracts.AudioChunk.timeline` for why the
    offset lives on the timeline rather than in these numbers.
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

    if boundary_overrun and not cover_instrumentals:
        logger.warning(
            "Ignoring boundary_overrun (issue #100): it moves boundaries across a "
            "contiguous timeline and measures every chunk's kept frames from the start "
            "of the track, and instrumental_coverage is off, so this run only renders "
            "the voiced spans and has no such timeline. Turn instrumental_coverage on "
            "(it is the default, and Stage 5 silently desyncs without it)."
        )
        boundary_overrun = False

    if phrase_aware_slicing and not cover_instrumentals:
        logger.warning(
            "Ignoring phrase_aware_slicing: it plans boundaries across a contiguous "
            "timeline, and instrumental_coverage is off, so this run only renders the "
            "voiced spans and has no such timeline. Turn instrumental_coverage on "
            "(it is the default, and Stage 5 silently desyncs without it)."
        )
        phrase_aware_slicing = False

    if phrase_aware_slicing and shot_lengths:
        # Both decide where boundaries fall, and a length request is anchored
        # to a boundary of the timeline its author saw -- which this mode
        # moves. Honouring one after the other would match anchors against
        # the wrong timeline, silently; refusing is the honest answer until
        # someone needs both and the anchor question is designed.
        logger.error(
            "phrase_aware_slicing cannot be combined with %d shot-plan length_seconds "
            "request(s) (issue #27): both decide where chunk boundaries fall, and each "
            "request is anchored to a boundary this mode moves.",
            len(shot_lengths),
        )
        raise ValueError(
            "phrase_aware_slicing cannot be combined with shot-plan length_seconds "
            f"requests ({len(shot_lengths)} given): both decide where chunk boundaries "
            "fall, and each request is anchored to a boundary this mode moves. Remove "
            "the plan's length_seconds or turn phrase_aware_slicing off"
        )

    eff_min, eff_max, grid = _effective_bounds(hardware)
    _log_effective_frame_window(eff_min, eff_max, grid)

    segments = tuple(sorted(alignment.segments, key=lambda s: s.start))

    # Issue #70: a phrase longer than this run's ceiling is cut mid-utterance
    # by every render there is, so it is worth naming before any of them --
    # and separately from the boundaries below, which are placement choices
    # rather than arithmetic. Runs whether or not instrumental coverage is on.
    _log_untenable_segments(segments, eff_max, grid)

    groups = _merge_for_minimum(segments, alignment.track_duration, eff_min)
    raw_pieces = _split_for_maximum(groups, eff_min, eff_max, grid)

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
            boundary_overrun=boundary_overrun,
            phrase_aware_slicing=phrase_aware_slicing,
        )

    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    master = AudioSegment.from_file(str(audio_path))

    chunks: list[AudioChunk] = []
    # Issues #96/#92. Kept as two separate registers because they answer two
    # different questions: "this chunk sings a lyric the report doubts" and
    # "this chunk's cross-character merge may not be a merge at all".
    suspects = frozenset(suspect_segment_indices)
    chunks_on_suspect_prompt: list[tuple[int, list[int]]] = []
    merges_on_suspect_drop: list[tuple[int, list[int]]] = []
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

        # The length H3 is asked for is the one that has to land on the grid.
        # Without issue #100 that is the chunk's own frame_count and the two
        # checks below are exactly what they always were.
        render_frames = (
            piece.render_frames if piece.render_frames is not None else piece.frame_count
        )
        if piece.frame_count is None or render_frames is None or not grid.is_valid(render_frames):
            logger.error(
                "Chunk %d (segments=%s) has no valid rendered length (frame_count=%r, "
                "render_frames=%r) at emission time.",
                idx,
                tuple(m.index for m in piece.members),
                piece.frame_count,
                piece.render_frames,
            )
            raise ChunkFrameMismatchError(
                f"chunk {idx}: rendered length {render_frames!r} is not a valid H3 grid point "
                f"(frame_count={piece.frame_count!r}, "
                f"source_segment_indices={tuple(m.index for m in piece.members)!r})"
            )

        if piece.render_frames is None:
            piece_duration = piece.end - piece.start
            expected_duration = grid.frames_to_seconds(piece.frame_count)
            if abs(piece_duration - expected_duration) > 1e-3:
                logger.error(
                    "Chunk %d (segments=%s): frame_count=%d implies %.6fs but the sliced "
                    "duration is %.6fs -- these must match exactly or Stage 4a's rendered "
                    "video will drift from its own audio stem.",
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
        else:
            # Issue #100's own version of the same guard, and it is stricter
            # rather than looser: the frames this chunk KEEPS must be exactly
            # the frames its span contains, measured from the start of the
            # track so the timeline telescopes (see FrameGrid.frames_between).
            # A chunk whose kept count disagrees with its own span by one
            # frame is a chunk that desyncs everything after it.
            expected_kept = grid.frames_between(piece.start, piece.end)
            if piece.frame_count != expected_kept or render_frames < piece.frame_count:
                logger.error(
                    "Chunk %d (segments=%s): keeps frame_count=%d of render_frames=%d over "
                    "%.6f-%.6fs, which contains %d frame(s) -- the kept count must equal the "
                    "span and must not exceed what was rendered, or Stage 5's trim desyncs "
                    "every chunk after this one (issue #100).",
                    idx,
                    tuple(m.index for m in piece.members),
                    piece.frame_count,
                    render_frames,
                    piece.start,
                    piece.end,
                    expected_kept,
                )
                raise ChunkFrameMismatchError(
                    f"chunk {idx}: keeps {piece.frame_count!r} frame(s) of {render_frames!r} "
                    f"rendered over a span containing {expected_kept!r} "
                    f"(source_segment_indices={tuple(m.index for m in piece.members)!r})"
                )

        start_ms = round(piece.start * 1000)
        # The stem has to cover what H3 renders, not what the chunk keeps:
        # the overrun frames are generated from this audio and then thrown
        # away, which is issue #100's stated cost -- up to one grid step of
        # the next phrase inside this chunk's conditioning.
        stem_end = (
            piece.end
            if piece.render_frames is None
            else piece.start + grid.frames_to_seconds(render_frames)
        )
        end_ms = round(stem_end * 1000)
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
                "silence so its duration still matches the rendered length of %d frame(s) "
                "exactly.",
                idx,
                shortfall_ms,
                render_frames,
            )
            sliced = sliced + AudioSegment.silent(
                duration=shortfall_ms, frame_rate=master.frame_rate
            ).set_channels(master.channels).set_sample_width(master.sample_width)

        # Issue #92: narrow to whichever character _dominant_character_member
        # picks BEFORE deriving text, so a chunk merging two singers is
        # prompted with only the one it is attributed to -- never both (see
        # _prompted_members's docstring for the empty-narrowing fallback).
        prompted = _prompted_members(piece.members, piece.start, piece.end)
        text = _text_within(prompted, piece.start, piece.end)
        is_instrumental = not piece.members or not text

        # Issues #96/#92: record, never act. See slice_audio's docstring.
        if suspects and not is_instrumental:
            prompted_suspects = sorted(m.index for m in prompted if m.index in suspects)
            if prompted_suspects:
                chunks_on_suspect_prompt.append((idx, prompted_suspects))

        # Issue #73 follow-up, measured on the v8 "Deathless" render (F26).
        # H3 is an audio-driven lip-sync model and this stem is its
        # conditioning signal, so an instrumental chunk handed full-level
        # music gets a mouth animated to the music. Measured: chunk 30
        # (instrumental, a viewer reported it mouthing words with no audio)
        # at -15.2 dB mean, against -16.8 dB for a *sung* chunk -- the same
        # level. Rewording the prompt's instrumental clause reduced the
        # effect and could not remove it, because the prompt is arguing with
        # the conditioning signal and the conditioning signal wins.
        #
        # A VOICED stem is never touched: it is the lip-sync ground truth.
        # Attenuation only, and level only -- duration and format are
        # preserved by construction, because a stem whose duration moved
        # desyncs its own chunk (issue #20).
        if is_instrumental and instrumental_audio_gain_db is not None:
            logger.info(
                "Chunk %d is instrumental; attenuating its conditioning stem by %.1f dB "
                "so H3 is not handed music to lip-sync (F26).",
                idx,
                instrumental_audio_gain_db,
            )
            sliced = sliced + instrumental_audio_gain_db

        out_path = chunks_dir / f"chunk_{idx:03d}.wav"
        sliced.export(str(out_path), format="wav")

        if is_instrumental:
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
                    render_frames=piece.render_frames,
                    is_instrumental=True,
                    timeline=timeline,
                )
            )
            prev_end = piece.end
            continue

        # Issue #40's dominant-voice rule, now measured by in-chunk voiced
        # overlap rather than whole-segment duration (issue #92) -- see
        # _dominant_character_member's docstring for why the two agree on
        # every real "Deathless" case but the old one over-reports how much
        # of the *chunk* a singer actually holds.
        distinct = {m.character for m in piece.members}
        dominant = _dominant_character_member(piece.members, piece.start, piece.end)
        if len(distinct) > 1:
            # Issue #92: this used to name only the winner and the
            # WHOLE-SEGMENT voiced-duration numbers that (issue #40) decided
            # it -- true about the attribution, silent about the actual
            # defect, which is that the chunk is then prompted with the
            # FULL span's text, including the other singer's words. This
            # version names, per character, how much of THIS CHUNK (not the
            # whole segment) it is audible for; who it is attributed to; the
            # text that IS prompted (already narrowed to `prompted` above);
            # the words that are audible in the stem but deliberately
            # dropped from the prompt; and the resulting leading vocal
            # offset for the attributed singer -- because H3 starts the
            # mouth at frame 0 regardless, so the dropped singer's seconds
            # at the head of the chunk are exactly what a viewer hears as
            # the attributed singer running late (issue #92, #79). One
            # WARNING per chunk, not per character.
            totals = _voiced_seconds_by_character(piece.members, piece.start, piece.end)
            total_voiced_in_chunk = sum(totals.values()) or _EPS
            breakdown = ", ".join(
                f"{character!r} {seconds:.3f}s ({100.0 * seconds / total_voiced_in_chunk:.0f}%)"
                for character, seconds in sorted(totals.items(), key=lambda kv: -kv[1])
            )
            dropped_members = tuple(
                m for m in piece.members if m.character != dominant.character
            )
            dropped_text = _text_within(dropped_members, piece.start, piece.end)
            dropped_suspects = sorted(m.index for m in dropped_members if m.index in suspects)
            if dropped_suspects:
                merges_on_suspect_drop.append((idx, dropped_suspects))
            dominant_onset = _first_prompted_word_onset(prompted, piece.start, piece.end)
            dominant_offset_text = (
                f" {dominant_onset - piece.start:.3f}s" if dominant_onset is not None else ""
            )
            logger.warning(
                "Chunk %d (%.3f-%.3fs) merges segments with differing characters %s -- "
                "in-chunk voiced time: %s. Attributing to %r, which contributes %.3fs of "
                "the chunk's %.3fs total in-chunk voiced duration. Prompted with %r; %r "
                "(aligned segment(s) %s) is "
                "audible in this chunk's stem but deliberately dropped from the prompt "
                "because it belongs to the other singer, not %r -- naming both risks #82's "
                "morphing defect. H3 starts the mouth at frame 0 regardless of where in the "
                "chunk the voice actually starts, so the dropped singer's time at the head "
                "of the chunk is what a viewer hears as %r running%s late (issue #92, #79).",
                idx,
                piece.start,
                piece.end,
                sorted(c for c in distinct if c is not None),
                breakdown,
                dominant.character,
                totals.get(dominant.character, 0.0),
                total_voiced_in_chunk,
                text,
                dropped_text,
                sorted(m.index for m in dropped_members),
                dominant.character,
                dominant.character,
                dominant_offset_text,
            )
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
                # Deliberately NOT narrowed to `prompted` (issue #92): this
                # records the provenance of the AUDIO, which genuinely
                # contains every singer merged into the chunk, not just the
                # one `text` above is narrowed to.
                source_segment_indices=tuple(m.index for m in piece.members),
                is_split_continuation=piece.is_split_continuation,
                frame_count=piece.frame_count,
                render_frames=piece.render_frames,
                timeline=timeline,
            )
        )
        prev_end = piece.end

    _log_suspect_segment_chunks(chunks_on_suspect_prompt, merges_on_suspect_drop, suspects)

    logger.info(
        "Sliced %d chunk(s) from %d aligned segment(s) into %s (frame-grid quantized, "
        "trained range %d-%d frames)",
        len(chunks),
        len(segments),
        chunks_dir,
        grid.trained_min_frames,
        grid.trained_max_frames,
    )
    _log_unmeasured_chunks(chunks, grid, measured_ceiling)
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


def _log_unmeasured_chunks(
    chunks: Sequence[AudioChunk],
    grid: FrameGrid,
    ceiling: MeasuredCeiling = CALIBRATED_CEILING,
) -> None:
    """Name, once per run, every chunk longer than the best-evidenced frame
    count available.

    Reported rather than refused, deliberately: 362 frames is inside H3's
    trained range and the whole point of issue #27 is that a long take is
    available. But VRAM above the measured ceiling is unknown, and an
    over-committed card here does not raise CUDA OOM -- it goes silent
    mid-load and wedges the host past SIGKILL (issues #23, #24). A run about
    to spend hours on frame counts nobody has measured should say so before
    it starts, not after. (The *refusal* on the same axis is
    ``envelope.check_render_envelope``, which additionally knows this run's
    resolution; this stays a warning because it does not.)

    **The message names its own evidence (issue #98).** It used to read
    "exceed 141 frames -- the longest anything ever rendered on this card",
    and that was false for months: the finished "Deathless" v13 render holds
    45 of 80 chunks above 141 frames, 15 of them at 192, confirmed with
    ``ffprobe -count_frames`` against the mp4s. ``MEASURED_MAX_FRAMES``
    records the largest frame count measured *when it was written*, and
    nothing updates it when a render quietly goes past. So the number now
    arrives as a :class:`~music_video_maker.envelope.MeasuredCeiling` that
    carries where it came from, and a caller holding a ``run_state.json``
    (``envelope.measured_ceiling``) can hand over a number that is evidence
    rather than a memory.

    Note this fires for filler chunks too: with ``max_chunk_seconds`` at H3's
    trained ceiling, a long instrumental already tiles into chunks well past
    the ceiling without anyone asking for a long take.
    """
    # Issue #100: VRAM is paid on what H3 was asked to render, which is not
    # the chunk's own length once an overrun is in play.
    over = [c for c in chunks if (c.rendered_frame_count or 0) > ceiling.frames]
    if not over:
        return
    longest = max(over, key=lambda c: c.rendered_frame_count or 0)
    logger.warning(
        "%d of %d chunk(s) exceed %d frames, which is %s. The longest is chunk %d at %d "
        "frames (%.3fs). This is inside H3's trained range and costs no extra wall clock, but "
        "VRAM behaviour above %d frames is unmeasured here and a card that runs out can wedge "
        "the host rather than raising CUDA OOM (issues #23, #24). Watch the first one -- "
        "docs/runbook-288-frame-proof.md is the attended way to measure it.",
        len(over),
        len(chunks),
        ceiling.frames,
        ceiling.provenance,
        longest.chunk_id,
        longest.rendered_frame_count,
        grid.frames_to_seconds(longest.rendered_frame_count or 0),
        ceiling.frames,
    )


# --------------------------------------------------------------------------- #
# Timeline-vs-track drift (issue #22)
# --------------------------------------------------------------------------- #


def timeline_track_drift_seconds(
    chunks: Sequence[AudioChunk], track_duration: float
) -> float:
    """How far the rendered chunk timeline's end sits from the master
    track's own duration.

    Positive means the timeline **overshoots** the track: rendered seconds
    past the end of the song. This is exactly what a music-video mux's
    ``-shortest`` throws away today with nothing logging it (CLAUDE.md's
    "-shortest" invariant, issue #22) -- 1.837s / 47 frames on "Deathless",
    because the final tile is padded up to H3's 124-frame trained floor when
    less than that much track remains. Negative means the timeline
    **undershoots**: the mux's ``-shortest`` ends the file at the
    end of the video, which is the *worse* defect for a music video (the
    song's own ending is cut out of the finished file). It was real until
    2026-09-13: a trailing gap under one trained-floor chunk (~5.167s) was
    dropped rather than covered. ``_cover_instrumentals`` now grows the last
    filler or appends a floor-length instrumental tile, so a negative value
    here means the invariant broke somewhere new.

    Reads only ``chunks[-1].end``, deliberately, rather than summing
    durations or re-deriving coverage: a contiguous timeline's last chunk end
    already *is* the total rendered runtime a viewer experiences, the same
    number :func:`_cover_instrumentals`'s own "tiling %.3fs of a %.3fs track"
    log line reports, and this reads correctly whether or not
    ``cover_instrumentals`` was on.

    Pure arithmetic: no I/O, no subprocess, safe to call from ``--prepare``
    (no GPU) as well as after a full render, at Stage 2's own boundary
    before any GPU time is spent. Returns ``0.0`` for an empty ``chunks`` --
    every caller here already refuses to proceed on zero chunks before this
    could be reached, so there is nothing to compare.
    """
    if not chunks:
        return 0.0
    return chunks[-1].end - track_duration
