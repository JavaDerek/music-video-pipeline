"""Stage 1: forced alignment of the master audio to lyric text (issue #3).

Uses ``stable-ts`` **forced alignment** (``model.align()``) -- never
``model.transcribe()``. The lyrics file is immutable truth (see project
CLAUDE.md); ASR transcription of sung vocals buried in an instrumental mix
hallucinates words that were never sung. ``suppress_silence=True`` enables
stable-ts's integrated Silero VAD so timestamps get clamped instead of
stretching a word across a guitar solo, and ``regroup=True`` lets stable-ts
restructure the forced words into logical segments.

``stable-ts`` (and the ``torch`` it depends on) is an optional, heavy
``[align]`` extra -- it must never be imported at module load time, or every
other lane's tests (which do not install it) break. It is imported lazily,
inside :func:`_load_model`, only when no model is injected.

Counterpoint (issue #33, level 3): ``model.align()`` maps audio to **one
linear text**, so a section where two voices sing *different* words at once
cannot be handed to it as two streams. The **alignment spine rule** resolves
that: the spine (the first sub-block of a ``[simultaneously]`` block) is the
only text stable-ts ever sees, and each concurrent
:class:`~music_video_maker.lyrics.CounterpointStream` *inherits that span*,
its words distributed proportionally across it (issue #33 open question 2 --
proportional, because no lip-sync accuracy claim can be made about a voice
the aligner never heard separately). Those derived segments come back as
:class:`ConcurrentSegment` on a :class:`CounterpointAlignmentResult`, which
**is** an ``AlignmentResult`` -- every stage written before #33 keeps seeing
exactly the spine timeline it always did, and a stage that understands
counterpoint reads the extra tuple.

Alignment quality (issue #35): every call to :func:`align` routes stable-ts's
own ``warnings`` (e.g. ``"1/27 segments failed to align."``) into ``logger``,
runs :func:`music_video_maker.alignment_quality.evaluate_alignment_quality`
over the result, and logs its summary at INFO unconditionally. Pass
``strict_alignment=True`` to additionally raise
:class:`music_video_maker.alignment_quality.AlignmentQualityError` when the
report has a critical finding -- see ``alignment_quality``'s module
docstring for the full API and the checks it runs.
"""

from __future__ import annotations

import logging
import warnings
import wave
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from music_video_maker import alignment_quality
from music_video_maker.contracts import (
    AlignedSegment,
    AlignmentOverride,
    AlignmentResult,
    LyricLine,
    WordTiming,
)
from music_video_maker.lyrics import CounterpointStream

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class ConcurrentSegment(AlignedSegment):
    """A counterpoint segment (issue #33): words sung *at the same time* as
    the spine, whose timings are **derived** from the spine's span rather
    than measured against the audio.

    A subclass of :class:`AlignedSegment` so anything that can read a segment
    can read one of these, with three extra facts a consumer needs in order
    not to confuse index spaces:

    * :attr:`index` continues the spine's numbering rather than restarting at
      0, so a bare segment index identifies exactly one segment across both
      tuples. It is still ``AlignedSegment`` index space -- never a
      ``LyricLine`` index and never a chunk id.
    * :attr:`spine_segment_indices` names the spine segments this one
      overlaps in time, which is how Stage 2a can tell that a chunk built
      from spine segment *n* also has another voice in it.
    * :attr:`stream_index` / :attr:`block_index` identify which concurrent
      voice of which ``[simultaneously]`` block produced it.

    Deliberately *not* in ``result.segments``: those are the spine, and the
    spine alone is the timeline the video is cut against. Merging the two
    tuples would double-count ``voiced_duration`` and make every concurrent
    segment look like an overlapping-segments defect.
    """

    spine_segment_indices: tuple[int, ...] = ()
    stream_index: int = 0
    """Which concurrent voice within its block. 1-based: 0 is the spine,
    which is never a ``ConcurrentSegment``."""
    block_index: int = 0
    """Which ``[simultaneously]`` block in the lyrics file, 0-based."""


@dataclass(frozen=True)
class CounterpointAlignmentResult(AlignmentResult):
    """An :class:`AlignmentResult` that also carries the concurrent voices.

    Returned by :func:`align` only when the lyrics actually contain
    counterpoint, so a level-1/level-2 file keeps getting a plain
    ``AlignmentResult`` and nothing about the common path changes.

    ``voiced_duration`` deliberately still counts the spine only:
    counterpoint occupies *the same seconds* of the master track, and adding
    them would report more vocal than the track contains.
    """

    concurrent_segments: tuple[ConcurrentSegment, ...] = ()

    @property
    def all_segments(self) -> tuple[AlignedSegment, ...]:
        """Spine + concurrent, in segment-index order. For reporting and
        diagnostics -- not a timeline."""
        return tuple(sorted(self.segments + self.concurrent_segments, key=lambda s: s.index))

    def concurrent_for_segment(self, segment_index: int) -> tuple[ConcurrentSegment, ...]:
        """Concurrent segments overlapping the spine segment ``segment_index``.

        ``segment_index`` is spine ``AlignedSegment`` index space. Provided so
        Stage 2a can ask "does this chunk have another voice in it?" without
        re-deriving overlap and getting the index space wrong -- the recurring
        bug this project keeps paying for.
        """
        return tuple(
            segment
            for segment in self.concurrent_segments
            if segment_index in segment.spine_segment_indices
        )

    def concurrent_in_span(self, start: float, end: float) -> tuple[ConcurrentSegment, ...]:
        """Concurrent segments overlapping the time span ``[start, end)``.

        The timeline-space companion to :meth:`concurrent_for_segment`, for a
        consumer (Stage 2a merges/splits chunks) that holds seconds rather
        than segment indices.
        """
        return tuple(
            segment
            for segment in self.concurrent_segments
            if segment.start < end and segment.end > start
        )


class AlignmentOverrideError(ValueError):
    """An authored :class:`~music_video_maker.contracts.AlignmentOverride`
    cannot be applied: unknown segment index, inverted span, or a span that
    would overlap a neighboring segment. Always a refusal -- a stale override
    (the lyrics changed, the indices moved) silently retiming the wrong
    segment is exactly the failure class the mandatory ``reason`` exists to
    make debuggable, not survivable."""


def _apply_overrides(
    segments: tuple[AlignedSegment, ...],
    overrides: Sequence[AlignmentOverride],
) -> tuple[AlignedSegment, ...]:
    """Retime the named segments, refusing anything incoherent. See
    :class:`~music_video_maker.contracts.AlignmentOverride`.

    All overrides are applied first and the FINAL sequence is validated for
    chronology -- not each step against not-yet-moved neighbors. Two
    overrides that are only coherent together (moving adjacent segments
    later in tandem) must not depend on the order they were written in the
    config file.
    """
    result = list(segments)
    for override in overrides:
        if not (0 <= override.segment_index < len(result)):
            raise AlignmentOverrideError(
                f"alignment override names segment {override.segment_index} but only "
                f"{len(result)} segment(s) were aligned (reason given: {override.reason!r})"
            )
        if override.end <= override.start:
            raise AlignmentOverrideError(
                f"alignment override for segment {override.segment_index} has an inverted "
                f"span {override.start!r}..{override.end!r} (reason: {override.reason!r})"
            )
        idx = override.segment_index
        old = result[idx]
        words = old.words
        if words:
            per = (override.end - override.start) / len(words)
            words = tuple(
                replace(w, start=override.start + i * per, end=override.start + (i + 1) * per)
                for i, w in enumerate(words)
            )
        result[idx] = replace(old, start=override.start, end=override.end, words=words)
        logger.warning(
            "Alignment override applied: segment %d [%.3f-%.3f] -> [%.3f-%.3f] (%r). "
            "Reason: %s",
            idx,
            old.start,
            old.end,
            override.start,
            override.end,
            old.text[:48],
            override.reason,
        )

    overridden = {o.segment_index for o in overrides}
    # strict=False deliberately: this is the adjacent-pairs idiom, so the two
    # operands differ in length by one *by construction*. strict=True would
    # raise on every well-formed timeline.
    for a, b in zip(result, result[1:], strict=False):
        if a.end > b.start + 1e-6:
            blame = [i for i in (a.index, b.index) if i in overridden]
            raise AlignmentOverrideError(
                f"after applying alignment overrides, segment {a.index} "
                f"(ends {a.end:.3f}) overlaps segment {b.index} (starts {b.start:.3f}). "
                f"Override(s) on segment(s) {blame or 'none -- pre-existing overlap?'} "
                f"retime against a timeline they disagree with; fix the neighboring "
                f"spans too"
            )
    return tuple(result)


def align(
    audio_file: Path | str,
    lyric_lines: Sequence[LyricLine],
    *,
    model: object | None = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str = DEFAULT_LANGUAGE,
    strict_alignment: bool = False,
    counterpoint: Sequence[CounterpointStream] | None = None,
    overrides: Sequence[AlignmentOverride] = (),
) -> AlignmentResult:
    """Force-align ``audio_file`` against tag-stripped ``lyric_lines``.

    ``lyric_lines`` must already be tag-stripped (issue #6's output) -- this
    function never parses ``[Name: Role]`` tags itself, it only carries each
    line's already-resolved ``character`` through onto the output segments.

    ``model`` is the injectable stable-ts model seam: tests pass a fake/mock
    exposing ``.align(audio, text, language=..., suppress_silence=...,
    regroup=...)`` shaped like stable-ts's ``WhisperResult``. When ``model``
    is ``None``, a real model is lazily loaded via stable-ts (requires the
    ``[align]`` extra -- not available/needed in the test environment).

    ``strict_alignment`` (issue #35, mirrors ``RunConfig.strict_alignment``
    -- callers such as cli.py pass their config value straight through) --
    when ``True``, raises ``alignment_quality.AlignmentQualityError`` if the
    computed quality report has a critical finding. Default ``False``
    matches the existing behaviour: report, never refuse. The quality
    summary is logged at INFO either way, on every call.

    ``counterpoint`` (issue #33 level 3) is the sequence of concurrent
    streams whose spine is in ``lyric_lines``. Leave it ``None`` and it is
    read off ``lyric_lines`` itself when that is a
    :class:`~music_video_maker.lyrics.LyricsDocument` -- which is what
    ``parse_lyrics`` returns, so an unmodified caller cannot silently drop a
    file's counterpoint. Pass it explicitly (including ``()`` to suppress)
    when the lines and the streams are held separately. With any stream
    present the return value is a :class:`CounterpointAlignmentResult` (with
    an empty ``concurrent_segments`` if none could be timed -- always
    logged); with none it is a plain ``AlignmentResult``, exactly as before.
    """
    audio_path = Path(audio_file)
    streams = _resolve_counterpoint(lyric_lines, counterpoint)

    non_empty = [line for line in lyric_lines if line.text and line.text.strip()]
    skipped = len(lyric_lines) - len(non_empty)
    if skipped:
        logger.info("Skipping %d empty lyric line(s) before alignment", skipped)

    if not non_empty:
        logger.warning("No non-empty lyric lines to align; returning an empty AlignmentResult")
        if streams:
            logger.error(
                "%d counterpoint stream(s) were supplied but there are no spine lyric "
                "lines to align them against; their timing cannot be derived and they "
                "are dropped",
                len(streams),
            )
        empty_result = AlignmentResult(segments=(), track_duration=_probe_duration(audio_path))
        empty_report = alignment_quality.evaluate_alignment_quality(empty_result)
        alignment_quality.log_report(empty_report, context=str(audio_path))
        return empty_result

    text_blob = "\n".join(line.text.strip() for line in non_empty)

    # Positional word -> source-character map. Forced alignment never adds,
    # drops, or reorders words -- regroup only changes how they are grouped
    # into segments -- so a flat left-to-right walk correctly attributes each
    # output segment's words back to the LyricLine(s) they came from even
    # when stable-ts merges or splits relative to our original line breaks.
    word_owner = [line.characters for line in non_empty for _ in line.text.split()]

    # The same positional walk, keyed by LyricLine.index, is what lets a
    # counterpoint stream find the *span* its spine lines were aligned to.
    # Built here, from the same `non_empty` list, so the two views can never
    # disagree about which word position belongs to which line.
    line_word_ranges = _line_word_ranges(non_empty)

    if model is None:
        model = _load_model(model_size)

    logger.info(
        "Running forced alignment: audio=%s lines=%d model_size=%s language=%s",
        audio_path,
        len(non_empty),
        model_size,
        language,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_result = model.align(  # type: ignore[attr-defined]
            str(audio_path),
            text_blob,
            language=language,
            suppress_silence=True,
            regroup=True,
        )
    for caught_warning in caught:
        logger.warning(
            "stable-ts warning during alignment of %s (%d line(s)): %s",
            audio_path,
            len(non_empty),
            caught_warning.message,
        )

    segments, segment_word_ranges = _build_segments(
        raw_result, word_owner, default_characters=non_empty[0].characters
    )
    if overrides:
        # Issue #42: authored corrections, applied BEFORE quality evaluation
        # and counterpoint derivation so both judge the corrected timeline.
        segments = _apply_overrides(segments, overrides)
    track_duration = _resolve_track_duration(audio_path, segments)

    concurrent_segments = (
        _build_concurrent_segments(streams, segments, segment_word_ranges, line_word_ranges)
        if streams
        else ()
    )

    result: AlignmentResult
    # Typed by what the *lyrics* contain, not by what could be derived: a
    # level-3 file whose streams were all dropped must still read as a
    # counterpoint file (with an empty tuple and a logged reason), never as a
    # plain one that happens to look fine.
    if streams:
        result = CounterpointAlignmentResult(
            segments=segments,
            track_duration=track_duration,
            concurrent_segments=concurrent_segments,
        )
    else:
        result = AlignmentResult(segments=segments, track_duration=track_duration)

    voiced_pct = (result.voiced_duration / track_duration * 100.0) if track_duration else 0.0
    logger.info(
        "Alignment complete: %d segment(s), voiced=%.2fs / track=%.2fs (%.1f%%)"
        "%s",
        len(result.segments),
        result.voiced_duration,
        result.track_duration,
        voiced_pct,
        (
            f", plus {len(concurrent_segments)} counterpoint segment(s) from "
            f"{len(streams)} concurrent stream(s)"
            if concurrent_segments
            else ""
        ),
    )

    word_confidences = _extract_word_confidences(raw_result)
    quality_report = alignment_quality.evaluate_alignment_quality(
        result, low_confidence_words=word_confidences
    )
    alignment_quality.log_report(quality_report, context=str(audio_path))
    alignment_quality.raise_if_blocking(quality_report, strict=strict_alignment)

    return result


def _resolve_counterpoint(
    lyric_lines: Sequence[LyricLine],
    counterpoint: Sequence[CounterpointStream] | None,
) -> tuple[CounterpointStream, ...]:
    """Explicit argument wins; otherwise read the streams off the lines when
    they arrived as a ``LyricsDocument`` (see this module's docstring)."""
    if counterpoint is not None:
        return tuple(counterpoint)
    return tuple(getattr(lyric_lines, "counterpoint", ()))


def _line_word_ranges(non_empty: Sequence[LyricLine]) -> dict[int, tuple[int, int]]:
    """``LyricLine.index`` -> half-open ``[first, last)`` **word position** in
    the flat text blob handed to stable-ts.

    Word position is a third index space, distinct from both ``LyricLine``
    and ``AlignedSegment`` indices; it exists only inside this module and is
    never exposed. Keyed by ``LyricLine.index`` rather than by position in
    ``non_empty`` on purpose -- a caller that dropped empty lines must not
    shift what a counterpoint stream's ``spine_line_indices`` refer to.
    """
    ranges: dict[int, tuple[int, int]] = {}
    position = 0
    for line in non_empty:
        count = len(line.text.split())
        ranges[line.index] = (position, position + count)
        position += count
    return ranges


def _word_time_by_position(
    segments: Sequence[AlignedSegment], segment_word_ranges: Sequence[tuple[int, int]]
) -> dict[int, WordTiming]:
    """Word position -> its aligned timing, for the positions that have one.

    A segment whose word count disagrees with its recorded range (stable-ts
    is not supposed to add or drop words, but a fake/degraded result can) is
    skipped rather than trusted, so a mismatch degrades to the coarser
    segment-bounds fallback instead of mis-timing every later word.
    """
    by_position: dict[int, WordTiming] = {}
    for segment, (start_pos, end_pos) in zip(segments, segment_word_ranges, strict=False):
        if len(segment.words) != end_pos - start_pos:
            if segment.words:
                logger.debug(
                    "Segment %d has %d word timing(s) for %d transcript word(s); "
                    "not using it for counterpoint word mapping",
                    segment.index,
                    len(segment.words),
                    end_pos - start_pos,
                )
            continue
        for offset, word in enumerate(segment.words):
            by_position[start_pos + offset] = word
    return by_position


def _spine_span(
    stream: CounterpointStream,
    segments: Sequence[AlignedSegment],
    segment_word_ranges: Sequence[tuple[int, int]],
    line_word_ranges: dict[int, tuple[int, int]],
) -> tuple[float, float] | None:
    """The ``(start, end)`` of the spine text this stream is sung over.

    Word timings first (tight, and immune to stable-ts having regrouped a
    spine line together with a neighbouring non-block line), segment bounds
    as the fallback. ``None`` -- always logged -- when the span cannot be
    established at all, in which case the stream is dropped rather than
    given invented timings.
    """
    known = [i for i in stream.spine_line_indices if i in line_word_ranges]
    missing = [i for i in stream.spine_line_indices if i not in line_word_ranges]
    if missing:
        logger.error(
            "Counterpoint stream (block %d, stream %d, voices=%s) names spine line "
            "index/indices %s that are not in the aligned lyric lines -- LyricLine "
            "index space and the lines actually handed to align() disagree",
            stream.block_index,
            stream.stream_index,
            list(stream.characters),
            missing,
        )
    if not known:
        return None

    first_pos = min(line_word_ranges[i][0] for i in known)
    end_pos = max(line_word_ranges[i][1] for i in known)
    if end_pos <= first_pos:
        logger.warning(
            "Counterpoint stream (block %d, stream %d) has a spine with no words; "
            "dropping it",
            stream.block_index,
            stream.stream_index,
        )
        return None

    by_position = _word_time_by_position(segments, segment_word_ranges)
    starts = [by_position[p].start for p in range(first_pos, end_pos) if p in by_position]
    ends = [by_position[p].end for p in range(first_pos, end_pos) if p in by_position]
    if starts and ends:
        return min(starts), max(ends)

    covering = [
        segment
        for segment, (a, b) in zip(segments, segment_word_ranges, strict=False)
        if a < end_pos and b > first_pos
    ]
    if not covering:
        logger.error(
            "Counterpoint stream (block %d, stream %d) has no aligned segment covering "
            "its spine words (positions %d-%d); dropping it",
            stream.block_index,
            stream.stream_index,
            first_pos,
            end_pos,
        )
        return None
    return min(s.start for s in covering), max(s.end for s in covering)


def _build_concurrent_segments(
    streams: Sequence[CounterpointStream],
    segments: Sequence[AlignedSegment],
    segment_word_ranges: Sequence[tuple[int, int]],
    line_word_ranges: dict[int, tuple[int, int]],
) -> tuple[ConcurrentSegment, ...]:
    """Give every concurrent stream timings inside its spine's span.

    The distribution is **proportional by word** across the span (issue #33
    open question 2): every word of the stream gets an equal slice, and each
    written line becomes one :class:`ConcurrentSegment` covering its words.
    That is a deliberate approximation -- the aligner never heard this voice
    on its own, so no lip-sync accuracy claim is made about it, and the honest
    cheap rule beats an invented precise-looking one.
    """
    next_index = max((s.index for s in segments), default=-1) + 1
    built: list[ConcurrentSegment] = []

    for stream in streams:
        span = _spine_span(stream, segments, segment_word_ranges, line_word_ranges)
        if span is None:
            continue
        start, end = span
        total_words = stream.word_count
        if total_words == 0:
            logger.warning(
                "Counterpoint stream (block %d, stream %d, voices=%s) has no words; "
                "skipping it",
                stream.block_index,
                stream.stream_index,
                list(stream.characters),
            )
            continue
        if end - start <= 0.0:
            logger.warning(
                "Counterpoint stream (block %d, stream %d, voices=%s) sits over a spine "
                "span of %.3fs (%.3f -> %.3f); its timings cannot be distributed, so it "
                "is dropped rather than given zero-length segments",
                stream.block_index,
                stream.stream_index,
                list(stream.characters),
                end - start,
                start,
                end,
            )
            continue

        per_word = (end - start) / total_words
        cursor = 0
        for text in stream.texts:
            words = text.split()
            if not words:
                continue
            seg_start = start + cursor * per_word
            seg_end = start + (cursor + len(words)) * per_word
            timings = tuple(
                WordTiming(
                    word=word,
                    start=start + (cursor + offset) * per_word,
                    end=start + (cursor + offset + 1) * per_word,
                )
                for offset, word in enumerate(words)
            )
            cursor += len(words)
            built.append(
                ConcurrentSegment(
                    index=next_index,
                    text=text,
                    start=seg_start,
                    end=seg_end,
                    words=timings,
                    characters=stream.characters,
                    spine_segment_indices=tuple(
                        s.index for s in segments if s.start < seg_end and s.end > seg_start
                    ),
                    stream_index=stream.stream_index,
                    block_index=stream.block_index,
                )
            )
            next_index += 1

    if built:
        logger.info(
            "Derived %d counterpoint segment(s) from %d concurrent stream(s) by "
            "proportional distribution across their spine spans",
            len(built),
            len(streams),
        )
    return tuple(built)


def _extract_word_confidences(
    raw_result: object,
) -> tuple[alignment_quality.LowConfidenceWord, ...]:
    """Pull stable-ts's own per-word alignment ``probability`` off every word
    in the raw result, before it's discarded by :func:`_build_segments`
    (this project's ``contracts.WordTiming`` has no field for it -- see
    ``alignment_quality``'s module docstring). Returns one entry per word
    that *has* a probability -- filtering down to the actually-low ones is
    ``alignment_quality.evaluate_alignment_quality``'s job, not this
    extraction step. Defensive: stable-ts types ``probability`` as
    ``Optional[float]``, so a word with none is skipped here, never treated
    as automatically low-confidence.
    """
    raw_segments = list(getattr(raw_result, "segments", None) or [])
    confidences: list[alignment_quality.LowConfidenceWord] = []
    for segment_index, raw_segment in enumerate(raw_segments):
        for raw_word in list(getattr(raw_segment, "words", None) or []):
            probability = getattr(raw_word, "probability", None)
            if probability is None:
                continue
            confidences.append(
                alignment_quality.LowConfidenceWord(
                    segment_index=segment_index,
                    word=str(getattr(raw_word, "word", "")),
                    start=float(getattr(raw_word, "start", 0.0)),
                    end=float(getattr(raw_word, "end", 0.0)),
                    probability=float(probability),
                )
            )
    return tuple(confidences)


def _build_segments(
    raw_result: object,
    word_owner: list[tuple[str, ...]],
    *,
    default_characters: tuple[str, ...],
) -> tuple[tuple[AlignedSegment, ...], tuple[tuple[int, int], ...]]:
    """Convert the raw stable-ts result into ``AlignedSegment``s, and report
    the half-open word-position range each one consumed.

    The ranges are the same left-to-right walk the character attribution
    already does, surfaced rather than thrown away: counterpoint (issue #33)
    needs to map a *lyric line* to the segments that timed it, and
    recomputing that walk somewhere else is how the two would drift apart.
    """
    raw_segments = list(getattr(raw_result, "segments", None) or [])
    segments: list[AlignedSegment] = []
    word_ranges: list[tuple[int, int]] = []
    word_ptr = 0
    # A second pointer, deliberately: character attribution has always
    # advanced only on segments that carry word timings, and changing that
    # would move faces around. The range pointer additionally counts the
    # words of a timing-less segment's *text*, so the transcript positions
    # stay aligned with the text blob even then.
    range_ptr = 0
    last_characters = default_characters

    for i, raw_segment in enumerate(raw_segments):
        raw_words = list(getattr(raw_segment, "words", None) or [])
        word_timings = tuple(
            WordTiming(word=w.word, start=float(w.start), end=float(w.end)) for w in raw_words
        )

        text = str(raw_segment.text).strip()
        span = len(raw_words) if raw_words else len(text.split())
        word_ranges.append((range_ptr, range_ptr + span))
        range_ptr += span

        if raw_words and word_owner:
            owner_index = min(word_ptr, len(word_owner) - 1)
            characters = word_owner[owner_index]
            word_ptr += len(raw_words)
        else:
            characters = last_characters
        last_characters = characters

        segments.append(
            AlignedSegment(
                index=i,
                text=text,
                start=float(raw_segment.start),
                end=float(raw_segment.end),
                words=word_timings,
                characters=characters,
            )
        )

    return tuple(segments), tuple(word_ranges)


def _resolve_track_duration(audio_path: Path, segments: tuple[AlignedSegment, ...]) -> float:
    track_duration = _probe_duration(audio_path)
    if segments:
        max_end = max(s.end for s in segments)
        if track_duration < max_end:
            logger.warning(
                "Probed track_duration=%.2fs is shorter than last aligned segment end=%.2fs "
                "(%s); using the segment end instead",
                track_duration,
                max_end,
                audio_path,
            )
            track_duration = max_end
    return track_duration


def _probe_duration(audio_path: Path) -> float:
    """Read WAV duration via the stdlib ``wave`` module -- no ffmpeg/pydub needed.

    Degrades gracefully (logs and returns ``0.0``) for non-WAV or unreadable
    files rather than crashing Stage 1 over a duration probe.
    """
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / float(rate)
    except (wave.Error, OSError) as exc:
        logger.warning(
            "Could not probe audio duration for %s (%s); defaulting track_duration to 0.0",
            audio_path,
            exc,
        )
        return 0.0


def _load_model(model_size: str) -> object:
    """Lazily import and load a stable-ts model. Requires the ``[align]`` extra."""
    try:
        import stable_whisper  # noqa: PLC0415 -- deliberately lazy, see module docstring
    except ImportError as exc:
        logger.error(
            "stable-ts is not installed; install the 'align' extra "
            "(pip install music-video-maker[align]) to run forced alignment"
        )
        raise RuntimeError(
            "stable-ts is not installed. Install the 'align' extra to run forced alignment, "
            "or inject a model= for testing."
        ) from exc

    logger.info("Loading stable-ts model size=%s", model_size)
    return stable_whisper.load_model(model_size)
