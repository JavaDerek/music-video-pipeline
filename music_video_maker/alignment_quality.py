"""Alignment quality checks (issue #35): a pure-function evaluator over an
:class:`~music_video_maker.contracts.AlignmentResult`, plus the small amount
of plumbing that turns its findings into log lines and, optionally, a refusal.

Forced alignment can fail badly while reporting nothing: zero-length
segments, five words crammed into 70ms, a single lyric line split across a
38-second gap, or a lyric placed in a passage with no singing in it at all.
None of that raises an exception -- it produces a plausible-looking
:class:`AlignmentResult` that downstream stages happily slice and render.
This module makes those failure modes visible and, on request, blocking.

Public API
----------
- :func:`evaluate_alignment_quality` -- the pure check. No audio, no model,
  no I/O; takes an ``AlignmentResult`` (plus an optional side-channel of
  :class:`LowConfidenceWord` -- see "Per-word confidence" below) and returns
  an :class:`AlignmentQualityReport`.
- :class:`AlignmentQualityReport` / :class:`Finding` / :class:`Severity` --
  the structured result. Each finding names the segment index, its timings,
  and what is wrong, and carries a :class:`Severity` so a caller can
  threshold.
- :func:`format_summary` -- one-line human-readable summary of a report.
- :func:`log_report` -- logs that summary at INFO (always -- this is what
  satisfies "render.log contains an alignment-quality summary on every
  run"), plus one WARNING/ERROR line per finding at its severity.
- :func:`raise_if_blocking` -- the "optionally refuse" half. A no-op unless
  ``strict=True`` *and* the report has a finding at/above ``threshold``
  (default :attr:`Severity.CRITICAL`), in which case it raises
  :class:`AlignmentQualityError` carrying the report. This is a typed
  exception, not a bare bool, specifically so a caller several layers up
  (cli.py) can catch it, log the report it carries, and choose its own exit
  code -- this module never calls ``sys.exit`` itself.

Wiring convention for callers (e.g. cli.py, which owns exit-code decisions
and is not touched by this module): call ``evaluate_alignment_quality(result)``
after alignment, then ``raise_if_blocking(report, strict=cfg.strict_alignment)``.
``music_video_maker.alignment.align()`` already does exactly this internally
on every call (logging the summary unconditionally, and raising only when
its own ``strict_alignment=`` argument is ``True``), so most callers never
need to call this module directly at all.

Per-word confidence (issue #35 item 4) -- VERIFIED, not assumed
-----------------------------------------------------------------
Read directly from the installed ``stable_whisper`` package
(``stable_whisper/result.py``'s ``WordTiming.__init__``): each aligned word
carries an ``Optional[float] probability`` attribute, populated by the same
forced-alignment code path ``model.align()`` uses
(``stable_whisper/alignment.py``, e.g. its ``fail_segs`` check reads
``word.probability``). So yes, per-word confidence exists and low
confidence *is* a cheaper, more direct signal than the timing heuristics
below for the "hallucinated word" case -- a word forced onto a time span
where the model finds no matching audio typically has near-zero
probability. But that field lives on stable-ts's own result objects, not on
this project's ``contracts.WordTiming`` (which this module does not own and
must not change) -- so there is nowhere on ``AlignmentResult`` to carry it
through. Instead, ``alignment.align()`` extracts it from the raw stable-ts
result *before* conversion and passes it into
``evaluate_alignment_quality(..., low_confidence_words=...)`` as a bag of
:class:`LowConfidenceWord`. The probability field is optional on stable-ts's
own object (``Optional[float]``) -- treated defensively here: a word with no
probability is silently skipped, never treated as automatically bad.

Counterpoint (issue #33 level 3)
--------------------------------
When the result carries concurrent segments (a
``alignment.CounterpointAlignmentResult``), they are evaluated too -- but
separately from the spine and never against it. Two reasons, both structural:

- A concurrent segment **overlaps the spine by definition**. Running the
  pairwise order/overlap checks across the two tuples would report the
  feature as a defect, once per segment.
- A concurrent segment's timings are **derived** (distributed proportionally
  across the spine's span), not measured against audio. So its findings are
  capped at WARNING and use their own codes: the honest signal they carry is
  about the *lyrics file* ("more counterpoint words than the spine span can
  hold"), not about the aligner, and ``--strict-alignment`` must not refuse a
  render over arithmetic this module did itself.

``segment_count`` stays the spine count -- that is the timeline the video is
cut against -- with the concurrent count reported alongside it in
``counterpoint_segment_count`` and in :func:`format_summary`.

Thresholds
----------
Every threshold below is a named module constant with a comment justifying
the number, and every one is overridable as a keyword argument to
:func:`evaluate_alignment_quality`. None of them are wired into
``config.py`` -- that file is not owned by this module.

Known heuristic limitations (documented, not hidden)
------------------------------------------------------
``split_lyric_line`` and ``isolated_segment`` cannot see the original
per-line lyric boundaries (``AlignedSegment`` does not carry one) or VAD
silence maps (not exposed on the contract either), so both rely on
timing + text heuristics that are necessarily imperfect:

- ``split_lyric_line`` flags a large gap only when the earlier segment's
  last word is a closed-class word that rarely ends a clause (a contracted
  subject, article, preposition, conjunction, ...). A genuine two-line gap
  that happens to end on one of those words will occasionally be
  over-flagged; that is judged a much smaller cost than staying silent on
  the 38-second split this issue exists to catch.
- ``isolated_segment`` deliberately does **not** evaluate the *first*
  segment in a result -- a long instrumental intro before the first vocal
  is completely normal and must never be flagged. The *last* segment is
  evaluated asymmetrically (a large gap-before alone is enough, since there
  is no "after" neighbor to compare against) because the real-world
  hallucination this issue documents is itself the last aligned segment,
  arriving long after the true vocal ended. This trades a small false-positive
  risk (a legitimately late final line after a long bridge) for catching the
  documented failure; there is no fixture in this project requiring the
  trailing edge to be false-positive-free, so this trade was made
  deliberately.
"""

from __future__ import annotations

import logging
import re
import shutil
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from music_video_maker.contracts import AlignedSegment, AlignmentResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Thresholds -- named constants, each justified, each overridable.
# --------------------------------------------------------------------------- #

# A segment shorter than this cannot contain a real spoken/sung syllable --
# the shortest phonemes run roughly 30-50ms. Exact 0.0s ("zero-length")
# segments are the clearest case (e.g. this issue's 198.580 -> 198.580
# "I'm"); this constant also catches near-zero ones without being so large
# it swallows the genuinely-short-but-real 70ms/5-word segment below (that
# one is caught by the words-per-second check instead).
NEAR_ZERO_DURATION_S = 0.03

# Fastest plausible sung/spoken delivery. Sustained fast rap can reach
# ~5-6 words/sec; auctioneer-fast speech tops out near 7-8. Set comfortably
# above real fast delivery but two orders of magnitude below the observed
# failure (5 words / 70ms = ~71 words/sec).
MAX_WORDS_PER_SECOND = 7.0

# Slowest plausible sustained delivery. Ballad-paced held notes can run a
# couple of seconds per word -- this project's own "normal song" fixture
# has a segment at ~0.62 words/sec. Below this it is more likely a
# stretched/misaligned span than a real held note.
MIN_WORDS_PER_SECOND = 0.2

# A pause *within* one lyric line rarely exceeds a few seconds of phrasing.
# 8s is generous headroom above ordinary in-line breathing pauses, while
# staying far below both a legitimate between-line/section gap (this
# project's own "long gap" fixture uses 30s) and the ~38s split-line
# failure this issue documents.
INTRA_LINE_GAP_S = 8.0

# How large a gap-before (see the "isolated_segment" docstring note above
# for why this is asymmetric, non-first/last-segment-only) has to be before
# a segment is treated as disconnected from the vocal activity around it --
# the fingerprint of a hallucinated lyric placed into an instrumental
# passage. Set well above ordinary between-line gaps and below the ~37s
# gap observed in the real failure.
ISOLATED_SEGMENT_GAP_S = 20.0

# stable-ts's own forced-alignment token probability (see the "Per-word
# confidence" docstring section). Below LOW, the aligner had weak evidence
# for the placement; below CRITICAL, essentially none.
LOW_WORD_CONFIDENCE = 0.35
CRITICAL_WORD_CONFIDENCE = 0.10

MIN_WORDS_FOR_COLLAPSE_CHECK = 20
"""Below this many words the whole-track fraction is noise, and the caller may
have supplied only the words it already suspected."""

TRACK_CONFIDENCE_COLLAPSE_FRACTION = 0.60
"""Share of the track's words below ``critical_word_confidence`` at which the
alignment stops being trustworthy *as a whole*.

Calibrated against a real measurement rather than taste: the corrected "Lucky
Ones" transcript -- a usable alignment with four genuine, locatable defects --
came in at 24% below 0.10. A threshold under that would condemn a workable
track; 60% says most of the song had essentially no acoustic support, which is
the signature of the wrong audio, the wrong lyrics, or the wrong language."""

# Closed-class function words that virtually never end a grammatical
# clause/sentence in English lyrics: contracted subjects, articles,
# prepositions, conjunctions. A segment ending in one of these, paired with
# a large gap to the next segment, is treated as *the same lyric line*
# continuing after a mis-timed jump rather than a legitimate pause between
# two distinct lines. Heuristic and necessarily imperfect -- see the module
# docstring's "Known heuristic limitations" section.
CLAUSE_CONTINUATION_WORDS = frozenset(
    {
        "i'm", "we're", "you're", "they're", "he's", "she's", "it's", "that's",
        "there's", "who's", "i", "we", "you", "he", "she", "they", "the", "a",
        "an", "and", "but", "or", "so", "because", "if", "when", "while", "to",
        "of", "in", "on", "at", "for", "with", "by", "my", "your", "his", "her",
        "our", "their", "its", "is", "are", "am", "was", "were", "be", "been",
    }
)

# The consonant band (issue #71): d/th/l/s/F/v/m/r and most of the rest of
# English consonant articulation put energy above this frequency; a tonal
# instrumental fadeout does not. See the "Vocal-energy check" docstring
# section below and VOCAL_ENERGY_RATIO_THRESHOLD's comment for the
# measurement this number and the threshold below it are based on.
CONSONANT_BAND_HZ = 3400.0

# Below this duration, a highpass-filtered RMS estimate is averaging over too
# few cycles to be trustworthy in either direction -- too short to safely
# treat as a real measurement of "no voice here" (a false CRITICAL) or to
# let it distort another segment's baseline (a false negative on that other
# segment). Every real segment measured for this issue (all >=1.28s, see the
# docstring) cleared this easily; the ~70ms 5-word pileup this module's own
# words-per-second check exists to catch would not, and is correctly left to
# that check instead.
MIN_SEGMENT_DURATION_FOR_VOCAL_ENERGY_S = 1.0

# A baseline built from fewer than this many other qualifying segments is a
# guess, not a measurement -- same reasoning as MIN_WORDS_FOR_COLLAPSE_CHECK
# above. Below this, the check is skipped for that segment entirely rather
# than compared against noise.
VOCAL_ENERGY_BASELINE_MIN_SEGMENTS = 3

VOCAL_ENERGY_RATIO_THRESHOLD = 0.10
"""How small a segment's consonant-band share can be, relative to the
*median* share across this track's other qualifying placed segments, before
it is flagged as having no vocal energy at all.

Median rather than mean or max: a track where many placed segments are
themselves hallucinated (the scenario this constant's docstring is required
to address) drags a mean down with them and lets a single loud "other"
segment set an unstably high ceiling for max: the number needs to still mean
something with a *minority* of the "other" segments contaminated, which is
what a median gives for free -- see
test_finding_still_fires_when_two_of_five_other_segments_are_also_phantom.

Calibrated 2026-08-15 against real masters decoded to mono 16kHz (matching
what this module actually does -- see _decode_to_mono_16khz), consonant band
>3.4kHz, share = highpass RMS power / total RMS power over each window:

  "Deathless" (issue #71's case), ratio against the median of 23 other real
  sung windows (0.0026-0.0460, median 0.0157):
    - quietest REAL vocal window measured, 386-391s ("Now the island is no
      longer..."): share 0.0026 -> ratio 0.166 (16.6% of baseline)
    - second-quietest real window, 54-62s: share 0.0057 -> ratio 0.287 (29%)
    - the phantom itself, "deathless, Forevermore!" `small` placed at
      495-502s, 12s into the fadeout: share 0.0009 -> ratio 0.057 (5.7%)
    - the same phantom's tighter 498.73-501.63s span: share 0.0003 ->
      ratio 0.019 (1.9%)

  "The Lucky Ones" (issue #42's case, already caught by isolated_segment's
  51s/37s gap -- this check is not required to also catch it, and on the
  measurement below it does not): ratio against the median of 21 other real
  sung windows (0.0164-0.0855, median 0.0267):
    - quietest real window: share 0.0164 -> ratio 0.614 (61%)
    - the "the lucky one." phantom at 235.68-238.56s: share 0.0135 ->
      ratio 0.506 (51%) -- NOT caught by this check. Unlike "Deathless"'s
      fadeout, this window sits inside a still-playing instrumental outro
      (percussion has its own energy above 3.4kHz), so silence-vs-voice
      is not this track's phantom's acoustic signature the way it is on
      "Deathless"; the gap heuristic is what catches it, and it does.

0.10 sits between the worst-measured phantom ratio (5.7%) and the
lowest-measured real ratio (16.6%) -- closer to the real floor than the
phantom ceiling, deliberately: a missed phantom costs a viewer-visible
defect after hours of GPU time, a false positive costs one log line (or,
under --strict-alignment, a refusal before any of those hours are spent --
still cheaper than the alternative). 0.15 or higher was rejected: it leaves
under a 2x margin from the 16.6% real floor, which is too little room for a
song whose quietest real line is quieter than "Deathless"'s. 0.05 was
rejected as too conservative given that cost asymmetry -- it would have
missed this issue's own tighter 1.9%-ratio measurement of the documented
failure had the wider chunk-level span (5.7%) not been the one evaluated."""

# --------------------------------------------------------------------------- #
# Finding codes.
# --------------------------------------------------------------------------- #

FINDING_ZERO_LENGTH = "zero_length_segment"
FINDING_NEGATIVE_DURATION = "negative_duration_segment"
FINDING_WPS_HIGH = "implausible_words_per_second_high"
FINDING_WPS_LOW = "implausible_words_per_second_low"
FINDING_OUT_OF_ORDER = "out_of_order_segment"
FINDING_OVERLAP = "overlapping_segments"
FINDING_SPLIT_LINE = "split_lyric_line"
FINDING_ISOLATED = "isolated_segment"
FINDING_LOW_CONFIDENCE = "low_word_confidence"
FINDING_CONFIDENCE_COLLAPSE = "track_confidence_collapse"
FINDING_COUNTERPOINT_RATE = "counterpoint_delivery_rate"
FINDING_COUNTERPOINT_SPAN = "counterpoint_degenerate_span"
FINDING_NO_VOCAL_ENERGY = "no_vocal_energy_in_placed_segment"


class Severity(IntEnum):
    """Ordered so ``severity >= Severity.WARNING`` etc. works directly."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass(frozen=True)
class LowConfidenceWord:
    """One word from the raw stable-ts result whose alignment ``probability``
    fell below a confidence threshold. Built by ``alignment.align()`` from
    the raw stable-ts result (which carries ``.probability``; see the module
    docstring) -- this module never touches stable-ts or ``contracts.py``
    directly."""

    segment_index: int
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Finding:
    """One quality issue. ``segment_index`` is the primary segment the
    finding is about; ``related_segment_index`` is set for pairwise checks
    (e.g. a gap finding names both the segment before and after the gap)."""

    segment_index: int | None
    start: float
    end: float
    severity: Severity
    code: str
    message: str
    related_segment_index: int | None = None


@dataclass(frozen=True)
class AlignmentQualityReport:
    """The full result of :func:`evaluate_alignment_quality`."""

    findings: tuple[Finding, ...] = field(default_factory=tuple)
    segment_count: int = 0
    """Spine segments -- the timeline. Counterpoint is counted separately."""
    counterpoint_segment_count: int = 0
    """Concurrent segments evaluated (issue #33). ``0`` for every level-1 and
    level-2 file, which is every file written before counterpoint existed."""

    @property
    def is_clean(self) -> bool:
        return not self.findings

    @property
    def max_severity(self) -> Severity | None:
        return max((f.severity for f in self.findings), default=None)

    def at_least(self, severity: Severity) -> tuple[Finding, ...]:
        """Findings at or above ``severity``."""
        return tuple(f for f in self.findings if f.severity >= severity)

    def counts_by_severity(self) -> dict[Severity, int]:
        counts = dict.fromkeys(Severity, 0)
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts


class AlignmentQualityError(RuntimeError):
    """Raised by :func:`raise_if_blocking` when strict mode is on and the
    report has a finding at/above the blocking severity threshold. Carries
    the full report so a caller several layers up (cli.py) can log/print
    details before choosing how to exit -- this module never exits the
    process itself."""

    def __init__(self, report: AlignmentQualityReport, *, threshold: Severity) -> None:
        self.report = report
        self.threshold = threshold
        blocking = report.at_least(threshold)
        super().__init__(
            f"Alignment quality check failed: {len(blocking)} finding(s) at or above "
            f"{threshold.name} severity (of {len(report.findings)} total across "
            f"{report.segment_count} segment(s))"
        )


# --------------------------------------------------------------------------- #
# The pure evaluator.
# --------------------------------------------------------------------------- #


def evaluate_alignment_quality(
    result: AlignmentResult,
    *,
    low_confidence_words: Sequence[LowConfidenceWord] = (),
    near_zero_duration_s: float = NEAR_ZERO_DURATION_S,
    max_words_per_second: float = MAX_WORDS_PER_SECOND,
    min_words_per_second: float = MIN_WORDS_PER_SECOND,
    intra_line_gap_s: float = INTRA_LINE_GAP_S,
    isolated_segment_gap_s: float = ISOLATED_SEGMENT_GAP_S,
    low_word_confidence: float = LOW_WORD_CONFIDENCE,
    critical_word_confidence: float = CRITICAL_WORD_CONFIDENCE,
    track_confidence_collapse_fraction: float = TRACK_CONFIDENCE_COLLAPSE_FRACTION,
    audio_path: str | Path | None = None,
    consonant_band_hz: float = CONSONANT_BAND_HZ,
    vocal_energy_ratio_threshold: float = VOCAL_ENERGY_RATIO_THRESHOLD,
    vocal_energy_min_duration_s: float = MIN_SEGMENT_DURATION_FOR_VOCAL_ENERGY_S,
    vocal_energy_min_baseline_segments: int = VOCAL_ENERGY_BASELINE_MIN_SEGMENTS,
    ffmpeg_runner: FfmpegRunner | None = None,
) -> AlignmentQualityReport:
    """Pure function *by default*: no audio, no model, no I/O. Runs every
    check below over ``result.segments`` (plus the optional
    ``low_confidence_words`` side channel -- see the module docstring's
    "Per-word confidence" section) and returns a structured
    :class:`AlignmentQualityReport`.

    The one exception is the vocal-energy check (issue #71): it is a no-op
    (and touches no audio, no ffmpeg, nothing) unless ``audio_path`` is
    given, which is why that parameter defaults to ``None`` -- every
    existing caller that never passes it keeps the pure-function contract
    exactly as before. See the module docstring's "Vocal-energy check"
    section for what it does and why it needs to be the one exception.

    Checks:
      - zero-length / near-zero-length segments
      - implausible words-per-second, both too high and too low
      - out-of-order or overlapping segments
      - a large gap within one lyric line (heuristic; see module docstring)
      - an isolated segment separated from the rest of the vocal by a very
        large gap (heuristic; see module docstring)
      - low-confidence words, if ``low_confidence_words`` is supplied
      - counterpoint segments, if the result carries any (issue #33) --
        on their own terms, see the module docstring
      - no vocal energy where a segment was placed, if ``audio_path`` is
        supplied (issue #71) -- see the module docstring
    """
    segments = result.segments
    # Duck-typed rather than imported: this module must not depend on the
    # alignment stage (which imports *this* one). A plain AlignmentResult
    # simply has no concurrent segments, which is the pre-#33 behaviour.
    concurrent: tuple[AlignedSegment, ...] = tuple(getattr(result, "concurrent_segments", ()))
    findings: list[Finding] = []

    for segment in segments:
        findings.extend(
            _check_duration_and_rate(
                segment, near_zero_duration_s, max_words_per_second, min_words_per_second
            )
        )

    for prev, cur in zip(segments, segments[1:], strict=False):
        findings.extend(_check_order_and_gap(prev, cur, intra_line_gap_s))

    findings.extend(_check_isolated_segments(segments, isolated_segment_gap_s))

    findings.extend(
        _check_low_confidence_words(
            low_confidence_words, low_word_confidence, critical_word_confidence
        )
    )
    findings.extend(
        _check_track_confidence(
            low_confidence_words, critical_word_confidence, track_confidence_collapse_fraction
        )
    )

    findings.extend(
        _check_counterpoint_segments(
            concurrent, near_zero_duration_s, max_words_per_second, min_words_per_second
        )
    )

    findings.extend(
        _check_vocal_energy(
            segments,
            audio_path,
            consonant_band_hz=consonant_band_hz,
            ratio_threshold=vocal_energy_ratio_threshold,
            min_duration_s=vocal_energy_min_duration_s,
            min_baseline_segments=vocal_energy_min_baseline_segments,
            runner=ffmpeg_runner,
        )
    )

    findings.sort(key=lambda f: (f.segment_index if f.segment_index is not None else -1, f.code))
    return AlignmentQualityReport(
        findings=tuple(findings),
        segment_count=len(segments),
        counterpoint_segment_count=len(concurrent),
    )


def _check_counterpoint_segments(
    concurrent: Sequence[AlignedSegment],
    near_zero_duration_s: float,
    max_words_per_second: float,
    min_words_per_second: float,
) -> list[Finding]:
    """Rate and span sanity for derived counterpoint segments (issue #33).

    Capped at WARNING, with counterpoint-specific codes, and never compared
    against the spine -- see the module docstring for why both of those are
    structural rather than a matter of taste. What this *can* catch is real
    and belongs to the lyrics file: a stream with more words than its spine
    span can plausibly hold means the block was mis-authored, and that is
    worth reading in the log before hours of GPU time.
    """
    findings: list[Finding] = []
    for segment in concurrent:
        duration = segment.end - segment.start
        if duration <= near_zero_duration_s:
            findings.append(
                Finding(
                    segment_index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    severity=Severity.WARNING,
                    code=FINDING_COUNTERPOINT_SPAN,
                    message=(
                        f"counterpoint segment {segment.index} ({segment.start:.3f}s -> "
                        f"{segment.end:.3f}s) is only {duration * 1000:.0f}ms long -- its "
                        "spine span is too short to distribute this stream across"
                    ),
                )
            )
            continue

        word_count = len(segment.words) if segment.words else len(segment.text.split())
        if word_count == 0:
            continue
        words_per_second = word_count / duration
        if not (min_words_per_second <= words_per_second <= max_words_per_second):
            findings.append(
                Finding(
                    segment_index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    severity=Severity.WARNING,
                    code=FINDING_COUNTERPOINT_RATE,
                    message=(
                        f"counterpoint segment {segment.index} ({segment.start:.3f}s -> "
                        f"{segment.end:.3f}s) places {word_count} word(s) in "
                        f"{duration:.3f}s ({words_per_second:.1f} words/sec, plausible "
                        f"range {min_words_per_second}-{max_words_per_second}). These "
                        "timings are derived from the spine's span, not measured, so "
                        "this points at the lyrics file: the concurrent stream and the "
                        "spine block it sits under are probably mismatched in length"
                    ),
                )
            )
    return findings


def _check_duration_and_rate(
    segment: AlignedSegment,
    near_zero_duration_s: float,
    max_words_per_second: float,
    min_words_per_second: float,
) -> list[Finding]:
    findings: list[Finding] = []
    duration = segment.end - segment.start

    if duration < 0:
        findings.append(
            Finding(
                segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                severity=Severity.CRITICAL,
                code=FINDING_NEGATIVE_DURATION,
                message=(
                    f"segment {segment.index} ends ({segment.end:.3f}s) before it "
                    f"starts ({segment.start:.3f}s)"
                ),
            )
        )
        return findings

    if duration <= near_zero_duration_s:
        findings.append(
            Finding(
                segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                severity=Severity.CRITICAL,
                code=FINDING_ZERO_LENGTH,
                message=(
                    f"segment {segment.index} ({segment.start:.3f}s -> {segment.end:.3f}s) "
                    f"is only {duration * 1000:.0f}ms long -- too short to contain real audio"
                ),
            )
        )
        return findings  # words-per-second is meaningless at this duration

    word_count = len(segment.words) if segment.words else len(segment.text.split())
    if word_count == 0:
        return findings

    words_per_second = word_count / duration
    if words_per_second > max_words_per_second:
        findings.append(
            Finding(
                segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                severity=Severity.CRITICAL,
                code=FINDING_WPS_HIGH,
                message=(
                    f"segment {segment.index} ({segment.start:.3f}s -> {segment.end:.3f}s) places "
                    f"{word_count} word(s) in {duration:.3f}s ({words_per_second:.1f} words/sec, "
                    f"max plausible {max_words_per_second})"
                ),
            )
        )
    elif words_per_second < min_words_per_second:
        findings.append(
            Finding(
                segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                severity=Severity.WARNING,
                code=FINDING_WPS_LOW,
                message=(
                    f"segment {segment.index} ({segment.start:.3f}s -> {segment.end:.3f}s) spends "
                    f"{duration:.3f}s on {word_count} word(s) ({words_per_second:.2f} words/sec, "
                    f"min plausible {min_words_per_second})"
                ),
            )
        )
    return findings


def _normalize_last_word(text: str) -> str:
    words = text.strip().split()
    if not words:
        return ""
    return words[-1].strip(".,!?;:\"'").lower()


def _check_order_and_gap(
    prev: AlignedSegment, cur: AlignedSegment, intra_line_gap_s: float
) -> list[Finding]:
    findings: list[Finding] = []

    if cur.start < prev.start:
        findings.append(
            Finding(
                segment_index=cur.index,
                related_segment_index=prev.index,
                start=cur.start,
                end=cur.end,
                severity=Severity.CRITICAL,
                code=FINDING_OUT_OF_ORDER,
                message=(
                    f"segment {cur.index} starts at {cur.start:.3f}s, before segment "
                    f"{prev.index} which starts at {prev.start:.3f}s"
                ),
            )
        )

    if cur.start < prev.end:
        findings.append(
            Finding(
                segment_index=cur.index,
                related_segment_index=prev.index,
                start=cur.start,
                end=cur.end,
                severity=Severity.CRITICAL,
                code=FINDING_OVERLAP,
                message=(
                    f"segment {cur.index} starts at {cur.start:.3f}s, before segment "
                    f"{prev.index} ends at {prev.end:.3f}s (overlap {prev.end - cur.start:.3f}s)"
                ),
            )
        )
        return findings  # the gap below is negative/meaningless while overlapping

    gap = cur.start - prev.end
    if gap > intra_line_gap_s:
        last_word = _normalize_last_word(prev.text)
        if last_word in CLAUSE_CONTINUATION_WORDS:
            findings.append(
                Finding(
                    segment_index=cur.index,
                    related_segment_index=prev.index,
                    start=cur.start,
                    end=cur.end,
                    severity=Severity.WARNING,
                    code=FINDING_SPLIT_LINE,
                    message=(
                        f"segment {prev.index} ends on '{last_word}' and segment {cur.index} "
                        f"begins {gap:.1f}s later -- looks like one lyric line split by a "
                        "mis-timed jump, not two distinct lines"
                    ),
                )
            )
    return findings


def _check_isolated_segments(
    segments: tuple[AlignedSegment, ...], isolated_segment_gap_s: float
) -> list[Finding]:
    """See the module docstring's "Known heuristic limitations" note: the
    first segment is never evaluated (a long intro is normal); the last
    segment is evaluated asymmetrically on gap-before alone."""
    findings: list[Finding] = []
    n = len(segments)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        gap_before = segment.start - segments[i - 1].end

        if i < n - 1:
            gap_after = segments[i + 1].start - segment.end
            isolated = gap_before > isolated_segment_gap_s and gap_after > isolated_segment_gap_s
        else:
            isolated = gap_before > isolated_segment_gap_s

        if isolated:
            findings.append(
                Finding(
                    segment_index=segment.index,
                    start=segment.start,
                    end=segment.end,
                    severity=Severity.CRITICAL,
                    code=FINDING_ISOLATED,
                    message=(
                        f"segment {segment.index} ({segment.start:.3f}s -> {segment.end:.3f}s) is "
                        f"separated from the nearest prior vocal activity by a {gap_before:.1f}s "
                        "gap -- consistent with a hallucinated lyric placed in an instrumental "
                        "passage"
                    ),
                )
            )
    return findings


def _check_low_confidence_words(
    low_confidence_words: Sequence[LowConfidenceWord],
    low_word_confidence: float,
    critical_word_confidence: float,
) -> list[Finding]:
    """One finding per *segment*, never one per word, and never CRITICAL on
    its own.

    Measured on the real track (2026-08-08, corrected transcript): stable-ts
    reported a median word probability of 0.76 -- but 24% of words sat below
    0.10 and 33% below 0.35. Forced alignment *places* a word whether or not
    the acoustic model is confident about it, and sung vocals over a full mix
    are exactly where that confidence collapses, so a low probability is
    weak evidence of a misplacement rather than proof of one. Emitting one
    CRITICAL per low word did two harmful things: it put hundreds of ERROR
    lines in front of the four findings that actually located a defect, and
    it would have made --strict-alignment refuse every real song.

    So: per-segment aggregation, capped at WARNING. The whole-track picture
    is the honest place for this signal, and it lives in
    :func:`_check_track_confidence` below.
    """
    by_segment: dict[int, list[LowConfidenceWord]] = {}
    for w in low_confidence_words:
        if w.probability < low_word_confidence:
            by_segment.setdefault(w.segment_index, []).append(w)

    findings: list[Finding] = []
    for segment_index, words in sorted(by_segment.items()):
        worst = min(words, key=lambda w: w.probability)
        findings.append(
            Finding(
                segment_index=segment_index,
                start=min(w.start for w in words),
                end=max(w.end for w in words),
                severity=Severity.WARNING,
                code=FINDING_LOW_CONFIDENCE,
                message=(
                    f"segment {segment_index}: {len(words)} word(s) below "
                    f"{low_word_confidence:.2f} alignment confidence "
                    f"(worst: '{worst.word.strip()}' at {worst.probability:.2f}). "
                    "Forced alignment places words it has weak evidence for, so this "
                    "is a hint about where to look, not proof of a misplacement"
                ),
            )
        )
    return findings


def _check_track_confidence(
    low_confidence_words: Sequence[LowConfidenceWord],
    critical_word_confidence: float,
    collapse_fraction: float,
) -> list[Finding]:
    """The aggregate that per-word findings cannot express: has confidence
    collapsed across the *whole* track?"""
    total = len(low_confidence_words)
    # A caller may hand over only the words it already suspects, which would
    # read as 100% starved on a perfectly good track. Require a real sample.
    if total < MIN_WORDS_FOR_COLLAPSE_CHECK:
        return []
    starved = sum(1 for w in low_confidence_words if w.probability < critical_word_confidence)
    fraction = starved / total
    if fraction < collapse_fraction:
        return []
    return [
        Finding(
            segment_index=None,
            start=0.0,
            end=0.0,
            severity=Severity.CRITICAL,
            code=FINDING_CONFIDENCE_COLLAPSE,
            message=(
                f"{starved}/{total} words ({fraction:.0%}) are below "
                f"{critical_word_confidence:.2f} alignment confidence -- the aligner had "
                "essentially no acoustic support across the track. Check that the audio, "
                "the lyrics and the language all match each other before rendering"
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# Vocal-energy check (issue #71): the one check in this module that looks at
# the master audio itself. Everything above reasons about timing and text;
# this asks the question none of them can: is there a voice where a segment
# claims a lyric was sung?
#
# Method (measured, not assumed -- see VOCAL_ENERGY_RATIO_THRESHOLD's
# comment for the real numbers this is calibrated against):
#   1. decode the master once to mono 16kHz (ffmpeg is already a hard
#      project dependency; no new one added);
#   2. for each placed segment long enough to trust
#      (MIN_SEGMENT_DURATION_FOR_VOCAL_ENERGY_S), measure the share of its
#      total spectral energy that sits above CONSONANT_BAND_HZ -- via
#      ffmpeg's own `highpass` + `astats` filters (RMS power ratio), not a
#      new FFT dependency;
#   3. compare each segment's share against the *median* share of this
#      track's other qualifying segments (its own self-calibrated
#      baseline -- see VOCAL_ENERGY_RATIO_THRESHOLD's docstring for why
#      median, not mean or max);
#   4. flag a segment sitting at a small fraction of that baseline.
#
# Impure by necessity (needs ffmpeg + the audio file), which is why it is
# the one exception to evaluate_alignment_quality's "pure function" contract
# -- see that function's docstring. Every failure mode degrades to "skip
# this check" with a logged reason, never a crash: a quality check must
# never be the thing that takes an alignment run down (global standard).
# --------------------------------------------------------------------------- #

FfmpegRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]

_RMS_LEVEL_DB_RE = re.compile(r"RMS level dB:\s*(-?[\d.]+|-inf)")


def _default_ffmpeg_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real ffmpeg invocation. Never used by unit tests -- injected out via
    ``ffmpeg_runner=``, same seam as assembly.py's ``SubprocessRunner``."""
    return subprocess.run(list(args), capture_output=True, check=False)


def _ffmpeg_stderr_text(proc: subprocess.CompletedProcess) -> str:
    stderr = proc.stderr if proc.stderr is not None else b""
    return stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")


def _rms_level_db(
    audio_path: Path,
    start: float,
    duration: float,
    *,
    highpass_hz: float | None,
    runner: FfmpegRunner,
) -> float | None:
    """Runs ffmpeg's ``astats`` filter over ``[start, start + duration)`` of
    ``audio_path``, optionally highpass-filtered at ``highpass_hz`` first,
    and returns the reported Overall RMS level in dB. ``None`` on any
    failure -- non-zero exit, an OSError launching the process, unparseable
    output, or pure digital silence (``-inf``, against/of which a ratio is
    meaningless) -- callers treat that as "this window can't be measured",
    never as evidence in either direction."""
    filt = "aformat=channel_layouts=mono"
    if highpass_hz is not None:
        filt += f",highpass=f={highpass_hz}"
    filt += ",astats=metadata=0:length=0"
    args = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(audio_path),
        "-af",
        filt,
        "-f",
        "null",
        "-",
    ]
    try:
        proc = runner(args)
    except OSError:
        logger.warning(
            "alignment_quality: ffmpeg invocation failed analyzing %s [%.3fs, %.3fs) -- "
            "skipping this window's vocal-energy measurement",
            audio_path,
            start,
            start + duration,
            exc_info=True,
        )
        return None
    if proc.returncode != 0:
        logger.warning(
            "alignment_quality: ffmpeg exited %s analyzing %s [%.3fs, %.3fs) -- skipping "
            "this window's vocal-energy measurement; stderr=%s",
            proc.returncode,
            audio_path,
            start,
            start + duration,
            _ffmpeg_stderr_text(proc)[-500:],
        )
        return None
    match = _RMS_LEVEL_DB_RE.search(_ffmpeg_stderr_text(proc))
    if not match or match.group(1) == "-inf":
        return None
    return float(match.group(1))


def _consonant_band_share(
    audio_path: Path,
    start: float,
    end: float,
    *,
    consonant_band_hz: float,
    runner: FfmpegRunner,
) -> float | None:
    """This window's share of total spectral energy sitting above
    ``consonant_band_hz``, as a power ratio (``10 ** ((hf_db - total_db) /
    10)``). ``None`` if either the total or the highpassed measurement
    failed."""
    duration = end - start
    if duration <= 0:
        return None
    total_db = _rms_level_db(audio_path, start, duration, highpass_hz=None, runner=runner)
    if total_db is None:
        return None
    hf_db = _rms_level_db(
        audio_path, start, duration, highpass_hz=consonant_band_hz, runner=runner
    )
    if hf_db is None:
        return None
    return 10 ** ((hf_db - total_db) / 10)


def _check_vocal_energy(
    segments: tuple[AlignedSegment, ...],
    audio_path: str | Path | None,
    *,
    consonant_band_hz: float,
    ratio_threshold: float,
    min_duration_s: float,
    min_baseline_segments: int,
    runner: FfmpegRunner | None,
) -> list[Finding]:
    """See the section docstring above. No-op (no I/O at all) when
    ``audio_path`` is ``None`` -- this is what keeps
    ``evaluate_alignment_quality`` a pure function for every caller that
    doesn't pass one."""
    if audio_path is None or not segments:
        return []

    active_runner: FfmpegRunner = runner if runner is not None else _default_ffmpeg_runner
    audio_path = Path(audio_path)

    if shutil.which("ffmpeg") is None:
        logger.warning(
            "alignment_quality: ffmpeg not found on PATH -- skipping the vocal-energy "
            "check (issue #71); every other alignment-quality check still ran"
        )
        return []
    if not audio_path.exists():
        logger.warning(
            "alignment_quality: master audio %s not found -- skipping the vocal-energy "
            "check (issue #71); every other alignment-quality check still ran",
            audio_path,
        )
        return []

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        decode_args = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(tmp_path),
        ]
        try:
            decode_proc = active_runner(decode_args)
        except OSError:
            logger.warning(
                "alignment_quality: could not launch ffmpeg to decode %s to mono 16kHz -- "
                "skipping the vocal-energy check (issue #71)",
                audio_path,
                exc_info=True,
            )
            return []
        if decode_proc.returncode != 0:
            logger.warning(
                "alignment_quality: ffmpeg exited %s decoding %s to mono 16kHz -- skipping "
                "the vocal-energy check (issue #71); stderr=%s",
                decode_proc.returncode,
                audio_path,
                _ffmpeg_stderr_text(decode_proc)[-500:],
            )
            return []

        shares: dict[int, float] = {}
        for segment in segments:
            if segment.end - segment.start < min_duration_s:
                continue  # too short for a trustworthy spectral estimate either way
            share = _consonant_band_share(
                tmp_path,
                segment.start,
                segment.end,
                consonant_band_hz=consonant_band_hz,
                runner=active_runner,
            )
            if share is not None:
                shares[segment.index] = share

        findings: list[Finding] = []
        for segment in segments:
            if segment.end - segment.start < min_duration_s:
                continue
            this_share = shares.get(segment.index)
            if this_share is None:
                continue  # this window's own measurement failed -- already logged
            others = [share for idx, share in shares.items() if idx != segment.index]
            if len(others) < min_baseline_segments:
                continue  # not enough of a baseline on this track to calibrate against
            baseline = statistics.median(others)
            if baseline <= 0:
                continue
            ratio = this_share / baseline
            if ratio < ratio_threshold:
                findings.append(
                    Finding(
                        segment_index=segment.index,
                        start=segment.start,
                        end=segment.end,
                        severity=Severity.CRITICAL,
                        code=FINDING_NO_VOCAL_ENERGY,
                        message=(
                            f"segment {segment.index} ({segment.start:.3f}s -> "
                            f"{segment.end:.3f}s) has almost no energy above "
                            f"{consonant_band_hz:.0f}Hz -- consonant-band share "
                            f"{this_share:.4f} against this track's own median of "
                            f"{baseline:.4f} across {len(others)} other placed "
                            f"segment(s) ({ratio:.0%} of it, threshold "
                            f"{ratio_threshold:.0%}). Consonants live above this band; a "
                            "lyric placed where essentially none survive is the acoustic "
                            "signature of a hallucinated placement, not a quiet one "
                            "(issue #71)"
                        ),
                    )
                )
        return findings
    except Exception:
        logger.exception(
            "alignment_quality: vocal-energy check failed unexpectedly for %s -- skipping "
            "it; every other alignment-quality check still ran",
            audio_path,
        )
        return []
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "alignment_quality: could not remove temp file %s", tmp_path, exc_info=True
                )


# --------------------------------------------------------------------------- #
# Logging + optional refusal.
# --------------------------------------------------------------------------- #


def format_summary(report: AlignmentQualityReport) -> str:
    """One-line human-readable summary, suitable for an INFO log line."""
    counts = report.counts_by_severity()
    counterpoint = (
        f" (+{report.counterpoint_segment_count} counterpoint segment(s))"
        if report.counterpoint_segment_count
        else ""
    )
    return (
        f"Alignment quality: {report.segment_count} segment(s){counterpoint}, "
        f"{counts[Severity.CRITICAL]} critical, {counts[Severity.WARNING]} warning "
        f"finding(s)"
    )


def log_report(report: AlignmentQualityReport, *, context: str = "") -> None:
    """Logs :func:`format_summary` at INFO unconditionally (this is what
    guarantees "render.log contains an alignment-quality summary on every
    run"), then one WARNING/ERROR line per finding at its severity. Never
    raises -- pair with :func:`raise_if_blocking` for refusal."""
    prefix = f"{context}: " if context else ""
    logger.info("%s%s", prefix, format_summary(report))
    for finding in report.findings:
        level = logging.ERROR if finding.severity is Severity.CRITICAL else logging.WARNING
        logger.log(level, "%s[%s] %s", prefix, finding.code, finding.message)


def raise_if_blocking(
    report: AlignmentQualityReport, *, strict: bool, threshold: Severity = Severity.CRITICAL
) -> None:
    """No-op unless ``strict`` is True and ``report`` has a finding at/above
    ``threshold``. Intended use: ``raise_if_blocking(report,
    strict=cfg.strict_alignment)``. Raises :class:`AlignmentQualityError`
    (never calls ``sys.exit`` -- that decision belongs to the caller, e.g.
    cli.py)."""
    if not strict:
        return
    blocking = report.at_least(threshold)
    if blocking:
        logger.error("Refusing to continue: %s", format_summary(report))
        raise AlignmentQualityError(report, threshold=threshold)
