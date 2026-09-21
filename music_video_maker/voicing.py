"""Voiced-vs-unvoiced measurement over a decoded mono PCM track (issue #96).

Why this exists
---------------
``alignment_quality``'s #71 check asks "is there a voice where you placed this
lyric" and answers it with a **consonant-band share**: the fraction of a
window's energy sitting above 3.4kHz. That statistic separates a lyric placed
over a *tonal fadeout* from a real one, and it is what caught "Deathless"'s
8:19 phantom. It cannot separate a lyric placed over a **plucked guitar
note** from a real one, because a plucked string's harmonics put real energy
in the same band -- FINDINGS F43 measured exactly that: the phantom
``'mushrooms grow.'`` at 228.590-230.150s was not even in the bottom 10 of
57 placed segments, and ``'you.'`` at 378.470-379.430s had the *lowest*
consonant-band share on the whole track and still did not clear #71's
threshold.

Brightness is not voice. This module measures **periodicity** instead: a
normalized cross-correlation function (NCCF) pitch/voicing detector of the
RAPT family, in pure stdlib Python over the mono 16kHz decode
``alignment_quality`` already makes for #71/#80. No new dependency (numpy is
deliberately *not* a core dependency of this project -- see pyproject.toml,
where the whole ML stack is an optional ``[align]`` extra), no second decode,
and no model.

What this CAN distinguish
-------------------------
- **A periodic source from an aperiodic one.** Drums, cymbals, hiss, room
  tone, applause, digital near-silence and broadband noise all score a low
  NCCF peak; anything with a stable pitch period scores a high one. This is
  the axis a consonant-band share is blind to, and it is the one the two
  F43 windows that are *not* a guitar note sit on.
- **How much of a span is audibly anything at all**, because every frame is
  level-gated before it is judged (see ``measure_span``'s ``level_floor_dbfs``)
  -- so ``voiced_fraction`` reads "of the frames in this span that are
  audible, how many are periodic", not "how much of this span is loud".
- **Coarse F0, period jitter and pitch movement**, reported for every span so
  they can be *calibrated* rather than assumed (see below).

What this CANNOT distinguish
----------------------------
- **A sung vowel from a sustained pitched instrument.** A plucked string, a
  bowed note, a synth pad and a vowel are all periodic; NCCF scores them all
  high. Periodicity alone therefore does **not** catch F43's headline case
  (``'mushrooms grow.'`` over a guitar note) and this module does not claim
  to. What it does is put the two statistics that *might* separate them --
  ``f0_jitter_pct`` (a voice has vibrato and micro-instability; a plucked
  string is metronomic) and ``f0_span_semitones`` (a sung phrase moves
  between notes; one plucked note does not) -- on the table with real
  numbers beside them, so the question can be settled by measurement on a
  track with a known answer instead of by taste. Until that happens they are
  **reported and never thresholded**: this project has shipped a keyword lint
  scored on a partial corpus twice (#60, #76) and had to retire it both
  times.
- **Which voice.** Nothing here attributes a speaker; see ``lyrics.py`` for
  the only thing in this pipeline that does (an authored ``[Name: Role]``
  tag).
- **A voice under a loud instrumental** in general. The NCCF sees whichever
  periodic source dominates the analysis band; a vocal buried under a
  sustained synth reads as the synth's period. This is why the finding built
  on it is a WARNING and self-calibrated against the track's own median,
  never an absolute score.

Cost
----
Deliberately cheap enough to run on every alignment. The decode is reused;
the analysis downsamples to 8kHz, correlates a 32ms window over 99 lags
(70-500Hz) every 20ms, and caps the frames per span
(:data:`MAX_FRAMES_PER_SPAN`). Measured on this repo's own benchmark: 0.28ms
per frame, so an 8-minute track with ~190s of placed segments costs well
under 3 seconds of CPU -- the same order as the ffmpeg calls #71 already
makes.

Octave errors are expected and harmless here: a 600Hz soprano note has a
strong secondary peak at twice its period, inside the search range, so
``voiced_fraction`` is unaffected while ``median_f0_hz`` may read an octave
low. Never treat ``median_f0_hz`` as a pitch transcription.
"""

from __future__ import annotations

import array
import logging
import math
import wave
from dataclasses import dataclass
from operator import mul
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Signal-processing constants. These describe the DETECTOR, not the policy
# built on it -- every threshold a *finding* is made from lives in
# alignment_quality.py beside the other named-and-justified constants.
# --------------------------------------------------------------------------- #

ANALYSIS_RATE_HZ = 8000
"""Rate the analysis runs at. The decode #71 already makes is mono 16kHz;
halving it halves the correlation cost and loses nothing, because everything
this module looks for lives below 500Hz plus a handful of harmonics. A
3-tap [0.25, 0.5, 0.25] lowpass runs before decimation so the drums and
cymbals above 4kHz do not alias down into the search band."""

MIN_F0_HZ = 70.0
MAX_F0_HZ = 500.0
"""Search range for the pitch period. Wide enough for a bass voice at the
bottom and a soprano at the top -- and a voice *above* 500Hz is still
detected, at a sub-harmonic lag (see the module docstring on octave errors),
which is what this module needs since it is measuring voicing, not pitch."""

CORRELATION_WINDOW_S = 0.032
"""Length of the window correlated against itself. 32ms is ~2.2 periods at
70Hz and ~16 at 500Hz -- the usual compromise: long enough for the lowest
period to be visible, short enough that a syllable boundary does not smear
two different periods together."""

HOP_S = 0.020
"""Analysis hop. 50 frames/second resolves a syllable (~150-250ms) into
several frames, which is what ``voiced_fraction`` needs in order to mean
anything on a span containing both a vowel and a fricative."""

VOICED_FRAME_NCCF = 0.45
"""NCCF peak at or above which one frame is called voiced.

Not a tuned constant: it is the middle of the range voicing detectors of this
family conventionally use (~0.3-0.5), chosen before any of this project's
audio was measured, and it is deliberately *not* the number the finding is
thresholded on -- ``alignment_quality`` compares a span's voiced fraction
against the same track's own median, so a track whose mix pushes every frame
up or down moves its own baseline with it. If a calibration run ever shows
this number matters more than the ratio does, that is a finding worth
recording, not a knob to turn quietly."""

MAX_FRAMES_PER_SPAN = 400
"""Upper bound on the frames analysed for one span (8 seconds at
:data:`HOP_S`). A longer span is strided evenly rather than truncated, so the
statistics still describe the whole span. Exists to bound the cost of a
pathological alignment that places one segment across half the track, not
because any real segment reaches it."""

_MIN_FRAME_SAMPLES = 2


class VoicingUnavailable(RuntimeError):
    """The audio could not be read in the form this module needs.

    Raised (and caught by callers, which degrade to "skip this check" with a
    logged reason) rather than returned as a sentinel, because the two
    situations it covers -- an unreadable/absent decode and an unexpected
    sample format -- are both "no measurement", and a quality check must
    never be the thing that takes an alignment run down."""


@dataclass(frozen=True)
class PcmAudio:
    """Mono 16-bit PCM held in memory, with its sample rate.

    An 8-minute 16kHz mono track is ~15MB as an ``array('h')``; holding it
    once beats re-seeking the file per segment, and it is thrown away with
    the caller's temp file."""

    samples: array.array
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class SegmentVoicing:
    """Voicing statistics for one span of audio.

    ``voiced_fraction`` is the one statistic a finding is built on today.
    Everything else is reported -- in the calibration table and in the
    finding's own message -- precisely so the next revision starts from data;
    see the module docstring on what is deliberately not thresholded."""

    start: float
    end: float
    frames_total: int
    """Frames the span was cut into before the level gate."""
    frames_measured: int
    """Frames that survived the level gate and were actually judged."""
    voiced_fraction: float
    """Share of ``frames_measured`` whose NCCF peak reached
    :data:`VOICED_FRAME_NCCF`. ``0.0`` when nothing was measured."""
    median_nccf: float
    """Median NCCF peak across the measured frames -- the threshold-free view
    of the same evidence ``voiced_fraction`` bins."""
    hnr_db: float
    """Harmonic-to-noise ratio implied by ``median_nccf``
    (``10*log10(r/(1-r))``, Boersma's relation), clamped. Reported because
    issue #96 names HNR explicitly; it is a monotone restatement of
    ``median_nccf``, not independent evidence."""
    level_dbfs: float
    """Full-span RMS, dB relative to full scale."""
    median_f0_hz: float | None
    """Median F0 over the voiced frames, octave-ambiguous (see the module
    docstring). ``None`` when no frame was voiced."""
    f0_jitter_pct: float | None
    """Median frame-to-frame relative change in the pitch *period*, over
    consecutive voiced frames, as a percentage. A voice carries vibrato and
    micro-instability; a plucked or bowed string is metronomic. A hypothesis
    with no corpus behind it yet -- reported, never thresholded."""
    f0_span_semitones: float | None
    """Spread between the 10th and 90th percentile voiced F0, in semitones.
    A sung phrase moves between notes; one plucked note does not. Same status
    as ``f0_jitter_pct``: reported, never thresholded.

    Read it with the octave ambiguity in mind: a source near the top of the
    search range flips between its true period and a sub-harmonic from frame
    to frame, which reads as a ~12-semitone spread that is an artefact, not a
    melody. This module's own synthetic probe shows it (a steady 400Hz tone
    reports 12.6 semitones), which is exactly why the statistic is reported
    for calibration rather than thresholded on."""


def load_mono_pcm16(path: str | Path) -> PcmAudio:
    """Read a mono 16-bit PCM WAV wholly into memory.

    This is exactly the shape ``alignment_quality``'s own
    ``ffmpeg -ac 1 -ar 16000`` decode produces, which is the only file this
    is ever pointed at in the render path. Anything else -- stereo, 8-bit,
    24-bit, a truncated or absent file -- raises :class:`VoicingUnavailable`
    rather than being silently reinterpreted.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error, EOFError) as exc:
        raise VoicingUnavailable(f"could not read {path} as a WAV: {exc}") from exc

    if channels != 1 or width != 2:
        raise VoicingUnavailable(
            f"{path} is {channels}-channel {width * 8}-bit; this measurement needs mono 16-bit "
            "PCM (the decode alignment_quality already makes)"
        )
    if rate <= 0:
        raise VoicingUnavailable(f"{path} reports a sample rate of {rate}")

    samples = array.array("h")
    usable = len(raw) - (len(raw) % 2)
    samples.frombytes(raw[:usable])
    return PcmAudio(samples=samples, sample_rate=rate)


def span_level_dbfs(audio: PcmAudio, start: float, end: float) -> float:
    """RMS of ``[start, end)`` in dB relative to full scale, or ``-inf`` for a
    span that is empty or digitally silent.

    Its own O(n) pass rather than a read of ffmpeg's ``astats`` output:
    callers need the *median across spans* before they can judge any one of
    them, and deriving that from the same samples the periodicity runs on
    keeps one definition of "how loud is this" instead of two that agree
    until one of them is changed."""
    rate = audio.sample_rate
    if rate <= 0 or end <= start:
        return -float("inf")
    first = max(0, int(start * rate))
    last = min(len(audio.samples), int(math.ceil(end * rate)))
    if last <= first:
        return -float("inf")
    window = audio.samples[first:last]
    energy = sum(float(v) * float(v) for v in window)
    if energy <= 0.0:
        return -float("inf")
    return 20.0 * math.log10(math.sqrt(energy / len(window)) / 32768.0)


def _decimate(samples: list[float], factor: int) -> list[float]:
    """Lowpass (3-tap [0.25, 0.5, 0.25]) then take every ``factor``-th sample.

    Crude on purpose: the filter only has to keep the drums and cymbals above
    the new Nyquist from folding down onto the pitch band, and a 3-tap kernel
    over ~15MB of samples is cheap enough to run inside an alignment."""
    if factor < 2:
        return samples
    n = len(samples)
    if n < 3:
        return samples[::factor]
    out: list[float] = []
    for i in range(0, n, factor):
        if i == 0 or i == n - 1:
            out.append(samples[i])
        else:
            out.append(0.25 * samples[i - 1] + 0.5 * samples[i] + 0.25 * samples[i + 1])
    return out


def _nccf_peak(frame: list[float], window: int, lag_min: int, lag_max: int) -> tuple[float, float]:
    """Peak normalized cross-correlation of ``frame`` against itself, and the
    (parabolically interpolated) lag it occurred at.

    ``frame`` must already be mean-removed and at least ``window + lag_max``
    samples long. Returns ``(0.0, 0.0)`` when the window carries no energy.

    Normalized by ``sqrt(e0 * e_lag)`` rather than by ``e0`` alone: that is
    what makes a decaying note (a pluck) score on its *shape* rather than
    being penalised for getting quieter across the window, which matters
    because a decaying note is one of the two things this measurement exists
    to reason about."""
    base = frame[:window]
    e0 = sum(v * v for v in base)
    if e0 <= 0.0:
        return 0.0, 0.0

    energy = e0
    scores: dict[int, float] = {}
    best = 0.0
    best_lag = 0
    for lag in range(1, lag_max + 1):
        leaving = frame[lag - 1]
        arriving = frame[lag + window - 1]
        energy += arriving * arriving - leaving * leaving
        if lag < lag_min:
            continue
        if energy <= 0.0:
            continue
        numerator = sum(map(mul, base, frame[lag : lag + window]))
        score = numerator / math.sqrt(e0 * energy)
        scores[lag] = score
        if score > best:
            best = score
            best_lag = lag

    if best_lag == 0:
        return 0.0, 0.0

    # Sub-sample peak location. Without it the period is quantized to whole
    # 8kHz samples, which is 2.5% at 200Hz -- larger than the jitter the
    # f0_jitter_pct statistic is trying to see, so the statistic would be
    # measuring the grid instead of the voice.
    left = scores.get(best_lag - 1)
    right = scores.get(best_lag + 1)
    lag = float(best_lag)
    if left is not None and right is not None:
        denominator = left - 2.0 * best + right
        if denominator < 0.0:
            offset = 0.5 * (left - right) / denominator
            if -1.0 < offset < 1.0:
                lag = best_lag + offset
    return best, lag


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def measure_span(
    audio: PcmAudio,
    start: float,
    end: float,
    *,
    level_floor_dbfs: float | None = None,
    voiced_frame_nccf: float = VOICED_FRAME_NCCF,
    max_frames: int = MAX_FRAMES_PER_SPAN,
) -> SegmentVoicing | None:
    """Measure ``[start, end)`` of ``audio``. ``None`` when the span is too
    short to frame at all (nothing is inferred from a span that could not be
    measured -- #93's rule that a zero from a detector is not evidence of
    absence unless something asked the second question).

    ``level_floor_dbfs`` gates individual frames: a frame quieter than this
    is dropped from both the numerator and the denominator of
    ``voiced_fraction``, so the statistic answers "of what is audible here,
    how much is periodic" rather than being diluted by the silence a segment
    boundary always carries at its edges. Callers pass a floor derived from
    the *track's own* level (see ``alignment_quality.VOICING_FRAME_LEVEL_DROP_DB``),
    which is #80's lesson: a ratio computed from two near-silent measurements
    means nothing in either direction.
    """
    rate = audio.sample_rate
    if rate <= 0 or end <= start:
        return None

    first = max(0, int(start * rate))
    last = min(len(audio.samples), int(math.ceil(end * rate)))
    if last - first < _MIN_FRAME_SAMPLES:
        return None

    raw = [float(v) for v in audio.samples[first:last]]
    span_energy = sum(v * v for v in raw)
    level_dbfs = (
        20.0 * math.log10(math.sqrt(span_energy / len(raw)) / 32768.0)
        if span_energy > 0.0
        else -float("inf")
    )

    factor = max(1, int(round(rate / ANALYSIS_RATE_HZ)))
    signal = _decimate(raw, factor)
    analysis_rate = rate / factor

    window = max(4, int(CORRELATION_WINDOW_S * analysis_rate))
    hop = max(1, int(HOP_S * analysis_rate))
    lag_min = max(1, int(analysis_rate / MAX_F0_HZ))
    lag_max = int(analysis_rate / MIN_F0_HZ)
    frame_len = window + lag_max + 1
    if len(signal) < frame_len:
        return None

    offsets = list(range(0, len(signal) - frame_len + 1, hop))
    if not offsets:
        return None
    if len(offsets) > max_frames:
        stride = len(offsets) / max_frames
        offsets = [offsets[int(i * stride)] for i in range(max_frames)]

    nccfs: list[float] = []
    periods: list[float | None] = []
    for offset in offsets:
        chunk = signal[offset : offset + frame_len]
        mean = sum(chunk) / len(chunk)
        centred = [v - mean for v in chunk]
        base_energy = sum(v * v for v in centred[:window])
        frame_dbfs = (
            20.0 * math.log10(math.sqrt(base_energy / window) / 32768.0)
            if base_energy > 0.0
            else -float("inf")
        )
        if level_floor_dbfs is not None and frame_dbfs < level_floor_dbfs:
            continue
        peak, lag = _nccf_peak(centred, window, lag_min, lag_max)
        nccfs.append(peak)
        periods.append(lag / analysis_rate if peak >= voiced_frame_nccf and lag > 0 else None)

    frames_total = len(offsets)
    frames_measured = len(nccfs)
    if frames_measured == 0:
        return SegmentVoicing(
            start=start,
            end=end,
            frames_total=frames_total,
            frames_measured=0,
            voiced_fraction=0.0,
            median_nccf=0.0,
            hnr_db=-float("inf"),
            level_dbfs=level_dbfs,
            median_f0_hz=None,
            f0_jitter_pct=None,
            f0_span_semitones=None,
        )

    voiced_periods = [p for p in periods if p is not None]
    voiced_fraction = len(voiced_periods) / frames_measured
    ordered = sorted(nccfs)
    median_nccf = _percentile(ordered, 0.5)
    clamped = min(max(median_nccf, 1e-6), 1.0 - 1e-6)
    hnr_db = 10.0 * math.log10(clamped / (1.0 - clamped))

    median_f0: float | None = None
    jitter: float | None = None
    f0_span: float | None = None
    if voiced_periods:
        sorted_f0 = sorted(1.0 / p for p in voiced_periods)
        median_f0 = _percentile(sorted_f0, 0.5)
        low = _percentile(sorted_f0, 0.1)
        high = _percentile(sorted_f0, 0.9)
        if low > 0.0 and high > 0.0:
            f0_span = 12.0 * math.log2(high / low)
        deltas = [
            abs(b - a) / ((a + b) / 2.0) * 100.0
            for a, b in zip(periods, periods[1:], strict=False)
            if a is not None and b is not None and a > 0.0 and b > 0.0
        ]
        if deltas:
            jitter = _percentile(sorted(deltas), 0.5)

    return SegmentVoicing(
        start=start,
        end=end,
        frames_total=frames_total,
        frames_measured=frames_measured,
        voiced_fraction=voiced_fraction,
        median_nccf=median_nccf,
        hnr_db=hnr_db,
        level_dbfs=level_dbfs,
        median_f0_hz=median_f0,
        f0_jitter_pct=jitter,
        f0_span_semitones=f0_span,
    )
