"""Is any chunk join landing inside a sung word? (pre-spend check)

A chunk's audio is cut at a join; the next chunk's audio starts there. When
the join falls *inside* a word, one chunk ends on the first half of it and the
next begins on the rest. The lip-sync model does not mouth half a word at a
clip's edge, so the singer stands silent on one half and the cut lands on the
other. ``phrase_aware_slicing`` and the mid-phrase warnings trust the forced
alignment's word times, so they cannot see this when that alignment is itself
off (forced ``align`` can return zero-length words for exactly the lines that
matter).

This check does not trust it. Before any spend it **independently transcribes
a short window of the vocal audio around every join** (+-:data:`WINDOW_SECONDS`
by default) with word timestamps -- stable-ts ``transcribe(...,
word_timestamps=True)``, which behaved better here than ``align`` -- and
reports every join that falls strictly inside a heard word, with the time, the
word, and the nearest pause as a suggestion.

**Reports and refuses; never moves a join.** A join is not a free parameter:
it is the running sum of grid-valid frame counts (5 + 17k), inside the
duration window, possibly pinned by a shot plan's ``length_seconds``, and
moving one changes two chunks at once. Choosing a new tiling is the slicer's
job (``phrase_aware_slicing``); this module only supplies evidence and a
suggested time. ``strict_alignment`` turns any finding into a refusal.

**Pure core, lazy ASR.** :func:`check_joins` takes a ``transcribe_window``
callable, so tests inject a fake. Only :func:`whisper_window_transcriber`
imports ``stable_whisper`` (the ``[align]`` extra) and decodes audio, on CPU.
ASR is a witness, not an oracle: a word it mishears or drops is invisible
here, and one clipped by a window's own edge is ignored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import AudioChunk

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 1.5
"""Each side of a join that is transcribed."""

GUARD_SECONDS = 0.04
"""A join this close to a word's edge is treated as on the edge, not inside:
whisper's word times are not accurate to better than this."""

EDGE_TRUST_SECONDS = 0.05
"""Words touching the window's own boundary are clipped by it and ignored."""

MIN_PAUSE_SECONDS = 0.1
"""A gap between heard words must be at least this to count as a pause."""

DEFAULT_REACH_SECONDS = 1.5
"""How far from the join a suggested pause may be."""

TranscribeWindow = Callable[[float, float], "list[HeardWord]"]
"""(start, end) in track seconds -> words with **track-absolute** times."""


class JoinInsideWordError(RuntimeError):
    """Strict mode refusal: at least one join falls inside a heard word."""

    def __init__(self, check: JoinCheck) -> None:
        self.check = check
        super().__init__(
            f"{len(check.findings)} chunk join(s) fall inside a sung word: "
            + "; ".join(_describe(f) for f in check.findings)
            + ". The lip-sync model will not mouth a half-word at a clip's edge."
        )


@dataclass(frozen=True)
class HeardWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class JoinFinding:
    chunk_before: int
    chunk_after: int
    join_seconds: float
    word: str
    word_start: float
    word_end: float
    suggested_seconds: float | None
    """Midpoint of the nearest heard pause, or None if none within reach."""


@dataclass(frozen=True)
class JoinCheck:
    joins_checked: int
    findings: tuple[JoinFinding, ...]
    unchecked: tuple[tuple[int, int, float], ...] = ()
    """Joins whose window could not be transcribed (never silently passed)."""


def _describe(f: JoinFinding) -> str:
    hint = (
        f", nearest pause {f.suggested_seconds:.3f}s"
        if f.suggested_seconds is not None
        else ", no pause within reach"
    )
    return (
        f"chunk {f.chunk_before}/{f.chunk_after} join at {f.join_seconds:.3f}s is inside "
        f"{f.word!r} ({f.word_start:.3f}-{f.word_end:.3f}s{hint})"
    )


# -- pure logic ----------------------------------------------------------- #


def word_containing(
    join: float, words: Sequence[HeardWord], *, guard: float = GUARD_SECONDS
) -> HeardWord | None:
    """The word the join falls strictly inside (beyond ``guard`` of both edges)."""
    for word in words:
        if word.start + guard < join < word.end - guard:
            return word
    return None


def nearest_clear_time(
    join: float,
    words: Sequence[HeardWord],
    *,
    reach: float = DEFAULT_REACH_SECONDS,
    min_pause: float = MIN_PAUSE_SECONDS,
) -> float | None:
    """Midpoint of the heard pause nearest ``join`` within ``reach``, if any."""
    ordered = sorted(words, key=lambda w: w.start)
    mids = [
        (a.end + b.start) / 2
        for a, b in zip(ordered, ordered[1:], strict=False)
        if b.start - a.end >= min_pause
    ]
    near = [m for m in mids if abs(m - join) <= reach]
    return min(near, key=lambda m: abs(m - join)) if near else None


def chunk_joins(chunks: Sequence[AudioChunk]) -> list[tuple[int, int, float]]:
    """(chunk_before, chunk_after, seconds) for every join, in order."""
    ordered = sorted(chunks, key=lambda c: c.start)
    return [(a.chunk_id, b.chunk_id, b.start) for a, b in zip(ordered, ordered[1:], strict=False)]


def check_joins(
    joins: Sequence[tuple[int, int, float]],
    transcribe_window: TranscribeWindow,
    *,
    window_seconds: float = WINDOW_SECONDS,
    guard: float = GUARD_SECONDS,
) -> JoinCheck:
    findings: list[JoinFinding] = []
    unchecked: list[tuple[int, int, float]] = []
    for before, after, join in joins:
        start = max(0.0, join - window_seconds)
        end = join + window_seconds
        try:
            heard = transcribe_window(start, end)
        except Exception:  # noqa: BLE001 -- a failed window must not pass silently
            logger.exception("Join check: could not transcribe %.3f-%.3fs", start, end)
            unchecked.append((before, after, join))
            continue
        trusted = [
            w for w in heard
            if w.start > start + EDGE_TRUST_SECONDS and w.end < end - EDGE_TRUST_SECONDS
        ]
        hit = word_containing(join, trusted, guard=guard)
        if hit is not None:
            findings.append(
                JoinFinding(
                    chunk_before=before,
                    chunk_after=after,
                    join_seconds=join,
                    word=hit.text,
                    word_start=hit.start,
                    word_end=hit.end,
                    suggested_seconds=nearest_clear_time(join, trusted),
                )
            )
    return JoinCheck(len(joins), tuple(findings), tuple(unchecked))


def log_findings(check: JoinCheck) -> None:
    logger.info(
        "Join check: %d join(s) transcribed, %d inside a sung word, %d unchecked",
        check.joins_checked, len(check.findings), len(check.unchecked),
    )
    for finding in check.findings:
        logger.warning("Join inside a word: %s", _describe(finding))
    for before, after, join in check.unchecked:
        logger.warning("Join chunk %d/%d at %.3fs could not be checked", before, after, join)


def raise_if_blocking(check: JoinCheck, *, strict: bool) -> None:
    if strict and check.findings:
        logger.error("Refusing to continue: %d join(s) inside a word", len(check.findings))
        raise JoinInsideWordError(check)


# -- the real transcriber (lazy, CPU) ------------------------------------- #


def words_from_result(result: object, *, offset: float) -> list[HeardWord]:
    """Flatten a stable-ts result into words with track-absolute times."""
    words: list[HeardWord] = []
    for segment in result.segments:  # type: ignore[attr-defined]
        for word in segment.words:
            text = str(word.word).strip()
            if text:
                words.append(HeardWord(text, float(word.start) + offset, float(word.end) + offset))
    return words


def whisper_window_transcriber(
    audio_path: str | Path,
    *,
    model_size: str = "base",
    language: str | None = None,
    model: object | None = None,
) -> TranscribeWindow:
    """A :data:`TranscribeWindow` over ``audio_path``, on CPU.

    The audio is decoded once (mono, 16 kHz); each call transcribes just its
    window. Requires the ``[align]`` extra."""
    try:
        import numpy as np  # noqa: PLC0415 -- heavy and optional
        import stable_whisper  # noqa: PLC0415 -- heavy and optional
        from pydub import AudioSegment  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "the join-inside-word check needs stable-ts: pip install -e '.[align]'"
        ) from exc
    if model is None:
        model = stable_whisper.load_model(model_size, device="cpu")
    segment = AudioSegment.from_file(str(audio_path)).set_channels(1).set_frame_rate(16000)
    segment = segment.set_sample_width(2)
    samples = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_window(start: float, end: float) -> list[HeardWord]:
        clip = samples[int(start * 16000): int(end * 16000)]
        result = model.transcribe(  # type: ignore[attr-defined]
            clip, word_timestamps=True, language=language, fp16=False, verbose=None
        )
        return words_from_result(result, offset=start)

    return transcribe_window
