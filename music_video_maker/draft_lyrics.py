"""Draft a lyrics file from the isolated vocal stem, for a person to correct
(issue #105, part 1).

``python -m music_video_maker.draft_lyrics --stem vocals.wav`` transcribes the
stem with stable-ts's ``model.transcribe()`` and writes ``lyrics.draft.txt``
in the normal lyrics format (``docs/lyrics-format.md``), plus a sidecar
``lyrics.draft.report.txt`` saying where to listen. It is the supervised
transcription step #57 named as the one way ASR can enter this project
without reopening the bug "lyrics are immutable truth" exists to prevent:

* **The draft is refused by Stage 1 until someone reviews it.** Every header
  line starts with :data:`~music_video_maker.lyrics.DRAFT_HEADER_PREFIX`, and
  :func:`~music_video_maker.lyrics.parse_lyrics_text` refuses any file
  containing one. Deleting the header is the explicit "I have reviewed this".
* **Nothing imports this module** (``tests/test_authoring_boundary.py``), so
  the only road from a transcript to Stage 1 runs through a file a person
  has edited. Forced alignment stays the only thing that produces timestamps.

Like ``--prepare`` (#52) and ``castgen``: no GPU render, no ComfyUI, no
custody handoff. It is its own entry point rather than a ``cli.py`` flag
because a new song has no lyrics file yet, and ``load_config`` refuses a
config whose ``lyrics_file`` does not exist.

**The stem, never the mix.** ASR on a full mix is the failure #57 describes;
on an isolated stem it is a much easier problem. This module cannot tell a
stem from a mix by listening, so it trusts the operator's ``--stem`` and says
so; it refuses a missing file and points at ``docs/vocal-stem-workflow.md``
rather than looking for anything else to transcribe.

**Words with no voice under them are removed.** Whisper's best-known failure
on music is filling silence with text, often a repeat of an earlier line. On
an isolated stem, silence really is near-silent, so the test is level: a
word whose span (padded by :data:`WORD_PAD_SECONDS`, because word timestamps
are coarse) never peaks above :data:`~music_video_maker.stems.DEFAULT_SILENT_VOCAL_DBFS`
-- the same "no voice on this stem slice" level Stage 2a-stem uses -- is
dropped, and the report lists it. What this gate **cannot** catch: a phantom
over separation leakage, which on a real stem can peak near
:data:`~music_video_maker.stems.DEFAULT_LEAKAGE_DBFS`. #71/#96's mix-calibrated
voice checks are not reused here; they judge a span against the *mix's*
median, and this file is not the mix.

**Thresholds are unmeasured.** :data:`LOW_PROBABILITY` decides which words the
report lists for a listen; it has not been scored against a corpus, and the
report says so.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from music_video_maker.config import ALIGNMENT_MODEL_SIZES
from music_video_maker.lyrics import DRAFT_HEADER_PREFIX
from music_video_maker.stems import DEFAULT_SILENT_VOCAL_DBFS
from music_video_maker.voicing import PcmAudio

logger = logging.getLogger(__name__)

DEFAULT_DRAFT_MODEL_SIZE = "medium"
"""Independent of ``alignment_model_size``. #105 proposed ``large-v3`` and left
"is medium enough?" to the WER measurement (2026-10-08, htdemucs stems, scored
against each song's authored lyrics, ~1 min per song on CPU for either model):

=============  ==========  ==============  ======
model          Deathless   The Lucky Ones  pooled
=============  ==========  ==============  ======
``medium``     28.0%       25.8%           26.9%
``large-v3``   51.2%       17.2%           34.9%
=============  ==========  ==============  ======

**A split, not a sweep.** large-v3 is clearly better on one song and clearly
worse on the other, and its loss is repetition loops -- the second verse sung
again, "back to you" eight times -- laid over real audio, where the voice gate
cannot see them. medium wins pooled and its errors are mishearings in place,
which are cheaper to review than plausible lines nobody sang; that is the
reason for the default, and two songs is all it rests on. ``--model-size``
is there to re-measure."""

LOW_PROBABILITY = 0.5
"""Words whisper scored below this are listed in the report. Unmeasured."""

WORD_PAD_SECONDS = 0.2
"""How far either side of a word's timestamps the voice gate looks, so a real
word whose coarse timing lands just off its own voice is not dropped."""

_STEM_WORKFLOW_DOC = "docs/vocal-stem-workflow.md"
_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|[♪♫♬]")
"""Whisper's own annotations, which are not anything sung: ``[Music]`` would
parse as a character tag, and a music note is not a word. Parentheses are
left alone -- ``(oh, oh)`` is how backing vocals are written."""


class DraftLyricsError(Exception):
    """A draft could not be written; the message says what to do instead."""


@dataclass(frozen=True)
class TranscribedWord:
    text: str
    start: float
    end: float
    probability: float | None


@dataclass(frozen=True)
class DraftResult:
    draft_path: Path
    report_path: Path
    words_path: Path
    lines: tuple[tuple[TranscribedWord, ...], ...]
    removed: tuple[TranscribedWord, ...]
    language: str | None


# -- loading ------------------------------------------------------------- #


def _load_model(model_size: str) -> object:
    """Lazily import and load a stable-ts model. Requires the ``[align]`` extra."""
    try:
        import stable_whisper  # noqa: PLC0415 -- heavy and optional, see alignment._load_model
    except ImportError as exc:
        raise DraftLyricsError(
            "--draft-lyrics needs stable-ts: pip install -e '.[align]'"
        ) from exc
    return stable_whisper.load_model(model_size)


def load_stem_audio(path: Path) -> PcmAudio:
    """Decode the stem to mono 16 kHz 16-bit PCM, the shape :mod:`voicing` reads."""
    from pydub import AudioSegment  # noqa: PLC0415 -- only needed on the real path

    segment = AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)
    segment = segment.set_sample_width(2)
    samples = array.array("h")
    samples.frombytes(segment.raw_data)
    return PcmAudio(samples=samples, sample_rate=16000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- transcription ------------------------------------------------------- #


def transcribe(
    model: object, stem: Path, *, language: str | None
) -> tuple[list[list[TranscribedWord]], str | None]:
    """One list of words per whisper segment, annotations removed."""
    result = model.transcribe(str(stem), language=language)  # type: ignore[attr-defined]
    lines: list[list[TranscribedWord]] = []
    for segment in result.segments:
        words = []
        for word in segment.words:
            text = _ANNOTATION_RE.sub("", word.word).strip()
            if not text:
                continue
            words.append(
                TranscribedWord(
                    text=text,
                    start=float(word.start),
                    end=float(word.end),
                    probability=getattr(word, "probability", None),
                )
            )
        if words:
            lines.append(words)
    return lines, getattr(result, "language", None)


def _span_peak_dbfs(audio: PcmAudio, start: float, end: float) -> float:
    rate = audio.sample_rate
    first = max(0, int(start * rate))
    last = min(len(audio.samples), int(math.ceil(end * rate)))
    if last <= first:
        return -float("inf")
    peak = max(abs(v) for v in audio.samples[first:last])
    return 20.0 * math.log10(peak / 32768.0) if peak else -float("inf")


def screen_silent_words(
    lines: Sequence[Sequence[TranscribedWord]],
    audio: PcmAudio,
    *,
    floor_dbfs: float = DEFAULT_SILENT_VOCAL_DBFS,
) -> tuple[list[list[TranscribedWord]], list[TranscribedWord]]:
    """Drop every word with no voice under it; a line left empty is dropped."""
    kept_lines: list[list[TranscribedWord]] = []
    removed: list[TranscribedWord] = []
    for line in lines:
        kept = []
        for word in line:
            peak = _span_peak_dbfs(
                audio, word.start - WORD_PAD_SECONDS, word.end + WORD_PAD_SECONDS
            )
            if peak < floor_dbfs:
                removed.append(word)
            else:
                kept.append(word)
        if kept:
            kept_lines.append(kept)
    return kept_lines, removed


# -- writing ------------------------------------------------------------- #


def _clock(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:05.2f}"


def _line_text(words: Sequence[TranscribedWord]) -> str:
    return " ".join(word.text for word in words)


def _header(
    *, stem: Path, stem_sha256: str, model_size: str, language: str | None,
    requested_language: str | None, report_name: str,
) -> list[str]:
    detected = "" if requested_language else " (detected)"
    written = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    p = DRAFT_HEADER_PREFIX
    return [
        f"{p} DRAFT LYRICS -- a machine transcription of the vocal stem, NOT yet reviewed (#105).",
        f"{p} Stage 1 refuses this file while any line starting {p!r} remains.",
        f"{p} Listen through it against the stem and correct every line ({report_name} lists",
        f"{p} where to listen), then delete every line starting {p!r} -- that is the act that",
        f'{p} says "I have reviewed this". Write repeats out as sung; tag singers as [Name: Role].',
        f"{p} model: {model_size}   language: {language or 'unknown'}{detected}",
        f"{p} stem: {stem.name}   sha256: {stem_sha256}",
        f"{p} written: {written}",
        f"{p} report: {report_name}",
    ]


def _report(
    *, draft_name: str, model_size: str, body_start_line: int,
    lines: Sequence[Sequence[TranscribedWord]], removed: Sequence[TranscribedWord],
) -> str:
    out = [
        f"Report for {draft_name} (model {model_size}).",
        "",
        "Removed -- no voice under them on the stem (peak below "
        f"{DEFAULT_SILENT_VOCAL_DBFS:g} dBFS",
        f"within {WORD_PAD_SECONDS:g} s of the word). Usually whisper filling silence; listen if",
        "a line looks short:",
    ]
    out += [f"  {_clock(w.start)}  {w.text!r}" for w in removed] or ["  (none)"]
    out += [
        "",
        f"Low-confidence -- whisper's probability below {LOW_PROBABILITY:g} (an unmeasured",
        "threshold; a word above it can still be wrong):",
    ]
    low = []
    for offset, line in enumerate(lines):
        for word in line:
            if word.probability is not None and word.probability < LOW_PROBABILITY:
                low.append(
                    f"  {_clock(word.start)}  line {body_start_line + offset}  "
                    f"{word.text!r}  p={word.probability:.2f}"
                )
    out += low or ["  (none)"]
    return "\n".join(out) + "\n"


def write_draft(
    stem: Path,
    out: Path,
    *,
    model_size: str = DEFAULT_DRAFT_MODEL_SIZE,
    language: str | None = None,
    force: bool = False,
    model: object | None = None,
    load_audio: Callable[[Path], PcmAudio] | None = None,
) -> DraftResult:
    stem = Path(stem)
    out = Path(out)
    if not stem.is_file():
        raise DraftLyricsError(
            f"no vocal stem at {stem}. --draft-lyrics transcribes the ISOLATED vocal stem, "
            f"never the mix; make one first -- see {_STEM_WORKFLOW_DOC}"
        )
    if out.exists() and not force:
        raise DraftLyricsError(
            f"{out} already exists; it may hold corrections. Pass --force to overwrite it"
        )
    report_path = out.with_name(f"{out.stem}.report.txt")
    words_path = out.with_name(f"{out.stem}.words.json")
    # Before the model runs: a transcription costs minutes, and an unwritable
    # destination discovered afterwards would throw all of them away.
    out.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = _load_model(model_size)
    audio = (load_audio or load_stem_audio)(stem)

    transcribed, detected = transcribe(model, stem, language=language)
    lines, removed = screen_silent_words(transcribed, audio)

    stem_sha256 = _sha256(stem)
    header = _header(
        stem=stem, stem_sha256=stem_sha256, model_size=model_size,
        language=language or detected, requested_language=language,
        report_name=report_path.name,
    )
    body_start_line = len(header) + 2  # header, one blank line, then the body
    out.write_text("\n".join([*header, "", *(_line_text(line) for line in lines)]) + "\n")
    report_path.write_text(
        _report(
            draft_name=out.name, model_size=model_size, body_start_line=body_start_line,
            lines=lines, removed=removed,
        )
    )
    logger.info(
        "Wrote %s: %d line(s); %d word(s) removed with no voice under them; report %s",
        out, len(lines), len(removed), report_path,
    )
    # #105 part 2: the transcript with its timings, every word including the
    # ones the voice gate removed (marked, never silently dropped), so a later
    # check can read where each word was HEARD. Only voiced words are evidence.
    removed_ids = {id(word) for word in removed}
    words_path.write_text(
        json.dumps(
            {
                "model": model_size,
                "language": language or detected,
                "stem": stem.name,
                "stem_sha256": stem_sha256,
                "words": [
                    {
                        "text": word.text,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                        "voiced": id(word) not in removed_ids,
                    }
                    for line in transcribed
                    for word in line
                ],
            },
            indent=1,
        )
        + "\n"
    )
    return DraftResult(
        draft_path=out, report_path=report_path, words_path=words_path,
        lines=tuple(tuple(line) for line in lines), removed=tuple(removed),
        language=language or detected,
    )


# -- CLI ----------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.draft_lyrics",
        description=(
            "Draft a lyrics file from the isolated vocal stem for a person to correct "
            "(issue #105). Stage 1 refuses the draft until its '#!' header is deleted."
        ),
    )
    parser.add_argument(
        "--stem", type=Path, required=True,
        help=f"the ISOLATED vocal stem, never the mix (see {_STEM_WORKFLOW_DOC})",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("lyrics.draft.txt"),
        help="where to write the draft (default: lyrics.draft.txt); the report goes beside it",
    )
    parser.add_argument(
        "--model-size", default=DEFAULT_DRAFT_MODEL_SIZE, choices=sorted(ALIGNMENT_MODEL_SIZES),
        help=(
            f"whisper model (default {DEFAULT_DRAFT_MODEL_SIZE}; "
            "independent of alignment_model_size)"
        ),
    )
    parser.add_argument("--language", help="language code, e.g. 'en' (default: detect)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing draft")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = write_draft(
            args.stem, args.out, model_size=args.model_size,
            language=args.language, force=args.force,
        )
    except DraftLyricsError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "Review %s before Stage 1 will read it; %s says where to listen.",
        result.draft_path, result.report_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
