"""The transcript as a second witness against forced alignment (issue #105, part 2).

Forced alignment places the lyrics file's words on the master. A transcript of
the isolated vocal stem (``draft_lyrics``' ``*.words.json``) says where the same
words were *heard*. Matching the two in order gives two independent placements
of one text, which answers two questions no other check here asks:

* **Did the aligner put this line where it was sung?**
  :attr:`Comparison.disagreements` -- a segment whose matched words sit more
  than :data:`DISAGREEMENT_SECONDS` from where the transcript heard them. #71's
  closing line 12 s into the fadeout and #42's ``base`` pile-up are this shape.
* **Is the lyrics file what was sung?** :attr:`Comparison.passages` -- a run of
  singing with no lyric line (an unwritten repeat chorus, an ad-lib long enough
  to matter) or a run of lyric lines nothing sang. Every other check here
  assumes the lyrics file is right and checks the alignment against it.

**Pure, and no ASR.** This module reads a transcript a person produced off the
render path with ``python -m music_video_maker.draft_lyrics``; it never
transcribes anything. Forced alignment stays the only thing that produces
timestamps: this only checks and reports, and never moves one.

**#42's caveat, and the two rules it forces.** ASR and the aligner are both
whisper, and whisper's silence fill tends to *repeat earlier lyric text*,
which would match the lyrics file perfectly and look like the strongest
evidence there is. So:

* only words ``draft_lyrics``' voice gate kept (``voiced``) are evidence;
* only runs of at least :data:`MIN_MATCH_RUN` consecutive matching words count
  as a match -- "the" and "back" match anywhere, and a draft gets roughly a
  quarter of its words wrong, so a lone matching word places nothing.

**Mishearings are not findings.** Between two matched runs, the lyrics file
and the transcript each have some unmatched words. When the counts are close,
that is the transcript getting words wrong in place; only a difference of at
least :data:`MIN_PASSAGE_WORDS` is reported as a passage. A gap is judged by
the difference, so an unwritten chorus that happens to sit beside a
mishearing is still seen.

**What it cannot see.** Two voices at once: a transcript is one linear text,
and on "The Lucky Ones"' closing counterpoint both models dropped most of one
voice, which reads here as lyric lines nothing sang. And a lyrics file missing
a repeat can bind its one written copy to the *wrong* sung copy -- the
passage report still names the missing one, but the disagreement check then
compares the aligner's placement against the wrong instance.

Thresholds are unmeasured until #105's measurement says otherwise.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from music_video_maker.contracts import AlignedSegment, AlignmentResult

MIN_MATCH_RUN = 3
"""Consecutive matching words needed before a match is evidence."""

MIN_SEGMENT_MATCHES = 3
"""Matched words a segment needs before its placement is judged."""

DISAGREEMENT_SECONDS = 3.0
"""How far a segment's placement may sit from where its words were heard.
Whisper's word times are coarse; this is well above that and well below the
documented misplacements (#71: 12 s, #42: 6-51 s)."""

MIN_PASSAGE_WORDS = 6
"""How many more words one side must have than the other, between two
matched runs, before the difference is a passage rather than mishearings."""

RESEMBLANCE_RATIO = 0.6
"""How close an unwritten passage must be to an existing lyric line for the
report to say it looks like a repeat of it."""

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class TranscriptError(ValueError):
    """A ``*.words.json`` file could not be read as a transcript."""


@dataclass(frozen=True)
class HeardWord:
    text: str
    start: float
    end: float
    probability: float | None
    voiced: bool


@dataclass(frozen=True)
class Transcript:
    words: tuple[HeardWord, ...]
    model: str | None = None
    stem: str | None = None
    stem_sha256: str | None = None


@dataclass(frozen=True)
class Disagreement:
    segment_index: int
    placed_start: float
    heard_start: float
    """Where the evidence says the line was sung: ``placed_start - offset`` for
    ``"matched"``, the violated neighbour bound for ``"bounds"``."""
    offset: float
    """Median of (placed - heard) over the segment's matched words; positive
    means the aligner placed the line late."""
    matched_words: int
    basis: str = "matched"
    """``"matched"``: the segment's own words were heard elsewhere. ``"bounds"``:
    too few of its words matched to judge them (a two-word closing line), so it
    is held to the heard times of its matched neighbours instead -- it must
    not start before the line before it was heard, nor after the line after it
    (or, for the last lines, after the last voiced word in the transcript)."""


@dataclass(frozen=True)
class Passage:
    kind: str
    """``"sung_not_written"`` or ``"written_not_sung"``."""
    start: float
    end: float
    text: str
    segment_indices: tuple[int, ...] = ()
    """The lyric segments a ``written_not_sung`` passage covers."""
    resembles_segment: int | None = None
    """For ``sung_not_written``: the lyric segment it most looks like a repeat of."""


@dataclass(frozen=True)
class Comparison:
    disagreements: tuple[Disagreement, ...]
    passages: tuple[Passage, ...]
    matched_words: int


def load_transcript(path: Path | str) -> Transcript:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_words = payload["words"]
        if not isinstance(raw_words, list):
            raise TypeError("'words' is not a list")
        words = tuple(
            HeardWord(
                text=str(raw["text"]),
                start=float(raw["start"]),
                end=float(raw["end"]),
                probability=None if raw.get("probability") is None else float(raw["probability"]),
                voiced=bool(raw["voiced"]),
            )
            for raw in raw_words
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TranscriptError(
            f"{path} is not a transcript as draft_lyrics writes it (*.words.json): {exc}"
        ) from exc
    return Transcript(
        words=words,
        model=payload.get("model"),
        stem=payload.get("stem"),
        stem_sha256=payload.get("stem_sha256"),
    )


def _tokens(text: str) -> list[str]:
    found = _TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [t.strip("'") for t in found if t.strip("'")]


@dataclass(frozen=True)
class _Token:
    text: str
    start: float
    end: float
    segment_index: int | None = None


def _lyric_tokens(segments: Sequence[AlignedSegment]) -> list[_Token]:
    out = []
    for segment in segments:
        for word in segment.words:
            for token in _tokens(word.word):
                out.append(_Token(token, word.start, word.end, segment.index))
    return out


def _heard_tokens(transcript: Transcript) -> list[_Token]:
    out = []
    for word in transcript.words:
        if not word.voiced:
            continue
        for token in _tokens(word.text):
            out.append(_Token(token, word.start, word.end))
    return out


def compare(result: AlignmentResult, transcript: Transcript) -> Comparison:
    segments = result.segments
    lyric = _lyric_tokens(segments)
    heard = _heard_tokens(transcript)
    matcher = SequenceMatcher(
        None, [t.text for t in lyric], [t.text for t in heard], autojunk=False
    )
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= MIN_MATCH_RUN]

    pairs: list[tuple[_Token, _Token]] = []
    heard_at: dict[int, _Token] = {}
    for block in blocks:
        for k in range(block.size):
            pairs.append((lyric[block.a + k], heard[block.b + k]))
            heard_at[block.a + k] = heard[block.b + k]

    matched = _disagreements(pairs)
    judged = {d.segment_index for d in matched}
    bounded = _bounded(segments, lyric, heard, heard_at, judged)
    return Comparison(
        disagreements=tuple(sorted([*matched, *bounded], key=lambda d: d.segment_index)),
        passages=tuple(_passages(segments, lyric, heard, blocks)),
        matched_words=len(pairs),
    )


def _disagreements(pairs: Sequence[tuple[_Token, _Token]]) -> list[Disagreement]:
    by_segment: dict[int, list[tuple[_Token, _Token]]] = {}
    for placed, sung in pairs:
        assert placed.segment_index is not None
        by_segment.setdefault(placed.segment_index, []).append((placed, sung))
    found = []
    for index, matched in sorted(by_segment.items()):
        if len(matched) < MIN_SEGMENT_MATCHES:
            continue
        offset = statistics.median(placed.start - sung.start for placed, sung in matched)
        if abs(offset) <= DISAGREEMENT_SECONDS:
            continue
        placed_start = min(p.start for p, _ in matched)
        found.append(
            Disagreement(
                segment_index=index,
                placed_start=placed_start,
                heard_start=placed_start - offset,
                offset=offset,
                matched_words=len(matched),
            )
        )
    return found


def _bounded(
    segments: Sequence[AlignedSegment],
    lyric: Sequence[_Token],
    heard: Sequence[_Token],
    heard_at: dict[int, _Token],
    judged: set[int],
) -> list[Disagreement]:
    """Hold each segment the matched evidence could not judge to the heard
    times of its matched neighbours. See :attr:`Disagreement.basis`."""
    if not heard:
        return []
    last_voiced_end = max(t.end for t in heard)
    positions: dict[int, list[int]] = {}
    for pos, token in enumerate(lyric):
        assert token.segment_index is not None
        positions.setdefault(token.segment_index, []).append(pos)
    matched_positions = sorted(heard_at)
    found = []
    for segment in segments:
        own = positions.get(segment.index)
        if not own or segment.index in judged:
            continue
        if sum(1 for pos in own if pos in heard_at) >= MIN_SEGMENT_MATCHES:
            continue  # judged on its own words, and found in place
        before = [pos for pos in matched_positions if pos < own[0]]
        after = [pos for pos in matched_positions if pos > own[-1]]
        lower = heard_at[before[-1]].start if before else None
        upper = heard_at[after[0]].start if after else last_voiced_end
        if segment.start > upper + DISAGREEMENT_SECONDS:
            edge = upper
        elif lower is not None and segment.start < lower - DISAGREEMENT_SECONDS:
            edge = lower
        else:
            continue
        found.append(
            Disagreement(
                segment_index=segment.index,
                placed_start=segment.start,
                heard_start=edge,
                offset=segment.start - edge,
                matched_words=sum(1 for pos in own if pos in heard_at),
                basis="bounds",
            )
        )
    return found


def _passages(
    segments: Sequence[AlignedSegment],
    lyric: Sequence[_Token],
    heard: Sequence[_Token],
    blocks: Sequence,
) -> list[Passage]:
    found = []
    edges = [(0, 0), *((b.a + b.size, b.b + b.size) for b in blocks)]
    starts = [*((b.a, b.b) for b in blocks), (len(lyric), len(heard))]
    for (a_from, b_from), (a_to, b_to) in zip(edges, starts, strict=True):
        lyric_gap = lyric[a_from:a_to]
        heard_gap = heard[b_from:b_to]
        repeat = _repeat_of_written_text(heard_gap, lyric, (a_from, a_to))
        unexplained_heard = len(heard_gap)
        if repeat is not None:
            first, last, resembles = repeat
            span = heard_gap[first:last]
            unexplained_heard -= len(span)
            found.append(
                Passage(
                    kind="sung_not_written",
                    start=span[0].start,
                    end=span[-1].end,
                    text=" ".join(t.text for t in span),
                    resembles_segment=resembles,
                )
            )
        excess = unexplained_heard - len(lyric_gap)
        if repeat is None and excess >= MIN_PASSAGE_WORDS:
            text = " ".join(t.text for t in heard_gap)
            found.append(
                Passage(
                    kind="sung_not_written",
                    start=heard_gap[0].start,
                    end=heard_gap[-1].end,
                    text=text,
                    resembles_segment=_resembles(text, segments),
                )
            )
        elif -excess >= MIN_PASSAGE_WORDS:
            indices = tuple(dict.fromkeys(t.segment_index for t in lyric_gap))
            found.append(
                Passage(
                    kind="written_not_sung",
                    start=lyric_gap[0].start,
                    end=lyric_gap[-1].end,
                    text=" ".join(t.text for t in lyric_gap),
                    segment_indices=tuple(i for i in indices if i is not None),
                )
            )
    return found


def _repeat_of_written_text(
    heard_gap: Sequence[_Token], lyric: Sequence[_Token], gap: tuple[int, int]
) -> tuple[int, int, int | None] | None:
    """The part of an unmatched stretch of singing that repeats lyric text
    written *elsewhere* -- the unwritten repeat chorus. Judged on content, not
    on how many words each side of the gap has: on "The Lucky Ones" an
    unwritten repeat and closing lines the transcript never heard fell into
    one gap, and equal counts hid both. Returns ``(first, last, segment)``:
    a half-open range into ``heard_gap`` and the lyric segment it repeats."""
    if len(heard_gap) < MIN_PASSAGE_WORDS:
        return None
    matcher = SequenceMatcher(
        None, [t.text for t in heard_gap], [t.text for t in lyric], autojunk=False
    )
    blocks = [
        b for b in matcher.get_matching_blocks()
        if b.size >= MIN_MATCH_RUN and not gap[0] <= b.b < gap[1]
    ]
    if sum(b.size for b in blocks) < MIN_PASSAGE_WORDS:
        return None
    return blocks[0].a, blocks[-1].a + blocks[-1].size, lyric[blocks[0].b].segment_index


def _resembles(text: str, segments: Sequence[AlignedSegment]) -> int | None:
    best_index, best_ratio = None, 0.0
    for segment in segments:
        candidate = " ".join(_tokens(segment.text))
        ratio = SequenceMatcher(None, text, candidate, autojunk=False).ratio()
        if ratio > best_ratio:
            best_index, best_ratio = segment.index, ratio
    return best_index if best_ratio >= RESEMBLANCE_RATIO else None
