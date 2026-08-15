"""Lyric character-tag parsing for dynamic cast switching (issue #6, #33).

Parses a lyrics file containing optional character tags into a sequence of
:class:`~music_video_maker.contracts.LyricLine`. A tag selects the "active"
cast member(s) for itself and every untagged line that follows, until the
next tag switches it again (falling back to the configured default lead
vocalist before any tag has appeared).

Three grammar levels are defined in issue #33 / ``docs/lyrics-format.md``,
and all three are implemented here:

- **Level 1** -- ``[Name]`` or ``[Name: Role]``. One voice audible. Original
  syntax (issue #6), unchanged.
- **Level 2** -- ``[Name & Name]`` or ``[Name & Name: Role]``. Two or more
  voices audible on the same words, in the written order (the first name is
  primary -- see ``LyricLine.character``). Whitespace around ``&`` is
  free-form. The role slot after the colon is accepted syntactically and, as
  with level 1, not otherwise used.
- **Level 3** -- ``[simultaneously] ... [/simultaneously]`` blocks. Two or
  more voices audible on *different* words at the same time.

Level 3 and the one-linear-transcript rule
------------------------------------------
Stage 1 forced alignment maps audio to **one linear text**, and the lyrics
file *is* that text. Counterpoint has two texts over one span, so something
has to give, and the rule (issue #33) is the **alignment spine**: inside a
``[simultaneously]`` block, the **first sub-block is the spine**. Its lines
are ordinary :class:`LyricLine` objects in the returned sequence -- the
transcript stable-ts actually times against. Every later sub-block becomes a
:class:`CounterpointStream`, carried *alongside* the transcript and never in
it, and inherits the spine's span downstream (see
:mod:`music_video_maker.alignment`).

Concurrent streams are deliberately **not** paired line by line: in "The
Lucky Ones" the spine block is four lines and the counterpoint is three. Any
pairing rule would be a fiction; the spine's *span* is the only shared fact.

Because the counterpoint text never enters the transcript,
:func:`parse_lyrics_text` returns a :class:`LyricsDocument` -- a plain
``tuple`` of ``LyricLine`` (so every existing caller is unaffected) that
additionally carries ``.counterpoint``. ``alignment.align()`` picks that up
automatically when it is handed a document rather than a bare tuple, so a
level-3 file cannot silently lose its counterpoint by passing through a
caller that predates it.

Every name in a tag -- solo, ``&``-joined, or inside a block -- is validated
against the cast before any ``LyricLine`` is produced from it, so a typo'd
singer fails at config load, before any GPU time is spent.

Tags are always stripped from ``LyricLine.text`` -- forced alignment (issue
#3) must never see a tag, or it pollutes word timestamps. ``LyricLine.raw``
keeps the untouched source line for diagnostics only.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import CastMember, LyricLine

logger = logging.getLogger(__name__)

# Matches a leading "[Name]", "[Name: Role]", "[Name & Name]" or
# "[Name & Name: Role]" tag, capturing any trailing text on the same physical
# line (e.g. "[Dianne: Lead] The lucky ones..."). The name group happily
# captures "Jan & Dianne" as one string -- splitting on "&" is done
# separately in _parse_names, once level 3 has been ruled out.
_TAG_RE = re.compile(r"^\[\s*(?P<name>[^\]:]+?)\s*(?::\s*[^\]]*)?\]\s*(?P<rest>.*)$")

# Level 3 (issue #33) block delimiters, matched case-insensitively. They are
# reserved words in the tag namespace: a cast member may not be named either
# of these, and a parser that does not understand them must fail rather than
# read "simultaneously" as a character name.
_BLOCK_OPEN_KEYWORD = "simultaneously"
_BLOCK_CLOSE_KEYWORD = "/simultaneously"
_LEVEL3_ISSUE_URL = "https://github.com/JavaDerek/music-video-pipeline/issues/33"


class LyricsError(Exception):
    """Raised when the lyrics source is malformed or references a character
    not in the cast."""


@dataclass(frozen=True)
class CounterpointStream:
    """One non-spine sub-block of a ``[simultaneously]`` block (issue #33).

    A stream is text sung *at the same time as* the spine, in different
    words. It is deliberately **not** a sequence of :class:`LyricLine`: a
    ``LyricLine`` lives in the transcript index space that Stage 1 aligns and
    that ``LyricCastResolver`` resolves against, and putting concurrent text
    in that space is exactly the index-space confusion this project keeps
    being bitten by. A stream's text is held as plain strings, and the only
    link back to the transcript is :attr:`spine_line_indices`.
    """

    block_index: int
    """Which ``[simultaneously]`` block in the file, 0-based."""
    stream_index: int
    """Position within the block. The spine is 0 (and is *not* represented by
    a ``CounterpointStream`` -- it is in the transcript); counterpoint streams
    are 1, 2, ... in written order."""
    characters: tuple[str, ...]
    """Every character audible on this stream, primary first."""
    texts: tuple[str, ...]
    """The stream's lines, tag-stripped, in written order. Never fed to
    forced alignment."""
    spine_line_indices: tuple[int, ...]
    """``LyricLine.index`` of every spine line this stream is concurrent with.

    This is **LyricLine index space**, not ``AlignedSegment`` index space and
    not a chunk id. The stream's span is derived from these lines' aligned
    timings downstream; the counts need not match (a four-line spine under a
    three-line counterpoint is the real case from "The Lucky Ones")."""

    @property
    def character(self) -> str | None:
        """Primary vocalist of this stream. Mirrors ``LyricLine.character``."""
        return self.characters[0] if self.characters else None

    @property
    def word_count(self) -> int:
        return sum(len(text.split()) for text in self.texts)


class LyricsDocument(tuple):  # noqa: SLOT001 -- tuple subclass on purpose, see docstring
    """The parser's output: a ``tuple`` of :class:`LyricLine` that also carries
    the level-3 counterpoint streams (issue #33).

    A ``tuple`` subclass rather than a new dataclass so that every caller
    written before level 3 -- ``cli.py``, the tests, anything that iterates,
    indexes or ``len()``s the parse result -- keeps working untouched, while
    ``alignment.align()`` can pick ``.counterpoint`` off the same object it
    was already being handed. That property is load-bearing: if counterpoint
    had to travel as a second return value, any caller not yet updated would
    silently drop it and render a file whose counterpoint quietly vanished.

    Note the one sharp edge: slicing (``doc[:3]``) yields a plain ``tuple``
    and therefore drops ``.counterpoint``. That is correct -- a subset of the
    transcript no longer has the spine lines a stream refers to -- but it is
    silent, so slice deliberately.
    """

    # No __slots__: a variable-length built-in subclass cannot have one, so
    # instances carry a __dict__ and the extra state lives there.

    def __new__(
        cls,
        lines: Iterable[LyricLine] = (),
        counterpoint: Iterable[CounterpointStream] = (),
    ) -> LyricsDocument:
        self = super().__new__(cls, lines)
        self._counterpoint = tuple(counterpoint)
        return self

    def __reduce__(self):  # noqa: D105 -- copy/pickle must keep the streams
        # Without this, copy/deepcopy/pickle rebuild a tuple subclass from its
        # items alone and the counterpoint would vanish silently -- the exact
        # failure mode this class exists to prevent.
        return (self.__class__, (tuple(self), self._counterpoint))

    @property
    def counterpoint(self) -> tuple[CounterpointStream, ...]:
        """Every non-spine stream in the file, in file order. Empty for a
        level-1/level-2 file, which is every file written before #33."""
        return self._counterpoint

    @property
    def lines(self) -> tuple[LyricLine, ...]:
        """The transcript itself, as a plain tuple."""
        return tuple(self)

    @property
    def has_counterpoint(self) -> bool:
        return bool(self._counterpoint)

    def counterpoint_for_line(self, line_index: int) -> tuple[CounterpointStream, ...]:
        """Every stream sung concurrently with the spine line ``line_index``.

        ``line_index`` is ``LyricLine.index`` -- transcript index space. The
        lookup exists so a downstream stage asking "is anyone else singing
        here?" does not have to re-derive the answer by scanning
        ``spine_line_indices`` itself and get the index space wrong.
        """
        return tuple(
            stream for stream in self._counterpoint if line_index in stream.spine_line_indices
        )


def parse_lyrics(
    lyrics_file: Path | str,
    cast: Mapping[str, CastMember],
    default_lead_vocalist: str,
) -> LyricsDocument:
    """Parse a lyrics file on disk into tag-stripped, character-resolved lines.

    ``cast`` and ``default_lead_vocalist`` are passed explicitly rather than a
    full ``RunConfig`` so this stays independently testable and reusable.
    """
    path = Path(lyrics_file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to read lyrics file: %s", path)
        raise
    return parse_lyrics_text(text, cast, default_lead_vocalist)


def parse_lyrics_text(
    text: str,
    cast: Mapping[str, CastMember],
    default_lead_vocalist: str,
) -> LyricsDocument:
    """Parse raw lyrics text into tag-stripped, character-resolved lines.

    Blank lines are skipped entirely (they neither switch the active
    character nor produce a :class:`LyricLine`). Every other non-blank line
    becomes exactly one ``LyricLine`` -- including bracketed instrumental
    markers such as ``(guitar solo)``, which forced alignment (issue #3) is
    expected to time against instrumental audio rather than vocals.

    Level 3 (issue #33): lines inside a ``[simultaneously]`` block belong to
    the block's *first* sub-block (the alignment spine) and are emitted as
    ordinary ``LyricLine`` objects in the normal index space; every later
    sub-block is returned as a :class:`CounterpointStream` on the
    :class:`LyricsDocument`, never as transcript text.
    """
    state = _ParseState(cast=dict(cast), default_lead_vocalist=default_lead_vocalist)

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        state.feed(stripped)

    state.finish()

    characters_used = sorted(
        {name for line in state.lines for name in line.characters}
        | {name for stream in state.streams for name in stream.characters}
    )
    logger.info(
        "Parsed %d lyric line(s); %d counterpoint stream(s) across %d "
        "[simultaneously] block(s); characters used: %s",
        len(state.lines),
        len(state.streams),
        state.blocks_seen,
        characters_used,
    )
    return LyricsDocument(state.lines, state.streams)


class _SubBlock:
    """One ``[Name]``-tagged sub-block inside a ``[simultaneously]`` block."""

    def __init__(self, characters: tuple[str, ...]) -> None:
        self.characters = characters
        self.texts: list[str] = []
        self.line_indices: list[int] = []


class _ParseState:
    """The line-at-a-time parser, with just enough state for level-3 blocks.

    Split out of :func:`parse_lyrics_text` so the flat (level 1/2) path and
    the block path can be read independently instead of as one nested loop.
    """

    def __init__(self, cast: Mapping[str, CastMember], default_lead_vocalist: str) -> None:
        self._cast = cast
        self._default_lead_vocalist = default_lead_vocalist
        self.lines: list[LyricLine] = []
        self.streams: list[CounterpointStream] = []
        self.blocks_seen = 0
        self._index = 0
        self._active_characters: tuple[str, ...] = ()

        # Level-3 block state. ``_block_subs`` is None when not inside a block.
        self._block_subs: list[_SubBlock] | None = None
        self._block_index = 0
        self._characters_before_block: tuple[str, ...] = ()

    # -- driving ---------------------------------------------------------- #

    def feed(self, stripped: str) -> None:
        match = _TAG_RE.match(stripped)
        if match:
            name_field = match.group("name").strip()
            rest = match.group("rest").strip()
            keyword = name_field.lower()
            if keyword == _BLOCK_OPEN_KEYWORD:
                self._open_block(rest)
                return
            if keyword == _BLOCK_CLOSE_KEYWORD:
                self._close_block(rest)
                return
            self._tag(name_field, rest, stripped)
            return
        self._content(stripped)

    def finish(self) -> None:
        if self._block_subs is not None:
            self._fail(
                f"unclosed '[{_BLOCK_OPEN_KEYWORD}]' block: the file ends without a "
                f"'[{_BLOCK_CLOSE_KEYWORD}]' -- every block must be closed explicitly "
                f"(see {_LEVEL3_ISSUE_URL})"
            )

    # -- level 1/2 -------------------------------------------------------- #

    def _tag(self, name_field: str, rest: str, raw: str) -> None:
        names = _parse_names(name_field)
        for name in names:
            _validate_character(name, self._cast)

        if self._block_subs is not None:
            # Inside a block, a tag opens a new sub-block rather than
            # switching the active character of the transcript.
            if rest:
                self._fail(
                    f"character tag '[{name_field}]' inside a '[{_BLOCK_OPEN_KEYWORD}]' "
                    "block must be on its own line, with its lyric lines beneath it"
                )
            self._block_subs.append(_SubBlock(names))
            return

        self._active_characters = names
        if rest:
            self._emit(rest, names, raw)

    def _content(self, stripped: str) -> None:
        if self._block_subs is not None:
            self._block_content(stripped)
            return
        characters = (
            self._active_characters
            if self._active_characters
            else (self._default_lead_vocalist,)
        )
        for name in characters:
            _validate_character(name, self._cast)
        self._emit(stripped, characters, stripped)

    def _emit(self, content: str, characters: tuple[str, ...], raw: str) -> int:
        line_index = self._index
        self.lines.append(
            LyricLine(index=line_index, text=content, characters=characters, raw=raw)
        )
        self._index += 1
        return line_index

    # -- level 3 ---------------------------------------------------------- #

    def _open_block(self, rest: str) -> None:
        if self._block_subs is not None:
            self._fail(
                f"'[{_BLOCK_OPEN_KEYWORD}]' blocks cannot nest -- close the open block "
                f"with '[{_BLOCK_CLOSE_KEYWORD}]' first"
            )
        if rest:
            self._fail(
                f"'[{_BLOCK_OPEN_KEYWORD}]' must be on its own line; found content after "
                f"it ({rest!r})"
            )
        self._characters_before_block = self._active_characters
        self._block_subs = []

    def _block_content(self, stripped: str) -> None:
        assert self._block_subs is not None
        if not self._block_subs:
            self._fail(
                f"a '[{_BLOCK_OPEN_KEYWORD}]' block must open with a character tag; "
                f"found the line {stripped!r} before any '[Name]' tag"
            )
        sub = self._block_subs[-1]
        if len(self._block_subs) == 1:
            # First sub-block == the alignment spine: it *is* the transcript
            # for this span, so it goes into the ordinary line index space.
            sub.line_indices.append(self._emit(stripped, sub.characters, stripped))
        else:
            sub.texts.append(stripped)

    def _close_block(self, rest: str) -> None:
        if self._block_subs is None:
            self._fail(
                f"'[{_BLOCK_CLOSE_KEYWORD}]' without a matching '[{_BLOCK_OPEN_KEYWORD}]'"
            )
        if rest:
            self._fail(
                f"'[{_BLOCK_CLOSE_KEYWORD}]' must be on its own line; found content after "
                f"it ({rest!r})"
            )
        subs = self._block_subs
        assert subs is not None
        if len(subs) < 2:
            self._fail(
                f"a '[{_BLOCK_OPEN_KEYWORD}]' block needs at least two voices singing "
                f"different words (found {len(subs)}); for one voice, use a plain "
                "'[Name]' tag instead"
            )
        spine = subs[0]
        for stream_index, sub in enumerate(subs):
            content = sub.line_indices if stream_index == 0 else sub.texts
            if not content:
                self._fail(
                    f"sub-block '[{' & '.join(sub.characters)}]' inside a "
                    f"'[{_BLOCK_OPEN_KEYWORD}]' block has no lyric lines"
                )
        for stream_index, sub in enumerate(subs[1:], start=1):
            self.streams.append(
                CounterpointStream(
                    block_index=self._block_index,
                    stream_index=stream_index,
                    characters=sub.characters,
                    texts=tuple(sub.texts),
                    spine_line_indices=tuple(spine.line_indices),
                )
            )

        self._block_index += 1
        self.blocks_seen += 1
        self._block_subs = None
        # A block has no single "last voice", so inheriting one of the
        # concurrent voices would silently privilege one of them. The tag in
        # force before the block is restored instead; anything else should be
        # tagged explicitly after the block.
        self._active_characters = self._characters_before_block

    def _fail(self, message: str) -> None:
        logger.error("Malformed lyrics file: %s", message)
        raise LyricsError(message)


def _parse_names(name_field: str) -> tuple[str, ...]:
    """Split a tag's name field on ``&`` into one or more character names.

    A bare ``[Name]`` (no ``&``) yields a single-element tuple, so level 1
    tags are handled by exactly the same code path as level 2 -- there is no
    separate "solo" branch to keep in sync. Whitespace around ``&`` is
    free-form (issue #33): ``"A&B"``, ``"A & B"`` and ``"A   &   B"`` are
    equivalent.
    """
    names = tuple(part.strip() for part in name_field.split("&"))
    if not names or any(not name for name in names):
        logger.error("Malformed character tag %r: empty name around '&'", name_field)
        raise LyricsError(f"malformed character tag {name_field!r}: empty name around '&'")
    return names


def _validate_character(name: str, cast: Mapping[str, CastMember]) -> None:
    if name not in cast:
        known = sorted(cast)
        logger.error("Unknown character %r referenced in lyrics (known cast: %s)", name, known)
        raise LyricsError(
            f"unknown character {name!r} in lyrics; known cast members: {known}"
        )


class LyricCastResolver:
    """Concrete :class:`music_video_maker.contracts.CastResolver` (issue #6).

    Resolves the :class:`CastMember` active for a given ``LyricLine.index``.
    Downstream stages carry that same ``index`` through to
    :class:`~music_video_maker.contracts.AlignedSegment` (issue #3), so this
    resolver is the seam Stage 2 prompt expansion (issue #5) uses to fetch the
    right reference image per segment.
    """

    def __init__(self, lines: Sequence[LyricLine], cast: Mapping[str, CastMember]) -> None:
        self._character_by_index: dict[int, str | None] = {
            line.index: line.character for line in lines
        }
        self._cast = dict(cast)

    def resolve(self, line_index: int) -> CastMember:
        if line_index not in self._character_by_index:
            logger.error("No parsed lyric line with index=%d to resolve cast for", line_index)
            raise LyricsError(f"no lyric line with index {line_index}")

        character = self._character_by_index[line_index]
        if character is None or character not in self._cast:
            known = sorted(self._cast)
            logger.error(
                "Line index=%d resolved to character %r, not in cast (known: %s)",
                line_index,
                character,
                known,
            )
            raise LyricsError(f"character {character!r} not found in cast: {known}")

        return self._cast[character]
