"""Tests for the repeat/section-directive lint in lyrics parsing (issue #106).

Published lyrics abbreviate what was sung -- ``(repeat chorus)``,
``Chorus x2``, ``[Verse 2]`` -- and none of those words are ever sung, so
none of them can be a correct line of the transcript forced alignment times
against. The parser refuses them at load, before any GPU time, rather than
handing them to the aligner as words. Documented instrumental parentheticals
and ordinary lyrics that merely start with a section word must still parse.

Fully offline, same as ``tests/test_lyrics.py``.
"""

from __future__ import annotations

import pytest

from music_video_maker.lyrics import LyricsError, parse_lyrics_text
from tests.harness.factories import make_cast_dict

CAST = make_cast_dict()
DEFAULT_LEAD = "Dianne"


def _parse(text: str):
    return parse_lyrics_text(text, CAST, DEFAULT_LEAD)


def _with(directive: str) -> str:
    return f"Count the days until the sun comes back\n{directive}\nNothing but the echo of a drum\n"


@pytest.mark.parametrize(
    "directive",
    [
        "(repeat chorus)",
        "(Repeat Chorus)",
        "(repeat)",
        "(repeat x2)",
        "(rpt chorus)",
        "(x2)",
        "(X3)",
        "(2x)",
        "(×2)",
        "(Chorus x2)",
        "Repeat chorus",
        "Repeat",
        "repeat x2",
        "Chorus x2",
        "Chorus (x2)",
        "Nothing but the echo x2",
        "Nothing but the echo (x4)",
        "Nothing but the echo ×3",
    ],
)
def test_a_repeat_directive_is_refused(directive):
    with pytest.raises(LyricsError, match="(?i)repeat"):
        _parse(_with(directive))


@pytest.mark.parametrize(
    "header",
    [
        "Chorus",
        "Chorus:",
        "CHORUS:",
        "Verse 2",
        "Verse 2:",
        "Pre-Chorus",
        "Pre-chorus:",
        "Post-Chorus",
        "Bridge:",
        "Refrain",
        "Hook:",
        "Intro:",
        "Outro",
        "(Chorus)",
        "(Verse 1)",
        "(Refrain)",
        "(Pre-Chorus)",
        "[Chorus]",
        "[Verse 2]",
        "[Chorus: Dianne]",
        "[Pre-Chorus]",
        "[Bridge]",
        "[Outro]",
    ],
)
def test_a_section_header_is_refused(header):
    with pytest.raises(LyricsError, match="(?i)section header"):
        _parse(_with(header))


def test_a_section_header_with_lyric_after_the_colon_is_refused():
    with pytest.raises(LyricsError, match="(?i)section header"):
        _parse("Chorus: Count the days until the sun comes back\n")


def test_a_bracketed_header_does_not_blame_the_cast():
    # Before #106, ``[Chorus]`` was refused as an unknown character, which
    # sent the operator to the cast table instead of the lyrics file.
    with pytest.raises(LyricsError) as excinfo:
        _parse(_with("[Chorus]"))
    assert "cast" not in str(excinfo.value).lower()


def test_a_directive_inside_a_simultaneously_block_is_refused():
    text = (
        "[simultaneously]\n"
        "[Dianne]\n"
        "Count the days\n"
        "(repeat)\n"
        "[Marcus]\n"
        "Nothing but the echo\n"
        "[/simultaneously]\n"
    )
    with pytest.raises(LyricsError, match="(?i)repeat"):
        _parse(text)


def test_the_refusal_names_the_line_and_says_to_write_it_out():
    with pytest.raises(LyricsError) as excinfo:
        _parse(_with("(repeat chorus)"))
    message = str(excinfo.value)
    assert "(repeat chorus)" in message
    assert "line 2" in message
    assert "as sung" in message


@pytest.mark.parametrize(
    "line",
    [
        # Documented instrumental markers (docs/lyrics-format.md).
        "(long instrumental intro)",
        "(extended instrumental bridge, guitar solo)",
        "(guitar solo)",
        "(instrumental)",
        "(Intro)",
        "(Outro)",
        "(Bridge)",
        "(Instrumental break)",
        # Ordinary lyrics that start with, or contain, a section word.
        "Repeat after me",
        "Repeat the words she said to me",
        "Bridge over troubled water",
        "The chorus of the morning birds",
        "Verse and chorus, all night long",
        "Hook, line and sinker",
        "Outro of a life, and the lights go down",
        "Malcolm X",
        "Generation X2 is waiting",
        "I counted 1, 2, 3",
        "(oh, oh, oh)",
        "(Come back to me)",
        # From real lyrics files on hand when #106 shipped.
        "This transmission will not be repeated.",
        "(sax solo)",
    ],
)
def test_instrumental_markers_and_ordinary_lyrics_still_parse(line):
    lines = _parse(_with(line))
    assert [entry.text for entry in lines][1] == line


def test_a_section_word_in_a_tags_role_slot_is_not_a_header():
    # Deathless's lyrics tag a group voice as "[The Dead: chorus]": the role
    # slot is free text, and only the name decides whether a tag is a header.
    lines = _parse("[Marcus: chorus]\nCount the days until the sun comes back\n")
    assert lines[0].character == "Marcus"


def test_a_cast_member_named_like_a_section_is_still_a_character_tag():
    cast = make_cast_dict()
    cast["Hook"] = next(iter(cast.values()))
    lines = parse_lyrics_text("[Hook]\nCount the days\n", cast, DEFAULT_LEAD)
    assert lines[0].character == "Hook"
