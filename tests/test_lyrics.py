"""Tests for lyric character-tag parsing (issue #6).

Fully offline: reads the committed lyrics fixtures under
``tests/fixtures/lyrics/`` and builds cast dicts via
``tests.harness.factories``. No network, no GPU, no stable-ts/torch import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from music_video_maker.contracts import LyricLine
from music_video_maker.lyrics import LyricCastResolver, LyricsError, parse_lyrics, parse_lyrics_text
from tests.harness.factories import (
    LYRICS_GAPS_PATH,
    LYRICS_MIXED_INHERIT_PATH,
    LYRICS_PLAIN_PATH,
    LYRICS_TAGGED_PATH,
    LYRICS_UNKNOWN_CHARACTER_PATH,
    make_cast_dict,
)

CAST = make_cast_dict()
DEFAULT_LEAD = "Dianne"

# Level 2 (`[Name & Name]`) fixtures. Not part of tests.harness.factories
# (that module is owned by another lane) -- built locally against the same
# tests/fixtures/lyrics/ directory those constants point at.
_LYRICS_DIR = Path(__file__).resolve().parent / "fixtures" / "lyrics"
LYRICS_DUET_PATH = _LYRICS_DIR / "duet.txt"
LYRICS_DUET_WHITESPACE_PATH = _LYRICS_DIR / "duet_whitespace.txt"
LYRICS_UNKNOWN_CHARACTER_DUET_PATH = _LYRICS_DIR / "unknown_character_duet.txt"
LYRICS_SIMULTANEOUSLY_BLOCK_PATH = _LYRICS_DIR / "simultaneously_block.txt"


def test_plain_untagged_lines_use_default_lead_vocalist():
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)

    assert len(lines) == 7
    assert all(line.character == "Dianne" for line in lines)
    assert lines[0].text == "Walking through the empty halls tonight"
    # No source line in plain.txt has a tag, so text and raw coincide.
    assert lines[0].raw == lines[0].text


def test_plain_lines_are_sequentially_indexed():
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)

    assert [line.index for line in lines] == list(range(len(lines)))
    assert all(isinstance(line, LyricLine) for line in lines)


def test_tagged_lines_switch_active_vocalist():
    lines = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)

    assert [line.character for line in lines] == [
        "Dianne",
        "Marcus",
        "Dianne",
        "Marcus",
        "Dianne",
    ]
    assert lines[0].text == "The lucky ones don't ever have to try"
    assert lines[1].text == "We're watching from the wings tonight"


def test_tags_never_reach_the_stripped_text():
    for path in (LYRICS_TAGGED_PATH, LYRICS_MIXED_INHERIT_PATH, LYRICS_GAPS_PATH):
        lines = parse_lyrics(path, CAST, DEFAULT_LEAD)
        for line in lines:
            assert "[" not in line.text
            assert "]" not in line.text


def test_untagged_lines_inherit_across_several_lines():
    lines = parse_lyrics(LYRICS_MIXED_INHERIT_PATH, CAST, DEFAULT_LEAD)

    # [Dianne] x2, [Marcus] x3, [Dianne] x2 -- inheritance spans multiple
    # untagged lines within each block, per mixed_inherit.txt.
    assert [line.character for line in lines] == [
        "Dianne",
        "Dianne",
        "Marcus",
        "Marcus",
        "Marcus",
        "Dianne",
        "Dianne",
    ]


def test_inline_tag_and_text_on_one_line_strips_tag_and_sets_character():
    text = "[Marcus: Backup] Yo, catch this line\nStill Marcus talking"
    lines = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert len(lines) == 2
    assert lines[0].character == "Marcus"
    assert lines[0].text == "Yo, catch this line"
    assert lines[0].raw == "[Marcus: Backup] Yo, catch this line"
    # Second line inherits Marcus with no tag present.
    assert lines[1].character == "Marcus"
    assert lines[1].raw == "Still Marcus talking"


def test_blank_lines_are_skipped_and_do_not_consume_index():
    text = "[Dianne: Lead]\nFirst line\n\n\nSecond line\n"
    lines = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert [line.index for line in lines] == [0, 1]
    assert [line.text for line in lines] == ["First line", "Second line"]


def test_instrumental_marker_lines_are_kept_as_lyric_lines_with_inherited_character():
    lines = parse_lyrics(LYRICS_GAPS_PATH, CAST, DEFAULT_LEAD)

    assert len(lines) == 6
    texts = [line.text for line in lines]
    assert "(long instrumental intro)" in texts
    assert "(extended instrumental bridge, guitar solo)" in texts

    intro = next(line for line in lines if line.text == "(long instrumental intro)")
    assert intro.character == "Dianne"

    bridge = next(
        line for line in lines if line.text == "(extended instrumental bridge, guitar solo)"
    )
    assert bridge.character == "Marcus"


def test_unknown_character_raises_and_logs(caplog):
    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.lyrics"),
        pytest.raises(LyricsError) as exc_info,
    ):
        parse_lyrics(LYRICS_UNKNOWN_CHARACTER_PATH, CAST, DEFAULT_LEAD)

    assert "Ghost" in str(exc_info.value)
    for name in sorted(CAST):
        assert name in str(exc_info.value)
    assert any("Ghost" in record.message for record in caplog.records)


def test_unknown_default_lead_vocalist_raises_on_first_untagged_line():
    text = "No tag here, just words"
    with pytest.raises(LyricsError):
        parse_lyrics_text(text, CAST, "NotInCast")


def test_first_untagged_lines_before_any_tag_use_default_lead():
    text = "Untagged opener\n[Marcus: Backup]\nNow Marcus"
    lines = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert lines[0].character == DEFAULT_LEAD
    assert lines[1].character == "Marcus"


def test_cast_resolver_resolves_cast_member_per_line_index():
    lines = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)
    resolver = LyricCastResolver(lines, CAST)

    dianne = resolver.resolve(0)
    marcus = resolver.resolve(1)

    assert dianne.name == "Dianne"
    assert marcus.name == "Marcus"
    assert dianne.image == CAST["Dianne"].image


def test_cast_resolver_raises_for_unknown_index():
    lines = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)
    resolver = LyricCastResolver(lines, CAST)

    with pytest.raises(LyricsError):
        resolver.resolve(999)


def test_cast_resolver_raises_when_resolved_character_not_in_cast():
    # Built directly (bypassing parse_lyrics's own validation) to exercise the
    # resolver's own defensive check against a cast dict it wasn't built from.
    lines = (LyricLine(index=0, text="hello", characters=("Nobody",)),)
    resolver = LyricCastResolver(lines, CAST)

    with pytest.raises(LyricsError):
        resolver.resolve(0)


def test_parse_lyrics_missing_file_logs_and_raises(tmp_path, caplog):
    missing = tmp_path / "does-not-exist.txt"

    with caplog.at_level(logging.ERROR, logger="music_video_maker.lyrics"), pytest.raises(OSError):
        parse_lyrics(missing, CAST, DEFAULT_LEAD)

    assert any("Failed to read lyrics file" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# Level 2 (issue #33): "[Name & Name]" -- same words, both voices audible.
# --------------------------------------------------------------------------- #


def test_ampersand_tag_produces_both_characters_in_written_order():
    lines = parse_lyrics(LYRICS_DUET_PATH, CAST, DEFAULT_LEAD)

    assert lines[0].characters == ("Dianne", "Marcus")
    assert lines[0].character == "Dianne"  # first name is primary
    assert lines[0].text == "Singing this together tonight"


def test_ampersand_tag_inherits_across_following_untagged_lines():
    lines = parse_lyrics(LYRICS_DUET_PATH, CAST, DEFAULT_LEAD)

    assert lines[1].characters == ("Dianne", "Marcus")
    assert lines[1].text == "Every word the same, every voice as one"


def test_ampersand_tag_with_role_slot_still_parses_and_role_is_unused():
    lines = parse_lyrics(LYRICS_DUET_PATH, CAST, DEFAULT_LEAD)

    # "[Dianne & Marcus: Chorus] Lord knows ..." -- inline tag+text, role text
    # ("Chorus") is accepted syntactically but does not appear anywhere.
    chorus_line = next(
        line for line in lines if line.text == "Lord knows when it's people like you"
    )
    assert chorus_line.characters == ("Dianne", "Marcus")
    assert "Chorus" in chorus_line.raw  # survives in raw (diagnostics only)
    for line in lines:
        assert "Chorus" not in line.text  # never in the tag-stripped text


def test_ampersand_tag_can_be_followed_by_a_solo_tag():
    lines = parse_lyrics(LYRICS_DUET_PATH, CAST, DEFAULT_LEAD)

    solo_line = next(line for line in lines if line.text == "But this part is mine alone")
    assert solo_line.characters == ("Dianne",)


def test_ampersand_whitespace_is_free_form():
    lines = parse_lyrics(LYRICS_DUET_WHITESPACE_PATH, CAST, DEFAULT_LEAD)

    assert len(lines) == 3
    assert all(line.characters == ("Dianne", "Marcus") for line in lines)


def test_ampersand_tag_validates_every_name_against_the_cast():
    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics(LYRICS_UNKNOWN_CHARACTER_DUET_PATH, CAST, DEFAULT_LEAD)

    assert "Ghost" in str(exc_info.value)


def test_ampersand_tag_validates_before_any_lines_are_produced():
    # A typo'd second singer must fail at config load, before any GPU time --
    # not partway through, and not silently dropping the bad name.
    text = "[Dianne & Ghost]\nSome words\n"
    with pytest.raises(LyricsError):
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)


def test_ampersand_tags_never_reach_the_stripped_text():
    for path in (LYRICS_DUET_PATH, LYRICS_DUET_WHITESPACE_PATH):
        lines = parse_lyrics(path, CAST, DEFAULT_LEAD)
        for line in lines:
            assert "[" not in line.text
            assert "]" not in line.text
            assert "&" not in line.text


def test_malformed_ampersand_tag_with_empty_name_raises():
    text = "[Dianne & ]\nSome words\n"
    with pytest.raises(LyricsError):
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)


# --------------------------------------------------------------------------- #
# Level 3 (issue #33): "[simultaneously]" blocks -- different words, same time.
#
# The first sub-block is the alignment spine: its lines are ordinary
# LyricLines in the returned sequence (that is the one linear transcript
# stable-ts gets). Every later sub-block is a CounterpointStream carried
# alongside, and never appears in the transcript.
# --------------------------------------------------------------------------- #

COUNTERPOINT_TEXT = """\
[Dianne]
Before the block
[simultaneously]
  [Dianne]
  There was a time
  when I hoped
  your printer
  would explode somehow.
  [Marcus]
  I know when it's people like you
  Hating people like me
  I'm the lucky one.
[/simultaneously]
[Marcus]
I'm the lucky one.
"""


def test_simultaneously_block_spine_lines_are_ordinary_lyric_lines():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert [line.text for line in doc] == [
        "Before the block",
        "There was a time",
        "when I hoped",
        "your printer",
        "would explode somehow.",
        "I'm the lucky one.",
    ]
    # Spine lines carry the spine sub-block's characters.
    assert [line.characters for line in doc[1:5]] == [("Dianne",)] * 4


def test_simultaneously_spine_lines_keep_one_contiguous_index_space():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert [line.index for line in doc] == list(range(len(doc)))


def test_counterpoint_text_never_appears_in_the_transcript_lines():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    transcript = "\n".join(line.text for line in doc)
    assert "Hating people like me" not in transcript
    assert "I know when it's people like you" not in transcript


def test_counterpoint_stream_is_captured_with_its_characters_and_spine_lines():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert len(doc.counterpoint) == 1
    stream = doc.counterpoint[0]
    assert stream.characters == ("Marcus",)
    assert stream.character == "Marcus"
    assert stream.texts == (
        "I know when it's people like you",
        "Hating people like me",
        "I'm the lucky one.",
    )
    assert stream.block_index == 0
    assert stream.stream_index == 1
    # Spine lines 1-4 of the document (index space: LyricLine.index).
    assert stream.spine_line_indices == (1, 2, 3, 4)


def test_counterpoint_streams_need_not_be_line_aligned_with_the_spine():
    # Dianne's block is 4 lines, Marcus's is 3 -- concurrent streams are not
    # paired line by line, and the parser must not try.
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert len(doc.counterpoint[0].spine_line_indices) == 4
    assert len(doc.counterpoint[0].texts) == 3


def test_document_is_a_plain_tuple_of_lyric_lines_for_every_existing_caller():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert isinstance(doc, tuple)
    assert all(isinstance(line, LyricLine) for line in doc)
    assert doc == tuple(doc.lines)


def test_file_without_any_block_has_no_counterpoint():
    doc = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)

    assert doc.counterpoint == ()


def test_simultaneously_block_fixture_parses_end_to_end():
    doc = parse_lyrics(LYRICS_SIMULTANEOUSLY_BLOCK_PATH, CAST, DEFAULT_LEAD)

    # "[Dianne & Marcus] Lord knows..." then a block with a Dianne spine and a
    # Marcus counterpoint stream.
    assert doc[0].characters == ("Dianne", "Marcus")
    assert len(doc) == 4  # 1 pre-block line + 3 spine lines
    assert len(doc.counterpoint) == 1
    assert doc.counterpoint[0].characters == ("Marcus",)
    assert doc.counterpoint[0].spine_line_indices == (1, 2, 3)


def test_three_sub_blocks_produce_two_counterpoint_streams():
    text = (
        "[simultaneously]\n"
        "[Dianne]\nSpine one\nSpine two\n"
        "[Marcus]\nSecond voice\n"
        "[Rex]\nThird voice\n"
        "[/simultaneously]\n"
    )
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert [line.text for line in doc] == ["Spine one", "Spine two"]
    assert [s.stream_index for s in doc.counterpoint] == [1, 2]
    assert [s.characters for s in doc.counterpoint] == [("Marcus",), ("Rex",)]


def test_level_2_tag_is_allowed_as_a_sub_block():
    text = (
        "[simultaneously]\n"
        "[Dianne]\nSpine line\n"
        "[Marcus & Rex]\nBoth of us underneath\n"
        "[/simultaneously]\n"
    )
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert doc.counterpoint[0].characters == ("Marcus", "Rex")


def test_block_tags_are_case_insensitive():
    text = "[SIMULTANEOUSLY]\n[Dianne]\nSpine\n[Marcus]\nUnder\n[/Simultaneously]\n"
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert [line.text for line in doc] == ["Spine"]
    assert doc.counterpoint[0].texts == ("Under",)


def test_blank_lines_inside_a_block_are_skipped():
    text = "[simultaneously]\n\n[Dianne]\n\nSpine\n\n[Marcus]\n\nUnder\n\n[/simultaneously]\n"
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert [line.text for line in doc] == ["Spine"]
    assert doc.counterpoint[0].texts == ("Under",)


def test_active_character_after_a_block_is_the_one_from_before_it():
    text = (
        "[Rex]\nBefore\n"
        "[simultaneously]\n[Dianne]\nSpine\n[Marcus]\nUnder\n[/simultaneously]\n"
        "After the block\n"
    )
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert doc[0].characters == ("Rex",)
    assert doc[1].characters == ("Dianne",)  # spine
    # Neither concurrent voice silently wins the inheritance: the pre-block
    # tag is restored, so an untagged line after the block is unambiguous.
    assert doc[2].text == "After the block"
    assert doc[2].characters == ("Rex",)


def test_block_tags_never_reach_the_stripped_text():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    for line in doc:
        assert "[" not in line.text
        assert "]" not in line.text
    for stream in doc.counterpoint:
        for stream_text in stream.texts:
            assert "[" not in stream_text
            assert "]" not in stream_text


def test_unknown_character_inside_a_block_fails_loudly(caplog):
    text = "[simultaneously]\n[Dianne]\nSpine\n[Ghost]\nUnder\n[/simultaneously]\n"

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.lyrics"),
        pytest.raises(LyricsError) as exc_info,
    ):
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "Ghost" in str(exc_info.value)


def test_unclosed_block_raises_and_names_the_block(caplog):
    text = "[simultaneously]\n[Dianne]\nSpine\n[Marcus]\nUnder\n"

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.lyrics"),
        pytest.raises(LyricsError) as exc_info,
    ):
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "simultaneously" in str(exc_info.value).lower()
    assert any("simultaneously" in r.message.lower() for r in caplog.records)


def test_closing_tag_without_an_open_block_raises():
    text = "[Dianne]\nA line\n[/simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "/simultaneously" in str(exc_info.value).lower()


def test_nested_block_raises():
    text = "[simultaneously]\n[Dianne]\nSpine\n[simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "nest" in str(exc_info.value).lower()


def test_block_content_before_the_first_sub_block_tag_raises():
    text = "[simultaneously]\nOrphan line\n[Dianne]\nSpine\n[/simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "orphan line" in str(exc_info.value).lower() or "tag" in str(exc_info.value).lower()


def test_block_with_a_single_sub_block_raises():
    text = "[simultaneously]\n[Dianne]\nOnly one voice here\n[/simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    message = str(exc_info.value).lower()
    assert "two" in message or "one" in message


def test_block_with_an_empty_sub_block_raises():
    text = "[simultaneously]\n[Dianne]\nSpine\n[Marcus]\n[/simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "Marcus" in str(exc_info.value)


def test_content_on_the_block_open_tag_line_raises():
    text = (
        "[simultaneously] with words after it\n"
        "[Dianne]\nSpine\n[Marcus]\nUnder\n[/simultaneously]\n"
    )

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "own line" in str(exc_info.value).lower() or "content" in str(exc_info.value).lower()


def test_cast_resolver_works_over_a_document_with_counterpoint():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)
    resolver = LyricCastResolver(doc, CAST)

    # Spine line -> spine voice. Counterpoint lines are not in this index
    # space at all, and must not be resolvable by a spine index.
    assert resolver.resolve(1).name == "Dianne"


def test_has_counterpoint_distinguishes_a_level_3_file():
    assert parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD).has_counterpoint
    assert not parse_lyrics(LYRICS_DUET_PATH, CAST, DEFAULT_LEAD).has_counterpoint


def test_inline_tag_and_text_inside_a_block_raises():
    # Ambiguous: is that line the sub-block's first line, or a level-1 line
    # that happens to sit in a block? Refuse rather than guess.
    text = "[simultaneously]\n[Dianne] Spine\n[Marcus]\nUnder\n[/simultaneously]\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "own line" in str(exc_info.value).lower()


def test_content_on_the_block_close_tag_line_raises():
    text = "[simultaneously]\n[Dianne]\nSpine\n[Marcus]\nUnder\n[/simultaneously] trailing\n"

    with pytest.raises(LyricsError) as exc_info:
        parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    assert "own line" in str(exc_info.value).lower()


def test_copying_a_document_keeps_its_counterpoint():
    import copy

    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert copy.deepcopy(doc).counterpoint == doc.counterpoint


def test_counterpoint_for_line_answers_in_lyric_line_index_space():
    doc = parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert doc.counterpoint_for_line(2) == (doc.counterpoint[0],)
    assert doc.counterpoint_for_line(0) == ()  # the line before the block
    assert doc.counterpoint_for_line(5) == ()  # the line after it
    assert doc.counterpoint_for_line(999) == ()


def test_parse_summary_logs_counterpoint_stream_count(caplog):
    with caplog.at_level(logging.INFO, logger="music_video_maker.lyrics"):
        parse_lyrics_text(COUNTERPOINT_TEXT, CAST, DEFAULT_LEAD)

    assert any("counterpoint" in r.message.lower() for r in caplog.records)


def test_lines_after_a_block_inherit_the_pre_block_voice_not_a_block_voice():
    """Design decision on issue #33 level 3: a ``[simultaneously]`` block has
    no single "last voice", so inheriting one of its sub-block tags would
    silently privilege one of two concurrent voices. The tag in force
    *before* the block is restored -- an untagged line after the block is the
    pre-block singer's, not the final sub-block's."""
    text = (
        "[Dianne]\n"
        "Before the block\n"
        "[simultaneously]\n"
        "  [Dianne]\n"
        "  There was a time\n"
        "  [Marcus]\n"
        "  I know when it's people like you\n"
        "[/simultaneously]\n"
        "After the block with no tag\n"
    )
    doc = parse_lyrics_text(text, CAST, DEFAULT_LEAD)

    after = [line for line in doc if line.text == "After the block with no tag"]
    assert len(after) == 1
    assert after[0].characters == ("Dianne",), (
        "the post-block line inherited a voice from inside the block instead of "
        "restoring the pre-block tag"
    )
