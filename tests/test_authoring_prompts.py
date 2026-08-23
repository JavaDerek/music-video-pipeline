"""Tests for the pure prompt-composition helpers in
:mod:`music_video_maker.authoring.prompts` -- issue #83's third continuity
axis (``conditions``) and issue #67's literalness band, both of which live
entirely in this module's static text and its three parameterised
``*_system_prompt`` composers.

No model, no driver, no filesystem beyond the real docs this module already
reads from disk (design section 9: "the docs are the system prompt") -- every
assertion here is a substring check on composed text.
"""

from __future__ import annotations

import pytest

from music_video_maker.authoring.prompts import (
    BEATS_PREAMBLE,
    CONCEPT_PREAMBLE,
    LITERALNESS_BLOCKS,
    beats_system_prompt,
    concept_system_prompt,
    literalness_block,
    photography_system_prompt,
    prose_system_prompt,
)
from music_video_maker.config import DEFAULT_LYRIC_LITERALNESS, LYRIC_LITERALNESS_BANDS

# --------------------------------------------------------------------------- #
# Issue #83: `conditions`, the third continuity axis
# --------------------------------------------------------------------------- #


def test_concept_preamble_reply_shape_includes_conditions():
    assert '"conditions"' in CONCEPT_PREAMBLE


def test_concept_preamble_explains_conditions_as_the_third_axis():
    lowered = CONCEPT_PREAMBLE.lower()
    assert "third continuity axis" in lowered
    assert "conditions" in lowered
    # The measured example from the real render that motivated #83.
    assert "snow" in lowered
    assert "debris" in lowered


def test_concept_preamble_says_name_at_least_one_condition():
    lowered = CONCEPT_PREAMBLE.lower()
    assert "nobody said" in lowered


def test_beats_preamble_reply_example_includes_conditions():
    assert '"conditions": "heavy falling snow"' in BEATS_PREAMBLE


def test_beats_preamble_has_rule_3c_for_conditions():
    assert "3c." in BEATS_PREAMBLE
    lowered = BEATS_PREAMBLE.lower()
    assert "approved `conditions`" in BEATS_PREAMBLE or "approved conditions" in lowered
    assert "consequence" in lowered


def test_beats_preamble_rule_3c_states_the_flip_flop_rule():
    lowered = BEATS_PREAMBLE.lower()
    assert "interrupted for exactly one chunk" in lowered or "exactly one" in lowered


def test_beats_preamble_rule_3c_states_the_one_way_regression_rule():
    lowered = BEATS_PREAMBLE.lower()
    assert "un-explode" in lowered
    assert "burnt valley" in lowered or "green over" in lowered


# --------------------------------------------------------------------------- #
# Issue #67: the literalness band
# --------------------------------------------------------------------------- #


def test_literalness_blocks_defines_every_band():
    assert set(LITERALNESS_BLOCKS) == set(LYRIC_LITERALNESS_BANDS)


@pytest.mark.parametrize("band", list(LYRIC_LITERALNESS_BANDS))
def test_literalness_block_returns_the_bands_own_text(band):
    assert literalness_block(band) == LITERALNESS_BLOCKS[band]
    assert band.upper() in LITERALNESS_BLOCKS[band]


def test_literalness_block_falls_back_to_the_default_for_an_unrecognised_band():
    """``load_config`` is the gate for the vocabulary (issue #67); a prompt
    composer must never be the thing that refuses a run over a bad string."""
    assert literalness_block("not-a-real-band") == LITERALNESS_BLOCKS[DEFAULT_LYRIC_LITERALNESS]
    assert literalness_block("") == LITERALNESS_BLOCKS[DEFAULT_LYRIC_LITERALNESS]
    assert literalness_block(None) == LITERALNESS_BLOCKS[DEFAULT_LYRIC_LITERALNESS]  # type: ignore[arg-type]


def test_free_band_says_the_video_may_share_no_imagery():
    lowered = LITERALNESS_BLOCKS["free"].lower()
    assert "free" in lowered
    assert "own idea" in lowered


def test_thematic_band_is_the_documented_middle_ground():
    lowered = LITERALNESS_BLOCKS["thematic"].lower()
    assert "thematic" in lowered
    assert "not a defect" in lowered or "surface where they land well" in lowered


def test_literal_band_says_every_named_object_must_appear():
    lowered = LITERALNESS_BLOCKS["literal"].lower()
    assert "literal" in lowered
    assert "error" in lowered


@pytest.mark.parametrize(
    ("system_prompt_fn", "expects_guide"),
    [
        (concept_system_prompt, False),
        (beats_system_prompt, True),
        (prose_system_prompt, True),
    ],
)
def test_each_parameterised_system_prompt_carries_its_bands_text(system_prompt_fn, expects_guide):
    for band in LYRIC_LITERALNESS_BANDS:
        system = system_prompt_fn(literalness=band)
        assert LITERALNESS_BLOCKS[band] in system


def test_the_literalness_section_sits_between_the_preamble_and_the_reference_doc():
    system = concept_system_prompt(literalness="literal")
    header = "# The brief: how literally this video reads its lyrics"
    assert header in system
    assert system.index(CONCEPT_PREAMBLE) < system.index(header)
    assert system.index(header) < system.index("# Reference: the lyrics file format")


def test_default_literalness_matches_config_default_with_no_argument_given():
    system = concept_system_prompt()
    assert LITERALNESS_BLOCKS[DEFAULT_LYRIC_LITERALNESS] in system


def test_beats_system_prompt_still_carries_the_shot_writing_guide_with_a_band_set():
    system = beats_system_prompt(literalness="free")
    assert "The three-beat rule" in system


def test_prose_system_prompt_still_carries_the_shot_writing_guide_with_a_band_set():
    system = prose_system_prompt(literalness="literal")
    assert "shot-writing guide" in system.lower()


def test_photography_system_prompt_carries_no_literalness_band():
    """Issue #67: photography decides framing, not what the words put on
    screen -- it is deliberately not parameterised by literalness at all."""
    system = photography_system_prompt()
    for block in LITERALNESS_BLOCKS.values():
        assert block not in system


def test_photography_system_prompt_takes_no_literalness_argument():
    import inspect

    params = inspect.signature(photography_system_prompt).parameters
    assert "literalness" not in params


def test_photography_system_prompt_docstring_explains_why():
    doc = (photography_system_prompt.__doc__ or "").lower()
    assert "literalness" in doc or "framing" in doc
