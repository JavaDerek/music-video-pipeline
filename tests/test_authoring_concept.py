"""Tests for Stage 1 -- concept (issue #54 design section 4).

No real model call anywhere in this file: :class:`ScriptedDriver` stands in
for the Claude CLI throughout. The interesting behaviour is the
retry-with-feedback loop -- a schema violation must feed the validator's own
error back into the next prompt and eventually give up, never loop forever
or silently accept a malformed reply.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from music_video_maker import contracts
from music_video_maker.authoring.chunks import load_chunk_skeleton
from music_video_maker.authoring.concept import (
    MAX_VALIDATION_ATTEMPTS,
    ConceptValidationError,
    build_concept_prompt,
    concept_input_hashes,
    generate_concept,
    validate_concept,
)
from music_video_maker.authoring.driver import DriverError, ScriptedDriver
from music_video_maker.authoring.prompts import LYRICS_FORMAT_DOC
from music_video_maker.config import RunConfig
from tests.harness.factories import make_cast_dict, make_raw_stablets_result, write_silent_wav

VALID_READING = {
    # Issue #69: the comprehension step, filled first and derived from
    # rather than a second freestanding pitch. `nouns`/`references` are
    # deliberately populated here (a happy-path fixture); the "empty is a
    # valid answer" behaviour has its own dedicated tests below.
    "subject": "a drummer waiting out a storm that finally breaks",
    "speaker": "the drummer",
    "addressee": "someone who has already left",
    "situation": "she is alone on the boardwalk as the weather turns",
    "change": "the storm builds, breaks, and clears, and she is still standing after",
    "register": "plain, contemporary",
    "period": "present day",
    "place": "a coastal town",
    "nouns": ["a snare drum", "a folding chair", "a length of rope"],
    "references": [
        {
            "reference": "a lighthouse",
            "what_it_is": "a navigational structure",
            "imagery": ["a beam sweeping the water", "a white tower against grey sky"],
        }
    ],
}

VALID_ACTS = [
    {"name": "situation", "function": "she waits alone as the storm gathers"},
    {"name": "resolution", "function": "the storm clears and what she planted pays off"},
]

VALID_CONCEPT = {
    "reading": VALID_READING,
    "logline": "A drummer walks through gathering weather as a storm builds and clears.",
    "setting": "a coastal town, contemporary",
    "tone": "elegiac, quiet",
    "motifs": ["gathering clouds", "an empty chair"],
    "avoid": ["literal lightning strikes"],
    # Issue #78: the closed vocabulary the beats stage assigns `location`
    # from -- a small, distinct set of places within `setting` the story
    # actually visits, not a monotonic progress scalar (the issue's own
    # "probably the version to build" framing).
    "locations": ["the boardwalk", "the empty pier", "her front porch"],
    # Issue #84: the video's dramatic shape, ordered.
    "acts": VALID_ACTS,
}


class _FakeAlignModel:
    """Stand-in for a stable-ts model -- only ever reached when lyrics.txt
    is non-empty, since align() short-circuits before touching the model
    otherwise (see chunks.py's docstring)."""

    def align(self, audio: str, text: str, **kwargs) -> SimpleNamespace:
        return make_raw_stablets_result()


def _config(tmp_path: Path, *, lyrics_text: str = "") -> RunConfig:
    master_audio = write_silent_wav(tmp_path / "audio" / "master.wav", seconds=20.0)
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text(lyrics_text, encoding="utf-8")
    return RunConfig(
        master_audio=master_audio,
        lyrics_file=lyrics_file,
        global_style="Refestramus progressive rock music video",
        narrative_concept="placeholder",
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "output" / "chunks",
        final_video_dir=tmp_path / "output" / "final",
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
    )


# --------------------------------------------------------------------------- #
# validate_concept
# --------------------------------------------------------------------------- #


def test_a_valid_concept_passes_silently():
    validate_concept(VALID_CONCEPT)  # must not raise


def test_a_non_dict_reply_is_rejected():
    with pytest.raises(ConceptValidationError):
        validate_concept(["not", "a", "dict"])


def test_a_missing_required_field_is_rejected():
    bad = {k: v for k, v in VALID_CONCEPT.items() if k != "tone"}
    with pytest.raises(ConceptValidationError, match="tone"):
        validate_concept(bad)


def test_an_empty_logline_is_rejected():
    bad = {**VALID_CONCEPT, "logline": "   "}
    with pytest.raises(ConceptValidationError):
        validate_concept(bad)


def test_motifs_must_be_a_list_of_strings():
    bad = {**VALID_CONCEPT, "motifs": "not a list"}
    with pytest.raises(ConceptValidationError, match="motifs"):
        validate_concept(bad)

    bad2 = {**VALID_CONCEPT, "motifs": [1, 2]}
    with pytest.raises(ConceptValidationError, match="motifs"):
        validate_concept(bad2)


# --------------------------------------------------------------------------- #
# `locations` -- the closed vocabulary issue #78's beats stage assigns
# `location` from. Required and non-empty: unlike `motifs`/`avoid`, an empty
# list here would leave the beats stage with no vocabulary at all to check
# against, silently disabling issue #78's whole mechanism.
# --------------------------------------------------------------------------- #


def test_locations_is_required():
    bad = {k: v for k, v in VALID_CONCEPT.items() if k != "locations"}
    with pytest.raises(ConceptValidationError, match="locations"):
        validate_concept(bad)


def test_locations_must_be_a_non_empty_list_of_strings():
    bad = {**VALID_CONCEPT, "locations": []}
    with pytest.raises(ConceptValidationError, match="locations"):
        validate_concept(bad)

    bad2 = {**VALID_CONCEPT, "locations": "the pier"}
    with pytest.raises(ConceptValidationError, match="locations"):
        validate_concept(bad2)

    bad3 = {**VALID_CONCEPT, "locations": [1, 2]}
    with pytest.raises(ConceptValidationError, match="locations"):
        validate_concept(bad3)

    bad4 = {**VALID_CONCEPT, "locations": ["  "]}
    with pytest.raises(ConceptValidationError, match="locations"):
        validate_concept(bad4)


# --------------------------------------------------------------------------- #
# `reading` -- issue #69's comprehension step. Structured and validated like
# every other stage's reply so `beats` can consume it directly, and empty
# `nouns`/`references` are a first-class, VALID answer -- a model forced to
# invent a reference for a song with none will invent one, and the invention
# gets staged.
# --------------------------------------------------------------------------- #


def test_reading_is_required():
    bad = {k: v for k, v in VALID_CONCEPT.items() if k != "reading"}
    with pytest.raises(ConceptValidationError, match="reading"):
        validate_concept(bad)


def test_reading_must_be_an_object():
    bad = {**VALID_CONCEPT, "reading": "just read it yourself"}
    with pytest.raises(ConceptValidationError, match="reading"):
        validate_concept(bad)


@pytest.mark.parametrize(
    "key",
    ["subject", "speaker", "addressee", "situation", "change", "register", "period", "place"],
)
def test_each_reading_string_field_is_required_and_non_empty(key):
    missing = {**VALID_CONCEPT, "reading": {k: v for k, v in VALID_READING.items() if k != key}}
    with pytest.raises(ConceptValidationError, match=key):
        validate_concept(missing)

    blank = {**VALID_CONCEPT, "reading": {**VALID_READING, key: "   "}}
    with pytest.raises(ConceptValidationError, match=key):
        validate_concept(blank)


def test_reading_nouns_may_be_an_empty_list():
    """Issue #69 requirement 2: a song with no concrete nouns worth naming is
    a real, valid answer -- never something the validator treats as missing
    or incomplete."""
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "nouns": []}}
    validate_concept(concept)  # must not raise


def test_reading_nouns_must_be_a_list_of_strings():
    bad = {**VALID_CONCEPT, "reading": {**VALID_READING, "nouns": "a drum, a chair"}}
    with pytest.raises(ConceptValidationError, match="nouns"):
        validate_concept(bad)

    bad2 = {**VALID_CONCEPT, "reading": {**VALID_READING, "nouns": [1, 2]}}
    with pytest.raises(ConceptValidationError, match="nouns"):
        validate_concept(bad2)


def test_reading_references_may_be_an_empty_list():
    """Issue #69 requirement 2, and the whole reason this issue exists: most
    songs -- a breakup song, say -- make no cultural/historical/mythological
    reference at all, and an empty list here is the CORRECT reply, not a
    gap."""
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": []}}
    validate_concept(concept)  # must not raise


def test_reading_references_must_be_a_list():
    bad = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": "Koschey the Deathless"}}
    with pytest.raises(ConceptValidationError, match="references"):
        validate_concept(bad)


def test_a_reference_missing_a_field_is_rejected():
    bad_ref = {"reference": "a lighthouse", "what_it_is": "a structure"}  # no imagery
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": [bad_ref]}}
    with pytest.raises(ConceptValidationError, match="imagery"):
        validate_concept(concept)


def test_a_reference_with_an_empty_name_is_rejected():
    bad_ref = {"reference": "  ", "what_it_is": "a structure", "imagery": ["a beam of light"]}
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": [bad_ref]}}
    with pytest.raises(ConceptValidationError, match="reference"):
        validate_concept(concept)


def test_a_references_imagery_may_itself_be_empty():
    """Only `references` and `nouns` are required to accept the empty case --
    this asserts that a reference's own `imagery` list is allowed to be
    empty too, since nothing in the issue requires it non-empty."""
    ref = {"reference": "a lighthouse", "what_it_is": "a structure", "imagery": []}
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": [ref]}}
    validate_concept(concept)  # must not raise


def test_a_references_imagery_must_be_a_list_of_strings():
    bad_ref = {"reference": "a lighthouse", "what_it_is": "a structure", "imagery": "a beam"}
    concept = {**VALID_CONCEPT, "reading": {**VALID_READING, "references": [bad_ref]}}
    with pytest.raises(ConceptValidationError, match="imagery"):
        validate_concept(concept)


def test_every_reading_problem_is_reported_at_once():
    """One retry round costs a whole model call -- three faults in `reading`
    must come back with three complaints, the same convention `beats` uses."""
    broken = {
        **VALID_READING,
        "subject": "",
        "nouns": "not a list",
        "references": [{"reference": "x"}],  # missing what_it_is, imagery
    }
    concept = {**VALID_CONCEPT, "reading": broken}
    with pytest.raises(ConceptValidationError) as excinfo:
        validate_concept(concept)

    message = str(excinfo.value)
    assert "subject" in message
    assert "nouns" in message
    assert "reading.references[0]" in message


# --------------------------------------------------------------------------- #
# `acts` -- issue #84's closed, ordered story-structure vocabulary. `beats`
# assigns every chunk's `act` from exactly this list; validated the way
# `locations` is: non-empty, and here also unique by name.
# --------------------------------------------------------------------------- #


def test_acts_is_required():
    bad = {k: v for k, v in VALID_CONCEPT.items() if k != "acts"}
    with pytest.raises(ConceptValidationError, match="acts"):
        validate_concept(bad)


def test_acts_must_be_a_non_empty_list():
    bad = {**VALID_CONCEPT, "acts": []}
    with pytest.raises(ConceptValidationError, match="acts"):
        validate_concept(bad)

    bad2 = {**VALID_CONCEPT, "acts": "situation, resolution"}
    with pytest.raises(ConceptValidationError, match="acts"):
        validate_concept(bad2)


def test_an_act_needs_a_name_and_a_function():
    bad = {**VALID_CONCEPT, "acts": [{"name": "situation"}]}  # no function
    with pytest.raises(ConceptValidationError, match="function"):
        validate_concept(bad)

    bad2 = {**VALID_CONCEPT, "acts": [{"function": "sets the scene"}]}  # no name
    with pytest.raises(ConceptValidationError, match="name"):
        validate_concept(bad2)


def test_an_act_with_a_blank_name_is_rejected():
    bad = {**VALID_CONCEPT, "acts": [{"name": "  ", "function": "sets the scene"}]}
    with pytest.raises(ConceptValidationError, match="name"):
        validate_concept(bad)


def test_duplicate_act_names_are_rejected():
    bad = {
        **VALID_CONCEPT,
        "acts": [
            {"name": "Situation", "function": "the opening"},
            {"name": "situation", "function": "a different one, badly renamed"},
        ],
    }
    with pytest.raises(ConceptValidationError, match="unique"):
        validate_concept(bad)


def test_a_well_formed_multi_act_structure_passes_silently():
    concept = {
        **VALID_CONCEPT,
        "acts": [
            {"name": "situation", "function": "she waits"},
            {"name": "complication", "function": "the storm builds"},
            {"name": "turn", "function": "it breaks"},
            {"name": "resolution", "function": "it clears, and what she planted pays off"},
        ],
    }
    validate_concept(concept)  # must not raise


# --------------------------------------------------------------------------- #
# build_concept_prompt
# --------------------------------------------------------------------------- #


def test_prompt_says_no_lyric_is_sung_when_lyrics_file_is_empty(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    # never touches align_model -- see _FakeAlignModel's docstring
    chunks = load_chunk_skeleton(config)

    prompt = build_concept_prompt(config, chunks)

    assert "no lyric is ever sung on screen" in prompt
    assert "Dianne" in prompt or "Rex" in prompt  # cast line present


def test_prompt_includes_the_real_lyric_text_when_present(tmp_path):
    config = _config(
        tmp_path,
        lyrics_text=(
            "Walking through the empty halls tonight\n"
            "Nobody's watching nobody cares\n"
            "The lights flicker but I don't mind\n"
        ),
    )
    chunks = load_chunk_skeleton(config, align_model=_FakeAlignModel())

    prompt = build_concept_prompt(config, chunks)

    assert "Walking through the empty halls tonight" in prompt
    assert "no lyric is ever sung on screen" not in prompt


def test_prompt_includes_setting_only_when_the_config_has_one(tmp_path):
    # global_style is a required RunConfig field, so its "already fixed"
    # line is always present -- only `setting` (optional) is conditional.
    empty_config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(empty_config)
    bare_prompt = build_concept_prompt(empty_config, chunks)
    assert "Setting (already fixed" not in bare_prompt
    assert "Style (already fixed" in bare_prompt

    from dataclasses import replace

    styled = replace(empty_config, setting="London, UK")
    styled_prompt = build_concept_prompt(styled, chunks)
    assert "Setting (already fixed -- work inside it): London, UK" in styled_prompt


def test_prompt_includes_hints_when_given(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)

    prompt = build_concept_prompt(config, chunks, hints="too bleak, and no rain")

    assert "too bleak, and no rain" in prompt


def test_prompt_omits_the_hints_section_when_none_given(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)

    prompt = build_concept_prompt(config, chunks, hints=None)

    assert "Hints from the person running this" not in prompt


# --------------------------------------------------------------------------- #
# concept_input_hashes
# --------------------------------------------------------------------------- #


def test_input_hashes_change_when_the_lyrics_file_changes(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    before = concept_input_hashes(config, chunks)

    config.lyrics_file.write_text("a whole new lyric line\n", encoding="utf-8")
    after = concept_input_hashes(config, chunks)

    assert before["lyrics"] != after["lyrics"]
    assert before["skeleton"] == after["skeleton"]  # same chunks object, unchanged


def test_input_hashes_include_the_lyrics_format_doc(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)

    hashes = concept_input_hashes(config, chunks)

    assert hashes["lyrics_format_doc"]
    assert LYRICS_FORMAT_DOC.exists()  # the doc this hash is actually of


# --------------------------------------------------------------------------- #
# generate_concept: the retry-with-feedback loop
# --------------------------------------------------------------------------- #


def test_generate_concept_happy_path(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    driver = ScriptedDriver([VALID_CONCEPT])

    result = generate_concept(config, chunks, driver)

    assert result.data == VALID_CONCEPT
    assert len(driver.calls) == 1
    assert result.input_hashes == concept_input_hashes(config, chunks)


def test_generate_concept_retries_with_the_validators_error_on_a_bad_reply(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    invalid = {k: v for k, v in VALID_CONCEPT.items() if k != "tone"}  # missing required field
    driver = ScriptedDriver([invalid, VALID_CONCEPT])

    result = generate_concept(config, chunks, driver)

    assert result.data == VALID_CONCEPT
    assert len(driver.calls) == 2
    # The retry's prompt carries the validator's own error, not a repeat of
    # the original prompt verbatim.
    assert "failed validation" in driver.calls[1]["prompt"]
    assert "tone" in driver.calls[1]["prompt"]
    # The system prompt is unchanged between attempts.
    assert driver.calls[0]["system"] == driver.calls[1]["system"]


def test_generate_concept_gives_up_after_exhausting_every_attempt(tmp_path):
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    invalid = {k: v for k, v in VALID_CONCEPT.items() if k != "tone"}
    driver = ScriptedDriver([invalid] * MAX_VALIDATION_ATTEMPTS)

    with pytest.raises(ConceptValidationError):
        generate_concept(config, chunks, driver)

    assert len(driver.calls) == MAX_VALIDATION_ATTEMPTS


def test_generate_concept_writes_nothing_on_failure(tmp_path):
    """Design section 3: "a half-written session is worse than no
    session." A failed generation must not return anything a caller could
    mistake for a usable result."""
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    driver = ScriptedDriver([None])  # DriverError, not a validation failure

    with pytest.raises(DriverError):
        generate_concept(config, chunks, driver)


def test_generate_concept_propagates_a_driver_error_without_extra_retries(tmp_path):
    """DriverError is the driver's OWN retry already exhausted -- this
    stage's retry loop is for schema violations only and must not catch or
    re-attempt a mechanical failure a second time."""
    config = _config(tmp_path, lyrics_text="")
    chunks = load_chunk_skeleton(config)
    driver = ScriptedDriver([None])

    with pytest.raises(DriverError):
        generate_concept(config, chunks, driver)

    assert len(driver.calls) == 1


# --------------------------------------------------------------------------- #
# The authoring layer is a THIRD align() call site. Its own docstring says it
# is "the same one source of truth run_pipeline and prepare_shot_plan already
# share" -- which was true of the audio, the lyrics and the overrides, and not
# of the model size. A plan authored against `base` describes a timeline the
# render (on `small`) never emits: 71 chunks vs 80, 15 voiced vs 41.
# --------------------------------------------------------------------------- #


def test_skeleton_uses_the_configured_alignment_model(tmp_path, monkeypatch):
    """Adding a config value means finding every consumer of it. `grep -rn
    'align('` is that check; a docstring claiming shared truth is not."""
    from dataclasses import replace

    from music_video_maker.authoring import chunks as chunks_mod

    config = replace(
        _config(tmp_path, lyrics_text="Walking through the empty halls tonight\n"),
        alignment_model_size="small",
    )

    seen: dict = {}
    real_align = chunks_mod.align

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_align(*args, **kwargs)

    monkeypatch.setattr(chunks_mod, "align", spy)
    load_chunk_skeleton(config, align_model=_FakeAlignModel())

    assert seen["model_size"] == "small"


# --------------------------------------------------------------------------- #
# Issue #69: the system prompt has to instruct comprehension BEFORE
# invention, and say plainly that an empty reference/noun list is correct.
# --------------------------------------------------------------------------- #


def test_the_concept_preamble_asks_for_reading_before_inventing():
    from music_video_maker.authoring.prompts import CONCEPT_PREAMBLE

    lowered = CONCEPT_PREAMBLE.lower()
    assert "reading" in lowered
    assert "fill" in lowered and "first" in lowered


def test_the_concept_preamble_says_empty_references_are_correct():
    from music_video_maker.authoring.prompts import CONCEPT_PREAMBLE

    lowered = CONCEPT_PREAMBLE.lower()
    assert "empty list is the correct answer" in lowered
    assert "invent a reference" in lowered


def test_the_concept_preamble_explains_the_acts_payoff_requirement():
    from music_video_maker.authoring.prompts import CONCEPT_PREAMBLE

    lowered = CONCEPT_PREAMBLE.lower()
    assert "acts" in lowered
    assert "pay off" in lowered or "payoff" in lowered
