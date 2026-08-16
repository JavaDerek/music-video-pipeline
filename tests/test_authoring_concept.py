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

VALID_CONCEPT = {
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
