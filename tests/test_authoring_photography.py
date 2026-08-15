"""Tests for Stage 3 -- photography (issue #54 design section 4).

The stage that chooses the film's look and each shot's framing. Its own
section of the design gives it two properties nothing else here has: it is
*optional per chunk* (an absent ``camera`` composes nothing, the same
conservative default as every other optional field), and it is **A/B-able**
-- ``--candidates`` runs it several times from deliberately different framing
stances so a human can pick, which is the discipline that found the
grammatical-subject rule in the first place.

The one hard validation rule is small and very specific: ``prompting`` composes
``camera`` as ``", camera <value>"``, so a value that starts with "camera"
composes as "..., camera the camera pushes in". That is the whole reason this
check exists and it is the first thing tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.driver import MODEL_OPUS, DriverError, ScriptedDriver
from music_video_maker.authoring.photography import (
    CANDIDATE_STANCES,
    PhotographyValidationError,
    build_photography_prompt,
    generate_photography,
    photography_candidates,
    photography_input_hashes,
    validate_photography,
)
from music_video_maker.config import RunConfig
from tests.harness.factories import make_cast_dict, write_silent_wav

CONCEPT = {
    "logline": "A drummer walks through gathering weather.",
    "setting": "a coastal town",
    "tone": "elegiac",
    "motifs": ["gathering clouds"],
    "avoid": ["literal lightning strikes"],
}


def _config(tmp_path: Path, *, cinematography: str | None = None) -> RunConfig:
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("", encoding="utf-8")
    return RunConfig(
        master_audio=write_silent_wav(tmp_path / "master.wav", seconds=20.0),
        lyrics_file=lyrics_file,
        global_style="Refestramus progressive rock music video",
        narrative_concept="placeholder",
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
        cinematography=cinematography,
    )


def _chunks(n: int = 3, width: float = 6.0):
    return tuple(
        contracts.AudioChunk(
            chunk_id=i + 1,
            audio_file=Path(f"chunk_{i + 1:04d}.wav"),
            start=i * width,
            end=(i + 1) * width,
            text=f"lyric {i + 1}",
        )
        for i in range(n)
    )


def _beat(chunk_id, *, group=1, role="transition", focus="subject"):
    start = (chunk_id - 1) * 6.0
    return Beat(
        chunk_id=chunk_id,
        start=start,
        end=start + 6.0,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=group,
        focus=focus,
    )


def _reply(*pairs, cinematography="35mm film, shallow depth of field, overcast light"):
    return {
        "cinematography": cinematography,
        "camera": [{"chunk_id": cid, "camera": text} for cid, text in pairs],
    }


GOOD = "tracking backwards ahead of her"


# --------------------------------------------------------------------------- #
# The one hard rule: "camera" is supplied by the composer, not the value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [
        "camera pushes in slowly",
        "Camera tracking backwards",
        "the camera pushes in",
        "The Camera cranes up over the roofline",
    ],
)
def test_a_camera_value_that_says_camera_itself_is_rejected(bad):
    """``prompting._apply_camera_clause`` composes ``", camera <value>"``, so
    this would render as "..., camera the camera pushes in"."""
    with pytest.raises(PhotographyValidationError, match="camera"):
        validate_photography(_reply((1, bad)), _chunks(), config_cinematography=None)


def test_a_value_that_merely_contains_the_word_later_is_fine():
    """Only the *leading* word collides with the composed one. "walking
    toward camera" is exactly how the guide's own examples read."""
    result = validate_photography(
        _reply((1, "held wide as she walks toward camera")), _chunks(), config_cinematography=None
    )

    assert result.camera[1] == "held wide as she walks toward camera"


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #


def test_a_well_formed_reply_validates():
    result = validate_photography(
        _reply((1, GOOD), (2, "locked off, low, the desk in the near foreground")),
        _chunks(),
        config_cinematography=None,
    )

    assert result.cinematography.startswith("35mm")
    assert set(result.camera) == {1, 2}


def test_partial_coverage_is_fine_and_is_not_padded_out():
    """An absent ``camera`` composes nothing -- the same conservative default
    as ``length_seconds`` and ``focus``. Demanding one per chunk would buy
    filler direction on every shot that does not need any."""
    result = validate_photography(_reply((2, GOOD)), _chunks(), config_cinematography=None)

    assert set(result.camera) == {2}


def test_an_invented_chunk_id_is_dropped(caplog):
    with caplog.at_level("WARNING"):
        result = validate_photography(
            _reply((1, GOOD), (99, GOOD)), _chunks(), config_cinematography=None
        )

    assert set(result.camera) == {1}
    assert "99" in caplog.text


def test_a_duplicate_chunk_id_is_an_error():
    with pytest.raises(PhotographyValidationError, match="more than once"):
        validate_photography(_reply((1, GOOD), (1, GOOD)), _chunks(), config_cinematography=None)


def test_a_blank_camera_value_is_an_error():
    with pytest.raises(PhotographyValidationError):
        validate_photography(_reply((1, "   ")), _chunks(), config_cinematography=None)


def test_a_reply_that_is_not_an_object_is_rejected():
    with pytest.raises(PhotographyValidationError):
        validate_photography(["nope"], _chunks(), config_cinematography=None)


def test_a_missing_cinematography_is_an_error_when_the_config_has_none():
    with pytest.raises(PhotographyValidationError, match="cinematography"):
        validate_photography({"camera": []}, _chunks(), config_cinematography=None)


def test_the_global_half_is_skipped_when_the_config_already_fixes_it():
    """Design section 4: if the look is already decided, only per-shot camera
    is generated. #55 will make that the normal case."""
    result = validate_photography(
        {"camera": [{"chunk_id": 1, "camera": GOOD}]},
        _chunks(),
        config_cinematography="16mm, high contrast, hard shadows",
    )

    assert result.cinematography is None
    assert result.camera == {1: GOOD}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_the_prompt_carries_the_concept_and_every_beat(tmp_path):
    beats = (_beat(1, role="consequence", focus="action"), _beat(2))

    prompt = build_photography_prompt(_config(tmp_path), CONCEPT, beats, _chunks())

    assert CONCEPT["logline"] in prompt
    assert "beat 1" in prompt and "beat 2" in prompt
    assert "consequence" in prompt


def test_the_prompt_tells_the_model_the_word_camera_is_supplied(tmp_path):
    prompt = build_photography_prompt(_config(tmp_path), CONCEPT, (_beat(1),), _chunks())

    assert "camera" in prompt.lower()
    assert "tracking backwards ahead of her" in prompt  # the shape it must write in


def test_the_beat_table_tags_voiced_and_instrumental_chunks(tmp_path):
    """Issue #58: the photography stage previously had no way to know which
    chunks were voiced, so it could not act on "keep her face available on a
    voiced chunk" -- the table now says so explicitly per beat."""
    voiced_chunk = contracts.AudioChunk(
        chunk_id=1, audio_file=Path("c1.wav"), start=0.0, end=6.0, text="a lyric line"
    )
    instrumental_chunk = contracts.AudioChunk(
        chunk_id=2,
        audio_file=Path("c2.wav"),
        start=6.0,
        end=12.0,
        text="",
        is_instrumental=True,
    )
    beats = (_beat(1), _beat(2))

    prompt = build_photography_prompt(
        _config(tmp_path), CONCEPT, beats, (voiced_chunk, instrumental_chunk)
    )

    assert "chunk_id=1 | 0.0-6.0s | voiced" in prompt
    assert "chunk_id=2 | 6.0-12.0s | instrumental" in prompt


def test_a_fixed_cinematography_is_stated_as_fixed(tmp_path):
    config = _config(tmp_path, cinematography="16mm, high contrast")

    prompt = build_photography_prompt(config, CONCEPT, (_beat(1),), _chunks())

    assert "16mm, high contrast" in prompt
    assert "already fixed" in prompt


def test_a_stance_reaches_the_prompt(tmp_path):
    prompt = build_photography_prompt(
        _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), stance=CANDIDATE_STANCES[1]
    )

    assert CANDIDATE_STANCES[1] in prompt


def test_input_hashes_cover_concept_and_beats(tmp_path):
    chunks = _chunks()
    beats = (_beat(1), _beat(2))

    before = photography_input_hashes(_config(tmp_path), chunks, CONCEPT, beats)
    after = photography_input_hashes(
        _config(tmp_path), chunks, {**CONCEPT, "tone": "comic"}, beats
    )

    assert before != after
    assert set(before) == {"skeleton", "concept", "beats"}


# --------------------------------------------------------------------------- #
# generate_photography
# --------------------------------------------------------------------------- #


def test_generate_photography_returns_a_validated_look(tmp_path):
    driver = ScriptedDriver([_reply((1, GOOD))])

    result = generate_photography(
        _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver
    )

    assert result.photography.camera == {1: GOOD}
    assert driver.calls[0]["model"] == MODEL_OPUS


def test_a_rejected_value_feeds_its_own_error_back_into_the_next_prompt(tmp_path):
    driver = ScriptedDriver([_reply((1, "the camera pushes in")), _reply((1, GOOD))])

    result = generate_photography(
        _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver
    )

    assert len(driver.calls) == 2
    assert "failed validation" in driver.calls[1]["prompt"]
    assert result.photography.camera == {1: GOOD}


def test_generate_photography_gives_up_after_the_bounded_attempts(tmp_path):
    bad = _reply((1, "camera pushes in"))
    driver = ScriptedDriver([bad, bad, bad])

    with pytest.raises(PhotographyValidationError):
        generate_photography(_config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver)

    assert len(driver.calls) == 3


def test_a_driver_error_propagates(tmp_path):
    with pytest.raises(DriverError):
        generate_photography(
            _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), ScriptedDriver([None])
        )


# --------------------------------------------------------------------------- #
# --candidates: different framing instructions, never different seeds
# --------------------------------------------------------------------------- #


def test_candidates_vary_the_framing_instruction_not_the_seed(tmp_path):
    """Design section 4 says this in as many words. Re-rolling the same
    prompt gives you noise; asking for a genuinely different stance gives you
    a choice worth making."""
    driver = ScriptedDriver([_reply((1, GOOD)), _reply((1, "locked off, low")), _reply((1, GOOD))])

    results = photography_candidates(
        _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver, count=3
    )

    assert len(results) == 3
    prompts = [call["prompt"] for call in driver.calls]
    assert len(set(prompts)) == 3
    for stance, prompt in zip(CANDIDATE_STANCES[:3], prompts, strict=False):
        assert stance in prompt


def test_asking_for_more_candidates_than_there_are_stances_is_capped(tmp_path):
    driver = ScriptedDriver([_reply((1, GOOD)) for _ in CANDIDATE_STANCES])

    results = photography_candidates(
        _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver, count=99
    )

    assert len(results) == len(CANDIDATE_STANCES)


def test_one_candidate_uses_no_stance_at_all(tmp_path):
    """A single run is the default path, not a one-item A/B -- pinning it to
    the first stance would quietly narrow every un-A/B'd run to one look."""
    driver = ScriptedDriver([_reply((1, GOOD))])

    photography_candidates(_config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver, count=1)

    assert CANDIDATE_STANCES[0] not in driver.calls[0]["prompt"]


def test_a_candidate_that_fails_validation_does_not_kill_the_others(tmp_path, caplog):
    """A/B is worth having precisely when one option is bad. Losing the whole
    run because stance 2 wrote "the camera" would make --candidates the least
    reliable way to use this stage."""
    driver = ScriptedDriver(
        [
            _reply((1, GOOD)),
            _reply((1, "the camera pushes in")),
            _reply((1, "the camera pushes in")),
            _reply((1, "the camera pushes in")),
            _reply((1, "locked off, low")),
        ]
    )

    with caplog.at_level("WARNING"):
        results = photography_candidates(
            _config(tmp_path), CONCEPT, (_beat(1),), _chunks(), driver, count=3
        )

    assert [r.stance_index for r in results] == [0, 2]
    assert "candidate" in caplog.text.lower()


def test_the_global_style_reaches_the_prompt(tmp_path):
    """Found by running this for real. ``global_style`` routinely carries
    editing and camera constraints -- this project's own says "ONE continuous
    unbroken take, a single camera following the subject, no cuts" -- because
    it is the junk drawer #53 split up. A photography stage that has not seen
    it proposes locked-off frames for a run whose style says the camera
    follows her, and it is composed into every prompt either way."""
    config = _config(tmp_path)

    prompt = build_photography_prompt(config, CONCEPT, (_beat(1),), _chunks())

    assert config.global_style in prompt
    assert "must not contradict it" in prompt
