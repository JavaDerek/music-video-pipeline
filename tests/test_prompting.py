"""Tests for the deterministic Stage 2b prompt expansion engine (issue #5).

No LLM, no network, no inference -- this is pure string composition, so every
test asserts the *exact* composed prompt string, not a substring match.

Character resolution is NOT this module's job: `chunk.character` is treated
as already-authoritative (issue #6's `LyricLine.character` ->
`AlignedSegment.character` -> `AudioChunk.character` pipeline owns that), so
these tests set `chunk.character` directly rather than injecting a resolver.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.config import RunConfig
from music_video_maker.prompting import (
    SubjectOnVoicedChunkError,
    UnknownCastMemberError,
    expand_prompt,
)
from tests.harness.factories import make_cast_dict

GLOBAL_STYLE = (
    "Refestramus progressive rock music video, atmospheric lighting, high quality cinematic"
)
NARRATIVE_CONCEPT = "Wandering through a surgery, kicking a life support plug out"
SETTING = "London, UK -- contemporary, overcast winter"


@pytest.fixture
def cast() -> dict[str, contracts.CastMember]:
    return make_cast_dict()


@pytest.fixture
def config(cast: dict[str, contracts.CastMember]) -> RunConfig:
    return RunConfig(
        master_audio=Path("audio/master.wav"),
        lyrics_file=Path("lyrics.txt"),
        global_style=GLOBAL_STYLE,
        narrative_concept=NARRATIVE_CONCEPT,
        cast=cast,
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=Path("output/chunks"),
        final_video_dir=Path("output/final"),
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
    )


def _chunk(
    chunk_id: int = 0,
    text: str = "",
    character: str | None = None,
    source_segment_indices: tuple[int, ...] = (),
    characters: tuple[str, ...] | None = None,
) -> contracts.AudioChunk:
    """Build a test chunk. ``character`` is the single-name convenience used
    by the pre-#33 tests; ``characters`` (plural) lets multi-vocalist tests
    pass a tuple directly. Passing both is a test bug, not a real case."""
    if characters is None:
        characters = (character,) if character else ()
    return contracts.AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunks/chunk_{chunk_id:04d}.wav"),
        start=0.0,
        end=6.0,
        text=text,
        characters=characters,
        source_segment_indices=source_segment_indices,
    )


def test_lead_vocal_segment_exact_prompt_and_image_ref(config: RunConfig, cast):
    chunk = _chunk(
        chunk_id=1,
        text="walking through the empty halls tonight",
        character="Dianne",
        source_segment_indices=(2,),
    )

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=1,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image,),
        characters=("Dianne",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
    )


def test_background_member_segment_wins_over_lead(config: RunConfig, cast):
    # A background member named explicitly on the chunk must be used as-is --
    # never silently overridden by the configured lead vocalist.
    chunk = _chunk(
        chunk_id=2,
        text="the drums crash like thunder before the fall",
        character="Rex",
        source_segment_indices=(5,),
    )

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=2,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Rex, Drummer, background, never sings, is "
            "the focus of this shot. The character is actively singing the "
            "lyric: 'the drums crash like thunder before the fall'."
        ),
        image_ref=cast["Rex"].image,
        image_refs=(cast["Rex"].image,),
        characters=("Rex",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Rex, Drummer, background, never sings, is "
            "the focus of this shot. The character is actively singing the "
            "lyric: 'the drums crash like thunder before the fall'."
        ),
    )


def test_instrumental_padded_segment_falls_back_to_default_lead(config: RunConfig, cast):
    # No character tag at all -- this chunk was not attributed to anyone
    # (e.g. pure instrumental padding). Must fall back to the configured
    # lead vocalist.
    chunk = _chunk(chunk_id=3, text="", character=None, source_segment_indices=())

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=3,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. Instrumental passage: the "
            "character stays silent throughout this shot."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image,),
        characters=("Dianne",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. Instrumental passage: the "
            "character stays silent throughout this shot."
        ),
    )


def test_character_none_defaults_to_configured_lead_vocalist(config: RunConfig, cast):
    # A chunk that does trace back to source segments but has no character
    # attributed yet must still default to the lead vocalist.
    chunk = _chunk(
        chunk_id=4,
        text="i've been here a thousand times before",
        character=None,
        source_segment_indices=(3,),
    )

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=4,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'i've been here a thousand times before'."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image,),
        characters=("Dianne",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'i've been here a thousand times before'."
        ),
    )


def test_unknown_character_raises_and_logs(config: RunConfig, cast, caplog):
    # A typo'd or stale character tag must fail loudly, never quietly
    # collapse into the lead vocalist.
    chunk = _chunk(chunk_id=5, text="some lyric", character="Ghostface")

    with caplog.at_level("ERROR"), pytest.raises(UnknownCastMemberError, match="Ghostface"):
        expand_prompt(config, chunk)

    assert any("Ghostface" in record.message for record in caplog.records)


def test_whitespace_only_text_is_treated_as_instrumental(config: RunConfig, cast):
    chunk = _chunk(chunk_id=6, text="   ", character="Dianne", source_segment_indices=(7,))

    result = expand_prompt(config, chunk)

    assert result.prompt.endswith(
        "Instrumental passage: the character stays silent throughout this shot."
    )


def test_result_is_deterministic_across_repeated_calls(config: RunConfig, cast):
    chunk = _chunk(
        chunk_id=1,
        text="walking through the empty halls tonight",
        character="Dianne",
        source_segment_indices=(2,),
    )

    first = expand_prompt(config, chunk)
    second = expand_prompt(config, chunk)

    assert first == second


# --------------------------------------------------------------------------- #
# Shot plan direction
# --------------------------------------------------------------------------- #


def test_authored_shot_replaces_the_global_narrative_concept(config: RunConfig, cast):
    """The whole point of a shot plan: chunk 12 gets chunk 12's direction,
    not the same global sentence every other chunk gets."""
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk, shot="She kicks the plug out without noticing")

    assert "She kicks the plug out without noticing" in result.prompt
    assert config.narrative_concept not in result.prompt


def test_global_style_and_lyric_survive_an_authored_shot(config: RunConfig, cast):
    """A shot replaces only the narrative concept -- the look of the film and
    the line being sung are still composed around it."""
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk, shot="A corridor, walking away from camera")

    assert config.global_style.rstrip(".") in result.prompt
    assert "the lucky ones" in result.prompt
    assert "Dianne" in result.prompt


def test_no_shot_falls_back_to_the_narrative_concept(config: RunConfig, cast):
    chunk = _chunk(text="the lucky ones", character="Dianne")

    assert expand_prompt(config, chunk, shot=None).prompt == expand_prompt(config, chunk).prompt
    assert config.narrative_concept.rstrip(".") in expand_prompt(config, chunk).prompt


def test_instrumental_chunk_with_a_shot_still_reads_as_instrumental(config: RunConfig, cast):
    """B-roll chunks are where a shot plan earns its keep -- they have no
    lyric to describe them, so the authored direction is all they get."""
    chunk = _chunk(text="", character=None)

    result = expand_prompt(config, chunk, shot="A bin lorry reverses into a parked car")

    assert "A bin lorry reverses into a parked car" in result.prompt
    assert "nstrumental" in result.prompt


# --------------------------------------------------------------------------- #
# Setting (issue #32) -- must anchor every chunk, voiced or instrumental
# --------------------------------------------------------------------------- #


def test_setting_qualifies_the_shot_instead_of_competing_with_it(config: RunConfig, cast):
    """Observed on the first Chicago render: a bare place-name sentence reads
    as a thing to DEPICT, not as a constraint on depiction, so it competed
    with the shot line and won. Shots written for an ICU corridor and a
    surgical gallery rendered as Loop sidewalks -- a medical cart and a mixing
    console standing on the pavement -- because the model relocated the action
    outdoors to get the skyline and the elevated tracks on screen.

    So the setting is composed as a conditional: it governs *which* place a
    location resolves to, never *whether* a location is shown. It also comes
    after the shot line, so the shot leads and the setting qualifies."""
    cfg = dataclasses.replace(config, setting=SETTING)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk, shot="An intensive care corridor, monitors either side")

    assert result.prompt == (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. An intensive care corridor, monitors either side. "
        "Location continuity: wherever the location is identifiable it is "
        "London, UK -- contemporary, overcast winter, and never anywhere else; "
        "do not relocate the action or add landmarks to establish it, this shot's "
        "own described location is what is on screen. Dianne, "
        "Lead Vocalist, smiling constantly, oblivious, is the focus of this "
        "shot. The character is actively singing the lyric: 'the lucky ones'."
    )


def test_the_setting_never_becomes_the_subject_of_the_shot(config: RunConfig, cast):
    """The failure mode in one assertion: the setting must not appear as a
    standalone sentence that could be read as 'show this place'."""
    cfg = dataclasses.replace(config, setting=SETTING)

    prompt = expand_prompt(cfg, _chunk(text="x", character="Dianne")).prompt

    assert f". {SETTING}." not in prompt
    assert "wherever the location is identifiable" in prompt


def test_setting_is_composed_on_voiced_and_instrumental_chunks_alike(config: RunConfig, cast):
    """The whole point of #32: geography must not be able to depend on
    whether the chunk has a lyric."""
    cfg = dataclasses.replace(config, setting=SETTING)
    voiced = _chunk(text="the lucky ones", character="Dianne")
    instrumental = _chunk(text="", character="Dianne")

    voiced_prompt = expand_prompt(cfg, voiced).prompt
    instrumental_prompt = expand_prompt(cfg, instrumental).prompt

    assert SETTING in voiced_prompt
    assert SETTING in instrumental_prompt
    assert "nstrumental" in instrumental_prompt


def test_setting_still_applies_when_a_shot_plan_replaces_the_concept(config: RunConfig, cast):
    cfg = dataclasses.replace(config, setting=SETTING)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk, shot="A corridor, walking away from camera")

    assert SETTING in result.prompt
    assert "A corridor, walking away from camera" in result.prompt


def test_setting_none_by_default_omits_any_setting_sentence(config: RunConfig, cast):
    # The shared `config` fixture never sets `setting`, so every other test
    # in this file already proves the None case renders unchanged -- this
    # test just says so explicitly.
    assert config.setting is None
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert SETTING not in result.prompt
    assert "London" not in result.prompt


# --------------------------------------------------------------------------- #
# Location (issue #78, composed since the "Deathless" nuclear-glow finding)
#
# `setting` is one string composed unchanged into all 80 chunks of a run; a
# song whose world changes over its own runtime (a detonation, an emptied
# planet) cannot express that change through a field that cannot vary by
# chunk. `location` already existed per-chunk (issue #78) but was deliberately
# never rendered. It is rendered now, substituted into the same guarded
# "Location continuity" sentence `setting` already uses -- narrowing it for
# this chunk, never adding a second place-naming sentence next to it, because
# #73/#74 measured that H3 renders whatever noun it is given regardless of
# the framing around it, and there is no negative-conditioning channel to
# retract one once it is named.
# --------------------------------------------------------------------------- #

CONTAMINATED_SETTING = (
    "A war-torn valley below a mountain watch-post -- medieval hosts, industrial "
    "armies, and nuclear glow all visible from the same watch-post, ending in a "
    "post-war planet eroded to a hill"
)
POST_DETONATION_LOCATION = "the eroded hill under a wind-still sky (the summit, after)"


def test_location_narrows_the_setting_clause_to_the_authored_place(config: RunConfig, cast):
    """The direct regression test for the Deathless nuclear-glow leak: a
    chunk authored with a post-detonation `location` must not still carry
    `setting`'s arc-locked nouns into the composed prompt."""
    cfg = dataclasses.replace(config, setting=CONTAMINATED_SETTING)
    chunk = _chunk(text="", character=None)

    result = expand_prompt(cfg, chunk, shot="Ash settles over the split mill wheel",
                            location=POST_DETONATION_LOCATION)

    assert POST_DETONATION_LOCATION in result.prompt
    assert "nuclear glow" not in result.prompt
    assert "industrial armies" not in result.prompt
    assert CONTAMINATED_SETTING not in result.prompt


def test_location_none_is_byte_identical_to_pre_location_prompt(config: RunConfig, cast):
    """Every pre-existing caller never passes `location` at all, and every
    chunk a plan leaves untagged resolves it to `None` -- both must compose
    exactly the golden string already pinned for the plain-`setting` case."""
    cfg = dataclasses.replace(config, setting=SETTING)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk, shot="An intensive care corridor, monitors either side")
    explicit_none = expand_prompt(
        cfg, chunk, shot="An intensive care corridor, monitors either side", location=None
    )

    golden = (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. An intensive care corridor, monitors either side. "
        "Location continuity: wherever the location is identifiable it is "
        "London, UK -- contemporary, overcast winter, and never anywhere else; "
        "do not relocate the action or add landmarks to establish it, this shot's "
        "own described location is what is on screen. Dianne, "
        "Lead Vocalist, smiling constantly, oblivious, is the focus of this "
        "shot. The character is actively singing the lyric: 'the lucky ones'."
    )
    assert result.prompt == golden
    assert explicit_none.prompt == golden


def test_location_used_when_setting_is_none(config: RunConfig, cast):
    """`place` must not require `setting` to be present -- an authored
    per-chunk location composes on its own."""
    assert config.setting is None
    chunk = _chunk(text="", character=None)

    result = expand_prompt(
        config, chunk, shot="Ash settles over the split mill wheel",
        location=POST_DETONATION_LOCATION,
    )

    assert POST_DETONATION_LOCATION in result.prompt
    assert "wherever the location is identifiable" in result.prompt


def test_location_composes_on_the_chained_variant_too(config: RunConfig, cast):
    """Location is about place, not identity, so -- unlike appearance -- it
    must survive on the chained I2V variant exactly like `camera` does."""
    cfg = dataclasses.replace(config, setting=CONTAMINATED_SETTING)
    chunk = _chunk(text="", character=None)

    result = expand_prompt(
        cfg, chunk, shot="Ash settles over the split mill wheel",
        location=POST_DETONATION_LOCATION,
    )

    assert result.chained_prompt is not None
    assert POST_DETONATION_LOCATION in result.chained_prompt
    assert "nuclear glow" not in result.chained_prompt


def test_location_still_positioned_after_the_shot_line_and_conditionally_phrased(
    config: RunConfig, cast
):
    """#32's protections must survive substitution: still no bare,
    unconditional place-naming sentence, still phrased as a conditional
    constraint, still after the shot line."""
    cfg = dataclasses.replace(config, setting=SETTING)
    chunk = _chunk(text="x", character="Dianne")

    prompt = expand_prompt(
        cfg, chunk, shot="A corridor, walking away from camera",
        location=POST_DETONATION_LOCATION,
    ).prompt

    assert f". {POST_DETONATION_LOCATION}." not in prompt
    assert "wherever the location is identifiable" in prompt
    shot_index = prompt.index("A corridor, walking away from camera")
    location_index = prompt.index(POST_DETONATION_LOCATION)
    assert shot_index < location_index


def test_location_composes_at_most_one_location_sentence(config: RunConfig, cast):
    """Guards against ever reintroducing #32 by concatenation: exactly one
    'Location continuity' sentence, never one for `setting` and a second for
    `location`."""
    cfg = dataclasses.replace(config, setting=CONTAMINATED_SETTING)
    chunk = _chunk(text="", character=None)

    prompt = expand_prompt(
        cfg, chunk, shot="Ash settles over the split mill wheel",
        location=POST_DETONATION_LOCATION,
    ).prompt

    assert prompt.count("Location continuity") == 1


# --------------------------------------------------------------------------- #
# Appearance (issue #31) -- attaches to the character, never to vocal action
# --------------------------------------------------------------------------- #


def test_member_appearance_is_composed_alongside_the_role_clause(config: RunConfig, cast):
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="looking a few years younger, softly lit, flattering"
    )
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.prompt == (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. Wandering through a surgery, kicking a "
        "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
        "oblivious, is the focus of this shot, looking a few years younger, "
        "softly lit, flattering. The character is actively singing the "
        "lyric: 'the lucky ones'."
    )


def test_global_appearance_and_member_appearance_both_compose(config: RunConfig, cast):
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="looking a few years younger, softly lit, flattering"
    )
    cfg = dataclasses.replace(
        config,
        global_appearance="everyone slim, trim and healthy looking",
        cast={**cast, "Dianne": dianne},
    )
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert (
        "is the focus of this shot, everyone slim, trim and healthy looking, "
        "looking a few years younger, softly lit, flattering." in result.prompt
    )


def test_appearance_never_mentions_singing_and_survives_instrumental_chunks(
    config: RunConfig, cast
):
    """The bug this issue exists to avoid: a character-attached instruction
    that names vocal action wins over the per-chunk instrumental clause. An
    appearance clause must be silent on singing so it can never do that."""
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="looking a few years younger, softly lit, flattering"
    )
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="", character="Dianne")

    result = expand_prompt(cfg, chunk)

    character_clause = result.prompt.split(". Instrumental passage")[0]
    assert "sing" not in character_clause.lower()
    assert "flattering" in result.prompt
    assert result.prompt.endswith(
        "Instrumental passage: the character stays silent throughout this shot."
    )


def test_global_appearance_alone_applies_to_a_member_with_no_appearance_of_their_own(
    config: RunConfig, cast
):
    cfg = dataclasses.replace(config, global_appearance="everyone slim, trim and healthy looking")
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert (
        "is the focus of this shot, everyone slim, trim and healthy looking. "
        "The character" in result.prompt
    )


# --------------------------------------------------------------------------- #
# All fields None -- no doubled periods, no stray separators
# --------------------------------------------------------------------------- #


def test_all_new_fields_none_produces_no_doubled_separators(config: RunConfig, cast):
    assert config.setting is None
    assert config.global_appearance is None
    assert cast["Dianne"].appearance is None
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert ".." not in result.prompt
    assert ",," not in result.prompt
    assert ", ." not in result.prompt
    assert ". ." not in result.prompt


def test_setting_present_appearance_absent_has_no_stray_trailing_comma(
    config: RunConfig, cast
):
    cfg = dataclasses.replace(config, setting=SETTING)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert "is the focus of this shot." in result.prompt
    assert ".." not in result.prompt
    assert ",." not in result.prompt


# --------------------------------------------------------------------------- #
# Concurrent vocalists (issue #33 levels 1-2) -- more than one active singer
# --------------------------------------------------------------------------- #


def test_single_character_via_characters_tuple_is_byte_identical_to_today(
    config: RunConfig, cast
):
    """Proves the widening did not change the level-1 (solo) case at all --
    same assertion as test_lead_vocal_segment_exact_prompt_and_image_ref, but
    built via the plural `characters` tuple directly rather than the old
    `character=` convenience, since that is what real chunks will carry."""
    chunk = _chunk(
        chunk_id=1,
        text="walking through the empty halls tonight",
        characters=("Dianne",),
        source_segment_indices=(2,),
    )

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=1,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image,),
        characters=("Dianne",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
    )


def test_two_characters_compose_both_into_one_shared_clause(config: RunConfig, cast):
    chunk = _chunk(
        chunk_id=10,
        text="Lord knows when it's people like you",
        characters=("Dianne", "Marcus"),
    )

    result = expand_prompt(config, chunk)

    assert result == contracts.ExpandedPrompt(
        chunk_id=10,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, and Marcus, Backup Vocalist, watching from the wings "
            "are the focus of this shot. The characters are actively singing "
            "the lyric: 'Lord knows when it's people like you'."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image, cast["Marcus"].image),
        characters=("Dianne", "Marcus"),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, and Marcus, Backup Vocalist, watching from the wings "
            "are the focus of this shot. The characters are actively singing "
            "the lyric: 'Lord knows when it's people like you'."
        ),
    )


def test_three_characters_compose_with_oxford_comma_list(config: RunConfig, cast):
    chunk = _chunk(chunk_id=11, text="all together now", characters=("Dianne", "Marcus", "Rex"))

    result = expand_prompt(config, chunk)

    assert result.prompt == (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. Wandering through a surgery, kicking a "
        "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
        "oblivious, Marcus, Backup Vocalist, watching from the wings, and Rex, "
        "Drummer, background, never sings are the focus of this shot. The "
        "characters are actively singing the lyric: 'all together now'."
    )
    assert result.image_ref == cast["Dianne"].image
    assert result.image_refs == (cast["Dianne"].image, cast["Marcus"].image, cast["Rex"].image)
    assert result.characters == ("Dianne", "Marcus", "Rex")


def test_primary_character_stays_first_regardless_of_who_is_listed_second(
    config: RunConfig, cast
):
    """Order is significant (#33): the first name is the primary vocalist and
    is what `image_ref` (the single-reference fallback) must resolve to."""
    chunk = _chunk(chunk_id=12, text="switch the order", characters=("Marcus", "Dianne"))

    result = expand_prompt(config, chunk)

    assert result.image_ref == cast["Marcus"].image
    assert result.image_refs == (cast["Marcus"].image, cast["Dianne"].image)
    assert result.prompt.index("Marcus") < result.prompt.index("Dianne")


def test_unknown_name_in_multi_character_tuple_raises_and_names_the_offender(
    config: RunConfig, cast, caplog
):
    """A typo among several real names must fail loudly and name exactly the
    bad one -- it must never quietly collapse to just the known names."""
    chunk = _chunk(chunk_id=13, text="some lyric", characters=("Dianne", "Ghostface"))

    with caplog.at_level("ERROR"), pytest.raises(UnknownCastMemberError, match="Ghostface"):
        expand_prompt(config, chunk)

    assert any("Ghostface" in record.message for record in caplog.records)


def test_each_members_own_appearance_stays_attached_to_that_member(config: RunConfig, cast):
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="looking a few years younger, softly lit, flattering"
    )
    marcus = dataclasses.replace(cast["Marcus"], appearance="a little more tired than usual")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne, "Marcus": marcus})
    chunk = _chunk(chunk_id=14, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)

    assert (
        "Dianne, Lead Vocalist, smiling constantly, oblivious, looking a few years "
        "younger, softly lit, flattering, and Marcus, Backup Vocalist, watching from "
        "the wings, a little more tired than usual are the focus of this shot."
        in result.prompt
    )


def test_global_appearance_applies_once_to_the_whole_cast_not_once_per_person(
    config: RunConfig, cast
):
    cfg = dataclasses.replace(config, global_appearance="everyone slim, trim and healthy looking")
    chunk = _chunk(chunk_id=15, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)

    # The global appearance sentence-fragment appears exactly once, not once
    # per cast member.
    assert result.prompt.count("everyone slim, trim and healthy looking") == 1
    assert (
        "are the focus of this shot, everyone slim, trim and healthy looking."
        in result.prompt
    )


def test_global_and_per_member_appearance_compose_together_for_two_singers(
    config: RunConfig, cast
):
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="looking a few years younger, softly lit, flattering"
    )
    cfg = dataclasses.replace(
        config,
        global_appearance="everyone slim, trim and healthy looking",
        cast={**cast, "Dianne": dianne},
    )
    chunk = _chunk(chunk_id=16, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)

    assert (
        "Dianne, Lead Vocalist, smiling constantly, oblivious, looking a few years "
        "younger, softly lit, flattering, and Marcus, Backup Vocalist, watching from "
        "the wings are the focus of this shot, everyone slim, trim and healthy "
        "looking." in result.prompt
    )


def test_instrumental_chunk_with_two_characters(config: RunConfig, cast):
    """No lyric text, two characters active -- the instrumental clause must
    still be the only thing saying nobody is singing, now in plural agreement
    since two people are on screen not singing."""
    chunk = _chunk(chunk_id=17, text="", characters=("Dianne", "Marcus"))

    result = expand_prompt(config, chunk)

    assert result.prompt.endswith(
        "Instrumental passage: the characters stay silent throughout this shot."
    )
    assert result.characters == ("Dianne", "Marcus")
    assert result.image_refs == (cast["Dianne"].image, cast["Marcus"].image)


def test_multi_character_clause_never_asserts_two_shot_or_split_screen(
    config: RunConfig, cast
):
    """Per #33's recorded decision: `&` states who is audible, not how they
    are framed. Staging language must never leak into the composed prompt --
    that is the shot plan's call, not the transcript's."""
    chunk = _chunk(chunk_id=18, text="the lucky ones", characters=("Dianne", "Marcus", "Rex"))

    result = expand_prompt(config, chunk)

    forbidden = ("side by side", "split screen", "split-screen", "two-shot", "two shot")
    lowered = result.prompt.lower()
    for phrase in forbidden:
        assert phrase not in lowered


def test_multi_character_clause_never_mentions_vocal_action(config: RunConfig, cast):
    """The same trap the single-character role bug hit: a character-attached
    clause must not assert who is singing -- that is part 5's job alone."""
    chunk = _chunk(chunk_id=19, text="", characters=("Dianne", "Marcus"))

    result = expand_prompt(config, chunk)

    character_clause = result.prompt.split(". Instrumental passage")[0]
    assert "sing" not in character_clause.lower()


def test_no_doubled_separators_with_two_characters_and_no_appearance(config: RunConfig, cast):
    chunk = _chunk(chunk_id=20, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(config, chunk)

    assert ".." not in result.prompt
    assert ",," not in result.prompt
    assert ", ." not in result.prompt
    assert ". ." not in result.prompt


# --------------------------------------------------------------------------- #
# Per-shot focus (issue #26)
# --------------------------------------------------------------------------- #


def test_default_still_says_the_character_is_the_focus(config: RunConfig, cast):
    """Byte-identical to today for every shot that does not opt out."""
    chunk = _chunk(text="the lucky ones", character="Dianne")

    assert "is the focus of this shot" in expand_prompt(config, chunk).prompt


def test_a_consequence_shot_can_hand_the_focus_to_the_action(config: RunConfig, cast):
    """Across four renders the consequence lost to the performer, because
    every prompt asserted the performer was the focus -- including on shots
    whose whole job was a consequence she had walked away from."""
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(
        config,
        chunk,
        shot="The printer erupts in a fireball behind her",
        subject_is_focus=False,
    )

    assert "is the focus of this shot" not in result.prompt
    assert "is not its subject" in result.prompt
    assert "not the performer" in result.prompt
    # The character is still named and still present -- this is a focus
    # change, not a removal.
    assert "Dianne" in result.prompt
    assert "The printer erupts in a fireball behind her" in result.prompt


def test_focus_handoff_still_carries_role_and_appearance(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], appearance="softly lit")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})

    result = expand_prompt(
        cfg, _chunk(text="x", character="Dianne"), shot="The window shatters",
        subject_is_focus=False,
    )

    assert "Lead Vocalist" in result.prompt
    assert "softly lit" in result.prompt


def test_focus_handoff_never_suppresses_the_lyric_clause(config: RunConfig, cast):
    """Whether she is singing is per-chunk state and stays the lyric clause's
    job -- handing the visual focus to the action must not touch it."""
    result = expand_prompt(
        config, _chunk(text="the lucky ones", character="Dianne"),
        shot="The window shatters", subject_is_focus=False,
    )

    assert "actively singing the lyric" in result.prompt


# --------------------------------------------------------------------------- #
# Issue #33 level 3: the second voice reaches the prompt
# --------------------------------------------------------------------------- #


def test_counterpoint_text_is_composed_into_the_prompt(config: RunConfig, cast):
    """A counterpoint chunk has two simultaneous lyric texts; the prompt must
    say both, attributed, or the second voice renders as a silent extra."""
    from dataclasses import replace as dc_replace

    chunk = dc_replace(
        _chunk(
            text="there was a time when i hoped",
            characters=("Dianne", "Marcus"),
        ),
        concurrent_texts=("i know when it's people like you",),
        concurrent_characters=(("Marcus",),),
    )

    result = expand_prompt(config, chunk)

    assert "there was a time when i hoped" in result.prompt
    assert "i know when it's people like you" in result.prompt
    assert "Marcus" in result.prompt
    assert "simultaneously" in result.prompt.lower()
    # Both faces stage their reference photos.
    assert result.image_refs == (cast["Dianne"].image, cast["Marcus"].image)


def test_chunks_without_counterpoint_compose_exactly_as_before(config: RunConfig):
    chunk = _chunk(text="a single line", character="Dianne")
    result = expand_prompt(config, chunk)
    assert "simultaneously" not in result.prompt.lower()


# --------------------------------------------------------------------------- #
# Chained I2V prompt variant (issue #46) -- no photo on that path, so an
# appearance clause is applied a second time to its own output. The chained
# variant must omit CastMember.appearance and global_appearance entirely
# while staying identical to `prompt` in every other respect.
# --------------------------------------------------------------------------- #


def test_chained_prompt_exact_string_for_relative_directive(config: RunConfig, cast):
    """The exact before/after for the issue's reported case: 'looking a few
    years younger' is a displacement, not an endpoint, so it must never be
    re-applied to a frame that is already its own output."""
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.prompt == (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. Wandering through a surgery, kicking a "
        "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
        "oblivious, is the focus of this shot, looking a few years younger. "
        "The character is actively singing the lyric: 'the lucky ones'."
    )
    assert result.chained_prompt == (
        "Refestramus progressive rock music video, atmospheric lighting, "
        "high quality cinematic. Wandering through a surgery, kicking a "
        "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
        "oblivious, is the focus of this shot. The character is actively "
        "singing the lyric: 'the lucky ones'."
    )


def test_chained_prompt_strips_an_absolute_appearance_directive_too(config: RunConfig, cast):
    """Even a non-relative directive is redundant on the chained path -- the
    seed frame already embodies the anchor's appearance decision, and the
    clause only competes with the frame for influence over a shot with no
    other identity signal. Absolute directives get stripped exactly like
    relative ones -- the stripping is not conditioned on wording."""
    dianne = dataclasses.replace(cast["Dianne"], appearance="in her late forties")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert "in her late forties" in result.prompt
    assert result.chained_prompt is not None
    assert "in her late forties" not in result.chained_prompt


def test_chained_prompt_strips_global_appearance(config: RunConfig, cast):
    cfg = dataclasses.replace(config, global_appearance="everyone slim, trim and healthy looking")
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert "everyone slim, trim and healthy looking" in result.prompt
    assert result.chained_prompt is not None
    assert "everyone slim, trim and healthy looking" not in result.chained_prompt


def test_chained_prompt_strips_appearance_for_multi_character_clause(config: RunConfig, cast):
    """Each member's own appearance AND the whole-cast global appearance are
    both stripped in the multi-character clause -- not just the primary
    member's."""
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    marcus = dataclasses.replace(cast["Marcus"], appearance="a little more tired than usual")
    cfg = dataclasses.replace(
        config,
        global_appearance="everyone slim, trim and healthy looking",
        cast={**cast, "Dianne": dianne, "Marcus": marcus},
    )
    chunk = _chunk(chunk_id=14, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)
    assert result.chained_prompt is not None

    for phrase in (
        "looking a few years younger",
        "a little more tired than usual",
        "everyone slim, trim and healthy looking",
    ):
        assert phrase in result.prompt
        assert phrase not in result.chained_prompt

    # Identity (name + role) still present -- only the appearance is gone.
    assert "Dianne" in result.chained_prompt
    assert "Marcus" in result.chained_prompt
    assert "Lead Vocalist" in result.chained_prompt
    assert "Backup Vocalist" in result.chained_prompt


def test_chained_prompt_strips_appearance_on_a_counterpoint_chunk(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = dataclasses.replace(
        _chunk(text="there was a time when i hoped", characters=("Dianne", "Marcus")),
        concurrent_texts=("i know when it's people like you",),
        concurrent_characters=(("Marcus",),),
    )

    result = expand_prompt(cfg, chunk)
    assert result.chained_prompt is not None

    assert "looking a few years younger" in result.prompt
    assert "looking a few years younger" not in result.chained_prompt
    # The counterpoint clause itself is untouched by the appearance strip.
    assert "i know when it's people like you" in result.chained_prompt
    assert "simultaneously" in result.chained_prompt.lower()


def test_chained_prompt_still_names_character_and_role(config: RunConfig, cast):
    """Stripping appearance must not strip identity -- the model still needs
    to know who is on screen and what they are doing, only not how they
    should look."""
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.chained_prompt is not None
    assert "Dianne" in result.chained_prompt
    assert "Lead Vocalist" in result.chained_prompt
    assert "actively singing the lyric" in result.chained_prompt


def test_chained_prompt_equals_prompt_when_no_appearance_exists(config: RunConfig, cast):
    """'No appearance to strip' and 'no variant composed' are different facts
    (see ExpandedPrompt.chained_prompt) -- chained_prompt is still populated
    even when there is nothing to remove, so it is identical to prompt rather
    than None."""
    assert config.global_appearance is None
    assert cast["Dianne"].appearance is None
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert result.chained_prompt is not None
    assert result.chained_prompt == result.prompt


def test_chained_prompt_differs_from_prompt_only_by_the_appearance_text(config: RunConfig, cast):
    """The load-bearing invariant: the chained variant must come from the SAME
    composition function, parameterised -- not a second hand-written composer
    that can drift out of step with the first. Proven by showing
    chained_prompt is byte-identical to what expand_prompt produces for a
    cast/config with the appearance fields removed entirely, i.e. the only
    thing that changed is the appearance text."""
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    cfg = dataclasses.replace(
        config,
        global_appearance="everyone slim, trim and healthy looking",
        cast={**cast, "Dianne": dianne},
    )
    no_appearance_cfg = dataclasses.replace(
        config,
        global_appearance=None,
        cast={**cast, "Dianne": dataclasses.replace(cast["Dianne"], appearance=None)},
    )
    chunk = _chunk(text="the lucky ones", character="Dianne")

    with_appearance = expand_prompt(cfg, chunk)
    baseline = expand_prompt(no_appearance_cfg, chunk)

    assert with_appearance.prompt != with_appearance.chained_prompt
    assert with_appearance.chained_prompt == baseline.prompt
    assert with_appearance.chained_prompt == baseline.chained_prompt


def test_chained_prompt_omits_appearance_on_an_instrumental_chunk(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], appearance="looking a few years younger")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.chained_prompt is not None
    assert "looking a few years younger" not in result.chained_prompt
    assert result.chained_prompt.endswith(
        "Instrumental passage: the character stays silent throughout this shot."
    )


# --------------------------------------------------------------------------- #
# Cinematography and camera direction (issue #53)
# --------------------------------------------------------------------------- #

CINEMATOGRAPHY = "35mm film, shallow depth of field, warm natural light"


def test_cinematography_composes_right_after_global_style(config: RunConfig, cast):
    cfg = dataclasses.replace(config, cinematography=CINEMATOGRAPHY)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.prompt == (
        f"{config.global_style}. {CINEMATOGRAPHY}. {config.narrative_concept}. "
        "Dianne, Lead Vocalist, smiling constantly, oblivious, is the focus of this shot. "
        "The character is actively singing the lyric: 'the lucky ones'."
    )


def test_cinematography_none_by_default_omits_the_sentence(config: RunConfig, cast):
    assert config.cinematography is None
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert CINEMATOGRAPHY not in result.prompt


def test_cinematography_survives_on_the_chained_variant_too(config: RunConfig, cast):
    """Cinematography is film-level, not identity-related, so unlike
    appearance it must NOT be stripped from the chained I2V prompt variant."""
    cfg = dataclasses.replace(config, cinematography=CINEMATOGRAPHY)
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.chained_prompt is not None
    assert CINEMATOGRAPHY in result.chained_prompt


def test_camera_is_composed_as_a_trailing_clause_on_the_concept(config: RunConfig, cast):
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(
        config, chunk, shot="She rounds the aisle", camera="tracking backwards ahead of her"
    )

    assert (
        "She rounds the aisle, camera tracking backwards ahead of her" in result.prompt
    )
    # Never its own sentence -- no period between the concept and "camera".
    assert "She rounds the aisle. Camera" not in result.prompt
    assert "She rounds the aisle." not in result.prompt


def test_camera_applies_to_the_global_concept_when_no_shot_is_authored(config: RunConfig, cast):
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk, shot=None, camera="pushing in slowly on her hands")

    assert (
        f"{config.narrative_concept}, camera pushing in slowly on her hands" in result.prompt
    )


def test_camera_none_leaves_the_concept_sentence_unchanged(config: RunConfig, cast):
    chunk = _chunk(text="the lucky ones", character="Dianne")

    with_camera = expand_prompt(config, chunk, camera=None).prompt
    without_camera_arg = expand_prompt(config, chunk).prompt

    assert with_camera == without_camera_arg
    assert "camera" not in with_camera.lower()


def test_camera_survives_on_the_chained_variant_too(config: RunConfig, cast):
    """Camera direction is about framing, not identity, so it must stay on
    the chained variant exactly like the unchained one."""
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(
        config, chunk, shot="She rounds the aisle", camera="tracking backwards ahead of her"
    )

    assert result.chained_prompt is not None
    assert "camera tracking backwards ahead of her" in result.chained_prompt


# --------------------------------------------------------------------------- #
# On-screen cast (issue #59)
#
# `chunk.characters` answers "who is singing", which the alignment knows. It
# was silently doing double duty as "who is on screen", which nothing knew.
# A real 36-chunk render made the gap visible: 9 shot lines referred to a
# second character as "him"/"he" -- because PROSE_PREAMBLE rule 2 forbids
# naming a cast member -- and on the 7 of those where he was not also the
# singer, no role, appearance or reference photo went into the prompt at all.
# H3 invented a different man each time.
# --------------------------------------------------------------------------- #


def test_present_cast_member_gets_name_role_and_photo(config: RunConfig, cast):
    """The whole point: a non-singing cast member in shot is conditioned."""
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(config, chunk, shot="She carries the box past him", present=("Rex",))

    assert "Also in shot, silent: Rex, Drummer, background, never sings" in result.prompt
    # His photo joins the reference set, after the singer's.
    assert result.image_refs == (cast["Dianne"].image, cast["Rex"].image)
    # ...but he is not credited with the vocal: `characters` stays the singers.
    assert result.characters == ("Dianne",)
    assert result.image_ref == cast["Dianne"].image


def test_present_cast_makes_the_lyric_clause_name_the_singer(config: RunConfig):
    """"The character is singing" is ambiguous once two people are on screen."""
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(config, chunk, present=("Rex",))

    assert "Dianne is actively singing the lyric: 'I'm the lucky one'" in result.prompt
    assert "The character is actively singing" not in result.prompt


def test_no_present_cast_leaves_the_prompt_byte_identical(config: RunConfig):
    """The established rule in this module: an untouched case stays untouched."""
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    assert expand_prompt(config, chunk, present=()) == expand_prompt(config, chunk)


def test_present_cast_member_who_is_also_singing_is_not_repeated(config: RunConfig, cast):
    """Jan sings chunk 18 and walks through chunk 4. Naming him in both places
    on the same chunk must not compose him twice or stage his photo twice."""
    chunk = _chunk(chunk_id=18, text="a bass line", character="Marcus")

    result = expand_prompt(config, chunk, present=("Marcus",))

    assert "Also in shot" not in result.prompt
    assert result.image_refs == (cast["Marcus"].image,)


def test_unknown_present_cast_member_raises(config: RunConfig):
    """Same rule as a singing name: a typo must fail loudly, never quietly
    become 'just the people we recognised'."""
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    with pytest.raises(UnknownCastMemberError, match="Jann"):
        expand_prompt(config, chunk, present=("Jann",))


def test_present_cast_appearance_is_dropped_on_the_chained_variant(config: RunConfig, cast):
    """Issue #46 applies to everyone in frame, not just the lead: the chained
    path has no photo, so appearance would compound against its own output."""
    described = dataclasses.replace(cast["Rex"], appearance="in his late forties")
    config = dataclasses.replace(config, cast={**cast, "Rex": described})
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(config, chunk, present=("Rex",))

    assert "in his late forties" in result.prompt
    assert "in his late forties" not in result.chained_prompt
    assert "Also in shot, silent: Rex" in result.chained_prompt


def test_two_present_cast_members_compose_as_one_clause(config: RunConfig):
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(config, chunk, present=("Marcus", "Rex"))

    assert result.prompt.count("Also in shot, silent:") == 1
    assert "Marcus, Backup Vocalist, watching from the wings, and Rex, Drummer" in result.prompt


def test_the_lora_trigger_word_leads_the_prompt(config: RunConfig):
    """Issue #62. A trigger word is a tag, not a statement: first, bare, in
    the style position, so it never takes the grammatical subject slot that
    docs/shot-writing-guide.md reserves for what the shot is about."""
    config = dataclasses.replace(config, lora="realism.safetensors", lora_trigger="r34l1sm")
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(config, chunk)

    assert result.prompt.startswith("r34l1sm. ")
    assert result.chained_prompt.startswith("r34l1sm. ")


def test_no_lora_means_no_trigger_word_even_if_one_is_set(config: RunConfig):
    """Config refuses this combination outright, but composition must not
    depend on that: a trigger with no adapter is a stray token in all 36
    prompts, in the position that most shapes the style."""
    config = dataclasses.replace(config, lora=None, lora_trigger="r34l1sm")
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    assert "r34l1sm" not in expand_prompt(config, chunk).prompt


# --------------------------------------------------------------------------- #
# Demeanour (issue #74) -- follows the appearance/global_appearance precedent
# (#31) exactly, one field over. Composed alongside role/appearance exactly
# as the module docstring's part 5 describes.
# --------------------------------------------------------------------------- #


def test_member_demeanour_is_composed_alongside_the_role_and_appearance_clause(
    config: RunConfig, cast
):
    dianne = dataclasses.replace(
        cast["Dianne"], appearance="in her late forties", demeanour="grave and unsmiling"
    )
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert (
        "Dianne, Lead Vocalist, smiling constantly, oblivious, is the focus of this "
        "shot, in her late forties, grave and unsmiling. The character is actively "
        "singing the lyric: 'the lucky ones'."
    ) in result.prompt


def test_global_demeanour_and_member_demeanour_both_compose(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], demeanour="grave and unsmiling, exhausted")
    cfg = dataclasses.replace(
        config,
        global_demeanour="nobody smiles; this is a war",
        cast={**cast, "Dianne": dianne},
    )
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert (
        "is the focus of this shot, nobody smiles; this is a war, grave and "
        "unsmiling, exhausted." in result.prompt
    )


def test_global_demeanour_alone_applies_to_a_member_with_no_demeanour_of_their_own(
    config: RunConfig, cast
):
    cfg = dataclasses.replace(config, global_demeanour="nobody smiles; this is a war")
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert (
        "is the focus of this shot, nobody smiles; this is a war. The character"
        in result.prompt
    )


def test_demeanour_survives_on_an_instrumental_chunk(config: RunConfig, cast):
    """Like appearance, demeanour is character-attached (part 5) so it must
    compose whether or not the chunk carries a lyric -- and, like appearance,
    it must never itself assert singing."""
    dianne = dataclasses.replace(cast["Dianne"], demeanour="grave and unsmiling")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="", character="Dianne")

    result = expand_prompt(cfg, chunk)

    character_clause = result.prompt.split(". Instrumental passage")[0]
    assert "sing" not in character_clause.lower()
    assert "grave and unsmiling" in result.prompt
    assert result.prompt.endswith(
        "Instrumental passage: the character stays silent throughout this shot."
    )


def test_no_doubled_separators_with_demeanour_and_no_appearance(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], demeanour="grave and unsmiling")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert ".." not in result.prompt
    assert ",," not in result.prompt
    assert ", ." not in result.prompt
    assert ". ." not in result.prompt


def test_each_members_own_demeanour_stays_attached_to_that_member(config: RunConfig, cast):
    dianne = dataclasses.replace(cast["Dianne"], demeanour="grave and unsmiling")
    marcus = dataclasses.replace(cast["Marcus"], demeanour="hollow-eyed, unspeaking")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne, "Marcus": marcus})
    chunk = _chunk(chunk_id=14, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)

    assert (
        "Dianne, Lead Vocalist, smiling constantly, oblivious, grave and unsmiling, "
        "and Marcus, Backup Vocalist, watching from the wings, hollow-eyed, "
        "unspeaking are the focus of this shot." in result.prompt
    )


def test_global_demeanour_applies_once_for_the_whole_cast_not_once_per_person(
    config: RunConfig, cast
):
    cfg = dataclasses.replace(config, global_demeanour="nobody smiles; this is a war")
    chunk = _chunk(chunk_id=15, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)

    assert result.prompt.count("nobody smiles; this is a war") == 1
    assert "are the focus of this shot, nobody smiles; this is a war." in result.prompt


def test_present_cast_member_demeanour_is_composed(config: RunConfig, cast):
    rex = dataclasses.replace(cast["Rex"], demeanour="impassive, watching")
    cfg = dataclasses.replace(config, cast={**cast, "Rex": rex})
    chunk = _chunk(chunk_id=4, text="I'm the lucky one", character="Dianne")

    result = expand_prompt(cfg, chunk, present=("Rex",))

    assert (
        "Also in shot, silent: Rex, Drummer, background, never sings, impassive, "
        "watching" in result.prompt
    )


# --------------------------------------------------------------------------- #
# `subject`: whose shot this is on an instrumental chunk (issue #82)
#
# `chunk.characters` is empty on an instrumental chunk, so this module falls
# back to `config.default_lead_vocalist` and composes THAT person as "the
# focus of this shot" -- a config default answering a question the shot line
# may already answer differently. On "Deathless" chunk 29, `present =
# ["Jan"]` staged his name/role/photo but the composed focus stayed Dianne,
# and H3 morphed one into the other. `subject` replaces the fallback with the
# actually-authored focus member; `present` (issue #59) is deliberately not
# this lever (CLAUDE.md: "`present` decides who a pronoun is, never whose
# shot this is").
# --------------------------------------------------------------------------- #


def test_subject_none_is_byte_identical_to_the_pre_82_prompt(config: RunConfig, cast):
    """The established rule in this module: an untouched case stays
    untouched -- compared against the exact golden prompt a solo singer
    composed before this field existed."""
    chunk = _chunk(
        chunk_id=1,
        text="walking through the empty halls tonight",
        character="Dianne",
        source_segment_indices=(2,),
    )

    result = expand_prompt(config, chunk, subject=None)

    assert result == contracts.ExpandedPrompt(
        chunk_id=1,
        prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
        image_ref=cast["Dianne"].image,
        image_refs=(cast["Dianne"].image,),
        characters=("Dianne",),
        chained_prompt=(
            "Refestramus progressive rock music video, atmospheric lighting, "
            "high quality cinematic. Wandering through a surgery, kicking a "
            "life support plug out. Dianne, Lead Vocalist, smiling constantly, "
            "oblivious, is the focus of this shot. The character is actively "
            "singing the lyric: 'walking through the empty halls tonight'."
        ),
    )
    # And the parameter's default (omitting it entirely) must agree.
    assert result == expand_prompt(config, chunk)


def test_subject_becomes_the_focus_on_an_instrumental_chunk(config: RunConfig, cast):
    """The chunk 29 fix itself: an instrumental chunk (no singer) whose shot
    line is about the drummer, not the default lead vocalist."""
    chunk = _chunk(chunk_id=29, text="")  # empty characters: the instrumental case

    result = expand_prompt(
        config, chunk, shot="His boot settles into a hollow", subject="Rex"
    )

    assert (
        "Rex, Drummer, background, never sings, is the focus of this shot" in result.prompt
    )
    assert "Dianne" not in result.prompt
    assert result.characters == ("Rex",)
    assert result.image_ref == cast["Rex"].image
    assert result.image_refs == (cast["Rex"].image,)


def test_subject_on_a_chunk_with_a_singer_raises(config: RunConfig):
    """Defence in depth: the shot-plan lint is the primary gate, but
    `expand_prompt` is a public function and must not silently do the wrong
    thing on a voiced chunk -- the singer owns the frame there (#58, #59,
    #60)."""
    chunk = _chunk(chunk_id=29, text="a lyric", character="Dianne")

    with pytest.raises(SubjectOnVoicedChunkError, match="29"):
        expand_prompt(config, chunk, subject="Rex")


def test_unknown_subject_raises(config: RunConfig):
    """Same rule as every other cast-valued field: a typo must fail loudly."""
    chunk = _chunk(chunk_id=29, text="")

    with pytest.raises(UnknownCastMemberError, match="Nope"):
        expand_prompt(config, chunk, subject="Nope")


def test_subject_is_dropped_from_present_when_also_listed_there(config: RunConfig, cast):
    """The focus member is not also a silent bystander -- same rule
    `_resolve_present_members` already applies to a singer named twice."""
    chunk = _chunk(chunk_id=29, text="")

    result = expand_prompt(config, chunk, subject="Rex", present=("Rex",))

    assert "Also in shot" not in result.prompt
    assert result.image_refs == (cast["Rex"].image,)
    assert result.present_cast == ()


def test_subject_appearance_is_dropped_on_the_chained_variant(config: RunConfig, cast):
    """Issue #46 applies to the focus member exactly as it always has --
    `subject` only changes WHO the focus member is, not that rule."""
    described = dataclasses.replace(cast["Rex"], appearance="grizzled, weathered")
    cfg = dataclasses.replace(config, cast={**cast, "Rex": described})
    chunk = _chunk(chunk_id=29, text="")

    result = expand_prompt(cfg, chunk, subject="Rex")

    assert "grizzled, weathered" in result.prompt
    assert "grizzled, weathered" not in result.chained_prompt
    assert (
        "Rex, Drummer, background, never sings, is the focus of this shot"
        in result.chained_prompt
    )


def test_demeanour_survives_the_chained_variant_unlike_appearance(config: RunConfig, cast):
    """The load-bearing difference from appearance (issue #46 vs #74):
    appearance describes how to read a photo the chained path does not have
    and is stripped; demeanour is behavioural direction, the same kind of
    statement as role, and is required to be phrased as an endpoint so
    restating it every chunk does not compound."""
    dianne = dataclasses.replace(
        cast["Dianne"],
        appearance="looking a few years younger",
        demeanour="grave and unsmiling",
    )
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne})
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert "looking a few years younger" in result.prompt
    assert "grave and unsmiling" in result.prompt
    assert result.chained_prompt is not None
    assert "looking a few years younger" not in result.chained_prompt
    assert "grave and unsmiling" in result.chained_prompt


def test_global_demeanour_survives_the_chained_variant_too(config: RunConfig, cast):
    cfg = dataclasses.replace(config, global_demeanour="nobody smiles; this is a war")
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert result.chained_prompt is not None
    assert "nobody smiles; this is a war" in result.chained_prompt


def test_demeanour_survives_the_chained_variant_for_multi_character_clause(
    config: RunConfig, cast
):
    """Mirrors test_chained_prompt_strips_appearance_for_multi_character_clause,
    proving demeanour is exempt from that stripping for every active member,
    not just the primary one."""
    dianne = dataclasses.replace(
        cast["Dianne"],
        appearance="looking a few years younger",
        demeanour="grave and unsmiling",
    )
    marcus = dataclasses.replace(cast["Marcus"], demeanour="hollow-eyed")
    cfg = dataclasses.replace(config, cast={**cast, "Dianne": dianne, "Marcus": marcus})
    chunk = _chunk(chunk_id=14, text="the lucky ones", characters=("Dianne", "Marcus"))

    result = expand_prompt(cfg, chunk)
    assert result.chained_prompt is not None

    assert "looking a few years younger" not in result.chained_prompt
    assert "grave and unsmiling" in result.chained_prompt
    assert "hollow-eyed" in result.chained_prompt


def test_no_demeanour_fields_set_leaves_the_prompt_byte_identical(config: RunConfig, cast):
    """The established rule this module follows everywhere: an untouched new
    field must not change anything for a config written before it existed."""
    assert config.global_demeanour is None
    assert all(member.demeanour is None for member in cast.values())
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert "focus of this shot." in result.prompt


# --------------------------------------------------------------------------- #
# The render-side avoid list (issue #73)
# --------------------------------------------------------------------------- #


def test_avoid_list_is_never_composed_into_a_prompt(config: RunConfig, cast):
    """Issue #73, reversed by measurement on "Deathless" chunk 72.

    An A/B at an identical seed, one clause different: WITH the avoid list
    ("modern climbing equipment, ropes, harnesses or safety gear") H3 put a
    full climbing harness, carabiners and a trailing rope on a character
    whose role has no climbing in it and whose shot line has no climbing in
    it. WITHOUT it, plain trousers and nothing else changed -- same framing,
    same background, same people.

    H3 exposes one prompt input into a single BasicGuider; there is no
    negative-conditioning channel. So a prohibition can only ever ADD its own
    nouns to the only channel there is, and where the scene supplies no other
    source for them, that is where they come from. On a chunk that already
    implied climbing the same clause changed nothing, which is why one chunk
    alone would have read as "inert" -- it manufactures what it forbids
    precisely where it has nothing to suppress.

    The authoring-side `avoid` (concept -> beats/prose, where a real model
    reads it as an instruction) is untouched and still useful. This is only
    about the render path."""
    cfg = dataclasses.replace(config, avoid=("bass guitar", "climbing harness"))
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(cfg, chunk)

    assert "bass guitar" not in result.prompt.lower()
    assert "climbing harness" not in result.prompt.lower()
    assert "never depict" not in result.prompt.lower()


def test_setting_avoid_changes_nothing_about_the_composed_prompt(config: RunConfig, cast):
    """The whole prompt, byte-for-byte, with and without an avoid list."""
    chunk = _chunk(text="the lucky ones", character="Dianne")
    with_avoid = dataclasses.replace(config, avoid=("bass guitar", "microphone"))

    assert expand_prompt(with_avoid, chunk).prompt == expand_prompt(config, chunk).prompt


def test_empty_avoid_list_produces_no_avoid_clause(config: RunConfig, cast):
    assert config.avoid == ()
    chunk = _chunk(text="the lucky ones", character="Dianne")

    result = expand_prompt(config, chunk)

    assert "must never depict" not in result.prompt
    assert "must never depict" not in result.chained_prompt


def test_instrumental_clause_is_a_positive_state_not_a_prohibition():
    """Issue #73, applied to the most-composed prohibition in the codebase.

    The clause read "the character performs silently, no lyric to sing" for
    every instrumental chunk -- 39 of 80 on "Deathless". H3 exposes exactly one
    ``prompt`` input into a single ``BasicGuider``: there is no
    negative-conditioning channel, so a prohibition cannot subtract, and the
    tokens actually reaching the model were *lyric* and *sing*. A viewer
    reported Jan mouthing words through instrumental passages at 3:14, 7:11 and
    8:27, and "his lips never stop moving after 3:49".

    ``_present_clause`` forty lines below already learned this -- its docstring
    reads "Says 'silent' positively rather than 'not singing'". This clause
    never got the same treatment.

    The second assertion is the trap on the other side: issue #74 measured that
    naming facial anatomy puts a camera on it (100% face presence against
    83/0/33% for manner-only phrasing). An instrumental chunk is frequently an
    authored landscape, so "mouth closed" would buy the fix by silently
    dragging 39 wide shots into close-ups.
    """
    from music_video_maker import prompting

    for clause in (
        prompting._INSTRUMENTAL_CLAUSE_SINGULAR,
        prompting._INSTRUMENTAL_CLAUSE_PLURAL,
    ):
        lowered = clause.lower()
        for prohibition_token in ("sing", "lyric", " no ", "not "):
            assert prohibition_token not in lowered, (
                f"{clause!r} still carries the prohibition token "
                f"{prohibition_token!r}; say what IS true (issue #73)"
            )
        for anatomy in ("mouth", "lips", "jaw", "teeth", "face", "eyes"):
            assert anatomy not in lowered, (
                f"{clause!r} names facial anatomy ({anatomy!r}), which pulls the "
                f"camera onto the face across every instrumental chunk (issue #74)"
            )
