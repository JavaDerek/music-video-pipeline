"""Tests for ``prompt_format = "structured"`` (issue #99).

MiniMax publishes a full-reference prompt grammar for H3
(``docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`` in the MiniMax-H3 model repo)
whose ``retention_analysis`` section says, per input, what that input is FOR.
This project composes prose. ``"structured"`` re-houses the *same* sentences
the prose path composes in that grammar, plus the declarations the grammar
exists to carry, so that an A/B between the two formats varies the format and
not the content. These tests pin both halves of that: the grammar is the
guide's, and nothing the prose says is lost or reworded on the way in.

Opt-in and unrendered at the time of writing -- the default stays ``"prose"``
and must compose byte-identically to before this field existed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.config import RunConfig
from music_video_maker.prompting import expand_prompt
from tests.harness.factories import make_cast_dict

GLOBAL_STYLE = "Refestramus progressive rock music video, 35mm film"
NARRATIVE_CONCEPT = "A mountain watch-post at dawn"

SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        master_audio=Path("audio/master.wav"),
        lyrics_file=Path("lyrics.txt"),
        global_style=GLOBAL_STYLE,
        narrative_concept=NARRATIVE_CONCEPT,
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=Path("output/chunks"),
        final_video_dir=Path("output/final"),
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
    )


@pytest.fixture
def structured(config: RunConfig) -> RunConfig:
    return dataclasses.replace(config, prompt_format="structured")


def _chunk(chunk_id: int = 0, text: str = "", characters: tuple[str, ...] = ()):
    return contracts.AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunks/chunk_{chunk_id:04d}.wav"),
        start=0.0,
        end=6.0,
        text=text,
        characters=characters,
    )


def _sections(prompt: str) -> dict[str, str]:
    """Split a structured prompt into {section name: body}."""
    found: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in prompt.splitlines():
        header = line.rstrip(":")
        if line.endswith(":") and header in SECTIONS:
            if current is not None:
                found[current] = "\n".join(lines).strip()
            current, lines = header, []
        elif current is not None:
            lines.append(line)
    if current is not None:
        found[current] = "\n".join(lines).strip()
    return found


# --------------------------------------------------------------------------- #
# The default is untouched
# --------------------------------------------------------------------------- #


def test_prose_is_the_default(config: RunConfig) -> None:
    assert config.prompt_format == "prose"


def test_explicit_prose_composes_byte_identically_to_the_default(config: RunConfig) -> None:
    """Every cached chunk and every fingerprint in existence is prose; naming
    the format explicitly must not move a single byte of it."""
    chunk = _chunk(text="Our lives are prisons,", characters=("Dianne",))
    explicit = dataclasses.replace(config, prompt_format="prose")
    assert expand_prompt(explicit, chunk, shot="She climbs") == expand_prompt(
        config, chunk, shot="She climbs"
    )


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #


def test_voiced_chunk_composes_the_guide_grammar_exactly(structured: RunConfig) -> None:
    chunk = _chunk(text="Our lives are prisons,", characters=("Dianne",))
    prompt = expand_prompt(structured, chunk, shot="She climbs the ridge").prompt
    assert prompt == (
        "subject_definitions:\n"
        "<Subject 1> is Dianne in <Picture 1>, Lead Vocalist, smiling constantly, oblivious.\n"
        "<Audio 1> is this shot's excerpt of the song, reused as its complete soundtrack; "
        "its sung vocal is performed by <Subject 1> (S1).\n"
        "\n"
        "summary:\n"
        "[reference generation + audio reuse] One continuous shot in which <Subject 1> "
        "sings to <Audio 1>.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - the likeness of Dianne in "
        "<Picture 1> is retained.\n"
        "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete "
        "final audio track.\n"
        "\n"
        "detailed_description:\n"
        "Refestramus progressive rock music video, 35mm film.\n"
        "[Shot 1] She climbs the ridge. Dianne, Lead Vocalist, smiling constantly, "
        "oblivious, is the focus of this shot. <Subject 1> (S1) sings, "
        "<d>[English] Our lives are prisons,</d>\n"
        "\n"
        "overall_soundscape:\n"
        "N/A\n"
        "\n"
        "non_diegetic_music:\n"
        "The instrumental layer of <Audio 1> is reused as the audience-only score."
    )


def test_instrumental_chunk_declares_the_audio_and_keeps_the_silent_clause(
    structured: RunConfig,
) -> None:
    prompt = expand_prompt(structured, _chunk(), shot="She climbs the ridge").prompt
    sections = _sections(prompt)
    assert "<d>" not in prompt
    assert "(S1)" not in prompt
    assert "it is an instrumental passage" in sections["subject_definitions"]
    assert sections["retention_analysis"].endswith(
        "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete "
        "final audio track."
    )
    # The prose path's own instrumental sentence, verbatim -- not a new,
    # stronger one. Adding "mouth closed" here would test content, not format.
    assert "Instrumental passage: the character stays silent throughout this shot" in (
        sections["detailed_description"]
    )
    assert sections["non_diegetic_music"] == (
        "<Audio 1> is directly reused as the complete audience-only score."
    )
    assert sections["summary"] == (
        "[reference generation + audio reuse] One continuous shot featuring <Subject 1>, "
        "set to <Audio 1>."
    )


def test_all_six_sections_appear_once_in_the_guide_order(structured: RunConfig) -> None:
    for chunk in (_chunk(), _chunk(text="la la", characters=("Dianne",))):
        prompt = expand_prompt(structured, chunk).prompt
        positions = [prompt.index(f"{name}:\n") for name in SECTIONS]
        assert positions == sorted(positions)
        for name in SECTIONS:
            assert prompt.count(f"{name}:\n") == 1


def test_speaker_ids_never_appear_in_retention_analysis(structured: RunConfig) -> None:
    """The guide: 'Do not write (Sx) in retention_analysis.'"""
    chunk = _chunk(text="together now", characters=("Dianne", "Marcus"))
    sections = _sections(expand_prompt(structured, chunk).prompt)
    assert "(S" not in sections["retention_analysis"]


def test_two_singers_get_their_own_speaker_ids(structured: RunConfig) -> None:
    chunk = _chunk(text="together now", characters=("Dianne", "Marcus"))
    detailed = _sections(expand_prompt(structured, chunk).prompt)["detailed_description"]
    assert "<Subject 1> (S1) and <Subject 2> (S2) sing, <d>[English] together now</d>" in (
        detailed
    )


def test_present_cast_is_numbered_after_the_singers_matching_the_staged_photos(
    structured: RunConfig,
) -> None:
    """<Picture k> must be the k-th photo the graph is actually handed:
    ``image_refs`` is singers first, then present cast, and that is the order
    ``ref_images.ref_image_0..N`` is wired in."""
    chunk = _chunk(text="la la", characters=("Dianne",))
    expanded = expand_prompt(structured, chunk, present=("Rex",))
    assert expanded.image_refs == (Path("cast/dianne_ref.png"), Path("cast/rex_ref.png"))
    definitions = _sections(expanded.prompt)["subject_definitions"]
    assert "<Subject 1> is Dianne in <Picture 1>," in definitions
    assert "<Subject 2> is Rex in <Picture 2>," in definitions
    summary = _sections(expanded.prompt)["summary"]
    assert summary.endswith("<Subject 2> is also in shot.")


def test_lyric_language_comes_from_config(structured: RunConfig) -> None:
    french = dataclasses.replace(structured, lyric_language="French")
    chunk = _chunk(text="Nos vies sont des prisons,", characters=("Dianne",))
    assert "<d>[French] Nos vies sont des prisons,</d>" in expand_prompt(french, chunk).prompt


def test_spoken_timeline_says_rather_than_sings(structured: RunConfig) -> None:
    chunk = _chunk(text="Once upon a time.", characters=("Dianne",))
    prompt = expand_prompt(structured, chunk, spoken=True).prompt
    assert "<Subject 1> (S1) says, <d>[English] Once upon a time.</d>" in prompt
    assert "its spoken line is performed by <Subject 1> (S1)" in prompt
    assert "sings" not in prompt


def test_chained_variant_is_not_composed(structured: RunConfig) -> None:
    """The grammar binds <Picture k> to staged photos and the chained graph
    has none; config refuses structured + i2v_continuity, and this is the
    defence in depth."""
    assert expand_prompt(structured, _chunk()).chained_prompt is None


def test_lora_trigger_leads_bare(structured: RunConfig) -> None:
    lora = dataclasses.replace(structured, lora="realism.safetensors", lora_trigger="R3AL")
    assert expand_prompt(lora, _chunk()).prompt.startswith("R3AL\n\nsubject_definitions:\n")


# --------------------------------------------------------------------------- #
# Format, not content
# --------------------------------------------------------------------------- #


def test_every_prose_sentence_but_the_lyric_survives_verbatim(structured: RunConfig) -> None:
    """The A/B this exists for compares formats. If the structured path
    reworded or dropped a sentence the prose path composes, a difference on
    pixels could be content, not format -- which is exactly what confounds
    the first two-chunk A/B (its arm B was hand-written and added "keeps her
    mouth closed", a sentence the prose path deliberately never composes)."""
    config = dataclasses.replace(
        structured,
        setting="A Slavic mountain watch-post",
        cinematography="anamorphic, heavy grain",
    )
    prose_config = dataclasses.replace(config, prompt_format="prose")
    kwargs = dict(
        shot="She climbs the ridge",
        camera="close on her face",
        present=("Rex",),
        location="the summit",
        conditions="snow",
        framing="close",
    )
    for chunk in (_chunk(text="la la", characters=("Dianne",)), _chunk()):
        prose = expand_prompt(prose_config, chunk, **kwargs).prompt
        structured_prompt = expand_prompt(config, chunk, **kwargs).prompt
        sentences = [s.strip().rstrip(".") for s in prose.split(". ")]
        kept = [s for s in sentences if "actively singing" not in s]
        assert len(kept) >= 8
        for sentence in kept:
            assert sentence in structured_prompt, sentence
