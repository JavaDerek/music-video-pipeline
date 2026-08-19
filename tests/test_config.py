"""Tests for the run configuration model (issue #2).

All tests run fully offline against ``tmp_path`` -- no network, no GPU, no
real ComfyUI server. The one shared fixture (a TOML template with ``{base}``
placeholders) lives in ``tests/fixtures/config/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from music_video_maker import config as config_module
from music_video_maker.config import (
    ALIGNMENT_MODEL_SIZES,
    DEFAULT_ALIGNMENT_MODEL_SIZE,
    REALISM_LORA,
    REALISM_LORA_STRENGTH,
    REALISM_LORA_TRIGGER,
    ConfigError,
    RunConfig,
    load_config,
)
from music_video_maker.contracts import CastMember, HardwareProfile
from music_video_maker.faces import DEFAULT_MIN_FACE_SIMILARITY, resolve_recognition_model_path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "config"

DEFAULT_CAST_TOML = """
[cast.Dianne]
role = "Lead Vocalist, smiling constantly, oblivious"
image = "{cast_dir}/dianne_ref_01.jpg"

[cast.Rex]
role = "Drummer, background, never sings"
image = "{cast_dir}/rex_ref.jpg"
"""

DEFAULT_HARDWARE_TOML = """
[hardware]
name = "RTX 4090"
vram_gb = 24.0
"""


def _create_default_assets(tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "master.wav").write_bytes(b"RIFF-fake-wav-data")
    (tmp_path / "lyrics.txt").write_text("la la la\n")
    (tmp_path / "cast").mkdir()
    (tmp_path / "cast" / "dianne_ref_01.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    (tmp_path / "cast" / "rex_ref.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    (tmp_path / "workflow_api.json").write_text("{}")


def _write_config(
    tmp_path: Path,
    *,
    master_audio: str | None = None,
    lyrics_file: str | None = None,
    workflow_template: str | None = None,
    default_lead_vocalist: str = "Dianne",
    comfyui_url: str = "http://doris:8188",
    cast_block: str | None = None,
    hardware_block: str | None = None,
    chunks_dir: str | None = None,
    final_video_dir: str | None = None,
    global_style: str = "Refestramus progressive rock music video, 35mm film",
    narrative_concept: str = "Wandering through a surgery, kicking a life support plug",
    filename: str = "run.toml",
    extra_toml: str = "",
) -> Path:
    """Assemble a run config TOML file directly in ``tmp_path``.

    Every field defaults to a valid value pointing at files created by
    :func:`_create_default_assets`, so a test only needs to override the one
    field it is exercising.
    """
    default_audio = str(tmp_path / "audio" / "master.wav")
    default_lyrics = str(tmp_path / "lyrics.txt")
    default_workflow = str(tmp_path / "workflow_api.json")
    default_chunks_dir = str(tmp_path / "output" / "chunks")
    default_final_dir = str(tmp_path / "output" / "final")
    default_cast_block = DEFAULT_CAST_TOML.format(cast_dir=tmp_path / "cast")

    master_audio = default_audio if master_audio is None else master_audio
    lyrics_file = default_lyrics if lyrics_file is None else lyrics_file
    workflow_template = default_workflow if workflow_template is None else workflow_template
    chunks_dir = default_chunks_dir if chunks_dir is None else chunks_dir
    final_video_dir = default_final_dir if final_video_dir is None else final_video_dir
    cast_block = default_cast_block if cast_block is None else cast_block
    hardware_block = DEFAULT_HARDWARE_TOML if hardware_block is None else hardware_block

    content = f"""
master_audio = "{master_audio}"
lyrics_file = "{lyrics_file}"
global_style = "{global_style}"
narrative_concept = "{narrative_concept}"
default_lead_vocalist = "{default_lead_vocalist}"
comfyui_url = "{comfyui_url}"
workflow_template = "{workflow_template}"
chunks_dir = "{chunks_dir}"
final_video_dir = "{final_video_dir}"
{extra_toml}

{cast_block}

{hardware_block}
"""
    config_path = tmp_path / filename
    config_path.write_text(content)
    return config_path


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_happy_path_loads_full_config(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(config_path)

    assert isinstance(cfg, RunConfig)
    assert cfg.master_audio == tmp_path / "audio" / "master.wav"
    assert cfg.lyrics_file == tmp_path / "lyrics.txt"
    assert cfg.global_style == "Refestramus progressive rock music video, 35mm film"
    assert cfg.narrative_concept.startswith("Wandering through a surgery")
    assert cfg.default_lead_vocalist == "Dianne"
    assert cfg.comfyui_url == "http://doris:8188"
    assert cfg.workflow_template == tmp_path / "workflow_api.json"
    assert cfg.chunks_dir == tmp_path / "output" / "chunks"
    assert cfg.final_video_dir == tmp_path / "output" / "final"
    assert isinstance(cfg.hardware, HardwareProfile)
    assert cfg.hardware.name == "RTX 4090"
    assert cfg.hardware.vram_gb == pytest.approx(24.0)


def test_cast_dictionary_supports_a_never_vocalist_member(tmp_path: Path) -> None:
    """A drummer who is never the active vocalist must still be representable."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(config_path)

    assert set(cfg.cast) == {"Dianne", "Rex"}
    assert isinstance(cfg.cast["Rex"], CastMember)
    assert cfg.cast["Rex"].role == "Drummer, background, never sings"
    assert cfg.cast["Rex"].image == tmp_path / "cast" / "rex_ref.jpg"
    # Rex is never the default lead vocalist and nothing requires he ever be one.
    assert cfg.default_lead_vocalist != "Rex"


def test_default_comfyui_url_assumes_co_location(tmp_path: Path) -> None:
    """Issue #50: the default is loopback, not a named machine. A remote host
    (e.g. a Tailscale hostname) is a config override, never the assumption."""
    _create_default_assets(tmp_path)
    # Write a config with no comfyui_url key at all by hand-crafting the file.
    content = f"""
master_audio = "{tmp_path / "audio" / "master.wav"}"
lyrics_file = "{tmp_path / "lyrics.txt"}"
global_style = "style"
narrative_concept = "concept"
default_lead_vocalist = "Dianne"
workflow_template = "{tmp_path / "workflow_api.json"}"
chunks_dir = "{tmp_path / "output" / "chunks"}"
final_video_dir = "{tmp_path / "output" / "final"}"

{DEFAULT_CAST_TOML.format(cast_dir=tmp_path / "cast")}

{DEFAULT_HARDWARE_TOML}
"""
    config_path = tmp_path / "run.toml"
    config_path.write_text(content)

    cfg = load_config(config_path)

    assert cfg.comfyui_url == "http://127.0.0.1:8188"


def test_relative_paths_resolve_against_config_file_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _create_default_assets(project_dir)

    config_path = _write_config(
        project_dir,
        master_audio="audio/master.wav",
        lyrics_file="lyrics.txt",
        workflow_template="workflow_api.json",
        chunks_dir="output/chunks",
        final_video_dir="output/final",
        cast_block=DEFAULT_CAST_TOML.format(cast_dir="cast"),
    )

    cfg = load_config(config_path)

    assert cfg.master_audio == project_dir / "audio" / "master.wav"
    assert cfg.lyrics_file == project_dir / "lyrics.txt"
    assert cfg.workflow_template == project_dir / "workflow_api.json"
    assert cfg.chunks_dir == project_dir / "output" / "chunks"
    assert cfg.cast["Dianne"].image == project_dir / "cast" / "dianne_ref_01.jpg"


def test_output_dirs_are_created_when_missing(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)
    chunks_dir = tmp_path / "output" / "chunks"
    final_video_dir = tmp_path / "output" / "final"
    assert not chunks_dir.exists()
    assert not final_video_dir.exists()

    load_config(config_path)

    assert chunks_dir.is_dir()
    assert final_video_dir.is_dir()


def test_fixture_template_loads(tmp_path: Path) -> None:
    """Exercise the shared TOML fixture under tests/fixtures/config/."""
    _create_default_assets(tmp_path)
    template = (FIXTURES_DIR / "valid_template.toml").read_text()
    config_path = tmp_path / "run.toml"
    config_path.write_text(template.format(base=tmp_path))

    cfg = load_config(config_path)

    assert cfg.default_lead_vocalist == "Dianne"
    assert set(cfg.cast) == {"Dianne", "Rex"}
    assert cfg.hardware.recommended_nodes == ("SageAttention",)


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #


def test_overrides_beat_file_values(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(
        config_path,
        comfyui_url="http://doris:9999",
        default_lead_vocalist="Rex",
        global_style="overridden style",
    )

    assert cfg.comfyui_url == "http://doris:9999"
    assert cfg.default_lead_vocalist == "Rex"
    assert cfg.global_style == "overridden style"


def test_override_path_field_is_resolved(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)
    other_workflow = tmp_path / "alt_workflow.json"
    other_workflow.write_text("{}")

    cfg = load_config(config_path, workflow_template=str(other_workflow))

    assert cfg.workflow_template == other_workflow


def test_unknown_override_key_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="overrides"):
        load_config(config_path, not_a_real_field="oops")

    assert "overrides" in caplog.text


def test_cast_demeanour_override_no_longer_exists(tmp_path: Path) -> None:
    """Issue #74's completed migration: ``cast_demeanour`` was a stand-in
    sibling dict on ``RunConfig`` because ``contracts.py`` was locked at the
    time. It is gone now that ``CastMember.demeanour`` is the one real field,
    so passing it as an override must be refused exactly like any other
    unknown override key -- there must be exactly one way to express this."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    with pytest.raises(ConfigError, match="overrides"):
        load_config(config_path, cast_demeanour={"Dianne": "grave and unsmiling"})


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


def test_missing_master_audio_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, master_audio=str(tmp_path / "audio" / "nope.wav"))

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="master_audio"):
        load_config(config_path)

    assert "master_audio" in caplog.text


def test_missing_lyrics_file_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, lyrics_file=str(tmp_path / "nope_lyrics.txt"))

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="lyrics_file"):
        load_config(config_path)

    assert "lyrics_file" in caplog.text


def test_missing_cast_image_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    missing_cast = f"""
[cast.Dianne]
role = "Lead Vocalist, smiling constantly, oblivious"
image = "{tmp_path / "cast" / "does_not_exist.jpg"}"
"""
    config_path = _write_config(tmp_path, cast_block=missing_cast)

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="cast.Dianne.image"):
        load_config(config_path)

    assert "cast.Dianne.image" in caplog.text


def test_unreadable_cast_image_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    image_path = tmp_path / "cast" / "dianne_ref_01.jpg"
    os.chmod(image_path, 0o000)
    config_path = _write_config(tmp_path)

    try:
        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(ConfigError, match="not readable"),
        ):
            load_config(config_path)
    finally:
        os.chmod(image_path, 0o644)

    assert "not readable" in caplog.text


def test_unknown_default_lead_vocalist_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, default_lead_vocalist="Nobody")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(ConfigError, match="default_lead_vocalist"),
    ):
        load_config(config_path)

    assert "default_lead_vocalist" in caplog.text


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://doris:8188", "doris:8188"])
def test_malformed_comfyui_url_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, bad_url: str
) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, comfyui_url=bad_url)

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="comfyui_url"):
        load_config(config_path)

    assert "comfyui_url" in caplog.text


def test_empty_comfyui_url_falls_back_to_default(tmp_path: Path) -> None:
    """An explicit empty string is treated the same as an absent key."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, comfyui_url="")

    cfg = load_config(config_path)

    assert cfg.comfyui_url == "http://127.0.0.1:8188"


def test_missing_workflow_template_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, workflow_template=str(tmp_path / "no_such_workflow.json")
    )

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="workflow_template"):
        load_config(config_path)

    assert "workflow_template" in caplog.text


def test_missing_hardware_table_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, hardware_block="")

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="hardware"):
        load_config(config_path)

    assert "hardware" in caplog.text


def test_malformed_toml_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = FIXTURES_DIR / "malformed.toml"

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_config_file_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "does_not_exist.toml"

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError):
        load_config(config_path)


def test_empty_cast_dictionary_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, cast_block="")

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match="cast"):
        load_config(config_path)


# --------------------------------------------------------------------------- #
# Wave 3 seams: resilience (#10) and I2V continuity (#12)
# --------------------------------------------------------------------------- #


def test_wave3_knobs_have_defaults_when_the_file_omits_them(tmp_path: Path) -> None:
    """Every Wave 3 knob is optional -- an existing config keeps working."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(config_path)

    assert cfg.watchdog_timeout_seconds == pytest.approx(900.0)
    assert cfg.max_render_attempts == 3
    assert cfg.retry_backoff_seconds == pytest.approx(5.0)
    assert cfg.min_free_disk_gb == pytest.approx(20.0)
    assert cfg.i2v_continuity is False
    assert cfg.i2v_workflow_template is None
    assert cfg.resume_ignore_prompt_changes is False


# --------------------------------------------------------------------------- #
# Cross-video continuity fields (#31, #32) and alignment strictness (#35).
#
# Each of these is a property that must hold across the WHOLE video but was
# previously expressible only as free text repeated in forty places -- or
# smuggled into `role`, which is what made the performer sing through every
# instrumental. Anything without a home drifts.
# --------------------------------------------------------------------------- #


def test_setting_is_read_and_applies_to_the_whole_video(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, extra_toml='setting = "London, UK -- contemporary, overcast winter"'
    )

    assert load_config(config_path).setting == "London, UK -- contemporary, overcast winter"


def test_a_missing_setting_warns_loudly_rather_than_passing_in_silence(
    tmp_path: Path, caplog
) -> None:
    """Silence is what produced the drift: the first render walked out of a
    British front door into an American park because nothing named a place."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    with caplog.at_level(logging.WARNING):
        cfg = load_config(config_path)

    assert cfg.setting is None
    assert any("setting" in r.getMessage().lower() for r in caplog.records)


def test_an_empty_setting_is_treated_as_unset(tmp_path: Path, caplog) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='setting = "   "')

    with caplog.at_level(logging.WARNING):
        cfg = load_config(config_path)

    assert cfg.setting is None
    assert any("setting" in r.getMessage().lower() for r in caplog.records)


def test_global_appearance_is_read(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, extra_toml='global_appearance = "everyone slim, trim and healthy looking"'
    )

    assert load_config(config_path).global_appearance == "everyone slim, trim and healthy looking"


def test_cinematography_is_read(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path,
        extra_toml='cinematography = "35mm film, shallow depth of field, warm natural light"',
    )

    assert (
        load_config(config_path).cinematography
        == "35mm film, shallow depth of field, warm natural light"
    )


def test_cinematography_defaults_to_none(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    assert load_config(config_path).cinematography is None


def test_an_empty_cinematography_is_treated_as_unset(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='cinematography = "   "')

    assert load_config(config_path).cinematography is None


def test_cast_appearance_is_read_and_is_separate_from_role(tmp_path: Path) -> None:
    """Appearance applies to every chunk the member is in, exactly like role --
    which is why it must not live inside role. See #31."""
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Lead Vocalist, smiling constantly, oblivious"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
appearance = "looking a few years younger, softly lit, flattering"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    member = load_config(config_path).cast["Dianne"]

    assert member.appearance == "looking a few years younger, softly lit, flattering"
    assert "younger" not in member.role


def test_cast_appearance_defaults_to_none(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    assert load_config(config_path).cast["Dianne"].appearance is None


def test_an_unknown_key_in_a_cast_table_is_rejected(tmp_path: Path) -> None:
    """``appearence`` silently doing nothing is the same failure mode as a
    misplaced top-level key: config that reads as applied but is ignored."""
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Lead Vocalist"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
appearence = "typo'd on purpose"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)

    assert "appearence" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Demeanour (issue #74)
# --------------------------------------------------------------------------- #


def test_global_demeanour_is_read(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, extra_toml='global_demeanour = "nobody smiles; this is a war"'
    )

    assert load_config(config_path).global_demeanour == "nobody smiles; this is a war"


def test_global_demeanour_defaults_to_none(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    assert load_config(config_path).global_demeanour is None


def test_an_empty_global_demeanour_is_treated_as_unset(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='global_demeanour = "   "')

    assert load_config(config_path).global_demeanour is None


def test_member_demeanour_is_read_and_is_separate_from_role_and_appearance(
    tmp_path: Path,
) -> None:
    """Follows the appearance/global_appearance precedent (#31): demeanour
    applies to every chunk the member is in, exactly like role and
    appearance, which is why it must not live inside either of them (#74).
    A plain ``CastMember.demeanour`` attribute, matching ``appearance``'s own
    shape -- not a sibling dict on ``RunConfig``."""
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Lead Vocalist, smiling constantly, oblivious"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
appearance = "in her late forties, softly lit"
demeanour = "grave and unsmiling, exhausted"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    cfg = load_config(config_path)

    assert cfg.cast["Dianne"].demeanour == "grave and unsmiling, exhausted"
    assert "grave" not in cfg.cast["Dianne"].role
    assert "grave" not in (cfg.cast["Dianne"].appearance or "")


def test_member_demeanour_defaults_to_none_when_nobody_sets_one(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(config_path)

    assert cfg.cast["Dianne"].demeanour is None
    assert cfg.cast["Rex"].demeanour is None


def test_only_members_with_a_demeanour_have_one_set(tmp_path: Path) -> None:
    """Rex (no demeanour) must resolve to ``None`` -- absence, not an empty
    string, is how "nobody set one" is represented."""
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Lead Vocalist, smiling constantly, oblivious"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
demeanour = "grave and unsmiling"

[cast.Rex]
role = "Drummer, background, never sings"
image = "{tmp_path / "cast" / "rex_ref.jpg"}"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    cast = load_config(config_path).cast

    assert cast["Dianne"].demeanour == "grave and unsmiling"
    assert cast["Rex"].demeanour is None


def test_role_prohibition_warns_at_load(tmp_path: Path, caplog) -> None:
    """The #73 finding: 'never holding anything' rendered a bass guitar twice
    anyway. Nothing warned that the fix did not work -- this closes that gap
    at load time."""
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Weathered watchman, patient and unhurried, never holding anything"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    with caplog.at_level(logging.WARNING):
        load_config(config_path)

    assert any(
        "prohibition" in r.getMessage().lower() and "Dianne" in r.getMessage()
        for r in caplog.records
    )


def test_role_prohibition_about_singing_does_not_warn(tmp_path: Path, caplog) -> None:
    """'never sings' is not the #73 bug: whether a member sings is decided by
    chunk.characters, never by the literal text of role, so the prohibition
    warning must not fire on the project's own shipped example."""
    _create_default_assets(tmp_path)  # DEFAULT_CAST_TOML's Rex reads "never sings"
    config_path = _write_config(tmp_path)

    with caplog.at_level(logging.WARNING):
        load_config(config_path)

    assert not any(
        "prohibition" in r.getMessage().lower() and "Rex" in r.getMessage()
        for r in caplog.records
    )


def test_demeanour_prohibition_warns(tmp_path: Path, caplog) -> None:
    _create_default_assets(tmp_path)
    cast_block = f"""
[cast.Dianne]
role = "Lead Vocalist"
image = "{tmp_path / "cast" / "dianne_ref_01.jpg"}"
demeanour = "never amused"
"""
    config_path = _write_config(tmp_path, cast_block=cast_block)

    with caplog.at_level(logging.WARNING):
        load_config(config_path)

    assert any(
        "prohibition" in r.getMessage().lower() and "demeanour" in r.getMessage().lower()
        for r in caplog.records
    )


def test_demeanour_displacement_warns(tmp_path: Path, caplog) -> None:
    """Issue #46: a displacement compounds on the chained I2V path, where each
    chunk conditions on the previous chunk's own output."""
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, extra_toml='global_demeanour = "increasingly grim as the war drags on"'
    )

    with caplog.at_level(logging.WARNING):
        load_config(config_path)

    assert any(
        "displacement" in r.getMessage().lower() and "global_demeanour" in r.getMessage()
        for r in caplog.records
    )


def test_demeanour_endpoint_phrasing_does_not_warn(tmp_path: Path, caplog) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='global_demeanour = "grave and unsmiling"')

    with caplog.at_level(logging.WARNING):
        load_config(config_path)

    assert not any(
        ("prohibition" in r.getMessage().lower() or "displacement" in r.getMessage().lower())
        for r in caplog.records
    )


# --------------------------------------------------------------------------- #
# The render-side avoid list (issue #73)
# --------------------------------------------------------------------------- #


def test_avoid_defaults_to_empty_tuple(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    assert load_config(config_path).avoid == ()


def test_avoid_is_read_from_toml(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path, extra_toml='avoid = ["bass guitar", "microphone", "stage"]'
    )

    assert load_config(config_path).avoid == ("bass guitar", "microphone", "stage")


def test_avoid_rejects_a_non_list(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='avoid = "bass guitar"')

    with pytest.raises(ConfigError, match="avoid"):
        load_config(config_path)


def test_avoid_rejects_a_non_string_item(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml="avoid = [1, 2]")

    with pytest.raises(ConfigError, match="avoid"):
        load_config(config_path)


def test_avoid_rejects_a_blank_item(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='avoid = ["   "]')

    with pytest.raises(ConfigError, match="avoid"):
        load_config(config_path)


def test_strict_alignment_defaults_off_and_is_readable(tmp_path: Path) -> None:
    """Report-only by default: a real song always has some odd segments, so
    refusing by default would be wrong (#35)."""
    _create_default_assets(tmp_path)

    assert load_config(_write_config(tmp_path)).strict_alignment is False
    assert (
        load_config(
            _write_config(tmp_path, extra_toml="strict_alignment = true", filename="strict.toml")
        ).strict_alignment
        is True
    )


def test_strict_alignment_rejects_a_non_boolean(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='strict_alignment = "yes"')

    with pytest.raises(ConfigError, match="strict_alignment"):
        load_config(config_path)


def test_resume_ignore_prompt_changes_is_readable_from_the_file(tmp_path: Path) -> None:
    """Issue #34's escape hatch: reuse chunks whose span is unchanged but whose
    prompt was edited. Off by default -- a resumed run reflects the config."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml="resume_ignore_prompt_changes = true")

    assert load_config(config_path).resume_ignore_prompt_changes is True


def test_resume_ignore_prompt_changes_rejects_a_non_boolean(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='resume_ignore_prompt_changes = "yes"')

    with pytest.raises(ConfigError, match="resume_ignore_prompt_changes"):
        load_config(config_path)


def test_run_state_file_defaults_under_chunks_dir(tmp_path: Path) -> None:
    """load_config always resolves run_state_file, so #10 never has to guess."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)

    cfg = load_config(config_path)

    assert cfg.run_state_file == cfg.chunks_dir / "run_state.json"


def test_run_state_file_is_resolved_against_the_config_directory(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml='run_state_file = "state/run.json"')

    cfg = load_config(config_path)

    assert cfg.run_state_file == tmp_path / "state" / "run.json"


def test_wave3_knobs_are_read_from_the_file(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path,
        extra_toml=(
            "watchdog_timeout_seconds = 120.5\n"
            "max_render_attempts = 5\n"
            "retry_backoff_seconds = 2.0\n"
            "min_free_disk_gb = 8.5\n"
        ),
    )

    cfg = load_config(config_path)

    assert cfg.watchdog_timeout_seconds == pytest.approx(120.5)
    assert cfg.max_render_attempts == 5
    assert cfg.retry_backoff_seconds == pytest.approx(2.0)
    assert cfg.min_free_disk_gb == pytest.approx(8.5)


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ("watchdog_timeout_seconds = 0", "watchdog_timeout_seconds"),
        ("watchdog_timeout_seconds = -1.0", "watchdog_timeout_seconds"),
        ("max_render_attempts = 0", "max_render_attempts"),
        ("retry_backoff_seconds = -0.5", "retry_backoff_seconds"),
        ("min_free_disk_gb = -1", "min_free_disk_gb"),
    ],
)
def test_nonsensical_wave3_numbers_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, line: str, field: str
) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml=line)

    with caplog.at_level(logging.ERROR), pytest.raises(ConfigError, match=field):
        load_config(config_path)


def test_i2v_continuity_without_a_template_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Fail loudly at load time rather than mid-run on chunk 1."""
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path, extra_toml="i2v_continuity = true")

    with caplog.at_level(logging.ERROR), pytest.raises(
        ConfigError, match="i2v_workflow_template"
    ):
        load_config(config_path)


def test_i2v_template_is_resolved_and_validated(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    (tmp_path / "workflow_i2v.json").write_text("{}")
    config_path = _write_config(
        tmp_path,
        extra_toml='i2v_continuity = true\ni2v_workflow_template = "workflow_i2v.json"',
    )

    cfg = load_config(config_path)

    assert cfg.i2v_continuity is True
    assert cfg.i2v_workflow_template == tmp_path / "workflow_i2v.json"


def test_missing_i2v_template_file_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(
        tmp_path,
        extra_toml='i2v_continuity = true\ni2v_workflow_template = "nope.json"',
    )

    with caplog.at_level(logging.ERROR), pytest.raises(
        ConfigError, match="i2v_workflow_template"
    ):
        load_config(config_path)


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #


def test_module_has_a_logger() -> None:
    assert config_module.logger.name == "music_video_maker.config"


def test_run_config_is_frozen(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    config_path = _write_config(tmp_path)
    cfg = load_config(config_path)

    with pytest.raises(AttributeError):
        cfg.comfyui_url = "http://other:8188"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Render resolution
#
# H3's width/height are a run-level choice, not a template constant: the same
# authored graph gets rendered small to preview a shot plan cheaply and large
# for the finished video. Cost scales with the latent volume -- 1344x768 was
# measured at 9.25 min/chunk on the 4090 against ~3 min at 864x480 -- so this
# is the single biggest lever on how long a run takes.
#
# The live node schema (docs/h3-node-schema.md) declares min=32, max=16384,
# step=32 for both, so anything off that grid is rejected here rather than
# silently rounded by ComfyUI at execution time.
# --------------------------------------------------------------------------- #


def test_render_dimensions_default_to_none_so_the_template_wins(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.render_width is None
    assert cfg.render_height is None


def test_render_dimensions_are_read_from_toml(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    path = _write_config(tmp_path, extra_toml="render_width = 864\nrender_height = 480")
    cfg = load_config(path)
    assert cfg.render_width == 864
    assert cfg.render_height == 480


@pytest.mark.parametrize("field", ["render_width", "render_height"])
def test_render_dimension_off_the_32_grid_is_rejected(tmp_path: Path, field: str) -> None:
    _create_default_assets(tmp_path)
    other = "render_height" if field == "render_width" else "render_width"
    path = _write_config(tmp_path, extra_toml=f"{field} = 850\n{other} = 480")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert field in str(excinfo.value)
    assert "32" in str(excinfo.value)


@pytest.mark.parametrize("field", ["render_width", "render_height"])
def test_render_dimension_below_the_minimum_is_rejected(tmp_path: Path, field: str) -> None:
    _create_default_assets(tmp_path)
    other = "render_height" if field == "render_width" else "render_width"
    path = _write_config(tmp_path, extra_toml=f"{field} = 0\n{other} = 480")
    with pytest.raises(ConfigError):
        load_config(path)


def test_render_dimensions_must_be_set_together(tmp_path: Path) -> None:
    """Half a resolution is a mistake, not a partial override: injecting a
    width while the template keeps its own height silently changes the aspect
    ratio of every shot."""
    _create_default_assets(tmp_path)
    path = _write_config(tmp_path, extra_toml="render_width = 864")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "render_height" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Misplaced top-level keys
#
# TOML assigns a bare key to whichever table precedes it, so a top-level
# setting written below `[hardware]` silently becomes `hardware.<key>` and the
# run uses the default instead. This is not hypothetical: examples/first-run.toml
# shipped with SIX settings below the table, including the 864x480 that is the
# whole point of the file, so every copy of it rendered at 1344x768 and took
# 3x as long as advertised. Nothing warned.
#
# A [hardware] table has a small, closed set of valid keys, so anything else in
# it is a misplaced top-level setting or a typo. Both deserve a loud failure
# naming the key -- silently ignoring config is exactly the class of bug this
# project's fail-loudly rule exists to prevent.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "misplaced",
    ["render_width = 864", "min_free_vram_gb = 20.0", "instrumental_coverage = true"],
)
def test_top_level_key_below_the_hardware_table_is_rejected(
    tmp_path: Path, misplaced: str
) -> None:
    _create_default_assets(tmp_path)
    key = misplaced.split(" =")[0]
    path = _write_config(
        tmp_path, hardware_block=f"{DEFAULT_HARDWARE_TOML}\n{misplaced}\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    message = str(excinfo.value)
    assert key in message
    assert "hardware" in message


def test_hardware_rejection_message_points_at_the_fix(tmp_path: Path) -> None:
    """The error has to say *what to do*: the key is real, it is just in the
    wrong place, and the fix is to move it above the table."""
    _create_default_assets(tmp_path)
    path = _write_config(
        tmp_path, hardware_block=f"{DEFAULT_HARDWARE_TOML}\nrender_width = 864\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert "above" in str(excinfo.value).lower()


def test_typo_inside_the_hardware_table_is_rejected(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    path = _write_config(
        tmp_path, hardware_block=f"{DEFAULT_HARDWARE_TOML}\nmax_chunk_secons = 8.0\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)

    assert "max_chunk_secons" in str(excinfo.value)


def test_valid_hardware_keys_still_load(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    path = _write_config(
        tmp_path,
        hardware_block=(
            "[hardware]\n"
            'name = "RTX 4090"\n'
            "vram_gb = 24.0\n"
            "min_chunk_seconds = 5.167\n"
            "max_chunk_seconds = 8.0\n"
            'recommended_nodes = ["SageAttention"]\n'
        ),
    )

    cfg = load_config(path)

    assert cfg.hardware.max_chunk_seconds == pytest.approx(8.0)
    assert cfg.hardware.recommended_nodes == ("SageAttention",)


def test_default_vram_floor_matches_the_lowest_proven_working_figure(tmp_path: Path) -> None:
    """The floor is set at the worst configuration actually observed to work
    (~16.4 GB free on doris), never below it on the theory that less might do.

    12.0 was below anything ever demonstrated, and on 2026-08-07 it green-lit
    a run onto a contended card; the load went silent rather than raising CUDA
    OOM and wedged the host. Equally it must not be raised to H3's 19995 MB
    staging figure -- renders demonstrably succeed at 16.4 GB because ComfyUI
    offloads between stages, and a floor of 20 would refuse working runs.
    """
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert 15.0 <= cfg.min_free_vram_gb <= 16.4


# --------------------------------------------------------------------------- #
# Issue #25: vocal-stem conditioning
# --------------------------------------------------------------------------- #


def test_vocal_stem_defaults_to_none(tmp_path: Path) -> None:
    """No stem configured means today's behaviour: condition on the mix."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.vocal_stem is None


def test_vocal_stem_resolves_relative_to_the_config_dir(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    (tmp_path / "audio" / "vocals.wav").write_bytes(b"RIFF-fake-wav-data")
    cfg = load_config(_write_config(tmp_path, extra_toml='vocal_stem = "audio/vocals.wav"'))
    assert cfg.vocal_stem == tmp_path / "audio" / "vocals.wav"


def test_missing_vocal_stem_file_is_refused_at_load(tmp_path: Path) -> None:
    """A stem path that does not exist must fail at config load, not hours
    later when staging reaches for the file mid-custody."""
    _create_default_assets(tmp_path)
    path = _write_config(tmp_path, extra_toml='vocal_stem = "audio/does_not_exist.wav"')
    with pytest.raises(ConfigError):
        load_config(path)


# --------------------------------------------------------------------------- #
# Issue #38: the noise seed is a run-level input, recorded and validated
# --------------------------------------------------------------------------- #


def test_noise_seed_defaults_to_zero(tmp_path: Path) -> None:
    """0 is what both committed templates have always carried -- the default
    renders byte-for-byte what this project has always rendered."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.noise_seed == 0


def test_noise_seed_loads_from_toml(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path, extra_toml="noise_seed = 42"))
    assert cfg.noise_seed == 42


def test_negative_noise_seed_is_refused_at_load(tmp_path: Path) -> None:
    """ComfyUI clamps out-of-range seeds server-side, which would make the
    recorded seed a lie -- refuse before custody is ever taken."""
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml="noise_seed = -1"))


def test_boolean_noise_seed_is_refused_at_load(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml="noise_seed = true"))


def test_config_seed_bound_mirrors_the_mutator_bound() -> None:
    """config.py keeps a copy of MAX_NOISE_SEED rather than importing the
    mutator for a constant; this is the assertion that keeps the copy honest."""
    from music_video_maker import workflow_graph

    assert config_module.MAX_NOISE_SEED == workflow_graph.MAX_NOISE_SEED


# --------------------------------------------------------------------------- #
# Issue #27: instrumental filler chunks get their own length ceiling
# --------------------------------------------------------------------------- #


def test_instrumental_shot_seconds_defaults_to_none(tmp_path: Path) -> None:
    """None means the same ceiling as max_chunk_seconds -- today's behaviour."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.instrumental_shot_seconds is None


def test_instrumental_shot_seconds_loads_from_toml(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path, extra_toml="instrumental_shot_seconds = 12.5"))
    assert cfg.instrumental_shot_seconds == pytest.approx(12.5)


def test_non_positive_instrumental_shot_seconds_is_refused(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml="instrumental_shot_seconds = 0"))


# --------------------------------------------------------------------------- #
# Issue #28: chaining knobs
# --------------------------------------------------------------------------- #


def test_chaining_knobs_default_to_reanchor_off_and_instrumental_scope(tmp_path: Path) -> None:
    """Default scope is 'instrumental': chaining is free on instrumentals and
    costs lip-sync on lyric lines (fl2va has no audio conditioning), so the
    default chains exactly where the trade is free."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.i2v_reanchor_interval is None
    assert cfg.i2v_chain_scope == "instrumental"


def test_chaining_knobs_load_from_toml(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    cfg = load_config(
        _write_config(
            tmp_path, extra_toml='i2v_reanchor_interval = 4\ni2v_chain_scope = "all"\n'
        )
    )
    assert cfg.i2v_reanchor_interval == 4
    assert cfg.i2v_chain_scope == "all"


def test_reanchor_interval_below_one_is_refused(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml="i2v_reanchor_interval = 0"))


def test_unknown_chain_scope_is_refused(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml='i2v_chain_scope = "everything"'))


# --------------------------------------------------------------------------- #
# Issue #49: seed-face recognition similarity floor
# --------------------------------------------------------------------------- #


def test_seed_face_similarity_defaults_to_none(tmp_path: Path) -> None:
    """Recognition is OFF by default -- a config that never sets this key
    must behave exactly as issue #47 shipped it, detection-only. Defaulting
    to the calibrated DEFAULT_MIN_FACE_SIMILARITY would silently make every
    pre-existing config with chaining enabled start refusing chain
    boundaries the moment it hit a checkout with no SFace model."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.i2v_min_seed_face_similarity is None


def test_seed_face_similarity_loads_from_toml_when_the_model_is_present(
    tmp_path: Path, monkeypatch
) -> None:
    fake_model = tmp_path / "fake_sface.onnx"
    fake_model.write_bytes(b"fake")
    monkeypatch.setattr(
        config_module, "resolve_recognition_model_path", lambda *a, **k: fake_model
    )
    _create_default_assets(tmp_path)

    cfg = load_config(
        _write_config(
            tmp_path, extra_toml=f"i2v_min_seed_face_similarity = {DEFAULT_MIN_FACE_SIMILARITY}"
        )
    )

    assert cfg.i2v_min_seed_face_similarity == DEFAULT_MIN_FACE_SIMILARITY


@pytest.mark.parametrize("value", [1.5, -1.5, "high"])
def test_seed_face_similarity_out_of_range_or_non_numeric_is_refused(
    tmp_path: Path, value
) -> None:
    """A cosine similarity lives in [-1.0, 1.0] -- unlike the face-AREA
    fraction, negative values are meaningful (a confidently wrong match), so
    the floor is -1.0, not 0.0."""
    _create_default_assets(tmp_path)
    literal = f'"{value}"' if isinstance(value, str) else value
    with pytest.raises(ConfigError, match="i2v_min_seed_face_similarity"):
        load_config(
            _write_config(tmp_path, extra_toml=f"i2v_min_seed_face_similarity = {literal}")
        )


def test_seed_face_similarity_boundary_values_are_accepted(
    tmp_path: Path, monkeypatch
) -> None:
    fake_model = tmp_path / "fake_sface.onnx"
    fake_model.write_bytes(b"fake")
    monkeypatch.setattr(
        config_module, "resolve_recognition_model_path", lambda *a, **k: fake_model
    )
    _create_default_assets(tmp_path)

    assert load_config(
        _write_config(tmp_path, extra_toml="i2v_min_seed_face_similarity = 1.0")
    ).i2v_min_seed_face_similarity == 1.0
    assert load_config(
        _write_config(tmp_path, extra_toml="i2v_min_seed_face_similarity = -1.0")
    ).i2v_min_seed_face_similarity == -1.0


def test_seed_face_similarity_set_but_recognition_model_missing_refuses_at_load(
    tmp_path: Path,
) -> None:
    """Recognition was explicitly requested but the 38.7 MB SFace model is --
    correctly -- absent from this checkout (issue #51: it is deliberately not
    committed). This must refuse loudly at config-load time, before any GPU
    time is spent, rather than silently degrade to a weaker detection-only
    gate at render time while reporting success."""
    _create_default_assets(tmp_path)
    assert not resolve_recognition_model_path().exists(), (
        "this test assumes the SFace model is not committed to the repository"
    )

    with pytest.raises(ConfigError, match="SFace") as excinfo:
        load_config(
            _write_config(tmp_path, extra_toml="i2v_min_seed_face_similarity = 0.34")
        )

    message = str(excinfo.value)
    assert "not committed" in message.lower()
    assert "docs/seed-face-recognition.md" in message or "opencv_zoo" in message


def test_seed_face_similarity_missing_model_check_is_independent_of_i2v_continuity(
    tmp_path: Path,
) -> None:
    """The pre-flight check fires purely on the knob being set -- the same
    'explicitly requested but cannot be honoured' logic applies whether or
    not i2v_continuity is also on, since a config is often prepared ahead of
    flipping that switch."""
    _create_default_assets(tmp_path)
    assert not resolve_recognition_model_path().exists()

    with pytest.raises(ConfigError, match="SFace"):
        load_config(
            _write_config(
                tmp_path,
                extra_toml="i2v_continuity = false\ni2v_min_seed_face_similarity = 0.34",
            )
        )


def test_unset_seed_face_similarity_never_touches_the_recognition_model_check(
    tmp_path: Path,
) -> None:
    """A config that never sets this key must load fine regardless of
    whether the (uncommitted) SFace model happens to be on disk -- the whole
    point of defaulting to None."""
    _create_default_assets(tmp_path)
    assert not resolve_recognition_model_path().exists()

    cfg = load_config(_write_config(tmp_path, extra_toml="i2v_continuity = false"))

    assert cfg.i2v_min_seed_face_similarity is None


# --------------------------------------------------------------------------- #
# Issue #42: authored alignment overrides
# --------------------------------------------------------------------------- #


def test_alignment_overrides_default_to_empty(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.alignment_overrides == ()


def test_alignment_override_parses_with_reason(tmp_path: Path) -> None:
    from music_video_maker.contracts import AlignmentOverride

    _create_default_assets(tmp_path)
    cfg = load_config(
        _write_config(
            tmp_path,
            extra_toml=(
                "[[alignment_override]]\n"
                "segment_index = 6\n"
                "start = 71.41\n"
                "end = 76.55\n"
                'reason = "base model piles the phrase at 65s; stem energy disagrees"\n'
            ),
        )
    )
    assert cfg.alignment_overrides == (
        AlignmentOverride(
            segment_index=6, start=71.41, end=76.55,
            reason="base model piles the phrase at 65s; stem energy disagrees",
        ),
    )


def test_alignment_override_without_a_reason_is_refused(tmp_path: Path) -> None:
    """An unexplained pin is indistinguishable from a stale one after the
    next lyrics edit -- provenance is mandatory."""
    _create_default_assets(tmp_path)
    path = _write_config(
        tmp_path,
        extra_toml=(
            "[[alignment_override]]\nsegment_index = 6\nstart = 71.41\nend = 76.55\n"
        ),
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_alignment_override_with_inverted_span_is_refused_at_load(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    path = _write_config(
        tmp_path,
        extra_toml=(
            "[[alignment_override]]\nsegment_index = 6\nstart = 76.0\nend = 71.0\n"
            'reason = "backwards"\n'
        ),
    )
    with pytest.raises(ConfigError):
        load_config(path)


# --------------------------------------------------------------------------- #
# Issue #39: the text encoder is a run-level choice, not a template constant
# --------------------------------------------------------------------------- #


def test_text_encoder_defaults_to_none(tmp_path: Path) -> None:
    """Unset means "whatever the authored template names" -- today's
    behaviour, unchanged."""
    _create_default_assets(tmp_path)
    cfg = load_config(_write_config(tmp_path))
    assert cfg.text_encoder is None


def test_text_encoder_loads_as_a_server_side_name_not_a_local_path(tmp_path: Path) -> None:
    """``clip_name`` is resolved by ComfyUI inside its own
    ``models/text_encoders/`` on doris. Resolving it against the config
    file's directory (as every real *path* field is) would produce a Mac
    path the render host cannot see."""
    _create_default_assets(tmp_path)
    cfg = load_config(
        _write_config(
            tmp_path,
            extra_toml='text_encoder = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"',
        )
    )
    assert cfg.text_encoder == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def test_blank_text_encoder_is_refused_at_load(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml='text_encoder = "   "'))


def test_non_string_text_encoder_is_refused_at_load(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, extra_toml="text_encoder = 4"))


def test_absolute_text_encoder_path_is_refused_at_load(tmp_path: Path) -> None:
    """An absolute path here is the local-path misreading: ComfyUI would
    look for it *under* its text_encoders directory and fail at load time,
    hours into a run with custody of the card already taken."""
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError):
        load_config(
            _write_config(
                tmp_path,
                extra_toml=(
                    'text_encoder = '
                    '"/home/derek/ComfyUI/models/text_encoders/x.safetensors"'
                ),
            )
        )


# --------------------------------------------------------------------------- #
# LoRA (issue #62)
# --------------------------------------------------------------------------- #


def _load_with(tmp_path, **kv):
    extra = "\n".join(f"{k} = {v}" for k, v in kv.items())
    _create_default_assets(tmp_path)
    return load_config(_write_config(tmp_path, extra_toml=extra))


def test_an_absolute_lora_path_is_refused(tmp_path):
    """Same rule and reason as text_encoder: ComfyUI resolves lora_name inside
    its own models/loras/, so an absolute path fails at load time on the render
    host -- after custody of the GPU has been taken."""
    with pytest.raises(ConfigError, match="absolute"):
        _load_with(tmp_path, lora='"/home/derek/realism.safetensors"')


def test_a_trigger_word_without_a_lora_is_refused(tmp_path):
    """Free to catch here; costs a whole render to notice."""
    with pytest.raises(ConfigError, match="trigger"):
        _load_with(tmp_path, lora_trigger='"r34l1sm"')


def test_lora_defaults_to_absent_at_full_strength(tmp_path):
    config = _load_with(tmp_path)
    assert config.lora is None
    assert config.lora_strength == 1.0
    assert config.lora_trigger is None


def test_a_top_level_key_misplaced_under_alignment_override_is_refused(tmp_path):
    """Issue #63, the same failure `HARDWARE_KEYS` exists to catch, in the
    other table this config format has.

    TOML binds a bare key to whichever table precedes it. `[[alignment_override]]`
    tables sit at the *end* of every real run config, so anything appended to
    the file lands inside the last one and is silently ignored. Hit for real
    while building issue #62's A/B: `lora` was appended below them, both arms
    loaded identically, and only a hand-check before the render caught it.
    """
    _create_default_assets(tmp_path)
    with pytest.raises(ConfigError, match="lora"):
        load_config(
            _write_config(
                tmp_path,
                extra_toml=(
                    "[[alignment_override]]\n"
                    "segment_index = 6\nstart = 71.4\nend = 76.5\n"
                    'reason = "measured"\n'
                    'lora = "realism.safetensors"\n'
                ),
            )
        )


def test_a_well_formed_alignment_override_still_loads(tmp_path):
    _create_default_assets(tmp_path)
    config = load_config(
        _write_config(
            tmp_path,
            extra_toml=(
                "[[alignment_override]]\n"
                "segment_index = 6\nstart = 71.4\nend = 76.5\n"
                'reason = "measured against stem energy"\n'
            ),
        )
    )
    assert config.alignment_overrides[0].segment_index == 6


# --------------------------------------------------------------------------- #
# face_treatment (issue #62): one switch per song, because the choice is
# per-song. Derek, deciding it: "for this song I want flattery, because I know
# that's what Dianne will want. For Storms, where the lead character is me, I
# want realistic."
# --------------------------------------------------------------------------- #


def test_face_treatment_defaults_to_flattering_with_no_adapter(tmp_path):
    """The default is what every render to date did, and what final_v10 --
    the current keeper -- was rendered with."""
    config = _load_with(tmp_path)
    assert config.face_treatment == "flattering"
    assert config.lora is None


def test_realistic_selects_the_realism_adapter_and_its_trigger(tmp_path):
    """One flag has to be one flag: if it still needed three `lora` lines
    beside it, it would not be the switch that was asked for."""
    config = _load_with(tmp_path, face_treatment='"realistic"')
    assert config.lora == REALISM_LORA
    assert config.lora_strength == REALISM_LORA_STRENGTH
    assert config.lora_trigger == REALISM_LORA_TRIGGER


def test_an_explicit_lora_overrides_the_preset(tmp_path):
    """The preset is a default, never a lock -- a future adapter must not
    need a code change to try."""
    config = _load_with(
        tmp_path,
        face_treatment='"realistic"',
        lora='"some-other.safetensors"',
        lora_strength="0.5",
    )
    assert config.lora == "some-other.safetensors"
    assert config.lora_strength == 0.5


def test_an_unknown_face_treatment_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="face_treatment"):
        _load_with(tmp_path, face_treatment='"photoreal"')


def test_realistic_with_flattering_appearance_warns(tmp_path, caplog):
    """The measured conflict, stated once where it is decided rather than
    guessed at from adjectives. A realism adapter and a de-aging appearance
    clause pull opposite ways and the stronger signal wins silently."""
    with caplog.at_level(logging.WARNING):
        _load_with(
            tmp_path,
            face_treatment='"realistic"',
            global_appearance='"everyone slim, trim and flatteringly lit"',
        )
    assert "pull in opposite directions" in caplog.text


def test_realistic_with_no_appearance_direction_is_quiet(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _load_with(tmp_path, face_treatment='"realistic"')
    assert "pull in opposite directions" not in caplog.text


# --------------------------------------------------------------------------- #
# Alignment model size. Stage 1's whisper model was hardcoded to "base": the
# parameter existed on align() but nothing ever passed it, so a run could not
# change it without editing the package. On "Deathless" (8:32) base placed
# 77.2s of vocal against small's 187.2s and believed the singing stopped at
# 3:11 of an 8:32 track -- 56 of 71 chunks would have rendered as "performs
# silently" while the master audio has her singing, with no error raised.
# --------------------------------------------------------------------------- #


def test_alignment_model_size_defaults_to_base(tmp_path):
    """Every config written before this key existed must align exactly as it
    always did -- the default is the old hardcoded value, not the better one."""
    config = _load_with(tmp_path)
    assert config.alignment_model_size == "base"


def test_alignment_model_size_can_be_set(tmp_path):
    config = _load_with(tmp_path, alignment_model_size='"small"')
    assert config.alignment_model_size == "small"


def test_an_unknown_alignment_model_size_is_refused(tmp_path):
    """A typo here costs the whole of Stage 1 and is only visible as a stack
    trace from inside stable-ts, after the model download."""
    with pytest.raises(ConfigError, match="alignment_model_size"):
        _load_with(tmp_path, alignment_model_size='"smal"')


def test_config_and_alignment_agree_on_the_default_model():
    """config mirrors the constant rather than importing the alignment module
    (the MAX_NOISE_SEED rule). Mirrors drift; this is what stops it."""
    from music_video_maker.alignment import DEFAULT_MODEL_SIZE

    assert DEFAULT_ALIGNMENT_MODEL_SIZE == DEFAULT_MODEL_SIZE
    assert DEFAULT_ALIGNMENT_MODEL_SIZE in ALIGNMENT_MODEL_SIZES



def test_instrumental_audio_gain_db_defaults_to_none(tmp_path: Path) -> None:
    """Every config written before F26 renders instrumental chunks at the
    master's own level, and this knob must not change that by existing."""
    _create_default_assets(tmp_path)

    cfg = load_config(_write_config(tmp_path))

    assert cfg.instrumental_audio_gain_db is None


def test_instrumental_audio_gain_db_accepts_attenuation(tmp_path: Path) -> None:
    _create_default_assets(tmp_path)

    cfg = load_config(
        _write_config(tmp_path, extra_toml="instrumental_audio_gain_db = -60.0\n")
    )

    assert cfg.instrumental_audio_gain_db == -60.0


def test_instrumental_audio_gain_db_refuses_a_boost(tmp_path: Path) -> None:
    """Attenuation only. A positive value makes the music H3 is lip-syncing
    *louder*, which is the F26 defect amplified rather than fixed."""
    _create_default_assets(tmp_path)

    with pytest.raises(ConfigError, match="instrumental_audio_gain_db"):
        load_config(_write_config(tmp_path, extra_toml="instrumental_audio_gain_db = 6.0\n"))
