"""Tests for the cinematography profile module (issue #55).

All tests run fully offline against ``tmp_path`` -- no network, no GPU. This
file, unlike ``profiles.py`` itself (D7: it imports nothing from
``music_video_maker``), may import anything, including
``music_video_maker.config`` and ``music_video_maker.contracts``, to check
the config-integration seam and the D4 fingerprint-evidence contract.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the project venv is 3.11
    import tomli as tomllib

from music_video_maker import config as config_module
from music_video_maker.config import ConfigError, load_config
from music_video_maker.contracts import ChunkFingerprint
from music_video_maker.profiles import (
    FINGERPRINT_EVIDENCE,
    LOOK_FIELDS,
    PROVENANCE_KEYS,
    TOP_LEVEL_KEYS,
    Profile,
    ProfileError,
    load_profile,
    main,
    promote_photography,
    render_profile_toml,
    resolve_look,
    write_profile_record,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "profiles"


# --------------------------------------------------------------------------- #
# D4: the closed field set must have fingerprint evidence
# --------------------------------------------------------------------------- #


def test_every_look_field_has_fingerprint_evidence():
    """D4's actual constraint: a look field admitted to LOOK_FIELDS with no
    fingerprint evidence would let a profile edit change the pixels while
    --resume reuses the old chunks and reports a clean match. Every entry
    must be either "prompt_hash" or a name ChunkFingerprint itself already
    tracks in one of its three comparison tiers."""
    known_fingerprint_fields = (
        ChunkFingerprint.CONDITIONING_FIELDS
        + ChunkFingerprint.CONTENT_FIELDS
        + ChunkFingerprint.TIMELINE_FIELDS
    )
    for field in LOOK_FIELDS:
        assert field in FINGERPRINT_EVIDENCE, f"{field} has no FINGERPRINT_EVIDENCE entry"
        evidence = FINGERPRINT_EVIDENCE[field]
        assert evidence == "prompt_hash" or evidence in known_fingerprint_fields, (
            f"{field} -> {evidence!r} is neither 'prompt_hash' nor a real "
            f"ChunkFingerprint field ({known_fingerprint_fields})"
        )


# --------------------------------------------------------------------------- #
# load_profile: happy paths
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, text: str, name: str = "profile.toml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_load_profile_happy_path_all_fields(tmp_path: Path):
    path = _write(
        tmp_path,
        """
version     = 2
name        = "refestramus-house"
description = "The locked look."

cinematography = "35mm film, shallow depth of field, warm natural light"
face_treatment = "realistic"
lora           = "h3-realism-people-t2v-i2v-r2v.safetensors"
lora_strength  = 0.8
lora_trigger   = "r34l1sm"

[provenance]
source        = "promoted"
song          = "Deathless"
promoted_from = ".authoring/photography.json"
model         = "claude-opus-5"
promoted_at   = "2026-08-16"
concept_hash  = "7fe8282199d1692a"
stance_index  = 1
notes         = "chosen over stances 0 and 2"
""",
    )
    profile = load_profile(path)
    assert profile.version == 2
    assert profile.name == "refestramus-house"
    assert profile.description == "The locked look."
    assert profile.values == {
        "cinematography": "35mm film, shallow depth of field, warm natural light",
        "face_treatment": "realistic",
        "lora": "h3-realism-people-t2v-i2v-r2v.safetensors",
        "lora_strength": 0.8,
        "lora_trigger": "r34l1sm",
    }
    assert list(profile.values) == [
        f for f in LOOK_FIELDS if f in profile.values
    ], "values must be in LOOK_FIELDS order"
    assert profile.provenance["source"] == "promoted"
    assert profile.provenance["stance_index"] == 1
    assert profile.path == path.resolve()


def test_load_profile_minimal_version_name_one_field(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\ncinematography = "grainy"\n')
    profile = load_profile(path)
    assert profile.values == {"cinematography": "grainy"}
    assert profile.description is None
    assert profile.provenance == {}


def test_load_profile_sha256_matches_file_bytes(tmp_path: Path):
    import hashlib

    path = _write(tmp_path, 'version = 1\nname = "x"\ncinematography = "grainy"\n')
    profile = load_profile(path)
    assert profile.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# load_profile: validation failures
# --------------------------------------------------------------------------- #


def test_load_profile_missing_version(tmp_path: Path):
    path = _write(tmp_path, 'name = "x"\ncinematography = "grainy"\n')
    with pytest.raises(ProfileError, match="version"):
        load_profile(path)


def test_load_profile_version_zero(tmp_path: Path):
    path = _write(tmp_path, 'version = 0\nname = "x"\ncinematography = "grainy"\n')
    with pytest.raises(ProfileError, match="version"):
        load_profile(path)


def test_load_profile_version_wrong_type(tmp_path: Path):
    path = _write(tmp_path, 'version = "1"\nname = "x"\ncinematography = "grainy"\n')
    with pytest.raises(ProfileError, match="version"):
        load_profile(path)


def test_load_profile_missing_name(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\ncinematography = "grainy"\n')
    with pytest.raises(ProfileError, match="name"):
        load_profile(path)


def test_load_profile_blank_name(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "   "\ncinematography = "grainy"\n')
    with pytest.raises(ProfileError, match="name"):
        load_profile(path)


def test_load_profile_no_look_field(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\n')
    with pytest.raises(ProfileError, match="none of"):
        load_profile(path)


def test_load_profile_unknown_top_level_key(tmp_path: Path):
    path = _write(
        tmp_path, 'version = 1\nname = "x"\ncinematography = "grainy"\ncamera = "wide"\n'
    )
    with pytest.raises(ProfileError, match="unknown top-level"):
        load_profile(path)


def test_load_profile_unknown_provenance_key(tmp_path: Path):
    path = _write(
        tmp_path,
        'version = 1\nname = "x"\ncinematography = "grainy"\n\n[provenance]\nbogus = "x"\n',
    )
    with pytest.raises(ProfileError, match="unknown key"):
        load_profile(path)


def test_load_profile_look_field_below_provenance_names_bare_key_rule(tmp_path: Path):
    """D8: TOML binds a bare key to whichever table precedes it, so a look
    field appended below [provenance] silently becomes provenance.<key>
    instead of a top-level field. The error must name this explicitly."""
    path = _write(
        tmp_path,
        """
version = 1
name = "x"

[provenance]
source = "authored"
cinematography = "grainy"
""",
    )
    with pytest.raises(ProfileError) as excinfo:
        load_profile(path)
    message = str(excinfo.value)
    assert "cinematography" in message
    assert "top-level look field" in message
    assert "[provenance]" in message
    assert "below that table" in message


def test_load_profile_nonexistent_path(tmp_path: Path):
    with pytest.raises(ProfileError, match="cannot read"):
        load_profile(tmp_path / "does-not-exist.toml")


def test_load_profile_malformed_toml(tmp_path: Path):
    path = _write(tmp_path, "this is not [ valid toml\n")
    with pytest.raises(ProfileError, match="malformed TOML"):
        load_profile(path)


def test_load_profile_lora_strength_wrong_type(tmp_path: Path):
    path = _write(
        tmp_path,
        'version = 1\nname = "x"\ncinematography = "grainy"\nlora_strength = "0.8"\n',
    )
    with pytest.raises(ProfileError, match="lora_strength"):
        load_profile(path)


def test_load_profile_blank_look_field_is_rejected(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\ncinematography = "   "\n')
    with pytest.raises(ProfileError, match="blank"):
        load_profile(path)


def test_top_level_and_provenance_key_sets_are_disjoint_by_construction():
    # Sanity: LOOK_FIELDS is a subset of TOP_LEVEL_KEYS (used by the module).
    assert set(LOOK_FIELDS) <= TOP_LEVEL_KEYS
    assert "version" in TOP_LEVEL_KEYS
    assert "source" in PROVENANCE_KEYS


# --------------------------------------------------------------------------- #
# resolve_look
# --------------------------------------------------------------------------- #


def _profile(tmp_path: Path, **look_values: object) -> Profile:
    lines = ['version = 1', 'name = "test-profile"']
    for key, value in look_values.items():
        if isinstance(value, str):
            lines.append(f'{key} = {json.dumps(value)}')
        else:
            lines.append(f"{key} = {value}")
    path = _write(tmp_path, "\n".join(lines) + "\n", name="p.toml")
    return load_profile(path)


def test_resolve_look_profile_only(tmp_path: Path):
    profile = _profile(tmp_path, cinematography="A", face_treatment="realistic")
    effective, overridden = resolve_look(profile, {})
    assert effective == {"cinematography": "A", "face_treatment": "realistic"}
    assert overridden == ()


def test_resolve_look_no_profile_returns_empty():
    effective, overridden = resolve_look(None, {"cinematography": "run-value"})
    assert effective == {}
    assert overridden == ()


def test_resolve_look_run_config_wins_and_is_reported(tmp_path: Path):
    profile = _profile(tmp_path, cinematography="from-profile")
    effective, overridden = resolve_look(profile, {"cinematography": "from-run-config"})
    assert effective["cinematography"] == "from-run-config"
    assert overridden == ("cinematography",)


def test_resolve_look_explicit_none_counts_as_unset(tmp_path: Path):
    profile = _profile(tmp_path, cinematography="from-profile")
    effective, overridden = resolve_look(profile, {"cinematography": None})
    assert effective["cinematography"] == "from-profile"
    assert overridden == ()


def test_resolve_look_run_field_profile_does_not_supply_is_not_overridden(tmp_path: Path):
    profile = _profile(tmp_path, cinematography="from-profile")
    effective, overridden = resolve_look(profile, {"lora_trigger": "r34l1sm"})
    assert effective["cinematography"] == "from-profile"
    assert effective["lora_trigger"] == "r34l1sm"
    assert overridden == ()


# --------------------------------------------------------------------------- #
# write_profile_record
# --------------------------------------------------------------------------- #


def test_write_profile_record_round_trips_and_creates_parents(tmp_path: Path):
    profile = _profile(tmp_path, cinematography="A")
    dest = tmp_path / "nested" / "record.json"
    result = write_profile_record(profile, {"cinematography": "A"}, ("cinematography",), dest)
    assert result == dest
    assert dest.exists()
    payload = json.loads(dest.read_text())
    assert payload["profile"]["name"] == "test-profile"
    assert payload["profile"]["values"] == {"cinematography": "A"}
    assert payload["effective"] == {"cinematography": "A"}
    assert payload["overridden_by_run_config"] == ["cinematography"]
    assert payload["profile_sha256"] == profile.sha256
    assert "recorded_at" in payload


# --------------------------------------------------------------------------- #
# render_profile_toml
# --------------------------------------------------------------------------- #


def test_render_profile_toml_round_trips_through_load(tmp_path: Path):
    text = render_profile_toml(
        version=3,
        name="rendered",
        description="a description",
        values={
            "cinematography": "35mm",
            "face_treatment": "realistic",
            "lora": "adapter.safetensors",
            "lora_strength": 0.8,
            "lora_trigger": "r34l1sm",
        },
        provenance={"source": "authored", "notes": "hi"},
    )
    path = _write(tmp_path, text, name="rendered.toml")
    profile = load_profile(path)
    assert profile.version == 3
    assert profile.name == "rendered"
    assert profile.description == "a description"
    assert profile.values["cinematography"] == "35mm"
    assert profile.values["lora_strength"] == 0.8
    assert profile.provenance == {"source": "authored", "notes": "hi"}


def test_render_profile_toml_escapes_quotes_backslashes_and_keeps_unicode(tmp_path: Path):
    tricky = 'He said "hi" \\ and kept walking — forever'
    text = render_profile_toml(version=1, name="x", values={"cinematography": tricky})
    path = _write(tmp_path, text, name="tricky.toml")
    profile = load_profile(path)
    assert profile.values["cinematography"] == tricky
    assert "—" in text  # em-dash left literal, not \u escaped


def test_render_profile_toml_via_tomllib_directly(tmp_path: Path):
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    text = render_profile_toml(version=1, name="x", values={"cinematography": "35mm"})
    parsed = tomllib.loads(text)
    assert parsed == {"version": 1, "name": "x", "cinematography": "35mm"}


def test_render_profile_toml_rejects_unknown_provenance_key():
    with pytest.raises(ProfileError, match="unknown key"):
        render_profile_toml(
            version=1, name="x", values={"cinematography": "35mm"}, provenance={"bogus": 1}
        )


# --------------------------------------------------------------------------- #
# promote_photography
# --------------------------------------------------------------------------- #


def _build_run_dir(
    tmp_path: Path,
    *,
    cinematography: str | None = "promoted look",
    with_session: bool = True,
) -> Path:
    run_dir = tmp_path / "run"
    authoring_dir = run_dir / ".authoring"
    authoring_dir.mkdir(parents=True)
    (authoring_dir / "photography.json").write_text(
        json.dumps({"cinematography": cinematography, "camera": {}})
    )
    if with_session:
        (authoring_dir / "session.json").write_text(
            json.dumps(
                {
                    "stages": {
                        "photography": {
                            "model": "claude-opus-5",
                            "completed_at": "2026-08-16T00:00:00+00:00",
                            "cost_usd": 0.42,
                            "input_hashes": {"concept": "7fe8282199d1692a"},
                        }
                    }
                }
            )
        )
    return run_dir


def test_promote_photography_happy_path(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    out_path = tmp_path / "out" / "profile.toml"
    result = promote_photography(
        run_dir, name="my-house", out_path=out_path, description="d"
    )
    assert result == out_path
    profile = load_profile(out_path)
    assert profile.name == "my-house"
    assert profile.values["cinematography"] == "promoted look"
    assert profile.provenance["source"] == "promoted"
    assert profile.provenance["model"] == "claude-opus-5"
    # Two dates, deliberately not one: `generated_at` is when the photography
    # stage produced the look, `promoted_at` is when a human kept it. The gap
    # between them is the review, so collapsing them loses which end you are
    # reading.
    assert profile.provenance["generated_at"] == "2026-08-16T00:00:00+00:00"
    assert profile.provenance["promoted_at"] == datetime.now(timezone.utc).date().isoformat()
    assert profile.provenance["concept_hash"] == "7fe8282199d1692a"
    assert profile.provenance["promoted_from"].endswith("photography.json")


def test_promote_photography_blank_cinematography_names_the_file(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path, cinematography=None)
    with pytest.raises(ProfileError) as excinfo:
        promote_photography(run_dir, name="x", out_path=tmp_path / "out.toml")
    message = str(excinfo.value)
    assert "photography.json" in message
    assert "config" in message.lower() or "run config" in message.lower()


def test_promote_photography_missing_session_warns_and_still_writes(tmp_path: Path, caplog):
    run_dir = _build_run_dir(tmp_path, with_session=False)
    out_path = tmp_path / "out.toml"
    with caplog.at_level("WARNING"):
        promote_photography(run_dir, name="x", out_path=out_path)
    assert out_path.exists()
    assert any("session.json" in record.message for record in caplog.records)
    profile = load_profile(out_path)
    assert "model" not in profile.provenance


def test_promote_photography_refuses_to_clobber(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    out_path = tmp_path / "out.toml"
    promote_photography(run_dir, name="x", out_path=out_path)
    with pytest.raises(ProfileError, match="already exists"):
        promote_photography(run_dir, name="x", out_path=out_path)


def test_promote_photography_overwrite_true_clobbers(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path, cinematography="first")
    out_path = tmp_path / "out.toml"
    promote_photography(run_dir, name="x", out_path=out_path)

    run_dir2 = _build_run_dir(tmp_path / "second", cinematography="second")
    promote_photography(run_dir2, name="x", out_path=out_path, overwrite=True)
    profile = load_profile(out_path)
    assert profile.values["cinematography"] == "second"


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def test_main_promote_returns_zero_and_writes(tmp_path: Path, capsys):
    run_dir = _build_run_dir(tmp_path)
    out_path = tmp_path / "cli-out.toml"
    exit_code = main(
        ["promote", "--run-dir", str(run_dir), "--name", "cli-house", "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.exists()
    captured = capsys.readouterr()
    assert captured.out == ""  # T20: never print()


def test_main_promote_returns_two_on_profile_error(tmp_path: Path, capsys):
    run_dir = _build_run_dir(tmp_path, cinematography=None)
    out_path = tmp_path / "cli-out.toml"
    exit_code = main(
        ["promote", "--run-dir", str(run_dir), "--name", "x", "--out", str(out_path)]
    )
    assert exit_code == 2
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert captured.out == ""


# --------------------------------------------------------------------------- #
# examples/profiles/refestramus-house-v1.toml
# --------------------------------------------------------------------------- #


def test_shipped_example_profile_loads():
    example_path = EXAMPLES_DIR / "refestramus-house-v1.toml"
    assert example_path.exists(), "examples/profiles/refestramus-house-v1.toml is missing"
    profile = load_profile(example_path)
    assert profile.name == "refestramus-house"
    assert profile.version == 1
    assert "cinematography" in profile.values


# --------------------------------------------------------------------------- #
# config.py integration
# --------------------------------------------------------------------------- #

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


def _write_run_config(
    tmp_path: Path,
    *,
    extra_toml: str = "",
    filename: str = "run.toml",
) -> Path:
    """Minimal helper local to this file (per the coding brief, not imported
    from tests/test_config.py) mirroring that file's own fixture style."""
    _create_default_assets(tmp_path)
    cast_block = DEFAULT_CAST_TOML.format(cast_dir=tmp_path / "cast")
    content = f"""
master_audio = "{tmp_path / "audio" / "master.wav"}"
lyrics_file = "{tmp_path / "lyrics.txt"}"
global_style = "Refestramus progressive rock music video, 35mm film"
narrative_concept = "Wandering through a surgery"
default_lead_vocalist = "Dianne"
comfyui_url = "http://doris:8188"
workflow_template = "{tmp_path / "workflow_api.json"}"
chunks_dir = "{tmp_path / "output" / "chunks"}"
final_video_dir = "{tmp_path / "output" / "final"}"
{extra_toml}

{cast_block}

{DEFAULT_HARDWARE_TOML}
"""
    path = tmp_path / filename
    path.write_text(content)
    return path


def _write_profile_file(tmp_path: Path, name: str = "profile.toml", **look_values) -> Path:
    lines = ['version = 1', 'name = "profile-under-test"']
    for key, value in look_values.items():
        if isinstance(value, str):
            lines.append(f"{key} = {json.dumps(value)}")
        else:
            lines.append(f"{key} = {value}")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def test_config_inherits_look_fields_from_profile(tmp_path: Path, caplog):
    _write_profile_file(
        tmp_path,
        cinematography="profile-look",
        face_treatment="realistic",
        lora="adapter.safetensors",
        lora_strength=0.8,
        lora_trigger="r34l1sm",
    )
    config_path = _write_run_config(
        tmp_path, extra_toml='cinematography_profile = "profile.toml"'
    )
    with caplog.at_level("INFO"):
        config = load_config(config_path)
    assert config.cinematography == "profile-look"
    assert config.face_treatment == "realistic"
    assert config.lora == "adapter.safetensors"
    assert config.lora_strength == 0.8
    assert config.lora_trigger == "r34l1sm"
    assert isinstance(config.cinematography_profile, Profile)
    assert config.cinematography_profile.name == "profile-under-test"
    assert any("cinematography_profile" in record.message for record in caplog.records)


def test_config_own_cinematography_wins_over_profile(tmp_path: Path):
    _write_profile_file(tmp_path, cinematography="profile-look")
    config_path = _write_run_config(
        tmp_path,
        extra_toml=(
            'cinematography_profile = "profile.toml"\n'
            'cinematography = "run-config-look"'
        ),
    )
    config = load_config(config_path)
    assert config.cinematography == "run-config-look"
    assert config.cinematography_profile is not None
    assert config.cinematography_profile.values["cinematography"] == "profile-look"


def test_config_nonexistent_profile_path_raises_config_error(tmp_path: Path):
    config_path = _write_run_config(
        tmp_path, extra_toml='cinematography_profile = "does-not-exist.toml"'
    )
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_config_profile_bogus_face_treatment_caught_by_configs_own_validation(tmp_path: Path):
    """D6: profiles.py never validates domain (face_treatment's closed set is
    config.py's alone), so a bogus value must surface as config's own
    face_treatment ConfigError, not a ProfileError."""
    _write_profile_file(tmp_path, cinematography="x", face_treatment="cartoonish")
    config_path = _write_run_config(
        tmp_path, extra_toml='cinematography_profile = "profile.toml"'
    )
    with pytest.raises(ConfigError, match="face_treatment"):
        load_config(config_path)


def test_config_no_profile_key_leaves_cinematography_profile_none(tmp_path: Path):
    config_path = _write_run_config(tmp_path)
    config = load_config(config_path)
    assert config.cinematography_profile is None


def test_config_module_does_not_reference_profiles_in_a_cycle():
    """Sanity check on D7: config.py imports profiles at module load without
    error (already implied by every test above importing config_module), and
    profiles.py itself must not import config back."""
    import ast

    profiles_path = Path(config_module.__file__).parent / "profiles.py"
    tree = ast.parse(profiles_path.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)
    violations = [name for name in top_level_imports if name.startswith("music_video_maker")]
    assert not violations, f"profiles.py imports music_video_maker at module scope: {violations}"


# --------------------------------------------------------------------------- #
# Remaining validation branches
#
# Every raise below is a *shape* check (D6) -- the kind that fires on a file
# somebody typed by hand at 1am, which is exactly when the message has to say
# what is wrong rather than surfacing a TypeError three frames deeper.
# --------------------------------------------------------------------------- #


def test_load_profile_look_field_wrong_type(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\nlora = 5\n')
    with pytest.raises(ProfileError, match="lora must be a string"):
        load_profile(path)


def test_load_profile_provenance_not_a_table(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\ncinematography = "g"\nprovenance = "nope"\n')
    with pytest.raises(ProfileError, match=r"\[provenance\] must be a table"):
        load_profile(path)


def test_load_profile_description_wrong_type(tmp_path: Path):
    path = _write(tmp_path, 'version = 1\nname = "x"\ndescription = 5\ncinematography = "g"\n')
    with pytest.raises(ProfileError, match="description must be a string"):
        load_profile(path)


def test_load_profile_not_utf8(tmp_path: Path):
    path = tmp_path / "profile.toml"
    path.write_bytes(b'version = 1\nname = "x"\ncinematography = "\xff\xfe"\n')
    with pytest.raises(ProfileError, match="not valid UTF-8"):
        load_profile(path)


def test_render_profile_toml_rejects_bad_version():
    with pytest.raises(ProfileError, match="version"):
        render_profile_toml(version=0, name="x", values={"cinematography": "g"})


def test_render_profile_toml_rejects_blank_name():
    with pytest.raises(ProfileError, match="name"):
        render_profile_toml(version=1, name="  ", values={"cinematography": "g"})


def test_render_profile_toml_rejects_non_numeric_strength():
    with pytest.raises(ProfileError, match="lora_strength must be a number"):
        render_profile_toml(version=1, name="x", values={"lora_strength": "strong"})


def test_render_profile_toml_rejects_non_string_look_value():
    with pytest.raises(ProfileError, match="cinematography must be a string"):
        render_profile_toml(version=1, name="x", values={"cinematography": 5})


def test_render_profile_toml_renders_scalar_provenance_types():
    text = render_profile_toml(
        version=1,
        name="x",
        values={"cinematography": "g"},
        provenance={"stance_index": 2, "notes": "kept"},
    )
    assert "stance_index = 2" in text
    reparsed = tomllib.loads(text)
    assert reparsed["provenance"] == {"stance_index": 2, "notes": "kept"}


def test_render_profile_toml_rejects_unrenderable_provenance_value():
    with pytest.raises(ProfileError, match="provenance.notes"):
        render_profile_toml(
            version=1, name="x", values={"cinematography": "g"}, provenance={"notes": ["a"]}
        )


def test_promote_photography_rejects_blank_name(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    with pytest.raises(ProfileError, match="name must be a non-empty string"):
        promote_photography(run_dir, name="  ", out_path=tmp_path / "out.toml")


def test_promote_photography_rejects_bad_version(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    with pytest.raises(ProfileError, match="version must be an int"):
        promote_photography(run_dir, name="x", version=0, out_path=tmp_path / "out.toml")


def test_promote_photography_missing_photography_json(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / ".authoring").mkdir(parents=True)
    with pytest.raises(ProfileError, match="cannot read"):
        promote_photography(run_dir, name="x", out_path=tmp_path / "out.toml")


def test_promote_photography_malformed_session_warns_and_still_writes(tmp_path: Path, caplog):
    run_dir = _build_run_dir(tmp_path)
    (run_dir / ".authoring" / "session.json").write_text("{not json", encoding="utf-8")
    out_path = tmp_path / "out.toml"
    with caplog.at_level("WARNING"):
        promote_photography(run_dir, name="x", out_path=out_path)
    # Partial provenance beats refusing the promotion over optional metadata.
    assert out_path.exists()
    assert "session.json" in caplog.text
    profile = load_profile(out_path)
    assert "model" not in profile.provenance


def test_promote_photography_rejects_unknown_provenance_extra(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    with pytest.raises(ProfileError, match="provenance_extra contains unknown key"):
        promote_photography(
            run_dir,
            name="x",
            out_path=tmp_path / "out.toml",
            provenance_extra={"director": "nobody"},
        )


def test_promote_photography_provenance_extra_is_merged(tmp_path: Path):
    run_dir = _build_run_dir(tmp_path)
    out_path = tmp_path / "out.toml"
    promote_photography(
        run_dir, name="x", out_path=out_path, provenance_extra={"song": "Deathless"}
    )
    assert load_profile(out_path).provenance["song"] == "Deathless"
