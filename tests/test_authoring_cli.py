"""Tests for the ``mvm-author`` CLI (issue #54).

Two layers, mirroring ``tests/test_cli.py``'s own split: argv/exit-code
wiring with the driver monkeypatched out, plus the two things worth
integration-testing end to end -- that ``concept`` actually writes
``.authoring/concept.json``/``session.json``, and that ``--dry-run`` writes
nothing and calls no driver at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from music_video_maker import alignment
from music_video_maker.authoring import cli as auth_cli
from music_video_maker.authoring.driver import ScriptedDriver
from tests.harness.factories import write_silent_wav

VALID_CONCEPT = {
    "logline": "A drummer walks through gathering weather as a storm builds and clears.",
    "setting": "a coastal town, contemporary",
    "tone": "elegiac, quiet",
    "motifs": ["gathering clouds"],
    "avoid": ["literal lightning strikes"],
    # Issue #78: the closed vocabulary the beats stage assigns `location`
    # from.
    "locations": ["the boardwalk", "the empty pier"],
}


class _FakeAlignModel:
    """Stand-in for a stable-ts model (issue #3's pattern, mirrored from
    ``tests/test_alignment.py`` and ``tests/test_cli.py``): a status check
    that edits the lyrics file re-runs real forced alignment, and this
    module -- unlike ``run_pipeline`` -- has no ``align_model=`` seam to
    inject through, so the fake is patched in at ``_load_model`` instead."""

    def __init__(self, text: str) -> None:
        self._text = text

    def align(self, _audio: str, _text: str, **_kwargs: object) -> SimpleNamespace:
        words = self._text.split()
        span = 4.0 / len(words)
        word_objs = [
            SimpleNamespace(
                word=w, start=i * span, end=(i + 1) * span, probability=0.99
            )
            for i, w in enumerate(words)
        ]
        segment = SimpleNamespace(text=self._text, start=0.0, end=4.0, words=word_objs)
        return SimpleNamespace(segments=[segment], language="en")


def _write_config(
    tmp_path: Path,
    *,
    lyrics_text: str = "",
    instrumental_coverage: bool = True,
    master_seconds: float = 20.0,
    instrumental_shot_seconds: float | None = None,
) -> Path:
    master_audio = write_silent_wav(tmp_path / "audio" / "master.wav", seconds=master_seconds)
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text(lyrics_text, encoding="utf-8")
    cast_dir = tmp_path / "cast"
    cast_dir.mkdir()
    (cast_dir / "dianne_ref.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    (cast_dir / "rex_ref.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    workflow = tmp_path / "workflow_api.json"
    workflow.write_text("{}")

    config_path = tmp_path / "run.toml"
    config_path.write_text(
        f"""
master_audio = "{master_audio}"
lyrics_file = "{lyrics_file}"
global_style = "Refestramus progressive rock music video"
narrative_concept = "placeholder"
default_lead_vocalist = "Dianne"
comfyui_url = "http://doris:8188"
workflow_template = "{workflow}"
chunks_dir = "{tmp_path / "output" / "chunks"}"
final_video_dir = "{tmp_path / "output" / "final"}"
instrumental_coverage = {"true" if instrumental_coverage else "false"}
{f"instrumental_shot_seconds = {instrumental_shot_seconds}" if instrumental_shot_seconds else ""}

[cast.Dianne]
role = "Lead Vocalist"
image = "{cast_dir / "dianne_ref.jpg"}"

[cast.Rex]
role = "Drummer, background, never sings"
image = "{cast_dir / "rex_ref.jpg"}"

[hardware]
name = "RTX 4090"
vram_gb = 24.0
"""
    )
    return config_path


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def test_config_is_required():
    with pytest.raises(SystemExit):
        auth_cli.main(["concept"])


def test_a_command_is_required(tmp_path):
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        auth_cli.main(["--config", str(config_path)])


def test_main_returns_error_on_missing_config_file(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.toml"
    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(missing), "status"])
    assert exit_code == auth_cli.EXIT_ERROR
    assert any("Failed to load run config" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# concept --dry-run: prints the prompts, calls nothing, writes nothing
# --------------------------------------------------------------------------- #


def test_dry_run_prints_both_prompts_and_writes_nothing(tmp_path, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")

    exit_code = auth_cli.main(["--config", str(config_path), "concept", "--dry-run"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "SYSTEM PROMPT" in out
    assert "USER PROMPT" in out
    assert "no lyric is ever sung on screen" in out  # this run's empty lyrics.txt
    assert not (config_path.parent / ".authoring").exists()


def test_dry_run_includes_notes(tmp_path, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")

    auth_cli.main(
        ["--config", str(config_path), "concept", "--dry-run", "--notes", "too bleak"]
    )

    assert "too bleak" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# concept: real run (driver monkeypatched to a ScriptedDriver)
# --------------------------------------------------------------------------- #


def test_concept_writes_the_expected_files(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))

    exit_code = auth_cli.main(["--config", str(config_path), "concept"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    authoring_dir = config_path.parent / ".authoring"
    concept_data = json.loads((authoring_dir / "concept.json").read_text())
    assert concept_data == VALID_CONCEPT
    assert (authoring_dir / "raw" / "concept.txt").exists()

    session = json.loads((authoring_dir / "session.json").read_text())
    assert session["stages"]["concept"]["model"] == "claude-fable-5"
    assert "lyrics" in session["stages"]["concept"]["input_hashes"]


def test_concept_prints_a_human_readable_summary(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))

    auth_cli.main(["--config", str(config_path), "concept"])

    out = capsys.readouterr().out
    assert VALID_CONCEPT["logline"] in out
    # Issue #78: the closed vocabulary is exactly what a human has to review
    # here -- everything downstream is validated against it, so a bad list
    # is worth catching at this review point, not forty beats later.
    assert "the boardwalk" in out
    assert "the empty pier" in out


def test_concept_notes_reach_the_prompt(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    driver = ScriptedDriver([VALID_CONCEPT])
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: driver)

    auth_cli.main(["--config", str(config_path), "concept", "--notes", "no rain"])

    assert "no rain" in driver.calls[0]["prompt"]


def test_concept_returns_error_when_the_skeleton_is_empty(tmp_path, caplog):
    """Empty lyrics + instrumental_coverage off is a genuine "nothing to
    author against" state (mirrors --prepare's own PipelineError) -- must
    fail loudly rather than call the driver with an empty skeleton."""
    config_path = _write_config(tmp_path, lyrics_text="", instrumental_coverage=False)

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "concept"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert any("Could not load the chunk skeleton" in r.message for r in caplog.records)


def test_concept_returns_error_when_generation_fails(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([None, None, None]))

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "concept"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert not (config_path.parent / ".authoring" / "concept.json").exists()


# --------------------------------------------------------------------------- #
# beats (issue #54 phase 2)
# --------------------------------------------------------------------------- #


def _run_concept(config_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))
    assert auth_cli.main(["--config", str(config_path), "concept"]) == auth_cli.EXIT_SUCCESS


def _beat_sheet(config_path: Path, *, lengths: dict[int, float] | None = None) -> dict:
    """A one-beat-per-chunk sheet for whatever the skeleton actually is."""
    from music_video_maker.authoring.chunks import load_chunk_skeleton
    from music_video_maker.config import load_config

    chunks = load_chunk_skeleton(load_config(config_path))
    lengths = lengths or {}
    return {
        "beats": [
            {
                "chunk_id": c.chunk_id,
                "beat": f"beat {c.chunk_id}",
                "beat_role": "instrumental" if c.is_instrumental else "transition",
                "beat_group": c.chunk_id + 1,
                "focus": "subject",
                # Issue #78: required, and validated against the concept's
                # own `locations` list when one is supplied (VALID_CONCEPT's
                # above, throughout this file).
                "location": "the boardwalk",
                **({"length_seconds": lengths[c.chunk_id]} if c.chunk_id in lengths else {}),
            }
            for c in chunks
        ]
    }


def test_beats_refuses_to_run_before_the_concept_exists(tmp_path, caplog):
    """Stage 2's whole input is the approved concept. Running without one
    would generate a beat sheet descending from nothing a human ever read."""
    config_path = _write_config(tmp_path, lyrics_text="")

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "beats"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert any("concept" in r.message for r in caplog.records)


def test_beats_dry_run_prints_the_prompts_and_calls_nothing(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    capsys.readouterr()
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: pytest.fail("--dry-run must call nothing")
    )

    exit_code = auth_cli.main(["--config", str(config_path), "beats", "--dry-run"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "SYSTEM PROMPT" in out
    assert "The three-beat rule" in out  # the guide, read from disk
    assert VALID_CONCEPT["logline"] in out
    assert not (config_path.parent / ".authoring" / "beats.json").exists()


def test_beats_writes_the_expected_files(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    sheet = _beat_sheet(config_path)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([sheet]))

    exit_code = auth_cli.main(["--config", str(config_path), "beats"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    authoring_dir = config_path.parent / ".authoring"
    payload = json.loads((authoring_dir / "beats.json").read_text())
    assert [b["chunk_id"] for b in payload["beats"]] == [
        b["chunk_id"] for b in sheet["beats"]
    ]
    assert payload["reanchored"] is False
    assert (authoring_dir / "raw" / "beats.txt").exists()

    session = json.loads((authoring_dir / "session.json").read_text())
    assert session["stages"]["beats"]["model"] == "claude-opus-5"
    assert "concept" in session["stages"]["beats"]["input_hashes"]


def test_beats_prints_each_beats_location(tmp_path, monkeypatch, capsys):
    """Issue #78: `location` is as much a per-beat review item as
    `beat_role`/`beat_group` already are -- a human skimming this stage's
    console output should see where every beat is set, not just what
    happens."""
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    sheet = _beat_sheet(config_path)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([sheet]))

    exit_code = auth_cli.main(["--config", str(config_path), "beats"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "the boardwalk" in out


def test_beats_records_the_re_cut_timeline_when_a_length_moves_it(tmp_path, monkeypatch):
    """The written beats carry the anchors of the timeline a render will
    actually produce, not the ones the sheet was authored against."""
    config_path = _write_config(
        tmp_path, lyrics_text="", master_seconds=60.0, instrumental_shot_seconds=6.0
    )
    first_chunk = _beat_sheet(config_path)["beats"][0]["chunk_id"]
    sheet = _beat_sheet(config_path, lengths={first_chunk: 15.0})
    _run_concept(config_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([sheet]))

    assert auth_cli.main(["--config", str(config_path), "beats"]) == auth_cli.EXIT_SUCCESS

    payload = json.loads((config_path.parent / ".authoring" / "beats.json").read_text())
    assert payload["reanchored"] is True
    assert len(payload["beats"]) < len(sheet["beats"])
    assert any(b["merged_from"] for b in payload["beats"])


def test_beats_notes_reach_the_prompt(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    driver = ScriptedDriver([_beat_sheet(config_path)])
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: driver)

    auth_cli.main(["--config", str(config_path), "beats", "--notes", "fewer doorways"])

    assert "fewer doorways" in driver.calls[0]["prompt"]


def test_beats_writes_nothing_when_generation_fails(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([None, None, None]))

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "beats"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert not (config_path.parent / ".authoring" / "beats.json").exists()


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def test_status_reports_an_error_row_when_the_skeleton_is_empty(tmp_path, capsys):
    config_path = _write_config(tmp_path, lyrics_text="", instrumental_coverage=False)

    exit_code = auth_cli.main(["--config", str(config_path), "status"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    concept_line = next(line for line in out.splitlines() if line.startswith("concept"))
    assert "ERROR" in concept_line


def test_status_before_anything_has_run(tmp_path, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")

    exit_code = auth_cli.main(["--config", str(config_path), "status"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "concept" in out and "not started" in out
    assert "beats" in out
    assert "photography" in out
    assert "prose" in out


def test_status_reports_ok_after_a_successful_concept_run(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))
    auth_cli.main(["--config", str(config_path), "concept"])
    capsys.readouterr()  # discard concept's own output

    auth_cli.main(["--config", str(config_path), "status"])

    out = capsys.readouterr().out
    concept_line = next(line for line in out.splitlines() if line.startswith("concept"))
    assert "ok" in concept_line
    assert "claude-fable-5" in concept_line


def test_status_reports_stale_after_the_lyrics_file_changes(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))
    auth_cli.main(["--config", str(config_path), "concept"])
    capsys.readouterr()

    lyrics_path = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in config_path.read_text().splitlines()
        if line.startswith("lyrics_file")
    )
    new_lyrics = "a whole new lyric line"
    Path(lyrics_path).write_text(new_lyrics + "\n", encoding="utf-8")

    monkeypatch.setattr(
        alignment, "_load_model", lambda _model_size: _FakeAlignModel(new_lyrics)
    )
    auth_cli.main(["--config", str(config_path), "status"])

    out = capsys.readouterr().out
    concept_line = next(line for line in out.splitlines() if line.startswith("concept"))
    assert "STALE" in concept_line
    assert "lyrics" in concept_line


def test_status_reports_beats_stale_when_the_concept_is_re_rolled(tmp_path, monkeypatch, capsys):
    """Staleness cascades, and is never auto-healed (design section 7).
    Re-rolling the concept does not silently leave a beat sheet descending
    from a paragraph nobody approved -- it says so and stops."""
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_beat_sheet(config_path)])
    )
    auth_cli.main(["--config", str(config_path), "beats"])
    capsys.readouterr()

    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver([{**VALID_CONCEPT, "tone": "comic, brisk"}]),
    )
    auth_cli.main(["--config", str(config_path), "concept"])
    capsys.readouterr()

    auth_cli.main(["--config", str(config_path), "status"])

    out = capsys.readouterr().out
    beats_line = next(line for line in out.splitlines() if line.startswith("beats"))
    assert "STALE" in beats_line
    assert "concept" in beats_line


def test_status_reports_beats_ok_after_a_successful_run(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_beat_sheet(config_path)])
    )
    auth_cli.main(["--config", str(config_path), "beats"])
    capsys.readouterr()

    auth_cli.main(["--config", str(config_path), "status"])

    beats_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("beats")
    )
    assert "ok" in beats_line
    assert "claude-opus-5" in beats_line


# --------------------------------------------------------------------------- #
# prose and write (issue #54 phase 3)
# --------------------------------------------------------------------------- #


def _run_beats(config_path: Path, monkeypatch, *, lengths=None) -> None:
    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver([_beat_sheet(config_path, lengths=lengths)]),
    )
    assert auth_cli.main(["--config", str(config_path), "beats"]) == auth_cli.EXIT_SUCCESS


def _prose_replies(config_path: Path, text: str = "Rain crosses the whole pavement") -> list:
    """One reply per beat group. Each carries every chunk; the ones outside a
    given window are dropped as read-only context, which is exactly the
    behaviour worth exercising here."""
    from music_video_maker.authoring.chunks import load_chunk_skeleton
    from music_video_maker.config import load_config

    chunks = load_chunk_skeleton(load_config(config_path))
    full = {"shots": [{"chunk_id": c.chunk_id, "shot": text} for c in chunks]}
    return [full for _ in chunks]


def test_prose_refuses_to_run_before_beats_exist(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "prose"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert any("beats" in r.message for r in caplog.records)


def test_prose_dry_run_prints_the_guide_and_calls_nothing(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    capsys.readouterr()
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: pytest.fail("--dry-run must call nothing")
    )

    exit_code = auth_cli.main(["--config", str(config_path), "prose", "--dry-run"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "grammatical subject" in out.lower()
    assert not (config_path.parent / ".authoring" / "prose.json").exists()


def test_prose_writes_the_expected_files(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver(_prose_replies(config_path))
    )

    assert auth_cli.main(["--config", str(config_path), "prose"]) == auth_cli.EXIT_SUCCESS

    authoring_dir = config_path.parent / ".authoring"
    payload = json.loads((authoring_dir / "prose.json").read_text())
    assert payload["shots"]
    assert (authoring_dir / "raw" / "prose_00.txt").exists()
    session = json.loads((authoring_dir / "session.json").read_text())
    assert session["stages"]["prose"]["model"] == "claude-sonnet-5"
    assert "beats" in session["stages"]["prose"]["input_hashes"]


def test_prose_for_one_group_keeps_the_prose_approved_for_the_others(tmp_path, monkeypatch):
    """`--groups` exists so a human can re-roll one sequence. Everything they
    already approved has to survive that."""
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver(_prose_replies(config_path, "The original line")),
    )
    auth_cli.main(["--config", str(config_path), "prose"])

    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver([{"shots": [{"chunk_id": 0, "shot": "A rewritten line"}]}]),
    )
    auth_cli.main(["--config", str(config_path), "prose", "--groups", "1"])

    shots = json.loads((config_path.parent / ".authoring" / "prose.json").read_text())["shots"]
    assert shots["0"] == "A rewritten line"
    assert any(text == "The original line" for text in shots.values())


def test_prose_rejects_a_beat_sheet_the_alignment_has_moved_under(tmp_path, monkeypatch, caplog):
    """Caught one layer earlier than ShotPlanDriftError, where re-running a
    stage is the cheap fix rather than a wasted render."""
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)

    beats_path = config_path.parent / ".authoring" / "beats.json"
    payload = json.loads(beats_path.read_text())
    payload["beats"][0]["start"] = 999.0
    beats_path.write_text(json.dumps(payload))

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "prose"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert any("no longer produces" in r.message for r in caplog.records)


def _author_through_prose(tmp_path, monkeypatch) -> Path:
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver(_prose_replies(config_path))
    )
    auth_cli.main(["--config", str(config_path), "prose"])
    return config_path


def test_write_produces_a_plan_that_loads_against_the_run_it_was_made_for(
    tmp_path, monkeypatch, capsys
):
    """The acceptance criterion from design section 13: it loads with zero
    errors and resolves against the same config without ShotPlanDriftError."""
    from music_video_maker.authoring.chunks import load_chunk_skeleton
    from music_video_maker.config import load_config
    from music_video_maker.shot_plan import load_shot_plan, resolve_shot

    config_path = _author_through_prose(tmp_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([])
    )  # a clean plan needs no revision

    assert auth_cli.main(["--config", str(config_path), "write"]) == auth_cli.EXIT_SUCCESS

    out = config_path.parent / "shot_plan.toml"
    config = load_config(config_path)
    plan = load_shot_plan(out, setting=config.setting, cast_names=tuple(config.cast))
    for chunk in load_chunk_skeleton(config):
        assert resolve_shot(plan, chunk)  # no drift, no blank
    assert "Wrote" in capsys.readouterr().out


def test_write_refuses_to_clobber_without_force(tmp_path, monkeypatch, caplog):
    config_path = _author_through_prose(tmp_path, monkeypatch)
    out = config_path.parent / "shot_plan.toml"
    out.write_text("# hand-authored\n")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "write"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert out.read_text() == "# hand-authored\n"


def test_write_records_provenance_in_the_file(tmp_path, monkeypatch):
    config_path = _author_through_prose(tmp_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))
    auth_cli.main(["--config", str(config_path), "write"])

    provenance = tomllib.loads((config_path.parent / "shot_plan.toml").read_text())["provenance"]
    assert provenance["concept_model"] == "claude-fable-5"
    assert provenance["beats_model"] == "claude-opus-5"
    assert provenance["prose_model"] == "claude-sonnet-5"
    assert len(provenance["lyrics_sha256"]) == 64


def test_write_refuses_before_prose_exists(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "write"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert not (config_path.parent / "shot_plan.toml").exists()


def test_status_reports_a_human_edit_to_a_generated_line(tmp_path, monkeypatch, capsys):
    config_path = _author_through_prose(tmp_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))
    auth_cli.main(["--config", str(config_path), "write"])
    capsys.readouterr()

    out = config_path.parent / "shot_plan.toml"
    out.write_text(
        out.read_text().replace("Rain crosses the whole pavement", "A human wrote this", 1)
    )

    auth_cli.main(["--config", str(config_path), "status"])

    assert "edited since they were generated" in capsys.readouterr().out


def test_status_is_quiet_about_edits_when_there_are_none(tmp_path, monkeypatch, capsys):
    config_path = _author_through_prose(tmp_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))
    auth_cli.main(["--config", str(config_path), "write"])
    capsys.readouterr()

    auth_cli.main(["--config", str(config_path), "status"])

    assert "edited since" not in capsys.readouterr().out


def test_status_reports_prose_ok_then_stale_when_beats_are_re_rolled(
    tmp_path, monkeypatch, capsys
):
    config_path = _author_through_prose(tmp_path, monkeypatch)
    capsys.readouterr()
    auth_cli.main(["--config", str(config_path), "status"])
    assert "ok" in next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("prose")
    )

    _run_beats(config_path, monkeypatch, lengths={0: 15.0})
    capsys.readouterr()
    auth_cli.main(["--config", str(config_path), "status"])

    prose_line = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("prose")
    )
    assert "STALE" in prose_line


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("3", {3}), ("3,5", {3, 5}), ("3-5", {3, 4, 5}), ("1,4-6", {1, 4, 5, 6})],
)
def test_groups_are_parsed_as_ids_and_ranges(raw, expected):
    assert auth_cli._parse_groups(raw) == expected


def test_no_groups_means_every_group():
    assert auth_cli._parse_groups(None) is None


# --------------------------------------------------------------------------- #
# photography and all (issue #54 phase 4)
# --------------------------------------------------------------------------- #


def _photography_reply(config_path: Path, look: str = "35mm, shallow focus, overcast") -> dict:
    from music_video_maker.authoring.chunks import load_chunk_skeleton
    from music_video_maker.config import load_config

    chunks = load_chunk_skeleton(load_config(config_path))
    return {
        "cinematography": look,
        "camera": [
            {"chunk_id": c.chunk_id, "camera": "tracking backwards ahead of her"}
            for c in chunks
        ],
    }


def test_photography_writes_the_expected_files(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_photography_reply(config_path)])
    )

    assert auth_cli.main(["--config", str(config_path), "photography"]) == auth_cli.EXIT_SUCCESS

    authoring_dir = config_path.parent / ".authoring"
    payload = json.loads((authoring_dir / "photography.json").read_text())
    assert payload["cinematography"].startswith("35mm")
    assert payload["camera"]["0"] == "tracking backwards ahead of her"
    session = json.loads((authoring_dir / "session.json").read_text())
    assert session["stages"]["photography"]["model"] == "claude-opus-5"


def test_candidates_write_a_candidates_file_and_adopt_nothing(tmp_path, monkeypatch, capsys):
    """Choosing is the human's move. Silently adopting one would skip the
    only review this stage has."""
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver(
            [_photography_reply(config_path, f"look {i}") for i in range(3)]
        ),
    )

    exit_code = auth_cli.main(
        ["--config", str(config_path), "photography", "--candidates", "3"]
    )

    assert exit_code == auth_cli.EXIT_SUCCESS
    authoring_dir = config_path.parent / ".authoring"
    candidates = json.loads((authoring_dir / "photography_candidates.json").read_text())
    assert len(candidates) == 3
    assert not (authoring_dir / "photography.json").exists()
    assert "CANDIDATE 1" in capsys.readouterr().out


def test_pick_freezes_the_chosen_candidate(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli,
        "ClaudeCliDriver",
        lambda: ScriptedDriver(
            [_photography_reply(config_path, f"look {i}") for i in range(3)]
        ),
    )
    auth_cli.main(["--config", str(config_path), "photography", "--candidates", "3"])
    capsys.readouterr()

    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: pytest.fail("--pick must call no model")
    )
    exit_code = auth_cli.main(["--config", str(config_path), "photography", "--pick", "2"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    payload = json.loads((config_path.parent / ".authoring" / "photography.json").read_text())
    assert payload["cinematography"] == "look 1"


def test_pick_out_of_range_is_an_error(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_photography_reply(config_path)])
    )
    auth_cli.main(["--config", str(config_path), "photography", "--candidates", "1"])

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "photography", "--pick", "9"])

    assert exit_code == auth_cli.EXIT_ERROR


def test_pick_before_any_candidates_is_an_error(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "photography", "--pick", "1"])

    assert exit_code == auth_cli.EXIT_ERROR


def test_write_puts_the_generated_camera_into_the_plan(tmp_path, monkeypatch):
    from music_video_maker.shot_plan import load_shot_plan

    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_photography_reply(config_path)])
    )
    auth_cli.main(["--config", str(config_path), "photography"])
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver(_prose_replies(config_path))
    )
    auth_cli.main(["--config", str(config_path), "prose"])
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))

    assert auth_cli.main(["--config", str(config_path), "write"]) == auth_cli.EXIT_SUCCESS

    plan = load_shot_plan(config_path.parent / "shot_plan.toml")
    assert plan[0].camera == "tracking backwards ahead of her"


def test_write_works_with_no_photography_at_all(tmp_path, monkeypatch):
    """Photography is optional -- an absent camera composes nothing, and a run
    that skips the stage must still produce a plan."""
    from music_video_maker.shot_plan import load_shot_plan

    config_path = _author_through_prose(tmp_path, monkeypatch)
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([]))

    assert auth_cli.main(["--config", str(config_path), "write"]) == auth_cli.EXIT_SUCCESS

    assert load_shot_plan(config_path.parent / "shot_plan.toml")[0].camera is None


def test_all_runs_every_stage_and_writes_the_plan(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    # One driver shared across every stage: each stage constructs its own, so
    # a fresh ScriptedDriver per call would replay the concept reply for beats.
    driver = ScriptedDriver(
        [VALID_CONCEPT, _beat_sheet(config_path), _photography_reply(config_path)]
        + _prose_replies(config_path)
    )
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: driver)

    exit_code = auth_cli.main(["--config", str(config_path), "all", "--yes"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    assert (config_path.parent / "shot_plan.toml").exists()
    out = capsys.readouterr().out
    for stage in ("CONCEPT", "BEATS", "PHOTOGRAPHY", "PROSE", "WRITE"):
        assert stage in out


def test_all_stops_rather_than_approving_itself_without_a_tty(tmp_path, monkeypatch, capsys):
    """An unattended `all` that approved its own output would spend four model
    calls and write a plan nobody looked at -- the one thing the review loop
    exists to prevent."""
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([VALID_CONCEPT]))

    exit_code = auth_cli.main(["--config", str(config_path), "all"])

    assert exit_code == auth_cli.EXIT_SUCCESS
    assert not (config_path.parent / "shot_plan.toml").exists()
    assert "stopping here" in capsys.readouterr().out


def test_all_stops_when_a_stage_fails(tmp_path, monkeypatch, caplog):
    config_path = _write_config(tmp_path, lyrics_text="")
    monkeypatch.setattr(auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([None, None, None]))

    with caplog.at_level("ERROR"):
        exit_code = auth_cli.main(["--config", str(config_path), "all", "--yes"])

    assert exit_code == auth_cli.EXIT_ERROR
    assert not (config_path.parent / "shot_plan.toml").exists()


def test_status_reports_photography(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, lyrics_text="")
    _run_concept(config_path, monkeypatch)
    _run_beats(config_path, monkeypatch)
    monkeypatch.setattr(
        auth_cli, "ClaudeCliDriver", lambda: ScriptedDriver([_photography_reply(config_path)])
    )
    auth_cli.main(["--config", str(config_path), "photography"])
    capsys.readouterr()

    auth_cli.main(["--config", str(config_path), "status"])

    line = next(
        row for row in capsys.readouterr().out.splitlines() if row.startswith("photography")
    )
    assert "ok" in line and "claude-opus-5" in line
