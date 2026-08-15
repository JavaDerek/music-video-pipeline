"""Tests for authoring session state (issue #54 design section 7).

Never auto-heals: :func:`stage_staleness` only reports. Whoever calls it
decides whether to regenerate -- that decision lives in the CLI layer
(``cli.py``'s ``status``/stage commands), not here.
"""

from __future__ import annotations

import json

import pytest

from music_video_maker.authoring.session import (
    STAGES,
    AuthoringSession,
    SessionError,
    StageRecord,
    session_path,
    stage_staleness,
)


def _record(**overrides) -> StageRecord:
    defaults = dict(
        stage="concept",
        model="claude-fable-5",
        completed_at="2026-08-12",
        cost_usd=0.04,
        input_hashes={"lyrics": "abc", "skeleton": "def"},
    )
    defaults.update(overrides)
    return StageRecord(**defaults)


# --------------------------------------------------------------------------- #
# session_path
# --------------------------------------------------------------------------- #


def test_session_path_lives_beside_the_run_directory(tmp_path):
    assert session_path(tmp_path) == tmp_path / ".authoring" / "session.json"


# --------------------------------------------------------------------------- #
# AuthoringSession: load / record / save round trip
# --------------------------------------------------------------------------- #


def test_load_of_a_missing_file_is_an_empty_session(tmp_path):
    session = AuthoringSession.load(tmp_path / "nope" / "session.json")
    assert session.get("concept") is None


def test_record_and_save_round_trips(tmp_path):
    path = tmp_path / ".authoring" / "session.json"
    session = AuthoringSession(path=path)
    session.record(_record())
    session.save()

    reloaded = AuthoringSession.load(path)
    record = reloaded.get("concept")
    assert record is not None
    assert record.model == "claude-fable-5"
    assert record.completed_at == "2026-08-12"
    assert record.cost_usd == pytest.approx(0.04)
    assert record.input_hashes == {"lyrics": "abc", "skeleton": "def"}


def test_save_creates_the_authoring_directory(tmp_path):
    path = tmp_path / "does" / "not" / "exist" / ".authoring" / "session.json"
    session = AuthoringSession(path=path)
    session.record(_record())
    session.save()
    assert path.exists()


def test_recording_an_unknown_stage_is_rejected(tmp_path):
    session = AuthoringSession(path=tmp_path / "session.json")
    with pytest.raises(SessionError):
        session.record(_record(stage="not-a-real-stage"))


def test_a_malformed_session_file_is_rejected_loudly(tmp_path):
    path = tmp_path / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SessionError):
        AuthoringSession.load(path)


def test_an_unknown_stage_in_the_file_is_ignored_not_fatal(tmp_path, caplog):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "stages": {
                    "concept": {
                        "model": "claude-fable-5",
                        "completed_at": "2026-08-12",
                        "cost_usd": 0.04,
                        "input_hashes": {},
                    },
                    "from_the_future": {
                        "model": "x",
                        "completed_at": "2026-08-12",
                        "cost_usd": None,
                        "input_hashes": {},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    import logging

    with caplog.at_level(logging.WARNING):
        session = AuthoringSession.load(path)

    assert session.get("concept") is not None
    assert "from_the_future" in caplog.text


def test_recording_the_same_stage_twice_overwrites(tmp_path):
    session = AuthoringSession(path=tmp_path / "session.json")
    session.record(_record(cost_usd=0.04))
    session.record(_record(cost_usd=0.09))
    assert session.get("concept").cost_usd == pytest.approx(0.09)


def test_stages_covers_all_four_design_stages_in_pipeline_order():
    assert STAGES == ("concept", "beats", "photography", "prose")


# --------------------------------------------------------------------------- #
# stage_staleness
# --------------------------------------------------------------------------- #


def test_no_record_is_not_started():
    check = stage_staleness(None, {"lyrics": "abc"})
    assert check.stale is True
    assert check.reason == "not started"


def test_matching_hashes_is_not_stale():
    record = _record(input_hashes={"lyrics": "abc", "skeleton": "def"})
    check = stage_staleness(record, {"lyrics": "abc", "skeleton": "def"})
    assert check.stale is False
    assert check.reason is None


def test_a_changed_hash_is_stale_and_names_the_input():
    record = _record(input_hashes={"lyrics": "abc", "skeleton": "def"})
    check = stage_staleness(record, {"lyrics": "CHANGED", "skeleton": "def"})
    assert check.stale is True
    assert "lyrics" in check.reason


def test_a_new_input_key_is_stale():
    """E.g. this stage started consuming a doc it didn't before -- a code
    change, not a content change, but still worth flagging rather than
    silently trusting the old record."""
    record = _record(input_hashes={"lyrics": "abc"})
    check = stage_staleness(record, {"lyrics": "abc", "guide": "xyz"})
    assert check.stale is True
    assert "guide" in check.reason


def test_staleness_never_raises_it_only_reports(tmp_path):
    """The module docstring's whole point: nothing here regenerates
    anything. Calling it never has a side effect on the session or the
    filesystem."""
    session = AuthoringSession(path=tmp_path / "session.json")
    session.record(_record(input_hashes={"lyrics": "abc"}))
    stage_staleness(session.get("concept"), {"lyrics": "CHANGED"})
    # The session itself is untouched -- still in memory, never saved.
    assert not (tmp_path / "session.json").exists()
