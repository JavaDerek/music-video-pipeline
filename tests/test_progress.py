"""Tests for the offline progress layer (issue #36).

``music_video_maker.progress`` turns ``run_state.json`` -- already persisted
atomically after every chunk by ``resilience.ResilientRunner`` -- into a
``RunProgress`` snapshot and a stream of diff events a browser-facing server
(not built here) could relay as SSE. Everything below is offline: no HTTP, no
sockets, no real files beyond ``tmp_path``, no GPU.

``RunState``/``ChunkResult``/``ChunkFingerprint`` have no shared factory in
``tests/harness/factories.py`` (checked before writing this file), so this
module builds them directly with a small local helper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from music_video_maker import resilience as resilience_module
from music_video_maker.contracts import ChunkResult, ChunkStatus, RunState
from music_video_maker.progress import (
    ChunkProgress,
    ProgressError,
    ProgressEvent,
    RunProgress,
    events_between,
    format_sse,
    read_run_state,
)


def _result(
    chunk_id: int,
    status: ChunkStatus,
    *,
    attempts: int = 1,
    errors: tuple[str, ...] = (),
    render_seconds: float | None = None,
    video_file: Path | None = None,
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        status=status,
        video_file=video_file,
        attempts=attempts,
        errors=errors,
        render_seconds=render_seconds,
    )


# --------------------------------------------------------------------------- #
# ChunkProgress
# --------------------------------------------------------------------------- #


def test_chunk_progress_to_dict_and_terminal_by_status():
    rendered = ChunkProgress(
        chunk_id=1,
        status="rendered",
        attempts=1,
        errors=(),
        render_seconds=12.0,
        video_file="chunk_0001.mp4",
    )
    assert rendered.terminal is True
    assert rendered.to_dict() == {
        "chunk_id": 1,
        "status": "rendered",
        "attempts": 1,
        "errors": [],
        "render_seconds": 12.0,
        "video_file": "chunk_0001.mp4",
    }

    cached = ChunkProgress(2, "cached", 1, (), 0.001, "chunk_0002.mp4")
    dead = ChunkProgress(3, "dead_lettered", 3, ("boom",), None, None)
    pending = ChunkProgress(4, "pending", 0, (), None, None)
    failed = ChunkProgress(5, "failed", 1, ("transient",), None, None)

    assert cached.terminal is True
    assert dead.terminal is True
    assert pending.terminal is False
    assert failed.terminal is False


# --------------------------------------------------------------------------- #
# RunProgress.from_run_state
# --------------------------------------------------------------------------- #


def test_from_run_state_all_pending_when_no_results():
    run_state = RunState(run_id="run-1")
    progress = RunProgress.from_run_state(run_state, [1, 2, 3])

    assert [c.chunk_id for c in progress.chunks] == [1, 2, 3]
    assert all(c.status == "pending" for c in progress.chunks)
    assert all(c.attempts == 0 for c in progress.chunks)
    assert all(c.errors == () for c in progress.chunks)
    assert all(c.render_seconds is None for c in progress.chunks)
    assert all(c.video_file is None for c in progress.chunks)
    assert progress.pending == 3
    assert progress.completed == 0
    assert progress.finished is False


def test_from_run_state_maps_each_chunk_status(tmp_path: Path):
    video = tmp_path / "chunk_0001.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    run_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.RENDERED, render_seconds=120.5, video_file=video),
            2: _result(2, ChunkStatus.CACHED, render_seconds=88.0, video_file=video),
            3: _result(3, ChunkStatus.DEAD_LETTERED, attempts=3, errors=("boom",)),
            # ChunkStatus.PENDING is never persisted by real code, but the
            # type allows it (e.g. a hand-edited or future-schema state
            # file) -- this is the "neither succeeded nor dead-lettered"
            # case the brief calls out explicitly.
            4: _result(4, ChunkStatus.PENDING, attempts=1, errors=("transient",)),
        },
    )

    progress = RunProgress.from_run_state(run_state, [1, 2, 3, 4, 5])
    by_id = {c.chunk_id: c for c in progress.chunks}

    assert by_id[1].status == "rendered"
    assert by_id[1].terminal is True
    assert by_id[2].status == "cached"
    assert by_id[2].terminal is True
    assert by_id[3].status == "dead_lettered"
    assert by_id[3].errors == ("boom",)
    assert by_id[3].terminal is True
    assert by_id[4].status == "failed"
    assert by_id[4].terminal is False
    assert by_id[5].status == "pending"
    assert by_id[5].terminal is False


def test_from_run_state_drops_unexpected_chunk_id_and_warns(caplog: pytest.LogCaptureFixture):
    run_state = RunState(run_id="run-1", results={99: _result(99, ChunkStatus.RENDERED)})

    with caplog.at_level(logging.WARNING):
        progress = RunProgress.from_run_state(run_state, [1, 2])

    assert [c.chunk_id for c in progress.chunks] == [1, 2]
    assert any("99" in r.message for r in caplog.records)


def test_from_run_state_orders_chunks_by_chunk_id_regardless_of_input_order():
    run_state = RunState(
        run_id="run-1",
        results={3: _result(3, ChunkStatus.RENDERED), 1: _result(1, ChunkStatus.RENDERED)},
    )
    progress = RunProgress.from_run_state(run_state, [3, 1, 2])
    assert [c.chunk_id for c in progress.chunks] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# mean_render_seconds / projected_remaining_seconds / finished
# --------------------------------------------------------------------------- #


def test_mean_render_seconds_ignores_cached_chunks():
    # D3 (issue #36): a CACHED chunk's render_seconds is whatever a PRIOR run
    # recorded when it actually rendered -- it cost this run's GPU nothing at
    # all, since --resume skipped it. Folding it into "how fast is this run
    # going" is the averaging bug that looks right in the happy case and lies
    # on exactly the run (--resume) where a human most needs the number. The
    # cached value here (0.001) is deliberately extreme -- if a future
    # "simplification" averages every terminal chunk's render_seconds
    # together instead of RENDERED chunks only, this assertion fails loudly
    # instead of drifting a few seconds off silently.
    run_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.RENDERED, render_seconds=200.0),
            2: _result(2, ChunkStatus.RENDERED, render_seconds=220.0),
            3: _result(3, ChunkStatus.CACHED, render_seconds=0.001),
        },
    )
    progress = RunProgress.from_run_state(run_state, [1, 2, 3])
    assert progress.mean_render_seconds == pytest.approx(210.0)


def test_mean_render_seconds_is_none_when_nothing_rendered():
    run_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.CACHED, render_seconds=50.0)},
    )
    progress = RunProgress.from_run_state(run_state, [1, 2])
    assert progress.mean_render_seconds is None
    assert progress.projected_remaining_seconds is None


def test_projected_remaining_seconds_multiplies_mean_by_remaining_chunk_count():
    run_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.RENDERED, render_seconds=100.0),
            2: _result(2, ChunkStatus.RENDERED, render_seconds=300.0),
            # Dead-lettered is terminal -- no more GPU work is coming for it
            # in this run -- so it must not count as "remaining".
            3: _result(3, ChunkStatus.DEAD_LETTERED, attempts=3, errors=("x",)),
        },
    )
    # Expected chunks 1..5: 1 and 2 are rendered, 3 is dead-lettered
    # (terminal), 4 and 5 are pending -- 2 chunks actually remain.
    progress = RunProgress.from_run_state(run_state, [1, 2, 3, 4, 5])
    assert progress.mean_render_seconds == pytest.approx(200.0)
    assert progress.projected_remaining_seconds == pytest.approx(400.0)


def test_finished_requires_every_expected_chunk_terminal_including_dead_letters():
    run_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.RENDERED),
            2: _result(2, ChunkStatus.DEAD_LETTERED, attempts=3, errors=("x",)),
        },
    )
    complete = RunProgress.from_run_state(run_state, [1, 2])
    assert complete.finished is True

    partial = RunProgress.from_run_state(run_state, [1, 2, 3])
    assert partial.finished is False


def test_run_progress_to_dict_shape():
    run_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.RENDERED, render_seconds=100.0)},
    )
    progress = RunProgress.from_run_state(run_state, [1, 2])
    data = progress.to_dict()

    assert data["run_id"] == "run-1"
    assert data["total"] == 2
    assert data["completed"] == 1
    assert data["rendered"] == 1
    assert data["cached"] == 0
    assert data["dead_lettered"] == []
    assert data["pending"] == 1
    assert data["finished"] is False
    assert [c["chunk_id"] for c in data["chunks"]] == [1, 2]


# --------------------------------------------------------------------------- #
# events_between
# --------------------------------------------------------------------------- #


def test_events_between_none_previous_emits_single_snapshot():
    run_state = RunState(run_id="run-1", results={1: _result(1, ChunkStatus.RENDERED)})
    current = RunProgress.from_run_state(run_state, [1, 2])

    events = events_between(None, current)

    assert len(events) == 1
    assert events[0].event == "run_snapshot"
    assert events[0].data == current.to_dict()


def test_events_between_no_change_emits_nothing():
    run_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.RENDERED, render_seconds=50.0)},
    )
    previous = RunProgress.from_run_state(run_state, [1, 2])
    current = RunProgress.from_run_state(run_state, [1, 2])

    assert events_between(previous, current) == ()


def test_events_between_newly_completed_chunk_emits_chunk_completed():
    previous = RunProgress.from_run_state(RunState(run_id="run-1"), [1, 2])
    current_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.RENDERED, render_seconds=100.0)},
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    events = events_between(previous, current)

    assert len(events) == 1
    event = events[0]
    assert event.event == "chunk_completed"
    assert event.data["chunk_id"] == 1
    assert event.data["status"] == "rendered"
    assert event.data["completed"] == current.completed
    assert event.data["total"] == current.total
    assert event.data["projected_remaining_seconds"] == current.projected_remaining_seconds


def test_events_between_multiple_completions_ordered_by_chunk_id():
    previous = RunProgress.from_run_state(RunState(run_id="run-1"), [1, 2, 3])
    current_state = RunState(
        run_id="run-1",
        results={
            3: _result(3, ChunkStatus.RENDERED, render_seconds=50.0),
            1: _result(1, ChunkStatus.CACHED, render_seconds=10.0),
        },
    )
    current = RunProgress.from_run_state(current_state, [1, 2, 3])

    events = events_between(previous, current)

    assert [e.event for e in events] == ["chunk_completed", "chunk_completed"]
    assert [e.data["chunk_id"] for e in events] == [1, 3]


def test_events_between_new_dead_letter_carries_error_history():
    # A second, still-pending chunk keeps the run unfinished so this test
    # isolates the dead-letter event from run_finished.
    previous_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.PENDING, attempts=1, errors=("first failure",))},
    )
    previous = RunProgress.from_run_state(previous_state, [1, 2])
    current_state = RunState(
        run_id="run-1",
        results={
            1: _result(
                1,
                ChunkStatus.DEAD_LETTERED,
                attempts=3,
                errors=("first failure", "second failure", "third failure"),
            )
        },
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    events = events_between(previous, current)

    assert len(events) == 1
    assert events[0].event == "chunk_dead_lettered"
    assert events[0].data["errors"] == ["first failure", "second failure", "third failure"]


def test_events_between_retry_without_terminal_status_emits_chunk_retried():
    previous_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.PENDING, attempts=1, errors=("e1",))},
    )
    previous = RunProgress.from_run_state(previous_state, [1, 2])
    current_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.PENDING, attempts=2, errors=("e1", "e2"))},
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    events = events_between(previous, current)

    assert len(events) == 1
    assert events[0].event == "chunk_retried"
    assert events[0].data["attempts"] == 2


def test_events_between_retry_that_lands_on_a_terminal_status_is_not_also_a_retry_event():
    previous_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.PENDING, attempts=1, errors=("e1",))},
    )
    previous = RunProgress.from_run_state(previous_state, [1, 2])
    current_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.RENDERED, attempts=2, render_seconds=90.0)},
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    events = events_between(previous, current)

    # attempts rose (1 -> 2) AND the chunk reached a terminal status this
    # poll -- it must be reported once, as a completion, not also as a retry.
    assert [e.event for e in events] == ["chunk_completed"]


def test_events_between_chunk_missing_from_previous_snapshot_is_not_treated_as_a_retry():
    # Defensive edge case: if a chunk_id current names has no counterpart in
    # previous at all (e.g. a poller's expected_chunk_ids widened between
    # polls), there is no "attempts rose" baseline to compare against -- it
    # must be skipped rather than guessed at as a retry.
    previous = RunProgress(
        run_id="run-1",
        chunks=(ChunkProgress(1, "failed", 1, ("e1",), None, None),),
    )
    current_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.PENDING, attempts=1, errors=("e1",)),
            2: _result(2, ChunkStatus.PENDING, attempts=1, errors=("e2",)),
        },
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    assert events_between(previous, current) == ()


def test_events_between_run_finished_emitted_once():
    previous = RunProgress.from_run_state(
        RunState(run_id="run-1", results={1: _result(1, ChunkStatus.RENDERED)}), [1, 2]
    )
    current_state = RunState(
        run_id="run-1",
        results={
            1: _result(1, ChunkStatus.RENDERED),
            2: _result(2, ChunkStatus.DEAD_LETTERED, attempts=3, errors=("x",)),
        },
    )
    current = RunProgress.from_run_state(current_state, [1, 2])

    events = events_between(previous, current)
    event_types = [e.event for e in events]
    assert event_types.count("run_finished") == 1

    # Polling again against the already-finished state must not re-emit.
    events_again = events_between(current, current)
    assert "run_finished" not in [e.event for e in events_again]


def test_events_between_run_id_change_re_snapshots_and_logs_error(
    caplog: pytest.LogCaptureFixture,
):
    previous = RunProgress.from_run_state(
        RunState(run_id="run-a", results={1: _result(1, ChunkStatus.RENDERED)}), [1]
    )
    current = RunProgress.from_run_state(
        RunState(run_id="run-b", results={1: _result(1, ChunkStatus.RENDERED)}), [1]
    )

    with caplog.at_level(logging.ERROR):
        events = events_between(previous, current)

    assert len(events) == 1
    assert events[0].event == "run_snapshot"
    assert events[0].data == current.to_dict()
    assert any("run-a" in r.message and "run-b" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# format_sse / to_json
# --------------------------------------------------------------------------- #


def test_format_sse_shape():
    event = ProgressEvent(event="run_snapshot", data={"a": 1})
    text = format_sse(event)
    assert text == f"event: run_snapshot\ndata: {json.dumps({'a': 1}, sort_keys=True)}\n\n"
    assert text.endswith("\n\n")


def test_to_json_is_stable_under_sort_keys_regardless_of_insertion_order():
    first = ProgressEvent(event="chunk_completed", data={"b": 2, "a": 1})
    second = ProgressEvent(event="chunk_completed", data={"a": 1, "b": 2})
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == {"a": 1, "b": 2}


def test_no_event_payload_contains_a_newline_even_with_a_multiline_error():
    run_state = RunState(
        run_id="run-1",
        results={
            1: _result(
                1,
                ChunkStatus.DEAD_LETTERED,
                attempts=2,
                errors=("boom\nwith an embedded traceback line",),
            )
        },
    )
    progress = RunProgress.from_run_state(run_state, [1])

    snapshot = events_between(None, progress)[0]
    assert "\n" not in snapshot.to_json()

    dead_letter_events = events_between(
        RunProgress.from_run_state(RunState(run_id="run-1"), [1]), progress
    )
    dead_letter_event = next(e for e in dead_letter_events if e.event == "chunk_dead_lettered")
    assert "\n" not in dead_letter_event.to_json()
    # format_sse's own frame is what a client actually parses -- confirm the
    # only newlines in it are the three the SSE format itself specifies
    # ("event: E\n" + "data: D\n" + the blank-line terminator "\n"), none of
    # them from the embedded "\n" in the error string (json.dumps escaped
    # that to the two characters "\\n" inside the payload instead).
    assert format_sse(dead_letter_event).count("\n") == 3


# --------------------------------------------------------------------------- #
# read_run_state
# --------------------------------------------------------------------------- #


def test_read_run_state_round_trips_a_file_written_by_resilience_serializer(tmp_path: Path):
    run_state = RunState(
        run_id="run-1",
        results={1: _result(1, ChunkStatus.RENDERED, render_seconds=12.5)},
    )
    # Build the payload with resilience's own serializer so this test breaks
    # if the schema moves, rather than re-encoding the format by hand here.
    payload = resilience_module._serialize_run_state(run_state)
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = read_run_state(path)

    assert loaded.run_id == "run-1"
    assert loaded.results[1].status == ChunkStatus.RENDERED
    assert loaded.results[1].render_seconds == 12.5


def test_read_run_state_accepts_a_string_path(tmp_path: Path):
    payload = resilience_module._serialize_run_state(RunState(run_id="run-1"))
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = read_run_state(str(path))
    assert loaded.run_id == "run-1"


def test_read_run_state_missing_file_raises_progress_error(tmp_path: Path):
    path = tmp_path / "does_not_exist.json"
    with pytest.raises(ProgressError) as exc_info:
        read_run_state(path)
    assert str(path) in str(exc_info.value)


def test_read_run_state_malformed_json_raises_progress_error(tmp_path: Path):
    path = tmp_path / "run_state.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ProgressError) as exc_info:
        read_run_state(path)
    assert str(path) in str(exc_info.value)


def test_read_run_state_wrong_schema_version_raises_progress_error(tmp_path: Path):
    path = tmp_path / "run_state.json"
    path.write_text(
        json.dumps({"schema_version": 1, "run_id": "old", "results": {}}), encoding="utf-8"
    )
    with pytest.raises(ProgressError) as exc_info:
        read_run_state(path)
    assert str(path) in str(exc_info.value)


def test_read_run_state_absent_schema_version_raises_progress_error(tmp_path: Path):
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps({"run_id": "old", "results": {}}), encoding="utf-8")
    with pytest.raises(ProgressError) as exc_info:
        read_run_state(path)
    assert str(path) in str(exc_info.value)


def test_read_run_state_logs_before_raising(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    path = tmp_path / "missing.json"
    with caplog.at_level(logging.ERROR), pytest.raises(ProgressError):
        read_run_state(path)
    assert any(str(path) in r.message for r in caplog.records)
