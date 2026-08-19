"""Tests for the Stage 4/resilience state machine (issue #10).

Covers the retry/backoff/dead-letter/resume state machine in
``music_video_maker.resilience`` entirely offline:

* Most tests drive :class:`~music_video_maker.resilience.ResilientRunner`
  against a small in-file ``StubExecutionClient`` double so retry counts,
  backoff values, and dead-letter bookkeeping can be asserted exactly without
  needing a full WS/HTTP script for every scenario.
* A second block of tests wires the *real* ``ComfyUIExecutionClient`` against
  ``tests/harness/comfyui_mock.FakeComfyUISession`` and
  ``tests/harness/ws.ScriptedWebSocket`` to cover the acceptance criteria
  from issue #10 literally: hang -> timeout -> interrupt/free/retry, OOM,
  retry exhaustion -> dead-letter, and resume-from-partial-run.

No real ``time.sleep`` (backoff is injected), no real socket, no GPU, no live
server.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from music_video_maker import resilience as resilience_module
from music_video_maker.config import RunConfig
from music_video_maker.contracts import (
    ChunkFingerprint,
    ChunkResult,
    ChunkStatus,
    HardwareProfile,
    RunState,
    Workflow,
)
from music_video_maker.execution import (
    ComfyUIExecutionClient,
    HistoryError,
    OutputRetrievalError,
    PromptSubmissionError,
    WebSocketTimeoutError,
    WorkflowExecutionError,
)
from music_video_maker.resilience import (
    ChunkIdMismatchError,
    DiskPreflightError,
    ResilienceError,
    ResilientRunner,
    VramBelowFloorError,
)
from tests.harness.comfyui_mock import FakeComfyUISession
from tests.harness.ws import (
    build_disconnect_sequence,
    build_hang_sequence,
    build_oom_sequence,
    build_success_sequence,
    make_ws_factory,
)

WORKFLOW: Workflow = {"1": {"class_type": "UNETLoader", "inputs": {}}}


# --------------------------------------------------------------------------- #
# Stub ExecutionClient -- fine-grained control over per-attempt outcomes
# --------------------------------------------------------------------------- #


@dataclass
class _ExecuteCall:
    workflow: Workflow
    chunk_id: int
    output_dir: Path


class StubExecutionClient:
    """Minimal ``contracts.ExecutionClient`` double.

    ``scripts`` maps chunk_id -> a list of items consumed in order, one per
    ``execute()`` call for that chunk id. Each item is either an exception
    *instance* to raise, or a ``ChunkResult`` to return.
    """

    def __init__(self, scripts: dict[int, list[Any]] | None = None) -> None:
        self.scripts: dict[int, deque[Any]] = {
            cid: deque(items) for cid, items in (scripts or {}).items()
        }
        self.calls: list[_ExecuteCall] = []
        self.interrupt_calls = 0
        self.free_calls: list[dict[str, Any]] = []
        self.ws_timeout: float | None = None

    def execute(self, workflow: Workflow, chunk_id: int, output_dir: Path) -> ChunkResult:
        self.calls.append(_ExecuteCall(workflow, chunk_id, output_dir))
        queue = self.scripts.get(chunk_id)
        if not queue:
            raise AssertionError(
                f"StubExecutionClient: no scripted item left for chunk_id={chunk_id}"
            )
        item = queue.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def free(self, *, unload_models: bool = True) -> None:
        self.free_calls.append({"unload_models": unload_models})


@dataclass
class RecordingSleeper:
    delays: list[float] = field(default_factory=list)

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class ScriptedVramProbe:
    """Fake ``custody.VramProbe`` seam: a zero-arg callable that returns the
    next scripted free-VRAM-in-GB reading (or ``None`` for an unreadable
    check) each time it's called, recording every call for assertions. Standing
    in for ``custody.build_vram_probe(session, base_url)`` so
    ``resilience.py`` tests never need a real/fake HTTP session -- the
    resilience layer only ever sees a plain callable."""

    def __init__(self, readings: list[float | None]) -> None:
        self._readings: deque[float | None] = deque(readings)
        self.calls: int = 0

    def __call__(self) -> float | None:
        self.calls += 1
        if not self._readings:
            raise AssertionError("ScriptedVramProbe: no scripted reading left")
        return self._readings.popleft()


def _rendered(chunk_id: int, tmp_path: Path, *, name: str | None = None) -> ChunkResult:
    video_file = tmp_path / (name or f"chunk_{chunk_id:04d}.mp4")
    video_file.write_bytes(b"fake-mp4-bytes")
    return ChunkResult(
        chunk_id=chunk_id,
        status=ChunkStatus.RENDERED,
        video_file=video_file,
        prompt_id=f"prompt-{chunk_id}",
        attempts=1,
    )


def _make_runner(
    execution_client,
    tmp_path: Path,
    *,
    max_render_attempts: int = 3,
    retry_backoff_seconds: float = 5.0,
    min_free_disk_gb: float = 1.0,
    sleeper=None,
    disk_usage=None,
    run_id: str | None = "test-run",
    ignore_prompt_changes: bool = False,
    vram_probe=None,
    min_free_vram_gb: float | None = None,
    between_chunk_min_free_vram_gb: float | None = None,
) -> ResilientRunner:
    sleeper = sleeper if sleeper is not None else RecordingSleeper()
    disk_usage = disk_usage if disk_usage is not None else _abundant_disk_usage
    kwargs: dict[str, Any] = {}
    if min_free_vram_gb is not None:
        kwargs["min_free_vram_gb"] = min_free_vram_gb
    if between_chunk_min_free_vram_gb is not None:
        kwargs["between_chunk_min_free_vram_gb"] = between_chunk_min_free_vram_gb
    return ResilientRunner(
        execution_client,
        run_state_file=tmp_path / "run_state.json",
        watchdog_timeout_seconds=900.0,
        max_render_attempts=max_render_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        min_free_disk_gb=min_free_disk_gb,
        run_id=run_id,
        sleeper=sleeper,
        disk_usage=disk_usage,
        ignore_prompt_changes=ignore_prompt_changes,
        vram_probe=vram_probe,
        **kwargs,
    )


class _FakeUsage:
    def __init__(self, free_bytes: int) -> None:
        self.free = free_bytes
        self.total = free_bytes * 2
        self.used = free_bytes


def _abundant_disk_usage(_path: str) -> _FakeUsage:
    return _FakeUsage(100 * 1024**3)


def _scarce_disk_usage(_path: str) -> _FakeUsage:
    return _FakeUsage(1024**2)  # ~0.001 GB


def _provider_returning(workflow: Workflow = WORKFLOW):
    calls: list[tuple[int, RunState]] = []

    def provider(chunk_id: int, run_state: RunState) -> Workflow:
        calls.append((chunk_id, run_state))
        return workflow

    provider.calls = calls  # type: ignore[attr-defined]
    return provider


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_render_run_happy_path_single_chunk(tmp_path: Path):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].chunk_id == 1
    assert run_state.results[1].attempts == 1
    assert len(client.calls) == 1
    assert client.calls[0].chunk_id == 1


def test_render_run_calls_provider_freshly_for_each_chunk(tmp_path: Path):
    client = StubExecutionClient(
        {1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]}
    )
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    runner.render_run([1, 2], provider, tmp_path / "chunks")

    assert [cid for cid, _ in provider.calls] == [1, 2]


def test_provider_receives_run_state_with_prior_results(tmp_path: Path):
    """The provider must see chunk 1's final RunState entry when deciding
    chunk 2's workflow, so a dead-lettered predecessor can be reacted to."""
    client = StubExecutionClient(
        {
            1: [HistoryError("no video")] * 3,  # dead-letters (max_render_attempts=3)
            2: [_rendered(2, tmp_path)],
        }
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    seen_predecessor_status = []

    def provider(chunk_id: int, run_state: RunState) -> Workflow:
        if chunk_id == 2:
            prior = run_state.results.get(1)
            seen_predecessor_status.append(prior.status if prior else None)
        return WORKFLOW

    runner.render_run([1, 2], provider, tmp_path / "chunks")

    assert seen_predecessor_status == [ChunkStatus.DEAD_LETTERED]


# --------------------------------------------------------------------------- #
# Provider re-invoked per attempt (not cached up front)
# --------------------------------------------------------------------------- #


def test_provider_is_called_once_per_attempt_not_once_up_front(tmp_path: Path):
    client = StubExecutionClient(
        {1: [WebSocketTimeoutError("hang"), _rendered(1, tmp_path)]}
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    call_count = 0

    def provider(chunk_id: int, run_state: RunState) -> Workflow:
        nonlocal call_count
        call_count += 1
        return WORKFLOW

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert call_count == 2  # once per attempt: failed attempt 1, succeeded attempt 2
    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2


def test_provider_raising_is_treated_as_a_failed_attempt_and_retried(tmp_path: Path, caplog):
    """A provider that raises must feed into the same retry/dead-letter path,
    not kill the run (issue #10 cross-lane contract)."""
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    attempts = []

    def flaky_provider(chunk_id: int, run_state: RunState) -> Workflow:
        attempts.append(1)
        if len(attempts) < 2:
            raise ValueError("provider is briefly broken")
        return WORKFLOW

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], flaky_provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2
    assert len(run_state.results[1].errors) == 1
    assert "provider" in run_state.results[1].errors[0].lower()


def test_provider_always_raising_dead_letters_without_killing_the_run(tmp_path: Path):
    client = StubExecutionClient({2: [_rendered(2, tmp_path)]})
    runner = _make_runner(client, tmp_path, max_render_attempts=2)

    def provider(chunk_id: int, run_state: RunState) -> Workflow:
        if chunk_id == 1:
            raise RuntimeError("always broken")
        return WORKFLOW

    run_state = runner.render_run([1, 2], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.DEAD_LETTERED
    assert run_state.results[1].attempts == 2
    assert run_state.results[2].status is ChunkStatus.RENDERED


# --------------------------------------------------------------------------- #
# Retry / recovery sequence / backoff
# --------------------------------------------------------------------------- #


def test_transient_failure_then_success_retries_with_recovery_and_backoff(tmp_path: Path):
    client = StubExecutionClient(
        {1: [WebSocketTimeoutError("watchdog fired"), _rendered(1, tmp_path)]}
    )
    sleeper = RecordingSleeper()
    runner = _make_runner(
        client, tmp_path, max_render_attempts=3, retry_backoff_seconds=5.0, sleeper=sleeper
    )
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2
    assert client.interrupt_calls == 1
    assert len(client.free_calls) == 1
    assert sleeper.delays == [5.0]


def test_backoff_grows_across_repeated_transient_failures(tmp_path: Path):
    client = StubExecutionClient(
        {
            1: [
                WebSocketTimeoutError("1"),
                WebSocketTimeoutError("2"),
                _rendered(1, tmp_path),
            ]
        }
    )
    sleeper = RecordingSleeper()
    runner = _make_runner(
        client, tmp_path, max_render_attempts=4, retry_backoff_seconds=5.0, sleeper=sleeper
    )
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert sleeper.delays == [5.0, 10.0]


def test_real_time_sleep_is_never_called(tmp_path: Path, monkeypatch):
    import time as time_module

    def _boom(*_a, **_k):
        raise AssertionError("resilience.py must never call the real time.sleep()")

    monkeypatch.setattr(time_module, "sleep", _boom)

    client = StubExecutionClient(
        {1: [WebSocketTimeoutError("hang"), _rendered(1, tmp_path)]}
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")
    assert run_state.results[1].status is ChunkStatus.RENDERED


# --------------------------------------------------------------------------- #
# Dead-letter after exhausting retries -- run must continue
# --------------------------------------------------------------------------- #


def test_retry_exhaustion_dead_letters_and_run_continues(tmp_path: Path, caplog):
    client = StubExecutionClient(
        {
            1: [HistoryError("no video") for _ in range(3)],
            2: [_rendered(2, tmp_path)],
        }
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1, 2], provider, tmp_path / "chunks")

    result1 = run_state.results[1]
    assert result1.status is ChunkStatus.DEAD_LETTERED
    assert result1.attempts == 3
    assert len(result1.errors) == 3
    assert result1.video_file is None

    result2 = run_state.results[2]
    assert result2.status is ChunkStatus.RENDERED

    assert 1 in run_state.dead_lettered
    assert any("dead" in r.message.lower() for r in caplog.records)


def test_oom_exhaustion_dead_letters(tmp_path: Path):
    client = StubExecutionClient(
        {
            1: [
                WorkflowExecutionError(
                    "oom", node_id="7", node_type="Sampler",
                    exception_type="torch.cuda.OutOfMemoryError", exception_message="oom",
                )
                for _ in range(2)
            ],
        }
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=2)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.DEAD_LETTERED
    assert run_state.results[1].attempts == 2
    assert client.interrupt_calls == 2
    assert len(client.free_calls) == 2


# --------------------------------------------------------------------------- #
# Non-retryable: PromptSubmissionError with node_errors -- fail fast
# --------------------------------------------------------------------------- #


def test_node_errors_prompt_submission_dead_letters_immediately_without_retry(
    tmp_path: Path, caplog
):
    client = StubExecutionClient(
        {1: [PromptSubmissionError("bad graph", node_errors={"7": {"errors": ["bad"]}})]}
    )
    sleeper = RecordingSleeper()
    runner = _make_runner(client, tmp_path, max_render_attempts=5, sleeper=sleeper)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], provider, tmp_path / "chunks")

    result = run_state.results[1]
    assert result.status is ChunkStatus.DEAD_LETTERED
    assert result.attempts == 1  # did not burn through max_render_attempts
    assert sleeper.delays == []  # no backoff -- fail fast
    assert client.interrupt_calls == 0  # no recovery sequence for a fatal graph error
    assert client.free_calls == []


def test_prompt_submission_error_without_node_errors_is_retryable(tmp_path: Path):
    """A bare submission failure (e.g. transient network hiccup on /prompt)
    carries no node_errors and should still be retried, unlike the graph
    validation failure case above."""
    client = StubExecutionClient(
        {
            1: [
                PromptSubmissionError("POST /prompt request failed: connection reset"),
                _rendered(1, tmp_path),
            ]
        }
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2


# --------------------------------------------------------------------------- #
# Non-retryable: disk-full -- fail fast
# --------------------------------------------------------------------------- #


def test_disk_full_signature_dead_letters_immediately_without_retry(tmp_path: Path, caplog):
    exc = OutputRetrievalError(
        "Failed to write video file /out/chunk_0001.mp4: [Errno 28] No space left on device"
    )
    client = StubExecutionClient({1: [exc]})
    sleeper = RecordingSleeper()
    runner = _make_runner(client, tmp_path, max_render_attempts=5, sleeper=sleeper)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], provider, tmp_path / "chunks")

    result = run_state.results[1]
    assert result.status is ChunkStatus.DEAD_LETTERED
    assert result.attempts == 1
    assert sleeper.delays == []
    assert client.interrupt_calls == 0
    assert any("disk" in r.message.lower() or "space" in r.message.lower() for r in caplog.records)


def test_generic_output_retrieval_error_without_disk_signature_is_retryable(tmp_path: Path):
    client = StubExecutionClient(
        {1: [OutputRetrievalError("GET /view returned status 500 for {}"), _rendered(1, tmp_path)]}
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2


# --------------------------------------------------------------------------- #
# Watchdog wiring
# --------------------------------------------------------------------------- #


def test_constructor_configures_execution_client_watchdog_timeout(tmp_path: Path):
    client = StubExecutionClient()
    runner = _make_runner(client, tmp_path)
    assert client.ws_timeout == runner.watchdog_timeout_seconds == 900.0


def test_watchdog_timeout_logged_distinctly(tmp_path: Path, caplog):
    client = StubExecutionClient({1: [WebSocketTimeoutError("timed out"), _rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        runner.render_run([1], provider, tmp_path / "chunks")

    assert any("watchdog" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Pre-flight disk check
# --------------------------------------------------------------------------- #


def test_preflight_disk_check_refuses_to_start_when_space_is_scarce(tmp_path: Path):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path, min_free_disk_gb=10.0, disk_usage=_scarce_disk_usage)
    provider = _provider_returning()

    with pytest.raises(DiskPreflightError):
        runner.render_run([1], provider, tmp_path / "chunks")

    assert len(provider.calls) == 0
    assert len(client.calls) == 0


def test_preflight_disk_check_passes_with_sufficient_space(tmp_path: Path):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path, min_free_disk_gb=1.0, disk_usage=_abundant_disk_usage)
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")
    assert run_state.results[1].status is ChunkStatus.RENDERED


def test_default_disk_usage_seam_is_shutil_disk_usage(tmp_path: Path):
    import shutil

    client = StubExecutionClient()
    runner = ResilientRunner(
        client,
        run_state_file=tmp_path / "run_state.json",
    )
    assert runner.disk_usage is shutil.disk_usage


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_resume_reuses_previously_rendered_chunk_as_cached(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)], 2: [_rendered(2, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    provider = _provider_returning()
    runner1.render_run([1, 2], provider, output_dir)

    # Fresh runner/client instance pointed at the same run_state_file, as a
    # second process resuming would be.
    client2 = StubExecutionClient({})  # no scripts -- must not be called for cached chunks
    runner2 = _make_runner(client2, tmp_path)
    provider2 = _provider_returning()

    run_state = runner2.render_run([1, 2], provider2, output_dir, resume=True)

    assert run_state.results[1].status is ChunkStatus.CACHED
    assert run_state.results[2].status is ChunkStatus.CACHED
    assert len(client2.calls) == 0
    assert len(provider2.calls) == 0


def test_resume_re_renders_chunk_whose_video_file_was_deleted(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    provider = _provider_returning()
    run_state1 = runner1.render_run([1], provider, output_dir)

    # Simulate the video going missing between runs.
    run_state1.results[1].video_file.unlink()

    # Different filename so building this stub doesn't eagerly recreate the
    # file we just deleted (the _rendered() helper writes bytes immediately).
    client2 = StubExecutionClient(
        {1: [_rendered(1, output_dir, name="chunk_0001_rerendered.mp4")]}
    )
    runner2 = _make_runner(client2, tmp_path)
    provider2 = _provider_returning()

    resumed_state = runner2.render_run([1], provider2, output_dir, resume=True)

    assert resumed_state.results[1].status is ChunkStatus.RENDERED
    assert len(client2.calls) == 1  # re-rendered, not silently skipped
    assert len(provider2.calls) == 1


def test_resume_missing_video_file_logs_a_warning(tmp_path: Path, caplog):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    run_state1 = runner1.render_run([1], _provider_returning(), output_dir)
    run_state1.results[1].video_file.unlink()

    client2 = StubExecutionClient(
        {1: [_rendered(1, output_dir, name="chunk_0001_rerendered.mp4")]}
    )
    runner2 = _make_runner(client2, tmp_path)

    with caplog.at_level(logging.WARNING):
        runner2.render_run([1], _provider_returning(), output_dir, resume=True)

    assert any("missing" in r.message.lower() for r in caplog.records)


def test_resume_dead_lettered_chunk_is_retried_not_skipped(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [HistoryError("no video")] * 3})
    runner1 = _make_runner(client1, tmp_path, max_render_attempts=3)
    run_state1 = runner1.render_run([1], _provider_returning(), output_dir)
    assert run_state1.results[1].status is ChunkStatus.DEAD_LETTERED

    client2 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner2 = _make_runner(client2, tmp_path, max_render_attempts=3)
    run_state2 = runner2.render_run([1], _provider_returning(), output_dir, resume=True)

    assert run_state2.results[1].status is ChunkStatus.RENDERED
    assert len(client2.calls) == 1


def test_resume_without_state_file_starts_fresh_and_warns(tmp_path: Path, caplog):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run([1], provider, tmp_path / "chunks", resume=True)

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert any(
        "no run state file" in r.message.lower() or "starting a fresh run" in r.message.lower()
        for r in caplog.records
    )


def test_resume_false_ignores_existing_state_file(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    runner1.render_run([1], _provider_returning(), output_dir)

    client2 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner2 = _make_runner(client2, tmp_path)
    provider2 = _provider_returning()

    run_state = runner2.render_run([1], provider2, output_dir, resume=False)

    assert len(client2.calls) == 1  # re-rendered, resume=False means fresh run
    assert run_state.results[1].status is ChunkStatus.RENDERED


# --------------------------------------------------------------------------- #
# Resume: chunk identity (issue #34)
#
# "the mp4 exists" is not "the mp4 is the right mp4". Every test below starts
# from a completed run and resumes with *something changed*, asserting the
# runner re-renders rather than assembling a silently desynced video.
# --------------------------------------------------------------------------- #


def _fp(
    start: float = 0.0,
    end: float = 6.0,
    *,
    frames: int | None = 141,
    width: int | None = 864,
    height: int | None = 480,
    prompt: str = "a composed prompt",
    character: str | None = "Dianne",
    image: str = "/cast/dianne.png",
    seed: int | None = 0,
) -> ChunkFingerprint:
    return ChunkFingerprint(
        start=start,
        end=end,
        frame_count=frames,
        render_width=width,
        render_height=height,
        prompt_hash=ChunkFingerprint.hash_prompt(prompt),
        character=character,
        image_ref=image,
        noise_seed=seed,
    )


def _complete_a_run(
    tmp_path: Path,
    output_dir: Path,
    fingerprints: dict[int, ChunkFingerprint],
) -> RunState:
    """Render every chunk in ``fingerprints`` once, leaving a state file."""
    chunk_ids = sorted(fingerprints)
    client = StubExecutionClient({cid: [_rendered(cid, output_dir)] for cid in chunk_ids})
    runner = _make_runner(client, tmp_path)
    return runner.render_run(
        chunk_ids, _provider_returning(), output_dir, fingerprints=fingerprints
    )


def test_resume_reuses_chunks_whose_fingerprint_is_unchanged(tmp_path: Path):
    """The fast path must stay fast: nothing changed, nothing re-renders."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    fingerprints = {0: _fp(0.0, 6.0), 1: _fp(6.0, 12.0)}
    _complete_a_run(tmp_path, output_dir, fingerprints)

    client = StubExecutionClient({})  # no scripts -- any render attempt fails loudly
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    run_state = runner.render_run(
        [0, 1], provider, output_dir, resume=True, fingerprints=fingerprints
    )

    assert [r.status for r in run_state.results.values()] == [ChunkStatus.CACHED] * 2
    assert client.calls == []
    assert provider.calls == []


def test_resume_re_renders_a_chunk_whose_span_moved(tmp_path: Path):
    """A lyrics edit moves chunk boundaries. The old mp4 exists and is valid --
    it just belongs to a different moment of the song."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(0.0, 6.0), 1: _fp(6.0, 12.0)})

    moved = {0: _fp(0.0, 6.0), 1: _fp(7.5, 13.5)}
    client = StubExecutionClient({1: [_rendered(1, output_dir, name="chunk_0001_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0, 1], _provider_returning(), output_dir, resume=True, fingerprints=moved
    )

    assert run_state.results[0].status is ChunkStatus.CACHED
    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert [c.chunk_id for c in client.calls] == [1]


def test_resume_re_renders_a_chunk_whose_frame_count_changed(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(0.0, 6.0, frames=141)})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(0.0, 6.0, frames=158)},
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


def test_resume_re_renders_when_the_render_resolution_changed(tmp_path: Path):
    """Not a desync but a corruption: Stage 5's concat demuxer runs ``-c:v
    copy`` and cannot join clips of differing dimensions."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(width=864, height=480)})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_big.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(width=1344, height=768)},
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert len(client.calls) == 1


def test_resume_re_renders_a_chunk_whose_prompt_changed(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(prompt="wide shot of the ward")})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(prompt="close up on the monitor")},
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


def test_resume_re_renders_a_chunk_whose_character_was_recast(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(character="Dianne", image="/cast/d.png")})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(character="Rex", image="/cast/r.png")},
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


def test_ignore_prompt_changes_reuses_a_chunk_whose_prompt_changed(tmp_path: Path, caplog):
    """Re-rendering 39 chunks to pick up a typo fix in one shot line is
    expensive, so the content tier is escapable -- loudly."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(prompt="wide shot")})

    client = StubExecutionClient({})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    with caplog.at_level(logging.INFO):
        run_state = runner.render_run(
            [0],
            _provider_returning(),
            output_dir,
            resume=True,
            fingerprints={0: _fp(prompt="close up")},
        )

    assert run_state.results[0].status is ChunkStatus.CACHED
    assert client.calls == []
    assert any("prompt_hash" in r.message for r in caplog.records)


def test_ignore_prompt_changes_does_not_excuse_a_moved_span(tmp_path: Path):
    """The escape hatch covers the content tier only -- a desync is never
    something the user can opt into by flag."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(0.0, 6.0, prompt="wide shot")})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(2.0, 8.0, prompt="close up")},
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


def test_reused_chunk_keeps_the_fingerprint_it_was_actually_rendered_for(tmp_path: Path):
    """Reusing under --ignore-prompt-changes must not re-stamp the cached chunk
    with the new prompt -- a later run without the flag would then believe the
    video matches a prompt it was never rendered from."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    original = _fp(prompt="wide shot")
    _complete_a_run(tmp_path, output_dir, {0: original})

    runner = _make_runner(StubExecutionClient({}), tmp_path, ignore_prompt_changes=True)
    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        resume=True,
        fingerprints={0: _fp(prompt="close up")},
    )

    assert run_state.results[0].fingerprint == original


def test_resume_re_renders_a_chunk_with_no_recorded_fingerprint(tmp_path: Path, caplog):
    """A result carrying no fingerprint cannot prove what it was rendered for,
    so it is re-rendered rather than trusted."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp()})

    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    payload["results"]["0"].pop("fingerprint")
    state_file.write_text(json.dumps(payload))

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp()}
        )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert any("fingerprint" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# The noise seed a chunk was rendered with (issue #38, recording half).
#
# Content tier: a re-seeded chunk still covers its own span, so it is never a
# desync -- just a different take of the same moment, and therefore escapable
# through the same flag a prompt edit is.
# --------------------------------------------------------------------------- #


def test_state_file_records_the_seed_each_chunk_was_rendered_with(tmp_path: Path):
    """'It happens to be 0 today; nothing in run_state.json says so.' Now it
    does -- a good take is reproducible only if the run wrote its seed down."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(seed=4242)})

    payload = json.loads((tmp_path / "run_state.json").read_text())

    assert payload["results"]["0"]["fingerprint"]["noise_seed"] == 4242


def test_resume_re_renders_a_chunk_whose_seed_changed(tmp_path: Path, caplog):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(seed=0)})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp(seed=99)}
        )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "noise_seed" in messages
    assert "99" in messages


def test_a_changed_seed_is_never_read_as_a_moved_timeline(tmp_path: Path):
    """The escape hatch has to work, so the seed must not leak into the
    inescapable tier: re-rolling every chunk's seed is a re-render by choice,
    not a desynced video."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(seed=0)})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp(seed=99)}
    )

    assert run_state.results[0].status is ChunkStatus.CACHED
    # Still recording the seed the video was really rendered with, so a later
    # run without the flag does not believe it matches.
    assert run_state.results[0].fingerprint.noise_seed == 0


def test_a_fingerprint_written_before_seeds_were_recorded_re_renders(tmp_path: Path, caplog):
    """The compat case, decided strictly: a state file with fingerprints but
    no ``noise_seed`` is *readable* (the schema version does not move -- every
    other field in it is still provable), but a chunk in it cannot prove which
    seed produced its pixels, so it is re-rendered rather than assumed to be
    the 0 the template used to carry."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    before = {0: _fp(0.0, 6.0, seed=0), 1: _fp(6.0, 12.0, seed=0)}
    _complete_a_run(tmp_path, output_dir, before)

    # Chunk 0 only: exactly what a file written before the field existed looks
    # like for that chunk, with chunk 1 left current so the *file* is still
    # provably readable rather than rejected wholesale.
    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    payload["results"]["0"]["fingerprint"].pop("noise_seed")
    state_file.write_text(json.dumps(payload))

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            [0, 1], _provider_returning(), output_dir, resume=True, fingerprints=before
        )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert run_state.results[1].status is ChunkStatus.CACHED  # the file was read, not rejected
    assert "noise_seed" in " ".join(r.getMessage() for r in caplog.records)


def test_a_pre_seed_fingerprint_is_reusable_for_a_run_that_manages_no_seed(tmp_path: Path):
    """The other half of the same decision: a caller that records no seed this
    run compares None against None and keeps its cached chunks. Old state files
    are not invalidated by the *existence* of the field, only by a run that
    actually claims a seed."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(seed=None)})

    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    payload["results"]["0"]["fingerprint"].pop("noise_seed")
    state_file.write_text(json.dumps(payload))

    runner = _make_runner(StubExecutionClient({}), tmp_path)
    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp(seed=None)}
    )

    assert run_state.results[0].status is ChunkStatus.CACHED


def test_mismatch_log_names_the_field_that_changed(tmp_path: Path, caplog):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(0.0, 6.0)})

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        runner.render_run(
            [0],
            _provider_returning(),
            output_dir,
            resume=True,
            fingerprints={0: _fp(3.25, 9.25)},
        )

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "start" in messages
    assert "3.25" in messages  # the new value
    assert "0.0" in messages  # the value it was rendered for


def test_a_whole_timeline_change_is_reported_once_not_once_per_chunk(tmp_path: Path, caplog):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    before = {cid: _fp(cid * 6.0, cid * 6.0 + 6.0) for cid in range(6)}
    _complete_a_run(tmp_path, output_dir, before)

    # Everything after chunk 0 slides by 1.5s -- the shape of a lyrics edit.
    after = {0: before[0], **{cid: _fp(cid * 6.0 + 1.5, cid * 6.0 + 7.5) for cid in range(1, 6)}}
    client = StubExecutionClient(
        {cid: [_rendered(cid, output_dir, name=f"chunk_{cid:04d}_new.mp4")] for cid in range(1, 6)}
    )
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            list(range(6)), _provider_returning(), output_dir, resume=True, fingerprints=after
        )

    assert [c.chunk_id for c in client.calls] == [1, 2, 3, 4, 5]
    assert run_state.results[0].status is ChunkStatus.CACHED
    # One summary line about the timeline, not five near-identical warnings.
    summaries = [r for r in caplog.records if "TIMELINE HAS CHANGED" in r.getMessage()]
    assert len(summaries) == 1
    assert "5 of 6" in summaries[0].getMessage()  # how many chunks moved
    assert not [r for r in caplog.records if "re-rendering" in r.getMessage()]


def test_a_shorter_timeline_reports_the_chunks_it_no_longer_renders(tmp_path: Path, caplog):
    """Prior state describing chunks this run does not produce means the
    timeline shrank -- those mp4s are now orphans and must not be assembled."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    before = {cid: _fp(cid * 6.0, cid * 6.0 + 6.0) for cid in range(4)}
    _complete_a_run(tmp_path, output_dir, before)

    runner = _make_runner(StubExecutionClient({}), tmp_path)
    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            [0, 1],
            _provider_returning(),
            output_dir,
            resume=True,
            fingerprints={0: before[0], 1: before[1]},
        )

    assert run_state.results[0].status is ChunkStatus.CACHED
    assert any("[2, 3]" in r.getMessage() for r in caplog.records)


def test_a_state_file_whose_fingerprints_were_stripped_re_renders_everything(
    tmp_path: Path, caplog
):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    before = {cid: _fp(cid * 6.0, cid * 6.0 + 6.0) for cid in range(4)}
    _complete_a_run(tmp_path, output_dir, before)

    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    for result in payload["results"].values():
        result["fingerprint"] = None
    state_file.write_text(json.dumps(payload))

    client = StubExecutionClient(
        {cid: [_rendered(cid, output_dir, name=f"chunk_{cid:04d}_new.mp4")] for cid in range(4)}
    )
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run(
            list(range(4)), _provider_returning(), output_dir, resume=True, fingerprints=before
        )

    assert [c.chunk_id for c in client.calls] == [0, 1, 2, 3]
    assert all(r.status is ChunkStatus.RENDERED for r in run_state.results.values())
    assert any("no fingerprint recorded" in r.getMessage() for r in caplog.records)


def test_resume_without_fingerprints_warns_that_identity_is_unverified(tmp_path: Path, caplog):
    """Backwards-compatible, but never silently: a caller that supplies no
    fingerprints gets the old existence-only behaviour and is told so."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp()})

    runner = _make_runner(StubExecutionClient({}), tmp_path)
    with caplog.at_level(logging.WARNING):
        run_state = runner.render_run([0], _provider_returning(), output_dir, resume=True)

    assert run_state.results[0].status is ChunkStatus.CACHED
    assert any("cannot be verified" in r.getMessage().lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Run state schema versioning (issue #34)
# --------------------------------------------------------------------------- #


def test_state_file_records_the_schema_version_and_fingerprints(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp(1.5, 7.5)})

    payload = json.loads((tmp_path / "run_state.json").read_text())
    assert payload["schema_version"] == resilience_module.RUN_STATE_SCHEMA_VERSION
    assert payload["results"]["0"]["fingerprint"]["start"] == 1.5
    assert payload["results"]["0"]["fingerprint"]["end"] == 7.5


def test_fingerprint_round_trips_through_the_state_file(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    fingerprints = {0: _fp(1.5, 7.5, frames=158, width=1344, height=768)}
    _complete_a_run(tmp_path, output_dir, fingerprints)

    runner = _make_runner(StubExecutionClient({}), tmp_path)
    reloaded = runner._read_run_state(tmp_path / "run_state.json")  # noqa: SLF001

    assert reloaded.results[0].fingerprint == fingerprints[0]


def test_state_file_from_an_older_schema_is_rejected_and_the_run_starts_fresh(
    tmp_path: Path, caplog
):
    """A pre-fingerprint state file describes chunks whose spans are unknowable
    -- it must be rejected, not misread as 'everything matches'."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    video = output_dir / "chunk_0000.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    (tmp_path / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "old-run",
                "results": {
                    "0": {
                        "chunk_id": 0,
                        "status": "rendered",
                        "video_file": str(video),
                        "prompt_id": "p",
                        "attempts": 1,
                        "errors": [],
                        "render_seconds": None,
                    }
                },
            }
        )
    )

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run(
            [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp()}
        )

    assert run_state.run_id != "old-run"
    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert any("schema" in r.getMessage().lower() for r in caplog.records)


def test_state_file_from_a_newer_schema_is_rejected(tmp_path: Path, caplog):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    _complete_a_run(tmp_path, output_dir, {0: _fp()})

    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    payload["schema_version"] = resilience_module.RUN_STATE_SCHEMA_VERSION + 1
    state_file.write_text(json.dumps(payload))

    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run(
            [0], _provider_returning(), output_dir, resume=True, fingerprints={0: _fp()}
        )

    assert run_state.results[0].status is ChunkStatus.RENDERED


# --------------------------------------------------------------------------- #
# Atomic persistence
# --------------------------------------------------------------------------- #


def test_run_state_persisted_atomically_after_each_chunk(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client = StubExecutionClient({1: [_rendered(1, output_dir)], 2: [_rendered(2, output_dir)]})
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    run_state = runner.render_run([1, 2], provider, output_dir)

    state_file = tmp_path / "run_state.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["run_id"] == run_state.run_id
    assert set(payload["results"].keys()) == {"1", "2"}
    # No stray temp files left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name not in ("chunks", state_file.name)]
    assert leftovers == []


def test_run_state_write_goes_through_os_replace(tmp_path: Path, monkeypatch):
    """Pin the *mechanism*, not just the absence of temp files.

    A plain ``path.write_text`` leaves no leftovers either, so the assertion
    above passes with or without atomicity. A crash midway through a direct
    write truncates the state file and loses the whole run's resume point --
    so assert the temp-file + ``os.replace`` rename actually happens.
    """
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    state_file = tmp_path / "run_state.json"

    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(resilience_module.os, "replace", spy_replace)

    client = StubExecutionClient({1: [_rendered(1, output_dir)], 2: [_rendered(2, output_dir)]})
    runner = _make_runner(client, tmp_path)

    runner.render_run([1, 2], _provider_returning(), output_dir)

    # One atomic rename per chunk, each landing on the real state file.
    assert len(replaced) == 2
    for src, dst in replaced:
        assert dst == str(state_file)
        assert src != dst
    assert json.loads(state_file.read_text())["results"]["2"]["chunk_id"] == 2


def test_persisted_state_round_trips_through_resume(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    runner1.render_run([1], _provider_returning(), output_dir)

    state_file = tmp_path / "run_state.json"
    payload = json.loads(state_file.read_text())
    assert payload["results"]["1"]["status"] == "rendered"
    assert payload["results"]["1"]["chunk_id"] == 1


# --------------------------------------------------------------------------- #
# Index-space guard
# --------------------------------------------------------------------------- #


def test_chunk_id_mismatch_error_names_both_values():
    from music_video_maker.resilience import _assert_chunk_id  # noqa: SLF001

    with pytest.raises(ChunkIdMismatchError) as exc_info:
        _assert_chunk_id(5, 6, context="test")
    message = str(exc_info.value)
    assert "5" in message
    assert "6" in message
    assert exc_info.value.expected == 5
    assert exc_info.value.actual == 6


def test_render_run_raises_when_execution_client_returns_wrong_chunk_id(tmp_path: Path):
    wrong = ChunkResult(chunk_id=999, status=ChunkStatus.RENDERED, video_file=tmp_path / "x.mp4")
    (tmp_path / "x.mp4").write_bytes(b"x")
    client = StubExecutionClient({1: [wrong]})
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    with pytest.raises(ChunkIdMismatchError) as exc_info:
        runner.render_run([1], provider, tmp_path / "chunks")

    assert exc_info.value.expected == 1
    assert exc_info.value.actual == 999


def test_corrupted_run_state_file_index_mismatch_logs_and_falls_back_to_fresh_run(
    tmp_path: Path, caplog
):
    state_file = tmp_path / "run_state.json"
    corrupted = {
        "schema_version": resilience_module.RUN_STATE_SCHEMA_VERSION,
        "run_id": "old-run",
        "results": {
            "5": {
                "chunk_id": 6,  # deliberately disagrees with the dict key "5"
                "status": "rendered",
                "video_file": None,
                "prompt_id": None,
                "attempts": 1,
                "errors": [],
                "render_seconds": None,
            }
        },
    }
    state_file.write_text(json.dumps(corrupted))

    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], provider, tmp_path / "chunks", resume=True)

    # Falls back to a fresh run rather than trusting the corrupted mapping.
    assert run_state.run_id != "old-run"
    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert any("index-space guard" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Robustness of the recovery sequence itself
# --------------------------------------------------------------------------- #


class _RecoveryRaisingExecutionClient(StubExecutionClient):
    """interrupt()/free() raising must not crash the run -- they're recovery
    best-effort actions, and the real ComfyUIExecutionClient already
    swallows its own transport errors, but this module must not assume that
    of every ExecutionClient implementation."""

    def interrupt(self) -> None:
        self.interrupt_calls += 1
        raise RuntimeError("interrupt endpoint unreachable")

    def free(self, *, unload_models: bool = True) -> None:
        self.free_calls.append({"unload_models": unload_models})
        raise RuntimeError("free endpoint unreachable")


def test_recovery_sequence_exceptions_are_logged_and_do_not_crash_the_run(tmp_path: Path, caplog):
    client = _RecoveryRaisingExecutionClient(
        {1: [WebSocketTimeoutError("hang"), _rendered(1, tmp_path)]}
    )
    runner = _make_runner(client, tmp_path, max_render_attempts=3)
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert client.interrupt_calls == 1
    assert len(client.free_calls) == 1
    assert any("interrupt()" in r.message for r in caplog.records)
    assert any("free()" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Disk stat failure during pre-flight
# --------------------------------------------------------------------------- #


def _raising_disk_usage(_path: str):
    raise OSError("statvfs failed")


def test_preflight_disk_check_stat_failure_raises_disk_preflight_error(tmp_path: Path):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path, disk_usage=_raising_disk_usage)
    provider = _provider_returning()

    with pytest.raises(DiskPreflightError):
        runner.render_run([1], provider, tmp_path / "chunks")

    assert len(provider.calls) == 0


# --------------------------------------------------------------------------- #
# Persist failure degrades gracefully rather than crashing the run
# --------------------------------------------------------------------------- #


def test_persist_failure_is_logged_and_does_not_crash_the_run(tmp_path: Path, caplog):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path)
    # Point run_state_file at a path whose parent can never be created (a
    # file standing where a directory needs to go).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    runner.run_state_file = blocker / "run_state.json"
    provider = _provider_returning()

    with caplog.at_level(logging.ERROR):
        run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert any("failed to persist run state" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Between-chunk VRAM re-check (issue #23)
#
# The pre-flight VRAM assertion (custody.py) runs once, before the run
# starts. It cannot see another process claiming the card afterwards -- which
# is exactly how the 2026-08-07 outage happened: H3 staged its ~20GB onto a
# contended card and went silent rather than raising CUDA OOM, wedging the
# host badly enough to need a power cycle. These tests cover the mid-run
# re-check that converts that into "run stopped after chunk N, resume when
# the card is free" instead.
# --------------------------------------------------------------------------- #


def test_no_vram_probe_keeps_current_behaviour(tmp_path: Path):
    """Default (no seam injected) must not change existing behaviour at all --
    every prior test in this file relies on that."""
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    runner = _make_runner(client, tmp_path)  # vram_probe defaults to None
    provider = _provider_returning()

    run_state = runner.render_run([1, 2], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[2].status is ChunkStatus.RENDERED


def test_healthy_vram_readings_render_every_chunk(tmp_path: Path):
    client = StubExecutionClient(
        {1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)], 3: [_rendered(3, tmp_path)]}
    )
    probe = ScriptedVramProbe([20.0, 19.0, 18.0])
    runner = _make_runner(client, tmp_path, vram_probe=probe, min_free_vram_gb=16.0)
    provider = _provider_returning()

    run_state = runner.render_run([1, 2, 3], provider, tmp_path / "chunks")

    assert [run_state.results[i].status for i in (1, 2, 3)] == [ChunkStatus.RENDERED] * 3
    assert probe.calls == 3


def test_a_low_between_chunk_reading_does_not_abort_by_default(tmp_path: Path):
    """Measured on doris 2026-08-08, mid-render, ComfyUI holding H3:

        Pre-chunk VRAM check for chunk 0: 20.72 GB free (need >= 16.00 GB)
        Chunk 0 rendered on attempt 1/3
        Pre-chunk VRAM check for chunk 1:  1.54 GB free (need >= 16.00 GB)

    Free VRAM between chunks is 1.5-5 GB *by design* -- ComfyUI keeps H3's
    19995 MB staged rather than unloading between prompts ("0 models
    unloaded" in its own log). Gating on the pre-flight floor therefore
    aborted every real run after its first chunk. The floor is a cold-card
    number and means nothing once the model is resident, so the between-chunk
    check reports by default and only gates when someone opts in with a
    number they measured on their own card."""
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    probe = ScriptedVramProbe([20.72, 1.54])
    runner = _make_runner(client, tmp_path, vram_probe=probe)  # no gate configured

    run_state = runner.render_run([1, 2], _provider_returning(), tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[2].status is ChunkStatus.RENDERED
    assert probe.calls == 2


def test_the_preflight_floor_is_never_reused_as_the_between_chunk_gate(tmp_path: Path):
    """``min_free_vram_gb`` is the custody pre-flight's cold-card floor. It
    must not leak into the between-chunk check as a default -- that equation
    is exactly what broke the first real run."""
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    probe = ScriptedVramProbe([20.72, 1.54])
    runner = _make_runner(client, tmp_path, vram_probe=probe, min_free_vram_gb=16.0)

    run_state = runner.render_run([1, 2], _provider_returning(), tmp_path / "chunks")

    assert run_state.results[2].status is ChunkStatus.RENDERED


def test_vram_below_floor_stops_the_run_before_submitting_the_next_chunk(tmp_path: Path):
    client = StubExecutionClient(
        {1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)], 3: [_rendered(3, tmp_path)]}
    )
    # Chunks 1 and 2 read healthy; the reading taken before chunk 3 has
    # dropped below the floor -- another process took the card mid-run.
    probe = ScriptedVramProbe([20.0, 19.0, 8.0])
    runner = _make_runner(
        client, tmp_path, vram_probe=probe, between_chunk_min_free_vram_gb=16.0
    )
    provider = _provider_returning()

    with pytest.raises(VramBelowFloorError, match="8.00 GB") as excinfo:
        runner.render_run([1, 2, 3], provider, tmp_path / "chunks")

    assert "16.00 GB" in str(excinfo.value)
    assert "resume" in str(excinfo.value).lower()

    # Chunk 3 was never submitted to the execution client.
    assert [c.chunk_id for c in client.calls] == [1, 2]


def test_vram_below_floor_persists_prior_chunks_so_resume_picks_up(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client = StubExecutionClient(
        {
            1: [_rendered(1, output_dir)],
            2: [_rendered(2, output_dir)],
            3: [_rendered(3, output_dir)],
        }
    )
    probe = ScriptedVramProbe([20.0, 19.0, 8.0])
    runner = _make_runner(
        client, tmp_path, vram_probe=probe, between_chunk_min_free_vram_gb=16.0
    )

    with pytest.raises(VramBelowFloorError):
        runner.render_run([1, 2, 3], _provider_returning(), output_dir)

    state_file = tmp_path / "run_state.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    # The state on disk must be exactly the two completed chunks -- not a
    # partial/fabricated entry for chunk 3, which never rendered.
    assert set(payload["results"].keys()) == {"1", "2"}
    assert payload["results"]["1"]["status"] == "rendered"
    assert payload["results"]["2"]["status"] == "rendered"

    # A resumed run picks up exactly where it stopped: 1 and 2 reused,
    # 3 rendered fresh once the card is free again.
    client2 = StubExecutionClient({3: [_rendered(3, output_dir, name="chunk_0003.mp4")]})
    probe2 = ScriptedVramProbe([20.0])
    runner2 = _make_runner(client2, tmp_path, vram_probe=probe2, min_free_vram_gb=16.0)
    resumed_state = runner2.render_run(
        [1, 2, 3], _provider_returning(), output_dir, resume=True
    )

    assert resumed_state.results[1].status is ChunkStatus.CACHED
    assert resumed_state.results[2].status is ChunkStatus.CACHED
    assert resumed_state.results[3].status is ChunkStatus.RENDERED
    assert [c.chunk_id for c in client2.calls] == [3]


def test_vram_unreadable_logs_and_continues(tmp_path: Path, caplog):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    probe = ScriptedVramProbe([None, 20.0])
    runner = _make_runner(client, tmp_path, vram_probe=probe, min_free_vram_gb=16.0)
    provider = _provider_returning()

    with caplog.at_level(logging.INFO):
        run_state = runner.render_run([1, 2], provider, tmp_path / "chunks")

    # A best-effort check must never block a run over its own flakiness.
    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[2].status is ChunkStatus.RENDERED
    assert probe.calls == 2
    assert any(
        "unavailable" in r.message.lower() or "unreadable" in r.message.lower()
        for r in caplog.records
    )


def test_aborting_on_the_very_first_chunk_still_leaves_a_readable_state_file(tmp_path: Path):
    """The loop persists after every *completed* chunk, so on chunk 2 onward
    the resume point already exists on disk. The first chunk is the one case
    where the abort path's own persist is load-bearing -- without it the run
    would raise having written nothing at all."""
    client = StubExecutionClient({})  # must never be reached
    probe = ScriptedVramProbe([5.0])
    runner = _make_runner(
        client, tmp_path, vram_probe=probe, between_chunk_min_free_vram_gb=16.0
    )

    with pytest.raises(resilience_module.VramBelowFloorError):
        runner.render_run([1, 2], _provider_returning(), tmp_path / "chunks")

    state_file = tmp_path / "run_state.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["schema_version"] == resilience_module.RUN_STATE_SCHEMA_VERSION
    assert client.calls == []


def test_a_probe_that_raises_is_treated_as_unreadable_not_as_a_run_failure(
    tmp_path: Path, caplog
):
    """``custody._fetch_free_vram_gb`` only catches ``requests.RequestException``,
    so an unexpected transport or session error reaches the runner as a raise
    rather than as ``None``. A raise is a *reading we could not take* -- the
    same situation ``None`` describes -- and killing a multi-hour render over
    a best-effort check would be a worse outcome than the check not existing."""
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    calls: list[int] = []

    def exploding_probe() -> float | None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("session blew up in a way requests does not classify")
        return 20.0

    runner = _make_runner(client, tmp_path, vram_probe=exploding_probe, min_free_vram_gb=16.0)

    with caplog.at_level(logging.INFO):
        run_state = runner.render_run([1, 2], _provider_returning(), tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[2].status is ChunkStatus.RENDERED
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


def test_vram_reading_is_logged_at_info_per_chunk_regardless_of_outcome(tmp_path: Path, caplog):
    client = StubExecutionClient({1: [_rendered(1, tmp_path)], 2: [_rendered(2, tmp_path)]})
    probe = ScriptedVramProbe([20.0, None])
    runner = _make_runner(client, tmp_path, vram_probe=probe, min_free_vram_gb=16.0)
    provider = _provider_returning()

    with caplog.at_level(logging.INFO):
        runner.render_run([1, 2], provider, tmp_path / "chunks")

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    # Chunk 1's healthy reading is reported...
    assert any("20.00" in m for m in info_messages)
    # ...and chunk 2's unreadable check is reported too -- "regardless of
    # outcome" per issue #23 ("the run's own logs said nothing about the
    # card" was the actual failure on the incident night).
    assert any(
        str(2) in m and ("unavailable" in m.lower() or "unreadable" in m.lower())
        for m in info_messages
    )


def test_vram_probe_is_not_called_for_chunks_reused_from_resume(tmp_path: Path):
    """A cached chunk never submits anything to the GPU, so there is nothing
    to protect it from -- the probe should only fire for chunks actually
    about to render."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client1 = StubExecutionClient({1: [_rendered(1, output_dir)], 2: [_rendered(2, output_dir)]})
    runner1 = _make_runner(client1, tmp_path)
    runner1.render_run([1, 2], _provider_returning(), output_dir)

    client2 = StubExecutionClient({})
    probe2 = ScriptedVramProbe([])
    runner2 = _make_runner(client2, tmp_path, vram_probe=probe2, min_free_vram_gb=16.0)

    run_state = runner2.render_run([1, 2], _provider_returning(), output_dir, resume=True)

    assert run_state.results[1].status is ChunkStatus.CACHED
    assert run_state.results[2].status is ChunkStatus.CACHED
    assert probe2.calls == 0


def test_vram_below_floor_error_is_a_resilience_error(tmp_path: Path):
    assert issubclass(VramBelowFloorError, ResilienceError)


def test_default_vram_probe_is_none(tmp_path: Path):
    client = StubExecutionClient()
    runner = ResilientRunner(client, run_state_file=tmp_path / "run_state.json")
    assert runner.vram_probe is None


def test_from_config_wires_vram_probe_and_min_free_vram_gb(tmp_path: Path):
    config = RunConfig(
        master_audio=tmp_path / "a.wav",
        lyrics_file=tmp_path / "l.txt",
        global_style="style",
        narrative_concept="concept",
        cast={},
        default_lead_vocalist="Lead",
        comfyui_url="http://doris:8188",
        workflow_template=tmp_path / "wf.json",
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=HardwareProfile(name="RTX 4090", vram_gb=24.0),
        run_state_file=tmp_path / "state.json",
        min_free_vram_gb=12.5,
    )
    client = StubExecutionClient()
    probe = ScriptedVramProbe([20.0])

    runner = ResilientRunner.from_config(config, client, vram_probe=probe)

    assert runner.min_free_vram_gb == pytest.approx(12.5)
    assert runner.vram_probe is probe


# --------------------------------------------------------------------------- #
# from_config convenience wiring
# --------------------------------------------------------------------------- #


def test_from_config_wires_knobs_from_run_config(tmp_path: Path):
    config = RunConfig(
        master_audio=tmp_path / "a.wav",
        lyrics_file=tmp_path / "l.txt",
        global_style="style",
        narrative_concept="concept",
        cast={},
        default_lead_vocalist="Lead",
        comfyui_url="http://doris:8188",
        workflow_template=tmp_path / "wf.json",
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=HardwareProfile(name="RTX 4090", vram_gb=24.0),
        watchdog_timeout_seconds=123.0,
        max_render_attempts=7,
        retry_backoff_seconds=2.5,
        min_free_disk_gb=3.0,
        run_state_file=tmp_path / "state.json",
        resume_ignore_prompt_changes=True,
    )
    client = StubExecutionClient()

    runner = ResilientRunner.from_config(config, client)

    assert runner.watchdog_timeout_seconds == 123.0
    assert runner.max_render_attempts == 7
    assert runner.retry_backoff_seconds == 2.5
    assert runner.min_free_disk_gb == 3.0
    assert runner.run_state_file == tmp_path / "state.json"
    assert runner.ignore_prompt_changes is True
    assert client.ws_timeout == 123.0


def test_from_config_rejects_config_with_unresolved_run_state_file(tmp_path: Path):
    """``load_config`` always resolves ``run_state_file``; a hand-constructed
    ``RunConfig`` that left it ``None`` is a programming error, not something
    to silently paper over with a made-up default path."""
    config = RunConfig(
        master_audio=tmp_path / "a.wav",
        lyrics_file=tmp_path / "l.txt",
        global_style="style",
        narrative_concept="concept",
        cast={},
        default_lead_vocalist="Lead",
        comfyui_url="http://doris:8188",
        workflow_template=tmp_path / "wf.json",
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=HardwareProfile(name="RTX 4090", vram_gb=24.0),
        run_state_file=None,
    )
    client = StubExecutionClient()

    with pytest.raises(ResilienceError):
        ResilientRunner.from_config(config, client)


def test_execution_client_without_ws_timeout_attribute_is_tolerated(tmp_path: Path):
    class _NoWsTimeoutClient:
        def execute(self, workflow, chunk_id, output_dir):
            raise AssertionError("not called in this test")

        def interrupt(self) -> None:
            pass

        def free(self, *, unload_models: bool = True) -> None:
            pass

    # Must not raise even though the client has no ws_timeout attribute.
    ResilientRunner(_NoWsTimeoutClient(), run_state_file=tmp_path / "run_state.json")


# =========================================================================== #
# Integration tests: real ComfyUIExecutionClient + comfyui_mock + ws harness
# =========================================================================== #


def _real_client(
    session: FakeComfyUISession, ws_factory, *, ws_timeout=900.0
) -> ComfyUIExecutionClient:
    import itertools

    return ComfyUIExecutionClient(
        base_url=session.base_url,
        session=session,
        ws_factory=ws_factory,
        client_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        clock=itertools.count(0.0, 1.0).__next__,
        ws_timeout=ws_timeout,
    )


def test_integration_hang_then_timeout_then_interrupt_free_retry_success(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0002", video_filename="chunk.mp4")

    scripts = deque(
        [
            build_hang_sequence("prompt-0001"),
            build_success_sequence("prompt-0002"),
        ]
    )

    def ws_factory(url, **kwargs):
        return make_ws_factory(scripts.popleft())(url, **kwargs)

    client = _real_client(session, ws_factory)
    sleeper = RecordingSleeper()
    runner = ResilientRunner(
        client,
        run_state_file=tmp_path / "run_state.json",
        watchdog_timeout_seconds=900.0,
        max_render_attempts=3,
        retry_backoff_seconds=1.0,
        min_free_disk_gb=0.0,
        sleeper=sleeper,
        disk_usage=_abundant_disk_usage,
    )
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert run_state.results[1].attempts == 2
    assert session.interrupt_calls == 1
    assert len(session.free_calls) == 1
    assert sleeper.delays == [1.0]


def test_integration_oom_then_retry_success(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0002", video_filename="chunk.mp4")

    scripts = deque(
        [
            build_oom_sequence("prompt-0001"),
            build_success_sequence("prompt-0002"),
        ]
    )

    def ws_factory(url, **kwargs):
        return make_ws_factory(scripts.popleft())(url, **kwargs)

    client = _real_client(session, ws_factory)
    runner = ResilientRunner(
        client,
        run_state_file=tmp_path / "run_state.json",
        max_render_attempts=3,
        retry_backoff_seconds=0.5,
        min_free_disk_gb=0.0,
        sleeper=RecordingSleeper(),
        disk_usage=_abundant_disk_usage,
    )
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert session.interrupt_calls == 1


def test_integration_retry_exhaustion_dead_letters_with_real_client(tmp_path: Path):
    session = FakeComfyUISession()
    scripts = deque(
        [
            build_disconnect_sequence("prompt-0001"),
            build_disconnect_sequence("prompt-0002"),
        ]
    )

    def ws_factory(url, **kwargs):
        return make_ws_factory(scripts.popleft())(url, **kwargs)

    client = _real_client(session, ws_factory)
    runner = ResilientRunner(
        client,
        run_state_file=tmp_path / "run_state.json",
        max_render_attempts=2,
        retry_backoff_seconds=0.1,
        min_free_disk_gb=0.0,
        sleeper=RecordingSleeper(),
        disk_usage=_abundant_disk_usage,
    )
    provider = _provider_returning()

    run_state = runner.render_run([1], provider, tmp_path / "chunks")

    result = run_state.results[1]
    assert result.status is ChunkStatus.DEAD_LETTERED
    assert result.attempts == 2
    assert len(result.errors) == 2
    assert session.interrupt_calls == 2
    assert len(session.free_calls) == 2


def test_integration_resume_from_partial_run(tmp_path: Path):
    output_dir = tmp_path / "chunks"
    session1 = FakeComfyUISession()
    session1.seed_history_success("prompt-0001", video_filename="chunk1.mp4")
    ws_factory1 = make_ws_factory(build_success_sequence("prompt-0001"))
    client1 = _real_client(session1, ws_factory1)
    runner1 = ResilientRunner(
        client1,
        run_state_file=tmp_path / "run_state.json",
        max_render_attempts=3,
        min_free_disk_gb=0.0,
        sleeper=RecordingSleeper(),
        disk_usage=_abundant_disk_usage,
    )
    run_state1 = runner1.render_run([1], _provider_returning(), output_dir)
    assert run_state1.results[1].status is ChunkStatus.RENDERED
    assert run_state1.results[1].video_file.exists()

    # Second run resumes: chunk 1 must be reused (CACHED), chunk 2 rendered fresh.
    # session2 is a brand-new FakeComfyUISession, so its own /prompt counter
    # starts at 1 -- the single real submission chunk 2 triggers gets
    # "prompt-0001" on *this* session, regardless of what session1 assigned.
    session2 = FakeComfyUISession()
    session2.seed_history_success("prompt-0001", video_filename="chunk2.mp4")
    ws_factory2 = make_ws_factory(build_success_sequence("prompt-0001"))
    client2 = _real_client(session2, ws_factory2)
    runner2 = ResilientRunner(
        client2,
        run_state_file=tmp_path / "run_state.json",
        max_render_attempts=3,
        min_free_disk_gb=0.0,
        sleeper=RecordingSleeper(),
        disk_usage=_abundant_disk_usage,
    )
    provider2 = _provider_returning()

    run_state2 = runner2.render_run([1, 2], provider2, output_dir, resume=True)

    assert run_state2.results[1].status is ChunkStatus.CACHED
    assert run_state2.results[2].status is ChunkStatus.RENDERED
    assert [cid for cid, _ in provider2.calls] == [2]  # chunk 1 never asked for a workflow
    assert len(session2.submitted_prompts) == 1  # only chunk 2 hit the real submission path


# --------------------------------------------------------------------------- #
# Issue #28: chained chunks and --resume
# --------------------------------------------------------------------------- #


def test_resume_re_renders_a_chunk_chained_from_a_rerendered_one(tmp_path: Path):
    """Chunk 1's first frame is pixels from chunk 0's video. When chunk 0
    re-renders, that footage no longer exists, so reusing chunk 1's cached mp4
    puts a hard cut exactly where the chain was meant to remove one -- even
    though chunk 1's own fingerprint matches perfectly."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    prior = {
        0: _fp(0.0, 6.0),
        1: dc_replace(_fp(6.0, 12.0), chained_from=0),
    }
    _complete_a_run(tmp_path, output_dir, prior)

    # Chunk 0's span moved; chunk 1 asks for exactly what it already has.
    moved = {
        0: _fp(0.0, 6.5),
        1: dc_replace(_fp(6.0, 12.0), chained_from=0),
    }
    client = StubExecutionClient(
        {
            0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")],
            1: [_rendered(1, output_dir, name="chunk_0001_new.mp4")],
        }
    )
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0, 1], _provider_returning(), output_dir, resume=True, fingerprints=moved
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert run_state.results[1].status is ChunkStatus.RENDERED
    assert [c.chunk_id for c in client.calls] == [0, 1]


def test_resume_reuses_an_unchained_chunk_after_its_predecessor_rerendered(tmp_path: Path):
    """The chain rule only reaches chained chunks: an unchained chunk does not
    depend on anyone's pixels and stays reusable."""
    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    prior = {0: _fp(0.0, 6.0), 1: _fp(6.0, 12.0)}
    _complete_a_run(tmp_path, output_dir, prior)

    moved = {0: _fp(0.0, 6.5), 1: _fp(6.0, 12.0)}
    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0, 1], _provider_returning(), output_dir, resume=True, fingerprints=moved
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert run_state.results[1].status is ChunkStatus.CACHED
    assert [c.chunk_id for c in client.calls] == [0]


def test_fingerprint_amender_records_what_actually_happened(tmp_path: Path):
    """The chaining decision is only known after the provider runs (a
    degraded chunk falls back to unchained); the amender lets the caller
    record the truth instead of the plan."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    client = StubExecutionClient({0: [_rendered(0, output_dir)]})
    runner = _make_runner(client, tmp_path)

    planned = {0: dc_replace(_fp(0.0, 6.0), chained_from=None)}

    def amender(chunk_id: int, fingerprint) -> object:
        return dc_replace(fingerprint, chained_from=7)

    run_state = runner.render_run(
        [0],
        _provider_returning(),
        output_dir,
        fingerprints=planned,
        fingerprint_amender=amender,
    )

    assert run_state.results[0].fingerprint.chained_from == 7
    # And it round-trips through persistence.
    state = json.loads((tmp_path / "run_state.json").read_text())
    assert state["results"]["0"]["fingerprint"]["chained_from"] == 7


# --------------------------------------------------------------------------- #
# Issue #25: the conditioning source is never escapable on resume
# --------------------------------------------------------------------------- #


def test_conditioning_source_change_re_renders_even_with_ignore_prompt_changes(
    tmp_path: Path, caplog
):
    """Swapping mix -> stem is the experiment variable of the stem A/B.
    resume_ignore_prompt_changes exists for cosmetic edits; letting it blur
    which conditioning produced a cached chunk hands the A/B its control
    twice and calls it a result."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    prior = {0: dc_replace(_fp(0.0, 6.0), conditioning_source="mix")}
    _complete_a_run(tmp_path, output_dir, prior)

    wanted = {0: dc_replace(_fp(0.0, 6.0), conditioning_source="stem:vocals.wav")}
    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints=wanted
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert [c.chunk_id for c in client.calls] == [0]


def test_unrecorded_conditioning_source_cannot_prove_a_match(tmp_path: Path):
    """A pre-#25 state file has no conditioning_source key: None means
    unrecorded, which is not evidence the chunk was mix-conditioned."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    prior = {0: _fp(0.0, 6.0)}  # conditioning_source unrecorded
    _complete_a_run(tmp_path, output_dir, prior)

    wanted = {0: dc_replace(_fp(0.0, 6.0), conditioning_source="mix")}
    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path)

    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints=wanted
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


# --------------------------------------------------------------------------- #
# Issue #39: which text encoder produced the conditioning
# --------------------------------------------------------------------------- #


def test_text_encoder_change_re_renders_even_with_ignore_prompt_changes(tmp_path: Path):
    """Swapping NVFP4 for INT8 is the experiment variable of #39's encoder
    A/B, exactly as mix->stem is #25's. Reusing an NVFP4 chunk inside an
    INT8 run hands the comparison its control twice and calls it a result --
    and, outside an A/B, assembles a video whose shots were understood by
    two different encoders."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    nvfp4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    prior = {0: dc_replace(_fp(0.0, 6.0), text_encoder=nvfp4)}
    _complete_a_run(tmp_path, output_dir, prior)

    wanted = {
        0: dc_replace(
            _fp(0.0, 6.0), text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        )
    }
    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints=wanted
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED
    assert [c.chunk_id for c in client.calls] == [0]


def test_unrecorded_text_encoder_cannot_prove_a_match(tmp_path: Path):
    """A state file written before issue #39 has no text_encoder key: None
    means unrecorded, which is not evidence that it was the encoder this run
    names."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    prior = {0: _fp(0.0, 6.0)}  # text_encoder unrecorded
    _complete_a_run(tmp_path, output_dir, prior)

    wanted = {
        0: dc_replace(
            _fp(0.0, 6.0), text_encoder="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        )
    }
    client = StubExecutionClient({0: [_rendered(0, output_dir, name="chunk_0000_new.mp4")]})
    runner = _make_runner(client, tmp_path, ignore_prompt_changes=True)

    run_state = runner.render_run(
        [0], _provider_returning(), output_dir, resume=True, fingerprints=wanted
    )

    assert run_state.results[0].status is ChunkStatus.RENDERED


def test_text_encoder_round_trips_through_run_state(tmp_path: Path):
    """The record only helps if it survives the file --resume reads."""
    from dataclasses import replace as dc_replace

    output_dir = tmp_path / "chunks"
    output_dir.mkdir()
    planned = {
        0: dc_replace(
            _fp(0.0, 6.0), text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        )
    }
    client = StubExecutionClient({0: [_rendered(0, output_dir)]})
    runner = _make_runner(client, tmp_path)
    runner.render_run([0], _provider_returning(), output_dir, fingerprints=planned)

    state = json.loads((tmp_path / "run_state.json").read_text())
    assert state["results"]["0"]["fingerprint"]["text_encoder"] == (
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )


def test_instrumental_audio_gain_is_conditioning_and_never_escapable():
    """F26: attenuating the instrumental stem changes what H3 was conditioned on.

    ``conditioning_source`` records *which* audio (mix vs isolated stem) and is
    blind to its level, so without this a ``--resume`` across a gain change
    would reuse chunks conditioned on full-level music inside a silenced run --
    handing the A/B its control twice, which is the exact failure that tier
    exists to prevent.

    ``None`` means unrecorded, and must never compare equal to a recorded value.
    """
    from music_video_maker.contracts import ChunkFingerprint

    assert "instrumental_audio_gain_db" in ChunkFingerprint.CONDITIONING_FIELDS

    loud = ChunkFingerprint(start=0.0, end=5.0, instrumental_audio_gain_db=None)
    quiet = ChunkFingerprint(start=0.0, end=5.0, instrumental_audio_gain_db=-60.0)

    assert "instrumental_audio_gain_db" in quiet.conditioning_differences(loud)
    assert "instrumental_audio_gain_db" in loud.conditioning_differences(quiet)
    assert quiet.conditioning_differences(quiet) == ()
