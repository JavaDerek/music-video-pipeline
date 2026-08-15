"""Tests for Stage 4b execution client (issue #9).

Drives :class:`ComfyUIExecutionClient` entirely offline against
``tests/harness/comfyui_mock.FakeComfyUISession`` (HTTP) and
``tests/harness/ws.ScriptedWebSocket`` (WebSocket). No real socket, no real
ComfyUI, no GPU, and -- per the project's "no sleep-polling" invariant --
this module must never call ``time.sleep``.
"""

from __future__ import annotations

import itertools
import json
import logging
import socket as socket_module
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.execution import (
    ComfyUIExecutionClient,
    HistoryError,
    OutputRetrievalError,
    PromptSubmissionError,
    WebSocketDisconnectedError,
    WebSocketTimeoutError,
    WorkflowExecutionError,
)
from tests.harness.comfyui_mock import FakeComfyUISession
from tests.harness.ws import (
    ScriptedWebSocket,
    build_noise_messages,
    build_oom_sequence,
    build_success_sequence,
    executing_message,
    make_ws_factory,
)

WORKFLOW = {"1": {"class_type": "UNETLoader", "inputs": {}}}
CLIENT_ID = "11111111-1111-4111-8111-111111111111"


def _make_client(
    session: FakeComfyUISession,
    ws_factory,
    *,
    client_id: str = CLIENT_ID,
    clock=None,
) -> ComfyUIExecutionClient:
    clock = clock or itertools.count(0.0, 1.0).__next__
    return ComfyUIExecutionClient(
        base_url=session.base_url,
        session=session,
        ws_factory=ws_factory,
        client_id=client_id,
        clock=clock,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_execute_happy_path_returns_rendered_result(tmp_path: Path, caplog):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))

    client = _make_client(session, ws_factory)
    output_dir = tmp_path / "chunks"

    with caplog.at_level(logging.INFO):
        result = client.execute(WORKFLOW, chunk_id=7, output_dir=output_dir)

    assert result.status is contracts.ChunkStatus.RENDERED
    assert result.chunk_id == 7
    assert result.prompt_id == "prompt-0001"
    assert result.attempts == 1
    assert result.render_seconds is not None and result.render_seconds > 0
    assert result.video_file is not None
    assert result.video_file.exists()
    assert result.video_file.name == "chunk_0007.mp4"
    assert result.video_file.read_bytes() == session._stored_outputs[("chunk.mp4", "", "output")]


def test_submitted_prompt_body_carries_workflow_and_client_id():
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))

    assert len(session.submitted_prompts) == 1
    submitted = session.submitted_prompts[0]
    assert submitted["workflow"] == WORKFLOW
    assert submitted["client_id"] == CLIENT_ID


def test_websocket_opens_before_prompt_is_submitted(tmp_path: Path):
    """Issue #48: on an idle queue ComfyUI can begin executing the instant
    ``POST /prompt`` is accepted, emitting ``execution_start`` before a socket
    opened *afterward* would ever connect -- so ``render_seconds`` stayed
    ``None`` on every render this project has done. Pin the ordering
    directly, since the failure depends on server timing that a mock can't
    reproduce by accident.
    """
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    events: list[str] = []

    real_post = session.post

    def _spy_post(url: str, **kwargs):
        if session._path(url) == "/prompt":
            events.append("prompt_submitted")
        return real_post(url, **kwargs)

    session.post = _spy_post  # type: ignore[method-assign]

    def _spy_ws_factory(url, **kwargs):
        events.append("ws_connected")
        return make_ws_factory(build_success_sequence("prompt-0001"))(url, **kwargs)

    client = _make_client(session, _spy_ws_factory)
    client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path)

    assert events == ["ws_connected", "prompt_submitted"]


def test_ws_url_carries_client_id_and_uses_ws_scheme():
    session = FakeComfyUISession(base_url="http://doris:8188")
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))

    # make_ws_factory's ScriptedWebSocket records the connect() url.
    # We recover the ws instance indirectly: build a fresh one to inspect the
    # factory's behavior directly (same factory, same script drained though --
    # so assert via a dedicated single-shot factory instead).
    seen_urls = []

    def _spy_factory(url, **kwargs):
        seen_urls.append(url)
        return make_ws_factory(build_success_sequence("prompt-0001"))(url, **kwargs)

    session2 = FakeComfyUISession()
    session2.seed_history_success("prompt-0001", video_filename="chunk2.mp4")
    client2 = _make_client(session2, _spy_factory)
    client2.execute(WORKFLOW, chunk_id=2, output_dir=Path("/tmp/unused2"))

    assert len(seen_urls) == 1
    url = seen_urls[0]
    assert url.startswith("ws://")
    assert f"clientId={CLIENT_ID}" in url


# --------------------------------------------------------------------------- #
# Completion detection / multi-tenancy noise filtering
# --------------------------------------------------------------------------- #


def test_only_matching_executing_none_ends_the_loop(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")

    foreign_prompt_id = "someone-elses-prompt-id"
    messages = [
        *build_noise_messages(prompt_id=foreign_prompt_id),
        # foreign completion-shaped message -- must NOT end our loop
        executing_message(foreign_prompt_id, None),
        *build_success_sequence("prompt-0001"),
    ]
    ws_factory = make_ws_factory(messages)
    client = _make_client(session, ws_factory)

    result = client.execute(WORKFLOW, chunk_id=3, output_dir=tmp_path)

    assert result.status is contracts.ChunkStatus.RENDERED
    assert result.prompt_id == "prompt-0001"


def test_id_less_completion_message_does_not_end_the_loop(tmp_path: Path):
    """Completion requires an *explicit* prompt_id match. An ``executing``
    frame carrying ``node: null`` and no prompt_id at all must not be mistaken
    for ours -- acting on it would send us to /history before our render
    exists, turning another tenant's traffic into a spurious failure."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")

    messages = [
        json.dumps({"type": "executing", "data": {"node": None}}),
        *build_success_sequence("prompt-0001"),
    ]
    ws_factory = make_ws_factory(messages)
    client = _make_client(session, ws_factory)

    result = client.execute(WORKFLOW, chunk_id=5, output_dir=tmp_path)

    assert result.status is contracts.ChunkStatus.RENDERED
    assert result.prompt_id == "prompt-0001"
    # The id-less frame precedes our execution_start, so a loop that ended on it
    # would never have started the render timer. This assertion -- not the
    # RENDERED status, which the seeded history would satisfy either way -- is
    # what actually distinguishes the correct behavior.
    assert result.render_seconds is not None


def test_noise_interleaved_with_real_sequence_is_ignored(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")

    real = build_success_sequence("prompt-0001")
    noise = build_noise_messages(prompt_id="other-tenant")
    # Interleave noise before, in the middle, and after the real sequence.
    messages = [*noise, *real[:3], *noise, *real[3:]]
    ws_factory = make_ws_factory(messages)
    client = _make_client(session, ws_factory)

    result = client.execute(WORKFLOW, chunk_id=4, output_dir=tmp_path)

    assert result.status is contracts.ChunkStatus.RENDERED


def test_execution_cached_and_progress_are_logged_not_fatal(tmp_path: Path, caplog):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    messages = build_success_sequence(
        "prompt-0001", include_cached=True, cached_nodes=["3"], progress_steps=2
    )
    ws_factory = make_ws_factory(messages)
    client = _make_client(session, ws_factory)

    with caplog.at_level(logging.INFO):
        result = client.execute(WORKFLOW, chunk_id=5, output_dir=tmp_path)

    assert result.status is contracts.ChunkStatus.RENDERED
    joined = "\n".join(r.message for r in caplog.records)
    assert "cache" in joined.lower()
    assert "progress" in joined.lower() or "Progress" in joined


# --------------------------------------------------------------------------- #
# WebSocket closing
# --------------------------------------------------------------------------- #


def test_ws_closed_on_success_path(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_instances = []

    def factory(url, **kwargs):
        ws = make_ws_factory(build_success_sequence("prompt-0001"))(url, **kwargs)
        ws_instances.append(ws)
        return ws

    client = _make_client(session, factory)
    client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path)

    assert ws_instances[0].close_calls == 1


def test_ws_closed_on_error_path():
    session = FakeComfyUISession()
    ws_instances = []

    def factory(url, **kwargs):
        ws = make_ws_factory(build_oom_sequence("prompt-0001"))(url, **kwargs)
        ws_instances.append(ws)
        return ws

    client = _make_client(session, factory)

    with pytest.raises(WorkflowExecutionError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))

    assert ws_instances[0].close_calls == 1


# --------------------------------------------------------------------------- #
# /prompt validation failure
# --------------------------------------------------------------------------- #


def test_prompt_validation_failure_closes_the_already_open_websocket(caplog):
    """Issue #48: the WebSocket now opens *before* ``POST /prompt`` (it is
    per-client, not per-prompt, so connecting first is safe), so a rejected
    submission must close that already-open socket rather than never seeing
    one -- and it must never reach ``_monitor`` since there is no prompt to
    watch."""
    session = FakeComfyUISession()
    session.queue_node_errors({"7": {"errors": [{"message": "required input missing"}]}})

    ws = ScriptedWebSocket([])  # empty script: recv() must never be called

    def factory(url, **kwargs):
        ws.connect(url, **kwargs)
        return ws

    client = _make_client(session, factory)

    with caplog.at_level(logging.ERROR), pytest.raises(PromptSubmissionError) as exc_info:
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))

    assert exc_info.value.node_errors == {"7": {"errors": [{"message": "required input missing"}]}}
    assert len(ws.connect_calls) == 1
    assert ws.close_calls == 1
    assert any("node_errors" in r.message or "rejected" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# execution_error (OOM)
# --------------------------------------------------------------------------- #


def test_execution_error_raises_typed_error_with_node_details():
    session = FakeComfyUISession()
    ws_factory = make_ws_factory(
        build_oom_sequence("prompt-0001", node_id="7", executing_nodes=("3",))
    )
    client = _make_client(session, ws_factory)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))

    err = exc_info.value
    assert err.node_id == "7"
    assert err.node_type == "MiniMaxH3Sampler"
    assert err.exception_type == "torch.cuda.OutOfMemoryError"
    assert "CUDA out of memory" in err.message


# --------------------------------------------------------------------------- #
# WebSocket timeout / disconnect
# --------------------------------------------------------------------------- #


def test_ws_timeout_raises_websocket_timeout_error():
    from tests.harness.ws import build_hang_sequence

    session = FakeComfyUISession()
    ws_factory = make_ws_factory(build_hang_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(WebSocketTimeoutError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))


def test_ws_disconnect_raises_websocket_disconnected_error():
    from tests.harness.ws import build_disconnect_sequence

    session = FakeComfyUISession()
    ws_factory = make_ws_factory(build_disconnect_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(WebSocketDisconnectedError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))


# --------------------------------------------------------------------------- #
# History / view failures
# --------------------------------------------------------------------------- #


def test_history_without_video_raises_history_error():
    session = FakeComfyUISession()
    session.seed_history_without_video("prompt-0001")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(HistoryError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))


def test_missing_history_entry_raises_history_error():
    session = FakeComfyUISession()
    # No seed at all -- FakeComfyUISession returns an empty {} body.
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(HistoryError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused"))


def test_view_empty_body_raises_output_retrieval_error(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    session.corrupt_view_output("chunk.mp4", mode="empty")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(OutputRetrievalError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path)

    assert not any(tmp_path.iterdir())


def test_view_http_error_raises_output_retrieval_error(tmp_path: Path):
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    session.corrupt_view_output("chunk.mp4", mode="error")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    with pytest.raises(OutputRetrievalError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path)

    assert not any(tmp_path.iterdir())


def test_history_output_scan_ignores_output_key_name(tmp_path: Path):
    """Video detection must not hardcode the 'videos' output key."""
    session = FakeComfyUISession()
    session.seed_history_success(
        "prompt-0001", video_filename="chunk.mp4", output_key="gifs_and_stuff"
    )
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    result = client.execute(WORKFLOW, chunk_id=9, output_dir=tmp_path)

    assert result.status is contracts.ChunkStatus.RENDERED
    assert result.video_file.name == "chunk_0009.mp4"


# --------------------------------------------------------------------------- #
# interrupt() / free()
# --------------------------------------------------------------------------- #


def test_interrupt_hits_interrupt_endpoint():
    session = FakeComfyUISession()
    client = _make_client(session, ws_factory=make_ws_factory([]))

    client.interrupt()

    assert session.interrupt_calls == 1


def test_free_hits_free_endpoint_with_expected_body():
    session = FakeComfyUISession()
    client = _make_client(session, ws_factory=make_ws_factory([]))

    client.free(unload_models=True)

    assert len(session.free_calls) == 1
    assert session.free_calls[0]["unload_models"] is True


def test_free_default_unloads_models():
    session = FakeComfyUISession()
    client = _make_client(session, ws_factory=make_ws_factory([]))

    client.free()

    assert session.free_calls[0]["unload_models"] is True


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #


def test_client_satisfies_execution_client_protocol():
    session = FakeComfyUISession()
    client = ComfyUIExecutionClient(base_url=session.base_url, session=session)
    assert isinstance(client, contracts.ExecutionClient)


# --------------------------------------------------------------------------- #
# No sleep-polling invariant
# --------------------------------------------------------------------------- #


def test_execution_module_never_sleeps(monkeypatch):
    import time as time_module

    def _boom(*_args, **_kwargs):
        raise AssertionError("execution.py must never call time.sleep()")

    monkeypatch.setattr(time_module, "sleep", _boom)

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)

    result = client.execute(WORKFLOW, chunk_id=1, output_dir=Path("/tmp/unused_sleep_test"))
    assert result.status is contracts.ChunkStatus.RENDERED


def test_execution_source_contains_no_sleep_call():
    import inspect

    from music_video_maker import execution

    source = inspect.getsource(execution)
    assert "time.sleep(" not in source
    assert "sleep(" not in source


# --------------------------------------------------------------------------- #
# Issue #43: ask ComfyUI before re-rendering what it may already have finished
# --------------------------------------------------------------------------- #


def test_ws_timeout_reconciles_a_render_comfyui_already_completed(tmp_path):
    """The 2026-08-11 defect. The host slept mid-chunk; ComfyUI finished the
    prompt 123 s later and wrote the mp4; this side heard nothing, and hours
    afterwards re-rendered work that was already on disk.

    A transport failure says nothing about whether the *render* failed, so the
    history is consulted before an attempt is written off."""
    from tests.harness.ws import build_hang_sequence

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="mvm_chunk_0001_00001.mp4")
    client = _make_client(session, make_ws_factory(build_hang_sequence("prompt-0001")))

    result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert result.video_file == tmp_path / "out" / "chunk_0001.mp4"
    assert result.video_file.read_bytes()
    # The whole point: no second submission.
    assert len(session.submitted_prompts) == 1


def test_ws_disconnect_reconciles_a_render_comfyui_already_completed(tmp_path):
    """Same rule for a dropped socket -- a Wi-Fi handover or a tailnet blip is
    not evidence the GPU failed."""
    from tests.harness.ws import build_disconnect_sequence

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="mvm_chunk_0001_00001.mp4")
    client = _make_client(session, make_ws_factory(build_disconnect_sequence("prompt-0001")))

    result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert len(session.submitted_prompts) == 1


def test_reconciled_result_keeps_the_transport_failure_as_evidence(tmp_path):
    """A reconciled chunk is not an uneventful one: the run must still be able
    to show that the socket died, or a systematically flaky link looks like a
    clean run."""
    from tests.harness.ws import build_hang_sequence

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="mvm_chunk_0001_00001.mp4")
    client = _make_client(session, make_ws_factory(build_hang_sequence("prompt-0001")))

    result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.errors, "the WebSocket failure was silently discarded"
    assert any("timed out" in e.lower() or "timeout" in e.lower() for e in result.errors)
    # render_seconds is honestly unknown: the completion was never witnessed.
    assert result.render_seconds is None


def test_ws_timeout_still_raises_when_the_render_really_did_not_finish(tmp_path):
    """History with no video output means ComfyUI did *not* produce the chunk
    -- still a genuine failure, and it must surface as the original transport
    error so the resilience layer classifies and retries it exactly as before,
    not as a HistoryError from the reconciliation attempt."""
    from tests.harness.ws import build_hang_sequence

    session = FakeComfyUISession()
    session.seed_history_without_video("prompt-0001")
    client = _make_client(session, make_ws_factory(build_hang_sequence("prompt-0001")))

    with pytest.raises(WebSocketTimeoutError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")


def test_ws_timeout_still_raises_when_the_prompt_is_absent_from_history(tmp_path):
    """Nothing in history at all -- still queued, or lost. The original
    transport error stands."""
    from tests.harness.ws import build_hang_sequence

    session = FakeComfyUISession()
    client = _make_client(session, make_ws_factory(build_hang_sequence("prompt-0001")))

    with pytest.raises(WebSocketTimeoutError):
        client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")


# --------------------------------------------------------------------------- #
# TCP keepalive tuning (issue #43 item 3)
#
# A watchdog governed by ws_timeout only fires when *our process* gets CPU to
# notice recv() overrun its deadline. When the host itself is asleep, nothing
# runs that clock, and the socket sits until the OS gives up on its own
# schedule -- observed 2026-08-10 at ~3.5h against a configured 900s watchdog.
# TCP keepalive (plus TCP_USER_TIMEOUT where the platform has it) makes the
# *kernel* notice the dead peer independent of our process being scheduled,
# bounded near the configured watchdog window instead of the OS default.
# --------------------------------------------------------------------------- #


class _FakeSocket:
    """Records ``setsockopt`` calls; can be told which optnames to reject,
    simulating a platform whose kernel doesn't actually support an option
    even though the Python ``socket`` module defines the constant."""

    def __init__(self, *, unsupported: set[int] | None = None):
        self.setsockopt_calls: list[tuple[int, int, int]] = []
        self._unsupported = unsupported or set()

    def setsockopt(self, level: int, optname: int, value: int) -> None:
        if optname in self._unsupported:
            raise OSError(f"protocol not supported for optname={optname}")
        self.setsockopt_calls.append((level, optname, value))


def _ws_factory_with_socket(messages, sock):
    """Like ``make_ws_factory``, but attaches ``sock`` to the returned
    ``ScriptedWebSocket`` so there is something for keepalive tuning to
    configure. Production ``websocket.WebSocket`` objects expose ``.sock``
    the same way once connected."""

    def factory(url, **kwargs):
        ws = ScriptedWebSocket(messages)
        ws.connect(url, **kwargs)
        ws.sock = sock
        return ws

    return factory


def test_tcp_keepalive_derives_timings_from_ws_timeout(tmp_path: Path, monkeypatch):
    """The knobs must be computed from the configured watchdog window, not
    hardcoded, so the two cannot drift apart (issue #43's core complaint)."""
    monkeypatch.setattr(socket_module, "TCP_KEEPIDLE", 4, raising=False)
    monkeypatch.setattr(socket_module, "TCP_KEEPINTVL", 5, raising=False)
    monkeypatch.setattr(socket_module, "TCP_KEEPCNT", 6, raising=False)
    monkeypatch.setattr(socket_module, "TCP_USER_TIMEOUT", 18, raising=False)
    monkeypatch.delattr(socket_module, "TCP_KEEPALIVE", raising=False)

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    fake_sock = _FakeSocket()
    ws_factory = _ws_factory_with_socket(build_success_sequence("prompt-0001"), fake_sock)
    client = _make_client(session, ws_factory)
    client.ws_timeout = 900.0

    result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    calls = {(level, optname): value for level, optname, value in fake_sock.setsockopt_calls}
    assert calls[(socket_module.SOL_SOCKET, socket_module.SO_KEEPALIVE)] == 1
    # idle and interval each 1/3 of the 900s watchdog window.
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPIDLE)] == 300
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPINTVL)] == 300
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPCNT)] == 3
    # (idle + interval * count) expressed in milliseconds for TCP_USER_TIMEOUT.
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_USER_TIMEOUT)] == 1_200_000


def test_tcp_keepalive_timings_scale_with_a_different_watchdog(tmp_path: Path):
    """Changing ws_timeout must change the derived timings -- pins that the
    values are computed, not a fixed constant (catches a hardcoding mutant)."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    fake_sock = _FakeSocket()
    ws_factory = _ws_factory_with_socket(build_success_sequence("prompt-0001"), fake_sock)
    client = _make_client(session, ws_factory)
    client.ws_timeout = 90.0

    client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    calls = {(level, optname): value for level, optname, value in fake_sock.setsockopt_calls}
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPINTVL)] == 30
    if hasattr(socket_module, "TCP_KEEPIDLE"):
        assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPIDLE)] == 30
    if hasattr(socket_module, "TCP_KEEPALIVE"):
        assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPALIVE)] == 30


def test_tcp_keepalive_skips_missing_platform_options_without_crashing(
    tmp_path: Path, monkeypatch, caplog
):
    """Simulate a platform (e.g. darwin, which really does lack both of
    these) missing some options but not others. Must not crash, must log
    what was skipped by name, and must still apply whatever it does have."""
    monkeypatch.delattr(socket_module, "TCP_KEEPIDLE", raising=False)
    monkeypatch.delattr(socket_module, "TCP_USER_TIMEOUT", raising=False)
    monkeypatch.setattr(socket_module, "TCP_KEEPALIVE", 16, raising=False)
    monkeypatch.setattr(socket_module, "TCP_KEEPINTVL", 257, raising=False)
    monkeypatch.setattr(socket_module, "TCP_KEEPCNT", 258, raising=False)

    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    fake_sock = _FakeSocket()
    ws_factory = _ws_factory_with_socket(build_success_sequence("prompt-0001"), fake_sock)
    client = _make_client(session, ws_factory)
    client.ws_timeout = 900.0

    with caplog.at_level(logging.WARNING):
        result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert (socket_module.SOL_SOCKET, socket_module.SO_KEEPALIVE, 1) in fake_sock.setsockopt_calls
    calls = {(level, optname): value for level, optname, value in fake_sock.setsockopt_calls}
    assert calls[(socket_module.IPPROTO_TCP, socket_module.TCP_KEEPALIVE)] == 300
    assert "TCP_KEEPIDLE" in caplog.text
    assert "TCP_USER_TIMEOUT" in caplog.text


def test_tcp_keepalive_setsockopt_failure_is_logged_and_does_not_crash(tmp_path: Path, caplog):
    """A constant existing doesn't guarantee the kernel accepts it (older
    kernels, containers). A rejected option must not abort the others."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    fake_sock = _FakeSocket(unsupported={socket_module.TCP_KEEPINTVL})
    ws_factory = _ws_factory_with_socket(build_success_sequence("prompt-0001"), fake_sock)
    client = _make_client(session, ws_factory)
    client.ws_timeout = 900.0

    with caplog.at_level(logging.WARNING):
        result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert (socket_module.SOL_SOCKET, socket_module.SO_KEEPALIVE, 1) in fake_sock.setsockopt_calls
    assert not any(
        optname == socket_module.TCP_KEEPINTVL for _l, optname, _v in fake_sock.setsockopt_calls
    )
    assert "TCP_KEEPINTVL" in caplog.text


def test_tcp_keepalive_skips_silently_but_logged_when_ws_has_no_socket(
    tmp_path: Path, caplog
):
    """The test harness's ScriptedWebSocket has no ``sock`` -- and neither
    might some future ws_factory. Must degrade, not crash."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    ws_factory = make_ws_factory(build_success_sequence("prompt-0001"))
    client = _make_client(session, ws_factory)
    client.ws_timeout = 900.0

    with caplog.at_level(logging.INFO):
        result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert "sock" in caplog.text.lower()


def test_tcp_keepalive_skipped_when_ws_timeout_not_configured(tmp_path: Path, caplog):
    """Nothing to derive timings from -- must not guess, must not crash, and
    must not touch the socket at all (leaves OS defaults untouched)."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="chunk.mp4")
    fake_sock = _FakeSocket()
    ws_factory = _ws_factory_with_socket(build_success_sequence("prompt-0001"), fake_sock)
    client = _make_client(session, ws_factory)
    client.ws_timeout = None

    with caplog.at_level(logging.WARNING):
        result = client.execute(WORKFLOW, chunk_id=1, output_dir=tmp_path / "out")

    assert result.status is contracts.ChunkStatus.RENDERED
    assert fake_sock.setsockopt_calls == []
    assert "ws_timeout" in caplog.text.lower()
