"""Scriptable fake WebSocket for ComfyUI execution monitoring (issue #16).

:class:`ScriptedWebSocket` duck-types the surface of ``websocket.WebSocket``
(the ``websocket-client`` package, already a project dependency) that Stage 4b's
execution client (issue #9) and the resilience state machine (issue #10) drive:
``connect``, ``settimeout``, ``send``, ``recv``, and ``close``. Messages are
scripted in advance -- as the JSON strings a real client would receive from
``recv()`` -- and popped one at a time, so every test is deterministic and
offline. No real socket, no ``time.sleep()``: watchdog timeouts are modeled by
scripting a :class:`websocket.WebSocketTimeoutException` instance that ``recv()``
raises immediately.

Message-builder functions below produce ComfyUI's real WS JSON shape
(``{"type": ..., "data": {...}}``) so tests exercise the client's actual parsing
code, not a simplified stand-in. :func:`build_success_sequence` assembles the
full happy path; :func:`build_hang_sequence`, :func:`build_oom_sequence`,
:func:`build_disconnect_sequence`, and :func:`build_noise_messages` cover the
failure modes issue #10 needs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

logger = logging.getLogger(__name__)

# A scripted item is either a pre-encoded JSON message string, an exception
# instance to raise from recv(), or a zero-arg callable producing a message
# string (for messages whose content must be computed lazily).
ScriptItem = str | BaseException | Callable[[], str]


class ScriptedWebSocket:
    """Fake of ``websocket.WebSocket``. Scripted messages are consumed by ``.recv()``.

    Construct with the full script up front, or use :meth:`append` / :meth:`hang`
    to extend it mid-test. Exception instances placed in the script are raised
    (not returned) when popped, which is how failure modes (timeout, OOM,
    disconnect) are injected inline with normal messages.
    """

    def __init__(self, messages: Sequence[ScriptItem] = (), *, client_id: str | None = None):
        self._messages: list[ScriptItem] = list(messages)
        self._closed = False
        self.client_id = client_id
        self.connect_calls: list[str] = []
        self.sent: list[str] = []
        self.timeout: float | None = None
        self.close_calls = 0

    # -- connection setup (websocket.WebSocket surface) --------------------- #
    def connect(self, url: str, **kwargs: Any) -> None:
        logger.debug("ScriptedWebSocket.connect(%s)", url)
        self.connect_calls.append(url)
        self._closed = False

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    @property
    def connected(self) -> bool:
        return not self._closed

    # -- messaging ------------------------------------------------------------ #
    def send(self, data: str, **kwargs: Any) -> None:
        if self._closed:
            raise WebSocketConnectionClosedException("Socket is already closed.")
        self.sent.append(data)

    def recv(self) -> str:
        if self._closed:
            raise WebSocketConnectionClosedException("Socket is already closed.")
        if not self._messages:
            raise WebSocketConnectionClosedException("Scripted message queue exhausted.")
        item = self._messages.pop(0)
        if isinstance(item, BaseException):
            logger.debug("ScriptedWebSocket.recv() raising scripted %s", type(item).__name__)
            raise item
        if callable(item):
            item = item()
        return item

    def close(self, **kwargs: Any) -> None:
        logger.debug("ScriptedWebSocket.close()")
        self._closed = True
        self.close_calls += 1

    # -- test-authoring convenience ------------------------------------------- #
    def append(self, item: ScriptItem) -> None:
        """Extend the script at runtime, e.g. to react mid-test."""
        self._messages.append(item)

    def hang(self) -> None:
        """Queue a scripted watchdog timeout. No real sleep -- recv() raises immediately."""
        self._messages.append(WebSocketTimeoutException("timed out"))

    def disconnect(self, message: str = "Connection to remote host was lost.") -> None:
        """Queue a scripted mid-stream disconnect."""
        self._messages.append(WebSocketConnectionClosedException(message))

    def __len__(self) -> int:
        return len(self._messages)


def make_ws_factory(
    messages: Sequence[ScriptItem], *, client_id: str | None = None
) -> Callable[..., ScriptedWebSocket]:
    """Build a ``create_connection(url) -> WebSocket``-shaped factory.

    Convenience for clients that build via ``websocket.create_connection(url)``
    in one call rather than ``WebSocket()`` + ``.connect(url)``.
    """

    def _factory(url: str, **kwargs: Any) -> ScriptedWebSocket:
        ws = ScriptedWebSocket(messages, client_id=client_id)
        ws.connect(url, **kwargs)
        return ws

    return _factory


# --------------------------------------------------------------------------- #
# Message builders -- ComfyUI's real {"type": ..., "data": {...}} WS shape
# --------------------------------------------------------------------------- #


def status_message(queue_remaining: int = 0, *, sid: str | None = None) -> str:
    payload: dict[str, Any] = {
        "type": "status",
        "data": {"status": {"exec_info": {"queue_remaining": queue_remaining}}},
    }
    if sid is not None:
        payload["sid"] = sid
    return json.dumps(payload)


def execution_start_message(prompt_id: str) -> str:
    return json.dumps({"type": "execution_start", "data": {"prompt_id": prompt_id}})


def executing_message(prompt_id: str, node: str | None) -> str:
    """``node is None`` is ComfyUI's completion signal for this prompt_id."""
    return json.dumps({"type": "executing", "data": {"node": node, "prompt_id": prompt_id}})


def progress_message(prompt_id: str, node: str, value: int, max_value: int) -> str:
    return json.dumps(
        {
            "type": "progress",
            "data": {"value": value, "max": max_value, "prompt_id": prompt_id, "node": node},
        }
    )


def execution_cached_message(prompt_id: str, nodes: Sequence[str]) -> str:
    return json.dumps(
        {"type": "execution_cached", "data": {"nodes": list(nodes), "prompt_id": prompt_id}}
    )


def execution_error_message(
    prompt_id: str,
    node_id: str = "7",
    *,
    node_type: str = "MiniMaxH3Sampler",
    exception_type: str = "torch.cuda.OutOfMemoryError",
    exception_message: str | None = None,
) -> str:
    """A realistic CUDA OOM ``execution_error`` message (issue #10)."""
    exception_message = exception_message or (
        "CUDA out of memory. Tried to allocate 2.44 GiB (GPU 0; 23.99 GiB total capacity; "
        "22.31 GiB already allocated; 512.00 MiB free; 23.02 GiB reserved in total by PyTorch)"
    )
    traceback = [
        "Traceback (most recent call last):",
        '  File "/home/derek/ComfyUI/execution.py", line 327, in execute',
        "    output_data, output_ui = get_output_data(obj, input_data_all)",
        '  File "/home/derek/ComfyUI/comfy_extras/nodes_minimax_h3.py", line 214, in sample',
        "    samples = model.sample(positive, negative, latent, seed=seed, steps=steps)",
        f"{exception_type}: {exception_message}",
    ]
    return json.dumps(
        {
            "type": "execution_error",
            "data": {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": node_type,
                "exception_type": exception_type,
                "exception_message": exception_message,
                "traceback": traceback,
            },
        }
    )


# --------------------------------------------------------------------------- #
# Full-sequence builders
# --------------------------------------------------------------------------- #


def build_success_sequence(
    prompt_id: str,
    *,
    node_ids: Sequence[str] = ("3", "7"),
    progress_steps: int = 3,
    include_cached: bool = False,
    cached_nodes: Sequence[str] = (),
    queue_remaining_start: int = 1,
) -> list[str]:
    """The realistic happy path: status -> execution_start -> executing/progress
    per node -> optional execution_cached -> executing(node=None) completion.
    """
    messages = [status_message(queue_remaining_start), execution_start_message(prompt_id)]
    if include_cached:
        messages.append(execution_cached_message(prompt_id, cached_nodes or node_ids[:1]))
    for node in node_ids:
        messages.append(executing_message(prompt_id, node))
        for step in range(1, progress_steps + 1):
            messages.append(progress_message(prompt_id, node, step, progress_steps))
    messages.append(status_message(0))
    messages.append(executing_message(prompt_id, None))  # completion signal
    return messages


def build_hang_sequence(prompt_id: str, *, node_id: str = "3") -> list[ScriptItem]:
    """Execution starts, then the backend goes silent -- watchdog must fire."""
    return [
        status_message(1),
        execution_start_message(prompt_id),
        executing_message(prompt_id, node_id),
        WebSocketTimeoutException("timed out"),
    ]


def build_oom_sequence(
    prompt_id: str,
    *,
    node_id: str = "7",
    executing_nodes: Sequence[str] = ("3",),
    progress_steps: int = 2,
) -> list[str]:
    """Execution starts, runs briefly, then CUDA OOMs."""
    messages = [status_message(1), execution_start_message(prompt_id)]
    for node in executing_nodes:
        messages.append(executing_message(prompt_id, node))
        for step in range(1, progress_steps + 1):
            messages.append(progress_message(prompt_id, node, step, progress_steps))
    messages.append(execution_error_message(prompt_id, node_id=node_id))
    return messages


def build_disconnect_sequence(
    prompt_id: str,
    *,
    node_id: str = "3",
    progress_steps: int = 1,
    disconnect_message: str = "Connection to remote host was lost.",
) -> list[ScriptItem]:
    """Execution starts, runs briefly, then the socket drops mid-stream."""
    messages: list[ScriptItem] = [
        status_message(1),
        execution_start_message(prompt_id),
        executing_message(prompt_id, node_id),
    ]
    for step in range(1, progress_steps + 1):
        messages.append(progress_message(prompt_id, node_id, step, progress_steps))
    messages.append(WebSocketConnectionClosedException(disconnect_message))
    return messages


def build_noise_messages(
    *, prompt_id: str = "someone-elses-prompt-id", node_id: str = "1"
) -> list[str]:
    """Messages belonging to another tenant's job on the same ComfyUI instance.

    ComfyUI's WS broadcasts to every connected client; a correct client filters
    on its own ``client_id``/``prompt_id`` and must silently ignore these.
    """
    return [
        executing_message(prompt_id, node_id),
        progress_message(prompt_id, node_id, 1, 10),
    ]
