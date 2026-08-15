"""Stage 4b: execution client (issue #9).

Submits a mutated MiniMax H3 workflow to ComfyUI's ``/prompt`` endpoint, then
monitors the render over ComfyUI's WebSocket event stream to completion
(event-driven -- **never** sleep-polling), and finally retrieves the
generated video through ``/history`` + ``/view``.

Every I/O seam is injectable so the whole client is testable offline against
``tests/harness/ws.ScriptedWebSocket`` and
``tests/harness/comfyui_mock.FakeComfyUISession`` (issue #16):

* ``session`` -- a ``requests.Session``-shaped object for the HTTP calls.
* ``ws_factory`` -- a ``websocket.create_connection(url, **kwargs) -> ws``-shaped
  callable for the WebSocket.
* ``client_id`` -- normally a UUID4, overridable for deterministic tests.
* ``clock`` -- a zero-arg monotonic clock for deterministic ``render_seconds``.

The WebSocket is how a completion is *witnessed*, not how it happens (issue
#43). If the socket dies mid-render -- the host slept, the tailnet blipped --
ComfyUI keeps going, finishes the prompt and writes the mp4, and a client that
equated "I lost the socket" with "the render failed" would re-render work
already sitting on the server. So a transport failure is investigated against
``/history`` before it is believed; only a prompt with no completed video
output surfaces as a failure, and then as the *original* transport exception,
because that is what issue #10's state machine classifies retries by. See
``ComfyUIExecutionClient._reconcile_after_transport_failure``.

Scope note (issue #9 vs issue #10): this module is happy-path-plus-basic-errors
only. Watchdog timeouts, OOM recovery, retries, and dead-lettering are the
resilience state machine built in issue #10 -- but the seams it needs already
exist here: ``interrupt()`` / ``free()`` are implemented, ``settimeout`` is
configurable via the constructor, and every failure mode surfaces as a
distinct, typed exception (see the hierarchy below) so #10 can dispatch on
``except`` clauses instead of string-matching messages.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import websocket
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

from music_video_maker.config import DEFAULT_COMFYUI_URL
from music_video_maker.contracts import ChunkResult, ChunkStatus, Workflow

logger = logging.getLogger(__name__)

# Injectable seams -- see module docstring.
WSFactory = Callable[..., Any]
Clock = Callable[[], float]

_VIDEO_EXTENSION = ".mp4"

# --------------------------------------------------------------------------- #
# TCP keepalive tuning (issue #43 item 3) -- bounding dead-peer detection.
#
# ``ws_timeout`` (recv()'s own SO_RCVTIMEO-style deadline) only fires when
# *our process* gets scheduled to notice it has elapsed. On 2026-08-10 the
# Mac slept mid-render: nothing ran recv()'s clock, and without any OS-level
# keepalive the kernel eventually gave up on its own schedule -- observed at
# ~3.5h against a configured 900s watchdog, roughly 14x the intended window.
# TCP keepalive (plus TCP_USER_TIMEOUT where the platform has it) makes the
# *kernel* itself notice the dead peer -- on wake, independent of whether our
# process is scheduled at all -- and return ETIMEDOUT/ECONNRESET from the
# blocked recv() within a bounded multiple of ws_timeout, instead of at
# whatever the platform's default retransmission ceiling happens to be.
#
# Fractions chosen: idle time and probe interval are each 1/3 of ws_timeout,
# with 3 probes after the idle period. Worst case before the kernel gives up:
#   idle + interval * count = ws_timeout/3 + ws_timeout = (4/3) * ws_timeout
# -- about 1.33x the configured watchdog window, not an OS default measured
# in hours. TCP_USER_TIMEOUT (Linux only) is set to that same bound directly,
# in milliseconds, as a second and independent cap on how long the kernel
# will wait for an ACK before declaring the connection dead.
_KEEPALIVE_FRACTION = 1 / 3
_KEEPALIVE_PROBE_COUNT = 3
_MIN_KEEPALIVE_SECONDS = 1


# --------------------------------------------------------------------------- #
# Typed exception hierarchy -- issue #10 dispatches on these, not on message
# text. Keep additions here in sync with the docstring above.
# --------------------------------------------------------------------------- #


class ExecutionError(RuntimeError):
    """Base class for every Stage 4b execution failure."""


class PromptSubmissionError(ExecutionError):
    """``POST /prompt`` was rejected by ComfyUI's graph validator, or the
    request/response itself was malformed (no ``prompt_id``)."""

    def __init__(self, message: str, *, node_errors: dict[str, Any] | None = None) -> None:
        self.node_errors: dict[str, Any] = node_errors or {}
        super().__init__(message)


class WebSocketMonitoringError(ExecutionError):
    """Base class for WebSocket-transport failures while monitoring a render."""


class WebSocketTimeoutError(WebSocketMonitoringError):
    """``recv()`` raised ``websocket.WebSocketTimeoutException`` (watchdog fires
    here in issue #10; this module never sets a watchdog timeout itself, only
    exposes ``ws_timeout`` for #10 to configure)."""


class WebSocketDisconnectedError(WebSocketMonitoringError):
    """The WebSocket connection was lost mid-render."""


class WorkflowExecutionError(ExecutionError):
    """ComfyUI reported an ``execution_error`` WS message (e.g. CUDA OOM)."""

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None,
        node_type: str | None,
        exception_type: str | None,
        exception_message: str | None,
    ) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.exception_type = exception_type
        self.message = exception_message
        super().__init__(message)


class HistoryError(ExecutionError):
    """``GET /history/{prompt_id}`` had no usable video output."""


class OutputRetrievalError(ExecutionError):
    """``GET /view`` failed, returned no bytes, or the video could not be
    written to disk."""


def _default_client_id() -> str:
    return str(uuid.uuid4())


class ComfyUIExecutionClient:
    """Stage 4b implementation of ``contracts.ExecutionClient``."""

    def __init__(
        self,
        base_url: str = DEFAULT_COMFYUI_URL,
        *,
        session: requests.Session | None = None,
        ws_factory: WSFactory | None = None,
        client_id: str | None = None,
        clock: Clock = time.monotonic,
        ws_timeout: float | None = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.ws_factory = ws_factory if ws_factory is not None else websocket.create_connection
        self.client_id = client_id if client_id is not None else _default_client_id()
        self.clock = clock
        self.ws_timeout = ws_timeout

    # -- public API (contracts.ExecutionClient) ------------------------------ #

    def execute(self, workflow: Workflow, chunk_id: int, output_dir: Path) -> ChunkResult:
        # Issue #48: the socket is opened *before* the prompt is submitted.
        # ComfyUI's WS stream is per-client (clientId), not per-prompt, so
        # connecting first is safe -- and on an idle queue it is necessary:
        # ComfyUI can start executing the instant POST /prompt is accepted,
        # emitting execution_start before a socket opened afterward would
        # ever connect, which is why render_seconds was None on every render
        # this project has ever done.
        ws_url = self._ws_url()
        logger.info("Opening WebSocket %s for chunk_id=%d", ws_url, chunk_id)
        ws = self.ws_factory(ws_url)
        ws.settimeout(self.ws_timeout)
        self._configure_tcp_keepalive(ws)

        try:
            prompt_id = self._submit(workflow)
        except PromptSubmissionError:
            self._close_ws(ws, "<unsubmitted>")
            raise

        logger.info("Monitoring prompt_id=%s for chunk_id=%d", prompt_id, chunk_id)
        transport_failure: WebSocketMonitoringError | None = None
        try:
            render_start = self._monitor(ws, prompt_id)
        except WebSocketMonitoringError as exc:
            # Issue #43: losing the socket says nothing about whether the
            # *render* failed. Ask ComfyUI before writing the attempt off --
            # see _reconcile_after_transport_failure.
            self._reconcile_after_transport_failure(prompt_id, chunk_id, exc)
            transport_failure = exc
            render_start = None
        finally:
            self._close_ws(ws, prompt_id)

        # Unknown, not zero: with the socket gone the completion was never
        # witnessed, and a fabricated duration would poison the only render-cost
        # measurements this project has.
        render_seconds = None if render_start is None else self.clock() - render_start

        video_meta = self._fetch_history_video(prompt_id)
        video_bytes = self._fetch_view_bytes(prompt_id, video_meta)
        video_path = self._write_video(output_dir, chunk_id, video_bytes)

        result = ChunkResult(
            chunk_id=chunk_id,
            status=ChunkStatus.RENDERED,
            video_file=video_path,
            prompt_id=prompt_id,
            attempts=1,
            errors=() if transport_failure is None else (str(transport_failure),),
            render_seconds=render_seconds,
        )
        logger.info(
            "Chunk %d rendered%s: prompt_id=%s video_file=%s render_seconds=%s",
            chunk_id,
            "" if transport_failure is None else " (reconciled from /history)",
            prompt_id,
            video_path,
            render_seconds,
        )
        return result

    def _reconcile_after_transport_failure(
        self, prompt_id: str, chunk_id: int, failure: WebSocketMonitoringError
    ) -> None:
        """Decide whether a lost WebSocket actually cost us a render (issue #43).

        The WebSocket is how completion is *witnessed*, not how it happens.
        When the socket dies -- the host slept, the tailnet blipped, the lid
        closed -- ComfyUI carries on, finishes the prompt, and writes the mp4.
        Before this, that outcome was indistinguishable from a dead render:
        the attempt was written off, the recovery sequence ran, and the chunk
        was submitted again. On 2026-08-11 that cost two full re-renders of
        chunks ComfyUI had already completed and saved (see issue #43's table).

        So: consult ``/history/{prompt_id}``. A completed entry with a video
        output means the render succeeded and only the *reporting* failed --
        this returns, and :meth:`execute` proceeds to fetch and write exactly
        the output it would have fetched had the socket survived.

        Anything else re-raises ``failure`` -- the original transport error,
        deliberately, not the :class:`HistoryError` raised while looking. The
        resilience layer classifies retries by exception type
        (``WebSocketTimeoutError`` is the watchdog firing), and a chunk that
        genuinely did not render must reach that layer wearing the same
        exception it always did. The reconciliation attempt is an
        investigation, and an investigation that finds nothing must not change
        the diagnosis.
        """
        try:
            self._fetch_history_video(prompt_id)
        except HistoryError as history_exc:
            logger.warning(
                "Chunk %d: %s, and /history has no completed video for prompt_id=%s (%s) -- "
                "treating it as a real render failure",
                chunk_id,
                failure,
                prompt_id,
                history_exc,
            )
            raise failure from None
        logger.warning(
            "Chunk %d: %s, but ComfyUI had already completed prompt_id=%s -- harvesting the "
            "finished video from /history instead of re-rendering it (issue #43)",
            chunk_id,
            failure,
            prompt_id,
        )

    def interrupt(self) -> None:
        url = f"{self.base_url}/interrupt"
        try:
            self.session.post(url)
            logger.info("Sent POST /interrupt to %s", self.base_url)
        except requests.RequestException as exc:
            logger.error("POST /interrupt to %s failed: %s", self.base_url, exc)

    def free(self, *, unload_models: bool = True) -> None:
        url = f"{self.base_url}/free"
        body = {"unload_models": unload_models, "free_memory": True}
        try:
            self.session.post(url, json=body)
            logger.info("Sent POST /free to %s (body=%s)", self.base_url, body)
        except requests.RequestException as exc:
            logger.error("POST /free to %s failed (body=%s): %s", self.base_url, body, exc)

    # -- submission ------------------------------------------------------------ #

    def _submit(self, workflow: Workflow) -> str:
        url = f"{self.base_url}/prompt"
        body = {"prompt": workflow, "client_id": self.client_id}
        try:
            response = self.session.post(url, json=body)
        except requests.RequestException as exc:
            logger.error(
                "POST /prompt to %s failed for client_id=%s: %s", url, self.client_id, exc
            )
            raise PromptSubmissionError(f"POST /prompt request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error(
                "POST /prompt to %s returned a non-JSON body (status=%s)",
                url,
                getattr(response, "status_code", None),
            )
            raise PromptSubmissionError("POST /prompt returned a non-JSON response") from exc

        node_errors = payload.get("node_errors") or {}
        prompt_id = payload.get("prompt_id")
        if node_errors or not prompt_id:
            logger.error(
                "ComfyUI rejected /prompt submission for client_id=%s: node_errors=%s prompt_id=%s",
                self.client_id,
                node_errors,
                prompt_id,
            )
            raise PromptSubmissionError(
                f"ComfyUI rejected the workflow submission: {node_errors}",
                node_errors=node_errors,
            )

        logger.info("Submitted workflow: client_id=%s prompt_id=%s", self.client_id, prompt_id)
        return prompt_id

    # -- WebSocket monitoring ---------------------------------------------------- #

    def _ws_url(self) -> str:
        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/ws?clientId={self.client_id}"

    def _tcp_keepalive_timings(self, ws_timeout: float) -> dict[str, int]:
        """Derive keepalive idle/interval/count and the TCP_USER_TIMEOUT bound
        from the configured watchdog window. See the module-level comment
        above ``_KEEPALIVE_FRACTION`` for the arithmetic and why."""
        idle = max(_MIN_KEEPALIVE_SECONDS, round(ws_timeout * _KEEPALIVE_FRACTION))
        interval = max(_MIN_KEEPALIVE_SECONDS, round(ws_timeout * _KEEPALIVE_FRACTION))
        count = _KEEPALIVE_PROBE_COUNT
        user_timeout_ms = round((idle + interval * count) * 1000)
        return {
            "idle": idle,
            "interval": interval,
            "count": count,
            "user_timeout_ms": user_timeout_ms,
        }

    def _configure_tcp_keepalive(self, ws: Any) -> None:
        """Apply OS-level TCP keepalive to the WebSocket's underlying socket
        (issue #43 item 3), bounding how long a dead peer can go unnoticed.

        ``ws`` is whatever ``ws_factory`` returned -- normally a real
        ``websocket.WebSocket`` exposing ``.sock`` once connected, but the
        test harness's ``ScriptedWebSocket`` has no socket at all, and a
        future ``ws_factory`` might not either. Both are expected, not
        exceptional: this degrades silently (logged, never raised) rather
        than assume a real socket exists.

        Option names are platform-specific (macOS spells the idle-time knob
        ``TCP_KEEPALIVE``; Linux spells it ``TCP_KEEPIDLE`` and additionally
        has ``TCP_USER_TIMEOUT``), so every option is probed with
        ``getattr(socket, name, None)`` and silently skipped -- logged at
        WARNING -- when the platform lacks it. What *was* applied is logged
        at INFO.
        """
        sock = getattr(ws, "sock", None)
        if sock is None:
            logger.info(
                "WebSocket object has no 'sock' attribute -- skipping TCP keepalive "
                "tuning (expected for the test harness and any non-socket ws_factory)"
            )
            return

        if not self.ws_timeout:
            logger.warning(
                "No ws_timeout configured -- cannot derive TCP keepalive timings from "
                "the watchdog window, leaving OS defaults in place"
            )
            return

        timings = self._tcp_keepalive_timings(self.ws_timeout)

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except (OSError, AttributeError) as exc:
            logger.warning("Failed to enable SO_KEEPALIVE: %s", exc)
            return

        applied: list[str] = ["SO_KEEPALIVE=1"]
        for opt_name, value in (
            ("TCP_KEEPIDLE", timings["idle"]),
            ("TCP_KEEPALIVE", timings["idle"]),  # macOS's spelling of the same knob
            ("TCP_KEEPINTVL", timings["interval"]),
            ("TCP_KEEPCNT", timings["count"]),
            ("TCP_USER_TIMEOUT", timings["user_timeout_ms"]),
        ):
            opt = getattr(socket, opt_name, None)
            if opt is None:
                logger.warning(
                    "socket.%s is not available on this platform -- skipping", opt_name
                )
                continue
            try:
                sock.setsockopt(socket.IPPROTO_TCP, opt, value)
            except (OSError, AttributeError) as exc:
                logger.warning("Failed to set socket.%s=%s: %s", opt_name, value, exc)
                continue
            applied.append(f"{opt_name}={value}")

        logger.info(
            "TCP keepalive configured from ws_timeout=%.1fs: %s",
            self.ws_timeout,
            ", ".join(applied),
        )

    def _monitor(self, ws: Any, prompt_id: str) -> float | None:
        """Drain WS messages until our ``prompt_id`` reports completion.

        Returns the ``clock()`` reading taken when our ``execution_start``
        was observed (the render timer), or ``None`` if it never arrived.
        Messages belonging to another client's job (multi-tenancy -- ComfyUI
        broadcasts to every connected client) are silently ignored, including
        a foreign ``executing`` message with ``node is None``: only *our*
        ``prompt_id`` reporting ``node is None`` ends the loop.
        """
        render_start: float | None = None
        while True:
            try:
                raw = ws.recv()
            except WebSocketTimeoutException as exc:
                logger.error("WebSocket timed out monitoring prompt_id=%s: %s", prompt_id, exc)
                raise WebSocketTimeoutError(
                    f"WebSocket timed out monitoring prompt_id={prompt_id}"
                ) from exc
            except WebSocketConnectionClosedException as exc:
                logger.error(
                    "WebSocket disconnected while monitoring prompt_id=%s: %s", prompt_id, exc
                )
                raise WebSocketDisconnectedError(
                    f"WebSocket disconnected monitoring prompt_id={prompt_id}: {exc}"
                ) from exc

            if isinstance(raw, (bytes, bytearray)):
                logger.debug(
                    "Ignoring binary WS frame (%d bytes) for prompt_id=%s", len(raw), prompt_id
                )
                continue

            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                logger.debug("Ignoring non-JSON WS frame for prompt_id=%s: %r", prompt_id, raw)
                continue

            if not isinstance(message, dict):
                logger.debug("Ignoring non-object WS message for prompt_id=%s", prompt_id)
                continue

            msg_type = message.get("type")
            data = message.get("data") or {}

            if msg_type == "status":
                queue_remaining = (
                    data.get("status", {}).get("exec_info", {}).get("queue_remaining")
                )
                logger.info(
                    "ComfyUI queue status for prompt_id=%s: remaining=%s",
                    prompt_id,
                    queue_remaining,
                )
                continue

            msg_prompt_id = data.get("prompt_id")
            if msg_prompt_id is not None and msg_prompt_id != prompt_id:
                logger.debug(
                    "Ignoring WS message type=%s for foreign prompt_id=%s (ours=%s)",
                    msg_type,
                    msg_prompt_id,
                    prompt_id,
                )
                continue

            if msg_type == "execution_start":
                render_start = self.clock()
                logger.info("Execution started for prompt_id=%s", prompt_id)
            elif msg_type == "executing":
                node = data.get("node")
                if node is None:
                    if msg_prompt_id != prompt_id:
                        # Completion is the one signal we require an *explicit*
                        # id match for. An id-less executing/None frame on a
                        # multi-tenant server would otherwise end our loop early
                        # and send us to /history before our render exists.
                        logger.debug(
                            "Ignoring completion-shaped WS message with prompt_id=%r (ours=%s)",
                            msg_prompt_id,
                            prompt_id,
                        )
                        continue
                    logger.info("Execution completed for prompt_id=%s", prompt_id)
                    return render_start
                logger.info("Executing node=%s for prompt_id=%s", node, prompt_id)
            elif msg_type == "progress":
                logger.info(
                    "Progress prompt_id=%s node=%s %s/%s",
                    prompt_id,
                    data.get("node"),
                    data.get("value"),
                    data.get("max"),
                )
            elif msg_type == "execution_cached":
                logger.info(
                    "Nodes served from cache for prompt_id=%s: %s", prompt_id, data.get("nodes")
                )
            elif msg_type == "execution_error":
                node_id = data.get("node_id")
                node_type = data.get("node_type")
                exception_type = data.get("exception_type")
                exception_message = data.get("exception_message")
                logger.error(
                    "ComfyUI execution_error prompt_id=%s node_id=%s node_type=%s "
                    "exception_type=%s message=%s",
                    prompt_id,
                    node_id,
                    node_type,
                    exception_type,
                    exception_message,
                )
                raise WorkflowExecutionError(
                    f"ComfyUI execution failed at node {node_id} ({node_type}): "
                    f"{exception_type}: {exception_message}",
                    node_id=node_id,
                    node_type=node_type,
                    exception_type=exception_type,
                    exception_message=exception_message,
                )
            else:
                logger.debug(
                    "Ignoring unrecognized WS message type=%s for prompt_id=%s",
                    msg_type,
                    prompt_id,
                )

    def _close_ws(self, ws: Any, prompt_id: str) -> None:
        try:
            ws.close()
        except Exception as exc:  # noqa: BLE001 -- close must never mask the real error
            logger.warning("Error closing WebSocket for prompt_id=%s: %s", prompt_id, exc)

    # -- output retrieval ------------------------------------------------------- #

    def _fetch_history_video(self, prompt_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/history/{prompt_id}"
        try:
            response = self.session.get(url)
        except requests.RequestException as exc:
            logger.error("GET %s failed: %s", url, exc)
            raise HistoryError(f"GET /history/{prompt_id} request failed: {exc}") from exc

        if response.status_code != 200:
            logger.error("GET %s returned status=%s", url, response.status_code)
            raise HistoryError(
                f"GET /history/{prompt_id} returned status {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("GET %s returned a non-JSON body", url)
            raise HistoryError(f"GET /history/{prompt_id} returned a non-JSON response") from exc

        entry = payload.get(prompt_id)
        if not entry:
            logger.error("GET %s: no history entry present for prompt_id=%s", url, prompt_id)
            raise HistoryError(f"No history entry found for prompt_id={prompt_id}")

        outputs = entry.get("outputs") or {}
        video_meta = _find_video_output(outputs)
        if video_meta is None:
            logger.error(
                "GET %s: no video output found among outputs for prompt_id=%s (node_ids=%s)",
                url,
                prompt_id,
                list(outputs),
            )
            raise HistoryError(f"No video output found in history for prompt_id={prompt_id}")

        return video_meta

    def _fetch_view_bytes(self, prompt_id: str, video_meta: dict[str, Any]) -> bytes:
        params = {
            "filename": video_meta.get("filename", ""),
            "subfolder": video_meta.get("subfolder", ""),
            "type": video_meta.get("type", "output"),
        }
        url = f"{self.base_url}/view"
        try:
            response = self.session.get(url, params=params)
        except requests.RequestException as exc:
            logger.error(
                "GET %s failed for prompt_id=%s params=%s: %s", url, prompt_id, params, exc
            )
            raise OutputRetrievalError(f"GET /view request failed: {exc}") from exc

        if response.status_code != 200:
            logger.error(
                "GET %s returned status=%s for prompt_id=%s params=%s",
                url,
                response.status_code,
                prompt_id,
                params,
            )
            raise OutputRetrievalError(
                f"GET /view returned status {response.status_code} for {params}"
            )

        content = response.content
        if not content:
            logger.error(
                "GET %s returned zero bytes for prompt_id=%s params=%s", url, prompt_id, params
            )
            raise OutputRetrievalError(f"GET /view returned an empty body for {params}")

        return content

    def _write_video(self, output_dir: Path, chunk_id: int, content: bytes) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"chunk_{chunk_id:04d}.mp4"
        try:
            video_path.write_bytes(content)
        except OSError as exc:
            logger.error(
                "Failed to write video for chunk_id=%d to %s: %s", chunk_id, video_path, exc
            )
            raise OutputRetrievalError(f"Failed to write video file {video_path}: {exc}") from exc
        return video_path


def _find_video_output(outputs: dict[str, Any]) -> dict[str, Any] | None:
    """Scan every output collection on every node for an entry that looks like
    a video: a ``.mp4`` filename, or a ``format`` field mentioning ``video``.

    Deliberately does **not** key on a fixed collection name (e.g.
    ``"videos"``) -- real ComfyUI SaveVideo-style nodes have been inconsistent
    about it, and the harness's default is just a convention, not a contract.
    """
    for node_outputs in outputs.values():
        if not isinstance(node_outputs, dict):
            continue
        for items in node_outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename", ""))
                fmt = str(item.get("format", ""))
                if filename.lower().endswith(_VIDEO_EXTENSION) or "video" in fmt.lower():
                    return item
    return None


__all__ = [
    "ComfyUIExecutionClient",
    "ExecutionError",
    "HistoryError",
    "OutputRetrievalError",
    "PromptSubmissionError",
    "WebSocketDisconnectedError",
    "WebSocketMonitoringError",
    "WebSocketTimeoutError",
    "WorkflowExecutionError",
]
