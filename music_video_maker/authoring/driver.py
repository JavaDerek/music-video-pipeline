"""The model-calling seam (issue #54, design section 3).

Every authoring stage talks to a model through :class:`ModelDriver` --
never directly to ``subprocess`` or a network client. That is what makes the
whole package testable offline: :class:`ScriptedDriver` replays canned
responses from :mod:`tests.fixtures.authoring` in every unit test, and
exactly one test (marked ``@pytest.mark.integration``, deselected in CI)
exercises :class:`ClaudeCliDriver` against the real ``claude`` binary.

**Driven through the Claude CLI, never a metered API** -- this project's
standing no-metered-API rule (see project ``CLAUDE.md``). ``--safe-mode``
rather than ``--bare`` is the load-bearing choice: ``--bare`` forces
``ANTHROPIC_API_KEY`` auth, which *is* the metered path. ``--safe-mode``
disables this repo's own customisation surface (``CLAUDE.md``, skills,
hooks, MCP -- none of which belong in an authoring prompt) while leaving
subscription auth alone. Authoring runs once per song, by hand, on the
existing subscription, and produces a file a human commits -- it is not the
render path's per-chunk, metered-by-construction problem this rule exists to
prevent.

``--tools ""`` is equally load-bearing: every stage in this package reads no
files and writes no files through the model. The driver hands it text and
gets text back; there is nothing to permission and no way for a stage to
edit the repository it is describing.

**On the exact JSON envelope ``--output-format json`` returns:** this module
was written against the CLI's documented shape as understood at design time
(a top-level object carrying the final reply under ``result`` and cost under
``total_cost_usd``/``cost_usd``), with tolerant field-name fallbacks because
that surface can move. The single ``integration``-marked test in
``tests/test_authoring_driver.py`` is the canary -- if the CLI's output shape
has drifted, that test fails loudly instead of a corrupt plan being written
silently. See design section 15, "Claude CLI flags moving."

``--system-prompt``/``--json-schema`` are passed **inline**, not via a
``--system-prompt-file``/schema-file path -- the design sketch assumed file
flags that turned out not to exist on the installed CLI (checked against
``claude --help`` before the first real call, 2026-08-12); only
``--system-prompt <text>`` and ``--json-schema <json>`` are real.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MODEL_FABLE = "claude-fable-5"
MODEL_OPUS = "claude-opus-5"
MODEL_SONNET = "claude-sonnet-5"
"""The three models the design's stages are pinned to (section 4): concept
on Fable, beats/photography on Opus, prose on Sonnet. ``MODEL_SONNET`` also
doubles as the driver's ``--fallback-model``."""

DEFAULT_MAX_BUDGET_USD = 2.0
"""A runaway guard, not a budget policy (design section 3) -- a stage that
wants 40x its expected spend has misunderstood the prompt, and this is what
stops it before it does that repeatedly across retries."""

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_ATTEMPTS = 3
"""Bounded retry on non-zero exit, timeout, or non-JSON output (design
section 3) -- schema violations are a *stage*-level concern (the stage owns
the local validator the driver knows nothing about) and are retried
separately, by the stage, via :meth:`ModelDriver.complete` called again with
the validator's error folded into the prompt."""


class DriverError(RuntimeError):
    """Raised when a model call could not be completed after every retry.

    Carries enough of the failure history in its message to diagnose without
    re-running anything -- the same convention as
    ``assembly.FfmpegError``/``staging.AssetStagingError``.
    """


class _RecoverableDriverError(DriverError):
    """Internal: a single attempt's failure that :meth:`ClaudeCliDriver.complete`
    retries automatically (non-zero exit, timeout, non-JSON output). Never
    raised across the public :class:`ModelDriver` boundary -- only
    :class:`DriverError` (after retries are exhausted) is."""


@dataclass(frozen=True)
class DriverResult:
    """One completed model call.

    ``data`` is the structured reply, already parsed from JSON -- callers
    still run it through their own stage-specific validator (design section
    3: "the result is re-validated locally"; the driver never assumes the
    model's ``--json-schema`` compliance is trustworthy on its own).
    ``raw`` is the full, unparsed stdout, kept for ``.authoring/raw/``
    forensics (design section 11). ``cost_usd`` is ``None`` when the CLI's
    output doesn't report one -- never a fabricated cost.
    """

    data: dict[str, Any]
    raw: str
    model: str
    session_id: str
    cost_usd: float | None
    wall_seconds: float


class ModelDriver(Protocol):
    """What every authoring stage calls through -- never ``subprocess``
    directly. See the module docstring for why this indirection is the whole
    ballgame for offline testing."""

    def complete(
        self, *, system: str, prompt: str, model: str, schema: dict[str, Any]
    ) -> DriverResult: ...


SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]
"""Mirrors ``assembly.SubprocessRunner`` -- same seam, same reason: unit
tests assert on the exact argv without a real binary ever running."""


def _text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def _extract_result_text(envelope: dict[str, Any]) -> str:
    for key in ("result", "response", "content"):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            # Some CLI versions nest the structured reply directly rather
            # than as a JSON-encoded string; re-serialize so the caller's
            # single `json.loads` codepath still works either way.
            return json.dumps(value)
    raise _RecoverableDriverError(
        "claude CLI's JSON envelope carried no recognizable result field "
        f"(looked for 'result'/'response'/'content'): {json.dumps(envelope)[:2000]}"
    )


def _extract_cost(envelope: dict[str, Any]) -> float | None:
    for key in ("total_cost_usd", "cost_usd"):
        value = envelope.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


class ClaudeCliDriver:
    """Real :class:`ModelDriver` -- shells out to the ``claude`` CLI.

    ``subprocess_runner`` is the injectable seam (mirrors
    ``assembly.SubprocessRunner``): unit tests supply a fake that records the
    exact argv and returns a scripted ``CompletedProcess``, so this class's
    argv-building and envelope-parsing logic is fully covered without a real
    binary ever running. Only ``tests/test_authoring_driver.py``'s single
    ``integration``-marked test uses the real default.
    """

    def __init__(
        self,
        *,
        binary: str = "claude",
        max_budget_usd: float = DEFAULT_MAX_BUDGET_USD,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        subprocess_runner: SubprocessRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._binary = binary
        self._max_budget_usd = max_budget_usd
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._run = subprocess_runner or self._default_run
        self._clock = clock

    def _default_run(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        """Real ``claude`` invocation. Never used by unit tests -- injected
        out, exactly like ``assembly._default_runner``."""
        return subprocess.run(
            list(args), capture_output=True, check=False, timeout=self._timeout_seconds
        )

    def complete(
        self, *, system: str, prompt: str, model: str, schema: dict[str, Any]
    ) -> DriverResult:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._attempt(system=system, prompt=prompt, model=model, schema=schema)
            except _RecoverableDriverError as exc:
                last_error = exc
                logger.warning(
                    "claude CLI call failed (attempt %d/%d, model=%s): %s",
                    attempt,
                    self._max_attempts,
                    model,
                    exc,
                )
        logger.error(
            "claude CLI call failed after %d attempt(s) (model=%s): %s",
            self._max_attempts,
            model,
            last_error,
        )
        raise DriverError(
            f"claude CLI call failed after {self._max_attempts} attempt(s) "
            f"(model={model}): {last_error}"
        ) from last_error

    def _attempt(
        self, *, system: str, prompt: str, model: str, schema: dict[str, Any]
    ) -> DriverResult:
        start = self._clock()
        argv = [
            self._binary,
            "-p",
            prompt,
            "--model",
            model,
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(schema),
            "--output-format",
            "json",
            "--tools",
            "",
            "--safe-mode",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--max-budget-usd",
            str(self._max_budget_usd),
            "--fallback-model",
            MODEL_SONNET,
        ]
        try:
            completed = self._run(argv)
        except subprocess.TimeoutExpired as exc:
            raise _RecoverableDriverError(
                f"claude CLI timed out after {self._timeout_seconds}s"
            ) from exc
        except OSError as exc:
            # Not recoverable by retrying -- the binary itself is
            # missing or unrunnable, and that will not change between
            # attempts. Fails the whole call immediately.
            raise DriverError(f"could not launch {self._binary!r}: {exc}") from exc

        wall_seconds = self._clock() - start

        if completed.returncode != 0:
            raise _RecoverableDriverError(
                f"claude CLI exited {completed.returncode}: {_text(completed.stderr)[:2000]}"
            )

        stdout = _text(completed.stdout)
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise _RecoverableDriverError(
                f"claude CLI did not return valid JSON on stdout: {exc}\n{stdout[:2000]}"
            ) from exc
        if not isinstance(envelope, dict):
            raise _RecoverableDriverError(
                f"claude CLI's top-level JSON was a {type(envelope).__name__}, not an object: "
                f"{stdout[:2000]}"
            )

        result_text = _extract_result_text(envelope)
        try:
            data = json.loads(result_text)
        except json.JSONDecodeError as exc:
            raise _RecoverableDriverError(
                f"claude CLI's structured result was not valid JSON: {exc}\n{result_text[:2000]}"
            ) from exc
        if not isinstance(data, dict):
            raise _RecoverableDriverError(
                f"claude CLI's structured result was a {type(data).__name__}, not an object: "
                f"{result_text[:2000]}"
            )

        return DriverResult(
            data=data,
            raw=stdout,
            model=model,
            session_id=str(envelope.get("session_id", "")),
            cost_usd=_extract_cost(envelope),
            wall_seconds=wall_seconds,
        )


@dataclass(frozen=True)
class _ScriptedReply:
    data: dict[str, Any] | None
    """The structured reply this call returns, or ``None`` to simulate a
    :class:`DriverError` (e.g. proving a stage's own retry-with-feedback
    loop actually retries)."""

    model: str = MODEL_SONNET
    cost_usd: float | None = 0.0
    wall_seconds: float = 0.1


class ScriptedDriver:
    """Test double: replays a fixed queue of canned replies, one per
    :meth:`complete` call, never touching a subprocess or a network.

    ``calls`` records every ``(system, prompt, model, schema)`` this driver
    was invoked with, in order, so a test can assert a retry actually
    appended the validator's error to the next prompt rather than repeating
    the first one verbatim.
    """

    def __init__(self, replies: Sequence[dict[str, Any] | None]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, *, system: str, prompt: str, model: str, schema: dict[str, Any]
    ) -> DriverResult:
        self.calls.append({"system": system, "prompt": prompt, "model": model, "schema": schema})
        if not self._replies:
            raise AssertionError("ScriptedDriver: more complete() calls than scripted replies")
        reply = self._replies.pop(0)
        if reply is None:
            raise DriverError("ScriptedDriver: this call was scripted to fail")
        return DriverResult(
            data=reply,
            raw=json.dumps({"result": json.dumps(reply)}),
            model=model,
            session_id="scripted-session",
            cost_usd=0.0,
            wall_seconds=0.01,
        )


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_BUDGET_USD",
    "DEFAULT_TIMEOUT_SECONDS",
    "MODEL_FABLE",
    "MODEL_OPUS",
    "MODEL_SONNET",
    "ClaudeCliDriver",
    "DriverError",
    "DriverResult",
    "ModelDriver",
    "ScriptedDriver",
    "SubprocessRunner",
]
