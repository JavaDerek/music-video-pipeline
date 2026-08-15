"""Tests for the model-calling seam (issue #54 design section 3).

Every test except the one marked ``integration`` injects a fake subprocess
runner -- no real ``claude`` binary, no network, no cost -- and asserts on
the exact argv built and on how the envelope gets parsed into a
:class:`DriverResult`. The retry behaviour (mechanical failures only; schema
violations are the calling *stage*'s job, tested in
``test_authoring_concept.py``) is the part most worth pinning down here.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from music_video_maker.authoring.driver import (
    MODEL_FABLE,
    MODEL_SONNET,
    ClaudeCliDriver,
    DriverError,
    DriverResult,
    ScriptedDriver,
)

SCHEMA = {"type": "object", "required": ["x"]}


# --------------------------------------------------------------------------- #
# ScriptedDriver
# --------------------------------------------------------------------------- #


def test_scripted_driver_returns_data_and_records_the_call():
    driver = ScriptedDriver([{"logline": "a story"}])

    result = driver.complete(system="sys", prompt="user", model=MODEL_FABLE, schema=SCHEMA)

    assert isinstance(result, DriverResult)
    assert result.data == {"logline": "a story"}
    assert result.model == MODEL_FABLE
    assert driver.calls == [
        {"system": "sys", "prompt": "user", "model": MODEL_FABLE, "schema": SCHEMA}
    ]


def test_scripted_driver_raises_when_reply_is_scripted_to_fail():
    driver = ScriptedDriver([None])
    with pytest.raises(DriverError):
        driver.complete(system="sys", prompt="user", model=MODEL_FABLE, schema=SCHEMA)


def test_scripted_driver_raises_when_queue_is_exhausted():
    driver = ScriptedDriver([{"a": 1}])
    driver.complete(system="sys", prompt="user", model=MODEL_FABLE, schema=SCHEMA)
    with pytest.raises(AssertionError):
        driver.complete(system="sys", prompt="user2", model=MODEL_FABLE, schema=SCHEMA)


def test_scripted_driver_replays_replies_in_order():
    driver = ScriptedDriver([{"n": 1}, {"n": 2}])
    first = driver.complete(system="s", prompt="p1", model=MODEL_FABLE, schema=SCHEMA)
    second = driver.complete(system="s", prompt="p2", model=MODEL_FABLE, schema=SCHEMA)
    assert first.data == {"n": 1}
    assert second.data == {"n": 2}


# --------------------------------------------------------------------------- #
# ClaudeCliDriver -- argv construction
# --------------------------------------------------------------------------- #


class _RecordingRunner:
    """Fake subprocess runner: records every argv, replays scripted results
    in order. A result may be a ``subprocess.CompletedProcess`` (return it)
    or an ``Exception`` instance (raise it) -- covers both success and every
    failure mode the driver retries on.

    ``--system-prompt``/``--json-schema`` are passed inline (verified against
    the real installed CLI's ``--help`` on 2026-08-12 -- there is no
    ``--system-prompt-file`` flag), so their content is just the next argv
    element, no file I/O involved."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[list[str]] = []
        self.system_prompt_contents: list[str] = []
        self.schema_contents: list[dict] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if "--system-prompt" in args:
            self.system_prompt_contents.append(args[args.index("--system-prompt") + 1])
        if "--json-schema" in args:
            self.schema_contents.append(json.loads(args[args.index("--json-schema") + 1]))
        if not self._results:
            raise AssertionError("_RecordingRunner: more calls than scripted results")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _envelope(data: dict, **extra) -> subprocess.CompletedProcess:
    payload = {"result": json.dumps(data), "session_id": "sess-1", "total_cost_usd": 0.05}
    payload.update(extra)
    return subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout=json.dumps(payload), stderr=""
    )


def test_claude_cli_driver_builds_the_expected_argv_shape():
    runner = _RecordingRunner([_envelope({"x": 1})])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_budget_usd=1.5)

    driver.complete(
        system="the system prompt", prompt="the user prompt", model=MODEL_FABLE, schema=SCHEMA
    )

    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[0] == "claude"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "the user prompt"
    assert "--model" in argv and argv[argv.index("--model") + 1] == MODEL_FABLE
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert "--tools" in argv and argv[argv.index("--tools") + 1] == ""
    assert "--safe-mode" in argv
    assert "--no-session-persistence" in argv
    assert "--disable-slash-commands" in argv
    assert "--max-budget-usd" in argv and argv[argv.index("--max-budget-usd") + 1] == "1.5"
    assert "--fallback-model" in argv and argv[argv.index("--fallback-model") + 1] == MODEL_SONNET

    # The system prompt and schema are passed inline via --system-prompt /
    # --json-schema, not written to files.
    assert runner.system_prompt_contents == ["the system prompt"]
    assert runner.schema_contents == [SCHEMA]


def test_claude_cli_driver_parses_the_envelope_into_a_driver_result():
    runner = _RecordingRunner([_envelope({"logline": "a story"})])
    driver = ClaudeCliDriver(subprocess_runner=runner)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"logline": "a story"}
    assert result.session_id == "sess-1"
    assert result.cost_usd == pytest.approx(0.05)
    assert result.model == MODEL_FABLE
    assert result.wall_seconds >= 0.0


def test_claude_cli_driver_accepts_a_nested_dict_result_field():
    """Some CLI versions may nest the structured reply as an object rather
    than a JSON-encoded string -- both must parse to the same DriverResult."""
    payload = {"result": {"logline": "a story"}, "session_id": "sess-2"}
    completed = subprocess.CompletedProcess(["claude"], 0, stdout=json.dumps(payload), stderr="")
    driver = ClaudeCliDriver(subprocess_runner=_RecordingRunner([completed]))

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"logline": "a story"}


def test_claude_cli_driver_cost_usd_is_none_when_not_reported():
    payload = {"result": json.dumps({"x": 1}), "session_id": "s"}
    completed = subprocess.CompletedProcess(["claude"], 0, stdout=json.dumps(payload), stderr="")
    driver = ClaudeCliDriver(subprocess_runner=_RecordingRunner([completed]))

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.cost_usd is None


# --------------------------------------------------------------------------- #
# ClaudeCliDriver -- retry on mechanical failures
# --------------------------------------------------------------------------- #


def test_retries_on_non_zero_exit_then_succeeds():
    failing = subprocess.CompletedProcess(["claude"], 1, stdout="", stderr="boom")
    runner = _RecordingRunner([failing, _envelope({"x": 1})])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"x": 1}
    assert len(runner.calls) == 2


def test_retries_on_non_json_stdout():
    garbled = subprocess.CompletedProcess(["claude"], 0, stdout="not json", stderr="")
    runner = _RecordingRunner([garbled, _envelope({"x": 1})])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"x": 1}


def test_retries_on_timeout():
    runner = _RecordingRunner(
        [subprocess.TimeoutExpired(cmd="claude", timeout=300.0), _envelope({"x": 1})]
    )
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"x": 1}


def test_raises_after_exhausting_every_retry():
    failing = subprocess.CompletedProcess(["claude"], 1, stdout="", stderr="boom")
    runner = _RecordingRunner([failing, failing, failing])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    with pytest.raises(DriverError, match="3 attempt"):
        driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert len(runner.calls) == 3


def test_missing_binary_fails_immediately_without_retrying():
    """The binary being absent will not change between attempts -- retrying
    would just burn the timeout three times for the same answer."""
    runner = _RecordingRunner([FileNotFoundError("no such file: claude")])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    with pytest.raises(DriverError, match="could not launch"):
        driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert len(runner.calls) == 1


def test_recognizable_result_field_missing_is_treated_as_a_recoverable_failure():
    no_result_field = subprocess.CompletedProcess(
        ["claude"], 0, stdout=json.dumps({"session_id": "s"}), stderr=""
    )
    runner = _RecordingRunner([no_result_field, _envelope({"x": 1})])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"x": 1}


def test_non_dict_top_level_json_is_treated_as_a_recoverable_failure():
    array_stdout = subprocess.CompletedProcess(["claude"], 0, stdout="[1, 2, 3]", stderr="")
    runner = _RecordingRunner([array_stdout, _envelope({"x": 1})])
    driver = ClaudeCliDriver(subprocess_runner=runner, max_attempts=3)

    result = driver.complete(system="s", prompt="p", model=MODEL_FABLE, schema=SCHEMA)

    assert result.data == {"x": 1}


# --------------------------------------------------------------------------- #
# Integration: the real `claude` CLI (deselected via -m in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_claude_cli_driver_integration_real_binary():
    """The canary for design section 15's "Claude CLI flags moving" risk --
    if ``--output-format json``'s envelope shape has drifted from what this
    module assumes, this is what catches it, not a corrupted shot plan."""
    binary = shutil.which("claude")
    if binary is None:
        pytest.skip("claude CLI not installed on this machine")

    driver = ClaudeCliDriver(max_budget_usd=0.05, timeout_seconds=60.0)
    result = driver.complete(
        system="Reply with JSON only, matching the schema exactly.",
        prompt="Reply with {\"x\": \"ok\"} and nothing else.",
        model=MODEL_FABLE,
        schema={"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}},
    )
    assert result.data.get("x") == "ok"
