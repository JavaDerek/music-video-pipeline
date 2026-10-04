"""Tests for the control half of the run monitor (issue #36).

What is under test is a *refusal machine*: five gates, a single-claim lock,
and one signal. So the suite is mostly "prove it says no, and prove it says
why" -- plus two tests that spawn a real child process, because the one
claim this module makes that cannot be faked is "stopping is exactly what
Ctrl-C does".

Offline by construction: the VRAM reading comes from
:class:`tests.harness.comfyui_mock.FakeComfyUISession`, the disk reading from
an injected ``disk_usage``, and every controller test but the two real-child
ones injects a spawner that starts nothing. No GPU, no ComfyUI, no network,
and no render ever runs.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from music_video_maker import control
from music_video_maker.config import load_config
from music_video_maker.prepare_report import (
    InputStamp,
    PrepareReport,
    write_prepare_report,
)
from tests.harness.comfyui_mock import FakeComfyUISession

# --------------------------------------------------------------------------- #
# Fixtures: a real, loadable run config whose files exist on disk
# --------------------------------------------------------------------------- #

_RUN_TOML = """
master_audio = "{tmp_path}/audio/master.wav"
lyrics_file = "{tmp_path}/lyrics.txt"
global_style = "test style"
narrative_concept = "test concept"
default_lead_vocalist = "Dianne"
comfyui_url = "{comfyui_url}"
workflow_template = "{tmp_path}/workflow_api.json"
chunks_dir = "{tmp_path}/output/chunks"
final_video_dir = "{tmp_path}/output/final"
render_width = 864
render_height = 480
min_free_disk_gb = 20.0
min_free_vram_gb = 16.0

[cast.Dianne]
role = "Lead Vocalist"
image = "{tmp_path}/cast/dianne_ref.jpg"

[hardware]
name = "RTX 4090 24GB (doris)"
vram_gb = 24.0
"""


def _write_config(tmp_path: Path, *, comfyui_url: str = "http://127.0.0.1:8188") -> Path:
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "master.wav").write_bytes(b"RIFF-fake-wav-data")
    (tmp_path / "lyrics.txt").write_text("la la la\n", encoding="utf-8")
    (tmp_path / "cast").mkdir(exist_ok=True)
    (tmp_path / "cast" / "dianne_ref.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    (tmp_path / "workflow_api.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        _RUN_TOML.format(tmp_path=tmp_path, comfyui_url=comfyui_url), encoding="utf-8"
    )
    return config_path


class _Usage:
    def __init__(self, free_gb: float) -> None:
        self.total = int(1000 * 1024**3)
        self.used = self.total - int(free_gb * 1024**3)
        self.free = int(free_gb * 1024**3)


def _disk_with(free_gb: float):
    def _usage(path: str) -> _Usage:
        return _Usage(free_gb)

    return _usage


class FakeProcess:
    """Stand-in for the ``Popen`` the default spawner returns."""

    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class RecordingSpawner:
    """Records what would have been spawned; starts nothing."""

    def __init__(self, *, pid: int = 4242, fail: bool = False) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, Path]] = []
        self.pid = pid
        self.fail = fail

    def __call__(self, argv, *, cwd: Path, log_path: Path) -> FakeProcess:
        self.calls.append((tuple(argv), cwd, log_path))
        if self.fail:
            raise OSError("simulated exec failure")
        return FakeProcess(pid=self.pid)


class RecordingSignaller:
    def __init__(self, *, missing: bool = False) -> None:
        self.calls: list[int] = []
        self.missing = missing

    def __call__(self, pgid: int) -> bool:
        self.calls.append(pgid)
        return not self.missing


def _report(**overrides) -> PrepareReport:
    defaults: dict[str, object] = dict(
        generated_at="2026-10-04",
        config_path="run.toml",
        chunk_count=80,
        voiced_chunk_count=41,
        instrumental_chunk_count=39,
        total_frames=12334,
        max_chunk_frames=192,
        max_chunk_frames_chunk_id=3,
        alignment_summary="alignment looked fine",
        alignment_finding_counts={},
    )
    defaults.update(overrides)
    return PrepareReport(**defaults)  # type: ignore[arg-type]


def _controller(
    tmp_path: Path,
    *,
    report: PrepareReport | None = None,
    free_vram_bytes: int = 23_000_000_000,
    free_disk_gb: float = 500.0,
    spawner: RecordingSpawner | None = None,
    signaller: RecordingSignaller | None = None,
    alive=None,
) -> tuple[control.RenderController, RecordingSpawner, RecordingSignaller]:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    session = FakeComfyUISession(base_url=config.comfyui_url)
    session.set_vram_free(free_vram_bytes)
    if report is not None:
        assert config.prepare_report_file is not None
        write_prepare_report(report, config.prepare_report_file)
    spawner = spawner or RecordingSpawner()
    signaller = signaller or RecordingSignaller()
    controller = control.RenderController(
        config,
        config_path,
        session=session,
        spawner=spawner,
        signaller=signaller,
        disk_usage=_disk_with(free_disk_gb),
        process_alive=alive if alive is not None else (lambda pid: True),
    )
    return controller, spawner, signaller


# --------------------------------------------------------------------------- #
# The command the page runs is the CLI, unchanged
# --------------------------------------------------------------------------- #


class TestRenderArgv:
    def test_it_runs_this_interpreters_own_cli_module(self, tmp_path: Path) -> None:
        """The whole point of spawning rather than calling in-process: the
        gates, the custody context manager and the unconditional ``POST
        /free`` are the CLI's, not a second copy of them."""
        controller, _, _ = _controller(tmp_path)
        argv = controller.render_argv(resume=False)
        assert argv[:3] == (sys.executable, "-m", "music_video_maker.cli")
        assert "--config" in argv
        assert "--resume" not in argv

    def test_resume_is_the_only_option_the_page_can_add(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path)
        assert "--resume" in controller.render_argv(resume=True)
        # No --only-chunks, no --reseed, no --acknowledge-*: a browser cannot
        # widen what the run does, only start the run this config describes.
        assert set(controller.render_argv(resume=True)) <= {
            sys.executable,
            "-m",
            "music_video_maker.cli",
            "--config",
            str(controller.config_path),
            "--resume",
        }


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class TestStartGate:
    def test_a_fully_prepared_healthy_run_is_allowed(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        gate = controller.evaluate_start()
        assert gate.allowed, gate.summary()
        assert {check.name for check in gate.checks} == {
            "one_run_at_a_time",
            "prepared",
            "disk",
            "vram",
            "envelope",
        }

    def test_an_unprepared_run_is_refused_and_names_the_command(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path)  # no report written
        gate = controller.evaluate_start()
        assert not gate.allowed
        failed = {check.name for check in gate.failures}
        assert failed == {"prepared"}
        assert "--prepare" in gate.summary()

    def test_a_stale_report_is_refused_naming_the_input_that_moved(
        self, tmp_path: Path
    ) -> None:
        config_path = _write_config(tmp_path)
        config = load_config(config_path)
        lyrics = Path(config.lyrics_file)
        stamp = InputStamp.of("lyrics_file", lyrics)
        assert config.prepare_report_file is not None
        write_prepare_report(_report(inputs=(stamp,)), config.prepare_report_file)
        # Edit the lyrics after the report was written: the report now
        # describes a timeline a render would no longer produce.
        time.sleep(0.01)
        lyrics.write_text("different words entirely\n", encoding="utf-8")

        session = FakeComfyUISession(base_url=config.comfyui_url)
        controller = control.RenderController(
            config,
            config_path,
            session=session,
            spawner=RecordingSpawner(),
            disk_usage=_disk_with(500.0),
        )
        gate = controller.evaluate_start()
        assert not gate.allowed
        assert "lyrics_file" in gate.summary()

    def test_a_report_with_no_frame_evidence_cannot_be_started_from(
        self, tmp_path: Path
    ) -> None:
        """A report written before ``max_chunk_frames`` existed cannot answer
        the issue #98 question, and "cannot answer" is not "nothing is too
        long" -- so the start is refused rather than ungated."""
        controller, _, _ = _controller(tmp_path, report=_report(max_chunk_frames=None))
        gate = controller.evaluate_start()
        assert not gate.allowed
        assert "max_chunk_frames" in gate.summary()

    def test_shot_plan_refusals_are_refused_here_rather_than_hours_in(
        self, tmp_path: Path
    ) -> None:
        controller, _, _ = _controller(
            tmp_path,
            report=_report(
                plan_checked="shot_plan.toml",
                plan_errors=("chunk 61 authored against start=460.333s",),
            ),
        )
        gate = controller.evaluate_start()
        assert not gate.allowed
        assert "460.333" in gate.summary()

    def test_critical_alignment_findings_only_block_under_strict_alignment(
        self, tmp_path: Path
    ) -> None:
        """Mirrors the CLI exactly: ``strict_alignment`` is what turns a
        CRITICAL finding into a refusal, and this gate does not invent a
        stricter policy than the run it is about to start."""
        lax, _, _ = _controller(
            tmp_path / "lax",
            report=_report(strict_alignment=False, alignment_finding_counts={"CRITICAL": 20}),
        )
        assert lax.evaluate_start().allowed

        strict, _, _ = _controller(
            tmp_path / "strict",
            report=_report(strict_alignment=True, alignment_finding_counts={"CRITICAL": 20}),
        )
        gate = strict.evaluate_start()
        assert not gate.allowed
        assert "strict_alignment" in gate.summary()

    def test_a_short_disk_refuses_with_the_reading(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report(), free_disk_gb=3.0)
        gate = controller.evaluate_start()
        assert not gate.allowed
        assert "3.00 GB free" in gate.summary()

    def test_a_held_card_refuses_with_the_reading(self, tmp_path: Path) -> None:
        """The issue #19 pre-flight, through ``custody.preflight_free_vram``
        -- the same function ``VramCustodyManager.__enter__`` calls."""
        controller, _, _ = _controller(
            tmp_path, report=_report(), free_vram_bytes=1_000_000_000
        )
        gate = controller.evaluate_start()
        assert not gate.allowed
        assert "0.93 GB free" in gate.summary()

    def test_an_unreachable_comfyui_degrades_exactly_as_the_cli_does(
        self, tmp_path: Path
    ) -> None:
        """Not a refusal, deliberately: the CLI's own pre-flight logs and
        proceeds on an unusable reading, and a gate stricter than the thing
        it gates is a policy nobody reviewed. The page says it could not
        read the card."""
        config_path = _write_config(tmp_path)
        config = load_config(config_path)
        assert config.prepare_report_file is not None
        write_prepare_report(_report(), config.prepare_report_file)

        class _Dead:
            def get(self, url: str, **kwargs):
                raise requests.ConnectionError("simulated: no route to host")

        controller = control.RenderController(
            config,
            config_path,
            session=_Dead(),
            spawner=RecordingSpawner(),
            disk_usage=_disk_with(500.0),
        )
        gate = controller.evaluate_start()
        assert gate.allowed
        vram = next(check for check in gate.checks if check.name == "vram")
        assert "no usable reading" in vram.detail

    def test_a_chunk_longer_than_anything_proven_is_refused(self, tmp_path: Path) -> None:
        """Issue #98's gate, evaluated from the report's own largest chunk --
        288 frames is ``max_chunk_seconds = 12.0``, which has never run."""
        controller, _, _ = _controller(
            tmp_path, report=_report(max_chunk_frames=288, max_chunk_frames_chunk_id=7)
        )
        gate = controller.evaluate_start()
        assert not gate.allowed
        summary = gate.summary()
        assert "288" in summary
        assert "chunk 7" in summary

    def test_the_envelope_gate_respects_the_same_acknowledgement_the_cli_does(
        self, tmp_path: Path
    ) -> None:
        config_path = _write_config(tmp_path)
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            # Above the [hardware] header: in TOML a bare key belongs to
            # whichever table precedes it.
            text.replace(
                "min_free_vram_gb = 16.0",
                "min_free_vram_gb = 16.0\nacknowledge_unproven_envelope = true",
            ),
            encoding="utf-8",
        )
        config = load_config(config_path)
        assert config.prepare_report_file is not None
        write_prepare_report(_report(max_chunk_frames=288), config.prepare_report_file)
        session = FakeComfyUISession(base_url=config.comfyui_url)
        controller = control.RenderController(
            config,
            config_path,
            session=session,
            spawner=RecordingSpawner(),
            disk_usage=_disk_with(500.0),
        )
        assert controller.evaluate_start().allowed

    def test_an_unresolvable_resolution_cannot_be_judged_and_says_so(
        self, tmp_path: Path
    ) -> None:
        """``read_render_dimensions`` answers ``(None, None)`` for a template
        with no H3 node, and a size gate that guesses the size is worse than
        no gate (``envelope.check_render_envelope``'s own rule)."""
        config_path = _write_config(tmp_path)
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            text.replace("render_width = 864\n", "").replace("render_height = 480\n", ""),
            encoding="utf-8",
        )
        config = load_config(config_path)
        assert config.prepare_report_file is not None
        write_prepare_report(_report(max_chunk_frames=288), config.prepare_report_file)
        controller = control.RenderController(
            config,
            config_path,
            session=FakeComfyUISession(base_url=config.comfyui_url),
            spawner=RecordingSpawner(),
            disk_usage=_disk_with(500.0),
        )
        gate = controller.evaluate_start()
        envelope = next(check for check in gate.checks if check.name == "envelope")
        assert envelope.passed
        assert "could not be resolved" in envelope.detail


# --------------------------------------------------------------------------- #
# Exactly one run at a time
# --------------------------------------------------------------------------- #


class TestOneRunAtATime:
    def test_starting_a_run_while_one_is_in_flight_is_refused_not_queued(
        self, tmp_path: Path
    ) -> None:
        """docs/design-web-ui.md's second day-one test, now that there is a
        start path for it to apply to."""
        controller, spawner, _ = _controller(tmp_path, report=_report())
        controller.start()
        assert len(spawner.calls) == 1

        with pytest.raises(control.StartRefused) as refused:
            controller.start()
        assert "one_run_at_a_time" in {c.name for c in refused.value.gate.failures}
        assert len(spawner.calls) == 1  # refused, never queued

    def test_a_second_controller_sees_the_claim_through_the_lock_file(
        self, tmp_path: Path
    ) -> None:
        """A restarted server must not start a second render over the first
        one, so the claim lives in a file beside the run state, not in
        memory."""
        first, spawner, _ = _controller(tmp_path, report=_report())
        first.start()

        second, second_spawner, _ = _controller(tmp_path, report=_report())
        assert second.status().state is control.RenderState.RUNNING
        with pytest.raises(control.StartRefused):
            second.start()
        assert second_spawner.calls == []

    def test_a_stale_lock_whose_process_is_gone_does_not_block_a_start(
        self, tmp_path: Path
    ) -> None:
        controller, spawner, _ = _controller(
            tmp_path, report=_report(), alive=lambda pid: False
        )
        controller.lock_path.parent.mkdir(parents=True, exist_ok=True)
        controller.lock_path.write_text(
            json.dumps({"pid": 999999, "pgid": 999999, "started_at": 1.0, "argv": []}),
            encoding="utf-8",
        )
        assert controller.status().state is control.RenderState.EXITED
        controller.start()
        assert len(spawner.calls) == 1

    def test_a_torn_lock_file_is_a_refusal_not_a_crash(self, tmp_path: Path) -> None:
        controller, spawner, _ = _controller(tmp_path, report=_report())
        controller.lock_path.parent.mkdir(parents=True, exist_ok=True)
        controller.lock_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(control.StartRefused):
            controller.start()
        assert spawner.calls == []

    def test_a_failed_spawn_releases_the_claim(self, tmp_path: Path) -> None:
        controller, spawner, _ = _controller(
            tmp_path, report=_report(), spawner=RecordingSpawner(fail=True)
        )
        with pytest.raises(control.ControlError):
            controller.start()
        assert not controller.lock_path.exists()
        assert controller.status().state is control.RenderState.IDLE

    def test_the_lock_and_log_live_beside_the_run_state(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        assert controller.lock_path.parent == controller.config.run_state_file.parent
        assert controller.log_path.parent == controller.config.run_state_file.parent


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


class TestStop:
    def test_stop_sends_exactly_one_sigint_to_the_childs_process_group(
        self, tmp_path: Path
    ) -> None:
        controller, _, signaller = _controller(tmp_path, report=_report())
        status = controller.start()
        assert status.claim is not None
        stopped = controller.stop()
        assert signaller.calls == [status.claim.pgid]
        assert stopped.claim is not None
        assert stopped.claim.stop_requested_at is not None

    def test_stopping_twice_sends_a_second_sigint_and_never_escalates(
        self, tmp_path: Path
    ) -> None:
        """Ctrl-C twice is a thing an operator does; SIGKILL is not something
        this module ever sends. Killing a render mid ``POST /free`` is how the
        card gets left holding 17 GB."""
        controller, _, signaller = _controller(tmp_path, report=_report())
        controller.start()
        controller.stop()
        controller.stop()
        assert len(signaller.calls) == 2
        assert set(signaller.calls) == {controller.status().claim.pgid}

    def test_stop_with_nothing_in_flight_is_refused(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        with pytest.raises(control.StopRefused):
            controller.stop()

    def test_stop_refuses_a_claim_this_process_did_not_create(self, tmp_path: Path) -> None:
        """Signalling a pid out of a file this process did not write is how a
        reused pid gets a SIGINT meant for a render. The refusal names the
        command to run instead."""
        first, _, _ = _controller(tmp_path, report=_report())
        first.start()

        second, _, second_signaller = _controller(tmp_path, report=_report())
        with pytest.raises(control.StopRefused) as refused:
            second.stop()
        assert "kill -INT" in str(refused.value)
        assert second_signaller.calls == []

    def test_stop_after_the_process_has_already_exited_is_refused(
        self, tmp_path: Path
    ) -> None:
        controller, _, signaller = _controller(tmp_path, report=_report())
        controller.start()
        controller._process = FakeProcess(pid=controller.status().claim.pid, returncode=0)
        controller._alive = lambda pid: False
        with pytest.raises(control.StopRefused):
            controller.stop()
        assert signaller.calls == []


# --------------------------------------------------------------------------- #
# Status and the log tail
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_an_untouched_run_is_idle(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        status = controller.status()
        assert status.state is control.RenderState.IDLE
        assert status.claim is None

    def test_a_finished_child_reports_its_exit_code(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        controller.start()
        controller._process = FakeProcess(pid=controller.status().claim.pid, returncode=2)
        controller._alive = lambda pid: False
        status = controller.status()
        assert status.state is control.RenderState.EXITED
        assert status.exit_code == 2

    def test_the_log_tail_is_the_last_lines_not_the_whole_file(self, tmp_path: Path) -> None:
        path = tmp_path / "render.log"
        path.write_text("".join(f"line {i}\n" for i in range(500)), encoding="utf-8")
        tail = control.read_log_tail(path, max_lines=5)
        assert tail.splitlines() == [f"line {i}" for i in range(495, 500)]

    def test_a_missing_log_tail_is_empty_never_an_error(self, tmp_path: Path) -> None:
        assert control.read_log_tail(tmp_path / "nope.log") == ""


# --------------------------------------------------------------------------- #
# The two claims that cannot be faked: a real child, a real SIGINT
# --------------------------------------------------------------------------- #

_CHILD = """
import signal, sys, time
from pathlib import Path
marker = Path(sys.argv[1])

def _on_sigint(signum, frame):
    marker.write_text("interrupted", encoding="utf-8")
    raise SystemExit(130)

signal.signal(signal.SIGINT, _on_sigint)
Path(sys.argv[2]).write_text("ready", encoding="utf-8")
for _ in range(600):
    time.sleep(0.05)
"""


@pytest.mark.skipif(os.name != "posix", reason="process groups and SIGINT are POSIX")
def test_a_real_child_is_its_own_process_group_leader_and_takes_sigint(
    tmp_path: Path,
) -> None:
    """``start_new_session=True`` plus ``killpg`` *is* Ctrl-C: a terminal
    sends SIGINT to the whole foreground process group, which is what gets
    ffmpeg and ComfyUI's client down with the orchestrator rather than
    leaving orphans behind. Nothing here renders anything -- the child is
    eight lines of python."""
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    marker = tmp_path / "marker.txt"
    ready = tmp_path / "ready.txt"
    log_path = tmp_path / "child.log"

    process = control.spawn_render(
        [sys.executable, str(script), str(marker), str(ready)],
        cwd=tmp_path,
        log_path=log_path,
    )
    try:
        deadline = time.monotonic() + 20.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "child never started"
        assert os.getpgid(process.pid) == process.pid

        assert control.signal_interrupt(process.pid) is True
        assert process.wait(timeout=20) == 130
        assert marker.read_text(encoding="utf-8") == "interrupted"
        assert log_path.exists()
    finally:
        if process.poll() is None:  # pragma: no cover - only on a failed assert above
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="process groups and SIGINT are POSIX")
def test_signalling_a_process_group_that_is_gone_reports_false_rather_than_raising() -> None:
    process = subprocess.Popen(  # noqa: S603 - this interpreter, a fixed argv
        [sys.executable, "-c", "pass"], start_new_session=True
    )
    process.wait(timeout=20)
    # The group is empty now; reaped children cannot be signalled.
    assert control.signal_interrupt(process.pid) is False


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_the_default_seams_are_the_real_ones(tmp_path: Path) -> None:
    """A controller built with no seams injected uses the real subprocess,
    the real signal and the real ``shutil.disk_usage`` -- so the test suite's
    injections are substitutions, not the only wiring that exists."""
    config_path = _write_config(tmp_path)
    controller = control.RenderController(load_config(config_path), config_path)
    assert controller._spawner is control.spawn_render
    assert controller._signaller is control.signal_interrupt
    assert controller._disk_usage is shutil.disk_usage
    assert controller._alive is control.process_alive


def test_process_alive_is_true_for_this_process_and_false_for_a_reaped_one() -> None:
    assert control.process_alive(os.getpid()) is True
    process = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    process.wait(timeout=20)
    assert control.process_alive(process.pid) is False


def test_signal_interrupt_sends_sigint_and_nothing_else(monkeypatch) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(control.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    assert control.signal_interrupt(1234) is True
    assert sent == [(1234, signal.SIGINT)]


# --------------------------------------------------------------------------- #
# The remaining edges
# --------------------------------------------------------------------------- #


class TestEdges:
    def test_a_passing_gate_summarises_as_such(self, tmp_path: Path) -> None:
        controller, _, _ = _controller(tmp_path, report=_report())
        assert controller.evaluate_start().summary() == "every pre-flight check passed"

    def test_a_lock_file_that_is_json_but_not_a_claim_is_refused(self, tmp_path: Path) -> None:
        controller, spawner, _ = _controller(tmp_path, report=_report())
        controller.lock_path.parent.mkdir(parents=True, exist_ok=True)
        controller.lock_path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        assert controller.status().state is control.RenderState.EXITED
        assert controller.status().claim is None
        with pytest.raises(control.StartRefused) as refused:
            controller.start()
        assert "does not contain a render claim" in refused.value.gate.summary()
        assert spawner.calls == []

    def test_a_claim_appearing_between_the_gate_and_the_spawn_starts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate and the claim are two different races: this is the one
        the exclusive O_EXCL create exists for (a second server, or a
        hand-run CLI, between the checks and the claim)."""
        controller, spawner, _ = _controller(
            tmp_path, report=_report(), alive=lambda pid: False
        )
        controller.lock_path.parent.mkdir(parents=True, exist_ok=True)
        controller.lock_path.write_text(
            json.dumps({"pid": 999999, "pgid": 999999, "started_at": 1.0}), encoding="utf-8"
        )
        # The claim is stale, so the gate passes -- but something recreates it
        # before the exclusive create.
        monkeypatch.setattr(controller, "_clear_exited_claim", lambda: None)
        with pytest.raises(control.ControlError, match="appeared between"):
            controller.start()
        assert spawner.calls == []

    def test_a_resolution_that_needs_an_unreadable_template_cannot_be_judged(
        self, tmp_path: Path
    ) -> None:
        config_path = _write_config(tmp_path)
        text = config_path.read_text(encoding="utf-8")
        config_path.write_text(
            text.replace("render_width = 864\n", "").replace("render_height = 480\n", ""),
            encoding="utf-8",
        )
        config = load_config(config_path)
        assert config.prepare_report_file is not None
        write_prepare_report(_report(max_chunk_frames=288), config.prepare_report_file)
        # Loadable at config time, gone by the time the gate looks: the gate
        # reports "cannot judge" rather than guessing or raising.
        Path(config.workflow_template).unlink()
        controller = control.RenderController(
            config,
            config_path,
            session=FakeComfyUISession(base_url=config.comfyui_url),
            spawner=RecordingSpawner(),
            disk_usage=_disk_with(500.0),
        )
        gate = controller.evaluate_start()
        envelope = next(check for check in gate.checks if check.name == "envelope")
        assert envelope.passed
        assert "could not be resolved" in envelope.detail

    def test_with_no_session_injected_it_builds_a_real_requests_session(
        self, tmp_path: Path
    ) -> None:
        """Nothing is sent here -- the session is built, not used. What this
        pins is that the default wiring is a real one, so the injected fakes
        above are substitutions rather than the only path that exists."""
        config_path = _write_config(tmp_path)
        controller = control.RenderController(load_config(config_path), config_path)
        assert isinstance(controller._session_or_default(), requests.Session)


def test_a_finished_child_this_server_spawned_is_not_read_as_still_running(
    tmp_path: Path,
) -> None:
    """Nothing here ``wait()``s the render, so a finished child is a zombie
    until it is reaped -- and ``os.kill(pid, 0)`` **succeeds** on a zombie.
    Asking the signal would report a finished render as in flight and refuse
    every later start for the life of the server, so liveness for a child we
    spawned comes from ``Popen.poll()`` (which also reaps it).

    The fake here is exactly that disagreement: the pid probe says alive, the
    process object says it exited."""
    controller, spawner, _ = _controller(
        tmp_path, report=_report(), alive=lambda pid: True
    )
    controller.start()
    pid = controller.status().claim.pid
    controller._process = FakeProcess(pid=pid, returncode=0)

    status = controller.status()
    assert status.state is control.RenderState.EXITED
    assert status.exit_code == 0
    # And the next start is not blocked by the claim of a run that is over.
    assert controller.evaluate_start().allowed
    controller.start()
    assert len(spawner.calls) == 2
