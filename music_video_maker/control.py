"""Start one run, stop one run -- the control half of issue #36's monitor.

:mod:`music_video_maker.webui` is the socket and the routes;
:mod:`music_video_maker.progress` is the reader; this module is the only
place in the project that will *begin* a render on somebody else's behalf,
and almost all of it is refusals. It imports no ``http`` and no ``socket``
for the same reason ``progress.py`` does not: a module that cannot listen
cannot be reached, and the thing that holds the gates should not also be
holding a connection.

Why it starts a subprocess rather than calling ``run_pipeline``
------------------------------------------------------------------
``cli.run_pipeline`` already does every dangerous part correctly: the issue
#19 custody pre-flight, ``prevent_host_sleep``, the issue #10 disk check,
the issues #24/#98 render-envelope refusal, the between-chunk VRAM re-check,
and -- the one that matters most -- the unconditional ``POST /free`` in a
``finally``. Calling it in a thread inside a long-lived HTTP server would
put all of that behind a shutdown path nobody has tested, and would make
"stop the render" mean "raise an exception in a worker thread", which Python
does not offer honestly.

So a start is ``python -m music_video_maker.cli --config <the same file>``,
spawned in its own session, and that gives three things for free:

* **the gates are the CLI's**, not a second copy that drifts from them;
* **a stop is a SIGINT to the child's process group, which is literally what
  Ctrl-C is** (a terminal sends SIGINT to the whole foreground group), so
  the pipeline's existing ``KeyboardInterrupt`` path -- persist run state,
  release custody in the ``finally`` -- runs exactly as it does today. There
  is no second shutdown path, because there did not need to be one;
* a render that wedges the host takes the server's child with it rather than
  the server, and the run state on disk is still the source of truth.

:func:`RenderController.render_argv` can add exactly one flag,
``--resume``. A browser cannot pass ``--only-chunks``, ``--reseed`` or
anything that acknowledges a safety gate.

The gate, and the one place it is deliberately stricter than the CLI
-----------------------------------------------------------------------
:meth:`RenderController.evaluate_start` runs five checks and refuses --
never queues -- if any fails. Four of them are the run's own gates, called
through the run's own functions so there is one copy of each rule:

* ``one_run_at_a_time`` -- GPU custody is exclusive. A claim file beside the
  run state holds the pid, so a *restarted* server still sees an in-flight
  render instead of starting a second one over it.
* ``disk`` -- :func:`resilience.preflight_disk_check`, the #10 check.
* ``vram`` -- :func:`custody.preflight_free_vram`, the #19 pre-flight.
  Unreadable degrades rather than refusing, because that is what the CLI's
  own pre-flight decides and a gate stricter than the thing it gates is a
  policy nobody reviewed. The page shows that it could not read the card.
* ``envelope`` -- :func:`envelope.check_render_envelope`, the #24/#98 size
  refusal, evaluated over the largest chunk the last ``--prepare``
  recorded. One chunk is enough: ``EnvelopePoint.covers`` is per axis, so at
  a fixed resolution the longest chunk decides the whole run.

The fifth, ``prepared``, is this module's own and has no CLI equivalent: a
start is refused unless a prepare report exists, this build can read it, it
records a frame count, and every input it was computed from is unchanged on
disk. Three reasons it is worth being stricter here than the CLI:

1. It is the half of issue #36 that matters more. "A 100-minute mistake
   turned into a 10-second one" is the whole argument for the page, and a
   button that skips the review is the mistake with a shorter path to it.
2. It is what makes the ``envelope`` check above possible at all without
   recomputing the timeline -- which the monitor must never do, because
   Stage 2 writes chunk audio into the live run's own directory.
3. The remedy is ~50 s of CPU and no GPU, and the page prints the command.

A shot-plan refusal or (under ``strict_alignment``) a CRITICAL alignment
finding also fails this check, because the run itself would refuse on both.
Everything else the report carries -- warnings, notices, drift -- is shown,
never gated on: a page that refuses what the CLI accepts teaches operators
to go round it.

What this module writes
--------------------------
Two files, beside the run state (``run_state.json``'s own directory, where
``--prepare`` already writes its report): ``webui_render.lock``, the claim,
and ``webui_render.log``, the child's stdout and stderr. This is the
deliberate exception to ``webui.py``'s "the monitor never writes into the
run's directory" -- which is exactly why it is a separate module: the
read-only half still writes nothing at all.

What it will not do
----------------------
* It will not **stop a claim it did not create.** A pid read out of a file
  this process did not write could have been reused by something unrelated,
  and SIGINT to a stranger's process group is not a recoverable mistake.
  The refusal names ``kill -INT <pid>`` instead.
* It will not **escalate.** Repeating a stop sends another SIGINT, the way
  pressing Ctrl-C twice does. Nothing here sends SIGKILL: killing a render
  inside its ``POST /free`` is how the card gets left holding 17 GB, which
  is the incident ``docs/design-web-ui.md`` records.
* It will not **touch the card itself.** Stopping the GPU's other tenants
  stays manual, per ``custody.py``'s module docstring. A button that paused
  somebody's inference server is the automation this project decided never
  to build.
* It will not **write the config.** Issue #36's "what the page collects"
  half is still not built; the TOML stays the source of truth and a human
  edits it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from music_video_maker import custody, envelope, resilience
from music_video_maker.config import RunConfig
from music_video_maker.prepare_report import (
    PrepareReport,
    PrepareReportError,
    read_prepare_report,
    stale_inputs,
)
from music_video_maker.workflow_graph import (
    WorkflowTemplateError,
    load_workflow_template,
    read_render_dimensions,
)

logger = logging.getLogger(__name__)

LOCK_FILENAME = "webui_render.lock"
"""The claim, beside the run state. Holds the child's pid and process-group
id so a restarted server sees an in-flight render rather than starting a
second one over it."""

LOG_FILENAME = "webui_render.log"
"""The child's stdout and stderr, appended. This is how a refusal that
happens *inside* the run -- a per-chunk envelope miss, a config error, a
strict-alignment stop -- reaches the page: the gate below cannot evaluate
those, and pretending otherwise would be a second copy of each."""

PREPARE_COMMAND = "music-video-maker --config <run.toml> --prepare"


class ControlError(RuntimeError):
    """A start or stop could not be carried out (as opposed to refused)."""


class StopRefused(ControlError):
    """There is nothing this server may stop. Carries the reason as text."""


class StartRefused(ControlError):
    """A gate said no. ``gate`` is the full :class:`StartGate` so the caller
    can show every check, not only the one that failed first."""

    def __init__(self, gate: StartGate) -> None:
        super().__init__(gate.summary())
        self.gate = gate


# --------------------------------------------------------------------------- #
# Process seams -- real by default, injected in tests
# --------------------------------------------------------------------------- #


class RenderProcess(Protocol):
    """The slice of ``subprocess.Popen`` this module uses."""

    pid: int

    def poll(self) -> int | None: ...  # pragma: no cover - Protocol


Spawner = Callable[..., RenderProcess]
Signaller = Callable[[int], bool]


def spawn_render(argv: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """Start ``argv`` detached into its own session, with stdout and stderr
    appended to ``log_path`` and stdin closed.

    ``start_new_session=True`` is load-bearing twice over. It makes the child
    a process-group leader, so :func:`signal_interrupt` can deliver SIGINT to
    the whole group -- the render plus any ffmpeg it has spawned -- which is
    what a terminal does on Ctrl-C. And it detaches the child from this
    server's own session, so an operator pressing Ctrl-C on ``mvm-webui``
    does not take a 100-minute render down with it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    try:
        return subprocess.Popen(  # noqa: S603 - argv is built by render_argv, never by a request
            list(argv),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # The child holds its own duplicate of the descriptor.
        handle.close()


def signal_interrupt(pgid: int) -> bool:
    """SIGINT to the process group ``pgid``; ``False`` if it is already gone.

    SIGINT and nothing else, ever. The pipeline's ``KeyboardInterrupt``
    handling is what releases the card in a ``finally``, and a signal that
    skips it is a worse outcome than a render that takes a few more seconds
    to stop."""
    try:
        os.killpg(pgid, signal.SIGINT)
    except (ProcessLookupError, PermissionError) as exc:
        logger.info("Could not signal process group %s (%s)", pgid, exc)
        return False
    return True


def process_alive(pid: int) -> bool:
    """Whether ``pid`` exists. ``PermissionError`` counts as alive: the
    process is there, it is simply not ours to signal."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover - defensive
        return False
    return True


def read_log_tail(path: Path | str, *, max_lines: int = 40, max_bytes: int = 64_000) -> str:
    """The last ``max_lines`` lines of ``path``, or ``""`` if it cannot be
    read. Never raises: a log tail is a convenience on a page, and a missing
    one is an ordinary state (nothing has been started yet)."""
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
        with resolved.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # drop the partial line the seek landed in
            raw = handle.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateCheck:
    """One pre-flight question, its verdict, and the numbers behind it.

    ``detail`` is written for an operator reading a browser page, so it
    carries the reading and the remedy rather than a code. A check that
    could not be *made* passes with a detail saying so -- "unknown" is never
    silently reported as "fine", but it is also not a refusal, which is the
    stance ``envelope.check_render_envelope`` and ``custody``'s pre-flight
    both already take."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class StartGate:
    checks: tuple[GateCheck, ...]

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def allowed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        """One line per failed check, or a line saying every check passed."""
        if self.allowed:
            return "every pre-flight check passed"
        return "\n".join(f"{check.name}: {check.detail}" for check in self.failures)


class RenderState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    EXITED = "exited"


@dataclass(frozen=True)
class RenderClaim:
    """The contents of the lock file: who holds the card, since when."""

    pid: int
    pgid: int
    started_at: float
    argv: tuple[str, ...] = ()
    resume: bool = False
    stop_requested_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "started_at": self.started_at,
            "argv": list(self.argv),
            "resume": self.resume,
            "stop_requested_at": self.stop_requested_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> RenderClaim:
        if not isinstance(raw, dict) or "pid" not in raw:
            raise ValueError("not a render claim")
        pid = int(raw["pid"])
        return cls(
            pid=pid,
            pgid=int(raw.get("pgid", pid)),
            started_at=float(raw.get("started_at", 0.0)),
            argv=tuple(str(one) for one in raw.get("argv", ()) or ()),
            resume=bool(raw.get("resume", False)),
            stop_requested_at=(
                None
                if raw.get("stop_requested_at") is None
                else float(raw["stop_requested_at"])
            ),
        )


@dataclass(frozen=True)
class RenderStatus:
    state: RenderState
    claim: RenderClaim | None = None
    exit_code: int | None = None
    """Only knowable for a child *this* process spawned. ``None`` after a
    server restart -- the log tail is the evidence then, and saying ``None``
    is more honest than inventing a zero."""
    owned: bool = False
    """Whether the claim belongs to a child this process spawned, which is
    the precondition for :meth:`RenderController.stop`."""
    log_path: Path | None = None


class LockError(ControlError):
    """The claim file exists but cannot be understood, so this module will
    not decide whether a render is in flight. Refusing is the safe direction:
    starting a second render over a live one is the unrecoverable mistake."""


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #


class RenderController:
    """One run's control surface: gate, start, stop, status."""

    def __init__(
        self,
        config: RunConfig,
        config_path: Path | str,
        *,
        session: Any = None,
        spawner: Spawner = spawn_render,
        signaller: Signaller = signal_interrupt,
        disk_usage: resilience.DiskUsage = shutil.disk_usage,
        process_alive: Callable[[int], bool] = process_alive,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if config.run_state_file is None:  # pragma: no cover - load_config always resolves it
            raise ControlError("config.run_state_file is unresolved; cannot site the claim file")
        self.config = config
        self.config_path = Path(config_path)
        self._session = session
        self._spawner = spawner
        self._signaller = signaller
        self._disk_usage = disk_usage
        self._alive = process_alive
        self._clock = clock
        self._mutex = threading.Lock()
        self._process: RenderProcess | None = None
        run_dir = Path(config.run_state_file).parent
        self.lock_path = run_dir / LOCK_FILENAME
        self.log_path = run_dir / LOG_FILENAME

    # -- the command ------------------------------------------------------- #

    def render_argv(self, *, resume: bool = False) -> tuple[str, ...]:
        """The CLI invocation a start runs. ``--resume`` is the only option a
        request can influence."""
        argv = [
            sys.executable,
            "-m",
            "music_video_maker.cli",
            "--config",
            str(self.config_path),
        ]
        if resume:
            argv.append("--resume")
        return tuple(argv)

    # -- status ------------------------------------------------------------ #

    def read_claim(self) -> RenderClaim | None:
        """The claim on disk, or ``None`` when there is no lock file.

        Raises :class:`LockError` for a lock file that exists and cannot be
        parsed -- see that class for why that is a refusal rather than a
        shrug."""
        try:
            raw = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise LockError(
                f"{self.lock_path} exists but could not be read ({exc}) -- refusing to decide "
                "whether a render is in flight. Check for a running "
                "'music_video_maker.cli' process, then delete the file if there is none."
            ) from exc
        try:
            return RenderClaim.from_dict(raw)
        except (TypeError, ValueError) as exc:
            raise LockError(
                f"{self.lock_path} does not contain a render claim ({exc}) -- refusing to "
                "decide whether a render is in flight. Delete it if no render is running."
            ) from exc

    def _liveness(self, claim: RenderClaim) -> tuple[bool, bool, int | None]:
        """``(owned, alive, exit_code)`` for ``claim``.

        For a child **this** process spawned, liveness comes from
        ``Popen.poll()`` and not from ``os.kill(pid, 0)``. That is not a
        preference: nothing here ever ``wait()``s the render, so a finished
        child is a **zombie** until it is reaped -- and a zombie is still a
        pid that ``os.kill(pid, 0)`` succeeds on. Asking the signal would
        report a finished render as still in flight and refuse every
        subsequent start for the life of the server. ``poll()`` both answers
        correctly and reaps it, and it is where the exit code comes from.

        For a claim written by some *other* process (a server restart, a
        hand-run CLI) there is no handle to poll, so the pid probe is all
        there is -- and an orphaned child is reparented to init and reaped
        there, so the zombie window does not apply to it."""
        if self._process is not None and self._process.pid == claim.pid:
            exit_code = self._process.poll()
            return True, exit_code is None, exit_code
        return False, self._alive(claim.pid), None

    def status(self) -> RenderStatus:
        """What this run's control half currently is. Never raises: a torn
        lock file reports as ``EXITED`` with no claim, and the *refusal* it
        causes happens in :meth:`start`, which is where it is actionable."""
        try:
            claim = self.read_claim()
        except LockError:
            logger.warning("Unreadable claim file at %s", self.lock_path, exc_info=True)
            return RenderStatus(state=RenderState.EXITED, log_path=self.log_path)
        if claim is None:
            return RenderStatus(state=RenderState.IDLE, log_path=self.log_path)
        owned, alive, exit_code = self._liveness(claim)
        if alive:
            return RenderStatus(
                state=RenderState.RUNNING,
                claim=claim,
                owned=owned,
                log_path=self.log_path,
            )
        return RenderStatus(
            state=RenderState.EXITED,
            claim=claim,
            exit_code=exit_code,
            owned=owned,
            log_path=self.log_path,
        )

    # -- the gate ---------------------------------------------------------- #

    def evaluate_start(self, *, resume: bool = False) -> StartGate:
        """Run every pre-flight check and report all of them.

        Every check runs even when an earlier one has already failed, so the
        page shows an operator the whole picture rather than one thing at a
        time -- except for ``one_run_at_a_time``, which short-circuits: there
        is nothing to say about free VRAM during a render that is using it,
        and probing ComfyUI once per page load while it renders is noise."""
        in_flight = self._one_run_check()
        if not in_flight.passed:
            return StartGate((in_flight,))
        report, prepared = self._prepared_check()
        return StartGate(
            (
                in_flight,
                prepared,
                self._disk_check(),
                self._vram_check(),
                self._envelope_check(report),
            )
        )

    def _one_run_check(self) -> GateCheck:
        try:
            claim = self.read_claim()
        except LockError as exc:
            return GateCheck("one_run_at_a_time", False, str(exc))
        if claim is not None and self._liveness(claim)[1]:
            since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(claim.started_at))
            return GateCheck(
                "one_run_at_a_time",
                False,
                f"a render is already in flight (pid {claim.pid}, started {since}) and GPU "
                "custody is exclusive -- this is refused, not queued. Stop that run first.",
            )
        if claim is not None:
            return GateCheck(
                "one_run_at_a_time",
                True,
                f"the previous run (pid {claim.pid}) has exited; its claim will be cleared.",
            )
        return GateCheck("one_run_at_a_time", True, "no render is in flight.")

    def _prepared_check(self) -> tuple[PrepareReport | None, GateCheck]:
        path = self.config.prepare_report_file
        if path is None:  # pragma: no cover - load_config always resolves it
            return None, GateCheck(
                "prepared", False, "this config resolves no prepare-report path."
            )
        try:
            report = read_prepare_report(path)
        except PrepareReportError as exc:
            return None, GateCheck(
                "prepared",
                False,
                f"{exc} -- run `{PREPARE_COMMAND}` (Stages 1-2 only: ~50 s of CPU, no GPU) "
                "and review /prepare before starting a render from here.",
            )
        changed = stale_inputs(report)
        if changed:
            return report, GateCheck(
                "prepared",
                False,
                f"{', '.join(changed)} changed on disk since this report was written, so it "
                f"describes a timeline a render would no longer produce. Re-run "
                f"`{PREPARE_COMMAND}`.",
            )
        if report.max_chunk_frames is None:
            return report, GateCheck(
                "prepared",
                False,
                "this report records no max_chunk_frames, so the render-envelope check "
                f"(issues #24/#98) cannot be evaluated from it. Re-run `{PREPARE_COMMAND}`.",
            )
        if report.plan_errors:
            return report, GateCheck(
                "prepared",
                False,
                f"the shot plan does not resolve against this timeline: "
                f"{'; '.join(report.plan_errors)}",
            )
        if report.strict_alignment and report.critical_finding_count:
            return report, GateCheck(
                "prepared",
                False,
                f"{report.critical_finding_count} CRITICAL alignment finding(s) and "
                "strict_alignment is on, so the run would refuse itself. See /prepare.",
            )
        return report, GateCheck(
            "prepared",
            True,
            f"prepared {report.generated_at}: {report.chunk_count} chunk(s), longest "
            f"{report.max_chunk_frames} frames, {report.critical_finding_count} critical and "
            f"{report.warning_finding_count} warning alignment finding(s). Review /prepare "
            "before you start -- nothing below re-checks what it found.",
        )

    def _disk_check(self) -> GateCheck:
        try:
            free_gb = resilience.preflight_disk_check(
                self.config.chunks_dir,
                min_free_disk_gb=self.config.min_free_disk_gb,
                disk_usage=self._disk_usage,
            )
        except resilience.DiskPreflightError as exc:
            return GateCheck("disk", False, str(exc))
        return GateCheck(
            "disk",
            True,
            f"{free_gb:.2f} GB free at {self.config.chunks_dir} "
            f"(>= {self.config.min_free_disk_gb:.2f} GB).",
        )

    def _vram_check(self) -> GateCheck:
        try:
            free_gb = custody.preflight_free_vram(
                self._session_or_default(),
                self.config.comfyui_url,
                min_free_vram_gb=self.config.min_free_vram_gb,
                hardware=self.config.hardware,
            )
        except custody.CustodyError as exc:
            return GateCheck("vram", False, str(exc))
        if free_gb is None:
            return GateCheck(
                "vram",
                True,
                f"no usable reading from {self.config.comfyui_url}/system_stats -- the same "
                "degradation the CLI's own pre-flight makes (it logs and proceeds). The run "
                "re-checks the card before every chunk it submits.",
            )
        return GateCheck(
            "vram",
            True,
            f"{free_gb:.2f} GB free on the card (>= {self.config.min_free_vram_gb:.2f} GB "
            "required). Point-in-time: anything else that uses this GPU must already be "
            "stopped by hand.",
        )

    def _envelope_check(self, report: PrepareReport | None) -> GateCheck:
        if report is None or report.max_chunk_frames is None:
            return GateCheck(
                "envelope",
                True,
                "not evaluated: no prepare report records this timeline's longest chunk. "
                "The run's own per-chunk check still runs before any GPU work.",
            )
        width, height = self._render_dimensions()
        largest = _LargestChunk(
            chunk_id=report.max_chunk_frames_chunk_id
            if report.max_chunk_frames_chunk_id is not None
            else -1,
            frame_count=report.max_chunk_frames,
        )
        try:
            envelope.check_render_envelope(
                [largest],
                hardware_name=self.config.hardware.name,
                width=width,
                height=height,
                acknowledged=self.config.acknowledge_unproven_envelope,
            )
        except envelope.UnprovenEnvelopeError as exc:
            # Prefixed because the gate hands the check exactly one chunk --
            # the longest -- so its own "1 of 1 chunk(s)" would otherwise
            # read as a claim about the timeline's size.
            return GateCheck(
                "envelope",
                False,
                "judged on the longest chunk the last --prepare recorded, which is the one "
                f"that decides every chunk at this resolution: {exc}",
            )
        if width is None or height is None:
            return GateCheck(
                "envelope",
                True,
                "this run's render resolution could not be resolved from the config or the "
                "workflow template, so the size gate cannot judge it -- a gate that guesses "
                "the size is worse than no gate. The run checks again before any GPU work.",
            )
        return GateCheck(
            "envelope",
            True,
            f"the longest chunk ({largest.frame_count} frames at {width}x{height}, chunk "
            f"{largest.chunk_id}) is inside what {self.config.hardware.name} has been shown "
            "to render. Per axis, so every shorter chunk is covered too.",
        )

    def _render_dimensions(self) -> tuple[int | None, int | None]:
        """The resolution this run would submit at, resolved the way
        ``cli._render_one_timeline`` resolves it: the config when it sets
        one, the template's own value when it does not."""
        if self.config.render_width is not None and self.config.render_height is not None:
            return self.config.render_width, self.config.render_height
        try:
            template = load_workflow_template(self.config.workflow_template)
        except WorkflowTemplateError:
            logger.warning(
                "Could not read %s to resolve this run's render resolution -- the "
                "render-envelope gate will report that it cannot judge the size",
                self.config.workflow_template,
                exc_info=True,
            )
            return self.config.render_width, self.config.render_height
        width, height = read_render_dimensions(template)
        return (
            self.config.render_width if self.config.render_width is not None else width,
            self.config.render_height if self.config.render_height is not None else height,
        )

    def _session_or_default(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    # -- start / stop ------------------------------------------------------ #

    def start(self, *, resume: bool = False) -> RenderStatus:
        """Gate, claim, spawn. Raises :class:`StartRefused` when any check
        fails, and :class:`ControlError` when the claim or the spawn itself
        could not be carried out.

        The mutex and the exclusive claim file are two different races: the
        first stops two requests to *this* server, the second stops two
        servers (or a server and a hand-run CLI that also wrote a claim)."""
        with self._mutex:
            gate = self.evaluate_start(resume=resume)
            if not gate.allowed:
                logger.warning("Refusing to start a render:\n%s", gate.summary())
                raise StartRefused(gate)
            self._clear_exited_claim()
            handle = self._claim_exclusively()
            argv = self.render_argv(resume=resume)
            try:
                process = self._spawner(argv, cwd=self.config_path.parent, log_path=self.log_path)
            except OSError as exc:
                os.close(handle)
                self.lock_path.unlink(missing_ok=True)
                raise ControlError(f"could not start {argv[0]}: {exc}") from exc
            self._process = process
            claim = RenderClaim(
                pid=process.pid,
                pgid=process.pid,  # start_new_session makes the child its own group leader
                started_at=self._clock(),
                argv=argv,
                resume=resume,
            )
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(claim.to_dict(), fh)
            logger.warning(
                "Started a render from the web monitor: pid %s, resume=%s, argv=%s, log=%s. "
                "This process now holds exclusive GPU custody.",
                claim.pid,
                resume,
                " ".join(argv),
                self.log_path,
            )
            return self.status()

    def stop(self) -> RenderStatus:
        """SIGINT the child's process group -- what Ctrl-C does -- and record
        that it was asked for. Raises :class:`StopRefused` when there is
        nothing this server may stop."""
        with self._mutex:
            status = self.status()
            if status.claim is None:
                raise StopRefused(
                    "no render is in flight, so there is nothing to stop. (If a render is "
                    "running that this page did not start, it has no claim file here.)"
                )
            claim = status.claim
            if status.state is not RenderState.RUNNING:
                raise StopRefused(
                    f"the render (pid {claim.pid}) has already exited; see {self.log_path}."
                )
            if not status.owned:
                raise StopRefused(
                    f"this server did not start the render in flight (pid {claim.pid} was "
                    f"claimed before this process began), and signalling a pid out of a file "
                    f"it did not write risks interrupting an unrelated process that reused "
                    f"that pid. Stop it with Ctrl-C in its own terminal, or `kill -INT "
                    f"{claim.pgid}` once you have confirmed what it is."
                )
            delivered = self._signaller(claim.pgid)
            stopped = RenderClaim(
                pid=claim.pid,
                pgid=claim.pgid,
                started_at=claim.started_at,
                argv=claim.argv,
                resume=claim.resume,
                stop_requested_at=self._clock(),
            )
            self._write_claim(stopped)
            logger.warning(
                "Sent SIGINT to the render's process group (%s, delivered=%s). The pipeline "
                "persists run state before every chunk and releases GPU custody in a finally, "
                "so this is the same stop as Ctrl-C. Nothing here will escalate to SIGKILL.",
                claim.pgid,
                delivered,
            )
            return self.status()

    # -- claim file -------------------------------------------------------- #

    def _clear_exited_claim(self) -> None:
        claim = self.read_claim()
        if claim is not None and not self._liveness(claim)[1]:
            logger.info(
                "Clearing the claim of pid %s, which is no longer running", claim.pid
            )
            self.lock_path.unlink(missing_ok=True)
            if self._process is not None and self._process.pid == claim.pid:
                self._process = None

    def _claim_exclusively(self) -> int:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ControlError(
                f"{self.lock_path} appeared between the pre-flight checks and the claim -- "
                "something else is starting a render for this config. Nothing was started."
            ) from exc
        except OSError as exc:
            raise ControlError(f"could not write the claim file {self.lock_path}: {exc}") from exc

    def _write_claim(self, claim: RenderClaim) -> None:
        self.lock_path.write_text(json.dumps(claim.to_dict()), encoding="utf-8")


@dataclass(frozen=True)
class _LargestChunk:
    """The one chunk the envelope gate judges: duck-typed to what
    ``envelope.unproven_chunks`` reads off a real ``AudioChunk``
    (``chunk_id``, ``frame_count``) and nothing else."""

    chunk_id: int
    frame_count: int


__all__ = [
    "LOCK_FILENAME",
    "LOG_FILENAME",
    "ControlError",
    "GateCheck",
    "LockError",
    "RenderClaim",
    "RenderController",
    "RenderState",
    "RenderStatus",
    "StartGate",
    "StartRefused",
    "StopRefused",
    "process_alive",
    "read_log_tail",
    "signal_interrupt",
    "spawn_render",
]
