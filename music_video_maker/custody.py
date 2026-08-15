"""GPU custody protocol (issue #19).

A render takes exclusive custody of the card for hours, and this module is
the *machine-checkable* half of that: assert the card is actually free before
submitting anything, and release it (``POST /free``) unconditionally when the
run ends, exception or not (context-manager semantics give the try/finally
for free).

Whatever else holds your GPU -- a desktop session, an inference server, a
scheduled job -- has to be stopped by hand before the run and started again
after. That step is deliberately outside this codebase: it is specific to a
machine's tenant list, which changes, and every attempt to automate one
particular tenant's pause/resume ends up shipping somebody's personal
infrastructure as everyone's default. What the pipeline *can* check is
whether the card is free right now, so that is what it checks -- and it
checks again between chunks (see :data:`VramProbe`), because a card that
reads free at the pre-flight is not one that stays free.

:class:`VramCustodyManager` runs a real, best-effort pre-flight free-VRAM
assertion against ``GET /system_stats`` on ``__enter__``, mirroring
``resilience.ResilientRunner._preflight_disk_check``: a clearly insufficient
reading refuses to start the run (custody is supposed to mean *exclusive*
GPU access -- see ``contracts.HardwareProfile``), while ComfyUI being
unreachable or returning something unparseable degrades gracefully (logged,
non-fatal) rather than blocking a run over a best-effort check.

It implements :class:`music_video_maker.contracts.CustodyManager`
structurally (a ``Protocol``, no explicit subclassing needed), and
:func:`build_custody_manager` is the single factory ``cli.py`` calls.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import requests

from music_video_maker.contracts import HardwareProfile

if TYPE_CHECKING:
    from music_video_maker.config import RunConfig

logger = logging.getLogger(__name__)

SYSTEM_STATS_PATH = "/system_stats"
FREE_PATH = "/free"

DEFAULT_MIN_FREE_VRAM_GB = 16.0
"""Absolute free-VRAM floor in GB, mirroring ``config.min_free_disk_gb``.

Set to **the lowest free-VRAM figure at which a render has ever demonstrably
succeeded**: ~16.4 GB, measured repeatedly on doris with the desktop session
and two other GPU services resident. That is the only defensible line, because
everything below it is untested and everything at or above it is proven.

For scale, ComfyUI's own load lines report ``MiniMaxH3`` staging 19995 MB,
plus a 14956 MB text encoder and ~5.5 GB of VAEs. Renders nonetheless succeed
at 16.4 GB free because ComfyUI offloads between stages rather than holding
them co-resident -- which is also why the blueprint's "~42.5 GB quantized
appetite" never materialized. Do **not** read 19995 MB as the floor; the
measurements say otherwise.

This number has now been wrong in both directions, which is the interesting
part. It began as a *fraction of nominal VRAM* (0.9, i.e. 21.6 GB) -- an
assumption, and too strict. It was then set to 12.0, which was below anything
ever demonstrated, and too loose in the dangerous direction: on 2026-08-07 it
green-lit a run onto a contended card, and the load went **silent** rather
than raising CUDA OOM, wedging the host badly enough to need a power cycle.

The lesson worth keeping is not a number but a rule: set the floor at the
worst configuration that has actually been observed to work, never below it on
the theory that less might do, and never above it on the theory that more
ought to be needed. A pre-flight that passes a run which cannot fit is worse
than no pre-flight at all, because it converts a fast, legible failure into a
hung machine.

This check is also *point-in-time*. It cannot see another process claiming the
card after the run starts, which is how that incident began -- doris shares
one 4090 between four independent workloads. See
``RunConfig.min_free_vram_gb``; this constant is only the fallback when no
config value is threaded through."""

class CustodyError(RuntimeError):
    """Raised when GPU custody could not be established or confirmed: the
    pre-flight free-VRAM assertion got a usable reading that is clearly
    insufficient. Mirrors ``resilience.DiskPreflightError`` -- refuse loudly
    up front rather than let a render run straight into a CUDA OOM issue #10
    would just retry and dead-letter anyway."""


# --------------------------------------------------------------------------- #
# VRAM reading -- one GET /system_stats shape, read here and by the
# between-chunk probe below.
# --------------------------------------------------------------------------- #


def _fetch_free_vram_gb(session: Any, base_url: str) -> float | None:
    """``GET {base_url}/system_stats`` and extract free VRAM in GB.

    Returns ``None`` -- after logging a WARNING explaining why -- whenever
    the reading can't be trusted: unreachable, non-200, non-JSON, or a body
    that doesn't have the expected ``devices[].vram_free`` shape. Callers
    decide what a ``None`` means for them.
    """
    url = f"{base_url}{SYSTEM_STATS_PATH}"
    try:
        response = session.get(url)
    except requests.RequestException as exc:
        logger.warning("VRAM check could not reach %s (%s)", url, exc)
        return None

    if response.status_code != 200:
        logger.warning("VRAM check: %s returned status=%s", url, response.status_code)
        return None

    try:
        payload = response.json()
    except ValueError:
        logger.warning("VRAM check: %s returned a non-JSON body", url)
        return None

    free_gb = _extract_free_vram_gb(payload)
    if free_gb is None:
        logger.warning(
            "VRAM check: could not find a usable 'devices[].vram_free' in the %s response", url
        )
        return None
    return free_gb


def _extract_free_vram_gb(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    devices = payload.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    try:
        free_bytes = max(float(device["vram_free"]) for device in devices)
    except (KeyError, TypeError, ValueError):
        return None
    return free_bytes / (1024**3)


VramProbe = Callable[[], "float | None"]
"""Zero-arg seam: call it, get a fresh free-VRAM reading in GB, or ``None``
when the reading can't be trusted. Built by :func:`build_vram_probe` below.

This is the shape ``resilience.ResilientRunner`` accepts (issue #23) to
re-check free VRAM between chunks, not just once before the run starts --
that module owns *what to do* with a reading (stop the run below the floor,
log and continue when unreadable); this module owns *how the reading is
taken*, which is the same ``GET /system_stats`` path the custody manager
below already uses via :func:`_fetch_free_vram_gb`."""


def build_vram_probe(session: Any, base_url: str) -> VramProbe:
    """Build a :data:`VramProbe` bound to ``session``/``base_url``.

    Every call re-issues the ``GET /system_stats`` request -- nothing here is
    memoized. A cached first reading would defeat the entire point of a
    between-chunk check: the 2026-08-07 incident was another process claiming
    the card *after* a pre-flight check had already passed once.
    """
    base_url = base_url.rstrip("/")

    def _probe() -> float | None:
        return _fetch_free_vram_gb(session, base_url)

    return _probe


def _free_comfyui(session: Any, base_url: str) -> None:
    """``POST /free`` (unload models, release VRAM). Never raises -- a
    network failure here is logged and swallowed, matching the "release must
    be unconditional" contract: a ComfyUI hiccup on teardown must not mask
    whatever outcome the run itself is already reporting."""
    url = f"{base_url}{FREE_PATH}"
    body = {"unload_models": True, "free_memory": True}
    try:
        session.post(url, json=body)
        logger.info(
            "Sent POST %s to %s (body=%s) to release VRAM after the run", url, base_url, body
        )
    except requests.RequestException as exc:
        logger.error("POST %s to %s failed (body=%s): %s", url, base_url, body, exc)


# --------------------------------------------------------------------------- #
# VramCustodyManager -- assert the card is free, release it afterwards
# --------------------------------------------------------------------------- #


class VramCustodyManager:
    """Assert free VRAM before the run, ``POST /free`` after it.

    Satisfies :class:`~music_video_maker.contracts.CustodyManager`
    structurally. It cannot stop or start anything else that uses the card --
    see this module's docstring for why that is deliberate -- so ``__enter__``
    names the manual step rather than pretending to have taken it.
    """

    def __init__(
        self,
        *,
        base_url: str,
        hardware: HardwareProfile,
        session: Any = None,
        min_free_vram_gb: float = DEFAULT_MIN_FREE_VRAM_GB,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hardware = hardware
        self.session = session if session is not None else requests.Session()
        self.min_free_vram_gb = min_free_vram_gb

    def __enter__(self) -> VramCustodyManager:
        logger.info(
            "Taking GPU custody for this run: anything else that uses this card must already "
            "be stopped by hand. Running a pre-flight free-VRAM assertion against %s%s.",
            self.base_url,
            SYSTEM_STATS_PATH,
        )
        self._preflight_vram_check()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        _free_comfyui(self.session, self.base_url)
        logger.info(
            "Released GPU custody: ComfyUI has been asked to unload its models. Restart "
            "whatever else you stopped for this run."
        )
        return False

    # -- pre-flight -------------------------------------------------------- #

    def _preflight_vram_check(self) -> None:
        free_gb = _fetch_free_vram_gb(self.session, self.base_url)
        if free_gb is None:
            logger.warning(
                "Pre-flight VRAM check against %s could not get a usable reading -- skipping "
                "(degraded, not fatal for the no-op custody manager)",
                self.base_url,
            )
            return

        required_gb = self.min_free_vram_gb
        if free_gb < required_gb:
            logger.error(
                "Pre-flight VRAM check FAILED: only %.2f GB free on %s, need >= %.2f GB "
                "(profile %r, %.2f GB nominal) -- refusing to start the run. Is something "
                "else still holding the GPU? Check with "
                "'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv', "
                "and note that a previous run's models stay resident until POST /free.",
                free_gb,
                self.base_url,
                required_gb,
                self.hardware.name,
                self.hardware.vram_gb,
            )
            raise CustodyError(
                f"only {free_gb:.2f} GB free on {self.base_url} "
                f"(need >= {required_gb:.2f} GB; hardware profile {self.hardware.name!r})"
            )

        logger.info(
            "Pre-flight VRAM check OK: %.2f GB free on %s (>= %.2f GB required)",
            free_gb,
            self.base_url,
            required_gb,
        )


# --------------------------------------------------------------------------- #
# Factory -- the one call cli.py needs
# --------------------------------------------------------------------------- #


def build_custody_manager(
    config: RunConfig,
    *,
    session: Any = None,
) -> VramCustodyManager:
    """Build the custody manager for a run.

    There is one, and this factory exists so ``cli.py`` has a single call
    site if that ever stops being true.
    """
    return VramCustodyManager(
        base_url=config.comfyui_url,
        hardware=config.hardware,
        session=session,
        min_free_vram_gb=config.min_free_vram_gb,
    )


SLEEP_ASSERTION_ARGV = ("caffeinate", "-d", "-i", "-m", "-s")
"""macOS sleep-prevention command, minus the ``-w <pid>`` appended per run.

``-d`` display, ``-i`` idle, ``-m`` disk, ``-s`` system-sleep-on-AC. Not a
substitute for judgement about lids: clamshell sleep overrides every assertion
here, which is why issue #43 item 4 wants the orchestrator off the laptop
entirely."""


def _default_assertion_spawner(argv: Sequence[str]) -> Any:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextmanager
def prevent_host_sleep(
    *,
    platform: str | None = None,
    spawner: Callable[[Sequence[str]], Any] | None = None,
    pid: int | None = None,
) -> Iterator[None]:
    """Keep the machine driving the render awake for the run's duration
    (issue #43).

    A render is hours of work that this process only *supervises* -- almost
    all of its life is spent blocked on a WebSocket, which reads to macOS as
    an idle machine. On 2026-08-11 that machine slept mid-chunk and froze the
    orchestrator for three and a half hours at a time while doris carried on
    finishing prompts nobody was listening for. Nothing failed; the run simply
    stopped advancing, which is the worst way for an overnight job to break.

    Scoped like the custody release it sits beside (issue #19): taken on entry,
    released unconditionally on exit, exception or not. The assertion is
    additionally tied to *this* process id (``caffeinate -w <pid>``), so it
    dies with us even when we are SIGKILLed and can never outlive the run --
    a machine left permanently pinned awake by a crashed render would be a
    worse bug than the one this fixes.

    Degrades loudly rather than refusing: a host with no ``caffeinate`` is
    logged and the render proceeds, per the project's fail-open-to-a-logged-
    reduced-state rule. Off darwin this is a deliberate no-op -- a headless
    Linux ComfyUI host does not idle-sleep the way a laptop does, and
    spawning a macOS-only binary there would log a failure on every run
    instead of the debug line below. This is what makes issue #50's
    co-located deployment (orchestrator and ComfyUI on the same Linux box)
    strictly simpler than the remote-laptop path: the whole assertion is
    moot, not merely satisfied.

    ``platform``/``spawner``/``pid`` are test seams; the defaults read the
    real ones.
    """
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        logger.debug("Host sleep assertion skipped: platform=%s does not idle-sleep", platform)
        yield
        return

    spawner = spawner if spawner is not None else _default_assertion_spawner
    pid = os.getpid() if pid is None else pid
    argv = [*SLEEP_ASSERTION_ARGV, "-w", str(pid)]

    process = None
    try:
        process = spawner(argv)
        logger.info("Holding a host sleep assertion for this run: %s", " ".join(argv))
    except Exception:  # noqa: BLE001 -- a best-effort guard must not fail the run
        logger.warning(
            "Could not hold a host sleep assertion (%s) -- the render will still run, but if "
            "this machine sleeps mid-chunk the run freezes until it wakes (issue #43). Keep "
            "the machine awake by hand, or run the orchestrator on the same host as ComfyUI "
            "(issue #50), where this assertion is unnecessary.",
            " ".join(argv),
            exc_info=True,
        )

    try:
        yield
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info("Released the host sleep assertion")
            except Exception:  # noqa: BLE001 -- teardown must never mask the run's own outcome
                logger.warning(
                    "Failed to release the host sleep assertion cleanly; it is tied to pid %s "
                    "and will lapse when this process exits",
                    pid,
                    exc_info=True,
                )


__all__ = [
    "CustodyError",
    "DEFAULT_MIN_FREE_VRAM_GB",
    "SLEEP_ASSERTION_ARGV",
    "VramCustodyManager",
    "VramProbe",
    "build_custody_manager",
    "build_vram_probe",
    "prevent_host_sleep",
]
