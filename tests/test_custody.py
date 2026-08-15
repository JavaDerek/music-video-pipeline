"""Tests for the GPU custody seam (:class:`VramCustodyManager`,
:func:`build_custody_manager`) and the host sleep assertion.

Everything runs against :class:`tests.harness.comfyui_mock.FakeComfyUISession`
-- no real network, no GPU, no real sleeping. The point is narrow: prove the
pre-flight VRAM assertion actually blocks a run when the reading is clearly
bad, degrades gracefully when ComfyUI can't be asked, and that teardown
(``POST /free``) always runs even when the wrapped work raises.
"""

from __future__ import annotations

import logging
import types

import pytest
import requests

from music_video_maker import custody
from music_video_maker.contracts import CustodyManager, HardwareProfile
from music_video_maker.custody import (
    CustodyError,
    VramCustodyManager,
    build_custody_manager,
    build_vram_probe,
)
from tests.harness.comfyui_mock import FakeComfyUISession

HARDWARE = HardwareProfile(name="RTX 4090 24GB (doris)", vram_gb=24.0)


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #








def _fake_config(**overrides: object) -> types.SimpleNamespace:
    """Duck-typed stand-in for ``config.RunConfig`` carrying only the
    attributes ``build_custody_manager`` reads. ``build_custody_manager``
    never does an ``isinstance(config, RunConfig)`` check -- it just reads
    attributes -- so this is a faithful, much cheaper substitute for a fully
    loaded config in these tests."""
    defaults: dict[str, object] = dict(
        comfyui_url="http://doris:8188",
        hardware=HARDWARE,
        min_free_vram_gb=12.0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class _NetworkErrorSession:
    """Minimal stand-in whose ``get``/``post`` always raise a network error."""

    def get(self, url: str, **kwargs):
        raise requests.ConnectionError("simulated network failure")

    def post(self, url: str, **kwargs):
        raise requests.ConnectionError("simulated network failure")


class _MalformedStatsSession:
    """Returns 200 with a body that isn't shaped like /system_stats."""

    def __init__(self):
        self.free_calls = 0

    def get(self, url: str, **kwargs):
        return _FakeJsonResponse(200, {"unexpected": "shape"})

    def post(self, url: str, **kwargs):
        self.free_calls += 1
        return _FakeJsonResponse(200, {})


class _FakeJsonResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


# --------------------------------------------------------------------------- #
# Pre-flight VRAM assertion
# --------------------------------------------------------------------------- #


def test_preflight_passes_when_vram_is_abundant():
    session = FakeComfyUISession()  # default: ~24 GB free of ~24 GB total
    custody_kwargs = {"base_url": session.base_url, "hardware": HARDWARE, "session": session}
    with VramCustodyManager(**custody_kwargs):
        pass


def test_preflight_raises_when_vram_is_insufficient():
    session = FakeComfyUISession()
    session.set_vram_free(1_000_000_000)  # ~0.93 GB, far below the 24 GB profile

    with (
        pytest.raises(CustodyError, match="only 0.93 GB free"),
        VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session),
    ):
        pytest.fail("must not enter the body when the pre-flight check fails")


def test_preflight_error_names_the_hardware_profile():
    session = FakeComfyUISession()
    session.set_vram_free(0)

    with (
        pytest.raises(CustodyError, match="RTX 4090 24GB \\(doris\\)"),
        VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session),
    ):
        pass


def test_preflight_floor_is_absolute_gb_not_a_fraction_of_the_card():
    """Regression guard. This threshold was 90% of the profile's *nominal*
    VRAM (21.6 GB on a 24 GB card), which encoded an assumption -- "custody
    means exclusive access" -- rather than a measurement, and refused runs
    doris handles comfortably: 5s clips render fine there with ~16.4 GB free
    and other services resident. A fraction-of-nominal rule cannot express
    that, so the floor is an absolute GB figure like ``min_free_disk_gb``.
    """
    session = FakeComfyUISession()
    session.set_vram_free(int(16.4 * 1024**3))  # the observed working figure

    # The default floor must ACCEPT the configuration that is known to work.
    with VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session):
        pass

    # It is still a real gate: raising the floor above the reading refuses.
    with (
        pytest.raises(CustodyError, match="need >= 20.00 GB"),
        VramCustodyManager(
            base_url=session.base_url,
            hardware=HARDWARE,
            session=session,
            min_free_vram_gb=20.0,
        ),
    ):
        pass


def test_preflight_still_refuses_a_gpu_that_was_never_released():
    """The case the check actually exists for: another tenant never let go, so the
    reading is under 1 GB. That must still be refused at the default floor."""
    session = FakeComfyUISession()
    session.set_vram_free(int(0.9 * 1024**3))

    with (
        pytest.raises(CustodyError),
        VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session),
    ):
        pass


def test_preflight_degrades_gracefully_when_comfyui_is_unreachable(caplog):
    session = _NetworkErrorSession()
    with (
        caplog.at_level(logging.WARNING),
        VramCustodyManager(base_url="http://doris:8188", hardware=HARDWARE, session=session),
    ):
        pass  # must not raise
    assert any("could not reach" in r.message for r in caplog.records)


def test_preflight_degrades_gracefully_on_malformed_response(caplog):
    session = _MalformedStatsSession()
    with (
        caplog.at_level(logging.WARNING),
        VramCustodyManager(base_url="http://doris:8188", hardware=HARDWARE, session=session),
    ):
        pass  # must not raise
    assert any("could not find a usable" in r.message for r in caplog.records)


def test_preflight_degrades_gracefully_on_non_200_status(caplog):
    class _ErrorStatusSession:
        def get(self, url, **kwargs):
            return _FakeJsonResponse(503, {})

        def post(self, url, **kwargs):
            return _FakeJsonResponse(200, {})

    with caplog.at_level(logging.WARNING), VramCustodyManager(
        base_url="http://doris:8188", hardware=HARDWARE, session=_ErrorStatusSession()
    ):
        pass  # must not raise


def test_preflight_degrades_gracefully_on_non_json_body(caplog):
    class _NonJsonResponse(_FakeJsonResponse):
        def json(self):
            raise ValueError("not JSON")

    class _NonJsonSession:
        def get(self, url, **kwargs):
            return _NonJsonResponse(200, {})

        def post(self, url, **kwargs):
            return _FakeJsonResponse(200, {})

    with caplog.at_level(logging.WARNING), VramCustodyManager(
        base_url="http://doris:8188", hardware=HARDWARE, session=_NonJsonSession()
    ):
        pass  # must not raise
    assert any("non-JSON body" in r.message for r in caplog.records)


def test_preflight_degrades_gracefully_when_devices_list_is_malformed(caplog):
    class _BadDevicesSession:
        def get(self, url, **kwargs):
            return _FakeJsonResponse(200, {"devices": [{"vram_free": "not-a-number"}]})

        def post(self, url, **kwargs):
            return _FakeJsonResponse(200, {})

    with caplog.at_level(logging.WARNING), VramCustodyManager(
        base_url="http://doris:8188", hardware=HARDWARE, session=_BadDevicesSession()
    ):
        pass  # must not raise
    assert any("could not find a usable" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Teardown: /free is unconditional
# --------------------------------------------------------------------------- #


def test_exit_calls_free_on_the_happy_path():
    session = FakeComfyUISession()
    with VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session):
        pass
    assert session.free_calls == [{"unload_models": True, "free_memory": True}]


def test_exit_calls_free_even_when_the_body_raises():
    session = FakeComfyUISession()

    with (
        pytest.raises(RuntimeError, match="boom"),
        VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session),
    ):
        raise RuntimeError("boom")

    assert session.free_calls == [{"unload_models": True, "free_memory": True}]


def test_exit_does_not_mask_free_network_errors_but_still_propagates_body_exception():
    session = _NetworkErrorSession()

    with (
        pytest.raises(RuntimeError, match="boom"),
        VramCustodyManager(base_url="http://doris:8188", hardware=HARDWARE, session=session),
    ):
        raise RuntimeError("boom")




# --------------------------------------------------------------------------- #
# Structural protocol conformance
# --------------------------------------------------------------------------- #


def test_satisfies_custody_manager_protocol():
    session = FakeComfyUISession()
    custody = VramCustodyManager(base_url=session.base_url, hardware=HARDWARE, session=session)
    assert isinstance(custody, CustodyManager)


# --------------------------------------------------------------------------- #
# build_custody_manager: the single factory cli.py calls
# --------------------------------------------------------------------------- #


def test_factory_builds_a_manager_bound_to_the_config():
    session = FakeComfyUISession()
    config = _fake_config(comfyui_url="http://gpu-host:8188/", min_free_vram_gb=9.5)

    manager = build_custody_manager(config, session=session)

    assert isinstance(manager, VramCustodyManager)
    assert isinstance(manager, CustodyManager)
    assert manager.base_url == "http://gpu-host:8188"
    assert manager.min_free_vram_gb == 9.5
    assert manager.hardware is config.hardware


# --------------------------------------------------------------------------- #
# build_vram_probe -- the seam resilience.ResilientRunner uses to re-check
# free VRAM between chunks (issue #23)
# --------------------------------------------------------------------------- #


def test_build_vram_probe_reads_the_current_value():
    session = FakeComfyUISession()
    session.set_vram_free(int(20 * 1024**3))
    probe = build_vram_probe(session, session.base_url)

    assert probe() == pytest.approx(20.0, abs=0.01)


def test_build_vram_probe_is_not_memoized_it_re_reads_every_call():
    """The whole point of issue #23 is a *fresh* reading each chunk -- a probe
    that cached its first result would be worthless between chunks."""
    session = FakeComfyUISession()
    session.set_vram_free(int(20 * 1024**3))
    probe = build_vram_probe(session, session.base_url)

    assert probe() == pytest.approx(20.0, abs=0.01)

    session.set_vram_free(int(5 * 1024**3))
    assert probe() == pytest.approx(5.0, abs=0.01)


def test_build_vram_probe_returns_none_when_unreadable(caplog):
    session = _NetworkErrorSession()
    probe = build_vram_probe(session, "http://doris:8188")

    with caplog.at_level(logging.WARNING):
        result = probe()

    assert result is None
    assert any("could not reach" in r.message for r in caplog.records)


def test_build_vram_probe_strips_a_trailing_slash_from_base_url():
    session = FakeComfyUISession()
    session.set_vram_free(int(20 * 1024**3))
    probe = build_vram_probe(session, session.base_url + "/")

    assert probe() == pytest.approx(20.0, abs=0.01)
    # Confirms it actually hit /system_stats rather than some doubled-slash
    # path FakeComfyUISession's router would 404 on and thus return None for.




# --------------------------------------------------------------------------- #
# Issue #43: the host must not sleep out from under a running render
# --------------------------------------------------------------------------- #


class _FakeAssertionProcess:
    """A stand-in for the spawned `caffeinate`."""

    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):  # noqa: ANN001 -- mirrors subprocess.Popen
        self.waited = True
        return 0

    def poll(self):
        return None


def _recording_spawner():
    calls: list[list[str]] = []
    procs: list[_FakeAssertionProcess] = []

    def spawn(argv):
        calls.append(list(argv))
        proc = _FakeAssertionProcess()
        procs.append(proc)
        return proc

    spawn.calls = calls  # type: ignore[attr-defined]
    spawn.procs = procs  # type: ignore[attr-defined]
    return spawn


def test_darwin_run_holds_a_sleep_assertion_naming_this_process():
    """The 2026-08-11 defect: a laptop that sleeps freezes the orchestrator
    mid-render for hours. -w <our pid> means the assertion dies with us even
    if we are SIGKILLed, so it can never outlive the run it belongs to."""
    spawn = _recording_spawner()

    with custody.prevent_host_sleep(platform="darwin", spawner=spawn, pid=4242):
        pass

    assert len(spawn.calls) == 1
    argv = spawn.calls[0]
    assert argv[0] == "caffeinate"
    assert "-w" in argv and argv[argv.index("-w") + 1] == "4242"


def test_the_sleep_assertion_is_released_when_the_run_ends():
    spawn = _recording_spawner()

    with custody.prevent_host_sleep(platform="darwin", spawner=spawn, pid=1):
        pass

    assert spawn.procs[0].terminated


def test_the_sleep_assertion_is_released_even_when_the_run_raises():
    """Unconditional, like the custody release it sits beside (issue #19): a
    crashed run must not leave the machine pinned awake."""
    spawn = _recording_spawner()

    with (
        pytest.raises(RuntimeError),
        custody.prevent_host_sleep(platform="darwin", spawner=spawn, pid=1),
    ):
        raise RuntimeError("render blew up")

    assert spawn.procs[0].terminated


def test_no_sleep_assertion_is_attempted_off_darwin():
    """doris is Linux and does not sleep; spawning a macOS-only binary there
    would log a spurious failure on every run."""
    spawn = _recording_spawner()

    with custody.prevent_host_sleep(platform="linux", spawner=spawn, pid=1):
        pass

    assert spawn.calls == []


def test_a_failed_sleep_assertion_degrades_loudly_but_does_not_fail_the_run(caplog):
    """Fail open to a logged, reduced state: a missing caffeinate is a reason
    to warn, not a reason to refuse a render that would otherwise work."""

    def exploding_spawner(argv):
        raise FileNotFoundError("caffeinate not found")

    with (
        caplog.at_level(logging.WARNING),
        custody.prevent_host_sleep(platform="darwin", spawner=exploding_spawner, pid=1),
    ):
        pass

    assert any("sleep" in r.message.lower() for r in caplog.records)
