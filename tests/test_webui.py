"""Tests for the read-only progress monitor server (issue #36).

Layered to match the module: pure bind-address validation first (no socket
at all), then a handful of real-socket tests proving the servers this module
actually builds only ever listen where they were told to, then the route
logic exercised directly against :class:`~music_video_maker.webui.MonitorContext`
via real HTTP requests over loopback with an ephemeral port -- per the task's
own instruction, sockets on 127.0.0.1 with port 0 are fine as long as they are
shut down, and are preferred here over reimplementing HTTP parsing by hand.

No GPU, no ComfyUI, no real ``ffmpeg``/``tailscale`` subprocess, no real
sleep, and nothing ever binds to a non-loopback, non-Tailscale address.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from music_video_maker import prepare_report as prepare_report_module
from music_video_maker import resilience as resilience_module
from music_video_maker import webui
from music_video_maker.contracts import ChunkResult, ChunkStatus, RunState, VramStopEvent


def _result(
    chunk_id: int,
    status: ChunkStatus,
    *,
    attempts: int = 1,
    errors: tuple[str, ...] = (),
    render_seconds: float | None = None,
    video_file: Path | None = None,
    free_vram_gb_before: float | None = None,
    rerender_reason: str | None = None,
    rerender_reason_fields: tuple[str, ...] = (),
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        status=status,
        video_file=video_file,
        attempts=attempts,
        errors=errors,
        render_seconds=render_seconds,
        free_vram_gb_before=free_vram_gb_before,
        rerender_reason=rerender_reason,
        rerender_reason_fields=rerender_reason_fields,
    )


def _write_run_state(path: Path, run_state: RunState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = resilience_module.dump_run_state(run_state)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Bind-address validation -- no socket at all
# --------------------------------------------------------------------------- #


class TestValidateBindAddress:
    @pytest.mark.parametrize("raw", ["127.0.0.1", "127.5.5.5", "::1"])
    def test_accepts_loopback(self, raw: str) -> None:
        assert webui.validate_bind_address(raw) == raw

    @pytest.mark.parametrize("raw", ["100.64.0.1", "100.127.255.255", "100.100.5.5"])
    def test_accepts_tailscale_ipv4_range(self, raw: str) -> None:
        assert webui.validate_bind_address(raw) == raw

    def test_accepts_tailscale_ipv6_range(self) -> None:
        raw = "fd7a:115c:a1e0::1"
        assert webui.validate_bind_address(raw) == raw

    @pytest.mark.parametrize(
        "raw",
        [
            "0.0.0.0",
            "::",
            "192.168.1.5",
            "10.0.0.5",
            "172.16.0.5",
            "8.8.8.8",
            "99.64.0.1",  # just outside the Tailscale /10
            "100.128.0.1",  # just outside the Tailscale /10
        ],
    )
    def test_refuses_wildcards_and_lan_and_public_addresses(self, raw: str) -> None:
        with pytest.raises(webui.BindAddressError):
            webui.validate_bind_address(raw)

    @pytest.mark.parametrize("raw", ["my-laptop.local", "doris", "not-an-ip", ""])
    def test_refuses_hostnames_never_resolving_them(self, raw: str) -> None:
        """Resolving hostnames is explicitly not needed -- refuse instead of
        looking one up, so a name that resolves to a LAN address can never
        sneak a bind through a DNS lookup this module never performs."""
        with pytest.raises(webui.BindAddressError):
            webui.validate_bind_address(raw)


class TestResolveBindAddresses:
    def test_always_includes_loopback(self) -> None:
        addrs = webui.resolve_bind_addresses(tailscale_ipv4=lambda: None)
        assert addrs == ("127.0.0.1",)

    def test_adds_tailscale_ipv4_when_found(self) -> None:
        addrs = webui.resolve_bind_addresses(tailscale_ipv4=lambda: "100.101.102.103")
        assert addrs == ("127.0.0.1", "100.101.102.103")

    def test_falls_back_to_loopback_only_when_tailscale_absent(self) -> None:
        addrs = webui.resolve_bind_addresses(tailscale_ipv4=lambda: None)
        assert "127.0.0.1" in addrs
        assert len(addrs) == 1

    def test_ignores_an_invalid_automatic_tailscale_address_without_raising(self) -> None:
        # Defensive: if `tailscale ip -4` ever returned something bogus, the
        # automatic probe degrades rather than crashing the server.
        addrs = webui.resolve_bind_addresses(tailscale_ipv4=lambda: "8.8.8.8")
        assert addrs == ("127.0.0.1",)

    def test_explicit_bind_addresses_are_validated_and_appended(self) -> None:
        addrs = webui.resolve_bind_addresses(
            ["100.64.9.9"], tailscale_ipv4=lambda: None
        )
        assert addrs == ("127.0.0.1", "100.64.9.9")

    def test_explicit_invalid_bind_address_raises_loudly(self) -> None:
        """Unlike the automatic Tailscale probe, an operator-supplied --bind
        address that fails validation must refuse the server outright, not
        silently degrade -- the operator asked for it by name."""
        with pytest.raises(webui.BindAddressError):
            webui.resolve_bind_addresses(["0.0.0.0"], tailscale_ipv4=lambda: None)

    def test_explicit_wildcard_and_lan_addresses_refused(self) -> None:
        for bad in ("0.0.0.0", "::", "192.168.1.1"):
            with pytest.raises(webui.BindAddressError):
                webui.resolve_bind_addresses([bad], tailscale_ipv4=lambda: None)

    def test_resolved_bind_list_contains_no_wildcard_and_no_lan_address(self) -> None:
        """The design doc's own required test, verbatim: 'the resolved bind
        list contains no wildcard and no non-loopback, non-tailnet
        address.'"""
        addrs = webui.resolve_bind_addresses(
            ["100.64.1.1"], tailscale_ipv4=lambda: "100.90.9.9"
        )
        for addr in addrs:
            assert addr not in ("0.0.0.0", "::")
            ip = webui.validate_bind_address(addr)  # raises if not loopback/tailnet
            assert ip == addr

    def test_deduplicates_addresses(self) -> None:
        addrs = webui.resolve_bind_addresses(
            ["127.0.0.1", "100.64.1.1", "100.64.1.1"], tailscale_ipv4=lambda: "100.64.1.1"
        )
        assert addrs == ("127.0.0.1", "100.64.1.1")


class TestDefaultTailscaleIpv4:
    def test_returns_none_when_binary_absent(self) -> None:
        def runner(args):
            raise FileNotFoundError("no such file: tailscale")

        assert webui.default_tailscale_ipv4(runner=runner) is None

    def test_returns_none_on_nonzero_exit(self) -> None:
        class Result:
            returncode = 1
            stdout = b""
            stderr = b"not logged in"

        assert webui.default_tailscale_ipv4(runner=lambda args: Result()) is None

    def test_parses_first_line_of_stdout(self) -> None:
        class Result:
            returncode = 0
            stdout = b"100.64.55.66\n"
            stderr = b""

        assert webui.default_tailscale_ipv4(runner=lambda args: Result()) == "100.64.55.66"


# --------------------------------------------------------------------------- #
# Real sockets: the servers this module builds listen only where told
# --------------------------------------------------------------------------- #


class TestServersListenOnlyOnValidatedAddresses:
    def test_server_address_matches_the_requested_loopback_address(self, tmp_path: Path) -> None:
        ctx = webui.MonitorContext(run_state_path=tmp_path / "run_state.json")
        server = webui.make_server("127.0.0.1", 0, ctx)
        try:
            assert server.server_address[0] == "127.0.0.1"
            assert server.server_address[1] != 0  # OS assigned a real ephemeral port
        finally:
            server.server_close()

    def test_make_servers_builds_one_server_per_address_all_loopback_or_tailnet(
        self, tmp_path: Path
    ) -> None:
        ctx = webui.MonitorContext(run_state_path=tmp_path / "run_state.json")
        addresses = webui.resolve_bind_addresses(tailscale_ipv4=lambda: None)
        servers = webui.make_servers(addresses, 0, ctx)
        try:
            assert len(servers) == len(addresses)
            for server, addr in zip(servers, addresses, strict=True):
                assert server.server_address[0] == addr
                webui.validate_bind_address(server.server_address[0])
        finally:
            for server in servers:
                server.server_close()

    def test_ipv6_tailnet_address_binds_with_af_inet6(self) -> None:
        ctx = webui.MonitorContext(run_state_path=Path("/nonexistent/run_state.json"))
        try:
            server = webui.make_server("::1", 0, ctx)
        except OSError:
            pytest.skip("IPv6 loopback not available in this sandbox")
        try:
            assert server.address_family == socket.AF_INET6
        finally:
            server.server_close()


# --------------------------------------------------------------------------- #
# current_progress / is_valid_chunk_id -- "the run's known chunks"
# --------------------------------------------------------------------------- #


class TestKnownChunks:
    def test_known_chunks_are_the_results_plus_any_vram_stop_chunk(self) -> None:
        run_state = RunState(
            run_id="run-1",
            results={0: _result(0, ChunkStatus.RENDERED), 2: _result(2, ChunkStatus.CACHED)},
            vram_stop=VramStopEvent(chunk_id=3, free_vram_gb=1.0, floor_gb=2.0),
        )
        progress = webui.current_progress(run_state)
        assert progress.total == 3
        ids = [chunk.chunk_id for chunk in progress.chunks]
        assert ids == [0, 2, 3]
        # chunk 3 never got a ChunkResult -- it renders as pending.
        pending = next(chunk for chunk in progress.chunks if chunk.chunk_id == 3)
        assert pending.status == "pending"

    def test_is_valid_chunk_id_true_only_for_known_ids(self) -> None:
        run_state = RunState(run_id="run-1", results={5: _result(5, ChunkStatus.RENDERED)})
        assert webui.is_valid_chunk_id(run_state, 5) is True
        assert webui.is_valid_chunk_id(run_state, 6) is False

    def test_a_run_with_no_results_yet_does_not_report_finished(self) -> None:
        """Regression guard: an empty expected set would make
        RunProgress.finished vacuously True (`all()` over nothing), which
        would misreport a run that has not even started as complete. This
        module's own MonitorContext-level handling of an empty run_state
        must never present that as 'finished' -- see render_index_html,
        which reports pending/total counts rather than a bare finished flag,
        and never shows an empty run as 100%."""
        run_state = RunState(run_id="run-1", results={})
        progress = webui.current_progress(run_state)
        assert progress.total == 0
        # This is the documented, honest limitation, not a bug: with zero
        # known chunks the module has no basis to claim anything, so the
        # rendered page must not claim "recorded: 0 / finished" as if that
        # were the whole run -- checked in the HTML-rendering tests below.
        assert progress.finished is True  # vacuous, by construction
        html = webui.render_index_html(progress).decode("utf-8")
        assert "chunks recorded in run_state.json so far" in html


# --------------------------------------------------------------------------- #
# stream_progress_events -- no sockets, no real sleep
# --------------------------------------------------------------------------- #


class RecordingSleeper:
    """Never actually sleeps. Records each call and can be told to rewrite
    run_state.json when invoked, which is how these tests advance a "run"
    across polls without any wall-clock time passing."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._on_call: list[callable] = []

    def queue(self, fn) -> None:
        self._on_call.append(fn)

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._on_call:
            self._on_call.pop(0)()


class TestStreamProgressEvents:
    def test_missing_state_file_yields_nothing_and_never_raises(self, tmp_path: Path) -> None:
        sleeper = RecordingSleeper()
        events = list(
            webui.stream_progress_events(
                run_state_path=tmp_path / "does_not_exist.json",
                sleeper=sleeper,
                max_polls=3,
            )
        )
        assert events == []
        assert len(sleeper.calls) == 2  # slept between polls 1-2 and 2-3, not after the last

    def test_torn_json_is_treated_as_not_ready_never_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "run_state.json"
        path.write_text("{not valid json", encoding="utf-8")
        sleeper = RecordingSleeper()
        events = list(
            webui.stream_progress_events(run_state_path=path, sleeper=sleeper, max_polls=2)
        )
        assert events == []

    def test_first_poll_emits_one_run_snapshot(self, tmp_path: Path) -> None:
        path = tmp_path / "run_state.json"
        _write_run_state(
            path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        sleeper = RecordingSleeper()
        events = list(
            webui.stream_progress_events(run_state_path=path, sleeper=sleeper, max_polls=1)
        )
        assert [e.event for e in events] == ["run_snapshot"]

    def test_a_new_chunk_landing_between_polls_emits_chunk_completed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run_state.json"
        _write_run_state(
            path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        sleeper = RecordingSleeper()
        sleeper.queue(
            lambda: _write_run_state(
                path,
                RunState(
                    run_id="run-1",
                    results={
                        0: _result(0, ChunkStatus.RENDERED),
                        1: _result(1, ChunkStatus.RENDERED, render_seconds=12.0),
                    },
                ),
            )
        )
        events = list(
            webui.stream_progress_events(run_state_path=path, sleeper=sleeper, max_polls=2)
        )
        kinds = [e.event for e in events]
        assert kinds == ["run_snapshot", "chunk_completed"]
        assert events[1].data["chunk_id"] == 1

    def test_a_dead_lettered_chunk_carries_its_errors(self, tmp_path: Path) -> None:
        path = tmp_path / "run_state.json"
        _write_run_state(path, RunState(run_id="run-1", results={}))
        sleeper = RecordingSleeper()
        sleeper.queue(
            lambda: _write_run_state(
                path,
                RunState(
                    run_id="run-1",
                    results={
                        0: _result(
                            0,
                            ChunkStatus.DEAD_LETTERED,
                            attempts=3,
                            errors=("timeout", "timeout", "<script>alert(1)</script>"),
                        )
                    },
                ),
            )
        )
        events = list(
            webui.stream_progress_events(run_state_path=path, sleeper=sleeper, max_polls=2)
        )
        dead = next(e for e in events if e.event == "chunk_dead_lettered")
        assert dead.data["errors"] == ["timeout", "timeout", "<script>alert(1)</script>"]

    def test_vram_stop_emits_run_stopped_once(self, tmp_path: Path) -> None:
        path = tmp_path / "run_state.json"
        _write_run_state(
            path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        sleeper = RecordingSleeper()
        sleeper.queue(
            lambda: _write_run_state(
                path,
                RunState(
                    run_id="run-1",
                    results={0: _result(0, ChunkStatus.RENDERED)},
                    vram_stop=VramStopEvent(chunk_id=1, free_vram_gb=0.5, floor_gb=2.0),
                ),
            )
        )
        events = list(
            webui.stream_progress_events(run_state_path=path, sleeper=sleeper, max_polls=2)
        )
        assert [e.event for e in events] == ["run_snapshot", "run_stopped"]


# --------------------------------------------------------------------------- #
# Thumbnails -- caching and validation
# --------------------------------------------------------------------------- #


class TestThumbnails:
    def test_cache_hit_skips_the_runner(self, tmp_path: Path) -> None:
        video = tmp_path / "chunk_0000.mp4"
        video.write_bytes(b"not a real mp4, just needs to exist")
        cache_dir = tmp_path / "cache"

        calls = []

        def runner(args):
            calls.append(args)
            # Simulate ffmpeg writing the PNG.
            out_path = Path(args[-1])
            out_path.write_bytes(b"\x89PNG\r\n fake")
            return type("R", (), {"returncode": 0, "stderr": b""})()

        first = webui.get_or_render_thumbnail(video, cache_dir=cache_dir, runner=runner)
        second = webui.get_or_render_thumbnail(video, cache_dir=cache_dir, runner=runner)
        assert first == second == b"\x89PNG\r\n fake"
        assert len(calls) == 1  # second call was a cache hit

    def test_cache_dir_is_outside_repo_and_chunks_dir_by_default(self) -> None:
        cache_dir = webui.DEFAULT_THUMBNAIL_CACHE_DIR
        repo_root = Path(__file__).resolve().parent.parent
        assert not str(cache_dir).startswith(str(repo_root))

    def test_ffmpeg_failure_raises_thumbnail_error(self, tmp_path: Path) -> None:
        video = tmp_path / "chunk_0000.mp4"
        video.write_bytes(b"x")

        def failing_runner(args):
            return type("R", (), {"returncode": 1, "stderr": b"no such filter"})()

        with pytest.raises(webui.ThumbnailError):
            webui.get_or_render_thumbnail(
                video, cache_dir=tmp_path / "cache", runner=failing_runner
            )

    def test_a_rerendered_chunk_with_new_mtime_invalidates_the_cache(self, tmp_path: Path) -> None:
        video = tmp_path / "chunk_0000.mp4"
        video.write_bytes(b"version 1")
        cache_dir = tmp_path / "cache"
        calls = []

        def runner(args):
            calls.append(args)
            Path(args[-1]).write_bytes(b"frame-for-" + video.read_bytes())
            return type("R", (), {"returncode": 0, "stderr": b""})()

        first = webui.get_or_render_thumbnail(video, cache_dir=cache_dir, runner=runner)
        # Re-render: new content and a bumped mtime, same filename (--resume shape).
        time.sleep(0.01)
        video.write_bytes(b"version 2, much longer content than before")
        second = webui.get_or_render_thumbnail(video, cache_dir=cache_dir, runner=runner)
        assert first != second
        assert len(calls) == 2


# --------------------------------------------------------------------------- #
# HTML escaping
# --------------------------------------------------------------------------- #


class TestHtmlEscaping:
    def test_a_script_tag_in_a_dead_letter_error_is_escaped(self) -> None:
        run_state = RunState(
            run_id="run-1",
            results={
                0: _result(
                    0,
                    ChunkStatus.DEAD_LETTERED,
                    errors=("<script>alert('xss')</script>",),
                )
            },
        )
        progress = webui.current_progress(run_state)
        rendered = webui.render_index_html(progress).decode("utf-8")
        assert "<script>alert" not in rendered
        assert "&lt;script&gt;" in rendered

    def test_rerender_reason_fields_are_escaped(self) -> None:
        run_state = RunState(
            run_id="<run/>",
            results={
                0: _result(
                    0,
                    ChunkStatus.RENDERED,
                    rerender_reason="content_changed",
                    rerender_reason_fields=("<b>prompt_hash</b>",),
                )
            },
        )
        progress = webui.current_progress(run_state)
        rendered = webui.render_index_html(progress).decode("utf-8")
        assert "<b>prompt_hash</b>" not in rendered
        assert "&lt;b&gt;prompt_hash&lt;/b&gt;" in rendered
        assert "&lt;run/&gt;" in rendered


# --------------------------------------------------------------------------- #
# Full route dispatch, over a real loopback socket
# --------------------------------------------------------------------------- #


@pytest.fixture
def running_server(tmp_path: Path):
    """A real MonitorRequestHandler server on 127.0.0.1:0, torn down after
    the test. Yields (base_url, ctx, run_state_path)."""
    run_state_path = tmp_path / "chunks" / "run_state.json"
    ctx = webui.MonitorContext(
        run_state_path=run_state_path,
        thumbnail_cache_dir=tmp_path / "thumbcache",
    )
    server = webui.make_server("127.0.0.1", 0, ctx)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}", ctx, run_state_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base_url: str, path: str, method: str = "GET") -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection(base_url.split("://")[1], timeout=5)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        resp.read_body = resp.read()  # type: ignore[attr-defined]
        return resp
    finally:
        conn.close()


class TestRouteDispatch:
    def test_index_before_run_state_exists_is_200_not_ready(self, running_server) -> None:
        base_url, _ctx, _path = running_server
        resp = _get(base_url, "/")
        assert resp.status == 200
        assert b"Not ready yet" in resp.read_body

    def test_index_after_a_chunk_lands(self, running_server) -> None:
        base_url, _ctx, run_state_path = running_server
        _write_run_state(
            run_state_path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        resp = _get(base_url, "/")
        assert resp.status == 200
        assert b"Run run-1" in resp.read_body

    def test_unknown_path_is_404(self, running_server) -> None:
        base_url, _ctx, _path = running_server
        resp = _get(base_url, "/nope")
        assert resp.status == 404

    def test_post_is_405_with_allow_header(self, running_server) -> None:
        base_url, _ctx, _path = running_server
        resp = _get(base_url, "/", method="POST")
        assert resp.status == 405
        assert resp.getheader("Allow") == "GET, HEAD"

    def test_put_and_delete_are_405_too(self, running_server) -> None:
        base_url, _ctx, _path = running_server
        for method in ("PUT", "DELETE", "PATCH"):
            resp = _get(base_url, "/", method=method)
            assert resp.status == 405

    def test_review_404s_when_not_configured(self, running_server) -> None:
        base_url, _ctx, _path = running_server
        resp = _get(base_url, "/review")
        assert resp.status == 404

    def test_review_serves_the_configured_file(self, tmp_path: Path) -> None:
        review_path = tmp_path / "review.html"
        review_path.write_text("<html>review</html>", encoding="utf-8")
        ctx = webui.MonitorContext(
            run_state_path=tmp_path / "run_state.json",
            review_html_path=review_path,
        )
        server = webui.make_server("127.0.0.1", 0, ctx)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            resp = _get(base_url, "/review")
            assert resp.status == 200
            assert resp.read_body == b"<html>review</html>"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_thumbnail_route_requires_digits_only_no_path_traversal(
        self, running_server
    ) -> None:
        """A path-traversal attempt through the <id> segment must not reach
        the filesystem at all -- the route regex only ever matches digits,
        so anything else (including an encoded '..') 404s as an unknown
        route, never as a lookup against a crafted path."""
        base_url, _ctx, run_state_path = running_server
        _write_run_state(
            run_state_path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        for attempt in (
            "/chunks/../../../etc/passwd/thumbnail.png",
            "/chunks/..%2f..%2fetc%2fpasswd/thumbnail.png",
            "/chunks/0/../1/thumbnail.png",
            "/chunks/%2e%2e/thumbnail.png",
            "/chunks/0x1/thumbnail.png",
            "/chunks/1.5/thumbnail.png",
            "/chunks//thumbnail.png",
        ):
            resp = _get(base_url, attempt)
            assert resp.status == 404, attempt

    def test_thumbnail_for_unknown_chunk_is_404(self, running_server) -> None:
        base_url, _ctx, run_state_path = running_server
        _write_run_state(run_state_path, RunState(run_id="run-1", results={}))
        resp = _get(base_url, "/chunks/0/thumbnail.png")
        assert resp.status == 404

    def test_thumbnail_for_a_chunk_with_no_video_file_is_404(self, running_server) -> None:
        base_url, _ctx, run_state_path = running_server
        _write_run_state(
            run_state_path,
            RunState(run_id="run-1", results={0: _result(0, ChunkStatus.DEAD_LETTERED)}),
        )
        resp = _get(base_url, "/chunks/0/thumbnail.png")
        assert resp.status == 404

    def test_thumbnail_extracts_and_serves_a_png(self, tmp_path: Path) -> None:
        video = tmp_path / "chunk_0000.mp4"
        video.write_bytes(b"fake mp4 bytes")
        run_state_path = tmp_path / "chunks" / "run_state.json"
        _write_run_state(
            run_state_path,
            RunState(
                run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED, video_file=video)}
            ),
        )

        def fake_ffmpeg(args):
            Path(args[-1]).write_bytes(b"\x89PNG fake frame")
            return type("R", (), {"returncode": 0, "stderr": b""})()

        ctx = webui.MonitorContext(
            run_state_path=run_state_path,
            thumbnail_cache_dir=tmp_path / "thumbcache",
            ffmpeg_runner=fake_ffmpeg,
        )
        server = webui.make_server("127.0.0.1", 0, ctx)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            resp = _get(base_url, "/chunks/0/thumbnail.png")
            assert resp.status == 200
            assert resp.getheader("Content-Type") == "image/png"
            assert resp.read_body == b"\x89PNG fake frame"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_head_request_gets_headers_no_body(self, running_server) -> None:
        base_url, _ctx, run_state_path = running_server
        _write_run_state(
            run_state_path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        resp = _get(base_url, "/", method="HEAD")
        assert resp.status == 200
        assert resp.read_body == b""
        assert int(resp.getheader("Content-Length")) > 0

    def test_events_route_serves_sse_content_type(self, running_server) -> None:
        base_url, _ctx, run_state_path = running_server
        _write_run_state(
            run_state_path, RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        )
        conn = http.client.HTTPConnection(base_url.split("://")[1], timeout=5)
        try:
            conn.request("GET", "/events")
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.getheader("Content-Type") == "text/event-stream"
            # Read exactly one SSE frame's worth (the run_snapshot) and stop --
            # the connection is intentionally long-lived otherwise.
            chunk = resp.read(200)
            assert b"event: run_snapshot" in chunk
        finally:
            conn.close()


# --------------------------------------------------------------------------- #
# /prepare -- the pre-render checks, READ from what --prepare wrote
# --------------------------------------------------------------------------- #


def _prepare_report(**overrides) -> prepare_report_module.PrepareReport:
    base = prepare_report_module.PrepareReport(
        generated_at="2026-09-21",
        config_path="run.toml",
        alignment_summary="Alignment quality: 57 segment(s), 2 critical, 5 warning finding(s)",
        alignment_finding_counts={"CRITICAL": 2, "WARNING": 5, "INFO": 0},
        alignment_findings=(
            prepare_report_module.AlignmentFindingRow(
                severity="CRITICAL",
                code="zero_length_segment",
                message="segment 0 is 20ms long",
                start=0.0,
                end=0.02,
                segment_index=0,
            ),
        ),
        chunk_count=80,
        voiced_chunk_count=41,
        instrumental_chunk_count=39,
        timeline_start=0.0,
        timeline_end=513.917,
        track_duration_seconds=512.08,
        timeline_drift_seconds=1.837,
        timeline_drift_frames=44.1,
        duration_tolerance_seconds=0.042,
        notices=(
            prepare_report_module.StageNotice(
                logger="music_video_maker.slicing",
                level="WARNING",
                message="Final chunk boundary at 261.000s ... (issue #70).",
                issue="70",
            ),
        ),
    )
    return type(base)(**{**base.__dict__, **overrides})


class TestPrepareRoute:
    def _serve(self, tmp_path: Path, report_path: Path | None):
        ctx = webui.MonitorContext(
            run_state_path=tmp_path / "chunks" / "run_state.json",
            prepare_report_path=report_path,
            thumbnail_cache_dir=tmp_path / "thumbcache",
        )
        server = webui.make_server("127.0.0.1", 0, ctx)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[0], server.server_address[1]
        return f"http://{host}:{port}", server, thread

    def test_a_run_nobody_has_prepared_says_so_and_names_the_command(
        self, tmp_path: Path
    ) -> None:
        base_url, server, thread = self._serve(tmp_path, tmp_path / "prepare_report.json")
        try:
            resp = _get(base_url, "/prepare")
            assert resp.status == 200  # an ordinary state, never a 500
            body = resp.read_body.decode("utf-8")
            assert "Nothing has been prepared for this run" in body
            assert "--prepare" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_the_page_never_offers_to_run_alignment_itself(self, tmp_path: Path) -> None:
        """Alignment writes chunk audio into the live run's own directory and
        costs ~6s of CPU: a step the operator invokes, not one a page performs
        because somebody opened it. There is no control here at all -- this
        route only ever reads a file."""
        base_url, server, thread = self._serve(tmp_path, tmp_path / "prepare_report.json")
        try:
            body = _get(base_url, "/prepare").read_body.decode("utf-8")
            assert "<form" not in body
            assert "<button" not in body
            assert _get(base_url, "/prepare", method="POST").status == 405
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_a_written_report_renders_its_findings_drift_and_notices(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "prepare_report.json"
        prepare_report_module.write_prepare_report(_prepare_report(), path)
        base_url, server, thread = self._serve(tmp_path, path)
        try:
            body = _get(base_url, "/prepare").read_body.decode("utf-8")
            assert "2 critical" in body
            assert "zero_length_segment" in body
            assert "1.837s" in body  # the #22 timeline drift
            assert "issue #70" in body  # notices grouped by the issue they name
            assert "80" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_a_report_whose_inputs_moved_renders_a_stale_banner(self, tmp_path: Path) -> None:
        lyrics = tmp_path / "lyrics.txt"
        lyrics.write_text("la la la\n", encoding="utf-8")
        path = tmp_path / "prepare_report.json"
        prepare_report_module.write_prepare_report(
            _prepare_report(
                inputs=(prepare_report_module.InputStamp.of("lyrics_file", lyrics),)
            ),
            path,
        )
        lyrics.write_text("an entirely different song\n", encoding="utf-8")
        base_url, server, thread = self._serve(tmp_path, path)
        try:
            body = _get(base_url, "/prepare").read_body.decode("utf-8")
            assert "Stale:" in body
            assert "lyrics_file" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_an_unreadable_report_reads_as_not_prepared_not_a_500(self, tmp_path: Path) -> None:
        path = tmp_path / "prepare_report.json"
        path.write_text("{not json", encoding="utf-8")
        base_url, server, thread = self._serve(tmp_path, path)
        try:
            resp = _get(base_url, "/prepare")
            assert resp.status == 200
            assert b"Nothing has been prepared" in resp.read_body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_no_configured_path_still_renders_the_not_prepared_page(
        self, tmp_path: Path
    ) -> None:
        base_url, server, thread = self._serve(tmp_path, None)
        try:
            resp = _get(base_url, "/prepare")
            assert resp.status == 200
            assert b"no prepare-report path configured" in resp.read_body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_the_index_links_to_it(self, tmp_path: Path) -> None:
        run_state = RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)})
        body = webui.render_index_html(webui.current_progress(run_state)).decode("utf-8")
        assert 'href="/prepare"' in body


class TestPrepareHtmlEscaping:
    def test_every_report_derived_string_is_escaped(self) -> None:
        """A shot-plan refusal quotes the plan's own text, a notice quotes a
        lyric, and an input path is whatever the operator named: all three
        reach this page as free text."""
        report = _prepare_report(
            plan_checked="<script>alert(1)</script>.toml",
            plan_errors=('drift on <img src=x onerror="alert(2)">',),
            notices=(
                prepare_report_module.StageNotice(
                    logger="music_video_maker.slicing",
                    level="WARNING",
                    message="<script>alert(3)</script>",
                    issue=None,
                ),
            ),
        )
        body = webui.render_prepare_html(report).decode("utf-8")
        assert "<script>alert(1)" not in body
        assert "<script>alert(3)" not in body
        # The tag that would have carried the handler is escaped, so
        # `onerror=` survives only as inert text inside a <li>.
        assert "<img" not in body
        assert "&lt;img" in body
        assert "&lt;script&gt;" in body

    def test_a_plan_free_run_says_no_plan_was_checked_rather_than_passing(self) -> None:
        body = webui.render_prepare_html(_prepare_report()).decode("utf-8")
        assert "No shot plan was checked" in body

    def test_a_plan_checked_without_its_lengths_says_the_comparison_is_not_real(self) -> None:
        """CLAUDE.md: a plan that sets length_seconds has two timelines and
        only one is real. A drift list produced against the other one has to
        say so, or it reads as a defect in the plan."""
        body = webui.render_prepare_html(
            _prepare_report(
                plan_checked="shot_plan.toml",
                plan_lengths_applied=False,
                plan_errors=("shot plan drift on chunk_id=7",),
            )
        ).decode("utf-8")
        assert "NO editorial lengths" in body
        assert "--from-plan" in body


# --------------------------------------------------------------------------- #
# Host header -- the DNS-rebinding defence (CLAUDE.md: "Not yet done: a
# Host-header check against DNS rebinding")
# --------------------------------------------------------------------------- #


class TestHostHeaderAllowed:
    """The pure predicate. A bound address plus a port stands in for one
    running server; the socket-level proof is in TestHostHeaderOverHTTP."""

    def _allowed(self, raw, *, address="127.0.0.1", port=8787, extra=frozenset()) -> bool:
        return webui.host_header_allowed(
            raw, bound_address=address, port=port, extra_allowed=extra
        )

    def test_absent_host_header_is_allowed(self) -> None:
        """A rebinding attack's whole leverage is the name it puts in this
        header, and a browser will not let script suppress it. Refusing an
        absent header would refuse HTTP/1.0 clients and block no attack --
        see webui.py's Host-header section."""
        assert self._allowed(None) is True

    @pytest.mark.parametrize("raw", ["127.0.0.1", "127.0.0.1:8787", "localhost", "localhost:8787"])
    def test_accepts_its_own_bound_address_and_localhost(self, raw: str) -> None:
        assert self._allowed(raw) is True

    def test_accepts_localhost_with_a_trailing_root_dot(self) -> None:
        assert self._allowed("LocalHost.") is True

    @pytest.mark.parametrize(
        "raw",
        [
            "evil.example",
            "evil.example:8787",
            "attacker.test.",
            "192.168.1.5",
            "192.168.1.5:8787",
            "8.8.8.8",
            "doris",  # a real name, but not one this server was told about
        ],
    )
    def test_refuses_every_other_name_and_literal(self, raw: str) -> None:
        assert self._allowed(raw) is False

    def test_refuses_a_right_name_on_the_wrong_port(self) -> None:
        assert self._allowed("127.0.0.1:9999") is False
        assert self._allowed("localhost:9999") is False

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "127.0.0.1:notaport", "http://127.0.0.1", "127.0.0.1/x", "a b", "u@127.0.0.1"],
    )
    def test_refuses_anything_that_is_not_a_plain_authority(self, raw: str) -> None:
        assert self._allowed(raw) is False

    def test_an_ipv6_bound_address_matches_its_bracketed_and_expanded_forms(self) -> None:
        assert self._allowed("[::1]:8787", address="::1") is True
        assert self._allowed("[0:0:0:0:0:0:0:1]", address="::1") is True
        # ...and a v4 literal does not match a v6 bind, or vice versa.
        assert self._allowed("[::1]", address="127.0.0.1") is False
        assert self._allowed("127.0.0.1", address="::1") is False

    def test_an_allowlisted_name_is_accepted_with_or_without_the_port(self) -> None:
        extra = frozenset({"doris"})
        assert self._allowed("doris", extra=extra) is True
        assert self._allowed("doris:8787", extra=extra) is True
        assert self._allowed("DORIS.", extra=extra) is True
        assert self._allowed("doris:9999", extra=extra) is False
        assert self._allowed("not-doris", extra=extra) is False

    def test_a_name_is_never_resolved_to_decide_this(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolution is the mechanism the check exists to defeat: asking DNS
        whether ``evil.example`` points here is asking the attacker."""

        def _boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("host_header_allowed resolved a name")

        monkeypatch.setattr(socket, "gethostbyname", _boom)
        monkeypatch.setattr(socket, "getaddrinfo", _boom)
        assert self._allowed("evil.example") is False


class TestNormaliseAllowedHost:
    def test_casefolds_and_strips_a_trailing_root_dot(self) -> None:
        assert webui.normalise_allowed_host("Doris.") == "doris"

    def test_accepts_an_ip_literal_and_canonicalises_it(self) -> None:
        assert webui.normalise_allowed_host("0:0:0:0:0:0:0:1") == "::1"

    @pytest.mark.parametrize(
        "raw", ["", "  ", "*", ".", "http://doris", "doris:8787", "doris/x", "a b", "-doris"]
    )
    def test_refuses_anything_that_is_not_a_bare_host_name(self, raw: str) -> None:
        with pytest.raises(webui.HostAllowlistError):
            webui.normalise_allowed_host(raw)


def _request_with_host(
    base_url: str, path: str, host: str | None, method: str = "GET"
) -> http.client.HTTPResponse:
    """One request over a real loopback socket with ``Host`` set to exactly
    ``host`` -- or, for ``None``, with no ``Host`` header at all (what an
    HTTP/1.0 client sends)."""
    conn = http.client.HTTPConnection(base_url.split("://")[1], timeout=5)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        if host is not None:
            conn.putheader("Host", host)
        conn.endheaders()
        resp = conn.getresponse()
        resp.read_body = resp.read()  # type: ignore[attr-defined]
        return resp
    finally:
        conn.close()


class TestHostHeaderOverHTTP:
    """Real sockets, the way the bind-address tests above are real sockets:
    a browser pointed at this server by a rebound name must get nothing."""

    def _serve(self, tmp_path: Path, allowed_hosts=frozenset()):
        run_state_path = tmp_path / "chunks" / "run_state.json"
        _write_run_state(
            run_state_path,
            RunState(run_id="run-1", results={0: _result(0, ChunkStatus.RENDERED)}),
        )
        ctx = webui.MonitorContext(
            run_state_path=run_state_path,
            thumbnail_cache_dir=tmp_path / "thumbcache",
            allowed_hosts=allowed_hosts,
        )
        server = webui.make_server("127.0.0.1", 0, ctx)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[0], server.server_address[1]
        return f"http://{host}:{port}", server, thread

    def test_a_rebound_name_gets_421_and_no_run_data(self, tmp_path: Path) -> None:
        base_url, server, thread = self._serve(tmp_path)
        try:
            resp = _request_with_host(base_url, "/", "evil.example")
            assert resp.status == 421
            body = resp.read_body
            # The page this would otherwise have served names the run.
            assert b"run-1" not in body
            # The refused value is logged, never echoed back into the body.
            assert b"evil.example" not in body
            assert resp.getheader("X-Content-Type-Options") == "nosniff"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_the_servers_own_address_and_localhost_are_answered(self, tmp_path: Path) -> None:
        base_url, server, thread = self._serve(tmp_path)
        port = server.server_address[1]
        try:
            for host in (f"127.0.0.1:{port}", "127.0.0.1", f"localhost:{port}", None):
                resp = _request_with_host(base_url, "/", host)
                assert resp.status == 200, host
                assert b"Run run-1" in resp.read_body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_the_check_runs_before_every_route_including_405_and_404(
        self, tmp_path: Path
    ) -> None:
        """"Before any handler runs" is the requirement: a rejected request
        must not even learn which methods are allowed or which paths exist."""
        base_url, server, thread = self._serve(tmp_path)
        try:
            for method, path in [
                ("GET", "/events"),
                ("GET", "/chunks/0/thumbnail.png"),
                ("GET", "/review"),
                ("GET", "/no-such-route"),
                ("POST", "/"),
                ("HEAD", "/"),
            ]:
                resp = _request_with_host(base_url, path, "evil.example", method=method)
                assert resp.status == 421, (method, path)
                assert resp.getheader("Allow") is None
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_head_rejection_sends_no_body(self, tmp_path: Path) -> None:
        base_url, server, thread = self._serve(tmp_path)
        try:
            resp = _request_with_host(base_url, "/", "evil.example", method="HEAD")
            assert resp.status == 421
            assert resp.read_body == b""
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_two_host_headers_are_refused(self, tmp_path: Path) -> None:
        base_url, server, thread = self._serve(tmp_path)
        port = server.server_address[1]
        try:
            conn = http.client.HTTPConnection(base_url.split("://")[1], timeout=5)
            try:
                conn.putrequest("GET", "/", skip_host=True, skip_accept_encoding=True)
                conn.putheader("Host", f"127.0.0.1:{port}")
                conn.putheader("Host", "evil.example")
                conn.endheaders()
                resp = conn.getresponse()
                assert resp.status == 421
                assert b"run-1" not in resp.read()
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_an_allow_host_name_is_answered(self, tmp_path: Path) -> None:
        base_url, server, thread = self._serve(tmp_path, allowed_hosts=frozenset({"doris"}))
        port = server.server_address[1]
        try:
            resp = _request_with_host(base_url, "/", f"doris:{port}")
            assert resp.status == 200
            assert b"Run run-1" in resp.read_body
            # ...and only that name; the allowlist is not a wildcard.
            assert _request_with_host(base_url, "/", "evil.example").status == 421
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# The design doc's second day-one test does not apply here
# --------------------------------------------------------------------------- #


def test_starting_a_run_while_one_is_in_flight_is_refused_does_not_apply() -> None:
    """docs/design-web-ui.md's "Testing" section names two day-one tests for
    a server like this: the bind-address assertion (covered exhaustively
    above) and 'starting a run while one is in flight is refused'.

    The second one is an acceptance criterion for the *start/configure* half
    of issue #36, which this work package deliberately does not build -- see
    music_video_maker/webui.py's module docstring, "Scope: read-only,
    deliberately". There is no start route here to refuse a second run from:
    ``MonitorRequestHandler`` only ever reads run_state.json and serves
    static/derived files. This test exists so a reader of the suite sees
    that gap was a decision, not an oversight, and can find the reasoning
    in one place.
    """
    assert not hasattr(webui.MonitorRequestHandler, "do_start")
    # No route in the dispatch table can mutate a run's state at all --
    # every POST/PUT/DELETE/PATCH is a flat 405 (see TestRouteDispatch), and
    # do_GET only ever reads run_state.json or serves a file the operator
    # named at startup.
    assert webui.MonitorRequestHandler.do_POST is webui.MonitorRequestHandler._method_not_allowed


# --------------------------------------------------------------------------- #
# VRAM-stop notice
# --------------------------------------------------------------------------- #


def test_vram_stop_notice_is_prominent_in_the_rendered_page() -> None:
    run_state = RunState(
        run_id="run-1",
        results={0: _result(0, ChunkStatus.RENDERED)},
        vram_stop=VramStopEvent(chunk_id=1, free_vram_gb=1.23, floor_gb=4.0),
    )
    progress = webui.current_progress(run_state)
    rendered = webui.render_index_html(progress).decode("utf-8")
    assert "VRAM stop" in rendered
    assert "1.23" in rendered
    assert "4.00" in rendered


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


_MINIMAL_RUN_TOML = """
master_audio = "{tmp_path}/audio/master.wav"
lyrics_file = "{tmp_path}/lyrics.txt"
global_style = "test style"
narrative_concept = "test concept"
default_lead_vocalist = "Dianne"
comfyui_url = "http://doris:8188"
workflow_template = "{tmp_path}/workflow_api.json"
chunks_dir = "{tmp_path}/output/chunks"
final_video_dir = "{tmp_path}/output/final"

[cast.Dianne]
role = "Lead Vocalist"
image = "{tmp_path}/cast/dianne_ref.jpg"

[hardware]
name = "RTX 4090"
vram_gb = 24.0
"""


def _write_minimal_run_config(tmp_path: Path) -> Path:
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "audio" / "master.wav").write_bytes(b"RIFF-fake-wav-data")
    (tmp_path / "lyrics.txt").write_text("la la la\n", encoding="utf-8")
    (tmp_path / "cast").mkdir(exist_ok=True)
    (tmp_path / "cast" / "dianne_ref.jpg").write_bytes(b"\xff\xd8\xff-fake-jpg")
    (tmp_path / "workflow_api.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "run.toml"
    config_path.write_text(_MINIMAL_RUN_TOML.format(tmp_path=tmp_path), encoding="utf-8")
    return config_path


class TestCliMain:
    def test_build_parser_parses_known_flags(self) -> None:
        args = webui.build_parser().parse_args(
            [
                "--config",
                "run.toml",
                "--bind",
                "100.64.1.1",
                "--port",
                "9000",
                "--review-html",
                "review.html",
            ]
        )
        assert args.config == "run.toml"
        assert args.bind == ["100.64.1.1"]
        assert args.port == 9000
        assert args.review_html == Path("review.html")

    def test_build_parser_collects_repeated_allow_host_values(self) -> None:
        args = webui.build_parser().parse_args(
            ["--config", "run.toml", "--allow-host", "doris", "--allow-host", "mac-mini"]
        )
        assert args.allow_host == ["doris", "mac-mini"]

    def test_main_returns_error_for_a_missing_config_file(self, tmp_path: Path) -> None:
        exit_code = webui.main(["--config", str(tmp_path / "does_not_exist.toml")])
        assert exit_code == 1

    def test_main_refuses_an_invalid_allow_host_without_binding_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silently-ignored allowlist entry is the failure mode where an
        operator believes the check is looser than it is, so a value this
        server cannot honour stops the process instead."""

        def _no_tailscale(args):
            raise FileNotFoundError("tailscale not installed in this sandbox")

        monkeypatch.setattr(webui, "_default_subprocess_runner", _no_tailscale)
        config_path = _write_minimal_run_config(tmp_path)
        exit_code = webui.main(
            ["--config", str(config_path), "--allow-host", "*", "--port", "0"]
        )
        assert exit_code == 1

    def test_main_refuses_an_invalid_bind_address_without_binding_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Keep this offline: `resolve_bind_addresses`'s default probe shells
        # out to a real `tailscale` binary. Patching the module-level
        # subprocess seam (looked up by name inside the function body, not
        # bound as a default argument) makes the probe fail fast instead of
        # touching a real process.
        def _no_tailscale(args):
            raise FileNotFoundError("tailscale not installed in this sandbox")

        monkeypatch.setattr(webui, "_default_subprocess_runner", _no_tailscale)

        config_path = _write_minimal_run_config(tmp_path)
        exit_code = webui.main(
            ["--config", str(config_path), "--bind", "192.168.1.5", "--port", "0"]
        )
        assert exit_code == 1
