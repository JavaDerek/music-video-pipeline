"""HTTP progress monitor for one run, and (opt-in) its start/stop control.

``python -m music_video_maker.webui --config run.toml`` (or the ``mvm-webui``
console script) serves a browser-facing view of a run that is already in
progress, or already finished. It is built entirely on top of
:mod:`music_video_maker.progress`, which does the reading and the diffing;
this module's own job is the socket, the routes, and the one constraint nothing
else in this project enforces for you: **what this process is allowed to bind
to**.

Read-only by default; ``--enable-control`` adds two POST routes
-----------------------------------------------------------------
Until 2026-10-04 this server had no write route at all, and the reason was
not "ran out of time": starting a render means exclusive GPU custody and the
host-wedge failure mode, and nobody had decided what a button may do. The
owner decided on 2026-10-04 that **the page may start and stop a run, with
guards** -- and nothing else. So:

* The control routes are **opt-in** (``--enable-control``). Without the flag
  this process is exactly the read-only monitor it has always been: ``GET``
  and ``HEAD`` only, every other method a flat ``405``, and no form in the
  page. An existing deployment does not acquire a write surface by being
  upgraded.
* The *write surface is two paths*, ``POST /start`` and ``POST /stop``, and
  neither takes anything from the request but a CSRF token and a ``resume``
  flag. Nothing configures a run from here: issue #36's "what the page
  collects" half is still not built, ``run.toml`` is still the source of
  truth, and a human still edits it.
* Every gate the CLI runs, the start runs -- because the start **is** the
  CLI (``python -m music_video_maker.cli --config ...``, spawned in its own
  session). The custody pre-flight, the issue #10 disk check, the issues
  #24/#98 render-envelope refusal, ``prevent_host_sleep`` and the
  unconditional ``POST /free`` in a ``finally`` are all the pipeline's own
  and are not reimplemented here. :mod:`music_video_maker.control` holds the
  pre-spawn gate (which calls the same functions), the one-claim lock, and
  the stop; read its module docstring before changing any of this.
* A stop is a SIGINT to the child's process group, which is what Ctrl-C is.
  There is no second shutdown path.

What a *request* can reach is therefore: read a run's own data, start the run
this config already describes, or interrupt it. It cannot name a path, a
chunk, a resolution or a config value. That matters because anything able to
start a render can write files and load custom nodes *by proxy through
ComfyUI*, which has no auth of its own -- so the control half is defended as
a write route in its own right (see "Reaching the control routes" below) and
not merely by the bind list.

Why this module never recomputes the chunk timeline
------------------------------------------------------
The obvious way to answer "how many chunks does this run have in total" is
to re-run Stage 1-2 (``alignment.align`` + ``slicing.slice_audio``) the way
``--prepare`` does. That function is **not** read-only: its own docstring
says it "Exports ``chunk_{idx:03d}.wav`` into ``chunks_dir``" -- the *same*
``chunks_dir`` a live render already reads and writes. A monitor that ran it
would race an in-flight run for that directory and burn disk on a machine
CLAUDE.md already flags as tight (doris runs high-90s%% full; the driving Mac
is not much better). So this module never imports ``alignment`` or
``slicing``, and "the run's known chunks" means exactly *the chunk ids that
already have an entry in ``run_state.json``* (plus, if the run stopped below
the VRAM floor, the one chunk id ``RunState.vram_stop`` names, which never
gets a ``ChunkResult`` of its own -- see :func:`_known_chunk_ids`). That is
strictly a subset of the run's true chunk plan until the last chunk lands, so
:class:`~music_video_maker.progress.RunProgress`'s ``total``/``finished``
here mean "recorded so far", never "the whole plan" -- the rendered page says
so in words rather than implying a total it cannot back up.

The true chunk plan belongs to the thing that runs Stage 1-2 once, on
demand, from the CLI -- and that is ``--prepare``. ``GET /prepare`` shows
what its last run found (alignment quality, every Stage-2 warning, the
timeline-versus-track drift, and any shot-plan drift) by **reading the JSON
report ``--prepare`` writes** (:mod:`music_video_maker.prepare_report`,
``RunConfig.prepare_report_file``). A run nobody has prepared renders a page
saying so, with the command; the page never offers to run it. The same rule
as above, stated from the other side: the measurement is a step the operator
invokes, and this process only ever reads what it left behind.

The bind-address constraint
------------------------------
ComfyUI has no authentication and is bound to loopback plus its Tailscale
address for exactly this reason (CLAUDE.md, "Infrastructure (doris)"). A
process serving this run's data over HTTP inherits the same threat model, so
:func:`resolve_bind_addresses` is the one function in this module that must
never be "simplified": it always includes ``127.0.0.1``, adds this host's
Tailscale IPv4 address if ``tailscale ip -4`` finds one (logged and skipped,
never fatal, if it does not), and validates every operator-supplied
``--bind`` address the same way -- loopback or inside Tailscale's own
address ranges (``100.64.0.0/10``, ``fd7a:115c:a1e0::/48``) or refused
outright. Nothing here resolves a hostname; an address must already be an IP
literal, so a DNS name that happens to resolve to a LAN address can never
sneak through by never being looked up.

The Host header, and why binding correctly is not enough
-----------------------------------------------------------
A correct bind list stops the *network* from reaching this server. It does
not stop a *browser* from being told to reach it: in a DNS-rebinding attack
a page on ``evil.example`` is served from a name whose DNS record flips to
``127.0.0.1`` a few seconds later, and the victim's own browser then issues
same-origin requests to this server carrying ``Host: evil.example`` -- and
reads the responses, because as far as it is concerned the origin never
changed. Loopback-only binding is exactly what that attack exists to
defeat, so "we only bind to 127.0.0.1" is the precondition for the attack,
not a defence against it.

:func:`host_header_allowed` is the defence, and it runs *before* any route
handler (see :meth:`MonitorRequestHandler._reject_unless_host_allowed`): a
request is answered only when its ``Host`` names **this server's own bound
address** (the literal, with an optional port that must match), the name
``localhost``, or something the operator explicitly listed with
``--allow-host`` -- a Tailscale MagicDNS name, typically, which is the one
real case a literal does not cover. Everything else gets ``421 Misdirected
Request`` with no run data in the body. The rejected value is logged, never
echoed back into the response.

Two deliberate choices in it:

* **An absent ``Host`` is allowed.** Rebinding cannot produce one: the
  attacker's leverage *is* the name in that header, and a browser will not
  let script suppress or forge it. Refusing an absent header would refuse
  only HTTP/1.0 clients (``curl -0``, a hand-rolled socket probe) while
  blocking no attack.
* **More than one ``Host`` header is refused.** One request with two
  authorities is a request-smuggling shape, not something a client this
  server should answer produces by accident.

Reaching the control routes
------------------------------
The Host check above stops a *rebound name*. It does not stop a form on a
page the operator is simply visiting: a cross-site ``POST`` of
``application/x-www-form-urlencoded`` needs no CORS preflight, so "the
browser is pointed at loopback" is all an attacker's page would need if the
only check were the bind list. :func:`control_request_refusal` is the
defence, and it is three checks that must all pass, deliberately layered
rather than picked:

1. **A per-process CSRF token**, generated at startup
   (:func:`new_control_token`), embedded in the page's forms and compared
   with ``secrets.compare_digest``. This is the load-bearing one: the token
   is only obtainable by *reading* ``GET /``, which a cross-origin script
   cannot do (no CORS headers are ever sent, and a rebound name is refused
   by the Host check before any handler runs).
2. **``Sec-Fetch-Site`` must be ``same-origin`` or ``none``** when present.
   Every current browser sends it and script cannot forge it; ``none`` is a
   typed navigation. ``same-site`` is refused too -- a sibling host is not
   this origin. Absent (``curl``, an old client) falls through to the token.
3. **``Origin``, when present, must be this server's own origin** -- the
   bound literal, ``localhost``, or an ``--allow-host`` name, on this port,
   over ``http``. ``null`` (a sandboxed iframe, a ``file://`` page) is
   refused rather than treated as absent.

Two smaller rules in the same spirit: only
``application/x-www-form-urlencoded`` is accepted (so a cross-origin
``fetch`` would need a content type requiring a preflight, and ``OPTIONS``
is a flat ``405`` with no CORS headers), and the index carries
``Cache-Control: no-store`` when control is on so the token is not stored by
anything on the way.

A refusal says which check failed and never echoes the value it rejected.

Testing (issue #36's day-one list)
-------------------------------------
The design doc names two tests that must exist on day one for a server like
this. Both exist:

* the bind-address assertion -- :func:`resolve_bind_addresses` and
  :func:`validate_bind_address` are pure and covered directly; a handful of
  tests also build real ``ThreadingHTTPServer`` instances on loopback with
  port 0 and assert ``server.server_address``, per the task's own
  instruction, rather than trusting the pure function alone.
* "starting a run while one is in flight is refused" -- this used to be
  recorded as *not applicable*, there being no start route. It applies now:
  ``tests/test_control.py::TestOneRunAtATime`` covers the gate and the
  cross-process claim, and ``tests/test_webui.py::TestControlStart`` covers
  the ``409`` a browser gets.
"""

from __future__ import annotations

import argparse
import html
import ipaddress
import logging
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from music_video_maker.config import ConfigError, load_config
from music_video_maker.contracts import RunState
from music_video_maker.control import (
    ControlError,
    RenderController,
    RenderState,
    RenderStatus,
    StartGate,
    StartRefused,
    StopRefused,
    read_log_tail,
)
from music_video_maker.logging_setup import configure_logging
from music_video_maker.prepare_report import (
    PrepareReport,
    PrepareReportError,
    read_prepare_report,
    stale_inputs,
)
from music_video_maker.progress import (
    ProgressError,
    ProgressEvent,
    RunProgress,
    events_between,
    format_sse,
    read_run_state,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Bind-address validation -- the non-negotiable (see module docstring)
# --------------------------------------------------------------------------- #

TAILSCALE_IPV4_RANGE = ipaddress.ip_network("100.64.0.0/10")
"""Tailscale's CGNAT range. See <https://tailscale.com/kb/1015/100.x-addresses>."""

TAILSCALE_IPV6_RANGE = ipaddress.ip_network("fd7a:115c:a1e0::/48")
"""Tailscale's ULA range for its IPv6 addresses."""


class BindAddressError(ValueError):
    """An address is neither loopback nor inside a Tailscale range.

    Raised for anything else, including ``0.0.0.0``, ``::``, a LAN address
    like ``192.168.x.x``, and any string that is not an IP literal at all
    (hostnames are deliberately never resolved -- see the module docstring)."""


def _parse_ip_literal(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(raw)
    except ValueError as exc:
        raise BindAddressError(
            f"{raw!r} is not an IP address literal -- hostnames are not resolved here, "
            "pass a literal loopback or Tailscale address"
        ) from exc


def _is_permitted(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback:
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in TAILSCALE_IPV4_RANGE
    return ip in TAILSCALE_IPV6_RANGE


def validate_bind_address(raw: str) -> str:
    """Return ``raw`` (canonicalised) if it is loopback or inside a Tailscale
    range; raise :class:`BindAddressError` otherwise.

    This is the one function standing between an operator's ``--bind`` flag
    and a real socket. ``0.0.0.0``, ``::``, ``192.168.1.5``, and
    ``"my-laptop.local"`` are all refused -- the first two are wildcards, the
    third is a LAN address, and the fourth is not an IP literal at all."""
    ip = _parse_ip_literal(raw)
    if not _is_permitted(ip):
        raise BindAddressError(
            f"{raw!r} is neither loopback nor inside a Tailscale range "
            f"({TAILSCALE_IPV4_RANGE}, {TAILSCALE_IPV6_RANGE}) -- refusing to bind"
        )
    return str(ip)


SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]
"""Same shape as ``assembly.SubprocessRunner`` / ``continuity.SubprocessRunner``
-- an injectable seam so tests never spawn a real ``tailscale`` or ``ffmpeg``."""


def _default_subprocess_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real subprocess invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False, timeout=10)


def default_tailscale_ipv4(*, runner: SubprocessRunner | None = None) -> str | None:
    """Best-effort ``tailscale ip -4``. ``None`` (logged, never raised) if the
    binary is absent, times out, or exits non-zero -- absence of Tailscale is
    an ordinary, expected case (a laptop with no tailnet), not a server error."""
    run = runner or _default_subprocess_runner
    try:
        result = run(["tailscale", "ip", "-4"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("`tailscale ip -4` could not run (%s) -- binding to loopback only", exc)
        return None
    if result.returncode != 0:
        logger.info(
            "`tailscale ip -4` exited %s -- binding to loopback only", result.returncode
        )
        return None
    stdout = result.stdout
    text = stdout if isinstance(stdout, str) else stdout.decode("utf-8", "replace")
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or None


def resolve_bind_addresses(
    explicit: Sequence[str] = (),
    *,
    tailscale_ipv4: Callable[[], str | None] = default_tailscale_ipv4,
) -> tuple[str, ...]:
    """The bind list: always ``127.0.0.1``, plus this host's Tailscale IPv4
    address if one is found, plus every validated ``explicit`` address.

    An invalid ``explicit`` address raises :class:`BindAddressError`
    immediately -- the operator asked for it by name, so refusing loudly is
    correct. An unusable *automatic* Tailscale probe (absent, failed, or --
    defensively -- an address that somehow fails validation) only logs and
    falls back to loopback-only, because "no tailnet on this machine" is not
    an operator mistake to refuse.

    Order is stable and de-duplicated: ``127.0.0.1`` first, then the
    Tailscale address if any, then ``explicit`` addresses in the order
    given, skipping anything already present."""
    addresses: list[str] = ["127.0.0.1"]

    auto = tailscale_ipv4()
    if auto:
        try:
            validated = validate_bind_address(auto)
        except BindAddressError:
            logger.warning(
                "`tailscale ip -4` returned %r, which is not a valid Tailscale address -- "
                "ignoring it and binding to loopback only",
                auto,
            )
        else:
            if validated not in addresses:
                addresses.append(validated)
    else:
        logger.info("No Tailscale IPv4 address found -- binding to loopback only.")

    for raw in explicit:
        validated = validate_bind_address(raw)
        if validated not in addresses:
            addresses.append(validated)

    return tuple(addresses)


# --------------------------------------------------------------------------- #
# Host-header validation -- the DNS-rebinding defence (see the module
# docstring's "The Host header, and why binding correctly is not enough")
# --------------------------------------------------------------------------- #

ALWAYS_ALLOWED_HOST_NAMES = frozenset({"localhost"})
"""Names always accepted in a ``Host`` header, whatever this server bound to.

``localhost`` only, and it is safe for the same reason an IP literal is: an
attacker's page cannot make a browser send this value at all. Reaching this
server as ``localhost`` requires typing (or linking) ``http://localhost:PORT``,
which is a first-party navigation whose response a cross-origin script still
cannot read. What rebinding actually delivers is the *attacker's own* name,
and that is what this set does not contain."""


class HostAllowlistError(ValueError):
    """An operator-supplied ``--allow-host`` value is not a usable host name.

    Refused rather than normalised away: a wildcard, an empty string, a URL,
    or anything carrying a port, path, scheme or userinfo is far more likely
    to be a misunderstanding of what this flag does than a name somebody
    meant, and a silently-ignored allowlist entry is the failure mode where
    an operator believes a check is looser than it is."""


_HOST_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")


def normalise_allowed_host(raw: str) -> str:
    """Canonicalise one ``--allow-host`` value: case-folded, with a trailing
    root dot removed (``doris.`` and ``doris`` are the same name, and a
    browser may send either).

    Raises :class:`HostAllowlistError` for anything that is not a bare host
    name or IP literal -- including ``*``, ``""``, ``http://doris``,
    ``doris:8787`` and ``doris/path``."""
    candidate = raw.strip().rstrip(".").casefold()
    if not candidate:
        raise HostAllowlistError(f"{raw!r} is not a host name")
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        pass
    if not _HOST_NAME_RE.match(candidate):
        raise HostAllowlistError(
            f"{raw!r} is not a bare host name -- pass just the name a browser would send in "
            "its Host header (no scheme, no port, no path, no wildcard)"
        )
    return candidate


def _split_host_header(raw: str) -> tuple[str, int | None] | None:
    """``"[::1]:8787"`` -> ``("::1", 8787)``; ``None`` when ``raw`` is not a
    plain authority this server should answer for.

    Parsed through :func:`urllib.parse.urlsplit` rather than by hand so the
    bracketed-IPv6 and port rules are the stdlib's, not a second guess at
    them. Anything carrying a path, scheme, userinfo or whitespace is
    rejected outright before parsing: a ``Host`` header is an authority, and
    a value shaped like anything else is not a client this server needs to
    understand."""
    candidate = raw.strip()
    if not candidate or any(ch in candidate for ch in "/\\@ \t"):
        return None
    try:
        parsed = urlsplit(f"//{candidate}")
        hostname, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if hostname is None or parsed.path or parsed.query or parsed.fragment:
        return None
    return hostname, port


def host_header_allowed(
    raw: str | None,
    *,
    bound_address: str,
    port: int,
    extra_allowed: frozenset[str] = frozenset(),
) -> bool:
    """Whether a request carrying ``Host: raw`` may be answered by a server
    listening on ``bound_address:port``.

    ``raw is None`` (no ``Host`` header at all) is **allowed** -- see the
    module docstring for why that refuses nothing an attacker can do. Every
    other value must be one of:

    * this server's own bound address as an IP literal, compared as an
      address rather than as text (so ``::1`` and ``0:0:0:0:0:0:0:1`` are the
      same host) and with an optional port that must equal ``port``;
    * ``localhost`` (:data:`ALWAYS_ALLOWED_HOST_NAMES`);
    * a name in ``extra_allowed``, which comes only from ``--allow-host``.

    A name is never resolved, exactly as :func:`validate_bind_address` never
    resolves one: resolution is the mechanism this check exists to defeat, so
    performing one here to "see if it points at us" would answer the
    attacker's own DNS query and let the attack through by design."""
    if raw is None:
        return True
    split = _split_host_header(raw)
    if split is None:
        return False
    hostname, host_port = split
    if host_port is not None and host_port != port:
        return False
    try:
        requested = ipaddress.ip_address(hostname)
    except ValueError:
        name = hostname.rstrip(".").casefold()
        return name in ALWAYS_ALLOWED_HOST_NAMES or name in extra_allowed
    try:
        bound = ipaddress.ip_address(bound_address)
    except ValueError:  # pragma: no cover - make_server only ever binds literals
        return False
    return requested == bound


# --------------------------------------------------------------------------- #
# Reaching the control routes -- the write-route defence (see the module
# docstring's "Reaching the control routes")
# --------------------------------------------------------------------------- #

CONTROL_PATHS = ("/start", "/stop")
"""The entire write surface. Two paths, no parameters that reach a file."""

CONTROL_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"

MAX_CONTROL_BODY_BYTES = 4096
"""A control form is two short fields. Anything larger is not one of ours."""

SAFE_FETCH_SITES = frozenset({"same-origin", "none"})
"""``Sec-Fetch-Site`` values this server will act on. ``none`` is a typed
navigation or a bookmark; ``same-site`` is deliberately **not** here -- a
sibling host is not this origin."""


def new_control_token() -> str:
    """A fresh CSRF token for one server process.

    Per process, never persisted: restarting ``mvm-webui`` invalidates every
    form a browser is still holding, which is the right direction for a
    token whose only job is to prove the submitter had read a page served by
    *this* process."""
    return secrets.token_urlsafe(32)


def _origin_is_own(
    raw: str,
    *,
    bound_address: str,
    port: int,
    extra_allowed: frozenset[str],
) -> bool:
    """Whether an ``Origin`` header names this server's own origin.

    Reuses :func:`host_header_allowed` for the authority so there is one
    definition of "a host this server answers for", and adds the two things
    an origin has that a ``Host`` does not: a scheme (``http`` only -- this
    server speaks no TLS, so an ``https`` origin on its port is not it) and
    the literal value ``null``, which a sandboxed iframe or a ``file://``
    page sends and which must be refused rather than read as absent."""
    parsed = urlsplit(raw.strip())
    if parsed.scheme != "http" or not parsed.netloc:
        return False
    return host_header_allowed(
        parsed.netloc,
        bound_address=bound_address,
        port=port,
        extra_allowed=extra_allowed,
    )


def control_request_refusal(
    *,
    origin: str | None,
    sec_fetch_site: str | None,
    token: str | None,
    expected_token: str,
    bound_address: str,
    port: int,
    extra_allowed: frozenset[str] = frozenset(),
) -> str | None:
    """``None`` when a control POST may be acted on, else the reason it may
    not -- operator-facing text, naming the check rather than the value.

    See the module docstring's "Reaching the control routes" for why all
    three checks are here and which one is load-bearing."""
    if not expected_token:
        return "this server was not started with --enable-control"
    if sec_fetch_site is not None and sec_fetch_site.strip().lower() not in SAFE_FETCH_SITES:
        return (
            "the browser's own Sec-Fetch-Site header says this request did not come from "
            "this page, so it is refused without looking at anything else in it"
        )
    if origin is not None and not _origin_is_own(
        origin, bound_address=bound_address, port=port, extra_allowed=extra_allowed
    ):
        return (
            "the Origin header is not this server's own origin. If you reach this monitor "
            "by a name, start it with --allow-host <name>"
        )
    if not token or not secrets.compare_digest(token, expected_token):
        return (
            "the control token in the submitted form is missing or does not match this "
            "server's. Reload the page (a token belongs to one server process) and try again"
        )
    return None


# --------------------------------------------------------------------------- #
# What "the run's known chunks" means here -- see the module docstring's
# "Why this module never recomputes the chunk timeline"
# --------------------------------------------------------------------------- #


def _known_chunk_ids(run_state: RunState) -> tuple[int, ...]:
    """Chunk ids this module will ever show a row for: every id with a
    ``ChunkResult``, plus the one id ``RunState.vram_stop`` names if the run
    stopped below the VRAM floor before that chunk got a result at all (see
    ``contracts.VramStopEvent``'s docstring) -- included here so that chunk's
    row renders as "pending" rather than not existing."""
    ids = set(run_state.results)
    if run_state.vram_stop is not None:
        ids.add(run_state.vram_stop.chunk_id)
    return tuple(sorted(ids))


def current_progress(run_state: RunState) -> RunProgress:
    """Build the :class:`RunProgress` this module actually shows: ``total``
    means "chunks recorded so far", not the run's full plan -- see the module
    docstring."""
    return RunProgress.from_run_state(run_state, _known_chunk_ids(run_state))


def is_valid_chunk_id(run_state: RunState, chunk_id: int) -> bool:
    """Whether ``chunk_id`` is one this run actually knows about. Used to
    validate the ``<id>`` path segment on the thumbnail route -- never used
    to build a filesystem path, only to look a dict key up."""
    return chunk_id in _known_chunk_ids(run_state)


# --------------------------------------------------------------------------- #
# SSE event stream -- pure generator, testable without a socket
# --------------------------------------------------------------------------- #


def stream_progress_events(
    *,
    run_state_path: Path,
    poll_interval_seconds: float = 2.0,
    sleeper: Callable[[float], None] = time.sleep,
    max_polls: int | None = None,
) -> Iterator[ProgressEvent]:
    """Poll ``run_state_path`` and yield the events issue #36's SSE route
    should relay, forever (``max_polls=None``) or a bounded number of times
    (tests only).

    A missing or torn state file is "not ready yet" -- it is silently
    skipped, this poll contributes no events, and the loop tries again next
    interval. It is never raised through to a caller, matching the module's
    scope note: a poller hitting any read failure treats it as "try again
    shortly", the same stance :func:`~music_video_maker.progress.read_run_state`
    already documents.

    Callers doing real work inject nothing but the default ``time.sleep``;
    tests inject a fake ``sleeper`` (which may itself rewrite
    ``run_state_path`` to the next fixture state when called) and a
    ``max_polls`` bound, so nothing here ever sleeps in a test."""
    previous: RunProgress | None = None
    polls = 0
    while max_polls is None or polls < max_polls:
        try:
            run_state = read_run_state(run_state_path)
        except ProgressError:
            run_state = None
        if run_state is not None:
            current = current_progress(run_state)
            yield from events_between(previous, current)
            previous = current
        polls += 1
        if max_polls is None or polls < max_polls:
            sleeper(poll_interval_seconds)


# --------------------------------------------------------------------------- #
# Thumbnails -- extracted on demand, cached outside the repo and outside
# chunks_dir (never write into either -- see the module docstring)
# --------------------------------------------------------------------------- #


class ThumbnailError(RuntimeError):
    """ffmpeg could not extract a frame from a chunk's video file."""


def _thumbnail_cache_key(video_path: Path) -> str:
    stat = video_path.stat()
    # mtime (ns) + size: a chunk re-rendered under --resume gets a new mtime
    # even if the filename is reused, so a stale cached frame is never served.
    return f"{video_path.stem}-{stat.st_mtime_ns}-{stat.st_size}.png"


def get_or_render_thumbnail(
    video_path: Path,
    *,
    cache_dir: Path,
    runner: SubprocessRunner,
    seek_seconds: float = 1.0,
) -> bytes:
    """Return a PNG frame from ``video_path``, extracting it through
    ``runner`` (an injected ffmpeg seam, never a direct ``subprocess`` call)
    and caching it under ``cache_dir``.

    ``cache_dir`` is the caller's choice -- see :data:`DEFAULT_THUMBNAIL_CACHE_DIR`
    for the default -- and must never be the repo or the run's ``chunks_dir``:
    this module writes to it on every cache miss, and a monitor is not
    allowed to write into either of those (see the module docstring)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / _thumbnail_cache_key(video_path)
    if cached_path.is_file():
        return cached_path.read_bytes()

    args = [
        "ffmpeg",
        "-y",
        "-ss",
        str(seek_seconds),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "4",
        str(cached_path),
    ]
    result = runner(args)
    if result.returncode != 0 or not cached_path.is_file():
        stderr = result.stderr if result.stderr is not None else b""
        stderr_text = (
            stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
        )
        raise ThumbnailError(
            f"ffmpeg could not extract a thumbnail from {video_path} (exit={result.returncode}): "
            f"{stderr_text}"
        )
    return cached_path.read_bytes()


DEFAULT_THUMBNAIL_CACHE_DIR = Path(tempfile.gettempdir()) / "mvm-webui-thumbnails"
"""Outside the repo, outside every run's ``chunks_dir``, and namespaced per
``run_id`` at request time (``DEFAULT_THUMBNAIL_CACHE_DIR / run_id``) so two
runs whose chunk ids collide never serve each other's frames."""


# --------------------------------------------------------------------------- #
# HTML rendering -- every piece of run data is escaped; nothing here trusts
# a lyric, an error string, or a filename to be safe markup
# --------------------------------------------------------------------------- #


def _e(value: object) -> str:
    """Shorthand: ``str(value)``, HTML-escaped. The one function every piece
    of run-derived text must pass through before landing in a page."""
    return html.escape(str(value))


_STATUS_CSS_CLASS = {
    "pending": "status-pending",
    "rendered": "status-rendered",
    "cached": "status-cached",
    "dead_lettered": "status-dead",
    "failed": "status-failed",
}


def _render_chunk_rows(progress: RunProgress) -> str:
    rows = []
    for chunk in progress.chunks:
        css = _STATUS_CSS_CLASS.get(chunk.status, "status-failed")
        errors = "; ".join(chunk.errors)
        reason = chunk.rerender_reason or ""
        if chunk.rerender_reason_fields:
            reason = f"{reason} ({', '.join(chunk.rerender_reason_fields)})"
        vram = (
            f"{chunk.free_vram_gb_before:.2f} GB" if chunk.free_vram_gb_before is not None else "-"
        )
        thumb = (
            f'<a href="/chunks/{chunk.chunk_id}/thumbnail.png">thumbnail</a>'
            if chunk.video_file
            else "-"
        )
        rows.append(
            "<tr class=\"{css}\">"
            "<td>{cid}</td><td>{status}</td><td>{attempts}</td>"
            "<td>{vram}</td><td>{reason}</td><td>{errors}</td><td>{thumb}</td>"
            "</tr>".format(
                css=_e(css),
                cid=_e(chunk.chunk_id),
                status=_e(chunk.status),
                attempts=_e(chunk.attempts),
                vram=_e(vram),
                reason=_e(reason) if reason else "-",
                errors=_e(errors) if errors else "-",
                thumb=thumb,  # built above from a validated int, not request text
            )
        )
    return "\n".join(rows)


def _render_dead_letters(progress: RunProgress) -> str:
    if not progress.dead_lettered:
        return "<p>No dead-lettered chunks.</p>"
    by_id = {chunk.chunk_id: chunk for chunk in progress.chunks}
    items = []
    for chunk_id in progress.dead_lettered:
        chunk = by_id[chunk_id]
        errors = "<br>".join(_e(e) for e in chunk.errors) or "(no error recorded)"
        items.append(
            f"<li><strong>Chunk {_e(chunk_id)}</strong> "
            f"({_e(chunk.attempts)} attempt(s)):<br>{errors}</li>"
        )
    return "<ul class=\"dead-letters\">" + "\n".join(items) + "</ul>"


def _render_vram_stop_notice(progress: RunProgress) -> str:
    if progress.vram_stop_chunk_id is None:
        return ""
    return (
        '<div class="vram-stop">'
        "<strong>VRAM stop:</strong> the run refused to submit chunk "
        f"{_e(progress.vram_stop_chunk_id)} because free VRAM read "
        f"{_e(f'{progress.vram_stop_free_vram_gb:.2f}')} GB, below the "
        f"{_e(f'{progress.vram_stop_floor_gb:.2f}')} GB floor. "
        "Resume once the card is clear again."
        "</div>"
    )


_PAGE_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; padding: 16px;
       background: #f7f7f8; color: #1a1a1a; }
h1 { font-size: 1.25rem; }
.summary { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.summary div { background: #fff; border: 1px solid #ddd; border-radius: 6px;
               padding: 8px 12px; }
.note { color: #555; font-size: 0.85rem; max-width: 60ch; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; font-size: 0.9rem; }
th { background: #eee; }
.status-rendered { background: #eaffea; }
.status-cached { background: #eaf2ff; }
.status-dead { background: #ffecec; }
.status-failed { background: #fff6e0; }
.status-pending { background: #fafafa; color: #888; }
.vram-stop { background: #ffecec; border: 1px solid #e88; border-radius: 6px;
             padding: 8px 12px; margin: 12px 0; }
.dead-letters { background: #fff; border: 1px solid #ddd; border-radius: 6px;
                padding: 8px 16px; }
.sev-CRITICAL { background: #ffecec; }
.sev-WARNING { background: #fff6e0; }
.level-ERROR { background: #ffecec; }
.level-WARNING { background: #fff6e0; }
.banner { border-radius: 6px; padding: 8px 12px; margin: 12px 0;
          background: #fff; border: 1px solid #ddd; }
.banner-bad { background: #ffecec; border-color: #e88; }
.msg { font-family: ui-monospace, monospace; font-size: 0.82rem; white-space: pre-wrap; }
.control { background: #fff; border: 1px solid #ddd; border-radius: 6px;
           padding: 8px 12px; margin: 12px 0; }
.control form { display: inline-block; margin-right: 16px; }
.gate-ok { background: #eaffea; }
.gate-bad { background: #ffecec; }
.logtail { background: #111; color: #eee; border-radius: 6px; padding: 8px 12px;
           font-family: ui-monospace, monospace; font-size: 0.78rem;
           white-space: pre-wrap; overflow-x: auto; }
"""

_SSE_SCRIPT = """
try {
  var box = document.getElementById('live-updates');
  var src = new EventSource('/events');
  src.onmessage = function () {};
  ['run_snapshot', 'chunk_completed', 'chunk_dead_lettered', 'chunk_retried',
   'run_stopped', 'run_finished'].forEach(function (name) {
    src.addEventListener(name, function (evt) {
      if (box) {
        var line = document.createElement('div');
        line.textContent = name + ': ' + evt.data;
        box.prepend(line);
      }
      // A full reload keeps the render logic in one place (server-side);
      // this script only proves the connection is live for anyone watching.
      if (name !== 'run_snapshot') { location.reload(); }
    });
  });
} catch (err) { /* EventSource unsupported: the static snapshot above still renders. */ }
"""


def _render_gate(gate: StartGate) -> str:
    rows = "\n".join(
        '<tr class="{css}"><td>{name}</td><td>{verdict}</td><td class="msg">{detail}</td></tr>'
        .format(
            css="gate-ok" if check.passed else "gate-bad",
            name=_e(check.name),
            verdict="ok" if check.passed else "REFUSED",
            detail=_e(check.detail),
        )
        for check in gate.checks
    )
    return (
        "<table><tr><th>pre-flight check</th><th></th><th>what it found</th></tr>"
        + rows
        + "</table>"
    )


def _render_log_tail(status: RenderStatus) -> str:
    if status.log_path is None:
        return ""
    tail = read_log_tail(status.log_path)
    if not tail:
        return (
            f'<p class="note">Nothing written to <code>{_e(status.log_path)}</code> yet.</p>'
        )
    return (
        f'<p class="note">The last lines of <code>{_e(status.log_path)}</code> -- this is '
        "where a refusal from inside the run itself appears (a per-chunk envelope miss, a "
        "config error, a strict-alignment stop):</p>"
        f'<div class="logtail">{_e(tail)}</div>'
    )


def render_control_section(
    status: RenderStatus, *, gate: StartGate | None, token: str
) -> str:
    """The control half of ``GET /``: what the run is doing, every pre-flight
    check with its reading, and the one or two buttons that are honest right
    now.

    ``gate`` is ``None`` while a render is in flight: there is nothing useful
    to say about free VRAM during a render that is using it, and probing
    ComfyUI once per page load while it renders is noise. Called only when
    ``--enable-control`` is on -- without it this never appears and there is
    no form in the page at all."""
    if status.state is RenderState.RUNNING and status.claim is not None:
        claim = status.claim
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(claim.started_at))
        lines = [
            f'<div class="banner"><strong>Rendering</strong> &mdash; pid {_e(claim.pid)}, '
            f"started {_e(started)}"
            + (" (resumed)" if claim.resume else "")
            + ". This process holds exclusive GPU custody.</div>"
        ]
        if claim.stop_requested_at is not None:
            asked = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(claim.stop_requested_at)
            )
            lines.append(
                f'<p class="note">A stop was requested at {_e(asked)}. The run finishes the '
                "chunk it is on, persists its state and releases the card in a "
                "<code>finally</code> &mdash; the same stop as Ctrl-C. Nothing here escalates "
                "to SIGKILL, because killing a render inside its <code>POST /free</code> is "
                "how the card gets left holding 17 GB.</p>"
            )
        if status.owned:
            lines.append(
                '<form method="post" action="/stop">'
                f'<input type="hidden" name="token" value="{_e(token)}">'
                "<button type=\"submit\">Stop this run (SIGINT, like Ctrl-C)</button>"
                "</form>"
            )
        else:
            lines.append(
                '<p class="note">This server did not start the run in flight, so it will not '
                "signal it: a pid read out of a file this process did not write could have "
                "been reused by something unrelated. Stop it with Ctrl-C in its own terminal, "
                f"or <code>kill -INT {_e(status.claim.pgid)}</code> once you have confirmed "
                "what it is.</p>"
            )
        lines.append(_render_log_tail(status))
        return '<h2>Control</h2><div class="control">' + "\n".join(lines) + "</div>"

    lines = []
    if status.state is RenderState.EXITED and status.claim is not None:
        code = (
            f"exit code {status.exit_code}"
            if status.exit_code is not None
            else "exit code unknown (this server did not spawn it, or was restarted)"
        )
        lines.append(
            f'<div class="banner"><strong>The last run this page started has exited</strong> '
            f"&mdash; pid {_e(status.claim.pid)}, {_e(code)}.</div>"
        )
    allowed = gate is not None and gate.allowed
    if gate is not None:
        lines.append(_render_gate(gate))
    lines.append(
        '<form method="post" action="/start">'
        f'<input type="hidden" name="token" value="{_e(token)}">'
        '<label><input type="checkbox" name="resume" value="1"> --resume '
        "(reuse chunks whose fingerprint still matches)</label> "
        f'<button type="submit"{"" if allowed else " disabled"}>Start this run</button>'
        "</form>"
    )
    if not allowed:
        lines.append(
            '<p class="note">At least one pre-flight check above refuses this run, so there '
            "is nothing to start yet. A start is refused, never queued &mdash; GPU custody is "
            "exclusive.</p>"
        )
    lines.append(
        '<p class="note">Starting here runs the same CLI you would run by hand '
        "(<code>music-video-maker --config &lt;run.toml&gt;</code>), so the custody "
        "pre-flight, the disk check, the render-envelope refusal and the unconditional "
        "<code>POST /free</code> are all the pipeline's own. This page cannot change what "
        "the run does &mdash; edit <code>run.toml</code> for that &mdash; and it does not "
        "stop the card's other tenants: that is still yours to do by hand.</p>"
    )
    lines.append(_render_log_tail(status))
    return '<h2>Control</h2><div class="control">' + "\n".join(lines) + "</div>"


def render_control_refusal_html(title: str, detail: str) -> bytes:
    """The page a refused ``POST /start``/``/stop`` returns: why, in words,
    with a way back. Every piece of it is escaped -- a gate's detail carries
    file paths and a shot plan's own error text."""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>refused</title><style>{_PAGE_CSS}</style></head>
<body><h1>{_e(title)}</h1>
<div class="banner banner-bad msg">{_e(detail)}</div>
<p><a href="/">&larr; run monitor</a></p>
</body></html>""".encode()


def render_index_html(
    progress: RunProgress | None,
    *,
    not_ready_reason: str | None = None,
    control_section: str = "",
) -> bytes:
    """Render ``GET /``. Renders a full, correct snapshot with no JavaScript
    required (issue #36's design doc: "the page must render its initial
    snapshot without JS"); the inline script only upgrades it to live-reload
    on new SSE events.

    ``progress is None`` means ``run_state.json`` does not exist or could not
    be parsed yet -- "not ready yet", per the module's own read contract,
    never a 500."""
    if progress is None:
        reason = _e(not_ready_reason or "waiting for the run to write its first chunk")
        body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>music-video-pipeline run monitor</title><style>{_PAGE_CSS}</style></head>
<body><h1>Run monitor</h1><p class="note">Not ready yet: {reason}.
This page will start showing chunks once <code>run_state.json</code> exists.</p>
{control_section}
</body></html>"""
        return body.encode("utf-8")

    mean = progress.mean_render_seconds
    mean_text = f"{mean:.1f}s" if mean is not None else "n/a (no chunk rendered yet this run)"
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>music-video-pipeline run monitor</title><style>{_PAGE_CSS}</style></head>
<body>
<h1>Run {_e(progress.run_id)}</h1>
<p class="note">"Total" below is <strong>chunks recorded in run_state.json so far</strong>,
not this run's full chunk plan -- this monitor deliberately never recomputes the chunk
timeline (it would write chunk audio into the live run's own directory). See
<code>docs/design-web-ui.md</code>.</p>
{_render_vram_stop_notice(progress)}
<div class="summary">
<div>Recorded: {_e(progress.total)}</div>
<div>Rendered: {_e(progress.rendered)}</div>
<div>Cached: {_e(progress.cached)}</div>
<div>Dead-lettered: {_e(len(progress.dead_lettered))}</div>
<div>Pending (known but not yet resolved): {_e(progress.pending)}</div>
<div>Mean render time (rendered only): {mean_text}</div>
</div>
<p class="note"><a href="/prepare">Pre-render checks</a> &mdash; what the last
<code>--prepare</code> found for this run (alignment quality, Stage-2 warnings, timeline
drift, shot-plan drift). Read from disk; this monitor never runs Stage 1-2 itself.</p>
{control_section}
<h2>Dead-lettered chunks</h2>
{_render_dead_letters(progress)}
<h2>Chunks</h2>
<table>
<tr><th>id</th><th>status</th><th>attempts</th><th>free VRAM before</th>
<th>re-render reason</th><th>errors</th><th>thumbnail</th></tr>
{_render_chunk_rows(progress)}
</table>
<div id="live-updates"></div>
<script>{_SSE_SCRIPT}</script>
</body></html>"""
    return body.encode("utf-8")


# --------------------------------------------------------------------------- #
# The pre-render checks page (issue #36) -- read from what --prepare wrote,
# never recomputed here. See prepare_report.py's module docstring.
# --------------------------------------------------------------------------- #


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def _frames(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} frames"


def _render_prepare_inputs(report: PrepareReport, changed: Sequence[str]) -> str:
    if not report.inputs:
        return "<p>This report records no inputs.</p>"
    rows = []
    for stamp in report.inputs:
        state = "changed since this report" if stamp.label in changed else (
            "unchanged" if stamp.exists else "missing when this report was written"
        )
        rows.append(
            "<tr class=\"{css}\"><td>{label}</td><td>{path}</td><td>{size}</td>"
            "<td>{state}</td></tr>".format(
                css="level-WARNING" if stamp.label in changed else "",
                label=_e(stamp.label),
                path=_e(stamp.path),
                size="-" if stamp.size_bytes is None else _e(f"{stamp.size_bytes:,} bytes"),
                state=_e(state),
            )
        )
    return (
        "<table><tr><th>input</th><th>path</th><th>size when prepared</th><th>now</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def _render_prepare_findings(report: PrepareReport) -> str:
    if not report.alignment_findings:
        return "<p>No alignment findings at WARNING or above.</p>"
    rows = [
        "<tr class=\"sev-{sev}\"><td>{sev}</td><td>{code}</td><td>{span}</td>"
        "<td class=\"msg\">{message}</td></tr>".format(
            sev=_e(row.severity),
            code=_e(row.code),
            span=_e(f"{row.start:.2f}-{row.end:.2f}s"),
            message=_e(row.message),
        )
        for row in report.alignment_findings
    ]
    return (
        "<table><tr><th>severity</th><th>code</th><th>span</th><th>message</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def _render_prepare_notices(report: PrepareReport) -> str:
    if not report.notices:
        return "<p>Stage 1-2 warned about nothing on this prepare.</p>"
    blocks = []
    for issue, notices in report.notices_by_issue():
        heading = (
            f"issue #{_e(issue)}" if issue != "other" else "not attributed to an issue"
        )
        items = "\n".join(
            f'<li class="level-{_e(notice.level)} msg">{_e(notice.message)}</li>'
            for notice in notices
        )
        blocks.append(
            f"<h3>{heading} &mdash; {_e(len(notices))} notice(s)</h3><ul>{items}</ul>"
        )
    return "\n".join(blocks)


def _render_prepare_plan(report: PrepareReport) -> str:
    if report.plan_checked is None:
        return (
            "<p>No shot plan was checked: this run's config sets no <code>shot_plan</code> "
            "and <code>--prepare</code> was given no <code>--from-plan</code>.</p>"
        )
        # (An unauthored run is a legitimate state, not a missing check.)
    basis = (
        "the timeline was re-anchored against this plan's own length_seconds "
        "(--prepare --from-plan)"
        if report.plan_lengths_applied
        else "the timeline was sliced with NO editorial lengths, so a plan that sets "
        "length_seconds describes a different timeline by construction -- re-run with "
        "--prepare --from-plan before acting on anything below"
    )
    if not report.plan_errors:
        return (
            f"<p>{_e(report.plan_checked)} resolves against every chunk in this timeline "
            f"({_e(basis)}).</p>"
        )
    items = "\n".join(f'<li class="msg">{_e(error)}</li>' for error in report.plan_errors)
    return (
        f'<div class="banner banner-bad"><strong>{_e(len(report.plan_errors))} shot-plan '
        f"refusal(s)</strong> resolving {_e(report.plan_checked)} against this timeline "
        f"&mdash; {_e(basis)}.</div><ul>{items}</ul>"
    )


def render_prepare_html(
    report: PrepareReport | None,
    *,
    not_prepared_reason: str | None = None,
    changed_inputs: Sequence[str] = (),
) -> bytes:
    """Render ``GET /prepare``: what the last ``--prepare`` found.

    ``report is None`` means no readable report exists for this run, which
    is an ordinary state (nobody has prepared it yet, or the file is from a
    schema this build does not read) and never a 500. The page then says so
    and gives the command -- it does **not** offer to run it: alignment is
    ~6 s of CPU that writes chunk audio into the live run's own directory,
    and that is a step an operator invokes, not something a page does when
    somebody opens it.

    ``changed_inputs`` comes from :func:`prepare_report.stale_inputs`, i.e.
    re-stat'ing now what the report stamped then. It is the difference
    between "these were the checks" and "these are the checks for the files
    you have"."""
    if report is None:
        reason = _e(not_prepared_reason or "no report file yet")
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>pre-render checks</title><style>{_PAGE_CSS}</style></head>
<body><h1>Pre-render checks</h1>
<p class="note">Nothing has been prepared for this run: {reason}.</p>
<p class="note">Run <code>music-video-maker --config &lt;run.toml&gt; --prepare</code> (Stages
1-2 only: ~50 s of CPU, no GPU, no ComfyUI) and reload this page. This monitor never runs
alignment itself &mdash; it writes chunk audio into the run's own directory, so it is a step
you invoke, not one a page performs when it is opened.</p>
<p><a href="/">&larr; run progress</a></p>
</body></html>""".encode()

    stale_banner = (
        '<div class="banner banner-bad"><strong>Stale:</strong> '
        f"{_e(', '.join(changed_inputs))} changed on disk since this report was written, so "
        "it describes a timeline a render would no longer produce. Re-run "
        "<code>--prepare</code>.</div>"
        if changed_inputs
        else ""
    )
    strict = (
        "on &mdash; a CRITICAL finding refuses the run (and would have refused this prepare)"
        if report.strict_alignment
        else "off &mdash; a CRITICAL finding is reported and the render proceeds anyway"
    )
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>pre-render checks</title><style>{_PAGE_CSS}</style></head>
<body>
<h1>Pre-render checks</h1>
<p class="note">What <code>--prepare</code> found on {_e(report.generated_at)} for
<code>{_e(report.config_path)}</code>. Read from
<code>prepare_report.json</code>; nothing on this page was recomputed when you opened it.</p>
{stale_banner}
<div class="summary">
<div>Chunks: {_e(report.chunk_count)} ({_e(report.voiced_chunk_count)} voiced,
{_e(report.instrumental_chunk_count)} instrumental)</div>
<div>Critical findings: {_e(report.critical_finding_count)}</div>
<div>Warning findings: {_e(report.warning_finding_count)}</div>
<div>Stage 1-2 notices: {_e(len(report.notices))}</div>
<div>Shot-plan refusals: {_e(len(report.plan_errors))}</div>
<div>Timeline drift: {_e(_seconds(report.timeline_drift_seconds))}</div>
</div>
<h2>Alignment (issue #35)</h2>
<p>{_e(report.alignment_summary)}</p>
<p class="note">Whisper model: <code>{_e(report.alignment_model_size or 'default')}</code>.
strict_alignment is {strict}.</p>
{_render_prepare_findings(report)}
<h2>Timeline (issue #22)</h2>
<p>{_e(report.chunk_count)} chunk(s) spanning
{_e(_seconds(report.timeline_start))}&ndash;{_e(_seconds(report.timeline_end))}
against a {_e(_seconds(report.track_duration_seconds))} master track: drift
{_e(_seconds(report.timeline_drift_seconds))}
({_e(_frames(report.timeline_drift_frames))}), tolerance
{_e(_seconds(report.duration_tolerance_seconds))}.</p>
<p class="note">Positive drift overshoots the track and the mux's <code>-shortest</code>
silently discards those rendered frames; negative drift cuts the song's own ending out of
the finished file.</p>
<h2>Shot plan</h2>
{_render_prepare_plan(report)}
<h2>What Stage 1-2 warned about</h2>
{_render_prepare_notices(report)}
<h2>Inputs this report was computed from</h2>
{_render_prepare_inputs(report, changed_inputs)}
<p><a href="/">&larr; run progress</a></p>
</body></html>"""
    return body.encode("utf-8")


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #


@dataclass
class MonitorContext:
    """Everything a request handler needs, attached to the server instance
    (``self.server.ctx``) rather than a module global -- one context is
    shared by every ``ThreadingHTTPServer`` this process runs, one per bound
    address (see :func:`make_servers`)."""

    run_state_path: Path
    prepare_report_path: Path | None = None
    """Where ``--prepare`` wrote this run's pre-render report
    (``RunConfig.prepare_report_file``), served at ``/prepare``. ``None``
    disables that route entirely -- it is a fixed path resolved from the
    config at startup, never a request parameter."""
    review_html_path: Path | None = None
    thumbnail_cache_dir: Path = field(default_factory=lambda: DEFAULT_THUMBNAIL_CACHE_DIR)
    ffmpeg_runner: SubprocessRunner = field(default=_default_subprocess_runner)
    poll_interval_seconds: float = 2.0
    sleeper: Callable[[float], None] = time.sleep
    allowed_hosts: frozenset[str] = frozenset()
    """Extra ``Host`` values this server answers for, from ``--allow-host``
    (normalised through :func:`normalise_allowed_host`). Empty by default:
    the server's own bound address and ``localhost`` are always accepted
    without anything being configured, so the usual case needs no entry at
    all -- this is for a name, typically a Tailscale MagicDNS one, that no
    literal covers."""
    controller: RenderController | None = None
    """The control half, from ``--enable-control``. ``None`` -- the default
    -- means this process has no write route at all: ``POST /start`` and
    ``POST /stop`` are a flat ``405`` like every other method, and the page
    contains no form. Control is opt-in because an existing deployment must
    not acquire a write surface by being upgraded."""
    control_token: str = ""
    """The CSRF token for this process, non-empty exactly when
    :attr:`controller` is set (:func:`new_control_token`)."""


_CHUNK_THUMBNAIL_RE = re.compile(r"^/chunks/(\d+)/thumbnail\.png$")


class MonitorRequestHandler(BaseHTTPRequestHandler):
    """Routes for the read-only monitor. ``self.server.ctx`` is a
    :class:`MonitorContext`, set by :func:`make_servers`."""

    server_version = "mvm-webui/1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        logger.info("%s - %s", self.address_string(), format % args)

    # -- dispatch -------------------------------------------------------- #

    def do_GET(self) -> None:
        if self._reject_unless_host_allowed():
            return
        self._dispatch(write_body=True)

    def do_HEAD(self) -> None:
        if self._reject_unless_host_allowed(write_body=False):
            return
        self._dispatch(write_body=False)

    # -- Host header ------------------------------------------------------ #

    def _reject_unless_host_allowed(self, *, write_body: bool = True) -> bool:
        """Answer ``421`` and return ``True`` when this request's ``Host``
        is not one this server answers for (see the module docstring).

        Called first in every method handler, including the ones that only
        ever return ``405``: "before any handler runs" is the whole point,
        and a rejected request must not be able to tell a ``405`` from a
        ``404`` either, since both would confirm what is listening here."""
        ctx: MonitorContext = self.server.ctx  # type: ignore[attr-defined]
        values = self.headers.get_all("Host") or []
        if len(values) > 1:
            logger.warning(
                "Refusing a request carrying %d Host headers from %s -- one request with two "
                "authorities is a request-smuggling shape, not a client to answer",
                len(values),
                self.address_string(),
            )
            self._send_misdirected(write_body=write_body)
            return True
        raw = values[0] if values else None
        if host_header_allowed(
            raw,
            bound_address=str(self.server.server_address[0]),
            port=int(self.server.server_address[1]),
            extra_allowed=ctx.allowed_hosts,
        ):
            return False
        logger.warning(
            # Logged, never echoed into the response body: the value is
            # attacker-chosen text, and the operator debugging a real
            # misconfiguration reads the log, not the browser.
            "Refusing a request from %s whose Host header (%r) is neither this server's own "
            "bound address (%s:%s) nor localhost nor an --allow-host value (%s) -- see "
            "music_video_maker/webui.py's Host-header section (DNS rebinding)",
            self.address_string(),
            raw,
            self.server.server_address[0],
            self.server.server_address[1],
            sorted(ctx.allowed_hosts) or "none configured",
        )
        self._send_misdirected(write_body=write_body)
        return True

    def _send_misdirected(self, *, write_body: bool) -> None:
        self._send_plain(
            421,
            "This monitor only answers requests addressed to its own bound address or to "
            "localhost. If you reach it by a name (a Tailscale MagicDNS name, say), start it "
            "with --allow-host <name>.\n",
            write_body=write_body,
        )

    def _method_not_allowed(self) -> None:
        if self._reject_unless_host_allowed():
            return
        self._send_method_not_allowed()

    def _send_method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- the control half -------------------------------------------------- #

    def do_POST(self) -> None:
        """``/start`` and ``/stop`` when ``--enable-control`` is on; a flat
        ``405`` otherwise, which is what every POST got before 2026-10-04.

        The Host check runs first, as it does for every other method, so a
        rebound name never reaches the cross-site checks below -- let alone
        the controller."""
        if self._reject_unless_host_allowed():
            return
        ctx: MonitorContext = self.server.ctx  # type: ignore[attr-defined]
        path = urlsplit(self.path).path
        if ctx.controller is None or path not in CONTROL_PATHS:
            self._send_method_not_allowed()
            return

        form = self._read_control_form()
        if form is None:
            return
        refusal = control_request_refusal(
            origin=self.headers.get("Origin"),
            sec_fetch_site=self.headers.get("Sec-Fetch-Site"),
            token=form.get("token"),
            expected_token=ctx.control_token,
            bound_address=str(self.server.server_address[0]),
            port=int(self.server.server_address[1]),
            extra_allowed=ctx.allowed_hosts,
        )
        if refusal is not None:
            # Logged without the rejected value: an Origin and a token are
            # both attacker-chosen text, and the operator debugging a real
            # misconfiguration needs the check's name, not the string.
            logger.warning(
                "Refusing %s from %s: %s", path, self.address_string(), refusal
            )
            self._send_bytes(
                403,
                "text/html; charset=utf-8",
                render_control_refusal_html("Refused", refusal),
                write_body=True,
            )
            return

        if path == "/start":
            self._do_start(ctx, resume=form.get("resume") == "1")
        else:
            self._do_stop(ctx)

    def _read_control_form(self) -> dict[str, str] | None:
        """The POST body as a flat mapping, or ``None`` having already sent
        the refusal.

        Only ``application/x-www-form-urlencoded`` is accepted: a
        cross-origin ``fetch`` with any other content type needs a CORS
        preflight, and this server answers ``OPTIONS`` with a flat ``405``
        carrying no CORS headers at all."""
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != CONTROL_FORM_CONTENT_TYPE:
            self._send_plain(
                415,
                f"this route accepts {CONTROL_FORM_CONTENT_TYPE} only\n",
                write_body=True,
            )
            return None
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length)  # type: ignore[arg-type]
            if length < 0:
                raise ValueError("a negative Content-Length")
        except (TypeError, ValueError):
            self._send_plain(
                411,
                "a control POST must carry a non-negative Content-Length\n",
                write_body=True,
            )
            return None
        if length > MAX_CONTROL_BODY_BYTES:
            # Drained first, up to a hard cap, so the client gets the status
            # line rather than a reset connection while it is still writing.
            self.rfile.read(min(length, MAX_CONTROL_BODY_BYTES * 16))
            self.close_connection = True
            self._send_plain(
                413,
                f"a control form is at most {MAX_CONTROL_BODY_BYTES} bytes\n",
                write_body=True,
            )
            return None
        raw = self.rfile.read(length)
        parsed = parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items() if values}

    def _do_start(self, ctx: MonitorContext, *, resume: bool) -> None:
        assert ctx.controller is not None  # checked by do_POST
        try:
            ctx.controller.start(resume=resume)
        except StartRefused as refused:
            only_in_flight = {check.name for check in refused.gate.failures} == {
                "one_run_at_a_time"
            }
            self._send_bytes(
                # 409 for the one refusal that is about *timing* ("not now"),
                # 412 for a pre-condition the run itself does not meet.
                409 if only_in_flight else 412,
                "text/html; charset=utf-8",
                render_control_refusal_html(
                    "A render is already in flight"
                    if only_in_flight
                    else "This run is not ready to start",
                    refused.gate.summary(),
                ),
                write_body=True,
            )
            return
        except ControlError as exc:
            logger.exception("Could not start a render")
            self._send_bytes(
                500,
                "text/html; charset=utf-8",
                render_control_refusal_html("Could not start the render", str(exc)),
                write_body=True,
            )
            return
        self._redirect_to_index()

    def _do_stop(self, ctx: MonitorContext) -> None:
        assert ctx.controller is not None  # checked by do_POST
        try:
            ctx.controller.stop()
        except StopRefused as refused:
            self._send_bytes(
                409,
                "text/html; charset=utf-8",
                render_control_refusal_html("Nothing to stop", str(refused)),
                write_body=True,
            )
            return
        except ControlError as exc:  # pragma: no cover - stop() raises only StopRefused today
            logger.exception("Could not stop the render")
            self._send_bytes(
                500,
                "text/html; charset=utf-8",
                render_control_refusal_html("Could not stop the render", str(exc)),
                write_body=True,
            )
            return
        self._redirect_to_index()

    def _redirect_to_index(self) -> None:
        """303 back to the page, so a reload does not re-submit the form."""
        self._send_bytes(
            303,
            "text/plain; charset=utf-8",
            b"",
            write_body=True,
            extra_headers=(("Location", "/"),),
        )

    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_CONNECT = _method_not_allowed
    do_TRACE = _method_not_allowed

    def _dispatch(self, *, write_body: bool) -> None:
        ctx: MonitorContext = self.server.ctx  # type: ignore[attr-defined]
        path = urlsplit(self.path).path

        if path == "/":
            self._serve_index(ctx, write_body=write_body)
            return
        if path == "/events":
            self._serve_events(ctx, write_body=write_body)
            return
        match = _CHUNK_THUMBNAIL_RE.match(path)
        if match is not None:
            self._serve_thumbnail(ctx, int(match.group(1)), write_body=write_body)
            return
        if path == "/prepare":
            self._serve_prepare(ctx, write_body=write_body)
            return
        if path == "/review":
            self._serve_review(ctx, write_body=write_body)
            return
        self._send_plain(404, "not found", write_body=write_body)

    # -- routes ------------------------------------------------------------ #

    def _serve_index(self, ctx: MonitorContext, *, write_body: bool) -> None:
        control_section = self._control_section(ctx)
        try:
            run_state = read_run_state(ctx.run_state_path)
        except ProgressError as exc:
            body = render_index_html(
                None, not_ready_reason=str(exc), control_section=control_section
            )
        else:
            body = render_index_html(
                current_progress(run_state), control_section=control_section
            )
        # The token lives in this page, so nothing on the way may keep a copy
        # of it. Only when control is on: the read-only page is unchanged.
        headers = (
            (("Cache-Control", "no-store"),) if ctx.controller is not None else ()
        )
        self._send_bytes(
            200,
            "text/html; charset=utf-8",
            body,
            write_body=write_body,
            extra_headers=headers,
        )

    def _control_section(self, ctx: MonitorContext) -> str:
        """The control markup for this request, or ``""`` when control is off.

        The gate is evaluated only when no render is in flight -- see
        :func:`render_control_section`. A gate evaluation reads the prepare
        report, stats a directory and asks ComfyUI for ``/system_stats``, so
        it is a page load's worth of work, not a render's."""
        if ctx.controller is None:
            return ""
        status = ctx.controller.status()
        gate = (
            None
            if status.state is RenderState.RUNNING
            else ctx.controller.evaluate_start()
        )
        return render_control_section(status, gate=gate, token=ctx.control_token)

    def _serve_events(self, ctx: MonitorContext, *, write_body: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if not write_body:
            return
        try:
            for event in stream_progress_events(
                run_state_path=ctx.run_state_path,
                poll_interval_seconds=ctx.poll_interval_seconds,
                sleeper=ctx.sleeper,
            ):
                self.wfile.write(format_sse(event).encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.info("SSE client %s disconnected", self.address_string())

    def _serve_thumbnail(self, ctx: MonitorContext, chunk_id: int, *, write_body: bool) -> None:
        try:
            run_state = read_run_state(ctx.run_state_path)
        except ProgressError:
            self._send_plain(404, "run state not available yet", write_body=write_body)
            return
        result = run_state.results.get(chunk_id)
        if result is None or result.video_file is None:
            self._send_plain(404, "no thumbnail for this chunk", write_body=write_body)
            return
        video_path = Path(result.video_file)
        if not video_path.is_file():
            self._send_plain(404, "chunk video file is missing on disk", write_body=write_body)
            return
        try:
            png_bytes = get_or_render_thumbnail(
                video_path,
                cache_dir=ctx.thumbnail_cache_dir / run_state.run_id,
                runner=ctx.ffmpeg_runner,
            )
        except ThumbnailError:
            logger.exception("Could not extract a thumbnail for chunk %s", chunk_id)
            self._send_plain(502, "could not extract a thumbnail", write_body=write_body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_bytes)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        if write_body:
            self.wfile.write(png_bytes)

    def _serve_prepare(self, ctx: MonitorContext, *, write_body: bool) -> None:
        """``GET /prepare``: the last ``--prepare``'s report, rendered.

        Reading is the whole of it. This handler never runs alignment or
        slicing -- see ``prepare_report.py``'s module docstring -- so a run
        nobody has prepared renders a page saying exactly that, with the
        command to run, rather than a page that quietly spends 6 s of CPU
        and writes into a live render's chunk directory."""
        if ctx.prepare_report_path is None:
            body = render_prepare_html(
                None, not_prepared_reason="this server has no prepare-report path configured"
            )
            self._send_bytes(200, "text/html; charset=utf-8", body, write_body=write_body)
            return
        try:
            report = read_prepare_report(ctx.prepare_report_path)
        except PrepareReportError as exc:
            body = render_prepare_html(None, not_prepared_reason=str(exc))
        else:
            body = render_prepare_html(report, changed_inputs=stale_inputs(report))
        self._send_bytes(200, "text/html; charset=utf-8", body, write_body=write_body)

    def _serve_review(self, ctx: MonitorContext, *, write_body: bool) -> None:
        if ctx.review_html_path is None:
            self._send_plain(
                404, "no --review-html configured for this server", write_body=write_body
            )
            return
        try:
            body = ctx.review_html_path.read_bytes()
        except OSError:
            self._send_plain(404, "review html not found", write_body=write_body)
            return
        self._send_bytes(200, "text/html; charset=utf-8", body, write_body=write_body)

    # -- helpers ------------------------------------------------------------ #

    def _send_plain(self, status: int, text: str, *, write_body: bool) -> None:
        self._send_bytes(
            status, "text/plain; charset=utf-8", text.encode("utf-8"), write_body=write_body
        )

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        *,
        write_body: bool,
        extra_headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in extra_headers:
            self.send_header(name, value)
        # Every response here has a content type this module chose itself, so
        # a browser has no reason to guess at one -- and the one place a guess
        # could matter is a text/plain error carrying operator-facing text.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if write_body:
            self.wfile.write(body)


class _MonitorHTTPServer(ThreadingHTTPServer):
    """One per bound address (see the module docstring's bind-address
    section). ``daemon_threads`` so a stray keep-alive connection never
    blocks process shutdown."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        ctx: MonitorContext,
        *,
        address_family: socket.AddressFamily,
    ) -> None:
        self.address_family = address_family
        self.ctx = ctx
        super().__init__(server_address, handler_cls)


def make_server(address: str, port: int, ctx: MonitorContext) -> _MonitorHTTPServer:
    """Build one already-listening server for ``address``. ``port=0`` asks
    the OS for an ephemeral port -- the pattern this project's own tests use
    for every real socket (CLAUDE.md's global no-network-in-tests standard),
    and the one this module's own tests use too."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return _MonitorHTTPServer((address, port), MonitorRequestHandler, ctx, address_family=family)


def make_servers(
    addresses: Sequence[str], port: int, ctx: MonitorContext
) -> tuple[_MonitorHTTPServer, ...]:
    return tuple(make_server(address, port, ctx) for address in addresses)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvm-webui",
        description=(
            "HTTP monitor for one music-video-pipeline run (issue #36). Read-only unless "
            "--enable-control is passed, which adds POST /start and POST /stop for this "
            "config's run and nothing else -- it never configures a run; run.toml stays the "
            "source of truth. See music_video_maker/webui.py's and control.py's module "
            "docstrings."
        ),
    )
    parser.add_argument("--config", required=True, help="the run.toml this run was started from")
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="ADDRESS",
        help="an additional loopback or Tailscale address to bind (repeatable). "
        "127.0.0.1 and this host's Tailscale IPv4 address are always included.",
    )
    parser.add_argument("--port", type=int, default=8787, help="TCP port for every bound address")
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="an additional value to accept in a request's Host header (repeatable). The "
        "server's own bound address and 'localhost' are always accepted; this is for a NAME "
        "-- typically a Tailscale MagicDNS name -- that no IP literal covers. Everything "
        "else is refused with 421 before any handler runs (DNS rebinding).",
    )
    parser.add_argument(
        "--review-html",
        type=Path,
        default=None,
        metavar="PATH",
        help="a pre-generated review HTML file to serve at /review (a fixed operator-chosen "
        "path; see docs/design-web-ui.md). Without it, /review 404s.",
    )
    parser.add_argument(
        "--thumbnail-cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"where extracted chunk thumbnails are cached "
        f"(default: {DEFAULT_THUMBNAIL_CACHE_DIR})",
    )
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="also serve POST /start and POST /stop for THIS config's run (issue #36). Off "
        "by default: without it this process has no write route at all. A start runs the "
        "same CLI you would run by hand, through the same custody/disk/render-envelope "
        "gates, and is refused -- never queued -- while a run is in flight or any gate "
        "fails; a stop is a SIGINT to the run's process group, exactly as Ctrl-C is. See "
        "music_video_maker/control.py's module docstring.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="how often /events re-reads run_state.json (default: 2.0)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper())

    try:
        config = load_config(Path(args.config))
    except ConfigError:
        logger.exception("Failed to load run config from %s", args.config)
        return 1
    if config.run_state_file is None:
        logger.error(
            "config.run_state_file is None -- load_config() always resolves it; this is a bug"
        )
        return 1

    try:
        bind_addresses = resolve_bind_addresses(args.bind)
    except BindAddressError:
        logger.exception("Refusing to start: invalid --bind address")
        return 1

    try:
        allowed_hosts = frozenset(normalise_allowed_host(name) for name in args.allow_host)
    except HostAllowlistError:
        logger.exception("Refusing to start: invalid --allow-host value")
        return 1

    controller: RenderController | None = None
    control_token = ""
    if args.enable_control:
        controller = RenderController(config, Path(args.config))
        control_token = new_control_token()
        logger.warning(
            "Control routes are ON: this server can START and STOP a render of %s "
            "(POST /start, POST /stop). Every start goes through the CLI's own custody "
            "pre-flight, disk check and render-envelope gate and is refused -- never queued "
            "-- if any of them fails or a run is already in flight. The claim file is %s and "
            "the run's own output goes to %s.",
            args.config,
            controller.lock_path,
            controller.log_path,
        )

    ctx = MonitorContext(
        run_state_path=config.run_state_file,
        # Issue #36: the same path --prepare writes, taken from the same
        # config resolution, so neither side has to derive the other's rule.
        prepare_report_path=config.prepare_report_file,
        review_html_path=args.review_html,
        thumbnail_cache_dir=args.thumbnail_cache_dir or DEFAULT_THUMBNAIL_CACHE_DIR,
        poll_interval_seconds=args.poll_interval_seconds,
        allowed_hosts=allowed_hosts,
        controller=controller,
        control_token=control_token,
    )

    servers = make_servers(bind_addresses, args.port, ctx)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True, name=f"mvm-webui-{addr}")
        for server, addr in zip(servers, bind_addresses, strict=True)
    ]
    for thread in threads:
        thread.start()

    logger.info(
        "music-video-pipeline monitor listening on: %s",
        # Each server's own `server_address` after binding, not `args.port` --
        # with `--port 0` (an ephemeral port, e.g. this module's own tests
        # and manual smoke-testing) the OS picks the real number only once
        # the socket exists, and logging the requested port would print the
        # one address nobody could actually reach.
        ", ".join(
            f"http://{server.server_address[0]}:{server.server_address[1]}" for server in servers
        ),
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ALWAYS_ALLOWED_HOST_NAMES",
    "CONTROL_PATHS",
    "MAX_CONTROL_BODY_BYTES",
    "BindAddressError",
    "DEFAULT_THUMBNAIL_CACHE_DIR",
    "HostAllowlistError",
    "MonitorContext",
    "MonitorRequestHandler",
    "TAILSCALE_IPV4_RANGE",
    "TAILSCALE_IPV6_RANGE",
    "ThumbnailError",
    "control_request_refusal",
    "current_progress",
    "default_tailscale_ipv4",
    "get_or_render_thumbnail",
    "host_header_allowed",
    "is_valid_chunk_id",
    "new_control_token",
    "normalise_allowed_host",
    "main",
    "make_server",
    "make_servers",
    "render_control_refusal_html",
    "render_control_section",
    "render_index_html",
    "render_prepare_html",
    "resolve_bind_addresses",
    "stream_progress_events",
    "validate_bind_address",
]
