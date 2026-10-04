# Web UI: collect a run, gate on the pre-render checks, show live progress (issue #36)

There is no UI. A run today is hand-edited TOML, a CLI, and `grep` over
`render.log`. That is workable for the person who wrote the pipeline and a wall
for anyone else — and it hides the two things that decide whether a run is
worth starting.

This document is the design. Five pieces of it are built:
`music_video_maker/progress.py` (the reader-plus-differ underneath the server),
`music_video_maker/review.py` (the read-only pre-render review page, `--review`;
see "The next slice — built", below), `music_video_maker/prepare_report.py`
(what `--prepare` found, written down so something other than a terminal can
read it; see "The pre-render gate, in the monitor — built"),
`music_video_maker/webui.py` (the server itself — the HTTP monitor; see
"What's built: the read-only monitor" below) and, since 2026-10-04,
`music_video_maker/control.py` (**start and stop one run, with guards** — see
"The control half — built" below, and read it before changing any of it).

**The *configure* half described in "What the page collects" is still not
built.** `run.toml` remains the source of truth and a human edits it; the page
can start the run that file already describes and interrupt it, and nothing
else.

## What is built, and why that piece first

Issue #36 identifies the progress half as "a plumbing problem, not a
measurement problem": the orchestrator already knows everything and emits it
only to a log. `progress.py` is that plumbing, with the server left out:

```
run_state.json  ──read_run_state──▶  RunState
                                        │
                     RunProgress.from_run_state(run_state, expected_chunk_ids)
                                        │
             events_between(previous_snapshot, current)  ──▶  (ProgressEvent, …)
                                        │
                                 format_sse(event)  ──▶  "event: …\ndata: {…}\n\n"
```

Five decisions in it are load-bearing:

* **The source of truth is `run_state.json`, polled.** `ResilientRunner` has
  no observer seam and adding one is a change to the most safety-critical file
  in the project. `run_state.json` is already written atomically after every
  chunk and is already what `--resume` trusts. A poller is strictly weaker than
  a callback and cannot corrupt a run.
* **A client that connects mid-run gets a snapshot, then diffs.** The first
  call emits one `run_snapshot`; later calls emit only changes. Without this a
  resumed run replays 40 chunks as though they had just finished.
* **Projected finish uses RENDERED chunks only, never CACHED.** A cached chunk
  cost this run nothing. Folding those into the mean tells you a 100-minute run
  has four minutes left — a lie that only appears on `--resume`, which is
  exactly the run where a human most needs the number.

  The mechanism is worth stating precisely, because it is not the obvious one:
  a cached chunk's `render_seconds` is **not** zero.
  `resilience._reusable_cached_result` does `dataclasses.replace(existing,
  status=ChunkStatus.CACHED)`, so the field carries forward whatever a
  *previous* run measured — possibly at a different resolution, on a different
  config. It is stale foreign data, not a zero being averaged in. Anyone
  "simplifying" this later will read the field, see a plausible number, and
  fold it back in.
* **Within-chunk progress is out of scope, and that is where a real server has
  work to do.** ComfyUI's WebSocket `progress` events (step N of 20) are
  consumed inside `execution.ComfyUIExecutionClient` and never reach
  `run_state.json`. A UI that wants a moving bar *inside* a chunk needs a seam
  there; everything else it needs already exists.
* **Free VRAM and the resume reason are carried, not computed.** Issue #36's
  "Gaps" section (below) used to name three things `run_state.json` did not
  yet carry; two are now real fields on `ChunkResult` /
  `RunState` — `free_vram_gb_before`, `rerender_reason` /
  `rerender_reason_fields`, and the run-level `VramStopEvent` for the
  stop-below-floor path — and `progress.py` does nothing but pass them
  through onto `ChunkProgress` / `RunProgress`. Within-chunk step progress
  is the one that remains open.

`progress.py` imports no `http`, no `socket`, no `asyncio`. A module that
cannot listen cannot get the security constraint below wrong.

## The constraint that shapes everything else

**This must not become an unauthenticated RCE surface.** ComfyUI has no
authentication, which is why on doris it is bound to loopback plus its
Tailscale address, with ufw allowing 8188 on `tailscale0` alone. Anything that
can start a render can write files and load custom nodes *by proxy through
ComfyUI*, so a UI inherits the same threat model exactly.

Non-negotiables for whoever writes the server:

1. Bind to `127.0.0.1` **and** the host's Tailscale address only. Never
   `0.0.0.0`. Not "in production" — never, including the first prototype,
   because the first prototype is what gets left running.
2. The bind address is a **test**, not a comment. Assert the resolved bind list
   contains no wildcard and no non-loopback, non-tailnet address.
3. No path parameter reaches the filesystem unvalidated. The page picks from a
   run directory the operator configured; it does not accept an arbitrary
   `master_audio=/etc/…`.
4. Serving chunk thumbnails means serving files. Serve them from the run's own
   `chunks_dir` by chunk id, never by path.
5. **Binding correctly is not enough.** A DNS-rebinding page can have a
   victim's own browser make same-origin requests to a loopback-bound server
   and read the responses — loopback binding is that attack's precondition,
   not a defence against it. Check the `Host` header before any handler runs.

All five are now enforced code, not just this list — see "What's built: the
read-only monitor" immediately below. One implementation note against #4: the
server extracts a still frame with ffmpeg and caches *that*, rather than
serving anything out of `chunks_dir` directly — see "Thumbnails" there for
why (the cache must be writable, and `chunks_dir` is a live render's own
directory, not this server's to write into).

## What's built: the read-only monitor (issue #36)

`music_video_maker/webui.py` — plus a console script and `python -m
music_video_maker.webui`, following `pyproject.toml`'s existing
`[project.scripts]` pattern — is the server the rest of this document used to
say did not exist. It is read-only unless started with `--enable-control`:
see "The control half — built" below for what that flag adds, what it
refuses, and why the boundary sits exactly there.

Run it against a config already driving a render, or a finished one:

```bash
mvm-webui --config run.toml --bind 100.x.y.z
```

`100.x.y.z` above is a **placeholder** — this project's own rule is that a
real Tailscale address never lands in a committed file (see CLAUDE.md,
"Everything committed here is intended to become public"). You do not
usually need `--bind` at all: `127.0.0.1` and the host's own Tailscale IPv4
address (found via `tailscale ip -4`) are always included automatically;
`--bind` only adds more validated addresses, e.g. for a second interface.

If you reach it by a *name* rather than an address — a Tailscale MagicDNS
name, typically — add `--allow-host <name>`, or the request is refused with
`421` by the Host check below. The server's own bound address and `localhost`
need no flag.

To be able to start and stop this run from the page, add `--enable-control`:

```bash
mvm-webui --config run.toml --enable-control
```

Without it there is no write route at all. With it there are exactly two, and
the startup log says so in a WARNING naming the claim file and the child's log.

### Bind addresses

`resolve_bind_addresses` always includes `127.0.0.1`, adds this host's
Tailscale IPv4 address if `tailscale ip -4` finds one (an injected subprocess
seam; absence or failure is logged and falls back to loopback-only, never
fatal — matching point 1 above), and validates every operator-supplied
`--bind` address through the same `validate_bind_address`: loopback, or
inside Tailscale's own ranges (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`), or
refused outright — `0.0.0.0`, `::`, a LAN address like `192.168.x.x`, and any
hostname (never resolved; an address must already be an IP literal) all
raise `BindAddressError` and refuse to start the process. `tests/test_webui.py`
checks this two ways, matching point 2 above: the pure function's output for
a wide range of addresses, and that a real `ThreadingHTTPServer` built from
it only ever reports `server_address` on the address it was given —
including one build with `AF_INET6` for a Tailscale IPv6 address.

### The Host header (DNS rebinding)

Non-negotiable 5 above, concretely. `host_header_allowed` runs at the top of
every method handler — including the ones that only ever return `405`, so a
refused request cannot learn which methods are allowed or which paths exist.
A request is answered only when its `Host` is:

* **this server's own bound address**, compared as an `ipaddress` value
  rather than as text (so `::1` and `0:0:0:0:0:0:0:1` are one host), with an
  optional port that must equal the port it is listening on;
* `localhost` (`ALWAYS_ALLOWED_HOST_NAMES`);
* a name the operator passed to `--allow-host` (repeatable, normalised, and
  refused at startup if it is not a bare host name) — the Tailscale MagicDNS
  case no IP literal covers.

Everything else gets `421 Misdirected Request` with no run data in the body;
the refused value is logged rather than echoed back. Two decisions in it:

* **An absent `Host` is allowed.** Rebinding's whole leverage *is* the name
  in that header, and a browser will not let script suppress or forge it.
  Refusing an absent header would refuse HTTP/1.0 clients (`curl -0`) and
  block no attack.
* **Two `Host` headers are refused outright** — one request with two
  authorities is a request-smuggling shape.

Nothing here resolves a name, for the same reason `validate_bind_address`
never does: resolution is the mechanism being defeated, so asking DNS whether
`evil.example` points at us is asking the attacker. A test pins that by making
`socket.gethostbyname`/`getaddrinfo` raise. Every response also carries
`X-Content-Type-Options: nosniff`.

### run_state.json and "the run's known chunks"

The server is started with `--config run.toml`, the same file the render
itself was started with. `config.load_config` already resolves
`RunConfig.run_state_file` (defaulting to `chunks_dir/run_state.json`,
overridable in the TOML directly) — that resolved path is what the monitor
polls, through `progress.read_run_state`; nothing in `webui.py` re-derives
or guesses the rule.

**Deliberately not built: recomputing the chunk timeline.** The obvious way
to answer "how many chunks will this run ever have" is to re-run Stage 1-2
(`alignment.align` + `slicing.slice_audio`) the way `--prepare` does. That
function is **not** read-only — its own docstring says it exports
`chunk_NNN.wav` into `chunks_dir`, the *same* directory a live render is
already reading and writing — so a monitor that called it would race an
in-flight run for that directory and spend disk on a machine this project's
own CLAUDE.md already flags as tight (doris and the driving Mac both). So
`webui.py` never imports `alignment` or `slicing`, and "the run's known
chunks" (`webui._known_chunk_ids`) means exactly the chunk ids that already
have an entry in `run_state.json`, plus the one id `RunState.vram_stop` names
if the run stopped below the VRAM floor before that chunk got a result of
its own (it renders as a "pending" row instead of not existing). This is
strictly a subset of the run's true plan until the last chunk lands — the
rendered page says "chunks recorded in run_state.json so far", explicitly
not "of N", and a run with zero results yet is never presented as finished
even though `RunProgress.finished` is vacuously `True` over an empty set
(`tests/test_webui.py::TestKnownChunks::test_a_run_with_no_results_yet_does_not_report_finished`
is the regression guard). The pre-render review page below is the right
place for the true chunk plan: it runs Stage 1-2 once, on demand, from the
CLI — not from a long-lived poller that might be watching a live render.

Point 3 above, concretely: chunk ids reaching the server in a request
(`/chunks/<id>/thumbnail.png`) are matched by a digits-only route regex,
parsed as `int`, and looked up as a dict key against `run_state.results` —
never concatenated into a path. `tests/test_webui.py` fires several
path-traversal shapes (`../../etc/passwd`, URL-encoded `..`, a decimal, a
double slash) at that route and asserts a flat 404 for every one, the same
404 an unrecognized route gets.

### Routes

* **`GET /`** — server-rendered HTML: recorded/rendered/cached/dead-lettered
  counts, mean render time over rendered chunks only (matching
  `progress.py`'s own exclusion of cached chunks' stale `render_seconds`), a
  per-chunk table (status, attempts, free VRAM before, re-render reason,
  errors, a thumbnail link), a prominent VRAM-stop notice when
  `RunState.vram_stop` is set, and dead-lettered chunks with their full error
  history. Renders a complete, correct snapshot with **no JavaScript
  required**; a small inline script upgrades it to reload on new SSE events.
  All run-derived text — errors, re-render-reason fields, the run id itself —
  is HTML-escaped (`html.escape`, via one `_e` helper every piece of
  run-derived text passes through); `tests/test_webui.py` proves a
  `<script>` string arriving in a dead-letter error, and a run id containing
  `<`/`>`, both survive unexecuted in the response body.
* **`GET /events`** — SSE, a thin wrapper (`stream_progress_events`) around
  `progress.events_between`/`format_sse`: one `run_snapshot` on connect,
  diffs after, polling `run_state.json` at `--poll-interval-seconds`
  (default 2.0). A missing or torn state file is silently "not ready yet",
  never a 500 — the same contract `progress.read_run_state` already
  documents — and a client disconnect is caught (`BrokenPipeError` /
  `ConnectionResetError` / `OSError`) rather than logged as a server error.
  `stream_progress_events` is itself a plain generator with an injectable
  sleeper and a `max_polls` bound, so `tests/test_webui.py` drives several
  polls of it with no real socket and no real `time.sleep` — a fake sleeper
  rewrites `run_state.json` to the next fixture state when called, which is
  what stands in for the passage of time.
* **`GET /chunks/<id>/thumbnail.png`** — a frame extracted with ffmpeg
  (`get_or_render_thumbnail`, an injected subprocess seam — never called
  directly in a test) from the chunk's `video_file` in `run_state.json`, and
  cached at `$TMPDIR/mvm-webui-thumbnails/<run_id>/<video-stem>-<mtime_ns>-<size>.png`
  by default (`DEFAULT_THUMBNAIL_CACHE_DIR`, overridable with
  `--thumbnail-cache-dir`) — **outside the repo and outside every run's
  `chunks_dir`** on purpose (see the note under point 4 above), namespaced by
  `run_id` so two runs whose chunk ids collide never serve each other's
  frames, and keyed by the video file's own mtime and size so a chunk
  re-rendered under `--resume` invalidates its cached frame automatically
  without anything having to notice the rename didn't happen. 404 if the
  chunk id is unknown to `run_state.json`, has no `video_file`, or the file
  is missing on disk; 502 if ffmpeg itself fails.
* **`GET /prepare`** — the pre-render checks, rendered from the JSON report
  `--prepare` writes (`RunConfig.prepare_report_file`; see "The pre-render
  gate, in the monitor — built" below). Reads a file and nothing else: no
  alignment, no slicing, no button that would run either. A run nobody has
  prepared renders a page saying exactly that, with the command to run, at
  `200` — an unprepared run is an ordinary state, not a server error, the
  same contract `/` has for a missing `run_state.json`.
* **`GET /review`** — serves the file at `--review-html PATH` verbatim if the
  operator passed one at startup (a fixed, operator-chosen path — never a
  request parameter); 404 otherwise. The page it serves is the one
  `cli --review` writes (see "The next slice — built" below); this route only
  ever serves whatever file is put there, never generates one, and never
  imports `review.py` — which is what keeps `webui.py` free of any import of
  `alignment`/`slicing`.
* **`POST /start`** and **`POST /stop`** — only with `--enable-control`; see
  "The control half — built" immediately below.
* Everything else 404s. Only `GET`, `HEAD` and (with `--enable-control`) a
  `POST` to those two paths are accepted; every other method — `PUT`,
  `DELETE`, `PATCH`, `OPTIONS`, `TRACE`, and `POST` to any other path — gets
  a flat 405 with an `Allow: GET, HEAD` header and no CORS headers at all.

### The control half — built (issue #36, 2026-10-04)

This section used to be called "What is deliberately not here" and said that
no start route existed, for two reasons: GPU custody, and the fact that
anything able to start a render can write files and load custom nodes *by
proxy through ComfyUI*. Both reasons are still true. What changed on
2026-10-04 is that the owner decided what a button may do: **start a run and
stop it, with guards, and nothing else.** So the scope sentence is now
narrower rather than absent — the page cannot configure a run, cannot name a
path, a chunk, a resolution or a config value, and cannot pass any flag but
`--resume`.

`music_video_maker/control.py` is that half, and it imports no `http` and no
`socket` for the same reason `progress.py` does not: the module holding the
gates should not also be holding a connection.

**A start is the CLI, spawned.** `python -m music_video_maker.cli --config
<the same file>`, in its own session. Not `run_pipeline` in a thread — that
would put the custody pre-flight, `prevent_host_sleep`, the between-chunk
VRAM re-check and the unconditional `POST /free` behind a shutdown path
nobody has tested, and would make "stop the render" mean "raise an exception
in a worker thread", which Python does not offer honestly. Spawning buys
three things:

* the gates are the CLI's own, not a second copy that drifts from them;
* a **stop is a SIGINT to the child's process group, which is literally what
  Ctrl-C is** (a terminal signals the whole foreground group, which is also
  how any ffmpeg the run has spawned goes down with it). The pipeline already
  persists run state before every chunk and releases custody in a `finally`,
  so there is no second shutdown path — and `start_new_session=True` also
  means Ctrl-C on `mvm-webui` itself does *not* take a 100-minute render with
  it;
* a render that wedges the host takes the server's child, not the server, and
  `run_state.json` is still the source of truth about what rendered.

**Opt-in.** Without `--enable-control` the process is the read-only monitor
it has always been — no POST route, no form in the page, and none of the
read-only tests needed to change. An existing deployment does not acquire a
write surface by being upgraded.

**The gate: five checks, all refusals, never a queue.** Four are the run's
own, called through the run's own functions so there is exactly one copy of
each rule — which is why this work refactored three of them into public
functions rather than re-deriving them:

| check | what runs | refuses when |
|---|---|---|
| `one_run_at_a_time` | a claim file beside the run state | a live pid holds the claim |
| `prepared` | `prepare_report.read_prepare_report` + `stale_inputs` | see below |
| `disk` | `resilience.preflight_disk_check` (issue #10) | short of `min_free_disk_gb` |
| `vram` | `custody.preflight_free_vram` (issue #19) | a usable reading below `min_free_vram_gb` |
| `envelope` | `envelope.check_render_envelope` (issues #24/#98) | the longest chunk is outside the proven table |

Every check's reading is shown on the page whether it passed or not, and a
refusal names the number and the remedy. Four notes on the shape of it:

* **`one_run_at_a_time` is a file, not memory.** A restarted server must not
  start a second render over the first one, so the claim (pid, process-group
  id, start time) lives in `webui_render.lock` beside the run state. It is
  created with `O_CREAT|O_EXCL` *after* the gate passes, which is the second
  of two different races: the in-process mutex stops two requests to this
  server, the exclusive create stops two servers. A claim whose pid is gone
  is cleared automatically; a claim file that exists and cannot be parsed is
  a **refusal**, because starting a second render over a live one is the
  unrecoverable direction. One thing worth writing down: the custody floor is
  itself a cross-process exclusion, measured — on this stack H3 resident
  leaves under 1 GB free (CLAUDE.md's between-chunk bullet), so a render
  started from a terminal makes the `vram` check refuse at ~0.8 GB against a
  16 GB floor. The lock is the fast, legible guard; the floor is the one that
  does not depend on a file.
* **`vram` degrades rather than refusing when ComfyUI cannot be read**,
  because that is what the CLI's own pre-flight decides (logged, non-fatal),
  and a gate stricter than the thing it gates is a policy nobody reviewed.
  The page says the card could not be read. This is the one asymmetry in the
  table and it is deliberate; a run that cannot reach ComfyUI fails in
  seconds without touching the card.
* **`envelope` is evaluated from published data, not a recomputed timeline.**
  The monitor must never run Stage 1-2 (it writes chunk audio into the live
  run's own directory), so the gate needs the longest chunk's frame count
  from somewhere else — and the lesson in the repos' shared CLAUDE.md is to
  check whether the other side already publishes what you were about to
  mirror. It nearly did: `--prepare` had `total_frames` but not the maximum.
  `PrepareReport.max_chunk_frames` / `max_chunk_frames_chunk_id` (optional on
  read, no schema bump — an old report reads as `None`) is that one number,
  and it is *sufficient* rather than convenient: `EnvelopePoint.covers` is
  per axis, so at one fixed resolution the longest chunk decides the whole
  run. Cover it and every shorter chunk is covered; miss it and the run is
  refused. Which *other* chunks also miss is reporting, and only the
  per-chunk check inside the run can enumerate those — it still runs, before
  any GPU work. The config's own ceiling was considered for this and
  rejected: `hardware.max_chunk_seconds` defaults to H3's trained maximum
  (362 frames), so a ceiling check would refuse *every* default config,
  which is a gate that cries wolf.
* **`prepared` is the one check with no CLI equivalent**, and it is
  deliberately stricter: a start is refused unless a prepare report exists,
  this build can read it, it records a frame count, and every input it was
  computed from is unchanged on disk. Reasons, in order: it is the half of
  issue #36 that matters more ("a 100-minute mistake turned into a 10-second
  one" is the whole argument for the page, and a button that skips the review
  is that mistake with a shorter path to it); it is what makes the `envelope`
  check above possible without recomputing the timeline; and the remedy is
  ~50 s of CPU and no GPU, with the page printing the command. A shot-plan
  refusal fails this check too, and a CRITICAL alignment finding fails it
  **only under `strict_alignment`** — mirroring the CLI exactly, because a
  page that refuses what the CLI accepts teaches operators to go round it.

**What the control half will not do**, each a refusal with words rather than
an omission:

* **It will not stop a claim it did not create.** A pid read out of a file
  this process did not write could have been reused by something unrelated,
  and SIGINT to a stranger's process group is not a recoverable mistake. The
  page says `kill -INT <pid>` instead. The cost is real and accepted: a
  server restarted during a 100-minute render shows a stop that refuses.
* **It will not escalate.** A second stop sends another SIGINT, the way
  pressing Ctrl-C twice does. Nothing sends SIGKILL: killing a render inside
  its `POST /free` is how the card gets left holding 17 GB, which is the
  incident recorded under "Custody and resume" below.
* **It will not touch the card.** Stopping the GPU's other tenants stays
  manual (`custody.py`'s module docstring). A button that paused somebody's
  inference server is the exact automation this project decided never to
  build.
* **It will not write `run.toml`.**

**Two files, beside the run state** (`run_state.json`'s own directory, where
`--prepare` already writes its report): `webui_render.lock`, the claim, and
`webui_render.log`, the child's stdout and stderr. That log is how a refusal
from *inside* the run reaches the page — a per-chunk envelope miss, a config
error, a strict-alignment stop — and the page shows its tail with the exit
code. This is the deliberate exception to "the monitor never writes into the
run's directory", and it is exactly why it is a separate module: the
read-only half still writes nothing at all.

**Defending a write route** (and why the bind list is not that defence) is in
`webui.py`'s module docstring under "Reaching the control routes". In short: a
cross-site `POST` of `application/x-www-form-urlencoded` needs no CORS
preflight, so three checks must all pass — a per-process CSRF token compared
with `secrets.compare_digest` (load-bearing: it is only obtainable by
*reading* `GET /`, which a cross-origin script cannot do and a rebound name
cannot reach), `Sec-Fetch-Site` in `{same-origin, none}` when present
(`same-site` refused — a sibling host is not this origin), and `Origin`
matching this server's own origin when present (`null` refused rather than
treated as absent). Plus: form content type only, a 4 KiB body cap, a `303`
redirect so a reload does not re-submit, `Cache-Control: no-store` on the
page carrying the token, and nothing ever echoed back into a response.

## Open questions, and the answers this design assumes

**Where does it run?** Assume **the Mac that drives the pipeline**, not doris.
Reasons: the chunk mp4s land there, the run config and shot plan live there,
and doris's disk runs high-90s % full. Consequence to accept: the UI cannot
show doris-side facts (GPU tenants, `nvidia-smi`) except through what the
pipeline already probes and records. That is a real limitation of this answer
and the reason to revisit it is thumbnails, not progress.

**SSE or an API + SPA?** **Server-rendered pages plus SSE for the live half.**
Progress is one-directional, the run is long-lived, and SSE reconnects on its
own. `progress.format_sse` already emits the wire format. An SPA buys
interactivity this application does not need and costs a build step in a repo
whose whole toolchain is currently "python and ffmpeg".

**Multi-run history?** "The current run plus the last one" is enough. Run
directories are already the unit of history and they are already on disk; a
database here would be a second source of truth about what rendered.

## The pre-render half, which matters more

Two expensive lessons this project paid for would have been ten-second catches
on a review page:

* **The alignment timeline.** `alignment_quality` (#35) computes zero-length
  segments, implausible word rates and lyrics hallucinated into instrumental
  passages, in the ~6 s alignment takes. On "Deathless", `base` believed the
  singing stopped at 3:11 of an 8:32 song and would have rendered 56 of 71
  chunks as "performs silently"; the report flagged it with 20 critical
  findings and nobody was looking at a report.
* **The shot plan against the chunks.** Seeing each chunk's span, its lyric and
  its shot line side by side is the review that catches a `setting` fighting
  its own shot lines (#32), a `location` tag that outlived the event that
  changed it (#78), and a plan authored against a different timeline
  (`ShotPlanDriftError`, which on Deathless named a 1.4 s discrepancy a human
  review of the plan could not have seen).

Suggested flow, unchanged from the issue: pick assets → align (cheap, CPU) →
**review timeline + quality findings + shot plan** → confirm → render with live
progress.

### The next slice — built (issue #36)

`music_video_maker/review.py`: a read-only `--review` CLI flag that writes one
self-contained HTML page plus the same data as JSON from a `--prepare`-style
input. No server, no forms, no start button, no socket import (same discipline
`progress.py` follows) — a file you open. `build_review` (inputs → a
`ReviewData` tree) and `render_review_html`/`render_review_json` (that tree →
a string) are two pure functions with nothing between them; `cli.main`'s own
wiring is the only part of this that touches a filesystem.

Per chunk: `chunk_id`, span, duration, frame count, voiced/instrumental, the
lyric text, the shot line, `camera`, `location`, `present`, `subject`,
`conditions`, every `alignment_quality` finding whose span touches the
chunk's, and every shot-plan warning that named it. Run-level: the alignment
quality summary, every shot-plan warning (including the ones that named no
chunk), any structural plan failure (`plan_errors` — a plan that would not
load, or that a raising lint refused, exactly as a real render would refuse,
surfaced instead of a stack trace), and `would_refuse_render` (below).

**The timeline it describes is the one a real render produces, not a second
approximation of it.** `run_pipeline` always slices with
`shot_length_requests(plan)` read from `config.shot_plan` directly; there is
no separate "from_plan" concept at render time. `build_review` defaults its
own `from_plan` to `config.shot_plan` for exactly this reason — an explicit
`--from-plan` (checking a *candidate* plan before it is wired into the
config, the same case `--prepare --from-plan` exists for) still overrides
it. A first cut of this page got this wrong: it ignored `config.shot_plan`'s
own lengths, so any plan setting a `length_seconds` got reviewed against the
*natural* (unmerged) timeline, and every chunk after the first long take
reported as `ShotPlanDriftError` in `plan_errors` instead of the merged
chunk the render actually produces.

**`strict_alignment` must never crash the review.** `prepare_timeline` ->
`align()` raises `AlignmentQualityError` once `strict_alignment` is set and a
finding reaches CRITICAL — exactly the run a reviewer most needs to see, not
a traceback for. The review's own call into `prepare_timeline` always aligns
non-strict (`--prepare` itself is untouched); when the *original* config was
strict and the report does have a CRITICAL-or-above finding,
`would_refuse_render` names the count and is rendered at the top of the HTML
page and included in the JSON, so "this run would in fact refuse" is not
lost along with the crash it no longer causes.

**Reused, not reimplemented**, on both axes this section originally flagged as
in flux:

* Stage 1-2 itself: `cli.prepare_timeline`, factored out of
  `cli.prepare_shot_plan` (the two now share one Stage 1-2 run rather than
  each describing their own), returns the chunk timeline *and* the
  `AlignmentQualityReport` Stage 1 already computes and, before this issue,
  only logged — `alignment.align()` grew one injectable seam
  (`on_quality_report`) for this, the same shape as every other Stage 1/2 I/O
  seam already takes.
* The shot-plan lints: `cli.run_shot_plan_lints`, factored out of
  `run_pipeline`'s own lint block, is the literal function a real render
  calls — same lints, same order. The review runs it with a logging handler
  attached, the identical technique `authoring/plan.check_plan` uses on the
  authoring side ("checked by the render's own loaders, never a copy of
  them"). `review.py` cannot import that collector directly —
  `tests/test_authoring_boundary.py` forbids anything outside
  `authoring/` from importing that package — so it carries its own
  nine-line `logging.Handler`, not a second copy of any lint.

Golden-file tests live in `tests/test_review.py` against
`tests/fixtures/review/review_golden.{json,html}`, built from a hand-written
`ReviewData` (not a real alignment/slicing run, so the fixture cannot drift
when those change) covering one voiced chunk, one instrumental, one with an
alignment finding, and one with a lint warning; `MVM_UPDATE_GOLDEN=1` on that
test file regenerates them after a deliberate format change. Separate wiring
tests exercise `build_review` through the same offline `Rig`
`tests/test_cli.py` uses for `--prepare`, with a real `FakeAlignModel`
injected.

What it does not do: nothing here talks to ComfyUI, GPU custody, or Stage 4/5
— a review is Stage 1-2 and nothing past it. It does not know within-chunk
step progress, the per-chunk resume reason, or free VRAM (`progress.py`'s own
"Gaps" section, below); those are run-time facts a finished run or an
in-flight one has, and a review is neither. It is one snapshot of one
`--prepare`-style input, not live — opening it twice after editing the shot
plan means running `--review` again.

### The pre-render gate, in the monitor — built (issue #36)

`--review` above is a file you generate and open. The monitor needed the same
gate as a *page of the run*, and the obvious implementation — have the page
run Stages 1-2 when somebody opens it — is the one thing it must not do:
`slice_audio` writes `chunk_NNN.wav` into the live run's own `chunks_dir`, and
alignment is ~6 s of CPU per open. Measuring is a step the operator invokes.

So the work is split at a file. `--prepare` already computed all of this and
logged it; now it also writes `prepare_report.json`
(`music_video_maker/prepare_report.py`, at `RunConfig.prepare_report_file` —
defaulting to `chunks_dir/prepare_report.json`, resolved by `load_config`
exactly the way `run_state_file` is, so the writer and the reader share one
rule rather than two that agree today). `GET /prepare` renders it.

What the report carries, all of it already produced by the one
`prepare_timeline` run that wrote the skeleton:

* the #35 alignment summary line verbatim (`format_summary` — one place
  decides how a report reads), the finding counts by severity, and every
  finding at WARNING or above;
* the timeline: chunk count, voiced/instrumental split, span, total frames,
  and the #22 drift against the master track in seconds and frames, with the
  run's own tolerance;
* every WARNING-or-above record `slicing` and `cli` emitted during Stage 1-2
  — which is #70's untenable segments and mid-phrase cuts, #79's surviving
  leading vocal offsets, and the drift line — captured *while they were
  emitted* (`collect_stage_notices`), never re-derived. Captured even when
  `--log-level ERROR` would have gated them before any handler saw them: the
  console's verbosity is the operator's choice, the report's contents are
  not. Each carries the issue number its own text names, purely so the page
  can group thirty mid-phrase-cut warnings under one heading;
* the shot plan resolved against *this* timeline through `load_shot_plan` /
  `resolve_shot` themselves — the `ShotPlanDriftError` a human review of the
  plan cannot perform, because the file reads perfectly well either way. The
  report records whether the timeline was re-anchored against that plan's own
  `length_seconds`, and the page says so, because a plan setting a length
  describes a different timeline *by construction* when it was not (CLAUDE.md:
  "a plan that sets `length_seconds` has two timelines, and only one is
  real") — a drift list produced against the other one would read as a defect
  in the plan;
* **an `InputStamp` per input** — resolved path, size, mtime. `stale_inputs`
  re-stats them when the page is rendered, so a report whose lyrics file has
  been edited since says so in a banner instead of quietly describing a
  timeline no render would now produce. This is #93's lesson as a field: a
  measurement artefact that cannot name what it was computed from can assert
  a subject it never opened.

Three things it deliberately is not. It is **not a behaviour change** —
`--prepare` slices the same timeline, writes the same skeleton, refuses the
same things; the order is Stage 1-2, skeleton, *then* report, so a prepare
that refuses leaves no report at all (one that did would be an artefact
describing something other than what its reader assumes), and a report that
cannot be written logs loudly rather than failing a prepare whose real output
already landed. It is **not read back into a render** — the same boundary an
authored `shot_plan.toml` sits on the other side of. And it is **not a second
implementation of any check**: every number in it came from the render's own
Stage 1-2 code.

`PREPARE_REPORT_SCHEMA_VERSION` is refused rather than guessed at on read,
unlike `run_state.json`'s — nothing is lost by refusing a report, since
re-running `--prepare` costs ~50 s and no GPU.

## What the page collects

Everything in `config.RunConfig`: master audio, lyrics, cast
(name/role/image/appearance/demeanour), `global_style`, `cinematography` or a
`cinematography_profile` (#55), `narrative_concept`, `setting`, render
resolution, chunk bounds, `instrumental_coverage`, `i2v_*`, and the resilience
knobs.

**The TOML file stays the source of truth and stays committable.** The page
reads and writes it; it does not replace it. A run must remain reproducible
from a file in a directory, by someone with no browser, because that is what
makes every finding in this project checkable.

## Custody and resume, which must be visible rather than hidden

The first two bullets below were written as requirements for a future
*start/configure* half and are now **satisfied** by the control half above;
they are kept in their original words, with what satisfies them noted, because
the requirement is what the next change has to keep true. The last two are
about *displaying* resume semantics, which the monitor already does.

* **GPU custody is exclusive.** A UI that can start a render must refuse to
  start a second one while one is in flight, and must not bypass the #19
  pre-flight. The stopping of the card's other tenants is deliberately manual
  and stays manual — see `custody.py`'s module docstring. A UI button that
  stops the Ollama container would be the exact automation this project
  decided never to build. **Satisfied:** the `one_run_at_a_time` claim file
  plus the `vram` check calling `custody.preflight_free_vram` itself, and no
  button anywhere touches another tenant.
* **Release is unconditional.** Any path that renders releases ComfyUI's VRAM
  in a `finally`. A direct-library test script that skipped it once left 17 GB
  held and starved every other tenant on the card. **Satisfied by not being
  reimplemented:** the control half spawns the CLI, so the `finally` that
  releases the card is the same one every hand-run render uses — which is
  also why stopping is a SIGINT (the `KeyboardInterrupt` path runs it) and
  why nothing here ever sends SIGKILL.
* **Resume semantics must be surfaced, not hidden.** When a chunk is
  re-rendered the page must say *why*: a `schema_version` rejection, or a
  fingerprint mismatch **naming the field that changed**. `ChunkFingerprint`
  already reports the field names through `timeline_differences` /
  `content_differences`; `resilience._reusable_cached_result` now records
  the category and field names on the resulting `ChunkResult` (issue #36) as
  `rerender_reason` / `rerender_reason_fields` — e.g. `"content_changed"` with
  `("prompt_hash",)` — instead of only logging them, and `progress.py` carries
  both through unchanged on every `ChunkProgress`. A freshly rendered chunk
  with no prior entry to reject reports `rerender_reason=None`, and so does
  every chunk after a whole-file `schema_version` rejection restarts the run
  from an empty `RunState` — deliberately: there is nothing to compare
  against in either case, and inventing a per-chunk reason for the second
  would be less honest than `None`, not more. See "Closed" below. **Built:**
  the monitor's per-chunk table shows this column verbatim.

  One correction from the 2026-09-20 real-render acceptance: a *cached* chunk
  used to carry a previous `--resume`'s reason forward
  (`replace(existing, status=CACHED)` copied every other field), so the table
  showed "cached … content_changed (prompt_hash)" beside a chunk this run had
  reused without comparing or rejecting anything. `resilience._as_cached` is
  now the one place a reused result is built and it clears both fields. Note
  the asymmetry, which is the point: `render_seconds` is *kept*, because it is
  still true of the mp4 that exists and `progress.py` already excludes cached
  chunks from its projection by status; `rerender_reason` answers "why did
  *this* run re-render this chunk", and this run did not.
* **Dead-lettered chunks** are called out with their error history.
  **Built:** the monitor shows the errors
  (`ProgressEvent("chunk_dead_lettered")` carries the full `errors` tuple),
  and since 2026-10-04 the resume this bullet originally wanted exists — the
  start form's `--resume` checkbox, through the same five gates as any other
  start. It is a checkbox rather than a per-chunk "retry this one" button on
  purpose: `--resume` reuses every chunk whose fingerprint still matches and
  re-renders the rest, which is what retrying a dead letter actually means,
  and a per-chunk control would be `--only-chunks` — a flag that widens what
  a run does and therefore one a browser does not get.

## Closed since this document was first written (issue #36)

The three gaps below, plus the private-reach fix, all landed in one pass —
each was "record what is already known", not new measurement, and all three
went onto `contracts.ChunkResult` / `RunState` without a `schema_version`
bump (every new field is optional on read, so a run_state.json written by
code before this change still loads and resumes).

1. **Free VRAM between chunks** (#23) — the most valuable of the three, now
   `ChunkResult.free_vram_gb_before`. Recorded **once per chunk, before its
   first attempt**: `ResilientRunner._check_vram_before_chunk` sits outside
   `_render_chunk`'s retry loop, so a chunk that fails twice before
   succeeding still carries exactly one reading rather than an ambiguous
   choice among several — re-probing per attempt would have conflated a VRAM
   condition with the recovery sequence's (`interrupt` → `free`) own effect
   on free VRAM. `None` when no `vram_probe` was injected or the probe
   returned an unreadable `None`; never set on a `CACHED` chunk, which never
   touches the GPU. The chunk that actually trips the floor and stops the
   run is the one case with no `ChunkResult` to record it on at all — see
   `contracts.VramStopEvent` below.
2. **The resume reason per chunk** — `ChunkResult.rerender_reason` (a short
   category: `"explicit_selection"`, `"video_missing"`, `"no_fingerprint"`,
   `"chain_blocked"`, `"conditioning_changed"`, `"timeline_changed"`,
   `"content_changed"`, `"stack_changed"`) plus `rerender_reason_fields` (the
   `ChunkFingerprint` field names, for the four reasons that come from a
   field comparison). `"stack_changed"` (issue #95) only ever appears when
   the operator set `resume_require_same_stack`; by default a chunk from
   another ComfyUI/torch build is reused and the *run* reports the split.
   `None` for a freshly rendered chunk with nothing to reject, including
   every chunk after a schema-version rejection — see the "Custody and
   resume" bullet above for why that collapse is deliberate rather than a
   missing case — and `None` on every `CACHED` chunk, which is a guarantee
   `resilience._as_cached` provides rather than a property of which paths
   happen to set it.
3. **Within-chunk step progress** — still open. Lives on ComfyUI's WebSocket
   inside `execution.py`, never persisted. Needs a seam there; nothing in
   this pass touched it, and it remains the one gap `run_state.json` cannot
   close because it is not the kind of fact that file was ever going to
   carry — see the module docstring's D4.

**The stop-below-floor path gets a run-level field, not a per-chunk one.**
`ResilientRunner._check_vram_before_chunk` persists the run state accumulated
so far and raises `VramBelowFloorError` *before* the chunk that tripped the
floor ever gets a `ChunkResult` — so there is no honest chunk to hang the
reading on. `RunState.vram_stop` (a `contracts.VramStopEvent`: `chunk_id`,
`free_vram_gb`, `floor_gb`) carries it instead, `RunProgress` surfaces it as
`vram_stop_chunk_id` / `vram_stop_free_vram_gb` / `vram_stop_floor_gb`, and
`events_between` emits a one-time `run_stopped` event the first poll after it
appears — mirroring how `run_finished` fires once when `RunProgress.finished`
flips. Because `VramBelowFloorError` propagates out of `render_run` rather
than returning, the only way a caller sees this is by polling
`run_state.json` after the process has stopped, exactly as issue #36
intended: the file, not a callback, is the source of truth.

Two smaller facts about `run_state.json` that a UI author will otherwise
rediscover:

* `ChunkStatus.PENDING` is a legal enum value that `resilience` never
  persists — a pending chunk simply has no entry. `progress.py` therefore
  treats a `PENDING`-status result (a hand-edited file, a future build) as
  `"failed"`, distinct from a genuinely absent entry.
* `resilience._serialize_run_state` / `_deserialize_run_state` are now
  implementation details behind public `resilience.load_run_state(path)` /
  `dump_run_state(run_state)`. `progress.read_run_state` calls
  `load_run_state` rather than reaching into the private function it used to
  — the reach this section apologised for is closed.

## Testing

The existing mock ComfyUI harness (#16) drives the control half's `vram`
check — `FakeComfyUISession.set_vram_free` is where every free-VRAM reading
in `tests/test_control.py` comes from — exactly as it drives the pipeline: no
GPU, no network, no live server in CI. Nothing about the read-only monitor
needed it, since it never renders anything, and the control half still never
renders anything either: every controller test but two injects a spawner that
starts nothing.

Those two exceptions are the one claim in this design that cannot be faked,
so they spawn a **real** child — eight lines of python, no ffmpeg, no
ComfyUI, no network — and assert that it is its own process-group leader and
that `killpg(…, SIGINT)` reaches it and runs its handler. "Stopping is
exactly what Ctrl-C does" is the whole safety argument for the stop button,
and a mock cannot support it.

For the server itself the two tests that must exist on day one are the
bind-address assertion and "starting a run while one is in flight is
refused". **Both are built:** the first is tested exhaustively (see "What's
built: the read-only monitor" → "Bind addresses"); the second was recorded
here as *not applicable* until 2026-10-04 and now exists for real, in
`tests/test_control.py::TestOneRunAtATime` (the gate, and a second controller
seeing the claim through the lock file) and
`tests/test_webui.py::TestControlStart` (the 409 a browser gets, refused and
not queued). The control routes' own cross-site defences are tested both ways
the Host check is: the pure predicate over a table of origins, fetch-site
values and tokens, and real loopback POSTs carrying each shape. Route dispatch, SSE diffing, thumbnail
caching, and HTML escaping are tested the way this project tests everything
else offline: pure functions directly where possible
(`resolve_bind_addresses`, `stream_progress_events`, `render_index_html`),
and a small number of real sockets on `127.0.0.1` with port 0, torn down in
a fixture, where HTTP semantics (status codes, headers, method dispatch)
are what's actually under test.

The `Host` check is tested both ways for the same reason the bind list is:
the pure predicate over a table of names, literals, ports and malformed
authorities, and real loopback requests carrying a hand-set `Host` (including
one with no `Host` at all, and one with two) asserting `421` with no run data
in the body across `/`, `/events`, `/chunks/<id>/thumbnail.png`, `/review`,
an unknown path, `POST` and `HEAD`.

`/prepare` is tested against real written report files, not a mocked reader:
an unprepared run, a torn one, a report whose stamped input has since been
edited (the stale banner), and one whose plan text contains `<script>`.
