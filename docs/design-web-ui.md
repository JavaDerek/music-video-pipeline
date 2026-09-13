# Web UI: collect a run, gate on the pre-render checks, show live progress (issue #36)

There is no UI. A run today is hand-edited TOML, a CLI, and `grep` over
`render.log`. That is workable for the person who wrote the pipeline and a wall
for anyone else — and it hides the two things that decide whether a run is
worth starting.

This document is the design. One piece of it is built:
`music_video_maker/progress.py`, the layer underneath the server. **There is
no server, and nothing in this repo binds a socket.**

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

### The next slice, specified

A read-only `review` command that renders one HTML page (and the same data as
JSON) from a `--prepare` output. No server, no forms, no start button — a file
you open. That keeps it honest: it can be built and tested offline today, it is
useful from the CLI before any server exists, and it becomes the server's
review page verbatim.

It needs, per chunk: `chunk_id`, span, duration, frame count, voiced /
instrumental, the lyric text, the shot line, `camera`, `location`, `present`,
`subject`, and every `alignment_quality` finding that touches the chunk's span,
plus the run-level quality summary and the `lint_*` warnings
`shot_plan.check`/`lint_shots_against_lyrics` raise. All of that already exists
as return values; none of it needs new measurement.

It was **not** built here on purpose. Its producers (`alignment_quality`,
`shot_plan`'s lints) are owned by other work in flight, and a renderer written
against them today is a merge conflict with no user. Build it after those
settle, in one commit, with a golden-file test.

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

* **GPU custody is exclusive.** The UI must refuse to start a second run while
  one is in flight, and must not bypass the #19 pre-flight. The stopping of the
  card's other tenants is deliberately manual and stays manual — see
  `custody.py`'s module docstring. A UI button that stops the Ollama container
  would be the exact automation this project decided never to build.
* **Release is unconditional.** Any path that renders releases ComfyUI's VRAM
  in a `finally`. A direct-library test script that skipped it once left 17 GB
  held and starved every other tenant on the card.
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
  would be less honest than `None`, not more. See "Closed" below.
* **Dead-lettered chunks** are called out with their error history and offered
  a resume. `ProgressEvent("chunk_dead_lettered")` carries the full `errors`
  tuple for exactly this.

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
   `"content_changed"`) plus `rerender_reason_fields` (the `ChunkFingerprint`
   field names, for the three reasons that come from a field comparison).
   `None` for a freshly rendered chunk with nothing to reject, including
   every chunk after a schema-version rejection — see the "Custody and
   resume" bullet above for why that collapse is deliberate rather than a
   missing case.
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

The existing mock ComfyUI harness (#16) should drive the UI's backend, exactly
as it drives the pipeline: no GPU, no network, no live server in CI. For the
server itself the two tests that must exist on day one are the bind-address
assertion above and "starting a run while one is in flight is refused".
