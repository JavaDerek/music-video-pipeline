# Runbook: the attended long-chunk VRAM proof (issue #98)

**What this is for.** `max_chunk_seconds = 12.0` is issue #70's measured lever
for mid-phrase cuts (80 chunks / 24 mid-phrase cuts → 64 / 18 on "Deathless"),
and it has **never run on doris**. The longest chunk this card has been shown
to render is **192 frames at 864×480**. #24's failure mode — H3 going silent
instead of raising CUDA OOM, wedging the host past SIGKILL, recoverable only by
a power cycle that frequently lands in Windows — is why the first attempt is one
chunk with a human at the keyboard, not an overnight render.

Read the GPU custody protocol in `CLAUDE.md` before starting. This runbook does
not restate the per-tenant stop/start commands; those are operator knowledge and
are deliberately not in the repo.

---

## 0. The number in the issue title is wrong, and it matters

Issue #98 says **288 frames**, which is `12.0 × 24`. H3's `length` is quantized
to `5 + 17k` (`contracts.FrameGrid`), and **288 is not on that grid**. The grid
points near it are:

| frames | seconds | note |
|---|---|---|
| 192 | 8.000 | **proven** — "Deathless" v13, 864×480 |
| 209 | 8.708 | |
| 226 | 9.417 | |
| 243 | 10.125 | |
| 260 | 10.8333… | write **10.834** — see the rounding trap below |
| **277** | **11.542** | **what `max_chunk_seconds = 12.0` actually produces** |
| 294 | 12.250 | needs `max_chunk_seconds ≥ 12.25` |
| 362 | 15.0833… | H3's trained ceiling; write **15.084** |

`slicing._grid_frames_at_or_below(12.0)` returns **277**. So the proof renders a
**277-frame** chunk, and 288 frames is not a thing this pipeline can ask for.
Record 277 in the issue, not 288.

**Rounding trap.** `max_chunk_seconds` is a floor, not a target: the slicer
takes the largest grid point *at or below* it. A frame duration that is not
exact in decimal must be written **rounded up**, or you silently get the
previous grid point. `max_chunk_seconds = 10.833` gives **243** frames, not 260,
and `15.083` gives **345**, not 362 — in both cases with no warning anywhere,
because the value asked for is perfectly legal. Check with:

```bash
python -c "from music_video_maker.slicing import _grid_frames_at_or_below as f; \
from music_video_maker.contracts import H3_FRAME_GRID as g; print(f(10.834, g))"
```

---

## 1. Build the proof config (no GPU)

Copy the current run config (e.g. `run_v13.toml`) to `run_proof277.toml` and
change exactly these lines:

```toml
# The lever under test. 12.0 s -> 277 frames (see the table above).
[hardware]
max_chunk_seconds = 12.0

# Somewhere else entirely: this run must not write into, or resume from, the
# keeper's cache. A fingerprint mismatch would re-render v13 chunks for free,
# but a shared chunks_dir also means a stray mp4 in the finished render's
# directory, which is how a measurement ends up pointed at the wrong render
# (#93).
chunks_dir     = "output/proof277"
run_state_file = "output/proof277/run_state.json"

# Issues #24/#98: the pipeline refuses to submit a chunk larger than anything
# in envelope.PROVEN_ENVELOPES. This run IS the measurement that extends it.
acknowledge_unproven_envelope = true
```

(`load_config` does **not** expand `~`; write paths as the existing config does,
relative to the run directory or fully spelled out.)

Then **comment out `shot_plan`**. Raising the ceiling re-anchors the whole
timeline, so a plan authored at 8.0 s raises `ShotPlanDriftError` against it —
and re-authoring a plan is not part of measuring VRAM. With no plan, each chunk
falls back to `narrative_concept`, which renders fine and costs nothing.

Leave `render_width` / `render_height` at **864×480**. The proof has one
variable. If 277 frames at 864×480 holds, *that* is what gets recorded; 1344×768
at 277 frames is a separate, much larger question and a separate proof.

---

## 2. Find the chunk (no GPU, ~50 s)

```bash
music-video-maker --config run_proof277.toml --prepare \
    --shot-plan-out /tmp/skeleton_proof277.toml 2>&1 | tee prepare_proof277.log
```

Two lines in that log tell you what you need.

**Which chunk is longest** — `slicing._log_unmeasured_chunks`:

```
N of M chunk(s) exceed 192 frames, which is <provenance>.
The longest is chunk 57 at 277 frames (11.542s). ...
```

That `chunk NN at 277 frames` is the chunk to render. If several reach 277,
take the **first** one: it is reached soonest if you later decide to let the run
continue, and its neighbours are already in the log above it.

**Whether 277 is reachable at all on this song** —
`slicing._log_untenable_segments` names any phrase this ceiling still cannot
hold whole. If the "longest is" line reports fewer than 277 frames, no chunk in
this song reaches the ceiling; the proof then measures whatever the largest
actually is, and you record *that* number, not the one you hoped for.

Sanity-check the skeleton before spending the card: chunk count should drop
(24 mid-phrase cuts → 18 on "Deathless" at this ceiling), and the timeline drift
line should still be small and positive.

---

## 3. Take custody

Per `CLAUDE.md`'s custody protocol, by hand:

1. Enumerate the card's tenants (`nvidia-smi --query-compute-apps=...`,
   `systemctl list-timers --all`). **Do not assume** the table in `CLAUDE.md` is
   current.
2. Stop them the sanctioned way — note in particular that Ollama's service is
   *paused*, never the container.
3. Confirm the card actually reads cold. With nothing but ComfyUI resident this
   has measured ~23.1 GB free (2026-09-20). Anything materially lower means
   something is still holding memory: find it before continuing, because this is
   the one pre-flight that still works on this stack.

A one-chunk run never reaches the between-chunk path, so
`release_vram_between_chunks` and `between_chunk_min_free_vram_gb` are not part
of this measurement. The cold `min_free_vram_gb` pre-flight still runs and still
gates.

---

## 4. Start the sampler (its own terminal, before submitting)

```bash
python -m music_video_maker.vramsample \
    --out ~/mvm-runs/deathless/measurements/proof277_vram.csv \
    --interval 0.5 \
    --label "277-frame proof, chunk 57, 864x480, run_proof277.toml"
```

It polls `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
every 0.5 s and flushes every row as it takes it — a wedge needs a power cycle,
so whatever is on disk at that instant is the whole record. The file opens with
its own provenance header (label, exact query, start time): a filename is not
provenance (#93).

Take a **baseline** first: let it run for ~15 s with nothing submitted and note
the idle figure. `memory.used` is the whole card's occupancy, not H3's appetite,
and every number below is only meaningful against that baseline.

Have ComfyUI's own stderr visible in a third terminal. Its `Requested to load
MiniMaxH3 … NNNNN MB Staged` line is the last thing that appeared before the
2026-08-07 wedge, and its per-step sampler progress is how you tell sampling
from decode.

---

## 5. Render exactly one chunk

```bash
music-video-maker --config run_proof277.toml --only-chunks 57 2>&1 \
    | tee render_proof277.log
```

`--only-chunks` deliberately skips Stage 5 — the deliverable is a clip to watch
and a CSV, not a video.

**Watch these three things, in this order:**

1. **Staging.** ComfyUI logs `… MB Staged`. The sampler's first big step is
   here.
2. **Sampling.** The orchestrator logs WebSocket `progress` events, step *k* of
   *n*. Note the sampler's reading while these are advancing — this is *peak
   VRAM during sampling*.
3. **VAE decode.** Progress events stop and the chunk has not finished yet. This
   is the phase nobody has measured above 192 frames, because temporal VAE
   decode memory scales non-linearly with frame count. Note the peak here
   separately — it is the number the whole proof exists for.

Extrapolating from the one measured point (141 frames at 864×480 ≈ 3.7 min),
277 frames should land around 7 minutes. Treat that as an expectation to compare
against, never as a deadline: a chunk taking 12 minutes is not by itself a
failure.

---

## 6. What success and failure look like

**Success.** The chunk finishes, an mp4 appears in `proof277/`, and
`run_state.json` records it `rendered`. Then:

```bash
ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of csv=p=0 \
    ~/mvm-runs/deathless/output/proof277/chunk_0057.mp4
```

It must print **277**. Count the frames in the file, not in `run_state.json` —
that is the #93 lesson, and it is what caught `MEASURED_MAX_FRAMES` being stale
in the first place. Watch the clip too: a frame count is not a usable shot, and
H3 at 277 frames has never been looked at.

**Clean failure (good).** ComfyUI raises `CUDA out of memory`, the runner
classifies it, retries, dead-letters the chunk and releases the card. Nothing is
wedged. This is a *result*, not an accident — go to §8 and bisect.

**The wedge (bad).** The signature, from 2026-08-07:

- ComfyUI's last line is `… MB Staged` (or a sampler step) and **nothing
  follows**;
- the sampler's `used_mib` column goes **flat at a high value** and stays there —
  no oscillation at all, where a live render's numbers move;
- no WebSocket `progress` event arrives;
- no `dmesg` line, no traceback, no CUDA error anywhere.

None of those numbers is a measured threshold, and this runbook does not invent
one. **Flatness plus silence is the signal, not any particular figure.**

---

## 7. The abort rule

> **If five minutes pass after submission with no resolution — no progress
> event, no error, no completion — stop. Do not wait longer. Do not
> `POST /interrupt`.**

This is issue #24's own stop condition and it is not negotiable by judgement in
the moment. The 2026-08-07 incident never recovered: `systemctl stop` hung, the
process survived SIGKILL (blocked in a driver call), the modules would not
unload, `nvidia-smi --gpu-reset` cannot fire while anything holds the device,
and the display went black for 40 minutes looking exactly like dead hardware.
Waiting longer buys nothing and costs the machine.

Ctrl-C the orchestrator (it will not help, but it costs nothing — and its HTTP
calls are now bounded at 60 s each rather than forever, so it *will* come back
and its run state is already on disk). Then:

1. **Stop the sampler with Ctrl-C** and keep the CSV. Its last rows are the only
   record of what the card was doing. Copy it somewhere outside the machine if
   you can.
2. Capture `nvidia-smi` output, `dmesg -T | tail -100`, and ComfyUI's last 50
   log lines — into a file on **another machine**, or a phone photo. A power
   cycle may lose anything only in a scrollback.
3. Power-cycle doris. Expect to land in Windows; the manual `bcdedit` recovery
   from another machine is described in `CLAUDE.md` / #23 / #24.
4. After it comes back: `dkms status` before trusting the driver (the kernel
   freeze in `CLAUDE.md` exists because a driver breakage arms silently and only
   fires on reboot), then a cold `nvidia-smi`.
5. Only then, cheaply, rule out the hardware hypothesis #24's comment names:
   `nvidia-smi -q -d POWER` — starved PCIe power delivery has produced an
   unkillable RTX 40-series process with no software cause.

---

## 8. If 277 fails: the decision rule for narrowing the ceiling

The ceiling is somewhere in `(192, 277]`. The only candidate values are the grid
points, so this is a **bisection over four unknowns**: 209, 226, 243, 260. Each
step is one attended one-chunk render, identical to the above with only
`max_chunk_seconds` changed. Worst case: three more renders.

| step | set `max_chunk_seconds` | frames | if it holds | if it fails |
|---|---|---|---|---|
| 1 | 12.0 | 277 | **done** — record 277 | go to step 2 |
| 2 | 9.417 | 226 | go to step 3 | go to step 4 |
| 3 | 10.834 | 260 | record 260 | try 10.125 → 243 |
| 4 | 8.709 | 209 | record 209 | record 192 — the ceiling is where it already was |

Any `max_chunk_seconds` in `[frame_seconds, next_frame_seconds)` gives the same
frame count — mind the rounding trap in §0 for 260. **Re-run
`--prepare` after every change** — each ceiling produces a different timeline
and therefore a different longest chunk and a different chunk id.

Two things to keep honest while bisecting:

- **A "holds" at 260 does not prove 243.** It does, physically, but this table
  records what *ran*. `envelope.EnvelopePoint.covers` compares per axis, so a
  260-frame point does cover a 243-frame chunk automatically — you do not need
  to render 243, and you must not record it as if you had.
- **Record failures too.** A frame count that failed is the more valuable half
  of the measurement and there is nowhere in the code for it to live. Put it in
  the issue.

---

## 9. Record the result

**In `music_video_maker/envelope.py`**, add the measured point to
`DORIS_4090_PROVEN`:

```python
EnvelopePoint(
    frames=277,
    width=864,
    height=480,
    source=(
        "issue #98's attended one-chunk proof, <date>: chunk 57 of 'Deathless' at "
        "max_chunk_seconds = 12.0, peak <N> MiB during sampling / <M> MiB during VAE "
        "decode against a <B> MiB idle baseline; 277 frames confirmed with "
        "`ffprobe -count_frames`"
    ),
),
```

That is what lets the next v14 run start without `acknowledge_unproven_envelope`,
and what the refusal message points at. **Do not leave the acknowledgement set**
in a config after the proof — that is the same mistake as leaving an assertion
disabled.

**In issue #98**, comment with:

- The frame count that ran (277, or whatever the bisection reached) and the
  resolution.
- Idle baseline MiB, peak during sampling, peak during VAE decode, all from the
  CSV — plus the CSV's path, since it carries its own provenance header.
- Wall clock for the chunk, against the 3.7 min/141 frames extrapolation.
- Frame count from `ffprobe -count_frames`, not from `run_state.json`.
- Whether the clip is *watchable* — 277 frames of H3 has never been looked at.
- Every frame count that **failed**, and how (clean OOM vs. wedge).
- A `General form:` line first, per `CLAUDE.md`'s issue-writing rule.

**`MEASURED_MAX_FRAMES` stays at 141** unless Derek decides otherwise: raising it
is a decision about what an operator wants warned, not a correction. The
*message* around it no longer claims to be a measurement, and the number a run
actually reports is now derived from `run_state.json` when one is available
(`envelope.measured_ceiling`).
