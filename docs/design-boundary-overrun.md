# Boundary overrun — quantize the work, not the boundary (issue #100)

**Status: built, opt-in, unrendered.** Every number below is timeline
arithmetic reproducible with `--prepare` and no GPU. Nothing here has been
measured on pixels, which is why `boundary_overrun` defaults to `false`.

## The idea, and where it came from

Issue #100 is a field report from another pipeline built on this one. Their
general form:

> When a generator only produces lengths on a grid, quantize the work, not the
> boundary — produce the next valid length and discard the overrun, so the
> boundary can stay where the content needs it.

MiniMax H3's `length` is a frame count ComfyUI quantizes to `5 + 17k`
(`docs/h3-node-schema.md`). This pipeline renders each chunk at exactly its own
length, so a chunk's length and its boundary have always been **one decision**.
That is the whole of #70: quantizing a chunk *moves its boundary*, which is
what cuts phrases mid-utterance. The depth-threshold preference #70 asked for
was built, scored against the real 80-chunk "Deathless" timeline, and **fired 0
times at every threshold tried** — clearing a phrase head costs 4–6 grid steps
of 0.708 s, and the chunk that has to pay sits at the 124-frame trained floor
in 20 of the 24 cases with nothing to give.

Ask H3 for the next valid length *past* the boundary and the two decisions come
apart. The boundary becomes a content decision; the length stays a VRAM
decision; the grid is paid in frames nobody watches. The 4–6 steps a move used
to cost become **zero**, because the neighbour is no longer paying for the grid.

## What it does

With `boundary_overrun = true`, after every other slicing pass has run
(`_overrun_timeline`, "pass 7"):

1. Every interior boundary that lands **inside** an aligned segment moves to
   that phrase's edge — its **start** first, its end only as a fallback.
2. The final boundary is shortened to the track's own end.
3. Each chunk then keeps `round(end·24) − round(start·24)` frames, both
   rounded from the start of the track, and is **rendered** at
   `cover_frames(kept) = clamp(quantize_up(kept), 124, 362)` — the same closed
   form the issue states, pinned by a test.
4. Stage 5 (`assembly.trim_overruns`) copies the first `kept` frames of each
   such chunk with `-c:v copy -frames:v N` and concatenates *those*.

The start edge is preferred over the nearer edge deliberately: both clear the
phrase, but moving back opens the *later* chunk on the phrase's first word,
which is issue #79's own objective (H3 starts the mouth at frame 0 regardless
of where the voice does). Moving forward leaves that chunk beginning in a vocal
gap — the "mouth moving with no vocal, then perfect once the singing starts"
defect a viewer has named twice. The issue's own pipeline makes the same choice
for the same reason.

### The invariant, and how it is held

`video offset == audio offset` for every chunk, by construction, including the
last one and the instrumental filler. Boundaries are held as **absolute
integer frame positions**, so the kept counts telescope: the frames before
boundary *i* sum to exactly `round(b_i · 24)` whatever moved and by how much.
`FrameGrid.frames_between` is deliberately `round(end) − round(start)` and
never `round(end − start)` — rounding the difference lets each chunk keep its
own residue, which is the drift #20 exists to eliminate.
`test_boundary_overrun.py` proves it with integer arithmetic over a whole
synthetic song rather than a seconds tolerance, because a tolerance would hide
exactly the sub-frame residue that accumulates.

Stage 2a additionally **refuses** a chunk whose kept count disagrees with its
own span, or whose rendered length is off the grid. Today those agree by
construction; the guard exists so the next edit to `slicing.py` cannot
introduce the disagreement silently.

### One defect found while measuring, worth keeping in mind

The first version rounded a phrase edge to the **nearest** frame. A phrase edge
is almost never on a frame boundary, so 72.920 s became 72.9166 s — still
0.003 s *inside* the phrase. Measured on Deathless: the 24 mid-phrase cuts went
to 23 while the deepest went from **86.0% to 99.9%** through its phrase. A pass
reporting success at clearing a phrase it had merely moved to the far end of —
which is #70's own first mistake arriving by a new road ("six snaps that every
one of them silently landed back inside the segment it claimed to avoid").
Rounding must go **outward**: floor for a start edge, ceil for an end edge.

### Four places that had to learn the difference

"Rendered" and "kept" were the same number everywhere until now, so every
consumer of `frame_count` had to be asked which one it meant. Three of the
four were wrong by default, and each would have been silent:

- **`cli.py`'s `chunk_frame_counts`** is H3's `length`. It must be the
  **rendered** count; the kept count would render exactly what the trim exists
  to avoid having to do, and leave the stem longer than the video.
- **`stems.py`** (issue #25's isolated vocal) cuts its own conditioning audio
  at the chunk's span and refused anything disagreeing with `frame_count`.
  That produces a stem *shorter* than the injected `length` — #20's drift from
  the other side, on the one path whose entire purpose is to be the
  conditioning signal. It now cuts to the rendered end, via one
  `_rendered_end` helper so the reach check, the slice and the duration guard
  cannot disagree about which end they mean.
- **`resilience.py`** serialises the fingerprint to `run_state.json`. A field
  written to a dataclass and dropped on serialisation is #38/#39/#45's blind
  spot one layer out: the number exists and no resumed run can read it.
- **`slicing._log_unmeasured_chunks`** warns about frame counts whose VRAM
  nobody has measured. VRAM is paid on what is rendered.

`AudioChunk.rendered_frame_count` is the single accessor for all of them, so
no caller has to know the `None`-means-no-overrun convention.

## Measured cost, on "Deathless"'s real timeline

`prepare_timeline` against `run_v14_cine.toml` (`alignment_model_size =
"small"`, 57 aligned segments, `shot_plan_v12.toml`'s lengths), both arms,
no GPU:

| | `max_chunk_seconds = 8.0` (the real run) | `= 12.0` (#70's measured lever) |
|---|---|---|
| chunks | 80 → 80 | 64 → 64 |
| frames kept | 12334 → **12290** | 12305 → **12290** |
| frames rendered | 12334 → **12419** | 12305 → **12509** |
| **overrun** | **+129 frames, +1.05%** | **+219 frames, +1.78%** |
| chunks with an overrun | 0 → 11 of 80 | 0 → 26 of 64 |
| largest single overrun | — → 44 frames (the tail) | — → 16 frames / 0.667 s |
| boundaries moved onto a phrase edge | — → 5 of 79 | — → **13 of 63** |
| **mid-phrase cuts (#70)** | 24 → **19** | 18 → **6** |
| deepest surviving cut | 86.0% → 44.5% | 92.9% → 99.5% |
| timeline-vs-track drift (#22) | +1.837 s → **+0.003 s** | +0.625 s → **+0.003 s** |

Read this honestly:

- **The cost is about a third of the issue's 3.7%** at this song's real
  settings, and about half of it at 12.0 s. Not because anything here is
  cleverer: our chunk lengths already sit on the grid, so only the chunks
  either side of a *moved* boundary pay anything at all. Their pipeline
  chooses every boundary freely, so every shot pays. The 0.667 s per-shot
  ceiling matches their "at most 16 frames (0.67 s) per shot" exactly, which
  is the part of their measurement that transfers.
- **At the song's real `max_chunk_seconds = 8.0` the win is small: 24 cuts
  → 19.** Nineteen boundaries could not move because one of the two
  neighbouring chunks would have fallen outside the 5.167–8.000 s window. The
  binding constraint on this song is still `max_chunk_seconds`, exactly as
  CLAUDE.md records.
- **The two levers compose, and that is the interesting result.** 12.0 s alone
  takes 24 cuts to 18; 12.0 s *with* overrun takes it to **6**, for +1.78%
  frames. At ~3.7 min per 141-frame chunk at 864×480, 219 frames is roughly
  **5–6 minutes of a 5-hour render**.
- The deepest *surviving* cut going up (92.9% → 99.5%) is not a regression:
  outward rounding means a boundary is either cleared or left exactly where it
  was, so the new worst case is a leftover that was previously ranked lower.
  Verified by a test that no moved boundary lands inside any segment.
- The drift column is a side effect worth naming. The grid-tiled timeline
  overshoots the master by up to one trained-floor chunk and `-shortest`
  discards those frames with nothing logging it (CLAUDE.md's `-shortest`
  bullet, #22). A free final boundary just stops where the song does, and the
  44 frames become a **recorded** overrun instead of an invisible one. The
  residual 0.003 s is 0.08 of a frame and not representable.

## What a resumed run can tell

`ChunkFingerprint.render_frames`, in **`CONDITIONING_FIELDS`** — the
inescapable, non-timeline tier, beside `conditioning_source` and
`instrumental_audio_gain_db`.

A rendered-and-trimmed chunk is in exactly the right *place*: it keeps
`frame_count` frames starting at `start`, and the timeline tier already proves
that. What differs is what H3 was conditioned on — the stem is one grid step
longer than the chunk's own span and carries up to 0.708 s of the next phrase,
so the kept frames are not the frames a render *to* `frame_count` would have
produced. That matters more here than for any other field on the list, because
the first thing anyone does with this flag is an A/B against a run without it,
and reusing the other arm's chunks hands that comparison its control twice.

`None` means "rendered exactly to length", which is what every pre-#100 state
file records *and* what a present-day run without the flag records — so an old
file and a default run compare equal and nothing re-renders. No
`schema_version` bump, the same coincidence that let `timeline` skip one. A
chunk whose content length happens to land on the grid anyway normalises to
`None` too: the pixels are identical, and re-rendering it to write down a
redundant number would be charging GPU hours for bookkeeping.

## Deliberate limits

- **Opt-in, and it must stay that way until something renders.** It re-cuts
  every boundary in the song, invalidating every mp4 on disk and every chunk a
  `--resume` would reuse.
- **Requires `instrumental_coverage`** (ignored, loudly, without it): the kept
  frames are measured from the start of the track, and there is no coherent
  way to do that over a timeline with holes in it.
- **Refused alongside `i2v_continuity`.** On the chained path a chunk's entire
  identity conditioning is its predecessor's last *rendered* frame
  (`continuity.extract_last_frame` probes the file and never trusts a
  requested length, by design) — and with an overrun that frame is in the
  discarded region, up to 0.708 s past where the next chunk begins. Every
  chain would be seeded from footage the finished video never shows, silently.
  Refused at config load rather than degraded at render time, the same split
  #49 draws: threading "keep only N frames" through chaining is real work, to
  be done when somebody wants both.
- **The trim is not opt-in at assembly, and that asymmetry is the point.** The
  opt-in lives where the timeline is *planned*. By the time a chunk says it was
  rendered past its own end, concatenating it whole is a desynced video with no
  error anywhere — and an invariant enforced as a side effect of a flag
  somebody else set is not enforced.

## What remains unproven until it renders on a real card

1. **Whether the conditioning cost is free.** H3 is audio-driven; each chunk's
   stem now carries up to 0.708 s of a line the chunk does not show. The
   project's own finding is that *a conditioning signal outranks the prompt
   describing it*, so the honest prior is that this does something. Nothing
   here can say what.
2. **Whether clearing a mid-phrase cut is visible.** #70's reopen rests on
   viewer reports, and depth-of-cut already failed to separate the labels it
   was drawn from (`docs/deathless-render-corpus.md`). 24 → 6 cuts may be
   worth nothing.
3. **Whether `-frames:v` with `-c:v copy` is exact on H3's own output.** It
   stops after N packets; a closed-GOP stream truncates cleanly, and these
   chunks are each a self-contained encode, but nobody has counted frames in a
   trimmed file with `ffprobe -count_frames` on this stack. That is the first
   thing to check, and it costs one chunk.
4. **The seam at a trimmed boundary.** The last kept frame of a trimmed chunk
   was generated with 0.7 s of future context the next chunk's first frame was
   not. Whether that reads as a cleaner cut or a worse one is a viewing.

The A/B that would settle 1 and 2 is cheap and well-defined: two configs
identical but for one line, separate output dirs, `render_frames` in the
fingerprint's inescapable tier so a resume cannot mix arms.
