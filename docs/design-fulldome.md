# Fulldome trailer (Refestramus): gap analysis, build plan, prototype

Target: a 60–120 s fulldome trailer set to one Refestramus track, later a
3–5 min version for festival submission (FullDome Festival Jena "New Talent",
≤5 min), as a **domemaster** — 4096×4096, 180° equidistant fisheye, front at
the bottom, 30 fps, lossless 16-bit image sequence master, H.265 distribution
copy at ~60 Mbit/s, six mono 48 kHz/24-bit stems in L R C LFE Ls Rs order
plus a stereo fold-down, frame-accurate with a 2-pop.

This document is three things: (1) what this pipeline does today measured
against that target, (2) the build plan for what is missing, and (3) the
record of the smallest end-to-end prototype — one verified 60-second test
render, made 2026-09-28 from the finished "Deathless" v13 render and the
album's 5.1 master, with no GPU time at all.

## Status

| Piece | State |
|---|---|
| Domemaster projection of a flat clip ("window" route) | **built** — `music_video_maker/dome.py`, `tests/test_dome.py` |
| 16-bit PNG master + H.265 distribution copy | **built, rendered** — see "The prototype run" |
| Six 5.1 stems + stereo fold-down, sample-accurate, with 2-pop | **built, rendered** |
| Automated checks (size, mask, count, rate, channels, length, sync) | **built**; every check passed on the prototype |
| Venue orientation as a parameter (no re-render of content) | **built** — `front_rotation_degrees`, applied at projection time |
| 24 → 30 fps retime | **built**, three modes; **shimmer not yet judged by eye at full size** |
| Credits / call to action with QR | **built** as a run asset; **QR not yet scanned from a projected preview** |
| Dome-native background | **built**, static starfield only |
| Preview in a dome simulator / VR headset | **not done** — a seat's-eye reprojection is built instead; not a substitute |
| 360° route (B) and 3D-scene route (C) | **evaluated, not built** — see "The three routes" |
| Full-dome D1: procedural, audio-reactive, native 4096² | **prototyped, 60 s verified** — see "Full-dome routes" |
| Full-dome D2: AI panorama + depth parallax | **prototyped, 25 s verified**; not good enough alone — see "Full-dome routes" |
| Combination: D2 land + D1 sky + H3 singer as a cut-out | **built, 80 s verified** (2026-09-30), after Derek's viewing |
| Fisheye base layer in `dome.py` (`--base`), H3 as an inset (`--window-opacity`) | **built** |
| §5 brightness and full-field flash checks | **built**, in `verify_render`; route A, D1 and D2 all pass |
| Dome-authored content (calm camera, dark palette, front-weighted) | **not done** — the prototype reuses a music-video render |
| Rights record per model | **open** — see "Rights" |
| Bogotá / Lunaria tech riders, 7.1 | **venue-dependent, not hard-coded** |

## What the pipeline is today, in the spec's terms

Measured facts, not the README's ambitions:

- **Generation:** MiniMax H3 through ComfyUI on doris (RTX 4090). Renders
  at **864×480, 24 fps** (H3's fixed rate: `contracts.H3_FRAME_GRID`), in
  chunks of 124–362 frames. 1344×768 is the template default and costs
  **3.1×** the GPU time (9 m 15 s vs 3.7 min per chunk, measured).
- **Camera control is prose.** The shot plan's `camera` field and the
  cinematography profile compose into the prompt; there is no camera-path
  input, no lens model, no determinism across seeds. "Slow, locked off" is
  an *instruction*, and `CLAUDE.md` records H3 ignoring instructions it was
  given for a whole render (#81: it cuts inside "one continuous take").
- **Output:** per-chunk H.264 mp4s concatenated with `-c:v copy` and the
  **stereo** master muxed in. Nothing writes an image sequence, nothing
  encodes H.265, nothing knows about more than two channels.
- **Audio:** the render's timeline is cut against the stereo master, and
  the album's 5.1 FLAC masters (`~/Downloads/5.1 FLAC/`, 48 kHz 24-bit,
  `5.1(side)`) are **different files with different heads** — on
  "Deathless", 505.560 s against 512.080 s.
- **Post-render checks that already exist:** darkness floor (`luminance.py`),
  scene cuts (`scenecuts.py`), face presence. All run on the flat render and
  all still apply — a cut inside a chunk is exactly the sudden full-field
  change §5 forbids, and the scene-cut check is the instrument for it.

## Gap analysis, section by section

The last column names where each gap lives, because "the pipeline" is three
different places: the **module** (`dome.py`, generic to any flat clip), the
**run** (a config, a shot plan, an asset in `~/mvm-runs/<song>/`), and the
**model** (what H3 can and cannot do, which no code here changes).

| Spec | Pipeline today | Gap | Lives in |
|---|---|---|---|
| §1 domemaster, 4096², equidistant, front at bottom, black outside | flat 864×480 | **closed.** ffmpeg `v360` `fisheye` output is equidistant (measured against a labelled pattern, see the module docstring); a hard disc mask guarantees pure black outside | module |
| §1 30 fps constant | 24 fps fixed by H3 | **closed by retime**, not by generation. `mci` (motion-compensated), `blend`, or `dup`. Which one holds up on a 23 m dome is a viewing question, below | module (mechanism), run (choice) |
| §1 16-bit PNG master, zero-padded | H.264 8-bit | **closed.** `rgb48be` PNGs, `dome_%06d.png` from 0. Note the extra 8 bits carry no information — the source is 8-bit — they satisfy the container, not the eye | module |
| §1 H.265 4096² ~60 Mbit/s, Rec.709 8-bit | — | **closed.** libx265 from the PNG master, explicit BT.709 matrix (swscale's RGB→YUV default is BT.601), `hvc1` tag | module |
| §2 front-weighted composition, 20–60° above the horizon | no notion of a dome | **closed for a window** (`WindowPlacement.elevation_degrees`, default 40). **Open for content**: nothing in authoring knows the front sector exists | module + run |
| §2 orientation as a parameter | — | **closed.** `front_rotation_degrees` is a `v360` roll on every layer; the flat source is untouched | module |
| §3 route A window | — | **built** (this document) | module |
| §3 route B 360° | — | **not viable with H3** — below | model |
| §3 route C 3D scene | — | **not built**; the right route for a festival piece if dome-native is the goal — below | new tool, not this repo |
| §4 detail holds at 4096² | 864×480 native | **partly open.** A 90°-wide window is ~2048 px across the frame, so the source is upscaled ~2.4× (1.5× from 1344×768). One Lanczos resample inside the projection, no ML upscaler, so no synthesised shimmer — and no synthesised detail either | run (resolution), module (resample) |
| §4 temporally stable upscaling | — | **stable by construction** (a spatial filter has no temporal term). Retime is the temporal risk; `dup` is the control arm | module |
| §4 no visible seams | — | **closed for the window edge** (feathered alpha, default 24 source px). No tiles, no stitching in route A | module |
| §5 slow motion, stable horizon, no roll | prose only | **open — content.** The prototype span was *chosen* for locked-off and slow-tilt cameras; nothing enforces it. The scene-cut check catches the worst case after the fact | run (authoring), model |
| §5 no strobing, no full-white frames | darkness floor exists, no brightness ceiling | **open — instrument.** A frame-to-frame mean-luminance jump check on the domemaster is a 40-line reuse of `luminance.py`'s probe; not built | module |
| §5 dark backgrounds, bright elements | — | **closed for the background** (dim starfield, ≤150/255 before blur). Open for the picture itself: "Deathless" is dark by authorship, another song may not be | run |
| §6 six mono WAVs 48 kHz 24-bit, L R C LFE Ls Rs, + stereo | stereo mux only | **closed.** Cut by sample count at a **measured** offset against the stereo master (1.288146 s on Deathless), 2-pop at −20 dBFS on all six, fold-down at fixed coefficients. Jena's 16-bit variant is a one-token change (`pcm_s16le`) not yet exposed | module |
| §6 frame-accurate sync, 2-pop | `-shortest` fixes the length silently (#22) | **closed and measured**: tone onset +0.021 ms from the flash frame, both prototype runs | module |
| §7 band name + CTA ≥5 s, front sector, readable | — | **closed for the mechanism** (a projected RGBA card, 8 s, faded in, elevation 35°). Readability from the back rows and **QR scan from a projected preview: not tested** | run asset |
| §8 dome simulator / VR preview | — | **open.** A seat's-eye perspective reprojection is built and proves orientation; it is not a dome | operator |
| §8 automated checks | none for this | **closed** — all seven in `verify_render`, JSON report beside the frames | module |
| §9 rights per model | `CLAUDE.md` #51 says record licences; H3's is not recorded | **open** | run/repo |
| Bogotá rider, 7.1 | — | **not hard-coded**, by design: every venue number is a parameter | operator |

## The three routes

### A. Window — built

Flat clip → feathered RGBA → `v360` flat→fisheye at `pitch = 90 − elevation`
→ overlaid on a dome-native background → hard disc mask. One `v360` pass per
layer (alpha passes straight through, measured), so a credits card is the
same mechanism at a different placement.

What the pipeline already supported: everything up to the flat clip. What was
missing: all of the above, ~500 lines, one day including tests and a
verified render. What it cannot do: the picture is a rectangle on the dome.
A 90°-wide window is large — half the field — but it is a screen in a
planetarium, not a planetarium show. For a *trailer* that is acceptable and
the spec says so.

### B. 360° — not viable with this generator

Equirectangular (2:1) or cubemap content needs the generator to produce a
seamless wrap, and H3 produces a 16:9 (or 1.8:1) rectangle from a prompt
with no camera geometry and no determinism across seeds (the same finding
that ruled out rendering a stereo pair twice — `docs/design-stereoscopic-3d.md`).
Assembling a 360° from several H3 clips means outpainting between them, and
every join is a seam that moves. The only honest version of route B is a
different model (a panoramic video model) or route C wearing a different
name. **Effort: not estimable with H3; do not start.** If a 360 model is
adopted later, the *output* side of this module (equirect→fisheye, mask,
master, audio, checks) is already what it needs: `build_background_args`
already projects an equirect texture.

### C. Scene — the festival route, and not this repo

H3 clips and stills as textures on surfaces in a 3D or real-time scene
(Blender, TouchDesigner), rendered through a 180° fisheye camera. Most
dome-native, most work, and the only route where "slow, deliberate camera
motion with a stable horizon" is a *camera path* rather than a prompt. What
the pipeline supplies: the clips, the `shot_plan.toml` as a shot list, and —
unchanged — the master/audio/verification half of `dome.py`, which does not
care where the fisheye frames came from. What is missing: the scene itself,
a fisheye render setup, and the compositing, none of which belong in a
Python orchestrator for a video model. **Effort: days to a first scene,
weeks to a 3–5 min piece; a Blender install is not on this machine.**
Recommendation: route A ships the trailer; decide on C only after the
trailer has been seen on a dome.

## Resolution and temporal quality

The window is 90° of a 180° field, so its width on the 4096 frame is about
2048 px (the mapping is not linear; the top of a low window is wider than
its bottom). From 864×480 that is a **2.4× upscale**; from 1344×768, 1.5×.
Both are done once, inside the projection, by `v360`'s Lanczos interpolator —
there is no separate upscale pass and no ML upscaler, so the two failure
modes the spec names (shimmer, crawl) cannot be introduced *here*. They can
be introduced by:

- **the retime.** `mci` synthesises one frame in five. On soft, hazy H3
  footage it should be benign; on a fast pan it will smear. `dup` (4:5
  duplication) synthesises nothing and judders. Both are one flag; the A/B is
  cheap and **has not been viewed at full size on a dome or a headset.**
- **H3 itself.** Its output has its own texture boil, which a 2.4× upscale
  magnifies. A 1344×768 render of the trailer's 60 s costs ~7 chunks × 9 m 15 s
  ≈ 65 GPU minutes and is the single most effective thing to do for §4 —
  once the card is free (it is not tonight: a 30B production model holds
  18.7 of 24.5 GB).

Whether "effective detail holds across a 23 m dome" is not a number this
pipeline can produce. It is a viewing. What *can* be said from the full-size
render: a 1:1 crop of the window (`crop_t33_window_1to1.png` beside the
copied outputs) is visibly soft — clean edges on the rocks, no ringing, no
blockiness, but no texture finer than the source had. That is the 2.4×
upscale doing exactly what it should and no more; it is the strongest
argument for step 2 of the build plan.

## Motion and comfort

The only lever on camera motion is prose, and it is unreliable. Three things
follow:

1. **Choose the span.** The prototype uses 2.0–62.0 s of "Deathless" v13
   because chunks 0–7 are authored "wide and low, locked off", "extreme wide,
   slow tilt up", "very wide and high" — and chunk 8 is "handheld close on
   her face, drifting", which is why the span stops at 62. Choosing content
   is worth more than any post-process.
2. **Author for the dome.** A dome run wants its own cinematography profile
   (`profiles.py` exists for exactly this): locked-off and slow-tilt camera
   vocabulary, dark palette, subject in the lower half of frame. Not built;
   it is a profile file plus a shot plan, no code.
3. **Measure after.** The scene-cut check (#81) already flags an authored
   single take that cut. A domemaster brightness-jump check (mean luminance
   across consecutive frames, sampled the way `luminance.py` samples) would
   catch strobing and full-white frames mechanically. Not built; the 2-pop
   flash is the one deliberate full-disc white frame and it is in the
   leader.

## Audio

Six stems, a fold-down, and one number that had to be measured:

- **The offset.** The 5.1 master is 6.52 s shorter than the stereo master
  the render was cut against, and cross-correlation puts its first sample
  **1.288146 s** into the stereo timeline (coarse at 8 kHz over 150 s, peak
  210× the RMS of the correlation; refined at 48 kHz over 60 s; independently
  confirmed at **lag 0 samples** by correlating the finished fold-down against
  the render's own audio track). `render` refuses a span that begins before
  the surround file does. The measurement script is a run asset
  (`~/mvm-runs/deathless/dome/`), not repo code — it needs numpy, and this
  package is stdlib-only on purpose.
- **Channel order.** ffprobe reports the FLACs as `5.1(side)`: FL FR FC LFE
  SL SR, which is the spec's L R C LFE Ls Rs. "Side" and "back" are labels
  for the same pair.
- **Cut by samples.** `atrim=start_sample`/`end_sample`, `adelay` for the
  leader, `apad=whole_len` to exactly `(leader + programme) × 48000`. No
  `-ss` seeks anywhere.
- **2-pop:** 1 kHz at −20 dBFS, one frame long, at 1.0 s; programme at 3.0 s.
  On all six channels.
- **Fold-down:** `0.6·L + 0.424·C + 0.424·Ls` (and mirror). Deliberately
  below unity so it cannot clip; it is a check copy, not a deliverable mix.
- **Jena's 16-bit:** `pcm_s24le` → `pcm_s16le` is one token; not yet a flag.
- **7.1 for Bogotá:** the ADM/Atmos master is the source if a venue takes
  7.1; nothing here reads ADM. Venue-dependent; do not build before a rider.

## Credits and call to action

A 1920×1080 RGBA card (band name, album, URL, QR to `https://refestramus.com`,
error-correction level H, 37×37 modules, on a 70%-black rounded panel) made by
`~/mvm-runs/deathless/dome/make_credits_card.py` (Pillow + `qrcode`, an
operator asset like the vocal stem — the package never imports either).
Projected at 70° wide, elevation 35°, from 52 s into the programme to its end
(8 s, ≥5 s as required), faded in over 1 s.

The panel exists because the first dry run put white text straight over the
singer's face — the card and the window share the front sector by design.

**Not tested:** scanning the QR from a projected dome preview. The seat's-eye
preview frame at 1920×1080 is the nearest thing available; scan that from a
phone before believing the size.

## Verification

Built, in `verify_render`, all seven from the spec's list, every failure a
row rather than an exception, JSON beside the frames:

| Check | How |
|---|---|
| frame size | ffprobe of the first PNG |
| circle mask | sampled frames decoded to raw `rgb48le`; every byte outside the disc must be 0, per row, using the *same* inequality the mask was drawn with |
| frame count vs duration | files on disk against `round((leader + programme) × fps)`; the mp4's `nb_frames` against the same |
| frame rate | the mp4's `r_frame_rate` |
| audio channels | 6 × mono + 1 × stereo, `pcm_s24le`, 48 kHz |
| audio length | `duration_ts` in samples, every file, exact |
| sync offset | the brightest leader frame must be the pop frame and the only bright one; the tone onset is the first sample over −44 dBFS; the difference must be within half a frame |

Not automated and not automatable here: viewing in a dome simulator or a
headset. The seat's-eye preview (`preview_seat_view.mp4`, a perspective
camera at the dome centre looking 30° up at the front, with the fold-down
and the pop) proves orientation and handedness; it does not prove comfort.

## Rights

Open. What to record, per `CLAUDE.md` #51 ("'It downloaded fine' is not a
licence"):

- MiniMax H3 weights as repackaged by Comfy-Org (`comfyui-setup-summary.md`
  lists the files) — the licence text shipped with the weights, and whether
  it permits commercial use and public exhibition of output.
- The fal H3 realism LoRA (#62), if the trailer's source render used it
  (v13 did not).
- Whisper via `stable-ts` (alignment only; no output pixels).
- Fonts rasterised into the credits card (Arial, from macOS) — a rendered
  image, but record it.

None of these are asserted here. They are the list.

## Build plan

The shortest path to one valid 60 s test render was **route A on an existing
render**, and it is done. What follows is ordered by what each step buys.

| # | Step | Buys | Cost | Where |
|---|---|---|---|---|
| 0 | **Done 2026-09-28**: `dome.py` + tests; 60 s verified render from Deathless v13 + 5.1 master | a dome-ready file to put in front of a projector | 1 day, no GPU | module |
| 1 | View it: dome simulator or headset; scan the QR from the projected preview; `mci` vs `dup` side by side | the answers to §4 and §5 nobody can compute | an evening | operator |
| 2 | Dome cinematography profile + a 60–120 s shot plan authored front-weighted, locked-off, dark; render at **1344×768** | §4 detail, §5 comfort *by content* | 8–15 chunks × 9 m 15 s ≈ 1.5–2.5 GPU hours, when the card is free | run |
| 3 | Brightness-jump check on the domemaster; `--audio-bit-depth 16`; `--fps 60` is already just a flag (retime cost ×2) | §5 strobing rule mechanised; Jena's audio spec | half a day | module |
| 4 | Background with slow, deliberate motion (drifting star field via `v360` yaw commands, or a second H3 clip as a sky) | dome-native feel without route C | a day | module |
| 5 | 3–5 min festival cut: the same module over a longer span of a dome-authored render | Jena submission | GPU hours from step 2 × 3 | run |
| 6 | Venue riders: Bogotá (Digistar 7, Christie Griffyn, 7.1), Lunaria (18° tilt: front-weight harder, `elevation` lower) | correct deliverables per venue | per rider | operator; `front_rotation`, `elevation`, codec are parameters |
| 7 | Route C, only if step 1 says a window is not enough | a real dome piece | weeks | outside this repo |

## The prototype run

Two runs, both from the same inputs: the finished Deathless v13 render
(`final_video.mp4`, h264 864×480 24 fps), the album's 5.1 FLAC master, the
credits card. Span 2.0–62.0 s of the render (surround 0.711854 s onward);
leader 3 s, pop at 1 s; window 90° wide at elevation 40°, feather 24 px;
credits from 52 s.

**Dry run, 1024², 8 s, `dup`, on the Mac** (M2 Pro, 8 threads): master
16 s for 330 frames; all checks passed; sync +0.021 ms. This is where the
credits panel and the preview's pitch sign were fixed.

**Full run, 4096², 60 s, `mci`, on doris** (32 cores, `nice 19`, 16 ffmpeg
threads, system ffmpeg 8.0.1, system Python 3.14 with `PYTHONPATH=.` — the
module needs nothing installed):

| Stage | Wall clock (doris log, UTC 02:13–02:27 on 2026-09-29) | Rate |
|---|---|---|
| background + disc (4096², once) | 1.3 s | — |
| master: trim, `mci` 24→30, two `v360` layers, overlay, mask, 16-bit PNG | **9 m 58 s** for 1890 frames | 3.2 frames/s |
| H.265 distribution copy from the PNGs (libx265 medium, 60 Mbit/s) | 2 m 11 s | 14 frames/s |
| six stems + fold-down | 46 s (includes verification) | — |
| verification (17 sampled frames decoded at 96 MB each; 90 leader probes; audio onset) | — | — |
| seat preview + contact sheet | 46 s | — |
| **total** | **13 m 45 s** | — |

| Output | Size |
|---|---|
| 1890 × 16-bit PNG master | **12 GB** (min 108 KB, median 7.1 MB, max 8.2 MB per frame — the leader frames are the small ones) |
| `dome_4096_h265.mp4` | 478 MB, measured bit rate 60.67 Mbit/s |
| six stems | 9.07 MB each (3 024 000 samples × 3 bytes); stereo 18.1 MB |

All eight checks passed — `frame_count` 1890, `frame_size` 4096×4096,
`bit_depth` rgb48be, `circle_mask` clean on all 17 sampled frames,
`distribution` hevc 4096² @ 30/1 with 1890 frames, `audio_files`,
`audio_length` 3 024 000 samples in every file, `sync` **+0.021 ms**
(the first sample of a sine is zero, so the onset is one sample late by
construction). The master ran at 12 GB on a machine with 193 GB free
afterwards; the Mac had 15 GB free, which is why doris rendered it.

Outputs (`~/mvm-dome/deathless_2-62_4096/` on doris; H.265, preview, contact
sheet, verification and manifest copied to
`~/mvm-runs/deathless/dome/deathless_2-62_4096/` on the Mac):

- `frames/dome_000000.png … dome_001889.png` — the 16-bit master
- `dome_4096_h265.mp4` — distribution copy
- `audio/{L,R,C,LFE,Ls,Rs,stereo}.wav`
- `preview_seat_view.mp4`, `contact_sheet.png`
- `dome_verification.json`, `dome_manifest.json` (every argv, every parameter)

To reproduce, or to render another span:

```bash
PYTHONPATH=. python3 -m music_video_maker.dome render \
  --source final_video.mp4 --source-start 2.0 --program-seconds 60 \
  --surround "06 - Deathless.flac" --surround-offset -1.288146 \
  --credits credits_card.png --credits-at 52 \
  --out-dir out/deathless_2-62_4096 --threads 16
python3 -m music_video_maker.dome verify --out-dir out/deathless_2-62_4096 --program-seconds 60
```

## Full-dome routes: D1 procedural and D2 panorama (addendum, 2026-09-28)

Route A was proven as a delivery pipeline and **rejected creatively**: it
reads as a curved screen, and the trailer has to fill the dome. Two
full-dome approaches were prototyped the same night. Both keep `dome.py`'s
back end unchanged — masking, 16-bit master, H.265, stems, 2-pop,
orientation, verification. They reach it through one new front-end input,
`--base`: a fisheye video already at the dome's size and rate, which
replaces the starfield. `dome.py` refuses a base of the wrong size or rate,
or one that runs short, rather than resampling it. The H3 clip becomes
optional, and when present it is an **inset**, not the frame: small `h_fov`,
heavy feather, `--window-opacity` below 1.

The generators are **run assets**, not package code, in
`~/mvm-runs/deathless/dome/fulldome/{d1,d2}/`. They need numpy, OpenGL and
torch; this package is stdlib-only on purpose (the same precedent as the
offset-measurement script). Every input, report, still and log from the
runs is beside them. The rendered outputs are in `…/fulldome/out/` on the
Mac (H.265, seat preview, contact sheet, verification, manifest) and in
`~/mvm-dome/fulldome/out/` on doris (the 16-bit masters).

### What changed in `dome.py`, and one correction

- **`--base`, source optional, `--window-opacity`.** Tests first; 3104 pass
  and ruff is clean.
- **A third measured sign.** A fisheye base is turned for the venue by a
  fisheye→fisheye `v360` roll. The preview's *pitch* flips when the
  fisheye is the input, so the roll's sign was measured rather than
  assumed. `FISHEYE_ROLL_SIGN = +1` is pinned by an integration test in
  which a red marker on the base and a green window at azimuth 0 must land
  at the same angle at 0° and 30° of rotation.
- **Correction: `v360` clamps a flat layer at its edge.** It does *not*
  write alpha 0 into the pixels the layer doesn't cover. The docstring
  said it did. Every uncovered pixel repeats the nearest edge pixel, alpha
  included, so a layer with an opaque border floods the whole dome. This
  was measured three ways: a solid 160×90 layer at any FOV covers 16384 of
  16384 pixels, and forcing its outermost pixel ring transparent covers
  280. Route A was right only because its window was feathered to zero at
  the border; `--feather 0`, or a credits card without a transparent
  margin, would have filled the dome. Every layer's outer ring is now
  forced transparent, the window at any feather and the credits card
  whatever its own alpha says.
- **§5 on the pixels.** `verify_render` now reads the distribution copy
  once, downscaled to 64², and averages luma over the disc only. It fails
  if any programme frame is brighter than **Y 96** (full range, 0–255).
  It also fails on more than **3 full-field flashes in any second**: a
  flash is a pair of opposing ≥20 Y swings, the ITU-R BT.1702 general
  rule. The leader is excluded because its flash is the 2-pop. The
  thresholds are a starting point, not a dome calibration, and a local
  strobe covering a quarter of the dome dilutes below the whole-disc
  mean; the docstring says both. **Control:** route A's accepted render
  passes with a peak of 23.7 and zero flashes.

### D1 — procedural, audio-reactive

A GLSL fragment shader evaluated **directly in equidistant-fisheye
coordinates at 4096²**. Each pixel's direction on the dome is computed
exactly, with nothing projected, resampled or upscaled. It renders
offline on the 4090 through headless EGL via `moderngl`, 16-bit per
channel, piped as FFV1 into `dome.py --base`.

**The audio drives the layers spatially.** `features.py` reads exactly the
span the dome plays (surround time 0.711854 s, 60 s, by sample count),
per 30 fps frame:

| Layer | Driven by | Why |
|---|---|---|
| Nebula, lit from five directions at 18° | L, R, Ls, Rs envelopes, each lighting the nebula from its own speaker's azimuth | the light comes from where the sound comes from |
| Front curtains (ember/ash) | the centre channel on its own scale | the voice, in the front sector where the singer inset sits |
| Horizon glow and ridge rim | LFE + the <150 Hz band | weight, not flicker |
| Star twinkle | 4–16 kHz band | point-sized, so it can't flash the field |
| Embers rising from the ridge | spectral-flux onsets, spawned at the azimuth of the channel that carried the onset | events have a place |
| Ridge silhouette | nothing | a fixed horizon, on purpose |

What the audio actually is matters more than the design. On this span the
centre channel sits **16 dB under L/R** (p95 −31.6 against −15.8 dBFS),
because 2–62 s is mostly the lead-in: its envelope is zero until 50 s and
then carries the voice. 113 of 123 onsets are L/R. So the curtains are
dark for 50 s and rise with the vocal, an effect the audio produced, not
one designed for.

**The comfort rules are enforced in code, not by taste** (`comfort.py`,
tested):

- **Brightness parameters** pass through an attack/release follower
  (attack floor 0.1 s, refused below it), then a hard slew cap of
  1.5 units/s. A full-scale swing takes ≥0.63 s, so no parameter can
  strobe whatever the music does.
- **Motion** is a speed integrated into a phase with capped acceleration.
  The sky turns 0.25°/s about the zenith, with no roll and no pitch, and
  nothing jumps when the music changes a speed.
- **An exposure governor with lookahead.** A 512² pre-pass of every frame
  measures disc-mean luma. A per-frame gain that starts falling a second
  *before* a bright passage keeps every frame under Y 48, and the gain
  itself is slew-capped, so the governor can't cause the flash it
  prevents.
- **The pre-pass is re-measured with the gain applied** by the same
  function `verify_render` uses (≤1 flash/s allowed, against the check's
  3), and the render stops before the long pass if it fails. It measured
  0 flashes, peak Y 48.0 (raw 61.0, gain floor 0.59).

**Result: 60 s verified.** "Deathless" 2–62 s, the same span as route A,
with the H3 render as an inset: 38° wide, elevation 26°, feather 90 px,
opacity 0.85, `mci`, in sync with the stems. Credits from 52 s. All ten
checks passed: frame count 1890, size, rgb48be, mask on 17 sampled frames,
HEVC 56.5 Mbit/s, **brightness peak 37.6 / mean 34.7, 0 flashes**, stems,
length, sync +0.021 ms.

| D1 stage (doris, 4090 + 32 cores) | Per 60 s at 4096² |
|---|---|
| audio features | 0.6 s |
| comfort pre-pass (2 × 1800 frames at 512²) | 11 s |
| shader render → FFV1 16-bit base | **7 m 38 s** (3.9 frames/s; GPU ~idle, readback- and encode-bound) |
| `dome.py` master (base + `mci` inset + credits + mask → 16-bit PNG) | 11 m 05 s |
| H.265, stems, verification, preview, contact sheet | ~4 m 45 s |
| **total wall clock** | **≈ 24 min**, of which ≈ 8 min touch the GPU |
| storage | base 27.7 GB (FFV1), master 17 GB, H.265 424 MB |

**Effort:** one session, about four hours, including the `dome.py` change.
A trailer-quality version is **2–4 days**, and nearly all of it is art
direction: the look, section changes that follow the song's structure,
grading the inset to the sky.

**Quality risks, D1:**

- **It is abstract.** Without the H3 insets it carries no story or place,
  and the insets are daylight music-video footage inside a night sky. The
  1:1 crop (`crop_inset_1to1.png`) shows the colour clash, which is a
  grade to author, not a pipeline gap.
- **16-bit gradients, 8-bit distribution.** The master has true 16-bit
  gradients, the first route where the 16 bits carry information. The
  H.265 copy is 8-bit 4:2:0, so banding in the dark sky is possible on a
  projector. Not yet looked at on one.
- **Not judged at scale.** Star size and twinkle at 23 m, and whether the
  0.25°/s sky turn reads as motion or as vertigo, are viewing questions.
- **Uncalibrated thresholds.** The governor's ceiling (Y 48) and the check's
  (Y 96) are starting values.

### D2 — AI panorama + depth parallax

Pipeline: 2:1 equirect panorama (2048×1024) → Real-ESRGAN ×4 to
8192×4096 → DA² 360 distance → per-pixel raymarch through the distance
surface from a slowly dollying camera, through an equidistant fisheye at
4096² → FFV1 → `dome.py --base`.

**Which model, and why this one** (full research, with a primary source
and date for every licence claim, in `…/fulldome/d2/d2_research.md`):

| Candidate | Seamless 360? | Resolution | Commercial use and public exhibition | Verdict |
|---|---|---|---|---|
| **Qwen-Image 2.1** (already on doris) | no 360 variant | — | **Qwen Research Licence, non-commercial only** | do not use |
| HunyuanWorld 1.0 / HY-World 2.0 (Tencent) | yes | ~2K | licence territory **excludes the EU, UK and South Korea**, and Jena is in the EU | excluded |
| DiT360, Matrix-3D, FLUX.1-dev / FLUX.2-klein-9B 360 LoRAs | yes (DiT360 designs for seam and poles) | ~2K | output clause allows commercial use, but *running* a FLUX non-commercial model for a ticketed or promotional piece is doubtful | grey; avoid without a BFL licence |
| **`ProGamerGov/qwen-360-diffusion` on Qwen-Image-2512** | yes (vetted training set) | 2048×1024 | Apache-2.0 base + MIT LoRA | **clean; best local quality**, but a 20B model needs CPU offload beside the resident LLM |
| **`ProGamerGov/sdxl-360-diffusion`** | yes | 2048×1024 | MIT UNet on SDXL (Open RAIL++-M, use-based restrictions only) | **clean; fits in 4 GiB — used here** |
| Blockade Labs Skybox AI (hosted) | yes (skybox-native) | up to 16K on Business | owned assets on a paid tier, but **its own pages disagree about which tier carries commercial rights** | viable if confirmed in writing |
| World Labs Marble (hosted) | a real 3D world, so parallax needs no depth estimate | export resolution not verified | ToS (2026-01-21): paid users own outputs and may publicly perform them commercially; the free tier may not | best overall if its resolution holds |

No open generator reaches 8192×4096 natively; every local route needs a
×4 upscale.

**What was measured:**

- **Generation:** 47–53 s per 2048×1024 panorama at a **4.0 GiB peak**
  with sequential offload. Plain model offload OOMed in our own process at
  load, and the resident LLM sharing the card was untouched. Four seeds; seed 11 used.
- **The wrap seam is closed by construction.** Every conv in the UNet and
  VAE pads circularly in x and with zeros in y. The pixel difference across
  the seam equals that of any interior column pair (seed 11: 4.27 against
  4.11; the other three seeds likewise).
- **The zenith is not pinched; it is empty.** The top rows are near-uniform
  haze (horizontal std 2.0), so the dome's centre is featureless rather
  than broken.
- **Upscale:** 12.5 s, 0.66 GiB, wrap-padded 64 px so the seam sees its
  true neighbour.
- **Depth:** DA², seconds, 1.54 GiB, at 1092×546. Ordering is right: rocks
  0.0015–0.003, valley 0.2–0.3, sky 0.4–1.2, in the model's units.
- **Render:** 282 s per 25 s at 4096² (2.7 frames/s). Composite: 2 m 42 s
  for the master (no retime), then about 2 min for the rest.

**Result: 25 s verified.** One panorama, one move: an eased dolly (zero
velocity at both ends) of 4% of the near ring's distance, rising 25%,
camera pitched 20° down toward the front, never rotating. All ten checks
passed: 840 frames, HEVC 42.9 Mbit/s, **brightness peak 42.7 / mean 42.6,
0 flashes**, sync +0.021 ms.

| D2, scaled to 60 s at 4096² | Time |
|---|---|
| panorama + upscale + depth, **per panorama** | ~1.3 min; a 60 s piece needs 2–3 (one move each) |
| parallax render | ≈ 11.3 min |
| `dome.py` master (no retime) + H.265 + stems + checks | ≈ 10 min |
| **total wall clock** | **≈ 25 min** |
| storage | base ≈ 57 GB/60 s (photographic texture is heavy in FFV1), master ≈ 36 GB |

**Effort:** one session, about three hours, most of it dependency and
licence work. Trailer quality is **1–2 weeks**. The Qwen-2512 generator
needs a free card or a slow offload; pole repair, depth-edge handling and
grading are each real work.

**Quality risks, D2 — why it is not good enough alone:**

1. **Most of the dome is sky, and the generator paints a flat sky.**
   "Night" came out as uniform dusk haze. Graded down (exposure 0.15) it is
   a featureless grey-brown bowl, disc mean Y 42.6 **with almost no
   contrast**. That passes the check and is still the washed-out uniform
   field the spec warns against. A procedural sky beats a generated one
   here.
2. **Effective resolution is a quarter of the target.** The 8K equirect is
   a ×4 GAN upscale of 2048 px, about 5.7 px/° of real detail against the
   22.8 px/° the domemaster samples. The GAN adds plausible texture, not
   information.
3. **The rubber sheet.** A single panorama has nothing behind its near
   objects. At 20% travel the near ground and boulders at the springline
   smeared into long streaks; even at 4% they visibly stretch at occlusion
   edges (`d2_crop.png`), while the valley barely moves. Parallax that
   reads comes from exactly the surfaces that tear. That caps D2 to small,
   slow moves: one short move per panorama, and never "walking through"
   anything.
4. **The generator decides the horizon.** Seed 11's horizon sits about 23°
   above the equator, so the landscape rings the whole dome edge. Combined
   with the 20° tilt that suits this panorama, but the layout is the
   generator's call, not the director's.
5. **Licences per model** (below). The clean local stack is the
   lower-quality one; the best quality is hosted and paid.

### The combination, and the recommendation

`…/fulldome/d2/assets/combo_f1650.png` is one still of the obvious
combination. D2's landscape is keyed by DA²'s own distance (far = sky)
over the same moment of D1: rock and valley in the front and sides, D1's
audio-lit nebula and voice curtains overhead. It shows the combination
works, and it shows the one integration it needs. D1's curtains are
anchored to D1's own low ridge, while the tilted landscape rises to about
40° in front, so the rays begin mid-sky. D1 has to take its horizon from
D2's depth, which is a uniform, not a redesign.

**Recommendation: build the trailer on D1, with H3 clips as graded insets,
and add D2 only as a front-sector landscape layer under D1's sky, only if
a viewing says the dome needs a place.** In order:

1. **D1 is the dome.** It is the only route that is natively 4096² with no
   upscale, and the only one where §5 is a property of the code rather than
   of the prompt. It is deterministic, locked to the 5.1 mix spatially and
   in time, the cheapest (≈24 min per 60 s, 8 GPU minutes), and entirely
   our own code with no model licence in the frame. Its weakness,
   abstraction, is exactly what the H3 clips supply.
2. **The H3 clips are the story, as insets.** Grade them to the night
   (step 2 of the build plan's 1344×768 dome-authored render makes this far
   better: dark palette, front-weighted, locked-off).
3. **D2 is a layer, never the dome.** Its sky is the weakest part of it and
   D1's is the strongest part of D1. Its landscape, kept to the front band
   and to tiny moves, gives a place without betting the dome on a
   generated sky or a large parallax move. For the festival version,
   replace the SDXL panorama with Qwen-Image-2512 + qwen-360-diffusion
   (the card free, or offload) or a paid Marble/Skybox tier *after* the
   commercial tier is confirmed in writing.
4. **Not D2 alone.** It fails §4 (a quarter of the resolution) and the
   spirit of §5 (a uniform bright-ish field), and its one lever for motion
   is the lever that tears the picture.

### The combination, built (2026-09-30)

Derek viewed D1 and D2 in a dome simulator (Domeport Pro; no sound) and then
with sound (a seat's-eye browser viewer, `…/fulldome/viewer/`). Three things
he saw changed the plan:

- **In D1, the inset still read as a screen:** "a square video broadcast
  onto the side of a dome". The feathered, 85%-opacity rectangle was route
  A's failure in a smaller size. What the brief said about H3 ("elements,
  not the frame") needs a *cut-out*, not a softened rectangle.
- **D2 read as a place:** "rocks and stuff all around". Being surrounded by
  somewhere outweighed its measured flaws on a laptop screen.
- **The embers read as a constant stream,** because every onset spawned one
  (about 2 a second on this span) and each lived 9 s. He liked it. Decision:
  keep the stream and add a distinct burst on the few biggest hits.

So the trailer is **both**. D2's land sits in front, keyed by DA²'s own
distance (0.355–0.365, a clean skyline) and written as 8-bit RGBA. D1's sky
sits behind it, with its curtains and embers anchored to the land's
skyline (`render_d2` writes it per 0.5° of azimuth). The singer sits
between them as a cut-out, not a rectangle (BiRefNet-matting, MIT, pinned
at `eccde0a8`), graded to night and placed so the land hides the frame's
cut bottom edge. The corrected card comes at the end.

**What `dome.py` gained for this** (tests first; 3136 pass):

- `--foreground`: a fisheye video with alpha, same contract as the base,
  laid over the window and under the credits.
- `--source-alpha`: the window keeps the clip's own alpha.

**Two defects found on the way, both fixed:**

- **`azimuth` never worked.** It was passed to `v360` as `yaw`, but the
  fisheye's optical axis is the zenith, so `yaw` tilts a layer sideways.
  Measured: a window at azimuth 90° landed on the frame's left springline,
  and one at 180° vanished. Turning about the zenith is `roll`. Now
  `roll = front_rotation + azimuth`, pinned by an integration test in which
  bottom, right, left and top all land where the audience expects. Route A
  only ever used 0, so nothing rendered showed it.
- **The credits card's QR code did not decode.** The band name was set at
  190 pt unconditionally and ran over the code's left third. OpenCV reads
  `''` from the old card, and route A and D1 both carried it. Every line is
  now fitted to the space beside the code, and the new card decodes both
  as a still and **from the seat's-eye view of the projected card** —
  the first test of that spec item.

**Result: 80 s verified** ("Deathless" 2–82 s). All ten checks pass:
2490 frames, HEVC 56.6 Mbit/s, brightness peak 34.0 / mean 30.2, 0
flashes, sync +0.021 ms. The bursts land at the 8 strongest onsets (raw
flux about 2× the median onset), spread across the span.

| Stage (doris, 80 s at 4096²) | Wall clock |
|---|---|
| land layer (D2 raymarch + key) | 10 m 20 s |
| sky layer (D1 shader, anchored) | 10 m 17 s |
| cut-out (retime, matte, smooth, grade) | ~1 min (the matte itself 30 s at 1.6 GiB) |
| `dome.py` composite (master 12 m 49 s, H.265 2 m 57 s, the rest 2 m 22 s) | 18 m 08 s |
| **total** | **≈ 40 min** per 80 s, ≈ 30 min per 60 s |

**Open, from the frames:**

- **The rubber sheet shows on the nearest rocks** at the bottom of the
  front: the camera's 80 s move is the same length as D2's 25 s one, but
  held longer. A smaller move, or none at all, is one flag.
- **The matte takes everyone.** Chunk 9 cuts out Jan with Dianne. Right
  for this shot; a shot where a stranger stands behind her would need a
  mask.
- **The apparition jumps between camera setups** (she is right of frame in
  one chunk and left in the next), because the source cuts. A dome-authored
  render, locked off with the figure placed, removes this; the build plan's
  step 2 is still the lever.
- **The land is dark by design (exposure 0.2)** and has not been seen large.

### Source resolution: the first custody night (2026-09-30)

A 4096 domemaster spends about 22.8 px on each degree of sky, so the
question for each layer is how much real detail it brings against that:

| Layer | Before | Shortfall |
|---|---|---|
| D1 sky | drawn at 4096 | none |
| Singer (H3) | 864 px wide, shown 80° across (needs ~1820) | ~2× upscaled |
| D2 land | 2048-wide panorama, ×4 Real-ESRGAN | ~4× short of real detail |

With the card taken into custody (the resident LLM evicted, per the GPU
custody protocol), three things were tried:

- **The render envelope gate refused chunk 7** (175 frames at 1344×768;
  proven here: 141 at 1344×768, 192 at 864×480). `docs/runbook-288-frame-proof.md`
  says the first render past the envelope is *attended* — "one chunk with
  a human at the keyboard, not an overnight render" — so it was **not
  attempted unattended** and the gate was not overridden. Chunk 7 stays at
  864×480 until an attended proof extends `PROVEN_ENVELOPES`.
- **Chunks 8 and 9 rendered at 1344×768** (124 frames, inside the proven
  point): 434 s and 445 s, first attempt each, about 7.3 min per chunk on
  ComfyUI 0.37.4, against CLAUDE.md's 9 m 15 s at 141 frames on the older
  stack (not the same frame count, so not a like-for-like speed claim).
  Same seed, **a different take**: H3 composes differently at a different
  resolution. Chunk 9 also came out **letterboxed** (black bars top and
  bottom). Harmless under a cut-out, since the matte removes them and the
  land hides the straight cut, but it is the model choosing the framing.
- **Placed at 60° instead of 80°, the singer is about 1:1** (60° × 22.8 ≈
  1365 px against a 1344 source). At a 1:1 crop of the composite the new
  shots resolve eyes, mouth and hair strands that the 864 version smears
  (`…/out/combo1344_deathless_2-82_4096/sing_compare.png`). The re-composite
  (same sky and land layers, only the singer changed) passed all ten checks.
- **Negative result: a tiled SDXL img2img detail pass on the land made it
  worse.** It covered the front band (lon −90..90, lat −40..30), with
  1024² tiles at strength 0.3, the same MIT sdxl-360 UNet, the tile's own
  low frequencies kept, and 17 s for 10 tiles. The rocks lost texture,
  and the valley shows **ghosting**, doubled tree shapes where
  neighbouring tiles disagreed (`detail_compare.png`). Independent tiles
  blended in pixel space are not enough. The real version is tile-coupled
  diffusion (MultiDiffusion-style latent averaging) plus a
  detail-preserving conditioning such as a tile ControlNet, each with its
  licence checked. That is a project, not a test. Switching the generator
  to Qwen-Image-2512 + its 360 LoRA does **not** fix resolution either
  (it is 2048×1024 native too). It would only improve the base image.
  The paid 16K route stays open, pending its commercial tier in writing.

Open decisions this adds:

- **D9. Run the attended envelope proof for 175 frames at 1344×768?**
  Decides whether chunk 7, and every other chunk over 141 frames, can be
  rendered at 1344×768. It needs a human at doris's keyboard for one chunk.
- **D10. Which land route: tile-coupled diffusion (days), or a paid 16K
  panorama (money, licence in writing)?** Decides whether the land ever
  reaches the dome's resolution.

### Rights record for these prototypes

Every model and weight file that touched a D1 or D2 pixel:

| Component | Source (revision) | Licence | Commercial / exhibition |
|---|---|---|---|
| D1 shader, features, comfort, renderer | this project's run assets | ours | yes |
| `moderngl` | PyPI | MIT | yes (a library; no output rights question) |
| SDXL 360 UNet | `ProGamerGov/sdxl-360-diffusion` @ `35658524`, UNet sha256 `39953ed8…adfca` | MIT | yes |
| SDXL base (text encoders, tokenizers, scheduler) | `stabilityai/stable-diffusion-xl-base-1.0` @ `46216598` | CreativeML Open RAIL++-M | yes, subject to its use-based restrictions |
| SDXL VAE fp16 fix | `madebyollin/sdxl-vae-fp16-fix` @ `207b116d` | **not verified** — record before a public showing | open |
| Real-ESRGAN x4plus | `xinntao/Real-ESRGAN` release v0.1.0, sha256 `4fa0d389…d682f1` | BSD-3-Clause (repo); separate weights licence not verified | yes (repo licence) |
| `spandrel` (ESRGAN loader) | PyPI | MIT | yes |
| DA² | `haodongli/DA-2` @ `0d55ccb5`, code `EnVision-Research/DA-2` @ `d6598385` | Apache-2.0 | yes |
| BiRefNet-matting (the singer cut-out) | `ZhengPeng7/BiRefNet-matting` @ `eccde0a8` (runs repo code; pinned) | MIT | yes |
| H3 inset | MiniMax H3 | **still open** (the Rights section above) | open |

### Decisions this adds

- **D6. D1 as the trailer's dome?** Decides whether the next work is art
  direction on the shader (days) or more route evaluation. Recommended:
  yes, after one viewing of `d1_deathless_2-62_4096` in a dome simulator or
  headset.
- **D7. Is a place needed?** Decides whether D2 is built as a landscape
  layer (1–2 weeks, a generator choice, the horizon handshake) or dropped.
  Answer by viewing D1 first.
- **D8. Which panorama source, if D2?** SDXL-360 (clean, here, weakest),
  Qwen-2512 + LoRA (clean, needs the card), or Marble/Skybox (best, paid,
  tier to confirm in writing). Decides the licence line in the rights
  record.

## Decisions

One question each; what the answer decides.

- **D1. Is a window enough for the trailer?** Decides whether step 7 (route
  C) is on the plan at all. Answer after step 1, not before.
- **D2. `mci` or `dup`?** Decides the retime for every dome render. Answer
  by viewing both; the module carries both.
- **D3. Render the trailer's span again at 1344×768, dome-authored (step 2),
  or ship the trailer from the music-video render?** Decides ~2 GPU hours
  and a shot-plan authoring session against a first viewing that costs
  nothing.
- **D4. Which venue first?** Decides `front_rotation`, `elevation`, the
  audio bit depth and whether 7.1 is needed. Nothing is hard-coded, so this
  can wait for a rider.
- **D5. Who records the model licences?** Decides whether §9 is closed
  before the first public showing.
