# "Deathless" full-render corpus (issues #76, #77)

Measured 2026-08-16, entirely offline against mp4s already on disk in
`~/mvm-runs/deathless/output/chunks/` -- no GPU, no ComfyUI, no doris. This
document is the data file issue #76 asked for ("the corpus and scores go in
`docs/`, so the next revision starts from data instead of from a fresh
guess") and the measured basis for issue #77's luminance floor
(`music_video_maker/luminance.py`). Numbers only; no frame, still, or clip
from the render is reproduced here, and no full lyric or shot line is quoted
-- only the short phrases the argument below actually needs, matching what
issue #76 itself already published.

## Which files this corpus is actually built from

`run.toml` in the run directory names `shot_plan = "shot_plan_v4.toml"`.
The run directory also contains an *older* `shot_plan.toml` left over from
an earlier authoring iteration -- same chunk anchors, different (earlier)
shot/camera text. Using the wrong one would have silently invalidated every
keyword-vs-face-presence number below: for example old `shot_plan.toml`'s
chunk 7 camera reads `"tracking backwards ahead of her on the switchback"`,
while `shot_plan_v4.toml`'s (the one actually rendered) reads
`"medium close, travelling with her, her closed fist and her face held in
the same frame"` -- the exact line issue #76 quotes. Verified, not assumed:

- `~/mvm-runs/deathless/output/chunks/run_state.json` (`schema_version: 2`)
  is authoritative for the chunk-id -> mp4 mapping and each chunk's actual
  rendered `start`/`end` (its `ChunkFingerprint`). It reports 80 chunks,
  ids 0-79 contiguous, `chunk_NNNN.mp4` filenames matching chunk id by
  simple zero-padding (`chunk_0043.mp4` is chunk 43) -- an identity mapping,
  but confirmed rather than assumed by cross-checking every span below.
- `shot_plan_v4.toml` was loaded with the repo's own
  `music_video_maker.shot_plan.load_shot_plan`, and **every one of its 80
  entries'** `start` was diffed against `run_state.json`'s fingerprint
  `start` for the same `chunk_id` (not a 3-chunk sample). Max discrepancy
  across all 80: **0.00033 s** -- float rounding, not drift. `shot_plan.toml`
  (the older file) was not used for anything in this document.
- Issue #77's own numbers were reproduced as an independent check: chunk 43
  spans 277.417-282.583 s (4:37.42-4:42.58), matching the issue's report
  exactly, and its shot/camera text below matches the issue's quotes
  verbatim.

## Instrument validation (do this before trusting anything below)

Face presence: 12 evenly-spaced frames per chunk, detected with
`music_video_maker.faces.detect_faces` (YuNet, the #47 gate's detector, at
its default 0.9 score threshold) -- the same instrument issue #76 used, so
the numbers are comparable to every prior measurement on this project.

| check | recorded | measured here | 
|---|---|---|
| chunk 7 face presence | ~80-83% | **83.3%** (10/12 frames) |
| chunk 35 face presence | 100% | **100%** (12/12 frames) |

Both reproduce. The instrument is trusted for everything below.

Luminance: mean of all decoded frames in a chunk's first quarter and its
last quarter (`cv2.VideoCapture`, BGR->gray, full resolution -- this
analysis script, not the cheaper 32x18-probe method
`music_video_maker.luminance` ships for the always-on assembly check).
Cross-checked against a second, independent method (`ffmpeg -ss <t> -frames:v
1 -s 32x18 -pix_fmt gray`, the exact method the shipped check uses) at
chunk 43: 76.5/14.9 (this script) vs. 77.7/15.9 (ffmpeg probe) -- agrees to
within about 1 Y, comfortably inside the >5 Y margin the floor is set with.

## Part 1: luminance -- the full 80-chunk distribution

Method note: "first quarter"/"last quarter" mean all decoded frames in that
quarter of the chunk (not the sparse 3-sample probe the shipped check uses).

- **Range:** Y 14.9 (chunk 43, darkest ending) to Y 124.3 (chunk 45).
- **Drift** (end - start): -61.6 to +54.5. **36 of 80 chunks** (45%) fall
  outside a +/-15 Y drift band. This reproduces issue #77's finding almost
  exactly (36 chunks, -61 to +55) and is why this document does **not**
  publish or recommend a drift-based check: "Deathless" is a night ending in
  dawn, and a chunk legitimately brightening or darkening by 30+ Y across its
  own span is the song's content on both halves of the arc (see the
  brightest chunks below, several of which are large *positive* swings
  during the dawn passage, not defects).
- **The gap a floor can sit in:** the darkest chunk ending (43, Y 14.9) and
  the *next* darkest chunk ending anywhere in the render (18, Y 36.4) are
  separated by **21.5 Y**, with nothing else in the corpus anywhere near
  either value. `music_video_maker.luminance.DEFAULT_DARK_FLOOR = 25.0`
  sits in the middle of that gap: 10.1 Y above chunk 43, 11.4 Y below chunk
  18 -- comfortable margin on both sides, not a value picked to land next to
  either edge.

5 darkest chunk endings:

| chunk | end Y | voiced? | span |
|---:|---:|:---:|---|
| 43 | 14.9 | voiced | 4:37.42-4:42.58 |
| 18 | 36.4 | voiced | 2:02.75-2:07.92 |
| 50 | 36.9 | voiced | 5:16.42-5:24.42 |
| 55 | 39.3 | instr | 5:55.71-6:03.00 |
| 25 | 45.1 | voiced | 2:41.04-2:49.04 |

5 brightest chunk endings (all late in the song, during the dawn passage --
positive drift, not a failure mode):

| chunk | end Y | voiced? |
|---:|---:|:---:|
| 40 | 99.9 | voiced |
| 68 | 100.3 | voiced |
| 59 | 105.4 | voiced |
| 20 | 105.4 | voiced |
| 45 | 124.3 | instr |

Chunk 43 is the only chunk in the entire render below Y 45 that is *also*
the chunk a viewer flagged; every other dark-ish chunk ending (18, 50, 55,
25 at 36-45 Y) is a legitimate low-light shot nobody has reported a problem
with. That is the evidence the shipped check acts on an absolute floor
rather than "how dark relative to this chunk's own start."

## Part 2: the full 80-chunk corpus

`face %`: percentage of 12 sampled frames with >=1 detected face (YuNet,
threshold 0.9). `start Y`/`end Y`: mean luminance, first/last quarter.
`drift`: end - start, informational only (see above -- never used to flag
anything).

| chunk | span | voiced | face % | start Y | end Y | drift |
|---:|---|:---:|---:|---:|---:|---:|
| 0 | 0:00.00-0:08.00 | instr | 0 | 52.7 | 66.3 | +13.6 |
| 1 | 0:08.00-0:16.00 | instr | 0 | 90.6 | 61.5 | -29.1 |
| 2 | 0:16.00-0:24.00 | instr | 0 | 82.2 | 45.8 | -36.3 |
| 3 | 0:24.00-0:32.00 | instr | 0 | 64.6 | 58.8 | -5.8 |
| 4 | 0:32.00-0:39.29 | instr | 0 | 38.5 | 93.1 | +54.5 |
| 5 | 0:39.29-0:47.29 | instr | 0 | 51.8 | 57.5 | +5.8 |
| 6 | 0:47.29-0:54.58 | instr | 0 | 46.1 | 99.5 | +53.4 |
| 7 | 0:54.58-1:01.88 | voiced | 83 | 60.1 | 98.3 | +38.2 |
| 8 | 1:01.88-1:07.04 | voiced | 8 | 64.6 | 61.0 | -3.6 |
| 9 | 1:07.04-1:12.21 | voiced | 0 | 54.7 | 61.3 | +6.6 |
| 10 | 1:12.21-1:18.79 | instr | 0 | 37.7 | 65.2 | +27.5 |
| 11 | 1:18.79-1:25.38 | instr | 0 | 51.2 | 74.2 | +23.0 |
| 12 | 1:25.38-1:31.96 | instr | 0 | 40.5 | 45.5 | +5.1 |
| 13 | 1:31.96-1:38.54 | instr | 0 | 58.3 | 58.9 | +0.6 |
| 14 | 1:38.54-1:45.12 | instr | 0 | 38.6 | 71.0 | +32.4 |
| 15 | 1:45.12-1:50.29 | voiced | 42 | 70.8 | 92.6 | +21.8 |
| 16 | 1:50.29-1:55.46 | voiced | 58 | 72.5 | 55.4 | -17.1 |
| 17 | 1:55.46-2:02.75 | voiced | 58 | 47.2 | 59.6 | +12.4 |
| 18 | 2:02.75-2:07.92 | voiced | 8 | 39.5 | 36.4 | -3.1 |
| 19 | 2:07.92-2:13.08 | voiced | 67 | 81.2 | 75.6 | -5.7 |
| 20 | 2:13.08-2:19.67 | voiced | 0 | 79.0 | 105.4 | +26.5 |
| 21 | 2:19.67-2:24.83 | voiced | 8 | 55.2 | 52.0 | -3.2 |
| 22 | 2:24.83-2:30.00 | voiced | 33 | 71.0 | 94.9 | +23.9 |
| 23 | 2:30.00-2:35.88 | instr | 0 | 79.7 | 77.5 | -2.1 |
| 24 | 2:35.88-2:41.04 | voiced | 17 | 61.6 | 93.7 | +32.1 |
| 25 | 2:41.04-2:49.04 | voiced | 0 | 73.4 | 45.1 | -28.3 |
| 26 | 2:49.04-2:55.62 | voiced | 0 | 43.6 | 50.9 | +7.4 |
| 27 | 2:55.62-3:02.21 | voiced | 42 | 58.3 | 78.1 | +19.8 |
| 28 | 3:02.21-3:07.38 | voiced | 8 | 63.7 | 52.5 | -11.2 |
| 29 | 3:07.38-3:13.96 | instr | 0 | 51.4 | 46.2 | -5.2 |
| 30 | 3:13.96-3:20.54 | instr | 0 | 59.1 | 57.3 | -1.8 |
| 31 | 3:20.54-3:27.83 | instr | 0 | 59.4 | 58.1 | -1.3 |
| 32 | 3:27.83-3:34.42 | instr | 0 | 79.6 | 88.3 | +8.8 |
| 33 | 3:34.42-3:41.71 | instr | 0 | 48.0 | 72.0 | +23.9 |
| 34 | 3:41.71-3:48.29 | instr | 0 | 76.9 | 75.4 | -1.5 |
| 35 | 3:48.29-3:54.17 | voiced | 100 | 80.7 | 92.0 | +11.3 |
| 36 | 3:54.17-3:59.33 | voiced | 75 | 63.0 | 57.2 | -5.8 |
| 37 | 3:59.33-4:07.33 | voiced | 42 | 76.4 | 78.6 | +2.2 |
| 38 | 4:07.33-4:13.21 | voiced | 75 | 74.5 | 93.2 | +18.7 |
| 39 | 4:13.21-4:19.79 | voiced | 100 | 67.9 | 64.2 | -3.7 |
| 40 | 4:19.79-4:27.08 | voiced | 25 | 73.4 | 99.9 | +26.5 |
| 41 | 4:27.08-4:32.25 | voiced | 83 | 84.4 | 61.3 | -23.1 |
| 42 | 4:32.25-4:37.42 | voiced | 100 | 66.0 | 66.1 | +0.1 |
| 43 | 4:37.42-4:42.58 | voiced | 33 | 76.5 | 14.9 | -61.6 |
| 44 | 4:42.58-4:47.75 | voiced | 67 | 85.1 | 76.7 | -8.4 |
| 45 | 4:47.75-4:52.92 | instr | 0 | 117.5 | 124.3 | +6.8 |
| 46 | 4:52.92-4:58.08 | instr | 0 | 80.9 | 71.5 | -9.3 |
| 47 | 4:58.08-5:03.25 | voiced | 67 | 74.9 | 84.4 | +9.4 |
| 48 | 5:03.25-5:08.42 | voiced | 100 | 72.0 | 66.4 | -5.6 |
| 49 | 5:08.42-5:16.42 | voiced | 92 | 69.9 | 82.2 | +12.3 |
| 50 | 5:16.42-5:24.42 | voiced | 83 | 68.6 | 36.9 | -31.6 |
| 51 | 5:24.42-5:32.42 | instr | 0 | 93.6 | 56.4 | -37.2 |
| 52 | 5:32.42-5:40.42 | instr | 0 | 56.3 | 49.4 | -6.9 |
| 53 | 5:40.42-5:47.71 | instr | 0 | 39.0 | 48.1 | +9.1 |
| 54 | 5:47.71-5:55.71 | instr | 0 | 70.0 | 53.7 | -16.3 |
| 55 | 5:55.71-6:03.00 | instr | 0 | 41.9 | 39.3 | -2.6 |
| 56 | 6:03.00-6:11.00 | instr | 0 | 68.9 | 72.9 | +4.0 |
| 57 | 6:11.00-6:18.29 | instr | 0 | 47.1 | 67.7 | +20.6 |
| 58 | 6:18.29-6:26.29 | voiced | 42 | 58.0 | 45.2 | -12.7 |
| 59 | 6:26.29-6:31.46 | voiced | 25 | 73.4 | 105.4 | +32.0 |
| 60 | 6:31.46-6:38.75 | instr | 0 | 44.6 | 70.8 | +26.2 |
| 61 | 6:38.75-6:46.04 | instr | 0 | 41.7 | 90.4 | +48.7 |
| 62 | 6:46.04-6:52.62 | instr | 0 | 69.7 | 62.4 | -7.2 |
| 63 | 6:52.62-6:57.79 | voiced | 75 | 74.3 | 76.8 | +2.5 |
| 64 | 6:57.79-7:02.96 | voiced | 92 | 47.1 | 60.0 | +12.9 |
| 65 | 7:02.96-7:08.83 | voiced | 25 | 109.3 | 63.1 | -46.2 |
| 66 | 7:08.83-7:14.00 | instr | 0 | 82.5 | 73.7 | -8.8 |
| 67 | 7:14.00-7:19.17 | voiced | 0 | 78.2 | 84.0 | +5.8 |
| 68 | 7:19.17-7:26.46 | voiced | 92 | 78.9 | 100.3 | +21.4 |
| 69 | 7:26.46-7:31.62 | voiced | 100 | 78.8 | 57.8 | -21.0 |
| 70 | 7:31.62-7:38.92 | instr | 17 | 51.4 | 71.7 | +20.4 |
| 71 | 7:38.92-7:46.21 | instr | 17 | 54.4 | 66.9 | +12.5 |
| 72 | 7:46.21-7:54.21 | instr | 42 | 93.2 | 62.9 | -30.3 |
| 73 | 7:54.21-7:59.38 | voiced | 83 | 65.3 | 68.8 | +3.6 |
| 74 | 7:59.38-8:05.25 | voiced | 75 | 79.9 | 60.9 | -19.0 |
| 75 | 8:05.25-8:11.83 | instr | 0 | 57.4 | 57.1 | -0.4 |
| 76 | 8:11.83-8:18.42 | instr | 0 | 38.3 | 80.5 | +42.2 |
| 77 | 8:18.42-8:23.58 | voiced | 100 | 40.4 | 81.6 | +41.2 |
| 78 | 8:23.58-8:28.75 | instr | 0 | 59.2 | 88.7 | +29.5 |
| 79 | 8:28.75-8:33.92 | instr | 0 | 86.0 | 85.8 | -0.3 |

Totals: **41 voiced, 39 instrumental** (matches issue #76's "41 voiced
chunks" exactly). Voiced-corpus mean face presence: **53.3%** (issue #76
reported 53.3%; this independent run of the same instrument against the
correctly-identified `shot_plan_v4.toml` measured 53.25%). A handful of
instrumental chunks (70-72) show nonzero face % because a cast member is
staged on screen (`present`) while not singing -- expected and not part of
the voiced-chunk analysis below.

## Part 3: `lint_voiced_framing` keyword-set separations (issue #76)

Scored with the **same sequential/exclusive precedence
`shot_plan.lint_voiced_framing` itself uses** (gaze-away checked against
`shot` first; only if it doesn't fire is behind-camera checked against
`camera`; only if neither fires is foot-level checked) -- not each keyword
set in isolation. This matters: chunk 15's camera text contains both a gaze
phrase (`"turns back"`, in `shot`) and a behind-camera phrase (`"behind her
shoulder"`, in `camera`). The real lint attributes it to gaze-away only
(`continue`s before reaching the behind-camera check), so an independent,
non-exclusive scoring double-counts it -- which is exactly the 1-chunk
discrepancy between an isolated re-scoring and issue #76's own reported n=2
for `_BEHIND_CAMERA_KEYWORDS`. Reproducing the issue's own table exactly
required matching its precedence, not just its keyword lists.

Voiced corpus mean face presence: 53.3%. "vs rest" = the flagged group's
average divided by the average of the **other** voiced chunks (matches issue
#76's own denominator, not the whole-corpus mean).

| keyword set | n | avg face % | vs rest | worst | best | chunks |
|---|---:|---:|---:|---:|---:|---|
| `_GAZE_AWAY_KEYWORDS` (shipped, on `shot`) | 9 | 40.7 | 0.72x | 0 | 100 | 9, 15, 20, 26, 42, 59, 64, 65, 73 |
| `_BEHIND_CAMERA_KEYWORDS` (shipped, on `camera`) | 2 | 41.7 | 0.77x | 0 | 83 | 7, 25 |
| `_FOOT_LEVEL_KEYWORDS` (shipped, on `camera`+`shot`) | 3 | 52.8 | 0.99x | 33 | 92 | 22, 43, 68 |
| close framing present, no warning | 27 | 58.3 | 1.34x | 0 | 100 | (27 chunks) |
| **`in profile`** (candidate, on `camera`) | 5 | 23.3 | 0.41x | 8 | 42 | 18, 21, 37, 43, 59 |

This reproduces issue #76's own table almost exactly (their reported
values: gaze 40.7%/0.72x; behind 41.6%/0.77x; foot 52.8%/0.99x; `in profile`
23.3%/0.41x). `_FOOT_LEVEL_KEYWORDS` is statistically indistinguishable from
the rest of the corpus (0.99x). `in profile` is the only tested set with
**no counter-example** -- every one of its 5 chunks lands below the corpus
mean, worst case 8%.

### Per-chunk detail, gaze-away (`_GAZE_AWAY_KEYWORDS`)

| chunk | face % | keyword | camera |
|---:|---:|---|---|
| 9 | 0 | "looks back" | "medium close, slightly above her, her face centre..." |
| 20 | 0 | "staring up" | "medium close, low, angled up past her shielding hand to her face" |
| 26 | 0 | "looks up at" | "close, low, tilted up onto her face" |
| 59 | 25 | "looks out" | "medium close, her face in profile against the flat horizon" |
| 65 | 25 | "gaze fixed" | "medium close past his shoulder onto her face" |
| 15 | 42 | "turns back" | "close on her face in three-quarter..." |
| 73 | 83 | "gaze lifts" | "close on his face, low, from the horizon side" |
| 64 | 92 | "lifts her gaze" | "close on her face, low, the light band behind her" |
| 42 | 100 | "gaze drops" | "close, straight on, static" |

Three of the worst four (9, 20, 26) all say "her face" in `camera` yet still
scored 0% -- the camera clause naming the face is not sufficient when the
gaze verb settles the head away from the lens. The best three (73, 64, 42)
also all name the face directly. See the counter-example discussion below;
naming the face is not on its own predictive across the full 41.

### Per-chunk detail, `in profile` (candidate)

| chunk | face % | camera |
|---:|---:|---|
| 18 | 8 | "medium close in profile, firelight raking one side of her face" |
| 21 | 8 | "close on her face in profile against the pre-dawn band" |
| 59 | 25 | "medium close, her face in profile against the flat horizon" |
| 43 | 33 | "medium close in profile against the dimming plain" |
| 37 | 42 | "medium close in profile, holding steady" |

## Part 4: the "does the camera clause name the face" hypothesis

Issue #76 proposed this as a possible replacement for the gaze-verb
heuristic, from two counter-examples on the shipped `_BEHIND_CAMERA_KEYWORDS`
set: chunk 7 (`"travelling with her, her closed fist and her face held in
the same frame"` -> 83%) vs. chunk 25 (`"travelling with her, the sails
passing through the background"` -> 0%) -- same phrase, opposite outcome,
and only one of the two names the face.

Tested against all 41 voiced chunks with a literal match for `"face"` / `"her
face"` / `"his face"` in the `camera` field:

| group | n | avg face % | vs rest |
|---|---:|---:|---:|
| camera clause names the face | 30 | 50.8 | 0.85x |
| camera clause does not name the face | 11 | 59.8 | 1.18x |

**This does not survive contact with the full 41** -- if anything the effect
runs backward from the hypothesis (naming the face correlates with slightly
*lower* face presence, not higher). Chunk 35 (100% face) never says "face"
at all (`"medium close on him from slightly below, the mushrooms at the very
bottom edge of frame"`); chunks 9, 20 and 26 (0%, see above) all explicitly
say "her face." A literal keyword match on the word "face" is not the right
operationalization of the chunk-7-vs-25 distinction -- it is closer to
"does the camera clause describe the frame in a way that keeps the head
inside it" (chunk 7's "held in the same frame" vs. chunk 25's "passing
through the background," which is about the *sails*, not her), which is a
judgment call a keyword list cannot make. **Recorded as tested and rejected
at n=41**, per issue #60's discipline: this was a hypothesis from n=2 and it
does not generalize as literal text matching.

## Part 5: evidence for the AUTHORING-lint half of issue #77

This is **not** an implementation -- shot_plan.py is out of scope for this
work (see the repo's file-ownership rules for this task). It is the measured
keyword evidence to hand to whoever writes "a shot line asking for a large
lighting transition on a voiced chunk is a warning."

Searched every one of all 80 chunks' `shot` + `camera` text (not just the
41 voiced ones, so a candidate's behavior on instrumental chunks is visible
too) for each candidate phrase:

| candidate | hits | chunks | verdict |
|---|---:|---|---|
| `"goes dark"` | 1 | 43 (drift -61.6, end Y 14.9) | **only real signal found**, but n=1 |
| `"dims"` | 1 | 43 (same) | same as above, n=1 |
| `"dimming"` | 1 | 43 (same) | same as above, n=1 |
| `"fading"` | 2 | 43 (drift -61.6); **48 (drift -5.6, end Y 66.4)** | **EXCLUDE** -- chunk 48 is a normal, non-dark chunk; the word alone does not discriminate |
| `"dawn breaks"` (literal) | 0 | -- | untested here (never appears verbatim); see "dawn" below |
| `"fades to black"` / `"fade to black"` | 0 | -- | untested, no evidence either way |
| `"the light dies"` / `"light dies"` | 0 | -- | untested, no evidence either way |
| `"darkens"` / `"darkening"` | 0 | -- | untested, no evidence either way |
| `"swallowed"` / `"blackout"` / `"goes out"` / `"burns out"` | 0 | -- | untested, no evidence either way |

**`"dawn"` specifically (broader than the literal "dawn breaks" phrase)**
appears in 9 chunks (0, 2, 21, 50, 58, 61, 64, 73, 76) with drift ranging
from -36.3 to +48.7 and **none** ending dark (all >= Y 36.9). Two of its
three largest-magnitude hits (chunk 61 at +48.7, chunk 76 at +42.2) are
large *brightening* swings, not darkening ones -- because in this song's
narrative "dawn" is the resolution, and the video gets brighter as it
approaches, not darker. **A "dawn breaks" keyword would fire in the wrong
direction on this corpus far more often than the right one; exclude it**,
or at minimum do not treat "dawn" language as evidence of a problematic
transition without also checking the sign of the luminance change.

**Bottom line for the next agent:** the one real, unambiguous case in this
render (`"goes dark"`/`"dims"`/`"dimming"`, all three matching only chunk
43) is a single data point -- real, but n=1, exactly the shape issue #60
warns against generalizing from. `"fading"` is a measured, excluded false
positive within this same corpus (chunk 48). Every other candidate phrase
from the issue text is simply absent from this song's 80 shot lines, so
this corpus offers no evidence for or against it -- test them against a
second song before shipping any of them as a keyword.

## Part 6: what actually shipped (issue #76 resolution)

Everything below was implemented in `music_video_maker/shot_plan.py`, with
per-word evidence re-derived directly against `shot_plan_v4.toml` via
`load_shot_plan` (not just the aggregate table in Part 3), so every kept or
excluded word below has its own chunk number, not just its group's average.

### `in profile` -- shipped

Superseding the earlier n=3 decision to leave it out (8%/0%/45%, "a real
risk but not a reliable one"). Unchanged from Part 3: **n=5, avg 23.3%,
0.41x vs rest, worst 8%, best 42%**, no counter-example. Chunks 18, 21, 37,
43, 59. Checked ahead of the close-framing check in the lint, since all 5
real occurrences also contain "close" or "medium close".

### `_FOOT_LEVEL_KEYWORDS` -- retired

Re-derived per-word (all 41 voiced chunks, both `camera` and `shot`, not
just the 3 chunks that happened to fire first under the lint's precedence):

| word | n | chunks (face %) | avg | verdict |
|---|---:|---|---:|---|
| "the ground" | 2 | 43 (33%), 68 (92%) | 62.5% | **EXCLUDE** -- same phrase, 59-point spread, no separation |
| "underfoot" | 1 | 22 (33%) | 33.0% | single data point, thin evidence |
| all other 10 words (boots, boot, feet, ankles, hem, at his/her feet, soil, the floor, knees) | 0 | -- | -- | never occur in this song's 41 voiced lines |

Combined shipped-set score: 0.99x (Part 3), indistinguishable from noise.
Retired outright rather than re-derived down to "underfoot" alone: one
word with one data point is the same thin evidence #60 warns against, and
this project's own standing rule is that a weak lint costs nothing at
runtime and a great deal the moment someone rewrites a correct shot to
please it. The observation that motivated the set -- a shot framed on a
needle in a raised fist kept the face 80% of the time, while a line reading
"pushing up between his boots" rendered legs, boots and mushrooms with no
face anywhere -- was measured on an earlier render, not this corpus, and
remains real; it is the keyword list built from it that failed to
generalise here.

### `_GAZE_AWAY_KEYWORDS` -- re-scored, 9 words down to 6

Per-word re-derivation (all 41 voiced chunks' `shot` text, not just the 9
that fired first under the lint's own precedence -- which surfaces the
co-occurrences the aggregate table can't show):

| word | n | chunk (face %) | verdict |
|---|---:|---|---|
| "looks back" | 1 | 9 (0%) | **KEEP** -- below mean |
| "looks up at" | 1 | 26 (0%) | **KEEP** -- below mean |
| "staring up" | 1 | 20 (0%) | **KEEP** -- below mean |
| "looks out" | 1 | 59 (25%) | **KEEP** -- below mean |
| "gaze fixed" | 1 | 65 (25%) | **KEEP** -- below mean |
| "turns back" | 1 | 15 (42%) | **KEEP** -- below mean (weakest survivor) |
| "gaze drops" | 1 | 42 (100%) | **EXCLUDE** -- above mean, false positive |
| "gaze lifts" | 1 | 73 (83%) | **EXCLUDE** -- above mean, false positive |
| "lifts her gaze" | 1 | 64 (92%) | **EXCLUDE** -- above mean, false positive |
| "keeps his gaze" | 1 | 65 (25%) | **EXCLUDE** -- redundant, only co-occurs with "gaze fixed" on the same chunk (65); no independent evidence |
| "staring down" | 1 | 65 (25%) | **EXCLUDE** -- redundant, same reason |
| "gaze drifts", "keeps her gaze", "lifts his gaze", "looking back", "looking down", "looking out", "looks down", "looks over", "stares down", "stares out", "staring out", "turns away" | 0 | -- | **EXCLUDE** -- never occur in this song's 41 voiced shot lines; #60's discipline applies to absence too, so an unmeasured word does not get to ship on the strength of the ones that were |

The 3 counter-examples above are the same one the issue calls out from the
other side: the gaze set's 3 best-scoring chunks (42 at 100%, 64 at 92%, 73
at 83%) all pair a gaze verb with a camera clause that separately names the
face ("close, straight on, static"; "close on her face, low, the light band
behind her"; "close on his face, low, from the horizon side") -- consistent
with Part 4's finding that naming the face is not a reliable predictor on
its own, but a gaze verb evidently loses less when something else in the
entry is also pinning the face in frame.

Final 6-word set, scored the same way as Part 3's table: **n=6 (chunks 9,
15, 20, 26, 59, 65), avg 15.3%, ~0.26x vs the rest of the voiced corpus**
(mean of the other 35 chunks: 59.8%) -- a materially cleaner separation than
the original 9-word set's 0.72x.

### `_BEHIND_CAMERA_KEYWORDS` -- flagged, not changed

Out of scope for what issue #76 asked (its "Do" list covers gaze,
foot-level, `in profile`, and the named-face hypothesis only), so the code
is unchanged. Recorded here so it isn't silently rediscovered later: this
set's two hits on the full corpus are **both the same phrase**, "travelling
with" -- chunk 7 (83% face, camera `"...travelling with her, her closed
fist and her face held in the same frame"`) and chunk 25 (0% face, camera
`"...travelling with her, the sails passing through the background"`). Same
word, opposite outcomes, exactly the shape that justified retiring
`_FOOT_LEVEL_KEYWORDS` above. The other 6 words in the set ("tracking
behind", "from behind", "following her", "following him", "behind her
shoulder", "over her shoulder from behind") never occur in this corpus --
untested, not falsified. **Candidate for a follow-up issue**, not fixed
here.

### The named-face hypothesis -- tested and rejected (issue #76 item 4)

Already covered in full in Part 4 above: a literal `"face"` match in
`camera` averages 50.8% (n=30) against 59.8% for entries that don't name it
(n=11) -- backwards from the hypothesis. Do not re-propose "does the camera
name the face" as a keyword; the best chunk in the entire corpus (35, 100%
face) never says "face" at all.

### Issue #77's lighting-transition authoring lint -- not shipped

Confirmed here rather than only in Part 5: `music_video_maker/shot_plan.py`
gained no lint for a voiced chunk requesting a large lighting transition.
The evidence in Part 5 does not support one on this corpus -- "goes
dark"/"dims"/"dimming" match only chunk 43 (n=1), "fading" is a measured
false positive (chunk 48, a normal chunk), and "dawn" trends toward
*brightening* on this song (its two largest-magnitude hits are +48.7 and
+42.2 Y), so a "dawn breaks" keyword would fire backwards more often than
not. Nothing in `shot_plan.py` implements this; test a second song before
shipping it.

### Net effect: how often the lint fires now

Not hand-tallied -- run directly through the real, updated
`lint_voiced_framing`, loading `shot_plan_v4.toml` with `load_shot_plan` and
feeding it 80 synthetic `AudioChunk`s carrying only the voiced/instrumental
flag from Part 2 (so the sequential/exclusive precedence the function itself
uses -- gaze-away checked before behind-camera, before `in profile`, before
close/wide-framing -- is exactly what produced this table, not a
reconstruction of it):

| chunk | old (9 gaze + 2 behind + 3 foot = 14) | new (6 gaze + `in profile`; behind unchanged, foot retired) | face % |
|---:|:---:|:---:|---:|
| 7 | behind-camera | **behind-camera (unchanged, out of scope)** | 83 |
| 9 | gaze-away | gaze-away (kept, "looks back") | 0 |
| 15 | gaze-away | gaze-away (kept, "turns back") | 42 |
| 18 | -- | **in profile (new)** | 8 |
| 20 | gaze-away | gaze-away (kept, "staring up") | 0 |
| 21 | -- | **in profile (new)** | 8 |
| 22 | foot-level | -- (retired) | 33 |
| 25 | behind-camera | behind-camera (unchanged, out of scope) | 0 |
| 26 | gaze-away | gaze-away (kept, "looks up at") | 0 |
| 37 | -- | **in profile (new)** | 42 |
| 42 | gaze-away | -- (excluded, false positive) | 100 |
| 43 | foot-level | **in profile (new)** -- same chunk, "the ground" no longer fires, "in profile" does | 33 |
| 59 | gaze-away | gaze-away (kept, "looks out") | 25 |
| 64 | gaze-away | -- (excluded, false positive) | 92 |
| 65 | gaze-away | gaze-away (kept, "gaze fixed") | 25 |
| 68 | foot-level | -- (retired) | 92 |
| 73 | gaze-away | -- (excluded, false positive) | 83 |

**12 of 41 voiced chunks fire under the revised lint** (7, 9, 15, 18, 20,
21, 25, 26, 37, 43, 59, 65), down from 14. Of those 12, **11 are true
positives** by this corpus's own face-presence numbers -- every fired chunk
except 7 scores at or below the 53.3% corpus mean, several well below it (0%
on four of them). **The one false positive left is chunk 7 (83% face),
fired by the untouched `_BEHIND_CAMERA_KEYWORDS` "travelling with"** -- the
same keyword flagged above as a candidate for a future pass and left alone
because it was out of this issue's scope. It is the single fired chunk this
revision did not fix. The three chunks the old gaze set fired on wrongly
(42 at 100%, 64 at 92%, 73 at 83%) no longer fire at all. Retiring
`_FOOT_LEVEL_KEYWORDS` removes both of its former hits, and they cut in
opposite directions: chunk 68 (92% face) was a false positive, gone for
good; chunk 22 (33% face) was a genuine true positive that the revised lint
no longer catches -- the honest cost of retiring a keyword set instead of
re-deriving it down to a single thin word. Net: 2 fewer warnings overall,
3 fewer false positives introduced by the gaze re-score, 1 true positive
lost to the foot-level retirement, and 1 known false positive (chunk 7)
remaining and explicitly flagged rather than silently left in.

## Reproducing this corpus

1. `run_state.json` for the chunk-id -> mp4 -> rendered-span mapping
   (ground truth; never assume identity without checking it).
2. `shot_plan_v4.toml` (check `run.toml`'s `shot_plan =` key on any future
   run -- do not assume the file literally named `shot_plan.toml` is the one
   that rendered), loaded via `music_video_maker.shot_plan.load_shot_plan`.
3. Face presence: 12 evenly-spaced frames per chunk through
   `music_video_maker.faces.detect_faces`.
4. Luminance: mean grayscale of all frames in the first/last quarter
   (`cv2`) for this document's numbers; `music_video_maker.luminance`'s
   sparser ffmpeg-based probe for the shipped, always-on assembly check --
   cross-checked to agree within ~1 Y at chunk 43.

No frame, still, clip, lyric line, or full shot line from the render is
reproduced in this repository; every number above is derived from run-local
data that stays in `~/mvm-runs/deathless/` and is not committed.
