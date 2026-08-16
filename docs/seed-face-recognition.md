# Seed-frame face recognition (issue #49)

`music_video_maker/faces.py` gates every chained (I2V) chunk boundary on
whether the predecessor's last frame can carry the performer's identity
forward — the seed frame is the *entire* identity conditioning on that path
(`MiniMaxH3ImageToVideo` has no `ref_images` input at all; see the project
`CLAUDE.md`, "On the chained path, the seed frame IS the identity
conditioning"). Issue #47 built the first gate: is there a face in the frame,
big and confident enough to be worth trusting? Issue #49 is this document —
closing the gap #47 named on purpose and recorded in `KNOWN_LIMITATION`:
detection cannot tell whose face it is.

## The motivating failure

lucky-ones v7, boundary 9 into 10. The frame contains two people: a man
holding a keyboard, and the performer (Dianne) with her back to camera in the
foreground. YuNet detects the man's face at 1.06% of frame area, score 0.922
— comfortably clears both of #47's floors (`DEFAULT_MIN_FACE_FRACTION` =
0.45%, `DEFAULT_SCORE_THRESHOLD` = 0.9). The gate would have approved
chaining from this frame, and the chained chunk's identity conditioning would
have been a stranger, not Dianne.

## What issue #49 adds

`recognize_face(frame_path, reference_photo, ...)` computes a cosine
similarity between the seed frame's largest detected face (selected exactly
as `detect_faces` selects it — same detector call, same `score_threshold`,
same largest-by-area rule, so recognition is guaranteed to be scoring the
*same* face detection already measured) and a face detected in the active
cast member's staged reference photo (`ExpandedPrompt.image_ref` /
`CastMember.image`), using SFace (`cv2.FaceRecognizerSF`).

`build_seed_face_gate()` returns a gate whose signature grew a second,
optional argument: `gate(frame_path, reference_photo=None)`.

- `reference_photo=None` (or omitted): identical to the pre-#49 gate.
  Detection-only, exactly as issue #47 shipped it. This is what keeps every
  existing caller correct without a code change.
- `reference_photo=<path>`: after the frame clears the existing area and
  confidence floors, its largest face must *also* clear
  `DEFAULT_MIN_FACE_SIMILARITY` against the reference photo's face, or the
  gate refuses — same as any other #47 refusal, degrading to the base
  reference path.

Recognition is attempted only after detection already passed. A frame too
small or unconfident to pass #47's gate never reaches the more expensive
recognition step.

## Why SFace, and why it is not committed

SFace is the recognition model documented as the plan in #47's own
`KNOWN_LIMITATION` at the time it shipped. It is 38.7 MB against YuNet's
232 KB — 165x larger, for a check that mattered in 1 of the 12 real seed-frame
boundaries examined during #47's own calibration. Per this project's
open-source-readiness rules (issue #51: check redistribution before
committing a third-party binary), that ratio is not worth shipping by
default.

So, unlike YuNet, **the SFace weights are not committed to this
repository.** `resolve_recognition_model_path()` still names a default
location (`models/face_recognition_sface_2021dec.onnx`, beside YuNet), and
its absence is not a special case anywhere in the code — every function in
`faces.py` treats a missing recognition model exactly like a missing
detection model: a `FaceDetectionError` that the gate catches and turns into
"cannot verify, refuse to chain." A checkout with no SFace weights on disk
still renders; it just runs with #47's detection-only gate until an operator
who wants the identity check fetches the model and points
`recognition_model_path` at it (or drops it at the default location).

**Source, licence, hash** (mirroring how `faces.py` records the same for
YuNet):

| | |
|---|---|
| Filename | `face_recognition_sface_2021dec.onnx` |
| Source | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> |
| Upstream authors | SFace: Yaoyao Zhong (<https://github.com/zhongyy/SFace>); ONNX conversion: Chengrui Wang |
| Licence | Apache License 2.0, per the model directory's own `LICENSE` file at the URL above |
| Size | 38,696,353 bytes (~36.9 MiB / 38.7 MB) |
| SHA-256 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

The hash was read directly from the file's own Git LFS pointer in the
`opencv_zoo` repository (`git lfs` stores large files as small pointer
objects naming the SHA-256 and byte size; that pointer, not a local
download, is the source of truth here), so it does not depend on trusting
any one mirror or download.

## Calibration

Real material, never copied into this repository: seed frames and cast
reference photos from `~/mvm-runs/lucky-ones` (Dianne, Jan) and
`~/mvm-runs/storms` (Derek — consented for publication, issue #51 #2; the
`storms` frame used here, `seed_frontal_chunk15.png`, is already a committed
regression fixture under `tests/fixtures/seed_frames/`). Only the resulting
numbers are recorded here and in `faces.py`.

### Method

For each seed frame: run YuNet at `score_threshold=0.9` (the production
floor `detect_faces` already uses) and take the largest-by-area detection —
identical to what the area/confidence gate measures. For each reference
photo: run YuNet at `score_threshold=0.5` and take the **highest-confidence**
detection, not the largest. This split exists because reference photos
behave differently from seed frames:

- All three real cast reference photos measured (Dianne, Jan, Derek) scored
  their genuine face **below** 0.9 — 0.827, 0.859 and 0.727 respectively —
  despite each being a single, clear, correctly-framed portrait. Reference
  photos are shot differently from in-scene render frames (tighter crop,
  different lighting/lens characteristics), so the seed-frame confidence
  floor is simply too strict for them.
- Lowering the threshold to catch these widens the candidate pool, and at a
  low threshold **area stops being a safe selector**: Dianne's reference
  photo produced a 17.2%-of-frame spurious box at score 0.11, versus the
  real face's 9.2% at score 0.83 — area-based selection would have picked
  the spurious box. Confidence-based selection does not have this failure
  mode anywhere in the calibration set: the genuine face was the
  highest-scoring candidate in all three reference photos.

SFace cosine similarity between the two crops (via `alignCrop` + `feature` +
`match(..., FaceRecognizerSF_FR_COSINE)`) was then computed for every pair
below.

### Genuine pairs (same person, 4 pairs, 2 subjects)

| pair | pose | face area | similarity |
|---|---|---|---|
| lucky-ones chunk 13→14 vs Dianne | near-frontal | 4.02% | **0.820** |
| lucky-ones chunk 32→33 vs Dianne | profile | 2.94% | 0.520 |
| lucky-ones chunk 1 vs Dianne | side/profile, walking | 0.76% | 0.494 |
| storms chunk 15 (committed fixture) vs Derek | 3/4 profile, looking down | 2.88% | **0.353** |

Each same-person judgement was made by eye against the reference photo before
computing any similarity number.

### Impostor pairs (different person, 12 frame/reference pairs + 3
reference/reference cross-checks)

| pair | similarity |
|---|---|
| lucky-ones chunk 32→33 vs Jan | **0.208** |
| lucky-ones chunk 30→31 vs Dianne (a rendered face that does not visually match the reference — either a different person or severe H3 identity drift) | 0.172 |
| lucky-ones chunk 9→10 ("the keyboard man" — the motivating #49 frame) vs Dianne | 0.140 |
| lucky-ones chunk 9→10 vs Derek | 0.156 |
| lucky-ones chunk 9→10 vs Jan | 0.127 |
| storms chunk 15 vs Dianne | 0.127 |
| lucky-ones chunk 13→14 vs Derek | 0.133 |
| lucky-ones chunk 32→33 vs Derek | 0.109 |
| lucky-ones chunk 13→14 vs Jan | 0.088 |
| lucky-ones chunk 1 vs Derek | 0.058 |
| lucky-ones chunk 30→31 vs Jan | -0.012 |
| lucky-ones chunk 1 vs Jan | 0.028 |
| storms chunk 15 vs Jan | -0.030 |
| Dianne's reference vs Jan's reference | 0.076 |
| Dianne's reference vs Derek's reference | 0.074 |
| Jan's reference vs Derek's reference | -0.023 |

Worst (highest) impostor similarity: **0.208**.

### The floor

Worst genuine pair 0.353, worst impostor pair 0.208 — a gap of 0.145.

`DEFAULT_MIN_FACE_SIMILARITY = 0.34` sits 0.013 below the weakest genuine
pair and 0.132 above the strongest impostor pair — close to the genuine
floor, not centered in the gap. This mirrors `DEFAULT_SCORE_THRESHOLD`'s own
placement (issue #47): a similarity threshold guards against a false
*approve*, and a false approve is the expensive mistake here — it throws
away the likeness for the rest of the chain, where a false *refuse* only
costs a shot that renders through the strictly-better-conditioned base
reference path instead. Per the project's own rule ("a guard's threshold
belongs near the evidence, on the side whose mistake is cheap"), margin
belongs on the refuse side.

For comparison, OpenCV Zoo's own SFace demo (`models/face_recognition_sface/sface.py`)
publishes `0.363` as a general "same identity" cosine threshold, calibrated
on frontal benchmark pairs. That value sits *above* this project's own
weakest genuine pair (0.353) — consistent with H3 seed frames routinely
being non-frontal in a way a curated benchmark pair is not, and it is
exactly why this floor comes from this project's own material rather than
someone else's calibration. Using 0.363 unmodified would have rejected a
real genuine pair in this dataset.

## What is still not covered

Recorded in `faces.KNOWN_LIMITATION`:

- A checkout with no SFace weights on disk, or a caller that does not pass a
  reference photo, still runs detection-only — exactly the pre-#49 gate.
  This is not a bug; it is the deliberate degrade path for the common case
  of not having fetched the (uncommitted) model.
- Recognition cannot help when the reference photo itself has no detectable
  face (an unusable cast photo) — that degrades to refuse, same as any other
  recognition failure.
- It cannot distinguish a close relative or otherwise closely-resembling
  impostor from the real cast member if their similarity clears the measured
  floor; the calibration set has no such pair to measure against.
- It can work *against* a run, not just for it: if the performer's own
  on-screen appearance has drifted far enough from her reference photo
  (heavy makeup, a costume change, or chained appearance-compounding drift —
  issue #46) a genuine frame of her could score below the floor, and a
  correct chain gets refused rather than a wrong one approved. Given the
  cost asymmetry above, that is the direction this floor is deliberately
  biased toward.

## Wiring, and why recognition is opt-in

The gate is wired: `continuity.py` passes chunk N's own
`ExpandedPrompt.image_ref` (never the predecessor's) down to the gate, and
`i2v_min_seed_face_similarity` sits in `RunConfig` alongside the existing
`i2v_min_seed_face_fraction`.

**It defaults to `None`, meaning recognition is off and the gate behaves
exactly as #47 shipped it — detection only.** That is deliberate. The SFace
weights are 38.7 MB and intentionally not committed (see the licence and
sha256 above), so defaulting the floor on would have meant every existing
config with chaining enabled silently refusing every chain boundary until
the operator downloaded a file nobody told them about. Refusing a chain is
*safe* — the base path is the better-conditioned one — but it is still a
behaviour change to runs that did not ask for one, and this project's
convention is that a new knob leaves every pre-existing config
byte-identical.

So the two failure modes are split deliberately, on which mistake is cheap:

- **Recognition requested but the model is missing** — refuse loudly at
  config-load time with a `ConfigError` naming the file, its source and its
  sha256. The operator asked for a guard; handing them a weaker one while
  reporting success is the expensive error. Failing before a run starts is
  cheap; discovering it hours in is not.
- **Recognition running but this frame cannot be judged** — unreadable
  frame, no face found, no face in the reference photo — degrade to the base
  path with a logged reason, never raise. "Cannot prove it is her" is
  exactly the case the base path exists for.

An operator opts in with `i2v_min_seed_face_similarity = 0.34`
(`faces.DEFAULT_MIN_FACE_SIMILARITY`, calibrated above).
