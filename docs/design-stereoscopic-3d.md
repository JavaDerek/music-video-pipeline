# Stereoscopic 3D, and beats that plant for a pop (issue #68)

Two separable halves: the **conversion** (a mono render becomes a stereo pair)
and the **planning** (the shot plan knows which beats are the money shots and
stages them for it). Design only — plus one measurement that changes the order
the planning half should be built in.

## Vocabulary, because the sign is a silent bug

**Negative (crossed) parallax** is what flies out of the screen: the left-eye
image sits to the *right* of the right-eye image, so the eyes converge in front
of the display. **Positive parallax** puts an object behind the screen plane,
which is where most of a comfortable frame should live. The knob is the
convergence plane: nearer than it pops out, further recedes.

Getting the sign backwards produces a headache, not an error. Any code that
warps must name the convention in its docstring and assert it in a test with a
synthetic depth ramp, because there is no runtime symptom to catch it.

## Conversion: exactly one route is available

**Render twice from two camera positions — ruled out.** H3 has no
camera-position input and no determinism that would survive one. Two renders of
"the same shot, 65 mm to the right" are two different videos, not a stereo
pair; they would not share a single object, let alone a disparity. Do not spend
a day on it.

**Depth-based synthesis from the mono render — the only route.** Monocular
depth estimation, then depth-image-based rendering (DIBR) to warp a second eye,
then inpainting the disocclusions. The hard parts are known:

* **Temporal consistency is the whole problem.** Frame-independent depth
  shimmers, and a shimmering depth map becomes a stereo pair that boils. This
  needs a *video* depth model, not an image one.
* **The showcase shot is the worst case.** A large foreground object moving
  fast toward the lens produces the biggest disocclusions, so the shot the
  feature exists for is exactly where the second eye has the most invented
  pixels. Budget for real inpainting, not edge-stretching, and evaluate on that
  shot first rather than on a calm one.
* **Window violations kill the illusion.** An object in negative parallax
  clipped by the frame edge gives the eyes contradictory evidence — in front of
  the screen and occluded by it — and the effect collapses. A popping object
  must come *at* the lens, never across it and out the side.

### Where it lands in the pipeline

Stage 5 never re-encodes, on purpose. Stereo conversion is a re-encode.
Resolve it by converting **per chunk, before concat**: the concat demuxer keeps
`-c:v copy`, the invariant survives intact, the pass is resumable and
dead-letterable like everything else, and a failed conversion costs one chunk
rather than the video.

That also means the converted chunks must match on every parameter the concat
copy preserves. Measured on the real "Deathless" render, all 80 chunks share
one signature — `h264 / High / level 30 / 864×480 / yuv420p / progressive /
24 fps / time_base 1/12288` — and a side-by-side output doubles the width to
1728, so *every* chunk must be converted or none. A mixed-width concat produces
a file that plays wrong rather than failing.

### Scale, measured

One 8:32 song is **12334 frames** (80 chunks at 864×480, 24 fps). A depth pass
plus a warp plus inpainting runs over every one of them. It does **not** change
render cost — resolution is the dominant cost and the H3 render is unchanged;
a side-by-side output is wider only *after* generation — but it is a second
full GPU pass on the same 4090, under the same custody protocol (#19), and it
should be costed on three chunks before anyone commits a whole video to it.

### Delivery format decides the container, and should be picked first

Full/half side-by-side, over-under, MV-HEVC for spatial-video playback, or
anaglyph. **Anaglyph is worth having regardless**, purely as a review artifact:
it is the only format that can be judged on the machine that rendered it, with
no glasses beyond a £2 pair and no player support.

## The cheap early experiment, specified

Issue #68 asks whether there is one. There is, and this is exactly what it
needs. It was **not run here**: the shared Python environment is in use by
other work, and installing a model family into it mid-flight is not a change to
make on someone else's machine. It needs no GPU and about twenty minutes.

**What it proves:** whether monocular depth on H3 output is good enough for
this content at all — before any pipeline work. H3 frames are soft, hazy and
often low-contrast by design (the "Deathless" grade is "cold and desaturated
toward slate, ash and iron, blacks lifted slightly with atmospheric haze"),
which is the condition where monocular depth is weakest. That is the risk the
experiment retires.

**Inputs.** Three chunks from `~/mvm-runs/deathless/output/chunks_v12`, chosen
to span the difficulty range rather than to look good:

| chunk | camera clause | why |
|---|---|---|
| 0 | "extreme wide, locked off, the silhouette small and central against the flat horizon" | almost no depth cues; the hardest case |
| 20 | "close and slightly low on her upturned face" (looking up at a distant silhouette) | where a wrong depth map is most visible — the cardboard cut-out effect — with a real near/far pair in one frame. Also the chunk with the song's largest leading vocal offset (+2.650 s, #79) that no viewer reported, so it is worth knowing what depth makes of a face pitched that far up |
| 73 | "medium close on his face and chest, locked off", above "an ash-grey, burnt-over valley lying still and quiet below" | a genuine two-plane composition, and the chunk whose far ground already caused trouble in the #78 location work; the case the feature is for |

**Procedure.**
1. A throwaway venv — **not** `~/mvm-runs/deathless/.venv`, which the test
   suite depends on.
2. `ffmpeg -i chunk_00NN.mp4 -vf fps=8 frames/%04d.png` (8 fps is enough to see
   boiling; a full 24 fps pass is not needed to answer the question).
3. Run the depth model per frame, save 16-bit depth PNGs.
4. DIBR: shift each pixel horizontally by `disparity = baseline * (1/depth -
   1/convergence)`, clamped to a maximum of ~1.5% of frame width (about 13 px
   at 864 wide — a comfortable ceiling; more is a headache, not more 3D).
   Fill disocclusions with a horizontal edge-stretch for this test only.
5. Anaglyph mux: left eye's red channel + right eye's green and blue.
6. Watch it. The three questions: does the depth boil frame to frame; do the
   two planes in chunk 73 separate; does the face in chunk 20 stay solid or
   turn to cardboard.

**Model candidates.** The licence must be checked before anything is committed
— this repo's rule is source, licence and sha256 recorded beside any
third-party binary, the way `faces.py` does for YuNet, and "it downloaded fine"
is not a licence. As of writing, and **verify each before use**:

| model | CPU-feasible? | licence to verify |
|---|---|---|
| MiDaS (small / hybrid) | yes, seconds per frame | MIT (Intel ISL) — the safest starting point |
| Depth Anything V2 **Small** | yes | Apache-2.0 — note the Base/Large variants are **not**, they are CC-BY-NC |
| Video Depth Anything | GPU, and it is the *right* model for the temporal problem | check; the family splits licences by size |
| DepthCrafter | GPU, diffusion-based, slow | check; restrictive in the versions seen |

For the experiment, an image model (MiDaS) is fine and correct: the point is to
see whether it boils. If it does not boil at 8 fps on MiDaS, a video model will
be better. If the *depth* is wrong — not shimmery, wrong — no video model
fixes that and the feature is dead for this content.

**Do not commit the weights.** Nothing about this experiment needs anything in
the repo.

## Planning: the measurement that reorders the work

The most useful finding this project has for 3D was found for an unrelated
reason. #58: an object staged "small against the tower far behind her" did not
render *at all*, twice; the same object staged near and large rendered
correctly. `docs/shot-writing-guide.md`'s "stage the object near, not far" and
`_lint_distant_staging` already exist — and near-and-large is precisely the
staging that produces negative parallax. **The rule that makes an object appear
is the rule that makes it pop.**

What is missing is *intent*: a beat should be able to say "this is a pop
moment", the way it already says `beat_role`. #68 proposes three consequences,
and one of them turns out not to be buildable yet.

### The lint cannot be scored, and that is a finding

Issue #68 asks for "a lint that checks the checkable half: a pop beat whose
prose sends the object *across* frame, or stages it distant", picking keywords
by measurement on a real corpus per #60.

Scored against the only corpus that exists — all 80 shot lines of
`~/mvm-runs/deathless/shot_plan_v12.toml`, plus their 59 `camera` clauses:

| candidate | shot lines (n=80) | camera clauses |
|---|---|---|
| `toward(s) the lens` | 0 | 0 |
| `at the lens` | 0 | 0 |
| `into the lens` | 0 | 0 |
| `toward(s) camera` / `at the camera` | 0 | 0 |
| `into frame` | 0 | 0 |
| `fills the frame` | 0 | 0 |
| `past the lens` | 0 | 0 |
| `foreground` / `in the foreground` | 0 | 0 |
| `closer` | 0 | 0 |
| `across the frame` | 0 | — |
| `out of frame` / `exits frame` / `off the edge` | 0 | — |

The only phrases with any hits are the generic ones: bare `across` (20 of 80)
and bare `past` (7 of 80), neither of which is about the frame edge — they are
"across the valley", "past the mill". For contrast, the distant-staging
vocabulary that *is* already linted scores `distant` 1 and `horizon` 11.

**Zero positives, zero negatives, nothing to score against.** A pop-beat lint
built today would be a hypothesis with no corpus — precisely the failure #76
recorded one level up, where `lint_voiced_framing`'s keyword sets were scored
mid-render and did not survive the finished one (`_GAZE_AWAY_KEYWORDS` at 0.72x
against 3.5x when it shipped).

So the build order is the reverse of the issue's own list:

1. **The field first.** `pop = true` (or `beat_role = "pop"`) on a beat,
   emitted by the beats stage, which knows whose beat it is. Following this
   project's own rule: a property that must hold across a video needs its own
   field, and whole-video stereo parameters (convergence, baseline, format) are
   a different thing again and belong next to `cinematography` in a locked
   house-style profile (#55) — see `docs/design-cinematography-profiles.md`,
   noting they would also need fingerprint evidence before being admitted
   there.
2. **The preamble second.** `BEATS_PREAMBLE` plants for it — the objects that
   fly out are motifs the concept established, so the payoff has a cause; that
   is the plant/payoff machinery the beats stage already enforces, aimed at a
   new axis. `PROSE_PREAMBLE` composes toward the lens for it, holds it long
   enough to read (a pop that lasts eight frames is a flicker) and keeps it
   clear of the frame edges.
3. **The lint last**, scored against a plan that actually contains pop beats.
   Only then is there a corpus with a known outcome, and only then can the
   excluded candidates be recorded with their reasons the way #60 requires.

### Two things worth stating now

* A pop beat should be *rare*. Every candidate above scoring zero is not only
  an absence of evidence — it is a reminder that 80 shots of a serious video
  contained no shot composed at the lens at all, and that a video where six of
  them are is a different video.
* The existing `_lint_distant_staging` is already half the pop lint. A pop beat
  whose prose stages its object distant is the same defect it catches, with a
  higher cost. Reuse it rather than writing a second vocabulary.

## Open questions, and the answers this design assumes

* **Whole video in 3D, or only the popping shots?** Whole video. The format is
  per-file, so "mixed" means a 3D file whose non-pop shots have near-zero
  disparity, not two files. That is fine — but it means the depth pass runs
  over all 12334 frames regardless, which is the cost line above.
* **Does a locked house style include a stereo profile?** Probably yes —
  comfort settings (convergence, maximum disparity) are a signature and a
  safety limit, not a per-song mood. But they only belong in a profile once
  they are fingerprinted; see the profile design doc's schema constraint.
* **Is the anaglyph review artifact a product or a tool?** A tool. It should
  never be the deliverable; it exists so the depth can be judged on the machine
  that rendered it.
