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

### The scaffold that exists, and what it is not

**Built (2026-09-21): `music_video_maker/stereo.py`, never run on a real
chunk** (its warp ran on 9 real frames on 2026-10-05; see "Not streaming"
below). It is the arithmetic and the seams, not the feature: the sign
convention with the synthetic-depth test this document asks for, a
convergence-plane parameter, the DIBR forward warp with a z-buffer for
collisions, nearest-neighbour hole filling, side-by-side and anaglyph output,
and the ffmpeg probe/decode/encode argv — with the depth model left as an
**injected callable** so no weights, no GPU and no licence question enter the
test suite.

What it deliberately is not:

* **Not streaming.** `decode_frames` still buffers a whole chunk (~240 MB at
  192 frames). The warp itself was the other half of this bullet and is done
  (2026-10-05): with numpy importable — declared as the `stereo` extra, and
  also arriving with the `faces` extra and any depth-model environment —
  `warp_eye` runs a vectorised path that is **byte-identical** to the
  original per-pixel loop, which stays as the fallback and as the test
  oracle. On 18 real 864x480 warps of the v14 render it measured ~35 ms per
  stereo pair against ~450 ms for the loop (12.8x; ~0.12 h against ~1.5 h of
  warping for a 12334-frame song), and every output matched the loop byte for
  byte: `~/mvm-runs/deathless/measurements/stereo68_2026-10-05/FINDINGS.md`.
  The loop was slow, not unrunnable — under half a second a pair — so the
  depth model, at ~0.5 s a frame on CPU, is now the dominant cost.
* **Not inpainting.** The hole fill copies the nearest written pixel along the
  row — the "edge-stretch, for this test only" this document specifies for the
  *experiment*. The showcase shot is still the worst case, and it is exactly
  where this will look worst.
* **Not temporally consistent.** Depth is per frame and will boil. The only
  numbers that exist are in `docs/pop-beat-corpus.md`.
* **Not in the render path.** It reads finished chunk mp4s and writes new
  files. No `ChunkFingerprint` moves, no `run_state.json` is written, and it
  cannot cause a re-render — a stereo pass able to invalidate a cached chunk
  would put hours of GPU custody behind a post-process.

One question it answers on the way past: #68 asks whether stereo comfort
settings belong in a locked house style (#55). **Not yet, and #55's own schema
constraint is what says so** — every field a profile may set must already be
recorded in a `ChunkFingerprint`, and a pass that runs *after* the render and
changes no chunk H3 produced has nothing there to move. They live in
`stereo.StereoParams` until that stops being true.

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

**Status (2026-09-21): steps 1 and 2 are built; step 3 is built in its
structural half only, and its keyword half is still blocked.** The corpus
that arrived on 2026-09-20 — three hand-written pop lines that rendered as
intended, chunks 45/46/66 — is recorded in `docs/pop-beat-corpus.md` with
every candidate and its reason, and it changes the answer for exactly one
check. What shipped is `prose.pop_object_named_in_shot_issues`: a pop beat
whose shot line never names its own `pop_object`. It needs **no vocabulary**,
which is why it could ship at n=3 — `pop_object` reaches no prompt (it rides
in the plan's `# beat:` comment like `act`), so the shot line is the only
channel by which the object a pop beat exists for can reach H3, and correct
authoring names it while incorrect authoring does not. It fires on 0 of the 3
known-good lines; it has **no measured true-positive rate**, because no
authored pop beat has ever been seen omitting its object. A keyword lint
still may not ship: the three lines' verbatim text is not published, so
nothing can be scored *on* them, and every candidate still scores zero on the
only 80-line corpus with per-chunk outcomes.

**Status (2026-09-13): steps 1 and 2 below are built, step 3 is still not.**
`Beat.pop_object` (not `pop = true`, and not `beat_role = "pop"` -- see the
field's own docstring in `authoring/beats.py` for why a named motif beats a
boolean: the preamble needs a concrete thing to plant, and a future lint
needs a concrete noun to look for), `BEATS_PREAMBLE`/`PROSE_PREAMBLE`/
`PHOTOGRAPHY_PREAMBLE`'s pop guidance, and the reuse of
`_lint_distant_staging`'s own predicate for a sharper pop-scoped warning all
shipped under issue #68. Step 3 -- a pop-specific keyword lint -- did not,
for exactly the reason this section already gives: the only real corpus still
contains zero pop beats, so there is still nothing to score a new vocabulary
against. That does not change until a real plan exists with `pop_object` set
on at least a few beats.

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

1. **The field first — BUILT.** `Beat.pop_object: str | None`, naming the
   motif rather than a bare flag, emitted by the beats stage, which knows
   whose beat it is. Structurally required to have an earlier `plant` in its
   own `beat_group` — the same plant/payoff shape `check_beat_structure`
   already enforces for a `consequence`, reused rather than reinvented — and
   deliberately not restricted by `beat_role` or voiced/instrumental status:
   neither restriction has evidence behind it (see the field's own docstring).
   `to_dict` omits the key when unset, so a pre-#68 beat sheet hashes
   byte-identically to before. Following this project's own rule — a property
   that must hold across a video needs its own field — whole-video stereo
   parameters (convergence, baseline, format) are a different thing again and
   belong next to `cinematography` in a locked house-style profile (#55) —
   see `docs/design-cinematography-profiles.md`, noting they would also need
   fingerprint evidence before being admitted there. **Not built here.**
2. **The preamble second — BUILT.** `BEATS_PREAMBLE` plants for it — the
   objects that fly out are motifs the concept established, so the payoff has
   a cause — and says to keep it rare, with no invented cap. `PROSE_PREAMBLE`
   composes toward the lens for it, holds it long enough to read (a pop that
   lasts eight frames is a flicker) and keeps it clear of the frame edges.
   `PHOTOGRAPHY_PREAMBLE` gets the same "toward the lens" instruction, since a
   pop beat's `camera` should push toward the object rather than away from it
   — `pop_object` reaches all three stages' prompts the same way
   `beat_role`/`focus`/`camera` already do.
3. **The lint last — STRUCTURAL HALF BUILT 2026-09-21, KEYWORD HALF STILL
   NOT.** See the status block at the top of this section and
   `docs/pop-beat-corpus.md`, which is the register the keyword half will be
   built from and which records the two other candidates considered and
   rejected (the object as grammatical subject; the object named in its own
   plant's prose), each with the reason. Scored against a plan that actually
   contains pop beats. Only then is there a corpus with a known outcome, and
   only then can the excluded candidates be recorded with their reasons the
   way #60 requires. What *is* built in the meantime: `write`'s advisory
   checks re-run `_lint_distant_staging`'s own predicate — imported, not
   copied — scoped to beats the sheet marks `pop_object`, and report a
   sharper warning when one trips it (`prose.pop_distant_staging_issues`).
   That is not step 3; it is the "already half the pop lint" observation
   below, wired up rather than only stated.

### Two things worth stating now

* A pop beat should be *rare*. Every candidate above scoring zero is not only
  an absence of evidence — it is a reminder that 80 shots of a serious video
  contained no shot composed at the lens at all, and that a video where six of
  them are is a different video.
* The existing `_lint_distant_staging` is already half the pop lint. A pop beat
  whose prose stages its object distant is the same defect it catches, with a
  higher cost. Reuse it rather than writing a second vocabulary — now wired up
  (`prose.pop_distant_staging_issues`, factored out as
  `shot_plan._distant_staging_match` so both call sites share one predicate),
  not just stated as an intention.

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
