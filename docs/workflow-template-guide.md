# Workflow template guide

How to export `workflow_api.json` from ComfyUI, exactly which nodes the
orchestrator locates and mutates per chunk, and why the I2V continuity path
is a second authored template rather than a tweak of the base one.

Ground truth for every node name and input in this doc is
[`docs/h3-node-schema.md`](h3-node-schema.md), pulled from a live
`GET /object_info` dump against ComfyUI v0.30.2 on doris. Read that first if
you haven't. The code implementing everything described here is
[`music_video_maker/workflow_graph.py`](../music_video_maker/workflow_graph.py).

## `workflow_api.json` and `workflow_i2v_api.json` are committed at the repo root

Issue #18 authored both templates against a live ComfyUI on doris and
validated each by submitting it to `POST /prompt` unmodified and letting it
actually render a real 5s clip end-to-end (GPU and all) -- `workflow_api.json`
for the base ref2va (reference-to-video-audio) path, `workflow_i2v_api.json`
for the I2V continuity path (issue #12). Both are the canonical templates;
`tests/fixtures/workflows/baseline_h3.json` and `i2v_h3.json` mirror them for
the offline test suite (see
[`tests/fixtures/workflows/README.md`](../tests/fixtures/workflows/README.md)).

They were authored directly against the API-format JSON and ComfyUI's own
graph validator (`execution.validate_prompt`, importable standalone -- no
running server, no GPU, no queued execution) rather than built by hand on the
canvas and exported. That's a legitimate, more reproducible alternative to
the Dev-mode "Save (API Format)" workflow described below, which remains the
right approach if you're iterating on the graph interactively in the browser.

## Exporting API-format JSON from ComfyUI

The orchestrator only understands ComfyUI's **API format** — a flat JSON
object mapping node id → `{"class_type": ..., "inputs": {...}, "_meta": {...}}`
— not the UI's native "workflow" save format (which nests everything under
a different schema for the canvas editor). To export API format:

1. In the ComfyUI web UI, open **Settings** and enable **Dev mode** (this
   adds an API-format save option to the menu).
2. Build/load your graph on the canvas.
3. Use **Save (API Format)** (in the menu that Dev mode exposes) to write the
   flat JSON.

## Which nodes the orchestrator mutates

`WorkflowGraphMutator.mutate()` (in `workflow_graph.py`) is the single place
that turns a loaded template into a chunk-ready workflow. It deep-copies the
template first — the caller's in-memory template is reused across every
chunk in a run and must never be mutated in place — then injects into
exactly four places:

| What | Where | How it's located |
|---|---|---|
| The expanded prompt string | `.inputs.prompt` on whichever H3 conditioning node is present | `_find_h3_conditioning_node()` — see below |
| The staged cast reference photo filename | `.inputs.image` on a `LoadImage` node | `find_one_node()`, or `find_titled_node()` if `cast_image_title` is given |
| The staged audio stem filename | `.inputs.audio` on the `LoadAudio` node | `find_one_node(workflow, "LoadAudio")` |
| The per-chunk output filename prefix | `.inputs.filename_prefix` on the `SaveVideo` node | `find_one_node(workflow, "SaveVideo")`, value = `"mvm_chunk_{chunk_id:04d}"` (e.g. `mvm_chunk_0007`) — issue #9's execution stage relies on this exact format to tell chunk outputs apart in ComfyUI's history / on disk |

A fifth injection, `length_frames`, is issue #20's frame-quantized `length`.
It goes into the **same H3 node the mutator already located**, so no caller
has to know which of the two H3 `class_type`s a given template uses. It is a
value, not a policy: `mutate()` never computes or quantizes a frame count
(slicing does, and puts the answer on `AudioChunk.frame_count`). What it does
enforce is that the value is one the node can honor exactly — see the
`length` frame-grid section below. Passing `None` leaves whatever `length`
the authored template carries.

A sixth, `noise_seed`, goes into the single `RandomNoise` node on every
mutation (issue #38) — a seed left to whatever the template file carries is a
render nobody can reproduce.

A seventh, `text_encoder`, is issue #39's `CLIPLoader.clip_name`. Unlike the
seed it is injected **only when a caller names one**: there is no
project-wide default encoder, and inventing one here would mean this module,
rather than the authored template, decides which weights doris loads. It is a
*server-side* file name resolved inside ComfyUI's own `models/text_encoders/`
— a blank, non-`str` or absolute value raises `InvalidTextEncoderError`, and
a template with no `CLIPLoader` raises `NodeNotFoundError` rather than
rendering the whole run through the encoder the caller was trying to replace.
`read_text_encoder(workflow)` is the read-only counterpart: it reports what a
graph *would* load (or `None`), which is how a run records its encoder in
each `ChunkFingerprint` even when it pins nothing.

An eighth, optional seam, `node_input_overrides: dict[class_type, dict[input_name, value]]`,
lets later stages push arbitrary additional inputs through (e.g. CLI-level
`width`/`height`/`seed`) without this module owning that policy — it resolves
each `class_type` via `find_one_node` (no title support), so it will raise
`AmbiguousNodeError` on a `class_type` that legitimately appears twice
(`VAELoader`) unless you're targeting one that doesn't.

### The prompt: no text-encode node exists

The issue #8 body originally assumed a `CLIPTextEncode` node pair (positive
and negative prompts) that would need disambiguating. **That graph shape does
not exist.** `MiniMaxH3ReferenceToVideo.inputs.prompt` (and, on the I2V
template, `MiniMaxH3ImageToVideo.inputs.prompt`) is a plain multiline
`STRING` input taken directly by the H3 node itself — there is no
`CLIPTextEncode` anywhere in this graph, and therefore no
positive/negative-prompt ambiguity to resolve.

`_find_h3_conditioning_node()` locates whichever of the two H3 node classes
is present — `MiniMaxH3ReferenceToVideo` (base) or `MiniMaxH3ImageToVideo`
(I2V) — and requires **exactly one** of the two. A template with both, or
neither, is treated as malformed (`AmbiguousNodeError` / `NodeNotFoundError`).

There is no `KSampler` downstream at all, and therefore no `negative`
`CONDITIONING` to source (issue #18, confirmed against ComfyUI's own
official MiniMax H3 template and a real render). The real sampling chain is
the generic ComfyUI "advanced sampling" pattern: `KSamplerSelect` +
`BasicScheduler` + `RandomNoise` feed `SamplerCustomAdvanced` alongside
`BasicGuider`, which takes a single (positive) `CONDITIONING` input — full
stop. See `docs/h3-node-schema.md` finding 2 for the exact wiring.

### The real ambiguity: two `VAELoader` nodes

The graph loads two separate VAE weight files through two separate
`VAELoader` node instances — the video VAE
(`minimax_h3_video_vae_fp16.safetensors`) and the audio VAE
(`minimax_h3_audio_vae_fp32.safetensors`). If both are left at ComfyUI's
un-customized default title, `_meta.title` can't tell them apart (both read
`"VAELoader"`). The orchestrator resolves this by **wiring**, not title:
`resolve_upstream_node(workflow, "MiniMaxH3ReferenceToVideo", "vae")` and
`resolve_upstream_node(workflow, "MiniMaxH3ReferenceToVideo", "audio_vae")`
trace which `VAELoader`'s output feeds which of the H3 node's two input
sockets. This is deliberately more robust than relying on human-authored
titles, which ComfyUI does not enforce.

## `find_one_node` vs. `find_titled_node`: hint vs. identity claim

This distinction is load-bearing — getting it backwards silently overwrites
the cast reference photo with the I2V seed frame.

- **`find_one_node(workflow, class_type, title=None)`** — if exactly **one**
  node of `class_type` exists in the workflow, it is returned **regardless of
  whether `title` was given or matches**. `title` is consulted only when
  there are 2+ candidates, to narrow them down to exactly one; it is a
  *tiebreaker*, never a requirement. This is right for the base template,
  which has exactly one `LoadImage` — the cast photo — so
  `mutate(..., cast_image_title=None)` finds it unambiguously with no title
  needed at all.

- **`find_titled_node(workflow, class_type, title)`** — always requires an
  **exact** `_meta.title` match, regardless of how many (or how few) nodes of
  `class_type` exist. Zero matches raises `NodeNotFoundError` (naming the
  titles that *are* present); multiple nodes sharing that title raises
  `AmbiguousNodeError`. `title` here is an *identity claim*, not a hint.

The I2V template carries **two** `LoadImage` nodes: the cast reference photo
(titled `"Load Cast Reference"`) and the seed frame from the previous chunk
(titled `"Seed Frame"`). `ContinuityWorkflowProvider` always calls `mutate()`
with both `cast_image_title` and `seed_frame_title` set on this path, and
`mutate()` resolves *both* through `find_titled_node()` — never
`find_one_node()` — specifically because of this scenario: if the seed-frame
injection used `find_one_node()` instead, and it were ever run against a
malformed or wrong template that happens to hold only **one** `LoadImage`,
`find_one_node()` would return that single node **without even checking
whether its title matches** — the same node the cast-photo injection just
wrote to. Since seed-frame injection runs after cast-photo injection in
`mutate()`, that would silently overwrite the cast reference photo with the
seed frame, and the render would carry no cast face at all — a defect
invisible until after the GPU hours are spent, not something a schema
validator would catch. Requiring `find_titled_node()`'s exact-match semantics
for the seed frame turns that into a loud `NodeNotFoundError` instead.

Rule of thumb: use `find_one_node(title=...)` when you expect exactly one
node and titles only matter to break a tie you don't expect to hit; use
`find_titled_node()` whenever a specific node's identity — not just its
`class_type` — is what makes an injection correct.

## `last_frame` dictates the final frame -- it is not an identity anchor

`workflow_i2v_api.json` once wired the cast reference photo into
`MiniMaxH3ImageToVideo.last_frame`, on the assumption that it would pull the
generation back toward the canonical cast appearance and stop drift
compounding across a chain.

It does not. `comfy_extras/nodes_minimax_h3.py` resolves `last_frame` to a
keyframe at `resolved_frame_index = frame_count - 1`, cover-crops it to the
canvas, and re-injects it at **every** sampling step. It is a mandate, not a
hint: whatever you wire there *is* the final frame of the clip.

On lucky-ones v7 that produced four distinct-looking defects with one cause --
a face flashing at every cut, shots dragged into a frontal smiling close-up for
their whole length, the performer spun round to camera to hit the mandated
pose, and (because the next chained chunk is seeded from that final frame)
consecutive chained shots cutting portrait-to-portrait.

So `last_frame` is left unwired. Identity on the chained path comes from the
seed frame; drift is answered by re-anchoring (`i2v_reanchor_interval`), which
routes a chunk back through the reference template where the cast photo is a
genuine `ref_images` reference rather than a mandated frame. The cast
`LoadImage` stays in the graph with its title intact -- the mutator injects
into it by title on every I2V mutation, and removing it would make that lookup
fall through to the seed-frame node -- but nothing consumes it, so ComfyUI
never executes it.

The general lesson, and it has bitten this project before: read the node's own
source for what an input *does*, rather than inferring it from the input's
name. See issue #44.

## Why the I2V path is a second authored template, not a mutation

`config.i2v_workflow_template` is a wholly separate file from
`workflow_template`, and `WorkflowGraphMutator` never transforms one into the
other. Concretely:

- The H3 conditioning node itself is a **different `class_type`** —
  `MiniMaxH3ImageToVideo` instead of `MiniMaxH3ReferenceToVideo`, confirmed
  against a live `/object_info` (issue #18): its required inputs are just
  `clip, vae, prompt, width, height, length` plus optional `first_frame` /
  `last_frame` `IMAGE` sockets — no `ref_image_size`, no
  `ref_images`/`ref_audios`, and no `audio_vae` (unlike the
  reference-to-video node).
- The I2V template adds a **second `LoadImage`** node (the seed frame) wired
  into `first_frame`, on top of the base template's single cast-reference
  `LoadImage` — which in this template is deliberately left **unwired**, for
  the reasons in "`last_frame` dictates the final frame" above. It stays in the
  graph so the mutator's title lookup still finds it; with nothing consuming
  its output, ComfyUI never executes it. A structurally different graph, not an
  extra input on the existing one.
- `ContinuityWorkflowProvider` (issue #12) decides *per chunk, at render
  time* whether to render through the base or I2V template — chunk 0, and
  any chunk whose predecessor didn't succeed, always falls back to the base
  template. A single mutated template couldn't represent both shapes at
  once.

`config.load_config()` enforces this pairing: setting `i2v_continuity = true`
without also setting `i2v_workflow_template` fails config validation loudly,
naming the reason (`config.py`'s `_validate()`).

`tests/fixtures/workflows/i2v_h3.json` mirrors this shape, matching the real,
committed `workflow_i2v_api.json` — issue #18 authored and validated it by
actually rendering a 5s clip through `MiniMaxH3ImageToVideo` on doris.

## The `length` frame-grid constraint

`MiniMaxH3ReferenceToVideo.inputs.length` (and presumably its I2V
counterpart) is an `INT` — a **frame count**, not a duration — quantized by
ComfyUI itself: `min=5`, `step=17`, so valid values are `5, 22, 39, ... 124,
... 362, ...` up to `max=3600`. At the model's fixed 24 fps, the **trained
range is 124–362 frames = ~5.167 s – ~15.083 s**; per the tooltip captured in
`docs/h3-node-schema.md`, going outside that range means the model is
extrapolating.

`music_video_maker/contracts.py`'s `FrameGrid` (and the `H3_FRAME_GRID`
constant) encode this grid as reusable arithmetic (`is_valid`,
`quantize_up`, `quantize_nearest`, `clamp_to_trained`) so Stage 2a slicing
and `continuity.py`'s frame math share one source of truth instead of two
copies of the same rule.

`WorkflowGraphMutator.mutate()` does **not** compute `length` — Stage 2a
does, and hands it over via `length_frames`. But the mutator *validates* it,
raising `InvalidLengthError` for either failure mode:

- **Off the grid.** ComfyUI does not reject an off-grid `length`; it silently
  rounds it, while the chunk's audio stem keeps its own duration. That silent
  rounding is precisely the drift this constraint exists to prevent, so it
  has to fail before submission rather than be discovered afterwards.
- **Outside 124–362.** These are valid grid points that ComfyUI accepts
  happily — they are refused because the model was never trained there, so
  the result is out-of-distribution rather than wrong-length.

Why this matters beyond the node schema: Stage 5 concatenates every chunk's
rendered video with `-c:v copy` and mux the pristine master audio over the
result once, at the end — it never re-corrects a per-chunk mismatch between
a chunk's audio-stem duration and its rendered video duration. An
unquantized chunk duration can be off from its nearest valid frame count by
up to half a step (17 frames = 0.7083 s), and that rounding error
accumulates chunk over chunk into progressive lip-sync drift against the
master track by the end of the song. Landing chunk boundaries on the frame
grid during slicing (rather than after the fact) is what keeps that from
happening; see the [README's chunk duration window
section](../README.md#chunk-duration-window) for the numbers this replaces.

## Resolved by issue #18

Three things this guide used to flag as provisional/unconfirmed are now
settled, each verified by an actual render on doris, not just graph
validation:

- **`ref_images` / `ref_audios` serialization.** These `COMFY_AUTOGROW_V3`
  dynamic multi-slot inputs serialize as ordinary flat, dotted keys directly
  in the node's `inputs` dict — e.g. `"ref_images.ref_image_0": ["50", 0]`
  — holding a single plain `[node_id, output_index]` link, exactly like any
  other input. No nesting, no list-of-links wrapper. Confirmed by reading
  ComfyUI's own dynamic-input expansion code (`comfy_api/latest/_io.py`'s
  `finalize_prefix`) and by a real render using exactly this shape.
  `WorkflowGraphMutator` still never writes into these fields directly; it
  only injects filenames into the `LoadImage`/`LoadAudio` nodes those
  dotted-key slots reference.
- **`MiniMaxH3ImageToVideo`'s real input schema**: `clip, vae, prompt, width,
  height, length` (required) plus `first_frame` / `last_frame` `IMAGE`
  sockets (optional) — no `start_image`, no `ref_image_size`, no
  `ref_images`/`ref_audios`, no `audio_vae`.
- **Where `KSampler`'s `negative` conditioning comes from**: nowhere, because
  there is no `KSampler` in the real graph. The sampling chain is
  `KSamplerSelect` + `BasicScheduler` + `RandomNoise` + `BasicGuider` (single
  positive `CONDITIONING`, no negative) + `SamplerCustomAdvanced`.
