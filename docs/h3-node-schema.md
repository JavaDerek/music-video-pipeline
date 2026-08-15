# MiniMax-H3 node schema (ground truth)

_Pulled from a live `GET /object_info` on doris (ComfyUI v0.30.2), 2026-08-07._

This is the authoritative reference for issues #8 (graph introspection), #12 (I2V
continuity) and #18 (authoring `workflow_api.json`). It supersedes every guessed
`class_type` name. Issue #18 is done: `workflow_api.json` and `workflow_i2v_api.json`
are committed at the repo root, each authored against this exact schema and validated
by actually rendering a 5s clip end-to-end on doris's GPU via `POST /prompt` --
see `docs/workflow-template-guide.md`. 831 nodes are registered; 10 match minimax/h3.

## Which nodes are ours

**Local H3 nodes (core ComfyUI, `comfy_extras/nodes_minimax_h3.py`) — use these:**

| class_type | category | outputs |
|---|---|---|
| `MiniMaxH3ReferenceToVideo` | `model/conditioning/minimax` | `CONDITIONING`, `LATENT` |
| `MiniMaxH3ImageToVideo` | `model/conditioning/minimax` | `CONDITIONING`, `LATENT` |
| `EmptyMiniMaxH3LatentAV` | `model/latent/minimax` | `LATENT` |
| `MiniMaxH3SigmaShift` | `model/patch/minimax` | `MODEL` |

Note the capitalization: **`MiniMaxH3…`** (capital M in "Max").

**Cloud API nodes — do NOT use** (`comfy_api_nodes/nodes_minimax.py`, billed, remote):
`MinimaxHailuo03FirstLastFrameNode`, `MinimaxHailuo03ReferenceNode`,
`MinimaxHailuo03TextToVideoNode`, `MinimaxHailuoVideoNode`, `MinimaxImageToVideoNode`,
`MinimaxTextToVideoNode`. These spell it `Minimax…` (lowercase m) — an easy misgrab.

## `MiniMaxH3ReferenceToVideo` — the main path

Maps to `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` ("ref2va"
= reference-to-video-audio). This is the cast-reference-photo + audio-stem path.

| input | kind | type |
|---|---|---|
| `clip` | required | `CLIP` |
| `vae` | required | `VAE` (video VAE) |
| `audio_vae` | required | `VAE` (audio VAE — separate file) |
| `prompt` | required | `STRING` (multiline, dynamicPrompts) |
| `width` | required | `INT` default 1344, step 32 |
| `height` | required | `INT` default 768, step 32 |
| `length` | required | `INT` default 124, min 5, max 3600, **step 17** |
| `ref_image_size` | required | COMBO |
| `ref_images` | optional | `COMFY_AUTOGROW_V3` |
| `ref_videos` | optional | `COMFY_AUTOGROW_V3` |
| `ref_video_audios` | optional | `COMFY_AUTOGROW_V3` |
| `ref_audios` | optional | `COMFY_AUTOGROW_V3` |

## Three findings that invalidate earlier assumptions

### 1. There is no text-encode node. The prompt is a STRING input.

`MiniMaxH3ReferenceToVideo.inputs.prompt` takes the expanded prompt **directly as a
string**. There is no `CLIPTextEncode` in this graph and no H3 text-encode node exists.

Consequences for #8:
- Mutate `MiniMaxH3ReferenceToVideo.inputs.prompt`, not a text-encode node.
- The "positive vs negative `CLIPTextEncode` disambiguation" problem described in #8's
  body **does not exist here** — there is no positive/negative text pair to disambiguate.
- Resolved by #18: the real graph has no `KSampler` and no `negative` `CONDITIONING` at
  all -- see finding 2 below.

### 2. There is no H3-specific sampler, VAE-decode or save node -- and no `KSampler` at all.

The graph uses stock ComfyUI nodes downstream, but **not** the classic `KSampler`.
Confirmed against ComfyUI's own official MiniMax H3 template
(`video_minimax_h3_r2v.json`, shipped in the `comfyui-workflow-templates` package
installed on doris) and a real render (issue #18): the real chain is the generic
"advanced sampling" pattern —

`UNETLoader` → (`BasicScheduler`, `BasicGuider`); `CLIPLoader` + `VAELoader` ×2 +
`LoadImage` + `LoadAudio` → the H3 conditioning node → positive `CONDITIONING` →
`BasicGuider`; `RandomNoise` (`NOISE`) + `BasicGuider` (`GUIDER`) + `KSamplerSelect`
(`SAMPLER`) + `BasicScheduler` (`SIGMAS`) + the H3 node's own `LATENT` output →
`SamplerCustomAdvanced` → (`VAEDecode`, `VAEDecodeAudio`) → `CreateVideo` → `SaveVideo`.

**There is no `negative` conditioning anywhere in this graph.** `BasicGuider` takes a
single (positive) `CONDITIONING` input, full stop -- the "where does KSampler.negative
come from" question from earlier drafts of this doc does not apply; `KSampler` was never
part of the real graph to begin with.

`MiniMaxH3SigmaShift` is a real node (`model/patch/minimax` category, `shift_video`/
`shift_audio` FLOAT inputs) but the canonical committed templates don't use it --
ComfyUI's own official template omits it too (`UNETLoader`'s `MODEL` feeds
`BasicScheduler`/`BasicGuider` directly). It's available for future quality tuning, not
required for a working render.

`CreateVideo(images, fps, audio?, bit_depth?)` → `SaveVideo(video, filename_prefix, format, codec)`.
`CLIPLoader` accepts `type="minimax"`. `UNETLoader.weight_dtype` ∈ default / fp8_e4m3fn /
fp8_e4m3fn_fast / fp8_e5m2.

### 3. `ref_images` / `ref_audios` are `COMFY_AUTOGROW_V3`, not plain sockets -- serialized shape confirmed.

They are dynamic multi-slot inputs. Confirmed (issue #18), two ways: reading ComfyUI's
own dynamic-input expansion code (`comfy_api/latest/_io.py`'s `finalize_prefix`, which
joins the field name and generated slot name with a dot and uses that as the live-input
lookup key) and a real render on doris using exactly this shape. Each populated slot is
an ordinary **flat, dotted key** directly in the node's `inputs` dict --
`"ref_images.ref_image_0": ["50", 0]` -- holding a single plain `[node_id, output_index]`
link, exactly like any other input. There is no nesting and no list-of-links wrapper.
Injecting a staged filename is still not a scalar assignment on `ref_images` itself
though: the mutator never writes into `ref_images`/`ref_audios` directly, it only injects
filenames into the `LoadImage`/`LoadAudio` nodes those dotted-key slots reference.

## ⚠️ `length` is frame-quantized — this constrains chunk durations

> tooltip: _"Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"_

- **24 fps**, `min=5`, **`step=17`** → valid lengths are `5 + 17k`, i.e. 5, 22, 39 … 124 … 362.
- **Trained range 124–362 frames = 5.17 s – 15.08 s.**

Two corrections follow:

**(a) The 4 s minimum is below the trained range.** `HardwareProfile.min_chunk_seconds`
is currently `4.0`; the model is only trained from ~5.17 s. The floor should be 124
frames (5.167 s), with 362 frames (15.083 s) as the ceiling.

**(b) Chunk durations must land on the frame grid, or the video drifts against the
master audio.** 17 frames = 0.7083 s, so an arbitrary chunk duration rounds by up to
±0.354 s. Stage 5 concatenates chunks and muxes the *pristine* master over them, so any
per-chunk mismatch between audio-slice duration and rendered video duration accumulates
as progressive lip-sync drift across the song — precisely the failure the
forced-alignment timeline exists to prevent.

The fix is for Stage 2a slicing to choose boundaries that land on valid frame counts, so
each chunk's audio duration exactly equals its rendered video duration. This is not yet
implemented — #4 and #13 were built to the blueprint's 4–15 s window before this schema
was available. Issue #20 revisited both: slicing now snaps every chunk onto this grid and
the profiles derive their window from it.

## Reproducing this

ComfyUI now runs as a systemd service reachable across the tailnet (issue #17), so this
is a one-liner from any machine on the tailnet — no SSH, no manual launch:

```bash
curl -s http://doris:8188/object_info -o /tmp/objinfo.json
```

If it does not answer, check the service rather than assuming the box is down:
`ssh doris 'systemctl status comfyui'`.

`/object_info` loads **no model weights** — node classes register at import — so it costs
a CUDA context, not the ~60 GB H3 weight set. No GPU custody handoff required.
