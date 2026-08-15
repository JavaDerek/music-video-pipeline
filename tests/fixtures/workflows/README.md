# Workflow fixtures — confirmed against the real H3 node schema and a real render

These API-format workflow JSONs model the MiniMax H3 graph. They mirror the
real, committed `workflow_api.json` / `workflow_i2v_api.json` at the repo
root — issue #18 authored those against a live ComfyUI on doris (v0.30.2)
and validated each by actually submitting it to `POST /prompt` and letting
it render a real 5s clip end-to-end, GPU and all. Every `class_type` name,
input name, and wiring below is confirmed, not guessed.

All `class_type` values here are generated from named constants in
[`tests/harness/factories.py`](../../harness/factories.py):

| Constant | Value | Status |
|---|---|---|
| `CLASS_TYPE_UNET_LOADER` | `UNETLoader` | confirmed, stock ComfyUI |
| `CLASS_TYPE_CLIP_LOADER` | `CLIPLoader` | confirmed, stock ComfyUI (`type="minimax"`) |
| `CLASS_TYPE_VAE_LOADER` | `VAELoader` | confirmed, stock ComfyUI — used twice (video VAE, audio VAE) |
| `CLASS_TYPE_IMAGE_LOAD` | `LoadImage` | confirmed, stock ComfyUI |
| `CLASS_TYPE_AUDIO_LOAD` | `LoadAudio` | confirmed, stock ComfyUI |
| `CLASS_TYPE_H3_REFERENCE_TO_VIDEO` | `MiniMaxH3ReferenceToVideo` | confirmed |
| `CLASS_TYPE_SAMPLER_SELECT` | `KSamplerSelect` | confirmed, stock ComfyUI |
| `CLASS_TYPE_SCHEDULER` | `BasicScheduler` | confirmed, stock ComfyUI |
| `CLASS_TYPE_NOISE` | `RandomNoise` | confirmed, stock ComfyUI |
| `CLASS_TYPE_GUIDER` | `BasicGuider` | confirmed, stock ComfyUI |
| `CLASS_TYPE_SAMPLER_ADVANCED` | `SamplerCustomAdvanced` | confirmed, stock ComfyUI |
| `CLASS_TYPE_VAE_DECODE` | `VAEDecode` | confirmed, stock ComfyUI |
| `CLASS_TYPE_VAE_DECODE_AUDIO` | `VAEDecodeAudio` | confirmed, stock ComfyUI |
| `CLASS_TYPE_VIDEO_CREATE` | `CreateVideo` | confirmed, stock ComfyUI |
| `CLASS_TYPE_VIDEO_SAVE` | `SaveVideo` | confirmed, stock ComfyUI |

Two more real local H3 nodes exist (`CLASS_TYPE_H3_IMAGE_TO_VIDEO` =
`MiniMaxH3ImageToVideo`, used by the I2V fixture below; `CLASS_TYPE_H3_EMPTY_LATENT_AV`
= `EmptyMiniMaxH3LatentAV`, unused — both H3 conditioning nodes emit their
own `LATENT` output) and one optional quality-tuning node
(`CLASS_TYPE_H3_SIGMA_SHIFT` = `MiniMaxH3SigmaShift`) that the canonical
committed templates do **not** use — ComfyUI's own official MiniMax H3
templates omit it too (`UNETLoader`'s `MODEL` feeds `BasicScheduler` /
`BasicGuider` directly).

`CLOUD_API_CLASS_TYPES` lists the cloud-API variants, from
`comfy_api_nodes/nodes_minimax.py` — lowercase "m" in "Minimax", billed,
remote — that must **never** appear in a graph built by this project:
`MinimaxHailuo03FirstLastFrameNode`, `MinimaxHailuo03ReferenceNode`,
`MinimaxHailuo03TextToVideoNode`, `MinimaxHailuoVideoNode`,
`MinimaxImageToVideoNode`, `MinimaxTextToVideoNode`. `baseline_h3.json`'s
fixture test asserts none of them are present — a real misgrab risk since
the names differ from the local nodes only by capitalization.

## Key findings baked into the graph shape

- **No text-encode node.** `prompt` is a plain `STRING` input directly on
  `MiniMaxH3ReferenceToVideo` (and `MiniMaxH3ImageToVideo` on the I2V path).
  There is no `CLIPTextEncode` and no H3 text-encode node anywhere in this
  graph.
- **No `KSampler`, no negative conditioning, at all.** The real sampling
  chain — confirmed against ComfyUI's own official MiniMax H3 template
  (`video_minimax_h3_r2v.json`, shipped in the `comfyui-workflow-templates`
  package) and a real render on doris — is the generic ComfyUI "advanced
  sampling" chain: `KSamplerSelect` + `BasicScheduler` (fed by the loaded
  `MODEL`) + `RandomNoise` feed `SamplerCustomAdvanced` alongside
  `BasicGuider`, which takes a **single** (positive) `CONDITIONING` input —
  there is no `negative` socket anywhere in this graph, so the old
  "where does KSampler.negative come from" question does not apply; it
  never existed.
- **Two decode nodes, not one.** `SamplerCustomAdvanced`'s joint audio+video
  `LATENT` output feeds both `VAEDecode` (video, `minimax_h3_video_vae_fp16`)
  and `VAEDecodeAudio` (audio, `minimax_h3_audio_vae_fp32`) — each pulls its
  own half out of the packed latent. `CreateVideo` then muxes both into one
  MP4 with synced sound (later discarded at Stage 5 in favor of the pristine
  master track, per this project's audio invariant).
- **`ref_images` / `ref_audios` are `COMFY_AUTOGROW_V3` dynamic multi-slot
  inputs, and their real serialized shape is confirmed.** Each populated
  slot is its own flat, dotted key directly in the node's `inputs` dict —
  e.g. `"ref_images.ref_image_0": ["50", 0]` — holding a single plain
  `[node_id, output_index]` link, exactly like any other input. There is no
  nesting and no list-of-links wrapper. This was verified two ways: reading
  ComfyUI's own dynamic-input expansion code
  (`comfy_api/latest/_io.py`'s `finalize_prefix`, which joins the field name
  and generated slot name with a dot and uses that as the live-input lookup
  key) and a real render on doris that used exactly this shape.
- **The real ambiguity is the two `VAELoader` nodes** (video VAE vs audio
  VAE, two different weight files loaded through two separate node
  instances). When both are left at ComfyUI's un-customized default title,
  they can only be told apart by tracing which input socket on the H3
  conditioning node each one's output feeds (`vae` vs `audio_vae`) — see
  `resolve_upstream_node()`.

## Files

- `baseline_h3.json` — the ref2va (reference-to-video-audio) graph: `UNETLoader`
  feeds `BasicScheduler` + `BasicGuider`; `CLIPLoader` + two `VAELoader`s
  (video, audio) + `LoadImage` + `LoadAudio` feed `MiniMaxH3ReferenceToVideo`,
  whose positive `CONDITIONING` feeds `BasicGuider` and whose own `LATENT`
  output feeds `SamplerCustomAdvanced` alongside `RandomNoise` (`NOISE`),
  `KSamplerSelect` (`SAMPLER`), and `BasicScheduler` (`SIGMAS`);
  `SamplerCustomAdvanced`'s output feeds `VAEDecode` + `VAEDecodeAudio` ->
  `CreateVideo` -> `SaveVideo`. The two `VAELoader` nodes have distinct,
  human-authored titles ("Load Video VAE" / "Load Audio VAE").
- `shifted_ids_h3.json` — the same graph, renumbered node ids (generated by
  remapping the baseline output, not hand-duplicated). Proves introspection
  never depends on node id, only `class_type` (+ title/wiring where
  ambiguous).
- `ambiguous_vae_h3.json` — baseline with both `VAELoader` nodes left at
  ComfyUI's un-customized default title (`_meta.title == "VAELoader"` for
  both, since that's what an un-renamed node's title equals). Title text
  alone can no longer tell the video VAE from the audio VAE apart; #8 must
  trace which downstream input on `MiniMaxH3ReferenceToVideo` each output
  feeds (`vae` vs `audio_vae`) instead.
- `missing_node_h3.json` — baseline with the audio-load node (`LoadAudio`)
  removed entirely, so introspection can be proven to fail loudly rather
  than silently when a required node is absent. The `ref_audios.ref_audio_0`
  reference that pointed at the removed node is left dangling on purpose.

## `i2v_h3.json` — the I2V continuity path (issue #12)

Generated from `make_workflow_i2v()` in `factories.py`. Matches the real,
committed `workflow_i2v_api.json`, differing from `baseline_h3.json` as
follows:

- The H3 node is `MiniMaxH3ImageToVideo` (`class_type` `CLASS_TYPE_H3_IMAGE_TO_VIDEO`,
  the `fl2va` weights), not `MiniMaxH3ReferenceToVideo`. Its real required
  inputs, confirmed from a live `/object_info` dump, are just
  `clip, vae, prompt, width, height, length` plus optional `first_frame` /
  `last_frame` `IMAGE` sockets — no `ref_image_size`, no
  `ref_images`/`ref_audios` AUTOGROW inputs, and no `audio_vae` (unlike the
  reference-to-video node).
- A second `LoadImage` node, titled `"Seed Frame"`
  (`music_video_maker.continuity.DEFAULT_SEED_FRAME_TITLE`), feeds
  `first_frame` — the previous chunk's last rendered frame, so this chunk's
  motion continues from where the last one left off. The existing
  cast-reference `LoadImage`, titled `"Load Cast Reference"`
  (`continuity.DEFAULT_CAST_IMAGE_TITLE`), feeds `last_frame` instead — a
  static identity anchor pulling the generation back toward the canonical
  cast appearance by the end of every chunk, so identity drift does not
  compound chunk over chunk purely from continuity chaining. The two
  `LoadImage` nodes are disambiguated by `_meta.title`, never by node id,
  matching this project's hard node-id-independence invariant.
- `MiniMaxH3ImageToVideo` has no audio-conditioning input at all, so this
  template has no `VAEDecodeAudio` / audio `VAELoader` stage — there is
  nothing to decode into a throwaway generated-audio track (Stage 5 always
  discards generated audio regardless of path). Instead `LoadAudio` (the
  sliced vocal stem — issue #18's body explicitly requires every template
  the orchestrator mutates to carry an "audio load for the sliced stems"
  node) is wired directly into `CreateVideo.audio`, so per-chunk I2V clips
  at least carry the real vocal stem for manual review.
