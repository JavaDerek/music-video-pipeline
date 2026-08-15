# ComfyUI Setup Summary

_Last updated: 2026-08-07_

A raw, ground-truth snapshot of this project's own ComfyUI install (the box
named `doris` elsewhere in this repo) — not a generic install guide. What
`music-video-maker` actually requires is just *a* ComfyUI host running this
version or newer with the MiniMax H3 weights installed; see the README's
["Deployment: co-located by
default"](README.md#deployment-co-located-by-default) for that.

## Install

- **Location:** `/home/derek/ComfyUI`
- **Source:** git clone of `https://github.com/comfyanonymous/ComfyUI.git`
- **Version:** `v0.30.2` (official stable release tag, checked out as detached HEAD — not tracking `master`, which runs ahead with unreleased/dev features)
- **Python:** isolated venv at `.venv` (Python 3.12), separate from system Python
- **GPU:** NVIDIA RTX 4090, torch 2.13.0+cu130, CUDA 13.0 confirmed working
- **Launch:** `./start.sh` — runs `.venv/bin/python main.py` and serves the UI at `http://127.0.0.1:8188`. Not currently set up as a system service; it's started/stopped manually (or by request) and does not survive a reboot on its own.

## Custom nodes (`custom_nodes/`)

- `ComfyUI-ConditioningKrea2Rebalance` — third-party custom node
- `websocket_image_save.py` — bundled example node, ships with ComfyUI
- `.claude/` — **not a real custom node**, just a stray folder (likely a Claude Code project marker) that ComfyUI tries and fails to import on every boot. Harmless warning, safe to ignore or remove if you want a clean startup log.

## Model library on disk (`models/`)

| Folder | Size |
|---|---|
| `text_encoders` | 71 GB |
| `diffusion_models` | 43 GB |
| `unet` | 20 GB |
| `vae` | 6.6 GB |
| `loras` | 3.3 GB |

## MiniMax-H3 model set

A MiniMax-H3 (video generation) pipeline has been downloaded into the standard ComfyUI model folders:

| File | Folder | Size |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `unet/` | 20 GB |
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | 20 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | 15 GB |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoders/` | 27 GB (added 2026-08-10, issue #39) |
| `minimax_h3_video_vae_fp16.safetensors` | `vae/` | 4.9 GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | 578 MB |

Total: **~87.6 GB** of model weights (60.5 GB before the INT8 encoder).

### The precision ladder (issue #39)

Comfy-Org publishes the same components at several precisions, all loadable by
the *same* core nodes — no custom loader, no GGUF detour:

| Component | Available builds |
|---|---|
| ref2va / fl2va DiT | `pruned_int8_convrot` 21 GB (installed) · `pruned_fp8_scaled` 21 GB · `pruned_bf16` 40 GB · `int8_convrot` 34 GB · `bf16` 66 GB |
| Qwen3-VL-32B text encoder | `nvfp4_awq` 15.7 GB (installed) · `int8_convrot` 27.1 GB (installed) · `bf16` 51.5 GB |

Two facts that were assumptions until they were checked on the files
themselves:

- **Pruning and quantisation are separate decisions.** Unpruned INT8 (34 GB)
  and pruned BF16 (40 GB) both exist, so "pruned INT8" is two independent
  reductions, not one artefact.
- **The vision tower is BF16 in *both* encoder builds** — 351 `visual.*`
  tensors, unquantised, in the 4-bit file as well as the 8-bit one. Only the
  50 language layers change precision (`I8` vs `U8`+`F8_E4M3` AWQ packing).
  Reference photos still reach the encoder (`MiniMaxH3ReferenceToVideo` passes
  them to `clip.tokenize()`, and their embeddings are consumed by the
  quantised language layers), but the image *encoding* itself is identical
  across the ladder. A precision swap is a comprehension experiment far more
  than an identity one.

Encoder swapping is a run-level config knob (`text_encoder`), not a template
edit — see the README. The pinned name is recorded in every chunk fingerprint,
so `--resume` will not mix encoders inside one video.

Node support for MiniMax-H3 is built into core ComfyUI (not a custom node):

- `comfy_extras/nodes_minimax_h3.py`
- `comfy_api_nodes/nodes_minimax.py` (MiniMax API-backed nodes, separate from the local H3 model)

No saved workflow (`.json`) referencing MiniMax was found under `user/` yet, so a workflow graph wiring these nodes together still needs to be built/loaded in the UI.

## Notes

- Disk usage has been tight lately (into the high-90s % full) mainly due to these large model downloads — worth checking `df -h /` before pulling in more models.
- VRAM is shared with other GPU processes on this machine (Ollama, a TTS service, scheduled jobs) — check `nvidia-smi` before large ComfyUI generations if other GPU work is running.
