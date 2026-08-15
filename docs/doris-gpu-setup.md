# doris / ComfyUI host setup

**This describes one specific, real setup — not a requirement.** All
`music-video-maker` needs is *a* ComfyUI host with the MiniMax H3 weights
installed, reachable over the network (`comfyui_url`, `http://127.0.0.1:8188`
by default if it's the same machine — see the README's ["Deployment: co-located
by default"](../README.md#deployment-co-located-by-default)). `doris` is this
project's own GPU box; everything below is what it took to run that box
reliably, kept because the specific numbers and failure modes are more useful
than a sanded-down generic version would be.

This builds on [`comfyui-setup-summary.md`](../comfyui-setup-summary.md) —
the ground-truth snapshot of the actual doris install — rather than
describing a from-scratch setup. Read that doc first for the raw facts
(versions, model files, disk usage); this doc adds the pieces specific to how
`music-video-maker` talks to that box: reaching it, the GPU custody protocol,
and the current known gaps.

## The box, in short

- ComfyUI **v0.30.2** at `/home/derek/ComfyUI`, isolated Python 3.12 `.venv`,
  RTX 4090, torch 2.13.0+cu130 / CUDA 13.0.
- The full MiniMax H3 model set is already downloaded (~60.5 GB across
  `unet/`, `diffusion_models/`, `text_encoders/`, `vae/`) — see
  `comfyui-setup-summary.md`'s table for the exact files.
- No saved MiniMax workflow JSON exists on doris yet — authoring
  `workflow_api.json` is open issue #18; see
  [`docs/workflow-template-guide.md`](workflow-template-guide.md).

## H3 support is core ComfyUI, not a custom node

`comfy_extras/nodes_minimax_h3.py` — where `MiniMaxH3ReferenceToVideo`,
`MiniMaxH3ImageToVideo`, `EmptyMiniMaxH3LatentAV`, and `MiniMaxH3SigmaShift`
live — ships with ComfyUI v0.30.2 itself. **Nothing needs to be installed to
use H3.** The only nodes this project would ever need to install into
`custom_nodes/` are the optional #13 optimization nodes, and only if VRAM
pressure or generation speed calls for them:

| Node | Repo | When it's worth installing |
|---|---|---|
| `SageAttention` | `ComfyUI-KJNodes` | Recommended unconditionally on doris's 24 GB card (`hardware.py`'s `PROFILE_RTX_4090_24GB`) — faster attention with no quality tradeoff. |
| `SpectrumMiniMaxH3` | `ComfyUI-Spectrum-MiniMax-H3` | Recommended on doris; skips expensive transformer evaluations via a Chebyshev ridge fit. Not needed on a 48 GB+ card, which has headroom to spare. |
| `EasyCache` | `Block/EasyCache` | Last-resort fallback if OOM errors persist even with the above — caches intermediate diffusion steps at some quality-degradation risk. |

`music_video_maker/hardware.py`'s `scan_workflow_for_missing_optimizations()`
checks a loaded workflow template against a `HardwareProfile`'s
`recommended_nodes` and logs a warning (naming the exact repo above) for
anything missing — it never installs anything itself.

### The cloud variant exists in the same file tree — never use it

`comfy_api_nodes/nodes_minimax.py` also ships with ComfyUI, registering a
**second**, unrelated set of MiniMax nodes:
`MinimaxHailuo03FirstLastFrameNode`, `MinimaxHailuo03ReferenceNode`,
`MinimaxHailuo03TextToVideoNode`, `MinimaxHailuoVideoNode`,
`MinimaxImageToVideoNode`, `MinimaxTextToVideoNode`. These are billed,
remote, cloud-API-backed nodes — nothing like the local H3 weights this
project runs. They're spelled `Minimax…` with a **lowercase m**, versus the
local nodes' `MiniMaxH3…` with a **capital M** — an easy misgrab when
skimming a node list.

`music_video_maker/workflow_graph.py`'s `assert_no_cloud_api_nodes()` is
called on every `mutate()` and raises `CloudApiNodeDetectedError` if any of
these six class_types are present in a workflow — this project must never
build or execute a graph containing them.

## Reaching the box

Always use the Tailscale hostname:

```
http://doris:8188
```

**Never** the raw LAN IP (a `192.168.x.x` address) — the Tailscale name
resolves both at home (over the LAN) and away (via relay); the raw IP only
works at home and silently hangs everything else. This is the value set for
`comfyui_url` in this project's own run configs, overriding the package
default of `http://127.0.0.1:8188` (co-located, issue #50) since the
orchestrator runs on a separate machine from doris. (doris's Tailscale IP is
the stable fallback if MagicDNS is ever unavailable — still not the LAN IP.)

## Running as a service, reachable over the tailnet (issue #17)

ComfyUI runs as the **`comfyui.service`** systemd unit — enabled, so it comes
back after a reboot, with `Restart=always` so it comes back after a crash.

```bash
systemctl status comfyui       # is it up?
sudo systemctl restart comfyui # after installing a custom node
journalctl -u comfyui -f       # logs, including render progress
```

It binds **two** addresses: `127.0.0.1` (so anything on doris itself keeps
working) and doris's own Tailscale address. That makes
`http://doris:8188` work from anywhere on the tailnet while doris's LAN
address stays refused.

Two independent guards keep it that way, and both matter:

1. **The bind list.** `--listen 127.0.0.1,<tailscale-addr>` — the process
   never listens on the LAN interface at all.
2. **ufw.** Default policy is `deny (incoming)`, with one interface-scoped
   exception: `8188/tcp on tailscale0 ALLOW`.

> **Never rebind this to `0.0.0.0`.** ComfyUI ships no authentication of any
> kind, and it can load arbitrary custom nodes and write files. An open bind
> is remote code execution for everyone on the network. If you need it
> somewhere new, add that address to the bind list and add a matching
> interface-scoped ufw rule — do not widen either one.

Because the bind list names the Tailscale address explicitly, the interface
has to exist before `main.py` starts; the unit's `ExecStartPre` waits for it
rather than restart-looping through boot.

## Disk pressure

Disk on doris runs into the **high-90s % full**, mostly from the H3 model
downloads. Check `df -h /` before writing anything large there — chunk
videos, staged uploads, and ComfyUI's own output directory all land on the
same disk. `music_video_maker.resilience.ResilientRunner` has a pre-flight
check for this on the *local* (orchestrator) side —
`config.min_free_disk_gb` (default 20.0 GB) is checked against the run's
output directory before any chunk renders, refusing to start rather than
hitting ENOSPC dozens of chunks in — but that check is local-disk only;
there's no ComfyUI endpoint that reports server-side free disk on doris
itself, so server-side disk-full is instead caught reactively (an
ENOSPC-shaped error message dead-letters the chunk immediately rather than
retrying it).

## GPU custody (issue #19)

The 4090 is shared. Other always-on services and the desktop session hold
VRAM, and the biggest tenant loads on demand. A render run needs the *whole*
card — `hardware.py`'s static `HardwareProfile`s assume full nominal VRAM
rather than querying live free VRAM, on the assumption that custody is
already exclusive by the time a render starts (this was weighed and rejected
as a live-VRAM-check strategy — see `hardware.py`'s module docstring). The
custody protocol is:

1. **Before rendering**: stop the other GPU tenants, by hand. On this host
   that means pausing the assistant service that holds Qwen resident, which
   is a Telegram command sent from Derek's own account — that step lives in
   Derek's own notes, not in this repo, because it is specific to this
   machine and it is not something the pipeline can or should do for you.
2. **Start the run.** `VramCustodyManager`'s pre-flight refuses it outright
   if the card is still held.
3. **After the run** (success, failure, dead-letter, Ctrl-C, or crash —
   unconditionally): the pipeline `POST /free`s ComfyUI's models. Restart
   what you stopped.

### What's automated today

`music_video_maker/custody.py` implements the machine-checkable half:

- **`VramCustodyManager`** — on `__enter__`, reads ComfyUI's
  `GET /system_stats` and refuses the run with `CustodyError` when free VRAM
  is below `min_free_vram_gb`. An unreachable or unparseable response
  degrades gracefully (logged, non-fatal) rather than blocking a run over a
  best-effort check. `__exit__` is unconditional: `POST /free` on normal
  completion and on any exception propagating out of the `with` block.
- **`build_custody_manager(config, ...)`** is the one factory `cli.py` calls.
- **`build_vram_probe(...)`** is the same reading, exposed as a seam
  `ResilientRunner` re-uses *between chunks* (issue #23) — the pre-flight is
  point-in-time and cannot see a process claiming the card after the run
  starts, which is exactly how the 2026-08-07 incident began.

`run_pipeline()` builds its manager through `build_custody_manager()`, so the
pre-flight brackets the whole render. `tests/test_cli.py` asserts the ordering
that actually matters: `GET /system_stats` happens *before* the first
`POST /prompt`, not merely that both happened.

**Automating the pause/resume of another service was tried and removed.** A
Telethon user-session integration lived here for months, enabled by default,
and never ran once outside its own tests — every real render used the manual
path. It is in the git history if it is ever wanted back.

What this does **not** do is free VRAM held by anything else on the card.
A TTS service and the desktop session (Chrome, nautilus) hold GPU memory
independently — a measurement right after pausing the largest tenant showed
~8 GB held, ~16.4 GB free (that reading also
included a scheduled stack retired 2026-08-08; the card is less contended
now).

**That is enough.** 5-second clips render at good quality in exactly that
state. It is also enough for *full speed*: a controlled measurement freed
5.4 GB by stopping the other GPU tenants and re-rendered the same chunk,
which took 554.78 s against 554.72 s before — a 0.06 s difference. VRAM
headroom above this floor buys nothing; resolution is what costs time (see
the README's render-resolution section). Note also that ComfyUI logs `Using
async weight offloading with 2 streams` at every startup: that is a
capability announcement, **not** a report of memory pressure, and it is not
evidence that the card is short of VRAM.

The pre-flight floor (`min_free_vram_gb`) is **16 GB**: the lowest free-VRAM
figure at which a render has ever demonstrably succeeded. It has now been
wrong in both directions — 90% of nominal VRAM (too strict, would have
refused every working run), then 12 GB, which was below anything ever
demonstrated and on 2026-08-07 green-lit a run onto a contended card that
went silent rather than raising CUDA OOM and wedged the host. Set the floor
at the worst configuration observed to work: never below because less might
do, never above because more ought to be needed.

Note the check is **point-in-time**. It cannot see a process claiming the card
mid-run, which is how that incident actually started. See issue #23.

Two caveats on that measurement:

- It is for **~124-frame (5s) clips at the default resolution only.** Nothing
  longer or larger has been rendered on this card. Temporal VAE decode memory
  scales non-linearly with frame count, and chunks may reach 362 frames.
- Extra VRAM headroom does **not** buy speed — see the 0.06 s measurement
  above. Stopping tenants is worth doing to avoid a *wedge*, not to go faster.
  Resolution is the lever on run time (see the README).

Check `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`
before blaming the pre-flight for a VRAM shortfall.

**Release must always run**, even on a crash, a Ctrl-C, or a dead-lettered
run — and restarting whatever you stopped is on you, so don't leave a tenant
paused because a render ended badly.
