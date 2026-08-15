# music-video-pipeline

Python orchestrator that turns a master audio track + lyrics file + cast reference photos + a narrative concept into a complete, lip-synced music video by driving a local ComfyUI instance running MiniMax H3.

## Source-of-truth documents

- `comfyui-setup-summary.md` — the *actual* state of the ComfyUI install on doris (versions, model files, limitations).
- GitHub issues #1–#19 — the work breakdown. Issue comments contain reality-check corrections to the original bodies; read the comments, not just the body.

## Architecture (5 stages)

1. **Forced alignment** — `stable-ts` `model.align()` maps audio to the lyric text with word-level timestamps (`suppress_silence=True` VAD, `regroup=True`). `alignment_quality.py` scores the result and logs a summary at INFO on every run; `strict_alignment` turns a critical finding into a refusal. Alignment is ~6 s on CPU and a render is hours — always re-align and read the summary before spending GPU time.
2. **Temporal slicing + prompt expansion** — `pydub` slices per-segment audio stems (4–15 s window, pad/merge below min, split above hardware max); prompt = Global Style + Narrative Concept (or this chunk's shot-plan line) + active cast member role + literal lyric line.
3. **Asset staging** — upload cast images + audio stems to ComfyUI `POST /upload/image` (multipart; it ingests audio too); the *server-returned filename* is what goes in the workflow payload.
4. **Execution** — mutate `workflow_api.json`, `POST /prompt` with a UUID4 `client_id`, monitor via WebSocket, fetch output through `/history/{prompt_id}` + `/view`.
5. **Assembly** — FFmpeg concat demuxer (`-c:v copy`, never re-encode), strip ALL generated audio, mux the pristine master track over the video.

`--prepare` runs Stages 1–2 only (no GPU, no ComfyUI, no custody handoff) and writes a `shot_plan.toml` skeleton with `chunk_id`/`start`/lyric filled in and `shot` left blank (issue #52) — the anchors a shot plan is authored against, and the two things a generator must never invent. `--prepare --from-plan <plan.toml>` reads an existing plan **for its `length_seconds` only** and re-anchors the skeleton against the timeline a render with those lengths actually produces; without it, any plan setting a length describes chunks no render will emit and raises `ShotPlanDriftError`.

## Authoring layer (issue #54, all four phases landed)

`music_video_maker/authoring/` + the separate `mvm-author` console script propose a narrative concept, a beat sheet, the photography, and the shot lines — ending in a `shot_plan.toml` a human reviews and commits — through the Claude CLI, the one place in this project that is allowed to call a model. Validated 2026-08-13 by re-authoring "The Lucky Ones" from scratch against the real CLI. A 3-chunk validation slice of that plan (`run_v9.toml`, chunks 4/26/29) was rendered the same day — issue #58's findings — and it is the only part of a generated plan that has been rendered; **the full 36-chunk render has not.** Strictly outside the 5 stages above: no GPU, no ComfyUI, no custody handoff, and its output is a JSON/TOML file a human reviews and commits, never something read back into a render automatically. The render/authoring import boundary is enforced mechanically, not just documented — see `tests/test_authoring_boundary.py` and `music_video_maker/authoring/__init__.py`'s docstring.

- **Stage 2's value is that it is checkable, not that it writes well.** `beat_role`/`beat_group` make three of `docs/shot-writing-guide.md`'s checklist items machine-checkable *before* any prose exists (a consequence needs an earlier plant and contact in its own group; a consequence needs `focus = "action"`; a lone consequence is a cause and its effect in one shot). Every problem is reported in one message, because a retry round costs a whole model call.
- **A generated anchor is never trusted.** `chunk_id` is matched against the skeleton and anything else is dropped; `start`/`end` are re-emitted from the skeleton every time. That is what makes drift structurally unreachable for a generated plan, rather than merely unlikely.
- **A `length_seconds` re-cuts the timeline, so beats re-anchor by song time.** `authoring/reanchor.py` maps each beat to the chunk containing its midpoint — never by chunk id, the same rule `ShotLength` follows. Getting this wrong raises nothing; it silently attaches shot 12's direction to shot 9's audio. A chunk no beat maps into goes back to the model, never left blank — blank falls through to `narrative_concept` and renders as something deliberate-looking nobody authored.
- **Staleness cascades through the input hashes, not a dependency graph.** `beats` hashes the concept it descended from and `prose` hashes the beats, so re-rolling an upstream stage reports the downstream one stale by the same comparison that catches an edited lyrics file. Reported, never auto-healed.
- **A generated plan is checked by the render's own loaders, never a copy of them.** `plan.check_plan` writes the candidate to a temp file and runs `load_shot_plan`, `shot_length_requests`, `resolve_shot`/`resolve_camera` and `lint_shots_against_lyrics` with a log handler attached. Those lints encode measured lessons; a second implementation drifts within a month.
- **Errors and warnings are two tiers on purpose, and asymmetric.** Errors get a targeted revision of the offending chunks, bounded at 2 rounds, then abort with nothing written. Warnings get exactly one round — written as a single `if`, not a bounded loop, so there is no number to tune upward — and survivors are annotated into the file as `# lint:` comments. Every one of these lints is documented as "a false positive must never block a run"; a loop that grinds them to zero will rewrite a correct shot to please a heuristic.
- **A targeted revision must return only what it was asked to change.** `prose.revise_prose` shows the model the whole beat group so the rewrite stays coherent, but validates against the objected chunks alone and returns only those; merging is the caller's. A retry that quietly rewords thirty approved shots destroys the value of the approval before it.
- **A lint that guesses loses to a stage that knows.** `_lint_consequence_focus` infers "this reads like a consequence" from keywords; the beat sheet has the answer. On the first real run the keyword lint fired on four chunks the beats stage had called plants and transitions, they went into the revision round, and the rewrite put "She" back into the subject slot of lines that had correctly led with the object. Where structure contradicts the heuristic, structure wins and no objection is raised.
- **Every stage that writes must see `global_style`.** It is the junk drawer #53 split up, so it carries editing and camera constraints ("ONE continuous unbroken take, no cuts") and it is composed into every rendered prompt regardless. A photography stage that has not seen it proposes locked-off frames for a run whose style says the camera follows her.
- **Anchors are copied from the chunks at every layer.** Stage 2 re-emits `start`/`end` from the skeleton, and `render_plan_toml` takes them from the frozen timeline, not from the beats. Both `prose` and `write` then re-slice and *verify* the beat sheet's anchors still match before doing anything — `ShotPlanDriftError` caught where re-running a stage is the cheap fix, not hours into a render.
- **Grammatical subject is necessary but not sufficient (issue #58).** The v9 validation slice put a printer in subject position for both its plant and payoff beat, exactly as `docs/shot-writing-guide.md` asks, and H3 rendered neither — both lines also staged it small and far away ("small against the tower far behind her", "far behind on the street"). A third beat in the same slice, staged near and large, rendered correctly. `docs/shot-writing-guide.md`'s "Stage the object near, not far" and `shot_plan._lint_distant_staging` are the fix; `prompts.PROSE_PREAMBLE` carries the same rule. Confirmed by hand-editing chunks 4 and 29 to stage the printer near and re-rendering: both produced the printer for the first time, on the second try.
- **A generalisation from one chunk is a hypothesis; the full render is the experiment (#60).** #58 made `focus = "action"` on a voiced chunk the primary face-away signal, generalising from a single chunk re-rendered three times. The first full 36-chunk render tested it on 7 at once: all 7 turned away as predicted and **all 7 kept their lip-sync**. The mechanism was real, the predicted cost was not. The one chunk that did lose sync was a `transition` beat with `focus` unset whose *prose* did the same job — "Behind them, ... as they keep walking". So the lint was wrong in both directions at once, and the predictor lives in the sentence, not the field. Its replacement keywords were **scored against those 36 real lines before shipping**: "keep walking"/"behind them" match only the true positive, "receding" was excluded because it matches a chunk that synced fine, and bare "behind her" was excluded because it is in 20 of 36 lines that were all correct. Pick lint keywords by measurement on a corpus with a known outcome, and record the *excluded* candidates with their reason.
- **A realism LoRA and a flattering `appearance` are not both satisfiable (#62).** The fal H3 realism adapter at strength 0.8 costs ~1.5% wall clock and does make faces more photoreal — and it fights `global_appearance = "flatteringly lit"` and a de-aged `appearance` band, because those ask for the opposite of realism. Consistent with the adapter's own metadata (`training_strategy: text_to_video`, `first_frame_conditioning_p: 0.0` — it has likely never seen the reference-image conditioning `appearance` exists to modulate). Not a defect: a run must decide whether it wants photoreal faces or flattering ones, and asking for both means the stronger signal wins silently. Measured as a proper A/B — two configs identical but for three lines, separate output dirs, `lora`/`lora_strength` in the fingerprint's inescapable tier so a resume cannot mix arms.
- **A prohibition can cause the bug it was meant to prevent (#59).** `PROSE_PREAMBLE` forbids naming a cast member in a shot line — correctly, since the render composes names itself — which leaves a pronoun as the only way to refer to a second character. 9 of 36 generated lines said "him"; on the 7 where he was not also the singer, nothing bound the pronoun to anybody and H3 invented a different man each time. `chunk.characters` answers "who is *singing*" (the alignment knows that); `ShotPlanEntry.present` is what answers "who is *on screen*", and it carries role, appearance and reference photo into the prompt. Same shape as "a property that must hold needs its own field", one level down: free text in `shot` resolves against nothing.
- **A shot line may not refer to another shot (#61).** Each chunk renders from its own prompt, by a model that has never seen any other shot. "The printer teeters on the sill of *that same empty frame*" pointed at a window broken two chunks earlier and rendered no printer at all, while the shots either side rendered theirs. `_lint_anaphora` and `PROSE_PREAMBLE` rule 10; re-describe every object in full, every time.
- **A voiced consequence chunk's lip-sync loss is not a wording problem.** The v9 slice's one working consequence shot left her back-to-camera for most of a chunk carrying a real lyric line. Three follow-up re-renders of that exact chunk each changed one thing — two rewrites of `camera`, one rewrite of `shot` to stop her walking and say she was still facing the lens — and all three still turned her away by roughly the shot's midpoint. The one thing held constant across all three failures was `focus = "action"`, which is what hands the sentence's subject to the consequence in the first place. The fix isn't camera wording (tried three times, failed three times); it's `prompts.BEATS_PREAMBLE` steering a `consequence` beat onto a nearby INSTRUMENTAL chunk when one is available. **Superseded in part by #60 above — read that first.** The camera-wording finding stands, but the `focus = "action"` lint built on it did not: the full render showed turning away is common and usually costs nothing, so that check is retired and `focus = "action"` on a voiced chunk is no longer something to avoid.

## Non-negotiable invariants

- **Lyrics are immutable truth.** Forced alignment only — never ASR transcription of the vocals.
- **The master audio track is the only audio in the final video.** Generated audio is always discarded; sync comes from alignment timestamps, not generated sound.
- **Never hardcode ComfyUI node IDs.** Locate nodes by `class_type` (plus title disambiguation) via graph introspection; user edits on the canvas renumber IDs.
- **No sleep-polling.** Execution tracking is event-driven over the WebSocket; completion = `executing` message with `node == null` and matching `prompt_id`.
- **One chunk failing must not kill the run.** Resilience state machine: watchdog timeout → `POST /interrupt` → `POST /free` → bounded retry → dead-letter queue; runs are resumable.
- **The chunk timeline must cover the whole track.** Rendering only voiced spans looks fine per chunk and is silently broken: Stage 5 concats end to end, so every skipped instrumental is squeezed out of the video while the muxed master keeps it. A typical song is under half vocals. `instrumental_coverage` (default on) fills the gaps and re-anchors every chunk so video offset == audio offset by construction. This includes the zero-segment case — a wholly unvoiced track, or a voiced one authored with nobody ever shown singing (an empty `lyrics_file`) — which `slice_audio` tiles as filler across the *entire* track rather than the pre-2026-08-12 behavior of returning no chunks at all.
- **A resumed chunk must prove it belongs.** "The mp4 exists" is not "the mp4 is
  the right mp4". Every `ChunkResult` records a `ChunkFingerprint` (span, frame
  count, resolution, prompt hash, character, reference photo) and `--resume`
  compares it; a chunk whose span moved is re-rendered, because reusing it
  assembles a desynced video with no error anywhere. Editing the lyrics file is
  enough to trigger this. `run_state.json` carries a `schema_version` so a file
  written before fingerprints existed is rejected, not misread.
- **A property that must hold across the whole video needs its own field.** Stated in forty places, it drifts: geography wandered between countries until `setting` existed, and presentation direction smuggled into `role` is what made the performer sing through every instrumental, because `role` is injected into every chunk. `setting`, `global_appearance` and `CastMember.appearance` compose into every prompt. Adding free text to an existing field to carry a new *kind* of statement is how that bug recurs.
- **On the chained path, the seed frame IS the identity conditioning.** `MiniMaxH3ImageToVideo` has no `ref_images` input at all, so a chained chunk's likeness comes entirely from the predecessor's final frame. A frame showing the back of the performer's head carries *nothing*, and the model invents a face when she turns round — 7 of 12 chained seeds in one real run were like that, because `i2v_chain_scope = "instrumental"` chains exactly the walking shots. Face-presence is a precondition for chaining (`i2v_require_seed_face`), not an assumption; `i2v_reanchor_interval` is only a positional guess at accumulated drift. Detection is not recognition — a large face belonging to someone else still passes (#47, #49).
- **Chained rendering makes every per-chunk instruction a recurrence relation.** Three bugs in one week came from this: `last_frame` correct as a keyframe and catastrophic as the next chunk's seed (#44); `appearance` correct against a photo and compounding against its own output (#46); a seed frame assumed to contain the thing it conditions (#47). Before adding any prompt component or graph input, ask what it does on the *fifth* application to its own output. Anything relative — anything that displaces rather than specifies — will drift.
- **A guard's threshold belongs near the evidence, on the side whose mistake is cheap.** #47's confidence floor shipped at 0.6 justified as "well clear" of detections scoring ≥0.835 — backwards, since a false-positive guard set *below* the evidence is exposure, not margin. It passed a blank office wall at 0.669 and cost a shot. Ask which error is cheap: refusing a chain costs a better-conditioned base-path shot, approving a bad one costs the likeness.
- **The GPU can be taken from under a running render.** The custody pre-flight is point-in-time; `ResilientRunner` re-reads free VRAM before every chunk it submits and stops the run cleanly below the floor, because H3 staging onto a contended card can go silent instead of raising CUDA OOM and wedge the host. A failed chunk is cheap; a power cycle that lands in Windows is not.
- **No LLM in the render path.** `prompting.py` is pure string composition. Per-chunk narrative direction comes from an authored `shot_plan.toml` written once and committed as data — never an inference call at render time, which would break determinism, `--resume`, and the no-metered-API rule. The sanctioned exception lives entirely outside the render path: `music_video_maker/authoring/` (issue #54) calls a model, once per song, by hand, and writes a file a human commits — see "Authoring layer" above. `tests/test_authoring_boundary.py` enforces the boundary between the two, not just this sentence.

## Everything committed here is intended to become public (#51)

This repo is a candidate for open sourcing. Both blockers found in the
2026-08-12 audit were introduced *the same day*, by changes that felt too small
to think about — a test fixture and a 232 KB model file. Assume anything
committed ships, and check before adding, not before releasing.
`tests/test_repo_assets.py` enforces the first two mechanically.

- **No likenesses of real people without consent.** A rendered H3 frame
  derived from a cast photo depicts a real, identifiable person; publishing it
  is *their* decision, not a repo-hygiene question. #47's regression fixtures
  originally depicted a cast member whose permission to publish had not been
  asked for — resolved 2026-08-14 (#51 §2) by re-rendering the same
  regressions from `storms` with Derek's own likeness instead, since he's both
  the subject and the one publishing the repo. `Asset.__init__` in `test_repo_assets.py` now
  refuses any `depicts_real_person=True` entry with no recorded `consent`. For
  anything new, prefer a synthetic or consented subject — and if only real
  footage proves the bug, keep the test skip-if-absent so the fixture can be
  removed without gutting the suite.
- **Check redistribution before committing a third-party binary** — models,
  weights, fonts, sample media. Record source, licence and sha256 beside it in
  code, the way `faces.py` does for YuNet. "It downloaded fine" is not a
  licence.
- **No credentials, ever.** `.env` is gitignored; keep it that way. Never
  commit a session file, token or key, including inside a test fixture.
- **Internal network detail gets a placeholder.** Tailscale hostnames are fine
  — they are meaningless outside the tailnet. Raw LAN IPs and anything that
  maps the home network are not.
- **Naming doris and the song is fine, and usually better.** A measured
  number is worth more attached to real hardware than sanded into "your GPU".
  What is not fine is a doc where a stranger cannot tell which details are
  essential and which are this setup.
- **Authoring session state stays out too.** `.authoring/` (issue #54) —
  `session.json`, generated stage JSON, and especially `raw/` (full,
  unredacted model replies, which routinely include the whole lyrics file) —
  is a run asset like `output/` and `run.toml`, not project code. It lives in
  `~/mvm-runs/<song>/`, never gets created inside a repo checkout, and is
  gitignored belt-and-braces anyway.

## Performance (measured, not assumed)

- **Resolution is the dominant cost**, not frame count — it scales with latent volume. At 141 frames on the 4090: **864×480 ≈ 3.7 min/chunk**, **1344×768 = 9 m 15 s** (measured). A 4:22 song is ~2.7 h vs ~6.8 h.
- `workflow_api.json` carries ComfyUI's template default of 1344×768. Set `render_width`/`render_height` explicitly; hands-on testing was done at 864×480.
- **VRAM headroom buys no speed.** Freeing 5.4 GB changed the same chunk from 554.72 s → 554.78 s. ComfyUI's startup line `Using async weight offloading with 2 streams` is a capability announcement, *not* memory pressure. Don't chase it.
- SageAttention is **not installed** on doris despite appearing in `recommended_nodes`; ComfyUI runs `pytorch attention`.

## Infrastructure (doris)

- ComfyUI v0.30.2 at `/home/derek/ComfyUI` (Python 3.12 venv, torch 2.13.0+cu130, RTX 4090). Reach it as **`http://doris:8188`** over Tailscale — never a raw LAN IP. Runs as the `comfyui.service` systemd unit (issue #17), bound to loopback + its Tailscale address with ufw allowing 8188 only on `tailscale0` — reachable across the tailnet, invisible to the LAN. Never rebind it to `0.0.0.0`: ComfyUI has no authentication and can load custom nodes and write files, so an open bind is RCE for the whole network.
- MiniMax H3 support is **core ComfyUI** (`comfy_extras/nodes_minimax_h3.py`) — not a custom node. `comfy_api_nodes/nodes_minimax.py` is the cloud-API variant; do not use it.
- H3 weights already installed (~87.6 GB: pruned-INT8-ConvRot DiT, plus **both** the NVFP4 AWQ and INT8 ConvRot text encoders since 2026-08-10, issue #39). `workflow_api.json` and `workflow_i2v_api.json` are committed (issue #18), authored from ComfyUI's built-in MiniMax H3 templates — which is why they carry its 1344×768 default; set `render_width`/`render_height` explicitly.
- **Which encoder a run uses is config, not a template edit.** `text_encoder` pins `CLIPLoader.clip_name` (a name inside doris's `models/text_encoders/`, never a local path) across *both* templates, and the resolved value is recorded in every `ChunkFingerprint` in the same inescapable tier as the conditioning audio — the encoder is #39's experiment variable, so `--resume` must never mix precisions inside one video. Only the encoder's 50 language layers are quantised; its vision tower is BF16 in both builds.
- Disk on doris runs high-90s % full; check `df -h` before writing large outputs there.

## GPU custody protocol (issue #19)

A render run takes **exclusive custody** of the 4090:

- Before rendering: stop the card's other tenants **by hand**, then verify VRAM actually freed. The pipeline does not automate this and deliberately never will — see `custody.py`'s module docstring. The specific commands for this host are operator knowledge, not repo content.
- Hardware profile (#13) therefore assumes the full 24 GB.
- After the run: the pipeline `POST /free`s ComfyUI to unload H3 weights; restarting what you stopped is yours.
- **Release is unconditional** — try/finally around `POST /free`, so a crash, Ctrl-C or dead-lettered run still returns the card.

### Stopping the tenant you know about is NOT sufficient (learned 2026-08-07)

The card has other tenants. Who they are changes over time, so **enumerate them, never assume**:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
systemctl list-timers --all          # anything scheduled can arrive mid-run
```

Current tenants (2026-08-08):

| Consumer | VRAM | Trigger |
|---|---|---|
| Ollama / Qwen (Docker container) | ~15 GB | on demand; `OLLAMA_KEEP_ALIVE` holds it resident |
| a TTS service | ~2.2 GB | always-on service |
| desktop session (Chrome, nautilus, ptyxis) | ~0.5 GB | always |

A stack of four scheduled services — the largest an image model on a 30-minute timer — was retired 2026-08-08, stopped and disabled permanently at Derek's request. It was the biggest scheduled hazard on the card, so a long run is materially safer now. Do not re-enable it to "restore" anything; its absence is deliberate.

That leaves Qwen as the one consumer big enough to matter. It is *on demand*, so a quiet card is not proof of a free card — one request pulls 15 GB back.

**Do not stop the Ollama container to keep it quiet.** Pausing the service that fronts it unloads Qwen from VRAM, which is the sanctioned move; stopping the container takes the endpoint down for *every* consumer and the service then refuses to resume until it can warm the model again. The per-service commands are operator knowledge.

**Failure mode to respect:** when H3 stages its 19995 MB onto a contended card it does not reliably raise CUDA OOM — it can go **silent** mid-load and wedge the host. The process then survives SIGKILL (stuck in a driver call), the modules will not unload, and only a power cycle recovers. Rebooting doris is expensive: the board ignores BootOrder and frequently lands in Windows, needing manual `bcdedit` recovery from another machine. A wedge can strand the box. See #23, #24.

### Kernel upgrades are frozen (2026-08-08)

`apt-daily`/`apt-daily-upgrade` are enabled, but `/etc/apt/apt.conf.d/50unattended-upgrades` now blacklists `linux-image`, `linux-headers`, `linux-generic`, `linux-hwe`, `linux-modules`. Everything else — including `linux-firmware-*` — still patches automatically.

Why: unattended-upgrades installed a kernel (`-28` → `-29`) while the box stayed up. The NVIDIA modules package remained pinned at `-28`, so the *next* reboot, days later, came up with no NVIDIA driver at all — `nvidia-smi` dead, GPU gone, display on a fallback framebuffer. The breakage is armed silently and only fires on reboot, which makes it very easy to blame on whatever was running at the time. It cost an hour of misdirected diagnosis: it presented as "the render killed the GPU."

Updating the kernel is now a deliberate act:

```bash
sudo apt update && sudo apt install linux-generic-hwe-26.04
dkms status          # confirm NVIDIA modules built for the NEW kernel
sudo reboot          # only after dkms looks right
```

Backup of the original config: `/etc/apt/apt.conf.d/50unattended-upgrades.bak-preblacklist`.

## Stack & testing

- Python 3.10+; deps: `stable-ts`, `pydub`, `websocket-client`, `requests`, `ffmpeg-python`, `torch`. FFmpeg must be installed at the OS level.
- Tests run fully offline (global standard): mock ComfyUI harness (issue #16) fakes `/upload/image`, `/prompt`, `/history`, `/view`, `/interrupt`, `/free` and replays scripted WebSocket sequences. No GPU, no network, no live server in CI.
- Telegram credentials (api_id, api_hash, session file) live in `.env` — never committed.
