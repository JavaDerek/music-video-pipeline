# Synthetic cast: characters that are nobody (issue #56)

`CastMember.image` requires a photograph of a real person. That is a hard
prerequisite for using this project at all, it is why #51's likeness question
exists, and it means every performer in every video must be someone who exists
and consented.

This needs an image model, and that is a real decision with a disk cost, a
licence question and an unsolved consistency problem — the wrong version of
this feature is a one-off script that makes a pretty portrait and leaves the
hard part untouched. That decision is still nobody's to make here.

> **The model decision was made on 2026-10-04: Krea 2.** Everything from
> "Part 2: generating the imagery" below is built; the sections before it are
> the design as it was written, kept because the reasoning is what the build
> follows. Where the two differ, the later section says so and why.

## Status (2026-09-13)

Built, in the work package that does not depend on the model decision:

* `synthetic`/`[cast.<name>.origin]` on `CastMember` (`contracts.py`,
  `config.py`) — real/invented is queryable in config, exactly as proposed
  below. `RunConfig.real_likenesses()` answers #51's own question; `cli.main`
  logs it at run start when non-empty.
* The consistency acceptance check this doc calls for (`music_video_maker/
  castcheck.py`, `python -m music_video_maker.castcheck`) — pairwise SFace
  similarity over a reference set, reusing `faces.recognize_face` exactly as
  proposed, with the mode-collapse caveat printed in every report rather than
  left to be remembered.

Still open as of 2026-09-13 — **and the first two were closed on 2026-10-04**;
see "Part 2: generating the imagery" below:

* ~~The model decision itself (SDXL+IP-Adapter / Flux / hosted API / reuse),
  and its licence.~~ **Krea 2**, already on the host, under the Krea 2
  Community License.
* ~~Actually generating a character~~ — `castgen.py` generates one; **no
  generated image is committed here**, by design (run assets).
* Re-deriving the 0.34 floor on synthetic pairs (the check enforces the
  caveat; it cannot do the re-derivation for you before synthetic material
  exists to measure). **Still open**, and now possible.
* The cross-video / profile question in "What 'done' looks like" item 4.
  **Still open.**

## What it actually solves

1. **The barrier to entry.** A stranger cannot run this today without
   photographs of willing performers. That is a bigger obstacle than any
   install step.
2. **#51's likeness blocker.** Fixtures depicting a real person are that
   person's decision to publish. A synthetic subject makes the question
   disappear for every future fixture. (The existing ones were resolved on
   2026-08-14 by re-rendering from Derek's own likeness — the subject and the
   publisher being the same person — so this is about the *next* fixture, not
   the current ones.)
3. **The supporting cast, which is the underrated one.** Every extra in a
   render is invented fresh by H3 in every shot, so they change between cuts —
   and a bystander is the direct cause of #49's hardest failure: the seed-face
   gate approved a frame because a *stranger's* face was large and confident
   while the lead had her back turned. Named, photo-backed supporting
   characters would make extras consistent **and** make identity checks
   meaningful.
4. **New performers cost nothing.** Today a new song means sourcing and
   possibly retouching photographs before any work can start.

## The problem is consistency, and it has a measurable definition

One flattering portrait is easy. A character who reads as the same person
across 80 shots is not — and "reads as the same person" is exactly the thing
this repo already has an instrument for.

`faces.recognize_face` computes SFace cosine similarity, calibrated on 16 real
pairs: **worst genuine 0.353, strongest impostor 0.208, floor set at 0.34**
(see `docs/seed-face-recognition.md`). That gives a synthetic character an
acceptance criterion that is not a matter of taste:

> A generated character is usable iff every reference image in its set scores
> **≥ 0.34** against every other, and iff frames rendered from it score ≥ 0.34
> against the reference the render was conditioned on.

That is an offline check, on a laptop, with weights this project already
documents. It also gives the feature an honest failure mode: if a generator
cannot clear that bar, the character is not ready, and you find out before a
render rather than after.

**Built:** the reference-set half — `music_video_maker/castcheck.py`
(`python -m music_video_maker.castcheck <character> <img1> <img2> ...`).
Scores every pair with `faces.recognize_face`, excludes and surfaces any
image whose `FaceObservation.verdict` is not `detected` rather than averaging
it in or scoring it as 0, refuses to pass vacuously on a set of one, and
prints the mode-collapse caveat in every report. The second half — "frames
rendered from it score ≥ the floor against the reference the render was
conditioned on" — is **not** a new instrument: `facescan.py` already has a
per-frame recognition-shaped path (it calls `faces.detect_faces` per sampled
frame today; wiring a reference photo through to `faces.recognize_face` the
way the #49 seed-face gate already does is the same shape of change, not a
second tool) and is where that half belongs once there is a rendered chunk
and a character to check it against.

Two caveats to carry into any such measurement:

* **A zero from the detector is not "no face".** A bare 0.0% from
  `faces.detect_faces` has only ever meant "nothing cleared the 0.9 gate"; use
  the `FaceObservation.verdict` (`detected` / `inconclusive` / `absent` /
  `unexamined`, issue #93) and `music_video_maker.facescan`, which stamps the
  input it actually read — the one documented "YuNet cannot see hats" case
  turned out to be a scan of the wrong render. A synthetic character that
  cannot be *detected* cannot be *recognised* either, so an `inconclusive`
  verdict must be looked at, not averaged in. Check the frame before believing
  a zero.
* **The floor was calibrated on photographs of real people.** Generated faces
  may have a different similarity distribution entirely (generated faces are
  often *more* similar to each other than real photographs of the same person
  are — a mode-collapse artefact, not identity). Re-derive the floor on
  synthetic pairs before trusting it; do not assume 0.34 transfers.

## One reference image, or a set?

`ref_images` is a `COMFY_AUTOGROW_V3` list and the pipeline already supports
multiple references per chunk (#33). So a set is possible without new graph
work, and a set is probably right: a single portrait fixes one pose, one lens
and one light, and #74 measured how strongly the *photo* decides things nobody
wrote down (a smiling reference photo carried eight and a half minutes of a
video about a massacre until `demeanour` existed to override it).

Decide early, because it changes the generator's job from "make a good
portrait" to "make N views of one person". Suggested v1 set, chosen so each
image answers a different question the render will ask:

* one neutral frontal, evenly lit — the identity anchor;
* one three-quarter — the angle most shots actually use;
* one at the video's own lighting (the `cinematography` grade) — because the
  reference is also a lighting reference, whether or not that was intended.

**Do not generate a set by re-rolling one prompt at different seeds.** That is
four different people. The methods that hold identity are img2img/inpaint from
the first image, an IP-Adapter-style conditioning on it, or a small LoRA
trained on it — each with a different cost and a different licence.

## The model decision

Three constraints, and they interact:

* **doris is at high-90s % disk.** "Install another model family" is not free;
  #39's storage squeeze is the precedent. H3's weights are already ~87.6 GB.
* **H3 cannot bootstrap this.** It generates video *from* a reference, so it
  cannot produce the reference. This is genuinely a new dependency.
* **The output's licence matters, not just the weights'.** A repo intended to
  go public cannot ship a fixture generated by a model whose licence restricts
  the use of its outputs. Record source, licence and sha256 beside anything
  committed, the way `faces.py` does for YuNet and `workflow_graph.KNOWN_LORAS`
  does for the realism adapter. "It downloaded fine" is not a licence.

Options, with the question each one answers:

| Option | Disk | Licence question | Consistency story |
|---|---|---|---|
| SDXL + IP-Adapter on doris | ~7–12 GB | SDXL weights are permissive; check the adapter's | good, well-trodden |
| Flux-class model | ~24 GB+ | several are non-commercial — check before generating a public fixture | very good |
| A hosted image API | 0 GB | outputs usually fine; **but** it is a metered model call and a network dependency | good |
| Reuse an existing local model | 0 GB | already resolved | unknown |

A hosted API is worth naming explicitly rather than dismissing: this is
authoring-time, once per character, entirely outside the render path — the
same boundary `mvm-author` already lives on. It does not violate "no LLM in the
render path", and it makes the disk problem disappear. It does add a second
metered dependency to a project whose cost story is currently "GPU hours and
electricity", which is a real thing to weigh, not a blocker.

**Recommendation: settle the licence question first, then pick.** The cheapest
path to a *decision* is to generate one character three ways and run the
similarity check above; the cheapest path to a *feature* is whichever of those
clears the bar.

## Provenance, and telling real from invented

Record how a character was made, so it can be regenerated or extended: the
model, the prompt, the seed, the sampler settings, and the date. Same reasoning
as #34, #38, #45 and #54 — an input that decides the pixels with nothing
writing it down is this project's most-repeated bug.

And answer #56's own open question **yes**: real and invented cast members must
be distinguishable in config, so a #51-style likeness question can be *queried*
rather than remembered.

**Built**, in `contracts.py` (`CastMember.synthetic`/`CastMember.origin`,
`CastOrigin`) and `config.py` (`_build_cast_origin`, `CAST_KEYS`):

```toml
[cast.Nobody]
role  = "Lead vocalist"
image = "cast/nobody_ref_01.jpg"
synthetic = true                     # default false: every existing config is real
[cast.Nobody.origin]                 # required when synthetic = true
model  = "…"
prompt = "…"
seed   = 12345
created = "2026-08-23"               # bare TOML date or a quoted ISO string; both normalize
# sampler = "…", steps = …           # free-form extra scalar keys, kept verbatim
```

`synthetic = true` with no `[cast.<name>.origin]` is refused at load, exactly
as proposed: "this character is invented" with no record of how is the
provenance bug in a new place. The mirror image is refused too, which this
doc left implicit — `[cast.<name>.origin]` present while `synthetic` is not
`true` is a config error, because an origin on a real person's photo is a
contradiction, not metadata. `synthetic = false` (the default) keeps every
existing config loading unchanged, and it is the value that means "a real
person's likeness is in this run" — which is the query #51 wants to be able
to run: `RunConfig.real_likenesses()`, logged at INFO when non-empty by
`cli.main` at run start.

**One interaction this doc didn't anticipate: `voiced_by` (issue #89, landed
after this doc was written).** A character with no `image` of its own borrows
its performer's photo — so whose photo is it, for `synthetic` purposes? Two
options were on the table: refuse `synthetic`/`origin` on such an entry
outright, or let it inherit the performer's facts. Shipped: **both, by case**.
An entry with no image of its own may not set `synthetic`/`origin`
explicitly — refused at load, naming #56 — and instead inherits the
performer's `synthetic`/`origin` automatically, along with the photo itself,
so `real_likenesses()` and the CLI log line are correct for it with no special
case. An entry that sets its *own* `image` (overriding the fallback, already
legal per #89) may set its own `synthetic`/`origin` too, independent of its
performer — it owns a different photo now, so it owns a different provenance
question.

The flag is also the natural place to hang a future consent field for the real
case, mirroring `tests/test_repo_assets.py`'s `Asset(depicts_real_person=…,
consent=…)`, which already refuses an unconsented likeness mechanically. Not
built here — `synthetic`/`origin` answer "real or invented", not "consented".

## Does an invented character still need `appearance`?

Probably much less, and this is testable rather than arguable. The README
already notes the photo is a far stronger lever than any adjective, and #62
measured the two fighting: a realism LoRA and a flattering `appearance` are not
both satisfiable, and the stronger signal wins silently. If the reference is
generated to specification, `appearance` has nothing left to compensate for,
and #46's warning applies with full force — an `appearance` that *displaces*
("looking a few years younger") rather than *specifies* is a recurrence
relation on the chained path.

`demeanour` is a different matter and should be kept: #74 measured that it is
behavioural direction, that it doubled face presence across a whole render
(29.2% → 59.1% mean over 80 chunks), and that its strength scales with how much
face it names. A generated reference does not fix expression across eight
minutes of video.

## What "done" looks like

1. A character is defined in a small authored file (prompt, model, seed,
   which views), generated once, and its images committed as **run assets** in
   `~/mvm-runs/<song>/cast/` — not in the repo, exactly like every other run
   asset. **Open** — depends on the model decision; nothing here generates
   anything.
2. The similarity check above passes over the set, and its numbers are
   recorded next to the character. **Built** (`castcheck.py`) for the
   reference-set half; nothing to run it against yet.
3. `synthetic = true` + `origin` in the run config; the render path is
   unchanged, because `ref_images` still just receives a photo — of nobody.
   **Built** — and confirmed unchanged: `synthetic`/`origin` are not
   fingerprinted (they don't move pixels; the image path/content already is
   fingerprinted) and are not profile fields (confirmed by
   `tests/test_profiles.py`), so neither `--resume` nor a cinematography
   profile is affected.
4. A cross-video story, since a character is a cross-video asset exactly like a
   locked house style (#55). Whether they share a mechanism is worth
   considering, but note the difference: a profile is *text* that resolves into
   config fields, while a character is *binary assets plus text*. The profile
   mechanism does not extend to that for free, and forcing it to would make the
   simple half worse. **Still open**, unchanged by this work package.

## Part 2: generating the imagery (2026-10-04, built)

`music_video_maker/castgen.py` (`python -m music_video_maker.castgen`) is the
generator this doc deferred. It is authoring-time, outside the render path, and
nothing in the package imports it — a test enforces that
(`tests/test_authoring_boundary.py::test_nothing_in_the_package_imports_the_image_generator`),
which is how "does the render binary ever call a model?" stays answerable by
reading an import graph rather than this paragraph.

### The model: Krea 2, because it is already on the host

Decided by Derek on 2026-10-04. It answers the disk constraint by being
already present (`/data/huggingface/hub/models--krea--Krea-2-Raw/snapshots/*/raw.safetensors`,
26.28 GB, plus a `Krea-2-Turbo` of the same size), and ComfyUI v0.37.4 on
doris supports it natively — `comfy/ldm/krea2/model.py`,
`comfy/text_encoders/krea2.py`, a `krea2` entry in `comfy/model_detection.py`
and `comfy.supported_models.Krea2`. No custom node is involved.

**Krea 2 Community License**, recorded in the repo's rights table (README's
own `## License` section) and in `castgen.KNOWN_IMAGE_MODELS` beside the code
that loads the weights, the way `faces.py` records YuNet's and
`workflow_graph.KNOWN_LORAS` records the realism adapter's:

| Term | What it obliges here |
|---|---|
| Commercial use free under **$1M annual revenue and under 50 seats** | Beyond either, the deployer needs Krea's commercial terms. Written into every manifest, so a published asset carries the condition it was made under. |
| A **derivative model's name must begin with "Krea"** | Nothing here trains or merges a model, so nothing here can breach it — but a cast LoRA ever trained on this output must be named `krea2-…`. |
| **No classifiers ship with the open weights**, so filtering is explicitly the deployer's responsibility | No classifier ships here either, and the provenance says so rather than being silent: every manifest records `filtering = "none — …"` and every generate run logs it at WARNING. A human looks at every image. |

That third term is why `castgen` does not pretend to a verdict it has not
earned. A provenance record that says nothing about filtering reads, later, as
"something checked this".

### The method: one anchor, then img2img views

This doc's "do not generate a set by re-rolling one prompt at different seeds"
is now **mechanical**, not advisory: a spec whose second view has no `from`, or
whose derived view sets `denoise >= 1.0` (which replaces the anchor latent
entirely — a re-roll with extra steps), is refused by `load_character_spec`
before anything is submitted. So is a set of one view, because `castcheck`
calls that `insufficient` and finding that out before the GPU time is the whole
point.

Of the three identity-holding methods named above, **img2img from the anchor is
the only one available with core nodes and the weights installed on doris**:

* ComfyUI's own shipped blueprint `Text to Image (Krea-2 Turbo)` gave the graph
  shape the committed templates follow (`UNETLoader` → `KSampler`,
  `CLIPLoader type="krea2"` → `CLIPTextEncode` → positive,
  `ConditioningZeroOut` → negative, `EmptyLatentImage`, `VAELoader` →
  `VAEDecode`). Its defaults for `sampler_name`, `scheduler` and 1024×1024 are
  taken from there rather than chosen here; `steps` and `cfg` are **required**
  in the spec, because the blueprint's 8 / 1.0 are *Turbo's* and applying them
  to Raw would be a generation nobody authored.
* The sibling blueprint `Image Style Reference (Krea-2 Turbo)` is the
  reference-conditioning route (`TextEncodeQwenImageEditPlus` +
  `FluxKontextMultiReferenceLatentMethod`), and it needs
  `krea2_style_reference.safetensors` — **not installed**. The full list of
  LoRAs on the host on 2026-10-04 was `Flux_2-Turbo-LoRA_comfyui`,
  `h3-realism-people-t2v-i2v-r2v`, `krea2_darkbrush`,
  `minimax_h3_fl2v_turbo_8step_v1.0`. **List the directory rather than trusting
  that list** (`ls /data/ComfyUI-models/loras/`) — the only Krea 2 entry in it
  is `krea2_darkbrush`, whose trigger phrase in ComfyUI's own blueprint is
  "muted minimalist sketch style": a style adapter, exactly the wrong thing for
  a photoreal cast portrait, and #62 already measured a LoRA and the prompt
  fighting over a face. Neither committed template carries a LoRA node, and a
  test records that as a decision rather than an omission.

So: view 1 is text-to-image, every other view is
`LoadImage → VAEEncode → KSampler(denoise < 1)` from an **earlier view's
rendered file**. Two committed templates, never one rewritten —
`workflow_cast_api.json` and `workflow_cast_view_api.json`, the same split
`docs/workflow-template-guide.md` describes for base vs I2V. Node ids are never
hardcoded; everything is located through `workflow_graph`'s `find_one_node`,
and a test renumbers every node in both templates and expects identical output.

There is **no default `denoise`**. What value holds a face while turning a head
is unmeasured for this model, and a default that looks calibrated is this
project's most-repeated bug in a new place.

### Prerequisites on doris, which are not yet satisfied

Read from the live server's own `GET /object_info` on 2026-10-04 (metadata
only — it loads no weights and touches no GPU):

1. **The DiT is invisible to `UNETLoader`.** `raw.safetensors` lives only in
   the HuggingFace cache, and the loader's enum lists no Krea 2 file at all.
   Link it into `models/diffusion_models/` (the templates name
   `krea2_raw_bf16.safetensors`) and restart or refresh ComfyUI. Reading the
   safetensors header says why nothing else will do: the file is the **DiT
   only** — 364 `blocks.*` tensors at 24.32 GB, plus `txtfusion`/`tproj`/`tmlp`,
   and no `vae.` or `text_encoders.` keys — so it cannot be loaded as a
   checkpoint.
2. **The text encoder is missing.** Krea 2 conditions on Qwen3-VL-**4B** at a
   12-layer tap (`comfy/text_encoders/krea2.py`), loaded by
   `CLIPLoader type="krea2"`. `models/text_encoders/` has the 8B and 32B H3
   encoders and no 4B; the HF cache holds `Qwen/Qwen3-VL-4B-Instruct` only as
   two diffusers shards, which that single-file loader cannot read. Download
   the single-file encoder ComfyUI's blueprint names,
   `qwen3vl_4b_fp8_scaled.safetensors`.
3. **The VAE is present**: `qwen_image_vae.safetensors`
   (`supported_models.Krea2` uses the Wan 2.1 latent format).

`castgen` checks all three *before* taking custody, by reading each loader's
own enum (`GET /object_info/{class_type}`) and comparing it with what the
templates name. A missing file is refused with its name and the directory it
belongs in — rather than arriving as a `node_errors` blob after the operator
has already stopped everything else on the card. An unreadable answer degrades
to a warning (the same decision `custody._fetch_free_vram_gb` makes about an
unreadable VRAM reading), never a refusal.

### Running it

Both commands write into a *run's* own cast directory, never this repository.

```bash
# offline: validate the spec, build one workflow per view, write nothing to the card
python -m music_video_maker.castgen plan cast/nobody.castgen.toml \
    --out-dir ~/mvm-runs/<song>/cast/plan

# on doris, with GPU custody taken by hand first
python -m music_video_maker.castgen generate cast/nobody.castgen.toml \
    --out-dir ~/mvm-runs/<song>/cast --comfyui-url http://doris:8188
```

A spec is small:

```toml
character = "Nobody"
prompt = "studio portrait photograph of a woman in her thirties, short dark hair"
seed = 4242
model = "krea2-raw"      # resolves the licence record in KNOWN_IMAGE_MODELS
steps = 28
cfg = 4.0

[[views]]                # the anchor: text-to-image, no `from`, no `denoise`
name = "frontal"
prompt = "looking straight into the lens, neutral expression, even soft light"

[[views]]                # img2img from the anchor's own rendered file
name = "three_quarter"
from = "frontal"
denoise = 0.45
prompt = "head turned thirty degrees to camera left, same person, same face"

[[views]]                # the reference is a lighting reference too
name = "graded"
from = "frontal"
denoise = 0.35
prompt = "lit by hard low side light, deep shadows, same person"
```

`generate` writes the images plus three records: a human-readable report, a
JSON manifest (per-image sha256, seed, denoise, prompt, `prompt_id`, server
filename, the weights it checked, the free-VRAM reading, ComfyUI and torch
versions, the licence, the filtering disclosure, and the consistency numbers),
and a `[cast.<name>]` block ready to paste into a run config — `synthetic =
true` with the `origin` table part 1 already requires. A test parses that block
and validates it through `config`'s own origin builder, so the two cannot
drift.

Exit code 0 only when every view generated **and** `castcheck` passed.
`unchecked` (no SFace weights, or `--skip-check`) is not a pass.

### VRAM and custody

**The DiT alone is 24.32 GB of bf16 weights against a 24 GB card.** ComfyUI
will offload rather than hold it resident, together with the ~4.5 GB fp8 text
encoder and the VAE. **No Krea 2 generation has been measured on doris**, so
there is no Krea-2-specific floor and `castgen` invents none (`envelope.py`'s
rule: no evidence means no gate, not a refusal). What it does run is the
existing `custody.preflight_free_vram` — H3's measured 16.0 GB floor, used here
as the *is the card actually free* check it has always been — and
`POST /free` in a `finally`, so a character that fails halfway does not leave
the model resident on a shared 4090.

The operator steps are the same as a render's and are deliberately not
automated (see `custody.py`'s docstring): enumerate the card's tenants
(`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`),
stop them by hand, confirm the VRAM actually freed, generate, then restart what
you stopped. A shared card with an LLM resident is the normal state of this
host, and one request pulls 15 GB back.

### Does the 0.34 floor transfer to generated faces? Still unknown

`castgen` calls `castcheck` — it does not score anything itself — and records
the status, the floor, and the floor's calibration *as a sentence*: real
photographs, 16 pairs, **not re-derived on synthetic pairs**. It additionally
reports the pairwise min/mean/max as numbers with no verdict attached, because
the one thing `castcheck`'s own docstring says it cannot see is mode collapse:
views authored to differ in angle and light that all score near-identical may
be one face three times, and no threshold separates that from success. Both
caveats are printed in every report.

So the honest reading of a `pass` from `castgen` today is "nothing here
contradicts this being one person", not "this is one person". Re-deriving the
floor is the work this part finally makes possible: generate several characters,
score genuine pairs (views of one character) against impostor pairs (views of
two different characters), and put the measurement in
`docs/seed-face-recognition.md` beside the real-photograph one.

### What cannot be known until a real generation runs

* Whether Krea 2 Raw at 1024² fits a 24 GB card in practice, how far ComfyUI
  has to offload, and what a view costs in wall clock.
* What `denoise` band holds a face while changing its angle — and whether
  img2img holds identity *at all* on this model, or only holds composition.
* Whether the 0.34 floor transfers, and what an impostor pair of *generated*
  faces scores.
* Whether a generated reference conditions H3 as well as a photograph does —
  the second half of the acceptance criterion ("frames rendered from it score
  ≥ the floor against the reference the render was conditioned on"), which
  needs a rendered chunk.
* Whether `EmptyLatentImage` is the right canvas node for a Wan21-format
  latent. ComfyUI's own Krea 2 blueprint uses it, which is why the template
  does; nothing here has run it.

## The trap to avoid

Do not design this around the fixture problem. #51 is the loudest reason and
the smallest one; a fix scoped to "regenerate three test images" helps this
repo once and looks finished. The defect is that **the pipeline requires a
photograph of a consenting human before it can do anything at all**, which is
true of every user who is not the author, and true of every extra in every
shot — none of whom consented to anything, because none of them exist.
