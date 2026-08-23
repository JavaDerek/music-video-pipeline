# Synthetic cast: characters that are nobody (issue #56)

`CastMember.image` requires a photograph of a real person. That is a hard
prerequisite for using this project at all, it is why #51's likeness question
exists, and it means every performer in every video must be someone who exists
and consented.

This is design only. It needs an image model, and that is a real decision with
a disk cost, a licence question and an unsolved consistency problem. Nothing
here has been built, deliberately — the wrong version of this feature is a
one-off script that makes a pretty portrait and leaves the hard part untouched.

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

Two caveats to carry into any such measurement:

* **The detector is the weak link.** `faces.detect_faces`/YuNet scores 0.0%
  face presence on a front-facing medium close-up filling a third of the frame
  when the subject wears a hat, beard and glasses, and on a large, fully-lit
  upturned face. A synthetic character in a hat that cannot be *detected*
  cannot be *recognised* either, and the number will read as "inconsistent"
  when it means "invisible to YuNet". Check the frame before believing a zero.
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
rather than remembered. Proposed shape (this is `contracts.py`/`config.py`,
owned elsewhere — proposed, not built):

```toml
[cast.Nobody]
role  = "Lead vocalist"
image = "cast/nobody_ref_01.jpg"
synthetic = true                     # default false: every existing config is real
[cast.Nobody.origin]                 # required when synthetic = true
model  = "…"
prompt = "…"
seed   = 12345
created = "2026-08-23"
```

`synthetic = true` with no `[cast.<name>.origin]` should be refused at load:
"this character is invented" with no record of how is the provenance bug in a
new place. `synthetic = false` (the default) keeps every existing config
loading unchanged, and it is the value that means "a real person's likeness is
in this run" — which is the query #51 wants to be able to run.

The flag is also the natural place to hang a future consent field for the real
case, mirroring `tests/test_repo_assets.py`'s `Asset(depicts_real_person=…,
consent=…)`, which already refuses an unconsented likeness mechanically.

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
   asset.
2. The similarity check above passes over the set, and its numbers are
   recorded next to the character.
3. `synthetic = true` + `origin` in the run config; the render path is
   unchanged, because `ref_images` still just receives a photo — of nobody.
4. A cross-video story, since a character is a cross-video asset exactly like a
   locked house style (#55). Whether they share a mechanism is worth
   considering, but note the difference: a profile is *text* that resolves into
   config fields, while a character is *binary assets plus text*. The profile
   mechanism does not extend to that for free, and forcing it to would make the
   simple half worse.

## The trap to avoid

Do not design this around the fixture problem. #51 is the loudest reason and
the smallest one; a fix scoped to "regenerate three test images" helps this
repo once and looks finished. The defect is that **the pipeline requires a
photograph of a consenting human before it can do anything at all**, which is
true of every user who is not the author, and true of every extra in every
shot — none of whom consented to anything, because none of them exist.
