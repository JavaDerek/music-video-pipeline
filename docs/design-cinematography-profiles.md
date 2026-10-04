# Locked house style: `cinematography_profile` (issue #55)

`music_video_maker/profiles.py` + one field on `RunConfig`. A named, versioned
file a run config points at, whose values fill in the look fields the run did
not set itself.

The point is not convenience. It is that a body of work should share a
recognisable signature — and that the look this project's own authoring layer
generates currently has **nowhere durable to live**.

## The measured motivation

"Deathless", the only full 80-chunk render this project has, was authored with
`mvm-author`. Its Stage-3 photography run produced 562 characters of film
direction and a human approved it:

> Shot on 35mm, anamorphic, heavy grain left in; long lenses for faces with
> shallow focus that holds the head sharp and dissolves the valley behind […]
> Grade is desaturated cold slate and ash-grey, crushed blacks, skin kept pale
> and matte, fire and flash the only warm values in the picture; no gloss, no
> flares, no lift.

Three things are true of that text, all checkable:

1. It is in `~/mvm-runs/deathless/.authoring/photography.json` under
   `"cinematography"`.
2. It appears **nowhere** in `shot_plan_v12.toml` (`grep -c cinematography`
   → 0) and **nowhere** in `run_v12.toml`.
3. `grep -c anamorphic render_v12.log` → **0**. The word appears in
   `photo_v6.log`, where it was generated, and in no line of the render that
   followed.

The *other* half of the same stage's output — the per-chunk `camera` clause —
is in all 80 shot-plan entries and was composed into all 80 prompts. So the
stage ran, the human approved, and exactly the half that is whole-video was
dropped on the floor, silently, because nothing carried it from
`.authoring/photography.json` into a field the render reads.

Meanwhile `run_v12.toml`'s `global_style` reads:

> `"Refestramus progressive rock music video, 35mm film, dark and mythic, epic
> in scale, serious tone, never camp. Each shot is ONE continuous unbroken
> take, a single camera, no cuts within the shot, no montage"`

— band, genre, film stock, tone, and an editing constraint, in one string.
That is the junk drawer issue #53 split up, growing back, for want of a place
to put the look. Issue #53 created `cinematography`; nothing has ever set it.

So #55's "promote the pick" is not the last refinement of a working loop. It
is the step that makes the loop close at all.

## Shape

```toml
# run.toml
cinematography_profile = "profiles/refestramus-house-v1.toml"
```

```toml
# profiles/refestramus-house-v1.toml
version     = 1                      # required; a bump is a deliberate act
name        = "refestramus-house"    # required; identity in logs and records
description = "..."                  # optional

cinematography = "35mm anamorphic, long lenses, cold slate grade …"
face_treatment = "realistic"
lora           = "h3-realism-people-t2v-i2v-r2v.safetensors"
lora_strength  = 0.8
lora_trigger   = "r34l1sm"

[provenance]
source        = "promoted"
song          = "Deathless"
promoted_from = ".authoring/photography.json"
model         = "claude-opus-5"
promoted_at   = "2026-08-16"
concept_hash  = "7fe8282199d1692a…"
```

## The five decisions the issue asks for

### 1. Reference, not copy — and the run config wins

A profile is referenced by path and read at config load. Any field the run
config sets itself beats the profile's value for that field.

Two reasons, and the second is the load-bearing one:

* It is the mechanism `load_config` already has. CLI `overrides` beat file
  values; file values beat profile values. One direction, one rule.
* The alternative — profile wins — means a run cannot deviate on one field
  without editing a file that every other video in the catalogue inherits.
  That is the coupling that makes a shared style dangerous rather than useful.

Every inherited field and every overridden field is logged at INFO, naming the
profile and its version. A silent inheritance is how a look drifts, which is
the thing this feature exists to stop.

### 2. Versioning: an explicit `version` field, not just `-v2` in the filename

The filename is a convention nothing checks; a missing `version` key is an
error at load. Bumping it is the deliberate act; regenerating a look and
overwriting a profile in place, without a bump, is the drift the issue names.

`version` is the *profile's* version, not a schema version. The format itself
is versioned by `profiles.PROFILE_FORMAT_VERSION` and by the closed key sets:
an unknown top-level key is an error, so a v2 format read by a v1 build fails
loudly instead of half-applying.

### 3. Partial locking is what a *field set* buys you

A profile that sets `cinematography` and nothing else locks the look and
leaves the LoRA decision per-song. A profile that sets `face_treatment` and
nothing else locks the photoreal-vs-flattering trade (#62) and leaves the
grade per-song. That is the partial locking the issue asks for, and it costs
no machinery.

Sub-locking *inside* `cinematography` — lock the grade, let lighting adapt to
a night-time song — is **not** built. If it is ever wanted, the shape that
does not break anything is a `[cinematography]` table of named parts composed
in a declared order, with the flat string staying legal. Do not build it
before a second song asks for it.

### 4. `prompt_hash` is not enough, so the resolved profile is recorded verbatim

A hash proves *change*. It does not preserve *content*. Six months from now,
with the profile at v3, a finished video's `run_state.json` can prove its
prompt hash differs from today's — and cannot say what look produced it. The
question #55 asks is exactly "can a finished video prove its look", and a hash
cannot answer it.

So `profiles.write_profile_record()` writes a JSON sidecar beside
`run_state.json`:

```json
{
  "recorded_at": "2026-08-23T…Z",
  "profile_path": "/…/profiles/refestramus-house-v1.toml",
  "profile_sha256": "…",
  "profile": { "version": 1, "name": "…", "values": {…}, "provenance": {…} },
  "effective": { "cinematography": "…", "face_treatment": "realistic", … },
  "overridden_by_run_config": ["lora_strength"]
}
```

`profile_sha256` is of the file bytes, so a later edit to the profile is
provable rather than merely suspected.

### 5. Provenance: which concept it was promoted from

`promote_photography()` reads `.authoring/session.json` and carries the
photography stage's `model`, `completed_at`, and its recorded `concept` input
hash into `[provenance]`. That hash is what ties the look back to the concept
that produced it — the same reasoning every other provenance decision here
follows (#34, #38, #45, #54).

## The schema constraint that is not obvious

**Every field a profile may set is already recorded in `ChunkFingerprint`, and
that is a rule, not a coincidence.**

| profile field   | what moves in the fingerprint | tier |
|---|---|---|
| `cinematography` | `prompt_hash` (composed into every prompt) | content |
| `face_treatment` | `lora`, `lora_strength` (it is a preset for them) | conditioning |
| `lora`           | `lora` | conditioning |
| `lora_strength`  | `lora_strength` | conditioning |
| `lora_trigger`   | `prompt_hash` (composed when a lora is set) | content |

A look field admitted to a profile that nothing fingerprints would let an edit
to a shared, cross-video file change the pixels while `--resume` reuses the
old ones and reports a clean match. That is #34's failure mode arriving down a
new road. `profiles.FINGERPRINT_EVIDENCE` states the mapping and a test
enforces that every entry of `LOOK_FIELDS` has one and that every value names
a real `ChunkFingerprint` field.

**Anything added to `LOOK_FIELDS` in future must come with its fingerprint
evidence in the same commit.**

## What is deliberately NOT a profile field

* **`global_style`.** Genre, band and tone: per-song. #53 split it out from
  the look precisely so it would stop being a junk drawer; putting it in a
  cross-video profile puts it straight back.
* **`render_width` / `render_height`.** Measured on the 4090 at 141 frames:
  864×480 ≈ 3.7 min/chunk, 1344×768 = 9 m 15 s. Resolution is the dominant
  cost lever in the whole project. A look profile that silently triples a
  run's GPU hours is the wrong kind of inheritance. If a future revision adds
  it, it must be loud at config load, not quiet.
* **per-shot `camera`.** The issue's central distinction: lock the look, keep
  the shots varied, or every video becomes the same video.
* **`setting`, `global_appearance`, `global_demeanour`, cast.** Whole-video,
  but about the story and the people in it, not the film's look. A profile
  that carried `setting` would relocate every future song to a Slavic
  mountain.

## Validation: shape here, domain in `config.py`

`profiles.py` checks types and non-emptiness. It does **not** know that
`face_treatment` ∈ `{"flattering", "realistic"}` — `config.py` owns that
closed set, and a second copy of it is a second thing to drift. The profile's
values are merged into the config's raw dict *before* validation runs, so an
illegal profile value is refused by exactly the check that would have refused
it in the run config, with the same message.

`profiles.py` imports nothing from `music_video_maker` (config imports it, so
the reverse would be circular). It raises `ProfileError`; `config.load_config`
catches that and raises `ConfigError`.

## The TOML hazard this format is shaped around

TOML binds a bare key to whichever table precedes it. This repo has paid for
that twice — see `config.HARDWARE_KEYS` and `ALIGNMENT_OVERRIDE_KEYS`, and
issue #62's A/B, where `lora` appended below the `[[alignment_override]]`
tables meant **both arms of the experiment loaded identically**.

A profile has the same shape: look fields at the top, `[provenance]` at the
bottom. A `cinematography` line appended to the end of the file lands inside
`[provenance]` and is silently dropped. So both key sets are closed, and when
an unknown `[provenance]` key is a known look field the error says so
explicitly and tells you to move it above the first table.

## Promoting a pick

```
python -m music_video_maker.profiles promote \
    --run-dir ~/mvm-runs/deathless \
    --name refestramus-house --version 1 \
    --out profiles/refestramus-house-v1.toml
```

Reads `<run-dir>/.authoring/photography.json` and `session.json` as plain
JSON. It does **not** import `music_video_maker.authoring` — the render/
authoring import boundary is enforced mechanically by
`tests/test_authoring_boundary.py`, and `profiles.py` is imported by
`config.py`, which is render-path.

It refuses to clobber an existing profile without `--overwrite`, for the same
reason `--prepare` refuses to clobber a shot plan: a locked house style is
real work and a promote is easy to run twice.

If `photography.json` has no `cinematography`, the error says why: the
photography stage returns nothing for the whole-video half when the run config
*already* fixes `cinematography`, so there is nothing generated to promote and
the run config's own value is what you want.

## Costs and limits, stated

* A profile is read at config load only. Editing it mid-run changes nothing
  until the next run; on `--resume` the changed text lands in `prompt_hash` and
  the affected chunks are correctly reported as content-changed.
* There is no profile registry, no search path, no `~/.mvm/profiles`. The path
  resolves against the run config's directory like every other path field.
  A shared house style is a file in a directory you control, or a checked-out
  repo; inventing a global namespace before there are two songs is premature.
* Nothing promotes automatically. Step 3 of #55's loop is a human saying "that
  one — keep it", and this command is the only way it happens.

## The committed house style: `profiles/`

A run config can point anywhere, but a house style that is only ever a path in
someone's home directory is not a catalogue's signature. `profiles/` is this
repo's copy, with `profiles/README.md` carrying the three rules a person needs
before editing anything in it: lock-vs-vary, versioning, and what proves which
look made a video.

**Immutability is a rule backed by evidence, not by a gate.** Once any run has
rendered against `name-vN.toml`, that file does not change again; a new look is
`name-v(N+1).toml` with `version = N+1`. Nothing can enforce that from a module
that only reads the file, so what backs it is the sidecar's `profile_sha256`
(an edit after the fact is *provable*) and `prompt_hash` (a resumed run reports
every affected chunk as content-changed). The version number is what a human
reads; the two hashes are what a machine can check.

**The first house profile is committed as a skeleton, and it refuses to load.**
`profiles/refestramus-house-v1.toml` carries the placeholder
`cinematography = "<TODO: …>"`, and `load_profile` raises on any look value
starting with `<TODO`, naming the promote command. The approved 562 characters
live in `~/mvm-runs/deathless/.authoring/photography.json` — a run asset, not
repo content, and not reproducible from a checkout.

The choice is about which mistake is cheap, the same question `i2v_require_seed_face`'s
recognition option answers one module over. A skeleton carrying a *plausible
but truncated* look renders a catalogue that claims the house style and does
not have it, with no symptom anywhere — this project's expensive kind of wrong.
A skeleton that refuses at config load costs a run that never started. So the
sentinel is a shape no real look has (a leading `<`), it is checked where every
other profile value is checked, and the message says how to fill it: promote,
never retype, because the promote carries the `[provenance]` block a retype
silently loses.

## What #68 asks of this file, and the answer

Issue #68 proposes that stereo comfort settings (convergence, maximum
disparity) belong in a locked house style, since "comfort settings are a
signature and a safety limit". **They do not, yet, and the schema constraint
above is exactly what says so.** `music_video_maker/stereo.py` is a
post-render pass: it reads finished chunk mp4s and writes new files, changes
no chunk H3 produced, and moves nothing in a `ChunkFingerprint`. There is
therefore no fingerprint evidence to admit it on, and it lives in
`stereo.StereoParams` until that changes. Worth recording as the first time
the constraint refused something rather than merely describing what was
already there.

## Verification status

Everything above is verified offline by `tests/test_profiles.py` (load,
precedence, record, promote, CLI, the placeholder refusal, the committed
skeleton, and the fingerprint-evidence guard) plus the config-integration
tests in the same file.

**A render has now been produced with `cinematography` set (2026-09-20).** The
confirmation this document specified was run as an A/B at identical seeds on
chunks 16/23/44/64, differing only in that field:

* mean |A−B| pixel distance **33–51 per chunk** — far more than a ComfyUI
  version change produces (4–22, #95);
* colour went cooler in all four (R−B down 4–30);
* fine detail dropped (Laplacian variance −18% to −52%), the shallow-focus
  clause softening backgrounds;
* motion dropped (−1% to −48%);
* and it **re-stages** shots rather than only re-grading them: a wide
  silhouette against the sky, an extreme close-up opening, a pull-back with a
  fire in frame.

Derek watched the A/B and chose it; the next full "Deathless" render sets it.
So the field is worth what #55 claimed, and the remaining question is the one
this document's `profiles/` section answers: a look that has been chosen needs
somewhere durable and versioned to live, or the next song re-rolls it.
