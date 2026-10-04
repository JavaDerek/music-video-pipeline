# House styles (issue #55)

Committed, versioned cinematography profiles. A run config points at one:

```toml
# run.toml
cinematography_profile = "../../profiles/refestramus-house-v1.toml"
```

The path resolves against **the run config's own directory** (absolute paths
work too). There is no registry, no search path and no `~/.mvm/profiles`: a
shared house style is a file in a directory you control, or a checked-out
repo. This directory is that repo's copy.

The machinery is `music_video_maker/profiles.py`; the reasoning behind each
decision is `docs/design-cinematography-profiles.md`. This file is the part a
person needs before editing anything in here.

## What is locked and what varies

The distinction is the whole feature. Two different kinds of statement:

| | field | scope | lock it? |
|---|---|---|---|
| **The look** | `cinematography` (+ `face_treatment`, `lora`, `lora_strength`, `lora_trigger`) | whole video, slow-changing, the calling card | **yes** — that is what this directory is |
| **The shots** | `ShotPlanEntry.camera` — framing and movement for one beat | per shot, per song | **never** — lock it and every video becomes the same video |

`profiles.LOOK_FIELDS` is the closed set of what a profile may set. Three
things are excluded on purpose and none of them is an oversight:

* **`global_style`** — genre, band, tone. Per-song. Issue #53 split
  cinematography *out* of it precisely so it would stop being a junk drawer;
  putting it in a cross-video file puts it straight back.
* **`render_width` / `render_height`** — resolution is the dominant cost lever
  measured on this project's own 4090 (864×480 ≈ 3.7 min/chunk against
  9 m 15 s at 1344×768). A look file that silently triples a catalogue's GPU
  hours is the wrong kind of inheritance.
* **`setting`, `global_appearance`, `global_demeanour`, the cast** —
  whole-video, but about the story and the people in it. A profile carrying
  `setting` would relocate every future song to a Slavic mountain.

**Partial locking costs no machinery: it is just which fields you set.** A
profile that sets only `cinematography` locks the grade and leaves the LoRA
decision per song, which is exactly what `refestramus-house-v1.toml` does and
why. Sub-locking *inside* `cinematography` (lock the grade, let lighting adapt
to a night song) is not built; don't build it before a second song asks.

**A run config always wins on a field it sets itself.** The profile fills in
only what the run left unset. Every inherited field and every overridden one
is logged at INFO at config load, naming the profile and its version — a
silent inheritance is how a locked look drifts, which is the thing this
feature exists to stop.

## Versioning: a profile file is immutable once a run has rendered against it

`version` is a required int ≥ 1 and it is the *house style's* version, not a
schema version. The rule:

> Once any run has rendered against `name-vN.toml`, that file does not change
> again. A new look is `name-v(N+1).toml` with `version = N+1`.

"Locked" does not mean "never changes" — it means changing it is a deliberate
act with a version bump, not a side effect of regenerating a look. Nothing
*enforces* immutability (a file on disk cannot be made read-only by a Python
module that only reads it), so what backs it up is evidence rather than a
gate:

* the run's sidecar records `profile_sha256` of the file bytes, so an edit
  after the fact is **provable**, not merely suspected;
* on `--resume`, an edited `cinematography` moves `prompt_hash`, and every
  affected chunk is correctly reported as content-changed and re-rendered.

A filename convention is checked by nothing, which is why `version` exists as
a key. Bump both together.

## Reproducibility: what proves which look made a video

`prompt_hash` proves a look **changed**. It cannot say what the look **was** —
six months from now, with the profile at v3, a finished `run_state.json` can
prove its hash differs from today's and nothing more. So every run that names
a profile writes `<chunks_dir>/cinematography_profile.json`:

```json
{
  "format_version": 1,
  "recorded_at": "…Z",
  "profile_path": "/…/profiles/refestramus-house-v1.toml",
  "profile_sha256": "…",
  "profile": { "version": 1, "name": "…", "values": {…}, "provenance": {…} },
  "effective": { "cinematography": "…", … },
  "overridden_by_run_config": ["lora_strength"]
}
```

`overridden_by_run_config` is `RunConfig.cinematography_profile_overrides`,
and it exists as its own field because **it cannot be recomputed afterwards**:
`lora_strength` defaults to `1.0`, not `None`, so a resolved config cannot
tell "the run set this" from "the dataclass defaulted it", and a
reconstruction reports it overridden on every run that names a profile at all.
The distinction only exists while the raw TOML is in scope, so `load_config`
writes it down as it goes.

Writing the sidecar can never abort a run: a provenance file failing is a
thing to log, not a reason to lose a render.

## Why this file refuses to load

`refestramus-house-v1.toml` is committed as a **skeleton**: its
`cinematography` is the placeholder `<TODO: …>`, and `load_profile` raises on
any look value starting with `<TODO`, naming the promote command.

The approved text — the 562 characters the "Deathless" photography stage
produced, which the 2026-09-20 A/B rendered at identical seeds on chunks
16/23/44/64 and which Derek chose — lives in
`~/mvm-runs/deathless/.authoring/photography.json`. That is a run asset, not
repo content, and it is not reproducible from a checkout.

Which leaves two ways to commit the file, and the choice is about which
mistake is cheap. A skeleton carrying a *plausible but truncated* look would
render a catalogue that claims the house style and does not have it, with no
symptom anywhere. A skeleton that refuses at config load costs a run that
never started. Same split `i2v_require_seed_face`'s recognition option uses:
refuse loudly at config load where the mistake is expensive, degrade quietly
where it is not.

Fill it by **promoting, never by retyping** — the promote command carries the
provenance block, which is the half a retype silently loses:

```
python -m music_video_maker.profiles promote \
    --run-dir ~/mvm-runs/deathless \
    --name refestramus-house --version 1 \
    --out profiles/refestramus-house-v1.toml --overwrite
```

`promote` reads `.authoring/photography.json` and `session.json` as plain JSON
(it never imports the authoring package — that boundary is mechanically
enforced), and records the photography stage's model, its `completed_at` as
`generated_at`, its recorded `concept` input hash, and today as `promoted_at`.
`generated_at` and `promoted_at` are two dates on purpose: **the gap between
them is the review.** It refuses to clobber an existing profile without
`--overwrite`, the same way `--prepare` refuses to overwrite a shot plan.

## The schema constraint, if you are adding a field

**Every field a profile may set must already be recorded in
`ChunkFingerprint`.** `profiles.FINGERPRINT_EVIDENCE` states the mapping and a
test enforces that every `LOOK_FIELDS` entry has one and that every value
names a real fingerprint field.

A look field in a shared cross-video file that nothing fingerprints lets an
edit change the pixels while `--resume` reuses the old ones and reports a
clean match — #34's failure mode arriving down a new road. **Anything added to
`LOOK_FIELDS` must come with its fingerprint evidence in the same commit.**

Worked example of the rule doing its job: issue #68 proposes that stereo
comfort settings (convergence, maximum disparity) belong here, since "comfort
settings are a signature and a safety limit". They do not, yet, and the
constraint is what says so — `music_video_maker/stereo.py` is a post-render
pass that changes no chunk H3 produced, so there is nothing in a
`ChunkFingerprint` for them to move. They stay in `stereo.StereoParams` until
that changes.

## TOML's bare-key hazard, which this format is shaped around

TOML binds a bare key to whichever table precedes it. **Every top-level key
must appear above `[provenance]`**; a `cinematography` line appended to the
end of a file lands inside that table and is silently dropped.

This project has paid for it twice (`config.HARDWARE_KEYS`,
`ALIGNMENT_OVERRIDE_KEYS`) and once expensively: in issue #62's A/B, a `lora`
key appended below the `[[alignment_override]]` tables meant **both arms of
the experiment loaded identically**. So both key sets here are closed, an
unknown key is an error rather than a shrug, and when an unknown
`[provenance]` key is a real look field the error says so and tells you to
move it up.

## Files

| file | what it is |
|---|---|
| `refestramus-house-v1.toml` | the catalogue house style. **Skeleton** — see above. |
| `../examples/profiles/refestramus-house-v1.toml` | a complete, loadable *example* for docs and tests. Hand-authored, never promoted from a real photography stage, and not the house style. |
