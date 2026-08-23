# A voice on the track is not a character in the story (issue #89)

**General form: a cast entry conflates a *performer* (a voice, a face, a
photograph) with a *character* (a part in the story), and one lookup answers
both questions. Whenever a lyric is written in a voice other than the
performer's own — a narrator singing as a chorus of the dead, one singer
voicing both halves of a dialogue, an unreliable narrator, a quoted speaker —
the two answers differ and the pipeline stages the performer.**

A stranger's test: does any vocal line in your song get sung by a performer
playing, quoting, or channeling someone who isn't them?

## The premise the issue rests on is false, and that is the finding

Issue #89 and the run-local F22 note both close the obvious fix before anyone
tries it:

> a third `[cast.*]` entry with its own reference photo would not close this
> gap. `chunk.characters` is derived from which cast member's lyric-file voice
> a line maps to — a property of the *audio and the alignment*, not of the
> story.

**`chunk.characters` is not a property of the audio.** It is authored text,
and the code says so:

- `lyrics.py` parses `[Name: Role]` tags and validates each name against the
  cast (`_validate_character`). Every `LyricLine.characters` is whatever the
  lyrics file *says*.
- `alignment.py` hands `model.align()` nothing but `text_blob` — the
  tag-stripped text — and gets timings back. `_build_segments` then attaches
  `characters` from `word_owner`, which was built from the parsed lines. The
  aligner is never asked who is singing and never answers.
- `slicing.py` carries those tuples onto `AudioChunk.characters` unchanged.

There is no speaker attribution from audio anywhere in this pipeline. So the
separation #89 asks for **already exists in the data model**: a lyric line can
already be tagged with any name the cast defines, and everything downstream —
`role`, `appearance`, `demeanour`, the staged reference photo, the composed
focus sentence — follows that name rather than any property of the recording.

What actually blocks `[cast."The Dead"]` today is one validation rule:

```python
# config.py
image = entry.get("image")
if not image:
    _fail(f"cast.{name}.image", "missing required 'image'")
```

`image` is mandatory on every cast entry, so **only a performer with a
photograph can be named**, and a story character that is nobody's face cannot
exist. #89 is one validation rule wide, not an architecture.

## What is genuinely missing

Two things, and neither is large:

1. **A character with no photograph has nothing to condition on.** The base
   `MiniMaxH3ReferenceToVideo` path requires `ref_images.ref_image_0`; it
   cannot render an identity from text alone. Something must say which face
   carries the part.
2. **Nothing records who is physically audible.** Once a line is tagged
   `[The Dead]`, no field anywhere answers "whose voice is that" — which any
   future vocal-range check, stem router, or the #92 character-merge billing
   would need.

One field answers both.

## The design

`CastMember.voiced_by: str | None = None`.

```toml
[cast.Jan]
role  = "Kashay Besmertny the Deathless, an immortal watchman"
image = "cast/jan_ref.jpg"

[cast."The Dead"]
voiced_by = "Jan"                       # whose voice physically sings these lines
role      = "the war's dead, speaking collectively"
demeanour = "flat, tireless, without appetite"
# image omitted -> inherits Jan's
```

and in the lyrics file, using the tag syntax that already exists:

```
[The Dead: chorus]
our lives are prisons
```

### Semantics

- **An entry with `voiced_by` is a character, not a performer.** It must name
  another cast entry, it may not name itself, and the entry it names may not
  itself set `voiced_by` — no chains. A chain is the recurrence-relation
  hazard CLAUDE.md warns about under "Chained rendering makes every per-chunk
  instruction a recurrence relation"; there is no case for one here, so it is
  refused at load rather than reasoned about later.
- **`image` becomes optional, and only for such an entry.** When omitted it
  resolves to the performer's image **at config load**, so `CastMember.image`
  is a valid path by the time anything downstream sees it. Staging, prompting,
  the graph mutation and the fingerprint all keep working on a plain
  `CastMember` and need no knowledge of this feature at all. An explicit
  `image` on a `voiced_by` entry wins — an invented character with its own
  generated portrait is the better-conditioned case, not a special one.
- **The render composes the character's own `role` / `appearance` /
  `demeanour`. Never the performer's.** That is the whole point, and it needs
  no new code: `_resolve_active_members` already looks up `config.cast[name]`
  and composes what it finds.
- **`chunk.characters` keeps reporting the character name**, because that is
  what the lyric file says. The performer is now derivable
  (`cast[name].voiced_by or name`) — the seam the issue says does not exist,
  and it is one attribute access.
- **`present` and `subject` are unaffected.** Both resolve a cast key; a
  `voiced_by` character is a legal value for either. `subject` stays refused
  on a voiced chunk (#82). `_resolve_present_members` already de-duplicates by
  cast key, so naming `Jan` as present on a chunk `The Dead` sings correctly
  keeps him — Jan standing beside the voice he is lending is a coherent shot,
  and the two entries are different characters.

### The fingerprint needs no change

`ChunkFingerprint.CONTENT_FIELDS` already carries `character` and `image_ref`.
Retagging a line from `[Jan]` to `[The Dead]` moves `character` and
`prompt_hash`; `image_ref` stays put when the photo is inherited. So a
`--resume` re-renders exactly those chunks and no others. That the existing
tiers already say the right thing is the strongest available evidence that
this design fits the model rather than fighting it.

### Inert unless used

No config in existence sets `voiced_by`, and with it unset every composed
prompt is byte-identical to before. That is the same standard `location` was
held to in commit 888b6d3 (`setting`/`location` both unset renders
byte-identically), and it is testable offline — it is a required test, not a
hope.

## Alternatives considered and rejected

- **`voiced_as` as a `ShotPlanEntry` field.** Wrong layer. Identity is a
  property of the *lyric line*, and chunk boundaries move — #79's onset
  preference moves them by design. A shot-plan field would silently misattach
  the way CLAUDE.md's re-anchoring bullet describes: attaching shot 12's
  identity to shot 9's audio, raising nothing.
- **A per-line override in a new tag syntax** (`[Jan as The Dead: ...]`).
  Rejected: the existing tag already selects a cast key, so the feature costs
  no new syntax, no `lyrics.py` change, and no new parse errors. Group A owns
  `lyrics.py`; this design deliberately requires nothing from them.
- **Refusing to render a `voiced_by` character with no image.** Rejected: it
  makes the field useless for the case that motivates it (#56, an invented
  character with no photograph), and the base path cannot run without an
  image at all.
- **A `gender` or `pronoun` field to help the #72 referent lint.** Out of
  scope and separately rejected in #72 — noted only so the next reader does
  not re-derive it.

## The hazard this introduces, and the guard for it

If a character inherits its performer's photograph **and** the performer is
also staged in the same shot via `present`, the graph conditions on the same
reference photo twice for two different characters. That is a real defect and
a plausible one, so it warns at config/prompt time rather than being
discovered in a render.

## What is NOT proven, and needs the 4090

Everything above is a statement about what the pipeline *composes*. Whether
H3 *honours* it is unmeasured, and this project's own record is that the two
come apart:

- #82 measured H3 resolving a contradiction between billing and sentence by
  **morphing one character into the other** mid-chunk (chunk 29, 3:11).
  Composing "The Dead" as the focus over Jan's reference photo is that
  contradiction deliberately introduced. The predicted failure is the one
  already observed.
- #73 measured a change that was "strictly additive" manufacturing the very
  thing it was meant to suppress.
- #60: a generalisation from one chunk is a hypothesis; the full render is the
  experiment.

So the honest status is: **the mechanism ships; the claim that it works does
not.**

### Verification plan (needs the card)

Song: "Deathless". The two passages F22 and #89 name are Jan's — the middle
accusation ("our lives are prisons") and the closing epitaph.

1. Copy `run_v12.toml` / `shot_plan_v12.toml`. Add a `[cast."The Dead"]`
   entry with `voiced_by = "Jan"`, its own `role` and `demeanour`, and **no**
   `image`. Retag only those lyric lines.
2. `--prepare --from-plan shot_plan_v12.toml` and confirm the chunk timeline
   is unchanged (it must be — the tag is stripped before alignment; a moved
   boundary means something is wrong with the change, not with H3).
3. Seed the output dir from `chunks_v12` and `--resume`. Only the retagged
   chunks should re-render — verify against `run_state.json` that the
   fingerprint differs on `character` and `prompt_hash` and **nothing else**.
   That is a free, GPU-less check of the whole design and should be run first.
4. Render the retagged chunks with `--only-chunks`. Arms, at identical seeds:
   - **A** — as authored today (Jan, his role, his demeanour).
   - **B** — `[The Dead]`, inheriting Jan's photo.
   - **C** — `[The Dead]` with its own generated portrait, to separate "the
     text did it" from "the photo did it". Without C, B's result cannot be
     attributed.
5. Measure, rather than watching: face presence and recognition similarity
   against Jan's reference photo (`faces.recognize_face`, floor 0.34, see
   `docs/seed-face-recognition.md`). **The prediction that would falsify the
   design is B scoring a *mid-range* similarity** — neither Jan nor a stable
   other person — which is what morphing looks like numerically. B holding
   Jan's likeness while the shot reads as the dead is the success case; B
   inventing a different face every take is the #59 failure returning.
6. Record the negative result either way. A design that composes correctly and
   renders wrong is a finding, not a regression to hide.
