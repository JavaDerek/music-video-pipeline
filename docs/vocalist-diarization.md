# Automatic vocalist diarization (issue #101)

Detect which cast member is singing from the audio, and land the answer in the
**same** `AlignedSegment.characters` field a manual `[Name: Role]` tag fills.
Built 2026-10-04. Opt in with `diarize = true`; off by default, and a run that
leaves it off is byte-identical to one from before this existed.

The design this implements is [`docs/design-multi-vocalist.md`](design-multi-vocalist.md);
the manual tags it sits beside are [`docs/lyrics-format.md`](lyrics-format.md)
and remain the whole mechanism when this is off. The input is the isolated
vocal stem from [`docs/vocal-stem-workflow.md`](vocal-stem-workflow.md).

**Status, stated plainly: the mechanism is built and tested; the claim is
not.** No multi-vocalist song has been diarized here, and the weights were
inaccessible when this shipped (see "What is still unverified" at the bottom).
Read that section before trusting a number this produces.

## Attribution — required

The diarizer is [`pyannote.audio`](https://github.com/pyannote/pyannote-audio).
Its **code** is MIT. Its **pretrained pipelines are CC-BY-4.0**, which requires
attribution from anyone who uses them:

> Speaker diarization by pyannote.audio (code: MIT) using the pretrained
> pipeline `pyannote/speaker-diarization-3.1` (models: CC-BY-4.0), by Hervé
> Bredin and contributors — <https://github.com/pyannote/pyannote-audio>.
> Cite: H. Bredin, *"pyannote.audio 2.1 speaker diarization pipeline:
> principle, benchmark, and recipe"*, Interspeech 2023.

That text is `music_video_maker.diarization.ATTRIBUTION`, and it is **logged at
INFO on every run that loads the weights** — not only recorded here. CC-BY asks
for credit from whoever uses the work, so the credit travels with the use.
It is also in the README's own [License](../README.md#license) section, which is
this repo's rights table for third-party models it does not commit.

**Nothing is redistributed.** No weights are committed here and none are
fetched by this project on anyone's behalf: the operator accepts the terms with
their own Hugging Face account and the files land in their own cache, exactly
as with the SFace weights `faces.py` points at. Derek accepted pyannote's terms
on 2026-10-04 (code MIT, models CC-BY-4.0, commercial use permitted with
attribution).

## One-time setup

### 1. Accept the terms — all three pages

`pyannote/speaker-diarization-3.1` is a *pipeline* that loads a segmentation
model and an embedding model, **each gated separately**. Sign in as the account
that will own the token and accept on every one of:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>
- <https://huggingface.co/pyannote/segmentation-3.0>
- <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>

Accepting only the first produces a 403 on the second, with a message naming a
repository you never asked for. The metadata API answers `200` for an
unaccepted repo and the **files** answer `403`, so "I can see the model page"
is not evidence of access.

### 2. Set the token

Create a read-scoped token at <https://huggingface.co/settings/tokens> and put
it in the gitignored `.env` as `HF_TOKEN` (`HUGGINGFACE_HUB_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` are also read, in that order). **Never in `run.toml`**
— that is committed run data, and CLAUDE.md's rule is that credentials live in
`.env` and nowhere else.

### 3. Install the extra

```bash
pip install -e ".[diarize]"
```

An extra for the same reason `[align]` is one: it pulls torch, and nothing in
the test suite needs it — every test injects the diarizer behind
`diarization.Diarizer` and the suite runs fully offline with no `pyannote`
installed at all.

## Per-song workflow

### 1. Produce and listen to the vocal stem

Diarization runs on the **isolated vocal stem**, never the full mix. That is
not a preference: in a mix a voice-like lead synth is indistinguishable from a
singer, and diarizing the mix is what made this feature not worth attempting in
the first place. `diarize = true` without `vocal_stem` is **refused at config
load**.

```bash
python -m demucs --two-stems=vocals -o stems/ audio/master.wav
# -> stems/htdemucs/master/vocals.wav
```

Listen to it. `stems.py`'s `StemQualityReport` already warns about chunks that
carry a lyric but are silent in the stem — a separator that filed a processed
voice under `other` will also hide it from the diarizer.

### 2. First pass: get the cluster table

Diarization produces **clusters**, never names. So the first run is expected to
assign nothing:

```toml
vocal_stem = "stems/htdemucs/master/vocals.wav"
diarize = true
```

```bash
music-video-maker --config run.toml --prepare
```

`--prepare` is the right place for this — Stages 1–2 only, no GPU, no ComfyUI,
no custody handoff — and it logs:

```
Detected speaker clusters:
  SPEAKER_00         187.44s    62 turn(s)  first at    8.133s  -> UNMAPPED -- add it to [diarization_speakers]  e.g. 'the lucky ones dont ever have to try'
  SPEAKER_01          41.09s    14 turn(s)  first at  121.900s  -> UNMAPPED -- add it to [diarization_speakers]  e.g. 'were watching from the wings tonight'
```

Longest cluster first, because the lead is almost always the biggest one. The
example lyric is the line that cluster's longest turn lands on, which is
usually enough to recognise a voice without opening the audio; `first at` is
where to listen if it is not.

### 3. Write the mapping, which is authored data

```toml
[diarization_speakers]
SPEAKER_00 = "Dianne"
SPEAKER_01 = "Marcus"
```

Authored once, committed, deterministic — the same shape as the shot plan and
`[[alignment_override]]`, and for the same reason: no inference at render time.
Every value is validated against `[cast]` at load, because a typo attributes a
whole verse to a character that does not exist and the symptom otherwise
arrives in Stage 2b as a missing reference photo, minutes into a run.

A label with no entry assigns nobody and is named in a warning. That is the
correct behaviour for a third cluster that turns out to be a doubled vocal or a
bleed, and it is also how you discover there is one.

### 4. Re-run `--prepare` and read the disagreements

The second pass reports what it did:

```
Diarization of vocals.wav: 2 speaker cluster(s) over 54 aligned segment(s) --
  31 assigned, 18 agreed with a tag, 2 disagreed with a tag (authored value kept),
  3 untagged segment(s) left on the default lead
```

## The rule: the tag wins, and the disagreement is reported

**Detection writes `characters` only where no `[Name]` tag is in force.** Where
a tag *is* in force and detection names somebody else, the segment is left
exactly as authored and the clash is logged at WARNING and recorded in
`DiarizationReport.disagreements`:

```
Diarization disagrees with an authored tag and the TAG WINS: segment 37
[142.100-148.400s] is tagged ['Dianne'] but SPEAKER_01 (mapped to ['Marcus'])
holds 88% of its detected voice (coverage 96%): 'were watching from the wings'.
Nothing was overwritten -- a silent overwrite of an authored fact is the
defect, not the fix. If the detection is right, fix the tag in the lyrics file.
```

Three reasons it resolves that way, in increasing order of how much they
decide:

1. **It is this project's existing hierarchy, one layer up.** Forced alignment
   fits timestamps to human-supplied text and never renegotiates the text. A
   `[Name: Role]` tag is the same kind of human-supplied fact about the same
   audio.
2. **The two mistakes cost differently.** Tag right / detection wrong, resolved
   the detector's way, puts the wrong face on screen and reports *nothing* —
   the silent-overwrite defect, and the one that actually happened on "The
   Lucky Ones". Tag wrong / detection right, resolved the tag's way, puts the
   authored face on screen and names the clash, so the operator is handed the
   cheap edit.
3. **Detection has no name of its own.** The name comes from
   `diarization_speakers`, which is also authored. So "detection overrides the
   tag" really means "one authored mapping overrides another authored tag" —
   and the tag is the more specific, more local statement.

A tag the detection *agrees* with is counted and not logged. Agreement is the
expected case; a line per segment would bury the clashes, which are the only
thing anybody has to act on.

### The scope of a tag is the scope of the protection

The lyrics format defines a tag as running until the next tag, so every line a
tag covers is authored, not just the line the tag sits on. One consequence
worth knowing: **a song where only the first verse is tagged gets no
assignments at all** — the rest of the file is still inside that tag's scope.
That is deliberately conservative, and the disagreement list is what tells you
it happened. A file with no tags anywhere is the case this feature is actually
for.

## Confidence, coverage and harmonies

pyannote's `Annotation` carries **no per-turn probability**, so this module does
not invent one. It measures two things it can compute from the turns
themselves, per aligned segment:

| Quantity | Meaning | Default floor |
|---|---|---|
| `coverage` | fraction of the segment any speaker turn covers | `0.5` |
| `share` | the winner's fraction of the **covered** time | `0.6` |

The winner is whichever speaker contributes the most overlapping duration
**inside the segment** — issue #40's dominant-voice rule as #92 re-measured it,
and deliberately the same measure `slicing._dominant_character_member` uses, so
the two cannot disagree about who a passage belongs to. Ties keep the
earliest-starting speaker, matching that function's strict `>`.

Where this and `slicing` part company is what happens when nobody dominates.
Slicing *must* pick someone — it has a chunk to render and one photograph to
stage. Diarization can decline, because declining leaves a known, nameable
value behind (the tag, or `default_lead_vocalist`) rather than a coin flip. So
a 50/50 harmony is reported `contested` and falls back loudly. **The wrong face
on screen is worse than the default face**: the default is a failure mode a
viewer can be warned about, a wrong guess looks confident and is found only by
someone who knows what the singer sounds like.

Emitting *both* names on a harmony was considered and rejected for now.
`characters` is plural and "both audible" is exactly what a harmony is — but a
thrashing diarizer produces spurious two-character segments, and plural
semantics invented from a classifier's noise is a worse failure than a named
fallback. Revisit it with a real overlapping passage measured.

### The thresholds are shapes, not measurements

`0.5` and `0.6` have never been scored against a hand-tagged multi-vocalist
song. They are the design document's third unmet precondition, and **the report
says so at WARNING on every run**:

```
Diarization thresholds are UNMEASURED: min_coverage=0.50 and min_share=0.60
have never been scored against a hand-tagged multi-vocalist song, so they are
shapes, not measurements.
```

What would replace them, in this project's usual form: take a song with a real
handoff, hand-tag every line, run diarization with the tags in place, and read
the `agreed` / `disagreed` split as a score — the tags are the labels, so this
is a scored corpus with a known outcome, not a preference. Then sweep
`min_coverage` and `min_share` and record the *excluded* values with the reason,
the way `scenecuts.py`'s threshold was chosen. Until that exists, a number here
is a guess wearing a number.

## When it cannot run

Four distinct failures, each with its own exception type and its own message
naming the remedy. **All four degrade**: the message goes to the log at ERROR,
the alignment comes back untouched, and the run proceeds exactly as it would
with `diarize = false` — manual tags and the default lead. A render is hours of
GPU custody, and ending one over an unset environment variable is the expensive
mistake; `--prepare` is where to find this out, in about a minute with no GPU.

| Cause | Exception | What the message says to do |
|---|---|---|
| package absent | `DiarizerUnavailableError` | `pip install -e ".[diarize]"` |
| no token | `DiarizerTokenError` | set `HF_TOKEN` (never `run.toml`), and accept the three pages |
| token rejected | `DiarizerTokenError` | make a new read-scoped token |
| terms not accepted (403) | `DiarizerAccessError` | accept all three gated pages **as the account owning the token** |
| weights not cached / offline | `DiarizerWeightsError` | run online once; cache lives under `HF_HOME` |

The 403 case is kept separate from the token case on purpose: the remedy is not
a new token, and reporting it as one sends the operator in the wrong direction.

## How it is wired, and what it does not touch

```
lyrics ──► parse_lyrics ──► align() ──► [diarization] ──► slice_audio ──► ...
             (tags)        (timeline)   (characters)      (chunks)
```

It runs between `align()` and `slice_audio()` inside
`cli._align_and_slice_timeline`, because `AlignedSegment.characters` is the
field slicing derives `AudioChunk.characters` from. Everything after Stage 2a —
`prompting`'s active-member resolution, the staged reference photo, #92's
dominant-voice attribution — is unchanged by construction and cannot tell
whether an attribution came from a tag or a classifier.

Song-only, like the vocal stem it reads: a prologue timeline (issue #66) has no
stem and no singers to tell apart.

Counterpoint segments (issue #33's `[simultaneously]` blocks) are **not**
diarized. The aligner never heard those voices separately, so their timings are
derived rather than measured, and their characters came from an explicit
sub-block tag either way.

`--resume` needs no new fingerprint field: changing the mapping changes
`characters`, which changes the composed prompt, which is already in
`ChunkFingerprint`'s `prompt_hash`. A run whose attribution moved re-renders
the chunks whose prompts moved and no others.

## What is still unverified

Honest status, so nobody reads a tested mechanism as a measured result.

**Blocked on the weights being accessible.** As of 2026-10-04 the terms are
accepted in principle but the account had not clicked through, and the Hugging
Face API returns `200` for metadata and `403` for the files. So nothing here
has ever loaded a real pipeline. Untestable until it can:

- that `pyannote/speaker-diarization-3.1` loads at all, on this stack, with
  these weights;
- that `Pipeline.from_pretrained`'s real failure text matches the substrings
  `_classify_load_failure` sorts on (`403`, `gated`, `awaiting`, `401`, …).
  Every branch still *refuses* if a match is missed — the classification only
  chooses which hint the operator gets — but the hint is the point;
- the real runtime and VRAM cost of a diarization pass, and whether it belongs
  on the 4090 at all or should be a CPU pass like alignment;
- whether `spans_from_annotation`'s duck-typing survives a real `Annotation`
  (it depends on exactly one method, `itertracks(yield_label=True)`).

**Blocked on a real multi-vocalist song.** "Deathless" is single-vocalist and
"The Lucky Ones" has never been rendered on a stem, so there is nothing here to
score against. Untestable until one exists:

- that the clusters correspond to the singers at all, rather than to recording
  sessions, processing chains or sections of the song;
- the `0.5` / `0.6` thresholds, as above;
- that the dominant-voice rule is right for a real overlapping-vocal passage,
  rather than inherited from the single-voice case;
- how many clusters a real pipeline emits for two singers (it may emit three,
  or one);
- whether the agreement rate on confidently-labelled segments is good enough to
  recommend this over hand-tagging at all. **Until that is measured,
  hand-tagging with `[Name: Role]` remains the recommended workflow**, and it
  is not a stopgap: it is fully specified, tested end to end, and already the
  documented answer.

Both halves of that list are #101's acceptance criterion, which asks for a real
multi-vocalist song — the same song #29's manual half still needs for its own
confirmation. The two can share one render.
