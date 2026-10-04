# Design: automatic vocalist detection (issue #29, built as #101)

**BUILT 2026-10-04.** This was design-only until the licence question below
was answered; it is now implemented in
[`music_video_maker/diarization.py`](../music_video_maker/diarization.py), and
the operator-facing half — the one-time gated-model setup, the per-song
workflow, the attribution CC-BY-4.0 requires, and exactly what is still
unverified — lives in
[`docs/vocalist-diarization.md`](vocalist-diarization.md). **Read that one if
you are using the feature; read this one for the shape and the reasoning**,
which the implementation follows and which is worth keeping because it is the
argument, not the manual.

Every load-bearing decision below was built as written, with two additions
the prose did not anticipate and one deliberate departure:

- *addition:* a tag's protection needed a field. "An explicit tag always wins"
  is unimplementable while `("Dianne",)` from a tag and `("Dianne",)` from
  `default_lead_vocalist` are the same value, so
  `LyricLine.characters_authored` / `AlignedSegment.characters_authored` record
  which it was. Same shape as this project's own "a resolved config cannot tell
  set from defaulted" (#55): a fact only knowable while parsing has to be
  written down while parsing.
- *addition:* the fifth open question — how a cluster label becomes a name —
  is answered as **authored data** (`[diarization_speakers]`), validated
  against `[cast]` at load, with the cluster table logged so a first run is the
  thing that produces the mapping. Not reference-clip matching: that is
  inference where a human's single listen is cheaper and exact.
- *departure:* the harmony rule. "The dominant voice wins" is implemented for
  a clear winner, but where nobody clears the share floor the segment is
  reported `contested` and **left alone** rather than attributed. Slicing must
  pick someone because it has a chunk to render; diarization can decline, and
  declining leaves a nameable value behind instead of a coin flip.

See [`docs/lyrics-format.md`](lyrics-format.md) for the manual
`[Name]` / `[Name: Role]` tags this sits beside, which work end to end
(`tests/test_multi_vocalist.py`), are not being replaced, and remain the
recommended workflow until the automatic path has been scored on a real song.

## The shape: an alternative front-end, not a parallel mechanism

`AlignedSegment.characters` is the one place "who is singing this" lives
downstream of Stage 1. Manual tags populate it today by a chain that is
already tested: `[Name]` tag → `LyricLine.characters` →
`align()`'s positional word-owner walk → `AlignedSegment.characters` →
`AudioChunk.characters` → `expand_prompt`'s active-member resolution → the
staged reference photo.

A diarization front-end's entire job is to populate that **same field**,
nothing else. It must not introduce a second character field, a second
resolution path in `prompting.py`, or a second reference-image lookup in
`staging.py`. If it needs to change anything downstream of
`AlignedSegment.characters` to work, that is a sign the design is wrong, not
a sign the downstream code needs a diarization special case — the entire
value of this shape is that the render path stays innocent of whether a
character assignment came from a human's tag or a classifier's guess.

Concretely: diarization is a second candidate producer of
`LyricLine.characters` (or, more precisely, a candidate producer of a
per-line/per-span character assignment that gets folded in at the same
point tags are today), not a new stage bolted onto the end of the pipeline.

## Manual tags always win

An explicit `[Name: Role]` tag is authored by a human who listened to the
song, or at least means to have. A diarization guess is a classifier's best
effort on a stem it may never have heard cleanly. Where both exist for the
same line, the tag wins, unconditionally — detection is a labour-saving
default for the lines nobody tagged, never an authority that can override
what a person wrote down. This mirrors the project's existing rule for
lyrics themselves (forced alignment fits timestamps to human-supplied text;
it does not renegotiate the text) at one layer up: human input is ground
truth, and an inference step is not allowed to contradict it.

## Confidence and fallback

A low-confidence assignment must fall back to `config.default_lead_vocalist`
and log loudly at the point of fallback — chunk id, the candidate speaker
label, and the confidence score that failed the bar. It must never guess
silently. The wrong face on screen is worse than the default face: the
default is a known, named failure mode a viewer can be warned about ("nobody
tagged this song, so untagged spans show the lead"); a wrong guess looks
confident and is discovered only by someone who knows what the singer
actually sounds like, which is usually nobody involved in the render.

## Harmonies and doubled vocals

Two singers at once is common in this material, and diarization will either
pick one voice or thrash between labels within a single chunk. The defined
behaviour: **the dominant voice in the chunk wins** — the same rule
`slicing.py`'s merge-attribution logic already uses for two different
singers' segments landing in one merged chunk (voiced duration, not word
count, decides the winner; see `tests/test_slicing.py`'s
`test_merged_chunk_*` family). Diarization should reuse that measure rather
than inventing a second one. A shot the author actually wants staged as a
deliberate two-shot (both singers on screen) is not a job for the automatic
path at all — that is what the shot plan's `present` field is for, and it
already overrides whatever chose the frame's primary singer.

## Dependency reality (resolved: an optional `[diarize]` extra)

The practical route is speaker diarization on an isolated vocal stem:
`pyannote.audio` for diarization, plus a Demucs-separated vocal stem as its
input (see [`docs/vocal-stem-workflow.md`](vocal-stem-workflow.md), issue
#25, which this would share the stem-separation step with).

**Half of that is no longer hypothetical: #25 shipped.** `vocal_stem` is a
real config field, `music_video_maker/stems.py` cuts the stem at the
master's own chunk spans, and separation is already defined as an off-path,
one-off, operator-run act (`python -m demucs --two-stems=vocals`) rather
than something the pipeline calls. So a diarizer's *input* exists today and
costs this project no new dependency — the operator produces the same file
either way. What is left to decide is only the diarizer itself.

`pyannote.audio` and Demucs are both heavy,
optional-extra-shaped dependencies with their own model weights. **That is how
it shipped:** `pyannote.audio` is the `[diarize]` extra (never a runtime
dependency, never imported unless `diarize = true`, and never imported at
module load even then), and Demucs stays off the render path entirely — the
operator produces the stem by hand, as issue #25 already defined. Building the real thing means answering,
up front, the same licensing/redistribution questions this project already
applies to every other model file (`CLAUDE.md`'s "Everything committed here
is intended to become public" section) — `pyannote.audio`'s pretrained
pipelines are gated behind their own license acceptance, which is a
one-time human step, not something the render path can silently satisfy.

## Offline testability

Whatever library ends up doing the diarization, its output must be injected
through a seam exactly like `alignment.py`'s `model` parameter: a plain
object or function returning speaker-labelled time ranges, so tests provide
a fake and never invoke `pyannote.audio`, download a model, or touch a GPU.
The same rule that makes `align()`'s tests fully offline today
(`tests/test_alignment.py`'s `_fake_model`) has to hold here, unchanged —
this project does not get to make an exception for its own next feature.

## What would have to be true before building this

- A song where diarization is worth it exists and has been listened to: at
  least two distinct singers, on a real isolated stem, not a hypothetical.
- The confidence signal `pyannote.audio` actually exposes has been measured
  against that song's known-correct tagging (the kind of scored-on-a-corpus
  check `CLAUDE.md`'s lint findings all insist on) — a threshold picked
  without that measurement is a guess wearing a number.
- The dominant-voice-wins rule for harmonies has been sanity-checked against
  at least one real overlapping-vocal passage, not assumed from the
  single-voice case.
- Someone has decided how a diarized speaker label ("SPEAKER_00") maps to a
  cast member name — by asking once per song, or by matching short reference
  clips — since diarization alone never produces a name, only a cluster.

A fifth thing is a *decision*, not a measurement, and it is the one that
actually blocked: **`pyannote.audio`'s pretrained pipelines are gated behind
accepting their terms on Hugging Face, and this repo is a candidate for open
sourcing.** That is a one-time human step nobody but the owner can take, it
cannot be satisfied from inside the render path, and it has to be answered
before any code is written — not discovered afterwards, the way CLAUDE.md's
"Check redistribution before committing a third-party binary" rule says.

**Answered 2026-10-04: accepted.** The code is MIT, the pretrained models are
CC-BY-4.0, and commercial use is permitted *with attribution* — so the terms
are compatible with publishing this repo, and the obligation is a credit, not
a restriction. Nothing is committed or redistributed here: the models stay
gated, the operator accepts the terms with their own account, and the weights
land in their own cache, which is the arrangement `faces.py` already uses for
the SFace weights. The attribution is recorded in the README's License section
*and logged at INFO on every run that loads the pipeline*, because CC-BY asks
for credit from whoever uses the work, not only from whoever reads the
repository.

**The other four are still not true, and that is the honest status of the
built feature.** The mechanism is implemented and tested offline; the *claim*
that it attributes a real song correctly is not made, because no
multi-vocalist song has been diarized here and the weights were still
inaccessible (403 on the files) when it shipped. The two thresholds say so at
WARNING on every run, and
[`docs/vocalist-diarization.md`](vocalist-diarization.md)'s "What is still
unverified" section is the full list.

So hand-tagging with `[Name: Role]` remains the correct answer, and it is not
a stopgap: it is fully specified, tested, and already the recommended workflow
in [`docs/lyrics-format.md`](lyrics-format.md).
