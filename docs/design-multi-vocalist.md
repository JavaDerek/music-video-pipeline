# Design: automatic vocalist detection (issue #29)

Design only — nothing here is implemented. This document exists so the next
person to pick up #29's automatic-detection half starts from a shape, not a
blank page. See [`docs/lyrics-format.md`](lyrics-format.md) for the manual
`[Name]` / `[Name: Role]` tags this is meant to sit beside, which already
work end to end (`tests/test_multi_vocalist.py`) and are not being replaced.

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

## Dependency reality: not being added now

The practical route is speaker diarization on an isolated vocal stem:
`pyannote.audio` for diarization, plus a Demucs-separated vocal stem as its
input (see [`docs/vocal-stem-workflow.md`](vocal-stem-workflow.md), issue
#25, which this would share the stem-separation step with). Both are heavy,
optional-extra-shaped dependencies with their own model weights, and neither
is being added by this document. Building the real thing means answering,
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

Until all four are true, hand-tagging with `[Name: Role]` remains the
correct answer, and it is not a stopgap: it is fully specified, tested, and
already the recommended workflow in
[`docs/lyrics-format.md`](lyrics-format.md).
