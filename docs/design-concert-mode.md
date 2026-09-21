# Concert mode: a click-track-synced backdrop video (issue #22)

A second output mode: rear-projection / LED-wall backdrop for a live band, cut
to the click track the band already plays to, conceptual rather than a literal
depiction of the band.

Three things change, and each of them changes an invariant, which is why this
is a mode and not a flag: the **timeline source**, the **conditioning**, and
the **output contract**.

This document records what has been built (two of the small pieces), what is
blocked and on whom, and one thing measured on the real render that changes
the priority of the third piece.

## Status

| Piece | State |
|---|---|
| The `(time, label)` marker contract, reader, fixture | **built** — `music_video_maker/markers.py`, `tests/test_markers.py` |
| An Ableton `.als` Locator reader producing that contract | **built, UNVERIFIED against a real set file** — `music_video_maker/ableton.py`; see "The Ableton reader" below |
| `format_marker_csv`: a DAW reader *produces* the contract rather than bypassing it | **built** — `markers.format_marker_csv` |
| Silent-output assembly + measured duration check | **built** — `assembly.assemble_final_video(master_audio=None, expected_duration=…)` |
| The duration check armed end to end on the silent path | **built** — `run_pipeline` now passes `expected_duration=alignment.track_duration` whenever `silent_output = true`; see "The duration check" below |
| Stage 2 timeline-drift reporting (before any GPU work) | **built** — `slicing.timeline_track_drift_seconds` + `cli._log_timeline_track_drift`, wired into both `run_pipeline` and `--prepare` |
| What the click tracks actually are | **blocked on the owner / the band** — the eight questions below are unchanged |
| t2v (or i2v) workflow template, projection aspect ratio | **blocked on the GPU** |
| Wiring the marker reader as an alternate Stage 1 | **deliberately not done** — see "Why nothing is wired up" |

## The measurement that reorders the work

Issue #22 says duration accuracy "becomes the hard requirement" and that the
acceptance criterion should be *measured* total duration, not computed. It is
sharper than that. Measured on the finished "Deathless" render
(`~/mvm-runs/deathless`, 2026-08-22):

| | seconds | frames @24 |
|---|---|---|
| master track (`audio/master.wav`) | 512.080 | — |
| chunk timeline (last chunk 79 ends at) | 513.917 | 12334 |
| `output/final_v12/_concat_intermediate.mp4` (silent concat) | 513.931 | 12334 |
| `output/final_v12/final_video.mp4` video stream | 511.973 | 12287 |

The chunk timeline **overshoots the master track by 1.837 s**. Chunk 79 spans
508.750–513.917 s: only 3.33 s of track remained, and 124 frames is H3's
trained minimum, so the final tile is padded past the end of the song. That is
correct behaviour for a music video — and it is invisible there, because the
mux's `-shortest` throws the overshoot away. 47 rendered frames were discarded
from `final_v12`, and nothing logged it.

**Remove the audio and nothing throws it away.** A silent concert output built
by today's pipeline is 1.84 s longer than the click track it was cut to.
Nothing in the pipeline would notice, no test would fail, and the first
symptom is a backdrop drifting against a live band on a stage.

So the duration check is not paperwork bolted onto the silent path. It is the
thing that replaces `-shortest`, and it is why the silent path was built with
it rather than after it.

## Change 1: the timeline source

Stage 1 exists to answer one question — where, in seconds, does each beat of
the song fall? For a live show that answer is *authored*: exact, identical
every night, and better than forced alignment can ever be.

### The contract is a marker file, not a DAW file

`music_video_maker/markers.py` reads a labelled-marker CSV:

```csv
# Deathless, click-track sections. Synthetic example.
time,label
0,Count-in
8.0,Intro
0:24.000,Verse 1
...
```

`Marker(time, label)` → `MarkerTrack` → `marker_sections(track, duration)` →
`to_alignment_result(track, duration, label_as_text=…)`, the same
`AlignmentResult` shape Stage 2 already consumes.

Times accept plain seconds or `[HH:]MM:SS[.mmm]`; both are common DAW exports.
Blank lines and `#` comments are skipped so a file can carry its own
provenance. Every error names the 1-based line and the offending text.

**Why a generic CSV and not an Ableton reader.** Issue #22's own step 1 —
"find out what the click tracks actually are" — is unanswered, and an `.als`
is gzipped XML whose schema changes between major versions. Writing a parser
against a guessed schema is the expensive kind of wrong. `MarkerSource` is a
`Protocol` with a single `read() -> MarkerTrack`, so an Ableton reader has a
named place to land the day the questions below are answered; there is no stub
class that raises, because a stub that raises is a half-built feature and this
repo is going public.

### Questions only the owner or the band can answer

These gate the reader that does not exist yet. They cost an afternoon and no
GPU:

1. Is it Ableton at all, and which major version? (`.als` schema is
   version-dependent.)
2. Are the section labels in **Locators**, in clip names, or in a separate
   cue/marker list?
3. Is tempo constant per song, or is there tempo automation? Constant is a
   much simpler beats→seconds conversion; automation means integrating a tempo
   map.
4. **Is there already a simpler export?** A marker/cue CSV, a `.mid` tempo
   map, or MIDI markers would avoid parsing `.als` entirely. Check this before
   writing any XML.
5. What is the authoritative song duration for a show — the arrangement
   length, or the audio file the playback rig fires? These differ, and the
   duration check above needs the one the rig honours.
6. What does the backdrop do at the **end** of a song and between songs? Held
   final frame, black, or a loopable idle. This is a playback-rig question as
   much as a rendering one, and it changes what the last chunk should contain.
7. What is the rig? Resolume / QLab / a video plugin in the same Ableton
   session — it dictates container, codec, frame rate and colour range, and it
   should be settled **before** rendering a show's worth of video.
8. What are the actual panel dimensions? (See "aspect ratio" below.)

### The trap in the adapter

A marker label is a **section name**, not a lyric. `AlignedSegment.text` is
consumed downstream as the literal lyric line composed into the render prompt.
So `to_alignment_result` takes `label_as_text` as a keyword argument **with no
default** — the caller must state which it means — and logs a WARNING every
time it is `True`. A concert-mode prompt composer must consume a label as
section identity; if "Chorus 2" ever reaches a prompt as a sung line, this is
where it came in.

### The Ableton reader (built 2026-09-21, and what it cannot claim)

`music_video_maker/ableton.py` reads a Live set's **Locators** and produces a
`MarkerTrack` — the same shape the CSV reader produces, so everything
downstream is unchanged. `markers.format_marker_csv` writes one back out, which
is what makes "the CSV is the contract, a DAW reader is one producer of it"
true in practice rather than in a docstring: an operator gets a file they can
read, diff, correct by hand and commit beside the run config.

**It has never been given a real `.als`.** Every test runs against a synthetic
fixture this repo authored (`tests/fixtures/markers/synthetic_live_set.als.xml`,
committed as plain XML precisely so it can be reviewed in a diff, and gzipped
at test time). It is written against Ableton's *documented* on-disk shape.
Step 1's questions 1, 2 and 4 are still unanswered, and the honest status is
"a starting point that will need correcting against a real export" — if the
band can export a marker CSV directly (question 4), that is still the cheaper
answer and this module is unnecessary.

Two things it refuses rather than guesses, which is the part worth having
before a real file arrives:

* **Tempo automation.** A Locator's `Time` is in **beats**. With one manual
  tempo, beats → seconds is a division; with automation it is an integral, and
  using the manual value anyway places every marker after the first tempo
  change wrong — progressively worse through the song, with nothing to notice
  it, which is this issue's own failure mode. That is design question 3,
  answered by refusing. `tempo_bpm` lets an operator supply the click's own
  tempo instead, and the reader says loudly when that disagrees with the file.
* **A set with no Locators.** Clip names are a different claim — a clip has a
  length and can be moved or duplicated — and reading them as section markers
  would produce a plausible-looking timeline nobody authored. The refusal names
  the alternatives, because *which of the three* is question 2 and it is open.

Both spellings of the tempo track are looked for (`MasterTrack`, and Live 12's
`MainTrack`), since question 1 — which major version — is also open.

## Change 2: conceptual conditioning, not the cast

Blocked on the GPU, and unchanged from the issue's own analysis:

* **No cast reference photos, no lip-sync.** `MiniMaxH3ReferenceToVideo`
  exists to drive a face from an audio stem; it is the wrong node here.
  ComfyUI ships `video_minimax_h3_t2v.json` and `video_minimax_h3_i2v.json`
  templates alongside the r2v one this project uses, both confirmed present on
  doris. Authoring a third template is the Stage-4 work.
* **Audio conditioning becomes optional**, and whether to feed the stem at all
  is worth measuring — it may be cheaper and it removes a class of artefact.
  Note that F26 already found the conditioning audio outranks the prompt
  describing it (`instrumental_audio_gain_db` exists because rewording could
  not beat the audio), so "feed music with no mouth in frame" is not obviously
  harmless.
* **The shot plan becomes mandatory rather than optional.** With no lyric and
  no face, the authored per-chunk direction is the entire prompt.
* **I2V continuity moves from optional to required.** A backdrop that
  re-layouts every six seconds is distracting behind a band. Note the
  precondition this project already learned: on the chained path the seed frame
  *is* the conditioning, and #47's face gate is meaningless here because there
  are no faces — the gate would refuse every chain. Concert mode needs
  `i2v_require_seed_face = false` and something else to decide whether a seed
  frame is worth chaining from, or nothing at all.

### Projection constraints (real, and easy to get wrong)

* **High contrast on near-black.** Rear projection competes with stage
  lighting; mid-grey and naturalistic daylight disappear under front light.
  This is a `cinematography` statement, and it is exactly the kind of
  whole-video look that belongs in a locked profile (#55) — a
  `profiles/stage-backdrop-v1.toml` is the natural home for it.
* **No fine detail, no text.** Lost at projection distance and at LED-wall
  pitch.
* **Aspect ratio is not 16:9.** Walls are frequently very wide (3840×1080) or
  non-standard. `render_width`/`render_height` already exist; what is unknown
  is (a) the rig's real dimensions and (b) whether H3 behaves at extreme
  aspect ratios — its trained range is not documented here. That is a cheap
  test *before* planning a show: one chunk at the target ratio, look at it.
  Remember resolution is the dominant cost lever (864×480 ≈ 3.7 min/chunk vs
  1344×768 = 9 m 15 s), and 3840×1080 is 10× the latent volume of 864×480.
* **Motion should be slow**, which argues for longer chunks — the same
  argument as #21, and worth more here than for a music video.

## Change 3: the output contract inverts

Two audio invariants change meaning, and exactly one of them is suspended:

* *"The master audio track is the only audio in the final video"* → for a
  backdrop there is **no audio at all**. The band is the audio; shipping a file
  with an audio track risks double-audio if a playback rig un-mutes it.
  `assemble_final_video(master_audio=None)` implements this, logs it at
  WARNING naming the suspended invariant, and sets `AssemblyResult.has_audio =
  False`.
* *"Generated audio is always discarded"* → **untouched.** The concat pass
  still passes `-an`, so H3's own generated audio never survives. This is the
  invariant that must not move, and it does not.
* *"Lyrics are immutable truth"* → still true. They are simply no longer the
  timeline; they are source material for the shot plan.

The concat pass writes straight to the final output when there is no mux —
there is no second pass to feed, and copying a 96 MB file again for nothing is
a real cost on an hours-long render.

### The duration check

`expected_duration` (opt-in) probes the finished file with `ffprobe
-show_entries format=duration` through the same injected runner, records
`AssemblyResult.measured_duration`, and **raises** `DurationMismatchError` if
the drift exceeds `duration_tolerance_seconds`.

* `format=duration` rather than a stream duration: the container duration is
  what a playback rig honours, and it is what a `-c:v copy` concat produces.
* It raises rather than warns, *after* the file is written: the file is still
  on disk to inspect, and for a show the operator has to be told now, not find
  it in a log tomorrow.
* The 0.05 s default is one frame at 24 fps rounded up. It is **not** a
  measured figure. A real rig's tolerance should come from the playback
  system — `duration_tolerance_seconds` in `run.toml` overrides it, and is
  only ever consulted when `silent_output = true`.
* When `expected_duration` is not given, nothing new happens — no probe, no
  subprocess, no behaviour change for the music-video path.

**Now armed end to end.** The gap this issue's own comment claimed closed and
didn't: `run_pipeline` (`cli.py`) now passes
`expected_duration=alignment.track_duration` whenever `config.silent_output`
is true, and leaves it `None` otherwise — so `silent_output = true` in
`run.toml` really does wire the check into a render, and the music-video path
still never probes anything. The master track's own duration (already known
from Stage 1, no second probe) stands in for the authoritative show duration
here — design doc question 5 above is still open, and the call site names
that explicitly rather than quietly answering it.

**Reported before GPU time, not just after assembly.** A new pure function,
`slicing.timeline_track_drift_seconds(chunks, track_duration)`, reads Stage
2's own finished chunk timeline against the master track and returns the
drift (positive = overshoot, negative = undershoot). `cli._log_timeline_track_drift`
wraps it with the logging policy and runs from **both** `run_pipeline` and
`--prepare` — the 50 s, no-GPU check this project already uses — right after
Stage 1-2 completes:

* Music-video overshoot: WARNING, naming the frame count `-shortest` will
  discard. Chosen over INFO because it quantifies rendered-but-discarded GPU
  time, and the entire reason this issue exists is that nothing surfaced this
  number before.
* Silent-path overshoot: ERROR, because it predicts the post-assembly
  duration check above will raise, and an operator should learn that before
  committing GPU hours, not after.
* Undershoot (either path): ERROR, always — the worse defect, since on the
  music-video path `-shortest` ends the file with the picture and cuts the
  song's ending off (and on the silent path the file is short of the track). Building this
  report found a real case — a trailing gap shorter than one trained-floor
  chunk (~5.167 s) was dropped rather than covered — fixed the same day in
  `slicing._cover_instrumentals`.

Given the overshoot measured above, the honest expectation is that the first
concert render **fails this check**, and now says so twice: once at Stage 2,
before any GPU time is spent, and again — authoritatively, on the real
assembled file — when `expected_duration` raises after assembly. The fix
itself is still upstream: either the final tile is trimmed to the track (a
Stage-2 change, and note H3's 124-frame floor means the last tile cannot
simply be made shorter), or the click track is padded to a legal grid length.
That is a decision for whoever owns Stage 2, and it should be made with the
click track in hand — nothing built here makes that call.

## Why nothing is wired up

There is still no `concert = true`, no `marker_file` config key, and no CLI
flag for the mode itself. Deliberately:

* The marker reader has no consumer until the t2v template exists, and the
  template needs the GPU.
* Wiring a mode switch that half of the mode does not honour is worse than no
  switch: it invites a run that reads markers, then composes labels as lyrics
  into an r2v graph conditioned on cast photos, and produces something that
  looks *nearly* right.
* Both built pieces are useful on their own today, and one of them is now
  useful on both output paths: silent output is what you hand a VJ; the
  duration check is what tells the operator before a show; and the Stage 2
  drift report above means a music-video run finally logs that GPU-rendered
  frames are being thrown away, which nothing did before this issue.

The order to finish in is the issue's own: answer the questions, then the
template + a cost measurement at the real aspect ratio, then the mode switch,
then a duration-accuracy test against a real click track.
