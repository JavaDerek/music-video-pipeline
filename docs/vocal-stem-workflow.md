# Conditioning H3 on an isolated vocal stem

MiniMax H3 lip-syncs to whatever waveform it is conditioned on. By default
that is a slice of the **full mix**, where the voice competes with drums, bass
and guitars — and where a voice-like lead synth is not distinguishable from a
singer. This document describes the optional workflow (issue #25) that
conditions each chunk on a slice of an isolated **vocal stem** instead.

Implementation:
[`music_video_maker/stems.py`](../music_video_maker/stems.py).

## This does not change what you hear

**The master audio track is still the only audio in the final video.** The
stem exists solely as the Stage 3 conditioning input:

- Stage 1 aligns the lyrics against the **master**. The stem never gets an
  alignment of its own.
- Stage 2a slices the **master** and plans the chunk timeline, exactly as
  before.
- Stage 2a-stem (this module) cuts the **stem** at those same chunk spans —
  the same millisecond rounding, byte offset for byte offset — and re-points
  each `AudioChunk.audio_file` at the resulting slice. No span, frame count,
  lyric or character changes.
- Stage 3 uploads that slice; Stage 4 renders against it.
- Stage 5 is untouched: the generated audio is discarded and the pristine
  master is muxed over the concatenated video, as always.

Sync therefore still comes from the master's alignment timestamps. The stem
only ever changes *what H3 listens to while animating a mouth*.

## Producing the stem

Separation is deliberately **off the render path** — it is a one-off,
cacheable, inspectable act performed by you, not an inference call inside the
pipeline. Demucs (Meta, MIT) is the tool of choice:

```bash
pip install demucs
python -m demucs --two-stems=vocals -o stems/ ~/mvm-runs/<song>/audio/master.wav
# -> stems/htdemucs/master/vocals.wav
# -> stems/htdemucs/master/no_vocals.wav
```

On doris this runs in well under a minute on the 4090 — but it is a GPU job,
so run it **before** taking custody for a render, not during one. It works on
CPU too.

Keeping it manual keeps a heavy ML dependency out of the render path, keeps
the run deterministic and resumable, and — the real reason — keeps the stem an
artefact you can listen to before spending GPU hours on it.

## Listen to `no_vocals.wav`, not just `vocals.wav`

The failure mode of this feature is **not** a noisy stem. It is a stem that
silently *omits* a voice.

`htdemucs` separates into drums/bass/other/vocals, and a heavily processed or
vocoded voice is genuinely ambiguous to that classifier — it can land in
`other`. "The Lucky Ones" ends on three lines sung by a vocoded version of
Jan's voice. If those land in `other`, conditioning on `vocals.wav` makes that
passage near-silence and H3 renders a **closed mouth over the most conspicuous
moment in the track** — a regression caused by this fix, and easy to misread
as "the stem improved instrumental behaviour".

So the check is: play `no_vocals.wav` and listen for voice that should not be
there. `vocals.wav` sounding clean proves nothing about what is missing from
it.

`stems.py` also checks this mechanically. Every chunk that carries a lyric but
whose stem slice peaks below −45 dBFS is named in a `WARNING`, and the
`StemQualityReport` returned by `slice_stem_for_chunks` lists them in
`silent_voiced_chunk_ids`. A run that logs that warning should be stopped and
the stem re-examined — those chunks will render mute mouths.

The report also lists `leaky_instrumental_chunk_ids`: instrumental chunks
whose stem slice still peaks above −30 dBFS, i.e. band audio that leaked into
the vocal stem. That weakens (but does not break) the near-silence this
conditioning relies on.

## Validation, and why it refuses rather than pads

`slice_stem_for_chunks` refuses the run if:

- the stem is missing or unreadable;
- the stem's duration differs from the master's by more than 100 ms
  (`DEFAULT_DURATION_TOLERANCE_SECONDS`);
- the stem is more than 1 s shorter than the end of the chunk timeline
  (`DEFAULT_MAX_TAIL_PAD_SECONDS`), even when no master is supplied;
- an emitted slice's duration disagrees with the chunk's span or with the
  duration its `frame_count` implies.

The duration check is the important one. Real separation output is
sample-identical to its input, so a stem of a different length was made from a
different cut of the song. Every chunk after the divergence would then be
conditioned on the wrong moment of the track *while still being exactly the
right length* — nothing downstream could notice. Padding or stretching it
would hide precisely the bug the check exists to catch, so this is a refusal
with both durations named.

The one place silence *is* added is the tail: a grid-quantized outro chunk can
legitimately land up to one grid step (~0.708 s) past the final sample, which
`slicing.py` already pads on the master for the same reason. That is logged at
INFO and reported in `padded_chunk_ids`.

## Output layout

Slices are written as `chunk_{chunk_id:03d}_vocal.wav` into their own
directory — **not** the mix's `chunks_dir`. Both sets are worth keeping: the
mix slices are the control for the A/B below, and having both on disk lets you
listen to what H3 was actually given for any chunk that renders badly.

## How to evaluate it

Do not change two variables at once, and do not decide this from one chunk.

1. Render a handful of sung chunks with the current (mix) setup as the
   control — chunks 17–19 of "The Lucky Ones" are the sharpest test: the
   shared-words chorus with both voices in the mix, where lip-sync failed and
   a two-reference experiment already ruled out reference *count* as the
   cause.
2. Re-render the same chunk ids with the vocal stem, same seed, same
   resolution, same shot plan.
3. Compare side by side. Chunks 20–23 (the 27 s instrumental) are the matching
   b-roll control — the 3:15 passage where a squealy lead synth was being
   lip-synced as if it were a voice is the other thing a stem should fix.

If the difference is not visible at that sample size, it is not worth the
added dependency and the extra failure mode.

**Re-render, do not `--resume`.** Changing the conditioning audio changes what
a chunk *is* without moving its span, so a resumed run would happily reuse the
mix-conditioned mp4s and quietly hand you the control twice. Until the run
fingerprint records the conditioning source, run the A/B into separate output
directories.
