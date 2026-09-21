# Voicing corpus (issue #96)

The register the voiced-vs-unvoiced lint gets calibrated from. Same role as
`docs/idiom-corpus.md` and `docs/deathless-render-corpus.md`: a place where the
numbers live so the next revision starts from data rather than from a rebuilt
experiment.

`alignment_quality.VOICING_RATIO_THRESHOLD` currently ships **uncalibrated**.
Everything below marked "to be filled in" is a hole in the evidence, not an
omission from the write-up.

## What the check measures

Per placed segment, over the same mono-16 kHz decode issues #71 and #80
already make:

| column | meaning |
|---|---|
| `vf` | fraction of the segment's *audible* analysis frames whose normalized autocorrelation peak reaches `voicing.VOICED_FRAME_NCCF` (0.45) |
| `vf_ratio` | `vf` over the median `vf` across the track's other measurable placed segments — the number the shipped threshold compares |
| `nccf` / `hnr_dB` | the threshold-free view of the same evidence; `hnr_dB` is a monotone restatement of `nccf` |
| `f0_Hz` | median F0 over the voiced frames. **Octave-ambiguous** — never read as a transcription |
| `jitter%` | median frame-to-frame change in the pitch *period*. Reported, never thresholded |
| `f0_span` | 10th-to-90th-percentile F0 spread in semitones. Reported, never thresholded |
| `frames` | measured / total. `0/N` means every frame was below the level floor, which is **unmeasured**, not unvoiced (#93) |

The level floor is the track's own median placed-segment RMS minus
`VOICING_FRAME_LEVEL_DROP_DB` (20 dB), so nothing here has an absolute
threshold in it except that drop.

## The only corpus so far: synthetic

Measured by `tests/test_voicing.py` on signals built in
`tests/harness/factories.py`, 2 s each at 16 kHz, level floor −45 dBFS. These
are *evidence about the statistics*, not about any song.

| source | `vf` | `jitter%` | `f0_span` | notes |
|---|---|---|---|---|
| harmonic stack, 180 Hz, ±3 % vibrato | 1.000 | ~1.6 | ~1.0 | what a sung vowel looks like |
| harmonic stack, 196 Hz, rigid F0, exponential decay | 1.000 | ~0.008 | ~0.003 | **a plucked string reads as voiced.** This is the documented limit, not a bug |
| broadband noise at the same level | 0.000 | – | – | drums, hiss, room tone |
| dither-level noise (−69 dBFS) | 0/98 frames | – | – | gated out; reported as unmeasured |
| harmonic stack at 400 Hz | 1.000 | ~1.8 | ~12.6 | octave flipping inflates `f0_span` — an artefact, not a melody |

Two things follow and both are already in the code:

1. `vf_ratio` catches a phantom over **anything aperiodic** and misses a
   phantom over **anything pitched**. F43's `'mushrooms grow.'` (a guitar
   note) is the second kind.
2. `jitter%` separates the two synthetic sources by **two orders of
   magnitude**. That is a hypothesis with no real-song evidence either way,
   which is exactly the state `docs/idiom-corpus.md` records for #75's
   keywords, and it does not become a lint until the table below is filled in.

## To be filled in: "Deathless"

One command, on the machine that holds the master, no GPU:

```bash
python -m music_video_maker.calibrate_voicing \
    --config ~/mvm-runs/deathless/run_v13.toml \
    --window 228.590-230.150 \
    --window 378.470-379.430 \
    --window 498.730-501.630 \
    --csv ~/mvm-runs/deathless/measurements/voicing_v13.csv
```

The three windows are FINDINGS F43's adjudicated phantoms, by ear on
exact-window clips:

| window | aligner's label | what is actually there |
|---|---|---|
| 228.590–230.150 | `'mushrooms grow.'` | a guitar note, then the word "So" at the very end |
| 378.470–379.430 | `'you.'` | instrumental |
| 498.730–501.630 | `'deathless, Forevermore!'` | silence — the known #71 phantom, already caught at 0.02× consonant-band baseline |

They are placed segments, so they appear in the table by index as well; the
`--window` rows are there so the exact adjudicated spans are scored even if
`alignment_model_size` moves a boundary.

Three questions the table answers, in order:

1. **Does `vf_ratio` separate the three from the other ~54?** If yes, record
   the lowest real segment's ratio and the highest phantom's, and set
   `VOICING_RATIO_THRESHOLD` between them the way
   `VOCAL_ENERGY_RATIO_THRESHOLD`'s docstring does — nearer the real floor,
   because a missed phantom costs GPU hours and a false positive costs a log
   line.
2. **If it separates only two of three** (the expected outcome: the guitar
   note should pass), that is the predicted result and it is worth recording
   as one rather than as a failure. Then go to 3.
3. **Do `jitter%` and `f0_span` separate `'mushrooms grow.'` from the ~54
   real segments?** Score *every* segment, not just the three — a statistic
   that separates n=1 from a hand-picked control is #76's mistake again. Record
   the excluded candidates with their reason, as #60 requires.

Anything the table shows that contradicts the synthetic numbers above is the
finding, and the synthetic numbers lose.

## Can the aligner be told, instead of only checked?

Read from the stable-ts **2.19.1** sdist (`stable_whisper/alignment.py`,
`options.py`, `stabilization/__init__.py`), not from memory. Confirm the
version actually installed on the render host before acting on any of it:

```bash
python -c "import stable_whisper, inspect; \
  print(stable_whisper._version.__version__); \
  print(inspect.signature(stable_whisper.alignment.align))"
```

**The correction first.** `alignment.py` claimed `suppress_silence=True`
"enables stable-ts's integrated Silero VAD". It does not. `vad` defaults to
`False` (`options.py:135`), and `stabilization/__init__.py` branches on it:
`predict_with_vad` runs Silero, `predict_with_nonvad` runs `wav2mask`, a
volume-quantization mask (`q_levels=20`, `k_size=5`). Every alignment this
project has ever run used a **loudness** threshold, not a speech detector --
which is precisely why a lyric can land on a guitar note: the note is not
quiet, so no silence mask can move it.

| option | default | what it would do about #96 |
|---|---|---|
| `vad=True`, `vad_threshold=0.35` | `False` | Silero VAD is a *speech* detector. The one lever aimed at the actual failure rather than at loudness. Needs PyTorch 1.12+, which the `[align]` extra already brings. |
| `only_voice_freq=True` | `False` | Restricts alignment to 200-5000 Hz. Removes bass and most cymbal energy before the model sees it; a plucked guitar's fundamental is inside that band, so expect partial help at best. |
| `denoiser="demucs"` | `None` | Source-separates before aligning. The heavyweight version of the right idea -- and this project already produces a vocal stem for conditioning (`vocal_stem`, `docs/vocal-stem-workflow.md`), so the cheap form is to align against **that file** rather than the master, with no new dependency. |
| `failure_threshold=<float>` | `None` | Aborts when the share of zero-duration words exceeds it. A refusal, not a detector; `alignment_quality`'s existing `track_confidence_collapse` already covers the same ground with more detail. |
| `nonspeech_skip=5.0` | `5.0` (already in force) | Skips non-speech runs >= 5 s. Only as good as the mask feeding it, i.e. row 1. |

**Why none of them is switched on here.** Each one re-cuts the timeline for
every song, which re-anchors every authored shot plan (`ShotPlanDriftError`
territory) and invalidates every chunk fingerprint. That is a decision taken on
evidence, not a correction -- and the evidence is one `--prepare` apart:

```bash
# arm A: today's behaviour
music-video-maker --config run_v13.toml --prepare
# arm B: same config, aligner told about speech -- one line changed in
# alignment.align()'s call, or aligned against the vocal stem instead
```

Compare, before any GPU time: segment count, `voiced=.../track=...`, the #71
`no_vocal_energy_in_placed_segment` findings, the #96
`no_voiced_periodicity_in_placed_segment` findings, and whether the three F43
windows above still carry a lyric at all. The last of those is the only
question that matters: an option that stops placing `'mushrooms grow.'` on a
guitar note has fixed the defect, where the check above can only report it.
