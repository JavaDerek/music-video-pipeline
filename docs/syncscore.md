# syncscore — lip-sync scoring of rendered chunks (#109)

```bash
mvm-syncscore output/chunks --syncnet ~/syncnet/syncnet_python \
    --syncnet-python ~/syncnet/.venv/bin/python --out sync.csv
# re-roll whatever is flagged, up to two new seeds each:
mvm-syncscore output/chunks/opening --syncnet ... --auto-reseed 2 \
    --config run.toml --timeline opening
```

## What it measures

H3 writes its own soundtrack into every chunk mp4. Scoring that track
measures the model against itself, so `syncscore` never reads it: each
`chunk_NNNN.mp4` is muxed with the chunk's **staged conditioning audio**,
`chunk_NNN.wav` (the file Stage 3 uploaded for that chunk), and SyncNet
scores the pair. The two names differ in zero-padding; they are matched on the
numeric id.

SyncNet ([syncnet_python](https://github.com/joonson/syncnet_python), MIT)
reports per face track an **AV offset** (in 25 fps frames; positive = video
lags audio), a **min distance**, and a **confidence** (median minus min
distance; higher is more certain). The track with the highest confidence is
the chunk's score.

| verdict | meaning |
|---|---|
| `ok` | `abs(offset) < 3` and `confidence >= 3.0` |
| `flagged` | either test fails |
| `no_face` | SyncNet found no face track (it needs one face ≥100 px tracked ≥100 frames ≈ 4 s) |
| `no_stem` | no staged `chunk_NNN.wav` survives for this chunk |

The exit code is 1 if any chunk is flagged, so a script can gate on it.

## Install (not a pip package)

```bash
git clone https://github.com/joonson/syncnet_python && cd syncnet_python
sh download_model.sh   # data/syncnet_v2.model, detectors/s3fd/weights/sfd_face.pth
python -m venv .venv && .venv/bin/pip install torch torchvision opencv-python \
    scenedetect python_speech_features scipy
```

`syncscore` itself needs only the standard library and `ffmpeg`; SyncNet runs
in its own venv via `--syncnet-python`. The report header records the sha256
of `syncnet_v2.model`.

## Re-rolls keep the run state honest

`--auto-reseed N` re-rolls flagged chunks with `--reseed` under generations
1..N, re-scoring after each, until every chunk passes. A chunk that never
passes is **re-rendered under its best generation**: the smallest offset, with confidence breaking ties. A confident 10-frame offset is confidently wrong, so confidence alone would pick the worst take. It is never copied back
from an archived take, because a copied file would contradict the seed
`run_state.json` records for it. That is the #93 defect, where the pixels
on disk did not match what the paperwork said. Seeds are deterministic, so
the re-render reproduces the take.

## Calibration

### Close-up synthesized speech (one performer, frontal, 1344×768)

19 chunks, each labelled in sync / out of sync by eye on the finished video,
scored against their own voice stems:

| label | n | flagged by the rule |
|---|---|---|
| bad | 6 | 6 |
| good | 13 | 0 |

One further chunk the viewer passed was flagged. On inspection it silently
re-mouths its line in trailing silence (#103's symptom), so the flag is
arguably right.

### Sung, multi-performer music video ("Deathless" v16) — the rule does not transfer

All 75 rendered chunks, each scored against its staged `chunk_NNN.wav`. That
file is a slice of the **master mix**, with the band under the voice, because
no isolated vocal stem survives for this render.

| verdict | chunks |
|---|---|
| `no_face` | 53 (wide, profile or multi-figure shots; no face ≥100 px held ≥4 s) |
| `flagged` | 20 |
| `ok` | 2 (35, 40) |

The viewer-reported desyncs: **33** flagged (offset −10, conf 0.36), **34**
flagged (−2, 1.23), **63** `no_face`. But 18 unreported chunks are flagged
too, almost all on confidence alone: 19 of the 22 scorable chunks sit below
2.0, against 3.5–7.7 for in-sync close-up speech. The labelled-bad chunks do
not stand out from the rest on confidence. Only 33's large offset is
distinctive, and offsets of −15 and +10/+11 appear on unreported chunks as
well.

**Conclusion:** on sung chunks scored against a master-mix slice, the 3.0
confidence floor is not a lip-sync detector. It mostly measures that SyncNet
cannot hear a voice through a band. Use `syncscore` to gate **speech
rendered from a clean voice stem**. On music videos, treat its numbers as a
pointer and not a verdict until it has been re-calibrated against an isolated
vocal stem (#25), which is the obvious next measurement.

General form: a threshold calibrated on one input distribution — here a clean
speech stem — is a claim about that distribution only. The 6/6 + 13/13 result
is real, and saying nothing about the mix is also real. The report header
names the audio it read so the two cannot be confused.
