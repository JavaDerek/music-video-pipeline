# Lip-sync backends, quality tiers and shot provenance

`music_video_maker/lipsync.py` decides *which model makes the talking
pixels*; `music_video_maker/provenance.py` records it per shot and lets a
final-grade step refuse what may not be exhibited. Both landed 2026-10-13.

## Why

The MiniMax H3 **open weights** are released under the MiniMax H3 Community
License (release date 2026-08-02). Its definitions name the United States,
the European Union, the United Kingdom and the Republic of Korea as
*Excluded Territories*, and section V.4 says you may not "use, reproduce,
modify, distribute, or display the MiniMax H3 Works or any of their Outputs
or results outside the Applicable Territory". Running the weights on a card
in an excluded territory is "use", whatever the footage is for, so every
shot that path makes is unusable there, drafts included. MiniMax's licence
Q&A (`huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md`)
says the hosted API is "globally available", and organisations in a
restricted region may apply for a formal licence at
`https://platform.minimax.io/h3-license`.

The **hosted API** (`https://api.minimax.io`) is governed instead by the
MiniMax Open Platform Terms of Service (effective 2026-03-30, provider
Nanonoble Pte. Ltd.), whose *Intellectual Property (Non-Transfer) and
Indemnity Obligations* clause 3 reads:

> As between you and us, and to the extent permitted by applicable laws, you
> retain your ownership rights in Client input and generated content. We may
> use the input and generated content to provide, maintain, develop, and
> improve our Services, comply with applicable law, enforce our terms and
> policies, and keep our Services safe.

`lipsync.MINIMAX_TERMS` carries that quotation, the URL and the effective
date into every provenance file an API run writes.

## The three backends (`lipsync_backend`)

| backend | what | status |
|---|---|---|
| `minimax_api` | reference-to-video on the hosted API: a reference image, a reference audio clip and the prompt in; a talking clip out | **default**; the only backend a `final` step accepts |
| `local_h3` | the open weights through ComfyUI, i.e. every path this project measured before 2026-10-13 | **locked**: refuses at config load unless `h3_authorization_file` names a file that exists (MiniMax's written authorization for the territory) |
| `local_draft` | a locally run open model for free iteration on a workstation card -- never MiniMax H3 | **not integrated**: `lipsync.LOCAL_DRAFT_MODELS` is empty, so the backend refuses until a model is chosen and its licence recorded beside it |

Candidates for `local_draft`, by licence first (none is integrated; the
licence lines are quoted from the primary sources as read 2026-10-13):

- **EchoMimicV3** (antgroup, 1.3 B, audio-driven portrait animation on
  Wan2.1-Fun-1.3B-InP; "12G VRAM is All YOU NEED", tested on an RTX 4090D).
  README: "The models in this repository are licensed under the Apache 2.0
  License." Weights `BadToBest/EchoMimicV3`: `License: apache-2.0`. Base
  model `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`: `apache-2.0`; audio encoder
  `facebook/wav2vec2-base-960h`: `apache-2.0`.
- **InfiniteTalk** (MeiGen-AI, image-audio-to-video on Wan2.1-I2V-14B-480P,
  "unlimited-length"; a low-VRAM mode and an fp8 variant exist, a single 24 GB
  card is not documented). README: "The models in this repository are
  licensed under the Apache 2.0 License." Weights `MeiGen-AI/InfiniteTalk`:
  `License: apache-2.0`. Audio encoder `TencentGameMate/chinese-wav2vec2-base`:
  `License: mit`.

Apache-2.0 §2 grants "a perpetual, worldwide, non-exclusive, no-charge,
royalty-free, irrevocable copyright license to reproduce, prepare Derivative
Works of, publicly display, publicly perform, sublicense, and distribute the
Work" -- no territorial, commercial or field-of-use restriction.

## Quality tier (`quality_tier`)

Orthogonal to the backend. `draft` is low resolution for iteration;
`final` is 2K through `minimax_api` only, and a `final` tier on a local
backend is refused at load.

| tier | hosted model | resolution | price (read 2026-10-13) |
|---|---|---|---|
| `draft` | `MiniMax-H3-Max` | 480P | $0.05 / s of output |
| `final` | `MiniMax-H3` | 2K | $0.13 / s of output |

Audio input is free. The API bills whole seconds, so a chunk's billable
length is its frame count rounded *up* to a second and clamped into the
model's range (H3 4-15 s, H3-Max 5-15 s). `estimate_cost` sums that over
the chunks about to be rendered and logs it before the first request; over
`CONFIRM_THRESHOLD_USD` ($10) the run refuses without `--yes`.

## What a hosted shot is

For each chunk the client pads the conditioning audio with silence to the
billed duration (the API's output length is the integer `duration`, so the
clip must be exactly that long), uploads the reference image and the padded
audio (`POST /v1/files/upload`, kept 7 days), creates the task
(`POST /v2/video_generation`, `reference_image` + `reference_audio`, ratio
chosen nearest the render size), polls `GET /v2/query/video_generation/{id}`
until `succeeded`, downloads the result, and **conforms** it: `fps=24`,
exactly the chunk's frame count (short outputs clone the last frame, long
ones are cut), with the conditioning audio muxed in place of the service's
own track -- the sync reference is what the shot was asked to follow. The
raw API file is kept beside the conformed one as `<stem>.api.mp4`.

The prompt is the pipeline's own composed prompt plus
`AUDIO_BINDING_SENTENCE`, which binds "reference audio 1" to the mouth.

## Provenance (`provenance.json`)

Every shot writes a record, after each shot (a run that dies on shot 7 still
proves 1-6): `shot_id`, `backend`, `model`, `tier`, `resolution`, `frames`,
`fps`, `duration_seconds`, `task_id`, request and completion timestamps,
SHA-256 of every input (reference image, reference audio, padded audio) and
of the conformed output, the cost, and notes (raw API file and its hash,
ratio, prompt, usage). `terms` carries the MiniMax terms above.

A post-render packager can consume it: `provenance.assert_final_grade` refuses, before
anything is rendered, any shot from `local_h3`/`local_draft` or at draft tier, and
`provenance.merge_provenance` joins several records into one.

## Secrets

`MINIMAX_API_KEY` is read from the process environment only. It is sent in
the `Authorization` header, redacted from any error body the client raises,
and never written to disk, a log or a provenance record. `.env` stays
gitignored; the package never loads it.
