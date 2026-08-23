# A spoken prologue: segments around the song (issue #66)

Thriller opens with four minutes of film before the song starts and closes with
a laugh after it ends. This pipeline structurally cannot make that: everything
is indexed on the master track. Alignment produces song time, slicing cuts song
time, `instrumental_coverage` guarantees the chunk timeline covers *the master
track* end to end, and Stage 5 concatenates chunks and muxes that one file over
them. A prologue is video with no position on that timeline and audio that is
not in that file.

Design only. Nothing here is built. What follows is the shape, the four places
it will actually break, and the decisions that must be made before code.

## The reframing

**A prologue is not a special chunk. It is a second timeline, rendered by the
same five stages, concatenated ahead of the first.**

Spoken dialogue is audio, and H3 conditions on audio — it does not know or care
that the audio is speech rather than singing. A script plus a recorded dialogue
track is exactly the (text, audio) pair Stage 1 already consumes, and
`stable-ts` `align()` is *better* at speech than at singing. So the prologue
reuses alignment, slicing, staging, execution and the cast verbatim. What is
new is joining two timelines.

**Make it a sequence, not a `prologue` field.** An epilogue is the same
machinery, and a bespoke `prologue` key guarantees a second implementation for
the laugh at the end. Proposed shape:

```toml
# the song itself stays exactly where it is: master_audio / lyrics_file
[[segment]]                       # rendered before the song
audio  = "audio/prologue.wav"
script = "prologue.txt"
shot_plan = "prologue_shot_plan.toml"

[[segment]]
position = "after"                # default "before" for the first, but say it
audio  = "audio/epilogue.wav"
script = "epilogue.txt"
```

Note the TOML hazard this repo has paid for twice: `[[segment]]` tables at the
end of a config capture any bare key written below them (`config.HARDWARE_KEYS`
and `ALIGNMENT_OVERRIDE_KEYS` exist for exactly this, and #62's A/B loaded both
arms identically because of it). A closed key set per segment table, with the
"this is a top-level setting, move it above the first table" hint, is not
optional.

## Where it will actually break

### 1. `-c:v copy` concat is unforgiving

The concat demuxer does not fail on mismatched inputs; it produces a file that
plays wrong. Every prologue chunk must match the song's chunks on every
parameter the copy preserves. Measured on the real "Deathless" render — all 80
chunks, one distinct signature:

```
h264, profile High, level 30, 864x480, yuv420p, progressive, 24/1 fps, time_base 1/12288
```

So the validation list, and it must be a *check* rather than an assumption:
codec, profile, level, width, height, pixel format, field order, frame rate,
and time base. Width and height are the two the config can predict
(`render_width`/`render_height`); the rest come out of the workflow template
and must be probed from the first chunk of each timeline and compared before
concat runs, not assumed from config.

One more thing the measurement shows: each chunk mp4 carries an AAC stream
(H3's own generated audio). The concat pass strips it with `-an` and that
must stay true for both timelines — "generated audio is always discarded" is
untouched by this feature.

### 2. Fingerprints must record which timeline a chunk belongs to

Prologue chunk 3 and song chunk 3 can trivially share a span, a frame count and
a resolution. Without a timeline discriminator in `ChunkFingerprint`,
`--resume` can hand a prologue shot to the song and report a clean match —
which is the exact failure #34 exists to prevent, reintroduced along a new
axis.

Proposal (this is `contracts.py`, owned elsewhere): a `timeline: str | None`
field, **timeline tier** — `None` means the song, so every existing state file
and every existing run is unchanged and compares equal to itself. Timeline tier
and not content tier, because a chunk from the wrong timeline is in the wrong
*place*: it is never escapable via `resume_ignore_prompt_changes`.

The chunk id space is the other half of the same question. Two options:

* **Separate id spaces per timeline** — prologue chunk 0..n, song chunk 0..m —
  which makes the discriminator load-bearing and makes chunk filenames collide
  unless the output directory is per timeline.
* **One id space, prologue first** — the song's chunk 0 becomes chunk 12. This
  is simpler for concat and fatal for everything else: it renumbers every
  authored shot plan the moment a prologue is added or its length changes, and
  `chunk_id` is the anchor a plan is authored against.

**Take separate id spaces and a per-timeline chunks directory.** The renumbering
failure is silent and destroys committed authoring work.

### 3. The audio is a concat too, and its total must equal the video's

Today the mux is one file over one video, and a duration mismatch is visible
immediately. With a seam, a few frames of drift at the join desyncs
*everything after it* while each half looks fine in isolation.

Assert `len(prologue_audio) + len(master) == len(video)` **before** muxing, not
after. And note the measurement that says this will not hold by default: on
"Deathless" the chunk timeline overshoots the master track by 1.837 s because
the last tile is padded up to H3's 124-frame minimum, and the mux's `-shortest`
silently threw away 47 rendered frames. That is harmless for one timeline. At a
seam it is not: an over-long prologue pushes the entire song out of sync by
whatever the padding was.

So the prologue's audio and its chunk timeline must be reconciled *at the
seam*, not at the end. Two honest options, both needing the decision written
down:

* **Pad the prologue audio** to the chunk timeline's length (append silence),
  so video and audio agree and the seam is exact. Costs a beat of silence
  before the song, which may even be wanted.
* **Trim the last prologue chunk's video** to the audio, which is a re-encode
  or a stream-copy cut on a keyframe, and `-c:v copy` concat is the invariant
  this project protects hardest.

Padding is almost certainly right. Say so in the config, with the padded amount
logged.

### 4. The seam is a chaining decision

If `i2v_chain_scope` is on, does the song's first chunk seed from the
prologue's last frame? Sometimes that is the effect (a transformation carrying
into the first verse); sometimes it is a hard cut on purpose. It needs to be a
setting — `chain_across_seam = false` as the default, because a hard cut is the
conventional edit and the surprising behaviour should be the one you ask for.

#47's face precondition applies to that seed like any other, and note the
compounding risk: a prologue's final shot is exactly the kind of dramatic
composition that ends on a back, a silhouette or an object. On the chained path
the seed frame **is** the identity conditioning, and a frame showing the back of
someone's head carries nothing.

## Authoring

`mvm-author` writes shots against lyric chunks. A prologue has a *script* — who
speaks, what they say — and the model would be writing dialogue, not
interpreting a line that already exists. Different stage, different output, and
it wants a human in the loop harder than the others, because the dialogue then
has to be **recorded** before anything can be aligned.

**v1 should not author the script.** The operator supplies script + recording;
the pipeline does the shots. Reasons: the recording is a physical dependency no
amount of generation removes; and a dialogue stage's output is not checkable
the way the beats stage's is — the whole value of that stage is that
`beat_role`/`beat_group`/`act` make it mechanically checkable, and "is this
dialogue any good" has no such handle.

Non-singing speakers are a cast question. A character who appears only in the
prologue never sings, so `default_lead_vocalist` and the singing clause must
stay out of its prompts. `present` (#59) is most of that answer, and #82's
`subject` — legal only on an instrumental chunk, refused on a voiced one — is
the rest of it: on a prologue chunk with dialogue, whoever is *speaking* owns
the frame, exactly as the singer does on a sung chunk. That rule (#58/#59/#60:
the sentence outranks any field that argues with it) transfers unchanged.

## Open questions, and the answers this design assumes

* **One run directory with two timelines, or two runs and a join step?**
  **One run.** The fingerprint and resume argument settles it: two runs cannot
  detect a prologue chunk being reused for the song, and the join step would
  need its own state.
* **Does `--prepare` emit a skeleton for the prologue?** **Yes.** The whole
  point of #52 is that anchors are never transcribed by hand, and a prologue's
  anchors come from its own alignment exactly like the song's.
* **What if there is no prologue audio yet?** A silent placeholder of a stated
  duration lets the shots be authored before the recording session. Silence
  aligns to nothing, so those chunks are pure filler — which is fine and worth
  saying out loud, because the shot plan authored against them will re-anchor
  when the real recording lands, and `ShotPlanDriftError` is what will catch
  anyone who forgets.
* **Does the prologue get its own `cinematography`?** It should be able to —
  the classic version of this device looks *different* from the song. That is
  a per-segment override of a whole-video field, which is the first case in
  this project where "whole-video" is not the whole video. Worth deciding
  deliberately rather than discovering.

## Cost

A prologue is chunks like any other: at 864×480, ~3.7 min/chunk on the 4090. A
four-minute prologue at ~6 s/chunk is ~40 chunks, ~2.5 h — comparable to a
whole song. It is not a small feature to *run*, whatever it costs to build.
