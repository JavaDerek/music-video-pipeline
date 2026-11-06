# Spoken-script guide

How to author a dialogue-led video (two or more people talking, no sung master)
so the render does what the script says. The companion of
[`shot-writing-guide.md`](shot-writing-guide.md), which covers music-video shot
lines; this one covers the *spoken* case, where the voice is a separate track and
the cast talks to each other.

Everything here is enforced, not advised:
[`music_video_maker/spokenscript.py`](../music_video_maker/spokenscript.py)
refuses a script that breaks a rule, before any GPU time is spent.

```bash
python -m music_video_maker.spokenscript my_script.toml   # 0 clean, 1 errors, 2 unreadable
python -m music_video_maker.spokenscript --for-render my_script.toml   # also: every still exists
```

**General form: a render takes hours and reports nothing about what it got wrong.**
A first end-to-end render of a hand-written 19-clip dialogue piece (several
characters per clip, about eight GPU-hours) failed in six ways, and every one was
readable from the script text alone. Each rule below is one of those failures turned
into a field and a refusal.

## The shape of a script

A cast, and an ordered list of shots. Every shot has **one speaker at most**, and the
speaker's words are a `line` that a separate voice stage speaks; the video model is
given a description and the finished audio, never the words as prose.

```toml
[script]
era = "pre-industrial"
style = "Warm natural light, handheld, a market square at dusk."

[cast.mara]
name = "Mara"
garment = "undyed linen tunic with a rope belt"
appearance = "weathered woman in her forties, grey-streaked braid"
voice = "low, dry, unhurried"

[setup.stall]                    # one camera angle: every shot on it starts from its still
characters = ["mara"]            # who the still shows
posture = { mara = "standing" }  # how the still shows them
still = "stills/stall.png"       # relative to this file; may be absent while authoring

[[shot]]
id = "s1"
seconds = 6
characters = ["mara"]            # everyone in frame, speaking or not
speaker = "mara"                 # at most one, and must be in `characters`
line = "We open at dawn."        # the voice track's words, not the video prompt's
text = "Close on Mara at her stall, hands on the counter."
posture = { mara = "standing" }
setup = "stall"
[shot.delivery]
pace = "measured"
pitch = "low"
volume = "normal"
pause_before = 0.4
```

A silent reaction shot has no `speaker`/`line`/`delivery`. A scene with several people
talking is several shots, cut together; the cut is the edit, not something the model
is asked to perform.

## One speaker per generated shot

When two or three faces share a clip and two voices are supplied, the model decides
which face the voice belongs to, and it does not decide reliably: in the failed render
one character's line came out of another's face, and in another clip a character
walked in who had no business there. Binding is a property of the *shot*, so it is
solved there. A shot's `speaker` must be in `characters` (`speaker-in-shot`), and a
`speaker` without a `line`, or the reverse, is refused (`speaker-line-pair`).

Cut to whoever is talking. A "wide shot with everyone" is a silent establishing shot
or an ambient one, not a place for dialogue.

## Delivery is data

The failed render's prose directions ("flat", "annoyed", "hopeful") produced the same
intonation on most lines, and for a comedy that is fatal: a joke is a contrast between
two ways of saying something. Nothing in a speech stage reads an adjective. So a spoken
line needs a `delivery` table from a closed vocabulary:

| field | values |
|---|---|
| `pace` | `slow`, `measured`, `normal`, `brisk`, `fast` |
| `pitch` | `low`, `mid`, `high` |
| `volume` | `whisper`, `soft`, `normal`, `loud` |
| `pause_before` | seconds of silence before the line (default 0) |

`delivery-missing` is an error. Two further checks report, but do not refuse, because
whether a reading is funny is a human call: `delivery-flat` fires when adjacent lines by
different speakers share one delivery, and when more than 60% of a script (five lines or
more) shares one. A pause is the cheapest comic tool there is; a deadpan answer to a
loud question is `soft`/`low` after a `pause_before` of a second.

`line-too-long` is arithmetic: words divided by the pace's words-per-second, against the
shot's seconds minus the pause. The words-per-second table is a starting point; replace
it with your synthesis engine's measured rate.

## Say the clothes, once, per character

Written as "tunic" in some prompts and left to chance in the rest, a period cast came out
in a modern button-down. This is the repo's rule that a property which must hold across
the whole video needs its own field. With an `era` set, every cast member needs a
`garment` (`garment-missing`), and a list of terms that cannot exist in that era is
refused wherever it appears, in the cast and in every shot's prose (`wardrobe-term`).
`pre-industrial` ships with a short, unambiguous list (button, collar, shirt, jeans,
zipper...); add your own with `banned_terms`. An unknown `era` warns (`era-unknown`)
rather than silently checking nothing.

State the garment's **opening**, not only its name (`garment-opening-unstated` warns when a
garment never says "round neck", "wrapped" and so on). A second render's portraits said
"tunic" and still came out with real buttons on a placket, because an image model fills an
unstated opening with the commonest one. `compose_portrait_prompt` is the one way to build a
cast image's prompt: it states the garment, the script's style and the era's lock (for
`pre-industrial`: a pullover tunic with a plain round neckline and no front opening, naming
slit, placket, buttons, toggles and collar as absent), and a cast member can replace the lock
with their own `wardrobe_lock`. Use it, so a portrait cannot omit what the linter checks.

The check is on text. It cannot see a rendered portrait: **zoom into the neckline of every
generated cast image at full size before spending video time on it**. At contact-sheet size a
buttoned placket and a plain slit look identical; at full size, two buttons were plain.

## State every posture, and mark every change

Left unsaid, posture is re-decided for each clip: people stood, sat, stood and sat
between cuts. Every character in frame needs a `posture`
(`standing`, `seated`, `kneeling`, `walking`, `lying`, `crouching`; `posture-missing`), and
a change from the last shot that character appeared in must be explained by the later
shot's `transition` (`posture-change-unmarked`):

```toml
transition = "Mara sits down on a stool"
```

## Cut between setups, not places

**General form: a cut back to an angle the audience has seen must open on what they saw,
and a prompt cannot make that true.** The second render cut back and forth between a few
angles (one speaker, the other, the first again), and every shot, rendered alone from a
portrait and a prompt, invented its own hillside, its own neighbours and its own posture. A
viewer read every cut as the camera jumping somewhere new. The script asked for nothing
else: it had one location for the whole piece, so no check on locations would have fired.

So the unit of continuity is the **setup**, one camera angle, and each setup has one
**still**, the frame every shot on it starts from:

* every shot names its `setup` (`setup-missing`, `unknown-setup`);
* the setup lists who is on its still, and the shot's speaker must be one of them
  (`setup-speaker-unframed`): the still is the shot's first frame, so it must carry the
  speaker's face, the same lesson as the chained path's seed frame;
* nobody may be in a shot who is not on its still (`setup-character-not-on-still`), and
  nobody may be in a posture the still contradicts (`setup-posture-mismatch`). The still
  decides, so a shot that needs someone seated who stands on the still needs another setup;
* a setup with no still warns while authoring and is an error with `--for-render`, which
  also requires the file to exist (`setup-still-missing`). A declared setup no shot uses
  warns (`setup-unused`).

Measured on three cut-backs to two setups, 30 renders, everything else identical: started
from the still, a shot's first frame scored 3.7-5.0 from it (ffmpeg thumbnail, mean absolute
difference, 0-255); rendered without it, 27.6-44.3, the closest being the still's own source
shot at another seed (14.7). The two setups' stills are 33.0 apart, so an unanchored cut-back
looked as unlike its own setup as a different setup does. Without the still the cut-backs
also dropped the people the script put beside the speaker. Check a render the same way:

```bash
python -m music_video_maker.setupmatch stills/stall.png takes/s1.mp4 takes/s7.mp4   # 1 on a mismatch
```

Cutting a lot is not the defect, and there is no limit on it: a quick back-and-forth can be
the joke. Name a few setups per scene, keep each person's eyeline the same way on every
return, and hold a speaker longer where the timing allows.

## Give every speaker a voice of their own

A voice taken from the video model's own audio gave every character the same voice. Put each
speaking character's synthesis voice id in `tts_voice`: two characters sharing one is an error
(`voice-shared`), and a speaker with none while others have one warns (`voice-missing`).
`voice` stays the prose description of what you are aiming for. Audition the voices on the
script's own hardest lines before authoring against them: a voice that cannot do a given
character is found in a minute with a synthesis engine and a day later in a render.

## Cast who is in frame, and no one else

A character named in a shot's prose but missing from its `characters` is a person the
model invents or imports from the previous shot (`unlisted-character-in-text`). List
them, with a posture, or stop naming them. More than four people in one shot warns
(`too-many-visible`): faces and postures blur in a crowd, so a crowd is a silent wide
shot.

## The prompt reads itself aloud

A video model that makes speech will speak what it is given. "Scene, 12 seconds:" and
"use it exactly for his lines" were both read out as dialogue. `compose_generation_prompt`
builds a prompt from description only (style, who is in frame with posture and garment,
then the shot's `text`) and never includes the `line`; `lint_generation_prompt` refuses
the two measured patterns. Also refused: quoted speech inside `text`
(`dialogue-in-prose`), which invites the model to say it. Put no label, duration or
instruction ahead of the description, and no dialogue in it.

## Rendering a single-speaker shot

`music_video_maker/spokenrender.py` renders a script that passes `--for-render`, one shot at a
time on the local graph, and cuts it together:

```bash
python -m music_video_maker.spokenrender my_script.toml --lines lines/ --out out/ \
    --h3-authorization license/minimax_h3_authorization.pdf [--only s7,s8] [--seed N]
```

`lines/` holds one finished WAV per spoken shot, named `<shot id>.wav`, made by whatever
synthesis engine you chose (not this module's job yet). Each shot gets the speaker's
`portrait` first among its reference images, the line as reference audio, and its setup's
still and the same line pinned at frame 0. Every take is checked with `setupmatch` and recorded
in `out/provenance.json` as it lands; `out/spoken_cut.mp4` is the cut, with the finished lines
as its only audio. It exits 1 if any shot failed to open on its still. It is `local_h3` only,
so the authorization lock applies, and it takes custody the way a render does: a free-VRAM
pre-flight first, `POST /free` however it ends. Not built yet: line synthesis, take scoring
and retakes, an ambience bed and loudness.

These are the lessons it encodes:

- **Start from the setup's still and pin the finished line to it.** On the local graph
  (`MiniMaxH3ReferenceToVideo` as before, then `MiniMaxH3AddGuide` with the still as `image`
  and the padded line as `audio`, both at `frame_idx = 0`, feeding the guider), measured over
  three shots and two seeds: the line given only as a *reference* audio passed SyncNet 0 of
  15 times (offsets up to 15 frames), the still alone 0 of 6, the still with the line pinned
  3 of 6 plus one at offset 0 and confidence 2.7. A viewer agreed with the meter on which
  speaker was in sync. The hosted path has not been tried this way.
- **Set the speaker's face size on the still, by cropping it.** One speaker stayed out of sync
  in every arm: his still was a wide frame with others in it (face 89 px of 480 tall). A
  re-generated still with the repo's `close` framing sentence came back waist-up (115 px) and
  did not fix it; the same still *cropped and enlarged* to head-and-shoulders (210 px) brought
  the timing to offset 0 or -1 on three of four takes, and a viewer called one of them
  perfect. The prompt did not set the size; the pixels did, as #97 found for sung chunks. The
  speaker who synced from the start had a 150 px face.
- **SyncNet's confidence floor fails takes a person passes.** Both shots the viewer called
  perfect scored offset 0 and -1 but confidence 2.35 and 1.59, under `syncscore`'s 3.0. On
  short single-speaker lines read the offset, and look before trusting the verdict.
- **Drive the mouth with the finished line.** Reference image of the speaker plus the
  synthesized line as the reference audio, one face and one voice per generated shot. The
  hosted and local reference-to-video paths both return the reference audio unchanged, so the
  voice track you assembled is the voice track you ship; discard everything else the model
  generated.
- **A short line still renders at the model's frame floor.** Quantize the work, not the cut
  (`boundary_overrun`'s principle): render the grid-valid length, then trim the shot at the edit
  to the last word plus a short tail. Padding the audio with silence to reach the floor leaves
  the mouth moving over nothing (#103, #107).
- **A pause is the edit's, not the audio's.** The model starts the mouth at frame 0, so leading
  silence in the audio is a mouth moving before the voice. Hold the shot's first frame for the
  pause instead.
- **Score a speaking take by whether the speaker's face is there** (a plain face detector over
  twelve frames) and retake the worst until the deadline; a speaker turned away or missing is
  the common failure, and it needs no model to find. Sync scoring is the next instrument.
- **Anything a person is "supposed to look at" needs a first look at the first shot.** Check
  one rendered speaking shot (identity, clothing, mouth) before releasing 80.

## What is not checked (open)

- **Whether a rendered cast image is period-correct.** Needs a vision check on the
  portraits; today it is a person looking.
- **Which engine speaks the lines, and whether the voices differ enough.** A voice
  cloned from the video model's own audio gave every character the same voice; use a
  separate synthesis stage with a distinct voice per character and audition it before
  authoring against it.
- **Whether the delivery is funny.** Only a listener can say.
- **Lip-sync of the finished video.** Check by eye and ear; the repo's lip-sync lessons
  (join placement, leading offset) apply unchanged to a spoken master.
- **The camera drifts inside a shot.** Started from the still, about two thirds of the
  shots pushed in, panned or tracked during the shot: same place, new framing by the time
  the edit cuts away. `setupmatch` checks the first frame only, because at the cut-away
  frame no threshold separates reframed from relocated. A style line that says "handheld"
  invites it; a locked-off wording is untried.
- **The still's face size and wardrobe.** Nothing measures the speaker's face on a still yet
  (crop to roughly 200 px of 480 tall, as above), and a still copies whatever it shows into
  every shot on its setup, a wrong garment included. Fix the still once, and every shot on
  it is fixed.
- **Consecutive shots on one setup.** Restarting both from the same still repeats the pose
  across the cut, which #118 found reads as a jump; chaining the second from the first's
  last frame is the obvious alternative and is untried for speech.
