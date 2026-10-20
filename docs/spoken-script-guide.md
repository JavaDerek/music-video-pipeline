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

[[shot]]
id = "s1"
seconds = 6
characters = ["mara"]            # everyone in frame, speaking or not
speaker = "mara"                 # at most one, and must be in `characters`
line = "We open at dawn."        # the voice track's words, not the video prompt's
text = "Close on Mara at her stall, hands on the counter."
posture = { mara = "standing" }
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

The check is on text. It cannot see a rendered portrait: **look at every generated cast
image for period correctness before spending video time on it**, and put the garment, in
words, in the portrait prompt too.

## State every posture, and mark every change

Left unsaid, posture is re-decided for each clip: people stood, sat, stood and sat
between cuts. Every character in frame needs a `posture`
(`standing`, `seated`, `kneeling`, `walking`, `lying`, `crouching`; `posture-missing`), and
a change from the last shot that character appeared in must be explained by the later
shot's `transition` (`posture-change-unmarked`):

```toml
transition = "Mara sits down on a stool"
```

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
