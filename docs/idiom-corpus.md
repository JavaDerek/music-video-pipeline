# Idiom corpus: figures of speech that name a physical object (issue #75)

**H3 has no idiom dictionary. Every noun in a shot line is a candidate for the
frame.** A figurative phrase that happens to name a physical object gets that
object put on screen, whatever the sentence meant by it.

The rule is already in force in two places — `PROSE_PREAMBLE` rule 12 and
[`shot-writing-guide.md`](shot-writing-guide.md#an-idiom-that-names-a-physical-object-gets-rendered-as-that-object-issue-75).
This file is the **register of measured cases** behind them, and the reason
there is no lint.

## Why there is no lint yet

Issue #60's discipline: a keyword list ships only with measured separation on
a corpus with a known outcome, and every excluded candidate recorded with its
number. This corpus is **n = 1**. One phrase, one render, one confirmed
misfire. #76 is what happens when a keyword set is shipped on less than that —
sets justified at 3.5x that scored 0.72x and 0.99x once the full render
existed, one of them retired outright.

So: the preamble rule ships (it works at the stage that writes the sentence,
costs nothing, and cannot rewrite a correct line), and the lint waits for
cases. **This file is awaiting cases.** Add one whenever a render puts an
object on screen that only a figure of speech asked for — and add the
controls too, because a phrase that did *not* misfire is the half of the
evidence that decides whether a candidate keyword separates anything.

## Measured: the phrase caused the object

| phrase | where | what rendered |
|---|---|---|
| **"holds her line"** | "Deathless", `shot_plan_v4.toml` / `shot_plan_v5.toml`, chunk 53, start 340.417 s (5:40) | A rope, a harness and climbing hardware, in a video whose `role` describes no equipment and where no other chunk has any. A viewer: *"why does Diane suddenly have actual mountain climbing safety equipment... Did Health and Safety finally arrive on the video shoot?"* |

The line, verbatim:

```toml
shot = "Rock dust and loose stones hiss down the cracked face in a spreading
        sheet, spattering off the ledge just beneath her grip as she holds her
        line above the crumbling rock."
```

"Holds her line" is climbing idiom for a route or a rope. Nothing else in the
prompt asked for equipment.

## Measured: the phrase did NOT cause an object (controls)

These are the reason the rule is "does this figure of speech name a physical
thing I don't want in frame", not "avoid figurative language". All from the
same song, several from the same render.

| phrase | where | outcome |
|---|---|---|
| "her shadow sprawling up the rock face" | Deathless chunk 18 | No misfire. "Shadow" names a real, intended visual. |
| "the climb behind her worn down now to little more than a bare hill" | Deathless chunk 64 | No misfire. The mountain eroding *is* the story. |
| "time is smeared" | `setting`, composed into **all 80** prompts | No misfire in any chunk. "Smeared" describes a visual quality, it does not stand in for an object. |

## A second route to the same object, and it is not this one

The same song rendered climbing hardware **twice**, from two different causes,
and only one of them is an idiom:

* this issue — "holds her line", a figure of speech naming a rope;
* issue #73 — `avoid = [..., "modern climbing equipment, ropes, harnesses or
  safety gear"]`, a *prohibition*, composed into every prompt. Measured as a
  single-variable A/B at an identical seed on chunk 72: with the clause, a full
  climbing harness, carabiners and a trailing rope; without it, plain trousers.
  There is no negative-conditioning channel in `MiniMaxH3ReferenceToVideo` —
  one `prompt` input into one `BasicGuider` — so a prohibition cannot subtract,
  it can only name.

Worth stating together because the diagnosis differs and the fix differs. If a
render produces an object nobody asked for, check both: what named it
figuratively, and what forbade it.

## Untested candidates

Proposed by issue #75 and by inspection. **None of them occurs in any of the
80 shot lines of `shot_plan_v4/v5/v6/v8/v10/v11/v12`** (grep-checked), so this
corpus has no evidence for or against a single one, and none of them may ship
in a lint on the strength of "holds her line". Recorded so the next case has
somewhere to land:

- `draws the line` / `draw the line`
- `takes the reins`
- `the ropes` (as in "knows the ropes", "on the ropes")
- `in the saddle`
- `holds the fort`
- `throws in the towel`
- `the writing on the wall`
- `under the hammer`
- `keeps her feet` / `finds her feet`
- `hits the wall`
- `bites the bullet`

Two general shapes worth watching, both unmeasured:

1. **Equipment idiom in a physical-activity song.** Climbing, sailing, riding
   and combat all have idiom registers full of hardware nouns, and a song set
   in one of them is exactly where a prose stage reaches for them.
2. **Body-part idiom.** "Turns her back on", "puts her foot down", "shoulders
   the weight" — these name anatomy the render will frame, and framing is the
   one thing this project has measured over and over (#58, #76, #74).

## What would have to be true to ship a lint

- At least **three** confirmed cases, from **more than one song** — a
  single-song list is a list of that song's habits.
- Each candidate keyword scored against a corpus of shot lines with a known
  per-chunk render outcome, keeps and excludes both recorded with numbers, per
  #60 and #76.
- The lint stays warning tier. A figure of speech may be exactly what the
  author meant, and `write`'s revision round has already been measured
  rewriting 37 of 80 approved lines to satisfy a lint that was wrong (#87).
