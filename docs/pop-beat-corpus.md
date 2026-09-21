# Pop-beat corpus: shot lines composed at the lens (issue #68)

A **pop beat** is one whose shot is composed for an object thrust at the lens,
so it reads as crossing in front of the screen plane — `Beat.pop_object` names
the motif. This file is the register of measured cases behind the checks, and
the record of every candidate that did **not** ship, with its reason. Same job
`docs/idiom-corpus.md` does for issue #75, and the same discipline: #60 and #76
both say a keyword list ships only with measured separation on a corpus with a
known outcome, and #76 is what happens when one ships on less.

## Provenance of every number below

Read this part first, because the honest status of this corpus is unusual.

* The **three positives** are reported in issue #68's comment of 2026-09-20,
  by the person who wrote and rendered them. The comment names the chunks and
  a short phrase for each; it does **not** publish the shot lines verbatim.
* The **negatives** (the zero-scoring keyword table) were measured 2026-09-13
  against `~/mvm-runs/deathless/shot_plan_v12.toml` — 80 shot lines and their
  59 `camera` clauses — and are reproduced from issue #68's comment and
  `docs/design-stereoscopic-3d.md`. They are **cited, not re-measured here.**
* The **depth-boil** numbers are from the same 2026-09-20 comment, measured
  with Depth Anything V2 Small against the 2026-09-13 stereo experiment.
* Nothing in this file was measured from a repo checkout. `~/mvm-runs` is a run
  asset, not project code (CLAUDE.md, "Authoring session state stays out too"),
  so the shot lines, the depth maps and the rendered frames are not here and
  cannot be re-derived from this repository. **A future re-score must name the
  run directory it read**, the way `docs/deathless-render-corpus.md` does in
  its opening lines — #93 is the standing evidence for why (an instrument
  pointed at the wrong render for a week, with only its output filename to
  say otherwise).

## Measured: pop beats that rendered as intended (n = 3)

"Deathless", 2026-09-20. Three hand-written shot lines staging an object at
the lens. All three rendered as intended — the first footage anywhere
containing a pop beat.

| chunk | object | phrase published in #68 | outcome | depth boil (fixed / per-frame) | max |
|---|---|---|---|---|---|
| 45 | needle | "needle" | rendered as intended | 0.0141 / 0.0190 | 0.117–0.130 |
| 46 | ember | "ember blooming out of black" | rendered as intended | 0.0271 / 0.0313 | 0.152–0.174 |
| 66 | mushroom cap | "mushroom cap filling the foreground" | rendered as intended | 0.0035 / 0.0074 | 0.014, lowest measured |

Two things this table is evidence for, and one it is not:

* **A held object at the lens is the easy case for depth.** Chunk 66 (held)
  boils a third as much as chunk 45 and an eighth as much as chunk 46 (an
  object *arriving* fast). Part of chunk 46's jitter may be the normalised
  depth re-ranging as the object arrives rather than true shimmer — that is
  the issue's own reading and it has not been separated.
* **The showcase shot is the worst case**, as the design doc predicted from
  the disocclusion argument, now with a second, independent reason: it is also
  where the depth estimate is least stable.
* It is **not** evidence about what a *generated* pop beat looks like. All
  three were hand-written by the person who added the field.

## What shipped, and what it can claim at n = 3

`prose.pop_object_named_in_shot_issues` — a pop beat whose shot line never
names its own `pop_object`.

**No vocabulary.** Correct authoring names the object and incorrect authoring
does not; there is no word list to be scored wrong. Same shape as
`authoring.worldstate.check_location_tags` (#78), which fires on 2 of 80 with
zero false positives and no vocabulary at all.

**Why it is worth having.** `pop_object` is composed into **no prompt**.
`plan.render_plan_toml` writes it into the `# beat:` comment beside `act`, and
the render reads neither, so the shot line is the only channel by which the
object a pop beat exists for can reach H3. A beat marked `pop_object` whose
line never names the object has the beat sheet paying for a plant, the
photography stage composing `camera` toward the object, and nothing to pop.
That is issue #55's general form one level down: a stage's output with nowhere
to land is silently discarded.

**What it claims:** it fires on **0 of 3** known-good lines. That is the
false-positive half, at n=3, and it is the half that matters for a check that
may never block a run.

**What it does not claim:** any true-positive rate. No authored pop beat has
ever been observed *omitting* its own object, because only three have ever
been authored. **Re-score when a generated plan carries pop beats** — that is
the missing arm, and the one where a model asked to compose at the lens might
well write around the noun rather than name it.

**What it declines to examine:** a `pop_object` whose every word is a stopword
or shorter than the lint's minimum length ("the axe", "an urn") normalises to
no stems, and the beat is logged as UNEXAMINED rather than passed. #93: a zero
from a detector is not evidence of absence unless something asked the second
question.

Already shipped under #68 and unchanged here: `prose.pop_distant_staging_issues`,
which re-runs `shot_plan._distant_staging_match` — a *measured* vocabulary
(#58, re-scored #76) — scoped to pop beats. None of the three rendered lines'
published phrases trips it, which is one more 0-of-3 and no more than that.

## Candidates registered and NOT shipped

Recorded with reasons, per #60. None of these may ship on the strength of
three hand-written lines.

### 1. A "toward the lens" keyword set

Scored 2026-09-13 against 80 shot lines and 59 `camera` clauses:

| candidate | shot lines (n=80) | camera clauses (n=59) |
|---|---|---|
| `toward(s) the lens` | 0 | 0 |
| `at the lens` | 0 | 0 |
| `into the lens` | 0 | 0 |
| `toward(s) camera` / `at the camera` | 0 | 0 |
| `into frame` | 0 | 0 |
| `fills the frame` | 0 | 0 |
| `past the lens` | 0 | 0 |
| `foreground` (bare, and `in the foreground`) | 0 | 0 |
| `closer` | 0 | 0 |
| `across the frame` | 0 | — |
| `out of frame` / `exits frame` / `off the edge` | 0 | — |

The only phrases with hits are the generic ones — bare `across` (20 of 80) and
bare `past` (7 of 80) — and reading them they are "across the valley" and
"past the mill", nothing to do with the frame edge. For contrast the
distant-staging vocabulary that *is* linted scores `distant` 1 and `horizon` 11.

**Still excluded, and the 2026-09-20 corpus does not change it.** Three
positives now exist, but their text is not published verbatim, so no candidate
can be scored *on* them; and 0 positives out of 80 on the negative side means
a keyword set would still be a hypothesis with a measured false-positive rate
of zero and an unmeasured true-positive rate of zero. Note what the published
phrases *hint* at — "filling the foreground" would be caught by `foreground`,
which scores 0 on all 80 — and note equally that a hint from a paraphrase is
not a measurement. Get the lines.

### 2. "The pop object is the grammatical subject of its line"

**Not shipped: it cannot be distinguished from the cheaper check it would sit
beside.** All three published phrases both *name* the object and *lead* with
it ("ember blooming…", "mushroom cap filling…"), so at n=3 a subject-position
rule and the naming rule agree on every case, and only the naming rule needs
no parsing. A lead-position rule would also fire on a line that is correct by
every other measure — "She turns as the needle swings at the lens" — which is
a false-positive surface bought for no separating evidence. #58's own finding
is the precedent in the other direction: grammatical subject is *necessary but
not sufficient*, and the thing that actually decided whether an object
rendered was near-vs-far staging, which `_lint_distant_staging` already covers.

### 3. "The pop object's own plant names it too"

`check_beat_structure` already requires a pop beat to have an earlier `plant`
in its own `beat_group` — but that is a check on *beats*, and nothing checks
that the plant's **prose** ever shows the object. A plant that does not is
planting nothing, and the object is invented as it flies at the lens, which is
exactly what `BEATS_PREAMBLE` rule 8 exists to prevent.

**Not shipped:** there is a named false-positive path with no evidence either
way — a plant may legitimately show the object under a different noun (a tray
of syringes plants the needle; a bed of coals plants the ember) — and this
corpus contains **zero plant lines**, because #68's comment publishes only the
three pop chunks. Build it when a corpus has plant/pop pairs, and score it the
way #85's plant-end-state rule was scored: over every plant→consequence pair
in every real plan, with the false positives counted.

### 4. A cap on pop beats per video

The design doc says "a video with six of them is a different video" and
`BEATS_PREAMBLE` asks for rarity. **Not shipped:** three is the only number
ever authored, so any cap would be invented. Rarity stays asked-for, not
counted — the same way it already is.

### 5. A window-violation keyword set (`across the frame`, `out a side`)

The illusion collapses when an object in negative parallax is clipped by the
frame edge, so this is the check with the highest value per hit. **Not
shipped:** every candidate scores 0 on all 80 lines (table above) and none of
the three positives is reported as a violation, so there is neither a positive
nor a negative to score against. `PROSE_PREAMBLE` rule 14 and
`PHOTOGRAPHY_PREAMBLE` already carry the instruction, which is the half that
costs nothing and cannot rewrite a correct line.

## What would have to be true to ship a keyword lint here

Deliberately the same bar `docs/idiom-corpus.md` sets, because the failure
mode is the same one:

* The **shot lines themselves**, verbatim, for every pop beat — not a
  paraphrase of them. Everything above is blocked on this and on nothing else.
* At least one **generated** plan carrying pop beats, so the corpus stops
  being three lines written by the person who invented the field.
* A per-chunk **rendered outcome** for each, so candidates can be scored
  against something other than intent: the same instrument
  `docs/deathless-render-corpus.md` uses, with its input directory named.
* Both halves recorded — keeps *and* excludes, with numbers — and the lint
  stays warning tier regardless. #87 is the standing evidence: `write`'s
  revision round rewrote 37 of 80 approved lines to satisfy a lint that was
  wrong about function words.
