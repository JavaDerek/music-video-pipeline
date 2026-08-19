# Shot-writing guide

How to write `shot_plan.toml` entries so a cause and its effect actually read
as connected on screen, instead of as two unrelated things that happened to
render in the same six seconds. This is issue #26's option (a): zero new
machinery, just what the first full render taught about writing the line
that goes in `shot = "..."`.

If you haven't read [`music_video_maker/shot_plan.py`](../music_video_maker/shot_plan.py)
and the README's [Shot plan](../README.md#shot-plan) section, start there —
this guide assumes you already know the mechanics (one TOML `[[shot]]` table
per chunk, keyed by `chunk_id`, `start` as a drift anchor, `shot` as free
text). This guide is only about what to put in `shot`.

**Camera direction has its own field now (issue #53).** Every worked example
below that ends a `shot` line with a trailing camera phrase — `"...,
camera tracking backwards ahead of her"` — still works exactly as written;
`shot` stays free text. But for new plans, prefer the structured `camera`
field beside `shot` (see the README's
[Cinematography and camera direction](../README.md#cinematography-and-camera-direction)):
it composes as the same trailing clause automatically, so "Make the effect
the grammatical subject" below can't be undone by camera language
accidentally landing in subject position of its own sentence.

## The defect this exists to fix

Across a whole video, the *state change* renders and the *agency* does not.
The disaster happens; the protagonist's causal role in it is invisible. Her
sleeve doesn't visibly hit anything, but the coffee is on the console anyway.
The window is broken, but nobody threw the chair through it as far as the
video is concerned. H3 will happily render "broken window." It will not
infer that the person walking past a moment earlier is the reason it broke.

This is the evidence table from the first full render of "The Lucky Ones"
(issue #26):

| Time | Authored intent | What rendered |
|---|---|---|
| 1:23 | snow plow crosses the path, jogger dives clear | jogger falls over, with no plow contact |
| 1:46 | her sleeve sweeps a coffee across a console, rocket fails | coffee **floats through midair** into her hands; no link to the failure |
| 2:36 | she presses mute in passing, the gallery erupts | mute pressed, **no visible reaction** from anyone |
| 2:47 | a chair goes through a plate glass window | windows shatter **out of nowhere** |

Every failing row has the same shape: two beats — a *cause* and an *effect*
— were asked for inside one shot, and H3 rendered the more visually dominant
one (the explosion, the shattering) while dropping the subtle one (the
sleeve, the foot). The one gag from that render that *did* work — the
printer catching fire — is the one whose beats were spread across three
separate chunks (shots 2 → 4 → 5) instead of compressed into one. That's the
whole thesis of this guide: **causation reads when it is given room, and
fails when compressed.**

The confirmed, in-repo half of that gag is chunk 4's actual authored line
(from the README):

```toml
[[shot]]
chunk_id = 4
start = 26.333333333333336
shot = "She rounds the end of the office aisle still singing to camera as, behind her, the printer erupts in a fireball"
```

Note what this line is *not* doing: it doesn't restate who she is (that's
`role`, composed in automatically) or what she's singing (that's the lyric
clause, also automatic — see `prompting.py`). It's pure staging: where she
is, what's behind her, what's happening there. Keep your own `shot` lines
that lean too — camera, staging, action, environment. Character identity and
the lyric line are the render loop's job, not yours.

## The three-beat rule

Plant → contact → consequence. Three beats, three shots, not one.

- **Plant** — establish the thing that's about to matter. A printer sits in
  the background. A coffee cup sits on the console edge. A chair sits by the
  window.
- **Contact** — the subject touches it. Name the body part and what it
  touches (see below). This is the beat every failing row in the table
  above was missing entirely.
- **Consequence** — the state change plays out, and ideally in frame with
  (or just after) the subject who caused it, not as an unexplained cutaway.
  The state change has to be *physical* — something in frame changing state,
  not only people reacting to it. See "Physical manifestation, not human
  reaction" below: a consequence beat that skips this can survive the
  three-beat split intact and still fail.

### Budget this in seconds, not shots

Chunks are not a fixed length: the slicer snaps each one to H3's frame grid
(steps of 17 frames from a minimum of 5, at a fixed 24 fps) inside the
window the hardware profile allows. The shipped `examples/first-run.toml`
caps that at 8.0 s, which gives five valid lengths — 124 / 141 / 158 / 175 /
192 frames, i.e. **~5.2 s to 8.0 s per chunk**. The 141-frame chunk (5.875 s)
is the one the render-time figures in `CLAUDE.md` were measured at. H3's full
trained range is wider, 124–362 frames (~5.167 s – ~15.083 s;
`docs/workflow-template-guide.md`), but nothing longer than 141 frames has
been rendered on this card yet.

So a three-beat gag built one beat per chunk costs roughly **three chunks,
~16–24 s of song**, not one 6-second shot. Check your own run's actual chunk
boundaries rather than assuming a length. Budget for that when you're deciding how many separate story
beats a given stretch of the timeline can afford — you cannot fit plant,
contact and consequence for two different gags into the same ~18 s window
without them competing for the same frames.

This also means the three shots don't have to be contiguous down to the
frame. A plant can sit a chunk or two before the contact if something else
needs to happen in between (a lyric line, a different beat) — what matters
is that plant and consequence are both anchored to the same object across
shots, not that no other content exists between them.

## One beat per shot

The single most common fix from the evidence table: stop asking one `shot`
line to carry both the cause and the effect. Split it into (at least) two
consecutive chunks, one beat each.

Below are the four failing rows from the table, rewritten as multi-shot
sequences. These are illustrative rewrites — the actual chunk numbers and
`start` values on a real render come from that run's alignment output
(`run_state.json`), never invented — but they show the shape a fix takes.

**Before** (one shot, cause and effect compressed — this is what produced
"jogger falls over, with no plow contact"):

```toml
[[shot]]
chunk_id = 14
start = 83.0
shot = "A snow plow crosses her path as she dives clear at the last second"
```

**After** (plant → contact → consequence, three chunks):

```toml
[[shot]]
chunk_id = 14
start = 83.0
shot = "She jogs along the path; foreground, a snow plow rumbles into frame from the left, still several strides away"

[[shot]]
chunk_id = 15
start = 88.9
shot = "Close on her feet planting hard and pushing off as the plow's blade fills the frame behind her, inches from her trailing leg"

[[shot]]
chunk_id = 16
start = 94.8
shot = "She lands the dive clear on the verge, breathing hard, as the plow rumbles past behind her along the path she'd just left"
```

**Before** (coffee and rocket failure in one shot — rendered as "coffee
floats through midair into her hands"):

```toml
[[shot]]
chunk_id = 24
start = 106.0
shot = "Her sleeve sweeps a coffee cup across the console and the rocket launch fails on the monitors behind her"
```

**After:**

```toml
[[shot]]
chunk_id = 24
start = 106.0
shot = "She leans across the console reaching for a switch; foreground, a full coffee cup sits inches from her trailing sleeve"

[[shot]]
chunk_id = 25
start = 111.9
shot = "Close on her sleeve catching the cup's rim, coffee arcing sideways across the console toward the keyboard"

[[shot]]
chunk_id = 26
start = 117.8
shot = "Coffee pooling across the console keys as, on the monitors behind her, the rocket launch telemetry flashes red and cuts out"
```

**Before** (mute press and gallery reaction in one shot — rendered as "no
visible reaction from anyone"):

```toml
[[shot]]
chunk_id = 41
start = 156.0
shot = "She presses mute in passing and the gallery erupts behind her"
```

**After:**

```toml
[[shot]]
chunk_id = 41
start = 156.0
shot = "Walking past a console, her finger presses a glowing mute button as she passes without breaking stride"

[[shot]]
chunk_id = 42
start = 161.9
shot = "The gallery behind her turns toward the now-silent speakers, several people rising out of their seats"
```

This rewrite fixes the compression problem in this shot — but it is not the
end of the story for this gag. The same shape (three beats, a dedicated
consequence chunk, worded differently but doing the same job) was tried for
real on a later render, and it still didn't work: a room reacting, nothing
in frame changing state, and the report back was that it "doesn't show
consequences at all." See "Physical manifestation, not human reaction"
below — the missing ingredient wasn't more beats, it was a *physical*
consequence, not only a social one.

**Before** (chair and window in one shot — rendered as "windows shatter out
of nowhere"):

```toml
[[shot]]
chunk_id = 44
start = 167.0
shot = "A chair goes through the plate glass window"
```

**After:**

```toml
[[shot]]
chunk_id = 44
start = 167.0
shot = "She grips the back of a chair, foreground, hauling it up off the floor as the plate glass window fills the background"

[[shot]]
chunk_id = 45
start = 172.9
shot = "The chair leaves her hands mid-throw, arms still extended from the release, glass already starring where it will hit"

[[shot]]
chunk_id = 46
start = 178.8
shot = "The window frame empty and jagged, glass scattered across the floor in the foreground where the chair landed"
```

## Name the contact explicitly

Every worked example above has one shot whose entire job is the *touch*: the
sleeve catching the cup's rim, the finger on the mute button, the foot
planting before the dive, the hands releasing the chair. Say which body part
and what it touches, in that shot's line, plainly. "She interacts with the
console" is not contact — H3 has nothing concrete to render. "Her sleeve
catches the cup's rim" is.

This is the single highest-value sentence in a three-beat sequence. If you
only have room to elaborate one of the three shots, elaborate the contact
shot.

## Describe the end state, not the motion

A failure shows up even inside a single, well-named contact shot. One
render's plan asked for:

> "the toe of her boot hooks under the taut power cable and drags it out of
> the wall socket"

and got the gesture only: "his foot brushes the machine cord, but doesn't
pull it out." Naming the body part and the object, per "Name the contact
explicitly" above, was not enough here — the line also asks the model to
carry an extended action through to its finish (hooks, drags, out), and a
single generated clip does not reliably resolve a process to its result. It
rendered the start of the gesture, not the outcome.

**Before** (the actual line from that render, not an illustrative rewrite):

```toml
shot = "The toe of her boot hooks under the taut power cable and drags it out of the wall socket"
```

**After:**

```toml
shot = "The plug now lies free on the floor, the socket empty behind it, her boot still close by"
```

"The plug now lies free, the socket empty" is a state a single frame can
satisfy. Where a shot line for a contact-and-consequence beat has to choose
between the motion (hooking, dragging, sweeping, throwing) and the result
(freed, spilled, shattered, dark), describe the result.

(#28's first/last-frame chaining is the more principled fix for this —
constrain the actual last frame instead of hoping prose lands on it — but
it's unbuilt; this is the zero-code version.)

## Name the outcome in plain physical terms

Same failure, different cause. One render staged a lyrical defenestration —
the lyric is "You would be defenestrated," a comic threat — as a mechanical
accident:

> "Close on her elbow clipping the emergency window release on the frame
> beside her, the catch springing open and the pane swinging wide onto the
> winter air"

and it rendered exactly that: a window opening. This isn't the model
softening violence on its own — the same catalogue's first render asked for
a chair through plate glass in plain terms ("the whole pane stars outward
from the impact point in a spreading web… the cracked window finally lets go
and collapses into the lobby"), and the evidence table at the top of this
guide already confirms the glass shattered; what was missing there was the
*agency*, this guide's whole subject, not the breakage. The glass broke when
the line asked it to break. It stayed shut here because the line, dressed up
as a release-and-swing mechanism, asked for a window opening — nothing more.

This is "describe the end state, not the motion" from another angle: name
the physical thing you actually want, not a mechanism or a decorous
stand-in for it.

**Before** (the actual line from that render):

```toml
shot = "Close on her elbow clipping the emergency window release on the frame beside her, the catch springing open and the pane swinging wide onto the winter air"
```

**After:**

```toml
shot = "Her elbow catches the window and the pane shatters outward in a spray of glass"
```

If the outcome you actually want *is* a window opening — no euphemism, that
is genuinely the beat — there's nothing wrong with the "Before" line. The
rule only bites when plain physical language is being avoided in favor of a
softer stand-in for something you actually want blunter, more violent, or
more comic than the line admits.

## Physical manifestation, not human reaction

The mute gag (see "One beat per shot" above) is this guide's clearest case
study in a failure that survives the three-beat split. It failed **twice**,
in two different stagings:

- The first render compressed it into one shot and got "mute pressed, no
  visible reaction from anyone" (the evidence table at the top of this
  guide).
- A later render gave it a dedicated consequence chunk of its own — "the
  whole office floor turns at once toward a dead speaker, several people
  rising out of their chairs" — and it still "doesn't show consequences at
  all."

Set that against what did render across the same renders: the printer
fireball, the plate-glass window, the coffee spilling across the console,
monitors going dark. Every one of those has something in frame *change
state*. The mute gag, in both stagings, asked only for people to notice
something happened:

> H3 renders physical state changes. It does not render social ones.

There is no pixel-level difference between a room of people about to react
and a room of people who already have — "people notice" isn't a visual
event the way "fire" or "broken glass" is.

**Every consequence beat needs a physical manifestation of its own** —
something in frame that changes state — not only, and not primarily, a
human reaction. The reaction can still be in the shot; it just can't be the
only thing carrying the beat. Below is an illustrative fix (untested, like
the rewrites in "One beat per shot" above) for the mute gag's consequence
chunk:

```toml
[[shot]]
chunk_id = 42
start = 161.9
shot = "Behind her, the console's VU meters drop to zero and the ON AIR light above the booth goes dark; the gallery turns toward the dead speaker, several people rising from their seats"
```

The reaction stays — it's just no longer the only thing being asked to
carry the shot.

If a chunk's whole job is a consequence the subject has already walked past
or away from, set `focus = "action"` on it
(`ShotPlanEntry.subject_is_focus` in `shot_plan.py`). Without it, the
character-attached "X is the focus of this shot" instruction that's
composed into every prompt competes with the consequence for the render's
attention — and across four renders, the character-attached instruction has
won.

## Carry the consequence into the next shot's background

A consequence that only exists in the shot where it happens is one lucky cut
away from reading as coincidence again. Once a state change has occurred —
spilled coffee, a broken window, a shut-off switch — keep it visible,
deliberately, in the background of whatever comes after it, the same way
`docs/workflow-template-guide.md`'s continuity path (I2V, first/last-frame
chaining) carries an image forward on purpose rather than by luck. You don't
need I2V continuity turned on to do this in prose: just write the aftermath
into the next shot's background clause, the way the rewritten sequences
above put "the plow rumbling past behind her" and "glass scattered across
the floor" into the shot that follows the consequence, not just the
consequence shot itself.

## Elaborate each shot

A one-clause shot line gives H3 nothing to anchor around except the subject.
Every `shot` line should carry, where relevant:

- **Camera framing** — close on the hands/feet, or a wider shot that holds
  the subject and the object of the beat in the same frame. The evidence
  table's failures are consistent with a camera that never framed cause and
  effect together; be explicit about what's in frame.
- **Foreground vs. background** — what's about to matter should usually be
  named as foreground (the thing she's approaching) or background (the
  thing about to react), not left implicit. Every worked example above
  states this explicitly.
- **Hands and feet** — H3 renders a person as a whole performance; without
  direction, hands and feet default to whatever's generic for the pose. If
  the beat depends on a foot planting or a hand releasing a chair, say so.

This doesn't mean padding every line with adjectives — `global_style` and
`narrative_concept` already carry the film's general look, and the cast
member's `role` already carries appearance and wardrobe (see
`docs/lyrics-format.md` and `prompting.py`'s module docstring — a `role`
must never describe vocal action, and the same discipline applies here: a
`shot` line's job is staging and action, not restating things the render
loop already composes for you).

## Keep locations inside the run's `setting`

`config.setting` anchors where and when the whole video takes place (issue
#32) — see `examples/first-run.toml`'s `setting` field and
`music_video_maker/config.py`. Leave it unset and each shot's geography is
whatever that shot's own line implies with nothing to keep them consistent;
the first full render walked out of a British terraced street into a
Central-Park-style US city scene this way, chunk to chunk. The config loader
already warns when `setting` is missing entirely.

Every `shot` line you write should describe a location and time of day that
plausibly exists *inside* that setting — don't name a place the run's
`setting` rules out (a US strip mall in a run whose `setting` is contemporary
London). This is an authoring discipline this guide is asking you to hold
yourself to, on top of whatever the loader checks for you: the three-beat
sequences above are also easier to keep coherent when every beat in the
sequence is understood to happen in the same real place, not just the same
`shot` line.

## Track where each character is, not just where the world is (issue #78)

`setting` is one fact for the whole video; it says nothing about where a
character IS inside it at a given moment, and a shot may not reference
another shot to say so (issue #61 — each chunk renders from its own prompt,
for a model that has never seen any other). A real "Deathless" render made
the gap concrete: chunk 7 (0:54) put Volokov's mill "in the valley below"
while she was still climbing toward it; chunk 13 (1:32), 37 seconds later
with every line between them describing continued ascent, had her palm
"trailing along its worn wood grain" at shoulder height. She cannot be
climbing away from a landmark and touching it a minute later. The same plan
put `present = ["Jan"]` on a hillside chunk that never mentions him, and
dropped him from the two chunks right after — the "hill where Jan is, cuts
to a different hill where Jan isn't" a viewer's first note complained about.

The fix is `ShotPlanEntry.location` — a tag drawn from a small closed set
the concept stage names for the song (`"valley floor"`, `"switchback"`,
`"mill"`, `"watch-post"`), assigned per chunk by the beats stage the same way
`beat_role`/`beat_group` already are: checkable before a word of prose
exists, never guessed from finished text afterward. It exists so two
mechanical checks can run:

* a chunk naming the same landmark as another chunk close by in song time,
  at contradictory distances (the mill "below" then "at shoulder height");
* `present` staging a companion at a location that contradicts where their
  own singing chunks elsewhere in the plan put them.

Both are warnings, not errors, for the reason every lint in this guide is:
a false positive on prose written deliberately must never block a run.
`location` has no bearing on wording — keep writing shot lines exactly as
the rest of this guide describes; just give each beat an honest answer to
"where is this happening" and let the two checks do the rest.

**`location` is also composed into the render now** (previously it was
"purely a checkable field, not a rendered one" — a "Deathless" render is why
that changed: its `setting` string named a detonation's own nouns, medieval
hosts, industrial armies, nuclear glow, and composed them into all 80
chunks unchanged, so the 17 shots authored for the aftermath — an eroded
hill, a bone-dry mill race — still carried the pre-detonation setting text
verbatim. `location`, when you set it, *substitutes* for `setting` in the
"Location continuity" sentence for that one chunk — it does not add a
second, qualifying sentence next to `setting`'s, because H3 renders whatever
noun it is given regardless of the framing placed around it (issues #73,
#74), so a second sentence cannot retract a noun the first one already
named. Practically: still author `location` from the closed vocabulary as
above, and know that for any chunk whose moment in the song has drifted from
`setting`'s own text — a war that has ended, a world that has emptied —
setting `location` is now what actually keeps the render honest, not only
the two lints.

## Make the effect the grammatical subject

This is the single highest-leverage sentence-level rule in this guide, and it
was found by A/B test rather than by reasoning. **H3 renders the grammatical
subject of the shot line and largely drops what sits in subordinate clauses.**

Three renders of the same chunk, same lyric, same everything except the
sentence:

| Shot line | What rendered |
|---|---|
| "**She rounds** the end of the office aisle as, behind her, the printer erupts in a fireball, staff scattering from their desks" | Nothing. A mild office, her smiling to camera. No printer, no fire, at any frame. |
| The same sentence, plus `focus = "action"` | Indistinguishable from the above. The flag changed nothing. |
| "**The printer** at the end of the office aisle **erupts in a fireball**, staff scattering back from their desks and papers catching light in the air, while she rounds the aisle in the foreground still singing to camera" | Papers flying through the air, sheets across the floor, staff ducking at their desks. |

The content of all three is identical. Only the grammar moved, and it moved
the render from "nothing happened" to "the room is mid-catastrophe".

This rule explains every consequence failure this project has recorded:

- "she presses mute in passing, **the gallery erupts**" -> she pressed mute; the gallery did nothing
- "her sleeve sweeps a coffee across a console, **the rocket fails**" -> the sleeve swept; the rocket was fine
- "she rounds the aisle as, behind her, **the printer erupts**" -> she rounded the aisle

In each, the beat that vanished was the one in the subordinate clause.

**So: when a shot is about something happening, that something is the
subject of the sentence, and the performer goes into a subordinate clause.**
She stays in frame, she keeps singing, she is still described — she is simply
not what the sentence is *about*.

```toml
# Wrong -- the consequence is subordinate and will be dropped:
shot = "She walks past the console as every monitor behind her goes black"

# Right -- the consequence leads, she is still present and still singing:
shot = "Every monitor along the console goes black one after another, as she
        walks past in the foreground still singing to camera"
```

Note what this rule is *not*. It is not "remove the performer" — a chunk
carrying a lyric still needs a mouth to sync, so writing her out costs you the
lip-sync for that whole chunk. Keep her; just stop making her the subject.

## Stage the object near, not far

Subject position is necessary but not sufficient. A validation slice of a
machine-authored plan (issue #58) put a printer in subject position for both
its plant and its payoff beat, exactly as the rule above asks, and H3
rendered neither:

- plant: "A beige printer sits blinking on the lit sill of that same upper
  window, **small against the tower far behind her**" — no printer rendered.
- payoff: "The printer lies smashed in a snowbank **far behind on the
  street**, its tray sprung and paper bursting up into the falling snow" —
  no smashed printer, just a correct smile.

A third beat in the same slice — a window and a chair, both objects staged
near and large — rendered both, in one frame. This guide's own successful
example was already consistent with that and nobody had noticed why: "The
printer at the end of the office **aisle** erupts in a fireball… while she
rounds the aisle in the foreground." Same room, mid-ground, performer in
front of it — not small and far behind her.

**So: subject position gets the model to attend to the thing; spatial
staging decides whether it is in frame at all.** Stage the object of a beat
in the **near or mid ground, in frame with her**, and never describe it as
small, distant, or far behind.

```toml
# Wrong -- subject position, but staged small and far away, and dropped:
shot = "A beige printer sits blinking on the lit sill of that same upper window, small against the tower far behind her"

# Right -- same beat, staged near:
shot = "A beige printer sits blinking on the office windowsill beside her, close enough to touch"
```

"Behind her" is about *narrative* obliviousness — she hasn't noticed it yet —
not about depth in the frame. A thing she hasn't noticed can still be large
and near.

## A voiced consequence chunk can cost the lip-sync, and wording doesn't fix it

A chunk carrying a lyric still needs a mouth to sync — stated above as a note
on the grammatical-subject rule. The first real render treated this as a
camera-wording problem; it isn't one, and three follow-up re-renders of the
same chunk proved it the hard way.

The validation slice (issue #58) gave its one successful three-object
consequence shot a camera direction: "craning up the building face to the
empty pane, then settling back down to her pace", combined with
`focus = "action"`. The shot bought its consequence at a real price: she is
back-to-camera for most of it, and it carries the lyric "the world was
breaking, always taking me". No mouth, no sync, on a chunk that is, by the
song's own timeline, supposed to be singing.

Three follow-up re-renders of that exact chunk, each changing one thing:

| Attempt | Change | Result |
|---|---|---|
| 1 | Rewrote `camera` to "pushing in past her shoulder... holding her face and the window together" | Face visible ~0-40%, back-to-camera the rest |
| 2 | Rewrote `camera` again, dropping "past her shoulder", to "holding steady... her face never leaving the shot" | Identical split — no measurable change |
| 3 | Rewrote the `shot` line itself to stop her walking and say "her face still turned toward us" | Turned away *earlier* (~20-30%), and by the shot's end she was walking away again despite the line saying she'd stopped |

Three independent wording changes, three failures, in the same direction.
**The thing held constant across all three was `focus = "action"`.** That
flag hands the sentence's grammatical subject to the consequence instead of
the performer — deliberately, per "Make the effect the grammatical subject"
above — and once she is not the subject, nothing tried here reliably kept
her oriented toward the lens either. Camera direction is a trailing clause
by design (issue #53); it cannot out-compete the sentence it trails.

### What the first full render did to that conclusion (issue #60)

**Read the above as history. The generalisation did not survive contact.**

The 36-chunk render that followed put the claim on 7 voiced chunks at once
instead of 1. All 7 turned away from the lens, exactly as predicted — and
**all 7 kept their lip-sync.** Turning away is real, common, and *usually
free*: H3 syncs a mouth in profile, and a performer who turns back within the
shot loses nothing. The inference above had confused a mechanism (she turns
away) with a consequence (the sync breaks) on the strength of a single chunk.

Worse, the one chunk in that render that *did* lose its sync was not flagged
at all. Chunk 13 is a `transition` beat with `focus` unset:

```toml
# lyric: "There was a time / But that's the past / You know I'm so much better"
shot = "Behind them, slush slides slowly off the buried car's roof as they
        keep walking, his bass still setting the pace beside her"
```

No `focus = "action"` anywhere — and the prose does the same job anyway. The
camera is behind them, the grammatical subject is "slush", and they are
walking on. **The predictor is in the sentence, not the field.**

So `focus = "action"` on a voiced chunk is no longer flagged, and is no
longer a thing to avoid: it is still required on every `consequence` beat
(issue #26), and the story, not this guide, decides which chunk carries one.
What to actually avoid on a chunk that carries a lyric is *walking her away
from the lens for its duration* — "as they keep walking", a fronted "Behind
them,". Where the movement matters, put it on a neighbouring instrumental
chunk.

One thing the correction does not touch: "behind her" describing where an
object sits is fine and always was. It appears in 20 of those 36 lines, all
of which synced. It means she has not noticed the thing yet, not that the
camera is behind her — the same distinction "Stage the object near, not far"
draws above.

```toml
# Risky: a consequence's focus="action" on a chunk that also carries a lyric --
# no camera wording measured so far has reliably kept her facing the lens:
[[shot]]
chunk_id = 26
# lyric: "the world was breaking, always taking me"
focus = "action"
shot = "..."

# Safer: the same consequence, moved to a nearby instrumental chunk in the
# same beat group -- nothing left to trade away:
[[shot]]
chunk_id = 27
# INSTRUMENTAL -- no lyric to sing
focus = "action"
shot = "..."
```

`lint_camera_face_away_on_voiced_chunks` runs once chunks exist (the same
point as `lint_shots_against_lyrics`, since it also needs to know which
chunks are voiced) and checks two signals. The primary one is now the
**shot line** reading as walking her away from the lens — "keep walking",
a fronted "behind them" — calibrated against those 36 real lines rather than
invented: both phrases match chunk 13 and no other voiced chunk. "receding"
is deliberately excluded despite sounding apt, because it matches chunk 7,
which turned away and kept its sync. The older, narrower signal — a `camera`
value that reads as turning her away ("back to camera", "away from her") —
is still checked for the chunks that spell it out. Both are warnings, never
errors, same bar as every other lint here, and the evidence base is one true
positive, so treat it as a prompt to look rather than a verdict.

## Framing choices on a voiced chunk, re-scored against the full render (issue #76)

`lint_voiced_framing`'s keyword sets (above, and the ones covering "camera
too far from the face") were scored mid-render, on a partial "Deathless"
corpus. The finished 80-chunk render — 41 voiced chunks, corpus mean face
presence 53.3% — does not support them as originally shipped. This is #60's
own lesson arriving one level up: a generalisation from a partial sample is
a hypothesis, and the full render is the experiment.

**Avoid framing a voiced chunk in profile.** Left out of the shipped keyword
list at first sample (three chunks, 8%/0%/45% face presence — "a real risk
but not a reliable one"), and reversed on a larger sample rather than
silently changed: across all 5 occurrences on the finished render (chunks
18, 21, 37, 43, 59) face presence never once cleared the corpus mean — worst
case 8%, best case 42%. It is the one framing choice tested here with no
counter-example, and it fires even on an otherwise-close shot: a profile
close-up is still a profile shot.

**A gaze verb that settles the eyes on something in the scene is still
risky, but the list is narrower than it was.** Of the original 9 gaze
phrases, only 6 actually separated on the full corpus (each one's single
measured occurrence landed below the 53.3% mean); the other 3 — "gaze
drops", "gaze lifts", "lifts her gaze" — measured *above* the mean, because
in each case the `camera` field also named the face directly and the camera
field won. The lesson isn't "gaze verbs are safe now" — it's that a gaze
verb only costs the face when nothing else in the entry is holding it there.

**Framing an object at foot level is no longer a checked keyword.** The
underlying finding was real — a line that read "pushing up between his
boots" once rendered legs, boots and mushrooms with no face anywhere — but
the keyword list built from it (boots, feet, ankles, the ground, …) scored
0.99x against the rest of the voiced corpus on the full render:
indistinguishable from noise. "the ground" alone ranged from 33% to 92%
face presence across its two occurrences, the same phrase in both. If a
sung chunk stages something at the performer's feet, treat it as a judgment
call rather than something the tooling will catch.

**A hypothesis that was tested and rejected: does the `camera` field name
the face at all?** Proposed from two examples where the identical camera
phrase "travelling with her" scored 83% face presence in one chunk and 0% in
another — one of the two named "her face... in the same frame", the other
did not. Tested as a literal keyword match for "face" across all 41 voiced
chunks: chunks whose `camera` names the face averaged **50.8%**, chunks that
don't averaged **59.8%** — backwards from the hypothesis. The best-scoring
chunk in the whole corpus (100% face) never says "face" at all. Naming the
face in the camera clause is not, on its own, a usable predictor; don't
re-propose it as a literal keyword without a different operationalisation.

Full numbers, per-chunk detail, and the excluded candidates for every one of
these decisions: `docs/deathless-render-corpus.md`.

## Never refer to another shot (issue #61)

Each chunk is rendered on its own, from its own prompt, by a model that has
not seen any other shot in the film. A line that points back at an earlier
one is pointing at nothing, and whatever clause is anchored to that reference
has no anchor.

Measured on the first full render. Chunk 27's line:

```toml
# Wrong -- "that same empty frame" refers to a window broken two shots ago,
# which this render has never seen. The printer did not appear at all:
shot = "The printer now teeters on the sill of that same empty frame, smoke
        pouring past it into the grey sky, while she continues along the
        sidewalk far below with him beside her"

# Right -- the frame is described, not referenced:
shot = "The printer teeters on the sill of a jagged, glassless upper window,
        smoke pouring past it into the grey sky, as she walks the sidewalk
        below with him beside her"
```

The shots either side of it rendered their printers; this one did not. Note
the second problem in the same line — "far below" — which the near/far rule
above covers and which the same edit fixes.

Re-describe the thing in full every time. The repetition costs you nothing:
nobody reads two shot lines side by side except you.

## An idiom that names a physical object gets rendered as that object (issue #75)

H3 has no idiom dictionary. Every noun in a shot line is a candidate for the
frame, whether you meant it literally or not — and idiom, metaphor and
figurative compression are exactly the constructions that make prose read
well, so a stage optimising for good writing is optimising straight into this
trap.

Measured on a real render. A viewer asked why the singer suddenly had
mountain-climbing safety equipment:

```toml
# Wrong -- "holds her line" is climbing idiom for a route or a rope. H3
# rendered the rope, and the harness and hardware that go with it. Her
# `role` describes no equipment anywhere, and no other chunk in the video
# has any:
shot = "Rock dust and loose stones hiss down the cracked face in a
        spreading sheet, spattering off the ledge just beneath her grip
        as she holds her line above the crumbling rock."

# Right -- the same beat, with nothing left that names an object you don't
# want on screen:
shot = "Rock dust and loose stones hiss down the cracked face in a
        spreading sheet, spattering off the ledge just beneath her grip
        as she keeps climbing, steady, above the crumbling rock."
```

The rule: **if a phrase names a physical object, H3 will render it.** Before
a line ships, ask what a viewer with no access to your intent would see if
every noun were taken at face value. If the answer includes something you did
not mean to put on screen, rewrite the line so it doesn't name that thing —
you rarely lose anything: "keeps climbing, steady" carries the same beat as
"holds her line" without inviting a harness into frame.

This is not a reason to write flat, literal prose everywhere. A shadow, a
worn-down mountain, and a smeared sense of time all survive contact with H3
just fine in the same render this example comes from — none of the following
misfired:

```toml
# Fine -- "shadow" names a real, intended visual; nothing here reads as
# hardware or equipment:
shot = "She climbs on through a slow fall of grey ash as firelight throws
        her shadow sprawling up the rock face beside her."

# Fine -- "worn down... to a hill" is the mountain itself changing over the
# song, which is the actual story, not an idiom smuggling in a second object:
shot = "A band of pre-dawn light lies unchanged along the horizon as she
        lifts her gaze to it, the climb behind her worn down now to little
        more than a bare hill."
```

and `setting = "...where time is smeared: medieval hosts, industrial armies,
and nuclear glow all visible from the same watch-post..."` composes into
*every* prompt in that render with no misfire anywhere, because "smeared" is
describing an actual visual quality of the scene, not standing in for an
unrelated object the way "line" stands in for a rope. The test is not "did I
use a figure of speech" — it's "does this figure of speech happen to be the
name of a physical thing I don't want in frame."

## Say who else is on screen, in `present` (issue #59)

You are told not to name a cast member in the shot line, because the render
composes each character's name, role and appearance into every prompt itself.
That leaves a pronoun as the only way to refer to a second character — and a
pronoun on its own reaches the model with no face attached to it.

`present` is what binds it:

```toml
[[shot]]
chunk_id = 4
# lyric: "There was a time when I hoped your printer"
present = ["Jan"]
shot = "A beige printer sits blinking on the lit sill of a ground-floor
        window right beside the sidewalk, close enough to touch, as she
        carries her box past it with him keeping pace just behind"
```

Every name in `present` gets its role, its appearance and its reference
photograph composed into that chunk's prompt, exactly as the singer's are.
Omit the field when she is alone. Never list whoever is singing — the
alignment already knows them, and naming them twice stages their photo twice.

Measured, and visible in the finished video: of 36 lines in the first
machine-authored plan, 9 referred to a man as "him". On the 7 where he was
not also the singer, nothing bound the pronoun to anybody, and H3 invented a
different man each time. The companion's face changes seven times over one
walk.

## Say whose shot it is, in `subject` (issue #82)

`present` binds a pronoun to a face. It does **not** say who the shot is
about — CLAUDE.md puts it bluntly: *"`present` decides who a pronoun is,
never whose shot this is."*

On a **voiced** chunk that question is already settled: the singer owns the
frame, and three separate measured findings (#58, #59, #60) say the sentence
outranks any field that argues otherwise. So `subject` is refused there —
setting it is an error, not an override.

On an **instrumental** chunk nobody is singing, so `chunk.characters` is
empty and the render falls back to `default_lead_vocalist` and bills *that*
person as "the focus of this shot". When the line is about somebody else,
the prompt then contradicts itself:

```toml
# Wrong -- an instrumental chunk whose sentence is entirely about Jan, with
# nothing telling the render that. The prompt billed Dianne as the focus and
# Jan as a silent bystander, and H3 resolved it by morphing one into the
# other at 3:11:
[[shot]]
chunk_id = 29
# INSTRUMENTAL -- no lyric to sing
present = ["Jan"]
shot = "His boot settles into a hollow already worn into the watch-post
        stone, the rock beneath him scarred smooth from years of the same
        unmoving stance."

# Right -- the same line, with the billing it always implied:
[[shot]]
chunk_id = 29
# INSTRUMENTAL -- no lyric to sing
subject = "Jan"
shot = "His boot settles into a hollow already worn into the watch-post
        stone, the rock beneath him scarred smooth from years of the same
        unmoving stance."
```

`subject` replaces the default focus outright: that member's name, role,
appearance and reference photo become the shot's, and they are no longer
listed as "also in shot, silent". Name at most one person — a shot has one
subject.

Measured on the "Deathless" plan the viewer reviewed: 10 of its 39
instrumental chunks (0, 1, 4, 29, 34, 45, 46, 56, 62, 66) describe only Jan
and the scenery while Dianne is composed as their focus.
`lint_instrumental_focus_mismatch` names them; it warns and never blocks.

## Put each joke where it is legible

A gag needs the room that makes it make sense. An exploding printer in an
intensive care unit is a non sequitur — the joke is an office joke, and it
needs desks and an aisle around it to read as one. The same is true in
reverse: a life-support plug belongs in a hospital, a snow plow belongs on a
street or in a park.

This bit us on a real render. One version of this song's plan put all six gags
inside a single hospital, and the feedback was immediate: *"it has her
wandering into a hospital and staying there… I feel like maybe it couldn't
quite get some of the jokes to land, for that reason — like the exploding
printer."* The video was coherent and the jokes stopped working.

`config.setting` deliberately constrains the **city**, not the venue (issue
#32). That freedom is yours to spend: write the video as a continuous walk
that *moves between* venues, and put each gag where its objects live.

```toml
# One journey, four venues, each joke where it reads:
#   hospital  -> life support plug
#   street    -> transition
#   park      -> snow plow
#   office    -> printer, muted mic, defenestration
```

Two practical notes. Name the venue in the shot line itself rather than
relying on `setting` to imply it — `setting` says *which city*, your line says
*which room*. And when a sequence changes venue, give the change a shot of its
own; a gag that begins in one place and ends in another inside a single
six-second chunk will not read.

## If the lyric names an object, put the object on screen

When the words say "printer", the audience looks for a printer. Nothing forces
you to show it — but that should be a decision, not an oversight.

This one is worth stating because it was missed in a render and nobody noticed
until the video was watched end to end. "The Lucky Ones" sings *"There was a
time when I hoped your printer would explode somehow"* **twice** — at 0:26 and
again at 3:03. The plan staged a printer only for the second. Over the first,
the lyric named a printer while the screen showed a hospital corridor.

**A repeated verse is a plant-and-payoff opportunity at full-song scale.** The
fix is not to duplicate the gag: plant the printer the first time the lyric
names it — jammed, blinking, ignored in the background — and let it explode
the second time, three minutes later. The song hands you the structure for
free; the printer becomes a running joke instead of a single sight gag.

`lint_shots_against_lyrics` warns at load when a chunk's lyric names an object
your plan stages in *other* chunks but not that one. It is quiet about objects
staged in the shots immediately either side, since a three-beat gag is one
visual moment spread over several chunks. It is a warning, never an error —
plenty of good shots do not echo their lyric.

## Checklist

Run down this list before committing a shot plan:

- [ ] Does any single `shot` line ask for both a cause and its effect?
      If yes, split it into plant / contact / consequence across
      consecutive chunks.
- [ ] For every consequence, is there a chunk that plants the object or
      person before it happens?
- [ ] For every consequence, is there a chunk that shows contact — a named
      body part touching a named object — before the state change?
- [ ] For every consequence, does something in frame change physical
      state — not only a person reacting to it?
- [ ] Does each contact/consequence line describe the end state — what's
      true when the shot is over — rather than only the motion that
      supposedly gets there?
- [ ] If the outcome is meant to be blunt, violent, or comic, does the line
      say so in plain physical terms, rather than through a euphemism or a
      stand-in mechanism?
- [ ] Does the shot immediately after a consequence keep it visible
      (typically in the background), rather than moving on as if it never
      happened?
- [ ] Does each shot line say what's in the foreground and what's in the
      background, rather than naming only the subject?
- [ ] Where the beat depends on it, do hands and feet have an explicit
      instruction rather than being left to default?
- [ ] Does every named location fit inside `config.setting`, with nothing
      that contradicts where and when the video is supposed to be
      happening?
- [ ] Does each gag happen somewhere it can read — the printer in an office,
      the plow on a street — rather than wherever the previous shot left off?
- [ ] In every shot that is about something *happening*, is that thing the
      grammatical subject of the sentence, with the performer in a
      subordinate clause?
- [ ] Is the object of every beat staged in the near or mid ground, in frame
      with her — never described as small, distant, or far behind?
- [ ] Does any `consequence` beat with `focus = "action"` land on a chunk
      that carries a lyric? If a nearby instrumental chunk in the same beat
      group is available, prefer it — camera wording alone has not been
      shown to protect that chunk's lip-sync.
- [ ] Where the lyric names a concrete object, is that object on screen, or
      have you decided deliberately that it should not be? If the verse
      repeats, is the first mention a plant and the second a payoff?
- [ ] Does any idiom or figure of speech in the line happen to name a
      physical object you don't want on screen ("holds her line", "draws
      the line")? H3 has no idiom dictionary and will render the noun.
- [ ] Is a voiced chunk's `camera` framed "in profile"? Measured with no
      counter-example on a real render (issue #76) — prefer facing the lens
      more directly, or move the profile framing to an instrumental chunk.
- [ ] On an instrumental chunk whose line is about somebody other than the
      default lead vocalist, is `subject` set to that cast member? `present`
      does not answer this, and without `subject` the render bills the
      config's default singer as the focus of a shot they are not in.
- [ ] Does any **voiced** chunk set `subject`? That is refused, not
      honoured — the singer owns the frame on a chunk they sing.
- [ ] Does every act declared by the concept actually happen, in order, in
      one unbroken run — and does the final act pay off something a
      previous act planted? A plan can satisfy every other check here and
      still be a sequence of events rather than a story.
- [ ] Does the `shot` line avoid restating things the render loop already
      composes — character identity, appearance, the lyric line, whether
      the character is singing?
- [ ] Is every `chunk_id` and `start` value taken from this run's actual
      alignment output (`run_state.json` / the shot-plan authoring pass),
      not guessed? A guessed `start` more than 0.25 s off raises
      `ShotPlanDriftError` at render time.

## What this guide is not

This is issue #26's option (a) only: better authoring, encoded as
convention, with no new code. It does not add a plan → elaboration step, and
it does not call any model — local or otherwise — at authoring or render
time. If a hand-written three-beat gag, elaborated per this guide, still
fails to read on a real render, that's the signal this guide's ceiling has
been reached and the answer is elsewhere (longer shots, first/last-frame
chaining, a bigger model) — see issue #26's "Related" section.
