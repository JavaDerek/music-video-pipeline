# Lyrics file format

This is the normative specification of the lyrics file format. It is the
public interface of this project for anyone who writes a song's lyrics file
without ever touching the Python — if this document and the code disagree,
that is a bug in the code (or in this document), and either way it should be
reported.

The lyrics file is plain text: one lyric line per physical line, with an
optional leading **character tag** that states which cast member(s) are
*audible* on that line and every untagged line after it, until the next tag.
Parsing lives in
[`music_video_maker/lyrics.py`](../music_video_maker/lyrics.py)
(`parse_lyrics_text` / `parse_lyrics`); real examples are in
[`tests/fixtures/lyrics/`](../tests/fixtures/lyrics/).

The format is defined in three levels, each degrading cleanly to the one
below it. A file that only uses level 1 parses exactly as it always has.

| Level | Syntax | Meaning | Status |
|---|---|---|---|
| 1 | `[Name]` / `[Name: Role]` | One voice audible | Implemented |
| 2 | `[Name & Name]` / `[Name & Name: Role]` | Two or more voices audible, same words | Implemented |
| 3 | `[simultaneously] ... [/simultaneously]` | Two or more voices audible, *different* words, same time | Implemented |

One rule applies to every level: **the tag states who is audible, never who
is visible.** Whether a `&` line is shot as a two-shot, a close-up on one
singer, split screen, or an intercut is the shot plan's decision, not the
transcript's. A tag also does not distinguish unison from harmony — both
render identically as "these names are singing this line."

## Plain lines, no tags at all

```
Walking through the empty halls tonight
Nobody's watching, nobody cares
The lights flicker but I don't mind
```

(`tests/fixtures/lyrics/plain.txt`.) Every non-blank line becomes one lyric
line. With no tag ever appearing, every line's active character is
`config.default_lead_vocalist`.

## Level 1: `[Character: Role]` tags

```
[Dianne: Lead]
The lucky ones don't ever have to try
[Marcus: Backup]
We're watching from the wings tonight
[Dianne: Lead]
But I'm the one who taught you how to fly
```

(`tests/fixtures/lyrics/tagged.txt`.) A tag on its own line switches the
active character for every following line (tagged or not) until the next
tag. The tag's role text (`Lead`, `Backup`, …) is accepted syntactically but
**not otherwise used** — the actual role text a chunk's prompt draws from
comes from that character's `role` field in the config's `[cast.<Name>]`
table, not from anything written after the colon in a lyrics tag. A tag can
also share a line with content, e.g. `[Dianne: Lead] The lucky ones...`, in
which case that line's text is tagged and no separate line is consumed.

Bracket-only tags (no trailing content, e.g. a lone `[Marcus: Backup]` line)
switch the active character but produce no lyric line of their own.

## Level 2: `[Character & Character]` tags

```
[Dianne & Marcus]
Singing this together tonight
Every word the same, every voice as one
[Dianne & Marcus: Chorus]
Lord knows when it's people like you
[Dianne]
But this part is mine alone
```

(`tests/fixtures/lyrics/duet.txt`.) `&` joins two or more names into one tag
when they sing the **same words** at the **same time**. It follows exactly
the same rules as a level 1 tag — it switches the active character(s) for
itself and every untagged line that follows, a role slot after a colon is
accepted and unused, and it can share a line with content
(`[Dianne & Marcus: Chorus] Lord knows...`). The only thing that's new is
that "active character" is now a *list*.

**Order is significant.** `[Dianne & Marcus]` produces `characters =
("Dianne", "Marcus")` — in that order. The first name is the *primary*
vocalist: it's what any part of the pipeline that can only reason about one
face (a fallback default, a log line, a future renderer that can't hold two
likenesses) falls back to. Writing `[Marcus & Dianne]` for the same line
would make Marcus primary instead — put the vocalist you'd want foregrounded
if forced to choose first.

Whitespace around `&` is free-form: `[Dianne&Marcus]`,
`[Dianne & Marcus]`, and `[  Dianne   &   Marcus  ]` all parse identically
(`tests/fixtures/lyrics/duet_whitespace.txt`). More than two names is
allowed the same way: `[Dianne & Marcus & Rex]`.

**Every name is validated independently**, exactly as a solo tag is — see
["Unknown characters fail loudly"](#unknown-characters-fail-loudly) below.
`[Dianne & Ghost]` fails just as loudly as `[Ghost]` does, naming `Ghost` as
the unknown one even though `Dianne` was spelled correctly.

A `&` tag states who is **audible**, never who is **visible**. It is not an
instruction to frame two people in the same shot — see the "shot plan"
callout at the top of this document. It also doesn't distinguish unison from
harmony; both are written and rendered the same way.

## Level 3: `[simultaneously]` blocks

Some songs have a section where two vocalists sing **different words at the
same time** — true counterpoint, not a shared line. Level 1 and level 2 both
assume the file, read top to bottom, is one linear transcript; counterpoint
breaks that assumption, because forced alignment (Stage 1) maps audio to
*one* text and cannot be handed two concurrent streams to align at once.

The syntax nests the existing tag grammar inside a block:

```
[simultaneously]
  [Dianne]
  There was a time
  when I hoped
  your printer
  would explode somehow.
  [Marcus]
  I know when it's people like you
  Hating people like me
  I'm the lucky one.
[/simultaneously]
```

(`tests/fixtures/lyrics/simultaneously_block.txt` is the committed fixture.)
Indentation inside the block is cosmetic — every line is whitespace-trimmed
like any other lyrics line — it's written indented purely so a human skimming
the file can see the block's extent at a glance. Both delimiters are
case-insensitive (`[SIMULTANEOUSLY]`, `[/Simultaneously]`).

### The alignment spine rule

**The first sub-block is the alignment spine.** Its lines are the transcript
for that span: they become ordinary lyric lines, in the ordinary line index
space, and they are the only text in the block that forced alignment ever
sees. Every later sub-block becomes a **counterpoint stream**, carried
alongside the transcript and never inside it.

Each counterpoint stream then **inherits the spine's span**: the aligned
start of the spine block's first word to the aligned end of its last, with
the stream's own words distributed **proportionally** across that span (every
word gets an equal share; each written line becomes one segment covering its
own words). That is a deliberate approximation. The aligner never heard the
non-spine voice on its own, so no lip-sync accuracy claim is made about it —
an honest cheap rule beats an invented precise-looking one.

**Concurrent streams are not paired line by line, and must not be written as
if they were.** In "The Lucky Ones", Dianne's block is four lines and Jan's
counterpoint under it is three. Only the *span* is shared. Put the front
voice — the one that is more audible, the one you would transcribe if you
could only keep one — first, because that is the voice whose timing is real.

### What you get downstream

- Spine lines: ordinary `LyricLine`s → `AlignedSegment`s, exactly as at
  levels 1 and 2. Nothing about the timeline changes.
- Counterpoint: `LyricsDocument.counterpoint`, a tuple of `CounterpointStream`
  (`characters`, `texts`, `spine_line_indices`, `block_index`,
  `stream_index`), which Stage 1 turns into `ConcurrentSegment`s on a
  `CounterpointAlignmentResult`. Those are kept in their own tuple, never
  merged into `result.segments`: they overlap the spine by design, so merging
  would double-count the voiced duration and make the feature look like an
  alignment defect.
- The alignment-quality summary counts them separately (`… 27 segment(s) (+3
  counterpoint segment(s)) …`) and checks them on their own terms. A stream
  whose words cannot plausibly fit its spine span is reported as a warning
  naming the lyrics file as the likely cause — but never as a blocking
  finding, because those timings were derived by arithmetic rather than
  measured against audio.

### Rules the parser enforces

Every one of these fails at parse time — before alignment, before any GPU
time — with a `LyricsError` that names the problem:

| Written | Result |
|---|---|
| A block that is never closed | error naming the missing `[/simultaneously]` |
| `[/simultaneously]` with no open block | error |
| A block inside a block | error: blocks cannot nest |
| A lyric line inside a block before any `[Name]` tag | error naming the orphan line |
| A block with only one sub-block | error: use a plain `[Name]` tag for one voice |
| A sub-block with no lyric lines under it | error naming that sub-block |
| Content on the same line as `[simultaneously]` or `[/simultaneously]` | error: delimiters go on their own line |
| An unknown character inside a block | the usual unknown-character error |

A sub-block tag may be a level 2 tag: `[Marcus & Rex]` inside a block means
those two sing that stream's words in unison, underneath the spine.

**Inheritance across a block.** A `[simultaneously]` block does not change
the active character for what follows it: the tag in force *before* the block
is restored afterwards. A block has no single "last voice", so inheriting one
of the concurrent voices would silently privilege one of them. Tag the lines
after a block explicitly if they belong to somebody else.

## The motivating bug

This isn't a hypothetical. The first full render of "The Lucky Ones" carried
a *single* `[Dianne: Lead]` tag at line 1 of the lyrics file and no tag after
it — even though the song has Dianne singing some verses, Jan singing
others, both singing the same words together in places, and a closing
section where they sing *different* words at once. With only one tag in the
whole file, level 1's inheritance rule (every untagged line keeps the
active character) meant Dianne's face rendered on every single line,
including all of Jan's. Nothing in the parser was wrong — the *lyrics file*
was incomplete, and there was no way for a reader to know that without
listening to the song. Level 2 exists so "two people, same words" can be
written down truthfully instead of approximated with one tag; level 3 exists
so "two people, different words" can be too. That second one turned out not
to be a nicety: with the counterpoint unwritable, the audio contained a fifth
"I'm the lucky one" that the transcript could not account for, and forced
alignment dumped the surplus into the instrumental outro — a *correctness*
bug in the timeline, not just a missing face.

## Character inheritance and mixing

```
[Dianne: Lead]
Waking up to another gray morning
Coffee's cold but I drink it anyway
[Marcus: Backup]
She never says a word about it
Just stares out at the rain
Nothing ever seems to change
[Dianne: Lead]
And still I find a reason to stay
```

(`tests/fixtures/lyrics/mixed_inherit.txt`.) Untagged lines inherit whichever
character (or, for a level 2 tag, character **tuple**) was most recently
tagged — here, the two lines after `[Dianne: Lead]` are both Dianne's, the
three lines after `[Marcus: Backup]` are all Marcus's, and so on. The same
rule applies to `&` tags: every untagged line after `[Dianne & Marcus]`
stays `("Dianne", "Marcus")` until the next tag switches it. Before any tag
has appeared in the file, untagged lines fall back to a single-element tuple
of `config.default_lead_vocalist`.

## Blank lines and instrumental passages

```
[Dianne: Lead]
Count the days until the sun comes back


(long instrumental intro)


Nothing but the echo of a drum

[Marcus: Backup]
Give the silence room to breathe


(extended instrumental bridge, guitar solo)
```

(`tests/fixtures/lyrics/gaps.txt`.) Blank lines are skipped entirely — they
neither switch the active character nor produce a lyric line. A bracketed
*parenthetical* instrumental marker like `(long instrumental intro)` is
**not** a character tag (character tags use square brackets, `[...]`) — it's
ordinary line content, parsed the same as any other line, carrying whatever
character is currently active. Forced alignment (Stage 1) is expected to
time markers like this against the instrumental audio itself rather than
vocals; `expand_prompt()` (Stage 2b) also has a dedicated fallback clause for
a chunk with no lyric text at all (an empty merged/split chunk), independent
of whether the source line literally said "(instrumental)".

## Tags never reach forced alignment

This is the load-bearing rule: **`LyricLine.text` is always tag-stripped**
before Stage 1 ever sees it. `parse_lyrics`/`parse_lyrics_text` strip the
`[Name: Role]` prefix and hand only the remaining content to alignment — a
raw tag surviving into `stable-ts`'s `model.align()` input would pollute
word-level timestamps, since the aligner would try to force-map audio to
literal bracket/colon tokens that were never sung. The untouched original
line (tag included) survives separately as `LyricLine.raw`, for diagnostics
only — never fed to alignment or into a prompt.

## Unknown characters fail loudly

```
[Dianne: Lead]
We started this together, side by side
[Ghost: Mystery Vocal]
A voice no one wrote down, no one hired
```

(`tests/fixtures/lyrics/unknown_character.txt`.) A tag naming a character
absent from the config's `cast` dictionary raises `LyricsError` immediately,
naming the unknown character and the known cast — a typo'd or stale tag must
never silently fall back to the lead vocalist, since that would put the
wrong face in frame with no error to catch it. This validation happens
**before parsing produces any lyric line**, so it fails at config load, long
before any GPU time is spent on a render.

The same check applies to every name in a level 2 tag independently —
`[Dianne & Ghost]` (`tests/fixtures/lyrics/unknown_character_duet.txt`) fails
exactly the same way and names `Ghost`, even though `Dianne` is a real cast
member. A correctly-spelled first name in a `&` tag provides no cover for a
typo'd second one.

The cast validator has a useful property that made level 3 safe to add: an
old parser meeting new syntax it doesn't understand reads a bracketed word it
doesn't recognise as an unrecognised *character name* and fails — new syntax
degrades to a clear error, never a silent mis-parse. So a file using
`[simultaneously]` blocks, opened by a copy of this project from before level
3 landed, stops at config load naming `simultaneously` as an unknown
character instead of quietly rendering the wrong thing. Conversely,
`simultaneously` and `/simultaneously` are reserved words in the tag
namespace here: a cast member cannot be named either.

## How the active character reaches the rest of the pipeline

`parse_lyrics` returns a `LyricsDocument`: a plain `tuple` of `LyricLine`
(so every caller that iterates, indexes or `len()`s it is unaffected) that
additionally carries `.counterpoint`. `align()` reads that off the object it
is already handed, so a level-3 file cannot lose its counterpoint by passing
through a caller written before level 3 existed. One sharp edge: *slicing*
the document (`doc[:3]`) yields a plain tuple and drops `.counterpoint` —
correct, since a subset of the transcript no longer contains the spine lines
a stream refers to, but silent.

`LyricLine.characters` (a tuple, in written order — first is primary) →
Stage 1's `AlignedSegment.characters` → Stage 2a's `AudioChunk.characters`.
Every one of those types also exposes a singular `.character` property
(the first name, or `None` if the tuple is empty) as a migration seam: any
stage still written for one singer keeps working unchanged against a level 1
or level 2 file — reading only `.character` sees exactly the primary
vocalist, since level 2 is level 1 with more than one name in the tuple.
Whatever a given stage does with the rest of the tuple (composing multiple
cast roles into a prompt, staging a second reference image, etc.) is that
stage's own concern and documented where that stage lives, not here. (This
project has been bitten once already by mixing `LyricLine` index space with
`AlignedSegment` index space — see `prompting.py`'s module docstring — so
those two are deliberately never cross-referenced downstream of parsing.)

Counterpoint keeps that discipline. `CounterpointStream.spine_line_indices`
is **`LyricLine` index space** and nothing else; `ConcurrentSegment.index`
and `ConcurrentSegment.spine_segment_indices` are **`AlignedSegment` index
space**, with concurrent segments numbered *after* the spine's segments so a
bare segment index still identifies exactly one segment across both tuples. A
counterpoint line never gets a `LyricLine` index at all — that space belongs
to the transcript that gets aligned, and putting concurrent text into it is
precisely the confusion this note exists to prevent.

## Summary: what happens on error

| Situation | Result |
|---|---|
| File uses only level 1 tags | Parses exactly as before |
| File uses level 2 (`&`) tags, all names known | Parses; `characters` has 2+ entries in written order |
| Any tag names a character absent from `config.cast` | `LyricsError` at parse time, naming the unknown character and the full known cast |
| File contains a well-formed `[simultaneously]` block | Parses; the spine becomes lyric lines, the other sub-blocks become `counterpoint` streams |
| A `[simultaneously]` block is malformed (unclosed, nested, one voice, empty sub-block, stray close, content on a delimiter line) | `LyricsError` at parse time, naming which rule was broken |
| Lyrics file is missing/unreadable | `OSError`, logged before re-raising |

In every failure case, the error surfaces at **config load / parse time** —
before forced alignment, before any asset staging, before any ComfyUI
request. No render time or GPU custody is ever spent on a lyrics file that
was going to fail anyway.
