"""Authored per-chunk narrative direction ("the shot plan").

``config.narrative_concept`` is a single global string, so without this every
chunk in a run is prompted with the same scene. That gives a video with no
progression -- and if the concept names several settings, H3 marches through
all of them inside every chunk, producing an identical montage forty times
over rather than a story.

A shot plan supplies one authored line per chunk instead, so a gag planted in
shot 4 can pay off in shot 12 and the instrumental breaks can carry the parts
of the story nobody sings.

**The planning is an authoring step, not a runtime one.** A plan is written
once -- by a human, or by a model in conversation with one -- and committed
as data. Nothing in this module calls an LLM or any external service, so the
render loop stays exactly what ``prompting.py``'s docstring promises: pure,
deterministic string composition. That also keeps ``--resume`` honest, since
re-rendering chunk 12 tomorrow composes byte-identical text to today.

Drift is the failure mode worth guarding. A plan is authored against one
alignment; re-running Stage 1 with edited lyrics or a different model
reshapes segments and renumbers chunks, at which point shot 12's direction
would land on some other chunk's audio. The result still looks deliberate,
which is precisely what makes it dangerous. Every entry therefore records the
chunk start time it was authored against, and a mismatch beyond
:data:`START_TOLERANCE_SECONDS` raises :class:`ShotPlanDriftError` rather
than warning.

**Geography lint against ``setting`` (issue #32).** A plan can still name
specific locations, but they must be consistent with ``config.setting`` --
"Central Park" alongside a London setting is almost certainly an authoring
slip, and one that cost real GPU time on a real run. ``load_shot_plan``
therefore takes the run's ``setting`` (and cast member names, so a performer
who happens to share a name with a landmark is never flagged) and warns, once
per offending entry, when a shot line names a well-known landmark whose city
is not mentioned anywhere in ``setting``. This is deliberately a small,
curated list of unambiguous landmarks rather than any attempt at general
place-name extraction -- false positives on ordinary prose are worse than
missed detections, since this is a load-time warning, never a load failure.
Silent when ``setting`` is ``None`` (nothing to check against) and silent
whenever no landmark word is matched.

**Shot-vs-lyric lint (issue #37).** A separate check, run by the orchestrator
once chunks exist: when a chunk's lyric names an object this plan stages in
*other* chunks but not this one, warn. "The Lucky Ones" sings "your printer
would explode somehow" twice and staged a printer only for the second, so over
the first the lyric named a printer while the video showed a hospital. See
:func:`lint_shots_against_lyrics` for why the "staged elsewhere" condition is
what keeps this precise rather than noisy.

**Consequence-vs-focus lint (issue #41).** ``focus = "action"`` (issue #26)
exists so a shot whose whole job is a consequence -- a printer erupting, a
bank of monitors flatlining -- can hand the shot's subject to that
consequence instead of the performer who already walked away. Nothing
enforced that an author actually reaches for it: a plan can describe the
world changing while still implicitly leaving the performer as the subject,
which is the same shape of bug as the performer singing through every
instrumental. ``load_shot_plan`` therefore also warns, once per offending
entry, when a shot's text matches a small curated list of unambiguous
physical/device words -- explosions, breakage, failing lights, malfunctioning
machinery, spills -- while the entry still leaves ``focus`` at its default.
Deliberately narrow for the same reason as the geography lint above: this
fires on prose a human wrote deliberately, so a false positive is worse than
a miss. Words that are just as at home describing a performer's own body or
mood ("collapses", "breaks down", "sparks fly") are left out on purpose --
see :data:`_CONSEQUENCE_KEYWORDS` for the exact list and why each entry
earned its place.

**Shot length (issue #27).** Every shot used to be 5-8 s, not because anyone
chose that but because chunk boundaries fell out of vocal-segment timing plus
a fixed ``max_chunk_seconds`` -- a scene change every six seconds for the
whole video, which is the most machine-generated thing about the output. An
entry may therefore also carry ``length_seconds``, and
:func:`shot_length_requests` projects those into anchored
:class:`ShotLength` requests that ``slicing.slice_audio`` honours.

Two properties of that design are load-bearing:

* **The anchor is the entry's ``start``, not its ``chunk_id``.** A time in the
  song is stable across a re-slice; a chunk number is not, and honouring one
  length renumbers every chunk after it. Anchoring on ``start`` is what lets a
  plan ask for several long takes at once without each one invalidating the
  next.
* **Length is a request, not a promise.** The frame grid, H3's trained range
  and the vocal timeline all outrank it, and ``slicing`` says so in the log
  when it clamps. Nothing here invents timing.

Anything longer than :data:`MEASURED_MAX_FRAMES` is warned about at load
time: it is inside H3's trained range but outside everything ever actually
rendered on this card.

**Blank shot lines (issue #52).** ``shot = ""`` (or whitespace-only) is
accepted at load time rather than rejected, and :func:`resolve_shot` treats
it exactly like a chunk with no entry at all -- warn, fall back to
``config.narrative_concept``. This is what makes :func:`--prepare
<music_video_maker.cli.prepare_shot_plan>`'s skeleton loadable: every entry
it writes has real ``chunk_id``/``start`` anchors but an empty ``shot`` for
the author to fill in, and a plan that only half-exists is the normal state
while writing one, not a malformed file. What still fails to load is the
``shot`` key being *absent* entirely -- that is a different failure (a
hand-edited or hand-written file missing a required field), not an unfilled
skeleton.

**Distant-staging lint (issue #58).** A first GPU render of a machine-authored
plan found that subject position is necessary but not sufficient: a printer
staged as the grammatical subject of its plant and payoff beats still failed
to render, because both lines also described it as small and far away
("small against the tower far behind her", "far behind on the street"). A
third beat in the same slice, staged near and large, rendered correctly.
``load_shot_plan`` therefore also warns, once per offending entry, when a
shot's text matches a small curated list of distance/smallness words -- see
:data:`_DISTANT_STAGING_KEYWORDS` and
``docs/shot-writing-guide.md``'s "Stage the object near, not far".

**Camera-vs-lip-sync lint (issue #58).** A separate check, run by the
orchestrator once chunks exist, the same shape as the shot-vs-lyric lint
above. Its primary signal is structural, not prose: ``focus = "action"`` on
a voiced chunk, because three independent real re-renders of the same chunk
-- two different ``camera`` rewrites, one ``shot``-line rewrite that stopped
her mid-stride and said so explicitly -- all still turned her away from the
lens by roughly the shot's midpoint. Wording did not fix it in any of the
three attempts; ``focus = "action"`` was the one thing held constant. A
narrower, older signal (an explicit "back to camera"-style ``camera`` value)
is also still checked. See :func:`lint_camera_face_away_on_voiced_chunks`.

**Camera direction (issue #53).** An entry may also carry ``camera``, this
shot's own framing and movement (e.g. ``"tracking backwards ahead of her"``).
It used to have nowhere to live except free text inside ``shot`` alongside
everything else -- which meant whoever wrote the line also decided where in
the sentence it landed, and H3 renders the grammatical subject of a shot line
(see ``docs/shot-writing-guide.md``'s "Make the effect the grammatical
subject"). ``prompting.py`` composes ``camera`` as a **trailing clause** on
the concept/shot sentence, never as its own sentence, so it can never compete
for the subject slot -- see ``prompting._apply_camera_clause``. Resolved by
:func:`resolve_camera`, independently of whether ``shot`` itself is filled
in: camera direction is just as meaningful paired with the global
``narrative_concept`` fallback as with an authored shot line.

**Referent-based companion lint (issue #72).** ``present`` (issue #59) and
the #64 lint above both ask a question about the *prompt*: does some cast
member's name appear somewhere in it? Chunk 65 of a real "Deathless" render
passed that question -- ``present = ["Dianne"]`` -- and still invented a
stranger, because Dianne was already the singer: naming her again binds
nothing that was not already bound, and the shot's ``he``/``his`` pointed at
Jan, who was in no field at all. The question that actually matters is about
the *sentence*: does this pronoun have a bound referent, distinct from
whoever is already accounted for? :func:`lint_unbound_companion_referent`
answers it structurally rather than by gender (no cast field carries gender
-- see the function's own docstring for the exact field this would need and
why the shipped version does not depend on one): it unions ``chunk.characters``
(who is actually singing; empty on an instrumental chunk, matching
``prompting._resolve_active_members``'s own fallback rule) with
``entry.present`` into one "who is already bound" set, and warns only when
that set names exactly one person while the shot's own prose insists on a
*second* one ("a few paces off", "between them", "just behind him"/"her").
Two or more bound names are left alone -- ambiguous, perhaps, but nobody is
missing an identity, appearance or reference photo. Measured against the
real 80-chunk "Deathless" plan: 2 of 80 chunks fire (chunk 65 itself, plus
chunk 66's mirror image -- Jan's own "he" left unbound while the approaching
"ash-crusted figure...just behind him" is Dianne, correctly bound). See the
function's docstring for the phrases measured and discarded, most notably
"beside her"/"beside him" -- suggested by the issue itself, and dropped after
it fired on a shadow, a roof beam and a rock face "beside her" with nobody
there at all (the very failure mode issue #64's own test suite already
documents for that phrase, one section up).

**Role-prohibition-vs-shot lint (issue #73).** A ``role`` written as a
prohibition -- ``"...never holding anything"`` -- does not work: it was in
force for an entire real render and a bass guitar still appeared in the same
two hands twice, because a diffusion prompt has no way to condition on the
*absence* of a token, only the presence of "holding" and "anything"
(``config._warn_if_prohibition`` now warns about this at load time).
:func:`lint_role_prohibition_contradiction` is the other half: it catches the
render actually asserting the forbidden thing. Chunk 72's shot line has
Dianne "close his fingers over" a needle she has just pressed into Jan's
palm -- text that never says "hold" at all, which is exactly why this lint
is honest about being narrow. It extracts the word right after
``never``/``no``/``without`` in a present cast member's ``role`` and checks
for it (or, for the specific and twice-measured "holding" concept only, a
small curated set of physical-contact synonyms -- grip, grasp, clutch,
cradle, "close(s) ... fingers") in that member's shot lines. Measured on the
same 80-chunk plan restricted to ``shot`` text only: the plain verb "hold" in
its many idiomatic and cinematographic senses ("holds his stance", "holding
steady" as a *camera* direction) produced double-digit false positives, and
even the exact word "holding" alone still caught two camera-direction uses
before ``camera`` was dropped from what this lint reads. What survived that
cutting -- the curated contact-verb pattern, ``shot`` text only -- fires
on exactly one chunk in the whole plan: chunk 72. This is a literal (or
near-literal, for the one concept measured twice) collision check, not
semantic understanding of what a `role` forbids; a differently-worded
violation of a differently-worded prohibition will still slip through, which
is why issue #73's own conclusion -- positive phrasing, plus the render-side
``avoid`` list -- is the durable fix and this lint is a backstop, not it.

**Voiced-framing keyword re-score (issue #76).** :func:`lint_voiced_framing`'s
keyword sets (:data:`_GAZE_AWAY_KEYWORDS`, :data:`_BEHIND_CAMERA_KEYWORDS`,
what was :data:`_FOOT_LEVEL_KEYWORDS`) were all scored mid-render, on a
partial "Deathless" corpus -- issue #60's own lesson, arriving one level up.
The finished 80-chunk render (41 voiced chunks, corpus mean face presence
53.3%, ``docs/deathless-render-corpus.md``) does not support them as
shipped: the gaze set separated at 0.72x, not the 3.5x it was justified at;
the foot-level set at 0.99x, statistically indistinguishable from noise; and
``"in profile"``, deliberately left unshipped at n=3, turned out to be the
one predictor with no counter-example across all 5 occurrences (0.41x).
Three changes followed, each backed by the full-corpus numbers rather than a
fresh guess: ``"in profile"`` now ships; ``_FOOT_LEVEL_KEYWORDS`` is retired
(the observation behind it was real, the keyword list did not generalise);
and ``_GAZE_AWAY_KEYWORDS`` is cut from 9 words to the 6 whose one measured
occurrence actually landed below the corpus mean, with every excluded
candidate recorded next to the chunk that falsified or never tested it. See
the constants themselves for the full per-word accounting, and
``docs/deathless-render-corpus.md`` for the numbers behind every one of
these calls, including the "does the camera clause name the face" hypothesis
tested and rejected at n=41 -- it runs backward from the hypothesis, not
merely absent.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from music_video_maker.contracts import H3_FRAME_GRID, AudioChunk, CastMember

logger = logging.getLogger(__name__)

MEASURED_MAX_FRAMES = 141
"""The longest chunk ever actually rendered on doris's 4090, at any resolution.

H3's *trained* range reaches 362 frames (15.083 s) and the cost model says a
long take is effectively free -- total render time tracks total latent volume,
which the song's length fixes, so longer shots just mean fewer of them. What
is unknown is VRAM: temporal VAE decode memory scales non-linearly with frame
count, and 362 frames is ~2.6x anything attempted here.

That is a reason to *name* the risk, not to gate on it: an over-committed card
on this box does not raise CUDA OOM, it goes silent and wedges the host past
SIGKILL (issues #23, #24), so the first long take is worth watching. Step up
gradually rather than jumping to the ceiling."""

_LANDMARK_LOCALES: dict[str, str] = {
    # A small, deliberately curated set of unambiguous, single-city
    # landmarks -- see the module docstring for why this stays narrow rather
    # than growing into general place-name extraction. Keys are matched as
    # whole words/phrases, case-insensitively, against shot text.
    "central park": "new york",
    "manhattan": "new york",
    "times square": "new york",
    "brooklyn bridge": "new york",
    "statue of liberty": "new york",
    "golden gate bridge": "san francisco",
    "hollywood sign": "los angeles",
    "eiffel tower": "paris",
    "big ben": "london",
    "tower bridge": "london",
    "buckingham palace": "london",
    "sydney opera house": "sydney",
    "colosseum": "rome",
    # An entry earns its place only if it names ONE place with no common
    # second reading. "Notre Dame" was dropped for failing that bar -- it is
    # equally a US university, so it would have warned about a perfectly
    # coherent American shot. A false positive here is worse than a miss:
    # this fires at load time on prose a human wrote deliberately.
}

START_TOLERANCE_SECONDS = 0.25
"""How far a chunk's start may sit from the value the plan was authored
against before it counts as drift rather than float arithmetic.

Chunk starts accumulate across the timeline as floats, so exact equality is
not a fair test. A quarter second is far wider than that accumulation and far
narrower than any real re-alignment, which moves boundaries by seconds."""


class ShotPlanError(ValueError):
    """Raised when a shot plan file is missing, malformed, or ambiguous."""


class ShotPlanDriftError(ShotPlanError):
    """Raised when a chunk's start no longer matches what the plan recorded.

    Means the alignment changed after the plan was authored, so the plan's
    chunk numbering no longer refers to the same moments of the song.
    """


@dataclass(frozen=True)
class ShotLength:
    """One editorial shot-length request, anchored in *song time* (issue #27).

    The unit ``slicing`` consumes. Deliberately not a shot plan: the slicer
    must not need to know about narrative direction, TOML, or drift checking
    to retile a timeline, and a caller with no plan at all (a test, an
    experiment stepping frame counts up gradually) can build these directly.
    """

    start: float
    """Where in the master track this shot begins, in seconds. The anchor.

    Matched against the start a chunk will actually *have*, which is why it is
    a time and not a chunk id -- see the module docstring."""

    length_seconds: float
    """How long the author wants this shot to run. Requested, not promised:
    ``slicing`` quantizes it onto H3's frame grid and clamps it into the
    trained range, saying so in the log when it does."""

    source_chunk_id: int | None = None
    """Which plan entry asked for this, for log messages only.

    **Never an index into anything.** Chunk ids are assigned by slicing after
    this request has already changed how many chunks there are, so using this
    to look a chunk up would be the same index-space mixup that has bitten
    ``AlignedSegment.index`` vs ``chunk_id`` before."""


@dataclass(frozen=True)
class ShotPlanEntry:
    """One authored chunk's direction."""

    chunk_id: int
    start: float
    """The chunk start this entry was authored against -- the drift anchor."""
    shot: str
    subject_is_focus: bool = True
    """Whether the active cast member is what this shot is *about* (issue #26).

    ``True`` (the default, and the historical behaviour) composes "X is the
    focus of this shot" into the prompt. ``focus = "action"`` in the TOML sets
    this ``False``, for the shots whose whole job is a consequence the subject
    has already walked away from -- a flatlining monitor, a burning printer.

    Those shots previously carried a contradiction: the shot line said the
    consequence was the beat while the character clause said the performer was
    the focus, and across four renders the character-attached instruction won.
    That is the same shape as the bug that made a performer sing through every
    instrumental, and the same remedy -- per-chunk state must not be asserted
    globally."""

    present: tuple[str, ...] = ()
    """Cast members visibly in this shot who are not necessarily singing it
    (issue #59).

    ``chunk.characters`` answers "who is singing", which the forced alignment
    knows. Nothing knew "who is on screen", and the two were silently the
    same field. A real 36-chunk render made the gap visible: nine shot lines
    referred to a second character as "him" -- because
    ``authoring.prompts.PROSE_PREAMBLE`` forbids naming a cast member, so a
    pronoun is the only way to reference one -- and on the seven of those
    where he was not also the singer, no role, no appearance and no reference
    photo went into the prompt at all. H3 invented a different man each time,
    and the video has a companion whose face changes seven times.

    Empty (the default) is "nobody else in shot", which composes
    byte-identically to the pre-#59 prompt. A name here that is not in the
    cast raises :class:`prompting.UnknownCastMemberError`, the same rule a
    singing name follows and for the same reason: a typo must never quietly
    become "just the people we recognised". A name that is *also* singing
    this chunk is not composed twice.

    This is the "a property that must hold needs its own field" pattern
    again. Writing "Jan walks beside her" into ``shot`` would not work: the
    shot line is free text that nothing resolves against the cast, so it
    would stage no photo, and it would fight rule 2 of the prose preamble."""

    subject: str | None = None
    """Whose shot this is (issue #82) -- a single cast name, or ``None``.

    Not :attr:`present`: that field answers "is this cast member visibly on
    screen" (and, via #72's referent lint, "does a pronoun bind to them"),
    but it never changes who the render composes as *"is the focus of this
    shot"* -- that stays whichever member is singing, or
    ``config.default_lead_vocalist`` on an instrumental chunk, regardless of
    what ``present`` says. On "Deathless" chunk 29 (instrumental, shot line
    entirely about Jan) ``present = ["Jan"]`` staged his name, role,
    appearance and photo -- and the prompt still said Dianne was the focus.
    H3 morphed one into the other at 3:11. ``subject`` is the field that
    actually reassigns focus.

    Not :attr:`subject_is_focus` either, though the names are easy to
    confuse: ``subject_is_focus`` (issue #26) is a boolean asking whether the
    active member is what the sentence is *about* (vs. a consequence they
    have walked away from, e.g. a burning printer) -- it never says *who*
    the active member is. ``subject`` says who the focus member IS. The two
    compose independently: which member is the focus, and whether that
    member is the sentence's subject, are different questions.

    Legal only on an **instrumental** chunk. A voiced chunk's singer owns the
    frame -- three separate measured findings (#58, #59, #60) say the
    sentence outranks any field that argues otherwise -- so `subject` there
    is refused by :func:`lint_subject_on_voiced_chunk`, never silently
    honoured: a field that could override the singer would be a new way to
    reintroduce the very desync it exists to prevent.

    ``None`` (the default) is "not authored", and composes exactly like the
    pre-#82 prompt: the instrumental fallback to ``default_lead_vocalist``
    is unchanged when nobody has said otherwise."""

    length_seconds: float | None = None
    """How long this shot should run, in seconds (issue #27).

    ``None`` (the default) means "however long the chunk timeline made it" --
    i.e. today's behaviour, which is where the uniform 5-8 s cadence comes
    from. Setting it makes shot length an editorial choice: long takes for
    calm passages and instrumental breaks, short ones for density and comic
    timing.

    It is a *request*. The frame grid, H3's trained range (5.167-15.083 s) and
    the vocal timeline all outrank it -- see :func:`shot_length_requests` and
    ``slicing.slice_audio``."""

    camera: str | None = None
    """This shot's own framing and movement (issue #53), e.g. ``"tracking
    backwards ahead of her"`` or ``"pushing in slowly on her hands"``.

    Composed by ``prompting.py`` as a **trailing clause** on the concept/shot
    sentence -- never as its own sentence -- so it cannot take the
    grammatical subject slot away from whatever the shot is actually about.
    Write it the way a shot-line's own trailing camera phrase already reads
    ("camera tracking backwards ahead of her"), assuming "camera" as the
    implied subject; ``prompting.py`` supplies that word.

    ``None`` (the default) composes nothing, exactly like every other
    optional field here. Independent of ``shot``: a blank ``shot`` still lets
    ``camera`` apply to whatever concept text this chunk falls back to."""

    location: str | None = None
    """Where the active cast member is inside the run's ``setting`` at this
    moment (issue #78), e.g. ``"switchback"`` or ``"mill"``.

    ``setting`` (issue #32) anchors the world's *contents* -- one geography
    for the whole video -- but nothing said where each character is inside
    that world at a given moment, and a viewer's first note on a real render
    was exactly this gap: "Diane climbs a hill where Jan is, cuts to her on
    the top of a different hill where Jan isn't, cuts to a windmill tower
    where they both are." A shot line may not reference another shot (issue
    #61: each chunk renders from its own prompt, for a model that has never
    seen any other), so nothing in the render itself can hold continuity of
    *position* the way ``setting`` holds continuity of *place*. This is that
    field's per-moment counterpart.

    Authored by the beats stage (``authoring/beats.py``) from a small closed
    vocabulary the concept stage defines for the song -- the same shape as
    ``beat_role``/``beat_group``: checkable before a word of prose exists,
    never guessed from finished text. A generated value is validated against
    that vocabulary at generation time and anything outside it is rejected,
    the same "a generated anchor is never trusted" rule ``chunk_id`` follows.
    Re-emitted here from the frozen timeline exactly like every other
    beat-derived field ``render_plan_toml`` writes.

    Purely a checkable field, not a rendered one: nothing in this module or
    ``prompting.py`` composes ``location`` into a prompt. It exists for
    :func:`~music_video_maker.shot_plan.lint_present_location_mismatch` and
    the load-time landmark-contradiction lint below to check consistency
    against, the same way ``beat_role`` exists to be checked rather than
    spoken. ``None`` (the default) is "not authored", never a fabricated
    value -- true of every pre-#78 entry and every hand-written plan that
    does not use the field."""


def load_shot_plan(
    path: str | Path,
    setting: str | None = None,
    cast_names: Iterable[str] = (),
) -> dict[int, ShotPlanEntry]:
    """Parse a shot plan TOML file into ``{chunk_id: ShotPlanEntry}``.

    ``setting`` and ``cast_names`` are optional and used only for the
    geography lint (issue #32): when ``setting`` is given, each entry's
    ``shot`` text is checked against a small curated list of well-known
    landmarks, and a mismatch (e.g. "Central Park" under a London setting)
    logs a warning -- never an error; a false positive here must not be able
    to block a run. ``cast_names`` excludes any cast member whose name
    happens to match a landmark word (e.g. a performer literally named
    "Manhattan") from being flagged. Omitting ``setting`` (the default)
    disables the lint entirely."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except FileNotFoundError:
        logger.error("Shot plan file does not exist: %s", path)
        raise ShotPlanError(f"shot plan file does not exist: {path}") from None
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.exception("Failed to read shot plan %s", path)
        raise ShotPlanError(f"could not read shot plan {path}: {exc}") from exc

    raw_entries = payload.get("shot")
    if not isinstance(raw_entries, list) or not raw_entries:
        logger.error("Shot plan %s has no [[shot]] entries", path)
        raise ShotPlanError(f"shot plan {path} contains no [[shot]] entries")

    plan: dict[int, ShotPlanEntry] = {}
    for index, raw in enumerate(raw_entries):
        entry = _parse_entry(raw, index, path, cast_names)
        if entry.chunk_id in plan:
            logger.error(
                "Shot plan %s defines chunk_id=%d more than once", path, entry.chunk_id
            )
            raise ShotPlanError(
                f"shot plan {path} defines chunk_id={entry.chunk_id} more than once; "
                "one of the two directions would be silently discarded"
            )
        plan[entry.chunk_id] = entry

    logger.info("Loaded shot plan from %s: %d chunk(s) directed", path, len(plan))
    _lint_setting_consistency(setting, plan, cast_names, path)
    _lint_landmark_position_contradiction(setting, plan, path)
    _lint_consequence_focus(plan, path)
    _lint_shot_lengths(plan, path)
    _lint_camera_double_composition(plan, path)
    _lint_distant_staging(plan, path)
    _lint_anaphora(plan, path)
    _lint_implied_companion_without_present(plan, path)
    return plan


def _lint_camera_double_composition(plan: Mapping[int, ShotPlanEntry], path: Path) -> None:
    """Warn (never raise) when a shot line carries its own camera language
    *and* the entry sets ``camera`` (issue #54 design section 6).

    ``prompting._apply_camera_clause`` composes ``camera`` as a trailing
    ``", camera <direction>"`` clause on the concept sentence. A ``shot`` line
    that already ends in a camera phrase -- the pre-#53 style, and still the
    style of every worked example in ``docs/shot-writing-guide.md`` -- then
    composes twice: "..., camera tracking backwards ahead of her, camera
    pushing in slowly on her hands". Issue #53 opened this door by giving
    camera a field of its own while ``shot`` stayed free text, and nothing
    checked that an author had *moved* the direction rather than duplicated
    it.

    Belongs here rather than in ``authoring/`` for two reasons the design
    states: a hand-written plan can make exactly the same mistake, and a
    generator should be checked by the same rules as a human.
    """
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        if entry.camera is None or not re.search(r"\bcameras?\b", entry.shot.lower()):
            continue
        logger.warning(
            "Shot plan %s: chunk_id=%d sets camera=%r while its shot line also says "
            "'camera' (%r), so the direction composes twice -- '..., camera <shot's own "
            "phrase>, camera %s'. Move the camera language out of the shot line and "
            "leave it in the camera field, which controls where in the sentence it "
            "lands (issue #53).",
            path,
            chunk_id,
            entry.camera,
            entry.shot,
            entry.camera,
        )


def _lint_setting_consistency(
    setting: str | None,
    plan: Mapping[int, ShotPlanEntry],
    cast_names: Iterable[str],
    path: Path,
) -> None:
    """Warn (never raise) when a shot names a landmark whose city is absent
    from ``setting``. See the module docstring for the design rationale."""
    if not setting:
        return

    setting_lower = setting.lower()
    cast_lower = {name.strip().lower() for name in cast_names if name and name.strip()}

    for entry in plan.values():
        shot_lower = entry.shot.lower()
        for landmark, locale in _LANDMARK_LOCALES.items():
            if landmark in cast_lower:
                continue
            if not re.search(rf"\b{re.escape(landmark)}\b", shot_lower):
                continue
            if locale in setting_lower:
                continue
            logger.warning(
                "Shot plan %s: chunk_id=%d names %r, which reads as %s -- but this "
                "run's setting is %r. This may be an authoring slip (issue #32); "
                "geography is meant to be consistent across the whole video.",
                path,
                entry.chunk_id,
                entry.shot,
                locale.title(),
                setting,
            )


# --------------------------------------------------------------------------- #
# Landmark position-contradiction lint (issue #78)
# --------------------------------------------------------------------------- #

LANDMARK_CONTRADICTION_WINDOW_SECONDS = 60.0
"""How close two mentions of the same landmark must be in song time before a
FAR-vs-NEAR contradiction between them counts as one (issue #78).

A landmark legitimately reads as far away in one shot and close enough to
touch in another an act of the song later -- that is the character closing
the distance, not a defect. The real "Deathless" true positive (chunk 7's
mill "in the valley below" at 54.583s, chunk 13's mill "at shoulder height"
at 91.958s, with every line between them describing continued climbing) is
37.4s apart; measured against the real 80-chunk plan, a window from 45s to
90s catches exactly that pair and nothing past it -- 120s starts pulling in
chunk 24 (157.583s later), which is a real arrival at the mill, not a
contradiction. 60s sits comfortably inside the gap that works."""

_LANDMARK_FAR_KEYWORDS: frozenset[str] = frozenset({
    # Deliberately excludes bare "below": measured against the real 80-chunk
    # "Deathless" plan, restricting candidate words to `setting`'s own
    # vocabulary (see `_landmark_candidates`) but keeping bare "below" in
    # this set still produced 32 fired pairs, the overwhelming majority
    # false -- "below" overwhelmingly describes some OTHER noun in the same
    # sentence (the valley, the battlefield) than the landmark word it
    # happens to share a shot line with. Every phrase below is specific
    # enough about elevation/distance that it survived that measurement with
    # the true positive (chunk 7's mill) intact and nothing else firing.
    "far below", "in the valley below", "far off", "far away",
    "in the distance", "far behind", "far above",
})

_LANDMARK_NEAR_KEYWORDS: frozenset[str] = frozenset({
    # Same measurement, same exclusion logic: generic body-part-contact
    # phrases ("her palm", "her fingertips", "beneath her boot") were cut
    # after they matched unrelated hand action (guiding a needle through a
    # sleeve cuff) that happened to share a sentence with a landmark mention
    # elsewhere in the plan. What survives are phrases specific enough to
    # the landmark itself -- a hand or body actually reaching a structure --
    # that they held the true positive (chunk 13's "at shoulder height...
    # trailing along its worn wood grain") with zero false positives across
    # the real 80-chunk plan.
    "at shoulder height", "trailing along", "close beside",
    "climbs up onto", "climbs onto", "reaches out and touches",
    "inches from", "at head height",
})


def _landmark_candidates(setting: str) -> frozenset[str]:
    """Candidate "landmark" words: content words that also appear in
    ``setting`` (issue #78).

    Restricting to ``setting``'s own vocabulary -- the same field #32 already
    uses to anchor geography -- is what keeps this lint from re-exploding
    into the noisy version measured and rejected during development: without
    it, any two chunks sharing an ordinary repeated word (a verb like "lies"
    or "rising", not a landmark at all) were candidates, and the false-
    positive count went from 1 to double digits on the same 80-chunk plan.
    A word the run has already named as fixed and significant in its own
    setting is a much stronger landmark signal than mere repetition."""
    return frozenset(_singularish(w) for w in _content_words(setting))


def _lint_landmark_position_contradiction(
    setting: str | None, plan: Mapping[int, ShotPlanEntry], path: Path
) -> None:
    """Warn (never raise) when two chunks close together in song time
    describe the same named landmark at contradictory distances (issue #78).

    The defect this exists for: chunk 7 of a real "Deathless" render put
    Volokov's mill "in the valley below" while she was still climbing toward
    it, and chunk 13 -- 37.4s later, with every line between them describing
    continued ascent -- had her palm "trailing along its worn wood grain" at
    shoulder height. She cannot be climbing away from a landmark and arrive
    at it. See the module's CLAUDE.md entry and
    :data:`LANDMARK_CONTRADICTION_WINDOW_SECONDS`'s docstring for exactly how
    narrow the surviving keyword sets and candidate-word restriction had to
    become before this stopped firing on ordinary prose: an early version
    matched any repeated word plus a bare "below" and fired on ~30 pairs
    across a third of the same 80-chunk plan, almost all of them a FAR/NEAR
    phrase attaching to some OTHER noun in the same sentence than the shared
    landmark word. What ships here fires on exactly the one true positive.

    Silent whenever ``setting`` is unset (nothing to derive landmark
    candidates from) -- the same convention :func:`_lint_setting_consistency`
    uses for the same reason.
    """
    if not setting:
        return
    candidates = _landmark_candidates(setting)
    if not candidates:
        return

    word_chunks: dict[str, list[int]] = {}
    for chunk_id, entry in plan.items():
        for word in _content_words(entry.shot):
            stem = _singularish(word)
            if stem in candidates:
                word_chunks.setdefault(stem, []).append(chunk_id)

    warned_pairs: set[tuple[int, int]] = set()
    for word in sorted(word_chunks):
        ordered = sorted(set(word_chunks[word]))
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                if abs(plan[a].start - plan[b].start) > LANDMARK_CONTRADICTION_WINDOW_SECONDS:
                    continue
                shot_a = plan[a].shot.lower()
                shot_b = plan[b].shot.lower()
                far_a = next((k for k in _LANDMARK_FAR_KEYWORDS if k in shot_a), None)
                near_a = next((k for k in _LANDMARK_NEAR_KEYWORDS if k in shot_a), None)
                far_b = next((k for k in _LANDMARK_FAR_KEYWORDS if k in shot_b), None)
                near_b = next((k for k in _LANDMARK_NEAR_KEYWORDS if k in shot_b), None)
                if far_a and near_b:
                    far_id, far_hit, near_id, near_hit = a, far_a, b, near_b
                elif near_a and far_b:
                    far_id, far_hit, near_id, near_hit = b, far_b, a, near_a
                else:
                    continue
                pair_key = (min(a, b), max(a, b))
                if pair_key in warned_pairs:
                    continue
                warned_pairs.add(pair_key)
                logger.warning(
                    "Shot plan %s: chunk_id=%d and chunk_id=%d both mention %r within "
                    "%.0fs of each other in the song, but these read at contradictory "
                    "distances: chunk_id=%d reads it far away (%r) while chunk_id=%d "
                    "reads it close enough to touch (%r). A landmark cannot be both at "
                    "once this close together in the timeline (issue #78) -- reword one, "
                    "or space them further apart if the character genuinely closed the "
                    "distance in between.",
                    path,
                    a,
                    b,
                    word,
                    LANDMARK_CONTRADICTION_WINDOW_SECONDS,
                    far_id,
                    far_hit,
                    near_id,
                    near_hit,
                )


# --------------------------------------------------------------------------- #
# Consequence-vs-focus lint (issue #41)
# --------------------------------------------------------------------------- #

_CONSEQUENCE_KEYWORDS: frozenset[str] = frozenset({
    # A small, deliberately curated list of unambiguous physical/device words
    # -- see the module docstring for why this stays narrow. Matched as whole
    # words/phrases, case-insensitively, against shot text (same technique as
    # `_LANDMARK_LOCALES` above). Grouped by the categories issue #41 itself
    # names: "physical outcome, breakage, fire, lights, machinery, spills, a
    # device acting".
    #
    # Measured against the real 38-entry plan authored for "The Lucky Ones"
    # rewrite 2 (issue #41's own test case), which has none of its entries
    # set focus = "action": the first cut of this list caught only chunk 30
    # (the printer fireball) -- 1 of the plan's 5 true consequence beats, the
    # rest walked past. The additions below (marked "added") were tuned
    # against that same file and raise it to all 5 -- chunks 4, 8, 29, 30, 33
    # -- with zero false positives on the other 33 entries, which include
    # the plan's own contact beats (her boot pulling the plug, her sleeve
    # tipping the cup, Jan's finger on the mute button) where the performer
    # correctly stays the subject.
    #
    # Fire / explosions -- "The printer erupts in a fireball" is issue #26's
    # own motivating example, and is exactly the shot this lint exists to
    # catch when an author forgets `focus = "action"`.
    "explode", "explodes", "exploding", "explosion",
    "erupt", "erupts", "erupting",
    "fireball", "catches fire", "catch fire", "bursts into flames",
    "smoke",  # added -- chunk 29: "smoke starting from its seams". Excludes
    # "smoking": the gerund is at least as likely to read as a person
    # smoking a cigarette, which is exactly the kind of miss this lint must
    # not make.
    # Breakage
    "shatter", "shatters", "shattering", "shattered",
    # Lights and machinery acting on their own -- deliberately excludes
    # "spark(s)", which is common as a romantic metaphor ("sparks fly
    # between them") and would have been the single noisiest false positive
    # in a lyric-video context.
    "flicker", "flickers", "flickering",
    "flatline", "flatlines", "flatlining", "flatlined",
    "flat line", "flat-line",  # added -- chunk 4 writes it as two words,
    # not the compound "flatline" already above; same meaning, just spelling.
    "malfunction", "malfunctions", "malfunctioning",
    "short-circuit", "short-circuits", "short-circuiting",
    "overheats", "overheating",
    "jammed",  # added -- chunk 29: "the printer ... jammed and juddering".
    # "jam(s)/jamming" stay excluded: re-evaluated on chunk 29's evidence
    # (issue #41 review), but the present-tense/gerund forms are ordinary
    # music-video prose for a band jamming, a reading "jammed" (a mechanical
    # stuck-state, not a performance) does not share.
    "window dark",  # added -- chunk 33: "its window dark ... an engineer
    # inside staring at the muted desk". Kept as this exact two-word phrase
    # rather than a bare "dark", which describes ordinary scene lighting
    # constantly ("dark alley", "her hair dark against the pillow") and
    # would have been the noisiest false positive in the whole list.
    # "goes dark" / "go dark" / "goes out" were also considered for this
    # same beat and rejected: none appear anywhere in the measured plan, and
    # without a device noun anchored to them they read just as easily as a
    # person's mood or a room's lights at the end of a party.
    # Spills / flooding
    "spill", "spills", "spilling",
    "overflow", "overflows", "overflowing",
    "flood", "floods", "flooding",  # added -- chunk 8: "coffee floods the
    # keyboard".
    # Deliberately excluded as too ambiguous even though they can describe a
    # physical outcome: "collapse(s)" and "breaks/breaking" are just as
    # often the performer's own body or an emotional beat ("her voice
    # breaks"), where the performer correctly stays the subject -- flagging
    # those would be wrong, not just noisy. "black(s) out"/"blacks out" was
    # considered for chunk 8's "monitors ... black out one after another"
    # (already caught via "floods" above) and rejected on the same
    # principle: a person can black out too, and that reading is at least
    # as common in shot prose as a bank of monitors failing.
})


def _lint_consequence_focus(plan: Mapping[int, ShotPlanEntry], path: Path) -> None:
    """Warn (never raise) when a shot reads as a consequence -- something in
    the world changing rather than the performer acting -- but the entry
    still leaves the performer as the shot's subject (issue #41).

    Fires only when BOTH hold: the shot text matches a word from
    :data:`_CONSEQUENCE_KEYWORDS`, and the entry does not set
    ``focus = "action"`` (i.e. ``subject_is_focus`` is still ``True``, the
    default). An entry that already set ``focus = "action"`` is exactly what
    this lint is asking authors to do, so it is silent there.

    This is a heuristic firing on prose a human wrote deliberately -- see the
    module docstring for the design rationale shared with the other lints in
    this file. A false positive here must never be able to block a run.
    """
    for entry in plan.values():
        if not entry.subject_is_focus:
            continue  # focus = "action" already set -- nothing to flag
        shot_lower = entry.shot.lower()
        for keyword in _CONSEQUENCE_KEYWORDS:
            if not re.search(rf"\b{re.escape(keyword)}\b", shot_lower):
                continue
            logger.warning(
                "Shot plan %s: chunk_id=%d reads as a consequence beat (%r) but "
                "does not set focus = \"action\" -- the performer stays this shot's "
                "subject by default (issue #26). If this shot is about something "
                "happening to the world rather than to the performer, set "
                "focus = \"action\".",
                path,
                entry.chunk_id,
                entry.shot,
            )
            break  # one warning per entry is enough


# --------------------------------------------------------------------------- #
# Distant-staging lint (issue #58)
# --------------------------------------------------------------------------- #

_DISTANT_STAGING_KEYWORDS: frozenset[str] = frozenset({
    # A small, deliberately curated list, matched as whole words/phrases,
    # case-insensitively, against shot text -- the same technique as
    # `_CONSEQUENCE_KEYWORDS` above. Issue #58: the grammatical-subject rule
    # gets the model to attend to a beat's object, but staging it small and
    # far away still drops it from the frame. Both printer beats in a real
    # render put the printer in subject position and it was still absent --
    # staged "small against the tower far behind her" and "far behind on the
    # street". The guide's own successful example keeps the object near: "at
    # the end of the office aisle", not across the street.
    "far behind", "far off", "far away", "in the distance", "distant",
    "tiny", "way behind", "small against",
    # Added after the first full 36-chunk render (issue #58 follow-up): the
    # list only knew how to say "far" along one axis. Chunk 27 staged its
    # printer with her "far below" and the printer did not render, exactly
    # like the two "far behind" beats that motivated the lint -- the failure
    # does not care which direction the distance runs in.
    "far below", "far above", "deep background", "high in the background",
    # Deliberately excluded: bare "far" ("so far tonight") and bare "small"
    # ("a small office") are ordinary prose far more often than they are
    # staging distance -- flagging every occurrence would be the noisiest
    # warning in this file for the least benefit.
})


def _lint_distant_staging(plan: Mapping[int, ShotPlanEntry], path: Path) -> None:
    """Warn (never raise) when a shot stages its subject as small or far away
    (issue #58).

    Subject position is necessary but not sufficient: a first GPU render of a
    machine-authored plan put a printer in the grammatical subject of both
    its plant and payoff beats and H3 rendered neither, because both lines
    also described it as small and distant. A third beat in the same slice,
    staged near and large, rendered correctly. See
    ``docs/shot-writing-guide.md``'s "Stage the object near, not far".

    Warning only, for the same reason as every other lint in this file: a
    false positive on prose a human wrote deliberately must never be able to
    block a run.
    """
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        shot_lower = entry.shot.lower()
        for keyword in _DISTANT_STAGING_KEYWORDS:
            if not re.search(rf"\b{re.escape(keyword)}\b", shot_lower):
                continue
            logger.warning(
                "Shot plan %s: chunk_id=%d stages its subject as %r, which reads as small "
                "or far away -- H3 has dropped beats staged this way even from grammatical-"
                "subject position (issue #58). Stage the object in the near or mid ground, "
                "in frame with her; \"behind her\" is about narrative obliviousness, not "
                "depth.",
                path,
                chunk_id,
                keyword,
            )
            break  # one warning per entry is enough


_ANAPHORA_KEYWORDS: frozenset[str] = frozenset({
    # Issue #61. A shot line is composed into a prompt on its own: H3 renders
    # each chunk from text plus (on the base path) a cast photo, and has no
    # memory whatsoever of the shot before it. A line that refers back to one
    # is therefore referring to nothing, and the clause anchored to that
    # reference has no anchor.
    #
    # Measured on the first full 36-chunk render: chunk 27's line began "The
    # printer now teeters on the sill of *that same empty frame*" -- pointing
    # at a window broken in chunk 25 -- and the printer did not render at all,
    # while the chunk either side of it rendered theirs. Say what is in
    # frame, every time, as if it were the first shot of the film.
    "that same", "those same", "the same window", "aforementioned",
    "previous shot", "earlier shot", "as before", "as we saw", "still there from",
})


_IMPLIED_COMPANION_KEYWORDS: frozenset[str] = frozenset({
    # Issue #64. #59 gave a shot a `present` list; nothing noticed when one
    # was *missing*, and the audit that populated the first real plan looked
    # for pronouns. That is only one of the three ways this prose refers to a
    # second character, because the prose stage is forbidden from naming one:
    #
    #   pronoun     "with him keeping pace just behind"
    #   instrument  "the steady pulse of a bass carried alongside her"
    #   plural      "the speaker still squawking beside them"
    #
    # Found by eye in a finished render: "she's walking with 3 different
    # people, none of whom are Jan" across 29 seconds whose three shot lines
    # contained no pronoun at all.
    #
    # "bass"/"bassline" earn their place here because an instrument does not
    # carry itself -- in this project's prose they are how the bass player
    # gets into a line without being named. Deliberately NOT included: "they"
    # and "their", which are frequently the performer plus an object ("they
    # settle onto the water"), and would fire on prose about nobody.
    "him", "his", "he",
    "them", "beside them", "the two of them", "two walkers",
    "bass", "bassline", "bass line",
})


def _lint_implied_companion_without_present(
    plan: Mapping[int, ShotPlanEntry], path: Path
) -> None:
    """Warn (never raise) when a shot line implies a second person on screen
    and no ``present`` binds one (issue #64).

    An unbound reference is not a cosmetic problem: the render composes only
    the singing cast, so H3 is asked for a companion with no role, no
    appearance and no reference photo, and invents one -- a different one each
    chunk. That is #59's whole defect, and #59 only supplied the cure. This is
    the diagnosis.

    Warning tier like every lint here: "a bass line pacing her step" may be
    describing the score rather than a person carrying an instrument, and only
    the author knows which.
    """
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        if entry.present:
            continue
        shot_lower = entry.shot.lower()
        for keyword in _IMPLIED_COMPANION_KEYWORDS:
            if not re.search(rf"\b{re.escape(keyword)}\b", shot_lower):
                continue
            logger.warning(
                "Shot plan %s: chunk_id=%d implies a second person on screen (%r) but sets "
                "no `present`, so nothing binds that reference to a cast member -- the "
                "prompt carries only whoever is singing, and H3 invents a different "
                "stranger for every such chunk (issue #64). Add present = [\"<name>\"], or "
                "reword if nobody else is meant to be there.",
                path,
                chunk_id,
                keyword,
            )
            break  # one warning per entry is enough


def _lint_anaphora(plan: Mapping[int, ShotPlanEntry], path: Path) -> None:
    """Warn (never raise) when a shot line refers back to another shot
    (issue #61).

    Warning rather than error for the usual reason: "that same" can appear
    in a line that is perfectly self-contained ("that same beige as the
    walls"), and a false positive must never block a run.
    """
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        shot_lower = entry.shot.lower()
        for keyword in _ANAPHORA_KEYWORDS:
            if not re.search(rf"\b{re.escape(keyword)}\b", shot_lower):
                continue
            logger.warning(
                "Shot plan %s: chunk_id=%d refers back to another shot (%r), but each "
                "chunk is rendered from its own prompt and H3 has no memory of any "
                "other shot -- the reference points at nothing and the clause built on "
                "it has no anchor (issue #61). Describe what is in frame as if this "
                "were the first shot of the film.",
                path,
                chunk_id,
                keyword,
            )
            break  # one warning per entry is enough


# --------------------------------------------------------------------------- #
# Shot length (issue #27)
# --------------------------------------------------------------------------- #

TRAINED_MIN_SECONDS = H3_FRAME_GRID.frames_to_seconds(H3_FRAME_GRID.trained_min_frames)
TRAINED_MAX_SECONDS = H3_FRAME_GRID.frames_to_seconds(H3_FRAME_GRID.trained_max_frames)
"""H3's trained duration window, ~5.167-15.083 s, derived from the grid rather
than hardcoded so it cannot drift out of sync with ``docs/h3-node-schema.md``."""


def _lint_shot_lengths(plan: Mapping[int, ShotPlanEntry], path: Path) -> None:
    """Warn (never raise) about a requested length the renderer will not give
    back verbatim -- one outside H3's trained range, or one longer than
    anything ever rendered on this card.

    Load time is the right place for this: the plan is read in ~milliseconds
    and the render it configures takes hours, so an author asking for 22 s
    should hear that they will get 15.083 s *before* the GPU is committed,
    not by measuring the output afterwards.
    """
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        length = entry.length_seconds
        if length is None:
            continue

        if length > TRAINED_MAX_SECONDS + 1e-6:
            logger.warning(
                "Shot plan %s: chunk_id=%d asks for length_seconds=%.3f, above H3's trained "
                "ceiling of %.3fs (%d frames); it will be clamped down to the ceiling. More "
                "VRAM does not extend the trained range (issue #20).",
                path,
                chunk_id,
                length,
                TRAINED_MAX_SECONDS,
                H3_FRAME_GRID.trained_max_frames,
            )
        elif length < TRAINED_MIN_SECONDS - 1e-6:
            logger.warning(
                "Shot plan %s: chunk_id=%d asks for length_seconds=%.3f, below H3's trained "
                "floor of %.3fs (%d frames); it will be clamped up to the floor rather than "
                "rendered out of distribution.",
                path,
                chunk_id,
                length,
                TRAINED_MIN_SECONDS,
                H3_FRAME_GRID.trained_min_frames,
            )

        frames = H3_FRAME_GRID.clamp_to_trained(
            H3_FRAME_GRID.quantize_nearest(round(H3_FRAME_GRID.seconds_to_frames(length)))
        )
        if frames > MEASURED_MAX_FRAMES:
            logger.warning(
                "Shot plan %s: chunk_id=%d asks for a %.3fs shot (%d frames), longer than the "
                "%d frames ever rendered on this card -- unmeasured territory. It is inside "
                "H3's trained range and costs no extra wall clock (fewer, longer chunks over "
                "the same total frames), but VRAM at this frame count has never been "
                "measured here. Watch the first one; step up gradually.",
                path,
                chunk_id,
                length,
                frames,
                MEASURED_MAX_FRAMES,
            )


def shot_length_requests(
    plan: Mapping[int, ShotPlanEntry] | None,
) -> tuple[ShotLength, ...]:
    """Project ``plan`` into the anchored :class:`ShotLength` requests
    ``slicing.slice_audio`` consumes, sorted by anchor.

    Entries with no ``length_seconds`` contribute nothing, so a plan authored
    before issue #27 yields ``()`` and slicing is unchanged -- the
    conservative default is the absence of a request, not a default length.

    Two entries claiming a length at the same ``start`` is a
    :class:`ShotPlanError`, not a warning: they describe the same moment of
    the song, exactly one of them could ever be applied, and which one it was
    would depend on dict ordering. That is the shape of bug this file's drift
    check exists to refuse.
    """
    if not plan:
        return ()

    requests: list[ShotLength] = []
    seen: dict[float, int] = {}
    for chunk_id in sorted(plan):
        entry = plan[chunk_id]
        if entry.length_seconds is None:
            continue
        if entry.start in seen:
            logger.error(
                "Shot plan defines a length at start=%.3fs on both chunk_id=%d and "
                "chunk_id=%d",
                entry.start,
                seen[entry.start],
                chunk_id,
            )
            raise ShotPlanError(
                f"shot plan requests a shot length at start={entry.start} twice "
                f"(chunk_id={seen[entry.start]} and chunk_id={chunk_id}); the anchor is a "
                "time in the song, so only one of the two could ever be applied"
            )
        seen[entry.start] = chunk_id
        requests.append(
            ShotLength(
                start=entry.start,
                length_seconds=entry.length_seconds,
                source_chunk_id=chunk_id,
            )
        )

    requests.sort(key=lambda request: request.start)
    if requests:
        logger.info(
            "Shot plan requests %d editorial shot length(s): %s",
            len(requests),
            ", ".join(
                f"chunk_id={r.source_chunk_id} @{r.start:.3f}s -> {r.length_seconds:.3f}s"
                for r in requests
            ),
        )
    return tuple(requests)


# --------------------------------------------------------------------------- #
# Shot-vs-lyric lint (issue #37)
# --------------------------------------------------------------------------- #

_LINT_STOPWORDS = frozenset({
    # Function words and the handful of high-frequency content words that
    # appear in almost every lyric AND almost every shot line, so matching on
    # them would say nothing. Deliberately short: the "staged elsewhere in
    # this plan" condition below does most of the filtering.
    "that", "this", "there", "then", "when", "what", "with", "your", "yours",
    "mine", "they", "them", "their", "have", "been", "were", "was", "will",
    "would", "could", "should", "from", "into", "onto", "over", "under",
    # "through" is the same part of speech as the five prepositions beside it
    # and was simply missed; it fired on a real generated plan against the
    # lyric "As you ran Through Central Park".
    "through",
    "about", "after", "before", "still", "just", "like", "know", "knows",
    "time", "away", "back", "down", "here", "much", "more", "some", "every",
    "camera", "frame", "shot", "foreground", "background", "close", "behind",
    "front", "walks", "walking", "walk", "singing", "sings", "song",
    # Generic people-words and homographs, added after measuring the lint
    # against a real plan: they are never stageable props, but they appear in
    # both lyrics and shot prose constantly. "people" alone produced four of
    # the eight warnings on the first run, and "past" fired because a lyric's
    # "that's the past" met a shot's "walks past" -- a different sense of the
    # same string, which no amount of matching can tell apart.
    "people", "person", "someone", "everyone", "everybody", "nobody",
    "past", "world", "life", "night", "thing", "things", "everything",
})

_MIN_LINT_WORD_LENGTH = 4
"""Shorter words are overwhelmingly function words, and the few real ones
("mic", "bin") are not worth the false positives they would bring."""


def _content_words(text: str) -> set[str]:
    """Lowercased alphabetic words worth matching on. No part-of-speech
    tagging and no new dependency -- see the module docstring for why the
    'staged elsewhere' condition is what makes this precise enough."""
    words = re.findall(r"[a-z']+", text.lower())
    return {
        w.rstrip("'s").rstrip("'")
        for w in words
        if len(w) >= _MIN_LINT_WORD_LENGTH and w not in _LINT_STOPWORDS
    }


def _singularish(word: str) -> str:
    """Crude 'printers' -> 'printer' so a plural in the lyric still matches a
    singular on screen. Deliberately not a stemmer."""
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


LINT_NEIGHBOUR_RADIUS = 2
"""How many chunks either side count as "already on screen".

Three-beat staging deliberately spreads a gag across consecutive chunks (see
``docs/shot-writing-guide.md``), so an object shown in the neighbouring shot is
part of the same visual moment and its absence from this one line is not a
defect. Measured: without this, the lint warned that chunk 12's lyric names a
park while chunk 11's shot -- the immediately preceding one -- establishes the
park it is standing in."""


_WIDE_FRAMING_KEYWORDS: frozenset[str] = frozenset({
    # Chosen by reading the 79 camera values a real generated plan produced,
    # not by imagination. Every one of these appeared there; each describes a
    # frame in which a person is small.
    "wide", "extreme wide", "very wide", "aerial", "locked off", "high above",
    "high and wide", "from a distance", "establishing",
})

# _FOOT_LEVEL_KEYWORDS is RETIRED (issue #76). It shipped as: "boots",
# "boot", "feet", "ankles", "underfoot", "knees", "hem", "at his feet",
# "at her feet", "soil", "the ground", "the floor" -- body parts and
# surfaces below the chin, on the real finding that a chunk framed on a
# needle in a raised fist kept the face for 80% of its frames while a line
# reading "pushing up between his boots" rendered legs, boots and perfect
# mushrooms with no face anywhere. That underlying observation was real and
# never retested here -- it's the KEYWORD LIST that failed to generalise,
# not the observation. Re-derived against all 41 voiced chunks of the
# finished "Deathless" render: only two of the twelve words ever occur.
# "the ground" fires on chunk 43 (33% face) and chunk 68 (92% face) -- the
# same phrase, a near-60-point spread, no separation at all. "underfoot"
# fires once, on chunk 22 (33% face) -- a single data point, the same thin
# evidence #60 warns against generalising from. Combined the shipped set
# scored 0.99x against the rest of the voiced corpus: statistically
# indistinguishable from noise. Retired rather than re-derived down to one
# thin word, on the standing warning that a weak lint costs nothing at
# runtime and a great deal the moment somebody rewrites a correct shot to
# please it. Full numbers: docs/deathless-render-corpus.md Part 3.

_GAZE_AWAY_KEYWORDS: frozenset[str] = frozenset({
    # A gaze that SETTLES on something in the scene turns the head away from
    # the lens, and the sentence beats the camera field every time: one chunk
    # asked for "medium close, her face centre" and read "as she looks back
    # down at them", and she was back-to-camera throughout.
    #
    # RE-SCORED against the full 41 voiced chunks of the finished "Deathless"
    # render (issue #76), superseding the mid-render n=9 set this replaced --
    # that set separated at only 0.72x on the full corpus, not the 3.5x it
    # was justified at. Kept below: each word's one measured occurrence in
    # this corpus landed BELOW the voiced-corpus mean (53.3%). The 6 kept
    # words together score 15.3% avg / ~0.26x vs the rest of the corpus --
    # a much cleaner separation than the original 9-word set ever had.
    "looks back",   # chunk 9,  0% face
    "looks up at",  # chunk 26, 0% face
    "staring up",   # chunk 20, 0% face
    "looks out",    # chunk 59, 25% face
    "gaze fixed",   # chunk 65, 25% face
    "turns back",   # chunk 15, 42% face -- below the mean, the weakest survivor
    #
    # EXCLUDED -- measured ABOVE the corpus mean: the gaze verb fired, but
    # the camera clause in the same entry also named the face, and the
    # camera won. These are false positives on this corpus, not merely
    # untested, so they are cut rather than kept on faith:
    #   "gaze drops"     -- chunk 42, 100% face ("close, straight on, static")
    #   "gaze lifts"     -- chunk 73, 83% face ("close on his face, low, from
    #                        the horizon side")
    #   "lifts her gaze" -- chunk 64, 92% face ("close on her face, low, the
    #                        light band behind her")
    #
    # EXCLUDED -- only ever co-occur with an already-kept word on the same
    # chunk (65, alongside "gaze fixed"), so keeping them tests nothing this
    # corpus didn't already test via that word:
    #   "keeps his gaze", "staring down"
    #
    # EXCLUDED -- never occur in any of this song's 41 voiced shot lines, so
    # this corpus has no evidence for or against them (#60's discipline
    # applies to absence, too: a word this run never exercised has not been
    # measured, and does not get to ship on the strength of the words that
    # did):
    #   "gaze drifts", "keeps her gaze", "lifts his gaze", "looking back",
    #   "looking down", "looking out", "looks down", "looks over",
    #   "stares down", "stares out", "staring out", "turns away"
    #
    # STILL DELIBERATELY EXCLUDED, unaffected by this re-score:
    #   "glancing" -- the highest-scoring chunk of the whole run (80-89% face
    #     presence) reads "glancing up at a motionless figure". A glance
    #     returns to camera; a gaze settles. Including it would have flagged
    #     the best line in the plan.
    #   "watches"/"watching" -- appears in lines that rendered fine and in
    #     lines that did not, so it does not discriminate. A word that fires
    #     on both is noise, and noise is what makes a lint block get skipped.
})

_IN_PROFILE_KEYWORDS: frozenset[str] = frozenset({
    # SHIPPED (issue #76), superseding the earlier decision to leave this
    # out at n=3 (8%, 0%, 45% -- "a real risk but not a reliable one"). A
    # larger sample reversed that call rather than silently changing it: on
    # the finished 80-chunk render this is the one predictor of the four
    # tested (gaze-away, behind-camera, foot-level, in-profile) with NO
    # counter-example across all 5 occurrences -- chunks 18, 21, 37, 43, 59,
    # face presence 8%/8%/42%/33%/25%, worst case 8%, best case 42%, every
    # one below the 53.3% corpus mean. Scores 23.3% avg / 0.41x vs the rest
    # of the corpus. Checked on `camera`, ahead of the close-framing check:
    # all 5 real occurrences also contain "close" or "medium close", so a
    # profile shot IS a close shot and would otherwise pass silently.
    "in profile",
})

_BEHIND_CAMERA_KEYWORDS: frozenset[str] = frozenset({
    # Where the lens is relative to the FACE, which is not the same as the
    # direction of travel. Measured on one run: "travelling with her" up a
    # climb gave 0% face presence, while "ahead of her" on the same run gave
    # 89% -- so a moving camera is fine and a following one is not.
    #
    # "in profile" now ships as its own check, _IN_PROFILE_KEYWORDS above --
    # not here (issue #76 reversed the earlier decision to leave it out).
    #
    # NOT re-scored against the full render, unlike the gaze and foot-level
    # sets above -- issue #76 did not ask for it, and this set's own two
    # measured hits on the full corpus are BOTH the same phrase,
    # "travelling with", on chunk 7 (83% face, "her closed fist and her face
    # held in the same frame") and chunk 25 (0% face, "the sails passing
    # through the background") -- the same word, opposite outcomes, one
    # combined score of 0.77x. That is the same shape of evidence that
    # retired the foot-level set above, and this set was left untouched only
    # because it was out of this issue's stated scope; flagged in
    # docs/deathless-render-corpus.md as a candidate for a future pass
    # rather than changed here.
    "travelling with", "tracking behind", "from behind", "following her",
    "following him", "behind her shoulder", "over her shoulder from behind",
})

_CLOSE_FRAMING_KEYWORDS: frozenset[str] = frozenset({
    # If any of these is present the framing is close enough to read a mouth,
    # even when a width word also appears ("medium wide on her face").
    "close", "tight", "macro", "medium", "three-quarter", "over the shoulder",
    "portrait", "head and shoulders",
})


def lint_voiced_framing(
    plan: Mapping[int, ShotPlanEntry],
    chunks: Sequence[AudioChunk],
) -> None:
    """Warn (never raise) when a chunk that carries a lyric is not framed
    close enough to read a mouth.

    Lip-sync is the whole reason this pipeline exists, and it needs a face
    big enough in frame to see. Nothing else in the authoring layer knows
    that: the photography stage optimises for the image, and for most of a
    song the better image genuinely is the wide one.

    Measured on the first machine-authored plan to reach a GPU. Of its 41
    voiced chunks, **11** had a close or medium framing; 1 was explicitly
    wide and 29 carried no ``camera`` value at all. Across the chunks
    rendered from it a face was detectable (YuNet, the #47 detector) in
    0-33% of sampled frames. One chunk was re-rendered four ways -- the
    second character bound with ``present`` and not, the shot line rewritten
    to make the singer the subject, the camera pointed at her -- and every
    variant lost the face, because the plan around it was landscape.

    Absent is warned about as loudly as wide, deliberately: ``camera`` is
    optional per shot, and an omitted framing on a sung chunk is not a
    neutral default. It hands the decision to H3, which measured wide.

    Warning only, like every lint here: a wide shot over a sung line is a
    real editorial choice (a held establishing shot under the first line of a
    verse), and a false positive must never be able to block a run.
    """
    voiced = {c.chunk_id for c in chunks if not c.is_instrumental and (c.text or "").strip()}
    for chunk_id in sorted(plan):
        if chunk_id not in voiced:
            continue
        camera = (plan[chunk_id].camera or "").strip().lower()
        if not camera:
            logger.warning(
                "Shot plan: chunk_id=%d carries a lyric but sets no `camera`, so nothing "
                "asks for the singer to be close enough to read a mouth -- the framing is "
                "left to H3, which measured wide. Lip-sync needs a face in frame; give a "
                "sung chunk a close or medium framing on whoever is singing it.",
                chunk_id,
            )
            continue
        # A gaze verb in the shot line settles the head away from the lens
        # regardless of what `camera` asks for -- the sentence outranks the
        # field. See _GAZE_AWAY_KEYWORDS for the full re-scored list.
        shot_lower = plan[chunk_id].shot.lower()
        gaze = next((k for k in sorted(_GAZE_AWAY_KEYWORDS) if k in shot_lower), None)
        if gaze:
            logger.warning(
                "Shot plan: chunk_id=%d carries a lyric but its shot line says %r, which "
                "settles the singer's gaze on something in the scene and turns the head "
                "away from the lens -- the sentence outranks the camera field, so no "
                "framing recovers it. Put what she is looking at near her, or move the "
                "looking to an instrumental chunk.",
                chunk_id,
                gaze,
            )
            continue
        behind = next((k for k in sorted(_BEHIND_CAMERA_KEYWORDS) if k in camera), None)
        if behind:
            logger.warning(
                "Shot plan: chunk_id=%d carries a lyric but the camera is %r -- a "
                "following shot is the back of a head. A moving camera is fine: the "
                "same run measured 89%% face presence on \"ahead of her\" and 0%% on "
                "\"travelling with her\". Put the lens in front of the singer.",
                chunk_id,
                behind,
            )
            continue
        # Checked before the close-framing check below: every one of this
        # song's 5 real "in profile" occurrences also contains "close" or
        # "medium close", so a profile shot IS a close shot and would
        # otherwise pass silently. Measured at 23.3% avg face presence / 0.41x
        # vs the rest of the voiced corpus, no counter-example (issue #76).
        profile = next((k for k in sorted(_IN_PROFILE_KEYWORDS) if k in camera), None)
        if profile:
            logger.warning(
                "Shot plan: chunk_id=%d carries a lyric but the camera is %r -- a "
                "profile framing loses the face far more often than a close shot "
                "generally does (23%% avg face presence across every measured "
                "occurrence on a real render, vs 53%% overall). Face the lens more "
                "directly, or move the profile framing to an instrumental chunk.",
                chunk_id,
                profile,
            )
            continue
        if any(k in camera for k in _CLOSE_FRAMING_KEYWORDS):
            continue
        hit = next((k for k in sorted(_WIDE_FRAMING_KEYWORDS) if k in camera), None)
        if hit:
            logger.warning(
                "Shot plan: chunk_id=%d carries a lyric but is framed %r, which puts the "
                "singer too small to read a mouth. Wide shots are what a music video is "
                "for -- spend them on the instrumental chunks, where no mouth has to "
                "match anything.",
                chunk_id,
                hit,
            )


def lint_shots_against_lyrics(
    plan: Mapping[int, ShotPlanEntry],
    chunks: Sequence[AudioChunk],
) -> None:
    """Warn when a chunk's lyric names an object this plan stages *elsewhere*.

    The failure this exists for (issue #37): "The Lucky Ones" sings "your
    printer would explode somehow" twice, and the plan staged a printer only
    for the second. Over the first, the lyric named a printer while the video
    showed a hospital. Both strings were already in hand and nothing compared
    them.

    Why "elsewhere in this plan" rather than "anywhere in the lyric": most
    shots legitimately do not echo their lyric, so requiring it would warn on
    nearly every chunk and be rightly ignored. A word that appears in the
    author's own shot text somewhere else is demonstrably an object they meant
    to put on screen, which makes its absence here worth a line. On the song
    that motivated this, it fires exactly once -- on the printer -- and stays
    silent on the snow plow and the muted mic, both staged where they are sung.

    Warning only, never an error: this is a heuristic firing on prose a human
    wrote deliberately, so a false positive must never block a run.
    """
    if not plan or not chunks:
        return

    # Every content word the author used anywhere in their own shot text, and
    # which chunks stage it. This is the vocabulary of things meant to be seen.
    staged: dict[str, list[int]] = {}
    for chunk_id, entry in plan.items():
        for word in _content_words(entry.shot):
            staged.setdefault(_singularish(word), []).append(chunk_id)

    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None or chunk.is_instrumental or not chunk.text.strip():
            continue
        # Everything on screen in this shot and its immediate neighbours: a
        # gag staged across consecutive chunks is one visual moment.
        here: set[str] = set()
        for offset in range(-LINT_NEIGHBOUR_RADIUS, LINT_NEIGHBOUR_RADIUS + 1):
            near = plan.get(chunk.chunk_id + offset)
            if near is not None:
                here |= {_singularish(w) for w in _content_words(near.shot)}
        for word in sorted(_singularish(w) for w in _content_words(chunk.text)):
            nearby = set(range(
                chunk.chunk_id - LINT_NEIGHBOUR_RADIUS, chunk.chunk_id + LINT_NEIGHBOUR_RADIUS + 1
            ))
            elsewhere = sorted(set(staged.get(word, ())) - nearby)
            if word in here or not elsewhere:
                continue
            logger.warning(
                "Shot plan chunk_id=%d: the lyric names %r but this shot does not show it "
                "-- the plan stages %r in chunk(s) %s. If the object belongs on screen here "
                "too, say so; a repeated verse is a chance to plant it early and pay it off "
                "later (issue #37). Shot: %r | lyric: %r",
                chunk.chunk_id,
                word,
                word,
                elsewhere,
                entry.shot[:80],
                chunk.text[:60],
            )


# --------------------------------------------------------------------------- #
# Camera-vs-lip-sync lint (issue #58)
# --------------------------------------------------------------------------- #

_CAMERA_AWAY_KEYWORDS: frozenset[str] = frozenset({
    # A small, deliberately curated set of unambiguous phrases, matched
    # case-insensitively against the entry's `camera` value only -- not
    # `shot` -- because `camera` is the structured field a photography stage
    # writes (issue #53). A false positive here must never block a run, same
    # discipline as every other lint in this file.
    #
    # The narrower of this function's two signals (see its docstring), kept
    # for the chunks that spell it out in the structured field.
    "back to camera", "her back to", "his back to", "their back to",
    "away from her", "away from him", "away from them",
    "turns away", "turning away", "faces away", "facing away",
})

_SHOT_WALKS_AWAY_KEYWORDS: frozenset[str] = frozenset({
    # Issue #60, and the primary signal now. Matched against the `shot` line,
    # because that is where the one measured failure actually lived.
    #
    # Calibrated against 36 real shot lines with a known per-chunk outcome,
    # not invented: "keep walking" and "behind them" each match chunk 13 --
    # the only chunk in that render that lost its lip-sync -- and no other
    # voiced chunk. The rest are the same idea said other ways and matched
    # nothing in that corpus, so they cost nothing to include.
    #
    # Two deliberate exclusions, both measured on the same corpus:
    # "receding" matches chunk 7, which turned away and kept its sync; and a
    # bare "behind her" appears in 20 of the 36 lines as narrative
    # obliviousness, not camera position.
    "keep walking", "keeps walking", "keep moving", "walking away",
    "walks away", "walk away", "behind them", "their backs", "her back to",
    "away from camera", "away down the",
})


def lint_camera_face_away_on_voiced_chunks(
    plan: Mapping[int, ShotPlanEntry], chunks: Sequence[AudioChunk]
) -> None:
    """Warn about a voiced chunk at risk of losing the performer's face to
    the lens (issue #58).

    A separate check, run by the orchestrator once chunks exist -- the same
    shape as :func:`lint_shots_against_lyrics` above, and for the same
    reason: it needs to know which chunks carry a lyric, which only the
    chunk timeline knows.

    Two independent signals:

    1. **A ``shot`` line that walks her away from the lens (the primary
       signal, issue #60).** Phrases like "as they keep walking" or a
       fronted "Behind them," put the camera at her back for the duration of
       a chunk that still needs a mouth to sync.
    2. **A ``camera`` value that reads as turning her away (the older,
       narrower signal).** Explicit phrasing like "back to camera" or "away
       from her", for the chunks that spell it out.

    **What this lint used to check, and why it does not any more.** #58 made
    ``focus = "action"`` on a voiced chunk the primary signal, generalising
    from one chunk re-rendered three times: two ``camera`` rewrites and a
    ``shot`` rewrite all failed identically, and ``focus = "action"`` was the
    one thing held constant. The first full 36-chunk render tested that on 7
    voiced chunks at once and it did not survive contact. All 7 turned away,
    as predicted -- and all 7 kept their lip-sync. Turning away is real,
    common, and *usually free*; the predicted cost simply is not there. The
    inference had confused the mechanism with its consequence.

    Worse, the one chunk in that render that did lose sync was not flagged at
    all: chunk 13 is a ``transition`` beat with ``focus`` unset, whose line
    reads "**Behind them**, slush slides slowly off the buried car's roof as
    **they keep walking**...". The prose did everything ``focus = "action"``
    does, while the field the lint watched stayed clean. So the check was
    aimed at the wrong variable in both directions -- 7 false positives and
    the one true case missed -- and it is the prose, not the field, that
    predicts it.

    Keywords were calibrated against that render's 36 real lines rather than
    invented: "keep walking" and "behind them" each match chunk 13 and
    nothing else. "receding" is deliberately excluded even though it sounds
    apt -- it matches chunk 7, which turned away and kept its sync. And a
    bare "behind her" is excluded because it appears in 20 of the 36 lines,
    where it means narrative obliviousness rather than camera position (the
    same distinction ``_lint_distant_staging`` documents).

    Warning only, for both signals: a curated keyword match cannot tell a
    deliberate choice from an oversight, and the evidence base here is one
    true positive.
    """
    for chunk in chunks:
        if chunk.is_instrumental or not chunk.text.strip():
            continue
        entry = plan.get(chunk.chunk_id)
        if entry is None:
            continue

        shot_lower = entry.shot.lower()
        for keyword in _SHOT_WALKS_AWAY_KEYWORDS:
            if not re.search(rf"\b{re.escape(keyword)}\b", shot_lower):
                continue
            logger.warning(
                "Shot plan chunk_id=%d: this chunk carries a lyric (%r) but its shot "
                "line reads as walking away from the lens (%r) -- a voiced chunk still "
                "needs a mouth to sync (issue #60). This is the phrasing of the one "
                "chunk in a real 36-chunk render that actually lost its lip-sync. Keep "
                "her turned toward the lens for a chunk carrying a lyric, or move the "
                "movement onto a nearby instrumental chunk.",
                chunk.chunk_id,
                chunk.text.strip()[:60],
                keyword,
            )
            break  # one warning per entry is enough

        if not entry.camera:
            continue
        camera_lower = entry.camera.lower()
        for keyword in _CAMERA_AWAY_KEYWORDS:
            if keyword not in camera_lower:
                continue
            logger.warning(
                "Shot plan chunk_id=%d: this chunk carries a lyric (%r) but its camera "
                "direction (%r) reads as turning her away from the lens -- a voiced chunk "
                "still needs a mouth to sync (issue #58). Camera direction that cranes away "
                "and holds is fine only on instrumental chunks.",
                chunk.chunk_id,
                chunk.text.strip()[:60],
                entry.camera,
            )
            break  # one warning per entry is enough


# --------------------------------------------------------------------------- #
# Referent-based companion lint (issue #72)
# --------------------------------------------------------------------------- #

_THIRD_PERSON_PRONOUN_PATTERN = re.compile(r"\b(he|him|his|she|her|hers)\b", re.IGNORECASE)
"""Every third-person singular personal pronoun this lint resolves. Wider
than issue #64's own list (which has no ``she``/``her`` at all, because #64
was built from an audit that happened to find only ``him``/``he`` cases) --
this lint needs both, since the case it exists for (chunk 65) is "he" left
unbound while "Dianne" (a ``she``) is the one already accounted for, and the
mirror case (chunk 66) is exactly the reverse."""

_SECOND_PERSON_DISTANCE_KEYWORDS: frozenset[str] = frozenset({
    # A small, deliberately curated set, matched case-insensitively against
    # `shot` text -- the same technique as `_DISTANT_STAGING_KEYWORDS` and
    # every other lint in this file. Each phrase has to assert TWO distinct
    # entities occupying different points in space, not merely something
    # near the one entity everybody already knows about.
    #
    # Measured against the real 80-chunk "Deathless" plan, gated to chunks
    # where exactly one cast member is already bound (see the function
    # docstring): "a few paces off" fires once, on chunk 65 -- the chunk this
    # lint exists for. "just behind him" fires once, on chunk 66, the mirror
    # case: Jan's own "He" is the unbound one there, while the "ash-crusted
    # figure...just behind him" (Dianne, correctly bound via `present`) is
    # what supplies the second-person evidence. "just behind her" is added
    # as the obvious symmetric form; it does not occur in the measured plan
    # in either direction.
    "a few paces off", "just behind him", "just behind her",
    "between them",
    # "between them" appears twice in the measured plan (chunks 69 and 71)
    # and is silent both times -- correctly, since both chunks already bind
    # two real names via `present`, so the single-candidate gate below never
    # reaches the phrase check for either. Kept because it is one of the
    # phrases issue #72 itself proposes and nothing in the corpus argues
    # against it.
    #
    # DELIBERATELY EXCLUDED, with reasons -- also proposed by issue #72:
    #   "beside her" / "beside him" / "next to her" / "next to him" /
    #   "at her side" / "at his side" -- measured 3 for 3 false positive on
    #   this same plan: "her shadow sprawling up the rock face beside her"
    #   (chunk 18, a shadow), "its roof beams collapsed and blackened close
    #   beside her" (chunk 24, a ruined terrace), "the sheer summit wall
    #   rising close beside her" (chunk 30, a cliff). "Beside" attaches to
    #   whatever inanimate noun is nearest in a landscape-heavy plan far more
    #   often than it introduces a second person -- this file's own #64 test
    #   suite already has a case for exactly this ("The printer lies smashed
    #   in a snowbank...beside her" must stay silent), which is the same
    #   lesson landing a second time.
    #   "a few steps away" / "a short distance away" / "off to the side" /
    #   "just ahead of her" / "just ahead of him" -- same shape as the
    #   shipped phrases, but never occur in the measured plan in either
    #   direction, so there is no evidence either way. Left out per issue
    #   #60's rule against shipping an unscored keyword; add them once a real
    #   plan supplies a case.
})


def lint_unbound_companion_referent(
    plan: Mapping[int, ShotPlanEntry], chunks: Sequence[AudioChunk]
) -> None:
    """Warn (never raise) when a shot's only bound cast member is not who a
    pronoun in the shot actually refers to (issue #72).

    Issue #64 (one section up) asks a question about the *prompt*: does some
    cast member's name appear in it at all? ``present = ["Dianne"]`` answers
    yes and #64 stays silent -- but on chunk 65 of a real "Deathless" render,
    Dianne was already the singer, so naming her again bound nothing that was
    not already bound, and the shot's own "he"/"his" pointed at Jan, who was
    in no field anywhere. The question that matters is about the *sentence*:
    is there a pronoun with no distinct referent among the names the prompt
    actually carries?

    The "candidates" a pronoun could resolve to are ``chunk.characters``
    (who is actually singing -- empty on an instrumental chunk, which is
    also exactly when ``prompting._resolve_active_members`` falls back to
    ``config.default_lead_vocalist``, so an instrumental chunk is never
    truly "nobody": it is whoever that fallback would compose) unioned with
    ``entry.present``. When that union names two or more people, a pronoun
    has somewhere real to land even if it is ambiguous which -- both
    identities are in the prompt regardless -- so this lint is silent. It
    only fires when exactly **one** name is bound and the shot's own prose
    still insists on a second, distinct entity via
    :data:`_SECOND_PERSON_DISTANCE_KEYWORDS`.

    **On gender.** The issue that motivated this lint proposes a cheaper
    version: warn when the only bound candidate's gender does not match the
    pronoun's. Nothing in this project's cast configuration carries a
    gender -- :class:`~music_video_maker.contracts.CastMember` has ``name``,
    ``role``, ``image`` and ``appearance``, and none of those is a
    structured "how do this member's own pronouns read" field. Rather than
    guess it from a name (unreliable, and the issue explicitly warns against
    exactly that), this lint uses the structural signal above instead, which
    needs no gender at all. If gender ever becomes worth adding formally,
    the field this lint would consume is a per-member
    ``CastMember.pronoun`` (or similar) that a lint could compare against
    the pronoun actually used -- until then, the phrase-based check above is
    the best available version, not a placeholder for a better one already
    in hand.

    Warning tier, like every lint in this file: a false positive on prose a
    human wrote deliberately must never be able to block a run.
    """
    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None:
            continue
        shot_lower = entry.shot.lower()
        if not _THIRD_PERSON_PRONOUN_PATTERN.search(shot_lower):
            continue
        candidates = set(chunk.characters) | set(entry.present)
        if len(candidates) != 1:
            continue
        hit = next(
            (k for k in sorted(_SECOND_PERSON_DISTANCE_KEYWORDS) if k in shot_lower), None
        )
        if hit is None:
            continue
        (only_bound,) = candidates
        logger.warning(
            "Shot plan chunk_id=%d: the only cast member bound to this shot is %r, but "
            "the shot line says %r -- which reads as a SECOND, distinct person, and this "
            "shot's own pronoun has nowhere else to resolve to (issue #72). If somebody "
            "else is meant to be on screen, add them to present = [...]; if %r is meant "
            "to be alone, reword so the pronoun clearly refers back to them.",
            chunk.chunk_id,
            only_bound,
            hit,
            only_bound,
        )


# --------------------------------------------------------------------------- #
# Role-prohibition-vs-shot lint (issue #73)
# --------------------------------------------------------------------------- #

_PROHIBITION_TRIGGER_PATTERN = re.compile(r"\b(?:never|without|no)\s+([a-z]+)", re.IGNORECASE)
"""Same trigger words as ``config._PROHIBITION_PATTERN`` (issue #73) --
"never"/"no"/"without" are what a human reaches for to write a prohibition.
This pattern additionally captures the single content word right after the
trigger, which is the term this lint tries to find asserted elsewhere."""

_PROHIBITION_EXTRACTION_STOPWORDS: frozenset[str] = frozenset({
    # Words a prohibition trigger is very often followed by that name nothing
    # concrete -- extracting these would just search for a stopword. "never
    # holding anything" must yield "holding", not "anything".
    "anything", "something", "nothing", "one", "ones", "it", "that", "this",
    "a", "an", "the", "one's",
})

_HOLD_CONTACT_TERMS: frozenset[str] = frozenset({"hold", "holds", "holding", "held"})
"""The one prohibited concept this lint has TWO independent measured real
bugs for (#31's "Bass player" role and #73's "never holding anything" role,
both on the same character, both putting an object in his hands) -- see
:data:`_HOLD_CONTACT_PATTERN` for why the plain words above are matched only
via a curated synonym set rather than searched for literally."""

_HOLD_CONTACT_PATTERN = re.compile(
    r"\b(?:grip|grips|gripping|gripped|grasp|grasps|grasping|grasped|"
    r"clutch|clutches|clutching|clutched|cradles|cradling|cradled)\b"
    r"|\bclose[sd]?\s+\w+\s+fingers\b|\bclosing\s+\w+\s+fingers\b",
    re.IGNORECASE,
)
"""What actually gets searched for when a `role` prohibits "holding"
something, instead of the bare words in :data:`_HOLD_CONTACT_TERMS`.

Chunk 72 of the real "Deathless" plan -- the clean test case issue #73 names
-- never uses the word "hold" at all: "Her fingers press the needle into his
open palm and close his fingers over it." A literal search for "hold" would
miss it entirely.

The opposite failure is just as real: searching `shot` text for the bare verb
"hold" in all its forms, measured against the same plan restricted to chunks
where Jan (the character whose role carries the prohibition) is present,
matched double digits of chunks -- "he holds his stance", "he holds his
ground", "the valley holds no armies now" -- none of which are Jan holding an
OBJECT. Even the single exact word "holding" (not the wider family) still
matched two `camera` values ("holding the far dust columns", "holding
steady") that are cinematography jargon for framing, not a hand. That is why
this pattern is (a) restricted to `shot` text only, never `camera`, and (b) a
small curated set of unambiguous physical-contact synonyms plus the specific
"closes ... fingers" construction, not the word "hold" itself. Scored
against the same 80-chunk plan, restricted to `shot` text on Jan's chunks:
exactly one match, chunk 72."""


def _extract_prohibited_terms(role: str) -> frozenset[str]:
    """Every content word immediately following a prohibition trigger
    ("never"/"no"/"without") in ``role``, singularized. See
    :data:`_PROHIBITION_TRIGGER_PATTERN`."""
    terms = set()
    for match in _PROHIBITION_TRIGGER_PATTERN.finditer(role):
        word = match.group(1).lower()
        if word in _PROHIBITION_EXTRACTION_STOPWORDS:
            continue
        terms.add(_singularish(word))
    return frozenset(terms)


def lint_role_prohibition_contradiction(
    plan: Mapping[int, ShotPlanEntry],
    chunks: Sequence[AudioChunk],
    cast: Mapping[str, CastMember],
) -> None:
    """Warn (never raise) when a chunk's ``shot`` text asserts something a
    present cast member's ``role`` says they never do (issue #73).

    The corrected `role` "...never holding anything" was in force for an
    entire real render and a bass guitar still appeared in the same hands
    twice, because a diffusion prompt has no channel for negation -- the only
    tokens it sees are "holding" and "anything"
    (``config._warn_if_prohibition`` now warns about the field in isolation,
    at load time). This is the other half: catching the shot line that
    actually describes the forbidden thing happening.

    For each cast member bound to a chunk (``chunk.characters | entry.present``
    -- the same union :func:`lint_unbound_companion_referent` uses), every
    prohibited term extracted from that member's ``role`` is checked against
    the chunk's ``shot`` text only, never ``camera`` -- see
    :data:`_HOLD_CONTACT_PATTERN`'s docstring for why ``camera`` was measured
    and dropped. A term that is a form of "hold" uses the curated
    physical-contact synonym set in :data:`_HOLD_CONTACT_PATTERN`, the one
    concept this project has two independent measured bugs for; every other
    extracted term is searched for literally (singularized, word-boundary).

    **Honesty about how far this generalises.** This is a literal (or, for
    "holding" specifically, near-literal) collision check on the term
    immediately following the prohibition trigger -- not comprehension of
    what a `role` forbids. A prohibition worded around a different concept
    ("no modern equipment") will only be caught if the shot text uses a
    plainly cognate word; a paraphrase will slip through exactly as "close
    his fingers over it" would have slipped through a literal "hold" search.
    Issue #73's own conclusion is that the durable fix is positive phrasing
    in `role` plus the render-side `avoid` list actually reaching the prompt
    -- this lint is a backstop for the one shape of violation it can see,
    not a substitute for either.

    Warning tier, like every lint in this file: a false positive on prose a
    human wrote deliberately must never be able to block a run.
    """
    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None:
            continue
        candidates = set(chunk.characters) | set(entry.present)
        if not candidates:
            continue
        shot_lower = entry.shot.lower()
        for name in sorted(candidates):
            member = cast.get(name)
            if member is None:
                continue
            terms = _extract_prohibited_terms(member.role)
            if not terms:
                continue
            for term in sorted(terms):
                if term in _HOLD_CONTACT_TERMS:
                    match = _HOLD_CONTACT_PATTERN.search(shot_lower)
                else:
                    match = re.search(rf"\b{re.escape(term)}s?\b", shot_lower)
                if match is None:
                    continue
                logger.warning(
                    "Shot plan chunk_id=%d: %s's role says %r (a prohibition -- issue "
                    "#73), but this chunk's shot line says %r, which reads as %s doing "
                    "exactly that (matched %r). A diffusion prompt cannot condition on "
                    "the ABSENCE of a token, so the prohibition in role has no effect "
                    "here; reword the shot, or reword the role to say what IS true "
                    "instead of what is not.",
                    chunk.chunk_id,
                    name,
                    member.role,
                    entry.shot,
                    name,
                    match.group(0),
                )
                break  # one warning per (chunk, member) is enough


# --------------------------------------------------------------------------- #
# present-vs-location mismatch lint (issue #78)
# --------------------------------------------------------------------------- #


def lint_present_location_mismatch(
    plan: Mapping[int, ShotPlanEntry], chunks: Sequence[AudioChunk]
) -> None:
    """Warn (never raise) when ``present`` stages a companion at a location
    that contradicts where their own singing chunks have placed them
    (issue #78).

    ``present`` (issue #59) answers "is this cast member on screen here"; it
    never answered "does the run even know the two characters are in the
    same place". Chunk 7 of a real "Deathless" render set
    ``present = ["Jan"]`` on a hillside scene that never mentions him -- the
    "hill where Jan is" a viewer's first note complained about -- while Jan's
    own singing chunks, dozens of chunks later, place him at the watch-post.
    Nothing caught that the two locations disagree.

    Structural, not textual: this compares the STRUCTURED ``location`` tag
    alone, never prose. For every name, ``own_locations`` is the set of
    ``location`` values recorded on the chunks where that name is actually
    singing (``chunk.characters``) -- the only chunks a name's location can
    be established from without guessing. A chunk that stages someone via
    ``present`` at a ``location`` outside that set, when the set is
    non-empty, is a genuine contradiction: nothing shows how they got from
    one place to the other, and #61 already rules out a shot line saying so
    itself.

    Deliberately silent when a present companion has **no** singing chunk of
    their own at all -- not a gap, a design choice. A companion who never
    sings solo (very common in a two-hander) would otherwise trip this on
    nearly every use of ``present``, for exactly the reason "absence of
    evidence" lints are excluded throughout this module: warning on "nothing
    is known yet" is not the same claim as warning on "this contradicts what
    is known", and only the second one is worth a human's attention.

    Silent, too, whenever ``location`` is unset on either side -- true of
    every plan authored before issue #78, including the real 80-chunk
    "Deathless" plan this project has on hand, which predates the field
    entirely and resolves every ``location`` to ``None``.
    """
    own_locations: dict[str, set[str]] = {}
    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None or not entry.location:
            continue
        for name in chunk.characters:
            own_locations.setdefault(name, set()).add(entry.location)

    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None or not entry.location or not entry.present:
            continue
        for name in entry.present:
            locations = own_locations.get(name)
            if not locations or entry.location in locations:
                continue
            logger.warning(
                "Shot plan chunk_id=%d: present=[%r] stages %s at location=%r, but %s's "
                "own singing chunk(s) elsewhere in the plan put them at %s -- nothing "
                "shows how %s got from one to the other, and a shot line may not "
                "reference another shot to explain it (issue #78, issue #61). Either "
                "%s is not really here, or an earlier/later chunk needs to show the "
                "move.",
                chunk.chunk_id,
                name,
                name,
                entry.location,
                name,
                sorted(locations),
                name,
                name,
            )


# --------------------------------------------------------------------------- #
# `subject`: whose shot this is (issue #82)
# --------------------------------------------------------------------------- #


def lint_subject_on_voiced_chunk(
    plan: Mapping[int, ShotPlanEntry],
    chunks: Sequence[AudioChunk],
    path: Path | None = None,
) -> None:
    """RAISE (not warn) when an entry sets ``subject`` and its chunk is
    voiced (issue #82).

    The one lint in this module that raises rather than warns, because
    ``ShotPlanEntry.subject``'s docstring states the asymmetry: on a voiced
    chunk the singer owns the frame, and three separate measured findings
    (#58, #59, #60) say the sentence outranks any field that argues
    otherwise. Honouring ``subject`` there would make the field a new way to
    reintroduce the very desync it exists to prevent, so a plan that tries
    it is refused before any GPU time is spent rather than silently
    mis-composed. ``expand_prompt`` also refuses this case
    (:class:`~music_video_maker.prompting.SubjectOnVoicedChunkError`) as
    defence in depth, but this is the primary gate: it runs at plan-load
    time, against the real chunk timeline, before Stage 3 ever stages
    anything.

    Silent whenever ``plan``/``chunks`` is empty, an entry never set
    ``subject``, or its chunk is instrumental -- the legal case.
    """
    if not plan or not chunks:
        return
    for chunk in chunks:
        entry = plan.get(chunk.chunk_id)
        if entry is None or entry.subject is None or chunk.is_instrumental:
            continue
        logger.error(
            "Shot plan %s: chunk_id=%d sets subject=%r but this chunk is voiced -- the "
            "singer owns the frame on a voiced chunk (issues #58, #59, #60), and "
            "honouring `subject` here would reintroduce the desync the field exists to "
            "avoid. `subject` is legal only on an instrumental chunk.",
            path,
            chunk.chunk_id,
            entry.subject,
        )
        raise ShotPlanError(
            f"shot plan chunk_id={chunk.chunk_id} sets subject={entry.subject!r} but this "
            "chunk is voiced; `subject` is legal only on an instrumental chunk -- the "
            "singer owns the frame on a voiced chunk (issues #58, #59, #60)"
        )


def _solo_voiced_pronoun_counts(
    plan: Mapping[int, ShotPlanEntry], chunks: Sequence[AudioChunk]
) -> dict[str, dict[str, int]]:
    """``{cast_name: {pronoun: count}}``, tallied from the shot line of every
    chunk where ``name`` is this chunk's ONLY singer.

    Not a guess from the name -- issue #72's own docstring already rejects
    that ("Nothing in this project's cast configuration carries a gender...
    Rather than guess it from a name (unreliable...)"). This measures
    instead: on a solo voiced chunk the sentence's subject is already the
    singer (#58, #59, #60), so whatever third-person pronoun the author
    reached for while writing that chunk's own shot line IS that singer's
    own pronoun, self-reference by construction -- evidence from the plan in
    hand, not an assumption about the name."""
    counts: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        if len(chunk.characters) != 1:
            continue
        entry = plan.get(chunk.chunk_id)
        if entry is None or not entry.shot:
            continue
        name = chunk.characters[0]
        for match in _THIRD_PERSON_PRONOUN_PATTERN.finditer(entry.shot):
            token = match.group(0).lower()
            bucket = counts.setdefault(name, {})
            bucket[token] = bucket.get(token, 0) + 1
    return counts


def _pronoun_ownership(counts: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    """``{pronoun: cast_name}`` for every pronoun with an unambiguous
    majority owner among :func:`_solo_voiced_pronoun_counts`' measurements,
    omitted when the evidence is too thin to trust.

    Two occurrences is the floor -- a single hit is noise, not a signature,
    issue #60's own lesson -- and the top name must strictly outrank every
    other name that also used the pronoun at all. Measured on the real
    80-chunk "Deathless" plan this cleanly separates "she"/"her" (Dianne: 22
    and 23 of the singer's own solo occurrences, against 2 and 1 noise) from
    "he"/"his"/"him" (Jan: 11, 17 and 3, against 2, 0 and 0 noise) with no
    tie either way."""
    totals: dict[str, dict[str, int]] = {}
    for name, pronoun_counts in counts.items():
        for pronoun, n in pronoun_counts.items():
            totals.setdefault(pronoun, {})[name] = n
    ownership: dict[str, str] = {}
    for pronoun, by_name in totals.items():
        ranked = sorted(by_name.items(), key=lambda item: -item[1])
        top_name, top_count = ranked[0]
        if top_count <= 1:
            continue
        if len(ranked) > 1 and ranked[1][1] >= top_count:
            continue
        ownership[pronoun] = top_name
    return ownership


def lint_instrumental_focus_mismatch(
    plan: Mapping[int, ShotPlanEntry],
    chunks: Sequence[AudioChunk],
    default_lead_vocalist: str,
) -> None:
    """Warn (never raise) when an instrumental chunk's shot line reads as
    entirely about a ``present`` bystander, with nothing in the plan telling
    the render that (issue #82) -- the general shape of the chunk 29 bug
    ``subject`` (above) exists to fix.

    ``chunk.characters`` is empty on an instrumental chunk, so
    ``prompting._resolve_active_members`` falls back to
    ``default_lead_vocalist`` and composes THAT person as "the focus of this
    shot" -- a config default answering a question the shot line already
    answers differently. ``present`` (issue #59) stages the real subject's
    name, role, appearance and photo, but does not change who is billed as
    the focus, and #72's own referent lint stays silent here too: its whole
    premise is that a pronoun needs a bound name to resolve to, and
    ``present`` already bound one. Only ``subject`` fixes it; this lint is a
    way to notice a plan is missing it before a render finds out.

    Fires only when EVERY third-person pronoun this chunk's shot line uses
    belongs (per :func:`_pronoun_ownership`'s measured evidence) to someone
    named in ``present``, and NONE belongs to ``default_lead_vocalist`` --
    i.e. the sentence gives the composed default focus no textual ground to
    stand on at all. Measured on the real 80-chunk "Deathless" plan: 10 of
    39 instrumental chunks fire this way, including chunk 29 itself and the
    other three (1, 4, 34) a manual audit (issue #72) had already found by
    eye before ``present`` existed to bind them halfway. Seven further
    chunks that also name a `present` bystander with a pronoun in the line
    (32, 33, 54, 70, 71, 72, 75) stay silent, correctly: each one also uses
    at least one pronoun the evidence attributes to the default focus, so
    those shots really are about the singer with a bystander alongside, not
    a bystander alone.

    Silent whenever the evidence is too thin to trust: no ``present``, no
    third-person pronoun in the line, ``subject`` already set (nothing left
    to notice), a voiced chunk (this lint's whole premise is the
    instrumental fallback), or fewer than two solo-voiced occurrences of a
    pronoun anywhere in the plan (see :func:`_pronoun_ownership`). Warning
    tier, like every lint in this module: a false positive on prose a human
    wrote deliberately must never be able to block a run.
    """
    if not plan or not chunks:
        return
    ownership = _pronoun_ownership(_solo_voiced_pronoun_counts(plan, chunks))
    if not ownership:
        return
    for chunk in chunks:
        if not chunk.is_instrumental:
            continue
        entry = plan.get(chunk.chunk_id)
        if entry is None or entry.subject is not None or not entry.present:
            continue
        pronouns = {
            match.group(0).lower()
            for match in _THIRD_PERSON_PRONOUN_PATTERN.finditer(entry.shot)
        }
        owners = {ownership[p] for p in pronouns if p in ownership}
        if not owners or default_lead_vocalist in owners:
            continue
        subjects = owners & set(entry.present)
        if not subjects:
            continue
        implied = sorted(subjects)[0]
        logger.warning(
            "Shot plan chunk_id=%d: this instrumental chunk's shot line reads as "
            "entirely about %s (present=%s), but nothing tells the render that -- the "
            "composed focus still falls back to default_lead_vocalist=%r, who has no "
            "pronoun in this line at all (issue #82). Set subject = %r, or reword so "
            "the line is genuinely about %r too.",
            chunk.chunk_id,
            implied,
            list(entry.present),
            default_lead_vocalist,
            implied,
            default_lead_vocalist,
        )


ENTRY_KEYS = frozenset(
    {
        "chunk_id", "start", "shot", "focus", "length_seconds", "camera", "present",
        "location", "subject",
    }
)
"""Every key this module actually reads out of a ``[[shot]]`` table."""

PROVENANCE_ENTRY_KEYS = frozenset({"generated_by", "content_sha256"})
"""Keys the authoring layer writes into an entry and the renderer ignores by
design (issue #54 design section 8). Provenance lives in the file rather than
in a sidecar precisely *because* nothing here reads it -- a sidecar gets
separated from the plan the first time someone copies it. Named here so
:func:`_warn_unknown_entry_keys` does not report the project's own convention
as a typo."""


def _warn_unknown_entry_keys(raw: dict, chunk_id: object, path: Path) -> None:
    """Warn about a ``[[shot]]`` key nothing reads (issue #54 design section 6).

    ``_parse_entry`` reads only the keys it knows, so ``camara = "..."`` or
    ``lenght_seconds = 9`` used to be dropped in silence. That was tolerable
    while every key was typed by the person who would notice it had no effect;
    it is not once a model is emitting them and a human is skimming the diff.

    A warning rather than an error, deliberately: unknown keys are how
    provenance and anything else the renderer ignores get to live in the file
    at all, so raising here would make the plan format closed in a way this
    project has chosen not to make it.
    """
    unknown = sorted(set(raw) - ENTRY_KEYS - PROVENANCE_ENTRY_KEYS)
    if not unknown:
        return
    logger.warning(
        "Shot plan %s: chunk_id=%s has key(s) nothing reads: %s. Nothing here fails, but "
        "whatever they were meant to do is not happening -- check for a typo against %s.",
        path,
        chunk_id,
        unknown,
        sorted(ENTRY_KEYS),
    )


def _parse_entry(
    raw: object, index: int, path: Path, cast_names: Iterable[str] = ()
) -> ShotPlanEntry:
    if not isinstance(raw, dict):
        raise ShotPlanError(f"shot plan {path}: [[shot]] #{index} is not a table")

    for field in ("chunk_id", "start", "shot"):
        if field not in raw:
            logger.error("Shot plan %s: [[shot]] #%d is missing %r", path, index, field)
            raise ShotPlanError(f"shot plan {path}: [[shot]] #{index} is missing {field!r}")

    chunk_id = raw["chunk_id"]
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise ShotPlanError(
            f"shot plan {path}: [[shot]] #{index} chunk_id must be an integer, "
            f"got {chunk_id!r}"
        )

    start = raw["start"]
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        raise ShotPlanError(
            f"shot plan {path}: [[shot]] #{index} start must be a number, got {start!r}"
        )

    shot = raw["shot"]
    if not isinstance(shot, str):
        logger.error("Shot plan %s: chunk_id=%s has non-string shot text %r", path, chunk_id, shot)
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has shot={shot!r}; it must be a string "
            '(blank -- shot = "" -- is fine and means "not authored yet", issue #52)'
        )

    _warn_unknown_entry_keys(raw, chunk_id, path)

    return ShotPlanEntry(
        chunk_id=int(chunk_id),
        start=float(start),
        shot=shot.strip(),
        subject_is_focus=_parse_focus(raw, chunk_id, path),
        length_seconds=_parse_length_seconds(raw, chunk_id, path),
        camera=_parse_camera(raw, chunk_id, path),
        present=_parse_present(raw, chunk_id, path),
        location=_parse_location(raw, chunk_id, path),
        subject=_parse_subject(raw, chunk_id, path, cast_names),
    )


def _parse_subject(
    raw: dict, chunk_id: object, path: Path, cast_names: Iterable[str] = ()
) -> str | None:
    """Read the optional ``subject`` field: whose shot this is (issue #82).

    Not :func:`_parse_present` -- that reads a *list*; a shot has exactly one
    subject, so a list here is the wrong shape, not merely a longer one. An
    error rather than a warning when malformed, the same rule ``present``
    follows and for the same reason: silently dropping this renders exactly
    the bug it exists to fix (a config default standing in for a stated
    focus), with nothing to show for the direction that was written.

    Unlike ``present`` (which defers its cast check to ``prompting``, the
    module that actually has the cast in scope), an unknown name is checked
    HERE, because ``load_shot_plan`` already has ``cast_names`` in scope and
    there is no "may also turn out to be a singer" ambiguity to defer for --
    a shot has one subject, named once. Skipped when ``cast_names`` is
    empty, exactly as :func:`_lint_setting_consistency`'s own cast check is:
    the "cast unknown" call sites (most tests, and any caller that has not
    wired the cast dict through) must not be forced to supply one just to
    parse a plan.
    """
    subject = raw.get("subject")
    if subject is None:
        return None
    if not isinstance(subject, str) or not subject.strip():
        logger.error(
            "Shot plan %s: chunk_id=%s has subject=%r, which must be a single non-blank "
            "cast name (not a list, unlike present)",
            path,
            chunk_id,
            subject,
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has subject={subject!r}; it must be a "
            'single cast name, e.g. subject = "Jan" -- not a list, unlike present'
        )
    name = subject.strip()
    known = {n.strip() for n in cast_names if n and n.strip()}
    if known and name not in known:
        logger.error(
            "Shot plan %s: chunk_id=%s has subject=%r, which is not in the known cast %s",
            path,
            chunk_id,
            name,
            sorted(known),
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has subject={name!r}, which is not a "
            f"known cast member; known cast: {sorted(known)}"
        )
    return name


def _parse_present(raw: dict, chunk_id: object, path: Path) -> tuple[str, ...]:
    """Read the optional ``present`` list of cast names (issue #59).

    An error rather than a warning when malformed, unlike ``camera``: the
    whole point of the field is that somebody is meant to appear on screen
    with an identity attached, and silently dropping it renders exactly the
    bug it exists to fix -- an unconditioned stranger -- with nothing to show
    for the direction that was written. Names are *not* checked against the
    cast here; this module never sees the cast. ``prompting`` raises on an
    unknown name, where the cast is in scope.
    """
    present = raw.get("present")
    if present is None:
        return ()
    if isinstance(present, str) or not isinstance(present, (list, tuple)):
        logger.error(
            "Shot plan %s: chunk_id=%s has present=%r, which is not a list of names",
            path,
            chunk_id,
            present,
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has present={present!r}; it must be a "
            'list of cast names, e.g. present = ["Jan"]'
        )
    names = []
    for name in present:
        if not isinstance(name, str) or not name.strip():
            logger.error(
                "Shot plan %s: chunk_id=%s has a non-string entry in present: %r",
                path,
                chunk_id,
                name,
            )
            raise ShotPlanError(
                f"shot plan {path}: chunk_id={chunk_id} has {name!r} in present; every "
                "entry must be a non-blank cast name"
            )
        names.append(name.strip())
    return tuple(names)


def _parse_camera(raw: dict, chunk_id: object, path: Path) -> str | None:
    """Read the optional ``camera`` field (issue #53). Absent/blank means no
    camera direction -- never a fabricated default, same convention as
    ``length_seconds``/``focus``."""
    camera = raw.get("camera")
    if camera is None:
        return None
    if not isinstance(camera, str):
        logger.error(
            "Shot plan %s: chunk_id=%s has camera=%r, which is not a string", path, chunk_id, camera
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has camera={camera!r}; it must be a string"
        )
    return camera.strip() or None


def _parse_location(raw: dict, chunk_id: object, path: Path) -> str | None:
    """Read the optional ``location`` field (issue #78). Absent/blank means
    "not authored" -- never a fabricated default, same convention as
    ``camera``. Not validated against any closed vocabulary here: this
    module never sees the concept that defines one, and a hand-written plan
    is free to use the field or not at all. The generation-time closed-set
    check lives in ``authoring/beats.py``, where the vocabulary actually is."""
    location = raw.get("location")
    if location is None:
        return None
    if not isinstance(location, str):
        logger.error(
            "Shot plan %s: chunk_id=%s has location=%r, which is not a string",
            path,
            chunk_id,
            location,
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has location={location!r}; it must be "
            "a string"
        )
    return location.strip() or None


def _parse_length_seconds(raw: dict, chunk_id: object, path: Path) -> float | None:
    """Read the optional ``length_seconds`` (issue #27).

    Absent means "no editorial opinion", which is not the same as zero and is
    the only value that leaves slicing untouched. A malformed one is an error
    rather than a warning: silently ignoring it would render the whole video
    at the cadence the author was trying to break, with nothing to show for
    the direction they wrote.
    """
    if "length_seconds" not in raw:
        return None

    value = raw["length_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.error(
            "Shot plan %s: chunk_id=%s has length_seconds=%r, which is not a number",
            path,
            chunk_id,
            value,
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has length_seconds={value!r}; it must "
            "be a number of seconds"
        )

    value = float(value)
    if value <= 0:
        logger.error(
            "Shot plan %s: chunk_id=%s has length_seconds=%.3f", path, chunk_id, value
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has length_seconds={value}; a shot "
            "length must be greater than zero. Omit the field entirely to leave this "
            "chunk's length to the slicer"
        )
    return value


FOCUS_VALUES = {"subject": True, "action": False}
"""Accepted ``focus`` values. ``subject`` (default) keeps the performer as
what the shot is about; ``action`` hands that role to the shot's own
description, for consequence beats."""


def _parse_focus(raw: dict, chunk_id: object, path: Path) -> bool:
    focus = raw.get("focus")
    if focus is None:
        return True
    if not isinstance(focus, str) or focus.strip().lower() not in FOCUS_VALUES:
        logger.error(
            "Shot plan %s: chunk_id=%s has focus=%r; valid values are %s",
            path,
            chunk_id,
            focus,
            ", ".join(sorted(FOCUS_VALUES)),
        )
        raise ShotPlanError(
            f"shot plan {path}: chunk_id={chunk_id} has focus={focus!r}. Valid values "
            f"are {', '.join(sorted(FOCUS_VALUES))} -- a typo here would silently mean "
            "'subject' and the consequence would keep losing to the performer"
        )
    return FOCUS_VALUES[focus.strip().lower()]


def _resolve_entry(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> ShotPlanEntry | None:
    """Shared lookup + drift check behind :func:`resolve_shot` and
    :func:`resolve_camera` (issue #53) -- both resolve the same entry against
    the same chunk, and a plan that has drifted must refuse both the same
    way, not just whichever one happened to be called.

    Raises :class:`ShotPlanDriftError` when an entry exists but was authored
    against a materially different chunk start -- see the module docstring on
    why that is an error and a missing entry is only a warning.
    """
    if not plan:
        return None

    entry = plan.get(chunk.chunk_id)
    if entry is None:
        logger.warning(
            "Shot plan has no entry for chunk_id=%d (%.3fs-%.3fs); falling back to the "
            "global narrative_concept for this chunk.",
            chunk.chunk_id,
            chunk.start,
            chunk.end,
        )
        return None

    drift = abs(chunk.start - entry.start)
    if drift > START_TOLERANCE_SECONDS:
        logger.error(
            "Shot plan drift on chunk_id=%d: authored against start=%.3fs but this run's "
            "chunk starts at %.3fs (drift %.3fs > %.3fs tolerance). The alignment has "
            "changed since the plan was written, so its chunk numbering no longer refers "
            "to the same moments of the song.",
            chunk.chunk_id,
            entry.start,
            chunk.start,
            drift,
            START_TOLERANCE_SECONDS,
        )
        raise ShotPlanDriftError(
            f"shot plan drift on chunk_id={chunk.chunk_id}: authored against "
            f"start={entry.start:.3f}s but this run's chunk starts at "
            f"{chunk.start:.3f}s (drift {drift:.3f}s). Re-author the plan against the "
            "current alignment, or restore the lyrics/model the plan was written for"
        )

    return entry


def resolve_shot(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> str | None:
    """The authored direction for ``chunk``, or ``None`` to fall back to
    ``config.narrative_concept``.

    ``None`` also covers an entry whose ``shot`` is blank (issue #52) -- an
    unfilled skeleton line, warned about exactly like a missing entry, since
    composing an empty direction would give the prompt a hole rather than a
    sensible fallback.

    Raises :class:`ShotPlanDriftError` via :func:`_resolve_entry` when an
    entry exists but was authored against a materially different chunk start.
    """
    entry = _resolve_entry(plan, chunk)
    if entry is None:
        return None

    if not entry.shot.strip():
        logger.warning(
            "Shot plan chunk_id=%d has an unfilled shot line (shot = \"\"); falling back "
            "to the global narrative_concept for this chunk (issue #52).",
            chunk.chunk_id,
        )
        return None

    return entry.shot


def resolve_camera(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> str | None:
    """This chunk's authored camera direction (issue #53), or ``None`` when
    the plan has no entry, the entry never set ``camera``, or (via
    :func:`_resolve_entry`) the plan itself is absent.

    Deliberately independent of :func:`resolve_shot`: camera direction can
    apply to a chunk even when its ``shot`` line is blank and the concept
    falls back to ``config.narrative_concept``. Shares
    :func:`_resolve_entry`'s drift check, so a stale plan refuses both the
    same way rather than silently applying a camera direction authored
    against audio that has since moved."""
    entry = _resolve_entry(plan, chunk)
    return entry.camera if entry is not None else None


def resolve_present(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> tuple[str, ...]:
    """This chunk's on-screen-but-not-singing cast (issue #59), or ``()``
    when the plan has no entry for it or the plan itself is absent.

    Empty is the honest answer for an unauthored chunk, and composes exactly
    as every chunk did before the field existed. Shares
    :func:`_resolve_entry`'s drift check for the same reason
    :func:`resolve_camera` does."""
    entry = _resolve_entry(plan, chunk)
    return entry.present if entry is not None else ()


def resolve_subject(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> str | None:
    """Whose shot this is (issue #82), or ``None`` when the plan has no
    entry for it, the entry never set ``subject``, or (via
    :func:`_resolve_entry`) the plan itself is absent.

    Not :func:`resolve_present`: ``present`` answers who a pronoun in the
    shot line binds to; this answers who the render composes as the focus of
    the shot. ``None`` is the honest answer for an unauthored chunk, and
    composes exactly as every instrumental chunk did before this field
    existed (the ``default_lead_vocalist`` fallback, unchanged). Shares
    :func:`_resolve_entry`'s drift check for the same reason
    :func:`resolve_camera`/:func:`resolve_present` do."""
    entry = _resolve_entry(plan, chunk)
    return entry.subject if entry is not None else None


def resolve_location(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk
) -> str | None:
    """This chunk's authored location tag (issue #78), or ``None`` when the
    plan has no entry for it, the entry never set ``location``, or (via
    :func:`_resolve_entry`) the plan itself is absent.

    Shares :func:`_resolve_entry`'s drift check for the same reason
    :func:`resolve_camera`/:func:`resolve_present` do: a stale plan must
    refuse every field the same way, not just the ones the render loop
    consumes directly. ``location`` is never composed into a prompt -- this
    exists only so the two lints below have something to compare."""
    entry = _resolve_entry(plan, chunk)
    return entry.location if entry is not None else None


# --------------------------------------------------------------------------- #
# Skeleton generation for --prepare (issue #52)
# --------------------------------------------------------------------------- #


def render_shot_plan_skeleton(
    chunks: Sequence[AudioChunk], *, source: str, generated_at: str
) -> str:
    """Compose the *text* of a shot_plan.toml skeleton for ``chunks``.

    Pure string composition, no file I/O -- mirrors how ``prompting.py``
    keeps composing text separate from anything happening to it afterwards.
    ``chunk_id`` and ``start`` come straight from ``chunks``, so a plan loaded
    back against the very chunks it was generated from can never drift
    (:class:`ShotPlanDriftError` only fires once the alignment underneath it
    changes). Every ``shot`` is emitted blank (``shot = ""``) for the author
    to fill in -- :func:`load_shot_plan`/:func:`resolve_shot` treat that
    exactly like no entry at all, so a half-finished skeleton is still
    loadable (issue #52).

    ``source`` and ``generated_at`` are caller-supplied (the run config path,
    and a date string) rather than read from the filesystem or the clock
    here, keeping this function a pure function of its arguments like every
    other composer in this module.
    """
    header = (
        f"# Generated by --prepare from {source} on {generated_at}. Anchors come from\n"
        "# the alignment; edit the shot lines, not the chunk_id/start values.\n"
    )
    blocks = []
    for chunk in chunks:
        duration = chunk.end - chunk.start
        frames = f", {chunk.frame_count} frames" if chunk.frame_count is not None else ""
        lines = [
            "[[shot]]",
            f"chunk_id = {chunk.chunk_id}",
            f"start = {chunk.start!r}"
            f"   # {chunk.start:.3f} - {chunk.end:.3f}  ({duration:.3f}s{frames})",
        ]
        if chunk.is_instrumental or not chunk.text.strip():
            lines.append("# INSTRUMENTAL -- no lyric to sing")
        else:
            lyric = " ".join(chunk.text.strip().split()).replace('"', "'")
            lines.append(f'# lyric: "{lyric}"')
        lines.append('shot = ""')
        blocks.append("\n".join(lines))
    return header + "\n" + "\n\n".join(blocks) + "\n"


def write_shot_plan_skeleton(
    chunks: Sequence[AudioChunk],
    output_path: str | Path,
    *,
    source: str,
    generated_at: str,
    force: bool = False,
) -> Path:
    """Write a shot_plan.toml skeleton for ``chunks`` to ``output_path``.

    Refuses to overwrite an existing file unless ``force=True``: an authored
    shot plan is real work an author may already have started, and
    ``--prepare`` must never silently discard it -- see the module docstring
    on why an unfilled skeleton and a partially-authored plan are treated the
    same way everywhere else in this module.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        logger.error(
            "Refusing to overwrite existing shot plan at %s (issue #52) -- pass "
            "force=True / --force to overwrite it",
            output_path,
        )
        raise ShotPlanError(
            f"{output_path} already exists -- an authored shot plan is real work; pass "
            "--force to overwrite it"
        )

    text = render_shot_plan_skeleton(chunks, source=source, generated_at=generated_at)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
    except OSError as exc:
        logger.exception("Failed to write shot plan skeleton to %s", output_path)
        raise ShotPlanError(
            f"could not write shot plan skeleton to {output_path}: {exc}"
        ) from exc

    logger.info("Wrote shot plan skeleton (%d chunk(s)) to %s", len(chunks), output_path)
    return output_path


__all__ = [
    "lint_camera_face_away_on_voiced_chunks",
    "lint_instrumental_focus_mismatch",
    "lint_present_location_mismatch",
    "lint_role_prohibition_contradiction",
    "lint_shots_against_lyrics",
    "lint_subject_on_voiced_chunk",
    "lint_unbound_companion_referent",
    "ENTRY_KEYS",
    "LANDMARK_CONTRADICTION_WINDOW_SECONDS",
    "MEASURED_MAX_FRAMES",
    "PROVENANCE_ENTRY_KEYS",
    "START_TOLERANCE_SECONDS",
    "TRAINED_MAX_SECONDS",
    "TRAINED_MIN_SECONDS",
    "ShotLength",
    "ShotPlanDriftError",
    "ShotPlanEntry",
    "ShotPlanError",
    "load_shot_plan",
    "render_shot_plan_skeleton",
    "resolve_camera",
    "resolve_location",
    "resolve_shot",
    "resolve_subject",
    "shot_length_requests",
    "write_shot_plan_skeleton",
]
