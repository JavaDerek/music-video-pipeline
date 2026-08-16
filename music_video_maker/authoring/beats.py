"""Stage 2 -- beats (issue #54 design section 4).

Reads the concept and the chunk skeleton and says *what happens* in each
chunk: how shot 12 pays off shot 4, which chunks are consequence beats, what
fills a 36-second solo. Structural and cross-shot, which is why it runs on
:data:`~music_video_maker.authoring.driver.MODEL_OPUS` and why it is worth its
own call rather than being folded into the prose stage.

**The two fields that earn this stage its keep are ``beat_role`` and
``beat_group``.** They make three of ``docs/shot-writing-guide.md``'s
checklist items machine-checkable for the first time, *before* a word of prose
exists:

* every ``consequence`` has a ``contact`` and a ``plant`` earlier in the same
  group -- the three-beat rule, which the guide's whole evidence table is
  about;
* every ``consequence`` carries ``focus = "action"`` -- derived from structure
  rather than guessed from keywords, which is strictly better than what
  ``shot_plan._lint_consequence_focus`` can do from finished prose;
* no group is a lone ``consequence`` -- a cause and its effect compressed into
  one shot, the original defect.

Those run locally on the JSON and their failures feed straight back into this
stage's retry, so a structurally broken beat sheet is never handed to the
prose stage to make eloquent.

**Anchors are never read from the model.** ``chunk_id`` is matched against the
skeleton and anything else is dropped; ``start``/``end`` are re-emitted from
the skeleton every time. That is what makes ``ShotPlanDriftError``
structurally unreachable for a generated plan rather than merely unlikely.

Validation is hand-rolled for the same reason :mod:`.concept` gives: this
package adds no pip dependencies, and ``--json-schema`` is a hint to the
model, never the gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from music_video_maker.authoring.chunks import skeleton_table_text
from music_video_maker.authoring.driver import MODEL_OPUS, DriverResult, ModelDriver
from music_video_maker.authoring.hashing import sha256_file, sha256_text
from music_video_maker.authoring.prompts import SHOT_WRITING_GUIDE_DOC, beats_system_prompt
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk
from music_video_maker.shot_plan import ShotLength

logger = logging.getLogger(__name__)

BEAT_ROLES: tuple[str, ...] = ("plant", "contact", "consequence", "transition", "instrumental")
"""The beat vocabulary, from ``docs/shot-writing-guide.md``'s three-beat rule
plus the two roles a real song needs and the guide does not name: a
``transition`` that only moves the story between venues, and an
``instrumental`` beat filling an unvoiced span."""

FOCUS_SUBJECT = "subject"
FOCUS_ACTION = "action"
FOCUS_VALUES: tuple[str, ...] = (FOCUS_SUBJECT, FOCUS_ACTION)
"""Deliberately the same two strings ``shot_plan.FOCUS_VALUES`` accepts in the
TOML, so nothing has to translate between vocabularies on the way to the file
a human commits."""

MAX_VALIDATION_ATTEMPTS = 3

BEATS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["beats"],
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "chunk_id", "beat", "beat_role", "beat_group", "focus", "location", "act",
                ],
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "beat": {"type": "string", "minLength": 1},
                    "beat_role": {"enum": list(BEAT_ROLES)},
                    "beat_group": {"type": "integer", "minimum": 1},
                    "focus": {"enum": list(FOCUS_VALUES)},
                    "length_seconds": {"type": "number", "exclusiveMinimum": 0},
                    # Issue #78: where the active cast member is inside the
                    # run's `setting` at this beat. Required, and checked at
                    # generation time against the concept's own closed
                    # `locations` list (see `_parse_entries`) -- the schema
                    # hint below covers only "is this a string", the way
                    # `--json-schema` covers every other field here.
                    "location": {"type": "string", "minLength": 1},
                    # Issue #84: which of the concept's ordered `acts` this
                    # beat belongs to. Required the same way `location` is,
                    # and checked against the concept's closed vocabulary in
                    # `_parse_entries` -- never guessed, never left blank.
                    "act": {"type": "string", "minLength": 1},
                    # Issue #82: whose shot this is, on an INSTRUMENTAL beat
                    # only. Optional -- most beats leave it out entirely --
                    # and checked against both the cast and this chunk's
                    # voiced/instrumental status in `_parse_entries`, never
                    # just "is this a string" the way this schema hint is.
                    "subject": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


class BeatsValidationError(ValueError):
    """Raised when a model's beat sheet is malformed or structurally broken.

    Carries **every** problem it found, not the first: one retry round costs a
    whole model call, so a sheet with three faults must come back with three
    complaints.
    """


@dataclass(frozen=True)
class Beat:
    """One chunk's story beat.

    ``start``/``end`` are the chunk's span, always copied from the skeleton --
    see the module docstring. They are what :mod:`.reanchor` maps by when a
    ``length_seconds`` re-cuts the timeline underneath these anchors.
    """

    chunk_id: int
    start: float
    end: float
    beat: str
    beat_role: str
    beat_group: int
    location: str
    """Where the active cast member is inside the run's ``setting`` at this
    beat (issue #78), e.g. ``"switchback"`` or ``"mill"``. Required, like
    ``beat_role`` -- ``setting`` fixes the world's geography for the whole
    video, but says nothing about where a character is inside it at a given
    moment, and this is the beats stage's answer, checkable the same way
    ``beat_role``/``beat_group`` are: structurally, before any prose exists.
    Assigned from the concept's closed ``locations`` list and validated
    against it in :func:`_parse_entries` -- never guessed from finished shot
    text (see ``shot_plan.py``'s "a lint that guesses loses to a stage that
    knows")."""
    act: str = ""
    """Which of the concept's ordered ``acts`` (issue #84) this beat belongs
    to, e.g. ``"situation"`` or ``"resolution"``. Assigned from the concept's
    closed act vocabulary and validated against it in :func:`_parse_entries`,
    the same treatment ``location`` gets. Defaults to ``""`` -- unlike
    ``location``, this field is given a dataclass default rather than made
    positionally required: a persisted pre-#84 beat sheet has no ``act`` key
    at all, and ``""`` is the same "not authored" default already used for an
    absent optional field elsewhere in this class, not a fabricated act. The
    actual gate is :data:`BEATS_SCHEMA` plus :func:`_parse_entries`, which
    require a real, non-empty ``act`` on every entry a *fresh* model reply
    produces -- the dataclass default only covers reading old data back in."""
    subject: str | None = None
    """Whose shot an INSTRUMENTAL beat is (issue #82) -- the field that
    replaces the render's ``default_lead_vocalist`` fallback (composed as
    "is the focus of this shot") when a beat on a chunk nobody sings is
    actually about someone else. ``None`` by default, unlike ``location``/
    ``act``'s ``""`` sentinel: those two are REQUIRED on every beat and
    ``""`` means "an old sheet predating the field", whereas ``subject`` is
    OPTIONAL on every beat and absent on almost all of them, so ``None`` --
    the same "not stated" default :attr:`length_seconds` already uses -- is
    the honest value, not a fabricated placeholder standing in for a real
    field the model chose not to set.

    Emitted by THIS stage rather than inferred later from pronouns, per the
    issue's own instruction: "the beats stage already knows whose beat it
    is". Legal only when this beat's chunk is instrumental -- checked in
    :func:`_parse_entries`, which has ``chunks`` in scope and so does not
    have to guess -- and refused by the render's own
    :func:`~music_video_maker.shot_plan.lint_subject_on_voiced_chunk` a
    second time as defence in depth. Not
    :attr:`~music_video_maker.shot_plan.ShotPlanEntry.subject_is_focus`
    (issue #26): that is a boolean asking whether the focus member is the
    sentence's grammatical subject; this says WHO the focus member is."""
    focus: str = FOCUS_SUBJECT
    length_seconds: float | None = None
    merged_from: tuple[int, ...] = ()
    """The pre-re-anchor chunk ids that merged into this beat, or ``()`` when
    nothing merged. Audit trail only -- never an index into anything, the same
    rule ``ShotLength.source_chunk_id`` follows."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "beat": self.beat,
            "beat_role": self.beat_role,
            "beat_group": self.beat_group,
            "location": self.location,
            "act": self.act,
            "subject": self.subject,
            "focus": self.focus,
            "length_seconds": self.length_seconds,
            "merged_from": list(self.merged_from),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Beat:
        return cls(
            chunk_id=int(payload["chunk_id"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            beat=str(payload["beat"]),
            beat_role=str(payload["beat_role"]),
            beat_group=int(payload["beat_group"]),
            # Pre-#78 persisted beat sheets have no `location` key at all;
            # "" is the same "not authored" default `shot_plan.ShotPlanEntry`
            # already uses for an absent optional field, not a fabricated
            # place.
            location=str(payload.get("location") or ""),
            # Pre-#84 persisted beat sheets have no `act` key at all -- same
            # "not authored" default as `location` above.
            act=str(payload.get("act") or ""),
            # Pre-#82 persisted beat sheets have no `subject` key at all;
            # `None` -- not `""` -- is the correct "not stated" value here,
            # see the field's own docstring.
            subject=(None if payload.get("subject") is None else str(payload["subject"])),
            focus=str(payload.get("focus", FOCUS_SUBJECT)),
            length_seconds=(
                None if payload.get("length_seconds") is None else float(payload["length_seconds"])
            ),
            merged_from=tuple(int(v) for v in payload.get("merged_from", ())),
        )


@dataclass(frozen=True)
class BeatsResult:
    """One successful beat-sheet generation, before any re-anchoring."""

    beats: tuple[Beat, ...]
    driver_result: DriverResult
    input_hashes: dict[str, str]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _canonical_vocabulary(values: Sequence[str]) -> dict[str, str]:
    """``{normalized: as-written}`` for a closed vocabulary (``locations``,
    issue #78; ``acts``, issue #84), keyed by a case/whitespace-insensitive
    form so a model that varies capitalization across beats ("Mill" vs
    "mill") does not fragment one entry into two in the written plan."""
    return {v.strip().lower(): v.strip() for v in values if v and v.strip()}


def _parse_entries(
    data: object,
    chunks: Sequence[AudioChunk],
    problems: list[str],
    locations: Sequence[str] = (),
    acts: Sequence[str] = (),
    cast_names: Sequence[str] = (),
) -> tuple[Beat, ...]:
    """Shape-check each entry and pin it to a real chunk. Appends to
    ``problems`` rather than raising, so one round reports everything.

    ``locations`` is the concept's closed vocabulary (issue #78, empty means
    no vocabulary supplied -- e.g. a pre-#78 concept, or a caller not using
    the field -- in which case any non-empty ``location`` string is accepted
    structurally and nothing is checked against a set). When non-empty, a
    beat's ``location`` outside it is a validation problem sent back to the
    model, the same "a generated anchor is never trusted" treatment
    ``chunk_id`` gets above: never silently accepted, never silently
    dropped.

    ``acts`` is issue #84's closed, ordered vocabulary for ``act`` -- the
    same "empty means no vocabulary supplied" convention ``locations``
    follows.

    ``cast_names`` is issue #82's known-cast set for ``subject``, exactly
    the same "empty means no vocabulary supplied" degradation -- a caller
    that has not threaded the cast through (most tests) must not be blocked
    from parsing at all."""
    canonical = _canonical_vocabulary(locations)
    canonical_acts = _canonical_vocabulary(acts)
    known_cast = {n.strip() for n in cast_names if n and n.strip()}
    if not isinstance(data, dict):
        raise BeatsValidationError(
            f"beats reply must be a JSON object, got {type(data).__name__}"
        )
    raw = data.get("beats")
    if not isinstance(raw, list):
        raise BeatsValidationError(
            "beats reply must carry a 'beats' array; got "
            f"{sorted(data) if isinstance(data, dict) else data!r}"
        )

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    beats: list[Beat] = []
    seen: set[int] = set()

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            problems.append(f"beats[{index}] is not an object")
            continue

        chunk_id = entry.get("chunk_id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            problems.append(f"beats[{index}] chunk_id must be an integer, got {chunk_id!r}")
            continue
        if chunk_id not in by_id:
            # Dropped, not an error: an invented anchor must not be able to
            # reach disk, and whatever real chunk it displaced is caught by
            # the coverage check below with a message the model can act on.
            logger.warning(
                "Beat sheet names chunk_id=%d, which is not in this song's skeleton "
                "(%d chunk(s), %d-%d); dropping it.",
                chunk_id,
                len(chunks),
                min(by_id, default=0),
                max(by_id, default=0),
            )
            continue
        if chunk_id in seen:
            problems.append(f"chunk_id={chunk_id} appears more than once")
            continue
        seen.add(chunk_id)

        beat_text = entry.get("beat")
        if not isinstance(beat_text, str) or not beat_text.strip():
            problems.append(f"chunk_id={chunk_id} beat must be a non-empty string")
            continue

        role = entry.get("beat_role")
        if role not in BEAT_ROLES:
            problems.append(
                f"chunk_id={chunk_id} beat_role={role!r} is not one of {list(BEAT_ROLES)}"
            )
            continue

        group = entry.get("beat_group")
        if isinstance(group, bool) or not isinstance(group, int):
            problems.append(f"chunk_id={chunk_id} beat_group must be an integer, got {group!r}")
            continue

        focus = entry.get("focus", FOCUS_SUBJECT)
        if focus not in FOCUS_VALUES:
            problems.append(
                f"chunk_id={chunk_id} focus={focus!r} is not one of {list(FOCUS_VALUES)}"
            )
            continue

        length = entry.get("length_seconds")
        if length is not None:
            if isinstance(length, bool) or not isinstance(length, (int, float)) or length <= 0:
                problems.append(
                    f"chunk_id={chunk_id} length_seconds={length!r} must be a number of "
                    "seconds greater than zero, or omitted entirely"
                )
                continue
            length = float(length)

        location = entry.get("location")
        if not isinstance(location, str) or not location.strip():
            problems.append(f"chunk_id={chunk_id} location must be a non-empty string")
            continue
        if canonical:
            resolved = canonical.get(location.strip().lower())
            if resolved is None:
                problems.append(
                    f"chunk_id={chunk_id} location={location!r} is not one of this song's "
                    f"approved locations {sorted(canonical.values())}. Every location a "
                    "chunk uses has to be on that list -- pick the closest match, or say "
                    "in your reply why a new one is needed so the concept can be revised"
                )
                continue
            location = resolved
        else:
            location = location.strip()

        act = entry.get("act")
        if not isinstance(act, str) or not act.strip():
            problems.append(f"chunk_id={chunk_id} act must be a non-empty string")
            continue
        if canonical_acts:
            resolved_act = canonical_acts.get(act.strip().lower())
            if resolved_act is None:
                problems.append(
                    f"chunk_id={chunk_id} act={act!r} is not one of this song's approved "
                    f"acts {sorted(canonical_acts.values())}. Every beat's `act` has to be "
                    "exactly one of the concept's ordered acts -- pick the closest match, "
                    "or say in your reply why the concept's act structure needs revising"
                )
                continue
            act = resolved_act
        else:
            act = act.strip()

        chunk = by_id[chunk_id]

        subject = entry.get("subject")
        if subject is not None:
            if not isinstance(subject, str) or not subject.strip():
                problems.append(
                    f"chunk_id={chunk_id} subject={subject!r} must be a single non-blank "
                    "cast name naming whose shot this is, or omitted entirely"
                )
                continue
            subject = subject.strip()
            if known_cast and subject not in known_cast:
                problems.append(
                    f"chunk_id={chunk_id} subject={subject!r} is not one of this song's "
                    f"cast members {sorted(known_cast)}"
                )
                continue
            if not chunk.is_instrumental:
                problems.append(
                    f"chunk_id={chunk_id} sets subject={subject!r} but this chunk is "
                    "voiced -- the singer already owns the frame on a voiced chunk "
                    "(issues #58, #59, #60); `subject` is legal only on an instrumental "
                    "chunk"
                )
                continue

        beats.append(
            Beat(
                chunk_id=chunk_id,
                start=chunk.start,
                end=chunk.end,
                beat=beat_text.strip(),
                beat_role=role,
                beat_group=group,
                location=location,
                act=act,
                subject=subject,
                focus=focus,
                length_seconds=length,
            )
        )

    return tuple(sorted(beats, key=lambda b: b.start))


def check_beat_structure(
    beats: Sequence[Beat], chunks: Sequence[AudioChunk], problems: list[str]
) -> None:
    """The three-beat rule and coverage, as machine checks.

    Pure and side-effect free apart from appending to ``problems`` -- this is
    the half of Stage 2 that has to be right, and it is much easier to trust
    when it can be exercised on a hand-built list with no model anywhere near
    it.
    """
    directed = {beat.chunk_id for beat in beats}
    missing = sorted(chunk.chunk_id for chunk in chunks if chunk.chunk_id not in directed)
    if missing:
        problems.append(
            f"these chunks have no beat: {missing}. Every chunk needs one -- an "
            "undirected chunk falls back to the global narrative_concept, which renders "
            "as something deliberate-looking that nobody authored"
        )

    ordered = sorted(beats, key=lambda b: b.start)
    group_sizes: dict[int, int] = {}
    for beat in ordered:
        group_sizes[beat.beat_group] = group_sizes.get(beat.beat_group, 0) + 1

    for position, beat in enumerate(ordered):
        if beat.beat_role != "consequence":
            continue

        if beat.focus != FOCUS_ACTION:
            problems.append(
                f'chunk_id={beat.chunk_id} is a consequence but does not set focus = "action". '
                "Without it the composed 'X is the focus of this shot' clause competes with "
                "the consequence, and across four renders the performer won"
            )

        if group_sizes[beat.beat_group] == 1:
            problems.append(
                f"chunk_id={beat.chunk_id} is a consequence on its own in beat_group="
                f"{beat.beat_group}: one shot carrying both a cause and its effect. Split it "
                "into plant / contact / consequence across consecutive chunks"
            )
            continue

        earlier = {
            other.beat_role
            for other in ordered[:position]
            if other.beat_group == beat.beat_group
        }
        for required in ("plant", "contact"):
            if required not in earlier:
                problems.append(
                    f"chunk_id={beat.chunk_id} is a consequence with no {required} earlier in "
                    f"beat_group={beat.beat_group}. Causation reads when it is given room and "
                    "fails when compressed"
                )


def check_act_structure(
    beats: Sequence[Beat], acts: Sequence[str], problems: list[str]
) -> None:
    """Issue #84's four mechanical arc checks, all independent and all
    appended to ``problems`` rather than raised on the first -- the same "one
    retry round costs a whole model call" convention :func:`check_beat_structure`
    already follows, one level up.

    ``acts`` is the concept's closed, ORDERED vocabulary. Empty means no
    vocabulary was supplied (a pre-#84 concept, or a caller not using the
    field) -- the same "empty means no check" convention ``locations``
    follows -- so this function does nothing at all in that case rather than
    inventing an order to check beats against.

    The four checks, each independently testable and independently failing:

    1. **Coverage** -- every declared act has at least one beat. An act named
       in the concept but never used is not part of this video's shape.
    2. **Order** -- the acts' first appearances, walking beats in song-time
       order, follow the concept's own stated sequence. This can fail even
       when every act's beats are individually contiguous (e.g. acts visited
       C, A, B when the concept declared A, B, C).
    3. **Contiguity** -- once a beat sequence leaves an act, that act must
       never come back. This can fail even when the *overall* order of first
       appearances is correct (e.g. A, B, A, C -- A's second appearance does
       not change when A/B/C were each *first* seen).
    4. **Payoff** -- the only real test of an arc, and the issue says so in
       as many words: the FINAL act must contain at least one ``consequence``
       beat whose ``beat_group`` also contains a ``plant`` from an EARLIER
       act. Skipped when there is only one act: with nothing earlier to plant
       from, the check would be unsatisfiable by construction rather than a
       real test of anything.
    """
    if not acts:
        return

    ordered = sorted(beats, key=lambda b: b.start)
    index = {name: position for position, name in enumerate(acts)}

    used = {beat.act for beat in ordered}
    empty_acts = [name for name in acts if name not in used]
    if empty_acts:
        problems.append(
            f"these act(s) from the concept's structure have no beat assigned to them: "
            f"{empty_acts}. Every act in {list(acts)} has to actually happen somewhere in "
            "the song, or it is not part of this video's shape"
        )

    # Collapse consecutive same-act beats into "runs" so order and
    # contiguity can be checked on the sequence of distinct blocks.
    runs: list[str] = []
    for beat in ordered:
        if not runs or runs[-1] != beat.act:
            runs.append(beat.act)

    seen: set[str] = set()
    reappeared: list[str] = []
    for act in runs:
        if act in seen and act not in reappeared:
            reappeared.append(act)
        seen.add(act)
    if reappeared:
        problems.append(
            f"act(s) {reappeared} reappear after a later act had already started. Each act's "
            "beats must form ONE unbroken run across the song -- an act that starts, stops "
            "for another act, then starts again is not one continuous movement of the story"
        )

    first_seen = list(dict.fromkeys(runs))
    positions = [index[act] for act in first_seen if act in index]
    if positions != sorted(positions):
        problems.append(
            f"the beats visit the acts in the order {first_seen}, which does not follow the "
            f"concept's stated order {list(acts)}. Acts must unfold in the order the concept "
            "declared them"
        )

    if len(acts) > 1:
        final_act = acts[-1]
        final_index = index[final_act]
        plant_groups_before_final = {
            beat.beat_group
            for beat in ordered
            if beat.beat_role == "plant" and index.get(beat.act, final_index) < final_index
        }
        payoff = any(
            beat.beat_role == "consequence" and beat.beat_group in plant_groups_before_final
            for beat in ordered
            if beat.act == final_act
        )
        if not payoff:
            problems.append(
                f"the final act ({final_act!r}) contains no consequence beat whose beat_group "
                "also has a plant in an earlier act. This is the only real test of an arc: "
                "something planted earlier has to be paid off at the end, not just a sequence "
                "of events"
            )


def validate_beats(
    data: object,
    chunks: Sequence[AudioChunk],
    locations: Sequence[str] = (),
    acts: Sequence[str] = (),
    cast_names: Sequence[str] = (),
) -> tuple[Beat, ...]:
    """Parse and check a model reply, or raise with everything that is wrong.

    ``locations`` is the concept's closed vocabulary for ``location`` (issue
    #78); ``acts`` is the concept's closed, ordered vocabulary for ``act``
    (issue #84); ``cast_names`` is the known cast for ``subject`` (issue
    #82). All three empty means no vocabulary was supplied and anything
    structurally valid is accepted. See :func:`_parse_entries` and
    :func:`check_act_structure`."""
    problems: list[str] = []
    beats = _parse_entries(data, chunks, problems, locations, acts, cast_names)
    check_beat_structure(beats, chunks, problems)
    check_act_structure(beats, acts, problems)
    if problems:
        raise BeatsValidationError(
            f"beat sheet has {len(problems)} problem(s):\n- " + "\n- ".join(problems)
        )
    return beats


# --------------------------------------------------------------------------- #
# Prompt + hashes
# --------------------------------------------------------------------------- #


def beats_input_hashes(
    config: RunConfig, chunks: Sequence[AudioChunk], concept: Mapping[str, Any]
) -> dict[str, str]:
    """Everything :func:`generate_beats` consumes, hashed.

    ``concept`` is in here on purpose: it is what makes staleness *cascade*
    (design section 7). Re-rolling the concept marks this stage stale rather
    than silently leaving a beat sheet that descends from a paragraph nobody
    approved -- reported, never auto-healed.
    """
    return {
        "skeleton": sha256_text(skeleton_table_text(chunks)),
        "concept": sha256_text(json.dumps(dict(concept), sort_keys=True)),
        "shot_writing_guide": sha256_file(SHOT_WRITING_GUIDE_DOC),
    }


def _reading_block(reading: Mapping[str, Any]) -> list[str]:
    """Issue #69 requirement 6: beats must actually SEE the reading, not just
    the concept's own invented pitch -- a field nothing downstream consumes
    is orphaned observability. Absent entirely for a pre-#69 concept (see
    :func:`_concept_block`'s caller), never fabricated."""
    lines = [
        "## What the song is actually about (read this before the beats below)",
        f"Subject: {reading.get('subject', '')}",
        f"Speaker -> addressee: {reading.get('speaker', '')} -> {reading.get('addressee', '')}",
        f"Situation: {reading.get('situation', '')}",
        f"What changes across the song: {reading.get('change', '')}",
        (
            f"Register/period/place: {reading.get('register', '')} / "
            f"{reading.get('period', '')} / {reading.get('place', '')}"
        ),
    ]
    nouns = reading.get("nouns") or []
    if nouns:
        lines.append("Concrete nouns the lyrics themselves name: " + "; ".join(map(str, nouns)))
    references = reading.get("references") or []
    if references:
        for ref in references:
            if not isinstance(ref, Mapping):
                continue
            imagery = "; ".join(str(i) for i in (ref.get("imagery") or []))
            lines.append(
                f"Reference -- {ref.get('reference', '')} ({ref.get('what_it_is', '')}): "
                f"{imagery}"
            )
    return lines


def _concept_block(concept: Mapping[str, Any]) -> str:
    parts = [
        f"Logline: {concept.get('logline', '')}",
        f"Setting: {concept.get('setting', '')}",
        f"Tone: {concept.get('tone', '')}",
    ]
    motifs = concept.get("motifs") or []
    avoid = concept.get("avoid") or []
    if motifs:
        parts.append("Motifs to plant and pay off: " + "; ".join(str(m) for m in motifs))
    if avoid:
        parts.append("Deliberately NOT on screen: " + "; ".join(str(a) for a in avoid))
    locations = concept.get("locations") or []
    if locations:
        parts.append(
            "Approved locations -- every beat's `location` MUST be exactly one of these: "
            + "; ".join(str(loc) for loc in locations)
        )
    acts = concept.get("acts") or []
    if acts:
        parts.append(
            "Approved acts, IN THIS ORDER -- every beat's `act` MUST be exactly one of these, "
            "the acts must run in this order across the song, and the final act must pay off "
            "something an earlier act planted: "
            + "; ".join(
                f"{a.get('name', '')} ({a.get('function', '')})"
                for a in acts
                if isinstance(a, Mapping)
            )
        )
    reading = concept.get("reading")
    if isinstance(reading, Mapping) and reading:
        parts.append("")
        parts.extend(_reading_block(reading))
    return "\n".join(parts)


def build_beats_prompt(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    concept: Mapping[str, Any],
    *,
    notes: str | None = None,
) -> str:
    """The Stage 2 user prompt: the approved concept, the whole chunk
    skeleton, the cast, and any freeform objection from the review loop."""
    voiced = sum(c.duration for c in chunks if not c.is_instrumental)
    instrumental = sum(c.duration for c in chunks if c.is_instrumental)
    cast_lines = "\n".join(f"- {name}: {member.role}" for name, member in config.cast.items())

    parts = [
        "## The approved concept (every beat below descends from this)",
        _concept_block(concept),
        "",
        "## The chunk skeleton -- these chunk_ids are the only ones that exist",
        (
            f"{len(chunks)} chunk(s), {voiced + instrumental:.1f}s total "
            f"({voiced:.1f}s voiced, {instrumental:.1f}s instrumental)"
        ),
        "chunk_id\tstart\tend\tframes\ttag\tsinger\tlyric",
        skeleton_table_text(chunks),
        "## Cast",
        cast_lines,
        (
            "Default lead vocalist -- who the render frames on an INSTRUMENTAL chunk "
            f"unless a beat's `subject` names someone else (issue #82): "
            f"{config.default_lead_vocalist}"
        ),
    ]
    if config.setting:
        parts += ["", f"## Setting (fixed for the whole video): {config.setting}"]
    if notes and notes.strip():
        parts += ["", f"## Notes from the person reviewing this: {notes.strip()}"]

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_beats(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    concept: Mapping[str, Any],
    driver: ModelDriver,
    *,
    notes: str | None = None,
    extra_instructions: str | None = None,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> BeatsResult:
    """Call the model, validate, and retry with the validator's own error
    folded into the prompt until it passes or attempts run out.

    ``extra_instructions`` is how :func:`~music_video_maker.authoring.reanchor.plan_beats`
    reports something this function cannot see -- that honouring the sheet's
    own ``length_seconds`` left a chunk with no direction -- without either
    module having to know the other's retry policy.
    """
    system = beats_system_prompt()
    prompt = build_beats_prompt(config, chunks, concept, notes=notes)
    if extra_instructions and extra_instructions.strip():
        prompt = f"{prompt}\n\n{extra_instructions.strip()}"
    # Issue #78: the closed vocabulary `location` is checked against, straight
    # from the approved concept -- never re-derived or guessed here.
    locations = tuple(str(loc) for loc in (concept.get("locations") or ()))
    # Issue #84: the closed, ORDERED act vocabulary, likewise straight from
    # the concept. A pre-#84 concept has no "acts" key at all -- `or ()`
    # degrades to "no vocabulary supplied" (see `check_act_structure`'s own
    # docstring), never a hard failure; that concept simply produced a beat
    # sheet with no arc structure to check, which is the same graceful
    # degradation `locations` already gets for a pre-#78 concept.
    acts = tuple(
        str(a["name"])
        for a in (concept.get("acts") or ())
        if isinstance(a, Mapping) and a.get("name")
    )
    if not acts:
        logger.warning(
            "This concept has no `acts` structure (issue #84) -- the beat sheet will have no "
            "story arc to check against. Re-run the concept stage to pick one up."
        )
    # Issue #82: the known cast for `subject`, straight from `config.cast` --
    # already in scope here, unlike `locations`/`acts` which have to come
    # from the concept because nothing else carries them.
    cast_names = tuple(config.cast)

    last_error: BeatsValidationError | None = None
    for attempt in range(1, max_validation_attempts + 1):
        result = driver.complete(
            system=system, prompt=prompt, model=MODEL_OPUS, schema=BEATS_SCHEMA
        )
        try:
            beats = validate_beats(result.data, chunks, locations, acts, cast_names)
        except BeatsValidationError as exc:
            last_error = exc
            logger.warning(
                "beat sheet failed validation (attempt %d/%d): %s",
                attempt,
                max_validation_attempts,
                exc,
            )
            prompt = (
                f"{prompt}\n\n"
                "## Your previous reply failed validation\n"
                f"{exc}\n"
                "Reply again with a single JSON object matching the required shape exactly, "
                "fixing every problem listed above."
            )
            continue
        logger.info(
            "Beat sheet accepted: %d beat(s) across %d group(s)",
            len(beats),
            len({b.beat_group for b in beats}),
        )
        return BeatsResult(
            beats=beats,
            driver_result=result,
            input_hashes=beats_input_hashes(config, chunks, concept),
        )

    logger.error(
        "beat generation failed validation after %d attempt(s): %s",
        max_validation_attempts,
        last_error,
    )
    raise BeatsValidationError(
        f"beat sheet failed validation after {max_validation_attempts} attempt(s): {last_error}"
    ) from last_error


def beat_length_requests(beats: Sequence[Beat]) -> tuple[ShotLength, ...]:
    """Project the sheet's ``length_seconds`` into anchored
    :class:`~music_video_maker.shot_plan.ShotLength` requests, sorted by
    anchor -- the exact unit ``slicing.slice_audio`` consumes.

    Mirrors ``shot_plan.shot_length_requests`` deliberately rather than
    reusing it: that one projects a *loaded TOML plan*, this one a beat sheet
    that has not been written to a file yet, and the duplicate-anchor case it
    guards against cannot arise here (one beat per chunk, by validation).
    """
    return tuple(
        ShotLength(
            start=beat.start, length_seconds=beat.length_seconds, source_chunk_id=beat.chunk_id
        )
        for beat in sorted(beats, key=lambda b: b.start)
        if beat.length_seconds is not None
    )


__all__ = [
    "BEATS_SCHEMA",
    "BEAT_ROLES",
    "FOCUS_ACTION",
    "FOCUS_SUBJECT",
    "FOCUS_VALUES",
    "MAX_VALIDATION_ATTEMPTS",
    "Beat",
    "BeatsResult",
    "BeatsValidationError",
    "beat_length_requests",
    "beats_input_hashes",
    "build_beats_prompt",
    "check_act_structure",
    "check_beat_structure",
    "generate_beats",
    "validate_beats",
]
