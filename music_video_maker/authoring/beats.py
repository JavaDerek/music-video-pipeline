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
                "required": ["chunk_id", "beat", "beat_role", "beat_group", "focus"],
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "beat": {"type": "string", "minLength": 1},
                    "beat_role": {"enum": list(BEAT_ROLES)},
                    "beat_group": {"type": "integer", "minimum": 1},
                    "focus": {"enum": list(FOCUS_VALUES)},
                    "length_seconds": {"type": "number", "exclusiveMinimum": 0},
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


def _parse_entries(
    data: object, chunks: Sequence[AudioChunk], problems: list[str]
) -> tuple[Beat, ...]:
    """Shape-check each entry and pin it to a real chunk. Appends to
    ``problems`` rather than raising, so one round reports everything."""
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

        chunk = by_id[chunk_id]
        beats.append(
            Beat(
                chunk_id=chunk_id,
                start=chunk.start,
                end=chunk.end,
                beat=beat_text.strip(),
                beat_role=role,
                beat_group=group,
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


def validate_beats(data: object, chunks: Sequence[AudioChunk]) -> tuple[Beat, ...]:
    """Parse and check a model reply, or raise with everything that is wrong."""
    problems: list[str] = []
    beats = _parse_entries(data, chunks, problems)
    check_beat_structure(beats, chunks, problems)
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
        "chunk_id\tstart\tend\tframes\ttag\tlyric",
        skeleton_table_text(chunks),
        "## Cast",
        cast_lines,
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

    last_error: BeatsValidationError | None = None
    for attempt in range(1, max_validation_attempts + 1):
        result = driver.complete(
            system=system, prompt=prompt, model=MODEL_OPUS, schema=BEATS_SCHEMA
        )
        try:
            beats = validate_beats(result.data, chunks)
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
    "check_beat_structure",
    "generate_beats",
    "validate_beats",
]
