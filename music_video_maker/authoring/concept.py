"""Stage 1 -- concept (issue #54 design section 4).

Reads the song's real lyric text (which may be empty -- see
``chunks.skeleton_table_text``'s docstring), its chunk structure, the cast,
and any style/setting the config already fixes, and proposes an original
narrative concept: a paragraph a human can judge in seconds. The tightest,
cheapest-to-re-roll review point in the whole design, which is why it runs
on :data:`~music_video_maker.authoring.driver.MODEL_FABLE`.

Validation is hand-rolled (mirrors ``shot_plan._parse_*``'s existing
convention) rather than a ``jsonschema`` dependency: this package adds zero
new pip dependencies on purpose, and a generic schema library is exactly the
kind of thing this design keeps out until a shape actually needs more than a
handful of field checks provide.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from music_video_maker.authoring.chunks import skeleton_table_text
from music_video_maker.authoring.driver import MODEL_FABLE, DriverResult, ModelDriver
from music_video_maker.authoring.hashing import sha256_file, sha256_text
from music_video_maker.authoring.prompts import LYRICS_FORMAT_DOC, concept_system_prompt
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk
from music_video_maker.lyrics import parse_lyrics

logger = logging.getLogger(__name__)

MAX_VALIDATION_ATTEMPTS = 3
"""Bounded retry on a schema violation (design section 3's "bounded retry
(3)"), separate from -- and layered on top of -- ``ClaudeCliDriver``'s own
retry on mechanical failures (non-zero exit, timeout, non-JSON). This stage
owns the local validator the driver knows nothing about, so this stage owns
the retry that feeds the validator's own error back into the prompt."""

READING_STRING_FIELDS: tuple[str, ...] = (
    "subject", "speaker", "addressee", "situation", "change", "register", "period", "place",
)
"""Issue #69's comprehension fields, every one a plain non-empty string:
what the song is about, who is speaking and to whom (this pipeline's own
hard-won audible/on-screen split), the situation and what changes across the
song (the raw material #84's arc is built from), and register/period/place
implied by the words -- the "American sanctions" lyric a viewer noticed
render as medieval."""

READING_REQUIRED_FIELDS: tuple[str, ...] = (
    *READING_STRING_FIELDS, "nouns", "references",
)

REFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reference", "what_it_is", "imagery"],
    "properties": {
        "reference": {"type": "string", "minLength": 1},
        "what_it_is": {"type": "string", "minLength": 1},
        "imagery": {"type": "array", "items": {"type": "string"}},
    },
}
"""One cultural/historical/mythological/religious/political reference the
lyrics themselves name, with the concrete imagery it carries -- Koschey's
death nested in a needle, in an egg, in a duck, in a hare, in a chest, under
an oak on an island. ``imagery`` may be empty; nothing about a reference's
*existence* requires its imagery to be non-empty, though a model that names
one without any is not doing this field's job."""

READING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(READING_REQUIRED_FIELDS),
    "properties": {
        "subject": {"type": "string", "minLength": 1},
        "speaker": {"type": "string", "minLength": 1},
        "addressee": {"type": "string", "minLength": 1},
        "situation": {"type": "string", "minLength": 1},
        "change": {"type": "string", "minLength": 1},
        "register": {"type": "string", "minLength": 1},
        "period": {"type": "string", "minLength": 1},
        "place": {"type": "string", "minLength": 1},
        # Issue #69 requirement 2: empty is a first-class, valid answer for
        # both of these -- a breakup song has no mythology and naming none is
        # the CORRECT reply, not an incomplete one. Never minItems here.
        "nouns": {"type": "array", "items": {"type": "string"}},
        "references": {"type": "array", "items": REFERENCE_SCHEMA},
    },
}
"""Issue #69: a comprehension step the concept stage answers *before*
inventing anything, so the concept -- and #84's acts, and everything
downstream -- descends from what the song actually says rather than a
generic music-video pitch. Structured and validated like every other stage's
reply (requirement 1), never prose, so :mod:`.beats` can consume it
directly."""

ACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "function"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "function": {"type": "string", "minLength": 1},
    },
}
"""One act of issue #84's story structure: a name and what it is FOR
("resolution" / "where the earlier plant pays off"), never a bare label. The
issue's own open question -- fixed vocabulary or chosen per song -- is
answered here as chosen per song: a ballad and a protest song do not have the
same shape, the same reasoning #67 already gives for a directorial choice
being config rather than a preamble opinion."""

CONCEPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reading", "logline", "setting", "tone", "motifs", "avoid", "locations", "acts"],
    "properties": {
        "reading": READING_SCHEMA,
        "logline": {"type": "string", "minLength": 1},
        "setting": {"type": "string", "minLength": 1},
        "tone": {"type": "string", "minLength": 1},
        "motifs": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
        "locations": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
        "acts": {
            "type": "array",
            "items": ACT_SCHEMA,
            "minItems": 1,
        },
    },
}
"""Handed to the driver for ``--json-schema`` -- a hint to the model, not a
guarantee. :func:`validate_concept` is the actual gate; see the module
docstring for why this project hand-rolls that rather than trusting the
flag alone (design section 3)."""

CONCEPT_REQUIRED_FIELDS = tuple(CONCEPT_SCHEMA["required"])


class ConceptValidationError(ValueError):
    """Raised when a model's concept reply does not match the required shape."""


def validate_concept(data: object) -> None:
    if not isinstance(data, dict):
        raise ConceptValidationError(
            f"concept reply must be a JSON object, got {type(data).__name__}"
        )
    missing = [key for key in CONCEPT_REQUIRED_FIELDS if key not in data]
    if missing:
        raise ConceptValidationError(f"concept reply is missing required field(s): {missing}")
    for key in ("logline", "setting", "tone"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ConceptValidationError(
                f"concept.{key} must be a non-empty string, got {value!r}"
            )
    for key in ("motifs", "avoid"):
        value = data[key]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConceptValidationError(
                f"concept.{key} must be a list of strings, got {value!r}"
            )
    locations = data["locations"]
    if (
        not isinstance(locations, list)
        or not locations
        or not all(isinstance(v, str) and v.strip() for v in locations)
    ):
        raise ConceptValidationError(
            "concept.locations must be a non-empty list of non-empty strings (issue #78: "
            f"the beats stage has no vocabulary to assign `location` from otherwise), got "
            f"{locations!r}"
        )

    problems = _validate_reading(data["reading"]) + _validate_acts(data["acts"])
    if problems:
        raise ConceptValidationError(
            f"concept reply has {len(problems)} problem(s):\n- " + "\n- ".join(problems)
        )


def _validate_reading(reading: object) -> list[str]:
    """Issue #69's comprehension object. Returns every problem found rather
    than raising on the first, the same "report everything in one message"
    convention :mod:`.beats` uses -- a retry round costs a whole model call."""
    if not isinstance(reading, dict):
        return [f"concept.reading must be a JSON object, got {type(reading).__name__}"]

    problems: list[str] = []
    missing = [key for key in READING_REQUIRED_FIELDS if key not in reading]
    if missing:
        problems.append(f"concept.reading is missing required field(s): {missing}")

    for key in READING_STRING_FIELDS:
        if key not in reading:
            continue
        value = reading[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"concept.reading.{key} must be a non-empty string, got {value!r}")

    if "nouns" in reading:
        nouns = reading["nouns"]
        # Issue #69 requirement 2: an empty list is the CORRECT answer for a
        # song with no concrete nouns worth naming -- never required non-empty.
        if not isinstance(nouns, list) or not all(isinstance(v, str) for v in nouns):
            problems.append(f"concept.reading.nouns must be a list of strings, got {nouns!r}")

    if "references" in reading:
        references = reading["references"]
        if not isinstance(references, list):
            # Issue #69 requirement 2: empty is fine and expected for most
            # songs -- only the wrong *type* is a problem, never an empty one.
            problems.append(f"concept.reading.references must be a list, got {references!r}")
        else:
            for index, ref in enumerate(references):
                problems.extend(_validate_reference(ref, index))

    return problems


def _validate_reference(ref: object, index: int) -> list[str]:
    if not isinstance(ref, dict):
        return [f"concept.reading.references[{index}] is not an object, got {ref!r}"]

    problems: list[str] = []
    missing = [key for key in ("reference", "what_it_is", "imagery") if key not in ref]
    if missing:
        problems.append(f"concept.reading.references[{index}] is missing field(s): {missing}")
        return problems

    for key in ("reference", "what_it_is"):
        value = ref[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(
                f"concept.reading.references[{index}].{key} must be a non-empty string, "
                f"got {value!r}"
            )

    imagery = ref["imagery"]
    if not isinstance(imagery, list) or not all(isinstance(v, str) for v in imagery):
        problems.append(
            f"concept.reading.references[{index}].imagery must be a list of strings, "
            f"got {imagery!r}"
        )
    return problems


def _validate_acts(acts: object) -> list[str]:
    """Issue #84's closed, ordered act vocabulary: non-empty, and every
    name unique (case/whitespace-insensitive, the same canonicalization
    :mod:`.beats` gives ``locations``). Order itself needs no validation here
    -- the list order IS the stated order; there is nothing to compare it
    against yet."""
    if not isinstance(acts, list) or not acts:
        return [
            "concept.acts must be a non-empty list of {'name', 'function'} objects (issue #84: "
            f"the beats stage has no act structure to assign `act` from otherwise), got {acts!r}"
        ]

    problems: list[str] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(acts):
        if not isinstance(entry, dict):
            problems.append(f"concept.acts[{index}] is not an object, got {entry!r}")
            continue
        name = entry.get("name")
        function = entry.get("function")
        if not isinstance(name, str) or not name.strip():
            problems.append(
                f"concept.acts[{index}].name must be a non-empty string, got {name!r}"
            )
        else:
            key = name.strip().lower()
            if key in seen:
                problems.append(
                    f"concept.acts names must be unique, but {name!r} appears more than once "
                    f"(acts[{seen[key]}] and acts[{index}])"
                )
            else:
                seen[key] = index
        if not isinstance(function, str) or not function.strip():
            problems.append(
                f"concept.acts[{index}].function must be a non-empty string, got {function!r}"
            )
    return problems


@dataclass(frozen=True)
class ConceptResult:
    """One successful concept generation."""

    data: dict[str, Any]
    driver_result: DriverResult
    input_hashes: dict[str, str]
    """What this run consumed, for :mod:`~music_video_maker.authoring.session`
    staleness -- the lyrics file, the chunk skeleton, and
    ``docs/lyrics-format.md``, each hashed by content."""


def _summary_stats(chunks: Sequence[AudioChunk]) -> dict[str, Any]:
    voiced = sum(chunk.duration for chunk in chunks if not chunk.is_instrumental)
    instrumental = sum(chunk.duration for chunk in chunks if chunk.is_instrumental)
    return {
        "total_runtime_seconds": round(voiced + instrumental, 3),
        "voiced_seconds": round(voiced, 3),
        "instrumental_seconds": round(instrumental, 3),
        "chunk_count": len(chunks),
    }


def concept_input_hashes(config: RunConfig, chunks: Sequence[AudioChunk]) -> dict[str, str]:
    """Every input :func:`generate_concept` actually consumes, hashed --
    shared between the real run and :mod:`~music_video_maker.authoring.session`
    staleness checks so the two can never disagree about what "the same
    inputs" means."""
    return {
        "lyrics": sha256_file(config.lyrics_file),
        "skeleton": sha256_text(skeleton_table_text(chunks)),
        "lyrics_format_doc": sha256_file(LYRICS_FORMAT_DOC),
    }


def build_concept_prompt(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    *,
    hints: str | None = None,
) -> str:
    """The Stage 1 user prompt: lyric text, chunk structure, cast, whatever
    style/setting the config already fixes, and freeform hints."""
    lines = parse_lyrics(config.lyrics_file, config.cast, config.default_lead_vocalist)
    lyric_text = "\n".join(line.text for line in lines.lines if line.text.strip())

    cast_lines = "\n".join(f"- {name}: {member.role}" for name, member in config.cast.items())
    stats = _summary_stats(chunks)

    parts = [
        "## Lyric text (verbatim, tag-stripped; empty means this song is "
        "authored with nobody shown singing)",
        lyric_text or "(none -- no lyric is ever sung on screen in this video)",
        "",
        "## Song structure",
        (
            f"Total runtime: {stats['total_runtime_seconds']:.1f}s "
            f"({stats['voiced_seconds']:.1f}s voiced, "
            f"{stats['instrumental_seconds']:.1f}s instrumental, "
            f"{stats['chunk_count']} chunks)"
        ),
        "chunk_id\tstart\tend\tframes\ttag\tsinger\tlyric",
        skeleton_table_text(chunks),
        "## Cast",
        cast_lines,
    ]
    if config.setting:
        parts += ["", f"## Setting (already fixed -- work inside it): {config.setting}"]
    if config.global_style:
        parts += ["", f"## Style (already fixed -- work inside it): {config.global_style}"]
    if hints and hints.strip():
        parts += ["", f"## Hints from the person running this: {hints.strip()}"]

    return "\n".join(parts)


def generate_concept(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    driver: ModelDriver,
    *,
    hints: str | None = None,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> ConceptResult:
    """Call the model, validate its reply, and retry (feeding the validator's
    own error back into the prompt) until it passes or attempts run out.

    Raises :class:`ConceptValidationError` if every attempt fails validation,
    or lets :class:`~music_video_maker.authoring.driver.DriverError` from the
    driver itself propagate -- either way, nothing is returned to write, per
    design section 3: "a half-written session is worse than no session."
    """
    system = concept_system_prompt()
    prompt = build_concept_prompt(config, chunks, hints=hints)

    last_error: ConceptValidationError | None = None
    result: DriverResult | None = None
    for attempt in range(1, max_validation_attempts + 1):
        result = driver.complete(
            system=system, prompt=prompt, model=MODEL_FABLE, schema=CONCEPT_SCHEMA
        )
        try:
            validate_concept(result.data)
        except ConceptValidationError as exc:
            last_error = exc
            logger.warning(
                "concept reply failed validation (attempt %d/%d): %s",
                attempt,
                max_validation_attempts,
                exc,
            )
            prompt = (
                f"{prompt}\n\n"
                "## Your previous reply failed validation\n"
                f"Error: {exc}\n"
                "Reply again with a single JSON object matching the required shape exactly."
            )
            continue
        return ConceptResult(
            data=result.data,
            driver_result=result,
            input_hashes=concept_input_hashes(config, chunks),
        )

    logger.error(
        "concept generation failed validation after %d attempt(s): %s",
        max_validation_attempts,
        last_error,
    )
    raise ConceptValidationError(
        f"concept reply failed validation after {max_validation_attempts} attempt(s): "
        f"{last_error}"
    ) from last_error


__all__ = [
    "ACT_SCHEMA",
    "CONCEPT_REQUIRED_FIELDS",
    "CONCEPT_SCHEMA",
    "MAX_VALIDATION_ATTEMPTS",
    "READING_REQUIRED_FIELDS",
    "READING_SCHEMA",
    "READING_STRING_FIELDS",
    "REFERENCE_SCHEMA",
    "ConceptResult",
    "ConceptValidationError",
    "build_concept_prompt",
    "concept_input_hashes",
    "generate_concept",
    "validate_concept",
]
