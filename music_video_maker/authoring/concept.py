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

CONCEPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["logline", "setting", "tone", "motifs", "avoid"],
    "properties": {
        "logline": {"type": "string", "minLength": 1},
        "setting": {"type": "string", "minLength": 1},
        "tone": {"type": "string", "minLength": 1},
        "motifs": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
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
    "CONCEPT_REQUIRED_FIELDS",
    "CONCEPT_SCHEMA",
    "MAX_VALIDATION_ATTEMPTS",
    "ConceptResult",
    "ConceptValidationError",
    "build_concept_prompt",
    "concept_input_hashes",
    "generate_concept",
    "validate_concept",
]
