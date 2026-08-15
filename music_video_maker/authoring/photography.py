"""Stage 3 -- photography (issue #54 design section 4).

Chooses the film's look (``config.cinematography``, issue #53) and each
shot's ``camera``. Its own stage, for three reasons the design gives: it is a
different craft from writing what happens, #55 wants to promote its output to
a locked profile, and isolating it makes it **A/B-able on its own** -- the
same discipline that found the grammatical-subject rule.

Two properties nothing else in this package has:

* **Per-chunk output is optional.** An absent ``camera`` composes nothing, the
  same conservative default as ``length_seconds`` and ``focus``. Demanding one
  direction per chunk would buy filler on every shot that does not need any,
  and every one of those composes a real trailing clause into a real prompt.
* **Candidates vary the framing instruction, never a seed.** Re-rolling one
  prompt gives noise; asking for a genuinely different stance
  (:data:`CANDIDATE_STANCES`) gives a choice worth making. That is #55's step
  3 -- "that one, keep it".

The one hard validation rule is small and very specific. ``prompting._apply_camera_clause``
composes ``", camera <value>"``, so a value that starts with "camera" or "the
camera" renders as "..., camera the camera pushes in". The model is told the
word is supplied for it, and then it is checked -- prohibitions in a prompt
are advisory; this is what holds the line.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.chunks import skeleton_table_text
from music_video_maker.authoring.driver import MODEL_OPUS, DriverError, DriverResult, ModelDriver
from music_video_maker.authoring.hashing import sha256_text
from music_video_maker.authoring.prompts import photography_system_prompt
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk

logger = logging.getLogger(__name__)

MAX_VALIDATION_ATTEMPTS = 3

CANDIDATE_STANCES: tuple[str, ...] = (
    "Long lenses and locked-off frames. The camera almost never moves; the "
    "world moves through it. Formal, patient, a little cold.",
    "Handheld and close, always moving with her. Short lenses, the frame "
    "breathing. Intimate and slightly unstable.",
    "Wide, symmetrical and observational -- the camera watches from across "
    "the room and lets the space do the work. Very few close-ups.",
    "Restless coverage: the camera finds the object of each beat before she "
    "reaches it, then leaves her behind for the consequence.",
)
"""The stances ``--candidates`` runs from, in order.

Deliberately *different instructions* rather than repeated rolls of one
(design section 4). Each is a coherent position a real cinematographer could
hold for a whole video, so the side-by-side is a genuine choice rather than a
sample of one distribution's variance."""

CAMERA_EXAMPLE = "tracking backwards ahead of her"

_LEADING_CAMERA = re.compile(r"^\s*(the\s+)?cameras?\b", re.IGNORECASE)

PHOTOGRAPHY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cinematography", "camera"],
    "properties": {
        "cinematography": {"type": "string"},
        "camera": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chunk_id", "camera"],
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "camera": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


class PhotographyValidationError(ValueError):
    """Raised when a photography reply is malformed or would compose badly."""


@dataclass(frozen=True)
class Photography:
    """One look: the whole-video film direction, and per-shot framing."""

    cinematography: str | None
    """``None`` when ``config.cinematography`` already fixes it -- the global
    half of this stage is skipped rather than second-guessed."""

    camera: dict[int, str]
    """Sparse by design: only the chunks that actually want direction."""


@dataclass(frozen=True)
class PhotographyResult:
    photography: Photography
    driver_result: DriverResult
    input_hashes: dict[str, str]
    stance_index: int | None = None
    """Which :data:`CANDIDATE_STANCES` entry produced this, or ``None`` for a
    plain single run."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_photography(
    data: object,
    chunks: Sequence[AudioChunk],
    *,
    config_cinematography: str | None,
) -> Photography:
    """Parse and check a reply, raising with everything that is wrong."""
    if not isinstance(data, dict):
        raise PhotographyValidationError(
            f"photography reply must be a JSON object, got {type(data).__name__}"
        )

    problems: list[str] = []
    look: str | None = None
    if config_cinematography is None:
        raw_look = data.get("cinematography")
        if not isinstance(raw_look, str) or not raw_look.strip():
            problems.append(
                "cinematography must be a non-empty string describing the whole video's "
                "look -- stock, lens, depth of field, lighting, grade"
            )
        else:
            look = raw_look.strip()

    raw_camera = data.get("camera")
    if not isinstance(raw_camera, list):
        raise PhotographyValidationError(
            f"photography reply must carry a 'camera' array; got keys {sorted(data)}"
        )

    known = {chunk.chunk_id for chunk in chunks}
    camera: dict[int, str] = {}
    for index, entry in enumerate(raw_camera):
        if not isinstance(entry, dict):
            problems.append(f"camera[{index}] is not an object")
            continue
        chunk_id = entry.get("chunk_id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            problems.append(f"camera[{index}] chunk_id must be an integer, got {chunk_id!r}")
            continue
        if chunk_id not in known:
            logger.warning(
                "Photography names chunk_id=%d, which is not in this song's timeline; "
                "dropping it.",
                chunk_id,
            )
            continue
        if chunk_id in camera:
            problems.append(f"chunk_id={chunk_id} appears more than once")
            continue

        value = entry.get("camera")
        if not isinstance(value, str) or not value.strip():
            problems.append(f"chunk_id={chunk_id} camera must be a non-empty string")
            continue
        value = value.strip()

        if _LEADING_CAMERA.match(value):
            problems.append(
                f"chunk_id={chunk_id} camera={value!r} starts with the word 'camera'. The "
                "renderer composes this as \", camera <your value>\", so it would read "
                f"'..., camera {value}'. Write it assuming 'camera' has already been said "
                f"-- e.g. {CAMERA_EXAMPLE!r}"
            )
            continue

        camera[chunk_id] = value

    if problems:
        raise PhotographyValidationError(
            f"photography reply has {len(problems)} problem(s):\n- " + "\n- ".join(problems)
        )
    return Photography(cinematography=look, camera=camera)


# --------------------------------------------------------------------------- #
# Prompt + hashes
# --------------------------------------------------------------------------- #


def photography_input_hashes(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
) -> dict[str, str]:
    return {
        "skeleton": sha256_text(skeleton_table_text(chunks)),
        "concept": sha256_text(json.dumps(dict(concept), sort_keys=True)),
        "beats": sha256_text(
            json.dumps([b.to_dict() for b in sorted(beats, key=lambda b: b.start)], sort_keys=True)
        ),
    }


def build_photography_prompt(
    config: RunConfig,
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    *,
    stance: str | None = None,
    notes: str | None = None,
) -> str:
    """The Stage 3 user prompt: the concept, every beat, and (for a candidate
    run) the framing stance this one is being asked to hold."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parts = [
        "## The approved concept",
        f"Logline: {concept.get('logline', '')}",
        f"Tone: {concept.get('tone', '')}",
    ]
    if concept.get("motifs"):
        parts.append("Motifs: " + "; ".join(str(m) for m in concept["motifs"]))
    if config.setting:
        parts.append(f"Setting: {config.setting}")

    if config.global_style:
        # Found by running this for real: `global_style` routinely carries
        # editing and camera constraints ("ONE continuous unbroken take, a
        # single camera following the subject, no cuts") because it was the
        # junk drawer #53 split up, and it is composed into every prompt. A
        # photography stage that has not seen it will happily propose
        # locked-off frames for a run whose style says the camera follows her.
        parts += [
            "",
            "## The run's style, already fixed and composed into every prompt -- your "
            "look and framing must not contradict it",
            config.global_style,
        ]

    if config.cinematography:
        parts += [
            "",
            "## The film's look is already fixed -- work inside it, do not replace it",
            config.cinematography,
            "Return an empty string for `cinematography`; only per-shot camera is wanted.",
        ]

    if stance:
        parts += ["", "## The stance you are holding for this whole video", stance]

    def _voiced_tag(beat: Beat) -> str:
        chunk = by_id.get(beat.chunk_id)
        is_voiced = chunk is not None and not chunk.is_instrumental and bool(
            (chunk.text or "").strip()
        )
        return "voiced" if is_voiced else "instrumental"

    parts += [
        "",
        "## Every beat in the song",
        *[
            (
                f"chunk_id={beat.chunk_id} | {beat.start:.1f}-{beat.end:.1f}s | "
                f"{_voiced_tag(beat)} | {beat.beat_role} | focus={beat.focus} | {beat.beat}"
                + (
                    ""
                    if by_id.get(beat.chunk_id) is None
                    or not (by_id[beat.chunk_id].text or "").strip()
                    else f' | lyric: "{by_id[beat.chunk_id].text.strip()}"'
                )
            )
            for beat in sorted(beats, key=lambda b: b.start)
        ],
        "",
        "Give a camera direction only to the shots that genuinely want one. An absent "
        "direction composes nothing, which is the right answer for most shots; a "
        "direction on every chunk is filler that still lands in every prompt.",
        'On a "voiced" chunk, keep her face available to the lens -- craning away and '
        'holding there is fine only on an "instrumental" chunk (a voiced chunk with no '
        "mouth on screen has no lip-sync for that whole chunk).",
        f'Write each one the way it reads after the word "camera" -- e.g. "{CAMERA_EXAMPLE}".',
    ]
    if notes and notes.strip():
        parts += ["", f"## Notes from the person reviewing this: {notes.strip()}"]

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_photography(
    config: RunConfig,
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    driver: ModelDriver,
    *,
    stance: str | None = None,
    stance_index: int | None = None,
    notes: str | None = None,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> PhotographyResult:
    """One look, validated, with the usual retry-with-feedback loop."""
    system = photography_system_prompt()
    base = build_photography_prompt(
        config, concept, beats, chunks, stance=stance, notes=notes
    )
    prompt = base

    last_error: PhotographyValidationError | None = None
    for attempt in range(1, max_validation_attempts + 1):
        result = driver.complete(
            system=system, prompt=prompt, model=MODEL_OPUS, schema=PHOTOGRAPHY_SCHEMA
        )
        try:
            photography = validate_photography(
                result.data, chunks, config_cinematography=config.cinematography
            )
        except PhotographyValidationError as exc:
            last_error = exc
            logger.warning(
                "photography reply failed validation (attempt %d/%d): %s",
                attempt,
                max_validation_attempts,
                exc,
            )
            prompt = (
                f"{base}\n\n"
                "## Your previous reply failed validation\n"
                f"{exc}\n"
                "Reply again with a single JSON object, fixing every problem above."
            )
            continue
        logger.info(
            "Photography accepted: %d chunk(s) given a camera direction out of %d",
            len(photography.camera),
            len(chunks),
        )
        return PhotographyResult(
            photography=photography,
            driver_result=result,
            input_hashes=photography_input_hashes(config, chunks, concept, beats),
            stance_index=stance_index,
        )

    logger.error(
        "photography failed validation after %d attempt(s): %s",
        max_validation_attempts,
        last_error,
    )
    raise PhotographyValidationError(
        f"photography reply failed validation after {max_validation_attempts} attempt(s): "
        f"{last_error}"
    ) from last_error


def photography_candidates(
    config: RunConfig,
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    driver: ModelDriver,
    *,
    count: int = 1,
    notes: str | None = None,
) -> tuple[PhotographyResult, ...]:
    """Run the stage ``count`` times from ``count`` different stances.

    ``count=1`` uses **no** stance at all rather than the first one: a single
    run is the default path, not a one-item A/B, and pinning it to
    ``CANDIDATE_STANCES[0]`` would quietly narrow every ordinary run to one
    look nobody chose.

    A candidate that cannot be made valid is logged and skipped rather than
    failing the whole call. A/B is worth having precisely when one option is
    bad, so losing the other two to it would make ``--candidates`` the least
    reliable way to use this stage.
    """
    if count <= 1:
        return (generate_photography(config, concept, beats, chunks, driver, notes=notes),)

    wanted = min(count, len(CANDIDATE_STANCES))
    if wanted < count:
        logger.info(
            "Asked for %d candidates but only %d distinct framing stances exist; running %d. "
            "More rolls of the same instruction would be noise, not a choice.",
            count,
            wanted,
            wanted,
        )

    results: list[PhotographyResult] = []
    for index in range(wanted):
        try:
            results.append(
                generate_photography(
                    config,
                    concept,
                    beats,
                    chunks,
                    driver,
                    stance=CANDIDATE_STANCES[index],
                    stance_index=index,
                    notes=notes,
                )
            )
        except (PhotographyValidationError, DriverError):
            logger.warning(
                "Photography candidate %d (stance %d) could not be generated; continuing "
                "with the others.",
                len(results) + 1,
                index,
                exc_info=True,
            )
    if not results:
        raise PhotographyValidationError(
            f"none of the {wanted} photography candidate(s) could be generated"
        )
    return tuple(results)


__all__ = [
    "CAMERA_EXAMPLE",
    "CANDIDATE_STANCES",
    "MAX_VALIDATION_ATTEMPTS",
    "PHOTOGRAPHY_SCHEMA",
    "Photography",
    "PhotographyResult",
    "PhotographyValidationError",
    "build_photography_prompt",
    "generate_photography",
    "photography_candidates",
    "photography_input_hashes",
    "validate_photography",
]
