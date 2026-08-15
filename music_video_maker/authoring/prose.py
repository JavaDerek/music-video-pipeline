"""Stage 4 -- prose (issue #54 design section 4).

Turns each beat into the ``shot`` line a render will actually compose,
obeying ``docs/shot-writing-guide.md``: effect as grammatical subject, one
beat per shot, name the physical contact.

**Windowed on beat groups, never on a fixed chunk count.** A window boundary
falling between a contact and its consequence produces exactly the defect the
three-beat rule exists to prevent, and a group is also the unit a human
objects in -- "regenerate group 3" is a sentence someone will actually say.
Each window is written with the two chunks either side supplied as read-only
context, so the payoff is written by a call that has seen the plant.

The design says prose is told not to write camera direction, not to restate
identity, and not to restate the lyric, and that "all four are then checked
rather than trusted". They are -- but in **two tiers**, and the reason is the
guide itself:

* **Errors** (this stage retries, feeding the objection back): naming a cast
  member, and quoting the chunk's own lyric verbatim. ``prompting.py``
  composes both already, the guide's checklist says so in as many words, and
  neither is ambiguous to check.
* **Warnings** (collected, never retried here): camera language in a shot
  line that has no ``camera`` field beside it, and saying the performer is
  singing. Both are *prohibited by the guide's checklist and demonstrated by
  the guide's own worked examples* -- every example line in it ends with a
  trailing camera phrase, and "still singing to camera" appears in the two
  lines the guide holds up as correct. Rejecting them while handing the model
  that document as its system prompt would be telling it to disobey its own
  reference. Worse for the singing one: the guide's grammatical-subject rule
  warns in as many words that writing the performer out of the line costs the
  lip-sync for that whole chunk, so a hard rejection pushes the model toward
  the more expensive mistake.

Camera language *is* an error once that chunk has a ``camera`` value (issue
#54 phase 4), which is the same condition ``shot_plan``'s own
double-composition lint uses on the render side.

Location-outside-``setting`` is the fourth prohibition and is deliberately
not re-implemented here: ``load_shot_plan``'s geography lint already checks
it, against a curated landmark list with measured reasons for every entry,
and it runs over the generated plan in
:mod:`~music_video_maker.authoring.plan`. A second implementation would drift
from it within a month.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.chunks import skeleton_table_text
from music_video_maker.authoring.driver import MODEL_SONNET, DriverResult, ModelDriver
from music_video_maker.authoring.hashing import sha256_file, sha256_text
from music_video_maker.authoring.prompts import SHOT_WRITING_GUIDE_DOC, prose_system_prompt
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk

logger = logging.getLogger(__name__)

MAX_VALIDATION_ATTEMPTS = 3

CONTEXT_RADIUS = 2
"""How many beats either side of a window are supplied as read-only context
-- the design's "the two chunks either side"."""

LYRIC_QUOTE_WORDS = 4
"""How many consecutive words shared with the chunk's own lyric count as
quoting it rather than as vocabulary the shot and the lyric happen to share.
Three is coincidence; the printer gag's own lyric and shot legitimately share
"printer"."""

_CAMERA_WORDS = re.compile(
    r"\b(camera|close on|wide shot|tracking (?:in|out|backwards|forwards)|"
    r"push(?:ing|es)? in|pull(?:ing|s)? back|pans? (?:left|right)|"
    r"crane|dolly|handheld|zoom(?:ing|s)? (?:in|out))\b",
    re.IGNORECASE,
)
"""Deliberately narrow: unambiguous camera *instructions*, not any word a
camera might also be described with. "close on her hands" is direction;
"the closing door" is staging."""

_SINGING_WORDS = re.compile(r"\b(sing|sings|singing|sung|lip.?sync\w*)\b", re.IGNORECASE)

PROSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chunk_id", "shot"],
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "shot": {"type": "string", "minLength": 1},
                    # Issue #59. The shot line may not name a cast member --
                    # the render composes names itself -- so a second
                    # character can only be referred to as "him"/"her". This
                    # is where that pronoun gets bound to somebody, and it is
                    # a list of names rather than prose precisely because
                    # nothing resolves prose against the cast.
                    "present": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


class ProseValidationError(ValueError):
    """Raised when a model's shot lines are malformed or break a prohibition
    this stage enforces as an error. Carries every problem, not the first."""


@dataclass(frozen=True)
class ProseIssue:
    """One objection to one shot line."""

    chunk_id: int
    severity: str
    """``"error"`` (retried here) or ``"warning"`` (collected, revised once at
    the plan level, then written into the file as a ``# lint:`` comment)."""

    message: str


@dataclass(frozen=True)
class ProseResult:
    shots: dict[int, str]
    present: dict[int, tuple[str, ...]]
    """Per chunk, the cast on screen but not singing (issue #59).

    Only chunks the model named anyone for appear here; a chunk with nobody
    else in shot is simply absent, which ``render_plan_toml`` writes as no
    ``present`` key at all."""

    issues: tuple[ProseIssue, ...]
    driver_results: tuple[DriverResult, ...]
    """One per window -- this stage makes several calls, and the cost of all
    of them belongs in the session record."""

    input_hashes: dict[str, str]


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #


def beat_windows(
    beats: Sequence[Beat], groups: Iterable[int] | None = None
) -> tuple[tuple[Beat, ...], ...]:
    """Split ``beats`` into one window per ``beat_group``, in song order.

    Song order is the order each group *first* appears, not the numeric order
    of the group ids -- a group number is a label the beats stage chose, and
    nothing makes it monotonic. A group whose members are spread across the
    song stays one window, because the guide explicitly allows a plant to sit
    a chunk or two before its contact and the payoff must be written by a call
    that saw the plant.
    """
    wanted = None if groups is None else set(groups)
    ordered: list[int] = []
    members: dict[int, list[Beat]] = {}
    for beat in sorted(beats, key=lambda b: b.start):
        if wanted is not None and beat.beat_group not in wanted:
            continue
        if beat.beat_group not in members:
            members[beat.beat_group] = []
            ordered.append(beat.beat_group)
        members[beat.beat_group].append(beat)
    return tuple(tuple(members[group]) for group in ordered)


def _context_for(
    window: Sequence[Beat], beats: Sequence[Beat]
) -> tuple[tuple[Beat, ...], tuple[Beat, ...]]:
    """The ``CONTEXT_RADIUS`` beats either side of this window's span."""
    ordered = sorted(beats, key=lambda b: b.start)
    in_window = {beat.chunk_id for beat in window}
    positions = [i for i, beat in enumerate(ordered) if beat.chunk_id in in_window]
    if not positions:
        return (), ()
    before = [b for b in ordered[max(0, positions[0] - CONTEXT_RADIUS) : positions[0]]]
    after = [b for b in ordered[positions[-1] + 1 : positions[-1] + 1 + CONTEXT_RADIUS]]
    return tuple(before), tuple(after)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _quotes_lyric(shot: str, lyric: str) -> str | None:
    """The verbatim run of ``LYRIC_QUOTE_WORDS`` words a shot shares with its
    own lyric, or ``None``."""
    lyric_words = _words(lyric)
    if len(lyric_words) < LYRIC_QUOTE_WORDS:
        return None
    shot_words = _words(shot)
    runs = {
        " ".join(lyric_words[i : i + LYRIC_QUOTE_WORDS])
        for i in range(len(lyric_words) - LYRIC_QUOTE_WORDS + 1)
    }
    for i in range(len(shot_words) - LYRIC_QUOTE_WORDS + 1):
        candidate = " ".join(shot_words[i : i + LYRIC_QUOTE_WORDS])
        if candidate in runs:
            return candidate
    return None


def advisory_issues(
    shots: Mapping[int, str], *, camera: Mapping[int, str] | None = None
) -> tuple[ProseIssue, ...]:
    """The warning-tier prohibitions, over finished shot text.

    Separate from :func:`validate_prose` and public because these have to be
    re-derived at *plan* time as well: the shot lines can change under a
    revision round, and annotating the written file with a warning about a
    sentence that has since been rewritten would be worse than not annotating
    it at all. Pure, and a function of the text alone, so it can be re-run as
    often as the text moves.
    """
    camera = dict(camera or {})
    issues: list[ProseIssue] = []
    for chunk_id in sorted(shots):
        shot = shots[chunk_id] or ""
        camera_match = _CAMERA_WORDS.search(shot)
        if camera_match and chunk_id not in camera:
            issues.append(
                ProseIssue(
                    chunk_id=chunk_id,
                    severity="warning",
                    message=(
                        f"shot line carries its own camera direction "
                        f"({camera_match.group(0)!r}); the camera field owns where that "
                        "lands in the sentence"
                    ),
                )
            )
        singing = _SINGING_WORDS.search(shot)
        if singing:
            issues.append(
                ProseIssue(
                    chunk_id=chunk_id,
                    severity="warning",
                    message=(
                        f"shot line restates that the performer is singing "
                        f"({singing.group(0)!r}); the lyric clause composes that already"
                    ),
                )
            )
    return tuple(issues)


def validate_prose(
    data: object,
    window: Sequence[Beat],
    *,
    chunks: Sequence[AudioChunk],
    cast_names: Sequence[str],
    camera: Mapping[int, str],
) -> tuple[dict[int, str], tuple[ProseIssue, ...]]:
    """Parse one window's reply, raising on errors and returning warnings.

    Returns ``(shots, warnings)``. Errors raise :class:`ProseValidationError`
    with all of them listed; warnings come back for the caller to collect --
    see the module docstring for why the two tiers are not the same loop.
    """
    if not isinstance(data, dict):
        raise ProseValidationError(
            f"prose reply must be a JSON object, got {type(data).__name__}"
        )
    raw = data.get("shots")
    if not isinstance(raw, list):
        raise ProseValidationError(
            f"prose reply must carry a 'shots' array; got keys {sorted(data)}"
        )

    in_window = {beat.chunk_id for beat in window}
    lyrics = {chunk.chunk_id: chunk.text or "" for chunk in chunks}
    errors: list[str] = []
    warnings: list[ProseIssue] = []
    shots: dict[int, str] = {}
    present: dict[int, tuple[str, ...]] = {}

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            errors.append(f"shots[{index}] is not an object")
            continue
        chunk_id = entry.get("chunk_id")
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            errors.append(f"shots[{index}] chunk_id must be an integer, got {chunk_id!r}")
            continue
        if chunk_id not in in_window:
            # Read-only context really is read-only: a reply that rewrites a
            # neighbouring chunk would silently overwrite prose another window
            # already produced and a human may already have approved.
            logger.warning(
                "Prose reply writes chunk_id=%d, which is context for this window rather "
                "than part of it; dropping it.",
                chunk_id,
            )
            continue
        if chunk_id in shots:
            errors.append(f"chunk_id={chunk_id} appears more than once")
            continue

        shot = entry.get("shot")
        if not isinstance(shot, str) or not shot.strip():
            errors.append(f"chunk_id={chunk_id} shot must be a non-empty string")
            continue
        shot = shot.strip()

        # Issue #59: bind whoever the line refers to as "him"/"her". Unknown
        # names are an error rather than a dropped entry -- a misspelt name
        # here stages nobody, which is exactly the silent failure the field
        # exists to end.
        raw_present = entry.get("present", [])
        if not isinstance(raw_present, list) or not all(
            isinstance(name, str) for name in raw_present
        ):
            errors.append(f"chunk_id={chunk_id} present must be an array of cast names")
        else:
            unknown = [name for name in raw_present if name not in cast_names]
            if unknown:
                errors.append(
                    f"chunk_id={chunk_id} lists {unknown} in present, which is not in the "
                    f"cast. Use exactly one of {sorted(cast_names)}, or omit the field if "
                    "nobody else is on screen"
                )
            elif raw_present:
                # Absent, not empty, when nobody is named: the renderer's
                # default is "she is alone in shot", and an empty list says
                # the same thing more loudly in every one of 36 blocks.
                present[chunk_id] = tuple(dict.fromkeys(raw_present))

        named = [name for name in cast_names if re.search(rf"\b{re.escape(name)}\b", shot, re.I)]
        if named:
            errors.append(
                f"chunk_id={chunk_id} names the cast member(s) {named} in the shot line. "
                "The render composes each character's name and role into every prompt "
                "already; naming them here spends the line on something it is given and "
                "competes with it. Refer to them as 'she'/'he'/'they' or not at all"
            )

        quoted = _quotes_lyric(shot, lyrics.get(chunk_id, ""))
        if quoted:
            errors.append(
                f"chunk_id={chunk_id} quotes its own lyric verbatim ({quoted!r}). The lyric "
                "clause is composed into the prompt separately; repeating it here costs the "
                "line its staging and tells the model the same thing twice"
            )

        camera_match = _CAMERA_WORDS.search(shot)
        if camera_match and chunk_id in camera:
            errors.append(
                f"chunk_id={chunk_id} writes camera direction ({camera_match.group(0)!r}) "
                f"into the shot line while this chunk already has camera="
                f"{camera[chunk_id]!r}. The two compose into one sentence, so the direction "
                "would land twice"
            )

        warnings.extend(advisory_issues({chunk_id: shot}, camera=camera))
        shots[chunk_id] = shot

    missing = sorted(in_window - set(shots))
    if missing:
        errors.append(
            f"these chunks in this window have no shot line: {missing}. Every chunk in the "
            "window needs one"
        )

    if errors:
        raise ProseValidationError(
            f"{len(errors)} problem(s) with these shot lines:\n- " + "\n- ".join(errors)
        )
    return shots, present, tuple(warnings)


# --------------------------------------------------------------------------- #
# Prompt + hashes
# --------------------------------------------------------------------------- #


def prose_input_hashes(
    config: RunConfig,
    chunks: Sequence[AudioChunk],
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
) -> dict[str, str]:
    """Everything this stage consumes, hashed -- including the beat sheet, so
    re-running beats reports prose stale rather than leaving shot lines
    written against a structure that has changed."""
    return {
        "skeleton": sha256_text(skeleton_table_text(chunks)),
        "concept": sha256_text(json.dumps(dict(concept), sort_keys=True)),
        "beats": sha256_text(
            json.dumps([b.to_dict() for b in sorted(beats, key=lambda b: b.start)], sort_keys=True)
        ),
        "shot_writing_guide": sha256_file(SHOT_WRITING_GUIDE_DOC),
    }


def _beat_line(beat: Beat, chunk: AudioChunk | None, camera: Mapping[int, str]) -> str:
    lyric = (chunk.text or "").strip() if chunk is not None else ""
    parts = [
        f"chunk_id={beat.chunk_id}",
        f"{beat.start:.3f}-{beat.end:.3f}s",
        f"role={beat.beat_role}",
        f"focus={beat.focus}",
        f"beat: {beat.beat}",
    ]
    if lyric:
        parts.append(f'lyric: "{lyric}"')
    else:
        parts.append("INSTRUMENTAL -- nothing sung here")
    if beat.chunk_id in camera:
        parts.append(f"camera (already decided, do not repeat): {camera[beat.chunk_id]}")
    return " | ".join(parts)


def build_prose_prompt(
    config: RunConfig,
    concept: Mapping[str, Any],
    window: Sequence[Beat],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    *,
    camera: Mapping[int, str],
    notes: str | None = None,
) -> str:
    """The Stage 4 user prompt for one window."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    before, after = _context_for(window, beats)

    parts = [
        "## The approved concept",
        f"Logline: {concept.get('logline', '')}",
        f"Tone: {concept.get('tone', '')}",
    ]
    motifs = concept.get("motifs") or []
    avoid = concept.get("avoid") or []
    if motifs:
        parts.append("Motifs: " + "; ".join(str(m) for m in motifs))
    if avoid:
        parts.append("Deliberately NOT on screen: " + "; ".join(str(a) for a in avoid))
    if config.setting:
        parts += ["", f"## Setting (every location must fit inside it): {config.setting}"]
    if config.global_style:
        # Same reason photography gets it: `global_style` carries editing
        # constraints ("one continuous unbroken take, no cuts, no scene
        # changes") that a shot line can contradict outright, and it is
        # composed into every prompt whether this stage has seen it or not.
        parts += [
            "",
            "## The run's style, already fixed and composed into every prompt -- do not "
            "restate it, but do not write anything that contradicts it",
            config.global_style,
        ]

    parts += [
        "",
        f"## Write a shot line for each of these {len(window)} chunk(s), and no others",
        *[_beat_line(beat, by_id.get(beat.chunk_id), camera) for beat in window],
    ]
    if before or after:
        parts += [
            "",
            "## Read-only context -- what surrounds this window. Do NOT write these.",
            *[_beat_line(beat, by_id.get(beat.chunk_id), camera) for beat in (*before, *after)],
        ]
    if notes and notes.strip():
        parts += ["", f"## Notes from the person reviewing this: {notes.strip()}"]

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_prose(
    config: RunConfig,
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    driver: ModelDriver,
    *,
    camera: Mapping[int, str] | None = None,
    notes: str | None = None,
    groups: Iterable[int] | None = None,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> ProseResult:
    """Write one shot line per chunk, one model call per beat group.

    ``groups`` restricts generation to those beat groups -- "regenerate group
    3", the sentence the design expects a human to say. Everything outside
    them is simply not generated, so the caller still holds whatever prose was
    approved for them.
    """
    camera = dict(camera or {})
    system = prose_system_prompt()
    cast_names = tuple(config.cast)

    shots: dict[int, str] = {}
    present: dict[int, tuple[str, ...]] = {}
    issues: list[ProseIssue] = []
    results: list[DriverResult] = []

    windows = beat_windows(beats, groups)
    logger.info(
        "Prose: %d window(s) over %d beat(s)%s",
        len(windows),
        len(beats),
        f", restricted to group(s) {sorted(set(groups))}" if groups is not None else "",
    )

    for window in windows:
        base = build_prose_prompt(
            config, concept, window, beats, chunks, camera=camera, notes=notes
        )
        prompt = base
        last_error: ProseValidationError | None = None
        for attempt in range(1, max_validation_attempts + 1):
            result = driver.complete(
                system=system, prompt=prompt, model=MODEL_SONNET, schema=PROSE_SCHEMA
            )
            try:
                window_shots, window_present, window_warnings = validate_prose(
                    result.data, window, chunks=chunks, cast_names=cast_names, camera=camera
                )
            except ProseValidationError as exc:
                last_error = exc
                logger.warning(
                    "prose for beat_group=%d failed validation (attempt %d/%d): %s",
                    window[0].beat_group,
                    attempt,
                    max_validation_attempts,
                    exc,
                )
                prompt = (
                    f"{base}\n\n"
                    "## Your previous reply failed validation\n"
                    f"{exc}\n"
                    "Reply again with a single JSON object, fixing every problem above and "
                    "leaving everything else as it was."
                )
                continue
            shots.update(window_shots)
            present.update(window_present)
            issues.extend(window_warnings)
            results.append(result)
            break
        else:
            logger.error(
                "prose for beat_group=%d failed validation after %d attempt(s): %s",
                window[0].beat_group,
                max_validation_attempts,
                last_error,
            )
            raise ProseValidationError(
                f"shot lines for beat_group={window[0].beat_group} failed validation after "
                f"{max_validation_attempts} attempt(s): {last_error}"
            ) from last_error

    return ProseResult(
        shots=shots,
        present=present,
        issues=tuple(issues),
        driver_results=tuple(results),
        input_hashes=prose_input_hashes(config, chunks, concept, beats),
    )


def revise_prose(
    config: RunConfig,
    concept: Mapping[str, Any],
    beats: Sequence[Beat],
    chunks: Sequence[AudioChunk],
    driver: ModelDriver,
    *,
    shots: Mapping[int, str],
    objections: Mapping[int, Sequence[str]],
    camera: Mapping[int, str] | None = None,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> ProseResult:
    """Rewrite only the chunks in ``objections``, and return only those.

    **Targeted means targeted** (design section 6). The model is shown its own
    prior line and the specific objection to it, still inside its beat group
    so the rewrite stays coherent with the plant and the payoff -- but the
    validation scope is the objected chunks alone, so a reply that also
    rewrites its neighbours has those lines dropped rather than merged.

    Returning only the revised chunks is the other half: merging is the
    caller's, which makes "every entry not in scope is copied through
    byte-identically" a property of the data flow rather than a promise. A
    revision that quietly reworded thirty approved shots would destroy the
    value of the approval that came before it.
    """
    camera = dict(camera or {})
    system = prose_system_prompt()
    cast_names = tuple(config.cast)

    revised: dict[int, str] = {}
    revised_present: dict[int, tuple[str, ...]] = {}
    issues: list[ProseIssue] = []
    results: list[DriverResult] = []

    for window in beat_windows(beats):
        scope = tuple(beat for beat in window if beat.chunk_id in objections)
        if not scope:
            continue

        base = "\n\n".join(
            [
                build_prose_prompt(
                    config, concept, window, beats, chunks, camera=camera, notes=None
                ),
                "## Revise ONLY these chunks. Do not rewrite any other line.",
                "\n".join(
                    f"chunk_id={beat.chunk_id}\n"
                    f"  your line: {shots.get(beat.chunk_id, '')!r}\n"
                    "  objection(s): " + "; ".join(objections[beat.chunk_id])
                    for beat in scope
                ),
                "Reply with a single JSON object carrying only those chunk ids.",
            ]
        )
        prompt = base
        last_error: ProseValidationError | None = None
        for attempt in range(1, max_validation_attempts + 1):
            result = driver.complete(
                system=system, prompt=prompt, model=MODEL_SONNET, schema=PROSE_SCHEMA
            )
            try:
                window_shots, window_present, window_warnings = validate_prose(
                    result.data, scope, chunks=chunks, cast_names=cast_names, camera=camera
                )
            except ProseValidationError as exc:
                last_error = exc
                logger.warning(
                    "revision for beat_group=%d failed validation (attempt %d/%d): %s",
                    window[0].beat_group,
                    attempt,
                    max_validation_attempts,
                    exc,
                )
                prompt = (
                    f"{base}\n\n## Your previous reply failed validation\n{exc}\n"
                    "Reply again, still carrying only the chunk ids listed above."
                )
                continue
            revised.update(window_shots)
            revised_present.update(window_present)
            issues.extend(window_warnings)
            results.append(result)
            break
        else:
            logger.error(
                "revision for beat_group=%d failed after %d attempt(s): %s",
                window[0].beat_group,
                max_validation_attempts,
                last_error,
            )
            raise ProseValidationError(
                f"revising beat_group={window[0].beat_group} failed after "
                f"{max_validation_attempts} attempt(s): {last_error}"
            ) from last_error

    return ProseResult(
        shots=revised,
        present=revised_present,
        issues=tuple(issues),
        driver_results=tuple(results),
        input_hashes=prose_input_hashes(config, chunks, concept, beats),
    )


__all__ = [
    "CONTEXT_RADIUS",
    "LYRIC_QUOTE_WORDS",
    "MAX_VALIDATION_ATTEMPTS",
    "PROSE_SCHEMA",
    "advisory_issues",
    "ProseIssue",
    "ProseResult",
    "ProseValidationError",
    "beat_windows",
    "build_prose_prompt",
    "generate_prose",
    "prose_input_hashes",
    "revise_prose",
    "validate_prose",
]
