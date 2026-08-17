"""Composing, checking and writing the generated ``shot_plan.toml``
(issue #54 design sections 6 and 8).

Three things happen here, in order:

1. **Compose.** :func:`render_plan_toml` turns the frozen chunk timeline, the
   beat sheet and the prose into TOML. Pure string composition, no I/O --
   the same discipline ``prompting.py`` and ``shot_plan.render_shot_plan_skeleton``
   already keep. ``chunk_id``/``start`` come from the *chunks*, never from
   anything a model said, which is what makes ``ShotPlanDriftError``
   unreachable for a generated plan.

2. **Check, through the real loaders.** :func:`check_plan` writes the
   candidate to a temp file and runs ``load_shot_plan``,
   ``shot_length_requests``, ``resolve_shot``/``resolve_camera`` over every
   chunk, and ``lint_shots_against_lyrics``/``lint_camera_face_away_on_voiced_chunks``
   -- the actual functions the renderer will use, with a ``logging.Handler``
   attached to collect what they say. **Not a reimplementation**: those lints
   encode measured lessons (a video that walked from a British street into
   Central Park; a printer sung about twice and staged once; a camera move
   that cost a chunk's lip-sync) and a second copy would drift from them
   within a month.

3. **Revise, in two deliberately different tiers.** Errors must be fixed:
   targeted revision of the offending chunk ids, bounded at 2 rounds, then
   abort and write nothing. Warnings get **one** round and then stop -- every
   one of those lints is documented as "a false positive must never block a
   run", and a loop that grinds them to zero will happily rewrite a correct
   shot to please a heuristic. Whatever survives is written into the file as a
   ``# lint:`` comment above its entry, where the human reviewing the plan
   sees it in context rather than in a log they have already scrolled past.

Provenance (design section 8) lives in the file rather than a sidecar,
because a sidecar gets separated from the plan the first time someone copies
it, and because ``load_shot_plan`` ignores keys it does not know. **Hashes,
not content**: what the inputs *were*, never what they *said* -- a committed
plan must not drag the whole lyric sheet into git (#51).
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.hashing import sha256_text
from music_video_maker.contracts import AudioChunk
from music_video_maker.shot_plan import (
    ShotPlanError,
    lint_camera_face_away_on_voiced_chunks,
    lint_present_location_mismatch,
    lint_role_prohibition_contradiction,
    lint_shots_against_lyrics,
    lint_subject_on_voiced_chunk,
    lint_unbound_companion_referent,
    lint_voiced_framing,
    load_shot_plan,
    resolve_camera,
    resolve_shot,
    shot_length_requests,
)

logger = logging.getLogger(__name__)

MAX_ERROR_ROUNDS = 2
"""Design section 6: "bounded at 2 rounds, then abort"."""

_CHUNK_ID_IN_MESSAGE = re.compile(r"chunk_id=(\d+)")

_LINT_LOGGERS = ("music_video_maker.shot_plan",)
"""Whose warnings :func:`check_plan` collects. Narrow on purpose: slicing and
config also warn during an authoring run, about things that are not this
plan's business."""


class PlanError(RuntimeError):
    """Raised when a candidate plan could not be made to load cleanly."""


@dataclass(frozen=True)
class PlanIssue:
    """One thing the real loaders said about a candidate plan."""

    chunk_id: int | None
    """``None`` for anything not attributable to one entry -- those become a
    file-level comment rather than being dropped."""

    severity: str
    message: str


@dataclass(frozen=True)
class PlanCheck:
    errors: tuple[PlanIssue, ...] = ()
    warnings: tuple[PlanIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class Provenance:
    """The ``[provenance]`` block: which model produced which part, from what.

    ``generated_at`` is caller-supplied rather than read from the clock here,
    the same reason ``write_shot_plan_skeleton`` gives: it keeps composition a
    pure function of its arguments.
    """

    generated_by: str
    generated_at: str
    models: Mapping[str, str] = field(default_factory=dict)
    """``{"concept": "claude-fable-5", "beats": ..., "prose": ...}``."""

    hashes: Mapping[str, str] = field(default_factory=dict)
    """``{"concept": ..., "guide": ..., "lyrics": ..., "skeleton": ...}``."""


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def _toml_string(value: str) -> str:
    """A TOML basic string. Escapes rather than substitutes: the skeleton
    writer can afford to turn a quote into an apostrophe inside a *comment*,
    but this is the shot text itself and must survive verbatim."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _comment(text: str) -> str:
    return " ".join(text.split())


def render_plan_toml(
    chunks: Sequence[AudioChunk],
    beats: Sequence[Beat],
    shots: Mapping[int, str],
    *,
    provenance: Provenance,
    camera: Mapping[int, str] | None = None,
    present: Mapping[int, Sequence[str]] | None = None,
    lint_comments: Mapping[int | None, Sequence[str]] | None = None,
) -> str:
    """Compose the text of a generated ``shot_plan.toml``.

    ``chunk_id``/``start`` come from ``chunks`` -- the frozen post-re-anchor
    timeline -- so a plan loaded back against the run it was generated for
    cannot drift. Every other field is optional and omitted when absent, the
    same convention the hand-authored format already uses: an absent
    ``length_seconds`` means "no editorial opinion", never a default.
    """
    camera = dict(camera or {})
    present = {k: list(v) for k, v in (present or {}).items()}
    lint_comments = {k: list(v) for k, v in (lint_comments or {}).items()}
    beats_by_id = {beat.chunk_id: beat for beat in beats}

    lines = [
        f"# Generated by {provenance.generated_by} on {provenance.generated_at}.",
        "# Anchors come from this run's own alignment -- edit the shot lines, not the",
        "# chunk_id/start values. A '# lint:' comment is a warning the real loaders",
        "# raised and nobody silenced; they are advisory, and some are false positives.",
        "",
        "[provenance]",
        f'generated_by = {_toml_string(provenance.generated_by)}',
        f'generated_at = {_toml_string(provenance.generated_at)}',
    ]
    for stage in sorted(provenance.models):
        lines.append(f"{stage}_model = {_toml_string(provenance.models[stage])}")
    for name in sorted(provenance.hashes):
        lines.append(f"{name}_sha256 = {_toml_string(provenance.hashes[name])}")

    for note in lint_comments.get(None, ()):
        lines.append(f"# lint: {_comment(note)}")

    blocks: list[str] = []
    for chunk in chunks:
        beat = beats_by_id.get(chunk.chunk_id)
        shot = (shots.get(chunk.chunk_id) or "").strip()
        block: list[str] = []

        for note in lint_comments.get(chunk.chunk_id, ()):
            block.append(f"# lint: {_comment(note)}")

        duration = chunk.end - chunk.start
        frames = f", {chunk.frame_count} frames" if chunk.frame_count is not None else ""
        block.append("[[shot]]")
        block.append(f"chunk_id = {chunk.chunk_id}")
        block.append(
            f"start = {chunk.start!r}"
            f"   # {chunk.start:.3f} - {chunk.end:.3f}  ({duration:.3f}s{frames})"
        )
        if chunk.is_instrumental or not chunk.text.strip():
            block.append("# INSTRUMENTAL -- no lyric to sing")
        else:
            block.append(f'# lyric: "{_comment(chunk.text).replace(chr(34), chr(39))}"')
        if beat is not None:
            # Issue #84: the act a human reviewing the plan as text can see
            # beside the beat's role and group -- the same comment line, not
            # a separate one, so the arc is visible at a glance rather than
            # requiring a second scan of the file. Omitted when absent (a
            # hand-built Beat, or one from a pre-#84 persisted sheet).
            tags = f"{beat.beat_role}, group {beat.beat_group}"
            if beat.act:
                tags += f', act "{beat.act}"'
            block.append(f"# beat: {_comment(beat.beat)}  [{tags}]")
            if beat.focus == "action":
                block.append('focus = "action"')
            if beat.length_seconds is not None:
                block.append(f"length_seconds = {beat.length_seconds}")
            # Issue #78: re-emitted from the beat, never from anything a
            # later stage (prose) could inject -- the same "anchors are
            # copied from the chunks/beats" rule every other beat-derived
            # field here follows. `beat.location` is always set on a
            # validated beat sheet; the guard is defensive for a hand-built
            # Beat (a test, a pre-#78 persisted sheet) that predates the
            # field entirely.
            if beat.location:
                block.append(f"location = {_toml_string(beat.location)}")
        if chunk.chunk_id in camera:
            block.append(f"camera = {_toml_string(camera[chunk.chunk_id])}")
        # Issue #59: omitted entirely when nobody else is in shot, the same
        # convention as every other optional field here -- an absent `present`
        # means "she is alone", never a default somebody has to read past.
        if present.get(chunk.chunk_id):
            names = ", ".join(_toml_string(n) for n in present[chunk.chunk_id])
            block.append(f"present = [{names}]")
        # Issue #82: re-emitted from the beat, never inferred here -- the
        # same "anchors are copied from the chunks/beats" rule `location`
        # follows. Emitted beside `present` because both answer "who is in
        # this shot", though `subject` answers a different question of it
        # (who the render composes as the FOCUS, not merely present).
        if beat is not None and beat.subject:
            block.append(f"subject = {_toml_string(beat.subject)}")
        block.append(f'generated_by = "{"prose" if shot else "skeleton"}"')
        block.append(f"content_sha256 = {_toml_string(sha256_text(shot))}")
        block.append(f"shot = {_toml_string(shot)}")
        blocks.append("\n".join(block))

    return "\n".join(lines) + "\n\n" + "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- #
# Checking, through the real loaders
# --------------------------------------------------------------------------- #


class _Collector(logging.Handler):
    """Collects what the real lints say while they run."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


_CONSEQUENCE_KEYWORD_WARNING = "reads as a consequence beat"


def _drop_keyword_consequence_warnings(
    warnings: Sequence[PlanIssue], beats: Sequence[Beat]
) -> tuple[PlanIssue, ...]:
    """Drop ``_lint_consequence_focus``'s guess where the beat sheet knows better.

    That lint infers "this reads like a consequence" from a curated keyword
    list, which is the best it can do from finished prose. A generated plan
    has the structural answer sitting right there in the beat sheet -- design
    section 4 says so in as many words: derived from structure is "strictly
    better than what ``_lint_consequence_focus`` can do from prose".

    This is not cosmetic tidying. On the first real re-authoring run it fired
    on four chunks the beats stage had classified as plants and transitions,
    purely because their lines mentioned smoke; those four then went into the
    warning revision round, and the rewrite put "She" back into the subject
    slot of a line that had correctly led with the object. Suppressing the
    contradicted warning is what stops a heuristic overruling the structure.
    """
    roles = {beat.chunk_id: beat.beat_role for beat in beats}
    kept = []
    for issue in warnings:
        if (
            _CONSEQUENCE_KEYWORD_WARNING in issue.message
            and issue.chunk_id is not None
            and roles.get(issue.chunk_id, "consequence") != "consequence"
        ):
            logger.debug(
                "Dropping the keyword consequence warning on chunk %d: the beat sheet "
                "calls it a %s.",
                issue.chunk_id,
                roles[issue.chunk_id],
            )
            continue
        kept.append(issue)
    return tuple(kept)


def check_plan(
    text: str,
    config,
    chunks: Sequence[AudioChunk],
    *,
    beats: Sequence[Beat] = (),
    scratch_dir: Path | None = None,
    stageable_nouns: Sequence[str] = (),
) -> PlanCheck:
    """Put a candidate plan through the functions the renderer will use.

    ``stageable_nouns`` (issue #87) is the concept's ``reading.nouns``
    (issue #69) -- the concrete objects this song's lyrics actually name.
    Passed straight through to ``lint_shots_against_lyrics``, which without
    it can only approximate "is this word a prop" and, measured on a real
    plan, spent most of its warnings on function words. The data flows
    authoring -> render here, never the reverse, so the import boundary
    ``tests/test_authoring_boundary.py`` enforces is untouched.

    Deliberately writes a temp file and calls ``load_shot_plan`` rather than
    parsing the TOML here: the plan's real behaviour is whatever those
    functions do with it, including the parts that raise, and a check that
    tested anything else would be checking a different plan.
    """
    errors: list[PlanIssue] = []
    collector = _Collector()
    loggers = [logging.getLogger(name) for name in _LINT_LOGGERS]

    candidate = Path("shot_plan.toml")
    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        candidate = Path(tmp) / "shot_plan.toml"
        candidate.write_text(text, encoding="utf-8")

        for log in loggers:
            log.addHandler(collector)
        try:
            plan = load_shot_plan(
                candidate, setting=config.setting, cast_names=tuple(config.cast)
            )
            # Issue #82: refused by the render's OWN rule, never a copy of
            # it -- run first, before any advisory lint, because it raises
            # and there is no point checking anything else on a plan this
            # will refuse (the same ordering `cli.py`'s render-side lint
            # block uses for the render path's own call to this function).
            lint_subject_on_voiced_chunk(plan, chunks, candidate)
            shot_length_requests(plan)
            for chunk in chunks:
                resolve_shot(plan, chunk)
                resolve_camera(plan, chunk)
            lint_shots_against_lyrics(plan, chunks, stageable_nouns)
            lint_camera_face_away_on_voiced_chunks(plan, chunks)
            lint_voiced_framing(plan, chunks)
            # Issue #72: a pronoun with only one bound candidate but text
            # that insists on a second, distinct person.
            lint_unbound_companion_referent(plan, chunks)
            # Issue #73: a role written as a prohibition, contradicted by
            # the shot line actually describing the forbidden thing.
            lint_role_prohibition_contradiction(plan, chunks, config.cast)
            # Issue #78: `present` staging a companion at a location that
            # contradicts where their own singing chunks place them.
            lint_present_location_mismatch(plan, chunks)
        except ShotPlanError as exc:
            # Drift, a duplicate chunk_id, a malformed entry: by construction
            # none of these can come from the model -- anchors are copied from
            # the chunks. Reaching here means *this module* composed something
            # wrong, so it is reported and never quietly retried at a model.
            errors.append(PlanIssue(chunk_id=None, severity="error", message=str(exc)))
            plan = {}
        finally:
            for log in loggers:
                log.removeHandler(collector)

    for chunk_id in sorted(plan):
        if not plan[chunk_id].shot.strip():
            errors.append(
                PlanIssue(
                    chunk_id=chunk_id,
                    severity="error",
                    message=(
                        "shot line is blank, so this chunk would fall back to the global "
                        "narrative_concept -- which renders as something deliberate-looking "
                        "that nobody authored"
                    ),
                )
            )

    warnings = tuple(
        PlanIssue(
            chunk_id=_chunk_id_from(message),
            severity="warning",
            message=message,
        )
        for message in (_strip_candidate_path(r.getMessage(), candidate) for r in collector.records)
    )
    return PlanCheck(
        errors=tuple(errors), warnings=_drop_keyword_consequence_warnings(warnings, beats)
    )


def _strip_candidate_path(message: str, candidate: Path) -> str:
    """Remove the temp path the candidate was checked in.

    Every loader message is prefixed ``"Shot plan <path>: "``, and here that
    path is a ``tempfile`` directory that stops existing the moment the check
    finishes. Writing it into the plan as a ``# lint:`` comment puts a dead
    local path in a file meant for a human to read and, per #51, one that may
    ship.
    """
    return message.replace(f"Shot plan {candidate}: ", "").replace(str(candidate), "this plan")


def _chunk_id_from(message: str) -> int | None:
    match = _CHUNK_ID_IN_MESSAGE.search(message)
    return int(match.group(1)) if match else None


def objections_by_chunk(issues: Sequence[PlanIssue]) -> dict[int, list[str]]:
    """Group attributable issues by chunk. Anything the loaders did not tie to
    a chunk is left out on purpose -- there is no targeted revision to make
    from it, and inventing a scope would rewrite approved lines."""
    grouped: dict[int, list[str]] = {}
    for issue in issues:
        if issue.chunk_id is None:
            continue
        grouped.setdefault(issue.chunk_id, []).append(issue.message)
    return grouped


def lint_comments_for(issues: Sequence[PlanIssue]) -> dict[int | None, list[str]]:
    """Surviving warnings, keyed for :func:`render_plan_toml` -- including the
    unattributable ones, which become a file-level comment rather than being
    dropped on the floor."""
    comments: dict[int | None, list[str]] = {}
    for issue in issues:
        comments.setdefault(issue.chunk_id, []).append(issue.message)
    return comments


@dataclass(frozen=True)
class BuiltPlan:
    """A candidate plan that loads cleanly, plus what it cost to get there."""

    text: str
    shots: dict[int, str]
    surviving_warnings: tuple[PlanIssue, ...]
    """Warnings still standing after the single revision round. They are in
    ``text`` as ``# lint:`` comments too; kept here so a caller can report the
    count without re-parsing the file it just composed."""

    revision_results: tuple[object, ...] = ()
    """Every ``DriverResult`` the revision rounds spent, for the session's
    cost record. Untyped here so this module stays free of any dependency on
    the prose stage -- it takes the reviser as a callable."""


def build_plan(
    config,
    chunks: Sequence[AudioChunk],
    beats: Sequence[Beat],
    shots: Mapping[int, str],
    *,
    provenance: Provenance,
    reviser,
    camera: Mapping[int, str] | None = None,
    present: Mapping[int, Sequence[str]] | None = None,
    extra_checks=None,
    scratch_dir: Path | None = None,
    stageable_nouns: Sequence[str] = (),
) -> BuiltPlan:
    """Compose, check, revise, and annotate -- the loop of design section 6.

    ``reviser`` is ``(shots, objections) -> ProseResult`` and ``extra_checks``
    is ``(shots) -> Sequence[PlanIssue]``: both injected rather than imported,
    so this module never depends on the prose stage and the two tiers can be
    tested without a model anywhere near them.

    ``extra_checks`` is how the prose stage's own advisory prohibitions (a
    camera phrase in the line, saying the performer is singing) reach the
    written file. Re-run on the *current* text each time rather than carried
    over from generation, because a revision round can rewrite the very
    sentence a warning was about -- annotating the file with a complaint about
    a sentence that no longer exists is worse than not annotating it.

    The tiers are not symmetric and must not be made so. Errors get up to
    :data:`MAX_ERROR_ROUNDS` and then abort with nothing written; warnings get
    exactly one round -- written as a single ``if`` rather than a bounded loop
    so there is no number to tune upward -- because every one of those lints is
    documented as a heuristic firing on prose a human wrote deliberately.
    """
    shots = dict(shots)
    present = dict(present or {})
    spent: list[object] = []

    def compose(comments: Mapping[int | None, Sequence[str]] | None = None) -> str:
        return render_plan_toml(
            chunks,
            beats,
            shots,
            provenance=provenance,
            camera=camera,
            present=present,
            lint_comments=comments,
        )

    def inspect() -> PlanCheck:
        found = check_plan(
            compose(), config, chunks, beats=beats, scratch_dir=scratch_dir,
            stageable_nouns=stageable_nouns,
        )
        if extra_checks is None:
            return found
        advisory = tuple(
            PlanIssue(chunk_id=issue.chunk_id, severity="warning", message=issue.message)
            for issue in extra_checks(shots)
        )
        return PlanCheck(errors=found.errors, warnings=found.warnings + advisory)

    check = inspect()
    for round_number in range(1, MAX_ERROR_ROUNDS + 1):
        if check.ok:
            break
        objections = objections_by_chunk(check.errors)
        if not objections:
            # Nothing attributable to a chunk means the composition itself is
            # broken, not the prose. No revision could fix it, so say so
            # rather than burning two rounds proving it.
            break
        logger.warning(
            "Candidate plan has %d error(s) on chunk(s) %s; revising (round %d/%d)",
            len(check.errors),
            sorted(objections),
            round_number,
            MAX_ERROR_ROUNDS,
        )
        revision = reviser(shots, objections)
        spent.extend(revision.driver_results)
        shots.update(revision.shots)
        check = inspect()

    if not check.ok:
        logger.error(
            "Candidate plan still has %d error(s) after %d revision round(s): %s",
            len(check.errors),
            MAX_ERROR_ROUNDS,
            [issue.message for issue in check.errors],
        )
        raise PlanError(
            f"generated plan still fails to load cleanly after {MAX_ERROR_ROUNDS} revision "
            f"round(s): {[issue.message for issue in check.errors]}. Nothing was written"
        )

    if check.warnings:
        objections = objections_by_chunk(check.warnings)
        if objections:
            logger.info(
                "Candidate plan has %d warning(s) on chunk(s) %s; one revision round, then "
                "whatever survives is written into the file as a comment",
                len(check.warnings),
                sorted(objections),
            )
            before_revision = dict(shots)
            revision = reviser(shots, objections)
            spent.extend(revision.driver_results)
            shots.update(revision.shots)
            after = inspect()
            if not after.ok:
                # A revision that fixed a heuristic and broke the loader is
                # strictly worse than the warning it was chasing. Roll the
                # *prose* back too, not just the verdict -- keeping the new
                # lines while reporting the old warnings would write out a
                # plan that neither check ever passed.
                logger.warning(
                    "The warning revision introduced %d error(s); rolling back to the "
                    "pre-revision prose and keeping its warnings instead.",
                    len(after.errors),
                )
                shots = before_revision
            else:
                check = after

    surviving = check.warnings
    if surviving:
        logger.info(
            "%d warning(s) survived and are written into the plan as '# lint:' comments -- "
            "they are advisory, and some of them are false positives",
            len(surviving),
        )
    return BuiltPlan(
        text=compose(lint_comments_for(surviving)),
        shots=shots,
        surviving_warnings=surviving,
        revision_results=tuple(spent),
    )


def human_edited_chunks(path: str | Path) -> tuple[int, ...]:
    """Chunk ids whose ``shot`` no longer matches the ``content_sha256``
    recorded beside it (design section 8).

    "Was this shot written or generated" is exactly the question #34/#38/#39/#45
    each needed answered after the fact. An entry with no recorded hash is not
    reported: a hand-written plan has none, and calling every line of one
    "edited" would make the signal useless.
    """
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    path = Path(path)
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PlanError(f"could not read shot plan {path}: {exc}") from exc

    edited = []
    for entry in payload.get("shot", []):
        if not isinstance(entry, dict):
            continue
        recorded = entry.get("content_sha256")
        chunk_id = entry.get("chunk_id")
        if not isinstance(recorded, str) or not isinstance(chunk_id, int):
            continue
        if sha256_text(str(entry.get("shot", "")).strip()) != recorded:
            edited.append(chunk_id)
    return tuple(sorted(edited))


def write_plan(text: str, output_path: str | Path, *, force: bool = False) -> Path:
    """Write a composed plan, refusing to clobber one without ``force`` --
    the same rule and the same reason as ``--prepare`` (issue #52): an
    authored shot plan is real work somebody may already have started."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        logger.error("Refusing to overwrite an existing shot plan at %s", output_path)
        raise PlanError(
            f"{output_path} already exists -- an authored shot plan is real work; pass "
            "--force to overwrite it"
        )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write shot plan to %s", output_path)
        raise PlanError(f"could not write shot plan to {output_path}: {exc}") from exc
    logger.info("Wrote generated shot plan to %s", output_path)
    return output_path


__all__ = [
    "MAX_ERROR_ROUNDS",
    "BuiltPlan",
    "PlanCheck",
    "PlanError",
    "PlanIssue",
    "Provenance",
    "build_plan",
    "check_plan",
    "human_edited_chunks",
    "lint_comments_for",
    "objections_by_chunk",
    "render_plan_toml",
    "write_plan",
]
