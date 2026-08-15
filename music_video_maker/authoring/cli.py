"""``mvm-author``: the authoring-layer CLI (issue #54).

A **separate console script** from ``music-video-maker``, not a flag on it
(design section 2). The reason is stronger than convenience: keeping every
model call behind a second binary means "does the render binary ever call a
model?" is answerable by reading ``pyproject.toml``'s ``[project.scripts]``
table, and stays answerable by construction as this package grows, rather
than by everyone remembering not to wire a stage into ``cli.py``.
``tests/test_authoring_boundary.py`` makes the same rule mechanically
enforced, not just documented, for imports; this is the console-script half
of the same discipline.

No GPU, no ComfyUI, no custody handoff, ever -- same rule as ``--prepare``
(``music_video_maker.cli.prepare_shot_plan``), which this reuses via
:mod:`~music_video_maker.authoring.chunks` for the chunk skeleton every
stage plans against.

All four stages are here (design section 12's phases 1-4): ``concept``,
``beats``, ``photography`` and ``prose``, plus ``write`` (compose, lint,
revise, write the plan), ``all`` (every stage in order, stopping for approval
between each), ``status``, and ``--dry-run`` on any stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from music_video_maker.authoring import beats as beats_module
from music_video_maker.authoring import concept as concept_module
from music_video_maker.authoring import photography as photography_module
from music_video_maker.authoring import plan as plan_module
from music_video_maker.authoring import prose as prose_module
from music_video_maker.authoring.beats import Beat, BeatsValidationError, beat_length_requests
from music_video_maker.authoring.chunks import SkeletonError, load_chunk_skeleton
from music_video_maker.authoring.concept import ConceptValidationError, generate_concept
from music_video_maker.authoring.driver import ClaudeCliDriver, DriverError
from music_video_maker.authoring.hashing import sha256_file, sha256_text
from music_video_maker.authoring.photography import PhotographyValidationError
from music_video_maker.authoring.plan import PlanError, Provenance
from music_video_maker.authoring.prompts import (
    SHOT_WRITING_GUIDE_DOC,
    beats_system_prompt,
    concept_system_prompt,
    photography_system_prompt,
    prose_system_prompt,
)
from music_video_maker.authoring.prose import ProseValidationError
from music_video_maker.authoring.reanchor import plan_beats
from music_video_maker.authoring.session import (
    STAGES,
    AuthoringSession,
    StageRecord,
    session_path,
    stage_staleness,
)
from music_video_maker.config import ConfigError, RunConfig, load_config
from music_video_maker.contracts import AudioChunk
from music_video_maker.logging_setup import configure_logging
from music_video_maker.shot_plan import START_TOLERANCE_SECONDS, ShotLength

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_ERROR = 1

AUTHORING_DIRNAME = ".authoring"
RAW_DIRNAME = "raw"


def _run_dir(config_path: Path) -> Path:
    """Where a run's ``.authoring/`` state lives -- beside the config, the
    same convention ``--prepare`` uses for ``shot_plan.toml``'s default
    output path."""
    return config_path.resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvm-author",
        description=(
            "Authoring layer (issue #54): invent the concept, plan the shots, direct the "
            "photography -- all before the GPU starts. Every stage here calls a model "
            "through the Claude CLI and writes a file a human reviews; nothing here ever "
            "touches the render path, the GPU, or ComfyUI."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the run config file.")
    parser.add_argument("--log-level", default="INFO", help="Root log level (default: INFO).")

    subparsers = parser.add_subparsers(dest="command", required=True)

    concept_parser = subparsers.add_parser("concept", help="Run (or re-run) the concept stage.")
    concept_parser.add_argument(
        "--notes",
        default=None,
        help="Freeform hints/notes for the model -- steering on a fresh run, or a targeted "
        "objection on a re-run ('too bleak, and no rain').",
    )
    concept_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact system + user prompt this call would send, and call nothing.",
    )

    beats_parser = subparsers.add_parser(
        "beats",
        help="Run (or re-run) the beats stage: what happens in each chunk, and how the "
        "plants, contacts and consequences group into sequences.",
    )
    beats_parser.add_argument(
        "--notes",
        default=None,
        help="Freeform notes/objections for the model ('the printer gag needs more room').",
    )
    beats_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact system + user prompt this call would send, and call nothing.",
    )

    prose_parser = subparsers.add_parser(
        "prose",
        help="Run (or re-run) the prose stage: turn each beat into the shot line a render "
        "will compose. One model call per beat group.",
    )
    prose_parser.add_argument(
        "--groups",
        default=None,
        help="Only (re)write these beat groups, e.g. '3' or '3,5-7'. Everything else keeps "
        "the prose already approved for it.",
    )
    prose_parser.add_argument(
        "--notes",
        default=None,
        help="Freeform notes/objections for the model ('stop putting her in doorways').",
    )
    prose_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact system + user prompt this call would send, and call nothing.",
    )

    photography_parser = subparsers.add_parser(
        "photography",
        help="Run (or re-run) the photography stage: the film's look, and each shot's "
        "framing and movement.",
    )
    photography_parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        metavar="N",
        help="Generate N looks from N deliberately different framing stances and print "
        "them side by side, writing nothing. Then freeze one with --pick.",
    )
    photography_parser.add_argument(
        "--pick",
        type=int,
        default=None,
        metavar="N",
        help="Freeze candidate N (1-based) from the last --candidates run.",
    )
    photography_parser.add_argument(
        "--notes",
        default=None,
        help="Freeform notes/objections for the model ('less handheld, hold the wides').",
    )
    photography_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact system + user prompt this call would send, and call nothing.",
    )

    write_parser = subparsers.add_parser(
        "write",
        help="Compose the shot_plan.toml, check it through the real loaders, revise what "
        "they object to, and write it.",
    )
    write_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write it. Defaults to 'shot_plan.toml' next to --config.",
    )
    write_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plan. An authored shot plan is real work, so this "
        "refuses to clobber one without it -- the same rule as --prepare.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run every stage in order and write the plan, stopping for approval between "
        "each one.",
    )
    all_parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not stop for approval. The review loop is the point of this design, so "
        "this is for a re-run of something already reviewed, not a first pass.",
    )
    all_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the plan. Defaults to 'shot_plan.toml' next to --config.",
    )
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plan at the output path.",
    )

    subparsers.add_parser("status", help="Report every stage's state: ok, stale, or not started.")

    return parser


def _parse_groups(raw: str | None) -> set[int] | None:
    """``"3,5-7"`` -> ``{3, 5, 6, 7}``. ``None`` means every group.

    A range syntax here where ``--only-chunks`` deliberately refuses one, and
    for the opposite reason: a validation slice is a hand-picked set, but beat
    groups run in story order, so "the middle of the song" really is a range.
    """
    if raw is None:
        return None
    groups: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            low, _, high = part.partition("-")
            groups.update(range(int(low), int(high) + 1))
        else:
            groups.add(int(part))
    if not groups:
        raise ValueError(f"--groups {raw!r} names no beat group")
    return groups


def _load_run_config(config_path: Path) -> RunConfig | None:
    try:
        return load_config(config_path)
    except ConfigError:
        logger.exception("Failed to load run config from %s", config_path)
        return None


def _cmd_concept(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    try:
        chunks = load_chunk_skeleton(config)
    except SkeletonError:
        logger.exception("Could not load the chunk skeleton for %s", config_path)
        return EXIT_ERROR

    if args.dry_run:
        _print_prompts(
            concept_system_prompt(),
            concept_module.build_concept_prompt(config, chunks, hints=args.notes),
        )
        return EXIT_SUCCESS

    driver = ClaudeCliDriver()
    try:
        result = generate_concept(config, chunks, driver, hints=args.notes)
    except (DriverError, ConceptValidationError):
        logger.exception("Concept generation failed for %s", config_path)
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    authoring_dir = run_dir / AUTHORING_DIRNAME
    authoring_dir.mkdir(parents=True, exist_ok=True)

    concept_path = authoring_dir / "concept.json"
    concept_path.write_text(json.dumps(result.data, indent=2, sort_keys=True) + "\n", "utf-8")

    raw_dir = authoring_dir / RAW_DIRNAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "concept.txt").write_text(result.driver_result.raw, encoding="utf-8")

    session = AuthoringSession.load(session_path(run_dir))
    session.record(
        StageRecord(
            stage="concept",
            model=result.driver_result.model,
            completed_at=date.today().isoformat(),
            cost_usd=result.driver_result.cost_usd,
            input_hashes=result.input_hashes,
        )
    )
    session.save()

    logger.info(
        "Concept written to %s (model=%s, cost=%s)",
        concept_path,
        result.driver_result.model,
        f"${result.driver_result.cost_usd:.4f}"
        if result.driver_result.cost_usd is not None
        else "not reported",
    )
    print(f"logline: {result.data['logline']}")
    print(f"setting: {result.data['setting']}")
    print(f"tone: {result.data['tone']}")
    return EXIT_SUCCESS


def _load_concept(run_dir: Path) -> dict | None:
    """The approved concept this stage descends from, or ``None`` with an
    explanation logged.

    Stage 2's entire input is the concept, so running without one would
    generate a beat sheet descending from a paragraph nobody ever read --
    which is exactly the review point the design calls the tightest.
    """
    path = run_dir / AUTHORING_DIRNAME / "concept.json"
    if not path.exists():
        logger.error(
            "No concept at %s -- run `mvm-author --config <run.toml> concept` first; the "
            "beats stage plans against an approved concept, never against nothing.",
            path,
        )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read the concept at %s", path)
        return None
    if not isinstance(data, dict):
        logger.error("Concept at %s is not a JSON object", path)
        return None
    return data


def _cmd_beats(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    concept = _load_concept(run_dir)
    if concept is None:
        return EXIT_ERROR

    try:
        chunks = load_chunk_skeleton(config)
    except SkeletonError:
        logger.exception("Could not load the chunk skeleton for %s", config_path)
        return EXIT_ERROR

    if args.dry_run:
        _print_prompts(
            beats_system_prompt(),
            beats_module.build_beats_prompt(config, chunks, concept, notes=args.notes),
        )
        return EXIT_SUCCESS

    def reslice(requests: Sequence[ShotLength]) -> Sequence[AudioChunk]:
        """Re-cut the timeline the way a render honouring these lengths will.

        The same call ``--prepare --from-plan`` makes on the render side, and
        deliberately so: if these two ever disagree about what a chunk list
        is, the beats are anchored to a timeline no render produces."""
        return load_chunk_skeleton(config, shot_lengths=requests)

    driver = ClaudeCliDriver()
    try:
        plan = plan_beats(
            config, chunks, concept, driver, reslice=reslice, notes=args.notes
        )
    except (DriverError, BeatsValidationError, SkeletonError):
        logger.exception("Beat generation failed for %s", config_path)
        return EXIT_ERROR

    authoring_dir = run_dir / AUTHORING_DIRNAME
    (authoring_dir / RAW_DIRNAME).mkdir(parents=True, exist_ok=True)

    payload = {
        "beats": [beat.to_dict() for beat in plan.beats],
        "reanchored": plan.reanchored,
        "merges": [{"chunk_id": new, "merged_from": list(old)} for new, old in plan.merges],
    }
    beats_path = authoring_dir / "beats.json"
    beats_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    (authoring_dir / RAW_DIRNAME / "beats.txt").write_text(
        plan.result.driver_result.raw, encoding="utf-8"
    )

    session = AuthoringSession.load(session_path(run_dir))
    session.record(
        StageRecord(
            stage="beats",
            model=plan.result.driver_result.model,
            completed_at=date.today().isoformat(),
            cost_usd=plan.result.driver_result.cost_usd,
            input_hashes=plan.result.input_hashes,
        )
    )
    session.save()

    logger.info(
        "Beat sheet written to %s (%d beat(s), timeline %s)",
        beats_path,
        len(plan.beats),
        "re-cut by this sheet's own lengths" if plan.reanchored else "unchanged",
    )
    for beat in plan.beats:
        merged = f"  [merged {list(beat.merged_from)}]" if beat.merged_from else ""
        print(
            f"{beat.chunk_id:>4}  {beat.start:>8.3f}  {beat.beat_role:<13}"
            f"g{beat.beat_group:<3} {beat.beat}{merged}"
        )
    return EXIT_SUCCESS


def _load_stage_json(run_dir: Path, name: str, *, quiet: bool = False) -> dict | None:
    """Read one stage's persisted output, or ``None`` with a reason logged."""
    path = run_dir / AUTHORING_DIRNAME / f"{name}.json"
    if not path.exists():
        if not quiet:
            logger.error(
                "No %s at %s -- run `mvm-author --config <run.toml> %s` first.",
                name,
                path,
                name,
            )
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read %s", path)
        return None
    return data if isinstance(data, dict) else None


def _load_beats(run_dir: Path, *, quiet: bool = False) -> tuple[Beat, ...] | None:
    payload = _load_stage_json(run_dir, "beats", quiet=quiet)
    if payload is None:
        return None
    try:
        return tuple(Beat.from_dict(entry) for entry in payload.get("beats", []))
    except (KeyError, TypeError, ValueError):
        logger.exception("Beat sheet in %s is malformed", run_dir / AUTHORING_DIRNAME)
        return None


def _timeline_for_beats(
    config: RunConfig, beats: Sequence[Beat]
) -> tuple[AudioChunk, ...] | None:
    """Re-derive the frozen ``chunks_v1`` the beat sheet is anchored to.

    Re-slicing with the sheet's own ``length_seconds`` must reproduce exactly
    the timeline it was re-anchored onto -- the same idempotence
    ``--prepare --from-plan`` relies on. **Verified rather than assumed**: if
    an anchor has moved, the alignment has changed underneath the beat sheet
    and everything downstream would be written against audio that has since
    moved. That is ``ShotPlanDriftError``'s failure, caught one layer earlier
    where re-running a stage is the cheap fix.
    """
    chunks = load_chunk_skeleton(config, shot_lengths=beat_length_requests(beats))
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    drifted = [
        beat.chunk_id
        for beat in beats
        if beat.chunk_id not in by_id
        or abs(by_id[beat.chunk_id].start - beat.start) > START_TOLERANCE_SECONDS
    ]
    if drifted:
        logger.error(
            "The beat sheet is anchored to a timeline this run no longer produces: chunk(s) "
            "%s have moved or no longer exist. The alignment has changed since the beats "
            "were written (an edited lyrics file, a different model). Re-run "
            "`mvm-author --config <run.toml> beats`.",
            drifted,
        )
        return None
    return chunks


def _cmd_prose(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    concept = _load_concept(run_dir)
    beats = _load_beats(run_dir)
    if concept is None or beats is None:
        return EXIT_ERROR

    try:
        groups = _parse_groups(args.groups)
    except ValueError:
        logger.exception("Could not read --groups %r", args.groups)
        return EXIT_ERROR

    try:
        chunks = _timeline_for_beats(config, beats)
    except SkeletonError:
        logger.exception("Could not load the chunk skeleton for %s", config_path)
        return EXIT_ERROR
    if chunks is None:
        return EXIT_ERROR

    if args.dry_run:
        windows = prose_module.beat_windows(beats, groups)
        if not windows:
            logger.error("No beat group matches --groups %r", args.groups)
            return EXIT_ERROR
        _print_prompts(
            prose_system_prompt(),
            prose_module.build_prose_prompt(
                config, concept, windows[0], beats, chunks, camera={}, notes=args.notes
            ),
        )
        print(f"\n({len(windows)} window(s) would be sent; the first is shown above.)")
        return EXIT_SUCCESS

    driver = ClaudeCliDriver()
    try:
        result = prose_module.generate_prose(
            config, concept, beats, chunks, driver, notes=args.notes, groups=groups
        )
    except (DriverError, ProseValidationError):
        logger.exception("Prose generation failed for %s", config_path)
        return EXIT_ERROR

    # Merge over whatever is already on disk: --groups exists so a human can
    # re-roll one sequence, and the prose they approved for every other group
    # must survive that (design section 6's "targeted means targeted").
    existing = _load_stage_json(run_dir, "prose", quiet=True) or {}
    shots = {int(k): v for k, v in (existing.get("shots") or {}).items()}
    shots.update(result.shots)
    # Issue #59, merged on the same rule as `shots`: a re-rolled group must
    # not drop the on-screen cast another group already resolved.
    present = {int(k): list(v) for k, v in (existing.get("present") or {}).items()}
    present.update({k: list(v) for k, v in result.present.items()})

    authoring_dir = run_dir / AUTHORING_DIRNAME
    (authoring_dir / RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
    (authoring_dir / "prose.json").write_text(
        json.dumps(
            {
                "shots": {str(k): v for k, v in sorted(shots.items())},
                "present": {str(k): v for k, v in sorted(present.items()) if v},
                "warnings": [
                    {"chunk_id": issue.chunk_id, "message": issue.message}
                    for issue in result.issues
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for index, driver_result in enumerate(result.driver_results):
        (authoring_dir / RAW_DIRNAME / f"prose_{index:02d}.txt").write_text(
            driver_result.raw, encoding="utf-8"
        )

    session = AuthoringSession.load(session_path(run_dir))
    session.record(
        StageRecord(
            stage="prose",
            model=result.driver_results[0].model if result.driver_results else "unknown",
            completed_at=date.today().isoformat(),
            cost_usd=_total_cost(result.driver_results),
            input_hashes=result.input_hashes,
        )
    )
    session.save()

    logger.info(
        "Prose written for %d chunk(s) over %d model call(s); %d advisory warning(s)",
        len(result.shots),
        len(result.driver_results),
        len(result.issues),
    )
    for chunk_id in sorted(result.shots):
        print(f"{chunk_id:>4}  {result.shots[chunk_id]}")
    return EXIT_SUCCESS


def _cmd_photography(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    authoring_dir = run_dir / AUTHORING_DIRNAME
    concept = _load_concept(run_dir)
    beats = _load_beats(run_dir)
    if concept is None or beats is None:
        return EXIT_ERROR

    if args.pick is not None:
        return _pick_photography(authoring_dir, run_dir, args.pick)

    try:
        chunks = _timeline_for_beats(config, beats)
    except SkeletonError:
        logger.exception("Could not load the chunk skeleton for %s", config_path)
        return EXIT_ERROR
    if chunks is None:
        return EXIT_ERROR

    if args.dry_run:
        _print_prompts(
            photography_system_prompt(),
            photography_module.build_photography_prompt(
                config, concept, beats, chunks, notes=args.notes
            ),
        )
        return EXIT_SUCCESS

    driver = ClaudeCliDriver()
    count = args.candidates or 1
    try:
        results = photography_module.photography_candidates(
            config, concept, beats, chunks, driver, count=count, notes=args.notes
        )
    except (DriverError, PhotographyValidationError):
        logger.exception("Photography generation failed for %s", config_path)
        return EXIT_ERROR

    authoring_dir.mkdir(parents=True, exist_ok=True)
    (authoring_dir / RAW_DIRNAME).mkdir(parents=True, exist_ok=True)

    if args.candidates:
        # Written to a *candidates* file, never to photography.json: choosing
        # is the human's move (design section 4's "that one, keep it"), and
        # silently adopting one would skip the only review this stage has.
        payload = [
            {
                "stance_index": result.stance_index,
                "stance": photography_module.CANDIDATE_STANCES[result.stance_index]
                if result.stance_index is not None
                else None,
                "model": result.driver_result.model,
                "cost_usd": result.driver_result.cost_usd,
                "photography": _photography_payload(result.photography),
                "input_hashes": result.input_hashes,
            }
            for result in results
        ]
        (authoring_dir / "photography_candidates.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for index, result in enumerate(results, start=1):
            (authoring_dir / RAW_DIRNAME / f"photography_candidate_{index:02d}.txt").write_text(
                result.driver_result.raw, encoding="utf-8"
            )
        _print_candidates(results)
        print(
            f"\nNothing has been adopted. Freeze one with: mvm-author --config "
            f"{args.config} photography --pick <1-{len(results)}>"
        )
        return EXIT_SUCCESS

    result = results[0]
    _write_photography(authoring_dir, run_dir, result.photography, result.input_hashes,
                       model=result.driver_result.model, cost=result.driver_result.cost_usd)
    (authoring_dir / RAW_DIRNAME / "photography.txt").write_text(
        result.driver_result.raw, encoding="utf-8"
    )
    _print_photography(result.photography)
    return EXIT_SUCCESS


def _photography_payload(photography) -> dict:
    return {
        "cinematography": photography.cinematography,
        "camera": {str(k): v for k, v in sorted(photography.camera.items())},
    }


def _write_photography(
    authoring_dir: Path,
    run_dir: Path,
    photography,
    input_hashes: dict[str, str],
    *,
    model: str,
    cost: float | None,
) -> None:
    (authoring_dir / "photography.json").write_text(
        json.dumps(_photography_payload(photography), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    session = AuthoringSession.load(session_path(run_dir))
    session.record(
        StageRecord(
            stage="photography",
            model=model,
            completed_at=date.today().isoformat(),
            cost_usd=cost,
            input_hashes=input_hashes,
        )
    )
    session.save()


def _pick_photography(authoring_dir: Path, run_dir: Path, pick: int) -> int:
    path = authoring_dir / "photography_candidates.json"
    if not path.exists():
        logger.error(
            "No candidates at %s -- run `photography --candidates N` before --pick.", path
        )
        return EXIT_ERROR
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read %s", path)
        return EXIT_ERROR
    if not isinstance(payload, list) or not 1 <= pick <= len(payload):
        logger.error(
            "--pick %d is out of range; %d candidate(s) available.",
            pick,
            len(payload) if isinstance(payload, list) else 0,
        )
        return EXIT_ERROR

    chosen = payload[pick - 1]
    photography = photography_module.Photography(
        cinematography=chosen["photography"]["cinematography"],
        camera={int(k): v for k, v in chosen["photography"]["camera"].items()},
    )
    _write_photography(
        authoring_dir,
        run_dir,
        photography,
        dict(chosen.get("input_hashes", {})),
        model=chosen.get("model", "unknown"),
        cost=chosen.get("cost_usd"),
    )
    print(f"Adopted candidate {pick}.")
    _print_photography(photography)
    return EXIT_SUCCESS


def _print_candidates(results: Sequence[object]) -> None:
    for index, result in enumerate(results, start=1):
        photography = result.photography
        stance = (
            photography_module.CANDIDATE_STANCES[result.stance_index]
            if result.stance_index is not None
            else "(no stance)"
        )
        print("=" * 80)
        print(f"CANDIDATE {index}")
        print("=" * 80)
        print(f"stance: {stance}")
        if photography.cinematography:
            print(f"look:   {photography.cinematography}")
        print(f"camera: {len(photography.camera)} shot(s) directed")
        for chunk_id in sorted(photography.camera)[:6]:
            print(f"  {chunk_id:>4}  {photography.camera[chunk_id]}")
        if len(photography.camera) > 6:
            print(f"  ... and {len(photography.camera) - 6} more")
        print()


def _print_photography(photography) -> None:
    if photography.cinematography:
        print(f"cinematography: {photography.cinematography}")
        print("  ^ this is a proposal for run.toml's `cinematography`; copy it there if you")
        print("    want it. It is a config value, not a shot-plan field, so `write` cannot")
        print("    put it in the plan for you.")
    print(f"camera directions: {len(photography.camera)} shot(s)")
    for chunk_id in sorted(photography.camera):
        print(f"  {chunk_id:>4}  {photography.camera[chunk_id]}")


def _load_camera(run_dir: Path) -> dict[int, str]:
    """Whatever photography has frozen, or ``{}``. Absent is normal: the
    photography stage is optional, and an absent ``camera`` composes
    nothing."""
    payload = _load_stage_json(run_dir, "photography", quiet=True)
    if payload is None:
        return {}
    return {int(k): v for k, v in (payload.get("camera") or {}).items()}


def _total_cost(results: Sequence[object]) -> float | None:
    """Summed cost, or ``None`` when nothing reported one -- never a
    fabricated zero, the same rule ``DriverResult.cost_usd`` follows."""
    costs = [r.cost_usd for r in results if getattr(r, "cost_usd", None) is not None]
    return sum(costs) if costs else None


def _cmd_write(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    concept = _load_concept(run_dir)
    beats = _load_beats(run_dir)
    prose_payload = _load_stage_json(run_dir, "prose")
    if concept is None or beats is None or prose_payload is None:
        return EXIT_ERROR
    shots = {int(k): v for k, v in (prose_payload.get("shots") or {}).items()}
    # Issue #59: who is on screen but not singing, resolved by the prose stage
    # into cast names the render can attach a role and a photo to.
    present = {int(k): list(v) for k, v in (prose_payload.get("present") or {}).items()}

    try:
        chunks = _timeline_for_beats(config, beats)
    except SkeletonError:
        logger.exception("Could not load the chunk skeleton for %s", config_path)
        return EXIT_ERROR
    if chunks is None:
        return EXIT_ERROR

    session = AuthoringSession.load(session_path(run_dir))
    provenance = Provenance(
        generated_by=f"mvm-author {_version()}",
        generated_at=date.today().isoformat(),
        models={
            stage: record.model
            for stage in ("concept", "beats", "prose")
            if (record := session.get(stage)) is not None
        },
        hashes={
            "concept": sha256_text(json.dumps(concept, sort_keys=True)),
            "guide": sha256_file(SHOT_WRITING_GUIDE_DOC),
            "lyrics": sha256_file(config.lyrics_file),
            "skeleton": sha256_text(beats_module.skeleton_table_text(chunks)),
        },
    )

    driver = ClaudeCliDriver()

    def reviser(current_shots, objections):
        return prose_module.revise_prose(
            config,
            concept,
            beats,
            chunks,
            driver,
            shots=current_shots,
            objections=objections,
            camera=camera,
        )

    camera = _load_camera(run_dir)

    def advisory(current_shots):
        """The prose stage's own warning-tier prohibitions, re-derived on the
        current text so a revision round cannot leave the file annotated with
        a complaint about a sentence that no longer exists."""
        return prose_module.advisory_issues(current_shots, camera=camera)

    try:
        built = plan_module.build_plan(
            config,
            chunks,
            beats,
            shots,
            provenance=provenance,
            reviser=reviser,
            camera=camera,
            present=present,
            extra_checks=advisory,
        )
    except (PlanError, DriverError, ProseValidationError):
        logger.exception("Could not compose a shot plan that loads cleanly")
        return EXIT_ERROR

    out_path = args.out or (run_dir / "shot_plan.toml")
    try:
        written = plan_module.write_plan(built.text, out_path, force=args.force)
    except PlanError:
        logger.exception("Failed to write the shot plan")
        return EXIT_ERROR

    print(f"Wrote {written} ({len(chunks)} chunk(s), {len(camera)} with camera direction)")
    if built.surviving_warnings:
        print(
            f"{len(built.surviving_warnings)} advisory warning(s) written into the file as "
            "'# lint:' comments -- read them, and expect some false positives:"
        )
        for issue in built.surviving_warnings:
            where = f"chunk {issue.chunk_id}" if issue.chunk_id is not None else "file"
            print(f"  [{where}] {issue.message}")
    return EXIT_SUCCESS


def _approve(stage: str, *, skip: bool) -> bool:
    """Ask whether to carry on to the next stage.

    Non-interactive stdin means **no**, not yes: an unattended ``all`` that
    silently approved its own output would spend four model calls and write a
    plan nobody looked at, which is the one thing this design's review loop
    exists to prevent. ``--yes`` is the deliberate way to say so.
    """
    if skip:
        return True
    if not sys.stdin.isatty():
        print(
            f"\n{stage} is done and nothing is reading stdin, so this is stopping here "
            "rather than approving itself. Re-run with --yes to go straight through."
        )
        return False
    answer = input(f"\nAccept {stage} and continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _cmd_all(args: argparse.Namespace) -> int:
    """Every stage in order, stopping for approval between each.

    Deliberately a thin sequence of the same command functions rather than a
    second code path: the stages are exactly what ``concept``/``beats``/
    ``photography``/``prose``/``write`` already do, and a separate
    implementation would be free to drift from them.
    """
    stages = (
        ("concept", _cmd_concept),
        ("beats", _cmd_beats),
        ("photography", _cmd_photography),
        ("prose", _cmd_prose),
    )
    for stage, handler in stages:
        stage_args = argparse.Namespace(
            config=args.config,
            notes=None,
            dry_run=False,
            groups=None,
            candidates=None,
            pick=None,
        )
        print(f"\n{'=' * 80}\n{stage.upper()}\n{'=' * 80}")
        exit_code = handler(stage_args)
        if exit_code != EXIT_SUCCESS:
            logger.error("Stopping: the %s stage failed.", stage)
            return exit_code
        if not _approve(stage, skip=args.yes):
            print(
                f"Stopped after {stage}. Everything up to here is saved in .authoring/; "
                f"re-run `mvm-author --config {args.config} {stage} --notes '...'` to "
                "revise it, or continue with the next stage when you are happy."
            )
            return EXIT_SUCCESS

    print(f"\n{'=' * 80}\nWRITE\n{'=' * 80}")
    return _cmd_write(
        argparse.Namespace(config=args.config, out=args.out, force=args.force)
    )


def _version() -> str:
    """This tool's version, for the provenance block.

    Read from the installed distribution's metadata rather than by importing
    ``music_video_maker.__version__``: the import-direction test
    (``tests/test_authoring_boundary.py``) allows this package to reach into
    a small named list of render-side modules, and the top-level package is
    not on it. Loosening that boundary to stamp a cosmetic string would be a
    bad trade -- it caught this on the first run.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("music-video-maker")
    except PackageNotFoundError:  # not pip-installed (a source checkout on sys.path)
        return "unknown"


def _print_prompts(system: str, prompt: str) -> None:
    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(system)
    print()
    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(prompt)


def _cmd_status(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = _load_run_config(config_path)
    if config is None:
        return EXIT_ERROR

    run_dir = _run_dir(config_path)
    session = AuthoringSession.load(session_path(run_dir))

    # Computed once and shared: every implemented stage's staleness is a
    # function of the same skeleton, and re-aligning per row would make one
    # `status` call do Stage 1 four times.
    chunks: tuple[AudioChunk, ...] | None = None
    skeleton_error: str | None = None
    try:
        chunks = load_chunk_skeleton(config)
    except SkeletonError as exc:
        skeleton_error = str(exc)

    concept = _peek_concept(run_dir)
    beats = _load_beats(run_dir, quiet=True)

    rows: list[tuple[str, str, str]] = []
    for stage in STAGES:
        record = session.get(stage)
        current: dict[str, str] | None = None
        if chunks is not None and stage == "concept":
            current = concept_module.concept_input_hashes(config, chunks)
        elif chunks is not None and stage == "beats" and concept is not None:
            current = beats_module.beats_input_hashes(config, chunks, concept)
        elif chunks is not None and stage == "photography" and concept is not None and beats:
            current = photography_module.photography_input_hashes(
                config, _prose_timeline(config, beats, chunks), concept, beats
            )
        elif chunks is not None and stage == "prose" and concept is not None and beats:
            # Prose is hashed against the *beats' own* timeline, not the
            # skeleton the concept was written from -- otherwise a beat sheet
            # that legitimately re-cut the timeline would report its own prose
            # stale forever.
            current = prose_module.prose_input_hashes(
                config, _prose_timeline(config, beats, chunks), concept, beats
            )

        if record is None:
            rows.append((stage, "not started", ""))
            continue
        cost = f"${record.cost_usd:.4f}" if record.cost_usd is not None else "?"
        detail = f"{record.model}  {record.completed_at}  {cost}"
        if current is None:
            # No current-hash provider: either Stage 1 failed (report why) or
            # this stage is not implemented yet, in which case reporting what
            # is persisted without a live staleness verdict is the honest
            # answer rather than a guessed one.
            rows.append((stage, "ERROR" if skeleton_error else "ok", skeleton_error or detail))
            continue
        check = stage_staleness(record, current)
        rows.append((stage, "STALE", check.reason or "") if check.stale else (stage, "ok", detail))

    if skeleton_error and not session.records:
        rows = [(stage, "ERROR", skeleton_error) for stage, _, _ in rows]

    for stage, state, detail in rows:
        print(f"{stage:<14}{state:<12}{detail}")

    _report_human_edits(config, run_dir)
    return EXIT_SUCCESS


def _prose_timeline(
    config: RunConfig, beats: Sequence[Beat], fallback: Sequence[AudioChunk]
) -> tuple[AudioChunk, ...]:
    """The beats' own timeline for staleness reporting, falling back to the
    skeleton when it cannot be re-derived. ``status`` must never fail because
    a re-slice did; the worst case is one row reading STALE, which is the
    honest answer when the timeline cannot be reproduced anyway."""
    try:
        recut = _timeline_for_beats(config, beats)
    except SkeletonError:
        return tuple(fallback)
    return recut if recut is not None else tuple(fallback)


def _report_human_edits(config: RunConfig, run_dir: Path) -> None:
    """Say which generated shot lines a human has since edited (design
    section 8) -- the question #34/#38/#39/#45 each needed answered after the
    fact. Reported, never reverted."""
    path = config.shot_plan or (run_dir / "shot_plan.toml")
    if not Path(path).exists():
        return
    try:
        edited = plan_module.human_edited_chunks(path)
    except PlanError:
        logger.warning("Could not check %s for human edits", path)
        return
    if edited:
        print(
            f"\n{len(edited)} shot line(s) in {path} have been edited since they were "
            f"generated: {list(edited)}"
        )


def _peek_concept(run_dir: Path) -> dict | None:
    """The concept on disk, or ``None`` -- quietly, unlike :func:`_load_concept`.

    ``status`` reporting "beats: not started" is not the place to complain
    that there is no concept yet; that is the normal state before anything
    has run."""
    path = run_dir / AUTHORING_DIRNAME / "concept.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable concept at %s for staleness reporting", path)
        return None
    return data if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper())

    if args.command == "concept":
        return _cmd_concept(args)
    if args.command == "beats":
        return _cmd_beats(args)
    if args.command == "photography":
        return _cmd_photography(args)
    if args.command == "prose":
        return _cmd_prose(args)
    if args.command == "write":
        return _cmd_write(args)
    if args.command == "all":
        return _cmd_all(args)
    if args.command == "status":
        return _cmd_status(args)
    raise AssertionError(f"unreachable: unknown command {args.command!r}")  # argparse enforces this


if __name__ == "__main__":
    sys.exit(main())
