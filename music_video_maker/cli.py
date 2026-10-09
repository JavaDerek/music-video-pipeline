"""CLI entrypoint: the ``main()`` conductor (issue #14, Wave 4).

Runs the full pipeline end-to-end from a validated :class:`~music_video_maker.config.RunConfig`
to a finished, lip-synced music video:

1. Load + validate config (#2).
2. Stage 1: parse lyric character tags (#6) and force-align to the master
   track (#3).
3. Stage 2: slice audio into chunks (#4) and expand each chunk's prompt (#5).
4. Load the workflow template(s) once; stage each chunk's assets (#7) and
   render every chunk under the fault-tolerant state machine (#10), with
   optional I2V continuity bridging (#12) deciding per chunk whether to
   render through the base or I2V template (#8, #9).
5. Stage 5: assemble + mux the final video (#11).

Steps 2-5 run under the GPU custody protocol (issue #19) -- the card is
confirmed actually free before anything is submitted, and ComfyUI's VRAM is
released unconditionally on the way out, via
:func:`music_video_maker.custody.build_custody_manager`. The split matches
the scope issue #14's own comment settled on: config loading itself needs
no GPU, everything after it
does.

Per-chunk render progress is the WebSocket ``progress`` events
``execution.ComfyUIExecutionClient`` already logs at ``INFO`` -- this module
adds no separate progress bar, since stderr logging *is* the progress
display (project ``CLAUDE.md``: all diagnostic output goes to stderr).

Every I/O seam a stage module already made injectable (the stable-ts model,
the ComfyUI HTTP session and WebSocket factory, the ffmpeg/ffprobe runner,
the resilience backoff sleeper and disk-usage probe, the wall-clock) is
threaded through :func:`run_pipeline` the same way, so an end-to-end dry run
against a fully mocked ComfyUI is possible with no real GPU, network, or
server -- see ``tests/test_cli.py``.

``--prepare`` (issue #52) is a separate, much smaller entry point:
:func:`prepare_shot_plan` runs Stages 1-2 only and writes a
``shot_plan.toml`` skeleton, never touching the GPU, ComfyUI, or GPU
custody. It shares Stage 1/2 with :func:`run_pipeline` but is not a mode of
it -- see :func:`prepare_shot_plan`'s own docstring.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import date
from pathlib import Path
from typing import Any

import requests

from music_video_maker.alignment import align
from music_video_maker.alignment_quality import (
    AlignmentQualityReport,
    evaluate_alignment_quality,
    suspect_segment_indices,
)
from music_video_maker.assembly import (
    TimelineAssembly,
    assemble_final_video,
    assemble_timelines,
)
from music_video_maker.config import ConfigError, RunConfig, load_config
from music_video_maker.continuity import ContinuityWorkflowProvider, planned_chain_source
from music_video_maker.contracts import (
    AlignmentResult,
    AudioChunk,
    ChunkFingerprint,
    ChunkStatus,
    RunState,
    Workflow,
)
from music_video_maker.custody import (
    RenderStack,
    build_custody_manager,
    build_vram_probe,
    build_vram_releaser,
    prevent_host_sleep,
    read_render_stack,
)
from music_video_maker.diarization import Diarizer, assign_characters, lazy_pyannote_diarizer
from music_video_maker.envelope import check_render_envelope, measured_ceiling
from music_video_maker.execution import ComfyUIExecutionClient
from music_video_maker.faces import build_seed_face_gate
from music_video_maker.hardware import scan_workflow_for_missing_optimizations
from music_video_maker.logging_setup import configure_logging
from music_video_maker.lyrics import parse_lyrics
from music_video_maker.prepare_report import (
    InputStamp,
    build_prepare_report,
    collect_stage_notices,
    plan_resolution_errors,
    write_prepare_report,
)
from music_video_maker.profiles import LOOK_FIELDS as PROFILE_LOOK_FIELDS
from music_video_maker.profiles import PROFILE_RECORD_FILENAME, write_profile_record
from music_video_maker.prompting import expand_prompt
from music_video_maker.resilience import DiskUsage, ResilientRunner, Sleeper
from music_video_maker.shot_plan import (
    ShotLength,
    ShotPlanEntry,
    ShotPlanError,
    lint_camera_face_away_on_voiced_chunks,
    lint_instrumental_focus_mismatch,
    lint_mouth_direction_on_instrumental_chunks,
    lint_present_location_mismatch,
    lint_role_prohibition_contradiction,
    lint_shots_against_lyrics,
    lint_subject_on_voiced_chunk,
    lint_unbound_companion_referent,
    lint_voiced_framing,
    load_shot_plan,
    resolve_camera,
    resolve_conditions,
    resolve_framing,
    resolve_location,
    resolve_present,
    resolve_shot,
    resolve_subject,
    shot_length_requests,
    write_shot_plan_skeleton,
)
from music_video_maker.slicing import slice_audio, timeline_track_drift_seconds
from music_video_maker.staging import ComfyUIAssetStager
from music_video_maker.stems import slice_stem_for_chunks
from music_video_maker.timelines import (
    SONG_TIMELINE_NAME,
    SeamOverrunError,
    Timeline,
    log_placements,
    place_timelines,
    plan_timelines,
    predicted_measurements,
)
from music_video_maker.workflow_graph import (
    PerChunkSeedMutator,
    WorkflowGraphMutator,
    graph_fingerprint,
    load_workflow_template,
    read_render_dimensions,
    read_text_encoder,
    resolve_chunk_seed,
)

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_PARTIAL_FAILURE = 2
"""Some chunk(s) dead-lettered; the run finished but is incomplete."""

DEFAULT_RESEED_GENERATION = 1
"""``--reseed``'s default alternate take (issue #38 CLI). Generation 0 is
what every chunk gets without ``--reseed`` at all, so 1 is the first value
guaranteed (see ``workflow_graph.resolve_chunk_seed``) to differ from a
chunk's existing take. A chunk reseeded once and still wrong can be reseeded
again with ``--reseed-generation 2``, 3, ... rather than silently repeating
the same alternate seed forever."""


class PipelineError(RuntimeError):
    """A Wave 4 orchestration failure not already a typed error from an
    earlier stage (e.g. slicing producing zero chunks to render)."""


def chain_scope_ids(scope: str, chunks: Sequence[Any]) -> frozenset[int] | None:
    """Resolve ``i2v_chain_scope`` into the provider's eligibility set
    (issue #28).

    ``None`` means every chunk is eligible (the provider warns loudly, since
    chained chunks cannot lip-sync); an empty set means none is. The default
    ``"instrumental"`` chains exactly the chunks with no lyric to sync."""
    if scope == "all":
        return None
    if scope == "none":
        return frozenset()
    return frozenset(c.chunk_id for c in chunks if c.is_instrumental)


def _resolve_seed_face_gate(
    config: RunConfig, injected: Callable[[Path, Path], bool] | None
) -> Callable[[Path, Path], bool] | None:
    """The seed-face predicate this run should use (issue #47), or ``None`` to
    chain from whatever the previous shot ended on.

    Injectable like every other I/O seam in this module, so the offline test
    suite exercises the *decision* without OpenCV, a model file, or a real face
    anywhere -- the detector itself is unit-tested separately.

    Issue #49: ``ContinuityWorkflowProvider`` always threads the active cast
    member's reference photo down to whatever gate it holds (see
    ``continuity.py``'s ``_stage_seed_frame``) -- that is just a field read
    and costs nothing to do unconditionally. Whether the photo actually gets
    *used* for recognition is decided here, from
    ``config.i2v_min_seed_face_similarity``, which defaults to ``None``
    (recognition off, matching every config written before this knob
    existed). When it is ``None`` the returned gate discards whatever photo
    it is handed and answers detection-only, exactly as issue #47 shipped
    it -- a chunk's reference photo happening to be available must not
    silently upgrade a run to a stricter check nobody asked for."""
    if injected is not None:
        return injected
    if not (config.i2v_continuity and config.i2v_require_seed_face):
        return None
    if config.i2v_min_seed_face_similarity is None:
        base_gate = build_seed_face_gate(min_fraction=config.i2v_min_seed_face_fraction)
        return lambda frame_path, _reference_photo=None: base_gate(frame_path)
    return build_seed_face_gate(
        min_fraction=config.i2v_min_seed_face_fraction,
        min_similarity=config.i2v_min_seed_face_similarity,
    )


def run_shot_plan_lints(
    plan: Mapping[int, ShotPlanEntry], chunks: Sequence[Any], config: RunConfig
) -> None:
    """Every shot-plan lint a real render runs against ``plan``, in the
    render's own order (issue #36).

    Factored out of :func:`run_pipeline` so the review page can run *this
    exact function* -- not a copy of it -- with a logging handler attached,
    the same "checked by the render's own loaders, never a copy of them"
    rule ``authoring/plan.check_plan`` already follows on the authoring side
    (see that module's docstring). See ``music_video_maker/review.py``.

    Raises whatever the first raising lint raises (today, only
    :func:`~music_video_maker.shot_plan.lint_subject_on_voiced_chunk`'s
    :class:`~music_video_maker.shot_plan.ShotPlanError`) -- callers that want
    to keep going past a structural plan error catch it themselves; this
    function does not soften it, the same as when this code lived inline in
    :func:`run_pipeline`.
    """
    # Issue #82: raises, so it goes first -- no point running
    # advisory lints on a plan a raising lint will refuse outright.
    # `subject` is legal only on an instrumental chunk; honouring it
    # on a voiced one would reintroduce the desync it exists to fix.
    lint_subject_on_voiced_chunk(plan, chunks)
    # Issue #37: both strings are in hand here -- a lyric naming an
    # object the plan stages only elsewhere is mechanically visible,
    # for free, before any GPU time is spent on it. Issue #67: the
    # run's own lyric_literalness decides how loudly this fires --
    # silenced at "free", promoted to ERROR at "literal" -- but the
    # render never refuses on it either way. "A false positive must
    # never block a run" and "one chunk failing must not kill the
    # run" both still apply here; the error tier only means
    # something in the authoring layer's revision round.
    lint_shots_against_lyrics(plan, chunks, literalness=config.lyric_literalness)
    # Issue #58: a camera direction that turns her away from the
    # lens on a voiced chunk costs that chunk's lip-sync.
    lint_camera_face_away_on_voiced_chunks(plan, chunks)
    # The mirror of the check above: an INSTRUMENTAL chunk whose own
    # text asks for the mouth the render is telling H3 to keep still.
    for finding in lint_mouth_direction_on_instrumental_chunks(plan, chunks):
        logger.warning(
            "Shot plan chunk_id=%d: this chunk is instrumental, so its prompt "
            "says the character stays silent -- but its %s names %r (%r). H3 "
            "renders the nouns it is given, so the prompt asks for the mouth it "
            "also forbids, and a viewer sees someone mouthing words with no "
            "audio. Describe what the shot shows without naming the mouth.",
            finding.chunk_id, finding.field, finding.matched, finding.text,
        )
    # A sung chunk whose `framing` is wide (or unset) has no face big
    # enough to read a mouth -- the one thing the whole pipeline is for
    # (issue #97: `framing` sets size, `camera` wording does not). Also
    # warns on a sung chunk turned away from the lens (#58, #76).
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
    # Issue #82: an instrumental chunk whose shot line reads as
    # entirely about a `present` bystander, with no `subject` set to
    # tell the render that -- the general shape of the chunk 29 bug.
    lint_instrumental_focus_mismatch(plan, chunks, config.default_lead_vocalist)


def _select_render_ids(chunks: Sequence[Any], only_chunks: Sequence[int] | None) -> list[int]:
    """Which chunk ids to actually render -- all of them, or a validation slice.

    Only the *render list* narrows. Stages 1-2 still run over the whole track,
    because a chunk's span is a function of the entire timeline: re-slicing
    around a selected chunk would give it a different start, a different frame
    count and a different prompt, and the slice would then validate a chunk the
    real run is never going to produce. This is the standing rule that prompt-
    shape and gate changes get proved on 1-3 chunks before a full run costs
    hours of exclusive GPU custody.

    An id that is not in the song is an error, not an empty render: a slice
    that silently renders nothing reports success exactly like one that
    finished.
    """
    available = [chunk.chunk_id for chunk in chunks]
    if not only_chunks:
        return available

    unknown = sorted(set(only_chunks) - set(available))
    if unknown:
        logger.error(
            "--only-chunks names chunk id(s) %s that this song does not have; it has %d "
            "chunk(s), %s..%s",
            unknown,
            len(available),
            available[0],
            available[-1],
        )
        raise PipelineError(
            f"--only-chunks names unknown chunk id(s) {unknown}; this run has "
            f"{len(available)} chunk(s), {available[0]}..{available[-1]}"
        )

    selected = [chunk_id for chunk_id in available if chunk_id in set(only_chunks)]
    logger.info(
        "Validation slice: rendering %d of %d chunk(s) -- %s",
        len(selected),
        len(available),
        ", ".join(str(i) for i in selected),
    )
    return selected


def _amend_from_render(
    provider: ContinuityWorkflowProvider, chunk_id: int, fp: ChunkFingerprint
) -> ChunkFingerprint:
    """Replace what the run *planned* for a chunk with what it actually did.

    Three of the fingerprint's fields cannot be known before render time,
    because they are decided by which of the two templates the provider routed
    this chunk through -- and that is not final until the moment it renders,
    since a chunk whose predecessor dead-lettered falls back to the base path:

    * ``chained_from`` (#28) -- degradations included, not the plan.
    * ``prompt_hash`` (#46) -- the chained path is deliberately told a
      different sentence, so hashing the planned prompt would record one the
      render never used.
    * ``template_hash`` (#45) -- the graph a chunk rendered through is the one
      of two it actually took.

    Each is amended only when the provider has something to say, so a chunk it
    never decided keeps its planned value rather than being overwritten with a
    fabricated ``None``.
    """
    amended = dc_replace(fp, chained_from=provider.chain_source(chunk_id))

    prompt_text = provider.prompt_text(chunk_id)
    if prompt_text is not None:
        amended = dc_replace(amended, prompt_hash=ChunkFingerprint.hash_prompt(prompt_text))

    template_hash = provider.template_hash(chunk_id)
    if template_hash is not None:
        amended = dc_replace(amended, template_hash=template_hash)

    return amended


def _resolve_text_encoder(
    config: RunConfig, base_template: Workflow, i2v_template: Workflow | None
) -> str | None:
    """Which text encoder this run's chunks are actually encoded with
    (issue #39), for the record every ``ChunkFingerprint`` carries.

    A pinned ``config.text_encoder`` is authoritative -- it is injected into
    every template's ``CLIPLoader``, so it is what every chunk uses whatever
    the templates say. Unpinned, the answer is what the base template names,
    which is what the render will load.

    The one case that cannot be recorded honestly is unpinned templates that
    disagree: the base and I2V paths would then encode at different
    precisions within one video, and a single fingerprint field cannot be
    true of both. That is a template-hygiene mistake rather than a run this
    project ever intends, so it is named loudly -- with the fix (pin
    ``text_encoder``) -- rather than silently recorded as the base value.
    """
    if config.text_encoder is not None:
        return config.text_encoder

    encoder = read_text_encoder(base_template)
    if encoder is None:
        logger.warning(
            "workflow template %s names no CLIPLoader.clip_name -- this run's chunks will be "
            "fingerprinted with an unrecorded text encoder, so --resume cannot prove a cached "
            "chunk was encoded the same way (issue #39)",
            config.workflow_template,
        )
        return None

    i2v_encoder = read_text_encoder(i2v_template) if i2v_template is not None else None
    if i2v_template is not None and i2v_encoder != encoder:
        logger.warning(
            "the two workflow templates name different text encoders (base=%s, i2v=%s) and "
            "this run pins neither, so chained and unchained chunks would be encoded at "
            "different precisions and only one of them can be recorded. Set text_encoder in "
            "the run config to pin both (issue #39).",
            encoder,
            i2v_encoder,
        )

    logger.info("text encoder for this run: %s (from %s)", encoder, config.workflow_template)
    return encoder


@dataclass(frozen=True)
class RunReport:
    """Issue #14's final report: chunks rendered/cached/dead-lettered, total
    wall time, and the output path."""

    run_state: RunState
    """The **song's** run state. Still the song's with a prologue configured,
    because that is what every existing caller means by it and because chunk
    ids are a separate space per timeline -- merging two states into one dict
    would collide ids that are not the same chunk (issue #66)."""
    total_chunks: int
    wall_seconds: float
    output_video: Path | None

    timeline_states: tuple[tuple[str, RunState], ...] = ()
    """``(timeline name, state)`` for every timeline this run rendered, in
    playback order, including the song (issue #66). An optional field: a
    hand-built report (or one from before segments existed) leaves it empty
    and every property below falls back to ``run_state`` alone, so nothing
    that reads this class changed meaning."""

    def _states(self) -> tuple[RunState, ...]:
        if self.timeline_states:
            return tuple(state for _name, state in self.timeline_states)
        return (self.run_state,)

    @property
    def rendered(self) -> int:
        return sum(
            1
            for state in self._states()
            for r in state.results.values()
            if r.status is ChunkStatus.RENDERED
        )

    @property
    def cached(self) -> int:
        return sum(
            1
            for state in self._states()
            for r in state.results.values()
            if r.status is ChunkStatus.CACHED
        )

    @property
    def dead_lettered(self) -> tuple[int, ...]:
        """Every dead-lettered chunk id across every timeline, deduplicated.

        Aggregated rather than song-only because this is what decides the
        process exit code: a prologue chunk that never rendered is a run that
        did not finish, and reporting success for it would be the same class
        of silence this project keeps finding. Which timeline each id came
        from is :attr:`dead_lettered_by_timeline` -- ids alone are ambiguous
        across id spaces, and that ambiguity is why both exist."""
        ids: set[int] = set()
        for state in self._states():
            ids.update(state.dead_lettered)
        return tuple(sorted(ids))

    @property
    def dead_lettered_by_timeline(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Dead-lettered ids attributed to the timeline they belong to."""
        if not self.timeline_states:
            return ((SONG_TIMELINE_NAME, self.run_state.dead_lettered),)
        return tuple(
            (name, state.dead_lettered)
            for name, state in self.timeline_states
            if state.dead_lettered
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-video-maker",
        description="Render a lip-synced music video from audio + lyrics + cast photos.",
    )
    parser.add_argument("--config", required=True, help="Path to the run config file.")
    parser.add_argument("--resume", action="store_true", help="Resume a partial run.")
    parser.add_argument(
        "--ignore-prompt-changes",
        action="store_true",
        help=(
            "On --resume, reuse cached chunks whose span is unchanged but whose prompt was "
            "edited (issue #34) -- a shot-plan tweak, a reworded cast role. Chunks whose span, "
            "frame count or render resolution moved are always re-rendered, flag or no flag."
        ),
    )
    parser.add_argument(
        "--strict-alignment",
        action="store_true",
        help=(
            "Refuse to render when forced alignment produces implausible timings (issue #35) "
            "-- zero-length segments, words placed where nothing is sung, a lyric line split "
            "across a huge gap. Alignment takes ~6s and a full render takes hours, so this is "
            "worth setting for an unattended run. Report-only without it."
        ),
    )
    parser.add_argument(
        "--only-chunks",
        type=_parse_chunk_ids,
        default=None,
        metavar="IDS",
        help=(
            "Render only these chunk ids (comma-separated, e.g. '32,33,34,35') and skip "
            "Stage 5 assembly. Stages 1-2 still run over the whole track, so each chunk "
            "gets exactly the span, prompt and frame count a full run would give it. This "
            "is the validation slice: prove a prompt-shape or gate change on a few chunks "
            "before committing hours of exclusive GPU custody to a full render. Without "
            "--resume the named chunks always re-render; with it they re-render only if "
            "their fingerprint changed, exactly as a full --resume would decide."
        ),
    )
    parser.add_argument(
        "--reseed",
        type=_parse_chunk_ids,
        default=None,
        metavar="IDS",
        help=(
            "Re-roll these chunk ids (comma-separated) under a different, deterministic "
            "seed (issue #38) and re-render just them, reusing every other cached chunk -- "
            "the common case of watching a render and finding one chunk bad. Implies "
            "--resume. The new seed is derived from --reseed-generation, never random, so a "
            "resumed --reseed run recomposes the same value rather than drifting further "
            "each time it is interrupted and restarted."
        ),
    )
    parser.add_argument(
        "--reseed-generation",
        type=int,
        default=DEFAULT_RESEED_GENERATION,
        metavar="N",
        help=(
            f"Which alternate take --reseed's chunks render (default {DEFAULT_RESEED_GENERATION}"
            "). Bump this if a previous --reseed of the same chunk(s) still was not right -- "
            "each generation is a distinct, reproducible seed, never the same one repeated. "
            "Ignored without --reseed."
        ),
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help=(
            "Run Stages 1-2 only -- alignment + slicing, ~6s, no GPU, no ComfyUI, no custody "
            "custody -- and write a shot_plan.toml skeleton with chunk_id/start/lyric filled "
            "in and shot left blank for you to author (issue #52). Every other flag except "
            "--shot-plan-out and --force is ignored."
        ),
    )
    parser.add_argument(
        "--shot-plan-out",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "With --prepare, where to write the shot-plan skeleton. Defaults to "
            "'shot_plan.toml' next to --config."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --prepare, overwrite an existing file at the output path. An authored "
            "shot plan is real work, so --prepare refuses to clobber one without this."
        ),
    )
    parser.add_argument(
        "--from-plan",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "With --prepare or --review, read an existing shot plan for its length_seconds "
            "only and re-anchor the timeline against the run that plan will actually "
            "produce. Without this, --prepare slices with no editorial lengths at all "
            "(there is usually no plan yet to read them from), and --review slices with "
            "--config's own shot_plan's lengths if it names one (matching what a real "
            "render does) or none if it doesn't. Pass this to check a candidate plan "
            "before it is wired into --config; it always overrides --config's own "
            "shot_plan for the lengths, whichever mode is running."
        ),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Run Stages 1-2 (like --prepare) and write a read-only review of the chunk "
            "timeline, the alignment-quality findings, and the shot plan's own warnings "
            "(issue #36) -- one self-contained HTML page plus the same data as JSON, no "
            "server, no GPU. Writes PATH with its suffix replaced by '.html' and '.json' "
            "(so 'out/review' or 'out/review.html' both produce 'out/review.html' and "
            "'out/review.json'). The timeline is sliced with --config's own shot_plan's "
            "editorial lengths, if it sets any -- the same lengths a real render with this "
            "config would use -- unless --from-plan names a different plan to check "
            "instead. Never raises on strict_alignment; if the config is strict and the "
            "report would in fact make a render refuse, the page says so at the top."
        ),
    )
    parser.add_argument(
        "--timeline",
        default=None,
        metavar="NAME",
        help=(
            "Which timeline --only-chunks/--reseed name chunk ids in (issue #66): "
            "'song' (the default) or a [[segment]] name. Chunk ids are a separate space "
            "per timeline -- prologue chunk 3 and song chunk 3 are different shots -- so "
            "a bare id is ambiguous the moment a config has segments, and this is how you "
            "say which one you mean. With --only-chunks, only the named timeline renders "
            "at all; the others are left alone, since a slice assembles nothing anyway."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Root log level (default: INFO)."
    )
    return parser


def _parse_chunk_ids(raw: str) -> tuple[int, ...]:
    """``"32,33, 35"`` -> ``(32, 33, 35)``, rejecting anything else loudly.

    Deliberately not a range syntax: a validation slice (``--only-chunks``) or
    a re-roll (``--reseed``) is usually a hand-picked set of chunks, not a
    contiguous run. Shared by both flags -- argparse prefixes any error this
    raises with the flag it was parsing for, so the message itself names
    neither."""
    try:
        ids = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expects comma-separated integers, got {raw!r}"
        ) from None
    if not ids:
        raise argparse.ArgumentTypeError("no chunk ids given")
    return ids


def _log_timeline_track_drift(
    chunks: Sequence[AudioChunk], track_duration: float, config: RunConfig
) -> float:
    """Issue #22: report Stage 2's chunk timeline against the master track's
    own duration -- before any GPU time is spent, from both ``run_pipeline``
    and ``--prepare`` (:func:`prepare_shot_plan`), which is the 50s no-GPU
    check this project already uses to catch a Stage-2 drift.

    CLAUDE.md's "-shortest" bullet: a music-video mux silently discards a
    timeline that overshoots the track -- harmless to watch, but real GPU
    seconds spent rendering frames nobody sees, and nothing has ever logged
    it. A silent concert backdrop (``silent_output``) has no mux to hide
    behind, so the same overshoot becomes a file that outruns the click
    track it was cut to, and the post-assembly ``expected_duration`` check
    wired up for that path (see ``run_pipeline``) WILL raise once the file
    is written. This function only reports -- it never refuses the run: the
    Stage-2 fix (trim the final tile, or pad the track to a legal grid
    length) is undecided and is a call for whoever owns Stage 2 with a real
    click track in hand (CLAUDE.md, ``docs/design-concert-mode.md``).

    An UNDERshoot -- the timeline finishing short of the track -- is
    reported too, and louder: it is the worse defect for a music video (the
    mux's ``-shortest`` stops at the end of the *video*, so the song's own
    ending is cut out of the finished file with no error). ``instrumental_coverage``
    now covers a short outro (see ``slicing._cover_instrumentals``), so this
    line firing means something upstream of that broke the invariant.

    Silent below ``config.duration_tolerance_seconds`` (default one frame at
    24 fps): that is the same window the post-assembly check itself treats
    as "no news," so this must not cry wolf inside it.

    Returns the drift in seconds (positive = overshoot, negative =
    undershoot) so a caller can reuse the number instead of recomputing it.
    """
    drift = timeline_track_drift_seconds(chunks, track_duration)
    if abs(drift) <= config.duration_tolerance_seconds:
        return drift

    fps = config.hardware.frame_grid.fps

    if drift > 0:
        if config.silent_output:
            logger.error(
                "Stage 2 timeline overshoots the master track by %.3fs (%.1f frames @%dfps): "
                "the silent_output file this run assembles will be %.3fs long against a "
                "%.3fs track, and the post-assembly duration check (tolerance=%.3fs) WILL "
                "raise once the file is written. Not refusing the run -- the fix is a Stage-2 "
                "decision (trim the final tile vs pad the click track) for whoever owns it "
                "with a click track in hand; see CLAUDE.md's '-shortest' bullet and "
                "docs/design-concert-mode.md.",
                drift,
                drift * fps,
                fps,
                track_duration + drift,
                track_duration,
                config.duration_tolerance_seconds,
            )
        else:
            logger.warning(
                "Stage 2 timeline overshoots the master track by %.3fs (%.1f frames @%dfps): "
                "the mux's -shortest will silently discard that many rendered frames from the "
                "final video. Harmless to watch, but it is GPU time spent rendering frames "
                "nobody will see -- CLAUDE.md's '-shortest' bullet, issue #22.",
                drift,
                drift * fps,
                fps,
            )
    else:
        undershoot = -drift
        if config.silent_output:
            logger.error(
                "Stage 2 timeline UNDERshoots the master track by %.3fs (%.1f frames @%dfps): "
                "the silent_output file this run assembles will be SHORTER than the %.3fs "
                "track, and the post-assembly duration check (tolerance=%.3fs) WILL raise "
                "once the file is written. Not refusing the run; see the overshoot log line "
                "for who owns the fix.",
                undershoot,
                undershoot * fps,
                fps,
                track_duration,
                config.duration_tolerance_seconds,
            )
        else:
            logger.error(
                "Stage 2 timeline UNDERshoots the master track by %.3fs (%.1f frames @%dfps): "
                "the mux's -shortest stops at the end of the video, so the last %.3fs of the "
                "song will be cut out of the final video. This is the worse defect of the two. Not "
                "refusing the run; a human should look at this before trusting the final cut.",
                undershoot,
                undershoot * fps,
                fps,
                undershoot,
            )
    return drift


@dataclass(frozen=True)
class TimelineRender:
    """One timeline's Stages 1-4 output (issue #66).

    Held rather than folded together because every consumer downstream needs
    it kept apart: Stage 5 concatenates the timelines in order and reconciles
    a seam between them, the run report attributes dead-lettered ids to the
    timeline whose id space they belong to, and a chunk's ``start`` is only
    meaningful against its own alignment."""

    timeline: Timeline
    alignment: AlignmentResult
    chunks: tuple[AudioChunk, ...]
    run_state: RunState


def _align_and_slice_timeline(
    config: RunConfig,
    timeline: Timeline,
    *,
    align_model: object | None,
    diarizer: Diarizer | None = None,
    from_plan: str | Path | None = None,
    load_plan: bool = True,
    on_quality_report: Callable[[AlignmentQualityReport], None] | None = None,
) -> tuple[AlignmentResult, tuple[AudioChunk, ...], Mapping[int, ShotPlanEntry] | None]:
    """Stages 1-2 for one timeline (issue #66).

    The whole point of the second-timeline design is that this is the *same*
    Stage 1 and Stage 2 the song gets: ``stable-ts``'s ``align()`` does not
    know or care that the audio is speech rather than singing (it is better
    at speech), and slicing is arithmetic over an ``AlignmentResult``. So a
    segment reuses both verbatim, with three deliberate exceptions:

    * ``alignment_overrides`` are **song-only**. An override names a
      ``segment_index`` in the song's own alignment (issue #42); applying the
      same indices to a prologue's alignment would pin arbitrary, unrelated
      segments to times someone measured against a different recording.
    * the **shot plan is the timeline's own**. The song's plan anchors
      ``chunk_id``s in the song's id space, and resolving them against a
      segment would attach the song's shot 3 to the prologue's shot 3 with
      nothing raising.
    * the **vocal stem is song-only** -- it is an isolated vocal cut from the
      master (issue #25), and there is no such thing for a dialogue take. So
      is the **diarization** that reads it (issue #101): a prologue has no
      stem to diarize and no singers to tell apart.

    ``from_plan`` is ``--prepare --from-plan``'s re-anchoring, applied to the
    timeline whose plan it is. ``load_plan=False`` is what
    :func:`prepare_timeline` passes: ``--prepare`` does not read the config's
    own shot plan at all -- not for its lengths and not for its lints --
    because there is usually no plan yet to read, and because a review whose
    job is to *surface* an unreadable plan must not crash on one. That
    asymmetry predates issue #66's refactor and is preserved by it rather
    than tidied away; see :func:`prepare_timeline` and ``--from-plan``'s help
    text.
    """
    lines = parse_lyrics(timeline.script, config.cast, config.default_lead_vocalist)
    # Issue #36: align() computes the quality report internally and only logs
    # it; this is the seam that hands the structured report to a caller (the
    # review page) that needs more than a log line. Issues #96/#92: it is
    # also the only thing that knows which segments may hold no voice at all,
    # so it is always captured here and handed to slicing, which maps those
    # segments onto the chunk ids a render actually emits.
    quality_reports: list[AlignmentQualityReport] = []

    def _capture(report: AlignmentQualityReport) -> None:
        quality_reports.append(report)
        if on_quality_report is not None:
            on_quality_report(report)

    alignment = align(
        timeline.audio,
        lines,
        model=align_model,
        model_size=config.alignment_model_size,
        strict_alignment=config.strict_alignment,
        overrides=config.alignment_overrides if timeline.is_song else (),
        on_quality_report=_capture,
        # #105 part 2: a transcript of the song's stem says nothing about a
        # prologue's audio, for the same reason the overrides do not apply.
        transcript_file=config.transcript_file if timeline.is_song else None,
    )
    if not timeline.is_song and config.alignment_overrides:
        logger.info(
            "Timeline %r: the run's %d alignment override(s) are NOT applied here -- they "
            "name segment indices in the song's own alignment (issue #42), which is a "
            "different recording with a different segmentation.",
            timeline.name,
            len(config.alignment_overrides),
        )

    if timeline.is_song and config.diarize and config.vocal_stem is not None:
        # Issue #101: an alternative FRONT-END to the manual [Name: Role]
        # tags, run here -- after align(), before slice_audio() -- because
        # AlignedSegment.characters is the field slicing derives
        # AudioChunk.characters from, so writing it here leaves every stage
        # after Stage 2a innocent of where the attribution came from. Reads
        # the isolated stem, never the master. Degrades rather than crashing:
        # a missing token or an unaccepted licence logs an ERROR naming the
        # remedy and leaves the alignment exactly as the tags left it.
        alignment = assign_characters(
            alignment,
            diarizer=diarizer if diarizer is not None else lazy_pyannote_diarizer(),
            audio_path=config.vocal_stem,
            speakers=config.diarization_speakers,
            default_lead_vocalist=config.default_lead_vocalist,
        ).alignment

    plan = (
        load_shot_plan(timeline.shot_plan, setting=config.setting, cast_names=config.cast)
        if timeline.shot_plan and load_plan
        else None
    )
    shot_lengths: tuple[ShotLength, ...] = shot_length_requests(plan)
    if from_plan is not None:
        shot_lengths = shot_length_requests(
            load_shot_plan(from_plan, setting=config.setting, cast_names=config.cast)
        )

    chunks = slice_audio(
        timeline.audio,
        alignment,
        config.hardware,
        timeline.chunks_dir,
        cover_instrumentals=config.instrumental_coverage,
        # Issue #100: opt-in, so every existing config slices byte-identically.
        boundary_overrun=config.boundary_overrun,
        # Opt-in, so every existing config slices byte-identically.
        phrase_aware_slicing=config.phrase_aware_slicing,
        shot_lengths=shot_lengths,
        instrumental_shot_seconds=config.instrumental_shot_seconds,
        instrumental_audio_gain_db=config.instrumental_audio_gain_db,
        # Issue #66: stamped on every chunk, and from there onto every
        # ChunkFingerprint, so --resume can never hand one timeline's mp4 to
        # another. None for the song, which is what every pre-#66 state file
        # already records.
        timeline=timeline.fingerprint_name,
        suspect_segment_indices=(
            suspect_segment_indices(quality_reports[0]) if quality_reports else ()
        ),
        # Issue #98: the "nothing longer has rendered here" warning gets its
        # number from what this timeline's own previous run actually
        # rendered, when there is one to read, instead of from a constant
        # that has been overtaken twice.
        measured_ceiling=measured_ceiling(timeline.run_state_file),
    )
    if not chunks:
        raise PipelineError(
            f"no chunks produced by slicing {timeline.audio} against {timeline.script} "
            f"(timeline {timeline.name!r}) -- nothing to render (no non-empty lines "
            "aligned to audio?)"
        )
    logger.info(
        "Stage 1-2 complete for timeline %r: %d chunk(s), %.3fs of track",
        timeline.name,
        len(chunks),
        alignment.track_duration,
    )
    _log_timeline_track_drift(chunks, alignment.track_duration, config)
    return alignment, tuple(chunks), plan


def _render_one_timeline(
    config: RunConfig,
    timeline: Timeline,
    chunks: tuple[AudioChunk, ...],
    plan: Mapping[int, ShotPlanEntry] | None,
    *,
    base_template: Workflow,
    i2v_template: Workflow | None,
    text_encoder: str | None,
    stager: ComfyUIAssetStager,
    execution_client: ComfyUIExecutionClient,
    session: Any,
    resume: bool,
    only_chunks: Sequence[int] | None,
    reseed_chunk_ids: Sequence[int] | None,
    reseed_generation: int,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None,
    sleeper: Sleeper,
    disk_usage: DiskUsage,
    seed_face_gate: Callable[[Path, Path], bool] | None,
    render_stack: RenderStack,
) -> RunState:
    """Stages 2b-4 for one timeline: prompts, staging, and the resilient
    render (issue #66).

    Lifted verbatim out of :func:`run_pipeline` rather than reimplemented --
    a prologue that renders through a *copy* of the render path is a prologue
    whose chunks stop matching the song's the first time one of them is
    edited, which is the whole class of defect ``ChunkFingerprint`` exists
    for.

    ``timeline.chunks_dir`` is per timeline, so chunk ids may (and do)
    collide across timelines without their files colliding, and
    ``timeline.run_state_file`` is per timeline for the same reason:
    ``RunState.results`` is keyed by chunk id, and merging two id spaces into
    one dict would silently overwrite.
    """
    if plan is not None:
        run_shot_plan_lints(plan, chunks, config)

    prompts = {
        chunk.chunk_id: expand_prompt(
            config,
            chunk,
            shot=resolve_shot(plan, chunk),
            subject_is_focus=(
                plan[chunk.chunk_id].subject_is_focus
                if plan is not None and chunk.chunk_id in plan
                else True
            ),
            camera=resolve_camera(plan, chunk),
            present=resolve_present(plan, chunk),
            subject=resolve_subject(plan, chunk),
            location=resolve_location(plan, chunk),
            conditions=resolve_conditions(plan, chunk),
            # Issue #97: how much of the frame this shot's focus member
            # should fill. The only field that says anything about delivered
            # face size -- `camera` is free text and #97 measured "close"
            # there spanning 0.0000-0.3561 of frame.
            framing=resolve_framing(plan, chunk),
            # Issue #66: a segment's audio is dialogue, so its prompt says
            # "speaking the line" rather than "singing the lyric". Taken from
            # the timeline, never inferred from the text -- a segment whose
            # script happens to be sung is still a segment.
            spoken=not timeline.is_song,
        )
        for chunk in chunks
    }

    reseed_generations: dict[int, int] = {}
    if reseed_chunk_ids:
        if reseed_generation < 1:
            raise PipelineError(
                f"--reseed-generation must be >= 1 (0 is the seed every chunk already "
                f"has without --reseed, not a re-roll of it), got {reseed_generation}"
            )
        available = [chunk.chunk_id for chunk in chunks]
        unknown = sorted(set(reseed_chunk_ids) - set(available))
        if unknown:
            logger.error(
                "--reseed names chunk id(s) %s that timeline %r does not have; it has %d "
                "chunk(s), %s..%s",
                unknown,
                timeline.name,
                len(available),
                available[0],
                available[-1],
            )
            raise PipelineError(
                f"--reseed names unknown chunk id(s) {unknown} on timeline "
                f"{timeline.name!r}; it has {len(available)} chunk(s), "
                f"{available[0]}..{available[-1]}"
            )
        reseed_generations = {chunk_id: reseed_generation for chunk_id in reseed_chunk_ids}
        logger.info(
            "Re-seeding chunk(s) %s of timeline %r at generation %d -- every other chunk "
            "keeps its existing seed and stays a cache hit under --resume",
            sorted(reseed_generations),
            timeline.name,
            reseed_generation,
        )

    assets = {
        chunk.chunk_id: stager.stage_chunk(prompts[chunk.chunk_id], chunk) for chunk in chunks
    }
    logger.info(
        "Stage 3 complete for timeline %r: %d chunk(s) staged to %s",
        timeline.name,
        len(assets),
        config.comfyui_url,
    )

    seeded_mutator = PerChunkSeedMutator(
        WorkflowGraphMutator(),
        base_seed=config.noise_seed,
        reseed_generations=reseed_generations,
    )

    provider = ContinuityWorkflowProvider(
        base_template=base_template,
        i2v_template=i2v_template,
        chunk_prompts=prompts,
        chunk_assets=assets,
        asset_stager=stager,
        # Per timeline, like the chunks themselves: a seed frame is named by
        # chunk id, and two timelines' chunk 3 would otherwise write to one
        # path -- handing one timeline's last frame to the other's chain.
        frames_dir=timeline.chunks_dir / "frames",
        continuity_enabled=config.i2v_continuity,
        mutator=seeded_mutator,
        subprocess_runner=ffmpeg_runner,
        # Issue #100: H3's `length` is what gets RENDERED, which is not a
        # chunk's own frame_count once an overrun is in play. Handing it the
        # kept count would render exactly what the trim was meant to avoid
        # having to do, and the stem (also cut to the rendered length) would
        # then be longer than the video -- issue #20's drift, reintroduced.
        chunk_frame_counts={c.chunk_id: c.rendered_frame_count for c in chunks},
        render_width=config.render_width,
        render_height=config.render_height,
        noise_seed=config.noise_seed,
        reanchor_interval=config.i2v_reanchor_interval,
        chainable_chunk_ids=chain_scope_ids(config.i2v_chain_scope, chunks),
        text_encoder=config.text_encoder,
        lora=config.lora,
        lora_strength=config.lora_strength,
        graph_hasher=graph_fingerprint,
        seed_face_gate=_resolve_seed_face_gate(config, seed_face_gate),
    )

    runner = ResilientRunner.from_config(
        # Issue #66: one run state file per timeline. RunState.results is keyed
        # by chunk id and the id spaces are separate, so one shared file would
        # have the prologue's chunk 3 overwrite the song's -- the same
        # collision the per-timeline chunks directory prevents on disk.
        dc_replace(config, run_state_file=timeline.run_state_file),
        execution_client,
        sleeper=sleeper,
        disk_usage=disk_usage,
        vram_probe=build_vram_probe(session, config.comfyui_url),
        vram_releaser=(
            build_vram_releaser(session, config.comfyui_url)
            if config.release_vram_between_chunks
            else None
        ),
    )

    fingerprints = {
        chunk.chunk_id: ChunkFingerprint.of(
            chunk,
            prompts[chunk.chunk_id],
            render_width=config.render_width,
            render_height=config.render_height,
            noise_seed=resolve_chunk_seed(
                config.noise_seed,
                chunk.chunk_id,
                reseed_generation=reseed_generations.get(chunk.chunk_id, 0),
            ),
            conditioning_source=(
                f"stem:{config.vocal_stem.name}"
                if config.vocal_stem and timeline.is_song
                else "mix"
            ),
            instrumental_audio_gain_db=config.instrumental_audio_gain_db,
            text_encoder=text_encoder,
            lora=config.lora,
            lora_strength=config.lora_strength if config.lora else None,
            # Issue #95: which ComfyUI/torch made the pixels. Reportable tier
            # -- a resumed run across an upgrade names the split and reuses
            # the chunks anyway, unless resume_require_same_stack says
            # otherwise.
            comfyui_version=render_stack.comfyui_version,
            torch_version=render_stack.torch_version,
        )
        for chunk in chunks
    }
    planned_chain = {
        chunk.chunk_id: planned_chain_source(
            chunk.chunk_id,
            continuity_enabled=config.i2v_continuity,
            reanchor_interval=config.i2v_reanchor_interval,
            chainable_chunk_ids=chain_scope_ids(config.i2v_chain_scope, chunks),
        )
        for chunk in chunks
    }
    fingerprints = {
        chunk_id: dc_replace(
            fp,
            chained_from=planned_chain[chunk_id],
            prompt_hash=ChunkFingerprint.hash_prompt(
                prompts[chunk_id].text_for(chained=planned_chain[chunk_id] is not None)
            ),
            template_hash=provider.planned_template_hash(
                chained=planned_chain[chunk_id] is not None
            ),
            fallback_template_hash=provider.planned_template_hash(chained=False),
        )
        for chunk_id, fp in fingerprints.items()
    }

    render_ids = _select_render_ids(chunks, only_chunks)
    slice_forced = () if resume else (only_chunks or ())
    force_chunk_ids = tuple(dict.fromkeys((*slice_forced, *(reseed_chunk_ids or ()))))

    # Issues #24, #98: the last of the three refusals, and the only one that
    # looks at the size of what is being submitted rather than at what else
    # holds the card. Run over the chunks this invocation would actually
    # render -- an --only-chunks slice of short chunks must not be refused
    # because some other chunk in the song is long, which is precisely how
    # the attended proof gets run. Resolution is taken from the config when
    # it sets one and from the template when it does not, because "unset"
    # means 1344x768 here, not "no resolution". No GPU work has happened at
    # this point -- staging is an upload.
    template_width, template_height = read_render_dimensions(base_template)
    check_render_envelope(
        [chunk for chunk in chunks if chunk.chunk_id in set(render_ids)],
        hardware_name=config.hardware.name,
        width=config.render_width if config.render_width is not None else template_width,
        height=config.render_height if config.render_height is not None else template_height,
        acknowledged=config.acknowledge_unproven_envelope,
    )

    return runner.render_run(
        render_ids,
        provider,
        timeline.chunks_dir,
        resume=resume or bool(only_chunks) or bool(reseed_chunk_ids),
        force_chunk_ids=force_chunk_ids,
        fingerprints=fingerprints,
        fingerprint_amender=lambda chunk_id, fp: _amend_from_render(provider, chunk_id, fp),
    )


def _resolve_flag_timeline(
    timelines: Sequence[Timeline], name: str | None, *, flag: str
) -> Timeline:
    """Which timeline ``--only-chunks``/``--reseed`` name ids in (issue #66).

    Defaults to the song, which is what those flags have always meant and
    what every config without segments still means. An unknown name is an
    error rather than a silent fall back to the song: rendering the wrong
    timeline's chunk 3 is exactly the confusion the separate id spaces exist
    to make impossible, and doing it because of a typo would be worse than
    doing it by accident."""
    if name is None:
        return next(t for t in timelines if t.is_song)
    matches = [t for t in timelines if t.name == name]
    if not matches:
        known = ", ".join(t.name for t in timelines)
        logger.error(
            "--timeline %r is not a timeline in this run; it has: %s", name, known
        )
        raise PipelineError(
            f"--timeline {name!r} is not a timeline in this run (used by {flag}); "
            f"this run has: {known}"
        )
    return matches[0]


def run_pipeline(
    config: RunConfig,
    *,
    resume: bool = False,
    align_model: object | None = None,
    diarizer: Diarizer | None = None,
    comfyui_session: Any = None,
    ws_factory: Callable[..., Any] | None = None,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None = None,
    sleeper: Sleeper = time.sleep,
    disk_usage: DiskUsage = shutil.disk_usage,
    clock: Callable[[], float] = time.monotonic,
    only_chunks: Sequence[int] | None = None,
    reseed_chunk_ids: Sequence[int] | None = None,
    reseed_generation: int = DEFAULT_RESEED_GENERATION,
    seed_face_gate: Callable[[Path, Path], bool] | None = None,
    flag_timeline: str | None = None,
) -> RunReport:
    """Run Stages 1-5 end-to-end against an already-loaded, validated config.

    ``comfyui_session`` is shared across staging, execution, and the custody
    seam -- one HTTP session for the whole run, not three, mirroring how a
    real ``requests.Session`` pools connections to the same host. Everything
    else mirrors the seam each stage module already exposes (``align``'s
    ``model``, ``ComfyUIExecutionClient``'s ``ws_factory``,
    ``ResilientRunner``'s ``sleeper``/``disk_usage``, the ffmpeg/ffprobe
    runner shared by continuity's frame extraction and Stage 5's mux).
    Raises :class:`PipelineError` if slicing produces zero chunks, and lets
    every stage's own typed exception (``ConfigError`` is the caller's
    problem, not this function's) propagate otherwise -- :func:`main` is
    where those turn into a log line and an exit code.

    ``reseed_chunk_ids`` is issue #38's ``--reseed``: force just these chunks
    to re-render under a seed that :func:`~music_video_maker.workflow_graph.
    resolve_chunk_seed` guarantees differs from ``reseed_generation``'s
    predecessors and from every other chunk's ordinary (generation-0) seed --
    the common "chunk 12 was the only bad one" case. Every chunk not named
    keeps generation 0, the exact seed it would have gotten without
    ``--reseed`` at all, so its fingerprint is unchanged and ``--resume``
    reuses it rather than re-rendering the whole song. Implies ``resume``
    (like ``only_chunks`` already does): a reseed with nothing to resume from
    just renders every chunk once, the named ones at their alternate
    generation.

    **Timelines (issue #66).** A config with ``[[segment]]`` tables renders
    more than one timeline -- a spoken prologue, an epilogue, or both -- each
    through the identical Stages 1-4 with its own audio, text, chunks
    directory, chunk id space and run state, and Stage 5 then concatenates
    them and reconciles the seam between them. A config with no segments
    renders exactly one timeline and takes the same Stage 5 path it always
    did, down to the same two ffmpeg calls.

    ``flag_timeline`` says which timeline ``only_chunks``/``reseed_chunk_ids``
    name ids in (default: the song). With ``only_chunks`` it also narrows the
    run to that one timeline -- a slice assembles nothing, so rendering the
    other timelines would be hours of GPU time for an artefact the run then
    deliberately does not produce.
    """
    start = clock()
    session = comfyui_session if comfyui_session is not None else requests.Session()
    custody = build_custody_manager(config, session=session)

    # Issue #55: record the resolved house style verbatim beside this run's
    # outputs, before any GPU time is spent. prompt_hash proves that a look
    # *changed*; it cannot say what the look *was* once the profile has moved
    # on to v3. Never allowed to abort a run -- a provenance sidecar failing
    # is a thing to log, not a reason to lose a render.
    if config.cinematography_profile is not None:
        try:
            # The effective value of a look field IS what is on the resolved
            # config -- that is what `load_config` composed and what the
            # prompts will carry. Which fields the run config *took back*,
            # though, is not recoverable here (lora_strength defaults to 1.0,
            # so "set" and "defaulted" look identical after load), which is
            # why load_config records it as it goes.
            effective = {
                name: getattr(config, name)
                for name in PROFILE_LOOK_FIELDS
                if getattr(config, name, None) is not None
            }
            write_profile_record(
                config.cinematography_profile,
                effective,
                config.cinematography_profile_overrides,
                Path(config.chunks_dir) / PROFILE_RECORD_FILENAME,
            )
        except Exception:  # noqa: BLE001 - provenance must never block a render.
            logger.exception(
                "Could not write the cinematography profile record -- continuing with the "
                "run. The look is still recoverable from %s, which prompt_hash pins.",
                config.cinematography_profile.path,
            )

    timelines = plan_timelines(config)
    flagged = _resolve_flag_timeline(
        timelines, flag_timeline, flag="--only-chunks/--reseed"
    )
    if only_chunks:
        # A slice renders named chunks and assembles nothing, so the other
        # timelines have no deliverable to contribute to -- rendering them
        # would be hours of exclusive GPU custody spent on an artefact this
        # run deliberately does not produce.
        rendered_timelines = [flagged]
        if len(timelines) > 1:
            logger.info(
                "Validation slice on timeline %r: the other %d timeline(s) are not "
                "rendered at all (a slice skips Stage 5, so they would contribute to "
                "nothing) -- %s",
                flagged.name,
                len(timelines) - 1,
                ", ".join(t.name for t in timelines if t is not flagged),
            )
    else:
        rendered_timelines = list(timelines)

    if len(timelines) > 1:
        logger.info(
            "This run has %d timelines (issue #66), in playback order: %s",
            len(timelines),
            ", ".join(f"{t.name} [{t.position}]" for t in timelines),
        )

    # Issue #43: a separate context manager from custody on purpose. Custody is
    # about the card; this is about the machine driving it staying awake long
    # enough to hear the card finish. Both are entered here, sleep prevention
    # first, so custody's free-VRAM pre-flight runs BEFORE Stage 1: every
    # timeline's alignment and slicing happen inside this block, under custody,
    # and are covered by sleep prevention along with everything after them.
    with prevent_host_sleep(), custody:
        base_template = load_workflow_template(config.workflow_template)
        i2v_template = (
            load_workflow_template(config.i2v_workflow_template)
            if config.i2v_continuity
            else None
        )
        scan_workflow_for_missing_optimizations(base_template, config.hardware)
        text_encoder = _resolve_text_encoder(config, base_template, i2v_template)

        # Issue #95: which build is about to make these pixels. One read per
        # run, over the session custody already uses -- a stack does not
        # change between chunks, and the only consumer is the fingerprint,
        # which a resumed run compares at the end rather than per chunk.
        # Best-effort by construction: an unreadable answer is recorded as
        # unknown and never as agreement.
        render_stack = read_render_stack(session, config.comfyui_url)
        if render_stack.known:
            logger.info(
                "Render stack for this run: comfyui %s / torch %s (issue #95)",
                render_stack.comfyui_version or "unknown",
                render_stack.torch_version or "unknown",
            )

        stager = ComfyUIAssetStager(base_url=config.comfyui_url, session=session)
        execution_client = ComfyUIExecutionClient(
            base_url=config.comfyui_url, session=session, ws_factory=ws_factory
        )

        renders: list[TimelineRender] = []
        for timeline in rendered_timelines:
            alignment, chunks, plan = _align_and_slice_timeline(
                config, timeline, align_model=align_model, diarizer=diarizer
            )
            if timeline.is_song and config.vocal_stem:
                # Issue #25: condition H3 on the isolated vocal stem, cut at
                # the very spans slicing just computed from the master.
                # Conditioning only -- Stage 5 still muxes the pristine master
                # over the video. Song-only: a dialogue take has no vocal stem
                # to isolate, and the field names one file.
                chunks = slice_stem_for_chunks(
                    config.vocal_stem,
                    chunks,
                    timeline.chunks_dir / "stem",
                    master_path=timeline.audio,
                ).chunks
            run_state = _render_one_timeline(
                config,
                timeline,
                chunks,
                plan,
                base_template=base_template,
                i2v_template=i2v_template,
                text_encoder=text_encoder,
                stager=stager,
                execution_client=execution_client,
                session=session,
                resume=resume,
                only_chunks=only_chunks if timeline is flagged else None,
                reseed_chunk_ids=reseed_chunk_ids if timeline is flagged else None,
                reseed_generation=reseed_generation,
                ffmpeg_runner=ffmpeg_runner,
                sleeper=sleeper,
                disk_usage=disk_usage,
                seed_face_gate=seed_face_gate,
                render_stack=render_stack,
            )
            renders.append(
                TimelineRender(
                    timeline=timeline,
                    alignment=alignment,
                    chunks=tuple(chunks),
                    run_state=run_state,
                )
            )

        output_video = _assemble_run(
            config,
            renders,
            only_chunks=only_chunks,
            ffmpeg_runner=ffmpeg_runner,
        )

    song_render = next(
        (r for r in renders if r.timeline.is_song), renders[0] if renders else None
    )
    return RunReport(
        run_state=song_render.run_state if song_render is not None else RunState(run_id=""),
        total_chunks=sum(len(r.chunks) for r in renders),
        wall_seconds=clock() - start,
        output_video=output_video,
        timeline_states=tuple((r.timeline.name, r.run_state) for r in renders),
    )


def _assemble_run(
    config: RunConfig,
    renders: Sequence[TimelineRender],
    *,
    only_chunks: Sequence[int] | None,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None,
) -> Path | None:
    """Stage 5 for however many timelines this run produced (issue #66).

    One timeline takes the path it always took -- :func:`assemble_final_video`
    with the same two ffmpeg calls, the same ``-shortest``, the same opt-in
    duration check -- so adding this branch changed no existing run's output.

    More than one takes :func:`~music_video_maker.assembly.assemble_timelines`,
    which concats each timeline separately, *measures* what it produced with
    ffprobe, pads each timeline's audio up to its own video, asserts that
    padding on a second probe, and only then joins them. The seam is where
    the ``-shortest`` safety net stops existing: it trims the end of a file,
    and a seam is in the middle of one.
    """
    if only_chunks:
        # Deliberately no Stage 5. Concatenating a subset would write a file
        # that looks like the finished song and is not -- the same silently
        # desynced artifact the chunk timeline and the fingerprints both
        # exist to prevent. A slice's deliverable is chunks to watch.
        render = renders[0]
        logger.info(
            "Rendered a %d-chunk slice of timeline %r -- skipping Stage 5 assembly. The "
            "clips are in %s; a partial concat would claim to be the whole song.",
            len(render.run_state.results),
            render.timeline.name,
            render.timeline.chunks_dir,
        )
        return None

    dead = [
        (r.timeline.name, r.run_state.dead_lettered)
        for r in renders
        if r.run_state.dead_lettered
    ]
    if dead:
        logger.error(
            "Skipping Stage 5 assembly: dead-lettered chunk(s) %s",
            dead,
        )
        return None

    if len(renders) == 1:
        render = renders[0]
        assembly_result = assemble_final_video(
            render.chunks,
            render.run_state,
            # Issue #22: None means no audio stream at all, for a concert
            # backdrop where the band is the audio. See the field's own
            # docstring for which invariant that suspends and which it
            # leaves alone.
            None if config.silent_output else config.master_audio,
            config.final_video_dir,
            runner=ffmpeg_runner,
            # Issue #22: arms the measured-duration check that replaces
            # -shortest, but ONLY on the silent path -- the music-video
            # path gets no new probe and no new subprocess, byte-for-byte
            # unchanged (expected_duration stays None). The master track's
            # own duration stands in for the authoritative show duration
            # here; design-concert-mode.md question 5 is still open, so
            # this is a stand-in, not the final answer.
            expected_duration=(
                render.alignment.track_duration if config.silent_output else None
            ),
            duration_tolerance_seconds=config.duration_tolerance_seconds,
        )
        return assembly_result.output_video

    multi = assemble_timelines(
        [
            TimelineAssembly(
                name=r.timeline.name,
                chunks=r.chunks,
                results=r.run_state,
                audio=None if config.silent_output else r.timeline.audio,
                fingerprint_name=r.timeline.fingerprint_name,
            )
            for r in renders
        ],
        config.final_video_dir,
        runner=ffmpeg_runner,
        duration_tolerance_seconds=config.duration_tolerance_seconds,
    )
    return multi.output_video


@dataclass(frozen=True)
class PreparedTimeline:
    """Stages 1-2's output, plus the quality report Stage 1 already computes
    internally (issue #36).

    ``--prepare`` (issue #52) only ever needed ``chunks``, to write the
    skeleton; ``alignment`` and ``quality_report`` are recorded here so a
    second caller -- the review page -- can use the exact same Stage 1-2 run
    rather than a second one, which is also what keeps the two honest about
    describing the same timeline."""

    alignment: AlignmentResult
    chunks: tuple[Any, ...]
    quality_report: AlignmentQualityReport


def prepare_timeline(
    config: RunConfig,
    *,
    align_model: object | None = None,
    diarizer: Diarizer | None = None,
    from_plan: str | Path | None = None,
) -> PreparedTimeline:
    """Run Stages 1-2 only for the **song** -- alignment + slicing, no GPU, no
    ComfyUI, no custody -- and return the chunk timeline, the raw alignment,
    and the alignment-quality report (issue #36).

    Shared by :func:`prepare_shot_plan` (issue #52, which only writes the
    skeleton) and the review page (issue #36's next slice, which additionally
    needs the quality report and the chunks themselves) -- both want exactly
    the same Stage 1-2 run, and a second implementation would risk describing
    a timeline the real one does not. Since issue #66 that sharing goes one
    level deeper: this runs ``_align_and_slice_timeline``, the same function
    ``run_pipeline`` renders through, rather than a parallel copy of it.

    ``from_plan`` re-anchors the timeline against the run a render with
    *that plan* will actually produce, by loading it **for its
    ``length_seconds`` only** and re-slicing with them (issue #54 design
    section 5). Without it this slices with no ``shot_lengths`` while
    :func:`run_pipeline` passes ``shot_length_requests(plan)``, so the moment
    a plan expresses one editorial length the timeline describes chunks no
    render will ever produce and every chunk after that length raises
    ``ShotPlanDriftError`` the moment something resolves against it. Nothing
    else in the source plan is read -- not the shot text, not the camera
    direction.

    A plan that cannot be read is a hard failure, never a silent fall back to
    a length-free timeline: that fallback would look exactly like success and
    drift hours later, on the GPU.

    A run with ``[[segment]]`` tables has more than one timeline;
    :func:`prepare_timelines` is the one that returns all of them. This
    function stays the song's, because that is what the review page means and
    what it has always returned.
    """
    song = next(t for t in plan_timelines(config) if t.is_song)
    return _prepare_one_timeline(
        config, song, align_model=align_model, diarizer=diarizer, from_plan=from_plan
    )


def _prepare_one_timeline(
    config: RunConfig,
    timeline: Timeline,
    *,
    align_model: object | None = None,
    diarizer: Diarizer | None = None,
    from_plan: str | Path | None = None,
) -> PreparedTimeline:
    quality_reports: list[AlignmentQualityReport] = []
    alignment, chunks, _plan = _align_and_slice_timeline(
        config,
        timeline,
        align_model=align_model,
        diarizer=diarizer,
        from_plan=from_plan,
        # --prepare deliberately does not read the config's own shot plan at
        # all; only --from-plan names one. Preserved from before issue #66's
        # refactor -- see _align_and_slice_timeline for why.
        load_plan=False,
        on_quality_report=quality_reports.append,
    )
    if from_plan is not None:
        logger.info(
            "Re-anchoring timeline %r against %s (issue #52 follow-up)",
            timeline.name,
            from_plan,
        )
    # `align()` always calls the callback exactly once before returning (see
    # its own docstring) -- this is defensive, not a real fallback path.
    quality_report = (
        quality_reports[0] if quality_reports else evaluate_alignment_quality(alignment)
    )
    return PreparedTimeline(
        alignment=alignment, chunks=tuple(chunks), quality_report=quality_report
    )


def prepare_timelines(
    config: RunConfig,
    *,
    align_model: object | None = None,
    diarizer: Diarizer | None = None,
    from_plan: str | Path | None = None,
) -> tuple[tuple[Timeline, PreparedTimeline], ...]:
    """Stages 1-2 for **every** timeline this run would render, in playback
    order, plus the seam report (issue #66).

    ``--prepare`` is this project's 50-second, no-GPU check, and with a
    prologue there are two things to check rather than one: each timeline's
    own chunk timeline against its own track (the issue #22 drift report,
    which ``_align_and_slice_timeline`` already emits per timeline), and the
    **seam** -- where each timeline starts in the finished video and how much
    silence its audio needs to match its own picture.

    The seam numbers here are *predicted* from Stage 2's chunk timeline;
    assembly re-derives them from ffprobe readings of the rendered files and
    raises if they disagree. Both are worth having and they are not the same
    claim, which is why the log line says which one it is.

    ``from_plan`` applies to the song, matching ``--prepare --from-plan``'s
    existing meaning: a segment's lengths come from its own ``shot_plan``.
    """
    timelines = plan_timelines(config)
    prepared: list[tuple[Timeline, PreparedTimeline]] = []
    for timeline in timelines:
        prepared.append(
            (
                timeline,
                _prepare_one_timeline(
                    config,
                    timeline,
                    align_model=align_model,
                    diarizer=diarizer,
                    from_plan=from_plan if timeline.is_song else None,
                ),
            )
        )

    if len(prepared) > 1:
        _report_predicted_seam(prepared)
    return tuple(prepared)


def _report_predicted_seam(prepared: Sequence[tuple[Timeline, PreparedTimeline]]) -> None:
    """Log where each timeline will start and what its seam costs, from Stage
    2's own numbers (issue #66).

    Reports; never refuses. The same arithmetic *does* refuse inside
    assembly, where it is measured against real files -- here it is a preview
    of a decision nobody has spent GPU time on yet, and a preview that raises
    is a preview nobody runs.
    """
    timelines = tuple(t for t, _ in prepared)
    chunk_timeline_seconds = {
        t.name: (p.chunks[-1].end if p.chunks else 0.0) for t, p in prepared
    }
    track_seconds = {t.name: p.alignment.track_duration for t, p in prepared}
    try:
        placements = place_timelines(
            predicted_measurements(timelines, chunk_timeline_seconds, track_seconds)
        )
    except SeamOverrunError:
        logger.exception(
            "Predicted seam is unreconcilable -- reporting it rather than refusing, "
            "because --prepare is a report. Assembly WILL refuse this run."
        )
        return
    log_placements(placements, measured=False)


def prepare_shot_plan(
    config: RunConfig,
    output_path: str | Path,
    *,
    source: str,
    generated_at: str,
    align_model: object | None = None,
    force: bool = False,
    from_plan: str | Path | None = None,
    report_path: str | Path | None = None,
) -> Path:
    """Issue #52: run Stages 1-2 only and write a ``shot_plan.toml`` skeleton.

    Deliberately does not call :func:`~music_video_maker.custody.build_custody_manager`
    or :func:`~music_video_maker.custody.prevent_host_sleep` -- alignment and
    slicing take ~6 s of CPU and touch neither the GPU nor ComfyUI, so there
    is no card to take custody of and no long run for the host to stay awake
    through. Claiming the GPU every time someone drafts a plan would be
    custody theatre, not custody.

    ``source``/``generated_at`` are threaded straight through to
    :func:`~music_video_maker.shot_plan.write_shot_plan_skeleton` for the
    skeleton's header comment -- see that function for why they are
    caller-supplied rather than read from the filesystem or the clock here.

    Stages 1-2 themselves live in :func:`prepare_timeline`, shared with the
    review page (issue #36) -- this function's own job is just the skeleton.

    ``report_path`` (issue #36) additionally writes a JSON *report* of what
    this prepare found -- the #35 alignment summary, every Stage-2 warning,
    the #22 timeline drift, and any shot-plan drift -- so the read-only
    monitor can show it without running Stage 1-2 itself (alignment is a
    step an operator invokes; a page opening must not be). It is ``None`` by
    default, so a library caller's behaviour is unchanged and only
    ``--prepare`` writes one.

    Order: Stage 1-2, then the skeleton, then the report. A prepare that
    refuses (``strict_alignment`` tripping on a CRITICAL finding, or a
    skeleton that would clobber an authored plan without ``--force``) writes
    no report at all, and that is deliberate -- a report is a description of
    a prepare that *happened*, and one left behind by a refused prepare is
    the ``face_presence_v12.csv`` failure over again: an artefact describing
    something other than what its reader assumes.

    **A run with ``[[segment]]`` tables gets a skeleton per timeline**
    (issue #66). ``--prepare`` emitting one for a prologue is not a
    convenience: the whole point of #52 is that a plan's anchors are never
    transcribed by hand, and a segment's anchors come from its own alignment
    exactly like the song's. Segment skeletons are written beside the song's
    as ``<stem>__<segment><suffix>`` -- one path in, one path per timeline
    out, derived rather than asked for, so nobody has to remember to pass a
    second ``--shot-plan-out``. The song's path is returned, unchanged, which
    is what every existing caller expects.

    Also reports the **seam** -- where each timeline starts in the finished
    video, and how much silence its audio needs to match its own picture.
    That is the number a prologue can get wrong in a way no per-timeline
    check can see, and it costs no GPU to learn here.
    """
    notices: list[logging.LogRecord] = []
    if report_path is None:
        prepared = prepare_timelines(config, align_model=align_model, from_plan=from_plan)
    else:
        # Capture what Stage 1-2 warns about while it warns -- see
        # prepare_report.collect_stage_notices. The console is unaffected.
        with collect_stage_notices() as notices:
            prepared = prepare_timelines(
                config, align_model=align_model, from_plan=from_plan
            )

    song_path = Path(output_path)

    def _destination(timeline: Timeline) -> Path:
        if timeline.is_song:
            return song_path
        return song_path.with_name(f"{song_path.stem}__{timeline.name}{song_path.suffix}")

    # Every destination is checked before any of them is written. Per-file
    # refusal alone would write the prologue's skeleton and then abort on the
    # song's, leaving one timeline's anchors from this run beside another's
    # from an older one -- and two skeletons that disagree about the same run
    # is exactly the drift anchors exist to prevent.
    if not force:
        clashes = [
            path for path in (_destination(timeline) for timeline, _ in prepared)
            if path.exists()
        ]
        if clashes:
            names = ", ".join(str(path) for path in clashes)
            logger.error(
                "Refusing to write any shot-plan skeleton: %s already exist(s). An authored "
                "shot plan is real work; pass --force to overwrite. Nothing was written.",
                names,
            )
            raise ShotPlanError(
                f"{names} already exist(s) -- an authored shot plan is real work; pass "
                "--force to overwrite. Nothing was written, so no run can leave one "
                "timeline's anchors beside another run's."
            )

    written: list[Path] = []
    song_result = song_path
    for timeline, timeline_data in prepared:
        destination = _destination(timeline)
        written_path = write_shot_plan_skeleton(
            timeline_data.chunks,
            destination,
            source=source,
            generated_at=generated_at,
            force=force,
        )
        written.append(written_path)
        if timeline.is_song:
            song_result = written_path

    if len(written) > 1:
        logger.info(
            "Wrote %d shot-plan skeleton(s), one per timeline (issue #66): %s. Each is "
            "authored against ITS OWN chunk ids -- the id spaces are separate, so the "
            "song's shot 3 and the prologue's shot 3 are different shots.",
            len(written),
            ", ".join(str(p) for p in written),
        )
    if report_path is not None:
        # Issue #36: the report describes the song's timeline -- the one the
        # monitor polls. Written last, after every skeleton landed, so a
        # refused prepare leaves no report behind.
        song_prepared = next(data for timeline, data in prepared if timeline.is_song)
        _write_prepare_report(
            config,
            report_path,
            config_path=source,
            generated_at=generated_at,
            timeline=song_prepared,
            notices=notices,
            from_plan=from_plan,
        )
    return song_result


def _write_prepare_report(
    config: RunConfig,
    report_path: str | Path,
    *,
    config_path: str,
    generated_at: str,
    timeline: PreparedTimeline,
    notices: Sequence[logging.LogRecord],
    from_plan: str | Path | None,
) -> None:
    """Assemble and write the issue #36 pre-render report.

    Never raises: the report is a description of work that already
    succeeded, so a full disk or an unwritable path must not turn a good
    prepare (whose skeleton is already on disk) into a failed one. It is
    logged loudly instead, because a report nobody can write is still worth
    knowing about."""
    plan_path = from_plan if from_plan is not None else config.shot_plan
    plan_errors = (
        plan_resolution_errors(
            plan_path,
            timeline.chunks,
            setting=config.setting,
            cast_names=tuple(config.cast),
        )
        if plan_path is not None
        else ()
    )
    drift = timeline_track_drift_seconds(timeline.chunks, timeline.alignment.track_duration)
    report = build_prepare_report(
        generated_at=generated_at,
        config_path=config_path,
        inputs=[
            InputStamp.of("master_audio", config.master_audio),
            InputStamp.of("lyrics_file", config.lyrics_file),
            InputStamp.of("shot_plan", config.shot_plan),
            InputStamp.of("from_plan", from_plan),
            InputStamp.of("vocal_stem", config.vocal_stem),
        ],
        alignment_model_size=config.alignment_model_size,
        strict_alignment=config.strict_alignment,
        chunks=timeline.chunks,
        quality_report=timeline.quality_report,
        track_duration_seconds=timeline.alignment.track_duration,
        timeline_drift_seconds=drift,
        fps=config.hardware.frame_grid.fps,
        duration_tolerance_seconds=config.duration_tolerance_seconds,
        notice_records=notices,
        plan_errors=plan_errors,
        plan_checked=plan_path,
        plan_lengths_applied=from_plan is not None,
    )
    try:
        write_prepare_report(report, report_path)
    except OSError:
        logger.exception(
            "Stages 1-2 completed and the skeleton was written, but the pre-render report "
            "could not be written to %s -- mvm-webui will report this run as not prepared",
            report_path,
        )


def _log_final_report(report: RunReport) -> None:
    dead = report.dead_lettered
    if len(report.timeline_states) > 1:
        logger.info(
            "Timelines rendered (issue #66): %s%s",
            ", ".join(
                f"{name} ({len(state.results)} chunk(s))"
                for name, state in report.timeline_states
            ),
            (
                f"; dead-lettered by timeline: {report.dead_lettered_by_timeline}"
                if dead
                else ""
            ),
        )
    logger.info(
        "Run finished in %.1fs: %d/%d chunk(s) rendered, %d cached, %d dead-lettered%s -- "
        "output: %s",
        report.wall_seconds,
        report.rendered,
        report.total_chunks,
        report.cached,
        len(dead),
        f" ({dead})" if dead else "",
        report.output_video if report.output_video is not None else "(not assembled)",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper())

    # One-way convention: --ignore-prompt-changes can only ever turn the
    # reuse-on-prompt-change behaviour *on*, so its absence leaves the config's
    # own setting alone.
    overrides: dict[str, object] = {}
    if args.ignore_prompt_changes:
        overrides["resume_ignore_prompt_changes"] = True
    if args.strict_alignment:
        overrides["strict_alignment"] = True

    if args.prepare and args.review:
        logger.error("--prepare and --review are two different entry points; pass only one")
        return EXIT_ERROR

    try:
        config = load_config(Path(args.config), **overrides)
    except ConfigError:
        logger.exception("Failed to load run config from %s", args.config)
        return EXIT_ERROR

    # Issue #56: #51's likeness question ("whose consent does this run
    # depend on?") made askable at run start rather than left to be
    # remembered. Only logged when the answer is non-empty -- a config with
    # an entirely synthetic cast has nothing to disclose here.
    real_likenesses = config.real_likenesses()
    if real_likenesses:
        logger.info(
            "this run conditions on real likenesses: %s", ", ".join(real_likenesses)
        )

    if args.prepare:
        output_path = args.shot_plan_out or (Path(args.config).resolve().parent / "shot_plan.toml")
        try:
            prepare_shot_plan(
                config,
                output_path,
                source=args.config,
                generated_at=date.today().isoformat(),
                force=args.force,
                from_plan=args.from_plan,
                # Issue #36: also write down what this prepare found, so
                # mvm-webui can show it without re-running Stage 1-2.
                report_path=config.prepare_report_file,
            )
        except (PipelineError, ShotPlanError):
            logger.exception("Failed to prepare shot plan")
            return EXIT_ERROR
        return EXIT_SUCCESS

    if args.review:
        # Local import: review.py imports this module (to reuse
        # prepare_timeline/run_shot_plan_lints, issue #36's whole point), so
        # importing it back at module scope here would be a cycle. Deferred
        # to this branch, it never runs at import time and the cycle never
        # forms.
        from music_video_maker.review import build_review, render_review_html, render_review_json

        try:
            data = build_review(config, from_plan=args.from_plan)
        except (PipelineError, ShotPlanError):
            logger.exception("Failed to build review")
            return EXIT_ERROR
        stem = args.review.with_suffix("")
        html_path = stem.with_suffix(".html")
        json_path = stem.with_suffix(".json")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_review_html(data), encoding="utf-8")
        json_path.write_text(render_review_json(data), encoding="utf-8")
        logger.info("Wrote review to %s and %s", html_path, json_path)
        return EXIT_SUCCESS

    try:
        report = run_pipeline(
            config,
            resume=args.resume,
            only_chunks=args.only_chunks,
            reseed_chunk_ids=args.reseed,
            reseed_generation=args.reseed_generation,
            flag_timeline=args.timeline,
        )
    except Exception:
        logger.exception("Pipeline run failed")
        return EXIT_ERROR

    _log_final_report(report)
    return EXIT_PARTIAL_FAILURE if report.dead_lettered else EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
