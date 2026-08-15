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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import date
from pathlib import Path
from typing import Any

import requests

from music_video_maker.alignment import align
from music_video_maker.assembly import assemble_final_video
from music_video_maker.config import ConfigError, RunConfig, load_config
from music_video_maker.continuity import ContinuityWorkflowProvider, planned_chain_source
from music_video_maker.contracts import ChunkFingerprint, ChunkStatus, RunState, Workflow
from music_video_maker.custody import (
    build_custody_manager,
    build_vram_probe,
    prevent_host_sleep,
)
from music_video_maker.execution import ComfyUIExecutionClient
from music_video_maker.faces import build_seed_face_gate
from music_video_maker.hardware import scan_workflow_for_missing_optimizations
from music_video_maker.logging_setup import configure_logging
from music_video_maker.lyrics import parse_lyrics
from music_video_maker.prompting import expand_prompt
from music_video_maker.resilience import DiskUsage, ResilientRunner, Sleeper
from music_video_maker.shot_plan import (
    ShotLength,
    ShotPlanError,
    lint_camera_face_away_on_voiced_chunks,
    lint_shots_against_lyrics,
    load_shot_plan,
    resolve_camera,
    resolve_present,
    resolve_shot,
    shot_length_requests,
    write_shot_plan_skeleton,
)
from music_video_maker.slicing import slice_audio
from music_video_maker.staging import ComfyUIAssetStager
from music_video_maker.stems import slice_stem_for_chunks
from music_video_maker.workflow_graph import (
    graph_fingerprint,
    load_workflow_template,
    read_text_encoder,
)

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_PARTIAL_FAILURE = 2
"""Some chunk(s) dead-lettered; the run finished but is incomplete."""


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
    config: RunConfig, injected: Callable[[Path], bool] | None
) -> Callable[[Path], bool] | None:
    """The seed-face predicate this run should use (issue #47), or ``None`` to
    chain from whatever the previous shot ended on.

    Injectable like every other I/O seam in this module, so the offline test
    suite exercises the *decision* without OpenCV, a model file, or a real face
    anywhere -- the detector itself is unit-tested separately."""
    if injected is not None:
        return injected
    if not (config.i2v_continuity and config.i2v_require_seed_face):
        return None
    return build_seed_face_gate(min_fraction=config.i2v_min_seed_face_fraction)


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
    total_chunks: int
    wall_seconds: float
    output_video: Path | None

    @property
    def rendered(self) -> int:
        return sum(1 for r in self.run_state.results.values() if r.status is ChunkStatus.RENDERED)

    @property
    def cached(self) -> int:
        return sum(1 for r in self.run_state.results.values() if r.status is ChunkStatus.CACHED)

    @property
    def dead_lettered(self) -> tuple[int, ...]:
        return self.run_state.dead_lettered


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
            "before committing hours of exclusive GPU custody to a full render."
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
            "With --prepare, read an existing shot plan for its length_seconds only and "
            "re-anchor the skeleton against the timeline a render with that plan will "
            "actually produce. Without this, a plan that sets any editorial shot length "
            "gets a skeleton describing chunks no render will ever emit, and drifts."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Root log level (default: INFO)."
    )
    return parser


def _parse_chunk_ids(raw: str) -> tuple[int, ...]:
    """``"32,33, 35"`` -> ``(32, 33, 35)``, rejecting anything else loudly.

    Deliberately not a range syntax: a validation slice is usually a
    hand-picked set of the chunks that exercise the change (a re-anchor plus
    the chunks chained off it, say), not a contiguous run."""
    try:
        ids = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--only-chunks expects comma-separated integers, got {raw!r}"
        ) from None
    if not ids:
        raise argparse.ArgumentTypeError("--only-chunks was given no chunk ids")
    return ids


def run_pipeline(
    config: RunConfig,
    *,
    resume: bool = False,
    align_model: object | None = None,
    comfyui_session: Any = None,
    ws_factory: Callable[..., Any] | None = None,
    ffmpeg_runner: Callable[[Sequence[str]], Any] | None = None,
    sleeper: Sleeper = time.sleep,
    disk_usage: DiskUsage = shutil.disk_usage,
    clock: Callable[[], float] = time.monotonic,
    only_chunks: Sequence[int] | None = None,
    seed_face_gate: Callable[[Path], bool] | None = None,
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
    """
    start = clock()
    session = comfyui_session if comfyui_session is not None else requests.Session()
    custody = build_custody_manager(config, session=session)

    # Issue #43: outside the GPU custody block on purpose. Custody is about the
    # card; this is about the machine driving it staying awake long enough to
    # hear the card finish -- and it must cover Stage 1 too, since alignment
    # runs before custody is ever taken.
    with prevent_host_sleep(), custody:
        lines = parse_lyrics(config.lyrics_file, config.cast, config.default_lead_vocalist)
        alignment = align(
            config.master_audio,
            lines,
            model=align_model,
            strict_alignment=config.strict_alignment,
            # Issue #42: authored corrections for provably mis-placed
            # segments; applied inside align(), before quality evaluation.
            overrides=config.alignment_overrides,
        )
        # setting + cast names feed the shot plan's geography lint (issue #32):
        # a shot naming Central Park while the run is set in London is an
        # authoring slip worth catching here, before any GPU time is spent on it.
        # Loaded BEFORE slicing (issue #27): the plan's length_seconds entries
        # are requests slicing must honor while it derives the chunk timeline;
        # loading it after the chunks exist made the whole mechanism inert.
        plan = (
            load_shot_plan(config.shot_plan, setting=config.setting, cast_names=config.cast)
            if config.shot_plan
            else None
        )
        chunks = slice_audio(
            config.master_audio,
            alignment,
            config.hardware,
            config.chunks_dir,
            cover_instrumentals=config.instrumental_coverage,
            shot_lengths=shot_length_requests(plan),
            instrumental_shot_seconds=config.instrumental_shot_seconds,
        )
        if not chunks:
            raise PipelineError(
                f"no chunks produced by slicing {config.master_audio} against "
                f"{config.lyrics_file} -- nothing to render (no non-empty lyric lines "
                "aligned to audio?)"
            )
        logger.info("Stage 1-2 complete: %d chunk(s) to render", len(chunks))

        if config.vocal_stem:
            # Issue #25: condition H3 on the isolated vocal stem, cut at the
            # very spans slicing just computed from the master. Conditioning
            # only -- Stage 5 still muxes the pristine master over the video.
            chunks = slice_stem_for_chunks(
                config.vocal_stem,
                chunks,
                config.chunks_dir / "stem",
                master_path=config.master_audio,
            ).chunks

        if plan is not None:
            # Issue #37: both strings are in hand here -- a lyric naming an
            # object the plan stages only elsewhere is mechanically visible,
            # for free, before any GPU time is spent on it.
            lint_shots_against_lyrics(plan, chunks)
            # Issue #58: a camera direction that turns her away from the
            # lens on a voiced chunk costs that chunk's lip-sync.
            lint_camera_face_away_on_voiced_chunks(plan, chunks)

        prompts = {
            chunk.chunk_id: expand_prompt(
                config,
                chunk,
                shot=resolve_shot(plan, chunk),
                # Issue #26: a consequence beat can hand the shot's subject to
                # the action, so the burning printer stops competing with an
                # assertion that the performer is the focus.
                subject_is_focus=(
                    plan[chunk.chunk_id].subject_is_focus
                    if plan is not None and chunk.chunk_id in plan
                    else True
                ),
                # Issue #53: this chunk's own camera framing/movement,
                # independent of whether `shot` itself is filled in.
                camera=resolve_camera(plan, chunk),
                # Issue #59: who is on screen, as opposed to who is singing.
                # Without this a shot line's "him" reaches H3 with no role,
                # appearance or reference photo behind it, and the model
                # invents a different person every chunk.
                present=resolve_present(plan, chunk),
            )
            for chunk in chunks
        }

        base_template = load_workflow_template(config.workflow_template)
        i2v_template = (
            load_workflow_template(config.i2v_workflow_template)
            if config.i2v_continuity
            else None
        )
        scan_workflow_for_missing_optimizations(base_template, config.hardware)
        text_encoder = _resolve_text_encoder(config, base_template, i2v_template)

        stager = ComfyUIAssetStager(base_url=config.comfyui_url, session=session)
        assets = {
            chunk.chunk_id: stager.stage_chunk(prompts[chunk.chunk_id], chunk) for chunk in chunks
        }
        logger.info("Stage 3 complete: %d chunk(s) staged to %s", len(assets), config.comfyui_url)

        provider = ContinuityWorkflowProvider(
            base_template=base_template,
            i2v_template=i2v_template,
            chunk_prompts=prompts,
            chunk_assets=assets,
            asset_stager=stager,
            frames_dir=config.chunks_dir / "frames",
            continuity_enabled=config.i2v_continuity,
            subprocess_runner=ffmpeg_runner,
            chunk_frame_counts={c.chunk_id: c.frame_count for c in chunks},
            render_width=config.render_width,
            render_height=config.render_height,
            noise_seed=config.noise_seed,
            reanchor_interval=config.i2v_reanchor_interval,
            chainable_chunk_ids=chain_scope_ids(config.i2v_chain_scope, chunks),
            text_encoder=config.text_encoder,
            lora=config.lora,
            lora_strength=config.lora_strength,
            # Issue #45: hash the graph on its way out, per chunk, because a
            # run holds two templates and a chunk renders through exactly one.
            # Passed in rather than imported inside the provider so a test can
            # substitute it, and so the provider owes nothing to this module.
            graph_hasher=graph_fingerprint,
            # Issue #47: a seed frame with no face in it carries no identity at
            # all on the chained path, which has no ref_images. Built here so
            # the provider never imports OpenCV and a test can pass a lambda.
            seed_face_gate=_resolve_seed_face_gate(config, seed_face_gate),
        )

        execution_client = ComfyUIExecutionClient(
            base_url=config.comfyui_url, session=session, ws_factory=ws_factory
        )
        # Issue #23: the custody pre-flight checks free VRAM once, before the
        # run starts, and cannot see another process claiming the card two
        # hours in -- which is exactly how the 2026-08-07 wedge happened. The
        # runner re-reads the same /system_stats path before every chunk it is
        # about to submit, over the session custody already uses.
        runner = ResilientRunner.from_config(
            config,
            execution_client,
            sleeper=sleeper,
            disk_usage=disk_usage,
            vram_probe=build_vram_probe(session, config.comfyui_url),
        )
        # Issue #34: what each chunk is being rendered *for*, composed from what
        # Stages 1-2 just produced. A resumed run compares this against what
        # each cached chunk was actually rendered for, so an edited lyrics file
        # (or shot plan, or resolution) re-renders the affected chunks instead
        # of assembling a silently desynced video from the old timeline.
        fingerprints = {
            chunk.chunk_id: ChunkFingerprint.of(
                chunk,
                prompts[chunk.chunk_id],
                render_width=config.render_width,
                render_height=config.render_height,
                # Issue #38: the same config value the provider injects into
                # RandomNoise -- two paths, one truth, pinned by test.
                noise_seed=config.noise_seed,
                # Issue #25: which audio drove the mouth. Never escapable on
                # resume -- it is the stem A/B's experiment variable.
                conditioning_source=(
                    f"stem:{config.vocal_stem.name}" if config.vocal_stem else "mix"
                ),
                # Issue #39: which encoder read the sentence -- the pinned one
                # if this run pins one, otherwise whatever the template names.
                # Recorded either way, because the encoder is the one
                # pixel-deciding input that a canvas edit can change with
                # nothing else in the fingerprint moving.
                text_encoder=text_encoder,
                # Issue #62: which adapter, at what strength. Conditioning tier --
                # template_hash cannot see it, because the node is spliced in
                # during mutation rather than authored into the template.
                lora=config.lora,
                lora_strength=config.lora_strength if config.lora else None,
            )
            for chunk in chunks
        }
        # Issue #28: plan the chain up front (what config alone predicts)...
        #
        # ...and with it, issue #45's graph and #46's prompt variant, because
        # all three answer the same question -- which of the two templates is
        # this chunk expected to take. ``--resume`` compares a cached chunk
        # against these *predictions*, so they have to be computable before
        # anything renders; the amender then corrects whichever ones the run
        # actually decided differently.
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
                # What this chunk would carry if the render declines to chain
                # -- a dead-lettered predecessor, an unextractable frame, or
                # #47's face gate. Lets the comparison tell a degradation from
                # an edited template instead of re-rendering it every resume.
                fallback_template_hash=provider.planned_template_hash(chained=False),
            )
            for chunk_id, fp in fingerprints.items()
        }

        render_ids = _select_render_ids(chunks, only_chunks)

        run_state = runner.render_run(
            render_ids,
            provider,
            config.chunks_dir,
            # A slice always loads prior state so it augments the run rather
            # than replacing it, and always re-renders its own chunks -- see
            # force_chunk_ids. Writing a state file containing only the sliced
            # chunks destroyed the record of every other chunk in the run.
            resume=resume or bool(only_chunks),
            force_chunk_ids=only_chunks,
            fingerprints=fingerprints,
            # ...and record what actually happened, degradations included: a
            # chunk whose predecessor dead-lettered fell back to unchained,
            # and its fingerprint must say so or --resume compares against a
            # chain the render never had.
            fingerprint_amender=lambda chunk_id, fp: _amend_from_render(provider, chunk_id, fp),
        )

        output_video: Path | None = None
        if only_chunks:
            # Deliberately no Stage 5. Concatenating a subset would write a
            # file that looks like the finished song and is not -- the same
            # silently-desynced artifact the chunk timeline and the
            # fingerprints both exist to prevent. A slice's deliverable is
            # chunks to watch.
            logger.info(
                "Rendered a %d-chunk slice (%s) -- skipping Stage 5 assembly. The clips are "
                "in %s; a partial concat would claim to be the whole song.",
                len(render_ids),
                ", ".join(str(i) for i in render_ids),
                config.chunks_dir,
            )
        elif run_state.dead_lettered:
            logger.error(
                "Skipping Stage 5 assembly: %d chunk(s) dead-lettered: %s",
                len(run_state.dead_lettered),
                run_state.dead_lettered,
            )
        else:
            assembly_result = assemble_final_video(
                chunks, run_state, config.master_audio, config.final_video_dir, runner=ffmpeg_runner
            )
            output_video = assembly_result.output_video

    return RunReport(
        run_state=run_state,
        total_chunks=len(chunks),
        wall_seconds=clock() - start,
        output_video=output_video,
    )


def prepare_shot_plan(
    config: RunConfig,
    output_path: str | Path,
    *,
    source: str,
    generated_at: str,
    align_model: object | None = None,
    force: bool = False,
    from_plan: str | Path | None = None,
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

    ``from_plan`` re-anchors the skeleton against the timeline a render with
    *that plan* will actually produce, by loading it **for its
    ``length_seconds`` only** and re-slicing with them (issue #54 design
    section 5). Without it this function slices with no ``shot_lengths`` while
    :func:`run_pipeline` passes ``shot_length_requests(plan)``, so the moment a
    plan expresses one editorial length the skeleton describes a timeline no
    render will ever produce and every chunk after that length raises
    ``ShotPlanDriftError``. Nothing else in the source plan is read -- not the
    shot text, not the camera direction -- so this never launders an authored
    line into a file that is meant to be a blank skeleton.

    A plan that cannot be read is a hard failure, never a silent fall back to a
    length-free skeleton: that fallback would look exactly like success and
    drift hours later, on the GPU.
    """
    lines = parse_lyrics(config.lyrics_file, config.cast, config.default_lead_vocalist)
    alignment = align(
        config.master_audio,
        lines,
        model=align_model,
        strict_alignment=config.strict_alignment,
        overrides=config.alignment_overrides,
    )
    shot_lengths: tuple[ShotLength, ...] = ()
    if from_plan is not None:
        shot_lengths = shot_length_requests(
            load_shot_plan(from_plan, setting=config.setting, cast_names=config.cast)
        )
        logger.info(
            "Re-anchoring the skeleton against %s: %d editorial shot length(s) applied "
            "(issue #52 follow-up)",
            from_plan,
            len(shot_lengths),
        )
    chunks = slice_audio(
        config.master_audio,
        alignment,
        config.hardware,
        config.chunks_dir,
        cover_instrumentals=config.instrumental_coverage,
        instrumental_shot_seconds=config.instrumental_shot_seconds,
        shot_lengths=shot_lengths,
    )
    if not chunks:
        raise PipelineError(
            f"no chunks produced by slicing {config.master_audio} against "
            f"{config.lyrics_file} -- nothing to prepare a shot plan for (no non-empty "
            "lyric lines aligned to audio?)"
        )
    logger.info("Stage 1-2 complete: %d chunk(s) available to plan (issue #52)", len(chunks))

    return write_shot_plan_skeleton(
        chunks, output_path, source=source, generated_at=generated_at, force=force
    )


def _log_final_report(report: RunReport) -> None:
    dead = report.dead_lettered
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

    try:
        config = load_config(Path(args.config), **overrides)
    except ConfigError:
        logger.exception("Failed to load run config from %s", args.config)
        return EXIT_ERROR

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
            )
        except (PipelineError, ShotPlanError):
            logger.exception("Failed to prepare shot plan")
            return EXIT_ERROR
        return EXIT_SUCCESS

    try:
        report = run_pipeline(config, resume=args.resume, only_chunks=args.only_chunks)
    except Exception:
        logger.exception("Pipeline run failed")
        return EXIT_ERROR

    _log_final_report(report)
    return EXIT_PARTIAL_FAILURE if report.dead_lettered else EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
