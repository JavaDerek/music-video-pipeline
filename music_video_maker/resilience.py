"""Stage 4/Level 3: fault-tolerant execution state machine (issue #10).

Wraps a ``contracts.ExecutionClient`` (issue #9's ``ComfyUIExecutionClient``,
or any test double) in a retry/backoff/dead-letter loop that renders a whole
run of chunks without ever letting one chunk's failure kill the run:

* **Watchdog.** The wrapped client's WebSocket timeout is configured from
  ``watchdog_timeout_seconds`` at construction time -- issue #9 already
  raises :class:`~music_video_maker.execution.WebSocketTimeoutError` when
  ``recv()`` doesn't hear from ComfyUI in time; this module treats that as
  the watchdog firing. No sleep-polling anywhere in this module.
* **Recovery.** On a transient failure: ``POST /interrupt`` -> ``POST
  /free`` -> backoff (through an injectable :data:`Sleeper`, never a real
  ``time.sleep``) -> retry. A *transport* failure never reaches this path
  unless the render genuinely did not finish: the execution client first
  reconciles against ``/history`` (issue #43), so a chunk ComfyUI completed
  while this side was asleep or disconnected is harvested rather than
  re-rendered. What arrives here is a real failure, wearing the same
  exception type it always did.
* **Classification.** Most execution failures (WS timeout/disconnect, CUDA
  OOM, missing history, bad ``/view`` output) are transient and retried.
  Two failure modes are *not* retried: a
  :class:`~music_video_maker.execution.PromptSubmissionError` carrying
  ``node_errors`` (the graph itself is wrong -- retrying just burns GPU
  hours) and anything that looks like disk-full (ENOSPC / "No space left on
  device" -- doris runs high-90s%% full, per issue #10's comment). Both
  dead-letter the chunk immediately.
* **Dead-letter queue.** After ``max_render_attempts`` a chunk gets a
  ``ChunkResult(status=DEAD_LETTERED, ...)`` recording the real attempt
  count and full error history, and the run moves on to the next chunk.
* **Resume.** ``RunState`` is persisted to ``run_state_file`` after *every*
  chunk, atomically (temp file + ``os.replace``). On ``resume=True``,
  previously-``RENDERED``/``CACHED`` chunks are reused as ``CACHED`` -- but
  only when their ``video_file`` still exists on disk *and* the
  :class:`~music_video_maker.contracts.ChunkFingerprint` it was rendered for
  still matches the one the caller is asking for (issue #34) -- including
  the ``noise_seed`` it was rendered under (issue #38), so a run is
  reproducible and a re-seed is a visible reason to re-render rather than a
  silent one. A state file pointing at a deleted video, or at a chunk
  rendered for a span the timeline no longer has, re-renders instead of
  silently producing a run with a missing chunk or a desynced one.
  Previously ``DEAD_LETTERED`` chunks are always retried on resume.
* **Run state schema version.** ``run_state.json`` carries
  :data:`RUN_STATE_SCHEMA_VERSION`. A file written by any other version is
  rejected outright rather than misread -- a v1 file records no fingerprints
  at all, so nothing in it can be proven to belong to the current timeline.
* **Pre-flight disk check.** Before any rendering starts, free space at
  ``output_dir`` is checked against ``min_free_disk_gb`` through an
  injectable ``disk_usage`` seam (default ``shutil.disk_usage``). This is a
  local-disk check only -- there is no ComfyUI endpoint that reports
  server-side free disk, so server-side disk-full is instead caught
  reactively via the ENOSPC-signature classification above.
* **Between-chunk VRAM re-check (issue #23).** ``custody.py``'s pre-flight
  free-VRAM assertion runs once, before the run starts, and cannot see
  another process claiming the card afterwards -- which is exactly how the
  2026-08-07 outage happened: H3 staged its ~20GB onto a contended card and
  went **silent** rather than raising CUDA OOM, wedging the host badly
  enough to need a power cycle. If a ``vram_probe`` seam is injected (a
  zero-arg callable returning free VRAM in GB or ``None`` -- see
  ``custody.build_vram_probe``), this module re-reads it immediately before
  submitting *each* chunk that actually needs to render (cached/resumed
  chunks never touch the GPU, so they're skipped). A reading below
  ``min_free_vram_gb`` persists the run state accumulated so far and raises
  :class:`VramBelowFloorError` -- loudly, and without returning a run state
  that looks complete -- rather than risking a wedge; ``--resume`` then
  picks up from the last chunk that actually rendered. An unreadable
  reading (ComfyUI unreachable, unparseable) is logged and the run
  continues, mirroring ``NullCustodyManager``'s degrade-gracefully
  behaviour -- a best-effort check must not fail a run over its own
  flakiness. Every reading is logged at INFO regardless of outcome: on the
  incident night the only VRAM evidence came from ``nvidia-smi`` run by hand
  afterwards, and the run's own logs said nothing about the card.
  ``vram_probe=None`` (the default) skips the check entirely, preserving
  prior behaviour for any caller that doesn't wire it up.

Index-space guard: chunk ids, :class:`~music_video_maker.contracts.AlignedSegment`
indices and :class:`~music_video_maker.contracts.LyricLine` indices are all
bare ``int`` and are *not* interchangeable (this has bitten the project
twice already). Wherever this module accepts two chunk-carrying values
together -- an ``ExecutionClient.execute()`` result against the chunk id it
was asked to render, or a ``RunState.results`` dict key against the
``ChunkResult.chunk_id`` stored under it -- it asserts they agree and raises
:class:`ChunkIdMismatchError` naming both values.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from music_video_maker.continuity import chain_reuse_blocked
from music_video_maker.contracts import (
    ChunkFingerprint,
    ChunkResult,
    ChunkStatus,
    ExecutionClient,
    RunState,
    Workflow,
)
from music_video_maker.custody import DEFAULT_MIN_FREE_VRAM_GB
from music_video_maker.execution import PromptSubmissionError, WebSocketTimeoutError

if TYPE_CHECKING:
    from music_video_maker.config import RunConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Injectable seams
# --------------------------------------------------------------------------- #

WorkflowProvider = Callable[[int, RunState], Workflow]
"""Given a chunk id and the ``RunState`` accumulated so far, returns the
workflow to submit for that chunk. Called immediately before *each* attempt
(never cached up front), so a chunk whose predecessor dead-lettered gets a
freshly-decided workflow. A provider that raises counts as that attempt
failing and is fed into the normal retry/dead-letter path."""

Sleeper = Callable[[float], None]
"""A ``time.sleep``-shaped backoff seam. Real ``time.sleep`` must never run
in a test -- inject a recording stub instead."""

DiskUsage = Callable[[str], Any]
"""A ``shutil.disk_usage``-shaped seam: takes a path, returns an object with
a ``.free`` attribute (bytes)."""

VramProbe = Callable[[], "float | None"]
"""A zero-arg seam: call it, get a fresh free-VRAM-in-GB reading, or ``None``
when the reading can't be trusted (issue #23). Deliberately a plain callable
rather than a ``(session, base_url)`` pair -- this module owns *what to do*
with a reading, not *how* it's taken. ``custody.build_vram_probe(session,
base_url)`` builds one of these bound to a real ComfyUI session; tests inject
a scripted stub instead. ``None`` here (the default) disables the check
entirely, preserving prior behaviour for a caller that doesn't wire it up."""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ResilienceError(RuntimeError):
    """Base class for every error raised by this module."""


class DiskPreflightError(ResilienceError):
    """Raised when free disk space is below ``min_free_disk_gb`` before a
    run starts. The whole point is to refuse loudly up front rather than
    hitting ENOSPC forty chunks in."""


class VramBelowFloorError(ResilienceError):
    """Raised when a between-chunk free-VRAM reading (issue #23) drops below
    ``min_free_vram_gb`` partway through a run.

    Unlike a transient chunk failure, this is deliberately **not** retried
    and **not** swallowed into a dead-lettered chunk: the failure mode it
    guards against is H3 staging its ~20GB onto a contended card and going
    silent rather than raising CUDA OOM, wedging the host badly enough to
    need a power cycle. The run state accumulated so far is persisted before
    this is raised, so ``--resume`` picks up from the last chunk that
    actually rendered -- converting "wedged host, power cycle, 90-minute
    outage" into "run stopped after chunk N, resume when the card is free".
    """


class RunStateSchemaError(ResilienceError):
    """Raised when ``run_state.json`` carries a ``schema_version`` this build
    does not write (issue #34).

    Refusing is the whole point: a v1 file records no per-chunk fingerprints,
    so every chunk in it is un-provable, and reading it as if it were current
    would reuse chunks that may belong to a timeline that no longer exists.
    """


class ChunkIdMismatchError(ResilienceError):
    """Index-space guard: two chunk-carrying values disagreed on chunk id.

    Raised, never silently coerced, whenever this module can cross-check a
    chunk id against another chunk id it should equal (an
    ``ExecutionClient.execute()`` result against the id it was asked to
    render, or a ``RunState.results`` dict key against the ``ChunkResult``
    stored under it).
    """

    def __init__(self, expected: int, actual: int, *, context: str = "") -> None:
        self.expected = expected
        self.actual = actual
        suffix = f" ({context})" if context else ""
        super().__init__(
            f"chunk id mismatch{suffix}: expected chunk_id={expected}, got chunk_id={actual}"
        )


def _assert_chunk_id(expected: int, actual: int, *, context: str) -> None:
    if expected != actual:
        logger.error(
            "Index-space guard failed (%s): expected chunk_id=%s, got chunk_id=%s",
            context,
            expected,
            actual,
        )
        raise ChunkIdMismatchError(expected, actual, context=context)


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #


class _FailureClass(str, Enum):
    RETRYABLE = "retryable"
    DISK_FULL = "disk_full"
    FATAL = "fatal"


_DISK_FULL_SIGNATURES = (
    "no space left on device",
    "enospc",
    "disk full",
    "errno 28",
)


def _looks_like_disk_full(exc: BaseException) -> bool:
    """Best-effort ENOSPC sniff across an exception and its cause chain.

    Catches the real path: ``execution.py``'s ``_write_video`` wraps an
    ``OSError`` (whose ``str()`` includes the OS's "No space left on
    device" text) in an ``OutputRetrievalError``. A generic non-200 from
    ``/view`` (e.g. an arbitrary server 500) does *not* match and stays
    generically retryable -- only an actual ENOSPC-shaped message dead-letters
    immediately.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).lower()
        if any(sig in text for sig in _DISK_FULL_SIGNATURES):
            return True
        current = current.__cause__ or current.__context__
    return False


def _classify_failure(exc: BaseException) -> _FailureClass:
    if _looks_like_disk_full(exc):
        return _FailureClass.DISK_FULL
    if isinstance(exc, PromptSubmissionError) and exc.node_errors:
        return _FailureClass.FATAL
    # Everything else -- WS timeout/disconnect, CUDA OOM, missing history,
    # bad /view output, a bare PromptSubmissionError with no node_errors
    # (e.g. a network hiccup on /prompt), or an unexpected exception raised
    # by a WorkflowProvider -- is treated as transient and retried; it will
    # still dead-letter once max_render_attempts is exhausted.
    return _FailureClass.RETRYABLE


# --------------------------------------------------------------------------- #
# RunState (de)serialization -- atomic persistence for --resume
# --------------------------------------------------------------------------- #


RUN_STATE_SCHEMA_VERSION = 2
"""Version stamped into ``run_state.json``.

* 1 -- no ``schema_version`` key and no per-chunk fingerprints (pre-#34).
* 2 -- ``schema_version`` plus a ``fingerprint`` on every result.

Bump this whenever a resumed run could *misread* an older file. A v1 file is
not upgradable: the information a v2 reader needs to prove a cached chunk
belongs to the current timeline was never written down.

Issue #38 added ``fingerprint.noise_seed`` **without** a bump, deliberately.
An early-v2 file simply has no ``noise_seed`` key, which deserializes to
``None`` -- *unrecorded*, which :class:`ChunkFingerprint` already treats as
"cannot prove a match" rather than as agreement. So such a chunk re-renders
against a run that names a seed, for the precise reason logged, and stays
reusable for one that does not; nothing is misread either way. Bumping instead
would have thrown away a whole file's worth of still-valid timeline evidence,
unescapably, to re-derive a conclusion the field comparison already reaches --
and ``resume_ignore_prompt_changes`` would have had nothing to forgive."""


def _serialize_fingerprint(fingerprint: ChunkFingerprint) -> dict[str, Any]:
    return {
        "start": fingerprint.start,
        "end": fingerprint.end,
        "frame_count": fingerprint.frame_count,
        "render_width": fingerprint.render_width,
        "render_height": fingerprint.render_height,
        "prompt_hash": fingerprint.prompt_hash,
        "character": fingerprint.character,
        "image_ref": fingerprint.image_ref,
        "present_cast": (
            list(fingerprint.present_cast) if fingerprint.present_cast is not None else None
        ),
        "noise_seed": fingerprint.noise_seed,
        "chained_from": fingerprint.chained_from,
        "conditioning_source": fingerprint.conditioning_source,
        "instrumental_audio_gain_db": fingerprint.instrumental_audio_gain_db,
        "text_encoder": fingerprint.text_encoder,
        "lora": fingerprint.lora,
        "lora_strength": fingerprint.lora_strength,
        "template_hash": fingerprint.template_hash,
    }


def _deserialize_fingerprint(raw: dict[str, Any] | None) -> ChunkFingerprint | None:
    if not raw:
        return None
    return ChunkFingerprint(
        start=float(raw["start"]),
        end=float(raw["end"]),
        frame_count=raw.get("frame_count"),
        render_width=raw.get("render_width"),
        render_height=raw.get("render_height"),
        prompt_hash=raw.get("prompt_hash"),
        character=raw.get("character"),
        image_ref=raw.get("image_ref"),
        # Absent in a file written before issue #59: None = unrecorded, which
        # is also how a run that staged nobody records it, so an old file and
        # a present-cast-free run compare equal and neither re-renders.
        present_cast=(
            tuple(raw["present_cast"]) if raw.get("present_cast") is not None else None
        ),
        # Absent in a file written before issue #38: ``None`` means the seed
        # was never recorded, which is not evidence that it was 0. See
        # RUN_STATE_SCHEMA_VERSION for why that is handled here rather than by
        # a version bump.
        noise_seed=raw.get("noise_seed"),
        # Absent in a file written before issue #28: None = unchained, which
        # is also the safe reading -- an unchained chunk depends on nobody's
        # pixels, so the chain-reuse rule simply never blocks it.
        chained_from=raw.get("chained_from"),
        # Absent in a file written before issue #25: None = unrecorded, which
        # proves nothing and never compares equal to a named source.
        conditioning_source=raw.get("conditioning_source"),
        # Absent in a file written before F26: None = unattenuated, which is
        # what every pre-F26 run actually did, so this one key is a true
        # default rather than "unrecorded".
        instrumental_audio_gain_db=raw.get("instrumental_audio_gain_db"),
        # Absent in a file written before issue #39: None = unrecorded, which
        # is not evidence that the chunk was encoded by the encoder this run
        # names. Same no-bump reasoning as noise_seed above -- an early-v2
        # reader cannot *misread* a missing key, it just cannot prove a match.
        text_encoder=raw.get("text_encoder"),
        # Absent before issue #62: None = no adapter, which is the same pixels
        # as unrecorded, so an old file and a LoRA-free run compare equal.
        lora=raw.get("lora"),
        lora_strength=raw.get("lora_strength"),
        # Absent in a file written before issue #45: None = unrecorded, which
        # is not evidence that the chunk rendered through the template this run
        # loaded. That reading is the whole point -- a state file from before
        # the graph was hashed cannot prove its chunks predate a template edit,
        # so it must not claim to.
        template_hash=raw.get("template_hash"),
    )


def _serialize_result(result: ChunkResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "status": result.status.value,
        "video_file": str(result.video_file) if result.video_file is not None else None,
        "prompt_id": result.prompt_id,
        "attempts": result.attempts,
        "errors": list(result.errors),
        "render_seconds": result.render_seconds,
        "fingerprint": (
            _serialize_fingerprint(result.fingerprint) if result.fingerprint is not None else None
        ),
    }


def _deserialize_result(raw: dict[str, Any]) -> ChunkResult:
    video_file = raw.get("video_file")
    return ChunkResult(
        chunk_id=int(raw["chunk_id"]),
        status=ChunkStatus(raw["status"]),
        video_file=Path(video_file) if video_file else None,
        prompt_id=raw.get("prompt_id"),
        attempts=int(raw.get("attempts", 1)),
        errors=tuple(raw.get("errors") or ()),
        render_seconds=raw.get("render_seconds"),
        fingerprint=_deserialize_fingerprint(raw.get("fingerprint")),
    )


def _serialize_run_state(run_state: RunState) -> dict[str, Any]:
    return {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "run_id": run_state.run_id,
        "results": {
            str(chunk_id): _serialize_result(result)
            for chunk_id, result in run_state.results.items()
        },
    }


def _deserialize_run_state(payload: dict[str, Any]) -> RunState:
    version = payload.get("schema_version")
    if version != RUN_STATE_SCHEMA_VERSION:
        raise RunStateSchemaError(
            f"run state schema_version={version!r}, this build reads and writes "
            f"{RUN_STATE_SCHEMA_VERSION}"
            + (
                " -- a file with no schema_version predates per-chunk fingerprints "
                "(issue #34), so nothing in it can be proven to belong to the current "
                "chunk timeline"
                if version is None
                else ""
            )
        )

    results: dict[int, ChunkResult] = {}
    for key, raw in (payload.get("results") or {}).items():
        result = _deserialize_result(raw)
        chunk_id = int(key)
        _assert_chunk_id(
            chunk_id, result.chunk_id, context="run state file key vs ChunkResult.chunk_id"
        )
        results[chunk_id] = result
    return RunState(
        run_id=payload["run_id"],
        results=results,
    )


def _first_existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor) if path.anchor else Path(".")


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #


class ResilientRunner:
    """Fault-tolerant wrapper around a ``contracts.ExecutionClient``."""

    def __init__(
        self,
        execution_client: ExecutionClient,
        *,
        run_state_file: Path,
        watchdog_timeout_seconds: float = 900.0,
        max_render_attempts: int = 3,
        retry_backoff_seconds: float = 5.0,
        min_free_disk_gb: float = 20.0,
        run_id: str | None = None,
        sleeper: Sleeper = time.sleep,
        disk_usage: DiskUsage = shutil.disk_usage,
        ignore_prompt_changes: bool = False,
        vram_probe: VramProbe | None = None,
        min_free_vram_gb: float = DEFAULT_MIN_FREE_VRAM_GB,
        between_chunk_min_free_vram_gb: float | None = None,
    ) -> None:
        self.execution_client = execution_client
        self.run_state_file = Path(run_state_file)
        self.watchdog_timeout_seconds = watchdog_timeout_seconds
        self.max_render_attempts = max_render_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_free_disk_gb = min_free_disk_gb
        self.run_id = run_id
        self.sleeper = sleeper
        self.disk_usage = disk_usage
        self.ignore_prompt_changes = ignore_prompt_changes
        """Issue #34: reuse a cached chunk whose span is unchanged but whose
        *prompt* changed. Covers the content tier only -- a moved span or a
        changed resolution is never reusable, by flag or otherwise."""
        self.vram_probe = vram_probe
        """Issue #23: zero-arg seam re-read immediately before submitting
        each chunk that actually needs to render. ``None`` (the default)
        disables the between-chunk check entirely -- see the module
        docstring's "Between-chunk VRAM re-check" section."""
        self.min_free_vram_gb = min_free_vram_gb
        """The custody pre-flight's COLD-CARD floor, kept here only for
        reporting. Deliberately *not* the between-chunk threshold -- see
        :attr:`between_chunk_min_free_vram_gb`."""
        self.between_chunk_min_free_vram_gb = between_chunk_min_free_vram_gb
        """Optional floor for the between-chunk reading. ``None`` (the
        default) means report, never gate.

        This is a separate knob from ``min_free_vram_gb`` because the two
        measure different worlds. Before the run the card is cold and ~20 GB
        is free; once ComfyUI stages H3 it holds ~19995 MB and *never unloads
        between prompts*, so a healthy mid-render reading on the 4090 is
        1.5-5 GB. Gating that against the cold-card floor aborted a real run
        after its first chunk (measured 2026-08-08: 20.72 GB before chunk 0,
        1.54 GB before chunk 1). Only set this to a number you have watched
        your own card hold steadily mid-render -- and note our own workload
        swings several GB between sampler and VAE decode, so a tight
        threshold will fire on us rather than on an intruder."""

        if hasattr(execution_client, "ws_timeout"):
            logger.info(
                "Configuring execution client watchdog: ws_timeout=%.1fs",
                watchdog_timeout_seconds,
            )
            execution_client.ws_timeout = watchdog_timeout_seconds  # type: ignore[attr-defined]
        else:
            logger.debug(
                "execution_client has no ws_timeout attribute -- watchdog timeout was not "
                "configured on it (fine for a test double that doesn't need it)"
            )

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        execution_client: ExecutionClient,
        *,
        run_id: str | None = None,
        sleeper: Sleeper = time.sleep,
        disk_usage: DiskUsage = shutil.disk_usage,
        vram_probe: VramProbe | None = None,
    ) -> ResilientRunner:
        """Convenience constructor reading the Wave 3 knobs off a loaded
        ``RunConfig`` (issue #2/#10). ``load_config`` always resolves
        ``run_state_file``, so this raises loudly if handed a
        hand-constructed config that left it ``None``.

        ``vram_probe`` (issue #23) is a caller-supplied seam, not something
        this method can build itself: constructing one needs an HTTP session
        bound to ``config.comfyui_url`` (``custody.build_vram_probe``), and
        that session lives with whatever already built the execution client
        -- this module stays free of ``requests``. Omitting it (the default)
        disables the between-chunk VRAM re-check.
        """
        if config.run_state_file is None:
            raise ResilienceError(
                "config.run_state_file is None -- load_config() always resolves it; "
                "pass a fully-loaded RunConfig, not a hand-constructed one"
            )
        return cls(
            execution_client,
            run_state_file=config.run_state_file,
            watchdog_timeout_seconds=config.watchdog_timeout_seconds,
            max_render_attempts=config.max_render_attempts,
            retry_backoff_seconds=config.retry_backoff_seconds,
            min_free_disk_gb=config.min_free_disk_gb,
            run_id=run_id,
            sleeper=sleeper,
            disk_usage=disk_usage,
            ignore_prompt_changes=config.resume_ignore_prompt_changes,
            vram_probe=vram_probe,
            min_free_vram_gb=config.min_free_vram_gb,
            between_chunk_min_free_vram_gb=config.between_chunk_min_free_vram_gb,
        )

    # -- public entry point -------------------------------------------------- #

    def render_run(
        self,
        chunk_ids: Sequence[int],
        provider: WorkflowProvider,
        output_dir: Path,
        *,
        resume: bool = False,
        fingerprints: Mapping[int, ChunkFingerprint] | None = None,
        fingerprint_amender: Callable[[int, ChunkFingerprint], ChunkFingerprint] | None = None,
        force_chunk_ids: Collection[int] | None = None,
    ) -> RunState:
        """Render every chunk in ``chunk_ids``, reusing prior work on ``resume``.

        ``fingerprints`` (issue #34) maps chunk id -> what that chunk is being
        rendered *for* this run: its span, frame count, resolution and prompt.
        It is recorded alongside each result and, on a resumed run, compared
        against what the cached chunk was actually rendered for -- a chunk
        whose span moved is re-rendered rather than reused into a silently
        desynced video. Omitting it keeps the old existence-only behaviour and
        says so in the log; it is never silently assumed to match.

        ``fingerprint_amender`` (issue #28) is called after a chunk actually
        renders, with the chunk id and the planned fingerprint, and its return
        value is what gets recorded. The chaining decision is only knowable
        after the provider runs -- a degraded chunk falls back to unchained --
        and recording the plan instead of the truth would make ``--resume``
        compare against a fingerprint the render never had. It is never called
        for reused chunks: their recorded fingerprint IS the truth.
        """
        output_dir = Path(output_dir)
        self._preflight_disk_check(output_dir)

        run_state = self._load_or_init_run_state(resume=resume)
        wholesale_change = (
            self._report_timeline_change(run_state, chunk_ids, fingerprints) if resume else False
        )

        self._force_chunk_ids = frozenset(force_chunk_ids or ())

        # Issue #28: chunk ids that re-rendered THIS run. A cached chunk whose
        # first frame came from one of these was seeded from footage that no
        # longer exists; chunks render in ascending id order and the chain
        # reaches back one chunk, so checking as we go propagates the rule
        # down the whole chain by itself.
        rerendered: set[int] = set()

        for chunk_id in chunk_ids:
            expected = fingerprints.get(chunk_id) if fingerprints is not None else None
            cached = (
                self._reusable_cached_result(
                    run_state,
                    chunk_id,
                    expected,
                    quiet=wholesale_change,
                    rerendered=rerendered,
                )
                if resume
                else None
            )
            if cached is not None:
                _store_result(run_state, chunk_id, cached)
                logger.info(
                    "Chunk %d reused from a previous run (resume): %s", chunk_id, cached.video_file
                )
            else:
                self._check_vram_before_chunk(chunk_id, run_state)
                result = self._render_chunk(chunk_id, provider, run_state, output_dir)
                if expected is not None:
                    recorded = (
                        fingerprint_amender(chunk_id, expected)
                        if fingerprint_amender is not None
                        else expected
                    )
                    result = replace(result, fingerprint=recorded)
                rerendered.add(chunk_id)
                _store_result(run_state, chunk_id, result)
            self._persist(run_state)

        dead = run_state.dead_lettered
        if dead:
            logger.error(
                "Run %s finished with %d dead-lettered chunk(s): %s",
                run_state.run_id,
                len(dead),
                dead,
            )
        else:
            logger.info(
                "Run %s finished: all %d chunk(s) succeeded", run_state.run_id, len(chunk_ids)
            )
        return run_state

    # -- per-chunk retry loop ------------------------------------------------- #

    def _render_chunk(
        self,
        chunk_id: int,
        provider: WorkflowProvider,
        run_state: RunState,
        output_dir: Path,
    ) -> ChunkResult:
        errors: list[str] = []
        for attempt in range(1, self.max_render_attempts + 1):
            result, failure_class = self._attempt_chunk(
                chunk_id, provider, run_state, output_dir, attempt, errors
            )
            if result is not None:
                logger.info(
                    "Chunk %d rendered on attempt %d/%d",
                    chunk_id,
                    attempt,
                    self.max_render_attempts,
                )
                return replace(result, attempts=attempt, errors=tuple(errors))

            assert failure_class is not None  # for type checkers; always set on failure

            if failure_class in (_FailureClass.DISK_FULL, _FailureClass.FATAL):
                logger.error(
                    "Chunk %d: non-retryable failure (%s) on attempt %d -- dead-lettering "
                    "immediately without further retries",
                    chunk_id,
                    failure_class.value,
                    attempt,
                )
                return self._dead_letter(chunk_id, attempt, errors)

            self._recover(chunk_id, attempt)

            if attempt >= self.max_render_attempts:
                logger.error(
                    "Chunk %d exhausted all %d attempt(s) -- dead-lettering",
                    chunk_id,
                    self.max_render_attempts,
                )
                return self._dead_letter(chunk_id, attempt, errors)

            self._backoff(attempt)

        # Unreachable given the loop bounds above; kept for type-checker sanity.
        return self._dead_letter(chunk_id, self.max_render_attempts, errors)

    def _attempt_chunk(
        self,
        chunk_id: int,
        provider: WorkflowProvider,
        run_state: RunState,
        output_dir: Path,
        attempt: int,
        errors: list[str],
    ) -> tuple[ChunkResult | None, _FailureClass | None]:
        try:
            workflow = provider(chunk_id, run_state)
        except Exception as exc:  # noqa: BLE001 -- a provider bug must not kill the run
            logger.exception(
                "WorkflowProvider raised for chunk_id=%d attempt=%d/%d",
                chunk_id,
                attempt,
                self.max_render_attempts,
            )
            errors.append(f"attempt {attempt}: provider raised {type(exc).__name__}: {exc}")
            return None, _classify_failure(exc)

        try:
            result = self.execution_client.execute(workflow, chunk_id, output_dir)
        except Exception as exc:  # noqa: BLE001 -- classified below, never left uncaught
            failure_class = _classify_failure(exc)
            if isinstance(exc, WebSocketTimeoutError):
                logger.error(
                    "Watchdog timeout fired for chunk_id=%d attempt=%d/%d "
                    "(no WS signal within %.1fs): %s",
                    chunk_id,
                    attempt,
                    self.max_render_attempts,
                    self.watchdog_timeout_seconds,
                    exc,
                )
            else:
                logger.error(
                    "Chunk %d attempt %d/%d failed (%s, %s): %s",
                    chunk_id,
                    attempt,
                    self.max_render_attempts,
                    failure_class.value,
                    type(exc).__name__,
                    exc,
                )
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            return None, failure_class
        else:
            # Index-space guard: the client must report back the exact chunk
            # id it was asked to render. Deliberately outside the try/except
            # above so it is never swallowed into the retry path -- this is
            # a wiring bug, not a transient execution failure.
            _assert_chunk_id(chunk_id, result.chunk_id, context="ExecutionClient.execute() result")
            return result, None

    def _recover(self, chunk_id: int, attempt: int) -> None:
        logger.warning(
            "Chunk %d attempt %d/%d: running recovery sequence (interrupt -> free)",
            chunk_id,
            attempt,
            self.max_render_attempts,
        )
        try:
            self.execution_client.interrupt()
        except Exception:  # noqa: BLE001 -- recovery must never mask the underlying failure
            logger.exception("Chunk %d: interrupt() raised during recovery", chunk_id)
        try:
            self.execution_client.free(unload_models=True)
        except Exception:  # noqa: BLE001
            logger.exception("Chunk %d: free() raised during recovery", chunk_id)

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
        logger.info("Backing off %.2fs before retry attempt %d", delay, attempt + 1)
        self.sleeper(delay)

    def _dead_letter(self, chunk_id: int, attempts: int, errors: list[str]) -> ChunkResult:
        logger.error(
            "Chunk %d DEAD-LETTERED after %d attempt(s). Error history: %s",
            chunk_id,
            attempts,
            errors,
        )
        return ChunkResult(
            chunk_id=chunk_id,
            status=ChunkStatus.DEAD_LETTERED,
            video_file=None,
            prompt_id=None,
            attempts=attempts,
            errors=tuple(errors),
        )

    # -- resume ---------------------------------------------------------------- #

    def _reusable_cached_result(
        self,
        run_state: RunState,
        chunk_id: int,
        expected: ChunkFingerprint | None = None,
        *,
        quiet: bool = False,
        rerendered: AbstractSet[int] = frozenset(),
    ) -> ChunkResult | None:
        if chunk_id in self._force_chunk_ids:
            # A validation slice (cli --only-chunks) re-renders its chunks on
            # purpose: the reason is usually something no fingerprint can see,
            # like a fixed detector threshold. It still loads and preserves
            # every other chunk's result, so the slice augments the run state
            # rather than replacing it.
            logger.info("Chunk %d: re-rendering because it was explicitly selected", chunk_id)
            return None
        existing = run_state.results.get(chunk_id)
        if existing is None or existing.status not in (ChunkStatus.RENDERED, ChunkStatus.CACHED):
            return None
        if existing.video_file is None or not Path(existing.video_file).exists():
            logger.warning(
                "Chunk %d was %s in the prior run state but video_file %s is missing on "
                "disk -- re-rendering instead of silently producing a run with a missing chunk",
                chunk_id,
                existing.status.value,
                existing.video_file,
            )
            return None

        if expected is None:
            # No fingerprint to check against -- the caller was warned once, up
            # front, that identity cannot be verified this run.
            return replace(existing, status=ChunkStatus.CACHED)

        stored = existing.fingerprint
        # ``quiet`` demotes the per-chunk detail to DEBUG when a single
        # whole-timeline summary has already been logged -- 39 near-identical
        # warnings bury the one line that explains what happened.
        log = logger.debug if quiet else logger.warning

        if stored is None:
            log(
                "Chunk %d has no recorded fingerprint in the prior run state, so there is no "
                "way to prove %s was rendered for this chunk's span -- re-rendering rather "
                "than risking a silently desynced video",
                chunk_id,
                existing.video_file,
            )
            return None

        if chain_reuse_blocked(stored.chained_from, rerendered):
            # Issue #28: not escapable via ignore_prompt_changes -- this is
            # not a content preference, it is a dependency on pixels that no
            # longer exist. Reusing it puts a hard visual cut exactly where
            # the chain was supposed to remove one.
            log(
                "Chunk %d was chained from chunk %d's last frame, and chunk %d re-rendered "
                "this run -- the footage this chunk was seeded from no longer exists, so it "
                "re-renders too",
                chunk_id,
                stored.chained_from,
                stored.chained_from,
            )
            return None

        conditioning = expected.conditioning_differences(stored)
        if conditioning:
            # Never escapable. Each field in this tier is some run's experiment
            # variable: the conditioning audio (#25, mix <-> stem), the encoder
            # that read the sentence (#39), the graph they were fed into (#45).
            # Reusing across any of them hands the comparison its control twice.
            #
            # The message names the *fields*, not a guess at which one moved --
            # it said "different conditioning audio" while the tier already held
            # the encoder too, so a #39 re-render was reported as an audio
            # change. A tier that grows must not keep describing itself by its
            # first member.
            log(
                "Chunk %d was rendered with different conditioning (%s) -- "
                "re-rendering; this is not escapable via resume_ignore_prompt_changes",
                chunk_id,
                _describe_changes(stored, expected, conditioning),
            )
            return None

        moved = expected.timeline_differences(stored)
        if moved:
            log(
                "Chunk %d was rendered for a different place in the song (%s) -- re-rendering; "
                "reusing it would put the clip at a moment it was never rendered for",
                chunk_id,
                _describe_changes(stored, expected, moved),
            )
            return None

        changed = expected.content_differences(stored)
        if changed:
            description = _describe_changes(stored, expected, changed)
            if self.ignore_prompt_changes:
                logger.info(
                    "Chunk %d covers the same span but its content changed (%s); reusing the "
                    "existing video anyway because resume_ignore_prompt_changes is set -- this "
                    "chunk does NOT reflect the current config",
                    chunk_id,
                    description,
                )
                # Deliberately keeps ``existing.fingerprint``: the state file
                # must keep recording what the video was really rendered from,
                # or a later run without the flag would believe it matches.
                return replace(existing, status=ChunkStatus.CACHED)
            log(
                "Chunk %d covers the same span but no longer matches the configured prompt, "
                "cast or noise seed (%s) -- re-rendering. Set resume_ignore_prompt_changes "
                "(or pass --ignore-prompt-changes) to reuse it as-is instead",
                chunk_id,
                description,
            )
            return None

        return replace(existing, status=ChunkStatus.CACHED)

    def _report_timeline_change(
        self,
        run_state: RunState,
        chunk_ids: Sequence[int],
        fingerprints: Mapping[int, ChunkFingerprint] | None,
    ) -> bool:
        """Log one summary when the *whole* timeline moved; return whether it did.

        A lyrics edit does not move one chunk, it moves every chunk after the
        edit. Reporting that as 38 separate warnings hides the single fact that
        matters, so the wholesale case gets one line and the per-chunk detail
        drops to DEBUG.
        """
        if fingerprints is None:
            logger.warning(
                "Resuming without per-chunk fingerprints: cached chunks cannot be verified "
                "against the current chunk timeline, so a change to the lyrics, alignment, "
                "chunk bounds or resolution since the last run would go undetected"
            )
            return False

        comparable = [
            chunk_id
            for chunk_id in chunk_ids
            if (result := run_state.results.get(chunk_id)) is not None and result.succeeded
        ]
        moved = [
            chunk_id
            for chunk_id in comparable
            if (stored := run_state.results[chunk_id].fingerprint) is None
            or fingerprints[chunk_id].timeline_differences(stored)
        ]
        orphaned = sorted(set(run_state.results) - set(chunk_ids))

        # Majority-moved, not "any moved": one moved chunk is best reported
        # against the chunk itself. ``max(1, ...)`` keeps a 1- or 2-chunk run
        # out of the summary path, where per-chunk detail is more useful.
        if not moved or len(moved) <= max(1, len(comparable) // 2):
            if orphaned:
                logger.warning(
                    "Prior run state has %d chunk id(s) this run does not render at all: %s "
                    "-- the timeline is shorter than it was; those entries are ignored",
                    len(orphaned),
                    orphaned,
                )
            return False

        logger.error(
            "THE CHUNK TIMELINE HAS CHANGED since run %s: %d of %d comparable chunk(s) no "
            "longer cover the span they were rendered for (e.g. chunk %d: %s)%s. Those chunks "
            "will be re-rendered -- reusing them would assemble a silently desynced video. "
            "Per-chunk detail is at DEBUG. Usual causes: the lyrics file, the alignment "
            "model, the chunk duration bounds, instrumental_coverage, or the master audio "
            "changed since the run being resumed",
            run_state.run_id,
            len(moved),
            len(comparable),
            moved[0],
            _describe_first_change(run_state, fingerprints, moved[0]),
            (
                f"; {len(orphaned)} prior chunk id(s) are no longer rendered at all"
                if orphaned
                else ""
            ),
        )
        return True

    def _load_or_init_run_state(self, *, resume: bool) -> RunState:
        if resume and self.run_state_file.exists():
            try:
                loaded = self._read_run_state(self.run_state_file)
            except RunStateSchemaError:
                logger.exception(
                    "Run state file %s was written by a different schema version -- refusing "
                    "to reuse chunks it cannot prove belong to this timeline. Starting a fresh "
                    "run: every chunk will be re-rendered",
                    self.run_state_file,
                )
                return self._new_run_state()
            except ChunkIdMismatchError:
                logger.exception(
                    "Run state file %s failed the chunk-id index-space guard -- refusing to "
                    "trust it, starting a fresh run instead",
                    self.run_state_file,
                )
                return self._new_run_state()
            except Exception:
                logger.exception(
                    "Failed to load run state from %s -- starting a fresh run instead",
                    self.run_state_file,
                )
                return self._new_run_state()
            else:
                logger.info(
                    "Resuming run %s from %s (%d prior result(s))",
                    loaded.run_id,
                    self.run_state_file,
                    len(loaded.results),
                )
                return loaded

        if resume:
            logger.warning(
                "resume=True but no run state file found at %s -- starting a fresh run",
                self.run_state_file,
            )
        return self._new_run_state()

    def _new_run_state(self) -> RunState:
        return RunState(run_id=self.run_id or str(uuid.uuid4()))

    def _read_run_state(self, path: Path) -> RunState:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _deserialize_run_state(payload)

    def _persist(self, run_state: RunState) -> None:
        path = self.run_state_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            payload = json.dumps(_serialize_run_state(run_state), indent=2)
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
            logger.debug(
                "Persisted run state run_id=%s to %s (%d result(s))",
                run_state.run_id,
                path,
                len(run_state.results),
            )
        except OSError:
            logger.exception(
                "Failed to persist run state run_id=%s to %s -- resume from this point may "
                "not be possible",
                run_state.run_id,
                path,
            )

    # -- between-chunk VRAM re-check (issue #23) ---------------------------------- #

    def _check_vram_before_chunk(self, chunk_id: int, run_state: RunState) -> None:
        """Re-read free VRAM immediately before submitting ``chunk_id``.

        No-op when ``vram_probe`` is unset (default). Below the floor:
        persist ``run_state`` (everything rendered so far) and raise
        :class:`VramBelowFloorError` -- never return a partial ``RunState``
        that looks complete, since Stage 5 would then assemble a truncated,
        desynced video with no error anywhere. Unreadable: log and continue,
        a best-effort check must not block a run over its own flakiness.
        """
        if self.vram_probe is None:
            return

        try:
            free_gb = self.vram_probe()
        except Exception as exc:  # noqa: BLE001 -- a best-effort check must not kill the run
            logger.exception(
                "Pre-chunk VRAM check for chunk %d raised %s -- treating it as an unreadable "
                "reading and continuing; the probe is best-effort and must not fail a run "
                "over its own flakiness",
                chunk_id,
                type(exc).__name__,
            )
            return

        if free_gb is None:
            logger.info(
                "Pre-chunk VRAM check for chunk %d: reading unavailable -- continuing without "
                "blocking (a best-effort check must not fail the run over its own flakiness)",
                chunk_id,
            )
            return

        floor = self.between_chunk_min_free_vram_gb
        logger.info(
            "Pre-chunk VRAM check for chunk %d: %.2f GB free%s",
            chunk_id,
            free_gb,
            f" (abort below {floor:.2f} GB)" if floor is not None else " (reporting only)",
        )

        if floor is not None and free_gb < floor:
            self._persist(run_state)
            message = (
                f"free VRAM dropped to {free_gb:.2f} GB before chunk {chunk_id} "
                f"(need >= {floor:.2f} GB) -- another process likely claimed "
                f"the card mid-run. Refusing to submit chunk {chunk_id}: staging H3's weights "
                f"onto a contended card can go silent instead of raising CUDA OOM and wedge "
                f"the host (issue #23). Run state through the last completed chunk has been "
                f"persisted -- re-run with --resume once the card is free again."
            )
            logger.error("%s", message)
            raise VramBelowFloorError(message)

    # -- pre-flight disk check --------------------------------------------------- #

    def _preflight_disk_check(self, output_dir: Path) -> None:
        check_path = _first_existing_ancestor(Path(output_dir))
        try:
            usage = self.disk_usage(str(check_path))
        except OSError as exc:
            logger.exception("Pre-flight disk check could not stat %s", check_path)
            raise DiskPreflightError(
                f"could not determine free disk space at {check_path}: {exc}"
            ) from exc

        free_gb = usage.free / (1024**3)
        if free_gb < self.min_free_disk_gb:
            logger.error(
                "Pre-flight disk check FAILED: only %.2f GB free at %s, need >= %.2f GB -- "
                "refusing to start the run",
                free_gb,
                check_path,
                self.min_free_disk_gb,
            )
            raise DiskPreflightError(
                f"only {free_gb:.2f} GB free at {check_path} "
                f"(need >= {self.min_free_disk_gb:.2f} GB)"
            )
        logger.info(
            "Pre-flight disk check OK: %.2f GB free at %s (>= %.2f GB required)",
            free_gb,
            check_path,
            self.min_free_disk_gb,
        )


def _describe_changes(
    stored: ChunkFingerprint, expected: ChunkFingerprint, fields: Sequence[str]
) -> str:
    """``"start 0.0 -> 3.25, end 6.0 -> 9.25"`` -- name the field *and* both
    values, so the log says what actually changed rather than just that
    something did."""
    return ", ".join(f"{f} {getattr(stored, f)!r} -> {getattr(expected, f)!r}" for f in fields)


def _describe_first_change(
    run_state: RunState, fingerprints: Mapping[int, ChunkFingerprint], chunk_id: int
) -> str:
    stored = run_state.results[chunk_id].fingerprint
    if stored is None:
        return "no fingerprint recorded"
    expected = fingerprints[chunk_id]
    return _describe_changes(stored, expected, expected.timeline_differences(stored))


def _store_result(run_state: RunState, chunk_id: int, result: ChunkResult) -> None:
    """Write ``result`` into ``run_state.results[chunk_id]``, guarding that
    the dict key and ``ChunkResult.chunk_id`` agree (index-space guard)."""
    _assert_chunk_id(chunk_id, result.chunk_id, context="RunState.results write")
    run_state.results[chunk_id] = result


__all__ = [
    "RUN_STATE_SCHEMA_VERSION",
    "ChunkIdMismatchError",
    "DiskPreflightError",
    "DiskUsage",
    "ResilienceError",
    "ResilientRunner",
    "RunStateSchemaError",
    "Sleeper",
    "VramBelowFloorError",
    "VramProbe",
    "WorkflowProvider",
]
