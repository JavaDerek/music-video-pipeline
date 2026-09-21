"""Stage 5: final assembly (issue #11).

Stitches the chronological chunk MP4s into one silent video via the FFmpeg
concat demuxer, then muxes the *original* master audio track over it.

Two non-negotiable invariants (see project ``CLAUDE.md``):

1. **Concat without re-encoding.** ``-c:v copy`` on both ffmpeg calls --
   re-encoding degrades quality and wastes hours on a long render.
2. **The master track is the only audio in the final video.** Generated
   per-chunk audio (even from H3's own audio VAE) is discarded entirely;
   the pristine master track -- the exact file used for Stage 1 alignment --
   is muxed in. Because chunk boundaries came from the forced-alignment
   timeline, this gives zero-generative-drift sync as long as every expected
   chunk is present, which is why gaps are validated *before* concat runs.

Nothing here imports ComfyUI, torch, or pydub -- only ``contracts``,
``luminance``/``scenecuts`` (themselves contracts-and-stdlib-only) and the
stdlib. The subprocess runner is injectable so unit tests can assert on the
exact argument lists without ever invoking a real ``ffmpeg`` binary.

**Post-render darkness check (issue #77).** Before concat runs, every
available chunk's video is sampled for its ending luminance via
:func:`music_video_maker.luminance.check_dark_chunks`, reusing the same
injected ``runner`` -- so it activates automatically wherever assembly
already runs against real ffmpeg, with nothing new to wire up. A flagged
chunk is logged loudly and carried on :class:`AssemblyResult` for a caller to
report; it never blocks assembly, the same asymmetric-warning discipline
every lint in ``shot_plan.py`` follows. The whole check is wrapped in its own
try/except here too, on top of ``luminance``'s own internal guards -- a
Stage-5 smell test must never be the reason a finished render doesn't get
written.

**Post-render scene-cut check (issue #81), same shape.** Also before concat,
every available chunk is scanned for a cut *inside* what was authored as one
continuous take, via :func:`music_video_maker.scenecuts.check_scene_cuts` --
same injected-runner reuse, same never-raises internal guards, same second
try/except here, same "logged loudly, never blocks" discipline. See
``scenecuts.py``'s module docstring for the two-corpus measurement behind
its default threshold.

A flagged chunk here is also a chained-path hazard the check does not
detect: on the chained I2V path (project ``CLAUDE.md``, "the seed frame IS
the identity conditioning"), a chunk that cut to a different scene hands the
*next* chunk a final frame from a shot nobody authored. Wiring the actual
``i2v_chain_scope`` lookup into this module to detect that mechanically was
judged not worth the coupling for this issue; the ERROR log line below names
the hazard in text instead, so it reaches whoever reads the log without this
module needing to know anything about continuity.
**Silent output for concert mode (issue #22).** Invariant 2 above inverts
for a rear-projection backdrop: the band playing live *is* the audio, so
shipping a file with an audio stream at all risks double-audio if someone's
playback rig un-mutes it. Passing ``master_audio=None`` produces a video
with **no audio stream whatsoever** -- the concat pass (still ``-an``, so
generated per-chunk audio is discarded exactly as before) writes straight to
the final output path and no mux pass runs at all. This is a *conditional*
suspension of invariant 2 for one mode, not a repeal of it: every caller that
still passes a real ``master_audio`` path gets byte-for-byte the same two
ffmpeg calls as before this existed. Invariant 1 (no re-encoding) and the
"generated audio is always discarded" rule are untouched either way.

**Measured duration check, also issue #22.** For a concert backdrop, drifting
against the click track is a show falling apart live with no chance to
correct it -- unlike a music video, where being 0.75 s short is merely
abrupt. ``expected_duration`` is opt-in (default ``None``, no behaviour
change for any existing caller); when given, the *finished* file's container
duration is probed with ffprobe through the same injected ``runner`` and
compared with tolerance. A mismatch raises :class:`DurationMismatchError`
*after* the file is already written, deliberately: the file stays on disk to
inspect, and an operator running a show needs to be told loudly, not have it
buried in a log line read the next day.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState
from music_video_maker.luminance import DEFAULT_DARK_FLOOR, DarkChunkWarning, check_dark_chunks
from music_video_maker.scenecuts import DEFAULT_SCENE_THRESHOLD, SceneCutWarning
from music_video_maker.scenecuts import check_scene_cuts as _run_scene_cut_check
from music_video_maker.timelines import TimelinePlacement, log_placements, place_timelines

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "final_video.mp4"
CONCAT_LIST_FILENAME = "concat_list.txt"
INTERMEDIATE_VIDEO_FILENAME = "_concat_intermediate.mp4"

DEFAULT_DURATION_TOLERANCE_SECONDS = 0.05
"""Issue #22: a starting point, not a measured figure -- roughly one frame at
24 fps (0.0417 s) rounded up. A real playback rig's tolerance should be set
from the actual show's frame rate and sync requirements, not left at this."""

# Injectable subprocess seam: unit tests supply a fake that never touches a
# real shell; only the (skippable) integration test exercises the default.
SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real ffmpeg invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False)


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg subprocess exits non-zero.

    Carries the full command and captured stderr so the caller (and the log
    line emitted before this is raised) has everything needed to diagnose
    without re-running anything.
    """

    def __init__(self, message: str, *, cmd: Sequence[str], returncode: int, stderr: bytes | str):
        self.cmd = tuple(cmd)
        self.returncode = returncode
        self.stderr = (
            stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
        )
        super().__init__(
            f"{message} (exit={returncode}): {' '.join(self.cmd)}\nstderr:\n{self.stderr}"
        )


class DurationMismatchError(RuntimeError):
    """Raised when a finished video's measured duration drifts beyond tolerance.

    Issue #22: for a concert backdrop, duration accuracy is the acceptance
    criterion, not cosmetic -- a video that drifts against the click track
    it's cut to is a show falling apart live, with no opportunity to correct
    it once the band is playing. Raised *after* the output file is already
    written, deliberately: the file stays on disk to inspect, and an
    operator running a show needs to be told loudly, not have this buried in
    a log line they read tomorrow.
    """

    def __init__(
        self,
        video_path: Path,
        expected_duration: float,
        measured_duration: float,
        tolerance: float,
    ):
        self.video_path = Path(video_path)
        self.expected_duration = expected_duration
        self.measured_duration = measured_duration
        self.tolerance = tolerance
        self.drift = measured_duration - expected_duration
        super().__init__(
            f"{self.video_path}: measured duration {measured_duration:.3f}s does not match "
            f"expected {expected_duration:.3f}s (drift={self.drift:+.3f}s, "
            f"tolerance={tolerance:.3f}s)"
        )


class MissingChunksError(RuntimeError):
    """Raised when the expected chunk set has gaps -- assembly refuses to run.

    Concatenating a video with a chunk silently missing would desync every
    subsequent chunk against the master audio's absolute timeline, which is
    exactly the drift Stage 5 exists to prevent. So any missing or
    dead-lettered chunk aborts assembly loudly rather than producing a
    misaligned video.
    """

    def __init__(
        self,
        missing_chunk_ids: Sequence[int],
        dead_lettered_chunk_ids: Sequence[int],
        *,
        timeline: str | None = None,
    ):
        self.missing_chunk_ids = tuple(missing_chunk_ids)
        self.dead_lettered_chunk_ids = tuple(dead_lettered_chunk_ids)
        self.timeline = timeline
        # Issue #66: chunk ids are a separate space per timeline, so "chunk 3
        # is missing" is ambiguous the moment there are two of them. Named
        # only when there IS a second timeline, so the single-timeline
        # message is unchanged.
        where = f" on timeline {timeline!r}" if timeline else ""
        super().__init__(
            f"cannot assemble final video: expected chunk(s) unavailable{where} "
            f"(missing={self.missing_chunk_ids}, dead_lettered={self.dead_lettered_chunk_ids})"
        )


@dataclass(frozen=True)
class AssemblyResult:
    """Everything produced by one :func:`assemble_final_video` run."""

    output_video: Path
    concat_file: Path
    intermediate_video: Path
    """The concat pass's own output. Equal to ``output_video`` when there is
    no mux pass (``master_audio=None``, issue #22 concert mode) -- there is
    no second pass to feed, so the concat pass writes the deliverable
    directly and the two are the same file."""
    chunk_ids: tuple[int, ...]
    """Chunk ids actually concatenated, in the chronological order used."""
    concat_args: tuple[str, ...]
    mux_args: tuple[str, ...]
    """Empty when ``master_audio=None`` -- no mux pass ran (issue #22)."""
    dark_chunk_warnings: tuple[DarkChunkWarning, ...] = ()
    """Chunks whose sampled ending luminance fell below the darkness floor
    (issue #77) -- informational only. The video was still assembled; a
    non-empty tuple here means a human should look at these chunks before
    trusting the final cut, not that anything failed."""
    scene_cut_warnings: tuple[SceneCutWarning, ...] = ()
    """Chunks containing at least one scene cut inside what was authored as
    a single continuous take (issue #81) -- informational only, same as
    :attr:`dark_chunk_warnings`. A non-empty tuple here means a human should
    look at these chunks -- and, on the chained I2V path, that the *next*
    chunk's seed frame may not depict what its prompt describes -- not that
    anything failed."""
    has_audio: bool = True
    """What the output actually contains, so a caller never has to re-derive
    it from whether ``mux_args`` is empty. ``False`` only for the issue #22
    silent-output path (``master_audio=None``)."""
    measured_duration: float | None = None
    """The probed container duration of the finished file (issue #22), or
    ``None`` when no duration check was requested (``expected_duration`` not
    given)."""


def _escape_concat_path(path: Path) -> str:
    """Escape a path for the ffmpeg concat demuxer's single-quoted directive.

    The concat demuxer's ``file '...'`` directive is a single-quoted token.
    A literal single quote inside the path must close the quote, emit an
    escaped quote, and reopen: ``it's.mp4`` -> ``it'\\''s.mp4``. Backslashes
    and other characters need no special handling inside single quotes.
    """
    return str(path).replace("'", "'\\''")


def write_concat_file(video_paths: Sequence[Path], dest: Path) -> Path:
    """Write an ffmpeg concat demuxer list file with one ``file '...'`` line
    per path, in the given order, as absolute paths."""
    lines = [f"file '{_escape_concat_path(Path(p).resolve())}'" for p in video_paths]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info("Wrote concat demuxer file %s with %d entries", dest, len(video_paths))
    return dest


def build_concat_args(concat_file: Path, output_path: Path) -> list[str]:
    """Args for the no-re-encode concat pass. ``-an`` explicitly strips any
    audio the individual chunk MP4s carry -- only the master track survives
    into the final mux."""
    return [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-an",
        "-c:v",
        "copy",
        str(output_path),
    ]


def build_mux_args(video_path: Path, master_audio: Path, output_path: Path) -> list[str]:
    """Args for muxing the pristine master track over the concatenated,
    now-silent video. ``-map 0:v -map 1:a`` guarantees the only audio stream
    in the output originates from ``master_audio``, never from the video
    input."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(master_audio),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-shortest",
        str(output_path),
    ]


def validate_chunk_availability(
    expected_chunk_ids: Sequence[int],
    results: Mapping[int, ChunkResult] | RunState,
) -> tuple[list[int], list[int], list[int]]:
    """Cross-reference the expected chunk ids against Stage 4 results.

    ``results`` may be a plain ``{chunk_id: ChunkResult}`` mapping or a
    :class:`RunState` (issue #10's resilience state) -- either conveys
    dead-lettered status via ``ChunkResult.status``, so nothing from a
    not-yet-existing dead-letter-queue module needs importing.

    Returns ``(available_ids, missing_ids, dead_lettered_ids)``, each sorted
    ascending. ``available`` means a succeeded :class:`ChunkResult` exists
    with a non-``None`` ``video_file``.
    """
    result_map: Mapping[int, ChunkResult] = (
        results.results if isinstance(results, RunState) else results
    )

    available: list[int] = []
    missing: list[int] = []
    dead: list[int] = []

    for cid in sorted(set(expected_chunk_ids)):
        result = result_map.get(cid)
        if result is None:
            missing.append(cid)
        elif result.status is ChunkStatus.DEAD_LETTERED:
            dead.append(cid)
        elif result.succeeded and result.video_file is not None:
            available.append(cid)
        else:
            missing.append(cid)

    return available, missing, dead


def build_duration_probe_args(video_path: Path) -> list[str]:
    """Args for probing a finished file's *container* duration via ffprobe.

    ``format=duration`` rather than a stream duration: the container
    duration is what a playback rig will actually honour, and it's what a
    ``-c:v copy`` concat pass produces (issue #22)."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def probe_duration_seconds(video_path: Path, runner: SubprocessRunner) -> float:
    """Measure a finished file's actual container duration via ffprobe.

    Issue #22: for a concert backdrop the acceptance criterion is *measured*
    duration against the click track, not the sum of chunk lengths computed
    upstream -- a ``-c:v copy`` concat is exact per chunk, but the container
    duration a playback rig actually honours is the only thing worth
    checking against.

    Raises :class:`FfmpegError` (logged first) if ffprobe exits non-zero or
    its stdout isn't parseable as a float.
    """
    video_path = Path(video_path)
    args = build_duration_probe_args(video_path)
    result = runner(args)
    if result.returncode != 0:
        stderr = _decode(result.stderr)
        logger.error(
            "ffprobe duration probe failed for %s (exit=%s): %s",
            video_path,
            result.returncode,
            stderr,
        )
        raise FfmpegError(
            "ffprobe duration probe failed", cmd=args, returncode=result.returncode, stderr=stderr
        )

    stdout = _decode(result.stdout).strip()
    try:
        return float(stdout)
    except ValueError as exc:
        logger.error(
            "ffprobe returned an unparseable duration for %s: %r", video_path, stdout
        )
        raise FfmpegError(
            f"ffprobe returned an unparseable duration for {video_path}: {stdout!r}",
            cmd=args,
            returncode=result.returncode,
            stderr=stdout,
        ) from exc


def _run_ffmpeg(args: Sequence[str], runner: SubprocessRunner, *, step: str) -> None:
    logger.info("Running ffmpeg step=%s: %s", step, " ".join(args))
    result = runner(args)
    if result.returncode != 0:
        stderr = result.stderr if result.stderr is not None else b""
        stderr_text = (
            stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
        )
        logger.error(
            "ffmpeg step=%s failed (exit=%s) cmd=%s stderr=%s",
            step,
            result.returncode,
            args,
            stderr_text,
        )
        raise FfmpegError(
            f"ffmpeg {step} step failed", cmd=args, returncode=result.returncode, stderr=stderr_text
        )


def assemble_final_video(
    chunks: Sequence[AudioChunk],
    results: Mapping[int, ChunkResult] | RunState,
    master_audio: Path | None,
    output_dir: Path,
    *,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    runner: SubprocessRunner | None = None,
    check_luminance: bool = True,
    luminance_floor: float = DEFAULT_DARK_FLOOR,
    luminance_runner: SubprocessRunner | None = None,
    check_scene_cuts: bool = True,
    scene_cut_threshold: float = DEFAULT_SCENE_THRESHOLD,
    scene_cut_runner: SubprocessRunner | None = None,
    expected_duration: float | None = None,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
) -> AssemblyResult:
    """Concat every chunk's rendered video (chronological ``chunk_id`` order)
    and mux the master audio track over it.

    Raises :class:`MissingChunksError` -- before any subprocess runs -- if
    any chunk in ``chunks`` lacks a succeeded, video-bearing result. Raises
    :class:`FfmpegError` if any ffmpeg/ffprobe subprocess exits non-zero.

    ``check_luminance`` (default ``True``) runs the issue #77 darkness check
    against every available chunk before concat, using ``luminance_runner``
    if given or ``runner`` otherwise -- so passing the same fake/real ffmpeg
    runner already used for concat/mux is enough to exercise or disable it in
    tests. The check never raises and never blocks assembly; see
    :func:`music_video_maker.luminance.check_dark_chunks`.

    ``check_scene_cuts`` (default ``True``) runs the issue #81 scene-cut
    check the same way, using ``scene_cut_runner`` if given or ``runner``
    otherwise. Also never raises and never blocks assembly; see
    :func:`music_video_maker.scenecuts.check_scene_cuts`.
    ``master_audio=None`` (issue #22 concert mode) produces a video with no
    audio stream at all: the concat pass writes straight to the final output
    path (there's no mux pass to feed) and :attr:`AssemblyResult.mux_args` is
    ``()``. This is a deliberate, per-run suspension of the "master track is
    the only audio" invariant -- a WARNING names it so a run that hits this
    by accident is discoverable in the log. ``-an`` on the concat pass still
    strips the chunks' own generated audio either way; "generated audio is
    always discarded" is untouched by this parameter.

    ``expected_duration`` (default ``None``) is an opt-in, *measured* check
    (issue #22): when given, ffprobe measures the finished file's real
    container duration and raises :class:`DurationMismatchError` if it drifts
    from ``expected_duration`` by more than ``duration_tolerance_seconds``
    (default 0.05 s -- roughly one frame at 24 fps, a starting point rather
    than a measured figure; set it from the real playback rig). The raise
    happens *after* the file is written, on purpose, so the file remains on
    disk for inspection. When ``expected_duration`` is ``None`` (every
    existing caller), no probe runs and :attr:`AssemblyResult.measured_duration`
    stays ``None``.
    """
    runner = runner or _default_runner
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = [c.chunk_id for c in chunks]
    available_ids, missing_ids, dead_ids = validate_chunk_availability(expected_ids, results)

    if missing_ids or dead_ids:
        logger.error(
            "Refusing to assemble final video: %d/%d chunks available, missing=%s dead_lettered=%s",
            len(available_ids),
            len(expected_ids),
            missing_ids,
            dead_ids,
        )
        raise MissingChunksError(missing_ids, dead_ids)

    result_map: Mapping[int, ChunkResult] = (
        results.results if isinstance(results, RunState) else results
    )
    ordered_ids = sorted(available_ids)
    video_paths = [result_map[cid].video_file for cid in ordered_ids]

    dark_chunk_warnings: tuple[DarkChunkWarning, ...] = ()
    if check_luminance:
        try:
            dark_chunk_warnings = check_dark_chunks(
                [c for c in chunks if c.chunk_id in available_ids],
                result_map,
                floor=luminance_floor,
                runner=luminance_runner or runner,
            )
        except Exception:  # noqa: BLE001 - a smell test must never block assembly.
            logger.exception(
                "The issue #77 darkness check raised unexpectedly -- skipping it and "
                "continuing with assembly. This must never abort a run."
            )
            dark_chunk_warnings = ()
        if dark_chunk_warnings:
            logger.error(
                "Assembly: %d chunk(s) end below the darkness floor (Y<%.1f) -- a human "
                "should check these before trusting the final video: %s",
                len(dark_chunk_warnings),
                luminance_floor,
                [w.chunk_id for w in dark_chunk_warnings],
            )

    scene_cut_warnings: tuple[SceneCutWarning, ...] = ()
    if check_scene_cuts:
        try:
            scene_cut_warnings = _run_scene_cut_check(
                [c for c in chunks if c.chunk_id in available_ids],
                result_map,
                threshold=scene_cut_threshold,
                runner=scene_cut_runner or runner,
            )
        except Exception:  # noqa: BLE001 - a smell test must never block assembly.
            logger.exception(
                "The issue #81 scene-cut check raised unexpectedly -- skipping it and "
                "continuing with assembly. This must never abort a run."
            )
            scene_cut_warnings = ()
        if scene_cut_warnings:
            logger.error(
                "Assembly: %d chunk(s) contain a scene cut inside a single take "
                "(scene>%.2f) -- a human should check these before trusting the final "
                "video, and a flagged chunk's final frame is an unauthored seed on the "
                "chained path (its identity conditioning for the next chunk may not "
                "depict what that chunk's prompt describes): %s",
                len(scene_cut_warnings),
                scene_cut_threshold,
                [w.chunk_id for w in scene_cut_warnings],
            )

    concat_file = output_dir / CONCAT_LIST_FILENAME
    write_concat_file(video_paths, concat_file)  # type: ignore[arg-type]

    output_path = output_dir / output_filename

    if master_audio is None:
        # Issue #22 concert mode: the band playing live is the audio, so
        # shipping a file WITH an audio stream risks double-audio if
        # someone's playback rig un-mutes it. There is no second (mux) pass
        # to feed, so the concat pass writes the deliverable directly --
        # copying the file again for nothing is a real cost on an
        # hours-long render.
        logger.warning(
            "master_audio=None: assembling %s as a SILENT video with NO AUDIO STREAM "
            "at all -- this suspends the CLAUDE.md invariant 'the master audio track "
            "is the only audio in the final video' for this run (issue #22 concert "
            "mode: a rear-projection backdrop's audio is the live band, not this "
            "file). If this run was meant to have audio, master_audio was passed as "
            "None by mistake.",
            output_path,
        )
        intermediate_video = output_path
        concat_args = build_concat_args(concat_file, output_path)
        _run_ffmpeg(concat_args, runner, step="concat")
        mux_args: tuple[str, ...] = ()
        has_audio = False
    else:
        intermediate_video = output_dir / INTERMEDIATE_VIDEO_FILENAME
        concat_args = build_concat_args(concat_file, intermediate_video)
        _run_ffmpeg(concat_args, runner, step="concat")

        mux_args_list = build_mux_args(intermediate_video, master_audio, output_path)
        _run_ffmpeg(mux_args_list, runner, step="mux")
        mux_args = tuple(mux_args_list)
        has_audio = True

    logger.info(
        "Assembled final video at %s from %d chunks (ids=%s) has_audio=%s",
        output_path,
        len(ordered_ids),
        ordered_ids,
        has_audio,
    )

    measured_duration: float | None = None
    if expected_duration is not None:
        measured_duration = probe_duration_seconds(output_path, runner)
        drift = measured_duration - expected_duration
        if abs(drift) > duration_tolerance_seconds:
            logger.error(
                "Duration mismatch for %s: measured=%.3fs expected=%.3fs drift=%+.3fs "
                "tolerance=%.3fs -- issue #22: for a concert backdrop this is a show "
                "falling apart against the click track, so this raises even though "
                "the file is already written and stays on disk for inspection.",
                output_path,
                measured_duration,
                expected_duration,
                drift,
                duration_tolerance_seconds,
            )
            raise DurationMismatchError(
                output_path, expected_duration, measured_duration, duration_tolerance_seconds
            )

    return AssemblyResult(
        output_video=output_path,
        concat_file=concat_file,
        intermediate_video=intermediate_video,
        chunk_ids=tuple(ordered_ids),
        concat_args=tuple(concat_args),
        mux_args=mux_args,
        dark_chunk_warnings=dark_chunk_warnings,
        scene_cut_warnings=scene_cut_warnings,
        has_audio=has_audio,
        measured_duration=measured_duration,
    )


# --------------------------------------------------------------------------- #
# Issue #66: joining two timelines
# --------------------------------------------------------------------------- #

VIDEO_SIGNATURE_ENTRIES: tuple[str, ...] = (
    "codec_name",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "field_order",
    "r_frame_rate",
    "time_base",
)
"""Every video-stream property the ``-c:v copy`` concat demuxer preserves and
does not check (issue #66).

The demuxer does not fail on mismatched inputs -- it produces a file that
plays wrong. Measured on the real "Deathless" render, all 80 chunks share
exactly one signature::

    h264, profile High, level 30, 864x480, yuv420p, progressive, 24/1 fps,
    time_base 1/12288

Only ``width`` and ``height`` are predictable from config
(``render_width``/``render_height``). Codec, profile, level, pixel format,
field order, frame rate and time base come out of the workflow template, so
they are **probed from the first chunk of each timeline and compared**, never
assumed. One timeline cannot disagree with itself in practice (same template,
same run); two timelines rendered days apart through an edited template can,
and that is the case this exists for."""

AUDIO_LAYOUT_ENTRIES: tuple[str, ...] = ("sample_rate", "channels")
"""What the seam's audio concat has to agree on. See
:func:`build_audio_pad_args` for why the *song's* layout is the target and
the segments are the ones resampled."""

TIMELINE_VIDEO_TEMPLATE = "_timeline_{name}.mp4"
TIMELINE_AUDIO_TEMPLATE = "_timeline_{name}.wav"
TIMELINE_CONCAT_LIST_TEMPLATE = "concat_list_{name}.txt"
COMBINED_AUDIO_FILENAME = "_seam_audio.wav"


class ConcatSignatureMismatchError(RuntimeError):
    """Two timelines' chunks cannot be concatenated with ``-c:v copy``.

    Raised before any concat runs, because the failure it prevents is not a
    crash: the demuxer accepts mismatched inputs and writes a file that plays
    wrong -- wrong speed, wrong colours, or a second half that does not
    decode at all. There is no ``-shortest``-style accident that saves this
    one, and the first symptom is watching the finished video."""

    def __init__(self, signatures: Mapping[str, Mapping[str, str]], differing: Sequence[str]):
        self.signatures = {name: dict(sig) for name, sig in signatures.items()}
        self.differing = tuple(differing)
        detail = "; ".join(
            f"{name}: " + ", ".join(f"{key}={sig.get(key)!r}" for key in self.differing)
            for name, sig in self.signatures.items()
        )
        super().__init__(
            "cannot concatenate timelines without re-encoding: their chunks disagree on "
            f"{', '.join(self.differing)} -- {detail}. The concat demuxer does not fail on "
            "mismatched inputs, it produces a file that plays wrong, so this refuses "
            "instead (issue #66). Re-render the odd timeline through the same workflow "
            "template and resolution as the others."
        )


class SeamMismatchError(RuntimeError):
    """A timeline's reconciled audio does not match the video rendered for it.

    The seam is closed by padding each timeline's audio up to its own
    *measured* video duration. This is the check that the padding actually
    happened: ffprobe reads the padded file back and compares. It exists
    because the failure it catches is invisible -- at a seam, a few frames of
    drift desyncs everything after it while each half still looks perfect in
    isolation, and ``-shortest`` has nothing to bite on in the middle of a
    file."""

    def __init__(self, name: str, expected: float, measured: float, tolerance: float):
        self.name = name
        self.expected = expected
        self.measured = measured
        self.tolerance = tolerance
        self.drift = measured - expected
        super().__init__(
            f"seam for timeline {name!r}: its reconciled audio measures {measured:.3f}s "
            f"against {expected:.3f}s of rendered video (drift={self.drift:+.3f}s, "
            f"tolerance={tolerance:.3f}s). Everything after this seam would play "
            f"{self.drift:+.3f}s out of sync while both halves looked correct on their own."
        )


@dataclass(frozen=True)
class TimelineAssembly:
    """One timeline's contribution to a multi-timeline assembly (issue #66)."""

    name: str
    chunks: Sequence[AudioChunk]
    results: Mapping[int, ChunkResult] | RunState
    audio: Path | None
    """This timeline's own audio: the segment's recording, or the song's
    master track. ``None`` only on the silent path (issue #22), where there
    is no audio anywhere in the output."""
    fingerprint_name: str | None = None
    """What :attr:`~music_video_maker.contracts.ChunkFingerprint.timeline`
    records for this timeline -- ``None`` for the song. Carried here purely
    so the assembly report names timelines the same way the run state does."""


@dataclass(frozen=True)
class MultiTimelineAssemblyResult:
    """Everything one :func:`assemble_timelines` run produced."""

    output_video: Path
    concat_file: Path
    intermediate_video: Path
    timeline_videos: tuple[Path, ...]
    placements: tuple[TimelinePlacement, ...]
    """Where each timeline landed and what its seam cost, from *measured*
    durations (issue #22's rule: measured, never computed)."""
    chunk_ids: tuple[tuple[str, tuple[int, ...]], ...]
    """``(timeline name, chunk ids)`` per timeline, in playback order."""
    concat_args: tuple[str, ...]
    mux_args: tuple[str, ...]
    dark_chunk_warnings: tuple[DarkChunkWarning, ...] = ()
    scene_cut_warnings: tuple[SceneCutWarning, ...] = ()
    has_audio: bool = True
    measured_duration: float | None = None
    video_signature: tuple[tuple[str, str], ...] = ()
    """The signature every timeline agreed on, as sorted key/value pairs --
    evidence that the check ran, not just that it did not raise."""


def build_stream_signature_args(video_path: Path) -> list[str]:
    """Args probing the video-stream properties ``-c:v copy`` preserves."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=" + ",".join(VIDEO_SIGNATURE_ENTRIES),
        "-of",
        "default=nw=1",
        str(video_path),
    ]


def build_audio_layout_args(audio_path: Path) -> list[str]:
    """Args probing an audio file's sample rate and channel count."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=" + ",".join(AUDIO_LAYOUT_ENTRIES),
        "-of",
        "default=nw=1",
        str(audio_path),
    ]


def _probe_key_values(
    args: Sequence[str], runner: SubprocessRunner, *, what: str, path: Path
) -> dict[str, str]:
    result = runner(args)
    if result.returncode != 0:
        stderr = _decode(result.stderr)
        logger.error(
            "ffprobe %s failed for %s (exit=%s): %s", what, path, result.returncode, stderr
        )
        raise FfmpegError(
            f"ffprobe {what} failed", cmd=args, returncode=result.returncode, stderr=stderr
        )
    values: dict[str, str] = {}
    for line in _decode(result.stdout).splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def probe_video_signature(video_path: Path, runner: SubprocessRunner) -> dict[str, str]:
    """Read the concat-relevant video properties off one rendered chunk."""
    video_path = Path(video_path)
    return _probe_key_values(
        build_stream_signature_args(video_path),
        runner,
        what="stream signature probe",
        path=video_path,
    )


def probe_audio_layout(audio_path: Path, runner: SubprocessRunner) -> dict[str, str]:
    """Read an audio file's sample rate and channel count."""
    audio_path = Path(audio_path)
    return _probe_key_values(
        build_audio_layout_args(audio_path),
        runner,
        what="audio layout probe",
        path=audio_path,
    )


def compare_video_signatures(signatures: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    """Raise :class:`ConcatSignatureMismatchError` if any two timelines'
    signatures disagree on a property ``-c:v copy`` preserves; return the
    agreed signature otherwise.

    Only keys ffprobe actually reported are compared, and a key missing from
    *every* timeline simply never differs -- an older ffmpeg that omits one
    (``field_order`` is the usual absentee) must not be read as a mismatch.
    """
    names = list(signatures)
    if not names:
        return {}
    reference = dict(signatures[names[0]])
    differing = [
        key
        for key in VIDEO_SIGNATURE_ENTRIES
        if any(sig.get(key) != reference.get(key) for sig in signatures.values())
    ]
    if differing:
        logger.error(
            "Timelines disagree on %s -- refusing to concat without re-encoding: %s",
            ", ".join(differing),
            {name: dict(sig) for name, sig in signatures.items()},
        )
        raise ConcatSignatureMismatchError(signatures, differing)
    return reference


def build_audio_pad_args(
    audio_path: Path,
    output_path: Path,
    *,
    pad_to_seconds: float,
    sample_rate: str | None,
    channels: str | None,
) -> list[str]:
    """Args padding one timeline's audio with silence up to its own video's
    measured duration (issue #66's seam reconciliation).

    ``apad=whole_dur`` pads and never trims, which is the asymmetry this
    reconciliation wants: audio *short* of its video is the case the chunk
    timeline's overshoot creates (1.837 s on "Deathless", where H3's
    124-frame floor padded the last tile past the end of the song), and audio
    *longer* than its video is a defect
    :class:`~music_video_maker.timelines.SeamOverrunError` refuses rather
    than silently cutting somebody off mid-sentence.

    The alternative reconciliation -- trimming the last chunk's video down to
    the audio -- is a re-encode or a keyframe cut, and ``-c:v copy`` is the
    invariant this project protects hardest. So: pad, and log the amount.

    ``sample_rate``/``channels`` are the **song's** layout, probed from the
    master track: every timeline is resampled to the song's rate rather than
    the song being resampled to anything, because the master is the one file
    this project refuses to degrade. The concat filter that joins these
    afterwards requires its inputs to agree, and a silent mismatch there
    produces garbled audio rather than an error.
    """
    args = ["ffmpeg", "-y", "-i", str(audio_path), "-af", f"apad=whole_dur={pad_to_seconds:.6f}"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if channels:
        args += ["-ac", str(channels)]
    args += ["-c:a", "pcm_s16le", str(output_path)]
    return args


def build_audio_concat_args(audio_paths: Sequence[Path], output_path: Path) -> list[str]:
    """Args joining the reconciled per-timeline audio into one track.

    The ``concat`` *filter* rather than the concat *demuxer*: the demuxer
    copies packets and would splice two differently-encoded files into
    something no decoder can read, while the filter works on decoded samples
    and produces one continuous PCM stream. Its inputs must already agree on
    sample rate and channel count, which :func:`build_audio_pad_args` is what
    guarantees.
    """
    args = ["ffmpeg", "-y"]
    for path in audio_paths:
        args += ["-i", str(path)]
    streams = "".join(f"[{i}:a]" for i in range(len(audio_paths)))
    args += [
        "-filter_complex",
        f"{streams}concat=n={len(audio_paths)}:v=0:a=1[seam]",
        "-map",
        "[seam]",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    return args


def _timeline_video_paths(timeline: TimelineAssembly) -> tuple[list[Path], list[int]]:
    """Validate one timeline's chunk availability and return its clips in
    chronological order. Raises :class:`MissingChunksError` naming the
    timeline -- chunk ids are a separate space per timeline, so a bare id is
    ambiguous the moment there are two."""
    expected_ids = [c.chunk_id for c in timeline.chunks]
    available, missing, dead = validate_chunk_availability(expected_ids, timeline.results)
    if missing or dead:
        logger.error(
            "Refusing to assemble: timeline %r has %d/%d chunks available, missing=%s "
            "dead_lettered=%s",
            timeline.name,
            len(available),
            len(expected_ids),
            missing,
            dead,
        )
        raise MissingChunksError(missing, dead, timeline=timeline.name)
    result_map: Mapping[int, ChunkResult] = (
        timeline.results.results
        if isinstance(timeline.results, RunState)
        else timeline.results
    )
    ordered = sorted(available)
    return [result_map[cid].video_file for cid in ordered], ordered  # type: ignore[misc]


def _run_chunk_checks(
    timeline: TimelineAssembly,
    available_ids: Sequence[int],
    *,
    check_luminance: bool,
    luminance_floor: float,
    luminance_runner: SubprocessRunner,
    check_scene_cuts: bool,
    scene_cut_threshold: float,
    scene_cut_runner: SubprocessRunner,
) -> tuple[tuple[DarkChunkWarning, ...], tuple[SceneCutWarning, ...]]:
    """The #77 darkness and #81 scene-cut checks for one timeline, with the
    same never-blocks-assembly discipline the single-timeline path uses: a
    smell test must never be the reason a finished render is not written."""
    result_map: Mapping[int, ChunkResult] = (
        timeline.results.results
        if isinstance(timeline.results, RunState)
        else timeline.results
    )
    chunks = [c for c in timeline.chunks if c.chunk_id in set(available_ids)]

    dark: tuple[DarkChunkWarning, ...] = ()
    if check_luminance:
        try:
            dark = check_dark_chunks(
                chunks, result_map, floor=luminance_floor, runner=luminance_runner
            )
        except Exception:  # noqa: BLE001 - a smell test must never block assembly.
            logger.exception(
                "The issue #77 darkness check raised unexpectedly on timeline %r -- "
                "skipping it and continuing with assembly.",
                timeline.name,
            )
            dark = ()

    cuts: tuple[SceneCutWarning, ...] = ()
    if check_scene_cuts:
        try:
            cuts = _run_scene_cut_check(
                chunks, result_map, threshold=scene_cut_threshold, runner=scene_cut_runner
            )
        except Exception:  # noqa: BLE001 - a smell test must never block assembly.
            logger.exception(
                "The issue #81 scene-cut check raised unexpectedly on timeline %r -- "
                "skipping it and continuing with assembly.",
                timeline.name,
            )
            cuts = ()

    if dark:
        logger.error(
            "Assembly: timeline %r has %d chunk(s) ending below the darkness floor "
            "(Y<%.1f): %s",
            timeline.name,
            len(dark),
            luminance_floor,
            [w.chunk_id for w in dark],
        )
    if cuts:
        logger.error(
            "Assembly: timeline %r has %d chunk(s) containing a scene cut inside a single "
            "take (scene>%.2f): %s",
            timeline.name,
            len(cuts),
            scene_cut_threshold,
            [w.chunk_id for w in cuts],
        )
    return dark, cuts


def assemble_timelines(
    timelines: Sequence[TimelineAssembly],
    output_dir: Path,
    *,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    runner: SubprocessRunner | None = None,
    check_luminance: bool = True,
    luminance_floor: float = DEFAULT_DARK_FLOOR,
    luminance_runner: SubprocessRunner | None = None,
    check_scene_cuts: bool = True,
    scene_cut_threshold: float = DEFAULT_SCENE_THRESHOLD,
    scene_cut_runner: SubprocessRunner | None = None,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
) -> MultiTimelineAssemblyResult:
    """Stage 5 for a run with more than one timeline (issue #66).

    ``timelines`` are in playback order -- "before" segments, the song,
    "after" segments -- and each carries its own chunks, its own Stage 4
    results and its own audio. A run with a single timeline does **not** come
    here: it calls :func:`assemble_final_video`, whose two ffmpeg calls are
    byte-for-byte what they have always been. Adding a prologue is what
    changes the assembly, and nothing else does.

    What this does that the single-timeline path cannot:

    1. **Probes and compares the concat signature** of the first chunk of
       every timeline (:data:`VIDEO_SIGNATURE_ENTRIES`). The concat demuxer
       does not fail on mismatched inputs; it writes a file that plays wrong.
    2. **Concats each timeline separately and measures the result**, so the
       seam arithmetic is done on ffprobe's reading of real files rather than
       on the sum of the spans those files were supposed to contain. That
       distinction is issue #22's whole finding: the chunk timeline
       overshoots its master by up to one trained-floor chunk, and for eight
       months nothing said so because ``-shortest`` hid it.
    3. **Reconciles each seam by padding audio up to video, and asserts it
       with a second probe.** A few frames of drift at a join desyncs
       everything after it while both halves look perfect in isolation, and
       there is no ``-shortest`` in the middle of a file to save it.
    4. **Checks the finished file** against the sum of the measured timeline
       durations, raising :class:`DurationMismatchError` like the silent path
       does.

    Every timeline's ``audio`` being ``None`` is issue #22's silent path:
    no mux, no audio anywhere, and the concat of timeline videos writes
    straight to the deliverable. Mixing (some audio, some not) is refused --
    a video with a hole where one timeline's audio should be is not something
    to guess about.
    """
    if not timelines:
        raise ValueError("assemble_timelines needs at least one timeline")

    runner = runner or _default_runner
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_present = [t.audio is not None for t in timelines]
    if any(audio_present) and not all(audio_present):
        silent_names = [t.name for t in timelines if t.audio is None]
        raise ValueError(
            "every timeline must have audio, or none of them (issue #22's silent path); "
            f"these have none: {silent_names}. A finished video with one timeline's audio "
            "missing is a hole nothing downstream can detect."
        )
    has_audio = all(audio_present)

    # 1. Availability first, before any subprocess, exactly as the
    #    single-timeline path does -- a missing chunk must not be discovered
    #    after an ffmpeg pass has already run.
    clips: dict[str, list[Path]] = {}
    ids: dict[str, list[int]] = {}
    for timeline in timelines:
        clips[timeline.name], ids[timeline.name] = _timeline_video_paths(timeline)

    dark_warnings: list[DarkChunkWarning] = []
    cut_warnings: list[SceneCutWarning] = []
    for timeline in timelines:
        dark, cuts = _run_chunk_checks(
            timeline,
            ids[timeline.name],
            check_luminance=check_luminance,
            luminance_floor=luminance_floor,
            luminance_runner=luminance_runner or runner,
            check_scene_cuts=check_scene_cuts,
            scene_cut_threshold=scene_cut_threshold,
            scene_cut_runner=scene_cut_runner or runner,
        )
        dark_warnings.extend(dark)
        cut_warnings.extend(cuts)

    # 2. The concat signature, probed from the first chunk of each timeline.
    signatures = {
        timeline.name: probe_video_signature(clips[timeline.name][0], runner)
        for timeline in timelines
    }
    agreed_signature = compare_video_signatures(signatures)
    logger.info(
        "Concat signature agreed across %d timeline(s): %s",
        len(timelines),
        agreed_signature,
    )

    # 3. One concat per timeline, then measure what it actually produced.
    timeline_videos: list[Path] = []
    measurements: list[tuple[str, str | None, float, float]] = []
    for timeline in timelines:
        concat_list = output_dir / TIMELINE_CONCAT_LIST_TEMPLATE.format(name=timeline.name)
        write_concat_file(clips[timeline.name], concat_list)
        video_path = output_dir / TIMELINE_VIDEO_TEMPLATE.format(name=timeline.name)
        _run_ffmpeg(
            build_concat_args(concat_list, video_path), runner, step=f"concat:{timeline.name}"
        )
        video_seconds = probe_duration_seconds(video_path, runner)
        audio_seconds = (
            probe_duration_seconds(timeline.audio, runner)
            if timeline.audio is not None
            else video_seconds
        )
        timeline_videos.append(video_path)
        measurements.append(
            (timeline.name, timeline.fingerprint_name, video_seconds, audio_seconds)
        )

    placements = place_timelines(measurements, tolerance_seconds=duration_tolerance_seconds)
    log_placements(placements, measured=True)

    # 4. Reconcile each seam by padding audio up to its own video, then
    #    measure the padded file back. Computing the pad is not evidence it
    #    happened.
    combined_audio: Path | None = None
    if has_audio:
        song_audio = next(
            (t.audio for t in timelines if t.fingerprint_name is None), timelines[0].audio
        )
        layout = probe_audio_layout(song_audio, runner)  # type: ignore[arg-type]
        padded: list[Path] = []
        for timeline, placement in zip(timelines, placements, strict=True):
            padded_path = output_dir / TIMELINE_AUDIO_TEMPLATE.format(name=timeline.name)
            _run_ffmpeg(
                build_audio_pad_args(
                    timeline.audio,  # type: ignore[arg-type]
                    padded_path,
                    pad_to_seconds=placement.video_seconds,
                    sample_rate=layout.get("sample_rate"),
                    channels=layout.get("channels"),
                ),
                runner,
                step=f"seam-pad:{timeline.name}",
            )
            measured = probe_duration_seconds(padded_path, runner)
            if abs(measured - placement.video_seconds) > duration_tolerance_seconds:
                logger.error(
                    "Seam mismatch on timeline %r: padded audio measures %.3fs against "
                    "%.3fs of video (%+.3fs, tolerance %.3fs) -- everything after this "
                    "seam would play out of sync while both halves looked correct alone.",
                    timeline.name,
                    measured,
                    placement.video_seconds,
                    measured - placement.video_seconds,
                    duration_tolerance_seconds,
                )
                raise SeamMismatchError(
                    timeline.name,
                    placement.video_seconds,
                    measured,
                    duration_tolerance_seconds,
                )
            logger.info(
                "Seam for timeline %r: audio %.3fs padded with %.3fs of silence to "
                "%.3fs, measured %.3fs against its own video (issue #66)",
                timeline.name,
                placement.audio_seconds,
                placement.pad_seconds,
                placement.video_seconds,
                measured,
            )
            padded.append(padded_path)

        combined_audio = output_dir / COMBINED_AUDIO_FILENAME
        _run_ffmpeg(
            build_audio_concat_args(padded, combined_audio), runner, step="seam-audio-concat"
        )

    # 5. Join the timelines, mux, and check the finished file.
    output_path = output_dir / output_filename
    concat_file = output_dir / CONCAT_LIST_FILENAME
    write_concat_file(timeline_videos, concat_file)

    expected_total = placements[-1].end_seconds
    if combined_audio is None:
        logger.warning(
            "No timeline has audio: assembling %s as a SILENT video with NO AUDIO STREAM "
            "at all -- this suspends the CLAUDE.md invariant 'the master audio track is "
            "the only audio in the final video' for this run (issue #22).",
            output_path,
        )
        intermediate_video = output_path
        concat_args = build_concat_args(concat_file, output_path)
        _run_ffmpeg(concat_args, runner, step="concat")
        mux_args: tuple[str, ...] = ()
    else:
        intermediate_video = output_dir / INTERMEDIATE_VIDEO_FILENAME
        concat_args = build_concat_args(concat_file, intermediate_video)
        _run_ffmpeg(concat_args, runner, step="concat")
        mux_list = build_mux_args(intermediate_video, combined_audio, output_path)
        _run_ffmpeg(mux_list, runner, step="mux")
        mux_args = tuple(mux_list)

    measured_duration = probe_duration_seconds(output_path, runner)
    drift = measured_duration - expected_total
    if abs(drift) > duration_tolerance_seconds:
        logger.error(
            "Duration mismatch for %s: measured=%.3fs expected=%.3fs drift=%+.3fs "
            "tolerance=%.3fs -- with a seam there is no -shortest to hide this behind "
            "(issue #66); the file is written and stays on disk for inspection.",
            output_path,
            measured_duration,
            expected_total,
            drift,
            duration_tolerance_seconds,
        )
        raise DurationMismatchError(
            output_path, expected_total, measured_duration, duration_tolerance_seconds
        )

    logger.info(
        "Assembled %d timeline(s) into %s: %.3fs, has_audio=%s, chunks=%s",
        len(timelines),
        output_path,
        measured_duration,
        has_audio,
        {name: len(ids[name]) for name in ids},
    )

    return MultiTimelineAssemblyResult(
        output_video=output_path,
        concat_file=concat_file,
        intermediate_video=intermediate_video,
        timeline_videos=tuple(timeline_videos),
        placements=placements,
        chunk_ids=tuple((t.name, tuple(ids[t.name])) for t in timelines),
        concat_args=tuple(concat_args),
        mux_args=mux_args,
        dark_chunk_warnings=tuple(dark_warnings),
        scene_cut_warnings=tuple(cut_warnings),
        has_audio=has_audio,
        measured_duration=measured_duration,
        video_signature=tuple(sorted(agreed_signature.items())),
    )
