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
``luminance`` (itself contracts-and-stdlib-only) and the stdlib. The
subprocess runner is injectable so unit tests can assert on the exact
argument lists without ever invoking a real ``ffmpeg`` binary.

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
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState
from music_video_maker.luminance import DEFAULT_DARK_FLOOR, DarkChunkWarning, check_dark_chunks

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "final_video.mp4"
CONCAT_LIST_FILENAME = "concat_list.txt"
INTERMEDIATE_VIDEO_FILENAME = "_concat_intermediate.mp4"

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


class MissingChunksError(RuntimeError):
    """Raised when the expected chunk set has gaps -- assembly refuses to run.

    Concatenating a video with a chunk silently missing would desync every
    subsequent chunk against the master audio's absolute timeline, which is
    exactly the drift Stage 5 exists to prevent. So any missing or
    dead-lettered chunk aborts assembly loudly rather than producing a
    misaligned video.
    """

    def __init__(self, missing_chunk_ids: Sequence[int], dead_lettered_chunk_ids: Sequence[int]):
        self.missing_chunk_ids = tuple(missing_chunk_ids)
        self.dead_lettered_chunk_ids = tuple(dead_lettered_chunk_ids)
        super().__init__(
            "cannot assemble final video: expected chunk(s) unavailable "
            f"(missing={self.missing_chunk_ids}, dead_lettered={self.dead_lettered_chunk_ids})"
        )


@dataclass(frozen=True)
class AssemblyResult:
    """Everything produced by one :func:`assemble_final_video` run."""

    output_video: Path
    concat_file: Path
    intermediate_video: Path
    chunk_ids: tuple[int, ...]
    """Chunk ids actually concatenated, in the chronological order used."""
    concat_args: tuple[str, ...]
    mux_args: tuple[str, ...]
    dark_chunk_warnings: tuple[DarkChunkWarning, ...] = ()
    """Chunks whose sampled ending luminance fell below the darkness floor
    (issue #77) -- informational only. The video was still assembled; a
    non-empty tuple here means a human should look at these chunks before
    trusting the final cut, not that anything failed."""


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
    master_audio: Path,
    output_dir: Path,
    *,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    runner: SubprocessRunner | None = None,
    check_luminance: bool = True,
    luminance_floor: float = DEFAULT_DARK_FLOOR,
    luminance_runner: SubprocessRunner | None = None,
) -> AssemblyResult:
    """Concat every chunk's rendered video (chronological ``chunk_id`` order)
    and mux the master audio track over it.

    Raises :class:`MissingChunksError` -- before any subprocess runs -- if
    any chunk in ``chunks`` lacks a succeeded, video-bearing result. Raises
    :class:`FfmpegError` if either ffmpeg subprocess exits non-zero.

    ``check_luminance`` (default ``True``) runs the issue #77 darkness check
    against every available chunk before concat, using ``luminance_runner``
    if given or ``runner`` otherwise -- so passing the same fake/real ffmpeg
    runner already used for concat/mux is enough to exercise or disable it in
    tests. The check never raises and never blocks assembly; see
    :func:`music_video_maker.luminance.check_dark_chunks`.
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

    concat_file = output_dir / CONCAT_LIST_FILENAME
    write_concat_file(video_paths, concat_file)  # type: ignore[arg-type]

    intermediate_video = output_dir / INTERMEDIATE_VIDEO_FILENAME
    concat_args = build_concat_args(concat_file, intermediate_video)
    _run_ffmpeg(concat_args, runner, step="concat")

    output_path = output_dir / output_filename
    mux_args = build_mux_args(intermediate_video, master_audio, output_path)
    _run_ffmpeg(mux_args, runner, step="mux")

    logger.info(
        "Assembled final video at %s from %d chunks (ids=%s)",
        output_path,
        len(ordered_ids),
        ordered_ids,
    )

    return AssemblyResult(
        output_video=output_path,
        concat_file=concat_file,
        intermediate_video=intermediate_video,
        chunk_ids=tuple(ordered_ids),
        concat_args=tuple(concat_args),
        mux_args=tuple(mux_args),
        dark_chunk_warnings=dark_chunk_warnings,
    )
