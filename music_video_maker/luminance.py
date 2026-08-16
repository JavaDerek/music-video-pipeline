"""Post-render darkness check (issue #77).

A viewer reported "sudden darkness for 2 seconds starting at 4:41" in the
"Deathless" render. It was authored: chunk 43's shot line asks for a glow
that "dims and finally goes dark" over a line Jan is singing. Measured mean
luminance, first quarter -> last quarter of that chunk: Y 76.4 -> 15.0.

Drift is the wrong statistic
-----------------------------
F11 proposed a *drift* band (start vs end luminance within about +/-15 Y)
from a 3-chunk validation slice, before the full render existed. Scanning all
80 chunks of the finished render: **36 fall outside +/-15**, ranging from -61
to +55. That is expected, not broken -- "Deathless" is a night ending in
dawn, so a chunk legitimately brightening or darkening by 30+ Y over its own
span is the song's content, not a defect. A drift-band check would fire on
nearly half the video.

What isolates chunk 43 is **absolute darkness, not drift**: it ends at Y
15.0, while the next darkest *chunk ending* anywhere in the render is Y 36.4
(chunk 18). That is a 21+ Y gap with exactly one chunk on the dark side of
it. See ``docs/deathless-render-corpus.md`` for the full 80-chunk
distribution this module's :data:`DEFAULT_DARK_FLOOR` is measured against --
it is not re-derived here, and it must not be re-derived as a +/-15 drift
band; that is precisely the statistic the full render showed was wrong.

This module therefore flags an absolute floor on a chunk's *ending*
luminance only. ``drift`` (the difference between the chunk's start and end
sampled means) is still computed and carried on :class:`ChunkLuminance` for
context in log lines and reports, but it never gates the flag.

Cheap, sampled, never fatal
----------------------------
Reads a handful of downscaled frames per chunk via ``ffmpeg`` (a 32x18
grayscale raw-pixel probe, not a full decode), never OpenCV -- ``assembly.py``
depends on nothing but the stdlib and ``contracts``, and this module keeps
that invariant so the check runs unconditionally, with no optional
dependency to install. The subprocess runner is injectable, the same seam
``assembly.py`` already uses for ffmpeg, so unit tests never spawn a real
process.

Every failure mode here degrades to "cannot prove this chunk is dark" rather
than raising: a missing chunk, an unreadable frame, ffmpeg not being on
PATH, a corrupt mp4. This check exists to warn a human before they watch an
8-minute video, never to abort an assembly that would otherwise succeed --
the same asymmetry every lint in ``shot_plan.py`` follows, and for the same
reason: a false positive here is cheap (someone glances at a chunk that was
actually fine) and a false *crash* is not (hours of GPU custody produced a
video Stage 5 then refused to hand back).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Measured, not invented -- see the module docstring and
# docs/deathless-render-corpus.md for the full 80-chunk distribution.
# --------------------------------------------------------------------------- #

DEFAULT_DARK_FLOOR = 25.0
"""Mean luminance (0-255, Y channel) below which a chunk's *ending* is
flagged.

Measured against all 80 rendered chunks of "Deathless" (see
``docs/deathless-render-corpus.md``): the flagged chunk (43, the viewer
report) ends at Y 14.9; the next darkest chunk ending anywhere in that render
is Y 36.4 (chunk 18, a legitimate night shot). 25.0 sits in the middle of
that gap with more than 5 Y of margin on both sides -- comfortably clear of
both the one real failure and the darkest *normal* chunk in the corpus, the
same "measured gap, not a round number someone liked" reasoning
``faces.DEFAULT_MIN_FACE_FRACTION`` uses."""

DEFAULT_START_FRACTIONS: tuple[float, ...] = (0.05, 0.15, 0.25)
"""Sample points inside a chunk's first quarter, as a fraction of its
duration. Three samples spread across the quarter rather than one frame at
the very edge, which could land on a black pre-roll or transition frame."""

DEFAULT_END_FRACTIONS: tuple[float, ...] = (0.75, 0.85, 0.95)
"""Sample points inside a chunk's last quarter -- the side the floor check
actually gates on."""

_PROBE_WIDTH = 32
_PROBE_HEIGHT = 18
"""Downscaled probe frame size. Luminance is a spatial average, so a 32x18
frame (576 px) approximates the full-resolution mean closely while keeping
each ffmpeg decode+scale trivial -- this is a sampled, cheap check, not a
frame-accurate one (see the module docstring)."""


SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]
"""Injectable seam for spawning ffmpeg -- identical shape to
``assembly.SubprocessRunner`` so ``assemble_final_video`` can pass its own
runner straight through without adapting it."""


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real ffmpeg invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False)


@dataclass(frozen=True)
class ChunkLuminance:
    """What was measured for one chunk."""

    chunk_id: int
    video_file: Path
    start_mean: float | None
    """Mean luminance (0-255) sampled across the chunk's first quarter.
    ``None`` when every sample in that quarter was unreadable."""
    end_mean: float | None
    """Mean luminance sampled across the chunk's last quarter. ``None`` when
    every sample was unreadable. This is the value :func:`check_dark_chunks`
    compares against the floor."""

    @property
    def drift(self) -> float | None:
        """``end_mean - start_mean``, for log/report context only -- never
        used to decide whether a chunk is flagged. See the module docstring
        for why a drift band is the wrong check for this render."""
        if self.start_mean is None or self.end_mean is None:
            return None
        return self.end_mean - self.start_mean


@dataclass(frozen=True)
class DarkChunkWarning:
    """One chunk whose ending luminance fell below the floor."""

    chunk_id: int
    video_file: Path
    start: float
    """The chunk's start time in the master track, in seconds -- carried
    through so a log line or report can say *where in the song* this is
    without a second lookup."""
    end: float
    end_mean: float
    floor: float
    start_mean: float | None = None
    drift: float | None = None


def build_frame_probe_args(
    video_path: Path, timestamp: float, *, width: int = _PROBE_WIDTH, height: int = _PROBE_HEIGHT
) -> list[str]:
    """ffmpeg args to decode exactly one downscaled grayscale frame at
    ``timestamp`` (seconds, relative to the start of ``video_path`` itself --
    each chunk mp4 is its own standalone clip) and write it to stdout as raw
    ``width*height`` gray8 bytes, one per pixel, no header.

    Downscales via ``-s`` (output frame size) rather than ``-vf scale=...``:
    same pixel result, but it keeps this probe's argument list free of
    ``-vf`` -- the flag ``continuity.py``'s seed-frame extraction uses, and
    which at least one caller greps for to tell "did continuity run" apart
    from "did assembly run". Two unrelated ffmpeg calls should not become
    indistinguishable by accident."""
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]


def _extract_frame_luminance(
    video_path: Path,
    timestamp: float,
    *,
    chunk_id: int,
    runner: SubprocessRunner,
    width: int = _PROBE_WIDTH,
    height: int = _PROBE_HEIGHT,
) -> float | None:
    """Mean luminance of one probed frame, or ``None`` (logged) if it could
    not be read. Never raises -- every failure here is "this one sample is
    unavailable", not "the check must abort"."""
    args = build_frame_probe_args(video_path, timestamp, width=width, height=height)
    try:
        result = runner(args)
    except Exception as exc:  # noqa: BLE001 - degrade, never let a broken
        # runner (ffmpeg missing, OS-level spawn failure) take assembly down.
        logger.warning(
            "chunk %d (%s): could not run ffmpeg to sample the frame at %.3fs -- %s. "
            "Skipping this sample; the darkness check degrades gracefully.",
            chunk_id,
            video_path,
            timestamp,
            exc,
        )
        return None

    if result.returncode != 0:
        stderr = result.stderr
        stderr_text = (
            stderr if isinstance(stderr, str) else (stderr or b"").decode("utf-8", errors="replace")
        )
        logger.warning(
            "chunk %d (%s): ffmpeg could not read the frame at %.3fs (exit=%s) -- %s. "
            "Skipping this sample; the darkness check degrades gracefully.",
            chunk_id,
            video_path,
            timestamp,
            result.returncode,
            stderr_text.strip(),
        )
        return None

    stdout = result.stdout or b""
    expected_len = width * height
    if len(stdout) != expected_len:
        logger.warning(
            "chunk %d (%s): frame probe at %.3fs returned %d bytes, expected %d -- "
            "treating as unreadable and skipping this sample.",
            chunk_id,
            video_path,
            timestamp,
            len(stdout),
            expected_len,
        )
        return None

    return sum(stdout) / len(stdout)


def measure_chunk_luminance(
    *,
    chunk_id: int,
    video_file: Path,
    duration: float,
    runner: SubprocessRunner | None = None,
    start_fractions: Sequence[float] = DEFAULT_START_FRACTIONS,
    end_fractions: Sequence[float] = DEFAULT_END_FRACTIONS,
) -> ChunkLuminance:
    """Sample a chunk's first and last quarters and average each side.

    Never raises: a sample that cannot be read is dropped from its side's
    average, and a side with zero readable samples reports ``None`` (logged)
    rather than raising or returning a fabricated number.
    """
    runner = runner or _default_runner

    def _side_mean(fractions: Sequence[float]) -> float | None:
        values = [
            v
            for f in fractions
            if (
                v := _extract_frame_luminance(
                    video_file, f * duration, chunk_id=chunk_id, runner=runner
                )
            )
            is not None
        ]
        if not values:
            logger.warning(
                "chunk %d (%s): none of %d sample frame(s) could be read -- "
                "luminance for that side of the chunk is unmeasured.",
                chunk_id,
                video_file,
                len(fractions),
            )
            return None
        return sum(values) / len(values)

    start_mean = _side_mean(start_fractions)
    end_mean = _side_mean(end_fractions)

    logger.info(
        "chunk %d (%s): luminance start=%s end=%s",
        chunk_id,
        video_file,
        f"{start_mean:.1f}" if start_mean is not None else "unmeasured",
        f"{end_mean:.1f}" if end_mean is not None else "unmeasured",
    )

    return ChunkLuminance(
        chunk_id=chunk_id, video_file=video_file, start_mean=start_mean, end_mean=end_mean
    )


def check_dark_chunks(
    chunks: Sequence[AudioChunk],
    results: Mapping[int, ChunkResult] | RunState,
    *,
    floor: float = DEFAULT_DARK_FLOOR,
    runner: SubprocessRunner | None = None,
) -> tuple[DarkChunkWarning, ...]:
    """Flag every chunk whose sampled ending luminance falls below ``floor``.

    Intended to run as part of Stage 5 assembly, once per chunk, so a chunk
    that renders a fade to black is visible in the log before a human sits
    through the whole video. Never raises: a chunk with no available result,
    an unreadable video, or an ffmpeg failure is logged and skipped -- it is
    never flagged (an unmeasured chunk is not evidence of darkness) and it
    never aborts the rest of the check.
    """
    result_map: Mapping[int, ChunkResult] = (
        results.results if isinstance(results, RunState) else results
    )
    runner = runner or _default_runner

    flags: list[DarkChunkWarning] = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        result = result_map.get(chunk.chunk_id)
        if (
            result is None
            or result.video_file is None
            or result.status is ChunkStatus.DEAD_LETTERED
            or not result.succeeded
        ):
            logger.info(
                "chunk %d: no rendered video available -- skipping the darkness check "
                "for this chunk.",
                chunk.chunk_id,
            )
            continue

        try:
            measurement = measure_chunk_luminance(
                chunk_id=chunk.chunk_id,
                video_file=result.video_file,
                duration=chunk.duration,
                runner=runner,
            )
        except Exception:  # noqa: BLE001 - a check must never take assembly down.
            logger.exception(
                "chunk %d (%s): the darkness check raised unexpectedly -- skipping it. "
                "This must never abort assembly.",
                chunk.chunk_id,
                result.video_file,
            )
            continue

        if measurement.end_mean is None:
            logger.warning(
                "chunk %d (%s): could not measure ending luminance -- skipping the "
                "darkness check for this chunk (cannot prove it is dark).",
                chunk.chunk_id,
                result.video_file,
            )
            continue

        if measurement.end_mean < floor:
            flag = DarkChunkWarning(
                chunk_id=chunk.chunk_id,
                video_file=result.video_file,
                start=chunk.start,
                end=chunk.end,
                end_mean=measurement.end_mean,
                floor=floor,
                start_mean=measurement.start_mean,
                drift=measurement.drift,
            )
            flags.append(flag)
            logger.warning(
                "chunk %d ends dark: mean luminance Y=%.1f (floor %.1f), span %.2fs-%.2fs, "
                "%s -- if unintended, check the shot line for an unplanned fade to black "
                "(issue #77).",
                chunk.chunk_id,
                measurement.end_mean,
                floor,
                chunk.start,
                chunk.end,
                result.video_file,
            )

    return tuple(flags)


__all__ = [
    "DEFAULT_DARK_FLOOR",
    "DEFAULT_END_FRACTIONS",
    "DEFAULT_START_FRACTIONS",
    "ChunkLuminance",
    "DarkChunkWarning",
    "SubprocessRunner",
    "build_frame_probe_args",
    "check_dark_chunks",
    "measure_chunk_luminance",
]
