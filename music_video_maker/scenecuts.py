"""Post-render scene-cut check (issue #81).

A viewer on the second full "Deathless" render:

    At 2:25 it flashes a snow-covered mountain top for all of 1 second, then
    by 2:26 it is an army facing a castle, then 2:28 it is Dianne walking
    through that snow-covered mountain top.

All three are **one chunk**. Chunk 22 spans 2:24.83-2:30.00 -- a single
5.17s take -- and `global_style` on that run reads "Each shot is ONE
continuous unbroken take, a single camera, no cuts within the shot, no
montage". H3 declined the instruction, and short of watching all 80 chunks
there was no way to know.

The instruction is not a property of the output
----------------------------------------------
Exactly the #81 general form: anything the model is merely *told* to do
needs a measurement of the rendered result, or nobody ever learns whether it
complied. This module is that measurement, and it is exactly the shape of
`luminance.py`'s darkness floor (issue #77) for the same reason: cheap,
sampled from the *rendered* video rather than the prompt, and never fatal.

The measurement
----------------
ffmpeg's own `scene` metric, scored over **two independent full 80-chunk
renders** of "Deathless":

Corpus A -- `~/mvm-runs/deathless/output/chunks_v3` (the 2026-08-16 render
issue #81 itself measured; its numbers reproduce exactly):

* At threshold 0.25: 3 chunks flagged (12, 22, 75), 4 cuts total. Chunk 22
  has 2 cuts, at +1.08s (score 0.752) and +3.71s (score 0.917) -- exactly
  where the viewer put them. Chunk 12: +2.17s @ 0.519. Chunk 75: +3.21s @
  0.370.
* Controls 7 / 29 / 45: max score 0.0061 / 0.0331 / 0.0241 -- an order of
  magnitude below.
* Largest score on any unflagged chunk: 0.112 (chunk 27, a continuous shot
  with a slight push-in, verified on pixels). Smallest flagged: 0.370
  (chunk 75). A 3.3x gap with nothing in between.

Corpus B -- `~/mvm-runs/deathless/output/chunks_v12` (the 2026-08-22 current
keeper render, an independent corpus):

* At threshold 0.25: 1 chunk flagged (4), 1 cut, at +6.08s, score 0.330.
* Largest score on any unflagged v12 chunk: 0.076 -- another clean gap,
  4.3x.
* Verified on pixels: chunk 4 @ +6.08s cuts from an extreme macro of cracked
  stone to a medium shot of the second cast member on a hilltop -- a real,
  previously unreported cut in the current best video.

Why 0.25, and what was excluded
--------------------------------
`DEFAULT_SCENE_THRESHOLD` sits inside a **plateau**: every threshold in
[0.15, 0.30] returns the identical answer on both corpora (3 chunks / 4 cuts
on A, 1 chunk / 1 cut on B). Insensitivity over a 2x range of the parameter
is what makes this a separating statistic rather than a tuned one --
threshold decisions scored against a single number, the way #60 and #76
warn against, do not survive contact with a second corpus; this one already
has two.

Excluded candidates, and why (per this project's #60/#76 convention of
recording what did *not* ship):

* **0.10 -- rejected.** Pulls in corpus A chunk 27 at 0.112, verified on
  pixels to be one continuous shot with a push-in. A false positive bought
  for no true positive.
* **0.35 -- rejected.** Loses corpus B's only true positive (chunk 4 at
  0.330) -- on the wrong side of the newer corpus's one real case.
* **0.40 and above -- rejected.** Also loses corpus A chunk 75 (0.370).
* **A first-frame guard (ignore scores in a chunk's first two frames) --
  measured inert, NOT shipped.** 0 of 160 chunks across both corpora have a
  >0.25 score in their first two frames. An inert guard is untested code,
  so it was not added.

Why H3 declining a "one continuous take" instruction is worse than cosmetic
-----------------------------------------------------------------------------
The seed-frame chaining path (see project `CLAUDE.md`, "On the chained path,
the seed frame IS the identity conditioning") takes a chunk's *final* frame
as the next chunk's identity conditioning. A chunk that cut to a different
scene mid-way hands the next chunk a final frame from a shot nobody
authored. Wiring the actual `i2v_chain_scope` lookup into `assembly.py` is
more than this check needs to do its job -- the hazard is stated in the
ERROR log line `assemble_final_video` emits and in this docstring, not
detected mechanically; see `assembly.py`'s module docstring for the same
note on the wiring boundary.

Same dependency and failure discipline as `luminance.py`
----------------------------------------------------------
stdlib + `music_video_maker.contracts` only -- no OpenCV, no numpy, no new
dependency, so `assembly.py` keeps depending on nothing else and this check
runs unconditionally. The subprocess runner is injectable with the identical
shape `luminance.SubprocessRunner` uses, so `assemble_final_video` passes
its own runner straight through and unit tests never spawn a real process.

Every failure mode here degrades to "cannot prove this chunk has a cut"
rather than raising: a missing chunk, ffmpeg not on PATH, a non-zero exit,
unparseable output, a corrupt mp4. This check exists to warn a human before
they watch an 8-minute video, never to abort an assembly that would
otherwise succeed -- hours of GPU custody must not produce a video Stage 5
then refuses to hand back.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Measured, not invented -- see the module docstring for the two-corpus basis
# and the excluded candidates.
# --------------------------------------------------------------------------- #

DEFAULT_SCENE_THRESHOLD = 0.25
"""ffmpeg `scene` score above which a frame is treated as a cut.

Measured over two independent full 80-chunk renders of "Deathless" (see the
module docstring): at 0.25, corpus A (2026-08-16 render) flags chunks 12,
22, 75 (4 cuts, matching issue #81's own report exactly) and corpus B
(2026-08-22 render) flags chunk 4 (1 cut, a previously unreported real
defect). The value sits inside a plateau -- every threshold in [0.15, 0.30]
gives the identical answer on both corpora -- rather than at a tuned edge:
corpus A's gap runs from 0.112 (largest score on any clean chunk) to 0.370
(smallest flagged score), and corpus B's from 0.076 to 0.330. 0.25 clears
both gaps with margin on both sides, the same "measured gap, not a round
number someone liked" reasoning `luminance.DEFAULT_DARK_FLOOR` and
`faces.DEFAULT_MIN_FACE_FRACTION` use. 0.10 and 0.35+ were scored and
rejected -- see the module docstring."""

SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]
"""Injectable seam for spawning ffmpeg -- identical shape to
``luminance.SubprocessRunner`` so ``assemble_final_video`` can pass its own
runner straight through without adapting it."""

_TIME_RE = re.compile(r"pts_time:\s*(\S+)")
_SCORE_RE = re.compile(r"lavfi\.scene_score=\s*(\S+)")


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real ffmpeg invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class SceneCut:
    """One frame ffmpeg's `scene` metric scored above the threshold."""

    time_seconds: float
    """Offset into the chunk's own clip, in seconds -- each chunk mp4 is a
    standalone file starting at 0, the same convention
    ``luminance.build_frame_probe_args`` documents."""
    score: float
    """ffmpeg's raw `scene` score for this frame (0.0-1.0-ish; not strictly
    bounded, ffmpeg's own metric)."""


@dataclass(frozen=True)
class ChunkSceneCuts:
    """What was measured for one chunk."""

    chunk_id: int
    video_file: Path
    cuts: tuple[SceneCut, ...]
    """Every frame scoring above the threshold this measurement was taken
    at. Empty means either "measured, genuinely no cut" or "could not be
    measured" -- disambiguate via :attr:`max_score`, the same way
    ``ChunkLuminance.end_mean`` distinguishes a real low reading from
    ``None``."""
    max_score: float | None
    """The highest `scene` score seen across every probed frame, regardless
    of the threshold -- so re-thresholding a single probe pass is possible
    and the plateau argument in the module docstring is checkable against
    every chunk, including clean ones. ``None`` means the probe produced no
    readable frame scores at all (missing file, ffmpeg failure, unparseable
    output) -- "cannot prove this chunk has no cut", not "zero cuts"."""


@dataclass(frozen=True)
class SceneCutWarning:
    """One chunk containing at least one cut above the threshold."""

    chunk_id: int
    video_file: Path
    start: float
    """The chunk's start time in the master track, in seconds -- carried
    through so a log line or report can say *where in the song* this is
    without a second lookup."""
    end: float
    cuts: tuple[SceneCut, ...]
    max_score: float
    threshold: float


def build_scene_probe_args(video_path: Path) -> list[str]:
    """ffmpeg args for one full-file scene-detection pass over ``video_path``.

    Prints `frame:N pts:N pts_time:T` / `lavfi.scene_score=S` pairs for
    *every* frame (the filter selects `gte(scene,0)`, true unconditionally)
    to the ``metadata=print:file=-`` sink. Filtering by threshold happens in
    Python (:func:`measure_chunk_scene_cuts`), not in the filter expression,
    so one probe pass supports re-thresholding and ``max_score`` is
    available even for a chunk with no cut. Verified against ffmpeg 8.1.2.
    """
    return [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "select='gte(scene,0)',metadata=print:file=-",
        "-an",
        "-f",
        "null",
        "-",
    ]


def parse_scene_scores(output: str) -> list[tuple[float, float]]:
    """Parse ``frame:N pts:N pts_time:T`` / ``lavfi.scene_score=S`` pairs out
    of ffmpeg's ``metadata=print`` text.

    Pure function, no I/O: a ``pts_time`` line sets the pending timestamp for
    the *next* score line that follows it; a score line with no pending
    timestamp (garbage, truncated output) is dropped rather than paired with
    a stale one, and a malformed number on either side drops that pending
    pair rather than raising. Never raises on any input, including empty or
    binary-garbage text.
    """
    pairs: list[tuple[float, float]] = []
    pending_time: float | None = None

    for line in output.splitlines():
        time_match = _TIME_RE.search(line)
        if time_match:
            try:
                pending_time = float(time_match.group(1))
            except ValueError:
                pending_time = None
            continue

        score_match = _SCORE_RE.search(line)
        if score_match and pending_time is not None:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pending_time = None
                continue
            pairs.append((pending_time, score))
            pending_time = None

    return pairs


def measure_chunk_scene_cuts(
    *,
    chunk_id: int,
    video_file: Path,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    runner: SubprocessRunner | None = None,
) -> ChunkSceneCuts:
    """Run one ffmpeg scene-detection pass over ``video_file`` and return
    every frame scoring above ``threshold``, plus the raw max score.

    Never raises: a runner exception, a non-zero exit, or output with no
    parseable frame scores all degrade to ``ChunkSceneCuts(cuts=(),
    max_score=None)`` -- "cannot prove this chunk has a cut" -- logged at
    WARNING, not "the chunk is clean".

    ffmpeg may emit ``metadata=print`` output on stdout or stderr depending
    on the build; the corpus scan behind :data:`DEFAULT_SCENE_THRESHOLD`
    read both streams and concatenated them, so this does too rather than
    risking a build-dependent silent miss.
    """
    runner = runner or _default_runner
    args = build_scene_probe_args(video_file)

    try:
        result = runner(args)
    except Exception as exc:  # noqa: BLE001 - degrade, never let a broken
        # runner (ffmpeg missing, OS-level spawn failure) take assembly down.
        logger.warning(
            "chunk %d (%s): could not run ffmpeg to probe scene cuts -- %s. "
            "Skipping this chunk; the scene-cut check degrades gracefully.",
            chunk_id,
            video_file,
            exc,
        )
        return ChunkSceneCuts(chunk_id=chunk_id, video_file=video_file, cuts=(), max_score=None)

    if result.returncode != 0:
        stderr_text = _decode(result.stderr)
        logger.warning(
            "chunk %d (%s): ffmpeg scene probe failed (exit=%s) -- %s. "
            "Skipping this chunk; the scene-cut check degrades gracefully.",
            chunk_id,
            video_file,
            result.returncode,
            stderr_text.strip(),
        )
        return ChunkSceneCuts(chunk_id=chunk_id, video_file=video_file, cuts=(), max_score=None)

    # Read both streams -- see the docstring above for why.
    combined = _decode(result.stdout) + _decode(result.stderr)
    pairs = parse_scene_scores(combined)

    if not pairs:
        logger.warning(
            "chunk %d (%s): scene probe produced no parseable frame scores -- "
            "cannot prove this chunk is cut-free. Skipping.",
            chunk_id,
            video_file,
        )
        return ChunkSceneCuts(chunk_id=chunk_id, video_file=video_file, cuts=(), max_score=None)

    max_score = max(score for _, score in pairs)
    cuts = tuple(SceneCut(time_seconds=t, score=s) for t, s in pairs if s > threshold)

    logger.info(
        "chunk %d (%s): scene probe max_score=%.3f cuts=%d (threshold=%.2f)",
        chunk_id,
        video_file,
        max_score,
        len(cuts),
        threshold,
    )

    return ChunkSceneCuts(chunk_id=chunk_id, video_file=video_file, cuts=cuts, max_score=max_score)


def check_scene_cuts(
    chunks: Sequence[AudioChunk],
    results: Mapping[int, ChunkResult] | RunState,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    runner: SubprocessRunner | None = None,
) -> tuple[SceneCutWarning, ...]:
    """Flag every chunk containing at least one frame scoring above
    ``threshold``.

    Intended to run as part of Stage 5 assembly, once per chunk, so a chunk
    that montages inside a single take is visible in the log before a human
    sits through the whole video. Never raises: a chunk with no available
    result, an unreadable video, or an ffmpeg failure is logged and skipped
    -- it is never flagged (an unmeasured chunk is not evidence of a cut)
    and it never aborts the rest of the check.
    """
    result_map: Mapping[int, ChunkResult] = (
        results.results if isinstance(results, RunState) else results
    )
    runner = runner or _default_runner

    flags: list[SceneCutWarning] = []
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        result = result_map.get(chunk.chunk_id)
        if (
            result is None
            or result.video_file is None
            or result.status is ChunkStatus.DEAD_LETTERED
            or not result.succeeded
        ):
            logger.info(
                "chunk %d: no rendered video available -- skipping the scene-cut check "
                "for this chunk.",
                chunk.chunk_id,
            )
            continue

        try:
            measurement = measure_chunk_scene_cuts(
                chunk_id=chunk.chunk_id,
                video_file=result.video_file,
                threshold=threshold,
                runner=runner,
            )
        except Exception:  # noqa: BLE001 - a check must never take assembly down.
            logger.exception(
                "chunk %d (%s): the scene-cut check raised unexpectedly -- skipping it. "
                "This must never abort assembly.",
                chunk.chunk_id,
                result.video_file,
            )
            continue

        if not measurement.cuts:
            continue

        flag = SceneCutWarning(
            chunk_id=chunk.chunk_id,
            video_file=result.video_file,
            start=chunk.start,
            end=chunk.end,
            cuts=measurement.cuts,
            max_score=measurement.max_score if measurement.max_score is not None else 0.0,
            threshold=threshold,
        )
        flags.append(flag)
        logger.warning(
            "chunk %d contains %d cut(s) inside a single take: %s (threshold %.2f), "
            "span %.2fs-%.2fs, %s -- if unintended, check whether global_style's "
            "'one continuous take' instruction was honoured; --reseed is the remedy "
            "(issue #81).",
            chunk.chunk_id,
            len(measurement.cuts),
            ", ".join(f"+{c.time_seconds:.2f}s@{c.score:.3f}" for c in measurement.cuts),
            threshold,
            chunk.start,
            chunk.end,
            result.video_file,
        )

    return tuple(flags)


# --------------------------------------------------------------------------- #
# Standalone entry point: python -m music_video_maker.scenecuts <chunks_dir>
#
# Deliberately not wired into cli.py's parser -- that file is contended with
# other in-flight work and its parser is flat with a required --config this
# tool has no use for. This is a self-contained scan of chunk mp4s already
# on disk, useful whenever a render exists (finished, partial, or a slice
# rendered via --only-chunks) without re-running any pipeline stage.
# --------------------------------------------------------------------------- #

_CHUNK_ID_RE = re.compile(r"chunk_(\d+)")


def _parse_chunk_id(video_path: Path) -> int:
    match = _CHUNK_ID_RE.search(video_path.stem)
    return int(match.group(1)) if match else -1


def _emit(line: str, *, stream: object | None = None) -> None:
    # sys.stdout.write, not print(): this is the tool's actual product (like
    # `git status`'s output), the same reasoning `authoring/cli.py` uses for
    # its own user-facing prints -- but that file's per-file ruff exemption
    # doesn't cover this new module, so the direct write sidesteps flake8-print
    # (T20) without asking for one. Looked up at call time (never bound as a
    # default argument) so pytest's ``capsys`` -- which swaps ``sys.stdout``
    # per-test -- actually sees it.
    (stream if stream is not None else sys.stdout).write(line + "\n")


def main(argv: list[str] | None = None, *, runner: SubprocessRunner | None = None) -> int:
    """Scan every ``chunk_*.mp4`` in a directory and report scene cuts.

    ``runner`` is accepted for testability (see ``tests/test_scenecuts.py``)
    -- never exposed as a CLI flag, since there is no real alternative to
    ffmpeg a user would select on the command line.
    """
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.scenecuts",
        description=(
            "Scan rendered chunk_*.mp4 files for scene cuts inside a single "
            "take (issue #81). Prints one line per flagged chunk and a "
            "summary count; exits nonzero if anything was flagged."
        ),
    )
    parser.add_argument("chunks_dir", type=Path, help="Directory containing chunk_*.mp4 files")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        help=(
            "ffmpeg 'scene' score above which a frame counts as a cut "
            f"(default {DEFAULT_SCENE_THRESHOLD})"
        ),
    )
    args = parser.parse_args(argv)

    chunks_dir: Path = args.chunks_dir
    if not chunks_dir.is_dir():
        _emit(f"error: {chunks_dir} is not a directory", stream=sys.stderr)
        return 1

    videos = sorted(chunks_dir.glob("chunk_*.mp4"))
    if not videos:
        _emit(f"no chunk_*.mp4 files found in {chunks_dir}")
        return 0

    flagged = 0
    total_cuts = 0
    for video in videos:
        measurement = measure_chunk_scene_cuts(
            chunk_id=_parse_chunk_id(video),
            video_file=video,
            threshold=args.threshold,
            runner=runner,
        )
        if measurement.cuts:
            flagged += 1
            total_cuts += len(measurement.cuts)
            times = ", ".join(
                f"+{c.time_seconds:.2f}s (score {c.score:.3f})" for c in measurement.cuts
            )
            _emit(f"{video.name}: {len(measurement.cuts)} cut(s) -- {times}")

    _emit(
        f"\n{flagged}/{len(videos)} chunk(s) flagged, {total_cuts} cut(s) total "
        f"(threshold={args.threshold})"
    )
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_SCENE_THRESHOLD",
    "ChunkSceneCuts",
    "SceneCut",
    "SceneCutWarning",
    "SubprocessRunner",
    "build_scene_probe_args",
    "check_scene_cuts",
    "main",
    "measure_chunk_scene_cuts",
    "parse_scene_scores",
]
