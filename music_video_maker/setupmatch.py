"""Does a rendered shot open on its camera setup's still? (a post-render continuity check)

**General form: a shot that cuts back to an angle the audience has already seen must open on
what they saw, and nothing about a prompt makes that true.** A dialogue piece cuts back and
forth between a few angles -- one speaker, then the other, then the first again. Rendered one
shot at a time from a reference portrait and a prompt, each return to an angle invented its own
hillside, its own trees and its own neighbours, and a viewer read every cut as the camera
jumping somewhere new. Starting each shot from one still per *setup* (an anchor at frame 0,
``MiniMaxH3AddGuide`` on the local graph) fixed it: every return opened on that still. This
module measures that property on the rendered file, because an instruction to the model is not
a property of the output (#81).

The measurement
---------------
Both images go through ffmpeg alone -- frame N, area-scaled to 54x30, a light Gaussian blur --
and the score is the mean absolute difference of the raw RGB bytes, 0 (identical) to 255. No
OpenCV, no numpy, nothing masked: the people stay in, and the result still separates cleanly.
Measured 2026-11-06 on 30 renders of three cut-backs to two setups, first frame against the
setup's still:

* **anchored at frame 0** (20 renders, two arms, four stills including a re-cropped one):
  3.7-5.0;
* **not anchored** (9 renders, same shots, same prompts, same seeds): 27.6-44.3;
* **the closest unanchored case**: the same shot and prompt as the still's own source render,
  a different seed: 14.7;
* for scale, the two setups' stills against each other: 33.0.

:data:`DEFAULT_MAX_DISTANCE` = 10.0 sits in the empty interval (5.0, 14.7). One corpus, so the
calibration is recorded beside the constant and pinned by a test, and the next piece should
re-measure rather than inherit it.

Excluded, and why
-----------------
* **The frame where the edit cuts away.** Anchored shots drift: in about two thirds of them the
  camera pushes in, pans or trucks during the shot, so the place is the same and the framing is
  not. At the cut-away frame anchored and unanchored takes overlap, and no threshold separates
  "reframed" from "relocated" with this statistic. Camera drift is a separate defect (it is the
  style text's "handheld" doing what it says) and is not measured here.
* **A people mask.** A first version masked every detected face and the body below it; it
  separated the same way (anchored 1.7-2.4, unanchored 39-53 on its own scale) and needed
  OpenCV, so it was dropped.

Standalone on purpose: it takes two paths and knows nothing about scripts, chunks or run state,
so any pipeline that renders shots against a still can call it. Never fatal to anything; the CLI
exits 1 on a mismatch so a script can gate on it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

THUMB_WIDTH = 54
THUMB_HEIGHT = 30
BLUR_SIGMA = 1.5

MEASURED_ANCHORED_MAX = 5.0
"""Largest first-frame distance of any render anchored at frame 0 to its setup's still (n=20)."""
MEASURED_UNANCHORED_MIN = 14.7
"""Smallest first-frame distance of any render NOT anchored to the still it is compared with
(n=10)."""
DEFAULT_MAX_DISTANCE = 10.0
"""A first frame further than this from its setup's still did not open on the setup."""

SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


class SetupMatchError(RuntimeError):
    """A frame could not be read."""


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, check=False)


def build_thumbnail_args(path: Path, frame: int = 0) -> list[str]:
    vf = (f"select=eq(n\\,{frame}),scale={THUMB_WIDTH}:{THUMB_HEIGHT}:flags=area,"
          f"gblur=sigma={BLUR_SIGMA}")
    return ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def thumbnail(path: Path, frame: int = 0, *, runner: SubprocessRunner | None = None) -> bytes:
    """Frame ``frame`` of ``path`` (a video or a still) as blurred 54x30 RGB bytes."""
    runner = runner or _default_runner
    try:
        result = runner(build_thumbnail_args(path, frame))
    except OSError as exc:
        raise SetupMatchError(f"{path}: ffmpeg could not run: {exc}") from exc
    if result.returncode != 0:
        err = result.stderr
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        raise SetupMatchError(f"{path}: ffmpeg failed: {str(err).strip()[-300:]}")
    data = result.stdout if isinstance(result.stdout, bytes) else bytes(result.stdout or b"")
    want = THUMB_WIDTH * THUMB_HEIGHT * 3
    if len(data) != want:
        raise SetupMatchError(f"{path}: frame {frame} gave {len(data)} bytes, expected {want}")
    return data


def distance(a: bytes, b: bytes) -> float:
    """Mean absolute difference of two thumbnails, 0-255, to one decimal."""
    if len(a) != len(b) or not a:
        raise SetupMatchError(f"thumbnails differ in size ({len(a)} vs {len(b)} bytes)")
    return round(sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a), 1)


@dataclass(frozen=True)
class SetupMatch:
    still: Path
    video: Path
    frame: int
    distance: float
    max_distance: float

    @property
    def matches(self) -> bool:
        return self.distance <= self.max_distance

    def describe(self) -> str:
        verdict = "ok" if self.matches else "MISMATCH"
        return (f"{verdict:8} {self.video} frame {self.frame}: {self.distance} from {self.still} "
                f"(max {self.max_distance})")


def check_shot(still: Path, video: Path, *, frame: int = 0,
               max_distance: float = DEFAULT_MAX_DISTANCE,
               runner: SubprocessRunner | None = None) -> SetupMatch:
    """Compare ``video``'s frame ``frame`` (default the first) with its setup's ``still``."""
    d = distance(thumbnail(Path(still), runner=runner),
                 thumbnail(Path(video), frame, runner=runner))
    return SetupMatch(Path(still), Path(video), frame, d, max_distance)


def main(argv: list[str] | None = None) -> int:
    """``python -m music_video_maker.setupmatch STILL VIDEO...``: exit 0 all open on the still,
    1 any mismatch, 2 a file could not be read."""
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.setupmatch",
        description="Check that each shot opens on its camera setup's still.")
    parser.add_argument("still", type=Path)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    worst = 0
    for video in args.videos:
        try:
            result = check_shot(args.still, video, max_distance=args.max_distance,
                                runner=_default_runner)
        except SetupMatchError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        sys.stdout.write(result.describe() + "\n")
        if not result.matches:
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
