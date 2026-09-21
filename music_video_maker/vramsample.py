"""Sample GPU memory to a CSV while somebody watches a render (issue #98).

This exists for one job: the attended 277-frame proof in
``docs/runbook-288-frame-proof.md`` needs a *number* for peak VRAM during
sampling and during VAE decode, and ComfyUI does not report one. So poll
``nvidia-smi`` beside the render and write every reading down with a
timestamp, which is all the instrument has to do.

**Inert unless invoked.** Nothing here runs at import, nothing in the render
path imports it, and it spawns no subprocess until :func:`sample_vram` is
called. It is deliberately *not* wired into ``ResilientRunner``: this project
takes its VRAM readings from ComfyUI's own ``GET /system_stats`` (see
``custody.build_vram_probe``), and a second, differently-sourced number
inside the render would be a measurement nobody could attribute. This one is
an operator's stopwatch, run in another terminal, and its output is a file
with its own provenance header.

**Yes, it polls.** The project's no-sleep-polling invariant is about
*execution tracking* -- completion is an event on the WebSocket, never a
status poll. Physical memory occupancy has no event to subscribe to, and
this is not in the render path.

**Provenance in the file, not in the filename** (issue #93): the CSV opens
with the exact argv that produced it, the interval, the host clock at start
and the label the operator gave it. A scan whose only distinguishing feature
is its filename can be pointed at the wrong render for a week with nothing
able to tell.

Usage, from a terminal that is *not* running the render::

    python -m music_video_maker.vramsample --out peak.csv --label "277-frame proof"

Ctrl-C stops it and logs the peak.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)

NVIDIA_SMI_ARGV: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
"""The one query this module runs. ``noheader,nounits`` makes each line a bare
integer count of MiB, one line per GPU, which is the whole reason for this
exact form -- anything else needs parsing that can silently half-succeed."""

DEFAULT_INTERVAL_SECONDS = 0.5
"""Half a second. Fast enough to catch a VAE-decode spike that lasts a few
seconds, slow enough that ``nvidia-smi``'s own ~20-50 ms startup is a small
fraction of the interval."""

CSV_COLUMNS = ("timestamp_utc", "elapsed_seconds", "used_mib", "raw")

Runner = Callable[[Sequence[str]], str]
"""Test seam: given argv, return the command's stdout. The default runs
``nvidia-smi``; a test substitutes a function and this module never touches a
GPU."""


def _default_runner(argv: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no user input
        list(argv),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


@dataclass(frozen=True)
class SampleRun:
    """What a sampling run measured."""

    samples: int
    peak_mib: int | None
    peak_at_elapsed: float | None
    failures: int
    out_file: Path

    def describe(self) -> str:
        if self.peak_mib is None:
            return (
                f"{self.samples} sample(s), {self.failures} failed reading(s), no usable "
                f"reading at all -> {self.out_file}"
            )
        return (
            f"{self.samples} sample(s), peak {self.peak_mib} MiB at t+{self.peak_at_elapsed:.1f}s, "
            f"{self.failures} failed reading(s) -> {self.out_file}"
        )


def _parse_used_mib(stdout: str, gpu_index: int) -> tuple[int | None, str]:
    """Pull one GPU's MiB figure out of ``nvidia-smi``'s output.

    Returns ``(None, raw)`` rather than raising when the line is missing or
    not an integer: a single unparseable reading must not end a sampling run
    that is babysitting an hours-long render. The raw text is written to the
    CSV either way, so a reader can see what arrived instead of a number.
    """
    raw = stdout.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if gpu_index >= len(lines):
        return None, raw
    try:
        return int(lines[gpu_index]), raw
    except ValueError:
        return None, raw


def _write_header(handle: TextIO, *, label: str, interval: float, gpu_index: int) -> None:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for line in (
        f"# music_video_maker.vramsample -- label={label!r}",
        f"# argv={' '.join(NVIDIA_SMI_ARGV)}",
        f"# gpu_index={gpu_index} interval_seconds={interval}",
        f"# started_utc={started}",
        "# memory.used is the WHOLE CARD's occupancy: this run's weights plus every other",
        "# tenant. It is not 'what H3 needed'. Read it against a baseline taken with the",
        "# card idle, which the runbook has you record first.",
    ):
        handle.write(line + "\n")


def sample_vram(
    out_file: Path | str,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    duration_seconds: float | None = None,
    gpu_index: int = 0,
    label: str = "",
    runner: Runner | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> SampleRun:
    """Poll GPU memory into ``out_file`` until ``duration_seconds`` elapses.

    ``duration_seconds=None`` samples until interrupted (Ctrl-C), which is
    the normal way to use it: start it before submitting the chunk, stop it
    when the chunk's mp4 appears.

    Every reading is flushed as it is taken. A sampler whose CSV is empty
    because the host wedged before the buffer flushed would be the one
    failure mode this instrument exists to survive -- issue #24's wedge
    needs a power cycle, and whatever is on disk at that moment is the entire
    record of what the card was doing.
    """
    runner = runner if runner is not None else _default_runner
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = 0
    failures = 0
    peak_mib: int | None = None
    peak_at: float | None = None
    start = clock()

    with path.open("w", encoding="utf-8", newline="") as handle:
        _write_header(handle, label=label, interval=interval_seconds, gpu_index=gpu_index)
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        handle.flush()

        try:
            while True:
                elapsed = clock() - start
                if duration_seconds is not None and elapsed > duration_seconds:
                    break
                try:
                    stdout = runner(NVIDIA_SMI_ARGV)
                except Exception as exc:  # noqa: BLE001 -- one bad reading is not the end
                    failures += 1
                    logger.warning(
                        "nvidia-smi reading %d failed (%s: %s) -- recording it as a gap and "
                        "continuing",
                        samples + 1,
                        type(exc).__name__,
                        exc,
                    )
                    used, raw = None, f"<{type(exc).__name__}: {exc}>"
                else:
                    used, raw = _parse_used_mib(stdout, gpu_index)
                    if used is None:
                        failures += 1

                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        f"{elapsed:.3f}",
                        "" if used is None else used,
                        raw,
                    ]
                )
                handle.flush()
                samples += 1
                if used is not None and (peak_mib is None or used > peak_mib):
                    peak_mib, peak_at = used, elapsed

                if duration_seconds is not None and clock() - start >= duration_seconds:
                    break
                sleeper(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted after %d sample(s)", samples)

    result = SampleRun(
        samples=samples,
        peak_mib=peak_mib,
        peak_at_elapsed=peak_at,
        failures=failures,
        out_file=path,
    )
    logger.info("VRAM sampling finished: %s", result.describe())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.vramsample",
        description=(
            "Poll nvidia-smi's memory.used into a CSV while an attended render runs "
            "(issue #98's 277-frame proof). Run it in its own terminal."
        ),
    )
    parser.add_argument("--out", required=True, help="CSV file to write (created/overwritten)")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"seconds between readings (default {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds (default: run until Ctrl-C)",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="which GPU's line to record")
    parser.add_argument(
        "--label",
        default="",
        help="recorded in the file's provenance header -- say which render this is",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = sample_vram(
        args.out,
        interval_seconds=args.interval,
        duration_seconds=args.duration,
        gpu_index=args.gpu_index,
        label=args.label,
    )
    return 0 if result.samples else 1


if __name__ == "__main__":  # pragma: no cover -- exercised through main()
    raise SystemExit(main())
