"""Print every placed segment's voicing statistics for one track (issue #96).

``alignment_quality``'s voiced-periodicity check ships with a threshold
(``VOICING_RATIO_THRESHOLD``) that has **not** been calibrated against a real
master, because the machine it was written on does not hold one. This is the
one command that calibrates it, on the machine that does::

    python -m music_video_maker.calibrate_voicing --config run_v13.toml

It runs Stage 1 exactly as a render would (same ``alignment_model_size``, same
``alignment_overrides``), decodes the master with the same ffmpeg argv the
check uses, measures every placed segment with the same function the check
calls, and prints one row per segment sorted with the most suspicious first.
No GPU, no ComfyUI, no model call beyond the aligner the pipeline already
runs; on an 8-minute song it is the alignment's own ~6 s plus a few seconds
of arithmetic.

What to look at
---------------
``vf_ratio`` is the number the shipped check thresholds: a segment's voiced
fraction over the median across the track's other measurable placed segments.
A phantom over drums, hiss or room tone sinks toward 0. A phantom over a
**sustained pitched instrument does not**, and that is the documented limit
of periodicity rather than a threshold to chase -- so ``jitter%`` and
``f0_span`` are printed beside it. A voice is unstable (vibrato, micro-jitter,
note changes); a plucked or bowed string is metronomic. Those two columns are
reported and deliberately never thresholded until a track with a known answer
says what they do; see ``docs/voicing-corpus.md``, which is where the answer
goes.

``--window 228.590-230.150`` scores an arbitrary span against the same decode
and the same level floor, so a window an operator has identified by ear can be
ranked against the segments without re-measuring the track differently.

Provenance, not a filename
--------------------------
Every run prints the master's resolved path, size and mtime, where the
segments came from, and every constant in force (#93: a measurement artefact
must not be able to assert a subject it never opened). ``--csv`` writes the
same header as comment lines above the rows.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from music_video_maker import alignment_quality, voicing
from music_video_maker.contracts import AlignedSegment

logger = logging.getLogger(__name__)

_COLUMNS = (
    "idx",
    "start",
    "end",
    "dur",
    "lvl_dBFS",
    "frames",
    "vf",
    "vf_ratio",
    "nccf",
    "hnr_dB",
    "f0_Hz",
    "jitter%",
    "f0_span",
    "text",
)

Decoder = Callable[[Path, Path], None]
SegmentSource = Callable[[Path], "tuple[Path, tuple[AlignedSegment, ...]]"]


class CalibrationError(RuntimeError):
    """Bad input: an unreadable config, an unparseable window, a missing file.

    Raised and turned into an exit code by :func:`main`; never a traceback in
    an operator's terminal."""


def _ffmpeg_decode(master: Path, out_path: Path) -> None:
    """Decode ``master`` to mono 16kHz PCM using the check's own argv."""
    if shutil.which("ffmpeg") is None:
        raise CalibrationError("ffmpeg is not on PATH; this command decodes the master with it")
    args = alignment_quality.decode_to_mono_16khz_args(master, out_path)
    proc = subprocess.run(args, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise CalibrationError(
            f"ffmpeg exited {proc.returncode} decoding {master}: {stderr[-400:]}"
        )


def _segments_from_config(config_path: Path) -> tuple[Path, tuple[AlignedSegment, ...]]:
    """Run Stage 1 the way a render does, and return its master and segments.

    Imported here rather than at module scope so that ``--segments`` works on
    a machine with no aligner installed: this is a diagnostic, and refusing to
    print a table because an optional heavy extra is missing would be the
    wrong failure."""
    from music_video_maker.alignment import align
    from music_video_maker.config import ConfigError, load_config
    from music_video_maker.lyrics import parse_lyrics

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise CalibrationError(f"could not load {config_path}: {exc}") from exc
    lines = parse_lyrics(config.lyrics_file, config.cast, config.default_lead_vocalist)
    result = align(
        config.master_audio,
        lines,
        model_size=config.alignment_model_size,
        overrides=config.alignment_overrides,
    )
    return Path(config.master_audio), result.segments


def _segments_from_json(path: Path) -> tuple[AlignedSegment, ...]:
    """Read segments from a JSON list of ``{index, text, start, end}`` objects.

    The escape hatch for a machine without the aligner, and the seam the unit
    tests use. A ``{"segments": [...]}`` wrapper is accepted too, because that
    is the shape anything dumping an ``AlignmentResult`` tends to produce."""
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"could not read segments from {path}: {exc}") from exc
    rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CalibrationError(f"{path} does not hold a list of segments")
    segments: list[AlignedSegment] = []
    for position, row in enumerate(rows):
        try:
            segments.append(
                AlignedSegment(
                    index=int(row.get("index", position)),
                    text=str(row.get("text", "")),
                    start=float(row["start"]),
                    end=float(row["end"]),
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"{path} segment {position} is malformed: {exc}") from exc
    return tuple(segments)


def _parse_window(text: str) -> tuple[float, float]:
    separator = "-" if "-" in text else ":"
    parts = text.split(separator)
    if len(parts) != 2:
        raise CalibrationError(f"--window {text!r} is not START-END (e.g. 228.590-230.150)")
    try:
        start, end = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise CalibrationError(f"--window {text!r} is not two numbers: {exc}") from exc
    if end <= start:
        raise CalibrationError(f"--window {text!r} ends at or before it starts")
    return start, end


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _row(
    label: str,
    measurement: voicing.SegmentVoicing,
    baseline: float,
    text: str,
) -> tuple[str, ...]:
    ratio = measurement.voiced_fraction / baseline if baseline > 0 else float("nan")
    return (
        label,
        f"{measurement.start:.3f}",
        f"{measurement.end:.3f}",
        f"{measurement.end - measurement.start:.2f}",
        f"{measurement.level_dbfs:.1f}",
        f"{measurement.frames_measured}/{measurement.frames_total}",
        f"{measurement.voiced_fraction:.3f}",
        f"{ratio:.2f}",
        f"{measurement.median_nccf:.3f}",
        f"{measurement.hnr_db:.1f}",
        _fmt(measurement.median_f0_hz, ".0f"),
        _fmt(measurement.f0_jitter_pct, ".2f"),
        _fmt(measurement.f0_span_semitones, ".2f"),
        text.strip()[:44],
    )


def _provenance(master: Path, source: str, floor_dbfs: float, judged: int) -> list[str]:
    try:
        stat = master.stat()
        size, mtime = stat.st_size, datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    except OSError:
        size, mtime = -1, "unknown"
    return [
        f"master           {master.resolve()}",
        f"master size      {size} bytes, mtime {mtime}",
        f"segments from    {source}",
        f"level floor      {floor_dbfs:.2f} dBFS "
        f"(track median placed-segment RMS - {alignment_quality.VOICING_FRAME_LEVEL_DROP_DB} dB)",
        f"judged segments  {judged} "
        f"(>= {alignment_quality.MIN_SEGMENT_DURATION_FOR_VOCAL_ENERGY_S}s long and "
        f">= {alignment_quality.VOICING_MIN_MEASURED_FRAMES} audible frames)",
        f"shipped threshold vf_ratio < {alignment_quality.VOICING_RATIO_THRESHOLD} "
        "(UNCALIBRATED -- this run is the calibration)",
        f"detector         NCCF {voicing.MIN_F0_HZ:.0f}-{voicing.MAX_F0_HZ:.0f}Hz at "
        f"{voicing.ANALYSIS_RATE_HZ}Hz, {voicing.CORRELATION_WINDOW_S * 1000:.0f}ms window / "
        f"{voicing.HOP_S * 1000:.0f}ms hop, voiced at NCCF >= {voicing.VOICED_FRAME_NCCF}",
    ]


def _print_table(rows: Sequence[tuple[str, ...]]) -> None:
    widths = [
        max(len(_COLUMNS[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(_COLUMNS))
    ]
    header = "  ".join(name.ljust(widths[i]) for i, name in enumerate(_COLUMNS))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def calibrate(
    master: Path,
    segments: Sequence[AlignedSegment],
    *,
    source_label: str,
    windows: Sequence[tuple[float, float]] = (),
    csv_path: Path | None = None,
    decoder: Decoder = _ffmpeg_decode,
) -> int:
    """Measure and print. Returns a process exit code."""
    if not segments:
        raise CalibrationError("no aligned segments to measure")
    if not master.exists():
        raise CalibrationError(f"master audio {master} does not exist")

    with tempfile.TemporaryDirectory(prefix="mvm-voicing-") as tmp_dir:
        decoded = Path(tmp_dir) / "master-mono16k.wav"
        decoder(master, decoded)
        try:
            track = alignment_quality.measure_placed_segment_voicing(decoded, segments)
        except voicing.VoicingUnavailable as exc:
            raise CalibrationError(str(exc)) from exc

        judged = {
            index: m
            for index, m in track.by_segment_index.items()
            if m.frames_measured >= alignment_quality.VOICING_MIN_MEASURED_FRAMES
        }
        if not judged:
            raise CalibrationError(
                "no placed segment had enough audible analysis frames to measure; check that "
                f"{master} is the right master for these segments"
            )
        fractions = sorted(m.voiced_fraction for m in judged.values())
        baseline = fractions[len(fractions) // 2]
        if baseline <= 0.0:
            # The level floor is derived from the track's own median, so a
            # track with nothing periodic anywhere clears its own gate and
            # every ratio comes out as a division by zero. Refusing is the
            # honest answer: printing a column of `nan` in the number an
            # operator is about to move a threshold on is exactly the kind of
            # measurement artefact #93 exists to stop. The shipped check makes
            # the same call (``if baseline <= 0: continue``).
            raise CalibrationError(
                f"not one placed segment on {master} is periodic, so there is no median to "
                "rank anything against -- this is the wrong master for these segments, or a "
                "track with no voice on it at all. The shipped check skips for the same "
                "reason rather than reporting every segment as a phantom"
            )

        texts = {s.index: s.text for s in segments}
        rows = [
            _row(str(index), judged[index], baseline, texts.get(index, ""))
            for index in sorted(judged, key=lambda i: judged[i].voiced_fraction)
        ]
        window_rows: list[tuple[str, ...]] = []
        for start, end in windows:
            measurement = voicing.measure_span(
                track.audio, start, end, level_floor_dbfs=track.level_floor_dbfs
            )
            if measurement is None:
                window_rows.append(
                    ("window", f"{start:.3f}", f"{end:.3f}", "-", "-", "0/0", "-", "-", "-", "-",
                     "-", "-", "-", "too short to measure")
                )
            else:
                window_rows.append(_row("window", measurement, baseline, "(--window)"))

        provenance = _provenance(master, source_label, track.level_floor_dbfs, len(judged))
        for line in provenance:
            print(line)
        print(f"median voiced fraction (the vf_ratio denominator)  {baseline:.3f}")
        print()
        _print_table(rows + window_rows)
        flagged = [
            index
            for index in sorted(judged)
            if baseline > 0
            and judged[index].voiced_fraction / baseline
            < alignment_quality.VOICING_RATIO_THRESHOLD
        ]
        print()
        print(
            f"the shipped threshold would flag {len(flagged)} of {len(judged)} segment(s): "
            f"{flagged}"
        )
        print(
            "Read jitter% and f0_span before moving the threshold: periodicity cannot "
            "separate a sung vowel from a sustained pitched instrument, and those two "
            "columns are the axis that might. Record the result in docs/voicing-corpus.md."
        )

        if csv_path is not None:
            _write_csv(csv_path, provenance, rows + window_rows)
            print(f"wrote {csv_path}")
    return 0


def _write_csv(
    csv_path: Path, provenance: Sequence[str], rows: Sequence[tuple[str, ...]]
) -> None:
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            for line in provenance:
                handle.write(f"# {line}\n")
            writer = csv.writer(handle)
            writer.writerow(_COLUMNS)
            writer.writerows(rows)
    except OSError as exc:
        raise CalibrationError(f"could not write {csv_path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.calibrate_voicing",
        description=(
            "Print per-segment voiced-vs-unvoiced statistics for one track (issue #96), so "
            "alignment_quality.VOICING_RATIO_THRESHOLD can be calibrated against a real "
            "master instead of assumed."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", type=Path, help="run TOML; aligns exactly as a render would, then measures"
    )
    source.add_argument(
        "--segments",
        type=Path,
        help="JSON list of {index, text, start, end}; use with --master when the aligner is "
        "not installed here",
    )
    parser.add_argument("--master", type=Path, help="master audio (required with --segments)")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START-END",
        help="also score this arbitrary span against the same decode and level floor; "
        "repeatable",
    )
    parser.add_argument("--csv", type=Path, help="also write the table here, with provenance")
    return parser


def main(argv: Sequence[str] | None = None, *, decoder: Decoder = _ffmpeg_decode) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        windows = [_parse_window(text) for text in args.window]
        if args.config is not None:
            master, segments = _segments_from_config(args.config)
            source_label = f"{args.config} (aligned by this command)"
        else:
            if args.master is None:
                raise CalibrationError("--segments needs --master")
            master, segments = args.master, _segments_from_json(args.segments)
            source_label = str(args.segments)
        return calibrate(
            master,
            segments,
            source_label=source_label,
            windows=windows,
            csv_path=args.csv,
            decoder=decoder,
        )
    except CalibrationError as exc:
        logger.error("calibrate_voicing: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
