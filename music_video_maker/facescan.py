"""Face-presence scan over a directory of rendered chunks (issue #93).

This replaces a run-local script, ``~/mvm-runs/deathless/measurements/
face_scan.py``, that hardcoded its input directory as ``output/chunks``.
Only its *output filename* was ever updated between renders -- so
``face_presence_v12.csv``, written the day of the v12 render, turned out to
be byte-identical to ``face_presence.csv``, a scan of a completely different,
week-old render. The CSV's own name asserted a subject it had never
actually read. Two chunks the run-local scan reported at 0.0% face presence
(the headline evidence behind issue #93's "the detector can't see hats or
head tilt" claim) were, on the render the filename claimed to describe,
really at 91.7% and 58.3% -- the detector was never wrong; the input it was
pointed at was.

The general defect is not "someone typed the wrong path once". It is that
**a measurement artefact's filename is not provenance** -- nothing about the
CSV itself said what it had read, so nothing could catch the mismatch until
a human went and looked at the pixels by hand. This module's whole reason to
exist is to make that structurally impossible: every row records the
resolved absolute path, size, and mtime of the file it actually opened, and
the report as a whole records the resolved input directory, the detector
model and its sha256, the thresholds, and the samples-per-chunk. A stale- or
wrong-directory bug like #93's is visible on the first row of the CSV, not
two years and a viewer complaint later.

Superset of the run-local script's columns, so old numbers stay comparable:
``chunk_id, frames, sampled, with_face, face_pct, carries_identity,
max_face_fraction`` are unchanged; ``inconclusive`` is new (issue #93's other
half -- see :meth:`music_video_maker.faces.FaceObservation.verdict`), as are
the five provenance columns.

Provenance as a header, not repeated columns
---------------------------------------------
The fields that are constant for the whole report -- the input directory,
the detector model + its sha256, the score threshold, the inspection floor,
samples-per-chunk -- are written once, as ``#``-prefixed comment lines above
the CSV table, rather than repeated on every row. Two reasons: a spreadsheet
or ``pandas.read_csv(..., comment="#")`` sees one clean table with no
constant columns to ignore, and a report that scanned 80 chunks does not
carry the same sha256 eighty times. The **per-row** provenance (the file
this specific row came from) is a real column, because that is exactly the
field #93 showed the run-local script never had.

Two injectable seams, no video and no OpenCV required to test them
---------------------------------------------------------------------
``cv2`` is imported lazily, exactly the way :mod:`music_video_maker.faces`
does it, so importing this module -- and testing its row-building and CSV
logic -- never requires OpenCV to be installed. Frame extraction
(:data:`FrameExtractor`) and detection (:data:`Detector`) are both injectable
callables, the same shape :mod:`music_video_maker.luminance` uses for its
ffmpeg subprocess runner: :func:`scan_chunk` and :func:`main` take them as
arguments (or factories, for ``main``) rather than reaching for cv2
themselves, so the row-building logic, the CSV/provenance writing, and
``main``'s argument handling are all exercised with fakes in CI. Only
:func:`extract_sample_frames` and :func:`build_default_detector` touch
OpenCV, and both are skipped-if-absent in the test suite the same way
:mod:`music_video_maker.faces`'s real-detector tests are.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from music_video_maker import faces

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 12
"""Frames sampled per chunk, evenly spaced. Matches the run-local script
this module replaces, so historical numbers stay comparable."""

FIELDNAMES: tuple[str, ...] = (
    "chunk_id",
    "frames",
    "sampled",
    "with_face",
    "face_pct",
    "carries_identity",
    "inconclusive",
    "max_face_fraction",
    "source_path",
    "source_size_bytes",
    "source_mtime",
)
"""CSV columns. The first seven (minus ``inconclusive``, new in #93) are the
run-local script's own columns, unchanged and in the same order, so a diff
against old numbers is a column subset, not a rewrite. ``inconclusive`` and
the three ``source_*`` columns are new -- see the module docstring."""


# --------------------------------------------------------------------------- #
# Chunk discovery -- pure, no cv2, no video content read
# --------------------------------------------------------------------------- #


def _chunk_id_from_filename(path: Path) -> int:
    """The numeric chunk id embedded in ``path``'s filename (e.g.
    ``chunk_0007.mp4`` -> ``7``), the same rule the run-local script used."""
    digits = "".join(c for c in path.stem if c.isdigit())
    if not digits:
        raise ValueError(f"{path}: no digits in filename to use as a chunk id")
    return int(digits)


def discover_chunk_videos(chunks_dir: Path) -> list[Path]:
    """Every chunk video in ``chunks_dir``, sorted by numeric chunk id.

    Prefers the ``chunk_*.mp4`` naming convention; falls back to any
    ``*.mp4`` if none match (mirroring the run-local script). A chunk
    directory also holds ``chunk_NNN.wav`` audio stems, which the glob
    excludes by extension, and possibly the odd file with no digits in its
    name at all, which is logged and skipped -- it cannot be assigned a
    chunk id, so it cannot become a row.
    """
    candidates = sorted(chunks_dir.glob("chunk_*.mp4"))
    if not candidates:
        candidates = sorted(chunks_dir.glob("*.mp4"))

    videos: list[Path] = []
    for path in candidates:
        try:
            _chunk_id_from_filename(path)
        except ValueError:
            logger.warning(
                "facescan: %s has no digits in its filename -- skipping, cannot assign "
                "it a chunk id.",
                path,
            )
            continue
        videos.append(path)

    return sorted(videos, key=_chunk_id_from_filename)


# --------------------------------------------------------------------------- #
# Frame extraction -- the one part of this module that needs OpenCV
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampledFrames:
    """What :data:`FrameExtractor` returns for one chunk video."""

    total_frames: int
    """The video's total frame count (``0`` if it could not be determined)."""
    sample_paths: tuple[Path, ...]
    """Paths to the sampled frame images actually extracted -- may be fewer
    than requested if some frames could not be read."""


FrameExtractor = Callable[[Path, int], SampledFrames]
"""Given a chunk video path and a sample count, returns the frames sampled
from it. Injectable so :func:`scan_chunk` never has to know whether it is
talking to real OpenCV or a test fake."""

Detector = Callable[[Path], faces.FaceObservation]
"""Given a frame image path, returns the :class:`faces.FaceObservation` for
it. Injectable for the same reason as :data:`FrameExtractor`."""


def _import_cv2():
    """Lazy import, mirroring ``faces._import_cv2`` -- so importing this
    module, and exercising its row-building/CSV logic through the injectable
    seams above, never requires OpenCV to be installed."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by the message, not CI
        raise faces.FaceDetectionError(
            "OpenCV is not installed, so chunk video frames cannot be sampled. Install "
            "opencv-python-headless (the 'faces' extra) to run music_video_maker.facescan."
        ) from exc
    return cv2


def extract_sample_frames(video_path: Path, samples: int, *, tmp_dir: Path) -> SampledFrames:
    """Real frame extractor: ``cv2.VideoCapture`` + evenly spaced frame
    indices, each written to its own temp PNG under ``tmp_dir``.

    Same sampling rule the run-local script used:
    ``round(i * (total - 1) / (samples - 1))`` for ``i in range(samples)``,
    so numbers stay comparable to the historical CSVs. Raises
    :class:`faces.FaceDetectionError` if the video cannot be opened at all
    -- a chunk this fails on is skipped by :func:`main`, not fatal to the
    whole scan.
    """
    cv2 = _import_cv2()

    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logger.warning(
                "facescan: %s reports %d frames -- treating as unreadable, no samples "
                "taken.",
                video_path,
                total,
            )
            return SampledFrames(total_frames=max(total, 0), sample_paths=())
        if samples <= 0:
            return SampledFrames(total_frames=total, sample_paths=())

        if samples == 1:
            indices = [0]
        else:
            indices = [round(i * (total - 1) / (samples - 1)) for i in range(samples)]

        paths: list[Path] = []
        for n, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                logger.warning(
                    "facescan: %s frame index %d (sample %d/%d) could not be read -- "
                    "skipping this sample.",
                    video_path,
                    index,
                    n + 1,
                    samples,
                )
                continue
            frame_path = tmp_dir / f"{video_path.stem}_{index:06d}_{n:02d}.png"
            cv2.imwrite(str(frame_path), frame)
            paths.append(frame_path)

        return SampledFrames(total_frames=total, sample_paths=tuple(paths))
    finally:
        capture.release()


def build_default_extractor(tmp_dir: Path) -> FrameExtractor:
    """A :data:`FrameExtractor` closure over :func:`extract_sample_frames`,
    binding ``tmp_dir`` so callers (:func:`main`) don't have to thread it
    through every call."""

    def extractor(video_path: Path, samples: int) -> SampledFrames:
        return extract_sample_frames(video_path, samples, tmp_dir=tmp_dir)

    return extractor


def build_default_detector(
    *,
    model_path: Path | str | None = None,
    score_threshold: float = faces.DEFAULT_SCORE_THRESHOLD,
    inspect_floor: float | None = faces.DEFAULT_INSPECTION_FLOOR,
) -> Detector:
    """A :data:`Detector` closure over :func:`faces.detect_faces`.

    ``inspect_floor`` defaults ON here (unlike ``detect_faces`` itself,
    which defaults it off to keep the #47 gate's call shape untouched) --
    the whole point of this scan is to make a zero inspectable rather than
    trusted (issue #93), so every row this instrument writes carries the
    ``inconclusive`` distinction the run-local CSV never could.
    """

    def detector(frame_path: Path) -> faces.FaceObservation:
        return faces.detect_faces(
            frame_path,
            model_path=model_path,
            score_threshold=score_threshold,
            inspect_floor=inspect_floor,
        )

    return detector


# --------------------------------------------------------------------------- #
# Row building -- pure given its two injected callables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkFaceScan:
    """One row of the report -- what was measured for one chunk video, plus
    the provenance of the file it was measured from."""

    chunk_id: int
    frames: int
    sampled: int
    with_face: int
    face_pct: float
    carries_identity: int
    inconclusive: int
    max_face_fraction: float
    source_path: Path
    """Resolved absolute path of the file this row actually scanned -- the
    field #93's run-local script never recorded, which is why a stale
    directory produced a CSV indistinguishable from a correct one."""
    source_size_bytes: int
    source_mtime: str
    """ISO-8601 UTC timestamp of the source file's mtime."""


def _format_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def scan_chunk(
    video_path: Path,
    *,
    samples: int,
    extractor: FrameExtractor,
    detector: Detector,
) -> ChunkFaceScan:
    """Sample ``video_path`` via ``extractor`` and score each frame via
    ``detector``, producing one report row.

    A frame the detector could not check (:class:`faces.FaceDetectionError`)
    is logged and skipped -- it does not count toward ``sampled`` or any
    other tally, and it never aborts the rest of the chunk's row. This
    mirrors the project-wide rule that one failure must not take down a
    whole run, scaled down to a single sampled frame.
    """
    chunk_id = _chunk_id_from_filename(video_path)
    extracted = extractor(video_path, samples)

    with_face = 0
    carries_identity = 0
    inconclusive = 0
    max_fraction = 0.0
    checked = 0

    for frame_path in extracted.sample_paths:
        try:
            observation = detector(frame_path)
        except faces.FaceDetectionError:
            logger.warning(
                "facescan: chunk %d frame %s could not be checked for a face -- "
                "skipping this sample.",
                chunk_id,
                frame_path,
            )
            continue

        checked += 1
        if observation.face_count > 0:
            with_face += 1
        if observation.carries_identity():
            carries_identity += 1
        if observation.verdict == "inconclusive":
            inconclusive += 1
        max_fraction = max(max_fraction, observation.largest_fraction)

    sampled = len(extracted.sample_paths)
    face_pct = round(100.0 * with_face / sampled, 1) if sampled else 0.0

    stat = video_path.stat()
    return ChunkFaceScan(
        chunk_id=chunk_id,
        frames=extracted.total_frames,
        sampled=sampled,
        with_face=with_face,
        face_pct=face_pct,
        carries_identity=carries_identity,
        inconclusive=inconclusive,
        max_face_fraction=round(max_fraction, 4),
        source_path=video_path.resolve(),
        source_size_bytes=stat.st_size,
        source_mtime=_format_mtime(stat.st_mtime),
    )


# --------------------------------------------------------------------------- #
# The report: a provenance header + the CSV table
# --------------------------------------------------------------------------- #


def write_report(
    rows: Sequence[ChunkFaceScan],
    out: TextIO,
    *,
    input_dir: Path,
    score_threshold: float,
    inspect_floor: float | None,
    samples: int,
) -> None:
    """Write the report's provenance header and CSV table to ``out``.

    The header -- the fields constant for the whole report -- is written as
    ``#``-prefixed comment lines, which both a spreadsheet and
    ``pandas.read_csv(..., comment="#")`` skip cleanly; see the module
    docstring for why this is a header rather than repeated columns.
    """
    out.write("# music_video_maker.facescan report (issue #93)\n")
    out.write(f"# input_dir={input_dir.resolve()}\n")
    out.write(f"# detector_model={faces.MODEL_FILENAME} sha256={faces.MODEL_SHA256}\n")
    out.write(
        "# score_threshold={score_threshold} inspection_floor={inspection_floor} "
        "samples_per_chunk={samples}\n".format(
            score_threshold=score_threshold,
            inspection_floor=inspect_floor if inspect_floor is not None else "none",
            samples=samples,
        )
    )

    writer = csv.writer(out)
    writer.writerow(FIELDNAMES)
    for row in rows:
        writer.writerow(
            [
                row.chunk_id,
                row.frames,
                row.sampled,
                row.with_face,
                row.face_pct,
                row.carries_identity,
                row.inconclusive,
                row.max_face_fraction,
                row.source_path,
                row.source_size_bytes,
                row.source_mtime,
            ]
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.facescan",
        description=(
            "Scan a directory of rendered chunk videos for face presence, with the "
            "input's own provenance recorded in the report (issue #93)."
        ),
    )
    parser.add_argument("chunks_dir", type=Path, help="directory containing chunk_*.mp4 files")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="frames sampled per chunk (evenly spaced)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="CSV output path (default: stdout)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=faces.DEFAULT_SCORE_THRESHOLD,
        help="YuNet confidence floor for the primary detection decision",
    )
    parser.add_argument(
        "--inspection-floor",
        type=float,
        default=faces.DEFAULT_INSPECTION_FLOOR,
        help="score floor for the second, inspection-only detector call",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None, help="override the YuNet model file location"
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    extractor_factory: Callable[[Path], FrameExtractor] = build_default_extractor,
    detector_factory: Callable[..., Detector] = build_default_detector,
) -> int:
    """Entry point for ``python -m music_video_maker.facescan``.

    ``extractor_factory``/``detector_factory`` are overridable purely so
    tests can drive the whole CLI -- argument parsing, chunk discovery,
    per-chunk degradation, report writing -- with fakes and no OpenCV. Real
    runs never pass them; the defaults are the real cv2-backed
    implementations.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    chunks_dir: Path = args.chunks_dir
    if not chunks_dir.is_dir():
        logger.error("facescan: %s is not a directory.", chunks_dir)
        return 1

    videos = discover_chunk_videos(chunks_dir)
    if not videos:
        logger.error(
            "facescan: no chunk_*.mp4 (or *.mp4) files found in %s -- nothing to scan.",
            chunks_dir,
        )
        return 1

    detector = detector_factory(
        model_path=args.model_path,
        score_threshold=args.score_threshold,
        inspect_floor=args.inspection_floor,
    )

    rows: list[ChunkFaceScan] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor = extractor_factory(Path(tmp_dir))
        for video in videos:
            try:
                rows.append(
                    scan_chunk(video, samples=args.samples, extractor=extractor, detector=detector)
                )
            except faces.FaceDetectionError:
                logger.exception(
                    "facescan: chunk video %s could not be scanned -- skipping it. This "
                    "must not abort the rest of the scan.",
                    video,
                )
                continue

    if not rows:
        logger.error(
            "facescan: every chunk video in %s failed to scan (see prior errors) -- "
            "nothing to write.",
            chunks_dir,
        )
        return 1

    report_kwargs = {
        "input_dir": chunks_dir,
        "score_threshold": args.score_threshold,
        "inspect_floor": args.inspection_floor,
        "samples": args.samples,
    }

    if args.out is not None:
        with args.out.open("w", newline="") as fh:
            write_report(rows, fh, **report_kwargs)
        logger.info("facescan: wrote %d row(s) to %s.", len(rows), args.out)
    else:
        write_report(rows, sys.stdout, **report_kwargs)

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main(), not this guard
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SAMPLES",
    "FIELDNAMES",
    "ChunkFaceScan",
    "Detector",
    "FrameExtractor",
    "SampledFrames",
    "build_default_detector",
    "build_default_extractor",
    "build_parser",
    "discover_chunk_videos",
    "extract_sample_frames",
    "main",
    "scan_chunk",
    "write_report",
]
