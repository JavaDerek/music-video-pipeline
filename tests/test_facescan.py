"""Tests for the in-repo face-presence scan instrument (issue #93).

#93 was filed as a YuNet recall bug -- "0.0% face presence on two obvious
faces". Measurement showed the detector was never the problem: the run-local
``measurements/face_scan.py`` script that produced those numbers hardcoded
its input directory, so a CSV *named* for the v12 render was actually a
byte-identical re-save of a scan of a week-old, different render. The
filename asserted a subject the scan had never read.

This module is the fix: bringing the instrument into the repo with
structural provenance -- every row records the resolved absolute path, size
and mtime of the file it actually opened, and the report as a whole records
the resolved input directory, the detector model + its sha256, the
thresholds, and the samples-per-chunk. A stale-directory bug like #93's
becomes visible on the first row, not after a viewer notices two years
later.

Three layers, matching ``tests/test_luminance.py``'s shape:

* frame-sampling and row-building logic, exercised with fake extractor/
  detector callables -- no cv2, no video, no model file;
* ``main()`` end to end, with fake factories injected -- no cv2, no video;
* the real detector/extractor against real fixtures and (optionally) a real
  chunk directory, skipped when OpenCV/the model/the directory are absent.
"""

from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker import faces, facescan

CALIBRATION = Path(__file__).parent / "fixtures" / "seed_frames"

# --------------------------------------------------------------------------- #
# Pure logic: chunk-id parsing and file discovery (no cv2, no video)
# --------------------------------------------------------------------------- #


def test_chunk_id_from_filename_extracts_digits():
    assert facescan._chunk_id_from_filename(Path("chunk_0007.mp4")) == 7
    assert facescan._chunk_id_from_filename(Path("chunk_0029.mp4")) == 29


def test_chunk_id_from_filename_raises_with_no_digits():
    with pytest.raises(ValueError, match="no digits"):
        facescan._chunk_id_from_filename(Path("mystery.mp4"))


def test_discover_chunk_videos_sorts_numerically_not_lexically(tmp_path):
    for name in ("chunk_0002.mp4", "chunk_0010.mp4", "chunk_0001.mp4"):
        (tmp_path / name).write_bytes(b"")
    videos = facescan.discover_chunk_videos(tmp_path)
    assert [p.name for p in videos] == ["chunk_0001.mp4", "chunk_0002.mp4", "chunk_0010.mp4"]


def test_discover_chunk_videos_falls_back_to_any_mp4_when_no_chunk_prefix(tmp_path):
    (tmp_path / "clip_2.mp4").write_bytes(b"")
    (tmp_path / "clip_1.mp4").write_bytes(b"")
    videos = facescan.discover_chunk_videos(tmp_path)
    assert {p.name for p in videos} == {"clip_1.mp4", "clip_2.mp4"}


def test_discover_chunk_videos_skips_files_it_cannot_assign_a_chunk_id_to(tmp_path, caplog):
    (tmp_path / "chunk_0005.mp4").write_bytes(b"")
    (tmp_path / "chunk_no_id.mp4").write_bytes(b"")
    with caplog.at_level(logging.WARNING):
        videos = facescan.discover_chunk_videos(tmp_path)
    assert [p.name for p in videos] == ["chunk_0005.mp4"]
    assert "chunk_no_id.mp4" in caplog.text


def test_discover_chunk_videos_ignores_non_video_siblings(tmp_path):
    """A chunks_v12-shaped directory also holds ``chunk_NNN.wav`` audio
    stems -- the glob must never pick those up."""
    (tmp_path / "chunk_0000.mp4").write_bytes(b"")
    (tmp_path / "chunk_000.wav").write_bytes(b"")
    videos = facescan.discover_chunk_videos(tmp_path)
    assert [p.name for p in videos] == ["chunk_0000.mp4"]


# --------------------------------------------------------------------------- #
# Row building: fake extractor + fake detector, no cv2, no video, no model
# --------------------------------------------------------------------------- #


class _FakeExtractor:
    """Records calls; returns a scripted SampledFrames for each video."""

    def __init__(self, plan: dict[Path, facescan.SampledFrames]):
        self.plan = plan
        self.calls: list[tuple[Path, int]] = []

    def __call__(self, video_path: Path, samples: int) -> facescan.SampledFrames:
        self.calls.append((video_path, samples))
        return self.plan[video_path]


class _FakeDetector:
    """Maps a frame path to a scripted FaceObservation, or an exception."""

    def __init__(self, plan: dict[Path, faces.FaceObservation | Exception]):
        self.plan = plan
        self.calls: list[Path] = []

    def __call__(self, frame_path: Path) -> faces.FaceObservation:
        self.calls.append(frame_path)
        outcome = self.plan[frame_path]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_scan_chunk_counts_face_presence_identity_and_inconclusive(tmp_path):
    video = tmp_path / "chunk_0007.mp4"
    video.write_bytes(b"fake video bytes")
    frame_a, frame_b, frame_c = (tmp_path / f"f{i}.png" for i in range(3))

    extractor = _FakeExtractor(
        {video: facescan.SampledFrames(total_frames=90, sample_paths=(frame_a, frame_b, frame_c))}
    )
    detector = _FakeDetector(
        {
            frame_a: faces.FaceObservation(1, 0.02, 0.95),  # detected, carries identity
            frame_b: faces.FaceObservation(
                0, 0.0, 0.0, candidate_scores=(0.253,), inspection_floor=0.15
            ),  # inconclusive
            frame_c: faces.FaceObservation(
                0, 0.0, 0.0, candidate_scores=(), inspection_floor=0.15
            ),  # absent
        }
    )

    row = facescan.scan_chunk(video, samples=3, extractor=extractor, detector=detector)

    assert row.chunk_id == 7
    assert row.frames == 90
    assert row.sampled == 3
    assert row.with_face == 1
    assert row.face_pct == pytest.approx(100.0 / 3, abs=0.05)
    assert row.carries_identity == 1
    assert row.inconclusive == 1
    assert row.max_face_fraction == pytest.approx(0.02)
    assert extractor.calls == [(video, 3)]
    assert detector.calls == [frame_a, frame_b, frame_c]


def test_scan_chunk_records_real_source_provenance(tmp_path):
    video = tmp_path / "chunk_0001.mp4"
    video.write_bytes(b"0123456789")  # 10 bytes, a known size to assert against

    extractor = _FakeExtractor({video: facescan.SampledFrames(total_frames=10, sample_paths=())})
    detector = _FakeDetector({})

    row = facescan.scan_chunk(video, samples=1, extractor=extractor, detector=detector)

    stat = video.stat()
    assert row.source_path == video.resolve()
    assert row.source_size_bytes == 10 == stat.st_size
    assert row.source_mtime  # non-empty, provenance recorded
    assert isinstance(row.source_mtime, str)


def test_scan_chunk_handles_zero_sampled_frames_without_dividing_by_zero(tmp_path):
    video = tmp_path / "chunk_0002.mp4"
    video.write_bytes(b"")
    extractor = _FakeExtractor({video: facescan.SampledFrames(total_frames=0, sample_paths=())})
    detector = _FakeDetector({})

    row = facescan.scan_chunk(video, samples=12, extractor=extractor, detector=detector)
    assert row.sampled == 0
    assert row.face_pct == 0.0
    assert row.with_face == 0
    assert row.max_face_fraction == 0.0


def test_scan_chunk_skips_a_frame_the_detector_could_not_check_and_keeps_going(tmp_path, caplog):
    """One frame failing must not lose the rest of the chunk's row -- the
    same 'one failure must not kill the run' shape the project uses
    everywhere else, scaled down to a single sampled frame."""
    video = tmp_path / "chunk_0003.mp4"
    video.write_bytes(b"x")
    frame_ok, frame_bad = tmp_path / "ok.png", tmp_path / "bad.png"

    extractor = _FakeExtractor(
        {video: facescan.SampledFrames(total_frames=24, sample_paths=(frame_ok, frame_bad))}
    )
    detector = _FakeDetector(
        {
            frame_ok: faces.FaceObservation(1, 0.05, 0.93),
            frame_bad: faces.FaceDetectionError("could not read frame"),
        }
    )

    with caplog.at_level(logging.WARNING):
        row = facescan.scan_chunk(video, samples=2, extractor=extractor, detector=detector)

    assert row.sampled == 2
    assert row.with_face == 1
    assert "bad.png" in caplog.text or str(frame_bad) in caplog.text


# --------------------------------------------------------------------------- #
# The provenance report: header + CSV rows
# --------------------------------------------------------------------------- #


def test_write_report_header_carries_provenance_once():
    row = facescan.ChunkFaceScan(
        chunk_id=1,
        frames=10,
        sampled=2,
        with_face=1,
        face_pct=50.0,
        carries_identity=1,
        inconclusive=0,
        max_face_fraction=0.01,
        source_path=Path("/abs/chunk_0001.mp4"),
        source_size_bytes=123,
        source_mtime="2026-08-22T12:00:00+00:00",
    )
    buf = io.StringIO()
    facescan.write_report(
        [row],
        buf,
        input_dir=Path("/abs/chunks_v12"),
        score_threshold=0.9,
        inspect_floor=0.15,
        samples=12,
    )
    text = buf.getvalue()
    header_lines = [line for line in text.splitlines() if line.startswith("#")]

    assert any("/abs/chunks_v12" in line for line in header_lines)
    assert any(faces.MODEL_FILENAME in line and faces.MODEL_SHA256 in line for line in header_lines)
    assert any("0.9" in line for line in header_lines)
    assert any("0.15" in line for line in header_lines)
    assert any("12" in line for line in header_lines)


def test_write_report_records_inspection_floor_as_none_when_absent():
    buf = io.StringIO()
    facescan.write_report(
        [], buf, input_dir=Path("/abs/chunks"), score_threshold=0.9, inspect_floor=None, samples=12
    )
    text = buf.getvalue()
    assert "none" in text.lower()


def test_write_report_rows_are_a_superset_of_the_run_local_script_columns():
    """The run-local face_scan.py this replaces wrote:
    chunk_id, frames, sampled, with_face, face_pct, carries_identity,
    max_face_fraction -- every one of those must still be a column, so old
    numbers stay comparable."""
    row = facescan.ChunkFaceScan(
        chunk_id=76,
        frames=175,
        sampled=12,
        with_face=7,
        face_pct=58.3,
        carries_identity=5,
        inconclusive=2,
        max_face_fraction=0.093,
        source_path=Path("/abs/chunk_0076.mp4"),
        source_size_bytes=999,
        source_mtime="2026-08-22T08:25:00+00:00",
    )
    buf = io.StringIO()
    facescan.write_report(
        [row],
        buf,
        input_dir=Path("/abs/chunks_v12"),
        score_threshold=0.9,
        inspect_floor=0.15,
        samples=12,
    )
    lines = [line for line in buf.getvalue().splitlines() if not line.startswith("#")]
    reader = csv.DictReader(lines)
    legacy_columns = {
        "chunk_id",
        "frames",
        "sampled",
        "with_face",
        "face_pct",
        "carries_identity",
        "max_face_fraction",
    }
    assert legacy_columns.issubset(set(reader.fieldnames))
    data = list(reader)
    assert len(data) == 1
    assert data[0]["chunk_id"] == "76"
    assert data[0]["source_path"] == "/abs/chunk_0076.mp4"
    assert data[0]["source_size_bytes"] == "999"
    assert data[0]["inconclusive"] == "2"


# --------------------------------------------------------------------------- #
# main(), end to end, with injected factories -- no cv2, no video
# --------------------------------------------------------------------------- #


def _factories_for(
    extractor: facescan.FrameExtractor, detector: facescan.Detector
):
    def extractor_factory(tmp_dir: Path) -> facescan.FrameExtractor:
        return extractor

    def detector_factory(**_kwargs) -> facescan.Detector:
        return detector

    return extractor_factory, detector_factory


def test_main_writes_a_report_to_the_requested_path(tmp_path, capsys):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    video = chunks_dir / "chunk_0000.mp4"
    video.write_bytes(b"v")

    frame = tmp_path / "frame.png"
    extractor = _FakeExtractor(
        {video: facescan.SampledFrames(total_frames=5, sample_paths=(frame,))}
    )
    detector = _FakeDetector({frame: faces.FaceObservation(1, 0.03, 0.97)})
    extractor_factory, detector_factory = _factories_for(extractor, detector)

    out_path = tmp_path / "report.csv"
    rc = facescan.main(
        [str(chunks_dir), "--samples", "1", "--out", str(out_path)],
        extractor_factory=extractor_factory,
        detector_factory=detector_factory,
    )

    assert rc == 0
    text = out_path.read_text()
    assert str(chunks_dir.resolve()) in text
    reader = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    rows = list(reader)
    assert rows[0]["chunk_id"] == "0"
    assert rows[0]["with_face"] == "1"


def test_main_writes_to_stdout_when_no_out_given(tmp_path, capsys):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    video = chunks_dir / "chunk_0000.mp4"
    video.write_bytes(b"v")

    extractor = _FakeExtractor({video: facescan.SampledFrames(total_frames=1, sample_paths=())})
    detector = _FakeDetector({})
    extractor_factory, detector_factory = _factories_for(extractor, detector)

    rc = facescan.main(
        [str(chunks_dir)],
        extractor_factory=extractor_factory,
        detector_factory=detector_factory,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "chunk_id" in out


def test_main_returns_nonzero_when_the_directory_does_not_exist(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        rc = facescan.main([str(tmp_path / "does_not_exist")])
    assert rc != 0


def test_main_returns_nonzero_when_no_chunk_videos_are_found(tmp_path, caplog):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with caplog.at_level(logging.ERROR):
        rc = facescan.main([str(empty_dir)])
    assert rc != 0


def test_main_skips_a_chunk_the_extractor_could_not_open_and_continues(tmp_path, caplog):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    good = chunks_dir / "chunk_0000.mp4"
    bad = chunks_dir / "chunk_0001.mp4"
    good.write_bytes(b"v")
    bad.write_bytes(b"v")

    frame = tmp_path / "frame.png"

    class _RaisingExtractor(_FakeExtractor):
        def __call__(self, video_path: Path, samples: int) -> facescan.SampledFrames:
            if video_path == bad:
                raise faces.FaceDetectionError("corrupt video")
            return super().__call__(video_path, samples)

    extractor = _RaisingExtractor(
        {good: facescan.SampledFrames(total_frames=1, sample_paths=(frame,))}
    )
    detector = _FakeDetector({frame: faces.FaceObservation(0, 0.0, 0.0)})
    extractor_factory, detector_factory = _factories_for(extractor, detector)

    out_path = tmp_path / "report.csv"
    with caplog.at_level(logging.ERROR):
        rc = facescan.main(
            [str(chunks_dir), "--out", str(out_path)],
            extractor_factory=extractor_factory,
            detector_factory=detector_factory,
        )
    assert rc == 0
    reader = csv.DictReader(
        line for line in out_path.read_text().splitlines() if not line.startswith("#")
    )
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "0"
    assert "chunk_0001" in caplog.text or "corrupt video" in caplog.text


# --------------------------------------------------------------------------- #
# The real detector/extractor (needs cv2 + the model; skipped otherwise)
# --------------------------------------------------------------------------- #


def test_build_default_detector_matches_faces_detect_faces_directly():
    pytest.importorskip("cv2")
    if not faces.resolve_model_path().exists():
        pytest.skip("YuNet model not available")

    detector = facescan.build_default_detector()
    path = CALIBRATION / "seed_frontal_chunk15.png"
    direct = faces.detect_faces(path, inspect_floor=faces.DEFAULT_INSPECTION_FLOOR)
    via_detector = detector(path)
    assert via_detector.face_count == direct.face_count
    assert via_detector.largest_fraction == direct.largest_fraction
    assert via_detector.inspection_floor == faces.DEFAULT_INSPECTION_FLOOR


@pytest.mark.integration
def test_extract_sample_frames_against_a_real_synthetic_clip(tmp_path):
    pytest.importorskip("cv2")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    video = tmp_path / "clip.mp4"
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:d=1:r=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    result = facescan.extract_sample_frames(video, 3, tmp_dir=tmp_path)
    assert result.total_frames > 0
    assert 1 <= len(result.sample_paths) <= 3
    for path in result.sample_paths:
        assert path.exists()


# --------------------------------------------------------------------------- #
# Optional integration test against the real "Deathless" v12 render.
#
# Never copies media into the repo: it symlinks a couple of real chunk files
# into pytest's tmp_path (a system temp directory), not into the checkout,
# purely so the scan stays fast (the real directory has ~80 chunks). Skips
# cleanly when the directory, cv2, or the model is absent -- this is the
# owner's local run data, not a repo fixture.
# --------------------------------------------------------------------------- #

REAL_CHUNKS_V12 = Path.home() / "mvm-runs" / "deathless" / "output" / "chunks_v12"


@pytest.mark.integration
def test_facescan_end_to_end_against_a_real_render_subset(tmp_path):
    pytest.importorskip("cv2")
    if not REAL_CHUNKS_V12.is_dir():
        pytest.skip("~/mvm-runs/deathless/output/chunks_v12 not present on this machine")
    if not faces.resolve_model_path().exists():
        pytest.skip("YuNet model not available")

    real_videos = sorted(REAL_CHUNKS_V12.glob("chunk_*.mp4"))[:2]
    if not real_videos:
        pytest.skip("no chunk_*.mp4 files found under chunks_v12")

    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    for video in real_videos:
        (subset_dir / video.name).symlink_to(video)

    out_path = tmp_path / "report.csv"
    rc = facescan.main([str(subset_dir), "--samples", "2", "--out", str(out_path)])
    assert rc == 0

    text = out_path.read_text()
    assert str(subset_dir.resolve()) in text
    assert faces.MODEL_SHA256 in text

    reader = csv.DictReader(line for line in text.splitlines() if not line.startswith("#"))
    rows = list(reader)
    assert len(rows) == len(real_videos)
    for row in rows:
        assert int(row["frames"]) > 0
        assert Path(row["source_path"]).is_absolute()
        assert int(row["source_size_bytes"]) > 0
