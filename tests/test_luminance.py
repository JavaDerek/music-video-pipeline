"""Tests for the post-render darkness check (issue #77).

A viewer reported "sudden darkness for 2 seconds" in the "Deathless" render.
The chunk responsible (43) ends at mean luminance Y 15.0 against a render-wide
"next darkest chunk ending" of Y 36.4 -- see ``docs/deathless-render-corpus.md``
for the full 80-chunk distribution this module's default floor is derived
from.

F11 proposed a **drift** band (start vs end luminance within about +/-15 Y)
before the full render existed. Scanning all 80 chunks showed 36 of them
outside that band, from -61 to +55 -- because the song's whole narrative is a
night ending in dawn, so large luminance swings are the *content*, not a
defect. This module deliberately does NOT implement a drift check; it flags
an **absolute floor** on the chunk's ending luminance only. ``drift`` is
still computed (and logged) for context, but never gates the flag.

Unit tests never invoke a real ffmpeg: the subprocess runner is injected as a
fake that returns scripted raw grayscale frame bytes. One integration test
(``@pytest.mark.integration``) exercises real ffmpeg against a tiny synthetic
lavfi clip, the same pattern ``tests/test_assembly.py`` already uses.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState
from music_video_maker.luminance import (
    DEFAULT_DARK_FLOOR,
    DEFAULT_END_FRACTIONS,
    DEFAULT_START_FRACTIONS,
    ChunkLuminance,
    DarkChunkWarning,
    build_frame_probe_args,
    check_dark_chunks,
    measure_chunk_luminance,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _gray_bytes(value: int, width: int = 32, height: int = 18) -> bytes:
    return bytes([value]) * (width * height)


class _ScriptedRunner:
    """Fake ffmpeg runner: maps an exact ``-ss`` timestamp to a grayscale
    byte value (or ``None`` for "ffmpeg failed to read this frame").
    Records every call for assertions.
    """

    def __init__(self, *, default_value: int | None = 100):
        self.calls: list[list[str]] = []
        self._by_ts: dict[float, int | None] = {}
        self._default_value = default_value

    def value_at(self, timestamp: float, value: int | None) -> None:
        self._by_ts[round(timestamp, 3)] = value

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        ts = round(float(args[args.index("-ss") + 1]), 3)
        value = self._by_ts.get(ts, self._default_value)
        if value is None:
            return subprocess.CompletedProcess(
                args, returncode=1, stdout=b"", stderr=b"no such filter"
            )
        return subprocess.CompletedProcess(
            args, returncode=0, stdout=_gray_bytes(value), stderr=b""
        )


class _RaisingRunner:
    """A runner whose __call__ raises -- simulating ffmpeg not being on PATH
    or the subprocess spawn itself failing (FileNotFoundError, OSError)."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        raise self._exc


def _chunk(chunk_id: int, start: float = 0.0, end: float = 8.0) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"/tmp/chunk_{chunk_id}.wav"),
        start=start,
        end=end,
        text=f"line {chunk_id}",
    )


def _result(
    chunk_id: int, video_file: Path, status: ChunkStatus = ChunkStatus.RENDERED
) -> ChunkResult:
    return ChunkResult(chunk_id=chunk_id, status=status, video_file=video_file)


# --------------------------------------------------------------------------- #
# build_frame_probe_args -- exact ffmpeg argument list
# --------------------------------------------------------------------------- #


def test_build_frame_probe_args_exact():
    args = build_frame_probe_args(Path("/chunks/chunk_0043.mp4"), 4.5, width=32, height=18)

    assert args == [
        "ffmpeg",
        "-y",
        "-ss",
        "4.500",
        "-i",
        "/chunks/chunk_0043.mp4",
        "-frames:v",
        "1",
        "-s",
        "32x18",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]


# --------------------------------------------------------------------------- #
# measure_chunk_luminance -- happy path
# --------------------------------------------------------------------------- #


def test_measure_chunk_luminance_bright_start_dark_end():
    duration = 8.0
    runner = _ScriptedRunner(default_value=None)
    for f in DEFAULT_START_FRACTIONS:
        runner.value_at(f * duration, 200)
    for f in DEFAULT_END_FRACTIONS:
        runner.value_at(f * duration, 10)

    result = measure_chunk_luminance(
        chunk_id=43, video_file=Path("/chunks/chunk_0043.mp4"), duration=duration, runner=runner
    )

    assert isinstance(result, ChunkLuminance)
    assert result.chunk_id == 43
    assert result.start_mean == pytest.approx(200.0)
    assert result.end_mean == pytest.approx(10.0)
    assert result.drift == pytest.approx(10.0 - 200.0)
    # one ffmpeg call per configured sample fraction
    assert len(runner.calls) == len(DEFAULT_START_FRACTIONS) + len(DEFAULT_END_FRACTIONS)


def test_measure_chunk_luminance_uses_correct_timestamps():
    runner = _ScriptedRunner()
    duration = 10.0

    measure_chunk_luminance(chunk_id=1, video_file=Path("/c.mp4"), duration=duration, runner=runner)

    seen_timestamps = {round(float(c[c.index("-ss") + 1]), 3) for c in runner.calls}
    expected = {round(f * duration, 3) for f in (*DEFAULT_START_FRACTIONS, *DEFAULT_END_FRACTIONS)}
    assert seen_timestamps == expected


def test_measure_chunk_luminance_averages_multiple_samples_per_side():
    duration = 100.0
    runner = _ScriptedRunner(default_value=None)
    start_values = [10, 20, 30]
    end_values = [90, 90, 90]
    for f, v in zip(DEFAULT_START_FRACTIONS, start_values, strict=True):
        runner.value_at(f * duration, v)
    for f, v in zip(DEFAULT_END_FRACTIONS, end_values, strict=True):
        runner.value_at(f * duration, v)

    result = measure_chunk_luminance(
        chunk_id=2, video_file=Path("/c.mp4"), duration=duration, runner=runner
    )

    assert result.start_mean == pytest.approx(sum(start_values) / len(start_values))
    assert result.end_mean == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# measure_chunk_luminance -- graceful degradation
# --------------------------------------------------------------------------- #


def test_measure_chunk_luminance_some_samples_unreadable_averages_the_rest():
    duration = 8.0
    runner = _ScriptedRunner(default_value=50)
    fractions = list(DEFAULT_START_FRACTIONS)
    runner.value_at(fractions[0] * duration, 100)
    runner.value_at(fractions[1] * duration, None)  # this one frame fails
    runner.value_at(fractions[2] * duration, 120)

    result = measure_chunk_luminance(
        chunk_id=3, video_file=Path("/c.mp4"), duration=duration, runner=runner
    )

    assert result.start_mean == pytest.approx((100 + 120) / 2)


def test_measure_chunk_luminance_all_samples_unreadable_yields_none_and_logs(
    caplog: pytest.LogCaptureFixture,
):
    runner = _ScriptedRunner(default_value=None)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        result = measure_chunk_luminance(
            chunk_id=7, video_file=Path("/chunks/chunk_0007.mp4"), duration=8.0, runner=runner
        )

    assert result.start_mean is None
    assert result.end_mean is None
    assert result.drift is None
    assert any(
        "7" in r.getMessage() and "chunk_0007.mp4" in r.getMessage() for r in caplog.records
    )


def test_measure_chunk_luminance_runner_exception_is_caught_and_logged(
    caplog: pytest.LogCaptureFixture,
):
    runner = _RaisingRunner(FileNotFoundError("ffmpeg not on PATH"))

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        result = measure_chunk_luminance(
            chunk_id=9, video_file=Path("/chunks/chunk_0009.mp4"), duration=8.0, runner=runner
        )

    assert result.start_mean is None
    assert result.end_mean is None
    assert any("ffmpeg not on PATH" in r.getMessage() for r in caplog.records)


def test_measure_chunk_luminance_malformed_stdout_length_is_treated_as_unreadable(
    caplog: pytest.LogCaptureFixture,
):
    def runner(args):
        return subprocess.CompletedProcess(list(args), returncode=0, stdout=b"\x10\x20", stderr=b"")

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        result = measure_chunk_luminance(
            chunk_id=11, video_file=Path("/c.mp4"), duration=8.0, runner=runner
        )

    assert result.start_mean is None
    assert result.end_mean is None


# --------------------------------------------------------------------------- #
# check_dark_chunks
# --------------------------------------------------------------------------- #


def test_check_dark_chunks_flags_a_chunk_ending_below_the_floor(caplog):
    start, end = 277.417, 282.583
    duration = end - start
    chunks = [_chunk(43, start=start, end=end)]
    runner = _ScriptedRunner(default_value=None)
    for f in DEFAULT_START_FRACTIONS:
        runner.value_at(f * duration, 200)
    for f in DEFAULT_END_FRACTIONS:
        runner.value_at(f * duration, 15)
    results = {43: _result(43, Path("/chunks/chunk_0043.mp4"))}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        flags = check_dark_chunks(chunks, results, runner=runner)

    assert len(flags) == 1
    flag = flags[0]
    assert isinstance(flag, DarkChunkWarning)
    assert flag.chunk_id == 43
    assert flag.end_mean == pytest.approx(15.0)
    assert flag.floor == DEFAULT_DARK_FLOOR
    assert any("43" in r.getMessage() for r in caplog.records)


def test_check_dark_chunks_does_not_flag_a_chunk_above_the_floor():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner(default_value=60)
    results = {0: _result(0, Path("/chunks/chunk_0000.mp4"))}

    flags = check_dark_chunks(chunks, results, runner=runner)

    assert flags == ()


def test_check_dark_chunks_custom_floor():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner(default_value=60)
    results = {0: _result(0, Path("/chunks/chunk_0000.mp4"))}

    flags = check_dark_chunks(chunks, results, floor=70.0, runner=runner)

    assert len(flags) == 1  # 60 < 70 with the raised floor


def test_check_dark_chunks_never_raises_when_a_chunk_is_unreadable(caplog):
    chunks = [_chunk(0), _chunk(1)]
    runner = _ScriptedRunner(default_value=None)  # every probe fails
    results = {
        0: _result(0, Path("/chunks/chunk_0000.mp4")),
        1: _result(1, Path("/chunks/chunk_0001.mp4")),
    }

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        flags = check_dark_chunks(chunks, results, runner=runner)

    assert flags == ()  # cannot prove darkness -- degrades to "not flagged"


def test_check_dark_chunks_skips_a_chunk_with_no_result_and_logs(caplog):
    chunks = [_chunk(0), _chunk(1)]
    runner = _ScriptedRunner(default_value=60)
    results = {0: _result(0, Path("/chunks/chunk_0000.mp4"))}  # chunk 1 has no result

    with caplog.at_level(logging.INFO, logger="music_video_maker.luminance"):
        flags = check_dark_chunks(chunks, results, runner=runner)

    assert flags == ()
    assert any("1" in r.getMessage() for r in caplog.records)
    # only chunk 0 actually got probed
    assert len(runner.calls) == len(DEFAULT_START_FRACTIONS) + len(DEFAULT_END_FRACTIONS)


def test_check_dark_chunks_accepts_run_state():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner(default_value=10)  # dark
    run_state = RunState(run_id="r1", results={0: _result(0, Path("/c0.mp4"))})

    flags = check_dark_chunks(chunks, run_state, runner=runner)

    assert len(flags) == 1


def test_check_dark_chunks_never_raises_on_unexpected_runner_exception(caplog):
    # The runner itself raises -- caught inside measure_chunk_luminance's own
    # per-sample guard, so this exercises the INNER degrade-gracefully layer.
    chunks = [_chunk(0)]
    runner = _RaisingRunner(RuntimeError("boom"))
    results = {0: _result(0, Path("/c0.mp4"))}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        flags = check_dark_chunks(chunks, results, runner=runner)

    assert flags == ()
    assert any("boom" in r.getMessage() for r in caplog.records)


def test_check_dark_chunks_survives_a_bug_in_the_measurement_itself(caplog, monkeypatch):
    # Defense in depth: even if measure_chunk_luminance itself misbehaves
    # (a bug unrelated to the runner/subprocess layer), check_dark_chunks
    # must still never raise -- this exercises the OUTER guard.
    import music_video_maker.luminance as luminance_module

    def boom(**_kwargs):
        raise ValueError("unexpected bug")

    monkeypatch.setattr(luminance_module, "measure_chunk_luminance", boom)
    chunks = [_chunk(0)]
    results = {0: _result(0, Path("/c0.mp4"))}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.luminance"):
        flags = check_dark_chunks(chunks, results, runner=_ScriptedRunner())

    assert flags == ()
    assert "unexpected bug" in caplog.text  # logger.exception attaches the traceback


# --------------------------------------------------------------------------- #
# The default floor is measured, not invented
# --------------------------------------------------------------------------- #


def test_default_floor_sits_inside_the_measured_gap():
    """chunk 43 of the "Deathless" full render ends at Y 14.9; the next
    darkest chunk ending anywhere in that 80-chunk render is Y 36.4 (chunk
    18) -- see docs/deathless-render-corpus.md. The floor must separate them
    with margin on both sides, not sit at either edge."""
    assert 14.9 < DEFAULT_DARK_FLOOR < 36.4
    # comfortable margin on both sides, not just barely inside the gap
    assert DEFAULT_DARK_FLOOR - 14.9 > 5.0
    assert 36.4 - DEFAULT_DARK_FLOOR > 5.0


# --------------------------------------------------------------------------- #
# Integration: real ffmpeg, a tiny synthetic clip
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_measure_chunk_luminance_integration_real_ffmpeg(tmp_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    # A clip that fades from white to black over 1 second.
    video = tmp_path / "fade.mp4"
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=64x64:d=1:r=10,fade=t=out:st=0:d=1:color=black",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    result = measure_chunk_luminance(chunk_id=0, video_file=video, duration=1.0)

    assert result.start_mean is not None
    assert result.end_mean is not None
    assert result.start_mean > result.end_mean  # fades from bright to dark
