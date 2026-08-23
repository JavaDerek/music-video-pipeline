"""Tests for the post-render scene-cut check (issue #81).

A viewer on the second full "Deathless" render reported three scenes inside
one 5.17s chunk whose `global_style` explicitly asks for "ONE continuous
unbroken take". ffmpeg's own `scene` metric, scored over two independent full
80-chunk renders (see `music_video_maker/scenecuts.py`'s module docstring for
the numbers), separates the reported cuts from every clean chunk with a 3x+
gap and no false positives at `DEFAULT_SCENE_THRESHOLD = 0.25`.

Unit tests never invoke a real ffmpeg: the subprocess runner is injected as a
fake returning scripted `metadata:print` text on stdout and/or stderr. One
integration test (`@pytest.mark.integration`) builds tiny synthetic clips
with real ffmpeg. An optional second integration test replays the exact
measurement against the real "Deathless" corpus on disk, skipping cleanly
when that run directory is absent (it is the owner's run dir, never repo
content).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState
from music_video_maker.scenecuts import (
    DEFAULT_SCENE_THRESHOLD,
    ChunkSceneCuts,
    SceneCut,
    SceneCutWarning,
    build_scene_probe_args,
    check_scene_cuts,
    main,
    measure_chunk_scene_cuts,
    parse_scene_scores,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _frame_lines(pairs: list[tuple[float, float]]) -> str:
    """Render (time, score) pairs the way ffmpeg's own metadata:print does."""
    lines = []
    for i, (t, s) in enumerate(pairs):
        lines.append(f"frame:{i}    pts:{int(t * 1000)}    pts_time:{t}")
        lines.append(f"lavfi.scene_score={s}")
    return "\n".join(lines) + "\n"


class _ScriptedRunner:
    """Fake ffmpeg runner: maps a video path to scripted output text, split
    across stdout/stderr as configured. Records every call for assertions."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self._by_path: dict[str, tuple[str, str, int]] = {}

    def script(self, video_path: Path, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self._by_path[str(video_path)] = (stdout, stderr, returncode)

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        video_path = args[args.index("-i") + 1]
        stdout, stderr, returncode = self._by_path.get(video_path, ("", "", 0))
        return subprocess.CompletedProcess(
            args, returncode=returncode, stdout=stdout.encode(), stderr=stderr.encode()
        )


class _RaisingRunner:
    """A runner whose __call__ raises -- ffmpeg missing from PATH, or the
    subprocess spawn itself failing (FileNotFoundError, OSError)."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def __call__(self, args):
        self.calls += 1
        raise self._exc


def _chunk(chunk_id: int, start: float = 0.0, end: float = 5.0) -> AudioChunk:
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
# build_scene_probe_args -- exact ffmpeg argument list
# --------------------------------------------------------------------------- #


def test_build_scene_probe_args_exact():
    args = build_scene_probe_args(Path("/chunks/chunk_0022.mp4"))

    assert args == [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "/chunks/chunk_0022.mp4",
        "-vf",
        "select='gte(scene,0)',metadata=print:file=-",
        "-an",
        "-f",
        "null",
        "-",
    ]


# --------------------------------------------------------------------------- #
# parse_scene_scores -- pure text parser
# --------------------------------------------------------------------------- #


def test_parse_scene_scores_basic_pairs():
    text = _frame_lines([(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)])

    pairs = parse_scene_scores(text)

    assert pairs == [(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)]


def test_parse_scene_scores_ignores_a_scene_score_line_with_no_preceding_time():
    text = "lavfi.scene_score=0.5\nframe:0 pts:0 pts_time:1.0\nlavfi.scene_score=0.3\n"

    pairs = parse_scene_scores(text)

    assert pairs == [(1.0, 0.3)]


def test_parse_scene_scores_ignores_a_time_line_with_no_following_score():
    text = "frame:0 pts:0 pts_time:1.0\nframe:1 pts:1 pts_time:2.0\nlavfi.scene_score=0.4\n"

    # the first pts_time is orphaned (overwritten before ever pairing) --
    # only the second, which is immediately followed by a score, survives.
    pairs = parse_scene_scores(text)

    assert pairs == [(2.0, 0.4)]


def test_parse_scene_scores_malformed_numbers_are_skipped_not_raised():
    text = (
        "frame:0 pts:0 pts_time:N/A\nlavfi.scene_score=0.4\n"
        "frame:1 pts:1 pts_time:1.0\nlavfi.scene_score=garbage\n"
    )

    pairs = parse_scene_scores(text)

    assert pairs == []


def test_parse_scene_scores_empty_text():
    assert parse_scene_scores("") == []


def test_parse_scene_scores_garbage_text_does_not_raise():
    assert parse_scene_scores("this is not ffmpeg output at all\n\x00\x01binary junk") == []


# --------------------------------------------------------------------------- #
# measure_chunk_scene_cuts -- happy path
# --------------------------------------------------------------------------- #


def test_measure_chunk_scene_cuts_finds_cuts_above_threshold():
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0022.mp4")
    runner.script(video, stdout=_frame_lines([(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)]))

    result = measure_chunk_scene_cuts(chunk_id=22, video_file=video, runner=runner)

    assert isinstance(result, ChunkSceneCuts)
    assert result.chunk_id == 22
    assert result.max_score == pytest.approx(0.917)
    assert len(result.cuts) == 2
    assert result.cuts[0] == SceneCut(time_seconds=pytest.approx(1.083), score=pytest.approx(0.752))
    assert result.cuts[1] == SceneCut(time_seconds=pytest.approx(3.708), score=pytest.approx(0.917))


def test_measure_chunk_scene_cuts_no_cuts_below_threshold():
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0027.mp4")
    runner.script(video, stdout=_frame_lines([(0.0, 0.0), (5.33, 0.112)]))

    result = measure_chunk_scene_cuts(chunk_id=27, video_file=video, runner=runner)

    assert result.cuts == ()
    assert result.max_score == pytest.approx(0.112)  # measured, just not a cut


def test_measure_chunk_scene_cuts_threshold_is_exclusive_at_the_boundary():
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_x.mp4")
    runner.script(video, stdout=_frame_lines([(1.0, 0.25)]))  # exactly at the default floor

    result = measure_chunk_scene_cuts(chunk_id=0, video_file=video, runner=runner)

    assert result.cuts == ()  # > threshold, not >=


def test_measure_chunk_scene_cuts_custom_threshold():
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_x.mp4")
    runner.script(video, stdout=_frame_lines([(1.0, 0.3)]))

    at_default = measure_chunk_scene_cuts(chunk_id=0, video_file=video, runner=runner)
    at_high = measure_chunk_scene_cuts(chunk_id=0, video_file=video, threshold=0.35, runner=runner)

    assert len(at_default.cuts) == 1
    assert at_high.cuts == ()


def test_measure_chunk_scene_cuts_reads_stderr_too():
    """ffmpeg may emit metadata:print output on stdout OR stderr depending on
    the build -- the corpus scan behind DEFAULT_SCENE_THRESHOLD read both and
    concatenated them, and this must match."""
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_x.mp4")
    runner.script(video, stderr=_frame_lines([(0.0, 0.0), (2.0, 0.9)]))

    result = measure_chunk_scene_cuts(chunk_id=0, video_file=video, runner=runner)

    assert len(result.cuts) == 1
    assert result.cuts[0].time_seconds == pytest.approx(2.0)


def test_measure_chunk_scene_cuts_concatenates_stdout_and_stderr():
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_x.mp4")
    runner.script(
        video,
        stdout=_frame_lines([(0.0, 0.0), (1.0, 0.9)]),
        stderr=_frame_lines([(2.0, 0.8)]),
    )

    result = measure_chunk_scene_cuts(chunk_id=0, video_file=video, runner=runner)

    assert len(result.cuts) == 2
    assert sorted(c.time_seconds for c in result.cuts) == pytest.approx([1.0, 2.0])


# --------------------------------------------------------------------------- #
# measure_chunk_scene_cuts -- graceful degradation
# --------------------------------------------------------------------------- #


def test_measure_chunk_scene_cuts_runner_exception_is_caught_and_logged(
    caplog: pytest.LogCaptureFixture,
):
    runner = _RaisingRunner(FileNotFoundError("ffmpeg not on PATH"))

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        result = measure_chunk_scene_cuts(
            chunk_id=9, video_file=Path("/chunks/chunk_0009.mp4"), runner=runner
        )

    assert result.cuts == ()
    assert result.max_score is None
    assert any("ffmpeg not on PATH" in r.getMessage() for r in caplog.records)


def test_measure_chunk_scene_cuts_nonzero_exit_degrades_and_logs(
    caplog: pytest.LogCaptureFixture,
):
    def runner(args):
        return subprocess.CompletedProcess(
            list(args), returncode=1, stdout=b"", stderr=b"no such file"
        )

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        result = measure_chunk_scene_cuts(
            chunk_id=5, video_file=Path("/chunks/chunk_0005.mp4"), runner=runner
        )

    assert result.cuts == ()
    assert result.max_score is None
    assert any("chunk_0005.mp4" in r.getMessage() for r in caplog.records)


def test_measure_chunk_scene_cuts_empty_output_degrades_and_logs(caplog: pytest.LogCaptureFixture):
    def runner(args):
        return subprocess.CompletedProcess(list(args), returncode=0, stdout=b"", stderr=b"")

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        result = measure_chunk_scene_cuts(
            chunk_id=1, video_file=Path("/chunks/chunk_0001.mp4"), runner=runner
        )

    assert result.cuts == ()
    assert result.max_score is None
    assert len(caplog.records) >= 1


def test_measure_chunk_scene_cuts_garbage_output_degrades_and_logs(
    caplog: pytest.LogCaptureFixture,
):
    def runner(args):
        return subprocess.CompletedProcess(
            list(args), returncode=0, stdout=b"\x00\x01\x02not parseable", stderr=b""
        )

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        result = measure_chunk_scene_cuts(
            chunk_id=2, video_file=Path("/chunks/chunk_0002.mp4"), runner=runner
        )

    assert result.cuts == ()
    assert result.max_score is None


def test_measure_chunk_scene_cuts_never_raises_regardless_of_failure_mode():
    # Belt and braces: none of the degradation paths above should ever
    # propagate an exception out of this function.
    for runner in (
        _RaisingRunner(OSError("boom")),
        lambda args: subprocess.CompletedProcess(list(args), 1, stdout=b"", stderr=b""),
        lambda args: subprocess.CompletedProcess(list(args), 0, stdout=b"", stderr=b""),
    ):
        measure_chunk_scene_cuts(chunk_id=0, video_file=Path("/c.mp4"), runner=runner)


# --------------------------------------------------------------------------- #
# check_scene_cuts
# --------------------------------------------------------------------------- #


def test_check_scene_cuts_flags_a_chunk_with_cuts(caplog):
    start, end = 144.83, 150.00
    chunks = [_chunk(22, start=start, end=end)]
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0022.mp4")
    runner.script(video, stdout=_frame_lines([(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)]))
    results = {22: _result(22, video)}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        flags = check_scene_cuts(chunks, results, runner=runner)

    assert len(flags) == 1
    flag = flags[0]
    assert isinstance(flag, SceneCutWarning)
    assert flag.chunk_id == 22
    assert len(flag.cuts) == 2
    assert flag.max_score == pytest.approx(0.917)
    assert flag.threshold == DEFAULT_SCENE_THRESHOLD
    assert any("22" in r.getMessage() for r in caplog.records)


def test_check_scene_cuts_does_not_flag_a_clean_chunk():
    chunks = [_chunk(7)]
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0007.mp4")
    runner.script(video, stdout=_frame_lines([(0.0, 0.0), (2.0, 0.006)]))
    results = {7: _result(7, video)}

    flags = check_scene_cuts(chunks, results, runner=runner)

    assert flags == ()


def test_check_scene_cuts_custom_threshold():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0000.mp4")
    runner.script(video, stdout=_frame_lines([(1.0, 0.3)]))
    results = {0: _result(0, video)}

    default_flags = check_scene_cuts(chunks, results, runner=runner)
    strict_flags = check_scene_cuts(chunks, results, threshold=0.35, runner=runner)

    assert len(default_flags) == 1
    assert strict_flags == ()


def test_check_scene_cuts_skips_a_chunk_with_no_result_and_logs(caplog):
    chunks = [_chunk(0), _chunk(1)]
    runner = _ScriptedRunner()
    video = Path("/chunks/chunk_0000.mp4")
    runner.script(video, stdout=_frame_lines([(1.0, 0.9)]))
    results = {0: _result(0, video)}  # chunk 1 has no result

    with caplog.at_level(logging.INFO, logger="music_video_maker.scenecuts"):
        flags = check_scene_cuts(chunks, results, runner=runner)

    assert len(flags) == 1
    assert flags[0].chunk_id == 0
    assert any("1" in r.getMessage() for r in caplog.records)
    assert len(runner.calls) == 1  # only chunk 0 actually got probed


def test_check_scene_cuts_skips_dead_lettered_chunk():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner()
    results = {0: _result(0, Path("/c0.mp4"), status=ChunkStatus.DEAD_LETTERED)}

    flags = check_scene_cuts(chunks, results, runner=runner)

    assert flags == ()
    assert runner.calls == []


def test_check_scene_cuts_accepts_run_state():
    chunks = [_chunk(0)]
    runner = _ScriptedRunner()
    video = Path("/c0.mp4")
    runner.script(video, stdout=_frame_lines([(1.0, 0.9)]))
    run_state = RunState(run_id="r1", results={0: _result(0, video)})

    flags = check_scene_cuts(chunks, run_state, runner=runner)

    assert len(flags) == 1


def test_check_scene_cuts_never_raises_on_runner_exception(caplog):
    chunks = [_chunk(0)]
    runner = _RaisingRunner(RuntimeError("boom"))
    results = {0: _result(0, Path("/c0.mp4"))}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        flags = check_scene_cuts(chunks, results, runner=runner)

    assert flags == ()
    assert any("boom" in r.getMessage() for r in caplog.records)


def test_check_scene_cuts_survives_a_bug_in_the_measurement_itself(caplog, monkeypatch):
    import music_video_maker.scenecuts as scenecuts_module

    def boom(**_kwargs):
        raise ValueError("unexpected bug")

    monkeypatch.setattr(scenecuts_module, "measure_chunk_scene_cuts", boom)
    chunks = [_chunk(0)]
    results = {0: _result(0, Path("/c0.mp4"))}

    with caplog.at_level(logging.WARNING, logger="music_video_maker.scenecuts"):
        flags = check_scene_cuts(chunks, results, runner=_ScriptedRunner())

    assert flags == ()
    assert "unexpected bug" in caplog.text


def test_check_scene_cuts_multiple_chunks_only_flags_the_bad_ones():
    chunks = [_chunk(7), _chunk(12), _chunk(22)]
    runner = _ScriptedRunner()
    v7, v12, v22 = Path("/c7.mp4"), Path("/c12.mp4"), Path("/c22.mp4")
    runner.script(v7, stdout=_frame_lines([(0.0, 0.0), (2.0, 0.006)]))
    runner.script(v12, stdout=_frame_lines([(0.0, 0.0), (2.17, 0.519)]))
    runner.script(v22, stdout=_frame_lines([(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)]))
    results = {7: _result(7, v7), 12: _result(12, v12), 22: _result(22, v22)}

    flags = check_scene_cuts(chunks, results, runner=runner)

    assert {f.chunk_id for f in flags} == {12, 22}


# --------------------------------------------------------------------------- #
# The default threshold sits inside a plateau, not a tuned point
# --------------------------------------------------------------------------- #


def test_default_threshold_is_the_measured_value():
    assert DEFAULT_SCENE_THRESHOLD == 0.25


def test_default_threshold_separates_corpus_a_and_b_with_margin():
    """corpus A: worst unflagged 0.112 (chunk 27), smallest flagged 0.370
    (chunk 75). corpus B: largest unflagged 0.076, smallest flagged 0.330
    (chunk 4). The default must sit strictly inside both gaps."""
    assert 0.112 < DEFAULT_SCENE_THRESHOLD < 0.370
    assert 0.076 < DEFAULT_SCENE_THRESHOLD < 0.330


# --------------------------------------------------------------------------- #
# main() -- standalone entry point, exercised in-process (no subprocess)
# --------------------------------------------------------------------------- #


def test_main_reports_flagged_chunks_and_returns_nonzero(tmp_path: Path, capsys):
    (tmp_path / "chunk_0022.mp4").write_bytes(b"")
    (tmp_path / "chunk_0027.mp4").write_bytes(b"")
    runner = _ScriptedRunner()
    runner.script(
        tmp_path / "chunk_0022.mp4",
        stdout=_frame_lines([(0.0, 0.0), (1.083, 0.752), (3.708, 0.917)]),
    )
    runner.script(tmp_path / "chunk_0027.mp4", stdout=_frame_lines([(0.0, 0.0), (5.33, 0.112)]))

    exit_code = main([str(tmp_path)], runner=runner)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "chunk_0022.mp4" in out
    assert "chunk_0027.mp4" not in out  # clean chunk isn't listed per-line
    assert "1/2" in out or "1 / 2" in out


def test_main_returns_zero_when_nothing_is_flagged(tmp_path: Path, capsys):
    (tmp_path / "chunk_0000.mp4").write_bytes(b"")
    runner = _ScriptedRunner()
    runner.script(tmp_path / "chunk_0000.mp4", stdout=_frame_lines([(0.0, 0.0), (2.0, 0.01)]))

    exit_code = main([str(tmp_path)], runner=runner)

    assert exit_code == 0


def test_main_accepts_custom_threshold(tmp_path: Path):
    (tmp_path / "chunk_0000.mp4").write_bytes(b"")
    runner = _ScriptedRunner()
    runner.script(tmp_path / "chunk_0000.mp4", stdout=_frame_lines([(1.0, 0.3)]))

    default_exit = main([str(tmp_path)], runner=runner)
    strict_exit = main([str(tmp_path), "--threshold", "0.35"], runner=runner)

    assert default_exit == 1
    assert strict_exit == 0


def test_main_missing_directory_returns_error(tmp_path: Path, capsys):
    exit_code = main([str(tmp_path / "does_not_exist")], runner=_ScriptedRunner())

    assert exit_code == 1
    assert "does_not_exist" in capsys.readouterr().err


def test_main_no_chunk_files_returns_zero(tmp_path: Path, capsys):
    exit_code = main([str(tmp_path)], runner=_ScriptedRunner())

    assert exit_code == 0
    assert "no chunk" in capsys.readouterr().out.lower()


def test_main_only_scans_chunk_named_mp4_files(tmp_path: Path, capsys):
    (tmp_path / "chunk_0001.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "final_video.mp4").write_bytes(b"")
    runner = _ScriptedRunner()
    runner.script(tmp_path / "chunk_0001.mp4", stdout=_frame_lines([(0.0, 0.0)]))

    main([str(tmp_path)], runner=runner)

    probed = {c[c.index("-i") + 1] for c in runner.calls}
    assert probed == {str(tmp_path / "chunk_0001.mp4")}


# --------------------------------------------------------------------------- #
# Integration: real ffmpeg, tiny synthetic clips
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_scene_cut_detection_integration_hard_cut_is_flagged(tmp_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    video = tmp_path / "cut.mp4"
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=64x64:rate=10:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    result = measure_chunk_scene_cuts(chunk_id=0, video_file=video)

    assert len(result.cuts) >= 1


@pytest.mark.integration
def test_scene_cut_detection_integration_continuous_pan_is_not_flagged(tmp_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    # One continuous source, slowly panned/zoomed -- no cut anywhere in it.
    video = tmp_path / "pan.mp4"
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=2",
            "-vf",
            "zoompan=z='min(zoom+0.002,1.2)':d=1:s=320x240",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    result = measure_chunk_scene_cuts(chunk_id=1, video_file=video)

    assert result.cuts == ()


# --------------------------------------------------------------------------- #
# Optional: replay against the real "Deathless" corpus (skip-if-absent)
# --------------------------------------------------------------------------- #

_DEATHLESS_CHUNKS_V3 = Path.home() / "mvm-runs" / "deathless" / "output" / "chunks_v3"
_DEATHLESS_CHUNKS_V12 = Path.home() / "mvm-runs" / "deathless" / "output" / "chunks_v12"


@pytest.mark.integration
def test_scene_cuts_against_real_corpus_chunk_22(tmp_path: Path):
    """Corpus A (issue #81's own measurement): chunk 22 has 2 cuts at
    +1.08s (0.752) and +3.71s (0.917). This is the owner's run directory,
    never repo content -- skip cleanly when it is not present."""
    ffmpeg_bin = shutil.which("ffmpeg")
    video = _DEATHLESS_CHUNKS_V3 / "chunk_0022.mp4"
    if ffmpeg_bin is None or not video.is_file():
        pytest.skip("real ffmpeg + ~/mvm-runs/deathless corpus not available")

    result = measure_chunk_scene_cuts(chunk_id=22, video_file=video)

    assert len(result.cuts) == 2
    assert result.cuts[0].time_seconds == pytest.approx(1.083, abs=0.01)
    assert result.cuts[0].score == pytest.approx(0.752, abs=0.01)
    assert result.cuts[1].time_seconds == pytest.approx(3.708, abs=0.01)
    assert result.cuts[1].score == pytest.approx(0.917, abs=0.01)


@pytest.mark.integration
def test_scene_cuts_against_real_corpus_v12_chunk_4():
    """Corpus B: chunk 4 has 1 cut at +6.08s (0.330) -- a real, previously
    unreported cut in the current best video."""
    ffmpeg_bin = shutil.which("ffmpeg")
    video = _DEATHLESS_CHUNKS_V12 / "chunk_0004.mp4"
    if ffmpeg_bin is None or not video.is_file():
        pytest.skip("real ffmpeg + ~/mvm-runs/deathless corpus not available")

    result = measure_chunk_scene_cuts(chunk_id=4, video_file=video)

    assert len(result.cuts) == 1
    assert result.cuts[0].time_seconds == pytest.approx(6.083, abs=0.01)
    assert result.cuts[0].score == pytest.approx(0.330, abs=0.01)
