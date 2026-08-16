"""Tests for Stage 5 final assembly (issue #11).

Unit tests never invoke a real ``ffmpeg``: the subprocess runner is injected
as a fake that records the exact argument list and returns a scripted
``subprocess.CompletedProcess``. They assert on the generated concat file
content (including single-quote escaping) and the exact ffmpeg argument
lists for both the concat pass and the mux pass.

One genuine end-to-end test exercises real ffmpeg against tiny synthetic
clips; it is marked ``@pytest.mark.integration`` (registered in
``pyproject.toml``) so CI can deselect it with ``-m "not integration"``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker.assembly import (
    AssemblyResult,
    FfmpegError,
    MissingChunksError,
    assemble_final_video,
    build_concat_args,
    build_mux_args,
    validate_chunk_availability,
    write_concat_file,
)
from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus, RunState

# --------------------------------------------------------------------------- #
# Helpers (local to this test module -- tests/harness is off-limits here)
# --------------------------------------------------------------------------- #


def _chunk(chunk_id: int, start: float = 0.0, end: float = 5.0) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"/tmp/chunk_{chunk_id}.wav"),
        start=start,
        end=end,
        text=f"line {chunk_id}",
    )


def _result(
    chunk_id: int,
    status: ChunkStatus = ChunkStatus.RENDERED,
    video_file: Path | None = None,
) -> ChunkResult:
    if video_file is None and status in (ChunkStatus.RENDERED, ChunkStatus.CACHED):
        video_file = Path(f"/tmp/chunk_{chunk_id}.mp4")
    return ChunkResult(chunk_id=chunk_id, status=status, video_file=video_file)


class _FakeRunner:
    """Records every invocation and returns a scripted CompletedProcess.

    By default succeeds (returncode 0). ``fail_on_step`` can be set to make
    the Nth call fail instead, simulating an ffmpeg error without ever
    spawning a process.
    """

    def __init__(self, *, fail_at_call: int | None = None, stderr: bytes = b""):
        self.calls: list[list[str]] = []
        self._fail_at_call = fail_at_call
        self._stderr = stderr

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        call_index = len(self.calls)
        if self._fail_at_call == call_index:
            return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=self._stderr)
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")


# --------------------------------------------------------------------------- #
# Concat file generation + path escaping
# --------------------------------------------------------------------------- #


def test_write_concat_file_basic_paths(tmp_path: Path):
    v1 = tmp_path / "chunk_0.mp4"
    v2 = tmp_path / "chunk_1.mp4"
    dest = tmp_path / "concat.txt"

    write_concat_file([v1, v2], dest)

    content = dest.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines == [
        f"file '{v1.resolve()}'",
        f"file '{v2.resolve()}'",
    ]


def test_write_concat_file_preserves_order(tmp_path: Path):
    # chunk_id order is the caller's responsibility; write_concat_file must
    # not reorder or dedupe -- it writes exactly what it's given.
    v2 = tmp_path / "chunk_2.mp4"
    v0 = tmp_path / "chunk_0.mp4"
    v1 = tmp_path / "chunk_1.mp4"
    dest = tmp_path / "concat.txt"

    write_concat_file([v2, v0, v1], dest)

    lines = dest.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"file '{v2.resolve()}'",
        f"file '{v0.resolve()}'",
        f"file '{v1.resolve()}'",
    ]


def test_write_concat_file_escapes_single_quote(tmp_path: Path):
    # A literal single quote in the path must close-escape-reopen per the
    # concat demuxer's quoting rules: it's.mp4 -> it'\''s.mp4
    tricky_dir = tmp_path / "cast's dir"
    tricky_dir.mkdir()
    video = tricky_dir / "chunk_it's_here.mp4"
    dest = tmp_path / "concat.txt"

    write_concat_file([video], dest)

    line = dest.read_text(encoding="utf-8").splitlines()[0]
    resolved = str(video.resolve())
    expected_escaped = resolved.replace("'", "'\\''")
    assert line == f"file '{expected_escaped}'"
    # Sanity: the raw single quotes from the path must never appear unescaped
    # (i.e. every ' in the body is immediately preceded by the escape).
    assert "'\\''" in line


def test_write_concat_file_handles_spaces(tmp_path: Path):
    space_dir = tmp_path / "my chunks dir"
    space_dir.mkdir()
    video = space_dir / "chunk with spaces.mp4"
    dest = tmp_path / "concat.txt"

    write_concat_file([video], dest)

    line = dest.read_text(encoding="utf-8").splitlines()[0]
    assert line == f"file '{video.resolve()}'"


def test_write_concat_file_writes_absolute_paths(tmp_path: Path, monkeypatch):
    # Even if given a relative Path, the written line must be absolute --
    # ffmpeg concat runs from an arbitrary cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chunk_0.mp4").write_bytes(b"")
    dest = tmp_path / "concat.txt"

    write_concat_file([Path("chunk_0.mp4")], dest)

    line = dest.read_text(encoding="utf-8").splitlines()[0]
    assert line.startswith("file '/") or line.startswith("file 'C:")  # posix vs win, defensive
    assert str(tmp_path) in line


# --------------------------------------------------------------------------- #
# Exact ffmpeg argument lists
# --------------------------------------------------------------------------- #


def test_build_concat_args_exact():
    concat_file = Path("/out/concat_list.txt")
    output = Path("/out/_concat_intermediate.mp4")

    args = build_concat_args(concat_file, output)

    assert args == [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "/out/concat_list.txt",
        "-an",
        "-c:v",
        "copy",
        "/out/_concat_intermediate.mp4",
    ]


def test_build_mux_args_exact():
    video = Path("/out/_concat_intermediate.mp4")
    master_audio = Path("/audio/master.wav")
    output = Path("/out/final_video.mp4")

    args = build_mux_args(video, master_audio, output)

    assert args == [
        "ffmpeg",
        "-y",
        "-i",
        "/out/_concat_intermediate.mp4",
        "-i",
        "/audio/master.wav",
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
        "/out/final_video.mp4",
    ]


# --------------------------------------------------------------------------- #
# validate_chunk_availability
# --------------------------------------------------------------------------- #


def test_validate_chunk_availability_all_present():
    results = {0: _result(0), 1: _result(1), 2: _result(2)}

    available, missing, dead = validate_chunk_availability([0, 1, 2], results)

    assert available == [0, 1, 2]
    assert missing == []
    assert dead == []


def test_validate_chunk_availability_reports_missing_result():
    # chunk 1 has no ChunkResult at all (never attempted / run crashed early)
    results = {0: _result(0), 2: _result(2)}

    available, missing, dead = validate_chunk_availability([0, 1, 2], results)

    assert available == [0, 2]
    assert missing == [1]
    assert dead == []


def test_validate_chunk_availability_reports_dead_lettered():
    results = {
        0: _result(0),
        1: _result(1, status=ChunkStatus.DEAD_LETTERED, video_file=None),
        2: _result(2),
    }

    available, missing, dead = validate_chunk_availability([0, 1, 2], results)

    assert available == [0, 2]
    assert missing == []
    assert dead == [1]


def test_validate_chunk_availability_pending_counts_as_missing():
    results = {0: _result(0), 1: _result(1, status=ChunkStatus.PENDING, video_file=None)}

    available, missing, dead = validate_chunk_availability([0, 1], results)

    assert available == [0]
    assert missing == [1]
    assert dead == []


def test_validate_chunk_availability_succeeded_without_video_file_is_missing():
    # Defensive: a RENDERED result with no video_file must not be treated as
    # available -- there's nothing to concat.
    bad = ChunkResult(chunk_id=0, status=ChunkStatus.RENDERED, video_file=None)

    available, missing, dead = validate_chunk_availability([0], {0: bad})

    assert available == []
    assert missing == [0]


def test_validate_chunk_availability_accepts_run_state():
    run_state = RunState(
        run_id="run-1",
        results={
            0: _result(0),
            1: _result(1, status=ChunkStatus.DEAD_LETTERED, video_file=None),
        },
    )

    available, missing, dead = validate_chunk_availability([0, 1], run_state)

    assert available == [0]
    assert dead == [1]


def test_validate_chunk_availability_cached_counts_as_available():
    results = {0: _result(0, status=ChunkStatus.CACHED)}

    available, missing, dead = validate_chunk_availability([0], results)

    assert available == [0]


# --------------------------------------------------------------------------- #
# assemble_final_video: happy path
# --------------------------------------------------------------------------- #


def test_assemble_final_video_happy_path_runs_concat_then_mux(tmp_path: Path):
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    video_files = {i: tmp_path / f"chunk_{i}.mp4" for i in range(3)}
    for p in video_files.values():
        p.write_bytes(b"fake-mp4")
    results = {i: _result(i, video_file=video_files[i]) for i in range(3)}
    master_audio = tmp_path / "master.wav"
    master_audio.write_bytes(b"fake-wav")
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    # check_luminance=False: this test asserts on the exact concat/mux call
    # count and sequence; the issue #77 darkness check is exercised by its
    # own dedicated tests below, using the same injected runner.
    result = assemble_final_video(
        chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
    )

    assert isinstance(result, AssemblyResult)
    assert len(runner.calls) == 2

    concat_call, mux_call = runner.calls
    assert concat_call[0] == "ffmpeg"
    assert "-an" in concat_call
    assert concat_call[-2:] == ["-c:v", "copy"] or "copy" in concat_call  # -c:v copy present
    assert str(result.intermediate_video) == concat_call[-1]

    assert mux_call[0] == "ffmpeg"
    assert "-map" in mux_call
    assert str(master_audio) in mux_call
    assert str(result.intermediate_video) in mux_call
    assert str(result.output_video) == mux_call[-1]

    assert result.chunk_ids == (0, 1, 2)
    assert result.output_video == output_dir / "final_video.mp4"
    assert result.concat_file == output_dir / "concat_list.txt"
    assert result.concat_file.exists()

    concat_content = result.concat_file.read_text(encoding="utf-8")
    # chronological order preserved regardless of dict insertion order
    for i in range(3):
        assert str(video_files[i].resolve()) in concat_content
    idx0 = concat_content.index(str(video_files[0].resolve()))
    idx1 = concat_content.index(str(video_files[1].resolve()))
    idx2 = concat_content.index(str(video_files[2].resolve()))
    assert idx0 < idx1 < idx2


def test_assemble_final_video_orders_chunks_chronologically_regardless_of_input_order(
    tmp_path: Path,
):
    # chunks handed in out of order; results dict also out of order
    chunks = [_chunk(2), _chunk(0), _chunk(1)]
    video_files = {i: tmp_path / f"chunk_{i}.mp4" for i in range(3)}
    results = {
        2: _result(2, video_file=video_files[2]),
        0: _result(0, video_file=video_files[0]),
        1: _result(1, video_file=video_files[1]),
    }
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    result = assemble_final_video(
        chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
    )

    assert result.chunk_ids == (0, 1, 2)


def test_assemble_final_video_accepts_run_state(tmp_path: Path):
    chunks = [_chunk(0), _chunk(1)]
    video_files = {i: tmp_path / f"chunk_{i}.mp4" for i in range(2)}
    run_state = RunState(
        run_id="run-1",
        results={i: _result(i, video_file=video_files[i]) for i in range(2)},
    )
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    result = assemble_final_video(
        chunks, run_state, master_audio, output_dir, runner=runner, check_luminance=False
    )

    assert result.chunk_ids == (0, 1)


def test_assemble_final_video_custom_output_filename(tmp_path: Path):
    chunks = [_chunk(0)]
    video_file = tmp_path / "chunk_0.mp4"
    results = {0: _result(0, video_file=video_file)}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    result = assemble_final_video(
        chunks,
        results,
        master_audio,
        output_dir,
        output_filename="my_video.mp4",
        runner=runner,
        check_luminance=False,
    )

    assert result.output_video == output_dir / "my_video.mp4"


# --------------------------------------------------------------------------- #
# assemble_final_video: missing / dead-lettered chunks
# --------------------------------------------------------------------------- #


def test_assemble_final_video_raises_on_missing_chunk_before_running_ffmpeg(tmp_path: Path):
    chunks = [_chunk(0), _chunk(1)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}  # chunk 1 missing
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    with pytest.raises(MissingChunksError) as excinfo:
        assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)

    assert excinfo.value.missing_chunk_ids == (1,)
    assert runner.calls == []  # never invoked ffmpeg


def test_assemble_final_video_raises_on_dead_lettered_chunk(tmp_path: Path):
    chunks = [_chunk(0), _chunk(1)]
    results = {
        0: _result(0, video_file=tmp_path / "chunk_0.mp4"),
        1: _result(1, status=ChunkStatus.DEAD_LETTERED, video_file=None),
    }
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    with pytest.raises(MissingChunksError) as excinfo:
        assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)

    assert excinfo.value.dead_lettered_chunk_ids == (1,)
    assert runner.calls == []


def test_assemble_final_video_missing_chunks_error_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    chunks = [_chunk(0), _chunk(1)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.assembly"),
        pytest.raises(MissingChunksError),
    ):
        assemble_final_video(chunks, results, master_audio, output_dir, runner=_FakeRunner())

    assert any("Refusing to assemble" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# assemble_final_video: ffmpeg failures (subprocess stderr must be captured
# and logged, never DEVNULL'd)
# --------------------------------------------------------------------------- #


def test_assemble_final_video_raises_ffmpeg_error_on_concat_failure(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner(fail_at_call=1, stderr=b"concat demuxer: no such file")

    with pytest.raises(FfmpegError) as excinfo:
        assemble_final_video(
            chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
        )

    assert excinfo.value.returncode == 1
    assert "no such file" in excinfo.value.stderr
    assert "no such file" in str(excinfo.value)
    assert len(runner.calls) == 1  # mux never attempted after concat failure


def test_assemble_final_video_raises_ffmpeg_error_on_mux_failure(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner(fail_at_call=2, stderr=b"Invalid data found when processing input")

    with pytest.raises(FfmpegError) as excinfo:
        assemble_final_video(
            chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
        )

    assert excinfo.value.returncode == 1
    assert "Invalid data" in excinfo.value.stderr
    assert len(runner.calls) == 2  # concat succeeded, mux attempted and failed


def test_assemble_final_video_ffmpeg_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner(fail_at_call=1, stderr=b"boom: disk full")

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.assembly"),
        pytest.raises(FfmpegError),
    ):
        assemble_final_video(
            chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
        )

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("boom: disk full" in r.getMessage() for r in error_records)
    assert any("concat" in r.getMessage() for r in error_records)


def test_ffmpeg_error_handles_str_stderr(tmp_path: Path):
    # Defensive: some runners might return str stderr (text=True) rather
    # than bytes -- FfmpegError must handle both without crashing.
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"

    def str_stderr_runner(args):
        return subprocess.CompletedProcess(
            list(args), returncode=1, stdout="", stderr="already a string"
        )

    with pytest.raises(FfmpegError) as excinfo:
        assemble_final_video(
            chunks,
            results,
            master_audio,
            output_dir,
            runner=str_stderr_runner,
            check_luminance=False,
        )

    assert excinfo.value.stderr == "already a string"


# --------------------------------------------------------------------------- #
# assemble_final_video: the issue #77 darkness check, wired in by default
# --------------------------------------------------------------------------- #


def _luminance_capable_runner(*, value: int):
    """A fake runner that answers BOTH kinds of call assemble_final_video
    now makes: an issue #77 frame probe (identified by ``-ss`` in the args,
    matching ``luminance.build_frame_probe_args``) returns ``value`` as every
    pixel of a 32x18 grayscale frame; anything else (concat/mux) succeeds
    with empty output, like ``_FakeRunner``'s default."""
    calls: list[list[str]] = []

    def runner(args):
        args = list(args)
        calls.append(args)
        if "-ss" in args:
            data = bytes([value]) * (32 * 18)
            return subprocess.CompletedProcess(args, returncode=0, stdout=data, stderr=b"")
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    runner.calls = calls
    return runner


def test_assemble_final_video_flags_a_dark_chunk_but_still_assembles(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    chunks = [_chunk(43, start=277.417, end=282.583)]
    results = {43: _result(43, video_file=tmp_path / "chunk_43.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _luminance_capable_runner(value=10)  # below DEFAULT_DARK_FLOOR

    with caplog.at_level(logging.ERROR, logger="music_video_maker.assembly"):
        result = assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)

    # The check never blocks assembly -- concat + mux still ran.
    assert isinstance(result, AssemblyResult)
    assert len(result.dark_chunk_warnings) == 1
    assert result.dark_chunk_warnings[0].chunk_id == 43
    assert any("43" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)
    # concat + mux calls (no -ss) still present alongside the probe calls
    assert any("-ss" not in c and "-i" in c for c in runner.calls)


def test_assemble_final_video_does_not_flag_bright_chunks(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _luminance_capable_runner(value=200)

    result = assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)

    assert result.dark_chunk_warnings == ()


def test_assemble_final_video_check_luminance_false_skips_probing_entirely(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _luminance_capable_runner(value=10)  # would be flagged if checked

    result = assemble_final_video(
        chunks, results, master_audio, output_dir, runner=runner, check_luminance=False
    )

    assert result.dark_chunk_warnings == ()
    assert not any("-ss" in c for c in runner.calls)  # no probe calls at all


def test_assemble_final_video_luminance_floor_is_configurable(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _luminance_capable_runner(value=60)

    default_result = assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)
    assert default_result.dark_chunk_warnings == ()  # 60 is above the default floor (25)

    raised_result = assemble_final_video(
        chunks, results, master_audio, output_dir, runner=runner, luminance_floor=70.0
    )
    assert len(raised_result.dark_chunk_warnings) == 1  # 60 < 70


def test_assemble_final_video_survives_a_broken_darkness_check(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
):
    # Even if the check itself misbehaves, assembly must still complete --
    # a Stage 5 smell test is never allowed to be the reason a finished
    # render doesn't get written.
    import music_video_maker.assembly as assembly_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("darkness check exploded")

    monkeypatch.setattr(assembly_module, "check_dark_chunks", boom)

    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    runner = _FakeRunner()

    with caplog.at_level(logging.ERROR, logger="music_video_maker.assembly"):
        result = assemble_final_video(chunks, results, master_audio, output_dir, runner=runner)

    assert isinstance(result, AssemblyResult)
    assert result.dark_chunk_warnings == ()
    assert "darkness check exploded" in caplog.text


def test_assemble_final_video_uses_separate_luminance_runner_if_given(tmp_path: Path):
    chunks = [_chunk(0)]
    results = {0: _result(0, video_file=tmp_path / "chunk_0.mp4")}
    master_audio = tmp_path / "master.wav"
    output_dir = tmp_path / "final"
    concat_runner = _FakeRunner()  # returns b"" stdout -- unreadable for probes
    luminance_runner = _luminance_capable_runner(value=10)

    result = assemble_final_video(
        chunks,
        results,
        master_audio,
        output_dir,
        runner=concat_runner,
        luminance_runner=luminance_runner,
    )

    assert len(result.dark_chunk_warnings) == 1
    assert len(concat_runner.calls) == 2  # only concat + mux, no probe calls
    assert any("-ss" in c for c in luminance_runner.calls)


# --------------------------------------------------------------------------- #
# Integration: real ffmpeg, tiny synthetic clips (deselected via -m in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_assemble_final_video_integration_real_ffmpeg(tmp_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunk_paths = []
    for i in range(2):
        p = chunk_dir / f"chunk_{i}.mp4"
        proc = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=64x64:d=1:r=10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(p),
            ],
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
        chunk_paths.append(p)

    master_audio = tmp_path / "master.wav"
    proc = subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(master_audio),
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    chunks = [_chunk(i) for i in range(2)]
    results = {i: _result(i, video_file=chunk_paths[i]) for i in range(2)}
    output_dir = tmp_path / "final"

    result = assemble_final_video(chunks, results, master_audio, output_dir)

    assert result.output_video.exists()
    assert result.output_video.stat().st_size > 0

    probe = subprocess.run(
        [
            ffmpeg_bin.replace("ffmpeg", "ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(result.output_video),
        ],
        capture_output=True,
        text=True,
    )
    stream_types = set(probe.stdout.split())
    assert "video" in stream_types
    assert "audio" in stream_types
