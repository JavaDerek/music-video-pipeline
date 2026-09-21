"""Tests for the #96 voicing calibration command.

Offline: the ffmpeg decode is an injected seam (``decoder=``), the segments
come from a JSON file rather than an aligner, and the waveforms are the same
synthetic voice/pluck/noise/near-silence fixtures ``tests/test_voicing.py``
uses. The one thing these tests are really guarding is that the *operator's*
single command cannot fail with a traceback or print a table whose provenance
is missing -- #93's rule, since this table is what a threshold will be moved
on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker import calibrate_voicing
from music_video_maker.contracts import AlignmentResult
from tests.harness.factories import (
    make_aligned_segment,
    near_silence_samples,
    noise_samples,
    voiced_samples,
    write_samples_wav,
)

SPAN = 2.0
LABELS = ["voice", "voice", "voice", "voice", "noise"]


def _copy_decoder(master: Path, out_path: Path) -> None:
    """Stand in for ffmpeg: the fixture master is already mono 16kHz PCM."""
    out_path.write_bytes(master.read_bytes())


def _write_master(tmp_path, labels=LABELS):
    sources = {"voice": voiced_samples, "noise": noise_samples, "quiet": near_silence_samples}
    blocks = []
    for index, label in enumerate(labels):
        blocks.append(sources[label](SPAN, seed=index + 10))
        blocks.append(near_silence_samples(1.0, seed=index + 50))
    return write_samples_wav(tmp_path / "master.wav", blocks)


def _write_segments(tmp_path, labels=LABELS, wrapped=False):
    rows = [
        {"index": i, "text": f"lyric {i}", "start": i * 3.0, "end": i * 3.0 + SPAN}
        for i, _ in enumerate(labels)
    ]
    payload = {"segments": rows} if wrapped else rows
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(tmp_path, extra=(), labels=LABELS, wrapped=False):
    master = _write_master(tmp_path, labels)
    segments = _write_segments(tmp_path, labels, wrapped)
    argv = ["--segments", str(segments), "--master", str(master), *extra]
    return calibrate_voicing.main(argv, decoder=_copy_decoder)


def test_the_table_ranks_the_least_voiced_segment_first(capsys, tmp_path):
    assert _run(tmp_path) == 0

    out = capsys.readouterr().out
    header_index = out.index("idx")
    rows = [line for line in out[header_index:].splitlines() if line[:1].isdigit()]
    assert rows[0].split()[0] == "4"  # the noise segment
    assert "would flag 1 of 5 segment(s): [4]" in out


def test_every_run_prints_its_provenance_and_its_constants(capsys, tmp_path):
    _run(tmp_path)

    out = capsys.readouterr().out
    assert "master           " in out
    assert "bytes, mtime" in out
    assert "segments from    " in out
    assert "level floor      " in out
    assert "UNCALIBRATED -- this run is the calibration" in out
    assert "NCCF 70-500Hz at 8000Hz" in out
    # The two columns the operator is told to read before moving anything.
    assert "jitter%" in out
    assert "f0_span" in out


def test_a_named_window_is_scored_against_the_same_decode(capsys, tmp_path):
    assert _run(tmp_path, ["--window", "12.000-14.000"]) == 0

    out = capsys.readouterr().out
    assert "window" in out
    assert "(--window)" in out


def test_a_window_too_short_to_measure_says_so_rather_than_scoring_zero(capsys, tmp_path):
    assert _run(tmp_path, ["--window", "1.000:1.010"]) == 0

    assert "too short to measure" in capsys.readouterr().out


def test_the_csv_carries_the_same_provenance_as_the_table(tmp_path):
    out_csv = tmp_path / "voicing.csv"

    assert _run(tmp_path, ["--csv", str(out_csv)]) == 0

    text = out_csv.read_text(encoding="utf-8")
    assert text.startswith("# master ")
    assert "# segments from" in text
    assert "idx,start,end,dur" in text
    assert len([line for line in text.splitlines() if line[:1].isdigit()]) == len(LABELS)


def test_a_wrapped_segments_payload_is_accepted(tmp_path):
    assert _run(tmp_path, wrapped=True) == 0


@pytest.mark.parametrize("window", ["228.590", "a-b", "230.0-228.0"])
def test_a_malformed_window_exits_two_without_a_traceback(tmp_path, window, caplog):
    assert _run(tmp_path, ["--window", window]) == 2
    assert "calibrate_voicing" in caplog.text


def test_segments_without_master_exits_two(tmp_path, caplog):
    segments = _write_segments(tmp_path)

    assert calibrate_voicing.main(["--segments", str(segments)], decoder=_copy_decoder) == 2
    assert "needs --master" in caplog.text


def test_a_malformed_segments_file_exits_two(tmp_path, caplog):
    master = _write_master(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('[{"index": 0, "text": "x"}]', encoding="utf-8")

    assert (
        calibrate_voicing.main(
            ["--segments", str(bad), "--master", str(master)], decoder=_copy_decoder
        )
        == 2
    )
    assert "malformed" in caplog.text


def test_an_unparseable_segments_file_exits_two(tmp_path, caplog):
    master = _write_master(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{{{", encoding="utf-8")

    assert (
        calibrate_voicing.main(
            ["--segments", str(bad), "--master", str(master)], decoder=_copy_decoder
        )
        == 2
    )
    assert "could not read segments" in caplog.text


def test_a_segments_file_that_is_not_a_list_exits_two(tmp_path, caplog):
    master = _write_master(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"segments": 3}', encoding="utf-8")

    assert (
        calibrate_voicing.main(
            ["--segments", str(bad), "--master", str(master)], decoder=_copy_decoder
        )
        == 2
    )
    assert "does not hold a list" in caplog.text


def test_an_absent_master_exits_two(tmp_path, caplog):
    segments = _write_segments(tmp_path)

    assert (
        calibrate_voicing.main(
            ["--segments", str(segments), "--master", str(tmp_path / "nope.wav")],
            decoder=_copy_decoder,
        )
        == 2
    )
    assert "does not exist" in caplog.text


def test_an_empty_segment_list_exits_two(tmp_path, caplog):
    master = _write_master(tmp_path)
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")

    assert (
        calibrate_voicing.main(
            ["--segments", str(empty), "--master", str(master)], decoder=_copy_decoder
        )
        == 2
    )
    assert "no aligned segments" in caplog.text


def test_a_decode_that_produces_nothing_readable_exits_two(tmp_path, caplog):
    def _empty_decoder(master, out_path):
        out_path.write_bytes(b"")

    master = _write_master(tmp_path)
    segments = _write_segments(tmp_path)

    assert (
        calibrate_voicing.main(
            ["--segments", str(segments), "--master", str(master)], decoder=_empty_decoder
        )
        == 2
    )
    assert "could not read" in caplog.text


def test_a_track_with_nothing_periodic_refuses_rather_than_printing_nan(tmp_path, caplog):
    # The level floor is relative to the track's own median, so a wholly
    # quiet track clears its own gate -- and then every vf_ratio is a
    # division by zero. Refusing beats handing an operator a column of `nan`
    # to move a threshold on.
    assert _run(tmp_path, labels=["quiet"] * 5) == 2
    assert "no median to rank anything against" in caplog.text


def test_an_unwritable_csv_exits_two(tmp_path, caplog):
    assert _run(tmp_path, ["--csv", str(tmp_path / "absent-dir" / "x.csv")]) == 2
    assert "could not write" in caplog.text


def test_the_real_decoder_refuses_when_ffmpeg_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(calibrate_voicing.CalibrationError, match="ffmpeg is not on PATH"):
        calibrate_voicing._ffmpeg_decode(tmp_path / "in.wav", tmp_path / "out.wav")


def test_the_real_decoder_reports_an_ffmpeg_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=1, stdout=b"", stderr=b"boom"),
    )

    with pytest.raises(calibrate_voicing.CalibrationError, match="ffmpeg exited 1"):
        calibrate_voicing._ffmpeg_decode(tmp_path / "in.wav", tmp_path / "out.wav")


def test_the_real_decoder_uses_the_checks_own_argv(tmp_path, monkeypatch):
    # The whole point of sharing decode_to_mono_16khz_args: a calibration run
    # against a different decode than the check uses is worse than none.
    seen = {}
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def _capture(args, **kwargs):
        seen["args"] = list(args)
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _capture)
    calibrate_voicing._ffmpeg_decode(tmp_path / "in.wav", tmp_path / "out.wav")

    assert seen["args"][:1] == ["ffmpeg"]
    assert "-ac" in seen["args"] and "1" in seen["args"]
    assert "-ar" in seen["args"] and "16000" in seen["args"]


def test_the_config_path_aligns_the_way_a_render_does(capsys, tmp_path, monkeypatch):
    master = _write_master(tmp_path)
    segments = tuple(
        make_aligned_segment(i, f"lyric {i}", i * 3.0, i * 3.0 + SPAN, "Dianne")
        for i, _ in enumerate(LABELS)
    )
    recorded = {}

    class _Config:
        master_audio = master
        lyrics_file = tmp_path / "lyrics.txt"
        cast: dict = {}
        default_lead_vocalist = "Dianne"
        alignment_model_size = "small"
        alignment_overrides: tuple = ()

    monkeypatch.setattr("music_video_maker.config.load_config", lambda path: _Config())
    monkeypatch.setattr("music_video_maker.lyrics.parse_lyrics", lambda *a, **k: ())

    def _fake_align(audio, lines, **kwargs):
        recorded.update(kwargs)
        return AlignmentResult(segments=segments, track_duration=len(LABELS) * 3.0)

    monkeypatch.setattr("music_video_maker.alignment.align", _fake_align)

    assert (
        calibrate_voicing.main(
            ["--config", str(tmp_path / "run.toml")], decoder=_copy_decoder
        )
        == 0
    )
    # Same alignment the render would produce, or the table describes a
    # different timeline than the check it is calibrating.
    assert recorded["model_size"] == "small"
    assert "aligned by this command" in capsys.readouterr().out


def test_a_bad_config_exits_two(tmp_path, monkeypatch, caplog):
    from music_video_maker.config import ConfigError

    def _boom(path):
        raise ConfigError("no such key")

    monkeypatch.setattr("music_video_maker.config.load_config", _boom)

    assert calibrate_voicing.main(["--config", str(tmp_path / "run.toml")]) == 2
    assert "could not load" in caplog.text
