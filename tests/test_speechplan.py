"""speechplan: synthesized lines laid out so every cut falls in silence."""

from __future__ import annotations

import array
import json
import math
import wave
from pathlib import Path

import pytest

from music_video_maker import speechplan
from music_video_maker.contracts import FrameGrid
from music_video_maker.shot_plan import load_shot_plan, shot_length_requests
from music_video_maker.speechplan import SpeechLine, SpeechPlanError, plan_speech

RATE = 24000


def _tone(seconds: float, *, lead: float = 0.0, tail: float = 0.0) -> array.array:
    """A loud tone (the 'speech') padded with exact digital silence."""
    out = array.array("h", bytes(2 * round(lead * RATE)))
    out.extend(
        int(12000 * math.sin(2 * math.pi * 220 * i / RATE)) or 1000
        for i in range(round(seconds * RATE))
    )
    out.extend(array.array("h", bytes(2 * round(tail * RATE))))
    return out


def _wav(path: Path, samples: array.array, *, rate: int = RATE, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def _lines(tmp_path: Path, seconds: list[float]) -> list[SpeechLine]:
    return [
        SpeechLine(
            text=f"line {i}",
            audio=_wav(tmp_path / f"l{i}.wav", _tone(s, lead=0.2, tail=0.3)),
        )
        for i, s in enumerate(seconds)
    ]


def _is_speech(sample: int) -> bool:
    return abs(sample) > int(speechplan.DEFAULT_SILENCE_THRESHOLD * 32767)


def test_every_chunk_starts_on_speech_and_ends_in_silence(tmp_path):
    plan = plan_speech(_lines(tmp_path, [2.0, 1.5, 3.0, 2.5, 1.0, 4.0]))
    grid = FrameGrid()
    for chunk in plan.chunks:
        assert grid.is_valid(chunk.frames)
        assert 124 <= chunk.frames <= 141
        first = round(chunk.start * RATE)
        assert _is_speech(plan.samples[first]), f"chunk {chunk.chunk_id} starts on silence"
        end = round((chunk.start + chunk.frames / grid.fps) * RATE)
        tail = plan.samples[end - round(speechplan.DEFAULT_MIN_TAIL * RATE) : end]
        assert not any(_is_speech(s) for s in tail), f"chunk {chunk.chunk_id} cut mid-speech"


def test_chunks_tile_the_master_exactly_on_the_frame_grid(tmp_path):
    plan = plan_speech(_lines(tmp_path, [2.0, 2.0, 2.0, 2.0, 2.0]))
    total_frames = sum(c.frames for c in plan.chunks)
    assert len(plan.samples) == round(total_frames * RATE / 24)
    starts = [c.start for c in plan.chunks]
    assert starts[0] == 0.0
    for previous, nxt in zip(plan.chunks, plan.chunks[1:], strict=False):
        assert nxt.start == pytest.approx(previous.start + previous.frames / 24, abs=1 / RATE)
    assert [i for c in plan.chunks for i in c.line_indices] == list(range(5))


def test_lines_are_grouped_greedily_under_the_ceiling(tmp_path):
    # 2.0 + 0.3 + 2.0 + 0.25 tail = 4.55s fits in 141 frames (5.875s); a third line does not.
    plan = plan_speech(_lines(tmp_path, [2.0, 2.0, 2.0]))
    assert [c.line_indices for c in plan.chunks] == [(0, 1), (2,)]


def test_short_chunks_are_padded_up_to_min_frames(tmp_path):
    plan = plan_speech(_lines(tmp_path, [0.5]))
    assert plan.chunks[0].frames == 124
    assert plan.chunks[0].speech_seconds == pytest.approx(0.5, abs=0.01)


def test_a_line_too_long_for_one_chunk_is_refused_by_name(tmp_path):
    with pytest.raises(SpeechPlanError, match="line 1 speaks for .*Split it"):
        plan_speech(_lines(tmp_path, [1.0, 5.8]))


def test_a_larger_ceiling_admits_the_same_line(tmp_path):
    plan = plan_speech(_lines(tmp_path, [1.0, 5.8]), max_frames=192)
    assert all(c.frames <= 192 for c in plan.chunks)


def test_off_grid_bounds_are_refused(tmp_path):
    with pytest.raises(SpeechPlanError, match="grid"):
        plan_speech(_lines(tmp_path, [1.0]), max_frames=140)
    with pytest.raises(SpeechPlanError, match="exceeds"):
        plan_speech(_lines(tmp_path, [1.0]), min_frames=141, max_frames=124)


def test_no_lines_is_refused():
    with pytest.raises(SpeechPlanError, match="no lines"):
        plan_speech([])


def test_mismatched_rates_are_refused(tmp_path):
    lines = _lines(tmp_path, [1.0])
    lines.append(SpeechLine("other", _wav(tmp_path / "x.wav", _tone(1.0), rate=22050)))
    with pytest.raises(SpeechPlanError, match="22050 Hz"):
        plan_speech(lines)


def test_stereo_and_silent_and_unreadable_clips_are_refused(tmp_path):
    stereo = _wav(tmp_path / "s.wav", _tone(1.0), channels=2)
    with pytest.raises(SpeechPlanError, match="16-bit mono"):
        plan_speech([SpeechLine("s", stereo)])
    silent = _wav(tmp_path / "q.wav", array.array("h", bytes(2 * RATE)))
    with pytest.raises(SpeechPlanError, match="silent"):
        plan_speech([SpeechLine("q", silent)])
    (tmp_path / "bad.wav").write_text("not audio")
    with pytest.raises(SpeechPlanError, match="cannot read"):
        plan_speech([SpeechLine("b", tmp_path / "bad.wav")])


def test_trim_silence_finds_the_speech():
    samples = _tone(1.0, lead=0.5, tail=0.5)
    first, end = speechplan.trim_silence(samples)
    assert first == pytest.approx(0.5 * RATE, abs=2)
    assert end == pytest.approx(1.5 * RATE, abs=2)
    assert speechplan.trim_silence(array.array("h", bytes(200))) == (0, 0)


def test_written_shot_plan_pins_every_chunk_through_the_real_loader(tmp_path):
    plan = plan_speech(_lines(tmp_path, [2.0, 3.0, 1.5, 4.0, 2.2]))
    path = speechplan.write_shot_plan(plan, tmp_path / "shot_plan.toml", shot="She speaks.")
    entries = load_shot_plan(path)
    assert sorted(entries) == [c.chunk_id for c in plan.chunks]
    requests = shot_length_requests(entries)
    assert [(r.start, r.length_seconds) for r in requests] == [
        (c.start, c.frames / 24) for c in plan.chunks
    ]
    assert all(e.shot == "She speaks." for e in entries.values())


def test_master_and_script_round_trip(tmp_path):
    plan = plan_speech(_lines(tmp_path, [1.0, 2.0]))
    master = speechplan.write_master(plan, tmp_path / "master.wav")
    with wave.open(str(master), "rb") as handle:
        assert (handle.getnchannels(), handle.getframerate()) == (1, RATE)
        assert handle.getnframes() == len(plan.samples)
    script = speechplan.write_script(plan, tmp_path / "script.txt")
    assert script.read_text().splitlines() == ["line 0", "line 1"]


def test_cli_writes_all_three_files(tmp_path, caplog):
    for i, s in enumerate([1.0, 2.0]):
        _wav(tmp_path / f"c{i}.wav", _tone(s, tail=0.2))
    lines = tmp_path / "lines.json"
    lines.write_text(
        json.dumps([{"text": "one", "audio": "c0.wav"}, {"text": "two", "audio": "c1.wav"}])
    )
    out = tmp_path / "out"
    assert speechplan.main([str(lines), "--out-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"master.wav", "script.txt", "shot_plan.toml"}
    assert "1 chunks" in caplog.text


def test_cli_reports_a_bad_lines_file(tmp_path):
    bad = tmp_path / "lines.json"
    bad.write_text(json.dumps({"text": "not a list"}))
    assert speechplan.main([str(bad), "--out-dir", str(tmp_path / "o")]) == 1
    (tmp_path / "broken.json").write_text("{")
    assert speechplan.main([str(tmp_path / "broken.json"), "--out-dir", str(tmp_path / "o")]) == 1
