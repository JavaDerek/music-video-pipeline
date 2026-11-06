"""Tests for the setup-still check: does a rendered shot open on its camera setup's still?

A dialogue piece cuts back to the same angle many times. Rendered one shot at a time from a
portrait and a prompt, every return to an angle invented its own background; started from one
still per setup, every return opened on that still. ``setupmatch`` measures the second property
on the rendered file, because an instruction to the model is not a property of the output (#81).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker import setupmatch as sm

SIZE = sm.THUMB_WIDTH * sm.THUMB_HEIGHT * 3


def _flat(value: int) -> bytes:
    return bytes([value]) * SIZE


class FakeRunner:
    """Answers each ffmpeg call with the thumbnail registered for its input path."""

    def __init__(self, thumbs: dict[str, bytes], returncode: int = 0) -> None:
        self.thumbs = thumbs
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        src = args[args.index("-i") + 1]
        out = self.thumbs.get(src, b"")
        err = b"boom" if self.returncode else b""
        return subprocess.CompletedProcess(args, self.returncode, out, err)


def test_thumbnail_args_exact():
    assert sm.build_thumbnail_args(Path("/takes/s07.mp4")) == [
        "ffmpeg", "-v", "error", "-i", "/takes/s07.mp4",
        "-vf", "select=eq(n\\,0),scale=54:30:flags=area,gblur=sigma=1.5",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]


def test_thumbnail_args_other_frame():
    args = sm.build_thumbnail_args(Path("a.mp4"), frame=71)
    assert "select=eq(n\\,71),scale=54:30:flags=area,gblur=sigma=1.5" in args


def test_distance_is_mean_absolute_difference():
    assert sm.distance(_flat(10), _flat(10)) == 0.0
    assert sm.distance(_flat(10), _flat(40)) == 30.0


def test_distance_refuses_mismatched_sizes():
    with pytest.raises(sm.SetupMatchError):
        sm.distance(_flat(1), b"\x00" * 3)


def test_thumbnail_refuses_short_output():
    runner = FakeRunner({"x.mp4": b"\x00" * 10})
    with pytest.raises(sm.SetupMatchError, match="x.mp4"):
        sm.thumbnail(Path("x.mp4"), runner=runner)


def test_thumbnail_refuses_ffmpeg_failure():
    runner = FakeRunner({"x.mp4": _flat(0)}, returncode=1)
    with pytest.raises(sm.SetupMatchError, match="boom"):
        sm.thumbnail(Path("x.mp4"), runner=runner)


def test_check_shot_matches_when_close():
    runner = FakeRunner({"still.png": _flat(100), "shot.mp4": _flat(104)})
    result = sm.check_shot(Path("still.png"), Path("shot.mp4"), runner=runner)
    assert result.distance == 4.0
    assert result.matches
    assert result.frame == 0


def test_check_shot_flags_a_new_background():
    runner = FakeRunner({"still.png": _flat(100), "shot.mp4": _flat(130)})
    result = sm.check_shot(Path("still.png"), Path("shot.mp4"), runner=runner)
    assert not result.matches
    assert "30.0" in result.describe()


def test_threshold_sits_in_the_measured_gap():
    # Calibration recorded beside the constant: every anchored first frame measured at or under
    # the first number, every unanchored one at or over the second.
    assert sm.MEASURED_ANCHORED_MAX < sm.DEFAULT_MAX_DISTANCE < sm.MEASURED_UNANCHORED_MIN


def test_cli_exit_codes(monkeypatch, capsys):
    runner = FakeRunner({"still.png": _flat(100), "a.mp4": _flat(101), "b.mp4": _flat(160)})
    monkeypatch.setattr(sm, "_default_runner", runner)
    assert sm.main(["still.png", "a.mp4"]) == 0
    assert sm.main(["still.png", "a.mp4", "b.mp4"]) == 1
    out = capsys.readouterr().out
    assert "b.mp4" in out and "MISMATCH" in out
    assert sm.main(["still.png", "missing.mp4"]) == 2


def test_cli_max_distance(monkeypatch):
    runner = FakeRunner({"still.png": _flat(100), "a.mp4": _flat(120)})
    monkeypatch.setattr(sm, "_default_runner", runner)
    assert sm.main(["still.png", "a.mp4"]) == 1
    assert sm.main(["--max-distance", "25", "still.png", "a.mp4"]) == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_real_ffmpeg_end_to_end(tmp_path):
    def make(name: str, color: str) -> Path:
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color={color}:size=160x90",
                        "-frames:v", "1", str(out)], check=True)
        return out

    still = make("still.png", "0x806040")
    same = make("same.png", "0x806040")
    other = make("other.png", "0x2060c0")
    assert sm.check_shot(still, same).matches
    assert not sm.check_shot(still, other).matches
