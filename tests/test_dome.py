"""Tests for the fulldome (domemaster) output path -- ``music_video_maker/dome.py``.

Everything here runs offline with the subprocess runner injected, the same seam
``assembly.py`` and ``luminance.py`` use: the argv builders are asserted on
their exact argument lists, and the verifier is exercised against a fake
runner that returns scripted ffprobe JSON and raw pixel/sample bytes. One
``@pytest.mark.integration`` test drives real ffmpeg end to end at a toy size
(a 128 px domemaster, ten frames a second, two seconds of programme) so the
filter graph itself is proven, not just its spelling.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from music_video_maker import dome
from music_video_maker.dome import (
    SURROUND_CHANNELS,
    DomeSpec,
    WindowPlacement,
    build_audio_args,
    build_background_args,
    build_contact_sheet_args,
    build_disc_mask_args,
    build_distribution_args,
    build_master_args,
    build_master_filter,
    build_preview_args,
    flat_v_fov_degrees,
    inside_span,
    retime_filter,
    verify_render,
)

# --------------------------------------------------------------------------- #
# Geometry and timing
# --------------------------------------------------------------------------- #


def test_spec_defaults_are_the_target_spec():
    spec = DomeSpec()
    assert spec.size == 4096
    assert spec.fps == 30
    assert spec.bit_depth == 16
    assert spec.sample_rate == 48000
    assert spec.leader_seconds == 3.0
    assert spec.pop_seconds == 1.0


def test_spec_frame_and_sample_counts_for_a_sixty_second_programme():
    spec = DomeSpec()
    assert spec.total_seconds(60.0) == 63.0
    assert spec.frame_count(60.0) == 1890
    assert spec.sample_count(60.0) == 63 * 48000
    assert spec.pop_frame == 30
    assert spec.program_start_frame == 90


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 4095},  # odd: the circle has no integer centre
        {"size": 0},
        {"fps": 0},
        {"pop_seconds": 3.0, "leader_seconds": 3.0},  # the pop must precede the programme
        {"pop_seconds": -1.0},
        {"bit_depth": 12},
        {"sample_rate": 0},
    ],
)
def test_spec_refuses_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DomeSpec(**kwargs)


def test_window_pitch_is_measured_from_the_zenith():
    # 40 degrees above the horizon is 50 degrees down from the dome's centre.
    assert WindowPlacement(elevation_degrees=40.0).pitch_degrees == 50.0
    assert WindowPlacement(elevation_degrees=90.0).pitch_degrees == 0.0
    # azimuth is a turn about the zenith -- roll -- and adds to the venue's
    spec = DomeSpec(front_rotation_degrees=10.0)
    assert WindowPlacement(azimuth_degrees=15.0).roll_degrees(spec) == pytest.approx(
        10.0 + dome.FISHEYE_ROLL_SIGN * 15.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"h_fov_degrees": 0.0}, {"h_fov_degrees": 180.0}, {"elevation_degrees": -1.0},
     {"elevation_degrees": 91.0}, {"feather_px": -1}],
)
def test_window_refuses_invalid_values(kwargs):
    with pytest.raises(ValueError):
        WindowPlacement(**kwargs)


def test_flat_v_fov_follows_the_tangent_not_the_aspect_ratio():
    # A gnomonic (flat) image's vertical FOV is NOT h_fov * h / w: at 90 degrees
    # wide an 864x480 (1.8:1) frame spans 58.1 degrees, not 50.0.
    v = flat_v_fov_degrees(864, 480, 90.0)
    assert v == pytest.approx(58.11, abs=0.01)
    assert flat_v_fov_degrees(100, 100, 90.0) == pytest.approx(90.0)
    assert flat_v_fov_degrees(1920, 1080, 60.0) == pytest.approx(35.98, abs=0.01)


def test_inside_span_matches_brute_force_on_a_small_disc():
    size = 16
    r = size / 2
    for row in range(size):
        brute = [x for x in range(size) if math.hypot(x + 0.5 - r, row + 0.5 - r) <= r]
        span = inside_span(row, size)
        if not brute:
            assert span is None
        else:
            assert span == (brute[0], brute[-1] + 1)


def test_inside_span_is_full_width_on_the_middle_rows_and_narrow_at_the_poles():
    assert inside_span(2047, 4096) == (0, 4096)
    assert inside_span(2048, 4096) == (0, 4096)
    top = inside_span(0, 4096)
    assert top is not None
    x0, x1 = top
    assert 0 < x1 - x0 < 200
    assert x0 + x1 == 4096  # symmetric about the centre


def test_retime_modes():
    assert retime_filter("mci", 30).startswith("minterpolate=fps=30:mi_mode=mci")
    assert retime_filter("blend", 30) == "minterpolate=fps=30:mi_mode=blend"
    assert retime_filter("dup", 30) == "fps=30"
    with pytest.raises(ValueError):
        retime_filter("magic", 30)


# --------------------------------------------------------------------------- #
# argv builders
# --------------------------------------------------------------------------- #


def test_disc_mask_args_draw_an_edge_to_edge_circle(tmp_path):
    out = tmp_path / "disc.png"
    args = build_disc_mask_args(out, size=4096)
    assert args[0] == "ffmpeg"
    assert str(out) == args[-1]
    lavfi = args[args.index("-i") + 1]
    assert "s=4096x4096" in lavfi
    assert "hypot(X+0.5-2048.0,Y+0.5-2048.0)" in lavfi
    assert "2048.0),255,0)" in lavfi
    assert "-frames:v" in args and args[args.index("-frames:v") + 1] == "1"


def test_background_args_project_an_equirect_texture_to_a_180_fisheye(tmp_path):
    out = tmp_path / "bg.png"
    args = build_background_args(out, size=512, star_density=0.001, max_brightness=140)
    lavfi = args[args.index("-i") + 1]
    assert "s=1024x512" in lavfi  # the equirect texture is 2:1 at the dome's width
    assert "v360=input=equirect:output=fisheye:h_fov=180:v_fov=180:w=512:h=512" in lavfi
    assert "140" in lavfi and "0.999" in lavfi
    assert str(out) == args[-1]


@pytest.fixture
def small_spec():
    return DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5)


def test_master_filter_places_the_window_below_the_zenith(small_spec):
    window = WindowPlacement(h_fov_degrees=90.0, elevation_degrees=40.0, feather_px=8)
    graph = build_master_filter(
        small_spec, window, source_width=864, source_height=480, source_start=2.0,
        program_seconds=6.0, retime="dup", credits=None,
    )
    assert "trim=start=2.0:end=8.0" in graph
    assert "fps=10" in graph
    # v360 flat -> fisheye with the tangent-derived vertical FOV, pitched down
    # from the zenith by (90 - elevation), no roll by default.
    assert (
        "v360=input=flat:ih_fov=90.0:iv_fov=58.11:output=fisheye:h_fov=180:v_fov=180"
        ":w=256:h=256:yaw=0.0:pitch=50.0:roll=0.0:interp=lanc" in graph
    )
    # feathered alpha on the flat clip, in pixels of the source
    assert "a='255*clip(min(min(X,W-1-X),min(Y,H-1-Y))/8,0,1)'" in graph
    # the leader, the pop frame and the hard disc mask
    assert "tpad=start_duration=1.0:color=black" in graph
    assert "enable='eq(n,5)'" in graph
    assert "blend=all_mode=multiply" in graph
    assert "[3:v]" not in graph  # no credits input when none is given
    assert graph.endswith("format=rgb48be[out]")


def test_master_filter_applies_the_venue_rotation_to_every_projected_layer():
    spec = DomeSpec(front_rotation_degrees=90.0)
    credits = dome.CreditsLayer(
        image_width=1920, image_height=1080,
        placement=WindowPlacement(h_fov_degrees=70.0, elevation_degrees=35.0),
        at_seconds=52.0, fade_seconds=1.0,
    )
    graph = build_master_filter(
        spec, WindowPlacement(), source_width=864, source_height=480, source_start=0.0,
        program_seconds=60.0, retime="mci", credits=credits,
    )
    assert graph.count("roll=90.0") == 2
    assert "[3:v]" in graph
    assert "a='alpha(X,Y)*gt(min(min(X,W-1-X),min(Y,H-1-Y)),0)*clip((T-52.0)/1.0,0,1)'" in graph
    assert "enable='gte(t,52.0)'" in graph
    assert "pitch=55.0" in graph  # 90 - 35 for the credits card
    assert "minterpolate=fps=30:mi_mode=mci" in graph


def test_master_args_wire_inputs_in_filter_order_and_cap_the_frame_count(tmp_path):
    spec = DomeSpec()
    args = build_master_args(
        source=tmp_path / "src.mp4", background_png=tmp_path / "bg.png",
        disc_png=tmp_path / "disc.png", credits_png=tmp_path / "credits.png",
        out_pattern=tmp_path / "frames" / "dome_%06d.png", spec=spec,
        window=WindowPlacement(), source_width=864, source_height=480,
        source_start=2.0, program_seconds=60.0, retime="mci",
        credits=dome.CreditsLayer(1920, 1080, WindowPlacement(), 52.0, 1.0),
    )
    inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    assert inputs == [
        str(tmp_path / "src.mp4"), str(tmp_path / "bg.png"),
        str(tmp_path / "disc.png"), str(tmp_path / "credits.png"),
    ]
    # every still is looped at the output rate and bounded to the total length
    assert args.count("-loop") == 3
    assert args.count("-framerate") == 3
    assert args[args.index("-frames:v") + 1] == "1890"
    assert args[args.index("-start_number") + 1] == "0"
    assert args[-1] == str(tmp_path / "frames" / "dome_%06d.png")
    assert "-filter_complex" in args
    assert args[args.index("-map") + 1] == "[out]"


def test_master_args_eight_bit_writes_rgb24(tmp_path):
    args = build_master_args(
        source=tmp_path / "src.mp4", background_png=tmp_path / "bg.png",
        disc_png=tmp_path / "disc.png", credits_png=None,
        out_pattern=tmp_path / "f_%06d.png", spec=DomeSpec(bit_depth=8),
        window=WindowPlacement(), source_width=864, source_height=480,
        source_start=0.0, program_seconds=1.0, retime="dup", credits=None,
    )
    graph = args[args.index("-filter_complex") + 1]
    assert graph.endswith("format=rgb24[out]")
    assert args.count("-loop") == 2


def test_master_args_refuse_a_credits_image_without_a_layer(tmp_path):
    with pytest.raises(ValueError):
        build_master_args(
            source=tmp_path / "src.mp4", background_png=tmp_path / "bg.png",
            disc_png=tmp_path / "disc.png", credits_png=tmp_path / "c.png",
            out_pattern=tmp_path / "f_%06d.png", spec=DomeSpec(),
            window=WindowPlacement(), source_width=864, source_height=480,
            source_start=0.0, program_seconds=1.0, retime="dup", credits=None,
        )


def test_distribution_args_are_h265_at_sixty_megabits_in_rec709(tmp_path):
    args = build_distribution_args(tmp_path / "f_%06d.png", tmp_path / "out.mp4", fps=30)
    assert args[args.index("-c:v") + 1] == "libx265"
    assert args[args.index("-b:v") + 1] == "60M"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-framerate") + 1] == "30"
    assert "out_color_matrix=bt709" in args[args.index("-vf") + 1]
    assert args[args.index("-colorspace") + 1] == "bt709"
    assert "-an" in args
    assert args[-1] == str(tmp_path / "out.mp4")


def test_audio_args_cut_the_surround_at_the_offset_and_split_six_monos(tmp_path):
    spec = DomeSpec()
    args = build_audio_args(
        surround=tmp_path / "mix.flac", surround_start_seconds=0.711854,
        program_seconds=60.0, spec=spec, out_dir=tmp_path / "audio",
    )
    graph = args[args.index("-filter_complex") + 1]
    # 0.711854 s * 48000 = 34169 samples, sample-accurate, not a -ss seek
    assert "atrim=start_sample=34169:end_sample=2914169" in graph
    assert "adelay=delays=3000:all=1" in graph
    assert f"apad=whole_len={63 * 48000}" in graph
    # -20 dBFS 1 kHz tone, one frame long, two seconds before programme start
    assert "0.1*sin(2*PI*1000*t)*between(t,1.0,1.033333)" in graph
    assert graph.count("between(") == 6  # one expression per channel
    assert "channelsplit=channel_layout=5.1(side)[FL][FR][FC][LFE][SL][SR]" in graph
    assert "pan=stereo|" in graph
    outputs = [a for a in args if a.endswith(".wav")]
    assert outputs == [str(tmp_path / "audio" / f"{name}.wav") for name in SURROUND_CHANNELS] + [
        str(tmp_path / "audio" / "stereo.wav")
    ]
    assert args.count("pcm_s24le") == 7
    assert args.count("48000") >= 7


def test_audio_args_refuse_a_negative_surround_start(tmp_path):
    with pytest.raises(ValueError):
        build_audio_args(
            surround=tmp_path / "mix.flac", surround_start_seconds=-0.5,
            program_seconds=60.0, spec=DomeSpec(), out_dir=tmp_path,
        )


def test_preview_args_look_at_the_front_from_the_seat(tmp_path):
    args = build_preview_args(
        tmp_path / "dist.mp4", tmp_path / "stereo.wav", tmp_path / "preview.mp4",
        elevation_degrees=30.0, h_fov_degrees=100.0,
    )
    vf = args[args.index("-vf") + 1]
    assert "v360=input=fisheye:ih_fov=180:iv_fov=180:output=flat:h_fov=100.0" in vf
    assert "pitch=-60.0" in vf  # fisheye INPUT: the sign mirrors the layers' (measured)
    assert args[-1] == str(tmp_path / "preview.mp4")
    assert "-shortest" in args


def test_contact_sheet_args_tile_evenly_spaced_frames(tmp_path):
    args = build_contact_sheet_args(tmp_path / "f_%06d.png", tmp_path / "sheet.png",
                                    frame_count=1890, columns=4, rows=2)
    vf = args[args.index("-vf") + 1]
    assert "select='not(mod(n,236))'" in vf  # 1890 // 8
    assert "tile=4x2" in vf
    assert args[-1] == str(tmp_path / "sheet.png")


# --------------------------------------------------------------------------- #
# Verification against a fake runner
# --------------------------------------------------------------------------- #


def _frame_bytes(size: int, bit_depth: int, *, inside_value: int, poison: tuple[int, int] | None):
    """Raw RGB bytes of one frame: ``inside_value`` inside the disc, zero outside,
    optionally one poisoned pixel at ``poison=(x, y)``."""
    bpc = 2 if bit_depth == 16 else 1
    row_len = size * 3 * bpc
    out = bytearray(size * row_len)
    for y in range(size):
        span = inside_span(y, size)
        if span is None:
            continue
        x0, x1 = span
        fill = bytes([inside_value]) * ((x1 - x0) * 3 * bpc)
        start = y * row_len + x0 * 3 * bpc
        out[start:start + len(fill)] = fill
    if poison is not None:
        px, py = poison
        out[py * row_len + px * 3 * bpc] = 1
    return bytes(out)


class FakeRunner:
    """Scripts every subprocess the verifier makes, keyed on what it asked for."""

    def __init__(self, *, spec: DomeSpec, program_seconds: float, frame_count: int,
                 nb_frames_mp4: int | None = None, poison_outside: bool = False,
                 audio_pop_sample: int | None = None, video_pop_frame: int | None = None,
                 audio_samples: int | None = None, channels: int = 1,
                 luma_at=None):
        self.spec = spec
        self.program_seconds = program_seconds
        self.frame_count = frame_count
        self.nb_frames_mp4 = frame_count if nb_frames_mp4 is None else nb_frames_mp4
        self.poison_outside = poison_outside
        self.audio_pop_sample = spec.pop_seconds * spec.sample_rate if audio_pop_sample is None \
            else audio_pop_sample
        self.video_pop_frame = spec.pop_frame if video_pop_frame is None else video_pop_frame
        self.audio_samples = spec.sample_count(program_seconds) if audio_samples is None \
            else audio_samples
        self.channels = channels
        pop = self.video_pop_frame
        self.luma_at = luma_at or (lambda i: 255 if i == pop else 20)
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if args[0] == "ffprobe":
            return self._probe(args)
        if "-f" in args and args[args.index("-f") + 1] == "rawvideo":
            if args[args.index("-i") + 1].endswith(".mp4"):  # the whole-programme luma scan
                n = dome.LUMA_SCAN_SIZE
                frames = [bytes([self.luma_at(i)]) * (n * n) for i in range(self.nb_frames_mp4)]
                return subprocess.CompletedProcess(args, 0, b"".join(frames), b"")
            if "-s" in args:  # the 32x18 luma probe
                frame = Path(args[args.index("-i") + 1])
                idx = int(frame.stem.split("_")[-1])
                value = 255 if idx == self.video_pop_frame else 4
                return subprocess.CompletedProcess(args, 0, bytes([value]) * (32 * 18), b"")
            frame = Path(args[args.index("-i") + 1])
            idx = int(frame.stem.split("_")[-1])
            poison = (0, 0) if (self.poison_outside and idx == self.frame_count - 1) else None
            return subprocess.CompletedProcess(
                args, 0,
                _frame_bytes(self.spec.size, self.spec.bit_depth, inside_value=9, poison=poison),
                b"",
            )
        if "-f" in args and args[args.index("-f") + 1] == "s16le":
            n = self.audio_samples
            buf = bytearray(n * 2)
            pop = int(self.audio_pop_sample)
            for i in range(pop, min(n, pop + 40)):
                buf[2 * i] = 0x10
                buf[2 * i + 1] = 0x27  # 0x2710 = 10000
            return subprocess.CompletedProcess(args, 0, bytes(buf), b"")
        raise AssertionError(f"unexpected command {args}")

    def _probe(self, args):
        path = Path(args[-1])
        if path.suffix == ".png":
            pix_fmt = "rgb48be" if self.spec.bit_depth == 16 else "rgb24"
            payload = {"streams": [{"codec_name": "png", "width": self.spec.size,
                                    "height": self.spec.size, "pix_fmt": pix_fmt}]}
        elif path.suffix == ".mp4":
            payload = {"streams": [{"codec_name": "hevc", "width": self.spec.size,
                                    "height": self.spec.size, "pix_fmt": "yuv420p",
                                    "r_frame_rate": f"{self.spec.fps}/1",
                                    "nb_frames": str(self.nb_frames_mp4)}],
                       "format": {"bit_rate": "60123456"}}
        else:
            channels = self.channels
            if path.name == "stereo.wav" and channels == 1:
                channels = 2
            payload = {"streams": [{"codec_name": "pcm_s24le", "channels": channels,
                                    "sample_rate": str(self.spec.sample_rate),
                                    "duration_ts": self.audio_samples,
                                    "time_base": f"1/{self.spec.sample_rate}"}]}
        return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")


def _layout(tmp_path: Path, spec: DomeSpec, program_seconds: float, frame_count: int | None = None):
    frames = tmp_path / "frames"
    frames.mkdir()
    n = spec.frame_count(program_seconds) if frame_count is None else frame_count
    for i in range(n):
        (frames / f"dome_{i:06d}.png").touch()
    audio = tmp_path / "audio"
    audio.mkdir()
    for name in SURROUND_CHANNELS:
        (audio / f"{name}.wav").touch()
    (audio / "stereo.wav").touch()
    (tmp_path / "dome_4096_h265.mp4").touch()
    return frames, audio, tmp_path / "dome_4096_h265.mp4"


def test_verify_passes_a_conforming_render(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=3)
    assert report.ok, [c for c in report.checks if not c.ok]
    names = {c.name for c in report.checks}
    assert {"frame_count", "frame_size", "bit_depth", "circle_mask", "distribution",
            "audio_files", "audio_length", "sync", "brightness", "flashes"} <= names
    sync = next(c for c in report.checks if c.name == "sync")
    assert sync.ok and "0.000 ms" in sync.detail
    # the mask check sampled first, pop, middle and last frames at least
    sampled = {Path(c[c.index("-i") + 1]).name for c in runner.calls
               if "-f" in c and c[c.index("-f") + 1] == "rawvideo" and "-s" not in c}
    assert {"dome_000000.png", "dome_000029.png"} <= sampled


def test_verify_fails_on_a_single_nonzero_pixel_outside_the_circle(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30, poison_outside=True)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=3)
    assert not report.ok
    mask = next(c for c in report.checks if c.name == "circle_mask")
    assert not mask.ok and "dome_000029.png" in mask.detail


def test_verify_fails_on_a_short_frame_sequence_and_a_short_mp4(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0, frame_count=29)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=29, nb_frames_mp4=29)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=2)
    failed = {c.name for c in report.checks if not c.ok}
    assert {"frame_count", "distribution"} <= failed


def test_verify_measures_a_sync_offset_and_fails_past_half_a_frame(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    # audio pop 0.3 s late: three frames at 10 fps
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30,
                        audio_pop_sample=int(0.8 * spec.sample_rate))
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=2)
    sync = next(c for c in report.checks if c.name == "sync")
    assert not sync.ok
    assert "+300.000 ms" in sync.detail
    assert report.sync_offset_seconds == pytest.approx(0.3, abs=1e-6)


def test_verify_fails_when_the_flash_frame_is_not_where_the_leader_says(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30, video_pop_frame=7)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=2)
    sync = next(c for c in report.checks if c.name == "sync")
    assert not sync.ok and "frame 7" in sync.detail


def test_verify_fails_on_wrong_channel_count_or_audio_length(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30, channels=2,
                        audio_samples=spec.sample_count(2.0) - 1)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=2)
    failed = {c.name for c in report.checks if not c.ok}
    assert {"audio_files", "audio_length"} <= failed


def test_verify_reports_a_missing_frames_directory_without_raising(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    report = verify_render(frames_dir=tmp_path / "nowhere", audio_dir=tmp_path / "nowhere",
                           distribution=tmp_path / "nowhere.mp4", spec=spec,
                           program_seconds=2.0, runner=FakeRunner(spec=spec, program_seconds=2.0,
                                                                  frame_count=0))
    assert not report.ok
    assert all(not c.ok for c in report.checks if c.name in {"frame_count", "audio_files"})


def test_verify_report_serialises_to_json(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    runner = FakeRunner(spec=spec, program_seconds=2.0, frame_count=30)
    report = verify_render(frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec,
                           program_seconds=2.0, runner=runner, sample_frames=2)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["ok"] is True
    assert payload["checks"][0]["name"]


def test_disc_mean_luma_ignores_the_black_corners():
    n = 8
    frame = bytearray(n * n)
    for y in range(n):
        span = inside_span(y, n)
        if span:
            frame[y * n + span[0]:y * n + span[1]] = bytes([100]) * (span[1] - span[0])
    assert dome.disc_mean_luma(bytes(frame), n) == pytest.approx(100.0)


def test_count_flashes_counts_opposing_swing_pairs_per_second():
    fps = 10
    steady = [20.0] * 30
    assert dome.max_flashes_per_second(steady, fps) == 0
    # a slow swell and fall (one transition each way over 3 s) is one flash at most
    swell = [20.0 + 60 * math.sin(math.pi * i / 30) for i in range(31)]
    assert dome.max_flashes_per_second(swell, fps) <= 1
    # 5 Hz strobe: dark/bright every frame pair -> ten transitions a second -> 5 flashes
    strobe = [20.0 if (i // 1) % 2 == 0 else 120.0 for i in range(30)]
    assert dome.max_flashes_per_second(strobe, fps) == 5
    # small flicker under the swing threshold is not a flash
    flicker = [20.0 + (dome.FLASH_DELTA_Y - 1) * (i % 2) for i in range(30)]
    assert dome.max_flashes_per_second(flicker, fps) == 0


def test_verify_fails_a_strobing_programme_and_a_white_dome(tmp_path):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    start = spec.program_start_frame

    def strobe(i):
        if i == spec.pop_frame:
            return 255
        return 150 if i >= start and i % 2 else 20

    frames, audio, mp4 = _layout(tmp_path, spec, program_seconds=2.0)
    report = verify_render(
        frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec, program_seconds=2.0,
        runner=FakeRunner(spec=spec, program_seconds=2.0, frame_count=30, luma_at=strobe),
        sample_frames=2)
    failed = {c.name for c in report.checks if not c.ok}
    assert "flashes" in failed and "brightness" in failed

    # the leader's own flash is the 2-pop and is never counted
    report = verify_render(
        frames_dir=frames, audio_dir=audio, distribution=mp4, spec=spec, program_seconds=2.0,
        runner=FakeRunner(spec=spec, program_seconds=2.0, frame_count=30), sample_frames=2)
    assert next(c for c in report.checks if c.name == "flashes").ok
    assert next(c for c in report.checks if c.name == "brightness").ok


# --------------------------------------------------------------------------- #
# Base layer (a fisheye sequence rendered elsewhere) and insets
# --------------------------------------------------------------------------- #


def test_an_unfeathered_layer_still_has_a_transparent_border():
    # v360 clamps a flat input at its edge (measured): an opaque border pixel
    # would repeat across every uncovered pixel of the dome.
    graph = build_master_filter(
        DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        WindowPlacement(feather_px=0), source_width=864, source_height=480,
        source_start=0.0, program_seconds=2.0, retime="dup", credits=None,
    )
    assert "a='255*gt(min(min(X,W-1-X),min(Y,H-1-Y)),0)'" in graph


def test_window_opacity_scales_the_feathered_alpha():
    graph = build_master_filter(
        DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        WindowPlacement(feather_px=8, opacity=0.5), source_width=864, source_height=480,
        source_start=0.0, program_seconds=2.0, retime="dup", credits=None,
    )
    assert "a='255*0.5*clip(min(min(X,W-1-X),min(Y,H-1-Y))/8,0,1)'" in graph
    with pytest.raises(ValueError):
        WindowPlacement(opacity=0.0)
    with pytest.raises(ValueError):
        WindowPlacement(opacity=1.5)


def test_master_filter_takes_a_fisheye_base_and_no_source():
    spec = DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    graph = build_master_filter(
        spec, None, source_width=0, source_height=0, source_start=0.0, program_seconds=6.0,
        retime="dup", credits=None, base=True,
    )
    assert graph.startswith("[0:v]trim=end_frame=60,setpts=PTS-STARTPTS,format=gbrp[bg]")
    assert "input=flat" not in graph  # no window
    assert "[bg]null[c1]" in graph
    assert "[1:v]format=gbrp,split=2[disc][pop]" in graph


def test_master_filter_rotates_a_fisheye_base_with_the_venue():
    spec = DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5,
                    front_rotation_degrees=30.0)
    credits = dome.CreditsLayer(64, 36, WindowPlacement(), 1.0, 0.5)
    graph = build_master_filter(
        spec, WindowPlacement(h_fov_degrees=40.0), source_width=864, source_height=480,
        source_start=2.0, program_seconds=6.0, retime="dup", credits=credits, base=True,
    )
    # inputs: source 0, base 1, disc 2, credits 3
    assert "[1:v]trim=end_frame=60,setpts=PTS-STARTPTS,format=gbrp,v360=input=fisheye" in graph
    assert f"roll={dome.FISHEYE_ROLL_SIGN * 30.0:.1f}" in graph.split("[bg]")[0]
    assert "[2:v]format=gbrp,split=2[disc][pop]" in graph and "[3:v]format=rgba" in graph


def test_master_args_read_a_base_video_once_and_loop_only_stills(tmp_path):
    args = build_master_args(
        source=None, background_png=None, base=tmp_path / "base.mkv",
        disc_png=tmp_path / "disc.png", credits_png=None,
        out_pattern=tmp_path / "f_%06d.png", spec=DomeSpec(), window=None,
        source_width=0, source_height=0, source_start=0.0, program_seconds=60.0,
        retime="dup", credits=None,
    )
    inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    assert inputs == [str(tmp_path / "base.mkv"), str(tmp_path / "disc.png")]
    assert args.count("-loop") == 1


def test_master_args_refuse_nothing_to_show(tmp_path):
    with pytest.raises(ValueError, match="source or a base"):
        build_master_args(
            source=None, background_png=tmp_path / "bg.png", disc_png=tmp_path / "d.png",
            credits_png=None, out_pattern=tmp_path / "f_%06d.png", spec=DomeSpec(),
            window=None, source_width=0, source_height=0, source_start=0.0,
            program_seconds=1.0, retime="dup", credits=None,
        )


def test_window_can_keep_the_sources_own_alpha():
    graph = build_master_filter(
        DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        WindowPlacement(feather_px=0, source_alpha=True), source_width=864,
        source_height=480, source_start=0.0, program_seconds=2.0, retime="dup", credits=None,
    )
    # the cut-out's alpha, times the transparent border the projection needs
    assert "a='alpha(X,Y)*gt(min(min(X,W-1-X),min(Y,H-1-Y)),0)'" in graph


def test_master_filter_puts_a_foreground_between_the_window_and_the_credits():
    spec = DomeSpec(size=256, fps=10, leader_seconds=1.0, pop_seconds=0.5,
                    front_rotation_degrees=30.0)
    credits = dome.CreditsLayer(64, 36, WindowPlacement(), 1.0, 0.5)
    graph = build_master_filter(
        spec, WindowPlacement(h_fov_degrees=40.0), source_width=864, source_height=480,
        source_start=2.0, program_seconds=6.0, retime="dup", credits=credits, base=True,
        foreground=True,
    )
    # inputs: source 0, base 1, foreground 2, disc 3, credits 4
    fg = graph.split("[fg]")[0].rsplit(";", 1)[-1]
    assert fg.startswith("[2:v]trim=end_frame=60,setpts=PTS-STARTPTS,format=gbrap")
    assert f"roll={dome.FISHEYE_ROLL_SIGN * 30.0:.1f}" in fg
    assert "[c1][fg]overlay=format=gbrp[c1f]" in graph
    assert "[c1f][cred]overlay" in graph
    assert "[3:v]format=gbrp,split=2[disc][pop]" in graph and "[4:v]format=rgba" in graph


def test_master_args_read_a_foreground_after_the_base(tmp_path):
    args = build_master_args(
        source=None, background_png=None, base=tmp_path / "base.mkv",
        foreground=tmp_path / "land.mkv", disc_png=tmp_path / "disc.png", credits_png=None,
        out_pattern=tmp_path / "f_%06d.png", spec=DomeSpec(), window=None,
        source_width=0, source_height=0, source_start=0.0, program_seconds=60.0,
        retime="dup", credits=None,
    )
    inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    assert inputs == [str(tmp_path / "base.mkv"), str(tmp_path / "land.mkv"),
                      str(tmp_path / "disc.png")]


def test_render_refuses_a_foreground_that_would_be_scaled(tmp_path):
    base = tmp_path / "base.mkv"
    base.touch()
    fg = tmp_path / "land.mkv"
    fg.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()

    def runner(args):
        args = list(args)
        if args[0] == "ffprobe":
            w = 64 if args[-1].endswith("land.mkv") else 32
            payload = {"streams": [{"codec_name": "ffv1", "width": w, "height": w,
                                    "r_frame_rate": "10/1", "nb_read_packets": "20"}]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    with pytest.raises(ValueError, match="land.mkv"):
        dome.render(
            source=None, base=base, foreground=fg, source_start=2.0, program_seconds=2.0,
            surround=surround, surround_offset_seconds=0.0, out_dir=tmp_path / "dome",
            spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
            window=None, credits_png=None, credits=None, retime="dup", runner=runner,
        )


def _base_runner(width, height, rate, packets):
    def runner(args):
        args = list(args)
        if args[0] == "ffprobe":
            payload = {"streams": [{"codec_name": "ffv1", "width": width, "height": height,
                                    "r_frame_rate": rate, "nb_read_packets": str(packets)}]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")
    return runner


@pytest.mark.parametrize(
    ("width", "rate", "packets", "match"),
    [(128, "10/1", 20, "4096|size"), (32, "24/1", 20, "rate"), (32, "10/1", 19, "frames")],
)
def test_render_refuses_a_base_that_would_be_scaled_retimed_or_run_short(
        tmp_path, width, rate, packets, match):
    base = tmp_path / "base.mkv"
    base.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()
    with pytest.raises(ValueError, match=match):
        dome.render(
            source=None, base=base, source_start=2.0, program_seconds=2.0, surround=surround,
            surround_offset_seconds=0.0, out_dir=tmp_path / "dome",
            spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
            window=None, credits_png=None, credits=None, retime="dup",
            runner=_base_runner(width, width, rate, packets),
        )


def test_render_with_a_base_and_no_source_skips_the_starfield(tmp_path, monkeypatch):
    base = tmp_path / "base.mkv"
    base.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()
    calls = []
    inner = _base_runner(32, 32, "10/1", 20)

    def runner(args):
        calls.append(list(args))
        return inner(args)

    monkeypatch.setattr(dome, "verify_render", lambda **kw: dome.VerificationReport(checks=()))
    dome.render(
        source=None, base=base, source_start=2.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=0.0, out_dir=tmp_path / "dome",
        spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        window=None, credits_png=None, credits=None, retime="dup", runner=runner,
    )
    kinds = [dome._describe(c) for c in calls]
    assert "background" not in kinds and "master" in kinds
    manifest = json.loads((tmp_path / "dome" / "dome_manifest.json").read_text())
    assert manifest["base"] == str(base) and manifest["source"] is None


# --------------------------------------------------------------------------- #
# The orchestrator, with a runner that records instead of rendering
# --------------------------------------------------------------------------- #


def test_render_runs_every_stage_in_order_and_writes_a_manifest(tmp_path, monkeypatch):
    spec = DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    source = tmp_path / "src.mp4"
    source.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()
    out_dir = tmp_path / "dome"
    calls: list[list[str]] = []

    def recording_runner(args):
        args = list(args)
        calls.append(args)
        if args[0] == "ffprobe":
            payload = {"streams": [{"codec_name": "h264", "width": 864, "height": 480}]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(dome, "verify_render", lambda **kw: dome.VerificationReport(checks=()))
    result = dome.render(
        source=source, source_start=2.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=-1.288146, out_dir=out_dir, spec=spec,
        window=WindowPlacement(), credits_png=None, credits=None, retime="dup",
        runner=recording_runner,
    )
    kinds = [dome._describe(c) for c in calls]
    assert kinds[:2] == ["ffprobe", "background"]
    assert "disc" in kinds and "master" in kinds and "distribution" in kinds
    assert kinds.index("master") < kinds.index("distribution")
    assert "audio" in kinds and "preview" in kinds and "contact_sheet" in kinds
    manifest = json.loads((out_dir / "dome_manifest.json").read_text())
    assert manifest["spec"]["size"] == 32
    assert manifest["surround_start_seconds"] == pytest.approx(2.0 - 1.288146)
    assert manifest["commands"]
    assert result.frames_dir == out_dir / "frames"
    assert result.verification.ok  # the stubbed verifier


def test_render_refuses_a_span_that_starts_before_the_surround_mix(tmp_path):
    source = tmp_path / "src.mp4"
    source.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()
    with pytest.raises(ValueError, match="surround"):
        dome.render(
            source=source, source_start=0.5, program_seconds=2.0, surround=surround,
            surround_offset_seconds=-1.288146, out_dir=tmp_path / "dome",
            spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
            window=WindowPlacement(), credits_png=None, credits=None, retime="dup",
            runner=lambda args: subprocess.CompletedProcess(list(args), 0, b"", b""),
        )


def test_render_raises_with_stderr_when_ffmpeg_fails(tmp_path):
    source = tmp_path / "src.mp4"
    source.touch()
    surround = tmp_path / "mix.flac"
    surround.touch()

    def failing_runner(args):
        args = list(args)
        if args[0] == "ffprobe":
            payload = {"streams": [{"codec_name": "h264", "width": 864, "height": 480}]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload).encode(), b"")
        return subprocess.CompletedProcess(args, 1, b"", b"boom: filter not found")

    with pytest.raises(dome.DomeCommandError, match="filter not found"):
        dome.render(
            source=source, source_start=2.0, program_seconds=2.0, surround=surround,
            surround_offset_seconds=-1.288146, out_dir=tmp_path / "dome",
            spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
            window=WindowPlacement(), credits_png=None, credits=None, retime="dup",
            runner=failing_runner,
        )


def test_cli_parses_the_spec_and_placement(tmp_path, monkeypatch):
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        return dome.RenderResult(
            out_dir=kwargs["out_dir"], frames_dir=kwargs["out_dir"] / "frames",
            distribution=kwargs["out_dir"] / "x.mp4", audio_dir=kwargs["out_dir"] / "audio",
            preview=None, contact_sheet=None, manifest=kwargs["out_dir"] / "m.json",
            verification=dome.VerificationReport(checks=()),
        )

    monkeypatch.setattr(dome, "render", fake_render)
    rc = dome.main([
        "render", "--source", str(tmp_path / "s.mp4"), "--source-start", "2",
        "--program-seconds", "60", "--surround", str(tmp_path / "m.flac"),
        "--surround-offset", "-1.288146", "--out-dir", str(tmp_path / "o"),
        "--size", "1024", "--fps", "30", "--elevation", "42", "--h-fov", "95",
        "--front-rotation", "180", "--retime", "dup", "--bit-depth", "8",
        "--credits", str(tmp_path / "c.png"), "--credits-width", "1920",
        "--credits-height", "1080", "--credits-at", "52", "--credits-elevation", "33",
    ])
    assert rc == 0
    assert captured["spec"] == DomeSpec(size=1024, fps=30, front_rotation_degrees=180.0,
                                        bit_depth=8)
    assert captured["window"].elevation_degrees == 42.0
    assert captured["window"].h_fov_degrees == 95.0
    assert captured["credits"].placement.elevation_degrees == 33.0
    assert captured["credits"].at_seconds == 52.0
    assert captured["retime"] == "dup"
    assert captured["base"] is None


def test_cli_takes_a_base_an_inset_opacity_and_no_source(tmp_path, monkeypatch):
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)
        return dome.RenderResult(
            out_dir=kwargs["out_dir"], frames_dir=kwargs["out_dir"] / "frames",
            distribution=kwargs["out_dir"] / "x.mp4", audio_dir=kwargs["out_dir"] / "audio",
            preview=None, contact_sheet=None, manifest=kwargs["out_dir"] / "m.json",
            verification=dome.VerificationReport(checks=()),
        )

    monkeypatch.setattr(dome, "render", fake_render)
    assert dome.main([
        "render", "--base", str(tmp_path / "b.mkv"), "--program-seconds", "60",
        "--surround", str(tmp_path / "m.flac"), "--out-dir", str(tmp_path / "o"),
    ]) == 0
    assert captured["source"] is None and captured["window"] is None
    assert captured["base"] == tmp_path / "b.mkv"
    assert dome.main([
        "render", "--base", str(tmp_path / "b.mkv"), "--source", str(tmp_path / "s.mp4"),
        "--program-seconds", "60", "--surround", str(tmp_path / "m.flac"),
        "--out-dir", str(tmp_path / "o"), "--window-opacity", "0.8", "--h-fov", "40",
    ]) == 0
    assert captured["window"].opacity == 0.8 and captured["window"].h_fov_degrees == 40.0
    assert dome.main([
        "render", "--base", str(tmp_path / "b.mkv"), "--foreground", str(tmp_path / "f.mkv"),
        "--source", str(tmp_path / "s.mov"), "--source-alpha", "--program-seconds", "60",
        "--surround", str(tmp_path / "m.flac"), "--out-dir", str(tmp_path / "o"),
    ]) == 0
    assert captured["foreground"] == tmp_path / "f.mkv"
    assert captured["window"].source_alpha is True


def test_cli_verify_returns_nonzero_when_a_check_fails(tmp_path, monkeypatch):
    bad = dome.VerificationReport(checks=(dome.CheckResult("frame_count", False, "29 != 30"),))
    monkeypatch.setattr(dome, "verify_render", lambda **kw: bad)
    rc = dome.main(["verify", "--out-dir", str(tmp_path), "--program-seconds", "2",
                    "--size", "32", "--fps", "10", "--leader", "1", "--pop", "0.5"])
    assert rc == 1


# --------------------------------------------------------------------------- #
# Real ffmpeg, toy size
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_toy_domemaster_end_to_end_with_real_ffmpeg(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=128x72:r=24:d=4",
         "-vf", "lutyuv=y=val/3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True, capture_output=True,
    )
    surround = tmp_path / "mix.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=220:sample_rate=48000:duration=4",
         "-af", "pan=5.1(side)|FL=c0|FR=c0|FC=c0|LFE=c0|SL=c0|SR=c0",
         "-c:a", "flac", str(surround)],
        check=True, capture_output=True,
    )
    credits = tmp_path / "credits.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white@0.8:s=64x36",
         "-vf", "format=rgba", "-frames:v", "1", str(credits)],
        check=True, capture_output=True,
    )
    spec = DomeSpec(size=128, fps=10, leader_seconds=1.0, pop_seconds=0.5, bit_depth=16)
    result = dome.render(
        source=source, source_start=1.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=-0.5, out_dir=tmp_path / "dome", spec=spec,
        window=WindowPlacement(h_fov_degrees=90.0, elevation_degrees=40.0, feather_px=4),
        credits_png=credits,
        credits=dome.CreditsLayer(64, 36, WindowPlacement(h_fov_degrees=60.0,
                                                          elevation_degrees=30.0), 1.0, 0.5),
        retime="dup",
    )
    assert result.verification.ok, [c for c in result.verification.checks
                                    if not c.ok]
    assert len(list(result.frames_dir.glob("dome_*.png"))) == 30
    assert result.preview is not None and result.preview.exists()
    assert result.contact_sheet is not None and result.contact_sheet.exists()
    assert (result.audio_dir / "LFE.wav").exists()


def _bright_centroid_angle(png: Path, size: int, *, channel: int) -> float:
    """Angle of the centroid of bright pixels in one colour channel, measured
    from the frame's bottom (the front), positive toward the frame's right."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(png), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    pts = [(i % size, i // size) for i in range(size * size)
           if raw[3 * i + channel] > 150 and raw[3 * i + (channel + 1) % 3] < 90]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return math.degrees(math.atan2(cx - size / 2, cy - size / 2))


@pytest.mark.integration
@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_a_fisheye_base_turns_with_the_projected_layers(tmp_path, rotation):
    """Measured, not reasoned: a red marker at the base's front and a green
    window at azimuth 0 must land at the same angle for any venue rotation.
    ``v360`` rotates the *input* sphere when the input is a fisheye, which
    is why the preview's pitch is the mirror of the layers' -- this is the
    same question for roll."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    size = 128
    base = tmp_path / "base.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=black:s={size}x{size}:r=10:d=3,format=rgb24,"
         f"geq=r='if(lt(hypot(X-{size // 2},Y-{size - 14}),6),255,0)':g=0:b=0",
         "-c:v", "ffv1", str(base)], check=True, capture_output=True)
    source = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x00ff00:s=160x90:r=24:d=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True, capture_output=True)
    surround = tmp_path / "mix.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=5.1(side)",
         "-t", "3", "-c:a", "flac", str(surround)], check=True, capture_output=True)
    spec = DomeSpec(size=size, fps=10, leader_seconds=1.0, pop_seconds=0.5,
                    front_rotation_degrees=rotation)
    result = dome.render(
        source=source, base=base, source_start=0.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=0.0, out_dir=tmp_path / "dome", spec=spec,
        window=WindowPlacement(h_fov_degrees=30.0, elevation_degrees=45.0, feather_px=0),
        credits_png=None, credits=None, retime="dup",
    )
    frame = result.frames_dir / "dome_000015.png"
    red = _bright_centroid_angle(frame, size, channel=0)
    green = _bright_centroid_angle(frame, size, channel=1)
    assert red == pytest.approx(green, abs=4.0)
    if rotation:
        assert abs(red) == pytest.approx(rotation, abs=4.0)


@pytest.mark.integration
def test_a_foreground_hides_the_window_and_a_cutout_keeps_its_shape(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    size = 128
    base = tmp_path / "base.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=0x000040:s={size}x{size}:r=10:d=3", "-c:v", "ffv1", str(base)],
                   check=True, capture_output=True)
    # foreground: opaque red over the lower half of the frame (the front), clear above
    land = tmp_path / "land.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=red:s={size}x{size}:r=10:d=3,format=rgba,"
                    f"geq=r='255':g='0':b='0':a='if(gt(Y,{size // 2}),255,0)'",
                    "-c:v", "ffv1", "-pix_fmt", "bgra", str(land)],
                   check=True, capture_output=True)
    # source: a green disc on a fully transparent frame
    src = tmp_path / "cutout.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x00ff00:s=160x90:r=10:d=3,format=rgba,"
                    "geq=r='0':g='255':b='0':a='if(lt(hypot(X-80,Y-45),30),255,0)'",
                    "-c:v", "qtrle", str(src)], check=True, capture_output=True)
    surround = tmp_path / "mix.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=5.1(side)", "-t", "3", "-c:a", "flac", str(surround)],
                   check=True, capture_output=True)
    spec = DomeSpec(size=size, fps=10, leader_seconds=1.0, pop_seconds=0.5)
    result = dome.render(
        source=src, base=base, foreground=land, source_start=0.0, program_seconds=2.0,
        surround=surround, surround_offset_seconds=0.0, out_dir=tmp_path / "dome", spec=spec,
        # at the back, just short of the zenith: the disc straddles the frame's
        # middle row, so the foreground must cut it and leave its top half
        window=WindowPlacement(h_fov_degrees=60.0, elevation_degrees=80.0, azimuth_degrees=180.0,
                               feather_px=0, source_alpha=True),
        credits_png=None, credits=None, retime="dup",
    )
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(result.frames_dir / "dome_000015.png"),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    px = lambda x, y: raw[3 * (y * size + x):3 * (y * size + x) + 3]  # noqa: E731
    green = [(x, y) for y in range(size) for x in range(size)
             if px(x, y)[1] > 150 and px(x, y)[0] < 90]
    red = [(x, y) for y in range(size) for x in range(size)
           if px(x, y)[0] > 150 and px(x, y)[1] < 90]
    assert green, "the cut-out vanished"
    assert all(y <= size // 2 + 1 for _, y in green), "the foreground did not cover the window"
    assert red and all(y > size // 2 - 1 for _, y in red)
    # a cut-out stays a disc: it fills about pi/4 of its bounding box, where the
    # clip's rectangle would fill all of it
    xs, ys = [x for x, _ in green], [y for _, y in green]
    box = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    assert len(green) / box < 0.9


@pytest.mark.integration
@pytest.mark.parametrize(("azimuth", "where"), [(0.0, "bottom"), (90.0, "right"),
                                                (-90.0, "left"), (180.0, "top")])
def test_a_window_lands_at_its_azimuth(tmp_path, azimuth, where):
    """Measured: the audience faces the frame's bottom, their right is the
    frame's right, and behind them is the frame's top."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    size = 128
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x00ff00:s=160x90:r=10:d=3", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(src)], check=True, capture_output=True)
    surround = tmp_path / "mix.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=5.1(side)", "-t", "3", "-c:a", "flac", str(surround)],
                   check=True, capture_output=True)
    result = dome.render(
        source=src, source_start=0.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=0.0, out_dir=tmp_path / "dome",
        spec=DomeSpec(size=size, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        window=WindowPlacement(h_fov_degrees=30.0, elevation_degrees=45.0,
                               azimuth_degrees=azimuth, feather_px=2),
        credits_png=None, credits=None, retime="dup",
    )
    angle = _bright_centroid_angle(result.frames_dir / "dome_000015.png", size, channel=1)
    expected = {"bottom": 0.0, "right": 90.0, "left": -90.0, "top": 180.0}[where]
    assert math.cos(math.radians(angle - expected)) > math.cos(math.radians(5.0))
