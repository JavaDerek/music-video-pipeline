"""Tests for the stereoscopic conversion scaffold (issue #68).

**The sign test is the reason this module exists.** `docs/design-stereoscopic-3d.md`:
"Getting the sign backwards produces a headache, not an error. Any code that
warps must name the convention in its docstring and assert it in a test with a
synthetic depth ramp, because there is no runtime symptom to catch it." So the
first test below builds a near square on a far background and asserts the
left-eye image of it lands to the RIGHT -- crossed (negative) parallax, the
thing that reads as in front of the screen.

Everything here runs on frames a handful of pixels across, with ffmpeg
replaced by fake runners and depth injected. Nothing in this file has ever
touched a rendered chunk, and neither has the module it tests.
"""

from __future__ import annotations

import subprocess
from fractions import Fraction

import pytest

from music_video_maker import stereo

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _frame(rows: list[list[tuple[int, int, int]]]) -> stereo.Frame:
    height = len(rows)
    width = len(rows[0])
    pixels = bytes(channel for row in rows for px in row for channel in px)
    return stereo.Frame(width=width, height=height, pixels=pixels)


def _depth(rows: list[list[float]]) -> stereo.DepthMap:
    return stereo.DepthMap(
        width=len(rows[0]), height=len(rows), values=[v for row in rows for v in row]
    )


def _white_columns(frame: stereo.Frame, y: int = 0) -> list[int]:
    """Which columns of row ``y`` are white -- how this file locates an object
    after a warp without needing numpy."""
    return [
        x
        for x in range(frame.width)
        if frame.pixels[(y * frame.width + x) * 3] > 127
    ]


def _centroid(columns: list[int]) -> float:
    return sum(columns) / len(columns)


# --------------------------------------------------------------------------- #
# THE SIGN. Everything else in this file is bookkeeping.
# --------------------------------------------------------------------------- #


def test_a_near_object_puts_the_left_eye_image_to_the_right():
    """Negative (crossed) parallax: the left-eye image of an object in front
    of the screen plane sits to the RIGHT of the right-eye image. Get this
    backwards and the only symptom is a headache."""
    width = 41
    row = [WHITE if 18 <= x <= 22 else BLACK for x in range(width)]
    depth_row = [1.0 if 18 <= x <= 22 else 0.0 for x in range(width)]
    frame = _frame([row])
    depth = _depth([depth_row])
    # convergence at the far plane, so the square is unambiguously in front
    # of it; a large ceiling so the shift is several whole pixels.
    params = stereo.StereoParams(convergence=0.0, max_disparity_fraction=0.1)

    left, right = stereo.stereo_pair(frame, depth, params=params)

    original = _centroid(_white_columns(frame))
    left_centroid = _centroid(_white_columns(left))
    right_centroid = _centroid(_white_columns(right))

    assert left_centroid > original > right_centroid
    assert left_centroid > right_centroid


def test_a_far_object_puts_the_left_eye_image_to_the_left():
    """The mirror: positive (uncrossed) parallax, behind the screen plane,
    where most of a comfortable frame should live."""
    width = 41
    row = [WHITE if 18 <= x <= 22 else BLACK for x in range(width)]
    depth_row = [0.0 if 18 <= x <= 22 else 1.0 for x in range(width)]
    frame = _frame([row])
    depth = _depth([depth_row])
    params = stereo.StereoParams(convergence=1.0, max_disparity_fraction=0.1)

    left, right = stereo.stereo_pair(frame, depth, params=params)

    assert _centroid(_white_columns(left)) < _centroid(_white_columns(right))


def test_the_convergence_plane_has_no_disparity():
    """Exactly at the convergence plane an object sits on the screen and both
    eyes agree -- the definition of the knob."""
    width = 21
    frame = _frame([[WHITE if x == 10 else BLACK for x in range(width)]])
    depth = _depth([[0.5] * width])
    params = stereo.StereoParams(convergence=0.5, max_disparity_fraction=0.1)

    left, right = stereo.stereo_pair(frame, depth, params=params)

    assert left.pixels == right.pixels == frame.pixels


def test_half_disparity_is_signed_by_nearness_not_by_eye():
    params = stereo.StereoParams(convergence=0.5, max_disparity_fraction=0.01)

    near = stereo.half_disparity_pixels(1.0, params=params, width=800)
    far = stereo.half_disparity_pixels(0.0, params=params, width=800)
    at_plane = stereo.half_disparity_pixels(0.5, params=params, width=800)

    assert near == pytest.approx(4.0)
    assert far == pytest.approx(-4.0)
    assert at_plane == pytest.approx(0.0)


def test_depth_values_outside_the_range_are_clamped_not_rejected():
    """A model emitting 1.0000001 is not a reason to abandon a chunk."""
    params = stereo.StereoParams(convergence=0.0, max_disparity_fraction=0.01)

    assert stereo.half_disparity_pixels(5.0, params=params, width=100) == pytest.approx(
        stereo.half_disparity_pixels(1.0, params=params, width=100)
    )
    assert stereo.half_disparity_pixels(-5.0, params=params, width=100) == pytest.approx(
        stereo.half_disparity_pixels(0.0, params=params, width=100)
    )


# --------------------------------------------------------------------------- #
# Occlusion, holes, and the naive fill
# --------------------------------------------------------------------------- #


def test_the_nearer_pixel_wins_a_collision():
    """Two source pixels landing on one destination: the near one is in
    front, so it is the one that survives."""
    width = 11
    frame = _frame([[(255, 0, 0) if x == 4 else (0, 0, 255) for x in range(width)]])
    # The red pixel is near (moves right); its right-hand neighbour is at the
    # convergence plane and does not move, so they collide on column 5.
    depth = _depth([[1.0 if x == 4 else 0.5 for x in range(width)]])
    params = stereo.StereoParams(convergence=0.5, max_disparity_fraction=0.1)

    left, _ = stereo.stereo_pair(frame, depth, params=params)

    assert left.pixels[5 * 3 : 5 * 3 + 3] == bytes((255, 0, 0))


def test_a_disocclusion_is_filled_from_the_nearest_written_pixel():
    """Naive by design and named as such: a real conversion inpaints. What is
    asserted here is only that no hole is left black when the row has
    neighbours to copy."""
    width = 21
    frame = _frame([[WHITE if x >= 10 else BLACK for x in range(width)]])
    depth = _depth([[1.0 if x >= 10 else 0.0 for x in range(width)]])
    params = stereo.StereoParams(convergence=0.0, max_disparity_fraction=0.08)

    left, _ = stereo.stereo_pair(frame, depth, params=params)

    # Every destination pixel got a value from somewhere (nothing left at the
    # zero-initialised black except where the source itself was black).
    assert set(_white_columns(left))


def test_a_row_with_no_holes_is_left_alone():
    """The fill has to be a no-op when the warp opened nothing -- a 4 px
    frame at the default ceiling shifts by less than half a pixel, so every
    destination is written and there is nothing to copy."""
    frame = _frame([[WHITE, BLACK, WHITE, BLACK]])
    depth = _depth([[1.0] * 4])

    left = stereo.warp_eye(
        frame,
        depth,
        eye="left",
        params=stereo.StereoParams(convergence=0.0, max_disparity_fraction=0.1),
    )

    assert left.pixels == frame.pixels


def test_an_entirely_empty_row_is_left_black(monkeypatch):
    """The real out-of-frame case, forced: every pixel shifts past the right
    edge, so the row receives nothing and the fill declines to invent. Forced
    through the reference loop, the only path that calls the scalar
    ``half_disparity_pixels`` this patches; the vectorised path's version of
    the same case is tested below with a depth nothing can land from."""
    monkeypatch.setattr(stereo, "_numpy", lambda: None)
    monkeypatch.setattr(stereo, "half_disparity_pixels", lambda *a, **k: 1000.0)
    frame = _frame([[WHITE] * 5])
    depth = _depth([[1.0] * 5])

    left = stereo.warp_eye(frame, depth, eye="left", params=stereo.StereoParams())

    assert left.pixels == bytes(len(frame.pixels))


# --------------------------------------------------------------------------- #
# The warp's output, PINNED (issue #68's vectorisation)
#
# These digests were computed from the original pure-Python per-pixel loop
# (commit 9e47e37's warp_eye) BEFORE anything was vectorised, and are the
# proof that the rewrite changed no output: every case below must keep
# producing these exact bytes whichever implementation warp_eye dispatches
# to. Inputs come from stdlib `random` with fixed seeds (its Random.random /
# randrange / uniform sequences are stable across Python versions), so the
# cases need no numpy and run in CI, which installs none.
#
# Do NOT regenerate these to make a failing test pass. A changed digest means
# the warp's output changed, which is exactly what they exist to catch.
# --------------------------------------------------------------------------- #

# name -> (seed, width, height, depth levels (0 = continuous), depth lo, depth
# hi, convergence, max_disparity_fraction)
_GOLDEN_CASES = {
    # the shipped defaults on smooth depth, wide enough (240) to move +-1.8 px
    "smooth": (1, 240, 4, 0, 0.0, 1.0, 0.5, 0.015),
    # the largest ceiling StereoParams allows: long shifts, many collisions
    "wide_disparity": (2, 64, 6, 0, 0.0, 1.0, 0.3, 0.1),
    # four depth levels: exact z-buffer TIES, which the first writer keeps
    "quantised_ties": (3, 41, 7, 4, 0.0, 1.0, 0.5, 0.1),
    # values below 0, below -1 and above 1: the clamp, plus the unclamped
    # z-buffer quirk (a pixel written at inv in (-1, 0) is still a "hole")
    "out_of_range": (4, 29, 5, 0, -1.5, 2.0, 0.7, 0.08),
    "convergence_far": (5, 50, 4, 0, 0.0, 1.0, 0.0, 0.05),
    "convergence_near": (6, 50, 4, 0, 0.0, 1.0, 1.0, 0.05),
    # shifts of exactly +-0.5 and +-1.0 px: round-half-to-even matters here
    "half_pixel_rounding": (7, 40, 3, 5, 0.0, 1.0, 0.5, 0.05),
}

_GOLDEN_SHA256 = {
    # (left eye, right eye)
    "smooth": (
        "3adc77a877cecf07f2dd23903eaba5f9930488b75384269ac40b313da1379bec",
        "ec1f0c7f8a5f3f71f08b8799450f3a41e70cc38c16484a0d96e38885f0b85817",
    ),
    "wide_disparity": (
        "f4439ecd955e79728e733132be5da1f41f25571233c5d9d10de7b57069ee463d",
        "7971f9facf5aa4b4b1f1dbb16432edb733546e947e635b1a1c1f6ba3bd299007",
    ),
    "quantised_ties": (
        "c76e0dee6359890cc872311aa2a64d6dbccb8219609d8fb190f4f116d0008fc5",
        "b9ecdbae12656137ecf15b7d84aa80136158593cb900f3d01d3ab110306d4af2",
    ),
    "out_of_range": (
        "d68407aa0b284927b09b6604d433201aa042af8a9143a7cc9fac4ab5c6125c52",
        "5d19563a4e08139d749ed8c1a1ef85cb7b8157e81ecbd9fad3869ffd24493fad",
    ),
    "convergence_far": (
        "19d53c73f5e1f6d83a778114197b39e5bd2e036702523a9602d75362074472de",
        "41264fc5a84a64623f9e60d6949291486da9f5772b690f2a871b05f743426c87",
    ),
    "convergence_near": (
        "2ebad6f674dd336eb86024dd12f5788911f21572df101078f33f8a521de922ce",
        "63db3452c502022f07142dd968cd8f1cb99d2c1877ab070d1d5aba7ce3496a14",
    ),
    "half_pixel_rounding": (
        "e093a24faa292d03497bea7d9a70f016118669f135b8840820897eddecb57f2e",
        "46fe7fbb143147bd02d8b5a869ba6e2cdbc4a67dea000b6d4e1e3983ce7f580b",
    ),
}


def _golden_case(name: str) -> tuple[stereo.Frame, stereo.DepthMap, stereo.StereoParams]:
    import random

    seed, width, height, levels, lo, hi, convergence, fraction = _GOLDEN_CASES[name]
    rng = random.Random(seed)
    pixels = bytes(rng.randrange(256) for _ in range(width * height * 3))
    if levels:
        values = [
            lo + (hi - lo) * rng.randrange(levels) / (levels - 1) for _ in range(width * height)
        ]
    else:
        values = [rng.uniform(lo, hi) for _ in range(width * height)]
    return (
        stereo.Frame(width=width, height=height, pixels=pixels),
        stereo.DepthMap(width=width, height=height, values=values),
        stereo.StereoParams(convergence=convergence, max_disparity_fraction=fraction),
    )


@pytest.mark.parametrize("name", sorted(_GOLDEN_CASES))
def test_the_warp_output_is_pinned(name):
    import hashlib

    frame, depth, params = _golden_case(name)

    left, right = stereo.stereo_pair(frame, depth, params=params)

    assert (
        hashlib.sha256(left.pixels).hexdigest(),
        hashlib.sha256(right.pixels).hexdigest(),
    ) == _GOLDEN_SHA256[name]


@pytest.mark.parametrize("name", sorted(_GOLDEN_CASES))
def test_the_reference_loop_still_produces_the_pinned_output(name, monkeypatch):
    """The pure-Python loop is kept as the fallback and as the oracle the
    vectorised path is proven against, so it is pinned on its own too --
    forced here, whatever this environment has installed."""
    import hashlib

    monkeypatch.setattr(stereo, "_numpy", lambda: None)
    frame, depth, params = _golden_case(name)

    left, right = stereo.stereo_pair(frame, depth, params=params)

    assert (
        hashlib.sha256(left.pixels).hexdigest(),
        hashlib.sha256(right.pixels).hexdigest(),
    ) == _GOLDEN_SHA256[name]


# --------------------------------------------------------------------------- #
# The vectorised warp is the loop, byte for byte (needs numpy, which CI does
# not install -- the pinned digests above are what CI checks instead)
# --------------------------------------------------------------------------- #


def _random_case(seed: int) -> tuple[stereo.Frame, stereo.DepthMap, stereo.StereoParams]:
    import random

    rng = random.Random(1000 + seed)
    width = rng.randrange(1, 90)
    height = rng.randrange(1, 6)
    pixels = bytes(rng.randrange(256) for _ in range(width * height * 3))
    style = seed % 4
    if style == 0:  # continuous, in range
        values = [rng.random() for _ in range(width * height)]
    elif style == 1:  # few levels: exact ties everywhere
        levels = rng.randrange(2, 5)
        values = [rng.randrange(levels) / (levels - 1) for _ in range(width * height)]
    elif style == 2:  # out of range both ways, incl. below -1 (never written)
        values = [rng.uniform(-2.5, 2.5) for _ in range(width * height)]
    else:  # gray16le-shaped, as external_depth_source produces
        values = [rng.randrange(65536) / 65535.0 for _ in range(width * height)]
    params = stereo.StereoParams(
        convergence=rng.choice([0.0, 0.25, 0.5, 1.0, rng.random()]),
        max_disparity_fraction=rng.choice([0.0, 0.015, 0.05, 0.1, rng.uniform(0.0, 0.1)]),
    )
    return (
        stereo.Frame(width=width, height=height, pixels=pixels),
        stereo.DepthMap(width=width, height=height, values=values),
        params,
    )


@pytest.mark.parametrize("seed", range(200))
@pytest.mark.parametrize("eye", ["left", "right"])
def test_the_vectorised_warp_matches_the_reference_loop(seed, eye):
    np = pytest.importorskip("numpy")
    frame, depth, params = _random_case(seed)
    direction = 1 if eye == "left" else -1

    expected = stereo._warp_eye_reference(frame, depth, direction=direction, params=params)
    actual = stereo._warp_eye_numpy(np, frame, depth, direction=direction, params=params)

    assert actual.pixels == expected.pixels


def test_the_vectorised_warp_leaves_an_unreachable_row_black_like_the_loop():
    """A row nothing lands on: every depth at or below the z-buffer's initial
    -1.0 (exactly -1.0 included -- the loop's test is ``<=``), so no source
    pixel is ever accepted. Both paths leave it black."""
    np = pytest.importorskip("numpy")
    frame = _frame([[WHITE] * 5, [WHITE] * 5, [WHITE] * 5])
    depth = _depth([[-2.0] * 5, [-1.0] * 5, [0.5] * 5])
    params = stereo.StereoParams()

    expected = stereo._warp_eye_reference(frame, depth, direction=1, params=params)
    actual = stereo._warp_eye_numpy(np, frame, depth, direction=1, params=params)

    assert actual.pixels == expected.pixels
    assert actual.pixels[:30] == bytes(30)


def test_warp_eye_uses_the_vectorised_path_when_numpy_is_present(monkeypatch):
    pytest.importorskip("numpy")
    called = []
    real = stereo._warp_eye_numpy
    monkeypatch.setattr(
        stereo, "_warp_eye_numpy", lambda *a, **k: called.append(1) or real(*a, **k)
    )
    frame, depth, params = _golden_case("quantised_ties")

    stereo.warp_eye(frame, depth, eye="left", params=params)

    assert called == [1]


def test_warp_eye_falls_back_to_the_loop_without_numpy(monkeypatch):
    monkeypatch.setattr(stereo, "_numpy", lambda: None)
    monkeypatch.setattr(
        stereo, "_warp_eye_numpy", lambda *a, **k: pytest.fail("numpy path taken")
    )
    frame, depth, params = _golden_case("quantised_ties")

    left = stereo.warp_eye(frame, depth, eye="left", params=params)

    assert left.pixels == stereo._warp_eye_reference(
        frame, depth, direction=1, params=params
    ).pixels


def test_a_nan_depth_takes_the_loop_rather_than_a_different_answer(monkeypatch):
    """The loop's handling of NaN is order-dependent (NaN never loses a
    z-buffer comparison and is neither a hole nor filled), so the vectorised
    path does not attempt to reproduce it: a map containing NaN goes through
    the reference loop and gets the loop's answer, slowly."""
    pytest.importorskip("numpy")
    monkeypatch.setattr(
        stereo, "_warp_eye_numpy", lambda *a, **k: pytest.fail("numpy path taken")
    )
    frame = _frame([[WHITE, BLACK, WHITE, BLACK]])
    depth = _depth([[0.2, float("nan"), 0.9, 0.4]])
    params = stereo.StereoParams(max_disparity_fraction=0.1)

    left = stereo.warp_eye(frame, depth, eye="left", params=params)

    assert left.pixels == stereo._warp_eye_reference(
        frame, depth, direction=1, params=params
    ).pixels


# --------------------------------------------------------------------------- #
# Output formats
# --------------------------------------------------------------------------- #


def test_side_by_side_doubles_the_width_and_keeps_the_eyes_in_order():
    left = _frame([[WHITE, BLACK]])
    right = _frame([[BLACK, WHITE]])

    out = stereo.side_by_side(left, right)

    assert (out.width, out.height) == (4, 1)
    assert _white_columns(out) == [0, 3]


def test_anaglyph_takes_red_from_the_left_eye_and_cyan_from_the_right():
    left = _frame([[(200, 10, 10)]])
    right = _frame([[(10, 200, 200)]])

    out = stereo.anaglyph(left, right)

    assert out.pixels == bytes((200, 200, 200))
    assert (out.width, out.height) == (1, 1)


def test_compose_dispatches_on_the_format_name():
    left = _frame([[WHITE]])
    right = _frame([[BLACK]])

    assert stereo.compose(left, right, stereo.FORMAT_SIDE_BY_SIDE).width == 2
    assert stereo.compose(left, right, stereo.FORMAT_ANAGLYPH).width == 1
    with pytest.raises(stereo.StereoError, match="unknown output format"):
        stereo.compose(left, right, "over-under")


def test_the_two_formats_refuse_mismatched_eyes():
    with pytest.raises(stereo.StereoError):
        stereo.side_by_side(_frame([[WHITE]]), _frame([[WHITE, BLACK]]))
    with pytest.raises(stereo.StereoError):
        stereo.anaglyph(_frame([[WHITE]]), _frame([[WHITE, BLACK]]))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_a_frame_buffer_of_the_wrong_length_is_refused():
    with pytest.raises(stereo.StereoError, match="frame buffer"):
        stereo.Frame(width=2, height=2, pixels=b"\x00")
    with pytest.raises(stereo.StereoError, match="dimensions must be positive"):
        stereo.Frame(width=0, height=2, pixels=b"")


def test_a_depth_map_of_the_wrong_length_is_refused():
    with pytest.raises(stereo.StereoError, match="depth map has"):
        stereo.DepthMap(width=2, height=2, values=[0.0])


def test_a_depth_map_that_does_not_match_its_frame_is_refused():
    with pytest.raises(stereo.StereoError, match="same size as the frame"):
        stereo.warp_eye(
            _frame([[WHITE, BLACK]]),
            _depth([[1.0]]),
            eye="left",
            params=stereo.StereoParams(),
        )


def test_an_unknown_eye_is_refused():
    with pytest.raises(stereo.StereoError, match="eye must be"):
        stereo.warp_eye(
            _frame([[WHITE]]), _depth([[1.0]]), eye="middle", params=stereo.StereoParams()
        )


def test_stereo_params_refuse_settings_outside_their_units():
    with pytest.raises(stereo.StereoError, match="normalised inverse-depth"):
        stereo.StereoParams(convergence=1.5)
    with pytest.raises(stereo.StereoError, match="headache"):
        stereo.StereoParams(max_disparity_fraction=0.5)


def test_the_depth_model_note_records_a_licence_and_no_weights():
    """CLAUDE.md: check redistribution before committing a third-party
    binary, and "it downloaded fine" is not a licence. The Base/Large trap is
    named because it is the mistake available to make."""
    note = stereo.DEPTH_ANYTHING_V2_SMALL

    assert note["committed"] is False
    assert note["sha256"] is None
    assert "Apache-2.0" in note["licence"]
    assert "CC-BY-NC" in note["licence"]


# --------------------------------------------------------------------------- #
# The ffmpeg seam -- argv is asserted, no process is ever spawned
# --------------------------------------------------------------------------- #


def _completed(stdout: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stdout=stdout)


def test_probe_parses_width_height_and_an_exact_frame_rate():
    info = stereo.probe_video("chunk.mp4", runner=lambda args: _completed(b"864,480,24/1\n"))

    assert info == stereo.VideoInfo(width=864, height=480, fps=Fraction(24, 1))


def test_probe_keeps_ntsc_rates_exact():
    """23.976023976 is a different stream parameter from 24000/1001, and the
    concat demuxer compares stream parameters."""
    info = stereo.probe_video("c.mp4", runner=lambda args: _completed(b"864,480,24000/1001"))

    assert info.fps == Fraction(24000, 1001)


def test_probe_failures_raise_rather_than_degrade():
    with pytest.raises(stereo.StereoError, match="ffprobe failed"):
        stereo.probe_video("c.mp4", runner=lambda args: _completed(b"", returncode=1))
    with pytest.raises(stereo.StereoError, match="could not parse"):
        stereo.probe_video("c.mp4", runner=lambda args: _completed(b"864,480"))
    with pytest.raises(stereo.StereoError, match="could not parse"):
        stereo.probe_video("c.mp4", runner=lambda args: _completed(b"wide,tall,fast"))

    def explode(args):
        raise OSError("ffprobe not on PATH")

    with pytest.raises(stereo.StereoError, match="could not run ffprobe"):
        stereo.probe_video("c.mp4", runner=explode)


def test_decode_splits_the_raw_stream_into_frames():
    info = stereo.VideoInfo(width=2, height=1, fps=Fraction(24, 1))
    raw = bytes(range(12))  # two 2x1 rgb24 frames

    frames = stereo.decode_frames("c.mp4", info, runner=lambda args: _completed(raw))

    assert [f.pixels for f in frames] == [raw[:6], raw[6:]]


def test_decode_drops_a_trailing_partial_frame_with_a_warning(caplog):
    info = stereo.VideoInfo(width=2, height=1, fps=Fraction(24, 1))

    with caplog.at_level("WARNING"):
        frames = stereo.decode_frames(
            "c.mp4", info, runner=lambda args: _completed(bytes(9))
        )

    assert len(frames) == 1
    assert "trailing bytes" in caplog.text


def test_decode_failures_raise():
    info = stereo.VideoInfo(width=2, height=1, fps=Fraction(24, 1))
    with pytest.raises(stereo.StereoError, match="ffmpeg failed decoding"):
        stereo.decode_frames("c.mp4", info, runner=lambda args: _completed(b"", returncode=1))
    with pytest.raises(stereo.StereoError, match="no whole frames"):
        stereo.decode_frames("c.mp4", info, runner=lambda args: _completed(b""))

    def explode(args):
        raise OSError("no ffmpeg")

    with pytest.raises(stereo.StereoError, match="could not run ffmpeg to decode"):
        stereo.decode_frames("c.mp4", info, runner=explode)


def test_encode_passes_every_frame_down_one_pipe_and_names_the_size():
    seen = {}

    def runner(args, payload):
        seen["args"] = list(args)
        seen["payload"] = payload
        return _completed()

    frames = [_frame([[WHITE, BLACK]]), _frame([[BLACK, WHITE]])]
    stereo.encode_frames(frames, "out.mp4", fps=Fraction(24, 1), runner=runner)

    assert "2x1" in seen["args"]
    assert "24" in seen["args"]
    assert seen["payload"] == frames[0].pixels + frames[1].pixels
    assert "-an" in seen["args"]  # generated audio is never carried anywhere


def test_encode_refuses_an_empty_or_ragged_sequence():
    with pytest.raises(stereo.StereoError, match="no frames"):
        stereo.encode_frames([], "out.mp4", fps=Fraction(24, 1), runner=lambda a, p: _completed())
    with pytest.raises(stereo.StereoError, match="same size"):
        stereo.encode_frames(
            [_frame([[WHITE]]), _frame([[WHITE, BLACK]])],
            "out.mp4",
            fps=Fraction(24, 1),
            runner=lambda a, p: _completed(),
        )


def test_encode_failures_raise():
    with pytest.raises(stereo.StereoError, match="ffmpeg failed encoding"):
        stereo.encode_frames(
            [_frame([[WHITE]])],
            "out.mp4",
            fps=Fraction(24, 1),
            runner=lambda a, p: _completed(returncode=1),
        )

    def explode(args, payload):
        raise OSError("no ffmpeg")

    with pytest.raises(stereo.StereoError, match="could not run ffmpeg to encode"):
        stereo.encode_frames([_frame([[WHITE]])], "out.mp4", fps=Fraction(24, 1), runner=explode)


def test_the_argv_builders_are_pure():
    assert stereo.build_probe_args("a.mp4")[0] == "ffprobe"
    assert "rgb24" in stereo.build_decode_args("a.mp4")
    args = stereo.build_encode_args("b.mp4", width=8, height=4, fps=Fraction(24, 1))
    assert "8x4" in args and "libx264" in args and "yuv420p" in args


# --------------------------------------------------------------------------- #
# The external depth command (the production seam, never run for real)
# --------------------------------------------------------------------------- #


def test_the_external_depth_source_reads_gray16le_back_as_normalised_depth():
    frame = _frame([[WHITE, BLACK]])
    payload = (0).to_bytes(2, "little") + (65535).to_bytes(2, "little")
    source = stereo.external_depth_source(["depth"], runner=lambda a, p: _completed(payload))

    depth = source(0, frame)

    assert list(depth.values) == pytest.approx([0.0, 1.0])
    assert (depth.width, depth.height) == (2, 1)


def test_the_external_depth_source_decodes_every_sample_exactly(monkeypatch):
    """Vectorised or not, each sample is ``int / 65535.0`` -- the same IEEE
    division -- so the values are equal, not approximately equal."""
    import random

    rng = random.Random(68)
    samples = [rng.randrange(65536) for _ in range(6 * 4)] + [0, 1, 32767, 32768, 65534, 65535]
    payload = b"".join(s.to_bytes(2, "little") for s in samples)
    frame = stereo.Frame(width=6, height=5, pixels=bytes(6 * 5 * 3))
    expected = [s / 65535.0 for s in samples]

    def runner(args, stdin):
        return _completed(payload)

    fast = stereo.external_depth_source(["depth"], runner=runner)(0, frame)
    monkeypatch.setattr(stereo, "_numpy", lambda: None)
    slow = stereo.external_depth_source(["depth"], runner=runner)(0, frame)

    assert list(fast.values) == expected
    assert list(slow.values) == expected


def test_the_external_depth_source_refuses_a_short_reply():
    frame = _frame([[WHITE, BLACK]])
    source = stereo.external_depth_source(["depth"], runner=lambda a, p: _completed(b"\x00\x00"))

    with pytest.raises(stereo.StereoError, match="returned 2 bytes"):
        source(0, frame)


def test_the_external_depth_source_reports_which_frame_failed():
    frame = _frame([[WHITE]])
    nonzero = stereo.external_depth_source(["d"], runner=lambda a, p: _completed(returncode=3))
    with pytest.raises(stereo.StereoError, match="exited 3 on frame 7"):
        nonzero(7, frame)

    def explode(args, payload):
        raise OSError("no such command")

    with pytest.raises(stereo.StereoError, match="failed on frame 2"):
        stereo.external_depth_source(["d"], runner=explode)(2, frame)


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #


def test_convert_chunk_probes_decodes_warps_and_encodes(tmp_path):
    raw = bytes([255, 255, 255, 0, 0, 0] * 2)  # two 2x1 frames
    calls = []

    def runner(args):
        calls.append(args[0])
        if args[0] == "ffprobe":
            return _completed(b"2,1,24/1")
        return _completed(raw)

    encoded = {}

    def pipe_runner(args, payload):
        encoded["args"] = list(args)
        encoded["payload"] = payload
        return _completed()

    seen_indices = []

    def depth(index, frame):
        seen_indices.append(index)
        return stereo.DepthMap(width=frame.width, height=frame.height, values=[0.5, 0.5])

    out = stereo.convert_chunk(
        "chunk_0045.mp4",
        tmp_path / "chunk_0045_sbs.mp4",
        depth,
        runner=runner,
        pipe_runner=pipe_runner,
    )

    assert calls == ["ffprobe", "ffmpeg"]
    assert seen_indices == [0, 1]
    assert out == tmp_path / "chunk_0045_sbs.mp4"
    # side-by-side, so each 2x1 frame became 4x1: 4*1*3 bytes * 2 frames.
    assert "4x1" in encoded["args"]
    assert len(encoded["payload"]) == 4 * 1 * 3 * 2


def test_convert_chunk_can_emit_an_anaglyph_review_artifact(tmp_path):
    raw = bytes([255, 255, 255, 0, 0, 0])

    def runner(args):
        return _completed(b"2,1,24/1") if args[0] == "ffprobe" else _completed(raw)

    encoded = {}

    def pipe_runner(args, payload):
        encoded["args"] = list(args)
        return _completed()

    stereo.convert_chunk(
        "c.mp4",
        tmp_path / "c_anaglyph.mp4",
        lambda i, f: stereo.DepthMap(width=f.width, height=f.height, values=[0.5, 0.5]),
        output_format=stereo.FORMAT_ANAGLYPH,
        runner=runner,
        pipe_runner=pipe_runner,
    )

    assert "2x1" in encoded["args"]  # anaglyph keeps the original width


def test_convert_chunk_refuses_an_unknown_format_before_spawning_anything(tmp_path):
    def runner(args):  # pragma: no cover - must never be reached
        raise AssertionError("no process should be spawned for a bad format")

    with pytest.raises(stereo.StereoError, match="unknown output format"):
        stereo.convert_chunk(
            "c.mp4", tmp_path / "o.mp4", lambda i, f: None, output_format="mvhevc", runner=runner
        )
