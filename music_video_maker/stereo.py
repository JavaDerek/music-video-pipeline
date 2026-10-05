"""Stereoscopic conversion scaffold -- issue #68's conversion half.

**ALMOST NOTHING HERE HAS BEEN RUN ON REAL FOOTAGE.** Every function in this
module is exercised by unit tests on synthetic frames a few pixels across,
with the depth source injected and ffmpeg replaced by a fake runner. The one
exception, 2026-10-05 (#68's vectorisation): :func:`stereo_pair`,
:func:`anaglyph` and :func:`external_depth_source` (with ``cat`` standing in
for the depth command) ran on 9 real 864x480 frames of the v14 render, with
Depth Anything V2 Small depth computed on CPU *outside* this module -- the
vectorised warp matched the original loop byte for byte on all of them, at
~35 ms per stereo pair against ~450 ms
(``~/mvm-runs/deathless/measurements/stereo68_2026-10-05/FINDINGS.md``).
:func:`convert_chunk`, :func:`decode_frames` and :func:`encode_frames` have
still never touched a real chunk, and nobody has judged an anaglyph *as
stereo*. Read every claim below as "this is what the code does", never as
"this is what it produces".

Why it exists anyway
--------------------
The sign convention is a silent bug. ``docs/design-stereoscopic-3d.md``:
"Getting the sign backwards produces a headache, not an error. Any code that
warps must name the convention in its docstring and assert it in a test with a
synthetic depth ramp, because there is no runtime symptom to catch it." That
assertion is cheap, it needs no GPU, no weights and no frames, and it is worth
having settled *before* someone with a depth model in hand starts writing a
warp at midnight. So: the convention, the arithmetic, the two output formats
and the ffmpeg seam, with the expensive half (the depth model) left as an
injected callable.

The convention, stated once
---------------------------
**Negative (crossed) parallax is what flies out of the screen: the left-eye
image sits to the RIGHT of the right-eye image.** Positive (uncrossed)
parallax puts an object behind the screen plane, which is where most of a
comfortable frame should live. The knob between them is the **convergence
plane**: nearer than it pops out, further recedes, exactly at it there is no
disparity at all.

So, for an object NEARER than the convergence plane, this module moves its
pixels **right** when synthesising the left eye and **left** when synthesising
the right eye. :func:`stereo_pair`'s test asserts exactly that on a synthetic
near square, and the assertion is the point of the test.

Depth units, which are the other half of the sign
-------------------------------------------------
:class:`DepthMap` carries **normalised inverse depth in [0, 1], where 1.0 is
NEAREST to the camera** -- the convention MiDaS and Depth Anything both emit
(they predict disparity-like inverse depth, not metres), so a production depth
source hands its output straight over with a min-max normalisation and nothing
else. :attr:`StereoParams.convergence` is a value in those same units, so
``convergence = 0.5`` puts the screen plane at the middle of the frame's own
depth range, and ``convergence = 1.0`` puts every pixel behind the screen.

A metric depth model (one that emits metres) must be converted by its caller:
``inv = 1/metres``, min-max normalised over the clip. Doing it per frame
re-ranges the depth every time the content changes, which is the mechanism
suspected behind chunk 46's boil in ``docs/pop-beat-corpus.md`` -- normalise
over the whole chunk, not per frame.

What is deliberately naive, and will show
-----------------------------------------
* **Hole filling is nearest-neighbour along the row.** A forward warp opens a
  disocclusion behind every depth edge, and this fills it by copying the
  nearest written pixel in the same row. That is the "edge-stretch, for this
  test only" the design doc specifies for the *experiment*, not what it
  specifies for the feature: "the showcase shot is the worst case ... budget
  for real inpainting, not edge-stretching, and evaluate on that shot first".
  A pop beat is precisely a large near object moving fast, so the feature's
  own money shot is where this will look worst.
* **Depth is per frame, so it will boil.** Temporal consistency is the whole
  problem (#68) and nothing here addresses it. The 2026-09-20 measurements in
  ``docs/pop-beat-corpus.md`` are the only numbers that exist: a *held* object
  at the lens boiled 0.0035, an *arriving* one 0.0271.
* **The warp is vectorised only when numpy is importable, and the chunk is
  still buffered whole.** numpy is optional: the ``stereo`` extra declares it
  (``pip install -e ".[stereo]"``), and it also arrives with the ``faces``
  extra (OpenCV requires it) and with any environment that can run a depth
  model. CI runs the core suite without it, then installs ``[stereo]`` and
  runs ``tests/test_stereo.py`` again so the equivalence cases execute. With it,
  :func:`warp_eye` runs :func:`_warp_eye_numpy`; without it, the original
  per-pixel loop, :func:`_warp_eye_reference`, which stays as the
  specification and as the oracle the fast path is tested against --
  **byte-identical output either way**, pinned by digest in
  ``tests/test_stereo.py`` from the loop as it stood before the rewrite
  (#68). :func:`decode_frames` still holds a whole chunk's raw RGB in a
  ``bytes`` (~1.2 MB a frame, ~240 MB for a 192-frame chunk at 864x480);
  **a streaming ``Popen`` pipe is still owed** before this runs over a song.

Where it sits in the pipeline
-----------------------------
**Strictly outside the render path.** It reads finished chunk mp4s and writes
new files somewhere else. It imports nothing from ``resilience``,
``execution`` or ``contracts``, touches no ``ChunkFingerprint``, writes no
``run_state.json``, and cannot cause a re-render: a stereo pass that could
invalidate a cached chunk would put hours of GPU custody behind a post-process.

Per chunk, before concat, is where the design doc puts it, so Stage 5's
concat demuxer keeps ``-c:v copy``. That carries a constraint this module
cannot enforce and states instead: **every chunk must be converted or none.**
All 80 chunks of the real render share one concat signature (h264 / High /
level 30 / 864x480 / yuv420p / 24 fps), and :data:`FORMAT_SIDE_BY_SIDE`
doubles the width to 1728. A mixed-width concat plays wrong rather than
failing.

The depth model, and the licence rule
-------------------------------------
:data:`DEPTH_ANYTHING_V2_SMALL` records the production candidate's source and
licence the way ``faces.py`` records YuNet's -- **without the weights**, which
are not committed and must not be. "It downloaded fine" is not a licence
(CLAUDE.md), and note the trap the design doc already found: Depth Anything V2
**Small** is Apache-2.0 while the Base and Large variants are CC-BY-NC.

:func:`external_depth_source` is the production seam: it shells out per frame
to a command that prints raw 16-bit grayscale depth to stdout. The exact
invocation is in that function's docstring. It has never been run.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Formats and defaults
# --------------------------------------------------------------------------- #

FORMAT_SIDE_BY_SIDE = "sbs"
"""Full side-by-side: left eye in the left half, right eye in the right half,
output width doubled. The delivery format; see the module docstring's warning
that a mixed-width concat plays wrong rather than failing."""

FORMAT_ANAGLYPH = "anaglyph"
"""Left eye's red channel, right eye's green and blue. A **tool, not a
product** (design doc's own answer to its open question): the only format that
can be judged on the machine that rendered it, with no player support and no
hardware beyond a cheap pair of glasses. Never ship it as the deliverable."""

OUTPUT_FORMATS = (FORMAT_SIDE_BY_SIDE, FORMAT_ANAGLYPH)

DEFAULT_MAX_DISPARITY_FRACTION = 0.015
"""Maximum half-disparity, as a fraction of frame width, at the extreme of the
depth range -- about 13 px at 864 wide. Straight from the design doc's own
procedure ("clamped to a maximum of ~1.5% of frame width ... more is a
headache, not more 3D"), which is a comfort convention from the stereography
literature rather than anything measured here. It has never been looked at on
this content; the first person to view an anaglyph should expect to move it."""

DEFAULT_CONVERGENCE = 0.5
"""The screen plane, in normalised inverse-depth units (1.0 = nearest). 0.5
puts it at the middle of the clip's own depth range, so roughly half the
frame recedes and half comes forward. Unmeasured, like the disparity ceiling
-- and unlike it, not even a convention: a shot whose depth histogram is
bimodal (a face against a far valley, which is most of this project's
content) wants it at the far mode, not the middle."""

DEPTH_ANYTHING_V2_SMALL = {
    "name": "Depth Anything V2 Small",
    "source": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small",
    "licence": (
        "Apache-2.0 for the SMALL variant only -- the Base and Large variants are "
        "CC-BY-NC and are NOT interchangeable with it. VERIFY against the model "
        "card before any weights are downloaded; this string is a note taken from "
        "issue #68, not a licence check performed by this project."
    ),
    "sha256": None,
    "committed": False,
}
"""Provenance note for the production depth model, in the shape ``faces.py``
records YuNet's -- with ``sha256`` deliberately ``None`` and ``committed``
``False``, because **the weights are not in this repo and must not be**
(CLAUDE.md: check redistribution before committing a third-party binary).
Whoever installs them on doris fills in the sha256 *there*, beside the file.
Nothing in this module loads them."""


class StereoError(RuntimeError):
    """Raised when a conversion cannot proceed: an unknown output format, a
    depth map whose dimensions disagree with its frame, a frame buffer of the
    wrong length, or ffmpeg failing to probe/decode/encode.

    Unlike ``luminance``/``scenecuts``, which degrade to "cannot prove this
    chunk is fine" because they run inside an assembly that must not be
    aborted, this is a **separate offline pass** whose only output is the
    converted file. Half a stereo conversion is not a usable artefact, so
    every failure here raises and the caller decides -- per chunk, the way the
    design doc asks ("a failed conversion costs one chunk rather than the
    video")."""


# --------------------------------------------------------------------------- #
# Pixel containers (plain bytes and sequences: numpy is optional, see the module docstring)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Frame:
    """One decoded frame, packed RGB24 (``rgb24`` is exactly what
    ``ffmpeg -f rawvideo -pix_fmt rgb24`` writes, so no conversion happens in
    Python)."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise StereoError(f"frame dimensions must be positive, got {self.width}x{self.height}")
        expected = self.width * self.height * 3
        if len(self.pixels) != expected:
            raise StereoError(
                f"frame buffer is {len(self.pixels)} bytes; a {self.width}x{self.height} "
                f"rgb24 frame is {expected}"
            )


@dataclass(frozen=True)
class DepthMap:
    """Per-pixel **normalised inverse depth in [0, 1], 1.0 nearest** -- see the
    module docstring's "Depth units" section, which is half of the sign
    convention. Values outside the range are clamped where they are used
    rather than rejected here: a model that emits a stray 1.0000001 is not a
    reason to abandon a chunk."""

    width: int
    height: int
    values: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.values) != self.width * self.height:
            raise StereoError(
                f"depth map has {len(self.values)} values; a {self.width}x{self.height} "
                f"map needs {self.width * self.height}"
            )


DepthSource = Callable[[int, Frame], DepthMap]
"""``(frame_index, frame) -> DepthMap``. Injected, always: the depth model is
the expensive, licence-encumbered, GPU-shaped half of this feature and nothing
in the test suite may need it. :func:`external_depth_source` builds the
production one."""


@dataclass(frozen=True)
class StereoParams:
    """The whole-video comfort settings.

    These are exactly the fields issue #68 suggests belong in a locked house
    style (#55) -- "comfort settings are a signature and a safety limit". They
    are **not** in :data:`music_video_maker.profiles.LOOK_FIELDS` and must not
    be added there until they have fingerprint evidence, which they cannot
    have while this pass runs *after* the render and changes no chunk H3
    produced. That is not an oversight; it is #55's schema constraint
    answering the question correctly.
    """

    convergence: float = DEFAULT_CONVERGENCE
    max_disparity_fraction: float = DEFAULT_MAX_DISPARITY_FRACTION

    def __post_init__(self) -> None:
        if not 0.0 <= self.convergence <= 1.0:
            raise StereoError(
                f"convergence is in normalised inverse-depth units [0, 1] (1.0 = "
                f"nearest), got {self.convergence!r}"
            )
        if not 0.0 <= self.max_disparity_fraction <= 0.1:
            raise StereoError(
                f"max_disparity_fraction is a fraction of frame width and anything "
                f"above a few percent is a headache rather than more depth; got "
                f"{self.max_disparity_fraction!r}"
            )


# --------------------------------------------------------------------------- #
# The warp
# --------------------------------------------------------------------------- #


def half_disparity_pixels(inv_depth: float, *, params: StereoParams, width: int) -> float:
    """Half the total disparity, in pixels, for one pixel's inverse depth --
    the amount ONE eye moves, since the two eyes move by this much in opposite
    directions.

    **Positive means nearer than the convergence plane**, i.e. the thing that
    pops out. This one sign is the whole convention; :func:`warp_eye` turns it
    into a direction and nothing else in the module makes that decision.

    This is the design doc's ``baseline * (1/depth - 1/convergence)`` written
    in the units the depth model actually emits: ``1/depth`` *is* inverse
    depth, so the subtraction is ``inv_depth - convergence`` and the baseline
    is folded into the disparity ceiling, which is the number a viewer's
    comfort is actually expressed in.
    """
    clamped = min(1.0, max(0.0, inv_depth))
    return (clamped - params.convergence) * params.max_disparity_fraction * width


def warp_eye(frame: Frame, depth: DepthMap, *, eye: str, params: StereoParams) -> Frame:
    """Forward-warp one eye's view out of a mono frame and its depth map.

    ``eye`` is ``"left"`` or ``"right"``. **A pixel nearer than the
    convergence plane moves RIGHT for the left eye and LEFT for the right
    eye**, which is negative (crossed) parallax: the left-eye image of a near
    object sits to the right of the right-eye image, the eyes converge in
    front of the display, and the object reads as being in front of the
    screen. A pixel further than the convergence plane moves the other way and
    recedes behind it.

    Occlusion is resolved by keeping the **nearer** source pixel when two land
    on the same destination -- a one-line z-buffer, and the only part of this
    that is not naive. Disocclusions (destinations nothing landed on) are
    filled by the nearest written pixel in the same row; see the module
    docstring on why that is a placeholder, not a design.
    """
    if eye not in ("left", "right"):
        raise StereoError(f"eye must be 'left' or 'right', got {eye!r}")
    _check_depth_matches(frame, depth)
    direction = 1 if eye == "left" else -1
    return _warp(frame, depth, direction, params, _depth_array(depth))


def _check_depth_matches(frame: Frame, depth: DepthMap) -> None:
    if (depth.width, depth.height) != (frame.width, frame.height):
        raise StereoError(
            f"depth map is {depth.width}x{depth.height} but the frame is "
            f"{frame.width}x{frame.height} -- a depth source must return a map the "
            "same size as the frame it was given"
        )


def _depth_array(depth: DepthMap):
    """The depth values as a float64 numpy array for the vectorised warp, or
    ``None`` to send the frame through the reference loop: when numpy is
    absent, or when the map holds NaN (whose loop behaviour is
    order-dependent and is not reproduced). Built once per frame and shared
    by both eyes."""
    np = _numpy()
    if np is None:
        return None
    values = np.asarray(depth.values, dtype=np.float64)
    if np.isnan(values).any():
        logger.warning(
            "Stereo: depth map contains NaN -- warping this frame with the slow "
            "reference loop, whose (order-dependent) NaN behaviour the vectorised "
            "path does not reproduce. A depth source should never emit NaN."
        )
        return None
    return values


def _warp(frame: Frame, depth: DepthMap, direction: int, params: StereoParams, values) -> Frame:
    if values is None:
        return _warp_eye_reference(frame, depth, direction=direction, params=params)
    np = _numpy()
    return _warp_eye_numpy(np, frame, depth, direction=direction, params=params, values=values)


def _numpy():
    """``numpy`` if it is importable, else ``None``.

    numpy is **not** a declared dependency of this project: it arrives with
    the ``faces`` extra (``opencv-python-headless`` requires it) and with
    every depth-model environment, and CI installs neither. So the
    vectorised warp is used whenever numpy is present and the reference loop
    otherwise -- same bytes either way, which the pinned tests assert. A
    function rather than a module-level import so a test can take it away.
    """
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _warp_eye_numpy(np, frame: Frame, depth: DepthMap, *, direction: int, params, values=None):
    """:func:`_warp_eye_reference`, vectorised, **byte-identical** to it on
    any depth map without NaN (tests/test_stereo.py proves it against the
    loop on 400 random cases and pins both to the same digests).

    Each step reproduces one rule of the loop exactly, not approximately:

    * the shift is the same float64 expression in the same order, and
      ``np.rint`` rounds half to even exactly as Python's ``round`` does, so
      every destination column is the loop's;
    * the z-buffer: the loop keeps a candidate only if its **unclamped**
      depth is strictly greater than what is already there (initially -1.0),
      visiting x left to right -- so each destination ends up holding the
      maximum depth, the *leftmost* source on a tie (possible only where
      round-half-to-even sends two equal-depth neighbours to one column),
      and nothing at all if every candidate was <= -1.0.
      ``np.maximum.at`` per destination, then
      ``np.minimum.at`` over the sources that reached that maximum, picks
      exactly that;
    * the hole fill copies from the nearest position whose written depth is
      >= 0.0, the **right-hand** one on a tie (the loop's ``<=`` advance), in
      rows that have at least one; a written depth in (-1, 0) still counts as
      a hole, as it does in the loop.
    """
    width, height = frame.width, frame.height
    size = width * height
    # One 3-byte item per pixel, so every gather/scatter below moves whole
    # pixels with one flat index instead of a (row, column, channel) triple.
    src = np.frombuffer(frame.pixels, dtype="V3")
    inv = np.asarray(depth.values, dtype=np.float64) if values is None else values

    columns = np.arange(width)
    flat_columns = np.tile(columns, height)
    clamped = np.minimum(1.0, np.maximum(0.0, inv))
    shift = (clamped - params.convergence) * params.max_disparity_fraction * width * direction
    dest = np.rint(flat_columns + shift)
    candidate = np.flatnonzero((dest >= 0) & (dest < width) & (inv > -1.0))

    depth_v = inv[candidate]
    source_x = flat_columns[candidate]
    slot = candidate - source_x + dest[candidate].astype(np.intp)
    # z-buffer: the maximum depth landing on each slot (-1.0 where none
    # does, the loop's initial value), then the leftmost source among the
    # candidates that reached it.
    written = np.full(size, -1.0)
    np.maximum.at(written, slot, depth_v)
    top = depth_v == written[slot]
    winner_x = np.full(size, width, dtype=np.intp)
    np.minimum.at(winner_x, slot[top], source_x[top])
    landed = np.flatnonzero(winner_x < width)

    out = np.zeros(size, dtype="V3")
    out[landed] = src[landed - flat_columns[landed] + winner_x[landed]]

    filled = (written >= 0.0).reshape(height, width)
    holes = ~filled & filled.any(axis=1, keepdims=True)
    if holes.any():
        left = np.maximum.accumulate(np.where(filled, columns, -1), axis=1)
        right = np.minimum.accumulate(np.where(filled, columns, width)[:, ::-1], axis=1)[:, ::-1]
        use_right = (right < width) & ((left < 0) | (right - columns <= columns - left))
        hole = np.flatnonzero(holes)
        nearest = np.where(use_right, right, left).reshape(-1)[hole]
        out[hole] = out[hole - flat_columns[hole] + nearest]

    return Frame(width=width, height=height, pixels=out.tobytes())


def _warp_eye_reference(frame: Frame, depth: DepthMap, *, direction: int, params) -> Frame:
    """The original per-pixel loop -- **the specification** of the warp, kept
    verbatim as the fallback when numpy is absent and as the oracle
    :func:`_warp_eye_numpy` is tested against. Several hundred thousand
    Python-level iterations per eye per frame at 864x480 -- about 0.22 s an
    eye on an M-series Mac, 13x the vectorised path; see
    ``~/mvm-runs/deathless/measurements/stereo68_2026-10-05/FINDINGS.md``
    for what that costs on a real frame."""
    width, height = frame.width, frame.height
    src = frame.pixels
    out = bytearray(len(src))

    for y in range(height):
        row_start = y * width
        written_depth: list[float] = [-1.0] * width
        for x in range(width):
            inv = depth.values[row_start + x]
            shift = half_disparity_pixels(inv, params=params, width=width) * direction
            dest = int(round(x + shift))
            if not 0 <= dest < width:
                continue
            if inv <= written_depth[dest]:
                continue
            written_depth[dest] = inv
            s = (row_start + x) * 3
            d = (row_start + dest) * 3
            out[d : d + 3] = src[s : s + 3]
        _fill_row_holes(out, written_depth, row_start, width)

    return Frame(width=width, height=height, pixels=bytes(out))


def _fill_row_holes(out: bytearray, written: list[float], row_start: int, width: int) -> None:
    """Nearest-neighbour hole fill along one row, in place.

    Naive on purpose and named as such everywhere: a real conversion inpaints.
    A row that received no pixels at all (possible only if every disparity
    pushed off the frame) is left black rather than invented -- a black band
    is a visible defect, which is the right failure for a scaffold.
    """
    holes = [x for x in range(width) if written[x] < 0.0]
    if not holes or len(holes) == width:
        return
    filled = [x for x in range(width) if written[x] >= 0.0]
    j = 0
    for x in holes:
        while j + 1 < len(filled) and abs(filled[j + 1] - x) <= abs(filled[j] - x):
            j += 1
        source = filled[j]
        s = (row_start + source) * 3
        d = (row_start + x) * 3
        out[d : d + 3] = out[s : s + 3]


def stereo_pair(
    frame: Frame, depth: DepthMap, *, params: StereoParams | None = None
) -> tuple[Frame, Frame]:
    """``(left_eye, right_eye)`` for one mono frame. See :func:`warp_eye` for
    the sign convention, which this function only fans out."""
    params = params or StereoParams()
    _check_depth_matches(frame, depth)
    values = _depth_array(depth)
    return (
        _warp(frame, depth, 1, params, values),
        _warp(frame, depth, -1, params, values),
    )


# --------------------------------------------------------------------------- #
# Output formats
# --------------------------------------------------------------------------- #


def side_by_side(left: Frame, right: Frame) -> Frame:
    """Full side-by-side: left eye then right eye, on every row. Doubles the
    width -- the constraint that makes stereo all-or-none across a run's
    chunks (module docstring)."""
    if (left.width, left.height) != (right.width, right.height):
        raise StereoError("side-by-side needs two frames of identical size")
    width, height = left.width, left.height
    out = bytearray(width * height * 6)
    stride = width * 3
    for y in range(height):
        src = y * stride
        dest = y * stride * 2
        out[dest : dest + stride] = left.pixels[src : src + stride]
        out[dest + stride : dest + stride * 2] = right.pixels[src : src + stride]
    return Frame(width=width * 2, height=height, pixels=bytes(out))


def anaglyph(left: Frame, right: Frame) -> Frame:
    """Red/cyan anaglyph: the left eye's RED channel with the right eye's
    GREEN and BLUE. A review tool, never a deliverable (see
    :data:`FORMAT_ANAGLYPH`)."""
    if (left.width, left.height) != (right.width, right.height):
        raise StereoError("an anaglyph needs two frames of identical size")
    out = bytearray(len(left.pixels))
    out[0::3] = left.pixels[0::3]
    out[1::3] = right.pixels[1::3]
    out[2::3] = right.pixels[2::3]
    return Frame(width=left.width, height=left.height, pixels=bytes(out))


def compose(left: Frame, right: Frame, output_format: str) -> Frame:
    """Dispatch to :func:`side_by_side` or :func:`anaglyph`."""
    if output_format == FORMAT_SIDE_BY_SIDE:
        return side_by_side(left, right)
    if output_format == FORMAT_ANAGLYPH:
        return anaglyph(left, right)
    raise StereoError(f"unknown output format {output_format!r}; valid: {OUTPUT_FORMATS}")


# --------------------------------------------------------------------------- #
# The ffmpeg seam -- injectable, identical shape to luminance/scenecuts
# --------------------------------------------------------------------------- #

SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]
"""Same shape ``luminance.SubprocessRunner`` and ``scenecuts`` already use, so
a caller that has one can pass it straight through."""

PipeRunner = Callable[[Sequence[str], bytes], "subprocess.CompletedProcess"]
"""``(argv, stdin_bytes) -> CompletedProcess``. Encoding needs to *write* to
ffmpeg, which the args-only runner above cannot express. Buffered, which is
one of the reasons the production path needs rewriting (module docstring)."""


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False)


def _default_pipe_runner(args: Sequence[str], payload: bytes) -> subprocess.CompletedProcess:
    """Real invocation with stdin. Never used by unit tests."""
    return subprocess.run(list(args), input=payload, capture_output=True, check=False)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    """Kept exact (``24/1``, ``24000/1001``) rather than floated: the whole
    point of converting per chunk is that the concat demuxer can still
    ``-c:v copy``, and a frame rate that arrives back as 23.976023976 is a
    different stream parameter from the one it left as."""


def build_probe_args(video_path: Path | str) -> list[str]:
    """ffprobe argv for one video stream's width, height and frame rate. Pure,
    so a test can assert on the argv without ffprobe being installed."""
    return [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]  # fmt: skip


def build_decode_args(video_path: Path | str) -> list[str]:
    """ffmpeg argv decoding a whole file to packed RGB24 on stdout."""
    return [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]  # fmt: skip


def build_encode_args(dest: Path | str, *, width: int, height: int, fps: Fraction) -> list[str]:
    """ffmpeg argv encoding packed RGB24 from stdin to an h264 mp4.

    ``yuv420p`` and h264 because the concat demuxer compares stream
    parameters and every chunk of the real render is h264/yuv420p already --
    see the module docstring's all-or-none note. **This re-encodes**, which is
    exactly why the design doc puts the pass before concat rather than after
    it: Stage 5's ``-c:v copy`` invariant stays intact because Stage 5 never
    sees these files.
    """
    return [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]  # fmt: skip


def probe_video(video_path: Path | str, *, runner: SubprocessRunner | None = None) -> VideoInfo:
    """Width, height and exact frame rate of a file's first video stream."""
    runner = runner or _default_runner
    args = build_probe_args(video_path)
    try:
        result = runner(args)
    except Exception as exc:  # noqa: BLE001 - one message, whatever failed.
        raise StereoError(f"could not run ffprobe on {video_path}: {exc}") from exc
    if result.returncode != 0:
        raise StereoError(f"ffprobe failed on {video_path} (exit {result.returncode})")
    text = (result.stdout or b"").decode("utf-8", "replace").strip()
    first = text.splitlines()[0] if text else ""
    parts = first.split(",")
    if len(parts) < 3:
        raise StereoError(f"could not parse ffprobe output for {video_path}: {text!r}")
    try:
        return VideoInfo(width=int(parts[0]), height=int(parts[1]), fps=Fraction(parts[2]))
    except (ValueError, ZeroDivisionError) as exc:
        raise StereoError(
            f"could not parse ffprobe output for {video_path}: {text!r} ({exc})"
        ) from exc


def decode_frames(
    video_path: Path | str, info: VideoInfo, *, runner: SubprocessRunner | None = None
) -> list[Frame]:
    """Every frame of a file as :class:`Frame`. Buffers the whole chunk (see
    the module docstring); a trailing partial frame is dropped with a warning
    rather than raising, because a truncated last frame is an ffmpeg/container
    artefact and not a reason to lose a conversion."""
    runner = runner or _default_runner
    try:
        result = runner(build_decode_args(video_path))
    except Exception as exc:  # noqa: BLE001
        raise StereoError(f"could not run ffmpeg to decode {video_path}: {exc}") from exc
    if result.returncode != 0:
        raise StereoError(f"ffmpeg failed decoding {video_path} (exit {result.returncode})")
    raw = result.stdout or b""
    stride = info.width * info.height * 3
    count, remainder = divmod(len(raw), stride)
    if remainder:
        logger.warning(
            "%s: decoded %d trailing bytes that do not make a whole %dx%d frame -- "
            "dropping them and converting the %d whole frames",
            video_path,
            remainder,
            info.width,
            info.height,
            count,
        )
    if count == 0:
        raise StereoError(f"{video_path} decoded to no whole frames")
    return [
        Frame(width=info.width, height=info.height, pixels=raw[i * stride : (i + 1) * stride])
        for i in range(count)
    ]


def encode_frames(
    frames: Sequence[Frame],
    dest: Path | str,
    *,
    fps: Fraction,
    runner: PipeRunner | None = None,
) -> Path:
    """Encode frames to ``dest``. All frames must share one size -- a stereo
    pass that changed size mid-file would produce exactly the mixed-parameter
    stream the concat demuxer cannot copy."""
    runner = runner or _default_pipe_runner
    if not frames:
        raise StereoError("nothing to encode: no frames")
    width, height = frames[0].width, frames[0].height
    if any((f.width, f.height) != (width, height) for f in frames):
        raise StereoError("every frame handed to the encoder must be the same size")
    payload = b"".join(f.pixels for f in frames)
    args = build_encode_args(dest, width=width, height=height, fps=fps)
    try:
        result = runner(args, payload)
    except Exception as exc:  # noqa: BLE001
        raise StereoError(f"could not run ffmpeg to encode {dest}: {exc}") from exc
    if result.returncode != 0:
        raise StereoError(f"ffmpeg failed encoding {dest} (exit {result.returncode})")
    return Path(dest)


def external_depth_source(
    command: Sequence[str], *, runner: PipeRunner | None = None
) -> DepthSource:
    """Build a :data:`DepthSource` that shells out, once per frame, to an
    external depth command.

    **The production path, documented and never run here.** The contract is
    deliberately dumb so that whatever ends up on doris only has to meet it:

    * this module writes the frame to the command's **stdin** as packed RGB24
      (``width*height*3`` bytes; the command is told the size by whatever
      wrapper script it is, since ffmpeg's own raw format carries none);
    * the command writes **stdout** as ``width*height`` little-endian
      ``uint16`` samples of normalised inverse depth, 0 = farthest,
      65535 = nearest -- i.e. ``gray16le``, which is what every depth demo
      writes and what ffmpeg can read back if anyone wants to look at it;
    * a non-zero exit, or the wrong number of bytes, raises
      :class:`StereoError` and costs one chunk.

    The intended invocation on doris, for the record and for whoever builds
    the wrapper (**this has not been run; the weights are not installed and
    are not committed** -- see :data:`DEPTH_ANYTHING_V2_SMALL` for the licence
    trap in the Base/Large variants)::

        python ~/depth-anything-v2/depth_stdio.py \\
            --encoder vits --width 864 --height 480

    Normalise over the **whole chunk**, not per frame: a per-frame min-max
    re-ranges the depth every time the content changes, which is the mechanism
    suspected behind the arriving-object boil in ``docs/pop-beat-corpus.md``.
    That is a property of the wrapper, not of this function, and this function
    cannot check it.
    """
    runner = runner or _default_pipe_runner

    def source(frame_index: int, frame: Frame) -> DepthMap:
        try:
            result = runner(list(command), frame.pixels)
        except Exception as exc:  # noqa: BLE001
            raise StereoError(f"depth command failed on frame {frame_index}: {exc}") from exc
        if result.returncode != 0:
            raise StereoError(
                f"depth command exited {result.returncode} on frame {frame_index}"
            )
        raw = result.stdout or b""
        expected = frame.width * frame.height * 2
        if len(raw) != expected:
            raise StereoError(
                f"depth command returned {len(raw)} bytes for frame {frame_index}; a "
                f"{frame.width}x{frame.height} gray16le map is {expected}"
            )
        np = _numpy()
        if np is not None:
            # Same IEEE division per sample as the loop below, so equal values.
            values = (np.frombuffer(raw, dtype="<u2").astype(np.float64) / 65535.0).tolist()
        else:
            values = [
                int.from_bytes(raw[i : i + 2], "little") / 65535.0
                for i in range(0, len(raw), 2)
            ]
        return DepthMap(width=frame.width, height=frame.height, values=values)

    return source


# --------------------------------------------------------------------------- #
# The one orchestrating entry point
# --------------------------------------------------------------------------- #


def convert_chunk(
    source: Path | str,
    dest: Path | str,
    depth_source: DepthSource,
    *,
    output_format: str = FORMAT_SIDE_BY_SIDE,
    params: StereoParams | None = None,
    runner: SubprocessRunner | None = None,
    pipe_runner: PipeRunner | None = None,
) -> Path:
    """Convert one rendered chunk mp4 to a stereo one. **Never run on a real
    chunk** -- see the module docstring.

    Reads ``source``, writes ``dest``, and touches nothing else: no fingerprint
    moves, no ``run_state.json`` is written, and the original chunk is left
    exactly as the render produced it. One chunk per call, so a failure costs
    one chunk (design doc), and the caller decides whether a video with one
    unconverted chunk is worth assembling -- it is not, and that is the
    all-or-none width constraint, which is the caller's to enforce because
    only the caller can see the other 79.
    """
    if output_format not in OUTPUT_FORMATS:
        raise StereoError(f"unknown output format {output_format!r}; valid: {OUTPUT_FORMATS}")
    params = params or StereoParams()
    info = probe_video(source, runner=runner)
    frames = decode_frames(source, info, runner=runner)
    logger.info(
        "Stereo: converting %s (%d frames, %dx%d @ %s) -> %s, format=%s, "
        "convergence=%.3f, max disparity %.3f%% of width",
        source,
        len(frames),
        info.width,
        info.height,
        info.fps,
        dest,
        output_format,
        params.convergence,
        params.max_disparity_fraction * 100,
    )
    converted = []
    for index, frame in enumerate(frames):
        depth = depth_source(index, frame)
        left, right = stereo_pair(frame, depth, params=params)
        converted.append(compose(left, right, output_format))
    return encode_frames(converted, dest, fps=info.fps, runner=pipe_runner)


__all__ = [
    "DEFAULT_CONVERGENCE",
    "DEFAULT_MAX_DISPARITY_FRACTION",
    "DEPTH_ANYTHING_V2_SMALL",
    "FORMAT_ANAGLYPH",
    "FORMAT_SIDE_BY_SIDE",
    "OUTPUT_FORMATS",
    "DepthMap",
    "DepthSource",
    "Frame",
    "PipeRunner",
    "StereoError",
    "StereoParams",
    "SubprocessRunner",
    "VideoInfo",
    "anaglyph",
    "build_decode_args",
    "build_encode_args",
    "build_probe_args",
    "compose",
    "convert_chunk",
    "decode_frames",
    "encode_frames",
    "external_depth_source",
    "half_disparity_pixels",
    "probe_video",
    "side_by_side",
    "stereo_pair",
    "warp_eye",
]
