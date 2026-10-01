"""Fulldome (domemaster) output: a rendered flat clip becomes a dome-ready
trailer -- 4096x4096 equidistant fisheye, front at the bottom, lossless
16-bit PNG master, H.265 distribution copy, six mono 5.1 stems with a 2-pop,
and an automated conformance check. Design and gap analysis:
``docs/design-fulldome.md``.

**It composites; it does not generate.** The "window" route (design doc,
route A) maps the flat H3 render onto a curved window in the front of the
dome over a dome-native starfield. The full-dome routes (design doc, D1
procedural and D2 panorama-with-depth) render their fisheye frames
somewhere else -- a shader, a parallax renderer -- and hand this module a
**base** video at exactly the dome's size and rate, which replaces the
starfield; an H3 clip then becomes an optional inset (``opacity`` < 1, a
small ``h_fov``) rather than the frame -- or, with ``source_alpha``, a cut-out
whose own alpha is kept. A **foreground** (a fisheye video *with* alpha, same
contract as the base) is laid over the window and under the credits, so a
landscape can stand in front of a figure. Masking, master, distribution copy,
stems, 2-pop, orientation and verification are the same code on every
route.

Strictly outside the render path
--------------------------------
Like ``stereo.py``, this reads a *finished* video (the assembled
``final_video.mp4`` or any chunk mp4) and writes new files somewhere else. It
imports nothing from ``resilience``, ``execution`` or ``assembly``, touches no
``run_state.json`` and moves no ``ChunkFingerprint``; it cannot cause a
re-render. It is stdlib-only on purpose, so it also runs on a host that has
ffmpeg and nothing else installed (the 4096x4096 render is CPU- and
disk-bound, not GPU-bound, and the box with 200 GB free is not the one with
the checkout).

Every ffmpeg call is an argv list built by a pure function and executed
through an injectable runner -- the same seam ``assembly.py`` and
``luminance.py`` use -- so the unit tests assert on argument lists and the
verifier is exercised against scripted bytes, and exactly one
``@pytest.mark.integration`` test runs real ffmpeg at a toy size.

Geometry, stated once
---------------------
ffmpeg's ``v360`` filter is the whole projection engine. Its ``fisheye``
format is *equidistant* (pixel radius proportional to the angle from the
optical axis -- ``xyz_to_fisheye`` in ``vf_v360.c`` writes ``theta / pi`` into
the radius), which is exactly the angular fisheye a domemaster is, and with
``h_fov=v_fov=180`` the circle's edge is 90 degrees from the centre: the
springline. The output's forward direction is the frame centre, the dome's
zenith.

Measured on a labelled test pattern (2026-09-28, ffmpeg 8.1.2), because the
sign of a rotation is a silent bug: **a positive ``pitch`` moves a flat layer
toward the bottom of the frame** with the layer's own top pointing at the
zenith and its left on the frame's left. A viewer standing at the dome's
centre facing the front (the bottom of the frame) sees the layer upright and
unmirrored, which is the domemaster convention: the file is a camera looking
up, not a map looked down on. So a window centred ``e`` degrees above the
dome's horizon is ``pitch = 90 - e``, and ``roll`` rotates the whole picture
about the zenith -- that is the per-venue orientation parameter the spec asks
for, applied to every projected layer and to the background, never by
re-rendering content.

A flat (gnomonic) layer's vertical field of view is **not** ``h_fov * h / w``.
It follows the tangent: ``2 * atan(tan(h_fov / 2) * h / w)``, 58.7 degrees for
a 16:9 frame at 90 wide, not 50.6. ``flat_v_fov_degrees`` is the one place
that is computed.

Alpha through the projection, also measured -- and corrected 2026-09-28:
``v360`` remaps the alpha plane like any other, and a flat input **clamps
at its edge**: every output pixel the input does not cover repeats the
nearest edge pixel, alpha included. It does *not* write alpha 0 there (an
earlier version of this docstring said it did; the route-A render was only
right because its window was feathered to zero at the border). A layer with
an opaque edge floods the whole dome. So every layer's alpha is forced to
0 on its outermost pixel ring before projection -- ``_EDGE_ZERO``, applied
to the window at any feather and to the credits card whatever its own alpha
says -- and a feathered edge is simply a feathered alpha on top of that:
one ``v360`` pass per layer, no mask pass, composited by ``overlay`` and
never by addition.

Frame-accurate sync
-------------------
The master is ``leader + programme`` long. The leader is black; the frame at
``pop_seconds`` (default 1.0) is a full-disc white flash, and the audio stems
carry a one-frame 1 kHz tone at -20 dBFS starting at exactly that time,
two seconds before the programme begins at ``leader_seconds`` (default 3.0).
Audio is cut by *sample count* (``atrim=start_sample``), never by a ``-ss``
seek, and padded to exactly ``total_seconds * sample_rate`` samples, so the
six stems and the image sequence are the same length to the sample.
``verify_render`` finds the flash frame by luminance and the tone onset by
threshold and reports the difference in milliseconds; more than half a frame
fails.

The 5.1 mix is not the stereo master
------------------------------------
The render's timeline was cut against the stereo master, and the 5.1 master
of the same song is a different file with a different head: on "Deathless"
it is 505.560 s against 512.080 s, and cross-correlation puts its first
sample **1.288146 s** into the stereo master's timeline. ``render`` takes
that as ``surround_offset_seconds`` (surround time minus stereo time, so
negative here) and refuses a span that would start before the surround file
does, because the alternative is a stem that is silently a second and a
quarter out. The number is measured per song, not assumed; the design doc
records how.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import subprocess
import sys
from array import array
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, check=False)


DOMEMASTER_SIZE = 4096
DEFAULT_FPS = 30
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DISTRIBUTION_MBITS = 60
"""Jena's guideline: H.265 mp4 at roughly 60 Mbit/s."""

SURROUND_CHANNELS: tuple[str, ...] = ("L", "R", "C", "LFE", "Ls", "Rs")
"""Deliverable stem names, in the spec's order."""
FFMPEG_SURROUND_LAYOUT = "5.1(side)"
"""What ffprobe reports for the Refestramus 5.1 FLAC masters. Its channel
order is FL FR FC LFE SL SR -- the same order as :data:`SURROUND_CHANNELS`,
with "side" and "back" surrounds being two labels for one speaker pair."""
_FFMPEG_CHANNEL_LABELS: tuple[str, ...] = ("FL", "FR", "FC", "LFE", "SL", "SR")

FRAME_PATTERN = "dome_%06d.png"
DISTRIBUTION_FILENAME = "dome_4096_h265.mp4"
MANIFEST_FILENAME = "dome_manifest.json"
VERIFICATION_FILENAME = "dome_verification.json"

RETIME_MODES: tuple[str, ...] = ("mci", "blend", "dup")
"""How 24 fps becomes the dome's 30. ``mci`` is motion-compensated
interpolation (``minterpolate``, the default), ``blend`` is frame blending,
``dup`` is plain 4:5 duplication -- kept because it is the only one with no
synthesised frames, and the one to compare against when judging shimmer."""

SYNC_TOLERANCE_FRAMES = 0.5
POP_LEVEL_DBFS = -20.0
POP_ONSET_THRESHOLD = 200
"""Of 32767: about -44 dBFS. The leader is digital silence, so anything above
this is the tone; the first sample of a sine is zero, so the onset lands one
sample late by construction (0.02 ms at 48 kHz)."""
LUMA_FLASH_FLOOR = 128
LUMA_DARK_CEILING = 40

LUMA_SCAN_SIZE = 64
"""Side of the downscale the whole-programme brightness scan decodes to."""
BRIGHT_FIELD_CEILING_Y = 96.0
"""Spec section 5: no large, very bright full-dome frames. The highest mean
luma (0-255, full range, disc only) any programme frame may have. A starting
value, not calibrated against a dome: the route-A Deathless render peaks far
below it, and a full-white dome is 255."""
FLASH_DELTA_Y = 20.0
"""A luminance swing of the whole disc's mean at least this large (0-255)
counts as one transition; a pair of opposing transitions is one flash."""
MAX_FLASHES_PER_SECOND = 3
"""ITU-R BT.1702 / Ofcom's general flash rule: more than three flashes in
any one-second period fails. Applied to the *whole-disc mean* only, so it
catches the spec's "rapid full-field flashing" and nothing smaller -- a
local strobe covering a quarter of the dome dilutes below the threshold."""

FISHEYE_ROLL_SIGN = 1
"""Sign of ``v360``'s ``roll`` on a fisheye->fisheye pass relative to the
projected layers' roll, so a base turns with the window over it. Measured by
``test_a_fisheye_base_turns_with_the_projected_layers``, never reasoned: the
preview's *pitch* flips for a fisheye input, and roll need not do the same."""


class DomeCommandError(RuntimeError):
    """An ffmpeg/ffprobe call exited non-zero. Carries argv and stderr."""

    def __init__(self, stage: str, cmd: Sequence[str], returncode: int, stderr: bytes | str):
        self.stage = stage
        self.cmd = tuple(cmd)
        self.returncode = returncode
        self.stderr = (
            stderr if isinstance(stderr, str) else stderr.decode("utf-8", errors="replace")
        )
        super().__init__(
            f"dome {stage} failed (exit={returncode}): {' '.join(self.cmd)}\n"
            f"stderr:\n{self.stderr}"
        )


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DomeSpec:
    """The image and timing contract for one render."""

    size: int = DOMEMASTER_SIZE
    fps: int = DEFAULT_FPS
    front_rotation_degrees: float = 0.0
    """Venue orientation: rotate the whole picture about the zenith. 0 puts
    the audience's front at the bottom of the frame, the standard convention."""
    leader_seconds: float = 3.0
    pop_seconds: float = 1.0
    pop_frequency_hz: int = 1000
    pop_level_dbfs: float = POP_LEVEL_DBFS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    bit_depth: int = 16

    def __post_init__(self) -> None:
        if self.size <= 0 or self.size % 2:
            raise ValueError(f"size must be a positive even number of pixels, got {self.size}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.leader_seconds < 0 or self.pop_seconds < 0:
            raise ValueError("leader_seconds and pop_seconds must be non-negative")
        if self.pop_seconds >= self.leader_seconds:
            raise ValueError(
                f"the 2-pop must fall inside the leader: pop_seconds={self.pop_seconds} "
                f"is not before leader_seconds={self.leader_seconds}"
            )
        if self.bit_depth not in (8, 16):
            raise ValueError(f"bit_depth must be 8 or 16, got {self.bit_depth}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def radius(self) -> float:
        return self.size / 2

    @property
    def pop_frame(self) -> int:
        return round(self.pop_seconds * self.fps)

    @property
    def program_start_frame(self) -> int:
        return round(self.leader_seconds * self.fps)

    @property
    def png_pix_fmt(self) -> str:
        return "rgb48be" if self.bit_depth == 16 else "rgb24"

    def total_seconds(self, program_seconds: float) -> float:
        return self.leader_seconds + program_seconds

    def frame_count(self, program_seconds: float) -> int:
        return round(self.total_seconds(program_seconds) * self.fps)

    def sample_count(self, program_seconds: float) -> int:
        return round(self.total_seconds(program_seconds) * self.sample_rate)


@dataclass(frozen=True)
class WindowPlacement:
    """Where a flat layer sits on the dome, in the audience's terms."""

    h_fov_degrees: float = 90.0
    elevation_degrees: float = 40.0
    """Centre of the layer above the dome's horizon. The spec's comfortable
    band is 20-60 degrees; the zenith (90) is where nobody is looking."""
    azimuth_degrees: float = 0.0
    """0 is the front; positive turns the layer to the audience's right."""
    feather_px: int = 24
    """Soft edge, in pixels of the *source* layer, painted into its alpha
    before projection so the window has no hard seam against the sky."""
    opacity: float = 1.0
    """Below 1 the layer is an inset seen *through*, not a screen in front of
    the base -- the full-dome routes' way of placing an H3 clip in the dome."""
    source_alpha: bool = False
    """Keep the clip's own alpha (a matted cut-out) instead of painting an
    opaque rectangle; the feather and opacity still multiply it."""

    def __post_init__(self) -> None:
        if not 0 < self.h_fov_degrees < 180:
            raise ValueError(f"h_fov_degrees must be in (0, 180), got {self.h_fov_degrees}")
        if not 0 <= self.elevation_degrees <= 90:
            raise ValueError(
                f"elevation_degrees must be in [0, 90], got {self.elevation_degrees}"
            )
        if self.feather_px < 0:
            raise ValueError(f"feather_px must be non-negative, got {self.feather_px}")
        if not 0 < self.opacity <= 1:
            raise ValueError(f"opacity must be in (0, 1], got {self.opacity}")

    @property
    def pitch_degrees(self) -> float:
        """v360 pitch: measured down from the zenith (see the module docstring
        for the measured sign)."""
        return 90.0 - self.elevation_degrees

    def roll_degrees(self, spec: DomeSpec) -> float:
        """v360 roll: the turn about the zenith, which is the fisheye's optical
        axis -- so a layer's azimuth *and* the venue rotation are both roll.
        (Corrected 2026-09-30: azimuth used to be passed as ``yaw``, which tilts
        the layer sideways about the frame's vertical instead; measured, a
        window at azimuth 90 landed on the frame's left springline and one at
        180 vanished. Route A only ever used azimuth 0, so nothing rendered
        showed it.)"""
        return spec.front_rotation_degrees + FISHEYE_ROLL_SIGN * self.azimuth_degrees


@dataclass(frozen=True)
class CreditsLayer:
    """The band name / call-to-action card: an RGBA still the operator makes
    (any tool; the run directory keeps the script that made this one), shown
    from ``at_seconds`` into the programme to its end, faded in."""

    image_width: int
    image_height: int
    placement: WindowPlacement
    at_seconds: float
    fade_seconds: float = 1.0


def flat_v_fov_degrees(width: int, height: int, h_fov_degrees: float) -> float:
    """Vertical FOV of a gnomonic image of the given aspect at ``h_fov``."""
    half = math.radians(h_fov_degrees) / 2
    return math.degrees(2 * math.atan(math.tan(half) * height / width))


def retime_filter(mode: str, fps: int) -> str:
    if mode == "mci":
        return f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    if mode == "blend":
        return f"minterpolate=fps={fps}:mi_mode=blend"
    if mode == "dup":
        return f"fps={fps}"
    raise ValueError(f"unknown retime mode {mode!r}; expected one of {RETIME_MODES}")


def inside_span(row: int, size: int) -> tuple[int, int] | None:
    """Pixel columns ``[x0, x1)`` of ``row`` whose centres lie inside the disc
    of diameter ``size`` -- the same inequality :func:`build_disc_mask_args`
    hands ffmpeg, so the verifier and the mask agree to the pixel."""
    r = size / 2
    dy = row + 0.5 - r
    if abs(dy) > r:
        return None
    half = math.sqrt(max(r * r - dy * dy, 0.0))
    x0 = max(0, math.ceil(r - half - 0.5))
    x1 = min(size, math.floor(r + half - 0.5) + 1)

    def inside(x: int) -> bool:
        return math.hypot(x + 0.5 - r, dy) <= r

    while x0 > 0 and inside(x0 - 1):
        x0 -= 1
    while x0 < x1 and not inside(x0):
        x0 += 1
    while x1 < size and inside(x1):
        x1 += 1
    while x1 > x0 and not inside(x1 - 1):
        x1 -= 1
    return (x0, x1) if x1 > x0 else None


# --------------------------------------------------------------------------- #
# argv builders
# --------------------------------------------------------------------------- #


def _projection(placement: WindowPlacement, spec: DomeSpec, width: int, height: int) -> str:
    v_fov = flat_v_fov_degrees(width, height, placement.h_fov_degrees)
    return (
        f"v360=input=flat:ih_fov={placement.h_fov_degrees:.1f}:iv_fov={v_fov:.2f}"
        f":output=fisheye:h_fov=180:v_fov=180:w={spec.size}:h={spec.size}"
        f":yaw=0.0:pitch={placement.pitch_degrees:.1f}"
        f":roll={placement.roll_degrees(spec):.1f}:interp=lanc"
    )


def build_disc_mask_args(out_png: Path, *, size: int) -> list[str]:
    """One white disc on black, edge to edge: the hard mask that guarantees
    pure black outside the circle, and doubles as the 2-pop flash frame."""
    r = size / 2
    lavfi = (
        f"color=c=black:s={size}x{size}:d=1,format=gray,"
        f"geq=lum='if(lte(hypot(X+0.5-{r},Y+0.5-{r}),{r}),255,0)'"
    )
    return ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", lavfi,
            "-frames:v", "1", str(out_png)]


def build_background_args(
    out_png: Path,
    *,
    size: int,
    star_density: float = 0.0006,
    max_brightness: int = 150,
    front_rotation_degrees: float = 0.0,
) -> list[str]:
    """A dome-native background: a sparse, dim starfield drawn once on a 2:1
    equirectangular texture and projected to the fisheye. Dark on purpose --
    the spec's rule that light reflects across a dome and washes out contrast
    -- and static on purpose, because a rotating sky around the viewer is the
    motion the comfort rules forbid."""
    lavfi = (
        f"nullsrc=s={2 * size}x{size}:d=1,"
        f"geq=lum='if(gt(random(1),{1 - star_density:g}),{max_brightness}*random(2),0)'"
        ":cb=128:cr=128,gblur=sigma=1.1,"
        f"v360=input=equirect:output=fisheye:h_fov=180:v_fov=180:w={size}:h={size}"
        f":roll={front_rotation_degrees:.1f}:interp=lanc,format=gray"
    )
    return ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", lavfi,
            "-frames:v", "1", str(out_png)]


_EDGE_ZERO = "gt(min(min(X,W-1-X),min(Y,H-1-Y)),0)"
"""1 inside, 0 on the outermost pixel ring: what the projection clamps to."""


def _feather_alpha(feather_px: int, opacity: float = 1.0, source_alpha: bool = False) -> str:
    peak = "alpha(X,Y)" if source_alpha else "255"
    if opacity != 1:
        peak += f"*{opacity:g}"
    if feather_px <= 0:
        return f"a='{peak}*{_EDGE_ZERO}'"
    return f"a='{peak}*clip(min(min(X,W-1-X),min(Y,H-1-Y))/{feather_px},0,1)'"


def _fisheye_roll(spec: DomeSpec) -> str:
    if not spec.front_rotation_degrees:
        return ""
    return (
        f",v360=input=fisheye:ih_fov=180:iv_fov=180:output=fisheye:h_fov=180:v_fov=180"
        f":w={spec.size}:h={spec.size}"
        f":roll={FISHEYE_ROLL_SIGN * spec.front_rotation_degrees:.1f}:interp=lanc"
    )


def build_master_filter(
    spec: DomeSpec,
    window: WindowPlacement | None,
    *,
    source_width: int,
    source_height: int,
    source_start: float,
    program_seconds: float,
    retime: str,
    credits: CreditsLayer | None,
    base: bool = False,
    foreground: bool = False,
) -> str:
    """The whole compositing graph. Inputs, in order: the source clip (only
    when ``window`` is given), the background -- a still, or with ``base``
    a fisheye video already at the dome's size and rate, read from its first
    frame -- the foreground (a fisheye video with alpha) when there is one,
    the disc mask, and the credits card when there is one. Output label
    ``[out]``."""
    has_source = window is not None
    i_bg = 1 if has_source else 0
    i_fg = i_bg + 1
    i_disc = i_bg + (2 if foreground else 1)
    i_credits = i_disc + 1
    if base:
        bg = (f"[{i_bg}:v]trim=end_frame={round(program_seconds * spec.fps)},"
              f"setpts=PTS-STARTPTS,format=gbrp{_fisheye_roll(spec)}[bg]")
    else:
        bg = f"[{i_bg}:v]format=gbrp[bg]"
    parts = []
    if has_source:
        end = source_start + program_seconds
        parts += [
            f"[0:v]trim=start={source_start}:end={end},setpts=PTS-STARTPTS,"
            f"{retime_filter(retime, spec.fps)},format=rgba,"
            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
            f":{_feather_alpha(window.feather_px, window.opacity, window.source_alpha)}[win_flat]",
            f"[win_flat]{_projection(window, spec, source_width, source_height)}[win]",
        ]
    parts += [bg, f"[{i_disc}:v]format=gbrp,split=2[disc][pop]"]
    parts.append("[bg][win]overlay=format=gbrp:shortest=1[c1]" if has_source
                 else "[bg]null[c1]")
    tail_in = "c1"
    if foreground:
        parts += [
            f"[{i_fg}:v]trim=end_frame={round(program_seconds * spec.fps)},"
            f"setpts=PTS-STARTPTS,format=gbrap{_fisheye_roll(spec)}[fg]",
            "[c1][fg]overlay=format=gbrp[c1f]",
        ]
        tail_in = "c1f"
    if credits is not None:
        parts += [
            f"[{i_credits}:v]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
            f":a='alpha(X,Y)*{_EDGE_ZERO}*clip((T-{credits.at_seconds})/{credits.fade_seconds},0,1)'"
            "[cred_flat]",
            "[cred_flat]"
            + _projection(credits.placement, spec, credits.image_width, credits.image_height)
            + "[cred]",
            f"[{tail_in}][cred]overlay=format=gbrp:enable='gte(t,{credits.at_seconds})'[c2]",
        ]
        tail_in = "c2"
    parts += [
        f"[{tail_in}][disc]blend=all_mode=multiply:shortest=1[c3]",
        f"[c3]tpad=start_duration={spec.leader_seconds}:color=black[c4]",
        f"[c4][pop]overlay=format=gbrp:enable='eq(n,{spec.pop_frame})'[c5]",
        f"[c5]tpad=stop_mode=clone:stop={spec.fps},format={spec.png_pix_fmt}[out]",
    ]
    return ";".join(parts)


def build_master_args(
    *,
    source: Path | None,
    background_png: Path | None,
    disc_png: Path,
    credits_png: Path | None,
    out_pattern: Path,
    spec: DomeSpec,
    window: WindowPlacement | None,
    source_width: int,
    source_height: int,
    source_start: float,
    program_seconds: float,
    retime: str,
    credits: CreditsLayer | None,
    threads: int | None = None,
    base: Path | None = None,
    foreground: Path | None = None,
) -> list[str]:
    """The master render: one PNG per frame, ``-frames:v`` capped at exactly
    the frame count the spec promises. ``base`` (a fisheye video) replaces
    ``background_png``; ``source`` and ``window`` come together or not at all."""
    if (credits_png is None) != (credits is None):
        raise ValueError("credits_png and credits must be given together or not at all")
    if (source is None) != (window is None):
        raise ValueError("source and window must be given together or not at all")
    if source is None and base is None:
        raise ValueError("nothing to show: give a source or a base (or both)")
    if base is None and background_png is None:
        raise ValueError("a render without a base needs the background still")
    total = spec.total_seconds(program_seconds)
    args = ["ffmpeg", "-v", "error", "-y"]
    if threads:
        args += ["-threads", str(threads), "-filter_threads", str(threads)]
    if source is not None:
        args += ["-i", str(source)]
    stills: tuple[Path, ...] = (disc_png,) + ((credits_png,) if credits_png else ())
    if base is not None:
        args += ["-i", str(base)]
    else:
        stills = (background_png,) + stills
        if foreground is not None:
            raise ValueError("a foreground needs a base video (it is read right after it)")
    if foreground is not None:
        args += ["-i", str(foreground)]
    for still in stills:
        args += ["-loop", "1", "-framerate", str(spec.fps), "-t", f"{total:.6f}", "-i", str(still)]
    graph = build_master_filter(
        spec, window, source_width=source_width, source_height=source_height,
        source_start=source_start, program_seconds=program_seconds, retime=retime,
        credits=credits, base=base is not None, foreground=foreground is not None,
    )
    args += [
        "-filter_complex", graph, "-map", "[out]",
        "-r", str(spec.fps), "-frames:v", str(spec.frame_count(program_seconds)),
        "-start_number", "0", "-pix_fmt", spec.png_pix_fmt, str(out_pattern),
    ]
    return args


def build_distribution_args(
    pattern: Path, out_mp4: Path, *, fps: int, mbits: int = DEFAULT_DISTRIBUTION_MBITS,
    threads: int | None = None,
) -> list[str]:
    """H.265 distribution copy encoded *from the master sequence*, Rec.709
    tagged and converted explicitly (swscale's RGB->YUV default is BT.601)."""
    args = ["ffmpeg", "-v", "error", "-y"]
    if threads:
        args += ["-threads", str(threads)]
    args += [
        "-framerate", str(fps), "-start_number", "0", "-i", str(pattern),
        "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
        "-c:v", "libx265", "-preset", "medium",
        "-b:v", f"{mbits}M", "-maxrate", f"{int(mbits * 1.2)}M", "-bufsize", f"{mbits * 2}M",
        "-x265-params", f"keyint={fps * 2}:min-keyint={fps}",
        "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", "-an", str(out_mp4),
    ]
    return args


def build_audio_args(
    *,
    surround: Path,
    surround_start_seconds: float,
    program_seconds: float,
    spec: DomeSpec,
    out_dir: Path,
) -> list[str]:
    """Six mono stems plus a stereo fold-down, each exactly
    ``spec.sample_count(program_seconds)`` samples of 48 kHz 24-bit PCM, with
    the leader's silence and the 2-pop in front of the programme."""
    if surround_start_seconds < 0:
        raise ValueError(
            f"surround_start_seconds must be non-negative, got {surround_start_seconds}"
        )
    sr = spec.sample_rate
    start = round(surround_start_seconds * sr)
    end = start + round(program_seconds * sr)
    total_samples = spec.sample_count(program_seconds)
    total = spec.total_seconds(program_seconds)
    delay_ms = round(spec.leader_seconds * 1000)
    amp = 10 ** (spec.pop_level_dbfs / 20)
    pop_expr = (
        f"{amp:g}*sin(2*PI*{spec.pop_frequency_hz}*t)"
        f"*between(t,{spec.pop_seconds:.1f},{spec.pop_seconds + 1 / spec.fps:.6f})"
    )
    exprs = "|".join([pop_expr] * 6)
    graph = ";".join([
        f"[0:a]atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS,"
        f"adelay=delays={delay_ms}:all=1,apad=whole_len={total_samples}[prog]",
        f"aevalsrc=exprs='{exprs}':s={sr}:c={FFMPEG_SURROUND_LAYOUT}:d={total:.6f}[pop]",
        "[prog][pop]amix=inputs=2:normalize=0:duration=first[mix]",
        "[mix]asplit=2[m1][m2]",
        f"[m1]channelsplit=channel_layout={FFMPEG_SURROUND_LAYOUT}"
        + "".join(f"[{label}]" for label in _FFMPEG_CHANNEL_LABELS),
        "[m2]pan=stereo|c0=0.6*FL+0.424*FC+0.424*SL|c1=0.6*FR+0.424*FC+0.424*SR[st]",
    ])
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(surround), "-filter_complex", graph]
    for name, label in zip(SURROUND_CHANNELS, _FFMPEG_CHANNEL_LABELS, strict=True):
        args += ["-map", f"[{label}]", "-c:a", "pcm_s24le", "-ar", str(sr),
                 str(out_dir / f"{name}.wav")]
    args += ["-map", "[st]", "-c:a", "pcm_s24le", "-ar", str(sr), str(out_dir / "stereo.wav")]
    return args


def build_preview_args(
    distribution: Path, stereo_wav: Path, out_mp4: Path, *,
    elevation_degrees: float = 30.0, h_fov_degrees: float = 100.0,
    width: int = 1920, height: int = 1080,
) -> list[str]:
    """A seat's-eye view: the domemaster reprojected to a perspective camera
    at the dome's centre looking at the front, ``elevation`` above the
    horizon. Not a dome simulator -- it proves the geometry reads the right
    way up and unmirrored, and carries the stereo fold-down so the 2-pop can
    be heard against the flash.

    The pitch sign is the mirror of the layers' (measured 2026-09-28): with the
    fisheye as the *input*, ``v360`` rotates the input sphere, so looking
    toward the frame's bottom -- the front -- is a **negative** pitch."""
    v_fov = flat_v_fov_degrees(width, height, h_fov_degrees)
    vf = (
        f"v360=input=fisheye:ih_fov=180:iv_fov=180:output=flat:h_fov={h_fov_degrees:.1f}"
        f":v_fov={v_fov:.2f}:w={width}:h={height}:pitch={elevation_degrees - 90:.1f}:interp=lanc"
    )
    return [
        "ffmpeg", "-v", "error", "-y", "-i", str(distribution), "-i", str(stereo_wav),
        "-vf", vf, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4),
    ]


def build_contact_sheet_args(
    pattern: Path, out_png: Path, *, frame_count: int, columns: int = 4, rows: int = 2,
    tile_px: int = 512,
) -> list[str]:
    step = max(1, frame_count // (columns * rows))
    vf = f"select='not(mod(n,{step}))',scale={tile_px}:{tile_px},tile={columns}x{rows}"
    return ["ffmpeg", "-v", "error", "-y", "-start_number", "0", "-i", str(pattern),
            "-vf", vf, "-frames:v", "1", str(out_png)]


# --------------------------------------------------------------------------- #
# Probes (runner-injected)
# --------------------------------------------------------------------------- #


def _probe_json(path: Path, runner: SubprocessRunner, *, stream: str, entries: str,
                format_entries: str | None = None) -> dict:
    show = f"stream={entries}" + (f":format={format_entries}" if format_entries else "")
    args = ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", show,
            "-of", "json", str(path)]
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError("ffprobe", args, result.returncode, result.stderr)
    return json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")


def probe_video(path: Path, runner: SubprocessRunner = _default_runner) -> dict:
    data = _probe_json(
        path, runner, stream="v:0",
        entries="codec_name,width,height,pix_fmt,r_frame_rate,nb_frames",
        format_entries="bit_rate",
    )
    stream = (data.get("streams") or [{}])[0]
    stream["bit_rate"] = (data.get("format") or {}).get("bit_rate")
    return stream


def probe_audio(path: Path, runner: SubprocessRunner = _default_runner) -> dict:
    data = _probe_json(
        path, runner, stream="a:0",
        entries="codec_name,channels,sample_rate,duration_ts,time_base",
    )
    return (data.get("streams") or [{}])[0]


def decode_frame(path: Path, pix_fmt: str, runner: SubprocessRunner) -> bytes:
    args = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError("decode", args, result.returncode, result.stderr)
    return result.stdout


def frame_mean_luma(path: Path, runner: SubprocessRunner) -> float:
    """The 32x18 grayscale probe ``luminance.py`` uses -- a downscale, not a
    full decode, so scanning a whole leader is cheap."""
    args = ["ffmpeg", "-v", "error", "-i", str(path), "-s", "32x18", "-f", "rawvideo",
            "-pix_fmt", "gray", "-"]
    result = runner(args)
    if result.returncode != 0 or not result.stdout:
        raise DomeCommandError("luma probe", args, result.returncode, result.stderr)
    return sum(result.stdout) / len(result.stdout)


def decode_mono_s16(path: Path, sample_rate: int, runner: SubprocessRunner) -> array:
    args = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
            "-ar", str(sample_rate), "-"]
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError("audio decode", args, result.returncode, result.stderr)
    samples = array("h")
    samples.frombytes(result.stdout[: len(result.stdout) // 2 * 2])
    if sys.byteorder != "little":  # pragma: no cover - s16le on a big-endian host
        samples.byteswap()
    return samples


def outside_circle_nonzero_bytes(frame: bytes, size: int, bit_depth: int) -> int:
    """How many bytes of the raw RGB frame are non-zero outside the disc.
    Row slices are C-speed, so a 4096x4096 16-bit frame (96 MB) takes well
    under a second with no numpy."""
    bpp = 3 * (2 if bit_depth == 16 else 1)
    row_len = size * bpp
    if len(frame) != size * row_len:
        raise ValueError(f"frame is {len(frame)} bytes, expected {size * row_len}")
    bad = 0
    for y in range(size):
        row = frame[y * row_len:(y + 1) * row_len]
        span = inside_span(y, size)
        if span is None:
            outside = (row,)
        else:
            x0, x1 = span
            outside = (row[: x0 * bpp], row[x1 * bpp:])
        for chunk in outside:
            bad += len(chunk) - chunk.count(b"\x00")
    return bad


def disc_mean_luma(frame: bytes, size: int) -> float:
    """Mean of a ``size``x``size`` gray frame over the disc only -- the black
    corners are 21% of the square and would flatter every reading."""
    total = count = 0
    for y in range(size):
        span = inside_span(y, size)
        if span is None:
            continue
        x0, x1 = span
        total += sum(frame[y * size + x0:y * size + x1])
        count += x1 - x0
    return total / count if count else 0.0


def programme_luma_series(distribution: Path, spec: DomeSpec,
                          runner: SubprocessRunner) -> list[float]:
    """Every frame's disc-mean luma from the distribution copy, in one ffmpeg
    pass (decoding 1890 16-bit PNGs one by one would take a quarter of an hour).
    Limited-range video luma is expanded to full range, so black reads 0."""
    n = LUMA_SCAN_SIZE
    args = ["ffmpeg", "-v", "error", "-i", str(distribution), "-vf",
            f"scale={n}:{n}:flags=area:in_range=tv:out_range=pc,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError("luma scan", args, result.returncode, result.stderr)
    data = result.stdout
    return [disc_mean_luma(data[i:i + n * n], n) for i in range(0, len(data) - n * n + 1, n * n)]


def max_flashes_per_second(series: Sequence[float], fps: int,
                           delta: float = FLASH_DELTA_Y) -> int:
    """The most flashes in any one-second window. A transition is a swing of
    at least ``delta`` away from the last extreme (a zig-zag with hysteresis,
    so slow drift and sub-threshold flicker count for nothing); a flash is a
    pair of opposing transitions."""
    if not series:
        return 0
    times: list[int] = []
    low = high = series[0]
    direction = 0  # 0 until the first transition, then +1 rising / -1 falling
    for i, value in enumerate(series[1:], start=1):
        low, high = min(low, value), max(high, value)
        if direction >= 0 and high - value >= delta:
            times.append(i)
            direction, low, high = -1, value, value
        elif direction <= 0 and value - low >= delta:
            times.append(i)
            direction, low, high = 1, value, value
    best = 0
    lo = 0
    for hi, t in enumerate(times):
        while t - times[lo] >= fps:
            lo += 1
        best = max(best, (hi - lo + 1) // 2)
    return best


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    checks: tuple[CheckResult, ...]
    sync_offset_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sync_offset_seconds": self.sync_offset_seconds,
            "checks": [asdict(c) for c in self.checks],
        }


def _sample_indices(frame_count: int, spec: DomeSpec, sample_frames: int) -> list[int]:
    wanted = {0, spec.pop_frame, spec.program_start_frame, frame_count // 2, frame_count - 1}
    if sample_frames > 0:
        step = max(1, frame_count // sample_frames)
        wanted.update(range(0, frame_count, step))
    return sorted(i for i in wanted if 0 <= i < frame_count)


def verify_render(
    *,
    frames_dir: Path,
    audio_dir: Path,
    distribution: Path,
    spec: DomeSpec,
    program_seconds: float,
    runner: SubprocessRunner = _default_runner,
    sample_frames: int = 12,
) -> VerificationReport:
    """The spec's automated checks: frame size, circle mask (pure black
    outside), frame count against duration, frame rate, audio channel count
    and length, and the 2-pop's sync offset -- plus section 5's comfort
    rules as far as a number can carry them: no very bright full-dome frame
    and no full-field flashing, both read over the programme only (the
    leader's flash is the 2-pop, on purpose). Never raises on a *finding* --
    every failed check is a row, so the report is complete rather than
    truncated at the first defect."""
    checks: list[CheckResult] = []
    expected_frames = spec.frame_count(program_seconds)
    frames = sorted(frames_dir.glob("dome_*.png")) if frames_dir.is_dir() else []
    checks.append(CheckResult(
        "frame_count", len(frames) == expected_frames,
        f"{len(frames)} frames in {frames_dir}, expected {expected_frames} "
        f"({spec.total_seconds(program_seconds):.3f} s x {spec.fps} fps)",
    ))

    sync_offset: float | None = None
    if frames:
        first = probe_video(frames[0], runner)
        checks.append(CheckResult(
            "frame_size",
            first.get("width") == spec.size and first.get("height") == spec.size,
            f"{first.get('width')}x{first.get('height')}, expected {spec.size}x{spec.size}",
        ))
        checks.append(CheckResult(
            "bit_depth", first.get("pix_fmt") == spec.png_pix_fmt,
            f"pix_fmt {first.get('pix_fmt')}, expected {spec.png_pix_fmt}",
        ))

        raw_fmt = "rgb48le" if spec.bit_depth == 16 else "rgb24"
        bad_frames: list[str] = []
        indices = _sample_indices(len(frames), spec, sample_frames)
        for i in indices:
            frame = frames[i]
            try:
                bad = outside_circle_nonzero_bytes(
                    decode_frame(frame, raw_fmt, runner), spec.size, spec.bit_depth
                )
            except (DomeCommandError, ValueError) as exc:  # pragma: no cover - defensive
                bad_frames.append(f"{frame.name}: {exc}")
                continue
            if bad:
                bad_frames.append(f"{frame.name}: {bad} non-zero bytes outside the circle")
        checks.append(CheckResult(
            "circle_mask", not bad_frames,
            f"sampled {len(indices)} frames; " + ("all pure black outside the circle"
                                                  if not bad_frames else "; ".join(bad_frames)),
        ))

        # -- the flash frame -------------------------------------------------- #
        leader_end = min(spec.program_start_frame, len(frames))
        lumas = [(frame_mean_luma(frames[i], runner), i) for i in range(leader_end)]
        flash_at: int | None = None
        flash_detail = "no leader frames"
        if lumas:
            brightest, flash_at = max(lumas)
            others = [value for value, i in lumas if i != flash_at]
            flash_ok = (
                flash_at == spec.pop_frame and brightest >= LUMA_FLASH_FLOOR
                and all(v <= LUMA_DARK_CEILING for v in others)
            )
            flash_detail = (
                f"flash frame found at frame {flash_at} (mean Y {brightest:.1f}), "
                f"expected frame {spec.pop_frame}"
            )
            if not flash_ok:
                flash_at = None
    else:
        flash_at = None
        flash_detail = "no frames to scan"

    # -- distribution copy ---------------------------------------------------- #
    if distribution.exists():
        dist = probe_video(distribution, runner)
        nb = dist.get("nb_frames")
        dist_ok = (
            dist.get("codec_name") == "hevc"
            and dist.get("width") == spec.size and dist.get("height") == spec.size
            and dist.get("r_frame_rate") == f"{spec.fps}/1"
            and str(nb) == str(expected_frames)
        )
        checks.append(CheckResult(
            "distribution", dist_ok,
            f"{dist.get('codec_name')} {dist.get('width')}x{dist.get('height')} "
            f"@ {dist.get('r_frame_rate')} fps, {nb} frames, bit_rate {dist.get('bit_rate')}; "
            f"expected hevc {spec.size}x{spec.size} @ {spec.fps}/1, {expected_frames} frames",
        ))
        start = spec.program_start_frame
        programme = programme_luma_series(distribution, spec, runner)[start:]
        if programme:
            peak = max(programme)
            peak_at = start + programme.index(peak)
            checks.append(CheckResult(
                "brightness", peak <= BRIGHT_FIELD_CEILING_Y,
                f"brightest programme frame {peak_at}: disc mean Y {peak:.1f} "
                f"(ceiling {BRIGHT_FIELD_CEILING_Y:.0f}); programme mean "
                f"{sum(programme) / len(programme):.1f}",
            ))
            flashes = max_flashes_per_second(programme, spec.fps)
            checks.append(CheckResult(
                "flashes", flashes <= MAX_FLASHES_PER_SECOND,
                f"at most {flashes} full-field flash(es) in any second "
                f"(swing >= {FLASH_DELTA_Y:.0f} Y; limit {MAX_FLASHES_PER_SECOND})",
            ))
        else:
            checks.append(CheckResult("brightness", False, "no programme frames decoded"))
            checks.append(CheckResult("flashes", False, "no programme frames decoded"))
    else:
        checks.append(CheckResult("distribution", False, f"{distribution} does not exist"))

    # -- audio ------------------------------------------------------------------ #
    expected_samples = spec.sample_count(program_seconds)
    stems = [audio_dir / f"{name}.wav" for name in SURROUND_CHANNELS]
    stereo = audio_dir / "stereo.wav"
    missing = [p.name for p in stems + [stereo] if not p.exists()]
    if missing:
        checks.append(CheckResult("audio_files", False, f"missing: {', '.join(missing)}"))
        checks.append(CheckResult("audio_length", False, "not measured: files missing"))
        audio_pop_seconds: float | None = None
    else:
        layout_problems: list[str] = []
        length_problems: list[str] = []
        for path, want_channels in [(p, 1) for p in stems] + [(stereo, 2)]:
            info = probe_audio(path, runner)
            if (int(info.get("channels", 0)) != want_channels
                    or str(info.get("sample_rate")) != str(spec.sample_rate)
                    or info.get("codec_name") != "pcm_s24le"):
                layout_problems.append(
                    f"{path.name}: {info.get('codec_name')} {info.get('channels')}ch "
                    f"@ {info.get('sample_rate')} Hz"
                )
            if str(info.get("duration_ts")) != str(expected_samples):
                length_problems.append(
                    f"{path.name}: {info.get('duration_ts')} samples"
                )
        checks.append(CheckResult(
            "audio_files", not layout_problems,
            "6 mono + stereo, pcm_s24le @ 48 kHz" if not layout_problems
            else "; ".join(layout_problems),
        ))
        checks.append(CheckResult(
            "audio_length", not length_problems,
            f"all {expected_samples} samples ({spec.total_seconds(program_seconds):.3f} s)"
            if not length_problems
            else "; ".join(length_problems) + f", expected {expected_samples}",
        ))
        samples = decode_mono_s16(stems[0], spec.sample_rate, runner)
        onset = next((i for i, v in enumerate(samples) if abs(v) > POP_ONSET_THRESHOLD), None)
        audio_pop_seconds = None if onset is None else onset / spec.sample_rate

    # -- sync ------------------------------------------------------------------- #
    if flash_at is None or audio_pop_seconds is None:
        checks.append(CheckResult(
            "sync", False,
            f"{flash_detail}; audio tone onset "
            + ("not found" if audio_pop_seconds is None else f"at {audio_pop_seconds:.6f} s"),
        ))
    else:
        video_pop_seconds = flash_at / spec.fps
        sync_offset = audio_pop_seconds - video_pop_seconds
        tolerance = SYNC_TOLERANCE_FRAMES / spec.fps
        checks.append(CheckResult(
            "sync", abs(sync_offset) <= tolerance,
            f"audio tone {sync_offset * 1000:+.3f} ms relative to the flash frame "
            f"(frame {flash_at}, tone onset {audio_pop_seconds:.6f} s); "
            f"tolerance +/-{tolerance * 1000:.3f} ms",
        ))
    return VerificationReport(checks=tuple(checks), sync_offset_seconds=sync_offset)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenderResult:
    out_dir: Path
    frames_dir: Path
    distribution: Path
    audio_dir: Path
    preview: Path | None
    contact_sheet: Path | None
    manifest: Path
    verification: VerificationReport
    commands: tuple[dict, ...] = field(default_factory=tuple)


def _describe(args: Sequence[str]) -> str:
    """Name the stage an argv belongs to -- for logs and the manifest."""
    joined = " ".join(args)
    if args and args[0] == "ffprobe":
        return "ffprobe"
    if "v360=input=equirect" in joined and "-filter_complex" not in args:
        return "background"
    if "geq=lum='if(lte(hypot" in joined:
        return "disc"
    if "libx265" in joined:
        return "distribution"
    if "channelsplit" in joined:
        return "audio"
    if "output=flat" in joined:
        return "preview"
    if "tile=" in joined:
        return "contact_sheet"
    if "-filter_complex" in args:
        return "master"
    return "other"


def _run(stage: str, args: Sequence[str], runner: SubprocessRunner) -> None:
    logger.info("dome %s: %s", stage, " ".join(args))
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError(stage, args, result.returncode, result.stderr)


def probe_base(path: Path, runner: SubprocessRunner = _default_runner) -> dict:
    """Size, rate and packet count of a base video. Packets are counted
    without decoding, so a 60 s 4096-pixel FFV1 file answers in seconds, and
    it works on containers (Matroska) that carry no ``nb_frames``."""
    args = ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_packets",
            "-of", "json", str(path)]
    result = runner(args)
    if result.returncode != 0:
        raise DomeCommandError("ffprobe", args, result.returncode, result.stderr)
    data = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    return (data.get("streams") or [{}])[0]


def _check_base(info: dict, spec: DomeSpec, program_seconds: float, path: Path) -> None:
    """A base is refused rather than scaled, retimed or padded: the point of
    rendering one natively is that nothing resamples it."""
    if info.get("width") != spec.size or info.get("height") != spec.size:
        raise ValueError(f"base {path} is {info.get('width')}x{info.get('height')}; it must be "
                         f"rendered at the dome's size, {spec.size}x{spec.size}")
    if info.get("r_frame_rate") != f"{spec.fps}/1":
        raise ValueError(f"base {path} runs at {info.get('r_frame_rate')}; the frame rate "
                         f"must be {spec.fps}/1")
    need = round(program_seconds * spec.fps)
    have = int(info.get("nb_read_packets") or 0)
    if have < need:
        raise ValueError(f"base {path} has {have} frames; the programme needs {need}")


def render(
    *,
    source: Path | None,
    source_start: float,
    program_seconds: float,
    surround: Path,
    surround_offset_seconds: float,
    out_dir: Path,
    spec: DomeSpec,
    window: WindowPlacement | None,
    credits_png: Path | None,
    credits: CreditsLayer | None,
    retime: str,
    runner: SubprocessRunner = _default_runner,
    threads: int | None = None,
    sample_frames: int = 12,
    base: Path | None = None,
    foreground: Path | None = None,
) -> RenderResult:
    """Background, disc, master sequence, distribution copy, audio stems,
    verification, seat preview, contact sheet, manifest -- in that order.
    ``surround_offset_seconds`` is surround-file time minus stereo-master
    time for the same musical moment (see the module docstring)."""
    if retime not in RETIME_MODES:
        raise ValueError(f"retime must be one of {RETIME_MODES}, got {retime!r}")
    surround_start = source_start + surround_offset_seconds
    if surround_start < 0:
        raise ValueError(
            f"source_start={source_start} s lies {-surround_start:.3f} s before the surround "
            f"mix begins (surround_offset_seconds={surround_offset_seconds}); start later"
        )
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    audio_dir = out_dir / "audio"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict] = []

    def run(stage: str, args: list[str]) -> None:
        commands.append({"stage": stage, "argv": list(args)})
        _run(stage, args, runner)

    if source is None and base is None:
        raise ValueError("nothing to show: give a source or a base (or both)")
    if source is not None and window is None:
        window = WindowPlacement()
    source_width = source_height = 0
    if source is not None:
        src = probe_video(source, runner)
        source_width, source_height = int(src["width"]), int(src["height"])
        logger.info("dome source %s is %dx%d (%s)", source, source_width, source_height,
                    src.get("codec_name"))
    else:
        window = None
    for layer in (base, foreground):
        if layer is not None:
            _check_base(probe_base(layer, runner), spec, program_seconds, layer)

    background_png: Path | None = None
    disc_png = out_dir / "disc.png"
    if base is None:
        background_png = out_dir / "background.png"
        run("background", build_background_args(
            background_png, size=spec.size, front_rotation_degrees=spec.front_rotation_degrees))
    run("disc", build_disc_mask_args(disc_png, size=spec.size))

    pattern = frames_dir / FRAME_PATTERN
    run("master", build_master_args(
        source=source, background_png=background_png, disc_png=disc_png,
        credits_png=credits_png, out_pattern=pattern, spec=spec, window=window,
        source_width=source_width, source_height=source_height, source_start=source_start,
        program_seconds=program_seconds, retime=retime, credits=credits, threads=threads,
        base=base, foreground=foreground,
    ))
    distribution = out_dir / DISTRIBUTION_FILENAME
    run("distribution", build_distribution_args(pattern, distribution, fps=spec.fps,
                                                 threads=threads))
    run("audio", build_audio_args(
        surround=surround, surround_start_seconds=surround_start,
        program_seconds=program_seconds, spec=spec, out_dir=audio_dir,
    ))

    verification = verify_render(
        frames_dir=frames_dir, audio_dir=audio_dir, distribution=distribution, spec=spec,
        program_seconds=program_seconds, runner=runner, sample_frames=sample_frames,
    )
    for check in verification.checks:
        logger.log(logging.INFO if check.ok else logging.ERROR, "dome check %s: %s -- %s",
                   check.name, "ok" if check.ok else "FAIL", check.detail)
    (out_dir / VERIFICATION_FILENAME).write_text(
        json.dumps(verification.to_dict(), indent=2), encoding="utf-8"
    )

    preview = out_dir / "preview_seat_view.mp4"
    run("preview", build_preview_args(distribution, audio_dir / "stereo.wav", preview))
    sheet = out_dir / "contact_sheet.png"
    run("contact_sheet", build_contact_sheet_args(
        pattern, sheet, frame_count=spec.frame_count(program_seconds)))

    manifest = out_dir / MANIFEST_FILENAME
    manifest.write_text(json.dumps({
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": None if source is None else str(source),
        "base": None if base is None else str(base),
        "foreground": None if foreground is None else str(foreground),
        "source_start_seconds": source_start,
        "program_seconds": program_seconds, "surround": str(surround),
        "surround_offset_seconds": surround_offset_seconds,
        "surround_start_seconds": surround_start,
        "spec": asdict(spec), "window": None if window is None else asdict(window),
        "credits": None if credits is None else {**asdict(credits), "image": str(credits_png)},
        "retime": retime, "frame_pattern": str(pattern), "distribution": str(distribution),
        "audio_dir": str(audio_dir), "verification": verification.to_dict(),
        "commands": commands,
    }, indent=2), encoding="utf-8")
    return RenderResult(
        out_dir=out_dir, frames_dir=frames_dir, distribution=distribution, audio_dir=audio_dir,
        preview=preview, contact_sheet=sheet, manifest=manifest, verification=verification,
        commands=tuple(commands),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=DOMEMASTER_SIZE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--leader", type=float, default=3.0, help="leader seconds before programme")
    parser.add_argument("--pop", type=float, default=1.0, help="2-pop position in the leader (s)")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=16)
    parser.add_argument("--front-rotation", type=float, default=0.0,
                        help="venue orientation, degrees about the zenith")


def _spec_from(ns: argparse.Namespace) -> DomeSpec:
    return DomeSpec(size=ns.size, fps=ns.fps, front_rotation_degrees=ns.front_rotation,
                    leader_seconds=ns.leader, pop_seconds=ns.pop, bit_depth=ns.bit_depth)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m music_video_maker.dome")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render a domemaster test trailer from a flat clip")
    r.add_argument("--source", type=Path, default=None,
                   help="flat clip for the window/inset; optional when --base is given")
    r.add_argument("--base", type=Path, default=None,
                   help="fisheye video at the dome's size and rate; replaces the starfield")
    r.add_argument("--foreground", type=Path, default=None,
                   help="fisheye video with alpha, same contract, laid over the window")
    r.add_argument("--source-start", type=float, default=0.0)
    r.add_argument("--program-seconds", type=float, required=True)
    r.add_argument("--surround", type=Path, required=True, help="the 5.1 master (6 channels)")
    r.add_argument("--surround-offset", type=float, default=0.0,
                   help="surround-file time minus stereo-master time, seconds")
    r.add_argument("--out-dir", type=Path, required=True)
    _add_spec_args(r)
    r.add_argument("--h-fov", type=float, default=90.0)
    r.add_argument("--elevation", type=float, default=40.0)
    r.add_argument("--azimuth", type=float, default=0.0)
    r.add_argument("--feather", type=int, default=24)
    r.add_argument("--window-opacity", type=float, default=1.0)
    r.add_argument("--source-alpha", action="store_true",
                   help="keep the source clip's own alpha (a matted cut-out)")
    r.add_argument("--retime", choices=RETIME_MODES, default="mci")
    r.add_argument("--credits", type=Path, default=None, help="RGBA card PNG")
    r.add_argument("--credits-width", type=int, default=1920)
    r.add_argument("--credits-height", type=int, default=1080)
    r.add_argument("--credits-at", type=float, default=52.0)
    r.add_argument("--credits-fade", type=float, default=1.0)
    r.add_argument("--credits-h-fov", type=float, default=70.0)
    r.add_argument("--credits-elevation", type=float, default=35.0)
    r.add_argument("--threads", type=int, default=None)
    r.add_argument("--sample-frames", type=int, default=12)

    v = sub.add_parser("verify", help="re-run the conformance checks on an output directory")
    v.add_argument("--out-dir", type=Path, required=True)
    v.add_argument("--program-seconds", type=float, required=True)
    _add_spec_args(v)
    v.add_argument("--sample-frames", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ns = build_parser().parse_args(argv)
    spec = _spec_from(ns)
    if ns.command == "verify":
        report = verify_render(
            frames_dir=ns.out_dir / "frames", audio_dir=ns.out_dir / "audio",
            distribution=ns.out_dir / DISTRIBUTION_FILENAME, spec=spec,
            program_seconds=ns.program_seconds, sample_frames=ns.sample_frames,
        )
        for check in report.checks:
            logger.log(logging.INFO if check.ok else logging.ERROR, "dome check %s: %s -- %s",
                       check.name, "ok" if check.ok else "FAIL", check.detail)
        return 0 if report.ok else 1

    credits = None
    if ns.credits is not None:
        credits = CreditsLayer(
            image_width=ns.credits_width, image_height=ns.credits_height,
            placement=WindowPlacement(h_fov_degrees=ns.credits_h_fov,
                                      elevation_degrees=ns.credits_elevation, feather_px=0),
            at_seconds=ns.credits_at, fade_seconds=ns.credits_fade,
        )
    result = render(
        source=ns.source, source_start=ns.source_start, program_seconds=ns.program_seconds,
        surround=ns.surround, surround_offset_seconds=ns.surround_offset, out_dir=ns.out_dir,
        spec=spec,
        window=None if ns.source is None else WindowPlacement(
            h_fov_degrees=ns.h_fov, elevation_degrees=ns.elevation,
            azimuth_degrees=ns.azimuth, feather_px=ns.feather, opacity=ns.window_opacity,
            source_alpha=ns.source_alpha),
        credits_png=ns.credits, credits=credits, retime=ns.retime, threads=ns.threads,
        sample_frames=ns.sample_frames, base=ns.base, foreground=ns.foreground,
    )
    logger.info("dome render %s: %s", "VERIFIED" if result.verification.ok else "FAILED checks",
                result.manifest)
    return 0 if result.verification.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
