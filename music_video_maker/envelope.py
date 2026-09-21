"""The proven render envelope: refuse to submit a chunk bigger than anything
ever measured on this card (issues #24, #98).

Issue #24's own conclusion is that a failure blocking *inside a driver call*
cannot be recovered by any application-level watchdog -- no retry, no
``POST /interrupt``, no ``POST /free`` reaches a process stuck below the
Python runtime -- so **the only defence is refusing to enter the
over-committed state at all**. The pipeline already refuses on one axis:
``custody``'s pre-flight floor and ``resilience``'s between-chunk re-check
both ask "is there enough free VRAM right now?" before submitting. Both are
about *what else is on the card*. Neither asks the other question -- **how
big is the thing we are about to submit** -- and that is the axis #98 is
about: nothing longer than 192 frames has ever rendered on doris, and
``max_chunk_seconds = 12.0`` would submit 277.

So this module holds the *evidence*, and the gate reads it:

* :data:`PROVEN_ENVELOPES` is a committed table of frame-count/resolution
  points that have demonstrably rendered on a named card, each with the
  measurement it comes from.
* :func:`check_render_envelope` refuses a run whose chunks fall outside it,
  naming the offending chunks and the remedy, unless the operator sets
  ``acknowledge_unproven_envelope``.
* ``docs/runbook-288-frame-proof.md`` is the sanctioned way to *extend* the
  table: an attended one-chunk proof, and its result becomes a new
  :class:`EnvelopePoint` here. That is the whole loop -- the gate is not a
  ceiling, it is a requirement that the ceiling be measured before it is
  spent hours of GPU time on.

**Dominance is per axis, never a product.** It is tempting to compare latent
*volume* (frames x width x height) and call a smaller number safe. Do not:
141 frames at 1344x768 is 145.5 Mpx of latent against 192 frames at 864x480's
79.6 Mpx, so a volume rule would declare the 192-frame point already proven by
the 1344x768 one -- and the thing that is actually unmeasured above 192 frames
is *temporal VAE decode memory, which scales non-linearly with frame count*.
A point proves another point only when it is at least as large on **every**
axis. The volume appears in the log message, as context for a human, and
never in the decision.

**No evidence means no gate, not a refusal.** The table is keyed by
:attr:`~music_video_maker.contracts.HardwareProfile.name`, and a profile with
no entry is *unmeasured*, not *forbidden*: refusing a stranger's H100 on the
strength of measurements taken on doris's 4090 would be inventing a
measurement, which is the failure this project spends most of its
``CLAUDE.md`` avoiding. Such a run is logged once at INFO and proceeds
exactly as it did before this module existed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_video_maker.contracts import ChunkStatus
from music_video_maker.shot_plan import MEASURED_MAX_FRAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvelopePoint:
    """One (frames, width, height) combination that has actually rendered.

    ``source`` is the measurement it comes from, in enough detail that a
    reader can go and check it. An entry with no traceable source is a
    remembered number, which is exactly what ``MEASURED_MAX_FRAMES`` turned
    into (issue #98).
    """

    frames: int
    width: int
    height: int
    source: str

    @property
    def latent_megapixels(self) -> float:
        """Frames x pixels, in millions. **Context for a human only** -- it is
        deliberately not what :meth:`covers` compares. See the module
        docstring."""
        return self.frames * self.width * self.height / 1_000_000

    def covers(self, frames: int, width: int, height: int) -> bool:
        """True when this point is at least as large on *every* axis."""
        return frames <= self.frames and width <= self.width and height <= self.height

    def describe(self) -> str:
        return (
            f"{self.frames} frames at {self.width}x{self.height} "
            f"({self.latent_megapixels:.1f} Mpx of latent; {self.source})"
        )


DORIS_4090_PROVEN = (
    EnvelopePoint(
        frames=192,
        width=864,
        height=480,
        source=(
            "the finished 'Deathless' v13 render -- 45 of 80 chunks above 141 frames, 15 of "
            "them at 192, confirmed with `ffprobe -count_frames` against the mp4s themselves "
            "(CLAUDE.md, 2026-09-16)"
        ),
    ),
    EnvelopePoint(
        frames=141,
        width=1344,
        height=768,
        source=(
            "CLAUDE.md's performance measurement: 141 frames at 1344x768 = 9m15s per chunk "
            "on this 4090"
        ),
    ),
)
"""Everything doris's RTX 4090 has been *shown* to render.

Two points, neither of which proves the other: the 864x480 one is the longest
chunk ever rendered here, the 1344x768 one is the largest frame ever rendered
here, and nothing has been measured at both at once. Add a third only from a
render that happened -- ``docs/runbook-288-frame-proof.md`` is the procedure,
and its result belongs here with its date."""

PROVEN_ENVELOPES: dict[str, tuple[EnvelopePoint, ...]] = {
    "RTX 4090 24GB (doris)": DORIS_4090_PROVEN,
}
"""Proven points per :attr:`HardwareProfile.name`. A name that is not a key
here has no recorded evidence and therefore no gate -- see the module
docstring for why that is not a refusal."""


class UnprovenEnvelopeError(RuntimeError):
    """A chunk about to be submitted is larger than anything proven on this
    card, and the operator has not acknowledged that.

    Raised *before* any GPU work, like
    :class:`~music_video_maker.custody.CustodyError` and
    :class:`~music_video_maker.resilience.VramBelowFloorError` -- the three
    are the same defence from three directions, and this is the only one that
    looks at the size of the thing being submitted rather than at what else
    holds the card."""


@dataclass(frozen=True)
class EnvelopeMiss:
    """One chunk that no proven point covers."""

    chunk_id: int
    frames: int
    width: int
    height: int

    def describe(self) -> str:
        return f"chunk {self.chunk_id} at {self.frames} frames ({self.width}x{self.height})"


def covering_point(
    points: Sequence[EnvelopePoint], frames: int, width: int, height: int
) -> EnvelopePoint | None:
    """The first proven point that covers ``(frames, width, height)``, or
    ``None`` when nothing does."""
    for point in points:
        if point.covers(frames, width, height):
            return point
    return None


def largest_proven_frames(points: Sequence[EnvelopePoint]) -> int | None:
    """The largest *frame count* any proven point reaches, at any resolution.

    This is the honest replacement for a remembered constant in a warning
    message. It says nothing about whether that frame count is available at
    *this* run's resolution -- :func:`covering_point` is what answers that."""
    if not points:
        return None
    return max(point.frames for point in points)


def envelope_for(hardware_name: str) -> tuple[EnvelopePoint, ...]:
    """Proven points for a hardware profile name; ``()`` when unrecorded."""
    return PROVEN_ENVELOPES.get(hardware_name, ())


# --------------------------------------------------------------------------- #
# Evidence from a finished run (issue #98's "make the number evidence")
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenderedExtreme:
    """The largest frame count a ``run_state.json`` can prove was rendered."""

    frames: int
    chunk_id: int
    source: Path

    def describe(self) -> str:
        return (
            f"{self.frames} frames (chunk {self.chunk_id} of {self.source}, which that run "
            "actually rendered)"
        )


@dataclass(frozen=True)
class MeasuredCeiling:
    """The frame count a "nothing longer has rendered here" warning may name,
    and **where that number comes from** (issue #98).

    The two fields exist together because separating them is the bug. The
    warning in ``slicing`` read "exceed 141 frames -- the longest anything
    ever rendered on this card" for months after the v13 render put 45 of 80
    chunks above 141 and 15 of them at 192: the number was a constant nobody
    updates, and the sentence around it claimed it was a measurement. A
    ceiling that has to carry its own provenance cannot make that claim by
    accident.
    """

    frames: int
    provenance: str

    def describe(self) -> str:
        return f"{self.frames} frames ({self.provenance})"


CALIBRATED_CEILING = MeasuredCeiling(
    frames=MEASURED_MAX_FRAMES,
    provenance=(
        "the largest frame count that had been measured when shot_plan.MEASURED_MAX_FRAMES was "
        "written -- a calibrated constant, not a live reading: nothing updates it when a render "
        "goes past, and on this card renders have (issue #98)"
    ),
)
"""What the warning falls back to when no run state is available to ask.

Deliberately not a claim about what has rendered. Raising
``MEASURED_MAX_FRAMES`` is a decision about what warning an operator wants,
not a correction -- so the constant stays where it is and the *sentence*
stops overstating it."""


def measured_ceiling(
    run_state_file: Path | str | None,
    *,
    calibrated: MeasuredCeiling = CALIBRATED_CEILING,
) -> MeasuredCeiling:
    """The best-evidenced frame ceiling available: a previous run's own
    longest rendered chunk when one can be read, the calibrated constant
    otherwise.

    Takes the *larger* of the two and names whichever won, so the number in
    the log is always the strongest available evidence and always says where
    it came from. A run state that only reaches shorter chunks does not lower
    the ceiling -- it is one run, not the history of the card.
    """
    rendered = largest_rendered_frame_count(run_state_file)
    if rendered is None or rendered.frames <= calibrated.frames:
        return calibrated
    return MeasuredCeiling(
        frames=rendered.frames,
        provenance=(
            f"the longest chunk {rendered.source} records as actually rendered -- chunk "
            f"{rendered.chunk_id}"
        ),
    )


def largest_rendered_frame_count(run_state_file: Path | str | None) -> RenderedExtreme | None:
    """Read back the longest chunk a previous run *actually rendered*.

    ``None`` whenever the question cannot be answered from evidence: no path,
    no file, an unreadable file, or a file whose results carry no frame counts
    (a v1 state file, written before fingerprints existed). A ``None`` here
    must never be read as "nothing long has rendered" -- it is "this did not
    ask anything that could answer", the same distinction
    ``faces.FaceObservation`` draws between *absent* and *unexamined*.

    Only ``RENDERED`` and ``CACHED`` results count. A dead-lettered chunk is
    the opposite of evidence: it is a frame count that was attempted and did
    not produce a video.
    """
    if run_state_file is None:
        return None
    path = Path(run_state_file)
    if not path.exists():
        return None

    # The render's own loader, never a second copy of it -- a state file's
    # schema rules (a v1 file is refused, not misread) live in exactly one
    # place. Imported here rather than at module scope to keep this module
    # importable from anywhere without dragging the execution stack in.
    from music_video_maker.resilience import load_run_state

    try:
        run_state = load_run_state(path)
    except Exception:  # noqa: BLE001 -- evidence gathering must never fail a run
        logger.warning(
            "Could not read %s to find the longest chunk a previous run rendered -- falling "
            "back to the calibrated constant. This is 'not measured', not 'nothing long has "
            "rendered'.",
            path,
            exc_info=True,
        )
        return None

    best: RenderedExtreme | None = None
    for chunk_id, result in sorted(run_state.results.items()):
        if result.status not in (ChunkStatus.RENDERED, ChunkStatus.CACHED):
            continue
        fingerprint = result.fingerprint
        frames = getattr(fingerprint, "frame_count", None) if fingerprint is not None else None
        if not isinstance(frames, int):
            continue
        if best is None or frames > best.frames:
            best = RenderedExtreme(frames=frames, chunk_id=chunk_id, source=path)
    return best


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def _chunk_frames(chunk: Any) -> int | None:
    frames = getattr(chunk, "frame_count", None)
    return frames if isinstance(frames, int) else None


def unproven_chunks(
    chunks: Iterable[Any],
    *,
    width: int,
    height: int,
    points: Sequence[EnvelopePoint],
) -> tuple[EnvelopeMiss, ...]:
    """Every chunk in ``chunks`` that no point in ``points`` covers.

    A chunk with no frame count is skipped rather than flagged: slicing
    guarantees one on everything it emits, and a caller passing something
    else is not evidence of a large chunk.
    """
    misses: list[EnvelopeMiss] = []
    for chunk in chunks:
        frames = _chunk_frames(chunk)
        if frames is None:
            continue
        if covering_point(points, frames, width, height) is None:
            misses.append(
                EnvelopeMiss(
                    chunk_id=getattr(chunk, "chunk_id", -1),
                    frames=frames,
                    width=width,
                    height=height,
                )
            )
    return tuple(misses)


def check_render_envelope(
    chunks: Sequence[Any],
    *,
    hardware_name: str,
    width: int | None,
    height: int | None,
    acknowledged: bool = False,
    runbook: str = "docs/runbook-288-frame-proof.md",
) -> tuple[EnvelopeMiss, ...]:
    """Refuse a run that would submit a chunk bigger than anything proven.

    Returns the misses it found (empty when everything is inside the
    envelope), so a caller can report them; raises
    :class:`UnprovenEnvelopeError` when there are misses and ``acknowledged``
    is false.

    Three ways this says nothing at all, each of them deliberate:

    * **No recorded envelope for ``hardware_name``.** Unmeasured is not
      forbidden -- see the module docstring.
    * **No resolved ``width``/``height``.** A run whose resolution could not
      be determined cannot be judged on it, and a gate that guesses is worse
      than no gate. Logged at WARNING, because it means something upstream
      failed to resolve the template's own dimensions.
    * **No chunks with frame counts.** Nothing to judge.
    """
    points = envelope_for(hardware_name)
    if not points:
        logger.info(
            "Render-envelope check: no proven frame-count/resolution measurements are recorded "
            "for hardware profile %r, so there is nothing to check this run against. That is "
            "'unmeasured', not 'safe' -- on a card whose limits are unknown, watch the first "
            "long chunk (issues #24, #98).",
            hardware_name,
        )
        return ()

    if width is None or height is None:
        logger.warning(
            "Render-envelope check skipped for hardware profile %r: this run's resolution could "
            "not be resolved (width=%r height=%r), and a size gate that guesses the size is "
            "worse than no gate.",
            hardware_name,
            width,
            height,
        )
        return ()

    misses = unproven_chunks(chunks, width=width, height=height, points=points)
    proven = "; ".join(point.describe() for point in points)
    if not misses:
        logger.info(
            "Render-envelope check OK: all %d chunk(s) at %dx%d are inside what this card has "
            "been shown to render (%s).",
            len(chunks),
            width,
            height,
            proven,
        )
        return ()

    largest = max(misses, key=lambda miss: miss.frames)
    shown = ", ".join(miss.describe() for miss in misses[:5])
    if len(misses) > 5:
        shown += f", ... ({len(misses) - 5} more)"
    detail = (
        f"{len(misses)} of {len(chunks)} chunk(s) are larger than anything ever rendered on "
        f"{hardware_name!r}: {shown}. The largest is {largest.frames} frames at "
        f"{width}x{height}. Proven here: {proven}. Nothing interpolates between those points "
        f"-- temporal VAE decode memory scales non-linearly with frame count, so a smaller "
        f"latent volume at a longer duration is not covered by a larger one at a shorter "
        f"duration. An over-committed card on this host does not raise CUDA OOM; it goes "
        f"silent mid-load and wedges the host past SIGKILL, and only a power cycle recovers "
        f"it (issues #23, #24)."
    )

    if acknowledged:
        logger.warning(
            "%s Proceeding anyway: acknowledge_unproven_envelope is set. Stay at the keyboard "
            "and follow %s -- if this holds, record the result as an EnvelopePoint in "
            "music_video_maker/envelope.py so the next run is checked against evidence "
            "instead of an acknowledgement.",
            detail,
            runbook,
        )
        return misses

    logger.error(
        "%s Refusing to start the run. The sanctioned way past this is to measure it: run the "
        "attended one-chunk proof in %s, then add what it measured to PROVEN_ENVELOPES in "
        "music_video_maker/envelope.py. To run the proof itself (or to accept the risk "
        "knowingly, at the keyboard), set acknowledge_unproven_envelope = true in the run "
        "config.",
        detail,
        runbook,
    )
    raise UnprovenEnvelopeError(detail)


__all__ = [
    "CALIBRATED_CEILING",
    "DORIS_4090_PROVEN",
    "EnvelopeMiss",
    "EnvelopePoint",
    "MeasuredCeiling",
    "PROVEN_ENVELOPES",
    "RenderedExtreme",
    "UnprovenEnvelopeError",
    "check_render_envelope",
    "covering_point",
    "envelope_for",
    "largest_proven_frames",
    "largest_rendered_frame_count",
    "measured_ceiling",
    "unproven_chunks",
]
