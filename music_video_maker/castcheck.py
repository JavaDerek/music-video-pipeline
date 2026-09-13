"""Consistency acceptance check for a synthetic cast member's reference set
(issue #56).

"One flattering portrait is easy. A character who reads as the same person
across 80 shots is not" -- and that is exactly the thing
:func:`music_video_maker.faces.recognize_face` already measures (issue #49):
SFace cosine similarity, calibrated on 16 real pairs to
:data:`~music_video_maker.faces.DEFAULT_MIN_FACE_SIMILARITY` = 0.34 (see
``docs/seed-face-recognition.md``). This module is the offline instrument the
design doc (``docs/design-synthetic-cast.md``, "The problem is consistency,
and it has a measurable definition") asks for: given a character's reference
set, score every pair and report whether the set clears the floor -- **before**
any GPU time is spent conditioning a render on it.

This module never picks an image model, never downloads anything, and never
generates an image. Issue #56 deliberately defers that decision (licence
first); this is the offline part that does not depend on it.

Two things this check must get right, both measured lessons from elsewhere in
this project, not invented for this module:

* **A face too small/unconfident to detect is not evidence of anything**
  (issue #93). ``FaceObservation.verdict`` (``detected`` / ``inconclusive`` /
  ``absent`` / ``unexamined``) is read for every image *before* any pair is
  scored, and anything other than ``detected`` is excluded from pairwise
  scoring entirely and surfaced in :attr:`ConsistencyReport.undetected` --
  never averaged in, never silently scored as a 0.34-failing 0.0.
* **A floor calibrated on real photographs may not transfer to generated
  ones.** Generated faces are often *more* similar to each other than two
  photographs of one real person are -- a mode-collapse artefact, not
  identity -- so :data:`~music_video_maker.faces.DEFAULT_MIN_FACE_SIMILARITY`
  is only this check's *default*, and :meth:`ConsistencyReport.render` says
  so in the report text itself, every time, not just in this docstring.

Reuses :mod:`music_video_maker.faces` for detection and recognition -- this
module does not touch cv2, ONNX, or SFace directly, so it stays exactly as
lazy-import-safe and offline-testable as ``faces.py`` and ``facescan.py``
already are. :data:`Detector`/:data:`Recognizer` are injectable callables in
the same shape ``facescan.py``'s ``Detector``/``FrameExtractor`` are, so the
row-building and reporting logic here is fully exercised in CI with fakes and
no OpenCV, the way ``tests/test_faces.py`` exercises ``build_seed_face_gate``
with a monkeypatched ``detect_faces``/``recognize_face``.

Why the detection/recognition floor here is 0.5, not faces.py's 0.9
---------------------------------------------------------------------
Every image compared by this module is a reference-photo-style portrait, not
an in-scene H3 seed frame -- and ``docs/seed-face-recognition.md``'s own
calibration found all three measured cast reference photos scored their
genuine face *below* 0.9 (0.727-0.859) despite being clear, correctly-framed
portraits. That is the same reason ``faces.py``'s own
``_REFERENCE_FACE_SCORE_THRESHOLD`` (0.5) exists for a reference photo's role
inside :func:`~music_video_maker.faces.recognize_face` -- but that constant
only ever applies to the *second* argument of that call. Calling
``recognize_face(image_a, image_b)`` with two portraits and the default
``score_threshold=0.9`` would refuse most genuine portrait pairs outright, for
the same reason a real cast reference photo would. :data:`
DEFAULT_PORTRAIT_SCORE_THRESHOLD` applies that same 0.5 floor to *both* sides
of a pairwise portrait comparison.

Provenance, stamped like ``facescan.py`` stamps it (issue #93's lesson)
--------------------------------------------------------------------------
"A filename is not provenance" -- ``facescan.py`` exists because a scan's own
output filename once asserted a subject it had never actually read. Every
image this module reports on carries its resolved source path, size and
mtime (:class:`ImageProvenance`); the report as a whole records the detection
and recognition model files and their sha256, and the thresholds used
(:class:`ConsistencyReport`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from music_video_maker import faces

logger = logging.getLogger(__name__)

DEFAULT_PORTRAIT_SCORE_THRESHOLD = 0.5
"""Detection floor for this module's comparisons -- not
:data:`faces.DEFAULT_SCORE_THRESHOLD` (0.9), which is calibrated against
in-scene H3 seed frames. Every image this module looks at is a
reference-photo-style portrait, and all three real cast reference photos
measured for issue #49 scored their genuine face below 0.9 (0.727-0.859)
despite being clear, correctly-framed portraits -- see the module
docstring's "Why the detection/recognition floor here is 0.5" section. This
is the same value as ``faces.py``'s own (private)
``_REFERENCE_FACE_SCORE_THRESHOLD``, applied to both sides of a pairwise
comparison rather than only the reference-photo side."""

DEFAULT_PORTRAIT_INSPECTION_FLOOR: float | None = None
"""Score floor for an optional *second*, inspection-only detector call
(issue #93: a bare zero from detection never meant "no face", only "nothing
cleared the primary floor"). **Off by default**, so an undetected image's
verdict is ``unexamined``.

No inspection floor has been calibrated for portraits. ``faces.DEFAULT_
INSPECTION_FLOOR`` (0.70) was calibrated on in-scene H3 frames against a 0.9
gate, so it sits *above* this module's 0.5 detection floor and could never
qualify anything here; and #93 measured that a low floor (0.15) fires on
nearly every zero and its extra candidates are not faces -- the one cited to
justify it was the back of a head. A guessed value between the two would be
a preference, not a calibration. It also buys nothing yet: every image that
is not ``detected`` is already excluded from pairing and makes the report
``needs_review`` whatever its verdict says, which is the "look at this"
outcome an inspection pass exists to trigger. Pass ``--inspection-floor``
once synthetic material exists to calibrate one against."""

_DETECTED = "detected"


# --------------------------------------------------------------------------- #
# Injectable seams -- no cv2 required to import or test this module.
# --------------------------------------------------------------------------- #

Detector = Callable[[Path], faces.FaceObservation]
"""Given an image path, returns the :class:`faces.FaceObservation` for it.
Injectable so :func:`check_consistency` never has to know whether it is
talking to real OpenCV or a test fake -- the same shape
``facescan.Detector`` uses."""

Recognizer = Callable[[Path, Path], float]
"""Given two image paths, returns their SFace cosine similarity. Raises
:class:`faces.FaceDetectionError` if either image has no usable face or a
model is missing -- callers degrade per-pair, never abort the whole report
for one bad pair (this project's "one failure must not kill the run" rule,
scaled down to a single comparison)."""


def build_default_detector(
    *,
    model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_PORTRAIT_SCORE_THRESHOLD,
    inspect_floor: float | None = DEFAULT_PORTRAIT_INSPECTION_FLOOR,
) -> Detector:
    """A :data:`Detector` closure over :func:`faces.detect_faces`, at the
    portrait-appropriate floor (see the module docstring) rather than the
    seed-frame floor ``detect_faces`` defaults to."""

    def detector(image_path: Path) -> faces.FaceObservation:
        return faces.detect_faces(
            image_path,
            model_path=model_path,
            score_threshold=score_threshold,
            inspect_floor=inspect_floor,
        )

    return detector


def build_default_recognizer(
    *,
    detection_model_path: Path | str | None = None,
    recognition_model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_PORTRAIT_SCORE_THRESHOLD,
) -> Recognizer:
    """A :data:`Recognizer` closure over :func:`faces.recognize_face`, at the
    same portrait-appropriate ``score_threshold`` :func:`build_default_detector`
    uses -- both images passed here are reference photos, never a seed frame,
    so both sides of the comparison want the lower floor (unlike
    ``recognize_face``'s own asymmetric default, tuned for a seed frame
    against a reference photo)."""

    def recognizer(image_a: Path, image_b: Path) -> float:
        return faces.recognize_face(
            image_a,
            image_b,
            model_path=detection_model_path,
            recognition_model_path=recognition_model_path,
            score_threshold=score_threshold,
        )

    return recognizer


# --------------------------------------------------------------------------- #
# Provenance + report data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageProvenance:
    """What #93 says every measurement must record about a file it actually
    opened, rather than trusting the caller's path or the report's own
    filename."""

    path: Path
    """As supplied by the caller."""
    resolved_path: Path
    """Absolute, resolved path -- the field a stale relative path or a
    working-directory mismatch would otherwise hide."""
    size_bytes: int
    mtime: str
    """ISO-8601 UTC timestamp of the source file's mtime."""


@dataclass(frozen=True)
class ImageObservation:
    """What detection saw for one reference image, plus its provenance."""

    provenance: ImageProvenance
    verdict: str
    """One of :attr:`faces.FaceObservation.verdict`'s four states
    (``detected`` / ``inconclusive`` / ``absent`` / ``unexamined``), or
    ``"error"`` -- this module's own fifth state, for an image the detector
    could not even attempt to examine (:class:`faces.FaceDetectionError`:
    an unreadable file, or a missing model). Only ``"detected"`` images are
    scored in :attr:`ConsistencyReport.pairs`; every other verdict is
    surfaced in :attr:`ConsistencyReport.undetected`, never averaged in and
    never scored as a similarity of 0.0 (issue #93's own lesson: a zero from
    a detector is not evidence of absence)."""
    detail: str | None = None
    """The exception message, when :attr:`verdict` is ``"error"``. ``None``
    otherwise."""


@dataclass(frozen=True)
class PairSimilarity:
    """One scored pair."""

    image_a: Path
    image_b: Path
    similarity: float


@dataclass(frozen=True)
class UnscoredPair:
    """A pair whose images both detected, but whose recognition comparison
    itself failed (a missing recognition model, most likely -- see the
    module docstring on why that is refused loudly for the run whose whole
    point is recognition, but degraded per-pair rather than aborting the
    rest of the report, matching this project's resilience rule)."""

    image_a: Path
    image_b: Path
    reason: str


@dataclass(frozen=True)
class ConsistencyReport:
    """The full result of :func:`check_consistency` -- both the scored pairs
    and everything needed to say *how* they were scored (issue #93's
    provenance lesson, applied to the report as a whole rather than just its
    rows)."""

    character: str
    floor: float
    score_threshold: float
    observations: tuple[ImageObservation, ...]
    pairs: tuple[PairSimilarity, ...]
    unscored_pairs: tuple[UnscoredPair, ...]
    detection_model_path: Path
    detection_model_sha256: str
    recognition_model_path: Path
    recognition_model_sha256: str
    generated_at: str

    @property
    def undetected(self) -> tuple[ImageObservation, ...]:
        """Images that did not clear detection -- "look at this", per the
        design doc, never averaged into :attr:`pairs` and never treated as a
        failing similarity of 0.0."""
        return tuple(o for o in self.observations if o.verdict != _DETECTED)

    @property
    def minimum_pair(self) -> PairSimilarity | None:
        """The pair that would sink the set, or ``None`` when there are no
        scored pairs at all (see :attr:`status` -- that is reported as
        ``"insufficient"``, not treated as a vacuous pass)."""
        if not self.pairs:
            return None
        return min(self.pairs, key=lambda p: p.similarity)

    @property
    def status(self) -> str:
        """One of four honest outcomes -- deliberately not a bare bool,
        because "no failing pair" and "passed" are different claims:

        * ``"insufficient"`` -- fewer than two images were supplied at all,
          so no pair could ever have existed. The design doc is explicit
          that a set of one must say so, not pass vacuously -- this is that
          rule, and the *only* thing it means: it is about how many images
          were handed in, never about what happened to them.
        * ``"needs_review"`` -- at least one image did not clear detection,
          or at least one pair could not be recognised (even if every
          *attempted* pair scored fine). A human must look at those before
          this set can be called consistent either way -- reported ahead of
          pass/fail because it is the more urgent fact. This covers "every
          image failed detection" too: that is not "insufficient data", it
          is a concrete thing to go look at, and more images would not fix
          it.
        * ``"fail"`` -- every image detected, every pair scored, and the
          minimum similarity is below :attr:`floor`.
        * ``"pass"`` -- every image detected, every pair scored, every pair
          at or above :attr:`floor`.
        """
        if len(self.observations) < 2:
            return "insufficient"
        if self.undetected or self.unscored_pairs:
            return "needs_review"
        return "pass" if all(p.similarity >= self.floor for p in self.pairs) else "fail"

    def render(self) -> str:
        """A human-readable report -- provenance header, per-image
        detection verdicts, every scored pair, the minimum pair, and the
        two caveats the design doc requires travel with any use of this
        check (mode collapse, and "a zero is not evidence of absence")."""
        lines: list[str] = []
        lines.append(f"# music_video_maker.castcheck report (issue #56): {self.character}")
        lines.append(f"# generated_at={self.generated_at}")
        lines.append(
            f"# detection_model={self.detection_model_path} sha256={self.detection_model_sha256}"
        )
        lines.append(
            f"# recognition_model={self.recognition_model_path} "
            f"sha256={self.recognition_model_sha256}"
        )
        lines.append(f"# floor={self.floor} score_threshold={self.score_threshold}")
        lines.append("#")
        lines.append(
            "# CAVEAT: this floor was calibrated on photographs of real people "
            "(docs/seed-face-recognition.md, 16 pairs). Generated faces are often MORE "
            "similar to each other than two photographs of one real person are -- a "
            "mode-collapse artefact, not identity. Re-derive this floor on synthetic "
            "pairs before trusting it; do not assume it transfers."
        )
        lines.append("")

        lines.append(f"images: {len(self.observations)}")
        for obs in self.observations:
            detail = f" ({obs.detail})" if obs.detail else ""
            lines.append(
                f"  [{obs.verdict}] {obs.provenance.path} "
                f"(resolved={obs.provenance.resolved_path}, "
                f"size={obs.provenance.size_bytes}B, mtime={obs.provenance.mtime}){detail}"
            )

        if len(self.observations) < 2:
            lines.append("")
            lines.append(
                "RESULT: insufficient -- a reference set of one image has no pairs to "
                "score. This check cannot assess consistency from a single image; add at "
                "least one more reference image."
            )
            return "\n".join(lines)

        if self.undetected:
            lines.append("")
            lines.append(
                f"{len(self.undetected)} image(s) did not clear detection -- look at "
                "these directly, they are excluded from every pair below, not averaged "
                "in and not scored as a failing 0.0 (issue #93: a zero from a detector "
                "is not evidence of absence):"
            )
            for obs in self.undetected:
                lines.append(f"  [{obs.verdict}] {obs.provenance.path}")

        lines.append("")
        lines.append(f"pairs scored: {len(self.pairs)}")
        for pair in sorted(self.pairs, key=lambda p: p.similarity):
            mark = "OK" if pair.similarity >= self.floor else "BELOW FLOOR"
            lines.append(f"  {pair.image_a} <-> {pair.image_b}: {pair.similarity:.4f} [{mark}]")

        if self.unscored_pairs:
            lines.append("")
            lines.append(f"{len(self.unscored_pairs)} pair(s) could not be scored:")
            for unscored in self.unscored_pairs:
                lines.append(f"  {unscored.image_a} <-> {unscored.image_b}: {unscored.reason}")

        minimum = self.minimum_pair
        lines.append("")
        if minimum is not None:
            lines.append(
                f"minimum pair: {minimum.image_a} <-> {minimum.image_b} "
                f"({minimum.similarity:.4f})"
            )
        elif not self.pairs:
            lines.append(
                "minimum pair: none -- no pair could be scored at all (see the images "
                "requiring review above)."
            )
        lines.append(f"RESULT: {self.status}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #


def _format_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _observe(path: Path, detector: Detector) -> ImageObservation:
    resolved = path.resolve()
    try:
        stat = path.stat()
        provenance = ImageProvenance(
            path=path,
            resolved_path=resolved,
            size_bytes=stat.st_size,
            mtime=_format_mtime(stat.st_mtime),
        )
    except OSError as exc:
        # Can't even stat it -- still produce a row (issue #93: every image
        # gets a provenance-stamped verdict, not a silent drop from the
        # report) with whatever provenance is knowable.
        logger.warning("castcheck: could not stat %s: %s", path, exc)
        provenance = ImageProvenance(
            path=path, resolved_path=resolved, size_bytes=-1, mtime="unknown"
        )

    try:
        observation = detector(path)
    except faces.FaceDetectionError as exc:
        logger.warning("castcheck: %s could not be checked for a face: %s", path, exc)
        return ImageObservation(provenance=provenance, verdict="error", detail=str(exc))

    return ImageObservation(provenance=provenance, verdict=observation.verdict)


def check_consistency(
    character: str,
    images: Sequence[Path | str],
    *,
    floor: float = faces.DEFAULT_MIN_FACE_SIMILARITY,
    score_threshold: float = DEFAULT_PORTRAIT_SCORE_THRESHOLD,
    detector: Detector,
    recognizer: Recognizer,
    detection_model_path: Path | str | None = None,
    detection_model_sha256: str = faces.MODEL_SHA256,
    recognition_model_path: Path | str | None = None,
    recognition_model_sha256: str = faces.RECOGNITION_MODEL_SHA256,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ConsistencyReport:
    """Score every pair of ``images`` for ``character`` with ``recognizer``,
    after excluding any image ``detector`` cannot find a face in.

    ``detector``/``recognizer`` are required, not defaulted to the real
    OpenCV-backed implementations -- callers that want the real thing pass
    :func:`build_default_detector`/:func:`build_default_recognizer` (or let
    :func:`main` do it); tests pass fakes and need no OpenCV at all, the same
    split ``facescan.scan_chunk`` uses.

    A set of one image is not silently a vacuous pass: it produces a report
    whose :attr:`ConsistencyReport.status` is ``"insufficient"`` and whose
    :meth:`~ConsistencyReport.render` says so in plain words (the design
    doc's own requirement). An image that fails detection is never scored as
    a similarity of 0.0 and never averaged into a pair -- it is excluded from
    pairing and surfaced in :attr:`ConsistencyReport.undetected`.
    """
    paths = [Path(p) for p in images]
    observations = tuple(_observe(p, detector) for p in paths)
    detected = [obs.provenance.path for obs in observations if obs.verdict == _DETECTED]

    pairs: list[PairSimilarity] = []
    unscored: list[UnscoredPair] = []
    for i in range(len(detected)):
        for j in range(i + 1, len(detected)):
            image_a, image_b = detected[i], detected[j]
            try:
                similarity = recognizer(image_a, image_b)
            except faces.FaceDetectionError as exc:
                logger.warning(
                    "castcheck: %s <-> %s could not be recognized: %s", image_a, image_b, exc
                )
                unscored.append(UnscoredPair(image_a=image_a, image_b=image_b, reason=str(exc)))
                continue
            pairs.append(PairSimilarity(image_a=image_a, image_b=image_b, similarity=similarity))

    return ConsistencyReport(
        character=character,
        floor=floor,
        score_threshold=score_threshold,
        observations=observations,
        pairs=tuple(pairs),
        unscored_pairs=tuple(unscored),
        detection_model_path=faces.resolve_model_path(detection_model_path),
        detection_model_sha256=detection_model_sha256,
        recognition_model_path=faces.resolve_recognition_model_path(recognition_model_path),
        recognition_model_sha256=recognition_model_sha256,
        generated_at=now().isoformat(),
    )


# --------------------------------------------------------------------------- #
# CLI -- `python -m music_video_maker.castcheck`, mirroring facescan.py's own
# entry-point pattern (a module CLI, not a second console_script -- neither
# tool is load-bearing for every run, the way music-video-maker/mvm-author
# in pyproject.toml's [project.scripts] are).
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.castcheck",
        description=(
            "Score a synthetic (or real) cast member's reference image set for "
            "consistency (issue #56) -- every pair via SFace cosine similarity, "
            "offline, before any GPU time is spent conditioning a render on it."
        ),
    )
    parser.add_argument("character", help="the cast member's name, for the report header")
    parser.add_argument(
        "images", nargs="+", type=Path, help="reference image paths (at least one)"
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=faces.DEFAULT_MIN_FACE_SIMILARITY,
        help=(
            "minimum acceptable pairwise similarity (default: "
            "faces.DEFAULT_MIN_FACE_SIMILARITY, calibrated on REAL photographs -- "
            "re-derive for a synthetic cast, see the report's own caveat)"
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_PORTRAIT_SCORE_THRESHOLD,
        help="YuNet confidence floor for detection/recognition (portrait-appropriate default)",
    )
    parser.add_argument(
        "--inspection-floor",
        type=float,
        default=DEFAULT_PORTRAIT_INSPECTION_FLOOR,
        help=(
            "score floor for an optional second, inspection-only detector call "
            "(default: off -- no portrait floor has been calibrated)"
        ),
    )
    parser.add_argument(
        "--model-path", type=Path, default=None, help="override the YuNet model file location"
    )
    parser.add_argument(
        "--recognition-model-path",
        type=Path,
        default=None,
        help="override the SFace model file location",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="report output path (default: stdout)"
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    detector_factory: Callable[..., Detector] = build_default_detector,
    recognizer_factory: Callable[..., Recognizer] = build_default_recognizer,
) -> int:
    """Entry point for ``python -m music_video_maker.castcheck``.

    Exit code 0 only for :attr:`ConsistencyReport.status` == ``"pass"`` --
    every other status (``"fail"``, ``"needs_review"``, ``"insufficient"``)
    means this reference set is not ready to condition a render on, which is
    the honest failure mode the design doc asks for.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    detector = detector_factory(
        model_path=args.model_path,
        score_threshold=args.score_threshold,
        inspect_floor=args.inspection_floor,
    )
    recognizer = recognizer_factory(
        detection_model_path=args.model_path,
        recognition_model_path=args.recognition_model_path,
        score_threshold=args.score_threshold,
    )

    report = check_consistency(
        args.character,
        args.images,
        floor=args.floor,
        score_threshold=args.score_threshold,
        detector=detector,
        recognizer=recognizer,
        detection_model_path=args.model_path,
        recognition_model_path=args.recognition_model_path,
    )

    text = report.render()
    if args.out is not None:
        args.out.write_text(text + "\n")
        logger.info("castcheck: wrote report to %s (status=%s)", args.out, report.status)
    else:
        sys.stdout.write(text + "\n")

    return 0 if report.status == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - exercised via main(), not this guard
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PORTRAIT_INSPECTION_FLOOR",
    "DEFAULT_PORTRAIT_SCORE_THRESHOLD",
    "ConsistencyReport",
    "Detector",
    "ImageObservation",
    "ImageProvenance",
    "PairSimilarity",
    "Recognizer",
    "UnscoredPair",
    "build_default_detector",
    "build_default_recognizer",
    "build_parser",
    "check_consistency",
    "main",
]
