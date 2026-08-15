"""Is the performer's face actually in the seed frame? (issue #47)

``MiniMaxH3ImageToVideo`` has no ``ref_images`` input. On the chained path the
*entire* identity conditioning is ``first_frame`` -- the predecessor chunk's
final frame. Nothing checked that that frame contained a face, and more than
half of one real run's seed frames did not: the performer walks away from
camera, the shot ends on the back of her head, and the next chunk has no
information about what she looks like. When she turns around a few seconds
later the model is not recalling a face it drifted from, it is inventing one it
was never given.

So face-presence is a **precondition for chaining**, not an assumption.
``i2v_reanchor_interval`` is a *positional* guess at when drift has piled up;
this is the *actual* precondition for chaining to preserve identity at all.
A frame that fails renders through the base reference path instead, where the
cast photo is a genuine ``ref_images`` reference and the vocal stem is back.

Detection only, deliberately
-----------------------------
This asks "is there a big enough face here", not "is it *her*". Answering the
second needs a recognition embedding, and the model for that is 38.7 MB against
YuNet's 232 KB -- see :data:`KNOWN_LIMITATION` for the one real case in the
calibration set where the difference shows.

Threshold, measured not invented
---------------------------------
Calibrated against all 12 seed frames of the lucky-ones v7 run, scored by eye
first and then by YuNet (largest detected face, as a percentage of frame area):

===========================  =========  ==========
seed frame                   by eye     face area
===========================  =========  ==========
chunk 14, 31                 good         4.02, 4.03
chunk 33                     good         2.94
chunk 10                     **no face**  1.06  <-- another person in frame
chunk 1                      good         0.76
chunk 9                      too distant  0.27
chunk 34                     **no face**  0.24  <-- the shot Derek reported
chunk 30                     **no face**  0.19
chunk 15, 21, 22, 35         no face      nothing detected
===========================  =========  ==========

Ignoring chunk 10, which no size threshold can fix, the worst *good* frame
(0.76%) and the worst *bad* one (0.27%) are separated by a factor of 2.8.
:data:`DEFAULT_MIN_FACE_FRACTION` sits in the middle of that gap, so it is a
measurement with a stated margin rather than a round number someone liked.

This table is the calibration history and does not change; the *committed*
regression fixtures under ``tests/fixtures/seed_frames/`` are a separate,
smaller set kept only to prove the gate still behaves against a real frame,
and were replaced 2026-08-14 with consented frames of the project's author
(``storms``) rather than the lead cast member who had never been asked
(issue #51 #2) -- see ``tests/test_faces.py`` for the current filenames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MIN_FACE_FRACTION = 0.0045
"""Minimum face bounding-box area, as a fraction of the whole frame, for a seed
frame to be considered capable of carrying identity.

0.45% -- the geometric midpoint of the 0.273%/0.759% gap measured across the v7
seed frames (see the module docstring). At 864x480 that is roughly a 43x43 px
face. Below it, every frame in the calibration set was one where a human said
"you cannot see who that is"."""

DEFAULT_SCORE_THRESHOLD = 0.9
"""YuNet's own confidence floor.

This was 0.6, justified as "every true detection scored >= 0.835, so this is
well clear of the evidence". That reasoning was backwards and it cost a shot.
A detector threshold guards against *false positives*, so putting it far
**below** the observed evidence is not margin, it is exposure -- and it let
YuNet's 0.669-confidence box on a blank patch of office wall count as the
performer's face. Chunk 21 of v8 chained from a frame showing nothing but the
back of her head, and she came back as a different woman.

Measured, from the 12 v8 boundaries: every genuine face scored >= 0.769, every
genuine face large enough to approve scored >= 0.925, and the one confirmed
false positive scored 0.669. 0.9 sits below every real approval and above
every spurious box.

The asymmetry decides which way to lean when in doubt. A false *approve*
throws away the likeness for a whole shot. A false *refuse* costs a shot that
renders through the base path instead -- with the cast photo and the vocal
stem, which is strictly better conditioned. Refusing is the cheap mistake."""

KNOWN_LIMITATION = (
    "detection cannot tell whose face it is. In a frame containing other people "
    "-- a surgeon, a second cast member, an office of extras -- a large face "
    "belonging to somebody else passes this gate while the performer has her "
    "back turned. Chunk 10 of lucky-ones v7 is exactly that case (1.06% face, "
    "and it is the man holding the keyboard). Closing it needs a recognition "
    "embedding matched against CastMember.image, not just a detector."
)

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_SOURCE = "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
"""Upstream source (issue #51 #3). Licensed MIT, copyright 2020 Shiqi Yu
(YuNet's original author) -- confirmed 2026-08-14 against the model
directory's own ``LICENSE`` file, which is more specific than, and takes
precedence over, opencv_zoo's repo-level Apache-2.0. Redistribution is
permitted; the required notice is carried alongside the committed weights at
``models/face_detection_yunet_2023mar.onnx.LICENSE``, not just asserted here."""

MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
"""Pinned so the gate cannot silently change its mind about a run's chunks when
the file on disk is replaced -- the same reasoning that made the H3 text
encoder a recorded fingerprint field (#39). Matches the 232,589-byte file at
:data:`MODEL_SOURCE` as of the date above."""


class FaceDetectionError(RuntimeError):
    """The detector could not run at all (missing model, unreadable frame).

    Callers are expected to treat this as "cannot prove a face is present" and
    degrade -- never to let it end a run. A chunk rendered through the base
    reference path is a slightly different shot; a dead run is hours of GPU
    custody thrown away.
    """


@dataclass(frozen=True)
class FaceObservation:
    """What the detector saw in one frame."""

    face_count: int
    largest_fraction: float
    """Largest face's bounding-box area as a fraction of the frame. ``0.0``
    when nothing was detected."""
    score: float
    """Detector confidence for that largest face; ``0.0`` when none."""

    def carries_identity(self, min_fraction: float = DEFAULT_MIN_FACE_FRACTION) -> bool:
        return self.face_count > 0 and self.largest_fraction >= min_fraction


def resolve_model_path(explicit: Path | str | None = None) -> Path:
    """Where the YuNet weights live. Explicit path wins; otherwise the copy
    committed alongside the package."""
    if explicit is not None:
        return Path(explicit)
    return Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def detect_faces(
    frame_path: Path | str,
    *,
    model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> FaceObservation:
    """Detect faces in ``frame_path`` with YuNet.

    Imports ``cv2`` lazily so the whole test suite -- and any run that never
    chains -- keeps working without OpenCV installed. Raises
    :class:`FaceDetectionError` rather than propagating OpenCV's own errors,
    so callers have one thing to catch.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by the message, not CI
        raise FaceDetectionError(
            "OpenCV is not installed, so seed frames cannot be checked for a face "
            "(issue #47). Install opencv-python-headless, or set "
            "i2v_require_seed_face = false to chain without the check."
        ) from exc

    model = resolve_model_path(model_path)
    if not model.exists():
        raise FaceDetectionError(
            f"YuNet face-detection model not found at {model}. Without it a chained "
            f"chunk cannot be shown to carry identity (issue #47)."
        )

    image = cv2.imread(str(frame_path))
    if image is None:
        raise FaceDetectionError(f"could not read seed frame {frame_path}")

    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (width, height), score_threshold=score_threshold
    )
    _retval, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return FaceObservation(face_count=0, largest_fraction=0.0, score=0.0)

    # YuNet rows are [x, y, w, h, <5 landmarks>, score]; compare by area so the
    # nearest face wins, which is the performer in every calibration frame
    # where she is facing camera at all.
    largest = max(faces, key=lambda row: float(row[2]) * float(row[3]))
    fraction = (float(largest[2]) * float(largest[3])) / float(width * height)
    return FaceObservation(
        face_count=len(faces), largest_fraction=fraction, score=float(largest[-1])
    )


def build_seed_face_gate(
    *,
    min_fraction: float = DEFAULT_MIN_FACE_FRACTION,
    model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
):
    """A ``Callable[[Path], bool]`` deciding whether a seed frame may be chained
    from -- the shape :class:`~music_video_maker.continuity.ContinuityWorkflowProvider`
    takes, so the provider owes nothing to OpenCV and a test can pass a lambda.

    A detector failure returns ``False``: "cannot prove a face is present"
    degrades to the base reference path, which is the safe direction. It costs
    a slightly different shot; the alternative costs the likeness.
    """

    def gate(frame_path: Path) -> bool:
        try:
            observation = detect_faces(
                frame_path, model_path=model_path, score_threshold=score_threshold
            )
        except FaceDetectionError:
            logger.exception(
                "seed frame %s could not be checked for a face -- refusing to chain from "
                "it, since an unverified seed is exactly the case that loses the "
                "performer's likeness (issue #47)",
                frame_path,
            )
            return False

        ok = observation.carries_identity(min_fraction)
        logger.info(
            "seed frame %s: %d face(s), largest %.3f%% of frame (score %.3f) -- %s "
            "(threshold %.3f%%)",
            frame_path,
            observation.face_count,
            observation.largest_fraction * 100,
            observation.score,
            "may chain" if ok else "NOT chainable",
            min_fraction * 100,
        )
        return ok

    return gate


__all__ = [
    "DEFAULT_MIN_FACE_FRACTION",
    "DEFAULT_SCORE_THRESHOLD",
    "KNOWN_LIMITATION",
    "MODEL_FILENAME",
    "MODEL_SHA256",
    "MODEL_SOURCE",
    "FaceDetectionError",
    "FaceObservation",
    "build_seed_face_gate",
    "detect_faces",
    "resolve_model_path",
]
