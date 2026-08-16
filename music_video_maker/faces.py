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

Recognition, on top of detection (issue #49)
----------------------------------------------
Detection alone cannot tell a big, confident face from *her* big, confident
face. lucky-ones v7, boundary 9 into 10: the gate above would have passed it
at 1.06% of frame area -- the face is the man holding the keyboard, not the
performer, whose back is to camera in the same frame. :func:`recognize_face`
adds a third floor, a cosine similarity between the seed frame's largest face
and the active cast member's staged reference photo, computed with SFace
(``cv2.FaceRecognizerSF``) over the same YuNet detection this module already
runs -- the reference photo is already loaded and staged per chunk, so the
comparison target costs nothing new.

SFace is 38.7 MB against YuNet's 232 KB (165x), so unlike YuNet it is **not
committed** to this repository (issue #51: prefer not shipping large binaries
that aren't load-bearing for every run). :func:`resolve_recognition_model_path`
still names a default location, and :data:`RECOGNITION_MODEL_SOURCE` /
:data:`RECOGNITION_MODEL_SHA256` record where to get it and how to verify it,
the way :data:`MODEL_SOURCE` / :data:`MODEL_SHA256` do for YuNet -- but a
checkout with no reference photo wired in, or no SFace weights on disk,
degrades to exactly the pre-#49 detection-only gate rather than failing
closed or open by accident. See ``docs/seed-face-recognition.md`` for the
full calibration writeup.

Calibrated against real seed frames and cast reference photos under
``~/mvm-runs/`` (lucky-ones: Dianne and Jan; storms: the committed
``seed_frontal_chunk15.png`` fixture against Derek's reference, both
consented for publication -- issue #51 #2). Cosine similarity, SFace,
largest face in the frame (score >= 0.9, same selection :func:`detect_faces`
uses) against the cast member's reference photo (score >= 0.5, highest
*confidence* rather than largest area -- see :func:`recognize_face`'s
docstring for why):

============================================  ===========  ==========
pair                                           same person  similarity
============================================  ===========  ==========
lucky-ones chunk 13->14 vs Dianne               yes          0.820
lucky-ones chunk 32->33 vs Dianne               yes          0.520
lucky-ones chunk 1 vs Dianne                    yes          0.494
storms chunk 15 (fixture) vs Derek              yes          0.353
lucky-ones chunk 32->33 vs Jan                  no           0.208
lucky-ones chunk 9->10 (keyboard man) vs Dianne no           0.140
lucky-ones chunk 30->31 vs Dianne               no           0.172
... 10 more cross-identity pairs                no           <= 0.156
============================================  ===========  ==========

16 pairs total: 4 genuine (2 subjects), 12 impostor, plus 3 reference-photo-
vs-reference-photo cross-checks (all < 0.08, not shown). The worst genuine
match (0.353, an extreme three-quarter profile looking down -- not a
frontal shot) and the worst impostor match (0.208, Dianne's reference
against the *other* cast member's face in a real chained frame) are
separated by 0.145. :data:`DEFAULT_MIN_FACE_SIMILARITY` sits close to the
genuine floor rather than centered in the gap, on purpose -- see its own
docstring.

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

DEFAULT_MIN_FACE_SIMILARITY = 0.34
"""Minimum SFace cosine similarity between a seed frame's largest face and the
active cast member's reference photo, for that frame to be trusted as *her*
face rather than merely *a* face (issue #49).

Measured across 16 real pairs (4 genuine, 12 impostor -- see the module
docstring's table): the worst genuine match was 0.353 (an extreme
three-quarter profile, not the frontal shots that scored 0.49-0.82); the
worst impostor match was 0.208 (a real chained frame's face matched against
the *wrong* cast member's reference). 0.34 sits 0.013 below the weakest
genuine pair and 0.132 above the strongest impostor pair -- close to the
genuine floor rather than centered in the gap, for the same reason
:data:`DEFAULT_SCORE_THRESHOLD` sits close to its own genuine floor rather
than its false-positive ceiling: a similarity threshold guards against a
false *approve* (it throws away the likeness for the rest of the chain), so
margin belongs on the refuse side, where the mistake is cheap -- a shot that
renders through the strictly-better-conditioned base path instead.

OpenCV Zoo's own SFace demo publishes 0.363 as a general-purpose "same
identity" cosine threshold (``models/face_recognition_sface/sface.py``),
calibrated on frontal benchmark pairs. It sits *above* this project's own
weakest genuine pair (0.353) -- consistent with H3 seed frames routinely
being non-frontal in a way a benchmark portrait pair is not, and exactly why
this constant is measured against this project's own material rather than
imported from someone else's calibration."""

RECOGNITION_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"
RECOGNITION_MODEL_SOURCE = "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
"""Upstream source (issue #49). SFace is contributed by Yaoyao Zhong
(https://github.com/zhongyy/SFace), ONNX conversion by Chengrui Wang;
licensed Apache-2.0 per the model directory's own ``LICENSE`` file at that
URL. Unlike YuNet, this weights file is **not committed** to this
repository -- at 38.7 MB it is 165x YuNet's 232 KB for one gate in twelve on
the material tested (issue #49's originating comment on #47), so it is
fetched by the operator and pointed at with ``recognition_model_path`` (or
placed at the default location :func:`resolve_recognition_model_path`
names) rather than shipped. Its absence is not an error: every caller in
this module treats a missing recognition model exactly like a missing
detection model -- a :class:`FaceDetectionError` that :func:`build_seed_face_gate`
turns into "cannot verify, refuse to chain" -- so a checkout with no SFace
weights on disk still renders, just without the identity check."""

RECOGNITION_MODEL_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
"""Matches the 38,696,353-byte file at :data:`RECOGNITION_MODEL_SOURCE` as of
2026-08-15 (verified against the file's own Git LFS pointer in the opencv_zoo
repository, not a local download that could have been tampered with)."""

KNOWN_LIMITATION = (
    "detection alone cannot tell whose face it is, and a checkout with no SFace "
    "weights on disk (they are not committed -- see RECOGNITION_MODEL_SOURCE) or "
    "no reference photo passed to the gate still runs detection-only, exactly as "
    "before issue #49. Recognition still cannot help when the reference photo "
    "itself has no detectable face (an unusable cast photo), when the real "
    "impostor is a close relative or otherwise resembles the cast member closely "
    "enough to clear the measured similarity floor, or when the performer's own "
    "appearance has drifted far enough from her reference photo (heavy makeup, a "
    "costume change, chained appearance-compounding drift per issue #46) that a "
    "genuine frame of her scores below the floor -- in that last case recognition "
    "actively works against a run rather than for it, since a correct chain is "
    "refused, not a wrong one approved. The floor is calibrated so that mistake "
    "stays the cheap one."
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


def resolve_recognition_model_path(explicit: Path | str | None = None) -> Path:
    """Where the SFace weights live (issue #49). Explicit path wins; otherwise
    the same directory YuNet's committed weights live in -- but unlike
    :func:`resolve_model_path`'s target, nothing is committed at this default
    location, so a fresh checkout resolves a path that does not exist. That is
    the expected state, not an error: see :data:`RECOGNITION_MODEL_SOURCE`."""
    if explicit is not None:
        return Path(explicit)
    return Path(__file__).resolve().parent.parent / "models" / RECOGNITION_MODEL_FILENAME


def _import_cv2(feature: str, issue: str):
    """Shared lazy import for :func:`detect_faces` and :func:`recognize_face`.

    Imports ``cv2`` lazily so the whole test suite -- and any run that never
    chains -- keeps working without OpenCV installed. Raises
    :class:`FaceDetectionError` rather than propagating an ``ImportError``, so
    callers have one thing to catch."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by the message, not CI
        raise FaceDetectionError(
            f"OpenCV is not installed, so {feature} ({issue}). Install "
            "opencv-python-headless, or set i2v_require_seed_face = false to "
            "chain without either check."
        ) from exc
    return cv2


def _detect_yunet_faces(cv2, image, model: Path, score_threshold: float):
    """Raw YuNet detections for ``image`` -- rows of ``[x, y, w, h, <5
    landmarks>, score]``, or ``None``/empty when nothing cleared
    ``score_threshold``. Shared by :func:`detect_faces` and
    :func:`recognize_face` so both pick a candidate face from the *identical*
    detector call -- the invariant :func:`recognize_face` depends on is that
    it recognizes the very face :func:`detect_faces` measured, not some other
    face in a multi-face frame."""
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(model), "", (width, height), score_threshold=score_threshold
    )
    _retval, detected = detector.detect(image)
    return detected


def _largest_face(faces):
    """The nearest-to-camera candidate: largest bounding-box area wins. This
    is the selection :func:`detect_faces` has always used for a *seed frame*,
    where the nearest face is the performer in every calibration frame where
    she is facing camera at all (issue #47)."""
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda row: float(row[2]) * float(row[3]))


def _most_confident_face(faces):
    """The highest-*score* candidate, used only for a **reference photo**
    (issue #49) -- a curated, single-subject portrait, not a scene. Area is
    the wrong selection there: at a threshold low enough to detect a face in
    every calibration reference photo (0.73-0.86 confidence -- see the module
    docstring), the largest spurious box in a cluttered background can exceed
    the real face's area by more than 2x while scoring under 0.15. Confidence
    does not have that failure mode in the calibration set."""
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda row: float(row[-1]))


_REFERENCE_FACE_SCORE_THRESHOLD = 0.5
"""Detection floor used only for extracting a **reference photo's** face,
never a seed frame's. :data:`DEFAULT_SCORE_THRESHOLD` (0.9) is calibrated
against in-scene seed frames and is too strict for a reference photo: all
three real cast reference photos measured (issue #49's calibration) scored
below it -- 0.73-0.86 -- despite each containing exactly one clear, correctly
identified face once :func:`_most_confident_face` (not area) does the
selecting. 0.5 is comfortably below every measured reference photo's real
face and is safe specifically *because* selection is by confidence, not by
membership above this floor: a low floor only widens the candidate pool,
and the genuine face wins every measured case by score even when a
same-image spurious box also clears 0.5."""


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
    cv2 = _import_cv2("seed frames cannot be checked for a face", "issue #47")

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
    faces = _detect_yunet_faces(cv2, image, model, score_threshold)

    if faces is None or len(faces) == 0:
        return FaceObservation(face_count=0, largest_fraction=0.0, score=0.0)

    # YuNet rows are [x, y, w, h, <5 landmarks>, score]; compare by area so the
    # nearest face wins, which is the performer in every calibration frame
    # where she is facing camera at all.
    largest = _largest_face(faces)
    fraction = (float(largest[2]) * float(largest[3])) / float(width * height)
    return FaceObservation(
        face_count=len(faces), largest_fraction=fraction, score=float(largest[-1])
    )


def recognize_face(
    frame_path: Path | str,
    reference_photo: Path | str,
    *,
    model_path: Path | str | None = None,
    recognition_model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> float:
    """Cosine similarity between the largest face in ``frame_path`` and the
    face in ``reference_photo`` -- the active cast member's staged reference
    photo (issue #49).

    The frame's face is selected exactly as :func:`detect_faces` selects it
    (largest area among detections scoring >= ``score_threshold``), so a
    caller that already ran :func:`detect_faces` on the same frame with the
    same ``score_threshold`` is guaranteed to be recognizing the same face it
    measured. The reference photo's face is selected by confidence instead
    (see :func:`_most_confident_face`), at :data:`_REFERENCE_FACE_SCORE_THRESHOLD`.

    Raises :class:`FaceDetectionError` for anything that stops the comparison
    from happening at all: OpenCV missing, either model file missing, either
    image unreadable, or no face found in either image. This deliberately
    covers the same failure shape :func:`detect_faces` already has one caller
    (:func:`build_seed_face_gate`) catching -- "cannot prove it is her" and
    "cannot prove there is a face" degrade identically, to the base reference
    path, never to an exception that ends a run.
    """
    cv2 = _import_cv2(
        "a seed frame cannot be matched against the cast reference photo", "issue #49"
    )

    detection_model = resolve_model_path(model_path)
    if not detection_model.exists():
        raise FaceDetectionError(
            f"YuNet face-detection model not found at {detection_model}. Without it "
            f"neither the seed frame nor the reference photo can be checked for a "
            f"face (issue #47)."
        )

    recognition_model = resolve_recognition_model_path(recognition_model_path)
    if not recognition_model.exists():
        raise FaceDetectionError(
            f"SFace face-recognition model not found at {recognition_model}. A big, "
            f"confidently-detected face is not enough on its own to prove it is the "
            f"performer's; without this model the seed frame's face cannot be matched "
            f"against the cast reference photo, so this boundary cannot be verified "
            f"(issue #49). See {RECOGNITION_MODEL_SOURCE} to obtain it -- sha256 "
            f"{RECOGNITION_MODEL_SHA256}."
        )

    frame_image = cv2.imread(str(frame_path))
    if frame_image is None:
        raise FaceDetectionError(f"could not read seed frame {frame_path}")

    reference_image = cv2.imread(str(reference_photo))
    if reference_image is None:
        raise FaceDetectionError(f"could not read reference photo {reference_photo}")

    frame_faces = _detect_yunet_faces(cv2, frame_image, detection_model, score_threshold)
    frame_face = _largest_face(frame_faces)
    if frame_face is None:
        raise FaceDetectionError(
            f"no face detected in seed frame {frame_path} at score_threshold="
            f"{score_threshold} -- nothing to match against the reference photo"
        )

    reference_faces = _detect_yunet_faces(
        cv2, reference_image, detection_model, _REFERENCE_FACE_SCORE_THRESHOLD
    )
    reference_face = _most_confident_face(reference_faces)
    if reference_face is None:
        raise FaceDetectionError(
            f"no face detected in reference photo {reference_photo} -- cannot build "
            f"a comparison target (issue #49)"
        )

    recognizer = cv2.FaceRecognizerSF.create(str(recognition_model), "")
    frame_feature = recognizer.feature(recognizer.alignCrop(frame_image, frame_face))
    reference_feature = recognizer.feature(recognizer.alignCrop(reference_image, reference_face))
    return float(
        recognizer.match(frame_feature, reference_feature, cv2.FaceRecognizerSF_FR_COSINE)
    )


def build_seed_face_gate(
    *,
    min_fraction: float = DEFAULT_MIN_FACE_FRACTION,
    model_path: Path | str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_similarity: float = DEFAULT_MIN_FACE_SIMILARITY,
    recognition_model_path: Path | str | None = None,
):
    """A ``Callable[[Path, Path | None], bool]`` deciding whether a seed frame
    may be chained from -- the shape
    :class:`~music_video_maker.continuity.ContinuityWorkflowProvider` takes, so
    the provider owes nothing to OpenCV and a test can pass a lambda.

    The second argument is the active cast member's reference photo (issue
    #49) -- optional and defaulting to ``None`` so every pre-#49 caller keeps
    the detection-only behaviour of issue #47 unchanged. Passing it adds a
    third floor, :data:`DEFAULT_MIN_FACE_SIMILARITY`, on top of the existing
    area and confidence floors: a face that is big and confident enough to
    pass detection is not automatically *hers* (lucky-ones v7, boundary 9
    into 10 -- see the module docstring).

    A detection failure, or -- once a reference photo is given -- a
    recognition failure, both return ``False``: "cannot prove a face is
    present" and "cannot prove it is her face" degrade identically, to the
    base reference path, which is the safe direction. It costs a slightly
    different shot; the alternative costs the likeness for the rest of the
    chain.
    """

    def gate(frame_path: Path, reference_photo: Path | str | None = None) -> bool:
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
        if not ok:
            return False

        if reference_photo is None:
            logger.info(
                "seed frame %s: no reference photo supplied -- detection-only decision "
                "(pre-issue-#49 behaviour)",
                frame_path,
            )
            return True

        try:
            similarity = recognize_face(
                frame_path,
                reference_photo,
                model_path=model_path,
                recognition_model_path=recognition_model_path,
                score_threshold=score_threshold,
            )
        except FaceDetectionError:
            logger.exception(
                "seed frame %s could not be matched against reference photo %s -- refusing "
                "to chain from it. A face big and confident enough to pass detection is not "
                "necessarily HERS (issue #49), and an unverified match is exactly the case "
                "that loses the performer's likeness",
                frame_path,
                reference_photo,
            )
            return False

        matched = similarity >= min_similarity
        logger.info(
            "seed frame %s vs reference photo %s: cosine similarity %.4f -- %s "
            "(floor %.4f, issue #49)",
            frame_path,
            reference_photo,
            similarity,
            "may chain" if matched else "NOT chainable",
            min_similarity,
        )
        return matched

    return gate


__all__ = [
    "DEFAULT_MIN_FACE_FRACTION",
    "DEFAULT_MIN_FACE_SIMILARITY",
    "DEFAULT_SCORE_THRESHOLD",
    "KNOWN_LIMITATION",
    "MODEL_FILENAME",
    "MODEL_SHA256",
    "MODEL_SOURCE",
    "RECOGNITION_MODEL_FILENAME",
    "RECOGNITION_MODEL_SHA256",
    "RECOGNITION_MODEL_SOURCE",
    "FaceDetectionError",
    "FaceObservation",
    "build_seed_face_gate",
    "detect_faces",
    "recognize_face",
    "resolve_model_path",
    "resolve_recognition_model_path",
]
