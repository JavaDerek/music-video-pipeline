"""Tests for the seed-frame face gate (issue #47).

Two layers, and only the first runs in CI:

* the *decision* -- thresholds, the degrade-on-failure rule, the observation
  arithmetic -- which is pure and needs neither OpenCV nor a model file.
* the *detector* itself, which needs both, and is therefore skipped unless
  they are present. The calibration case is pinned there so that if the model
  or OpenCV ever changes its mind about the frame that started all this, a
  test says so rather than a video.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from music_video_maker import faces
from music_video_maker.faces import (
    DEFAULT_MIN_FACE_FRACTION,
    DEFAULT_MIN_FACE_SIMILARITY,
    FaceDetectionError,
    FaceObservation,
    build_seed_face_gate,
)

# --------------------------------------------------------------------------- #
# The decision (no OpenCV, no model, no image)
# --------------------------------------------------------------------------- #


def test_no_face_never_carries_identity():
    assert not FaceObservation(0, 0.0, 0.0).carries_identity()


def test_a_face_below_the_threshold_is_treated_as_absent():
    """Chunks 9, 30 and 34 of lucky-ones v7 all *had* a detectable face and
    were all frames a human said you could not identify anyone from -- 0.27%,
    0.19% and 0.24% of frame area. Presence alone is not the question."""
    assert not FaceObservation(1, 0.0027, 0.9).carries_identity()
    assert not FaceObservation(1, 0.0024, 0.83).carries_identity()


def test_a_face_at_or_above_the_threshold_carries_identity():
    assert FaceObservation(1, DEFAULT_MIN_FACE_FRACTION, 0.9).carries_identity()
    assert FaceObservation(1, 0.0076, 0.94).carries_identity()  # chunk 1, the worst good one


def test_the_calibrated_threshold_sits_inside_the_measured_gap():
    """The default is a measurement, not a round number. It must separate the
    worst *good* frame (0.759%) from the worst *bad* one (0.273%) -- and if
    someone retunes it, this says which evidence they are contradicting."""
    assert 0.00273 < DEFAULT_MIN_FACE_FRACTION < 0.00759


def test_the_gate_refuses_when_detection_fails(monkeypatch):
    """'Cannot prove a face is present' must read as 'do not chain'. The
    unverified seed is exactly the case that loses the likeness, so the safe
    direction is the base reference path."""

    def boom(*_a, **_k):
        raise FaceDetectionError("model file is corrupt")

    monkeypatch.setattr(faces, "detect_faces", boom)
    assert build_seed_face_gate()(Path("seed.png")) is False


def test_the_gate_passes_the_configured_threshold_through(monkeypatch):
    seen = {}

    def fake_detect(path, *, model_path=None, score_threshold=0.6):
        seen["path"] = path
        return FaceObservation(1, 0.005, 0.9)

    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    assert build_seed_face_gate(min_fraction=0.004)(Path("seed.png")) is True
    assert build_seed_face_gate(min_fraction=0.006)(Path("seed.png")) is False
    assert seen["path"] == Path("seed.png")


def test_detect_faces_raises_a_typed_error_for_a_missing_model(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FaceDetectionError, match="not found"):
        faces.detect_faces(tmp_path / "frame.png", model_path=tmp_path / "absent.onnx")


# --------------------------------------------------------------------------- #
# Recognition on top of detection (issue #49) -- the decision, no OpenCV,
# no models, no images. build_seed_face_gate's second (reference_photo)
# argument is optional, so every one of these can drive it with a plain
# monkeypatched recognize_face.
# --------------------------------------------------------------------------- #


def test_the_gate_keeps_the_pre_49_behaviour_when_no_reference_photo_is_given(monkeypatch):
    """The whole point of making ``reference_photo`` optional: every caller
    that has not been updated for issue #49 (and any test exercising #47
    alone) must see identical behaviour to before recognition existed --
    including never even importing/calling the recognition machinery."""

    def fake_detect(path, *, model_path=None, score_threshold=None):
        return FaceObservation(1, 0.01, 0.95)

    def boom(*_a, **_k):
        raise AssertionError("recognize_face must not be called when no reference_photo is given")

    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    monkeypatch.setattr(faces, "recognize_face", boom)
    assert build_seed_face_gate()(Path("seed.png")) is True


def test_the_gate_never_recognizes_a_frame_that_already_failed_detection(monkeypatch):
    """Recognition is a *third* floor on top of area and confidence, not a
    replacement for them -- a frame too small/unconfident to pass #47's gate
    must never even reach the (more expensive) recognition step."""

    def fake_detect(path, *, model_path=None, score_threshold=None):
        return FaceObservation(1, 0.0001, 0.95)  # far below DEFAULT_MIN_FACE_FRACTION

    def boom(*_a, **_k):
        raise AssertionError("recognize_face must not be called after a detection refusal")

    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    monkeypatch.setattr(faces, "recognize_face", boom)
    assert build_seed_face_gate()(Path("seed.png"), Path("ref.jpg")) is False


def test_the_gate_requires_the_similarity_floor_when_a_reference_photo_is_given(monkeypatch):
    def fake_detect(path, *, model_path=None, score_threshold=None):
        return FaceObservation(1, 0.01, 0.95)  # clears detection comfortably

    monkeypatch.setattr(faces, "detect_faces", fake_detect)

    monkeypatch.setattr(faces, "recognize_face", lambda *_a, **_k: DEFAULT_MIN_FACE_SIMILARITY)
    assert build_seed_face_gate()(Path("seed.png"), Path("ref.jpg")) is True

    monkeypatch.setattr(
        faces, "recognize_face", lambda *_a, **_k: DEFAULT_MIN_FACE_SIMILARITY - 0.001
    )
    assert build_seed_face_gate()(Path("seed.png"), Path("ref.jpg")) is False


def test_the_gate_passes_the_reference_photo_and_configured_similarity_through(monkeypatch):
    seen = {}

    def fake_detect(path, *, model_path=None, score_threshold=None):
        return FaceObservation(1, 0.01, 0.95)

    def fake_recognize(frame_path, reference_photo, **kwargs):
        seen["frame_path"] = frame_path
        seen["reference_photo"] = reference_photo
        return 0.5

    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    monkeypatch.setattr(faces, "recognize_face", fake_recognize)

    assert build_seed_face_gate(min_similarity=0.4)(Path("seed.png"), Path("ref.jpg")) is True
    assert build_seed_face_gate(min_similarity=0.6)(Path("seed.png"), Path("ref.jpg")) is False
    assert seen["frame_path"] == Path("seed.png")
    assert seen["reference_photo"] == Path("ref.jpg")


def test_the_gate_refuses_when_recognition_cannot_be_verified(monkeypatch):
    """'Cannot prove it is her' must read as 'do not chain', exactly like
    #47's detection-failure case -- a face that passed area and confidence is
    not enough on its own once a reference photo is supplied (issue #49)."""

    def fake_detect(path, *, model_path=None, score_threshold=None):
        return FaceObservation(1, 0.01, 0.95)

    def boom(*_a, **_k):
        raise FaceDetectionError("SFace model file is corrupt")

    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    monkeypatch.setattr(faces, "recognize_face", boom)
    assert build_seed_face_gate()(Path("seed.png"), Path("ref.jpg")) is False


def test_default_min_face_similarity_sits_inside_the_measured_gap():
    """The default is a measurement, not a round number -- see the module
    docstring's calibration table. Worst genuine pair measured: 0.353 (an
    extreme three-quarter profile). Worst impostor pair measured: 0.208 (a
    real chained frame's face against the *wrong* cast member's reference).
    If someone retunes this, this test says which evidence they are
    contradicting."""
    assert 0.208 < DEFAULT_MIN_FACE_SIMILARITY < 0.353


def test_recognize_face_raises_a_typed_error_for_a_missing_detection_model(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FaceDetectionError, match="YuNet"):
        faces.recognize_face(
            tmp_path / "frame.png",
            tmp_path / "ref.jpg",
            model_path=tmp_path / "absent.onnx",
        )


def test_recognize_face_raises_a_typed_error_for_a_missing_recognition_model(tmp_path):
    """The YuNet model (committed) resolves fine; the SFace model (never
    committed -- issue #49) does not, and that must be reported distinctly
    from a missing *detection* model so the degrade reason is legible in
    logs."""
    pytest.importorskip("cv2")
    if not faces.resolve_model_path().exists():
        pytest.skip("YuNet model not available")
    with pytest.raises(FaceDetectionError, match="SFace"):
        faces.recognize_face(
            tmp_path / "frame.png",
            tmp_path / "ref.jpg",
            recognition_model_path=tmp_path / "absent.onnx",
        )


def test_resolve_recognition_model_path_defaults_beside_the_package():
    default = faces.resolve_recognition_model_path()
    assert default.name == faces.RECOGNITION_MODEL_FILENAME
    assert default.parent.name == "models"


def test_resolve_recognition_model_path_respects_an_explicit_override(tmp_path):
    explicit = tmp_path / "somewhere-else.onnx"
    assert faces.resolve_recognition_model_path(explicit) == explicit


# --------------------------------------------------------------------------- #
# The detector (needs OpenCV + the model; skipped otherwise)
# --------------------------------------------------------------------------- #

CALIBRATION = Path(__file__).parent / "fixtures" / "seed_frames"


@pytest.mark.parametrize(
    ("frame", "should_chain"),
    [
        ("seed_faceless_chunk21.png", False),  # back to camera, drums obscuring
        ("seed_frontal_chunk15.png", True),  # a clean 2.88% frontal face
    ],
)
def test_the_detector_agrees_with_the_human_on_real_seed_frames(frame: str, should_chain: bool):
    """The two ends of the calibration set, kept as fixtures.

    Frames from the ``storms`` render (Derek's own likeness, consented for
    publication -- issue #51 #2) replaced the original lucky-ones fixtures of
    Dianne, which depicted a real person who had not been asked. Same
    regression, different source: a detector or model update that started
    passing the faceless frame would silently reintroduce #47's defect."""
    pytest.importorskip("cv2")
    path = CALIBRATION / frame
    if not path.exists() or not faces.resolve_model_path().exists():
        pytest.skip("calibration fixtures or YuNet model not available")

    assert build_seed_face_gate()(path) is should_chain


def test_a_low_confidence_detection_does_not_count_as_a_face():
    """Issue #47 follow-up. The gate's first threshold (0.6) was set *below*
    the observed evidence, which is the wrong direction for a false-positive
    guard: YuNet scored a blank patch of office wall at 0.669 and the gate
    chained from a frame showing only the back of the performer's head.

    Pinned as an observation rather than an image so the rule is explicit --
    confidence is a gate in its own right, not a formality on the way to
    measuring area. A big spurious box is still spurious."""
    spurious = FaceObservation(1, 0.01749, 0.669)
    assert spurious.largest_fraction > DEFAULT_MIN_FACE_FRACTION, (
        "the wall box was four times the area threshold -- area alone cannot catch it"
    )
    assert faces.DEFAULT_SCORE_THRESHOLD > 0.669


def test_the_score_threshold_sits_above_every_confirmed_false_positive():
    """0.669 is the confirmed spurious detection; 0.925 the lowest genuine
    face the gate approved. Retuning into that band means arguing with the
    v8 measurements."""
    assert 0.669 < faces.DEFAULT_SCORE_THRESHOLD <= 0.925


def test_the_detector_rejects_a_spurious_detection():
    """A real frame from the ``storms`` render: a pair of cymbals leaning in a
    doorway, no person in shot. At ``score_threshold=0.5`` YuNet reports a
    7.31%-of-frame "face" (the cymbal's rim) at score 0.501 -- the same shape
    as #47's original bug, a big spurious box a low threshold would accept.
    At the production default (0.9) the detector doesn't return it as a
    candidate at all, so this and the faceless frame both gate identically
    today; kept as a named fixture anyway, so a threshold regression toward
    0.5 has something concrete to fail against rather than only the
    hardcoded-observation tests below."""
    pytest.importorskip("cv2")
    path = CALIBRATION / "seed_spurious_chunk10.png"
    if not path.exists() or not faces.resolve_model_path().exists():
        pytest.skip("calibration fixtures or YuNet model not available")

    assert build_seed_face_gate()(path) is False
    assert build_seed_face_gate(score_threshold=0.5)(path) is True, (
        "if this starts failing, the spurious-detection regression case is gone -- "
        "pick a new frame from a fresh render rather than deleting the test"
    )


# --------------------------------------------------------------------------- #
# Recognition against a real model (issue #49). SFace is 38.7 MB and, unlike
# YuNet, is deliberately NOT committed to this repository (issue #51: not
# worth 165x YuNet's size for one gate in twelve on the material tested) --
# see faces.RECOGNITION_MODEL_SOURCE. Every test below skips cleanly when it
# is absent, which is the expected state of a fresh checkout; place the file
# at faces.resolve_recognition_model_path() to actually exercise them.
#
# The committed fixtures under tests/fixtures/seed_frames/ have no matching
# committed *reference photo* (issue #51 #2 forbids adding another real-person
# image just for this), so the "same person" case here is a frame matched
# against itself -- a real, non-trivial exercise of detection + alignCrop +
# embedding + cosine match end to end, just not an independent second photo.
# The full genuine/impostor separation is measured against material outside
# the repo (~/mvm-runs/) and recorded in the module docstring and
# docs/seed-face-recognition.md instead.
# --------------------------------------------------------------------------- #


def _recognition_ready() -> bool:
    return (
        faces.resolve_model_path().exists() and faces.resolve_recognition_model_path().exists()
    )


def _write_blank_image(path: Path) -> Path:
    """A synthetic, single-colour image with no face in it at all -- used
    instead of a real photo so "no face in the reference photo" doesn't need
    a new real-person fixture (issue #51 #2 forbids adding one just for this).
    Confirmed against the production detector: every candidate box YuNet
    proposes on a blank frame scores well under even the permissive
    reference-photo threshold (0.5), let alone the seed-frame one (0.9)."""
    import cv2
    import numpy as np

    image = np.full((480, 864, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def test_recognize_face_matches_a_frame_against_itself():
    pytest.importorskip("cv2")
    path = CALIBRATION / "seed_frontal_chunk15.png"
    if not path.exists() or not _recognition_ready():
        pytest.skip("calibration fixture or SFace/YuNet models not available")

    similarity = faces.recognize_face(path, path)
    assert similarity > 0.99, "a photo matched against itself should be ~1.0 cosine similarity"


def test_recognize_face_raises_when_the_reference_photo_has_no_face(tmp_path):
    """Recognition cannot build a comparison target from a reference photo
    with no detectable face in it, which must degrade the same way an
    unreadable photo would (issue #49) -- not raise something uncaught."""
    pytest.importorskip("cv2")
    frame = CALIBRATION / "seed_frontal_chunk15.png"
    if not frame.exists() or not _recognition_ready():
        pytest.skip("calibration fixture or SFace/YuNet models not available")
    reference = _write_blank_image(tmp_path / "blank.png")

    with pytest.raises(FaceDetectionError, match="reference photo"):
        faces.recognize_face(frame, reference)


def test_the_gate_chains_a_frame_verified_against_its_own_reference_photo():
    """End-to-end through the public gate API, real detector and recognizer,
    self-match standing in for "the correct cast member" per the module note
    above."""
    pytest.importorskip("cv2")
    path = CALIBRATION / "seed_frontal_chunk15.png"
    if not path.exists() or not _recognition_ready():
        pytest.skip("calibration fixture or SFace/YuNet models not available")

    assert build_seed_face_gate()(path, path) is True


def test_the_gate_refuses_a_frame_that_cannot_be_verified_against_a_faceless_reference(tmp_path):
    pytest.importorskip("cv2")
    frame = CALIBRATION / "seed_frontal_chunk15.png"
    if not frame.exists() or not _recognition_ready():
        pytest.skip("calibration fixture or SFace/YuNet models not available")
    reference = _write_blank_image(tmp_path / "blank.png")

    assert build_seed_face_gate()(frame, reference) is False


# --------------------------------------------------------------------------- #
# A FaceObservation in a boolean context (found using this API to measure a
# render: `if detect_faces(frame):` scored EVERY sampled frame as a hit,
# silently turning a face-presence measurement into a flat 100%).
# --------------------------------------------------------------------------- #


def test_face_observation_is_falsey_when_nothing_was_detected():
    assert not faces.FaceObservation(face_count=0, largest_fraction=0.0, score=0.0)


def test_face_observation_is_truthy_when_a_face_was_detected():
    assert faces.FaceObservation(face_count=1, largest_fraction=0.02, score=0.95)


def test_face_observation_truthiness_ignores_size_and_confidence():
    """`bool()` answers "was anything detected", nothing more. The area and
    confidence floors are `carries_identity`'s job, and conflating the two is
    how a gate ends up passing a blank wall (#47)."""
    tiny = faces.FaceObservation(face_count=1, largest_fraction=0.0001, score=0.1)
    assert tiny
    assert not tiny.carries_identity()
