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
