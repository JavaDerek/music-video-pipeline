"""Tests for the synthetic-cast consistency check (issue #56).

Three layers, matching ``tests/test_facescan.py``'s own shape:

* the pure report logic (:class:`ConsistencyReport.status`/``render``) --
  no cv2, no model, no image;
* :func:`check_consistency` and :func:`main` end to end, with injected
  detector/recognizer callables -- still no cv2, no model, no real image
  (the way ``tests/test_faces.py`` drives ``build_seed_face_gate`` with a
  monkeypatched ``detect_faces``/``recognize_face``);
* :func:`build_default_detector`/:func:`build_default_recognizer`, which
  really do call into :mod:`music_video_maker.faces` -- checked by
  monkeypatching ``faces.detect_faces``/``faces.recognize_face`` and
  confirming the closure forwards the right arguments, never against a real
  model file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from music_video_maker import castcheck, faces

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeDetector:
    """Maps an image path to a fixed :class:`faces.FaceObservation`, or
    raises :class:`faces.FaceDetectionError` if the path is mapped to one."""

    def __init__(self, verdicts: dict[Path, object]):
        self._verdicts = verdicts

    def __call__(self, image_path: Path) -> object:
        result = self._verdicts[image_path]
        if isinstance(result, BaseException):
            raise result
        return result


class _FakeRecognizer:
    """Maps an unordered pair of image paths to a fixed similarity, or an
    exception to raise."""

    def __init__(self, similarities: dict[frozenset, object]):
        self._similarities = similarities

    def __call__(self, image_a: Path, image_b: Path) -> float:
        result = self._similarities[frozenset((image_a, image_b))]
        if isinstance(result, BaseException):
            raise result
        return result


def _detected(verdict: str = "detected") -> faces.FaceObservation:
    if verdict == "detected":
        return faces.FaceObservation(face_count=1, largest_fraction=0.1, score=0.95)
    if verdict == "inconclusive":
        return faces.FaceObservation(
            face_count=0, largest_fraction=0.0, score=0.0,
            candidate_scores=(0.3,), inspection_floor=0.25,
        )
    if verdict == "absent":
        return faces.FaceObservation(
            face_count=0, largest_fraction=0.0, score=0.0,
            candidate_scores=(), inspection_floor=0.25,
        )
    raise ValueError(verdict)


# --------------------------------------------------------------------------- #
# check_consistency -- pure logic, fakes only
# --------------------------------------------------------------------------- #


def test_a_single_image_is_reported_insufficient_not_a_vacuous_pass(tmp_path):
    """The design doc's own rule: a set of one has no pairs, and must say so
    explicitly rather than passing vacuously."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    detector = _FakeDetector({img: _detected()})
    recognizer = _FakeRecognizer({})

    report = castcheck.check_consistency(
        "Nobody", [img], detector=detector, recognizer=recognizer
    )

    assert report.status == "insufficient"
    assert report.pairs == ()
    assert "one image" in report.render() or "no pairs" in report.render()


def test_two_images_above_the_floor_pass(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.5})

    report = castcheck.check_consistency(
        "Nobody", [a, b], floor=0.34, detector=detector, recognizer=recognizer
    )

    assert report.status == "pass"
    assert len(report.pairs) == 1
    assert report.pairs[0].similarity == 0.5
    assert report.minimum_pair.similarity == 0.5


def test_a_pair_below_the_floor_fails(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.1})

    report = castcheck.check_consistency(
        "Nobody", [a, b], floor=0.34, detector=detector, recognizer=recognizer
    )

    assert report.status == "fail"


def test_every_pair_is_scored_exactly_once(tmp_path):
    """Three images -> C(3,2) = 3 pairs, not 6 (no double-counting either
    direction of a pair, and no self-pairs)."""
    paths = [tmp_path / f"{n}.jpg" for n in "abc"]
    for p in paths:
        p.write_bytes(b"x")
    detector = _FakeDetector({p: _detected() for p in paths})
    recognizer = _FakeRecognizer(
        {frozenset((paths[i], paths[j])): 0.9 for i in range(3) for j in range(i + 1, 3)}
    )

    report = castcheck.check_consistency(
        "Nobody", paths, floor=0.34, detector=detector, recognizer=recognizer
    )

    assert len(report.pairs) == 3
    assert report.status == "pass"


def test_an_undetected_image_is_excluded_from_pairing_not_scored_as_zero(tmp_path):
    """Issue #93's lesson, applied here: a face too small/unconfident to
    detect must be surfaced, never averaged into the similarity numbers or
    treated as a failing 0.0."""
    good_a, good_b, bad = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "bad.jpg"
    for p in (good_a, good_b, bad):
        p.write_bytes(b"x")

    def boom(*_a, **_k):
        raise AssertionError("the undetected image must never reach the recognizer")

    detector = _FakeDetector(
        {good_a: _detected(), good_b: _detected(), bad: _detected("absent")}
    )
    recognizer = _FakeRecognizer({frozenset((good_a, good_b)): 0.9})

    report = castcheck.check_consistency(
        "Nobody", [good_a, good_b, bad], floor=0.34, detector=detector, recognizer=recognizer
    )

    assert len(report.pairs) == 1  # only good_a <-> good_b
    assert len(report.undetected) == 1
    assert report.undetected[0].provenance.path == bad
    assert report.undetected[0].verdict == "absent"
    # A real, passing pair exists, but the undetected image still demands
    # review -- surfaced, not silently overridden by a clean pair elsewhere.
    assert report.status == "needs_review"


def test_all_images_undetected_needs_review_not_insufficient(tmp_path):
    """'insufficient' means too few images were even supplied; two images
    that both failed detection is a concrete problem to go look at -- more
    images would not fix it, so this is 'needs_review', not 'insufficient'."""
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected("absent"), b: _detected("inconclusive")})
    recognizer = _FakeRecognizer({})

    report = castcheck.check_consistency(
        "Nobody", [a, b], detector=detector, recognizer=recognizer
    )

    assert report.status == "needs_review"
    assert report.minimum_pair is None
    assert len(report.undetected) == 2
    assert "RESULT: needs_review" in report.render()


def test_a_recognition_failure_is_reported_not_fatal(tmp_path):
    """One pair failing recognition (e.g. missing model) must not crash the
    whole report -- this project's 'one failure must not kill the run' rule,
    scaled to a single comparison."""
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer(
        {frozenset((a, b)): faces.FaceDetectionError("SFace model not found")}
    )

    report = castcheck.check_consistency(
        "Nobody", [a, b], detector=detector, recognizer=recognizer
    )

    assert report.pairs == ()
    assert len(report.unscored_pairs) == 1
    assert "SFace model not found" in report.unscored_pairs[0].reason
    assert report.status == "needs_review"
    text = report.render()
    assert "could not be scored" in text
    assert "SFace model not found" in text
    assert "RESULT: needs_review" in text


def test_an_image_that_cannot_even_be_stat_ed_still_gets_a_row(tmp_path):
    """issue #93's lesson taken to its edge case: even a path that vanishes
    out from under the check (or was never real) gets a provenance-stamped
    row, not a silent drop from the report."""
    missing = tmp_path / "does-not-exist.jpg"
    detector = _FakeDetector({missing: faces.FaceDetectionError("could not read image")})

    report = castcheck.check_consistency(
        "Nobody", [missing], detector=detector, recognizer=_FakeRecognizer({})
    )

    obs = report.observations[0]
    assert obs.provenance.path == missing
    assert obs.provenance.size_bytes == -1
    assert obs.provenance.mtime == "unknown"
    assert obs.verdict == "error"


def test_a_detection_error_is_recorded_as_its_own_verdict(tmp_path, caplog):
    """An unreadable image is not silently dropped from the report -- it
    gets its own row, with the exception message attached."""
    unreadable = tmp_path / "corrupt.jpg"
    unreadable.write_bytes(b"x")
    other = tmp_path / "ok.jpg"
    other.write_bytes(b"y")
    detector = _FakeDetector(
        {unreadable: faces.FaceDetectionError("could not read image"), other: _detected()}
    )
    recognizer = _FakeRecognizer({})

    with caplog.at_level(logging.WARNING):
        report = castcheck.check_consistency(
            "Nobody", [unreadable, other], detector=detector, recognizer=recognizer
        )

    error_obs = [o for o in report.observations if o.provenance.path == unreadable][0]
    assert error_obs.verdict == "error"
    assert "could not read image" in error_obs.detail
    assert error_obs in report.undetected


def test_provenance_is_stamped_per_image(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"some bytes")
    detector = _FakeDetector({img: _detected()})

    report = castcheck.check_consistency(
        "Nobody", [img], detector=detector, recognizer=_FakeRecognizer({})
    )

    obs = report.observations[0]
    assert obs.provenance.resolved_path == img.resolve()
    assert obs.provenance.size_bytes == len(b"some bytes")
    assert obs.provenance.mtime  # non-empty, ISO-ish


def test_render_includes_the_mode_collapse_caveat(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.5})

    report = castcheck.check_consistency(
        "Nobody", [a, b], floor=0.34, detector=detector, recognizer=recognizer
    )
    text = report.render()

    assert "mode-collapse" in text.lower() or "mode collapse" in text.lower()
    assert "real people" in text.lower() or "real photographs" in text.lower()
    assert "Nobody" in text
    assert "RESULT: pass" in text


# --------------------------------------------------------------------------- #
# main(), end to end, with injected factories -- no cv2
# --------------------------------------------------------------------------- #


def _factories_for(detector, recognizer):
    def detector_factory(**_kwargs):
        return detector

    def recognizer_factory(**_kwargs):
        return recognizer

    return detector_factory, recognizer_factory


def test_main_returns_zero_when_the_set_passes(tmp_path, capsys):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.9})
    detector_factory, recognizer_factory = _factories_for(detector, recognizer)

    rc = castcheck.main(
        ["Nobody", str(a), str(b)],
        detector_factory=detector_factory,
        recognizer_factory=recognizer_factory,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "RESULT: pass" in out


def test_main_returns_nonzero_when_the_set_fails(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.01})
    detector_factory, recognizer_factory = _factories_for(detector, recognizer)

    rc = castcheck.main(
        ["Nobody", str(a), str(b), "--floor", "0.34"],
        detector_factory=detector_factory,
        recognizer_factory=recognizer_factory,
    )

    assert rc == 1


def test_main_returns_nonzero_for_a_single_image(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"x")
    detector = _FakeDetector({a: _detected()})
    recognizer = _FakeRecognizer({})
    detector_factory, recognizer_factory = _factories_for(detector, recognizer)

    rc = castcheck.main(
        ["Nobody", str(a)],
        detector_factory=detector_factory,
        recognizer_factory=recognizer_factory,
    )

    assert rc == 1


def test_main_writes_a_report_to_the_requested_path(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    detector = _FakeDetector({a: _detected(), b: _detected()})
    recognizer = _FakeRecognizer({frozenset((a, b)): 0.9})
    detector_factory, recognizer_factory = _factories_for(detector, recognizer)

    out_path = tmp_path / "report.txt"
    rc = castcheck.main(
        ["Nobody", str(a), str(b), "--out", str(out_path)],
        detector_factory=detector_factory,
        recognizer_factory=recognizer_factory,
    )

    assert rc == 0
    text = out_path.read_text()
    assert "Nobody" in text
    assert "RESULT: pass" in text


# --------------------------------------------------------------------------- #
# build_default_detector / build_default_recognizer -- wired to faces.py,
# checked with a monkeypatch, never a real model file.
# --------------------------------------------------------------------------- #


def test_build_default_detector_uses_the_portrait_floor_by_default(monkeypatch):
    seen = {}

    def fake_detect_faces(path, *, model_path=None, score_threshold=None, inspect_floor=None):
        seen["path"] = path
        seen["score_threshold"] = score_threshold
        seen["inspect_floor"] = inspect_floor
        return _detected()

    monkeypatch.setattr(faces, "detect_faces", fake_detect_faces)
    detector = castcheck.build_default_detector()
    detector(Path("a.jpg"))

    assert seen["path"] == Path("a.jpg")
    assert seen["score_threshold"] == castcheck.DEFAULT_PORTRAIT_SCORE_THRESHOLD
    assert seen["inspect_floor"] == castcheck.DEFAULT_PORTRAIT_INSPECTION_FLOOR
    # Portrait floor, not faces.py's own seed-frame floor:
    assert castcheck.DEFAULT_PORTRAIT_SCORE_THRESHOLD < faces.DEFAULT_SCORE_THRESHOLD


def test_build_default_recognizer_uses_the_portrait_floor_on_both_sides(monkeypatch):
    seen = {}

    def fake_recognize_face(
        image_a, image_b, *, model_path=None, recognition_model_path=None, score_threshold=None
    ):
        seen["args"] = (image_a, image_b, score_threshold)
        return 0.42

    monkeypatch.setattr(faces, "recognize_face", fake_recognize_face)
    recognizer = castcheck.build_default_recognizer()
    result = recognizer(Path("a.jpg"), Path("b.jpg"))

    assert result == 0.42
    assert seen["args"] == (
        Path("a.jpg"), Path("b.jpg"), castcheck.DEFAULT_PORTRAIT_SCORE_THRESHOLD,
    )
