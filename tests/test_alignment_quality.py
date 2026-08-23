"""Tests for the alignment-quality evaluator (issue #35).

Fully offline: every fixture is a synthetic ``contracts.AlignmentResult``
built either from ``tests.harness.factories`` or directly from
``contracts`` objects. No model, no audio, no I/O -- matches
``evaluate_alignment_quality``'s own contract as a pure function.

The real-timing fixtures below are taken verbatim from issue #35's evidence
on "The Lucky Ones" (2026-08-08 alignment run).
"""

import math
import subprocess
from pathlib import Path

import pytest

from music_video_maker.alignment_quality import (
    FINDING_ISOLATED,
    FINDING_NO_VOCAL_ENERGY,
    FINDING_OUT_OF_ORDER,
    FINDING_OVERLAP,
    FINDING_SPLIT_LINE,
    FINDING_VOICE_IN_UNPLACED_GAP,
    FINDING_WPS_HIGH,
    FINDING_WPS_LOW,
    FINDING_ZERO_LENGTH,
    AlignmentQualityError,
    AlignmentQualityReport,
    LowConfidenceWord,
    Severity,
    evaluate_alignment_quality,
    format_summary,
    log_report,
    raise_if_blocking,
)
from music_video_maker.contracts import AlignmentResult, WordTiming
from tests.harness.factories import (
    make_aligned_segment,
    make_alignment_result_normal_song,
    make_alignment_result_with_gaps,
)

# --------------------------------------------------------------------------- #
# "The Lucky Ones" fixtures -- real timings from issue #35's evidence.
# --------------------------------------------------------------------------- #


def _corrected_transcript_segments():
    """Corrected-transcript excerpt: a zero-length segment followed, after a
    ~37s gap, by a hallucinated lyric in an instrumental passage."""
    return (
        make_aligned_segment(0, "I'm the lucky one.", 192.840, 194.120, "Dianne"),
        make_aligned_segment(1, "I'm the lucky one.", 194.920, 197.740, "Dianne"),
        make_aligned_segment(2, "I'm the lucky one.", 198.100, 198.580, "Dianne"),
        make_aligned_segment(3, "I'm", 198.580, 198.580, "Dianne"),  # zero-length
        make_aligned_segment(4, "the lucky one.", 235.680, 238.560, "Dianne"),  # hallucination
    )


def _original_lyrics_segments():
    """Original-committed-lyrics excerpt: one lyric line ("...We're the lucky
    ones.") split across a 38-second gap, then the same line correctly sung
    with 5 words crammed into 70ms."""
    seg_a = make_aligned_segment(
        24,
        "I know when it's people like you Hating people like us We're",
        194.920,
        200.040,
        "Dianne",
    )
    seg_b = make_aligned_segment(25, "the lucky ones.", 238.160, 243.000, "Dianne")
    fast_words = (
        WordTiming(word="We", start=247.230, end=247.244),
        WordTiming(word="'re", start=247.244, end=247.258),
        WordTiming(word="the", start=247.258, end=247.272),
        WordTiming(word="lucky", start=247.272, end=247.286),
        WordTiming(word="ones.", start=247.286, end=247.300),
    )
    seg_c = make_aligned_segment(
        26, "We're the lucky ones.", 247.230, 247.300, "Dianne", words=fast_words
    )
    return (seg_a, seg_b, seg_c)


def test_zero_length_segment_is_flagged_critical():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)

    report = evaluate_alignment_quality(result)

    zero_length = [f for f in report.findings if f.code == FINDING_ZERO_LENGTH]
    assert any(f.segment_index == 3 for f in zero_length)
    assert all(f.severity is Severity.CRITICAL for f in zero_length)


def test_hallucinated_lyric_in_instrumental_passage_is_flagged_isolated():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)

    report = evaluate_alignment_quality(result)

    isolated = [f for f in report.findings if f.code == FINDING_ISOLATED]
    assert any(f.segment_index == 4 for f in isolated)
    assert all(f.severity is Severity.CRITICAL for f in isolated)


def test_five_words_in_seventy_milliseconds_is_flagged_implausible_rate():
    result = AlignmentResult(segments=_original_lyrics_segments(), track_duration=250.03)

    report = evaluate_alignment_quality(result)

    too_fast = [f for f in report.findings if f.code == FINDING_WPS_HIGH]
    assert any(f.segment_index == 26 for f in too_fast)
    assert all(f.severity is Severity.CRITICAL for f in too_fast)


def test_lyric_line_split_across_thirty_eight_second_gap_is_flagged():
    result = AlignmentResult(segments=_original_lyrics_segments(), track_duration=250.03)

    report = evaluate_alignment_quality(result)

    split = [f for f in report.findings if f.code == FINDING_SPLIT_LINE]
    assert any(
        f.segment_index == 25 and f.related_segment_index == 24 for f in split
    )


# --------------------------------------------------------------------------- #
# No false positives on well-behaved alignments.
# --------------------------------------------------------------------------- #


def test_normal_song_produces_a_clean_report():
    result = make_alignment_result_normal_song()

    report = evaluate_alignment_quality(result)

    assert report.findings == ()
    assert report.is_clean
    assert report.max_severity is None


def test_legitimate_instrumental_gap_is_not_flagged_as_split_or_isolated():
    # 30s gap between segments -- a real bridge/solo, not a bug. This project
    # explicitly treats large instrumental gaps as normal (see CLAUDE.md's
    # "chunk timeline must cover the whole track" invariant); the quality
    # checker must agree.
    result = make_alignment_result_with_gaps()

    report = evaluate_alignment_quality(result)

    assert not [f for f in report.findings if f.code == FINDING_SPLIT_LINE]
    assert not [f for f in report.findings if f.code == FINDING_ISOLATED]


def test_an_instrumental_intro_before_the_first_vocal_is_not_flagged_isolated():
    # The very first segment starting late (a long intro) is normal and must
    # never be treated as "isolated" -- see the module docstring's
    # documented asymmetry.
    segments = (
        make_aligned_segment(0, "here we go now", 45.0, 48.0, "Dianne"),
        make_aligned_segment(1, "singing every word", 49.0, 52.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=60.0)

    report = evaluate_alignment_quality(result)

    assert not [f for f in report.findings if f.code == FINDING_ISOLATED]


# --------------------------------------------------------------------------- #
# Out-of-order / overlapping segments.
# --------------------------------------------------------------------------- #


def test_overlapping_segments_are_flagged_critical():
    segments = (
        make_aligned_segment(0, "one two three", 0.0, 5.0, "Dianne"),
        make_aligned_segment(1, "four five six", 3.0, 8.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=10.0)

    report = evaluate_alignment_quality(result)

    overlaps = [f for f in report.findings if f.code == FINDING_OVERLAP]
    assert len(overlaps) == 1
    assert overlaps[0].severity is Severity.CRITICAL
    assert overlaps[0].segment_index == 1
    assert overlaps[0].related_segment_index == 0


def test_out_of_order_segments_are_flagged_critical():
    segments = (
        make_aligned_segment(0, "second in time", 10.0, 15.0, "Dianne"),
        make_aligned_segment(1, "first in time", 0.0, 4.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=20.0)

    report = evaluate_alignment_quality(result)

    out_of_order = [f for f in report.findings if f.code == FINDING_OUT_OF_ORDER]
    assert len(out_of_order) == 1
    assert out_of_order[0].severity is Severity.CRITICAL


# --------------------------------------------------------------------------- #
# Implausibly slow delivery (the "too low" half of the words-per-second check).
# --------------------------------------------------------------------------- #


def test_implausibly_slow_segment_is_flagged_warning():
    # One word held for 20 seconds -- far below MIN_WORDS_PER_SECOND.
    segments = (make_aligned_segment(0, "held", 0.0, 20.0, "Dianne"),)
    result = AlignmentResult(segments=segments, track_duration=25.0)

    report = evaluate_alignment_quality(result)

    too_slow = [f for f in report.findings if f.code == FINDING_WPS_LOW]
    assert len(too_slow) == 1
    assert too_slow[0].severity is Severity.WARNING


# --------------------------------------------------------------------------- #
# Per-word confidence (issue #35 item 4).
# --------------------------------------------------------------------------- #


def test_low_confidence_word_is_flagged_when_supplied():
    result = make_alignment_result_normal_song()
    low_conf = (
        LowConfidenceWord(segment_index=0, word="halls", start=3.0, end=3.5, probability=0.05),
    )

    report = evaluate_alignment_quality(result, low_confidence_words=low_conf)

    from music_video_maker.alignment_quality import FINDING_LOW_CONFIDENCE

    findings = [f for f in report.findings if f.code == FINDING_LOW_CONFIDENCE]
    assert len(findings) == 1
    # Never CRITICAL on its own: forced alignment places words it has weak
    # evidence for, so one low probability locates a place to look, it does
    # not prove a misplacement. See the whole-track check for the escalation.
    assert findings[0].severity is Severity.WARNING


def test_absent_confidence_produces_no_low_confidence_findings():
    # Default: no low_confidence_words supplied at all -- must not crash or
    # fabricate findings out of nothing.
    result = make_alignment_result_normal_song()

    report = evaluate_alignment_quality(result)

    from music_video_maker.alignment_quality import FINDING_LOW_CONFIDENCE

    assert not [f for f in report.findings if f.code == FINDING_LOW_CONFIDENCE]


# --------------------------------------------------------------------------- #
# Report / severity plumbing.
# --------------------------------------------------------------------------- #


def test_report_counts_by_severity_and_at_least():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)

    report = evaluate_alignment_quality(result)

    counts = report.counts_by_severity()
    assert counts[Severity.CRITICAL] >= 1
    assert report.max_severity is Severity.CRITICAL
    assert report.at_least(Severity.CRITICAL) == tuple(
        f for f in report.findings if f.severity is Severity.CRITICAL
    )


def test_format_summary_reports_segment_and_finding_counts():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)
    report = evaluate_alignment_quality(result)

    summary = format_summary(report)

    assert "5 segment(s)" in summary
    assert "critical" in summary


def test_log_report_logs_summary_at_info_and_findings_at_warning_or_error(caplog):
    import logging

    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)
    report = evaluate_alignment_quality(result)

    with caplog.at_level(logging.INFO, logger="music_video_maker.alignment_quality"):
        log_report(report, context="track.wav")

    assert any("Alignment quality" in r.message for r in caplog.records)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_clean_report_logs_only_the_info_summary(caplog):
    import logging

    report = evaluate_alignment_quality(make_alignment_result_normal_song())

    with caplog.at_level(logging.INFO, logger="music_video_maker.alignment_quality"):
        log_report(report)

    assert all(r.levelno == logging.INFO for r in caplog.records)


# --------------------------------------------------------------------------- #
# strict_alignment / refusal.
# --------------------------------------------------------------------------- #


def test_raise_if_blocking_is_a_noop_when_strict_is_false():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)
    report = evaluate_alignment_quality(result)

    raise_if_blocking(report, strict=False)  # must not raise


def test_raise_if_blocking_raises_alignment_quality_error_when_strict_and_critical():
    result = AlignmentResult(segments=_corrected_transcript_segments(), track_duration=250.03)
    report = evaluate_alignment_quality(result)

    try:
        raise_if_blocking(report, strict=True)
    except AlignmentQualityError as exc:
        assert exc.report is report
        assert exc.threshold is Severity.CRITICAL
    else:
        raise AssertionError("expected AlignmentQualityError")


def test_raise_if_blocking_is_a_noop_on_a_clean_report_even_when_strict():
    report = evaluate_alignment_quality(make_alignment_result_normal_song())

    raise_if_blocking(report, strict=True)  # must not raise -- nothing to refuse


def test_empty_alignment_result_is_a_clean_report():
    result = AlignmentResult(segments=(), track_duration=0.0)

    report = evaluate_alignment_quality(result)

    assert isinstance(report, AlignmentQualityReport)
    assert report.is_clean
    assert report.segment_count == 0


# --------------------------------------------------------------------------- #
# Per-word confidence: aggregated, not one finding per word.
#
# Calibrated against a real measurement (doris, 2026-08-08). On the corrected
# "Lucky Ones" transcript stable-ts reported a median word probability of 0.76
# with 24% of words below 0.10 -- a usable alignment carrying four genuine,
# locatable defects. Emitting one CRITICAL per low word buried those four
# behind hundreds of ERROR lines and would have made --strict-alignment refuse
# every real song.
# --------------------------------------------------------------------------- #


def _words(*probabilities: float, segment_index: int = 0) -> tuple[LowConfidenceWord, ...]:
    return tuple(
        LowConfidenceWord(
            segment_index=segment_index,
            word=f"w{i}",
            start=float(i),
            end=float(i) + 0.4,
            probability=p,
        )
        for i, p in enumerate(probabilities)
    )


def test_many_low_words_in_one_segment_produce_one_finding_not_many():
    from music_video_maker.alignment_quality import FINDING_LOW_CONFIDENCE

    report = evaluate_alignment_quality(
        make_alignment_result_normal_song(),
        low_confidence_words=_words(0.01, 0.02, 0.03, 0.04, 0.05),
    )

    findings = [f for f in report.findings if f.code == FINDING_LOW_CONFIDENCE]
    assert len(findings) == 1
    assert "5 word(s)" in findings[0].message


def test_a_real_songs_confidence_profile_does_not_raise_a_critical():
    """The measured profile: median 0.76, 24% below 0.10. This alignment has
    real defects elsewhere, but its *confidence* must not be what condemns
    it -- otherwise --strict-alignment refuses every song ever rendered."""
    from music_video_maker.alignment_quality import FINDING_CONFIDENCE_COLLAPSE

    profile = _words(*([0.02] * 24 + [0.5] * 26 + [0.9] * 50))
    report = evaluate_alignment_quality(
        make_alignment_result_normal_song(), low_confidence_words=profile
    )

    assert not [f for f in report.findings if f.code == FINDING_CONFIDENCE_COLLAPSE]
    assert not [f for f in report.findings if f.severity is Severity.CRITICAL]


def test_confidence_collapsing_across_the_whole_track_is_critical():
    """The case the aggregate exists for: not "some words are hard" but "the
    aligner had no acoustic support anywhere" -- wrong audio, wrong lyrics,
    or wrong language."""
    from music_video_maker.alignment_quality import FINDING_CONFIDENCE_COLLAPSE

    report = evaluate_alignment_quality(
        make_alignment_result_normal_song(),
        low_confidence_words=_words(*([0.01] * 90 + [0.95] * 10)),
    )

    collapse = [f for f in report.findings if f.code == FINDING_CONFIDENCE_COLLAPSE]
    assert len(collapse) == 1
    assert collapse[0].severity is Severity.CRITICAL
    assert "90%" in collapse[0].message


def test_the_collapse_check_needs_a_real_sample_before_it_fires():
    """A caller that hands over only the words it already considers suspect
    would otherwise read as 100% starved on a perfectly good track."""
    from music_video_maker.alignment_quality import FINDING_CONFIDENCE_COLLAPSE

    report = evaluate_alignment_quality(
        make_alignment_result_normal_song(), low_confidence_words=_words(0.01, 0.02, 0.01)
    )

    assert not [f for f in report.findings if f.code == FINDING_CONFIDENCE_COLLAPSE]


# --------------------------------------------------------------------------- #
# Level 3 counterpoint (issue #33): concurrent segments are evaluated, but on
# their own terms -- they overlap the spine by definition, and their timings
# are derived from it rather than measured.
# --------------------------------------------------------------------------- #


def _spine_result():
    return (
        make_aligned_segment(0, "There was a time", 0.0, 5.0, "Dianne"),
        make_aligned_segment(1, "when I hoped", 5.0, 10.0, "Dianne"),
    )


def _counterpoint_result(*concurrent):
    from music_video_maker.alignment import CounterpointAlignmentResult

    return CounterpointAlignmentResult(
        segments=_spine_result(),
        track_duration=250.0,
        concurrent_segments=tuple(concurrent),
    )


def _concurrent(index, text, start, end, character="Marcus"):
    from music_video_maker.alignment import ConcurrentSegment

    return ConcurrentSegment(
        index=index,
        text=text,
        start=start,
        end=end,
        characters=(character,),
        spine_segment_indices=(0, 1),
        stream_index=1,
        block_index=0,
    )


def test_counterpoint_overlapping_the_spine_is_not_an_overlap_defect():
    # The whole point of a [simultaneously] block: these seconds are shared.
    result = _counterpoint_result(
        _concurrent(2, "I know when it's people like you", 0.0, 5.0),
        _concurrent(3, "Hating people like me", 5.0, 10.0),
    )

    report = evaluate_alignment_quality(result)

    assert not [f for f in report.findings if f.code == FINDING_OVERLAP]
    assert not [f for f in report.findings if f.code == FINDING_OUT_OF_ORDER]


def test_report_counts_counterpoint_segments_separately_from_the_spine():
    result = _counterpoint_result(
        _concurrent(2, "I know when it's people like you", 0.0, 5.0),
        _concurrent(3, "Hating people like me", 5.0, 10.0),
    )

    report = evaluate_alignment_quality(result)

    assert report.segment_count == 2  # spine only -- that is the timeline
    assert report.counterpoint_segment_count == 2
    summary = format_summary(report)
    assert "2 counterpoint segment(s)" in summary


def test_summary_is_unchanged_when_there_is_no_counterpoint():
    result = AlignmentResult(segments=_spine_result(), track_duration=250.0)

    report = evaluate_alignment_quality(result)

    assert report.counterpoint_segment_count == 0
    assert "counterpoint" not in format_summary(report)


def test_counterpoint_crammed_into_its_spine_span_is_flagged_but_never_blocking():
    from music_video_maker.alignment_quality import FINDING_COUNTERPOINT_RATE

    # Nine words distributed across 0.5s: the author wrote more counterpoint
    # than the spine span can hold. Worth surfacing -- but the timings are
    # derived, not measured, so it must not refuse a strict run.
    result = _counterpoint_result(
        _concurrent(2, "I know when it's people like you right now", 0.0, 0.5)
    )

    report = evaluate_alignment_quality(result)

    rate = [f for f in report.findings if f.code == FINDING_COUNTERPOINT_RATE]
    assert len(rate) == 1
    assert rate[0].severity is Severity.WARNING
    assert "derived" in rate[0].message
    raise_if_blocking(report, strict=True)  # must not raise


def test_counterpoint_with_a_degenerate_span_is_flagged():
    from music_video_maker.alignment_quality import FINDING_COUNTERPOINT_SPAN

    result = _counterpoint_result(_concurrent(2, "I'm the lucky one", 4.0, 4.0))

    report = evaluate_alignment_quality(result)

    span = [f for f in report.findings if f.code == FINDING_COUNTERPOINT_SPAN]
    assert len(span) == 1
    assert span[0].severity is Severity.WARNING


def test_counterpoint_findings_are_logged_with_the_rest(caplog):
    import logging

    from music_video_maker.alignment_quality import FINDING_COUNTERPOINT_RATE

    result = _counterpoint_result(
        _concurrent(2, "I know when it's people like you right now", 0.0, 0.5)
    )
    report = evaluate_alignment_quality(result)

    with caplog.at_level(logging.INFO, logger="music_video_maker.alignment_quality"):
        log_report(report, context="track.wav")

    assert any(FINDING_COUNTERPOINT_RATE in r.message for r in caplog.records)
    assert any("counterpoint segment(s)" in r.message for r in caplog.records)


def test_a_counterpoint_segment_with_no_words_is_not_a_finding():
    # An empty derived segment carries no rate signal; it must not invent one.
    result = _counterpoint_result(_concurrent(2, "", 0.0, 5.0))

    report = evaluate_alignment_quality(result)

    assert report.findings == ()
    assert report.counterpoint_segment_count == 1


# --------------------------------------------------------------------------- #
# Vocal-energy check (issue #71): "is there a voice where this lyric was
# placed?" -- the one check in this module that looks at the master audio
# itself rather than reasoning about timing and text.
#
# All of these (bar the one marked ``@pytest.mark.integration`` at the very
# end) inject a fake ffmpeg runner and never touch a real ffmpeg binary or a
# real audio file -- same convention as assembly.py's ``SubprocessRunner``.
# The fake simulates both the decode-once-to-mono-16kHz call (identified by
# the absence of ``-ss`` in its argv) and the per-window ``astats`` calls
# (identified by ``-ss``/``-af``), keyed by each window's *intended* HF
# share so tests can express "this window sounds like X" directly rather
# than reverse-engineering dB arithmetic by hand.
# --------------------------------------------------------------------------- #


def _share_to_db_pair(share: float, total_db: float = -15.0) -> tuple[float, float]:
    """Inverse of the module's own ``10 ** ((hf_db - total_db) / 10)``
    share computation: picks a highpass dB level that reproduces ``share``
    exactly for a given ``total_db``."""
    hf_db = total_db + 10 * math.log10(share)
    return total_db, hf_db


class _FakeVocalEnergyRunner:
    """``profile`` maps a window's ``start`` (an exact float used by the
    test's segments) to its intended HF share. The whole-file decode call
    (no ``-ss`` in argv) reports success unless ``decode_ok`` is False.
    ``fail_starts`` makes specific per-window ``astats`` calls fail (exit 1)
    to test that one bad window doesn't take down the others.

    Issue #80's gap check tiles windows inside the (large, by construction)
    gaps between these tests' tiny synthetic segments, at ``-ss`` starts no
    pre-#80 test ever anticipated. Falling back to ``DEFAULT_SHARE`` for any
    unlisted start (rather than a bare ``self.profile[start]`` that would
    raise ``KeyError``) keeps every test written before #80 existed passing
    unchanged: the default is far enough below any of these fixtures'
    baselines that it never trips GAP_VOCAL_ENERGY_RATIO_THRESHOLD."""

    DEFAULT_SHARE = 0.001

    def __init__(self, profile, *, decode_ok=True, fail_starts=frozenset()):
        self.profile = profile
        self.decode_ok = decode_ok
        self.fail_starts = fail_starts
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if "-ss" not in args:
            rc = 0 if self.decode_ok else 1
            return subprocess.CompletedProcess(args, returncode=rc, stdout=b"", stderr=b"")
        start = float(args[args.index("-ss") + 1])
        if start in self.fail_starts:
            return subprocess.CompletedProcess(
                args, returncode=1, stdout=b"", stderr=b"synthetic failure"
            )
        af = args[args.index("-af") + 1]
        highpass = "highpass" in af
        entry = self.profile.get(start, self.DEFAULT_SHARE)
        # An entry may be a bare share (uses the -15.0dB default full-band
        # level) or a (share, total_db) pair -- the latter is how issue #80's
        # "silent tail" trap is expressed: a high share at a level nowhere
        # near the placed-segment baseline, which GAP_LEVEL_DROP_DB exists to
        # gate out regardless of how high the share looks.
        share, total_db = entry if isinstance(entry, tuple) else (entry, -15.0)
        total_db, hf_db = _share_to_db_pair(share, total_db)
        db = hf_db if highpass else total_db
        stderr = f"[Parsed_astats_0] Overall\n[Parsed_astats_0] RMS level dB: {db:.4f}\n".encode()
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=stderr)


def _energy_segments():
    return (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 20.0, 26.0, "Dianne"),
        make_aligned_segment(2, "real three", 40.0, 46.0, "Dianne"),
        make_aligned_segment(3, "phantom", 60.0, 66.0, "Dianne"),
    )


def _write_stub_audio(tmp_path):
    audio = tmp_path / "master.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")  # only existence is checked before ffmpeg runs
    return audio


def test_audio_path_none_is_a_pure_noop_and_never_touches_ffmpeg():
    def _boom(args):
        raise AssertionError("ffmpeg must not be invoked when audio_path is None")

    result = make_alignment_result_normal_song()

    report = evaluate_alignment_quality(result, audio_path=None, ffmpeg_runner=_boom)

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]


def test_segment_with_no_vocal_energy_is_flagged_critical(tmp_path):
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0015}  # phantom well under baseline
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert len(hits) == 1
    assert hits[0].segment_index == 3
    assert hits[0].severity is Severity.CRITICAL


def test_uniform_vocal_energy_produces_no_finding(tmp_path):
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.022}  # all four alike
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]


def test_too_few_other_segments_skips_the_check(tmp_path):
    # Only one "other" segment exists to calibrate against -- not enough of
    # a baseline to trust, even though the ratio would otherwise trip.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "phantom", 20.0, 26.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=30.0)
    profile = {0.0: 0.02, 20.0: 0.0005}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result,
        audio_path=_write_stub_audio(tmp_path),
        ffmpeg_runner=runner,
        vocal_energy_min_baseline_segments=3,
    )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]


def test_a_very_short_segment_is_excluded_from_evaluation_and_baseline(tmp_path):
    # A ~5-word-in-70ms style near-instant segment can't produce a
    # trustworthy spectral estimate -- it must be skipped both as something
    # this check evaluates AND as a contributor to other segments' baseline.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 20.0, 26.0, "Dianne"),
        make_aligned_segment(2, "real three", 40.0, 46.0, "Dianne"),
        make_aligned_segment(3, "blip", 60.0, 60.4, "Dianne"),  # 0.4s: below the 1.0s floor
    )
    result = AlignmentResult(segments=segments, track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0001}  # would trip if ever evaluated
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    ss_starts = {float(c[c.index("-ss") + 1]) for c in runner.calls if "-ss" in c}
    assert 60.0 not in ss_starts  # never even measured


def test_finding_still_fires_when_two_of_five_other_segments_are_also_phantom(tmp_path):
    # A track where MANY placed segments are themselves hallucinated must
    # not blind the check: the median baseline stays anchored to the real
    # segments as long as they aren't the minority.
    segments = (
        make_aligned_segment(0, "phantom a", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "phantom b", 20.0, 26.0, "Dianne"),
        make_aligned_segment(2, "real one", 40.0, 46.0, "Dianne"),
        make_aligned_segment(3, "real two", 60.0, 66.0, "Dianne"),
        make_aligned_segment(4, "real three", 80.0, 86.0, "Dianne"),
        make_aligned_segment(5, "phantom under test", 100.0, 106.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=110.0)
    profile = {
        0.0: 0.001,
        20.0: 0.001,
        40.0: 0.02,
        60.0: 0.02,
        80.0: 0.02,
        100.0: 0.001,
    }
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = {f.segment_index for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY}
    assert 5 in hits


def test_missing_ffmpeg_binary_skips_gracefully(tmp_path, monkeypatch, caplog):
    import logging

    from music_video_maker import alignment_quality as aq

    monkeypatch.setattr(aq.shutil, "which", lambda _name: None)
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)

    def _boom(args):
        raise AssertionError("ffmpeg must not be invoked when the binary is absent")

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment_quality"):
        report = evaluate_alignment_quality(
            result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=_boom
        )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert any("ffmpeg" in r.message.lower() for r in caplog.records)


def test_missing_audio_file_skips_gracefully(caplog):
    import logging

    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)

    def _boom(args):
        raise AssertionError("ffmpeg must not be invoked when the audio file is absent")

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment_quality"):
        report = evaluate_alignment_quality(
            result, audio_path="/nonexistent/path/master.wav", ffmpeg_runner=_boom
        )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_decode_failure_skips_gracefully(tmp_path, caplog):
    import logging

    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    runner = _FakeVocalEnergyRunner({}, decode_ok=False)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment_quality"):
        report = evaluate_alignment_quality(
            result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
        )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_one_windows_measurement_failure_does_not_crash_the_others(tmp_path):
    # 5 segments so that even with segment 2's window failing to measure,
    # segment 3 (the phantom under test) still has 3 valid "other" shares to
    # calibrate against.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 20.0, 26.0, "Dianne"),
        make_aligned_segment(2, "real three (fails to measure)", 40.0, 46.0, "Dianne"),
        make_aligned_segment(3, "phantom", 60.0, 66.0, "Dianne"),
        make_aligned_segment(4, "real four", 80.0, 86.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=90.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0015, 80.0: 0.02}
    runner = _FakeVocalEnergyRunner(profile, fail_starts={40.0})

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert len(hits) == 1
    assert hits[0].segment_index == 3
    # segment 2's own failed window must not itself be (falsely) flagged
    assert not any(f.segment_index == 2 for f in hits)


def test_ffmpeg_launch_oserror_on_the_decode_step_skips_gracefully(tmp_path, caplog):
    import logging

    def _raises(args):
        raise OSError("ffmpeg binary vanished mid-run")

    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment_quality"):
        report = evaluate_alignment_quality(
            result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=_raises
        )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_pure_digital_silence_window_is_skipped_not_flagged(tmp_path):
    # ffmpeg's astats reports "-inf" dB for a window with literally zero
    # signal -- a ratio against/of that is meaningless, so the window must
    # be excluded (not treated as evidence in either direction), same as a
    # decode failure.
    class _SilentRunner:
        def __init__(self):
            self.calls: list[list[str]] = []

        def __call__(self, args):
            args = list(args)
            self.calls.append(args)
            if "-ss" not in args:
                return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")
            start = float(args[args.index("-ss") + 1])
            if start == 60.0:
                stderr = b"[Parsed_astats_0] RMS level dB: -inf\n"
                return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=stderr)
            total_db, hf_db = _share_to_db_pair(0.02)
            highpass = "highpass" in args[args.index("-af") + 1]
            db = hf_db if highpass else total_db
            stderr = f"[Parsed_astats_0] RMS level dB: {db:.4f}\n".encode()
            return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=stderr)

    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    runner = _SilentRunner()

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    # The silent segment can't be measured, so it's neither flagged itself
    # nor does it crash evaluation of the others.
    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]


def test_an_unexpected_internal_error_is_caught_logged_and_degrades_gracefully(
    tmp_path, monkeypatch, caplog
):
    import logging

    from music_video_maker import alignment_quality as aq

    def _boom(*_args, **_kwargs):
        raise ValueError("unexpected")

    monkeypatch.setattr(aq.statistics, "median", _boom)
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0015}
    runner = _FakeVocalEnergyRunner(profile)

    with caplog.at_level(logging.ERROR, logger="music_video_maker.alignment_quality"):
        report = evaluate_alignment_quality(
            result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
        )

    assert not [f for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY]
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_strict_alignment_refuses_on_a_vocal_energy_finding(tmp_path):
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0015}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    try:
        raise_if_blocking(report, strict=True)
    except AlignmentQualityError as exc:
        blocking_codes = {f.code for f in exc.report.at_least(Severity.CRITICAL)}
        assert FINDING_NO_VOCAL_ENERGY in blocking_codes
    else:
        raise AssertionError("expected AlignmentQualityError")


def test_vocal_energy_finding_is_counted_in_the_info_summary(tmp_path):
    result = AlignmentResult(segments=_energy_segments(), track_duration=70.0)
    profile = {0.0: 0.02, 20.0: 0.021, 40.0: 0.019, 60.0: 0.0015}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert "1 critical" in format_summary(report)


def test_real_deathless_phantom_measurement_is_flagged(tmp_path):
    """Locks in the actual calibration data (2026-08-15, doris master decoded
    to mono 16kHz, see VOCAL_ENERGY_RATIO_THRESHOLD's comment): four real
    sung windows from "Deathless" -- including its quietest measured real
    vocal window, 386.292s-391.458s ("Now the island is no longer...") -- and
    the closing 'deathless, Forevermore!' phantom `small` placed at
    495.375s-501.958s, 12s into the fadeout where issue #71 was filed."""
    segments = (
        make_aligned_segment(
            0, "High upon a mountain, past Volokov's mill,", 54.583, 61.875, "Dianne"
        ),
        make_aligned_segment(
            1, "Kashay prays for an endless night", 133.083, 139.667, "Dianne"
        ),
        make_aligned_segment(
            2, "Now the island is no longer, so he will always be", 386.292, 391.458, "Dianne"
        ),
        make_aligned_segment(3, "the war. Born to be deathless.", 474.208, 479.375, "Dianne"),
        make_aligned_segment(4, "deathless, Forevermore!", 495.375, 501.958, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=513.917)
    profile = {
        54.583: 0.0057,
        133.083: 0.0460,
        386.292: 0.0026,  # the quietest real vocal window measured on this track
        474.208: 0.0182,
        495.375: 0.0009,  # the phantom issue #71 documents
    }
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = {f.segment_index for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY}
    assert hits == {4}  # the phantom, and only the phantom -- not the quiet-but-real segment


@pytest.mark.integration
def test_vocal_energy_integration_real_ffmpeg(tmp_path):
    """Real ffmpeg, tiny synthesized audio -- deselected via ``-m`` in CI,
    same convention as assembly.py's equivalent. Three "vocal-like" windows
    (a tone plus broadband noise, so there's real energy above 3.4kHz) and
    one "phantom" window (a bare low tone, no noise -- nothing above
    3.4kHz) built with ffmpeg's own lavfi sources, so nothing here depends
    on a real recording."""
    import shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        pytest.skip("ffmpeg not installed on this machine")

    master = tmp_path / "synth_master.wav"
    args = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
        "-f", "lavfi", "-i", "anoisesrc=color=white:duration=6:amplitude=0.05",
        "-f", "lavfi", "-i", "sine=frequency=320:duration=6",
        "-f", "lavfi", "-i", "anoisesrc=color=white:duration=6:amplitude=0.05",
        "-f", "lavfi", "-i", "sine=frequency=280:duration=6",
        "-f", "lavfi", "-i", "anoisesrc=color=white:duration=6:amplitude=0.05",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
        "-filter_complex",
        "[0][1]amix=inputs=2:duration=first[r1];"
        "[2][3]amix=inputs=2:duration=first[r2];"
        "[4][5]amix=inputs=2:duration=first[r3];"
        "[r1][r2][r3][6]concat=n=4:v=0:a=1[out]",
        "-map", "[out]", str(master),
    ]
    proc = subprocess.run(args, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
        make_aligned_segment(3, "phantom", 18.0, 24.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=24.0)

    report = evaluate_alignment_quality(result, audio_path=master)  # real ffmpeg_runner

    hits = {f.segment_index for f in report.findings if f.code == FINDING_NO_VOCAL_ENERGY}
    assert hits == {3}


# --------------------------------------------------------------------------- #
# Voice-in-unplaced-gap check (issue #80): #71's mirror. "Is there a voice
# where you placed NOTHING", instead of "is there a voice where you placed
# this lyric". Same fake-runner convention as the section above; every gap
# in these fixtures is deliberately sized as a whole multiple of
# GAP_WINDOW_S (4.0s) so window boundaries land on exact floats and the
# fake runner's dict lookups need no floating-point tolerance.
# --------------------------------------------------------------------------- #


def test_a_high_share_window_inside_a_gap_at_a_healthy_level_is_flagged_warning(tmp_path):
    # Three contiguous baseline segments (0-18s, share ~0.02), an 8s gap
    # (18-26s, two 4s windows), then a closing segment so the gap under test
    # isn't the trailing tail. The second window (22-26s) sounds like a
    # voice; the first (18-22s) is left at the fake's quiet default.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
        make_aligned_segment(3, "real four", 26.0, 32.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=32.0)
    profile = {0.0: 0.02, 6.0: 0.021, 12.0: 0.019, 26.0: 0.02, 22.0: 0.08}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]
    assert len(hits) == 1
    assert hits[0].severity is Severity.WARNING
    assert hits[0].segment_index is None
    assert hits[0].start == pytest.approx(18.0)
    assert hits[0].end == pytest.approx(26.0)
    assert "22.000" in hits[0].message or "22.0" in hits[0].message


def test_high_share_gated_out_by_silent_level_produces_no_finding_the_silent_tail_trap(tmp_path):
    # Same layout as above, but the high-share window is also very quiet
    # (-67.8dB, the real trailing-tail level this issue's calibration
    # measured) -- well below the placed-segment level floor. A share ratio
    # computed from two near-nothing measurements is meaningless, so this
    # window must be gated out rather than flagged, regardless of how high
    # its share looks.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
        make_aligned_segment(3, "real four", 26.0, 32.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=32.0)
    profile = {
        0.0: 0.02,
        6.0: 0.021,
        12.0: 0.019,
        26.0: 0.02,
        22.0: (0.08, -67.8),  # high share, but effectively silent
    }
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert not [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]


def test_a_gap_shorter_than_the_minimum_duration_is_never_measured(tmp_path):
    # 18.0 -> 20.0 is only 2.0s, below GAP_MIN_DURATION_S (4.0s): it must not
    # even be enumerated, let alone measured.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
        make_aligned_segment(3, "real four", 20.0, 26.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=26.0)
    profile = {0.0: 0.02, 6.0: 0.021, 12.0: 0.019, 20.0: 0.02}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert not [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]
    ss_starts = {float(c[c.index("-ss") + 1]) for c in runner.calls if "-ss" in c}
    assert 18.0 not in ss_starts  # the short gap was never even measured


def test_fewer_than_the_baseline_minimum_placed_segments_skips_the_gap_check_entirely(tmp_path):
    # Only two placed segments exist on this track -- below
    # VOCAL_ENERGY_BASELINE_MIN_SEGMENTS -- even though the gap between them
    # is large and a window in it would score high if it were ever measured.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 20.0, 26.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=26.0)
    profile = {0.0: 0.02, 20.0: 0.021, 10.0: 0.09}  # 10.0 would trip it if ever measured
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    assert not [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]
    ss_starts = {float(c[c.index("-ss") + 1]) for c in runner.calls if "-ss" in c}
    assert ss_starts == {0.0, 20.0}  # only the two segments -- the gap was never touched


def test_a_long_gap_with_several_high_windows_yields_exactly_one_finding(tmp_path):
    # 18.0 -> 34.0 is 16s, tiled into four 4s windows (18, 22, 26, 30). Three
    # of the four score high; this must still produce exactly one finding
    # for the gap, not one per window.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
        make_aligned_segment(3, "real four", 34.0, 40.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=40.0)
    profile = {
        0.0: 0.02,
        6.0: 0.021,
        12.0: 0.019,
        34.0: 0.02,
        22.0: 0.09,
        26.0: 0.07,
        30.0: 0.08,
    }
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]
    assert len(hits) == 1
    assert hits[0].start == pytest.approx(18.0)
    assert hits[0].end == pytest.approx(34.0)
    # the peak window (22-26s, share 0.09) is the one named in the message
    assert "22.000" in hits[0].message or "22.0" in hits[0].message


def test_the_trailing_span_after_the_last_segment_is_measured_and_can_fire(tmp_path):
    # Three contiguous baseline segments end at 18.0s; the track runs to
    # 26.0s, an 8s trailing tail with no segment after it at all. This is
    # exactly the span #79/#71's isolated_segment check would never look at
    # past the last segment -- it must still be tiled and measured here.
    segments = (
        make_aligned_segment(0, "real one", 0.0, 6.0, "Dianne"),
        make_aligned_segment(1, "real two", 6.0, 12.0, "Dianne"),
        make_aligned_segment(2, "real three", 12.0, 18.0, "Dianne"),
    )
    result = AlignmentResult(segments=segments, track_duration=26.0)
    profile = {0.0: 0.02, 6.0: 0.021, 12.0: 0.019, 22.0: 0.09}
    runner = _FakeVocalEnergyRunner(profile)

    report = evaluate_alignment_quality(
        result, audio_path=_write_stub_audio(tmp_path), ffmpeg_runner=runner
    )

    hits = [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]
    assert len(hits) == 1
    assert hits[0].start == pytest.approx(18.0)
    assert hits[0].end == pytest.approx(26.0)
    ss_starts = {float(c[c.index("-ss") + 1]) for c in runner.calls if "-ss" in c}
    assert 18.0 in ss_starts and 22.0 in ss_starts  # the trailing tail WAS measured


def test_gap_check_audio_path_none_is_a_pure_noop_and_never_touches_ffmpeg():
    # Regression: an existing pure call must stay unchanged by #80 -- no I/O
    # at all when audio_path is None, same guarantee #71 already gave.
    def _boom(args):
        raise AssertionError("ffmpeg must not be invoked when audio_path is None")

    result = make_alignment_result_normal_song()

    report = evaluate_alignment_quality(result, audio_path=None, ffmpeg_runner=_boom)

    assert not [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]


# --------------------------------------------------------------------------- #
# Real "Deathless" master, real ffmpeg -- skipped when the master (deliberately
# never committed to this repo) is absent from the machine running the tests.
# --------------------------------------------------------------------------- #

_DEATHLESS_MASTER = Path.home() / "mvm-runs" / "deathless" / "audio" / "master.wav"


@pytest.mark.skipif(
    not _DEATHLESS_MASTER.exists(),
    reason="real 'Deathless' master audio is not present on this machine (deliberately not "
    "committed to the repo -- see CLAUDE.md's 'Everything committed here is intended to "
    "become public')",
)
def test_real_deathless_gap_check_flags_the_495s_silent_tower_and_not_the_fadeout_tail():
    """Locks in issue #80's own real measurement: the aligner never placed a
    lyric across 288.65s-297.41s, and the viewer who watched the render said
    "4:55 he is back on non-snowy tower, with singing playing, but he isn't
    singing" -- 295.0s falls inside that exact gap. Segment spans are the
    ones bracketing the interesting region (issue #80's own list); this does
    NOT run whisper -- alignment in a test is not allowed.

    The two assertions this test exists for: the gap containing 295.0s
    fires, and the trailing tail after the last placed segment (501.630s ->
    the track's own 512.080s end) does not -- that tail is the silent-tail
    trap GAP_LEVEL_DROP_DB exists to gate out.
    """
    spans = [
        (281.570, 286.870),
        (287.010, 288.650),
        (297.410, 300.670),
        (301.260, 307.110),
        (307.110, 312.090),
        (314.390, 320.970),
        (473.990, 475.490),
        (475.490, 477.050),
        (481.150, 483.930),
        (485.580, 486.550),
        (498.730, 501.630),
    ]
    segments = tuple(
        make_aligned_segment(i, f"lyric segment {i}", start, end, "Dianne")
        for i, (start, end) in enumerate(spans)
    )
    result = AlignmentResult(segments=segments, track_duration=512.080)

    report = evaluate_alignment_quality(result, audio_path=_DEATHLESS_MASTER)  # real ffmpeg

    gap_hits = [f for f in report.findings if f.code == FINDING_VOICE_IN_UNPLACED_GAP]

    contains_295 = [f for f in gap_hits if f.start <= 295.0 <= f.end]
    assert contains_295, f"expected a gap finding containing 295.0s; got {gap_hits}"
    assert contains_295[0].severity is Severity.WARNING
    assert contains_295[0].start == pytest.approx(288.650, abs=0.01)
    assert contains_295[0].end == pytest.approx(297.410, abs=0.01)

    trailing_tail_start = 501.630
    assert not any(f.start >= trailing_tail_start - 0.01 for f in gap_hits), (
        f"the trailing tail after the last placed segment must not fire; got {gap_hits}"
    )
