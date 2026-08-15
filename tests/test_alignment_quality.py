"""Tests for the alignment-quality evaluator (issue #35).

Fully offline: every fixture is a synthetic ``contracts.AlignmentResult``
built either from ``tests.harness.factories`` or directly from
``contracts`` objects. No model, no audio, no I/O -- matches
``evaluate_alignment_quality``'s own contract as a pure function.

The real-timing fixtures below are taken verbatim from issue #35's evidence
on "The Lucky Ones" (2026-08-08 alignment run).
"""

from __future__ import annotations

from music_video_maker.alignment_quality import (
    FINDING_ISOLATED,
    FINDING_OUT_OF_ORDER,
    FINDING_OVERLAP,
    FINDING_SPLIT_LINE,
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
