"""Tests for Stage 1 forced alignment (issue #3).

Fully offline: the stable-ts model is always injected as a fake/mock that
duck-types ``model.align()``'s return shape. ``stable-ts``/``torch`` are not
installed in this environment on purpose -- this file (and
``music_video_maker.alignment`` itself) must import cleanly without them.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from music_video_maker.alignment import CounterpointAlignmentResult, align
from music_video_maker.alignment_quality import AlignmentQualityError
from music_video_maker.contracts import AlignedSegment, AlignmentResult, LyricLine
from music_video_maker.lyrics import CounterpointStream, parse_lyrics, parse_lyrics_text
from tests.harness.factories import (
    LYRICS_PLAIN_PATH,
    LYRICS_TAGGED_PATH,
    make_cast_dict,
    make_raw_stablets_result,
    write_silent_wav,
)

CAST = make_cast_dict()
DEFAULT_LEAD = "Dianne"


def _word(word: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=word, start=start, end=end, probability=0.99)


def _segment(text: str, start: float, end: float) -> SimpleNamespace:
    words = text.split()
    span = (end - start) / max(len(words), 1)
    word_objs = [_word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    return SimpleNamespace(text=text, start=start, end=end, words=word_objs)


def _raw_result(*segments: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(segments=list(segments), language="en")


def _fake_model(raw_result: SimpleNamespace) -> MagicMock:
    model = MagicMock()
    model.align.return_value = raw_result
    return model


def test_align_calls_model_align_with_vad_and_regroup_kwargs(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    align(audio, lines, model=model)

    model.align.assert_called_once()
    _, kwargs = model.align.call_args
    assert kwargs["suppress_silence"] is True
    assert kwargs["regroup"] is True
    assert kwargs["language"] == "en"


def test_align_never_calls_transcribe(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    align(audio, lines, model=model)

    model.transcribe.assert_not_called()


def test_align_output_shape_and_character_carry_through(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    result = align(audio, lines, model=model)

    assert isinstance(result, AlignmentResult)
    assert len(result.segments) == 3
    for segment in result.segments:
        assert isinstance(segment, AlignedSegment)
        assert segment.character == "Dianne"
        assert segment.words
        assert segment.end > segment.start

    assert result.segments[0].text == "walking through the empty halls tonight"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 6.5


def test_align_uses_probed_wav_track_duration(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=32.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    result = align(audio, lines, model=model)

    assert result.track_duration == 32.0
    assert result.voiced_duration < result.track_duration


def test_align_track_duration_falls_back_to_last_segment_end_when_probe_is_shorter(tmp_path):
    # WAV reports 5s but the (fake) alignment produced a segment ending later --
    # must not silently truncate the timeline other stages depend on.
    audio = write_silent_wav(tmp_path / "master.wav", seconds=5.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:1]
    model = _fake_model(_raw_result(_segment("walking through the empty halls tonight", 0.0, 6.5)))

    result = align(audio, lines, model=model)

    assert result.track_duration == 6.5


def test_align_character_switches_with_source_lyric_lines(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=15.0)
    lines = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)[:2]
    assert [line.character for line in lines] == ["Dianne", "Marcus"]

    raw = _raw_result(
        _segment(lines[0].text, 0.0, 4.0),
        _segment(lines[1].text, 5.0, 9.0),
    )
    model = _fake_model(raw)

    result = align(audio, lines, model=model)

    assert [segment.character for segment in result.segments] == ["Dianne", "Marcus"]


def test_align_handles_long_instrumental_gap_without_crashing(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=55.0)
    lines = [
        LyricLine(index=0, text="count the days until the sun comes back", characters=("Dianne",)),
        LyricLine(index=1, text="give the silence room to breathe", characters=("Marcus",)),
    ]
    raw = _raw_result(
        _segment(lines[0].text, 0.0, 5.0),
        _segment(lines[1].text, 41.0, 47.0),  # 36s instrumental gap
    )
    model = _fake_model(raw)

    result = align(audio, lines, model=model)

    assert len(result.segments) == 2
    gap = result.segments[1].start - result.segments[0].end
    assert gap == pytest.approx(36.0)
    assert result.track_duration == 55.0


def test_align_skips_empty_lyric_lines(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=10.0)
    lines = [
        LyricLine(index=0, text="", characters=("Dianne",)),
        LyricLine(index=1, text="   ", characters=("Dianne",)),
        LyricLine(index=2, text="only real line", characters=("Dianne",)),
    ]
    model = _fake_model(_raw_result(_segment("only real line", 0.0, 3.0)))

    align(audio, lines, model=model)

    args, kwargs = model.align.call_args
    text_blob = args[1]
    assert text_blob == "only real line"


def test_align_all_empty_lyric_lines_short_circuits_without_calling_model(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=10.0)
    lines = [
        LyricLine(index=0, text="", characters=("Dianne",)),
        LyricLine(index=1, text="   ", characters=("Dianne",)),
    ]
    model = _fake_model(_raw_result())

    result = align(audio, lines, model=model)

    model.align.assert_not_called()
    assert result.segments == ()
    assert result.track_duration == 10.0


def test_align_tag_stripped_text_never_reaches_the_model(tmp_path):
    # Integration with issue #6: parse tagged.txt, then verify the text blob
    # handed to model.align() contains no bracket characters at all.
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_TAGGED_PATH, CAST, DEFAULT_LEAD)
    raw = _raw_result(
        *(_segment(line.text, float(i * 5), float(i * 5 + 4)) for i, line in enumerate(lines))
    )
    model = _fake_model(raw)

    align(audio, lines, model=model)

    args, _ = model.align.call_args
    text_blob = args[1]
    assert "[" not in text_blob
    assert "]" not in text_blob


def test_align_segment_with_no_words_carries_forward_last_character(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=15.0)
    lines = [LyricLine(index=0, text="only real words", characters=("Marcus",))]
    wordless = SimpleNamespace(text="", start=4.0, end=4.2, words=[])
    raw = _raw_result(_segment("only real words", 0.0, 3.5), wordless)
    model = _fake_model(raw)

    result = align(audio, lines, model=model)

    assert len(result.segments) == 2
    assert result.segments[0].character == "Marcus"
    assert result.segments[1].character == "Marcus"
    assert result.segments[1].words == ()


def test_align_zero_segments_from_model_falls_back_to_probed_duration(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=12.0)
    lines = [LyricLine(index=0, text="something sung", characters=("Dianne",))]
    model = _fake_model(_raw_result())

    result = align(audio, lines, model=model)

    assert result.segments == ()
    assert result.track_duration == 12.0


# Both tests below used to assert the *absence* of the [align] extra --
# `"stable_whisper" not in sys.modules`, and align() raising because the
# import fails. CI installs `.[dev]` only, so they were green there forever
# while failing on every machine that can actually run Stage 1. Worse, the
# first one did not merely fail with the extra installed: align() fell
# through to _load_model() and ran a REAL whisper alignment inside a unit
# test, downloading ~145 MB of weights on a machine with a cold cache --
# against CONTRIBUTING.md's "no network" rule.
#
# The property each one is really about does not depend on the environment,
# so neither test does now.


def test_align_without_injected_model_raises_clear_error_when_stable_ts_missing(
    tmp_path, monkeypatch
):
    """Simulated absence, not observed absence. A None in sys.modules makes
    `import stable_whisper` raise ImportError on demand, so this exercises the
    missing-extra path identically whether or not stable-ts is installed --
    and never loads a model."""
    monkeypatch.setitem(sys.modules, "stable_whisper", None)

    audio = write_silent_wav(tmp_path / "master.wav", seconds=10.0)
    lines = [LyricLine(index=0, text="hello there", characters=("Dianne",))]

    with pytest.raises(RuntimeError, match="align.*extra|extra.*align"):
        align(audio, lines)


def test_module_import_does_not_pull_in_stable_whisper_or_torch():
    """The real property is that alignment.py's stable-ts import is LAZY, so
    every other stage stays light. Asserting `not in sys.modules` in-process
    only tested that nothing else had imported it yet. A fresh interpreter
    tests the import itself, and gives the same answer either way."""
    probe = (
        "import sys; import music_video_maker.alignment as a; "
        "print('stable_whisper' in sys.modules, 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False", out.stdout


def test_probe_duration_degrades_gracefully_for_missing_file(caplog):
    from music_video_maker.alignment import _probe_duration

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        duration = _probe_duration(Path("/nonexistent/does-not-exist.wav"))

    assert duration == 0.0
    assert any("Could not probe" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# Issue #35: stable-ts warnings routed into the log, and quality reporting.
# --------------------------------------------------------------------------- #


def _fake_model_that_warns(raw_result: SimpleNamespace, message: str) -> MagicMock:
    """A fake model whose .align() emits a real UserWarning before returning,
    the same way stable-ts's own non_whisper/alignment.py does
    (`warnings.warn(f'{fail_segs}/{len(...)} segments failed to align.')`) --
    no real stable-ts involved."""
    model = MagicMock()

    def _align(*_args, **_kwargs):
        warnings.warn(message, UserWarning, stacklevel=2)
        return raw_result

    model.align.side_effect = _align
    return model


def test_align_routes_stable_ts_warnings_into_the_logger(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model_that_warns(make_raw_stablets_result(), "1/27 segments failed to align.")

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        align(audio, lines, model=model)

    matching = [r for r in caplog.records if "failed to align" in r.message]
    assert matching, "expected the stable-ts UserWarning to be routed through logger.warning"
    assert str(audio) in matching[0].message
    assert "3" in matching[0].message  # line count context


def test_align_does_not_swallow_multiple_warnings(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    raw_result = make_raw_stablets_result()
    model = MagicMock()

    def _align(*_args, **_kwargs):
        warnings.warn("first warning", UserWarning, stacklevel=2)
        warnings.warn("second warning", UserWarning, stacklevel=2)
        return raw_result

    model.align.side_effect = _align

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        align(audio, lines, model=model)

    messages = "\n".join(r.message for r in caplog.records)
    assert "first warning" in messages
    assert "second warning" in messages


def test_align_logs_alignment_quality_summary_at_info_on_every_run(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    with caplog.at_level(logging.INFO, logger="music_video_maker.alignment_quality"):
        align(audio, lines, model=model)

    assert any("Alignment quality" in r.message for r in caplog.records)


def test_align_strict_alignment_false_by_default_does_not_raise_on_bad_alignment(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=250.0)
    lines = [LyricLine(index=0, text="I'm the lucky one", characters=("Dianne",))]
    # A zero-length segment -- definitely a critical finding -- but
    # strict_alignment defaults to False, so this must not raise.
    raw = _raw_result(_segment("I'm the lucky one", 5.0, 5.0))
    model = _fake_model(raw)

    result = align(audio, lines, model=model)

    assert result.segments[0].start == result.segments[0].end == 5.0


def test_align_strict_alignment_true_raises_on_critical_finding(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=250.0)
    lines = [LyricLine(index=0, text="I'm the lucky one", characters=("Dianne",))]
    raw = _raw_result(_segment("I'm the lucky one", 5.0, 5.0))  # zero-length -> critical
    model = _fake_model(raw)

    with pytest.raises(AlignmentQualityError):
        align(audio, lines, model=model, strict_alignment=True)


def test_align_extracts_word_confidence_from_raw_stablets_words():
    # stable_whisper.result.WordTiming carries an Optional[float]
    # `probability`; verified by reading the installed package directly
    # (stable_whisper/result.py). Confirm align() reads it defensively and
    # feeds it into the quality report rather than ignoring it.
    from music_video_maker.alignment import _extract_word_confidences

    raw = make_raw_stablets_result()

    confidences = _extract_word_confidences(raw)

    assert confidences  # make_raw_stablets_result's words all carry probability=0.99
    assert all(c.probability == pytest.approx(0.99) for c in confidences)


def test_extract_word_confidences_skips_words_with_no_probability_attribute():
    from music_video_maker.alignment import _extract_word_confidences

    word_without_probability = SimpleNamespace(word="hello", start=0.0, end=0.5)
    segment = SimpleNamespace(text="hello", start=0.0, end=0.5, words=[word_without_probability])
    raw = SimpleNamespace(segments=[segment])

    confidences = _extract_word_confidences(raw)

    assert confidences == ()


# --------------------------------------------------------------------------- #
# Level 3 counterpoint (issue #33): the spine is aligned; concurrent streams
# inherit the spine's span. stable-ts only ever sees the spine text.
# --------------------------------------------------------------------------- #

COUNTERPOINT_LYRICS = """\
[simultaneously]
[Dianne]
spine line one
spine line two
[Marcus]
under one
under two
[/simultaneously]
"""


def _counterpoint_doc():
    return parse_lyrics_text(COUNTERPOINT_LYRICS, CAST, DEFAULT_LEAD)


def _spine_raw():
    # Four spine words, 0-10s: "spine line one" 0->6, "spine line two" 6->10.
    return _raw_result(
        _segment("spine line one", 0.0, 6.0),
        _segment("spine line two", 6.0, 10.0),
    )


def test_counterpoint_text_never_reaches_the_model(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    model = _fake_model(_spine_raw())

    align(audio, doc, model=model)

    args, _ = model.align.call_args
    text_blob = args[1]
    assert text_blob == "spine line one\nspine line two"
    assert "under" not in text_blob


def test_counterpoint_is_picked_up_from_the_document_without_a_caller_change(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    assert isinstance(result, CounterpointAlignmentResult)
    assert isinstance(result, AlignmentResult)  # every existing consumer still works
    assert len(result.concurrent_segments) == 2


def test_counterpoint_can_also_be_passed_explicitly(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    model = _fake_model(_spine_raw())

    result = align(audio, doc.lines, model=model, counterpoint=doc.counterpoint)

    assert len(result.concurrent_segments) == 2


def test_no_counterpoint_still_returns_a_plain_alignment_result(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    model = _fake_model(make_raw_stablets_result())

    result = align(audio, lines, model=model)

    assert type(result) is AlignmentResult


def test_counterpoint_segments_span_the_spine_and_are_distributed_proportionally(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    first, second = result.concurrent_segments
    # The spine's block runs 0 -> 10s; the counterpoint has 4 words in two
    # equal lines, so each line gets half the span.
    assert first.start == pytest.approx(0.0)
    assert first.end == pytest.approx(5.0)
    assert second.start == pytest.approx(5.0)
    assert second.end == pytest.approx(10.0)
    assert [s.text for s in result.concurrent_segments] == ["under one", "under two"]


def test_counterpoint_segments_carry_word_timings_inside_their_span(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    for segment in result.concurrent_segments:
        assert [w.word for w in segment.words] == segment.text.split()
        assert segment.words[0].start == pytest.approx(segment.start)
        assert segment.words[-1].end == pytest.approx(segment.end)
        for prev, cur in zip(segment.words, segment.words[1:], strict=False):
            assert cur.start >= prev.end - 1e-9


def test_counterpoint_segments_carry_their_own_voice_not_the_spines(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    assert all(s.characters == ("Marcus",) for s in result.concurrent_segments)
    assert all(s.character == "Marcus" for s in result.concurrent_segments)
    assert all(s.character == "Dianne" for s in result.segments)


def test_counterpoint_segment_indices_never_collide_with_spine_indices(tmp_path):
    # AlignedSegment index space is a bare int shared by both tuples -- the
    # recurring index-space bug in this project. Concurrent segments continue
    # the spine's numbering rather than restarting at 0.
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    spine_indices = {s.index for s in result.segments}
    concurrent_indices = {s.index for s in result.concurrent_segments}
    assert spine_indices == {0, 1}
    assert concurrent_indices == {2, 3}
    assert not (spine_indices & concurrent_indices)


def test_counterpoint_segments_record_the_spine_segments_they_overlap(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    first, second = result.concurrent_segments
    assert first.spine_segment_indices == (0,)  # 0 -> 5s sits inside spine segment 0
    assert second.spine_segment_indices == (0, 1)  # 5 -> 10s straddles both
    assert all(s.stream_index == 1 for s in result.concurrent_segments)
    assert all(s.block_index == 0 for s in result.concurrent_segments)


def test_lookup_helpers_answer_in_their_own_index_and_time_spaces(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)
    first, second = result.concurrent_segments

    # Spine segment 0 (0 -> 6s) is under both; spine segment 1 (6 -> 10s)
    # only under the second.
    assert result.concurrent_for_segment(0) == (first, second)
    assert result.concurrent_for_segment(1) == (second,)
    assert result.concurrent_for_segment(99) == ()

    assert result.concurrent_in_span(0.0, 1.0) == (first,)
    assert result.concurrent_in_span(4.0, 6.0) == (first, second)
    assert result.concurrent_in_span(11.0, 12.0) == ()


def test_counterpoint_does_not_inflate_voiced_duration(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    # voiced_duration is time on the master track with singing on it; the
    # counterpoint is *the same seconds*, so counting it twice would report
    # 20s of vocal in a 10s span.
    assert result.voiced_duration == pytest.approx(10.0)


def test_counterpoint_referencing_an_unknown_spine_line_logs_and_skips(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    stray = CounterpointStream(
        block_index=0,
        stream_index=1,
        characters=("Marcus",),
        texts=("under one",),
        spine_line_indices=(97, 98),  # not LyricLine indices in this document
    )
    model = _fake_model(_spine_raw())

    with caplog.at_level(logging.ERROR, logger="music_video_maker.alignment"):
        result = align(audio, doc.lines, model=model, counterpoint=(stray,))

    assert result.concurrent_segments == ()
    assert any("spine line" in r.message.lower() for r in caplog.records)


def test_counterpoint_over_a_zero_length_spine_span_is_skipped_with_a_warning(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    collapsed = _raw_result(
        _segment("spine line one", 4.0, 4.0),
        _segment("spine line two", 4.0, 4.0),
    )
    model = _fake_model(collapsed)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        result = align(audio, doc, model=model)

    assert result.concurrent_segments == ()
    assert any("counterpoint" in r.message.lower() for r in caplog.records)


def test_counterpoint_falls_back_to_segment_bounds_when_words_are_missing(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    wordless = _raw_result(
        SimpleNamespace(text="spine line one", start=0.0, end=6.0, words=[]),
        SimpleNamespace(text="spine line two", start=6.0, end=10.0, words=[]),
    )
    model = _fake_model(wordless)

    result = align(audio, doc, model=model)

    assert len(result.concurrent_segments) == 2
    assert result.concurrent_segments[0].start == pytest.approx(0.0)
    assert result.concurrent_segments[-1].end == pytest.approx(10.0)


def test_counterpoint_with_no_aligned_segments_at_all_logs_and_drops(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_raw_result())  # the model returned nothing

    with caplog.at_level(logging.ERROR, logger="music_video_maker.alignment"):
        result = align(audio, _counterpoint_doc(), model=model)

    assert result.concurrent_segments == ()
    assert any("no aligned segment covering" in r.message for r in caplog.records)


def test_counterpoint_stream_with_no_words_logs_and_is_skipped(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    wordless_stream = CounterpointStream(
        block_index=0,
        stream_index=1,
        characters=("Marcus",),
        texts=("", "   "),
        spine_line_indices=(0, 1),
    )
    model = _fake_model(_spine_raw())

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        result = align(audio, doc.lines, model=model, counterpoint=(wordless_stream,))

    assert result.concurrent_segments == ()
    assert any("has no words" in r.message for r in caplog.records)


def test_spine_span_of_a_wordless_spine_line_is_refused_not_invented(caplog):
    # Defensive: align() filters empty lines out before this point, so this
    # can only be reached by a direct caller -- it must still refuse rather
    # than return a degenerate span.
    from music_video_maker.alignment import _spine_span

    stream = CounterpointStream(
        block_index=0,
        stream_index=1,
        characters=("Marcus",),
        texts=("under one",),
        spine_line_indices=(0,),
    )

    with caplog.at_level(logging.WARNING, logger="music_video_maker.alignment"):
        span = _spine_span(stream, (), (), {0: (0, 0)})

    assert span is None
    assert any("no words" in r.message for r in caplog.records)


def test_counterpoint_quality_report_still_logs_a_summary(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    with caplog.at_level(logging.INFO, logger="music_video_maker.alignment_quality"):
        align(audio, _counterpoint_doc(), model=model)

    summaries = [r.message for r in caplog.records if "Alignment quality" in r.message]
    assert summaries
    assert "counterpoint" in summaries[0].lower()


def test_all_segments_merges_both_tuples_in_index_order(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    model = _fake_model(_spine_raw())

    result = align(audio, _counterpoint_doc(), model=model)

    assert [s.index for s in result.all_segments] == [0, 1, 2, 3]
    assert [s.character for s in result.all_segments] == [
        "Dianne",
        "Dianne",
        "Marcus",
        "Marcus",
    ]


def test_blank_text_inside_a_counterpoint_stream_is_skipped_not_timed(tmp_path):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    doc = _counterpoint_doc()
    stream = CounterpointStream(
        block_index=0,
        stream_index=1,
        characters=("Marcus",),
        texts=("under one", "   ", "under two"),
        spine_line_indices=(0, 1),
    )
    model = _fake_model(_spine_raw())

    result = align(audio, doc.lines, model=model, counterpoint=(stream,))

    assert [s.text for s in result.concurrent_segments] == ["under one", "under two"]


def test_counterpoint_with_no_lyric_lines_at_all_logs_and_returns_empty(tmp_path, caplog):
    audio = write_silent_wav(tmp_path / "master.wav", seconds=10.0)
    stream = CounterpointStream(
        block_index=0,
        stream_index=1,
        characters=("Marcus",),
        texts=("under one",),
        spine_line_indices=(0,),
    )
    model = _fake_model(_raw_result())

    with caplog.at_level(logging.ERROR, logger="music_video_maker.alignment"):
        result = align(audio, [], model=model, counterpoint=(stream,))

    model.align.assert_not_called()
    assert result.segments == ()
    assert any("no spine lyric" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Authored alignment overrides (issue #42): pin a segment's span when the
# model provably mis-places it and no single model is right everywhere.
# --------------------------------------------------------------------------- #


def test_alignment_override_retimes_the_segment_and_its_words(tmp_path):
    from music_video_maker.contracts import AlignmentOverride

    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    override = AlignmentOverride(
        segment_index=1, start=8.0, end=12.5,
        reason="model piles the phrase early; stem energy and a second model agree on 8-12.5",
    )

    result = align(
        audio, lines, model=_fake_model(make_raw_stablets_result()), overrides=(override,)
    )

    seg = result.segments[1]
    assert seg.start == pytest.approx(8.0)
    assert seg.end == pytest.approx(12.5)
    # Words spread proportionally across the new span, chronological.
    starts = [w.start for w in seg.words]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(8.0)
    assert seg.words[-1].end == pytest.approx(12.5)
    # Neighbors untouched.
    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[2].start == pytest.approx(13.0)


def test_alignment_override_refuses_an_overlap_with_a_neighbor(tmp_path):
    from music_video_maker.alignment import AlignmentOverrideError
    from music_video_maker.contracts import AlignmentOverride

    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    override = AlignmentOverride(
        segment_index=1, start=5.0, end=12.0, reason="would overlap segment 0 (ends 6.5)"
    )

    with pytest.raises(AlignmentOverrideError):
        align(audio, lines, model=_fake_model(make_raw_stablets_result()), overrides=(override,))


def test_alignment_override_refuses_an_unknown_segment_index(tmp_path):
    from music_video_maker.alignment import AlignmentOverrideError
    from music_video_maker.contracts import AlignmentOverride

    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    override = AlignmentOverride(segment_index=9, start=8.0, end=12.5, reason="no such segment")

    with pytest.raises(AlignmentOverrideError):
        align(audio, lines, model=_fake_model(make_raw_stablets_result()), overrides=(override,))


def test_alignment_overrides_validate_the_final_sequence_not_each_step(tmp_path):
    """Two overrides that are coherent TOGETHER must apply regardless of
    order, even when either alone would overlap a not-yet-moved neighbor.
    Validation runs on the final sequence."""
    from music_video_maker.contracts import AlignmentOverride

    audio = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    lines = parse_lyrics(LYRICS_PLAIN_PATH, CAST, DEFAULT_LEAD)[:3]
    # Segment 1 currently 7.2-12.0, segment 2 currently 13.0-18.0. Move both
    # later; segment 1's new end (14.0) overlaps segment 2's OLD start.
    overrides = (
        AlignmentOverride(segment_index=1, start=9.0, end=14.0, reason="move later"),
        AlignmentOverride(segment_index=2, start=14.5, end=19.0, reason="follows"),
    )

    result = align(audio, lines, model=_fake_model(make_raw_stablets_result()), overrides=overrides)

    assert result.segments[1].end == pytest.approx(14.0)
    assert result.segments[2].start == pytest.approx(14.5)
