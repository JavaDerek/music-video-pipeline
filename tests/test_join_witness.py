"""Tests for the chunk-join-inside-a-word check.

Pure and offline: the transcriber is a fake that hands back hand-written
words. No Whisper runs here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_video_maker.contracts import AudioChunk
from music_video_maker.join_witness import (
    HeardWord,
    JoinCheck,
    JoinInsideWordError,
    check_joins,
    chunk_joins,
    log_findings,
    nearest_clear_time,
    raise_if_blocking,
    word_containing,
    words_from_result,
)


def _w(text: str, start: float, end: float) -> HeardWord:
    return HeardWord(text=text, start=start, end=end)


def _chunk(chunk_id: int, start: float, end: float) -> AudioChunk:
    return AudioChunk(chunk_id=chunk_id, audio_file=Path(f"c{chunk_id}.wav"),
                      start=start, end=end, text="x")


# -- word_containing ------------------------------------------------------ #


def test_join_strictly_inside_a_word_is_found() -> None:
    words = [_w("hello", 9.0, 9.4), _w("forever", 9.8, 10.6)]
    assert word_containing(10.2, words) == words[1]


def test_join_in_a_pause_is_clear() -> None:
    words = [_w("hello", 9.0, 9.4), _w("forever", 9.8, 10.6)]
    assert word_containing(9.6, words) is None


def test_join_at_a_word_edge_is_clear_within_the_guard() -> None:
    words = [_w("forever", 9.8, 10.6)]
    assert word_containing(9.82, words, guard=0.04) is None
    assert word_containing(10.58, words, guard=0.04) is None
    assert word_containing(9.9, words, guard=0.04) == words[0]


def test_zero_length_words_never_contain_a_join() -> None:
    # The forced aligner's failure mode: it returned zero-length words.
    assert word_containing(5.0, [_w("a", 5.0, 5.0)]) is None


# -- nearest_clear_time --------------------------------------------------- #


def test_suggestion_is_the_nearest_pause_edge() -> None:
    words = [_w("hello", 9.0, 9.4), _w("forever", 9.8, 10.6), _w("more", 10.6, 11.0)]
    # join inside "forever": the pause before it (midpoint 9.6) is nearer
    # than the end of it, which abuts "more".
    assert nearest_clear_time(10.0, words) == pytest.approx(9.6)


def test_suggestion_respects_reach() -> None:
    words = [_w("sustained", 5.0, 15.0)]
    assert nearest_clear_time(10.0, words, reach=1.5) is None


def test_suggestion_none_without_words() -> None:
    assert nearest_clear_time(10.0, []) is None


# -- chunk_joins ---------------------------------------------------------- #


def test_chunk_joins_are_between_consecutive_chunks() -> None:
    chunks = [_chunk(0, 0.0, 5.0), _chunk(1, 5.0, 11.0), _chunk(2, 11.0, 15.0)]
    assert chunk_joins(chunks) == [(0, 1, 5.0), (1, 2, 11.0)]


def test_chunk_joins_empty_for_one_chunk() -> None:
    assert chunk_joins([_chunk(0, 0.0, 5.0)]) == []


# -- check_joins ---------------------------------------------------------- #


def test_check_joins_reports_each_join_inside_a_word() -> None:
    heard = {
        5.0: [_w("up", 4.2, 4.5), _w("beyond", 4.9, 5.6)],
        11.0: [_w("again", 11.3, 11.8)],
    }
    windows: list[tuple[float, float]] = []

    def fake(start: float, end: float) -> list[HeardWord]:
        windows.append((start, end))
        return heard[round((start + end) / 2, 1)]

    joins = [(0, 1, 5.0), (1, 2, 11.0)]
    check = check_joins(joins, fake, window_seconds=1.5)
    assert windows == [(3.5, 6.5), (9.5, 12.5)]
    assert check.joins_checked == 2
    assert len(check.findings) == 1
    finding = check.findings[0]
    assert (finding.chunk_before, finding.chunk_after) == (0, 1)
    assert finding.word == "beyond"
    assert finding.join_seconds == 5.0
    assert finding.suggested_seconds == pytest.approx(4.7)


def test_window_is_clamped_at_the_start_of_the_track() -> None:
    seen: list[tuple[float, float]] = []

    def fake(start: float, end: float) -> list[HeardWord]:
        seen.append((start, end))
        return []

    check_joins([(0, 1, 1.0)], fake, window_seconds=1.5)
    assert seen == [(0.0, 2.5)]


def test_window_edge_words_are_not_trusted() -> None:
    # A word clipped by the window's own edge says nothing about the song.
    def fake(start: float, end: float) -> list[HeardWord]:
        return [_w("clipped", start, end)]

    check = check_joins([(0, 1, 5.0)], fake, window_seconds=1.5)
    assert check.findings == ()


def test_transcriber_failure_is_recorded_not_raised() -> None:
    def fake(start: float, end: float) -> list[HeardWord]:
        raise RuntimeError("boom")

    check = check_joins([(0, 1, 5.0)], fake)
    assert check.findings == ()
    assert check.unchecked == ((0, 1, 5.0),)


# -- refusal and reporting ------------------------------------------------ #


def _check_with_finding() -> JoinCheck:
    return check_joins(
        [(3, 4, 5.0)], lambda s, e: [_w("beyond", 4.9, 5.6)], window_seconds=1.5
    )


def test_strict_refuses_naming_time_and_word() -> None:
    check = _check_with_finding()
    with pytest.raises(JoinInsideWordError) as info:
        raise_if_blocking(check, strict=True)
    assert "5.000" in str(info.value)
    assert "beyond" in str(info.value)
    assert info.value.check is check


def test_not_strict_does_not_refuse() -> None:
    raise_if_blocking(_check_with_finding(), strict=False)


def test_strict_with_no_findings_passes() -> None:
    raise_if_blocking(check_joins([(0, 1, 5.0)], lambda s, e: []), strict=True)


def test_log_findings_warns_per_finding(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        log_findings(_check_with_finding())
    text = caplog.text
    assert "beyond" in text and "chunk 3/4" in text


# -- reading a stable-ts result ------------------------------------------- #


def test_words_from_result_offsets_and_cleans() -> None:
    result = SimpleNamespace(
        segments=[
            SimpleNamespace(
                words=[
                    SimpleNamespace(word=" hello", start=0.5, end=0.9),
                    SimpleNamespace(word="  ", start=1.0, end=1.1),
                    SimpleNamespace(word="you", start=1.2, end=1.6),
                ]
            )
        ]
    )
    assert words_from_result(result, offset=10.0) == [
        _w("hello", 10.5, 10.9),
        _w("you", 11.2, 11.6),
    ]
