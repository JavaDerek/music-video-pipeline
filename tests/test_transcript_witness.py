"""Tests for the transcript as a second witness against the aligner (#105 part 2).

Pure and offline: aligned segments and transcripts are built by hand. The
transcript is what ``draft_lyrics`` writes as ``*.words.json``; nothing here
runs ASR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_video_maker.contracts import AlignedSegment, AlignmentResult, WordTiming
from music_video_maker.transcript_witness import (
    HeardWord,
    Transcript,
    TranscriptError,
    compare,
    load_transcript,
)


def _segment(index: int, words: list[tuple[str, float]], *, step: float = 0.4) -> AlignedSegment:
    timings = tuple(WordTiming(word=w, start=t, end=t + step) for w, t in words)
    return AlignedSegment(
        index=index,
        text=" ".join(w for w, _ in words),
        start=timings[0].start,
        end=timings[-1].end,
        words=timings,
    )


def _sung(text: str, start: float, *, step: float = 0.4, voiced: bool = True) -> list[HeardWord]:
    return [
        HeardWord(text=w, start=start + i * step, end=start + i * step + step,
                  probability=0.9, voiced=voiced)
        for i, w in enumerate(text.split())
    ]


def _placed(index: int, text: str, start: float, *, step: float = 0.4) -> AlignedSegment:
    return _segment(index, [(w, start + i * step) for i, w in enumerate(text.split())], step=step)


def _result(*segments: AlignedSegment) -> AlignmentResult:
    return AlignmentResult(segments=segments, track_duration=600.0)


def _transcript(*runs: list[HeardWord]) -> Transcript:
    return Transcript(words=tuple(w for run in runs for w in run), model="medium",
                      stem="vocals.wav", stem_sha256="0" * 64)


VERSE = "count the days until the sun comes back"
CHORUS = "nothing but the echo of a drum tonight"
BRIDGE = "give the silence room to breathe again"


# -- agreement -------------------------------------------------------------- #


def test_a_segment_placed_where_it_was_heard_raises_nothing():
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0))
    transcript = _transcript(_sung(VERSE, 10.2), _sung(CHORUS, 19.8))

    comparison = compare(result, transcript)

    assert comparison.disagreements == ()
    assert comparison.passages == ()


def test_mishearings_in_place_are_not_findings():
    # A draft gets a quarter of its words wrong; a word swapped one-for-one
    # is a transcription error, not a timing or lyrics-file problem.
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0))
    transcript = _transcript(
        _sung("count the daze until the son comes back", 10.0),
        _sung("nothing but the echo of a drum to night", 20.0),
    )

    comparison = compare(result, transcript)

    assert comparison.disagreements == ()
    assert comparison.passages == ()


# -- disagreement ----------------------------------------------------------- #


def test_a_segment_placed_far_from_where_it_was_heard_is_reported():
    # #71's shape: the closing line thrown 12 s into the fadeout.
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 32.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 20.0))

    comparison = compare(result, transcript)

    assert [d.segment_index for d in comparison.disagreements] == [1]
    found = comparison.disagreements[0]
    assert found.placed_start == pytest.approx(32.0)
    assert found.heard_start == pytest.approx(20.0)
    assert found.offset == pytest.approx(12.0)
    assert found.matched_words == len(CHORUS.split())


def test_an_early_pile_up_is_reported_too():
    # #42's base model: every word stamped at one instant, seconds early.
    piled = _segment(1, [(w, 65.4) for w in CHORUS.split()], step=0.0)
    result = _result(_placed(0, VERSE, 50.0), piled)
    transcript = _transcript(_sung(VERSE, 50.0), _sung(CHORUS, 71.5))

    comparison = compare(result, transcript)

    assert [d.segment_index for d in comparison.disagreements] == [1]
    assert comparison.disagreements[0].offset == pytest.approx(65.4 - 71.5 - 0.4 * 3.5, abs=0.5)


def test_a_small_timing_difference_is_not_a_disagreement():
    result = _result(_placed(0, VERSE, 10.0))
    transcript = _transcript(_sung(VERSE, 11.5))

    assert compare(result, transcript).disagreements == ()


def test_silent_words_are_never_evidence():
    # Whisper's silence fill repeats earlier lyric text; it would match the
    # lyrics file perfectly. Only words with a voice under them count.
    result = _result(_placed(0, CHORUS, 40.0))
    transcript = _transcript(_sung(CHORUS, 20.0, voiced=False), _sung(CHORUS, 40.0))

    comparison = compare(result, transcript)

    assert comparison.disagreements == ()
    assert comparison.passages == ()


def test_one_or_two_matched_words_are_not_evidence():
    # "the" and "back" match anywhere; a lone common word is not a placement.
    result = _result(_placed(0, "the sun comes back", 10.0))
    transcript = _transcript(_sung("the cat sat", 30.0))

    assert compare(result, transcript).disagreements == ()


# -- the lyrics file against the singing ------------------------------------ #


def test_a_passage_sung_but_not_in_the_lyrics_file_is_reported():
    # The unwritten repeat chorus: sung twice, written once.
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0), _placed(2, BRIDGE, 40.0))
    transcript = _transcript(
        _sung(VERSE, 10.0), _sung(CHORUS, 20.0), _sung(CHORUS, 30.0), _sung(BRIDGE, 40.0)
    )

    comparison = compare(result, transcript)

    assert len(comparison.passages) == 1
    passage = comparison.passages[0]
    assert passage.kind == "sung_not_written"
    assert passage.start == pytest.approx(30.0, abs=0.5)
    assert passage.text == CHORUS
    assert passage.resembles_segment == 1


def test_a_trailing_unwritten_repeat_is_reported():
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 20.0), _sung(CHORUS, 30.0))

    passages = compare(result, transcript).passages

    assert [(p.kind, round(p.start)) for p in passages] == [("sung_not_written", 30)]


def test_a_lyric_line_nothing_sang_is_reported():
    result = _result(_placed(0, VERSE, 10.0), _placed(1, BRIDGE, 20.0), _placed(2, CHORUS, 30.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 30.0))

    passages = compare(result, transcript).passages

    assert [(p.kind, p.segment_indices) for p in passages] == [("written_not_sung", (1,))]


def test_a_short_unmatched_run_is_not_a_passage():
    # An ad-lib "oh yeah" is not a missing line.
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung("oh yeah", 16.0), _sung(CHORUS, 20.0))

    assert compare(result, transcript).passages == ()


def test_punctuation_and_case_do_not_matter():
    result = _result(_placed(0, "Count, the days! Until the sun's back.", 10.0))
    transcript = _transcript(_sung("count the days until the sun's back", 10.0))

    comparison = compare(result, transcript)

    assert comparison.disagreements == ()
    assert comparison.passages == ()


# -- loading ---------------------------------------------------------------- #


def test_load_reads_what_draft_lyrics_writes(tmp_path: Path):
    path = tmp_path / "lyrics.draft.words.json"
    path.write_text(json.dumps({
        "model": "medium", "language": "en", "stem": "vocals.wav", "stem_sha256": "a" * 64,
        "words": [
            {"text": "Count", "start": 1.0, "end": 1.4, "probability": 0.9, "voiced": True},
            {"text": "phantom", "start": 8.0, "end": 8.5, "probability": None, "voiced": False},
        ],
    }))

    transcript = load_transcript(path)

    assert transcript.model == "medium"
    assert transcript.stem_sha256 == "a" * 64
    assert transcript.words == (
        HeardWord("Count", 1.0, 1.4, 0.9, True),
        HeardWord("phantom", 8.0, 8.5, None, False),
    )


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"words": "nope"}),
        json.dumps({"words": [{"text": "a", "start": 1.0}]}),
        json.dumps({"words": [{"text": "a", "start": "x", "end": 1.0, "voiced": True}]}),
    ],
)
def test_a_malformed_transcript_is_refused(tmp_path: Path, payload: str):
    path = tmp_path / "words.json"
    path.write_text(payload)
    with pytest.raises(TranscriptError, match=str(path)):
        load_transcript(path)


# -- short lines: bounded by their matched neighbours ----------------------- #


def test_a_short_line_placed_after_the_singing_ended_is_reported():
    # #71 and #42's `small`: a two-word closing line thrown into the outro.
    # Too short to match on its own words; its neighbours bound it anyway.
    result = _result(
        _placed(0, VERSE, 10.0),
        _placed(1, CHORUS, 20.0),
        _placed(2, "deathless, forevermore!", 45.0),
    )
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 20.0), _sung("forever more", 23.4))

    disagreements = compare(result, transcript).disagreements

    assert [(d.segment_index, d.basis) for d in disagreements] == [(2, "bounds")]
    found = disagreements[0]
    assert found.placed_start == pytest.approx(45.0)
    assert found.heard_start == pytest.approx(24.2)  # the last voiced word's end
    assert found.offset == pytest.approx(45.0 - 24.2)


def test_a_short_line_placed_before_its_heard_predecessor_is_reported():
    result = _result(_placed(0, VERSE, 30.0), _placed(1, "oh no", 12.0), _placed(2, CHORUS, 40.0))
    transcript = _transcript(_sung(VERSE, 30.0), _sung("oh no", 34.0), _sung(CHORUS, 40.0))

    disagreements = compare(result, transcript).disagreements

    assert [(d.segment_index, d.basis) for d in disagreements] == [(1, "bounds")]
    assert disagreements[0].offset < 0


def test_a_short_line_between_its_neighbours_is_fine():
    result = _result(_placed(0, VERSE, 10.0), _placed(1, "oh no", 14.0), _placed(2, CHORUS, 20.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung("whoa oh", 14.5), _sung(CHORUS, 20.0))

    assert compare(result, transcript).disagreements == ()


def test_matched_evidence_reports_with_its_basis():
    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 32.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 20.0))

    assert [d.basis for d in compare(result, transcript).disagreements] == ["matched"]


def test_an_unwritten_repeat_is_found_even_when_an_unheard_line_cancels_its_count():
    # "The Lucky Ones": the deleted repeat and the closing lines the transcript
    # never heard (counterpoint) fell in one gap, and equal counts hid both.
    TAIL = "we are the ones the lucky lucky ones"
    result = _result(
        _placed(0, VERSE, 10.0), _placed(1, CHORUS, 20.0), _placed(2, BRIDGE, 30.0),
        _placed(3, TAIL, 40.0),
    )
    transcript = _transcript(
        _sung(VERSE, 10.0), _sung(CHORUS, 20.0), _sung(BRIDGE, 30.0), _sung(CHORUS, 40.0)
    )

    passages = compare(result, transcript).passages

    assert [(p.kind, p.resembles_segment, p.segment_indices) for p in passages] == [
        ("sung_not_written", 1, ()),
        ("written_not_sung", None, (3,)),
    ]
    assert passages[0].start == pytest.approx(40.0)
    assert passages[0].text == CHORUS


def test_a_misheard_gap_is_not_mistaken_for_a_repeat():
    # Unmatched words on both sides that do not repeat written text elsewhere
    # are mishearings, and equal counts mean nothing to report.
    result = _result(_placed(0, VERSE, 10.0), _placed(1, BRIDGE, 20.0), _placed(2, CHORUS, 30.0))
    transcript = _transcript(
        _sung(VERSE, 10.0), _sung("give us silence brew to breed a gain", 20.0), _sung(CHORUS, 30.0)
    )

    assert compare(result, transcript).passages == ()


# -- reported through the alignment quality report ------------------------- #


def test_the_quality_report_carries_the_witness_as_warnings():
    from music_video_maker.alignment_quality import Severity, evaluate_alignment_quality

    result = _result(
        _placed(0, VERSE, 10.0), _placed(1, CHORUS, 32.0), _placed(2, BRIDGE, 40.0)
    )
    transcript = _transcript(
        _sung(VERSE, 10.0), _sung(CHORUS, 20.0), _sung(CHORUS, 30.0), _sung(BRIDGE, 40.0)
    )

    report = evaluate_alignment_quality(result, transcript=transcript)

    witness = [f for f in report.findings if f.code.startswith("transcript_")]
    assert {f.code for f in witness} == {"transcript_disagreement", "transcript_sung_not_written"}
    assert all(f.severity == Severity.WARNING for f in witness)
    disagreement = next(f for f in witness if f.code == "transcript_disagreement")
    assert disagreement.segment_index == 1
    assert "32.0" in disagreement.message and "20.0" in disagreement.message
    sung = next(f for f in witness if f.code == "transcript_sung_not_written")
    assert sung.related_segment_index == 1
    assert "repeat" in sung.message


def test_written_not_sung_names_its_first_segment():
    from music_video_maker.alignment_quality import evaluate_alignment_quality

    result = _result(_placed(0, VERSE, 10.0), _placed(1, BRIDGE, 20.0), _placed(2, CHORUS, 30.0))
    transcript = _transcript(_sung(VERSE, 10.0), _sung(CHORUS, 30.0))

    report = evaluate_alignment_quality(result, transcript=transcript)

    found = [f for f in report.findings if f.code == "transcript_written_not_sung"]
    assert [f.segment_index for f in found] == [1]


def test_without_a_transcript_the_report_is_unchanged():
    from music_video_maker.alignment_quality import evaluate_alignment_quality

    result = _result(_placed(0, VERSE, 10.0), _placed(1, CHORUS, 32.0))

    report = evaluate_alignment_quality(result)

    assert not [f for f in report.findings if f.code.startswith("transcript_")]


def test_the_reported_heard_time_is_the_one_the_offset_implies():
    # The message reads "placed at X, heard at Y (offset)"; Y must be X - offset,
    # not the earliest matched word, or a split match reads as a contradiction.
    words = [(w, 10.0 + i * 0.4) for i, w in enumerate(CHORUS.split())]
    result = _result(_segment(0, words))
    transcript = _transcript(_sung("nothing but the", 3.0), _sung("echo of a drum tonight", 2.0))

    found = compare(result, transcript).disagreements[0]

    assert found.heard_start == pytest.approx(found.placed_start - found.offset)
