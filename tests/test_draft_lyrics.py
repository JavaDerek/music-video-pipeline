"""Tests for ``--draft-lyrics`` (issue #105, part 1).

Fully offline: stable-ts is replaced by a fake model whose ``transcribe``
returns scripted segments, and the stem is a synthetic :class:`PcmAudio`
built in memory -- loud where a voice is, digital silence elsewhere. No
torch, no whisper weights, no ffmpeg.
"""

from __future__ import annotations

import array
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from music_video_maker import draft_lyrics
from music_video_maker.draft_lyrics import (
    DRAFT_HEADER_PREFIX,
    DraftLyricsError,
    TranscribedWord,
    screen_silent_words,
    write_draft,
)
from music_video_maker.lyrics import LyricsError, parse_lyrics_text
from music_video_maker.voicing import PcmAudio
from tests.harness.factories import make_cast_dict

RATE = 16000


# -- fakes --------------------------------------------------------------- #


@dataclass
class _Word:
    word: str
    start: float
    end: float
    probability: float = 0.95


@dataclass
class _Segment:
    words: list[_Word]


@dataclass
class _Result:
    segments: list[_Segment]
    language: str = "en"


@dataclass
class _FakeModel:
    result: _Result
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def transcribe(self, audio: str, **kwargs):
        self.calls.append((audio, kwargs))
        return self.result


def _stem_audio(duration: float, voiced: list[tuple[float, float]]) -> PcmAudio:
    """Digital silence, with a loud 220 Hz square wave over each voiced span."""
    samples = array.array("h", [0]) * int(duration * RATE)
    for start, end in voiced:
        for i in range(int(start * RATE), min(len(samples), int(end * RATE))):
            samples[i] = 8000 if (i // 36) % 2 else -8000
    return PcmAudio(samples=samples, sample_rate=RATE)


def _segments(*lines: list[tuple]):
    return [_Segment([_Word(*w) for w in line]) for line in lines]


@pytest.fixture
def stem(tmp_path: Path) -> Path:
    path = tmp_path / "vocals.wav"
    path.write_bytes(b"not really audio; the loader is injected")
    return path


def _run(stem: Path, out: Path, model: _FakeModel, audio: PcmAudio, **kwargs):
    return write_draft(
        stem,
        out,
        model=model,
        load_audio=lambda _path: audio,
        **kwargs,
    )


# -- the draft file ------------------------------------------------------ #


def test_each_transcribed_segment_becomes_one_lyric_line(stem, tmp_path):
    model = _FakeModel(
        _Result(
            _segments(
                [(" Count", 1.0, 1.4), (" the", 1.4, 1.6), (" days", 1.6, 2.0)],
                [
                    (" Nothing", 3.0, 3.5),
                    (" but", 3.5, 3.7),
                    (" the", 3.7, 3.8),
                    (" echo", 3.8, 4.4),
                ],
            )
        )
    )
    audio = _stem_audio(6.0, [(0.9, 2.1), (2.9, 4.5)])
    out = tmp_path / "lyrics.draft.txt"

    _run(stem, out, model, audio)

    body = [line for line in out.read_text().splitlines() if not line.startswith("#!")]
    assert [line for line in body if line] == ["Count the days", "Nothing but the echo"]


def test_the_draft_opens_with_a_provenance_header(stem, tmp_path):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)]), language="en"))
    out = tmp_path / "lyrics.draft.txt"

    _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]), model_size="medium")

    lines = out.read_text().splitlines()
    header = [line for line in lines if line.startswith(DRAFT_HEADER_PREFIX)]
    assert lines[0].startswith(DRAFT_HEADER_PREFIX)
    text = "\n".join(header)
    assert "medium" in text
    assert "vocals.wav" in text
    assert "sha256" in text
    assert "language: en" in text
    assert "lyrics.draft.report.txt" in text
    # The header says how to accept the file.
    assert "delete every" in text.lower()


def test_the_header_names_the_stem_but_not_its_directory(stem, tmp_path):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))
    out = tmp_path / "lyrics.draft.txt"

    _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]))

    assert str(stem.parent) not in out.read_text()


def test_the_model_is_asked_to_transcribe_the_stem_with_the_chosen_language(stem, tmp_path):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))

    _run(stem, tmp_path / "d.txt", model, _stem_audio(3.0, [(0.9, 1.5)]), language="en")

    assert len(model.calls) == 1
    audio_arg, kwargs = model.calls[0]
    assert audio_arg == str(stem)
    assert kwargs["language"] == "en"


def test_the_default_draft_model_is_the_measured_one():
    # #105's WER measurement, pooled over two songs: medium 26.9%, large-v3 34.9%.
    assert draft_lyrics.DEFAULT_DRAFT_MODEL_SIZE == "medium"


# -- the voice gate ------------------------------------------------------ #


def test_a_word_with_no_voice_under_it_is_removed_and_reported(stem, tmp_path):
    # Whisper's best-known failure on music: filling silence with an earlier
    # line. The repeat at 8-9 s sits over digital silence on the stem.
    model = _FakeModel(
        _Result(
            _segments(
                [(" Count", 1.0, 1.4), (" the", 1.4, 1.6), (" days", 1.6, 2.0)],
                [(" Count", 8.0, 8.4), (" the", 8.4, 8.6), (" days", 8.6, 9.0)],
            )
        )
    )
    audio = _stem_audio(10.0, [(0.9, 2.1)])
    out = tmp_path / "lyrics.draft.txt"

    result = _run(stem, out, model, audio)

    body = [line for line in out.read_text().splitlines() if line and not line.startswith("#!")]
    assert body == ["Count the days"]
    assert [w.text for w in result.removed] == ["Count", "the", "days"]
    report = result.report_path.read_text()
    assert "0:08.00" in report
    assert "no voice" in report.lower()


def test_a_word_beside_a_voice_survives_the_gate():
    # Word timestamps are coarse; a real word whose span lands just outside
    # the voiced region must not be dropped for it.
    audio = _stem_audio(5.0, [(1.0, 2.0)])
    words = [[TranscribedWord("late", 2.05, 2.3, 0.9)]]

    kept, removed = screen_silent_words(words, audio)

    assert removed == []
    assert kept == words


def test_a_line_whose_every_word_is_removed_is_dropped_entirely():
    audio = _stem_audio(10.0, [(1.0, 2.0)])
    words = [
        [TranscribedWord("real", 1.1, 1.5, 0.9)],
        [TranscribedWord("phantom", 6.0, 6.5, 0.9)],
    ]

    kept, removed = screen_silent_words(words, audio)

    assert kept == [words[0]]
    assert [w.text for w in removed] == ["phantom"]


# -- the report ---------------------------------------------------------- #


def test_low_probability_words_are_listed_with_time_and_draft_line(stem, tmp_path):
    model = _FakeModel(
        _Result(
            _segments(
                [(" Count", 1.0, 1.4, 0.98), (" the", 1.4, 1.6, 0.97), (" daze", 1.6, 2.0, 0.21)],
            )
        )
    )
    out = tmp_path / "lyrics.draft.txt"

    result = _run(stem, out, model, _stem_audio(3.0, [(0.9, 2.1)]))

    report = result.report_path.read_text()
    draft_lines = out.read_text().splitlines()
    line_number = draft_lines.index("Count the daze") + 1
    assert "daze" in report
    assert "0:01.60" in report
    assert f"line {line_number}" in report
    assert "0.21" in report
    assert "Count" not in report.split("Low-confidence")[1]


def test_the_report_sits_beside_the_draft(stem, tmp_path):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))
    out = tmp_path / "lyrics.draft.txt"

    result = _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]))

    assert result.report_path == tmp_path / "lyrics.draft.report.txt"
    assert result.report_path.exists()


# -- what whisper writes that is not lyrics ------------------------------ #


def test_whisper_annotations_never_reach_the_draft(stem, tmp_path):
    # "[Music]" would parse as a character tag, and a music note is not a
    # word; both are whisper's own annotations, not anything sung.
    model = _FakeModel(
        _Result(
            _segments(
                [(" [Music]", 0.5, 0.9)],
                [
                    (" ♪", 1.0, 1.1),
                    (" Count", 1.1, 1.4),
                    (" the", 1.4, 1.6),
                    (" days", 1.6, 2.0),
                    (" ♪", 2.0, 2.1),
                ],
            )
        )
    )
    out = tmp_path / "lyrics.draft.txt"

    _run(stem, out, model, _stem_audio(3.0, [(0.4, 2.2)]))

    body = [line for line in out.read_text().splitlines() if line and not line.startswith("#!")]
    assert body == ["Count the days"]


# -- refusals ------------------------------------------------------------ #


def test_a_missing_stem_is_refused_with_a_pointer_to_the_stem_workflow(tmp_path):
    model = _FakeModel(_Result([]))
    with pytest.raises(DraftLyricsError, match="vocal-stem-workflow"):
        write_draft(
            tmp_path / "nope.wav",
            tmp_path / "d.txt",
            model=model,
            load_audio=lambda _p: _stem_audio(1.0, []),
        )
    assert model.calls == []


def test_an_existing_output_is_not_overwritten_without_force(stem, tmp_path):
    out = tmp_path / "lyrics.draft.txt"
    out.write_text("hand-corrected work\n")
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))

    with pytest.raises(DraftLyricsError, match="--force"):
        _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]))
    assert out.read_text() == "hand-corrected work\n"
    assert model.calls == []


def test_a_missing_output_directory_is_created_before_transcribing(stem, tmp_path):
    # A transcription costs minutes; failing to write afterwards wastes all of them.
    out = tmp_path / "not" / "yet" / "lyrics.draft.txt"
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))

    _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]))

    assert "Count" in out.read_text()


def test_force_overwrites(stem, tmp_path):
    out = tmp_path / "lyrics.draft.txt"
    out.write_text("old\n")
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))

    _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.5)]), force=True)

    assert "Count" in out.read_text()


# -- Stage 1 refuses an unreviewed draft --------------------------------- #


def test_stage_1_refuses_a_draft_until_its_header_is_deleted(stem, tmp_path):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4), (" the", 1.4, 1.6)])))
    out = tmp_path / "lyrics.draft.txt"
    _run(stem, out, model, _stem_audio(3.0, [(0.9, 1.7)]))
    cast = make_cast_dict()

    with pytest.raises(LyricsError) as excinfo:
        parse_lyrics_text(out.read_text(), cast, "Dianne")
    message = str(excinfo.value)
    assert "draft" in message.lower()
    assert "#!" in message

    reviewed = "\n".join(
        line for line in out.read_text().splitlines() if not line.startswith("#!")
    )
    lines = parse_lyrics_text(reviewed, cast, "Dianne")
    assert [line.text for line in lines] == ["Count the"]


def test_a_single_leftover_header_line_still_refuses():
    # Deleting only part of the header must not turn the rest into lyrics.
    text = "Count the days\n#! model: large-v3\nNothing but the echo\n"
    with pytest.raises(LyricsError, match="line 2"):
        parse_lyrics_text(text, make_cast_dict(), "Dianne")


# -- CLI ----------------------------------------------------------------- #


def test_main_returns_nonzero_and_logs_the_reason_for_a_missing_stem(tmp_path, caplog):
    out = tmp_path / "d.txt"
    code = draft_lyrics.main(["--stem", str(tmp_path / "nope.wav"), "--out", str(out)])

    assert code == 1
    assert "vocal-stem-workflow" in caplog.text
    assert not out.exists()


def test_main_rejects_an_unknown_model_size(stem, tmp_path):
    with pytest.raises(SystemExit):
        draft_lyrics.main(["--stem", str(stem), "--model-size", "enormous"])


def test_main_writes_the_draft(stem, tmp_path, monkeypatch, caplog):
    model = _FakeModel(_Result(_segments([(" Count", 1.0, 1.4)])))
    audio = _stem_audio(3.0, [(0.9, 1.5)])
    monkeypatch.setattr(draft_lyrics, "_load_model", lambda size: model)
    monkeypatch.setattr(draft_lyrics, "load_stem_audio", lambda path: audio)
    caplog.set_level(logging.INFO, logger="music_video_maker.draft_lyrics")
    out = tmp_path / "lyrics.draft.txt"

    code = draft_lyrics.main(["--stem", str(stem), "--out", str(out)])

    assert code == 0
    assert "Count" in out.read_text()
    assert str(out) in caplog.text


# -- the transcript, kept for the second witness (#105 part 2) ----------- #


def test_the_word_timings_are_written_beside_the_draft(stem, tmp_path):
    model = _FakeModel(
        _Result(
            _segments(
                [(" Count", 1.0, 1.4, 0.9), (" the", 1.4, 1.6, 0.8)],
                [(" phantom", 8.0, 8.5, 0.7)],
            )
        )
    )
    out = tmp_path / "lyrics.draft.txt"

    result = _run(stem, out, model, _stem_audio(10.0, [(0.9, 1.7)]), model_size="medium")

    assert result.words_path == tmp_path / "lyrics.draft.words.json"
    payload = json.loads(result.words_path.read_text())
    assert payload["model"] == "medium"
    assert payload["stem"] == "vocals.wav"
    assert len(payload["stem_sha256"]) == 64
    assert payload["words"] == [
        {"text": "Count", "start": 1.0, "end": 1.4, "probability": 0.9, "voiced": True},
        {"text": "the", "start": 1.4, "end": 1.6, "probability": 0.8, "voiced": True},
        {"text": "phantom", "start": 8.0, "end": 8.5, "probability": 0.7, "voiced": False},
    ]
