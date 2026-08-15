"""Tests for Stage 2a-stem: cutting an isolated vocal stem at the chunk spans
the pipeline already computed from the master (issue #25).

Everything is WAV-only and synthesized in-test with stdlib ``wave`` -- no
ffmpeg, no network, no GPU, and no committed binaries. pydub's wav read/write
path uses the stdlib ``wave`` module, so nothing here shells out.

The invariant under test throughout: the stem is **conditioning input only**.
It never gets its own alignment, it is cut at exactly the spans
``slicing.slice_audio`` produced from the master, and Stage 5 is untouched --
the pristine master is still the only audio in the finished video.
"""

from __future__ import annotations

import logging
import math
import struct
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from pydub import AudioSegment

from music_video_maker import hardware
from music_video_maker.contracts import AudioChunk
from music_video_maker.slicing import slice_audio
from music_video_maker.stems import (
    StemSliceError,
    StemSliceResult,
    StemValidationError,
    slice_stem_for_chunks,
    validate_stem_against_master,
)
from tests.harness.factories import (
    make_alignment_result_normal_song,
    write_silent_wav,
)

DEFAULT_HARDWARE = hardware.PROFILE_RTX_4090_24GB
GRID = DEFAULT_HARDWARE.frame_grid

SAMPLE_RATE = 8000
"""Low but legal -- these fixtures are minutes long and only ever inspected
for loudness and duration, so the sample rate is chosen for test speed."""


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _write_wav_with_tone_spans(
    path: Path,
    seconds: float,
    spans: tuple[tuple[float, float], ...] = (),
    *,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """A 16-bit mono WAV of ``seconds``, audible inside ``spans``, silent elsewhere.

    Stands in for a separated vocal stem: real voice where the singer sings,
    near-digital-silence over the instrumental passages.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(0, round(seconds * sample_rate))
    peak = int(max(0.0, min(1.0, amplitude)) * 32767)
    samples = [0] * n_frames
    for start, end in spans:
        first = max(0, round(start * sample_rate))
        last = min(n_frames, round(end * sample_rate))
        for i in range(first, last):
            samples[i] = int(peak * math.sin(2 * math.pi * frequency * (i / sample_rate)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *samples))
    return path


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _max_dbfs(path: Path) -> float:
    return AudioSegment.from_file(str(path)).max_dBFS


def _chunk(
    chunk_id: int,
    start: float,
    end: float,
    *,
    text: str = "a lyric line",
    is_instrumental: bool = False,
    audio_file: Path | None = None,
    frame_count: int | None = None,
) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=audio_file if audio_file is not None else Path(f"mix/chunk_{chunk_id:03d}.wav"),
        start=start,
        end=end,
        text="" if is_instrumental else text,
        characters=() if is_instrumental else ("Dianne",),
        frame_count=frame_count,
        is_instrumental=is_instrumental,
    )


# A 20 s track whose voice is only present from 10 s to 15 s. The four chunks
# tile it contiguously in 5 s pieces, so exactly one of them (chunk 2) sits
# over the sung passage.
VOICED_SPAN = (10.0, 15.0)
TRACK_SECONDS = 20.0


def _voiced_stem(tmp_path: Path) -> Path:
    return _write_wav_with_tone_spans(tmp_path / "vocals.wav", TRACK_SECONDS, (VOICED_SPAN,))


def _full_mix(tmp_path: Path) -> Path:
    """A band playing for the whole track -- loud everywhere, unlike the stem."""
    return _write_wav_with_tone_spans(
        tmp_path / "master.wav", TRACK_SECONDS, ((0.0, TRACK_SECONDS),), frequency=110.0
    )


def _tiled_chunks() -> tuple[AudioChunk, ...]:
    return (
        _chunk(0, 0.0, 5.0, is_instrumental=True),
        _chunk(1, 5.0, 10.0, is_instrumental=True),
        _chunk(2, 10.0, 15.0, text="here is the chorus"),
        _chunk(3, 15.0, 20.0, is_instrumental=True),
    )


# --------------------------------------------------------------------------- #
# Happy path: the stem is cut at exactly the spans the master produced
# --------------------------------------------------------------------------- #


def test_slices_the_stem_at_exactly_the_chunk_spans(tmp_path):
    stem = _voiced_stem(tmp_path)
    chunks = _tiled_chunks()

    result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks")

    assert isinstance(result, StemSliceResult)
    assert len(result.chunks) == len(chunks)
    for original, sliced in zip(chunks, result.chunks, strict=True):
        assert sliced.audio_file.exists()
        assert sliced.audio_file.parent == tmp_path / "stem_chunks"
        duration = _wav_duration_seconds(sliced.audio_file)
        assert duration == pytest.approx(original.duration, abs=1e-3)


def test_returns_chunks_identical_apart_from_the_conditioning_audio_file(tmp_path):
    """Nothing about the timeline may move: the stem is a swap of the
    conditioning waveform, not a re-planning of the chunks."""
    stem = _voiced_stem(tmp_path)
    chunks = _tiled_chunks()

    result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks")

    for original, sliced in zip(chunks, result.chunks, strict=True):
        assert sliced.audio_file != original.audio_file
        assert replace(sliced, audio_file=original.audio_file) == original


def test_conditioning_audio_comes_from_the_stem_not_the_mix(tmp_path):
    """The whole point of issue #25: an instrumental span must be near-silence
    in the conditioning audio even though the mix is loud there."""
    stem = _voiced_stem(tmp_path)
    mix = _full_mix(tmp_path)
    chunks = _tiled_chunks()

    result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks", master_path=mix)

    sung = result.chunks[2].audio_file
    instrumental = result.chunks[3].audio_file
    assert _max_dbfs(sung) > -12.0
    assert _max_dbfs(instrumental) < -45.0
    # ... while the mix is loud across that very same span.
    mix_tail = AudioSegment.from_file(str(mix))[15_000:20_000]
    assert mix_tail.max_dBFS > -12.0


def test_creates_the_output_directory_and_leaves_the_mix_slices_untouched(tmp_path):
    stem = _voiced_stem(tmp_path)
    mix_dir = tmp_path / "chunks"
    mix_slices = [write_silent_wav(mix_dir / f"chunk_{i:03d}.wav", 5.0) for i in range(4)]
    before = [p.read_bytes() for p in mix_slices]
    chunks = tuple(
        replace(chunk, audio_file=mix_slices[i]) for i, chunk in enumerate(_tiled_chunks())
    )

    result = slice_stem_for_chunks(stem, chunks, tmp_path / "nested" / "stem_chunks")

    assert (tmp_path / "nested" / "stem_chunks").is_dir()
    assert [p.read_bytes() for p in mix_slices] == before
    assert all(c.audio_file.parent.name == "stem_chunks" for c in result.chunks)


def test_output_filenames_are_keyed_on_chunk_id_not_position(tmp_path):
    """Chunk ids are the identity every other stage keys on; a rendered chunk
    resumed by id must find the stem cut for *that* id."""
    stem = _voiced_stem(tmp_path)
    chunks = (_chunk(7, 10.0, 15.0), _chunk(11, 15.0, 20.0, is_instrumental=True))

    result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks")

    names = [c.audio_file.name for c in result.chunks]
    assert names[0].startswith("chunk_007")
    assert names[1].startswith("chunk_011")


# --------------------------------------------------------------------------- #
# Validation against the master: refuse loudly, never silently pad
# --------------------------------------------------------------------------- #


def test_refuses_a_stem_whose_duration_does_not_match_the_master(tmp_path, caplog):
    stem = _write_wav_with_tone_spans(tmp_path / "vocals.wav", TRACK_SECONDS - 3.0)
    mix = _full_mix(tmp_path)
    out_dir = tmp_path / "stem_chunks"

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemValidationError) as excinfo,
    ):
        slice_stem_for_chunks(stem, _tiled_chunks(), out_dir, master_path=mix)

    assert "duration" in str(excinfo.value).lower()
    assert caplog.records
    # Nothing was written: a refusal must not leave half a conditioning set
    # behind for a later --resume to pick up.
    assert not out_dir.exists() or not list(out_dir.glob("*.wav"))


def test_accepts_a_stem_within_the_duration_tolerance(tmp_path):
    """Separation tools can land a few milliseconds off; that is not a desync."""
    stem = _write_wav_with_tone_spans(
        tmp_path / "vocals.wav", TRACK_SECONDS - 0.03, (VOICED_SPAN,)
    )
    mix = _full_mix(tmp_path)

    result = slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "stem_chunks", master_path=mix)

    assert len(result.chunks) == 4


def test_validate_stem_against_master_returns_the_stem_duration(tmp_path):
    stem = _voiced_stem(tmp_path)
    mix = _full_mix(tmp_path)

    assert validate_stem_against_master(stem, mix) == pytest.approx(TRACK_SECONDS, abs=1e-3)


def test_validate_stem_against_master_raises_on_a_mismatch(tmp_path, caplog):
    stem = _write_wav_with_tone_spans(tmp_path / "vocals.wav", TRACK_SECONDS + 5.0)
    mix = _full_mix(tmp_path)

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemValidationError),
    ):
            validate_stem_against_master(stem, mix)
    assert caplog.records


def test_missing_stem_file_is_refused(tmp_path, caplog):
    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemValidationError),
    ):
            slice_stem_for_chunks(tmp_path / "nope.wav", _tiled_chunks(), tmp_path / "out")
    assert caplog.records


def test_unreadable_stem_file_is_refused(tmp_path, caplog):
    stem = _voiced_stem(tmp_path)
    stem.chmod(0o000)
    try:
        with (
            caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
            pytest.raises(StemValidationError),
        ):
                slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "out")
        assert caplog.records
    finally:
        stem.chmod(0o644)


def test_a_stem_far_shorter_than_the_timeline_is_refused_even_without_a_master(tmp_path, caplog):
    """Without a master to compare against, the chunk timeline is still a
    check: a stem that cannot cover it is the wrong file, not a short tail."""
    stem = _write_wav_with_tone_spans(tmp_path / "vocals.wav", 12.0)

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemValidationError),
    ):
            slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "out")
    assert caplog.records


def test_a_chunk_running_a_hair_past_the_stem_end_is_padded_with_silence(tmp_path, caplog):
    """The outro chunk is grid-quantized and can land past the final sample --
    exactly as in slicing.py. The stem slice must still be its full length or
    Stage 4a gets audio shorter than its own frame_count implies."""
    stem = _write_wav_with_tone_spans(tmp_path / "vocals.wav", TRACK_SECONDS, (VOICED_SPAN,))
    chunks = (*_tiled_chunks(), _chunk(4, 20.0, 20.5, is_instrumental=True))

    with caplog.at_level(logging.INFO, logger="music_video_maker.stems"):
        result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks")

    tail = result.chunks[-1].audio_file
    assert _wav_duration_seconds(tail) == pytest.approx(0.5, abs=1e-3)
    assert result.report.padded_chunk_ids == (4,)
    assert any("pad" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Quality reporting: the failure mode is a stem that drops a real voice
# --------------------------------------------------------------------------- #


def test_reports_and_warns_when_a_sung_chunk_has_a_silent_stem(tmp_path, caplog):
    """htdemucs can file a vocoded/processed voice under ``other`` rather than
    ``vocals``. Conditioning on that stem renders a closed mouth over a sung
    line -- a regression caused by this feature, and invisible unless checked.
    """
    stem = _write_wav_with_tone_spans(tmp_path / "vocals.wav", TRACK_SECONDS, ((10.0, 15.0),))
    chunks = (
        _chunk(0, 0.0, 5.0, is_instrumental=True),
        _chunk(1, 5.0, 10.0, text="the robot voice sings here"),
        _chunk(2, 10.0, 15.0, text="here is the chorus"),
        _chunk(3, 15.0, 20.0, text="and the vocoded outro"),
    )

    with caplog.at_level(logging.WARNING, logger="music_video_maker.stems"):
        result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks")

    assert result.report.silent_voiced_chunk_ids == (1, 3)
    assert result.report.voiced_chunk_count == 3
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("silent" in r.message.lower() for r in warnings)


def test_no_warning_when_every_sung_chunk_carries_voice(tmp_path, caplog):
    stem = _voiced_stem(tmp_path)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.stems"):
        result = slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "stem_chunks")

    assert result.report.silent_voiced_chunk_ids == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_reports_instrumental_chunks_whose_stem_still_carries_signal(tmp_path):
    """Leakage the other way: separation left band audio in the vocal stem, so
    an instrumental chunk is not the near-silence this issue is banking on."""
    stem = _write_wav_with_tone_spans(
        tmp_path / "vocals.wav", TRACK_SECONDS, ((0.0, 5.0), VOICED_SPAN)
    )

    result = slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "stem_chunks")

    assert result.report.leaky_instrumental_chunk_ids == (0,)


def test_logs_a_summary_at_info_on_every_run(tmp_path, caplog):
    stem = _voiced_stem(tmp_path)

    with caplog.at_level(logging.INFO, logger="music_video_maker.stems"):
        result = slice_stem_for_chunks(stem, _tiled_chunks(), tmp_path / "stem_chunks")

    assert "4" in result.report.summary()
    assert any(result.report.summary() in r.getMessage() for r in caplog.records)


def test_no_chunks_returns_empty_and_warns(tmp_path, caplog):
    stem = _voiced_stem(tmp_path)

    with caplog.at_level(logging.WARNING, logger="music_video_maker.stems"):
        result = slice_stem_for_chunks(stem, (), tmp_path / "stem_chunks")

    assert result.chunks == ()
    assert result.report.chunk_count == 0
    assert caplog.records


# --------------------------------------------------------------------------- #
# Frame-grid identity: the stem slice must match frame_count exactly
# --------------------------------------------------------------------------- #


def test_refuses_a_slice_whose_duration_disagrees_with_its_frame_count(tmp_path, caplog):
    """The same guard slicing.py carries: a conditioning stem that is not
    exactly ``frame_count`` frames long hands Stage 4a a length mismatch, which
    is the drift issue #20 exists to eliminate."""
    stem = _voiced_stem(tmp_path)
    bad = (_chunk(0, 0.0, 5.0, frame_count=GRID.trained_min_frames),)

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemSliceError),
    ):
            slice_stem_for_chunks(stem, bad, tmp_path / "stem_chunks")
    assert caplog.records


def test_refuses_a_chunk_whose_span_runs_backwards(tmp_path, caplog):
    """A reversed span slices to nothing; emitting a zero-length conditioning
    file for a chunk H3 will render is worse than refusing."""
    stem = _voiced_stem(tmp_path)

    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
        pytest.raises(StemSliceError),
    ):
            slice_stem_for_chunks(stem, (_chunk(0, 6.0, 5.0),), tmp_path / "stem_chunks")
    assert caplog.records


def test_a_write_failure_is_logged_and_raised_not_swallowed(tmp_path, caplog):
    stem = _voiced_stem(tmp_path)
    out_dir = tmp_path / "stem_chunks"
    out_dir.mkdir()
    out_dir.chmod(0o500)
    try:
        with (
            caplog.at_level(logging.ERROR, logger="music_video_maker.stems"),
            pytest.raises(StemSliceError),
        ):
                slice_stem_for_chunks(stem, _tiled_chunks(), out_dir)
        assert caplog.records
    finally:
        out_dir.chmod(0o755)


def test_frame_count_identity_holds_across_a_real_sliced_timeline(tmp_path):
    """End to end against Stage 2a's own output: same spans, same durations,
    same frame counts -- the stem tiles the track exactly as the mix did."""
    alignment = make_alignment_result_normal_song()
    master = write_silent_wav(tmp_path / "master.wav", alignment.track_duration)
    mix_chunks = slice_audio(
        master, alignment, DEFAULT_HARDWARE, tmp_path / "chunks", cover_instrumentals=True
    )
    stem = _write_wav_with_tone_spans(
        tmp_path / "vocals.wav",
        alignment.track_duration,
        tuple((s.start, s.end) for s in alignment.segments),
    )

    result = slice_stem_for_chunks(
        stem, mix_chunks, tmp_path / "stem_chunks", master_path=master
    )

    assert len(result.chunks) == len(mix_chunks)
    prev_end = 0.0
    for mix_chunk, stem_chunk in zip(mix_chunks, result.chunks, strict=True):
        assert (stem_chunk.start, stem_chunk.end) == (mix_chunk.start, mix_chunk.end)
        assert stem_chunk.frame_count == mix_chunk.frame_count
        assert stem_chunk.start == pytest.approx(prev_end, abs=1e-6)
        assert _wav_duration_seconds(stem_chunk.audio_file) == pytest.approx(
            GRID.frames_to_seconds(stem_chunk.frame_count), abs=1e-3
        )
        prev_end = stem_chunk.end
