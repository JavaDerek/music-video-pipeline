"""Tests for the NCCF voicing measurement (issue #96).

Offline by construction: every waveform is synthesized by
``tests.harness.factories`` from stdlib ``math``/``random``, written with
stdlib ``wave``, and read back by the module under test. No ffmpeg, no
network, no committed binary.

The fixtures differ on **periodicity**, which is the axis #96 exists to
measure -- not on level and not on brightness, which is what issue #71's
consonant-band share already sees. One of them, ``plucked_samples``, is here
to pin a *limit* rather than a capability: a plucked string is periodic and
must read as voiced, because periodicity genuinely cannot tell it from a sung
vowel. A test that asserted otherwise would be asserting a discrimination the
measurement does not have.
"""

from __future__ import annotations

import wave

import pytest

from music_video_maker import voicing
from tests.harness.factories import (
    VOICING_SAMPLE_RATE,
    near_silence_samples,
    noise_samples,
    plucked_samples,
    voiced_samples,
    write_samples_wav,
)

SPAN_SECONDS = 2.0
# Chosen once, from the near-silence fixture's own level: dither at amplitude
# 20/32768 measures around -69dBFS and every real fixture here sits near
# -20dBFS, so -45 separates them with ~24dB of margin on both sides. The real
# check derives its floor from the track's own median instead (see
# alignment_quality.VOICING_FRAME_LEVEL_DROP_DB); this is the fixed number
# these unit tests need to be about periodicity alone.
LEVEL_FLOOR_DBFS = -45.0


def _four_source_wav(tmp_path):
    """One track: voice, pluck, noise, near-silence, two seconds each."""
    return write_samples_wav(
        tmp_path / "track.wav",
        [
            voiced_samples(SPAN_SECONDS),
            plucked_samples(SPAN_SECONDS),
            noise_samples(SPAN_SECONDS),
            near_silence_samples(SPAN_SECONDS),
        ],
    )


def _measure(audio, index):
    return voicing.measure_span(
        audio,
        index * SPAN_SECONDS,
        (index + 1) * SPAN_SECONDS,
        level_floor_dbfs=LEVEL_FLOOR_DBFS,
    )


def test_a_harmonic_voice_is_voiced_and_broadband_noise_is_not(tmp_path):
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    voice = _measure(audio, 0)
    noise = _measure(audio, 2)

    assert voice.voiced_fraction > 0.9
    assert noise.voiced_fraction == 0.0
    # The threshold-free view of the same evidence must agree with the binned
    # one, or the VOICED_FRAME_NCCF constant is doing the separating.
    assert voice.median_nccf > noise.median_nccf
    assert voice.hnr_db > noise.hnr_db


def test_loud_noise_is_not_saved_by_being_loud(tmp_path):
    # The thing #71 cannot do: these two are at comparable levels and the
    # noise is the brighter of the pair. Only periodicity separates them.
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    voice = _measure(audio, 0)
    noise = _measure(audio, 2)

    assert abs(voice.level_dbfs - noise.level_dbfs) < 6.0
    assert voice.voiced_fraction - noise.voiced_fraction > 0.9


def test_a_plucked_string_reads_as_voiced_which_is_the_documented_limit(tmp_path):
    # Issue #96's headline case is a phantom lyric over a guitar note. This
    # check CANNOT catch it, and that is recorded here as an assertion rather
    # than only in prose: a periodicity measure scores a pluck exactly as high
    # as a vowel. See music_video_maker/voicing.py's "What this CANNOT
    # distinguish".
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    pluck = _measure(audio, 1)

    assert pluck.voiced_fraction > 0.9


def test_jitter_and_pitch_spread_separate_a_pluck_from_a_voice(tmp_path):
    # The two statistics the module reports and deliberately does NOT
    # threshold. They are the candidate answer to the case above, and this
    # test records what they do on synthetic signals -- which is evidence
    # about the statistics, not about any song. Calibration on a real master
    # is what would justify a lint; see docs/voicing-corpus.md.
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    voice = _measure(audio, 0)
    pluck = _measure(audio, 1)

    assert voice.f0_jitter_pct is not None
    assert pluck.f0_jitter_pct is not None
    assert voice.f0_jitter_pct > 10 * pluck.f0_jitter_pct
    assert voice.f0_span_semitones > pluck.f0_span_semitones


def test_near_silence_is_unmeasured_not_unvoiced(tmp_path):
    # #93's rule: a zero from a detector is not evidence of absence unless
    # something asked the second question. Every frame is gated out, so the
    # honest report is "nothing was measured here".
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    quiet = _measure(audio, 3)

    assert quiet.frames_measured == 0
    assert quiet.frames_total > 0
    assert quiet.median_f0_hz is None
    assert quiet.f0_jitter_pct is None


def test_without_a_level_floor_nothing_is_gated_out(tmp_path):
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    quiet = voicing.measure_span(audio, 6.0, 8.0, level_floor_dbfs=None)

    assert quiet.frames_measured == quiet.frames_total > 0
    assert quiet.voiced_fraction == 0.0  # dither is aperiodic, it is just quiet


def test_f0_is_recovered_within_a_few_percent(tmp_path):
    path = write_samples_wav(tmp_path / "tone.wav", [voiced_samples(SPAN_SECONDS, f0=147.0)])
    audio = voicing.load_mono_pcm16(path)

    measurement = voicing.measure_span(audio, 0.0, SPAN_SECONDS, level_floor_dbfs=LEVEL_FLOOR_DBFS)

    assert abs(measurement.median_f0_hz - 147.0) / 147.0 < 0.05


def test_a_span_too_short_to_frame_returns_none_rather_than_zero(tmp_path):
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    assert voicing.measure_span(audio, 0.0, 0.01) is None
    assert voicing.measure_span(audio, 1.0, 1.0) is None
    assert voicing.measure_span(audio, 5.0, 4.0) is None


def test_frames_are_strided_not_truncated_when_a_span_is_long(tmp_path):
    path = write_samples_wav(tmp_path / "long.wav", [voiced_samples(6.0)])
    audio = voicing.load_mono_pcm16(path)

    capped = voicing.measure_span(audio, 0.0, 6.0, max_frames=12)

    assert capped.frames_total == 12
    # Strided across the whole span rather than stopping after 12 hops: a
    # truncating cap would describe the first 0.25s and call it the segment.
    assert capped.voiced_fraction > 0.9


def test_span_level_dbfs_reports_minus_infinity_for_digital_silence(tmp_path):
    path = write_samples_wav(tmp_path / "silent.wav", [[0.0] * VOICING_SAMPLE_RATE])
    audio = voicing.load_mono_pcm16(path)

    assert voicing.span_level_dbfs(audio, 0.0, 1.0) == -float("inf")
    assert voicing.span_level_dbfs(audio, 1.0, 0.5) == -float("inf")
    assert voicing.span_level_dbfs(audio, 5.0, 6.0) == -float("inf")


def test_a_digitally_silent_span_measures_as_unvoiced_without_dividing_by_zero(tmp_path):
    path = write_samples_wav(tmp_path / "silent.wav", [[0.0] * (VOICING_SAMPLE_RATE * 2)])
    audio = voicing.load_mono_pcm16(path)

    measurement = voicing.measure_span(audio, 0.0, 2.0, level_floor_dbfs=None)

    assert measurement.level_dbfs == -float("inf")
    assert measurement.voiced_fraction == 0.0
    assert measurement.median_nccf == 0.0


def test_a_stereo_decode_is_refused_rather_than_reinterpreted(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(VOICING_SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * 2 * VOICING_SAMPLE_RATE)

    with pytest.raises(voicing.VoicingUnavailable, match="mono 16-bit"):
        voicing.load_mono_pcm16(path)


def test_an_unreadable_file_raises_the_typed_error(tmp_path):
    path = tmp_path / "not-a-wav.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")

    with pytest.raises(voicing.VoicingUnavailable):
        voicing.load_mono_pcm16(path)

    with pytest.raises(voicing.VoicingUnavailable):
        voicing.load_mono_pcm16(tmp_path / "absent.wav")


def test_pcm_audio_duration_is_derived_from_the_samples(tmp_path):
    audio = voicing.load_mono_pcm16(_four_source_wav(tmp_path))

    assert audio.duration == pytest.approx(4 * SPAN_SECONDS, abs=0.01)
    assert voicing.PcmAudio(samples=audio.samples, sample_rate=0).duration == 0.0


def test_an_eight_minute_track_costs_seconds_not_minutes(tmp_path):
    # Not a wall-clock assertion (a shared CI box makes those flaky); an
    # arithmetic one about how much work the frame cap admits. 41 voiced
    # segments of 5s each is the shape of "Deathless", and the module's own
    # benchmark measured 0.28ms/frame.
    frames_per_segment = min(
        voicing.MAX_FRAMES_PER_SPAN, int(5.0 / voicing.HOP_S)
    )
    assert frames_per_segment * 41 * 0.00028 < 5.0
