"""Tests for the second-timeline model and the seam (issue #66).

``timelines.py`` is pure -- no ffmpeg, no pydub, no file I/O -- so everything
here is arithmetic and validation. The parts that need a subprocess live in
``tests/test_assembly_timelines.py`` (the measured seam) and
``tests/test_prologue_cli.py`` (the whole pipeline through the mock ComfyUI).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.timelines import (
    DEFAULT_SEGMENT_POSITION,
    SONG_TIMELINE_NAME,
    SeamOverrunError,
    Segment,
    TimelineError,
    build_segments,
    log_placements,
    place_timelines,
    plan_timelines,
    predicted_measurements,
)


def _raw(**overrides) -> dict:
    raw = {"name": "prologue", "audio": "audio/prologue.wav", "script": "prologue.txt"}
    raw.update(overrides)
    return raw


# --------------------------------------------------------------------------- #
# Segment validation
# --------------------------------------------------------------------------- #


def test_a_minimal_segment_defaults_to_playing_before_the_song():
    (segment,) = build_segments([_raw()])
    assert segment.name == "prologue"
    assert segment.position == DEFAULT_SEGMENT_POSITION == "before"
    assert segment.shot_plan is None
    assert segment.order == 0


def test_segments_resolve_their_paths_through_the_callers_resolver():
    (segment,) = build_segments(
        [_raw(shot_plan="plans/prologue.toml")],
        resolve=lambda value: Path("/run") / value,
    )
    assert segment.audio == Path("/run/audio/prologue.wav")
    assert segment.script == Path("/run/prologue.txt")
    assert segment.shot_plan == Path("/run/plans/prologue.toml")


def test_an_unknown_segment_key_is_refused_and_named():
    """The TOML hazard this repo has paid for twice: a bare key written below
    a table binds to it. A ``[[segment]]`` table is the last thing anybody
    adds to a config, so it is the most exposed of the three."""
    with pytest.raises(TimelineError) as excinfo:
        build_segments([_raw(render_width=864)])
    message = str(excinfo.value)
    assert "render_width" in message
    assert "ABOVE the first table" in message


def test_a_reserved_segment_name_is_refused():
    """``chunks_dir/frames`` is continuity's seed frames; a segment called
    "frames" would put one timeline's wavs where another stage writes."""
    with pytest.raises(TimelineError, match="reserved"):
        build_segments([_raw(name="frames")])


@pytest.mark.parametrize("name", ["Prologue", "pro logue", "", "../escape", "prol/ogue"])
def test_a_segment_name_that_is_not_safe_as_a_directory_is_refused(name: str):
    with pytest.raises(TimelineError, match="name must be"):
        build_segments([_raw(name=name)])


def test_two_segments_may_not_share_a_name():
    with pytest.raises(TimelineError, match="duplicate segment name"):
        build_segments([_raw(), _raw(audio="audio/other.wav")])


def test_an_unknown_position_is_refused():
    with pytest.raises(TimelineError, match="position must be"):
        build_segments([_raw(position="during")])


@pytest.mark.parametrize("missing", ["audio", "script"])
def test_a_segment_without_its_two_required_files_is_refused(missing: str):
    raw = _raw()
    del raw[missing]
    with pytest.raises(TimelineError, match=f"{missing} is required"):
        build_segments([raw])


def test_no_segment_table_at_all_is_no_segments():
    assert build_segments(None) == ()
    assert build_segments([]) == ()


def test_a_segment_array_that_is_not_an_array_is_refused():
    with pytest.raises(TimelineError, match=r"\[\[segment\]\] tables"):
        build_segments({"name": "prologue"})


def test_a_segment_entry_that_is_not_a_table_is_refused():
    with pytest.raises(TimelineError, match="must be a table"):
        build_segments(["prologue.wav"])


def test_a_non_string_path_is_refused_rather_than_coerced():
    with pytest.raises(TimelineError, match="must be a path string"):
        build_segments([_raw(audio=3)])


# --------------------------------------------------------------------------- #
# plan_timelines
# --------------------------------------------------------------------------- #


class _Config:
    """The four fields ``plan_timelines`` reads. Deliberately not a real
    ``RunConfig``: this function's contract is those four fields and nothing
    else, and a full config would hide that."""

    def __init__(self, tmp_path: Path, segments=()):
        self.master_audio = tmp_path / "master.wav"
        self.lyrics_file = tmp_path / "lyrics.txt"
        self.chunks_dir = tmp_path / "output" / "chunks"
        self.run_state_file = tmp_path / "output" / "chunks" / "run_state.json"
        self.shot_plan = None
        self.segments = tuple(segments)


def test_a_config_with_no_segments_is_one_timeline_in_the_same_places_as_before(tmp_path: Path):
    config = _Config(tmp_path)
    (song,) = plan_timelines(config)  # type: ignore[arg-type]
    assert song.name == SONG_TIMELINE_NAME
    assert song.is_song
    assert song.fingerprint_name is None
    assert song.chunks_dir == config.chunks_dir
    assert song.run_state_file == config.run_state_file


def test_timelines_are_ordered_before_song_after_in_declaration_order(tmp_path: Path):
    segments = build_segments(
        [
            _raw(name="prologue"),
            _raw(name="coda", position="after", audio="audio/coda.wav", script="coda.txt"),
            _raw(name="titles", audio="audio/titles.wav", script="titles.txt"),
        ]
    )
    timelines = plan_timelines(_Config(tmp_path, segments))  # type: ignore[arg-type]
    assert [t.name for t in timelines] == ["prologue", "titles", "song", "coda"]


def test_each_segment_gets_its_own_chunks_directory_and_run_state(tmp_path: Path):
    """The separate id space is only safe because the files are separate too:
    prologue chunk 3 and song chunk 3 are different shots, and
    ``RunState.results`` is keyed by chunk id."""
    segments = build_segments([_raw()])
    timelines = plan_timelines(_Config(tmp_path, segments))  # type: ignore[arg-type]
    prologue = timelines[0]
    song = timelines[1]
    assert prologue.chunks_dir == song.chunks_dir / "prologue"
    assert prologue.run_state_file == song.chunks_dir / "prologue" / "run_state.json"
    assert prologue.run_state_file != song.run_state_file
    assert prologue.fingerprint_name == "prologue"


def test_a_config_with_no_resolved_run_state_file_still_places_the_song(tmp_path: Path):
    config = _Config(tmp_path)
    config.run_state_file = None
    (song,) = plan_timelines(config)  # type: ignore[arg-type]
    assert song.run_state_file == config.chunks_dir / "run_state.json"


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


def test_placements_lay_timelines_end_to_end_and_pad_audio_up_to_video():
    """The measured "Deathless" case, one timeline in front of it: the chunk
    timeline overshoots its master by 1.837s, so the song's audio is padded
    by that much rather than the whole song being pushed 1.837s late."""
    placements = place_timelines(
        [
            ("prologue", "prologue", 30.0, 28.5),
            ("song", None, 513.917, 512.080),
        ]
    )
    prologue, song = placements
    assert prologue.offset_seconds == 0.0
    assert prologue.pad_seconds == pytest.approx(1.5)
    assert song.offset_seconds == pytest.approx(30.0)
    assert song.pad_seconds == pytest.approx(1.837, abs=1e-6)
    assert song.end_seconds == pytest.approx(543.917)


def test_audio_that_outruns_its_own_video_is_refused_rather_than_trimmed():
    """Padding is the reconciliation; trimming picture is not. Audio longer
    than its video cannot be padded into agreement, and silently cutting the
    end off somebody's dialogue is the failure a seam check exists to stop."""
    with pytest.raises(SeamOverrunError) as excinfo:
        place_timelines([("prologue", "prologue", 10.0, 12.0), ("song", None, 20.0, 20.0)])
    assert "prologue" in str(excinfo.value)
    assert excinfo.value.audio_seconds == 12.0


def test_a_sub_tolerance_overrun_is_float_noise_and_clamps_to_no_pad():
    (placement,) = place_timelines([("song", None, 10.0, 10.0000001)])
    assert placement.pad_seconds == 0.0


def test_log_placements_says_whether_it_measured_or_predicted(caplog):
    """Both callers are worth having and they are not the same claim: one is
    ffprobe on real files, the other is Stage 2 before any GPU time."""
    placements = place_timelines([("prologue", "prologue", 10.0, 9.0), ("song", None, 20.0, 20.0)])
    with caplog.at_level(logging.INFO, logger="music_video_maker.timelines"):
        log_placements(placements, measured=False)
    assert "predicted (Stage 2, no GPU)" in caplog.text
    assert "1.000s of silence inserted at 1 seam(s)" in caplog.text.replace("\n", " ")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="music_video_maker.timelines"):
        log_placements(placements, measured=True)
    assert "[measured]" in caplog.text


def test_log_placements_of_one_timeline_reports_no_seam(caplog):
    placements = place_timelines([("song", None, 20.0, 20.0)])
    with caplog.at_level(logging.INFO, logger="music_video_maker.timelines"):
        log_placements(placements, measured=True)
    assert "Seam total" not in caplog.text


def test_predicted_measurements_reads_stage_2s_own_numbers(tmp_path: Path):
    segments = build_segments([_raw()])
    timelines = plan_timelines(_Config(tmp_path, segments))  # type: ignore[arg-type]
    measurements = predicted_measurements(
        timelines,
        {"prologue": 12.0, "song": 513.917},
        {"prologue": 11.5, "song": 512.080},
    )
    assert measurements == [
        ("prologue", "prologue", 12.0, 11.5),
        ("song", None, 513.917, 512.080),
    ]


# --------------------------------------------------------------------------- #
# The fingerprint discriminator (issue #66's half of issue #34)
# --------------------------------------------------------------------------- #


def _chunk(chunk_id: int, timeline: str | None) -> contracts.AudioChunk:
    return contracts.AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id:03d}.wav"),
        start=0.0,
        end=5.167,
        text="",
        frame_count=124,
        timeline=timeline,
    )


def test_two_timelines_chunks_with_identical_spans_do_not_compare_equal():
    """Prologue chunk 3 and song chunk 3 can trivially share a span, a frame
    count and a resolution -- every field the fingerprint had before. Without
    the discriminator, --resume would hand one to the other and report a
    clean match, which is #34's failure along a new axis."""
    song = contracts.ChunkFingerprint.of(_chunk(3, None))
    prologue = contracts.ChunkFingerprint.of(_chunk(3, "prologue"))
    assert song != prologue
    assert song.timeline_differences(prologue) == ("timeline",)


def test_timeline_is_in_the_timeline_tier_so_no_resume_flag_forgives_it():
    """A chunk from the wrong timeline is in the wrong *place*, not merely
    stale -- it is somebody else's clip, and --ignore-prompt-changes must not
    reach it."""
    assert "timeline" in contracts.ChunkFingerprint.TIMELINE_FIELDS
    assert "timeline" not in contracts.ChunkFingerprint.CONTENT_FIELDS
    assert "timeline" not in contracts.ChunkFingerprint.CONDITIONING_FIELDS


def test_a_song_chunk_records_none_so_every_pre_segment_state_file_still_matches():
    """The reason this needed no schema_version bump: an old file has no
    ``timeline`` key, which deserializes to None, which is exactly what a
    song chunk records today."""
    from music_video_maker.resilience import _deserialize_fingerprint, _serialize_fingerprint

    song = contracts.ChunkFingerprint.of(_chunk(3, None))
    old_file_payload = _serialize_fingerprint(song)
    del old_file_payload["timeline"]
    assert _deserialize_fingerprint(old_file_payload) == song


def test_a_segments_fingerprint_round_trips_through_the_state_file():
    from music_video_maker.resilience import _deserialize_fingerprint, _serialize_fingerprint

    prologue = contracts.ChunkFingerprint.of(_chunk(3, "prologue"))
    assert _deserialize_fingerprint(_serialize_fingerprint(prologue)) == prologue


def test_a_segment_is_a_frozen_value_so_nothing_edits_it_after_validation():
    (segment,) = build_segments([_raw()])
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        segment.name = "other"  # type: ignore[misc]
    assert isinstance(segment, Segment)
