"""Tests for the generic (time, label) marker contract (issue #22, step 2).

Concert mode's timeline source is a labelled-marker CSV, not an Ableton
``.als`` parser -- issue #22's own step 1 never confirmed the band uses
Ableton, and the schema is version-dependent even if they do. Everything
here is pure stdlib string/CSV work: no subprocess, no filesystem writes,
no GPU, no ComfyUI. See :mod:`music_video_maker.markers` for the contract
this exercises.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest

from music_video_maker.contracts import AlignmentResult
from music_video_maker.markers import (
    Marker,
    MarkerCsvSource,
    MarkerError,
    MarkerSection,
    MarkerSource,
    MarkerTrack,
    marker_sections,
    parse_marker_csv,
    parse_marker_time,
    read_marker_csv,
    to_alignment_result,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "markers" / "example_click.csv"

EXPECTED_FIXTURE_LABELS = (
    "Count-in",
    "Intro",
    "Verse 1",
    "Chorus 1",
    "Verse 2",
    "Chorus 2",
    "Solo",
    "Outro",
)


# --------------------------------------------------------------------------- #
# parse_marker_time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.5", 12.5),
        ("12", 12.0),
        ("0", 0.0),
        ("0.0", 0.0),
        ("  8.25  ", 8.25),
    ],
)
def test_parse_marker_time_plain_seconds(raw: str, expected: float) -> None:
    assert parse_marker_time(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1:02.5", 62.5),
        ("0:00", 0.0),
        ("2:00", 120.0),
        ("01:30", 90.0),
    ],
)
def test_parse_marker_time_mmss(raw: str, expected: float) -> None:
    assert parse_marker_time(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("00:01:02.500", 62.5),
        ("1:00:00", 3600.0),
        ("00:00:00", 0.0),
    ],
)
def test_parse_marker_time_hhmmss(raw: str, expected: float) -> None:
    assert parse_marker_time(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "abc",
        "1:2:3:4",
        "1:",
        ":30",
        "1:xy",
        "nan",
        "inf",
        "-inf",
        "-5",
        "-1:30",
    ],
)
def test_parse_marker_time_rejects(raw: str) -> None:
    with pytest.raises(MarkerError):
        parse_marker_time(raw)


# --------------------------------------------------------------------------- #
# parse_marker_csv -- happy paths
# --------------------------------------------------------------------------- #


def test_parse_marker_csv_happy_path() -> None:
    text = "0.0,Intro\n10.0,Verse\n20.0,Chorus\n"
    track = parse_marker_csv(text, source="<inline>")

    assert track.source == "<inline>"
    assert track.markers == (
        Marker(time=0.0, label="Intro"),
        Marker(time=10.0, label="Verse"),
        Marker(time=20.0, label="Chorus"),
    )
    assert track.labels == ("Intro", "Verse", "Chorus")


def test_parse_marker_csv_header_row_skipped() -> None:
    text = "time,label\n0.0,Intro\n10.0,Verse\n"
    track = parse_marker_csv(text)
    assert track.labels == ("Intro", "Verse")


def test_parse_marker_csv_header_row_case_insensitive() -> None:
    text = "TIME,label\n0.0,Intro\n"
    track = parse_marker_csv(text)
    assert track.labels == ("Intro",)


def test_parse_marker_csv_comments_and_blanks_skipped() -> None:
    text = "\n".join(
        [
            "# this is a provenance comment",
            "",
            "time,label",
            "0.0,Intro",
            "   ",
            "# another comment",
            "10.0,Verse",
            "",
        ]
    )
    track = parse_marker_csv(text)
    assert track.labels == ("Intro", "Verse")


def test_parse_marker_csv_quoted_label_with_comma() -> None:
    text = '0.0,"Verse, reprise"\n10.0,Outro\n'
    track = parse_marker_csv(text)
    assert track.markers[0].label == "Verse, reprise"


def test_parse_marker_csv_both_time_formats_in_one_file() -> None:
    text = "0.0,Intro\n1:02.5,Verse\n90,Chorus\n"
    track = parse_marker_csv(text)
    assert [m.time for m in track.markers] == pytest.approx([0.0, 62.5, 90.0])


# --------------------------------------------------------------------------- #
# parse_marker_csv -- errors (all MarkerError, all name the 1-based line
# number and the offending text)
# --------------------------------------------------------------------------- #


def test_parse_marker_csv_empty_text_raises() -> None:
    with pytest.raises(MarkerError):
        parse_marker_csv("")


def test_parse_marker_csv_only_header_and_comments_raises() -> None:
    with pytest.raises(MarkerError):
        parse_marker_csv("time,label\n# nothing here\n\n")


def test_parse_marker_csv_row_with_too_few_fields_raises(caplog: pytest.LogCaptureFixture) -> None:
    text = "0.0,Intro\n10.0\n20.0,Chorus\n"
    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.markers"),
        pytest.raises(MarkerError) as exc_info,
    ):
        parse_marker_csv(text)
    assert "2" in str(exc_info.value)
    assert "10.0" in str(exc_info.value)
    assert any("2" in r.getMessage() for r in caplog.records)


def test_parse_marker_csv_unparseable_time_raises(caplog: pytest.LogCaptureFixture) -> None:
    text = "0.0,Intro\nabc,Verse\n"
    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.markers"),
        pytest.raises(MarkerError) as exc_info,
    ):
        parse_marker_csv(text)
    assert "2" in str(exc_info.value)
    assert "abc" in str(exc_info.value)


def test_parse_marker_csv_negative_time_raises() -> None:
    text = "0.0,Intro\n-5.0,Verse\n"
    with pytest.raises(MarkerError) as exc_info:
        parse_marker_csv(text)
    assert "2" in str(exc_info.value)


def test_parse_marker_csv_blank_label_raises() -> None:
    text = "0.0,Intro\n10.0,\n"
    with pytest.raises(MarkerError) as exc_info:
        parse_marker_csv(text)
    assert "2" in str(exc_info.value)


def test_parse_marker_csv_blank_label_whitespace_only_raises() -> None:
    text = "0.0,Intro\n10.0,   \n"
    with pytest.raises(MarkerError):
        parse_marker_csv(text)


def test_parse_marker_csv_non_increasing_times_raises(caplog: pytest.LogCaptureFixture) -> None:
    text = "0.0,Intro\n10.0,Verse\n5.0,Chorus\n"
    with (
        caplog.at_level(logging.ERROR, logger="music_video_maker.markers"),
        pytest.raises(MarkerError) as exc_info,
    ):
        parse_marker_csv(text)
    assert "3" in str(exc_info.value)


def test_parse_marker_csv_equal_times_raises() -> None:
    """Two markers at the same instant would produce a zero-length section --
    a class of bug this project has already paid for (CLAUDE.md's chunk
    timeline invariants). Equal is rejected, not just decreasing."""
    text = "0.0,Intro\n10.0,Verse\n10.0,Chorus\n"
    with pytest.raises(MarkerError) as exc_info:
        parse_marker_csv(text)
    assert "3" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# read_marker_csv
# --------------------------------------------------------------------------- #


def test_read_marker_csv_matches_parse_marker_csv_on_same_text(tmp_path: Path) -> None:
    text = "time,label\n0.0,Intro\n10.0,Verse\n"
    path = tmp_path / "markers.csv"
    path.write_text(text, encoding="utf-8")

    from_file = read_marker_csv(path)
    from_text = parse_marker_csv(text)

    assert from_file.markers == from_text.markers
    assert from_file.source == str(path)


def test_read_marker_csv_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "markers.csv"
    path.write_text("0.0,Intro\n10.0,Verse\n", encoding="utf-8")
    track = read_marker_csv(str(path))
    assert track.labels == ("Intro", "Verse")


# --------------------------------------------------------------------------- #
# marker_sections
# --------------------------------------------------------------------------- #


def test_marker_sections_contiguous_and_ordered() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n25.0,Chorus\n")
    sections = marker_sections(track, track_duration=40.0)

    assert len(sections) == 3
    assert sections[0] == MarkerSection(index=0, label="Intro", start=0.0, end=10.0)
    assert sections[1] == MarkerSection(index=1, label="Verse", start=10.0, end=25.0)
    assert sections[2] == MarkerSection(index=2, label="Chorus", start=25.0, end=40.0)

    for i in range(len(sections) - 1):
        assert sections[i].end == sections[i + 1].start

    assert sections[-1].end == pytest.approx(40.0)


def test_marker_sections_duration_property() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    sections = marker_sections(track, track_duration=15.0)
    assert sections[0].duration == pytest.approx(10.0)
    assert sections[1].duration == pytest.approx(5.0)


def test_marker_sections_durations_sum_to_track_minus_first_marker() -> None:
    track = parse_marker_csv("5.0,Intro\n10.0,Verse\n25.0,Chorus\n")
    sections = marker_sections(track, track_duration=42.0)
    assert sum(s.duration for s in sections) == pytest.approx(42.0 - 5.0)


def test_marker_sections_uncovered_head_logs_info_not_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    track = parse_marker_csv("5.0,Intro\n10.0,Verse\n")
    with caplog.at_level(logging.INFO, logger="music_video_maker.markers"):
        sections = marker_sections(track, track_duration=20.0)
    assert sections[0].start == pytest.approx(5.0)
    assert any(
        r.levelno == logging.INFO and "5" in r.getMessage() for r in caplog.records
    )
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_marker_sections_first_marker_at_zero_does_not_log_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with caplog.at_level(logging.INFO, logger="music_video_maker.markers"):
        marker_sections(track, track_duration=20.0)
    assert not any(r.levelno == logging.INFO for r in caplog.records)


@pytest.mark.parametrize("bad_duration", [math.inf, -math.inf, math.nan])
def test_marker_sections_rejects_non_finite_track_duration(bad_duration: float) -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with pytest.raises(MarkerError):
        marker_sections(track, track_duration=bad_duration)


def test_marker_sections_rejects_duration_at_or_before_last_marker() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with pytest.raises(MarkerError):
        marker_sections(track, track_duration=10.0)  # equal to last marker
    with pytest.raises(MarkerError):
        marker_sections(track, track_duration=9.0)  # before last marker


def test_marker_sections_empty_track_raises() -> None:
    track = MarkerTrack(markers=(), source="<empty>")
    with pytest.raises(MarkerError):
        marker_sections(track, track_duration=10.0)


# --------------------------------------------------------------------------- #
# to_alignment_result
# --------------------------------------------------------------------------- #


def test_to_alignment_result_label_as_text_true_sets_text_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with caplog.at_level(logging.WARNING, logger="music_video_maker.markers"):
        result = to_alignment_result(track, track_duration=20.0, label_as_text=True)

    assert [s.text for s in result.segments] == ["Intro", "Verse"]
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert any("lyric" in r.getMessage() for r in caplog.records)


def test_to_alignment_result_label_as_text_false_leaves_text_empty_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with caplog.at_level(logging.WARNING, logger="music_video_maker.markers"):
        result = to_alignment_result(track, track_duration=20.0, label_as_text=False)

    assert [s.text for s in result.segments] == ["", ""]
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_to_alignment_result_requires_label_as_text_keyword() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    with pytest.raises(TypeError):
        to_alignment_result(track, 20.0, True)  # type: ignore[misc]


def test_to_alignment_result_track_duration_roundtrips() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n")
    result = to_alignment_result(track, track_duration=33.5, label_as_text=False)
    assert isinstance(result, AlignmentResult)
    assert result.track_duration == pytest.approx(33.5)


def test_to_alignment_result_voiced_duration() -> None:
    track = parse_marker_csv("5.0,Intro\n10.0,Verse\n25.0,Chorus\n")
    result = to_alignment_result(track, track_duration=42.0, label_as_text=False)
    # sections are contiguous from the first marker to track_duration, so
    # voiced_duration (sum of segment durations) covers exactly that span.
    assert result.voiced_duration == pytest.approx(42.0 - 5.0)


def test_to_alignment_result_segments_contiguous_and_empty_words_and_characters() -> None:
    track = parse_marker_csv("0.0,Intro\n10.0,Verse\n25.0,Chorus\n")
    result = to_alignment_result(track, track_duration=40.0, label_as_text=False)

    segments = result.segments
    assert segments[0].start == pytest.approx(0.0)
    for i in range(len(segments) - 1):
        assert segments[i].end == segments[i + 1].start
    assert segments[-1].end == pytest.approx(40.0)

    for i, segment in enumerate(segments):
        assert segment.index == i
        assert segment.words == ()
        assert segment.characters == ()


# --------------------------------------------------------------------------- #
# MarkerSource protocol / MarkerCsvSource
# --------------------------------------------------------------------------- #


def test_marker_csv_source_satisfies_marker_source_protocol(tmp_path: Path) -> None:
    path = tmp_path / "markers.csv"
    path.write_text("0.0,Intro\n10.0,Verse\n", encoding="utf-8")
    source = MarkerCsvSource(path=path)
    assert isinstance(source, MarkerSource)


def test_marker_csv_source_read_matches_read_marker_csv(tmp_path: Path) -> None:
    path = tmp_path / "markers.csv"
    path.write_text("0.0,Intro\n10.0,Verse\n", encoding="utf-8")
    source = MarkerCsvSource(path=path)
    assert source.read() == read_marker_csv(path)


# --------------------------------------------------------------------------- #
# shipped fixture
# --------------------------------------------------------------------------- #


def test_shipped_fixture_parses() -> None:
    track = read_marker_csv(FIXTURE_PATH)
    assert track.labels == EXPECTED_FIXTURE_LABELS
    assert len(track.markers) == 8


def test_shipped_fixture_yields_eight_contiguous_sections() -> None:
    track = read_marker_csv(FIXTURE_PATH)
    last_marker_time = track.markers[-1].time
    track_duration = last_marker_time + 25.0  # outro tail, > last marker
    sections = marker_sections(track, track_duration=track_duration)

    assert len(sections) == 8
    assert tuple(s.label for s in sections) == EXPECTED_FIXTURE_LABELS
    for i in range(len(sections) - 1):
        assert sections[i].end == sections[i + 1].start
    assert sections[-1].end == pytest.approx(track_duration)


def test_shipped_fixture_converts_cleanly_to_alignment_result() -> None:
    track = read_marker_csv(FIXTURE_PATH)
    track_duration = track.markers[-1].time + 25.0
    result = to_alignment_result(track, track_duration=track_duration, label_as_text=False)

    assert len(result.segments) == 8
    assert result.track_duration == pytest.approx(track_duration)
    assert result.segments[0].start == pytest.approx(track.markers[0].time)


def test_shipped_fixture_has_at_least_one_mmss_time_format() -> None:
    """The fixture must exercise both time formats parse_marker_time accepts
    (issue #22 step 2's explicit requirement) -- assert the raw file actually
    contains a colon-separated timestamp, not just plain seconds."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    data_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.lower().startswith("time")
    ]
    assert any(":" in line.split(",", 1)[0] for line in data_lines)
    assert any(":" not in line.split(",", 1)[0] for line in data_lines)
