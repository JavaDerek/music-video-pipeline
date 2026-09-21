"""Reading section markers out of an Ableton set (issue #22, step 1).

Every test here runs against a **synthetic** fixture this repo authored
(``tests/fixtures/markers/synthetic_live_set.als.xml``), because issue #22's
step 1 is unanswered: nobody has confirmed the band uses Ableton, which major
version, or whether the sections are in Locators at all. So these tests pin
what the reader does with the format as *documented* -- they are not evidence
that a real export parses, and the module says so in its own docstring.

What they do prove is the two refusals, which is the part worth having
before a real file arrives: a set with no Locators, and a set with tempo
automation, both fail loudly rather than producing a plausible timeline
nobody authored.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from music_video_maker.ableton import (
    AbletonError,
    AbletonLocatorSource,
    read_ableton_locators,
    read_tempo_bpm,
)
from music_video_maker.markers import (
    MarkerError,
    MarkerSource,
    format_marker_csv,
    marker_sections,
    parse_marker_csv,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "markers" / "synthetic_live_set.als.xml"

EXPECTED = [
    (0.0, "Count-in"),
    (4.0, "Intro"),
    (12.0, "Verse 1"),
    (28.0, "Chorus 1, first"),
]
"""0, 8, 24, 56 beats at 120 BPM. The arithmetic is the whole reason this
module exists rather than a generic XML walk: a Live set stores marker
positions in BEATS and a video timeline is in seconds."""


def _als(tmp_path: Path, xml: str | None = None) -> Path:
    """Write the fixture out as a real (gzipped) ``.als``."""
    path = tmp_path / "set.als"
    body = xml if xml is not None else FIXTURE.read_text()
    path.write_bytes(gzip.compress(body.encode("utf-8")))
    return path


def _fixture_without(tag: str) -> str:
    root = ET.fromstring(FIXTURE.read_text())
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_locator_beats_become_seconds_at_the_sets_own_tempo(tmp_path: Path):
    track = read_ableton_locators(_als(tmp_path))
    assert [(m.time, m.label) for m in track.markers] == EXPECTED


def test_plain_xml_is_read_too_so_the_fixture_can_live_in_a_diff():
    """A ``.als`` is gzipped, and a gzipped fixture cannot be reviewed,
    corrected by hand, or committed without a provenance-manifest entry for a
    file whose provenance is "we made it up"."""
    track = read_ableton_locators(FIXTURE)
    assert [m.label for m in track.markers] == [label for _t, label in EXPECTED]


def test_a_label_containing_a_comma_survives_the_round_trip_to_csv(tmp_path: Path):
    """The CSV is the contract (issue #22); this reader is one producer of
    it, so what it produces has to be readable by the reader that consumes
    it -- including "Chorus 1, first"."""
    track = read_ableton_locators(_als(tmp_path))
    text = format_marker_csv(track, provenance="from set.als, 120 BPM (synthetic)")
    assert text.startswith("# from set.als")
    reparsed = parse_marker_csv(text, source="round-trip")
    assert [(m.time, m.label) for m in reparsed.markers] == EXPECTED


def test_the_track_it_produces_is_what_stage_2_already_consumes(tmp_path: Path):
    """The point of landing on ``MarkerTrack`` rather than inventing a second
    shape: ``marker_sections`` works on it unchanged."""
    track = read_ableton_locators(_als(tmp_path))
    sections = marker_sections(track, track_duration=40.0)
    assert [s.label for s in sections] == [label for _t, label in EXPECTED]
    assert sections[-1].end == 40.0
    assert all(a.end == b.start for a, b in zip(sections, sections[1:], strict=False))


def test_the_source_is_a_marker_source(tmp_path: Path):
    source = AbletonLocatorSource(path=_als(tmp_path))
    assert isinstance(source, MarkerSource)
    assert [m.label for m in source.read().markers] == [label for _t, label in EXPECTED]


def test_the_source_records_where_it_came_from(tmp_path: Path):
    path = _als(tmp_path)
    assert read_ableton_locators(path).source == str(path)


# --------------------------------------------------------------------------- #
# The refusals -- the part worth having before a real file arrives
# --------------------------------------------------------------------------- #


def test_tempo_automation_is_refused_rather_than_converted_with_the_manual_value(
    tmp_path: Path,
):
    """A Locator's time is in beats, so an automated tempo makes beats ->
    seconds an integral. Using the manual value anyway places every marker
    after the first tempo change wrong, progressively worse through the song,
    with nothing to notice it -- exactly the drift issue #22 exists to stop."""
    xml = FIXTURE.read_text().replace(
        "<AutomationTarget Id=\"8\">",
        "<ArrangerAutomation><Events>"
        "<FloatEvent Id=\"1\" Time=\"0\" Value=\"120\" />"
        "<FloatEvent Id=\"2\" Time=\"64\" Value=\"96\" />"
        "</Events></ArrangerAutomation><AutomationTarget Id=\"8\">",
    )
    with pytest.raises(AbletonError) as excinfo:
        read_ableton_locators(_als(tmp_path, xml))
    assert "tempo automation" in str(excinfo.value)
    assert "2 tempo events" in str(excinfo.value)


def test_a_set_with_no_locators_is_refused_and_names_the_alternatives(tmp_path: Path):
    """Reading clip names instead would produce a plausible-looking timeline
    nobody authored -- and which of the three places the sections live in is
    issue #22's own open question 2."""
    with pytest.raises(AbletonError) as excinfo:
        read_ableton_locators(_als(tmp_path, _fixture_without("Locators")))
    assert "Locators only" in str(excinfo.value)
    assert "clip name" in str(excinfo.value)


def test_a_set_with_no_tempo_is_refused_unless_the_caller_supplies_one(tmp_path: Path):
    path = _als(tmp_path, _fixture_without("MainTrack"))
    with pytest.raises(AbletonError, match="no <Tempo> found"):
        read_ableton_locators(path)

    track = read_ableton_locators(path, tempo_bpm=60.0)
    assert [m.time for m in track.markers] == [0.0, 8.0, 24.0, 56.0]


def test_an_override_that_disagrees_with_the_file_says_so(tmp_path: Path, caplog):
    with caplog.at_level("WARNING"):
        track = read_ableton_locators(_als(tmp_path), tempo_bpm=60.0)
    assert "overriding the set's own tempo" in caplog.text
    assert [m.time for m in track.markers] == [0.0, 8.0, 24.0, 56.0]


def test_two_locators_at_one_instant_are_refused_like_two_csv_rows_are(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Time Value="24" />', '<Time Value="8" />')
    with pytest.raises(AbletonError, match="zero-length section"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_locator_before_the_start_of_the_arrangement_is_refused(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Time Value="8" />', '<Time Value="-4" />')
    with pytest.raises(AbletonError, match="before the start"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_locator_missing_its_name_is_refused(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Name Value="Intro" />', '<Name Value="  " />')
    with pytest.raises(AbletonError, match="missing a Time or a Name"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_non_numeric_locator_time_is_refused(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Time Value="8" />', '<Time Value="bar 3" />')
    with pytest.raises(AbletonError, match="non-numeric Time"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_non_numeric_tempo_is_refused(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Manual Value="120" />', '<Manual Value="fast" />')
    with pytest.raises(AbletonError, match="not a number"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_zero_tempo_is_refused_before_it_divides_by_zero(tmp_path: Path):
    xml = FIXTURE.read_text().replace('<Manual Value="120" />', '<Manual Value="0" />')
    with pytest.raises(AbletonError, match="must be positive"):
        read_ableton_locators(_als(tmp_path, xml))


def test_a_file_that_is_not_xml_is_refused(tmp_path: Path):
    path = tmp_path / "set.als"
    path.write_bytes(gzip.compress(b"this is not a Live set"))
    with pytest.raises(AbletonError, match="not parseable as XML"):
        read_ableton_locators(path)


def test_a_file_that_claims_to_be_gzipped_and_is_not_is_refused(tmp_path: Path):
    path = tmp_path / "set.als"
    path.write_bytes(b"\x1f\x8b" + b"truncated")
    with pytest.raises(AbletonError, match="could not be decompressed"):
        read_ableton_locators(path)


def test_an_ableton_error_is_a_marker_error_so_one_except_catches_both(tmp_path: Path):
    """A caller handling "the marker source was unusable" should not have to
    know which producer it was handed."""
    assert issubclass(AbletonError, MarkerError)


def test_the_master_track_spelling_is_read_too(tmp_path: Path):
    """Live renamed Master to Main in 12, and which version the band runs is
    exactly the question step 1 has not answered."""
    xml = FIXTURE.read_text().replace("MainTrack", "MasterTrack")
    assert read_tempo_bpm(ET.fromstring(xml), source="test") == 120.0
