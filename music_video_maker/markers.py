"""Generic (time, label) marker contract for concert mode (issue #22, step 2).

Concert mode replaces Stage 1 forced alignment with a timeline the band
already authored in a DAW: click-track locators marking "Verse 1", "Chorus
2", "Solo", and so on. Issue #22's own step 1 flags that this is *believed*
to be Ableton Live but never confirmed, and an Ableton ``.als`` set is
gzipped XML whose schema changes between major versions -- writing a parser
for it now would be guessing at a format nobody has verified the band even
produces. So the input contract this module defines is a labelled-marker
CSV, not a DAW file: "A generic labelled-marker format (time, label CSV)
should probably be the *actual* input contract, with an Ableton reader as
one producer of it. That keeps the pipeline honest if the band ever changes
DAW, and makes the whole thing testable offline with no Ableton installed"
(issue #22). :class:`MarkerSource` is the seam an Ableton (or Logic, or
Cubase) reader lands on later, once step 1 is actually settled -- it is a
``Protocol``, not a stub that raises ``NotImplementedError``, because there
is nothing to implement yet and a stub would only be able to lie about that.

A label is a section name, not a lyric -- and it is easy to fall into this
trap silently. :class:`~music_video_maker.contracts.AlignedSegment.text` is
consumed downstream as the literal lyric line composed into the render
prompt; a marker label like "Chorus 2" or "Stab" is a *section identity*,
never a sung word. :func:`to_alignment_result` therefore takes a
keyword-only ``label_as_text`` with no default -- the caller must say which
it means -- and logs a WARNING every time a caller sets it ``True``, because
concert mode's prompt composer (not built by this step) must never read a
label as a lyric line.

Concert mode has no cast and no lip-sync (issue #22, Change 2): a projection
behind a live band should not show animated versions of the band, and
``MiniMaxH3ReferenceToVideo`` exists specifically to drive a face from an
audio stem. **A marker-derived timeline must never be run against that
node.** The contract this module produces is timeline-shaped
(``AlignmentResult``) so Stage 2 slicing is reusable as-is, not because
concert mode wants a lip-synced face -- it has none.

CSV format
----------
UTF-8 text, one marker per line: ``<time>,<label>``.

* An optional header line whose first field is exactly ``time``
  (case-insensitive) is skipped.
* Blank lines, and lines whose first non-space character is ``#``, are
  skipped -- so a file can carry a provenance comment (see the shipped
  fixture, ``tests/fixtures/markers/example_click.csv``).
* The label may contain a comma if quoted (``12.5,"Verse, reprise"``); this
  is parsed with the stdlib ``csv`` module, never by hand-splitting on commas.
* ``<time>`` accepts either plain seconds (``12.5``, ``12``) or
  ``[HH:]MM:SS[.mmm]`` (``1:02.5``, ``00:01:02.500``) -- both are common DAW
  marker/locator export formats. :func:`parse_marker_time` is the one place
  that knows this and is tested standalone.

Example::

    # click track export for "Song Title", synthetic
    time,label
    0.0,Count-in
    8.0,Intro
    24.0,Verse 1
    1:04.500,Chorus 1

Contiguity is the invariant this whole feature exists to protect: "A
backdrop that drifts against a click track is a show that falls apart in
front of an audience, with no opportunity to correct" (issue #22, Change 3).
So :func:`marker_sections` and :func:`to_alignment_result` both *raise*,
never warn, if the produced sections would not start at the first marker,
be strictly ordered, tile with no gap or overlap, and end exactly at
``track_duration`` -- the same contiguity discipline
``instrumental_coverage`` enforces for an aligned timeline.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol, runtime_checkable

from music_video_maker.contracts import AlignedSegment, AlignmentResult

logger = logging.getLogger(__name__)


class MarkerError(ValueError):
    """A marker file or timeline is malformed.

    Every raise site logs first (with the 1-based line number and the
    offending text, where one exists) so a bad export is diagnosable from
    the log alone, before the traceback is even read.
    """


@dataclass(frozen=True)
class Marker:
    """One authored point on the timeline: a time and a section label."""

    time: float
    """Seconds from the start of the track."""

    label: str
    """Section identity ("Verse 1", "Solo") -- not a lyric. See the module
    docstring's ``label_as_text`` warning before treating this as one."""


@dataclass(frozen=True)
class MarkerTrack:
    """An ordered, validated set of markers plus where they came from."""

    markers: tuple[Marker, ...]
    source: str
    """A path, or ``"<text>"`` for markers parsed from an in-memory string --
    carried through into every log line and error for provenance."""

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(m.label for m in self.markers)


@runtime_checkable
class MarkerSource(Protocol):
    """Anything that can produce a :class:`MarkerTrack`.

    ``MarkerCsvSource`` is the one implementation today. This is the named
    place an Ableton (or Logic, or Cubase) locator reader lands once issue
    #22's step 1 confirms what the band actually exports -- deliberately a
    ``Protocol`` and not a stub class that raises ``NotImplementedError``,
    because there is nothing to stub until that is known.
    """

    def read(self) -> MarkerTrack: ...


@dataclass(frozen=True)
class MarkerCsvSource:
    """Reads a marker track from a CSV file on disk (see module docstring)."""

    path: Path

    def read(self) -> MarkerTrack:
        return read_marker_csv(self.path)


_HHMMSS_COMPONENT_RE = re.compile(r"^\d+$")
_SECONDS_COMPONENT_RE = re.compile(r"^\d+(?:\.\d+)?$")


def parse_marker_time(raw: str) -> float:
    """Parse one timestamp in either format a DAW marker export uses.

    Accepts plain seconds (``"12.5"``, ``"12"``) or ``[HH:]MM:SS[.mmm]``
    (``"1:02.5"``, ``"00:01:02.500"``). This is the single place that knows
    both forms exist, kept out of the CSV row parser and tested on its own,
    because a caller that assumes only one form silently misreads the other.

    Raises :class:`MarkerError` for anything unparseable, non-finite
    (``"nan"``, ``"inf"`` both parse as floats but describe no point in a
    track), or negative -- a marker time is seconds from the start of the
    track and cannot be before it.
    """
    text = raw.strip()
    if not text:
        message = f"empty time value: {raw!r}"
        logger.error("markers: %s", message)
        raise MarkerError(message)

    if ":" in text:
        parts = text.split(":")
        *head, seconds_part = parts
        malformed = len(parts) not in (2, 3) or not all(
            _HHMMSS_COMPONENT_RE.match(p) for p in head
        ) or not _SECONDS_COMPONENT_RE.match(seconds_part)
        if malformed:
            message = f"unparseable time (expected [HH:]MM:SS[.mmm]): {raw!r}"
            logger.error("markers: %s", message)
            raise MarkerError(message)
        value = 0.0
        for component in (*(int(p) for p in head), float(seconds_part)):
            value = value * 60 + component
    else:
        try:
            value = float(text)
        except ValueError as exc:
            message = f"unparseable time: {raw!r}"
            logger.error("markers: %s", message)
            raise MarkerError(message) from exc

    if not math.isfinite(value):
        message = f"non-finite time: {raw!r}"
        logger.error("markers: %s", message)
        raise MarkerError(message)
    if value < 0:
        message = f"negative time: {raw!r}"
        logger.error("markers: %s", message)
        raise MarkerError(message)
    return value


def _fail(lineno: int, raw_line: str, reason: str) -> NoReturn:
    message = f"line {lineno}: {reason}: {raw_line!r}"
    logger.error("markers: %s", message)
    raise MarkerError(message)


def read_marker_csv(path: Path | str) -> MarkerTrack:
    """Read and parse a marker CSV file. See the module docstring for format."""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    return parse_marker_csv(text, source=str(resolved))


def parse_marker_csv(text: str, *, source: str = "<text>") -> MarkerTrack:
    """Parse marker CSV text already in memory. See the module docstring for
    format, and :class:`MarkerError` for what is rejected and why."""
    markers: list[Marker] = []
    last_time: float | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        try:
            row = next(csv.reader([raw_line]))
        except csv.Error as exc:
            _fail(lineno, raw_line, f"malformed CSV row ({exc})")

        if not row or not row[0].strip():
            continue

        if row[0].strip().lower() == "time":
            continue  # optional header line

        if len(row) < 2:
            _fail(lineno, raw_line, "expected 2 fields '<time>,<label>'")

        label = ",".join(row[1:]).strip()
        if not label:
            _fail(lineno, raw_line, "blank label")

        try:
            time = parse_marker_time(row[0])
        except MarkerError as exc:
            _fail(lineno, raw_line, str(exc))

        if last_time is not None and time <= last_time:
            _fail(
                lineno,
                raw_line,
                f"marker times must be strictly increasing (got {time}, "
                f"previous {last_time}) -- two markers at the same instant "
                "would open a zero-length section",
            )

        markers.append(Marker(time=time, label=label))
        last_time = time

    if not markers:
        message = f"no markers found in {source!r}"
        logger.error("markers: %s", message)
        raise MarkerError(message)

    return MarkerTrack(markers=tuple(markers), source=source)


@dataclass(frozen=True)
class MarkerSection:
    """The honest intermediate between raw markers and a chunk-shaped
    timeline: each marker opens a section that runs until the next marker
    (or, for the last one, until ``track_duration``). A caller that only
    wants "where are the sections" should not have to go through an
    :class:`~music_video_maker.contracts.AlignmentResult` to get one --
    :func:`to_alignment_result` is a thin adapter over this."""

    index: int
    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def marker_sections(track: MarkerTrack, track_duration: float) -> tuple[MarkerSection, ...]:
    """Turn a validated marker track into contiguous sections.

    Raises :class:`MarkerError` (never just warns -- see the module
    docstring's contiguity discussion) if ``track_duration`` is not finite,
    or is at or before the last marker's time: the last marker opens a
    section, and a section needs a positive duration to exist at all.

    A first marker at ``time > 0`` is not an error -- it is logged at INFO,
    because the uncovered head is Stage 2's ``instrumental_coverage``
    problem, exactly as it is for a forced-alignment timeline that starts
    mid-song.
    """
    if not track.markers:
        message = f"marker track {track.source!r} has no markers"
        logger.error("markers: %s", message)
        raise MarkerError(message)

    if not math.isfinite(track_duration):
        message = f"track_duration must be finite, got {track_duration!r}"
        logger.error("markers: %s", message)
        raise MarkerError(message)

    last_marker_time = track.markers[-1].time
    if track_duration <= last_marker_time:
        message = (
            f"track_duration ({track_duration}s) must be strictly after the last "
            f"marker ({last_marker_time}s) in {track.source!r} -- the last marker "
            "opens a section, which must run past it to have any duration"
        )
        logger.error("markers: %s", message)
        raise MarkerError(message)

    first_marker_time = track.markers[0].time
    if first_marker_time > 0.0:
        logger.info(
            "markers: %s: first marker at %.3fs, not 0.0 -- the uncovered head is "
            "Stage 2's instrumental_coverage problem, same as an aligned timeline",
            track.source,
            first_marker_time,
        )

    sections: list[MarkerSection] = []
    for i, marker in enumerate(track.markers):
        end = track.markers[i + 1].time if i + 1 < len(track.markers) else track_duration
        sections.append(MarkerSection(index=i, label=marker.label, start=marker.time, end=end))
    return tuple(sections)


def to_alignment_result(
    track: MarkerTrack, track_duration: float, *, label_as_text: bool
) -> AlignmentResult:
    """Adapt a marker track into the ``AlignmentResult`` shape Stage 2 already
    consumes, so slicing/prompt-expansion is reusable as-is for concert mode.

    ``label_as_text`` has no default: the caller must state whether section
    labels should become ``AlignedSegment.text``. Setting it ``True`` logs a
    WARNING every time, because ``text`` is consumed downstream as the
    literal lyric line composed into the render prompt (see the module
    docstring) -- concert mode's prompt composer must never do that, and any
    other caller reading a label as a sung line is very likely a bug.

    Every segment's ``words`` is empty: word timings exist so issue #79's
    leading-vocal-offset preference can anchor a chunk start at a prompted
    word's onset, and a section label has no words that are sung -- there is
    nothing for that mechanism to anchor to here. ``characters`` is empty
    for the same reason concert mode has no cast: see the module docstring's
    "no cast and no lip-sync" note.
    """
    sections = marker_sections(track, track_duration)

    if label_as_text:
        logger.warning(
            "markers: %s: label_as_text=True -- section labels (%s) will be "
            "composed as lyric text by any downstream stage that reads "
            "AlignedSegment.text. A marker label is a section identity, not a "
            "sung line; concert mode's prompt composer must never do this "
            "(issue #22).",
            track.source,
            ", ".join(track.labels),
        )

    segments = tuple(
        AlignedSegment(
            index=section.index,
            text=section.label if label_as_text else "",
            start=section.start,
            end=section.end,
        )
        for section in sections
    )
    return AlignmentResult(segments=segments, track_duration=track_duration)
