"""Read section markers out of an Ableton Live set (issue #22, step 1).

``markers.py`` defines the input contract concert mode actually consumes -- a
labelled ``(time, label)`` CSV -- and names :class:`~music_video_maker.markers.MarkerSource`
as the seam a DAW reader lands on. This is that reader, for the DAW issue #22
*believes* the band uses.

**Unverified against a real set file.** Nobody has yet answered issue #22's
step-1 questions: whether it is Ableton at all, which major version, and
whether the section labels live in Locators, in clip names, or in a separate
cue list (``docs/design-concert-mode.md``). This module is written against
Ableton's *documented* on-disk shape and is exercised only by a **synthetic**
fixture this repo authored (``tests/fixtures/markers/``). Treat it as a
starting point that will need correcting against a real export, not as a
format this project has confirmed -- and prefer the CSV contract if the band
can export one, which is question 4 and is still the cheapest answer.

What it does read, and what it refuses:

* **Locators only.** A Locator is the only place in a Live set that is
  unambiguously "a named point on the arrangement timeline". Clip names are a
  different claim (a clip has a length and can be moved or duplicated), and
  reading them as markers would produce a plausible-looking timeline nobody
  authored. A set with no Locators is refused, naming the alternatives, rather
  than returning an empty track.
* **Constant tempo only.** A Locator's ``Time`` is in **beats**, not seconds,
  so every marker's position depends on the tempo map. With one manual tempo
  the conversion is ``beats * 60 / bpm``. With tempo automation it is an
  integral, and using the manual value anyway would place every marker after
  the first tempo change wrong -- silently, and progressively worse through
  the song, which is the shape of defect this whole issue exists to prevent.
  So automation is **refused**, naming the number of tempo events found.
  That is issue #22's question 3 answered by refusing rather than guessing.

The conversion to seconds is the only arithmetic here, and it is the reason
this module exists rather than a generic XML walk: a marker file is authored
in beats and a video timeline is in seconds, and the tempo that converts one
to the other is a property of the set that nothing else in this pipeline
would ever see.
"""

from __future__ import annotations

import gzip
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.markers import Marker, MarkerError, MarkerTrack

logger = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"
"""A ``.als`` is gzipped XML. Plain XML is accepted too -- some tools export
it, and a *fixture* that can be read in a diff is worth more than one that
cannot (this repo's is plain XML, gzipped by the test that uses it)."""

TEMPO_TRACK_TAGS: tuple[str, ...] = ("MasterTrack", "MainTrack")
"""Live renamed the master track to "Main" in version 12. Both are looked for
because which one a file uses is exactly the version question issue #22's
step 1 has not answered, and guessing one would fail on half the answers."""

SECONDS_PER_MINUTE = 60.0


class AbletonError(MarkerError):
    """An Ableton set could not be read as a marker track.

    A subclass of :class:`~music_video_maker.markers.MarkerError` so a caller
    handling "the marker source was unusable" catches both without knowing
    which producer it was handed."""


@dataclass(frozen=True)
class AbletonLocatorSource:
    """A :class:`~music_video_maker.markers.MarkerSource` backed by a ``.als``.

    ``tempo_bpm`` overrides whatever the file says -- for the case where the
    set has no readable tempo, or where the band's click is authoritative and
    the set is not. It is never applied silently on top of a tempo the file
    *does* state without saying so in the log.
    """

    path: Path
    tempo_bpm: float | None = None

    def read(self) -> MarkerTrack:
        return read_ableton_locators(self.path, tempo_bpm=self.tempo_bpm)


def _load_xml(path: Path) -> ET.Element:
    raw = Path(path).read_bytes()
    if raw[:2] == GZIP_MAGIC:
        try:
            raw = gzip.decompress(raw)
        except OSError as exc:
            message = f"{path}: looks gzipped but could not be decompressed ({exc})"
            logger.error("ableton: %s", message)
            raise AbletonError(message) from exc
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        message = f"{path}: not parseable as XML ({exc})"
        logger.error("ableton: %s", message)
        raise AbletonError(message) from exc


def _value_of(element: ET.Element | None) -> str | None:
    """Live stores scalars as ``<Name Value="Verse 1"/>`` throughout."""
    if element is None:
        return None
    return element.get("Value")


def read_tempo_bpm(root: ET.Element, *, source: str) -> float:
    """The set's single constant tempo, in BPM.

    Raises :class:`AbletonError` if the set has tempo *automation* (more than
    one tempo event): a Locator's time is in beats, so an automated tempo
    makes beats->seconds an integral rather than a division, and every marker
    after the first tempo change would land wrong. Refusing is issue #22's
    question 3 answered honestly; the alternative is a backdrop that drifts
    further from the band the longer the song runs.
    """
    tempo = None
    for tag in TEMPO_TRACK_TAGS:
        for track in root.iter(tag):
            found = next(iter(track.iter("Tempo")), None)
            if found is not None:
                tempo = found
                break
        if tempo is not None:
            break
    if tempo is None:
        message = (
            f"{source}: no <Tempo> found under {' or '.join(TEMPO_TRACK_TAGS)}. A Locator's "
            "time is in BEATS, so without a tempo there is nothing to convert it to seconds "
            "with -- pass tempo_bpm explicitly if the band's click is authoritative"
        )
        logger.error("ableton: %s", message)
        raise AbletonError(message)

    events = [
        event
        for event in tempo.iter()
        if event.tag.endswith("FloatEvent") or event.tag.endswith("EnumEvent")
    ]
    if len(events) > 1:
        message = (
            f"{source}: this set has tempo automation ({len(events)} tempo events). A "
            "Locator's time is in beats, so converting with the manual tempo would place "
            "every marker after the first tempo change wrong -- progressively worse through "
            "the song, with nothing to notice it. Refusing rather than guessing (issue #22, "
            "design question 3): export a marker CSV from the set instead, or supply the "
            "click track's own section times"
        )
        logger.error("ableton: %s", message)
        raise AbletonError(message)

    raw = _value_of(next(iter(tempo.iter("Manual")), None))
    if raw is None and events:
        raw = events[0].get("Value")
    try:
        bpm = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        message = f"{source}: tempo value is not a number: {raw!r}"
        logger.error("ableton: %s", message)
        raise AbletonError(message) from None
    if not bpm > 0:
        message = f"{source}: tempo must be positive, got {bpm}"
        logger.error("ableton: %s", message)
        raise AbletonError(message)
    return bpm


def read_ableton_locators(
    path: Path | str, *, tempo_bpm: float | None = None
) -> MarkerTrack:
    """Read a Live set's Locators as a :class:`~music_video_maker.markers.MarkerTrack`.

    See the module docstring for what is refused and why, and for the standing
    caveat that no real ``.als`` has ever been read by this code.
    """
    path = Path(path)
    root = _load_xml(path)
    source = str(path)

    file_bpm: float | None = None
    try:
        file_bpm = read_tempo_bpm(root, source=source)
    except AbletonError:
        if tempo_bpm is None:
            raise
        logger.warning(
            "ableton: %s: could not read a tempo from the set; using the caller's "
            "tempo_bpm=%.3f instead. Every marker's position depends on this number.",
            source,
            tempo_bpm,
        )

    if tempo_bpm is not None and file_bpm is not None and tempo_bpm != file_bpm:
        logger.warning(
            "ableton: %s: overriding the set's own tempo (%.3f BPM) with tempo_bpm=%.3f. "
            "Every marker moves by the ratio between them.",
            source,
            file_bpm,
            tempo_bpm,
        )
    bpm = tempo_bpm if tempo_bpm is not None else file_bpm
    assert bpm is not None  # one of the two branches above always set it

    seconds_per_beat = SECONDS_PER_MINUTE / bpm

    raw_locators = list(root.iter("Locator"))
    if not raw_locators:
        message = (
            f"{source}: no <Locator> elements found. This reader reads Locators only -- a "
            "clip name is a different claim (a clip has a length and can be moved or "
            "duplicated), and reading clip names as section markers would produce a "
            "plausible-looking timeline nobody authored. If this set keeps its sections "
            "somewhere else, that is issue #22's step-1 question 2 and the answer belongs "
            "in docs/design-concert-mode.md before any more code is written"
        )
        logger.error("ableton: %s", message)
        raise AbletonError(message)

    markers: list[Marker] = []
    for index, locator in enumerate(raw_locators):
        raw_time = _value_of(next(iter(locator.iter("Time")), None))
        label = _value_of(next(iter(locator.iter("Name")), None))
        if raw_time is None or label is None or not label.strip():
            message = (
                f"{source}: Locator #{index} is missing a Time or a Name "
                f"(time={raw_time!r}, name={label!r})"
            )
            logger.error("ableton: %s", message)
            raise AbletonError(message)
        try:
            beats = float(raw_time)
        except ValueError:
            message = f"{source}: Locator #{index} ({label!r}) has a non-numeric Time {raw_time!r}"
            logger.error("ableton: %s", message)
            raise AbletonError(message) from None
        if beats < 0:
            message = (
                f"{source}: Locator #{index} ({label!r}) is at beat {beats}, before the start "
                "of the arrangement"
            )
            logger.error("ableton: %s", message)
            raise AbletonError(message)
        markers.append(Marker(time=beats * seconds_per_beat, label=label.strip()))

    markers.sort(key=lambda m: m.time)
    for previous, current in zip(markers, markers[1:], strict=False):
        if current.time <= previous.time:
            message = (
                f"{source}: two Locators land at the same time ({current.time:.3f}s: "
                f"{previous.label!r} and {current.label!r}) -- that would open a "
                "zero-length section, the same rule the CSV reader enforces"
            )
            logger.error("ableton: %s", message)
            raise AbletonError(message)

    logger.info(
        "ableton: %s: %d locator(s) read at %.3f BPM (%.4f s/beat), %.3f-%.3fs. UNVERIFIED "
        "against a real Ableton export -- see this module's docstring and issue #22 step 1.",
        source,
        len(markers),
        bpm,
        seconds_per_beat,
        markers[0].time,
        markers[-1].time,
    )
    return MarkerTrack(markers=tuple(markers), source=source)
