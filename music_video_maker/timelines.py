"""Second timelines: a spoken prologue (or epilogue) around the song (#66).

Everything in this pipeline is indexed on one master track. Alignment
produces song time, slicing cuts song time, ``instrumental_coverage``
guarantees the chunk timeline covers *that track* end to end, and Stage 5
concatenates the chunks and muxes that one file over them. A prologue is
video with no position on that timeline and audio that is not in that file.

The reframing ``docs/design-prologue-timelines.md`` settles on, and what this
module implements: **a prologue is not a special chunk, it is a second
timeline, rendered by the same five stages and concatenated ahead of the
first.** Spoken dialogue is audio and H3 conditions on audio; a script plus a
recording is exactly the (text, audio) pair Stage 1 already consumes. So a
segment reuses alignment, slicing, staging, execution and the cast verbatim,
and what is new is only *joining* two timelines.

This module owns the joining, and nothing else. It is pure: no ffmpeg, no
pydub, no file I/O. :func:`place_timelines` takes durations somebody else
measured and returns where each timeline starts; :mod:`assembly` measures
those durations with ffprobe and calls it, and ``cli`` calls it again from
``--prepare`` with Stage 2's own predicted numbers so both timelines are
reported before any GPU time is spent.

Three decisions are load-bearing enough to restate here.

**Chunk ids are a separate space per timeline, with a per-timeline chunks
directory.** The alternative -- one id space with the prologue first -- is
simpler for concat and fatal for everything else: it renumbers every authored
shot plan the moment a prologue is added or its length changes, and
``chunk_id`` is the anchor a plan is authored against. The renumbering
failure is silent and destroys committed authoring work, so the id spaces are
separate, the mp4s live in separate directories, and
:attr:`~music_video_maker.contracts.ChunkFingerprint.timeline` is what keeps
``--resume`` from handing one timeline's clip to another.

**The offset is a property of the timeline, not of a chunk.** A chunk's
``start``/``end`` stay seconds from the start of its own audio. Every anchor
in the project is measured that way -- a shot plan's ``chunk_id``/``start``,
``ShotPlanDriftError``'s comparison, the stem slicing -- so folding a
whole-video offset into them would move every authored anchor in the song the
moment the prologue's length changed. It would also be a *content* change to
chunks whose pixels are identical, re-rendering the entire song for an edit
in front of it.

**The seam is reconciled by padding audio, and it is measured, never
computed.** The chunk timeline overshoots its own master track by up to one
trained-floor chunk (1.837 s on "Deathless", where H3's 124-frame minimum
padded the final tile past the end of the song). For one timeline the mux's
``-shortest`` silently throws that away -- 47 rendered frames, unlogged. At a
seam there is nothing for ``-shortest`` to bite on: an over-long prologue
pushes the *entire song* out of sync by whatever the padding was, while both
halves still look perfect in isolation. So each timeline's audio is padded
with silence up to its own rendered video duration, the pad is logged, and
the duration it is padded to is the one ffprobe reports for the assembled
video -- not the sum of the chunk spans that video was supposed to contain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, and never a runtime cycle
    from music_video_maker.config import RunConfig

logger = logging.getLogger(__name__)

SEGMENT_POSITIONS: tuple[str, ...] = ("before", "after")
"""Where a segment sits relative to the song. Deliberately not a number: two
"before" segments run in the order they are written, which is the order a
reader of the config already assumes."""

DEFAULT_SEGMENT_POSITION = "before"
"""What ``[[segment]]`` means with no ``position``. A prologue is the case
that exists; an epilogue says so."""

SONG_TIMELINE_NAME = "song"
"""The song's own name in a log line and in a per-timeline path. Never
written into a fingerprint -- see :attr:`Timeline.fingerprint_name`."""

SEGMENT_KEYS = frozenset({"name", "position", "audio", "script", "shot_plan"})
"""Every key a ``[[segment]]`` table may contain. Closed for the same reason
:data:`~music_video_maker.config.HARDWARE_KEYS` and
:data:`~music_video_maker.config.ALIGNMENT_OVERRIDE_KEYS` are, and with the
same sharp edge: TOML binds a bare key to whichever table precedes it, so a
top-level setting written below a ``[[segment]]`` table silently becomes a
segment key. This repo has paid for that twice -- ``examples/first-run.toml``
rendered at 1344x768 for months, and issue #62's A/B loaded both arms
identically -- and a segment table is *more* exposed than either, because a
prologue is the last thing anybody adds to a config."""

RESERVED_SEGMENT_NAMES = frozenset({SONG_TIMELINE_NAME, "frames", "stem", "final"})
"""Names a segment may not take, because its chunks directory is a
subdirectory of ``chunks_dir`` and these are already in use there:
``frames/`` is continuity's extracted seed frames, ``stem/`` is issue #25's
sliced vocal stems. A collision would put one timeline's wavs where another
stage is about to write."""

_SEGMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
"""A segment name becomes a directory name and a fingerprint value, so it is
restricted to what is safe in both, rather than quoted at each use."""


class TimelineError(ValueError):
    """A segment definition or a seam is malformed. Logged before it is
    raised, with the segment named, because a config error found at load time
    is the cheapest one there is."""


@dataclass(frozen=True)
class Segment:
    """One authored ``[[segment]]`` table: a second (text, audio) pair.

    ``script`` is a lyrics-format file (``docs/lyrics-format.md``) -- the
    same parser, the same ``[Name: Role]`` attribution, because a prologue's
    speaker is a cast question and #59's ``present`` / #82's ``subject``
    already answer "who is on screen" and "whose shot is this".

    ``shot_plan`` is this segment's own plan, authored against this segment's
    own chunk ids. The song's ``shot_plan`` is never consulted for a segment:
    its ``chunk_id`` anchors name chunks in a different id space, and
    resolving them against a segment would attach the song's shot 3 to the
    prologue's shot 3 with nothing raising."""

    name: str
    audio: Path
    script: Path
    position: str = DEFAULT_SEGMENT_POSITION
    shot_plan: Path | None = None
    order: int = 0
    """Declaration order within the config, which is the order two segments
    on the same side play in."""


@dataclass(frozen=True)
class Timeline:
    """One timeline to render: the song, or one segment.

    Built by :func:`plan_timelines` so that every consumer -- Stage 1-2,
    Stage 4's run state, Stage 5's concat, ``--prepare``'s report -- is
    looking at the same answer to "where do this timeline's files live and
    what is it made of", rather than each re-deriving it."""

    name: str
    position: str
    """``"before"``, ``"song"`` or ``"after"``."""
    audio: Path
    """This timeline's own master: the song's ``master_audio``, or the
    segment's recording."""
    script: Path
    """This timeline's own text: ``lyrics_file``, or the segment's script."""
    chunks_dir: Path
    run_state_file: Path
    shot_plan: Path | None = None
    is_song: bool = False
    sort_key: tuple[int, int] = (0, 0)

    @property
    def fingerprint_name(self) -> str | None:
        """What goes in :attr:`~music_video_maker.contracts.ChunkFingerprint.timeline`.

        ``None`` for the song, deliberately: that is what every state file
        written before segments existed records, and what a song chunk
        records today, so the two compare equal and a run that adds a
        prologue does not re-render the song."""
        return None if self.is_song else self.name


def build_segment(raw: object, index: int, *, resolve: object = None) -> Segment:
    """Validate one ``[[segment]]`` table. ``resolve`` is a callable turning a
    config-relative path string into a :class:`~pathlib.Path`; ``None`` leaves
    the strings as given (used by tests and by callers that already resolved).

    Every failure names the segment by index *and* by name when it has one:
    a config with three segments and a typo in the second must not send
    anyone reading all three.
    """
    if not isinstance(raw, dict):
        message = (
            f"[[segment]] #{index} must be a table with {sorted(SEGMENT_KEYS)}, got "
            f"{type(raw).__name__}"
        )
        logger.error("segments: %s", message)
        raise TimelineError(message)

    unknown = sorted(set(raw) - SEGMENT_KEYS)
    if unknown:
        message = (
            f"[[segment]] #{index} contains unknown key(s): {', '.join(unknown)}. Valid "
            f"keys are {', '.join(sorted(SEGMENT_KEYS))}. In TOML a bare key belongs to "
            "whichever table precedes it, so a top-level setting written below a "
            "[[segment]] table binds to it -- move it ABOVE the first table."
        )
        logger.error("segments: %s", message)
        raise TimelineError(message)

    name = raw.get("name")
    if not isinstance(name, str) or not _SEGMENT_NAME_RE.match(name):
        message = (
            f"[[segment]] #{index}: name must be lowercase letters, digits, '_' or '-' "
            f"(it becomes a directory name and a fingerprint value), got {name!r}"
        )
        logger.error("segments: %s", message)
        raise TimelineError(message)
    if name in RESERVED_SEGMENT_NAMES:
        message = (
            f"[[segment]] #{index}: {name!r} is reserved -- a segment's chunks live in "
            f"chunks_dir/<name>/, and {sorted(RESERVED_SEGMENT_NAMES)} already name "
            "something there"
        )
        logger.error("segments: %s", message)
        raise TimelineError(message)

    position = raw.get("position", DEFAULT_SEGMENT_POSITION)
    if position not in SEGMENT_POSITIONS:
        message = (
            f"[[segment]] #{index} ({name}): position must be one of "
            f"{list(SEGMENT_POSITIONS)}, got {position!r}"
        )
        logger.error("segments: %s", message)
        raise TimelineError(message)

    def _path(key: str, required: bool) -> Path | None:
        value = raw.get(key)
        if value is None or value == "":
            if not required:
                return None
            message = f"[[segment]] #{index} ({name}): {key} is required"
            logger.error("segments: %s", message)
            raise TimelineError(message)
        if not isinstance(value, str):
            message = (
                f"[[segment]] #{index} ({name}): {key} must be a path string, got {value!r}"
            )
            logger.error("segments: %s", message)
            raise TimelineError(message)
        return resolve(value) if callable(resolve) else Path(value)

    # Both required paths: `_path(..., required=True)` raises rather than
    # returning None, so these are Paths by construction.
    audio = _path("audio", True)
    script = _path("script", True)
    shot_plan = _path("shot_plan", False)

    return Segment(
        name=name,
        audio=audio,  # type: ignore[arg-type]
        script=script,  # type: ignore[arg-type]
        position=position,
        shot_plan=shot_plan,
        order=index,
    )


def build_segments(raw: object, *, resolve: object = None) -> tuple[Segment, ...]:
    """Validate the whole ``[[segment]]`` array, rejecting duplicate names.

    Two segments with one name would share a chunks directory and a
    fingerprint value -- the exact confusion the discriminator exists to
    prevent, arriving from the config side instead."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        message = f"segment must be [[segment]] tables, got {type(raw).__name__}"
        logger.error("segments: %s", message)
        raise TimelineError(message)

    segments = tuple(build_segment(entry, i, resolve=resolve) for i, entry in enumerate(raw))
    seen: set[str] = set()
    for segment in segments:
        if segment.name in seen:
            message = (
                f"duplicate segment name {segment.name!r} -- each segment needs its own "
                "name: it is the chunks directory and the fingerprint discriminator that "
                "stop one timeline's clips being reused for another"
            )
            logger.error("segments: %s", message)
            raise TimelineError(message)
        seen.add(segment.name)
    return segments


def plan_timelines(config: RunConfig) -> tuple[Timeline, ...]:
    """Every timeline this run renders, in playback order.

    "Before" segments in declaration order, then the song, then "after"
    segments in declaration order. A config with no ``[[segment]]`` table
    returns exactly one :class:`Timeline` -- the song, with ``chunks_dir``
    and ``run_state_file`` untouched -- which is what keeps every existing
    run byte-identical.
    """
    chunks_dir = Path(config.chunks_dir)
    song = Timeline(
        name=SONG_TIMELINE_NAME,
        position=SONG_TIMELINE_NAME,
        audio=Path(config.master_audio),
        script=Path(config.lyrics_file),
        chunks_dir=chunks_dir,
        run_state_file=Path(config.run_state_file)
        if config.run_state_file is not None
        else chunks_dir / "run_state.json",
        shot_plan=Path(config.shot_plan) if config.shot_plan else None,
        is_song=True,
        sort_key=(0, 0),
    )
    timelines = [song]
    for segment in config.segments:
        side = -1 if segment.position == "before" else 1
        timelines.append(
            Timeline(
                name=segment.name,
                position=segment.position,
                audio=Path(segment.audio),
                script=Path(segment.script),
                chunks_dir=chunks_dir / segment.name,
                run_state_file=chunks_dir / segment.name / "run_state.json",
                shot_plan=Path(segment.shot_plan) if segment.shot_plan else None,
                is_song=False,
                sort_key=(side, segment.order),
            )
        )
    return tuple(sorted(timelines, key=lambda t: t.sort_key))


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


DEFAULT_SEAM_TOLERANCE_SECONDS = 0.05
"""One frame at 24 fps rounded up, the same figure
:data:`~music_video_maker.assembly.DEFAULT_DURATION_TOLERANCE_SECONDS` uses
and for the same reason: it is a starting point, not a measured rig
tolerance."""


@dataclass(frozen=True)
class TimelinePlacement:
    """Where one timeline lands in the finished video, and what the seam in
    front of it cost."""

    name: str
    fingerprint_name: str | None
    offset_seconds: float
    """Seconds from the start of the finished video to this timeline's first
    frame. The number a chunk's ``start`` would have to be shifted by to be
    read as video time -- which is exactly why it is here and not there."""
    video_seconds: float
    audio_seconds: float
    pad_seconds: float
    """Silence appended to this timeline's audio so it matches its own
    video. Always >= 0: padding is the reconciliation this project takes,
    because the alternative -- trimming the last chunk's video -- is a
    re-encode or a keyframe cut, and ``-c:v copy`` is the invariant this
    project protects hardest."""

    @property
    def end_seconds(self) -> float:
        return self.offset_seconds + self.video_seconds


class SeamOverrunError(TimelineError):
    """A timeline's audio is *longer* than the video rendered for it, so the
    seam cannot be closed by padding.

    This is not the case the overshoot creates -- a chunk timeline overshoots
    its master, never the reverse -- so it means something upstream is wrong:
    a segment's recording was replaced with a longer take after its chunks
    were rendered, or a chunk is missing from the concat. Raised rather than
    trimmed: silently dropping the end of somebody's dialogue is the failure
    mode a seam check exists to make impossible."""

    def __init__(self, name: str, video_seconds: float, audio_seconds: float, tolerance: float):
        self.name = name
        self.video_seconds = video_seconds
        self.audio_seconds = audio_seconds
        self.tolerance = tolerance
        super().__init__(
            f"timeline {name!r}: its audio is {audio_seconds:.3f}s but only "
            f"{video_seconds:.3f}s of video was rendered for it (over by "
            f"{audio_seconds - video_seconds:+.3f}s, tolerance {tolerance:.3f}s). The seam "
            "is closed by padding audio up to video, so audio that OUTRUNS its video "
            "cannot be reconciled without trimming the picture -- which would either "
            "re-encode (breaking -c:v copy) or cut somebody off mid-sentence. Re-render "
            "this timeline against the audio it actually has."
        )


def place_timelines(
    measurements: list[tuple[str, str | None, float, float]],
    *,
    tolerance_seconds: float = DEFAULT_SEAM_TOLERANCE_SECONDS,
) -> tuple[TimelinePlacement, ...]:
    """Lay measured timelines end to end and compute each one's audio pad.

    ``measurements`` is ``(name, fingerprint_name, video_seconds,
    audio_seconds)`` in playback order -- both durations *measured*, not
    summed from what the chunks were supposed to be. That distinction is the
    whole point: on "Deathless" the chunk timeline ends at 513.917 s against
    a 512.080 s master, and the 1.837 s difference is real rendered video
    that a seam has to account for.

    Raises :class:`SeamOverrunError` for a timeline whose audio outruns its
    video by more than ``tolerance_seconds``. Within tolerance the pad is
    clamped at zero rather than negative -- a few sample-frames of float
    noise is not a seam defect, and an ``apad`` to less than the input's own
    length would do nothing anyway.
    """
    placements: list[TimelinePlacement] = []
    offset = 0.0
    for name, fingerprint_name, video_seconds, audio_seconds in measurements:
        if audio_seconds - video_seconds > tolerance_seconds:
            logger.error(
                "Seam: timeline %r has %.3fs of audio against %.3fs of rendered video "
                "(+%.3fs) -- refusing to assemble.",
                name,
                audio_seconds,
                video_seconds,
                audio_seconds - video_seconds,
            )
            raise SeamOverrunError(name, video_seconds, audio_seconds, tolerance_seconds)
        pad = max(0.0, video_seconds - audio_seconds)
        placements.append(
            TimelinePlacement(
                name=name,
                fingerprint_name=fingerprint_name,
                offset_seconds=offset,
                video_seconds=video_seconds,
                audio_seconds=audio_seconds,
                pad_seconds=pad,
            )
        )
        offset += video_seconds
    return tuple(placements)


def log_placements(placements: tuple[TimelinePlacement, ...], *, measured: bool) -> None:
    """Report the seam: where each timeline starts and what it was padded by.

    ``measured`` says which of the two callers this is -- ``--prepare``'s
    prediction from Stage 2's chunk timeline (no GPU), or assembly's real
    ffprobe reading of the rendered files. Both are worth having and they
    are not the same claim, so the log line says which one it is rather than
    leaving a reader to guess from context.
    """
    source = "measured" if measured else "predicted (Stage 2, no GPU)"
    for placement in placements:
        logger.info(
            "Timeline %r [%s]: starts at %.3fs, %.3fs of video, %.3fs of audio, padded "
            "by %.3fs of silence",
            placement.name,
            source,
            placement.offset_seconds,
            placement.video_seconds,
            placement.audio_seconds,
            placement.pad_seconds,
        )
    if len(placements) > 1:
        total_pad = sum(p.pad_seconds for p in placements)
        logger.info(
            "Seam total [%s]: %d timelines, %.3fs of finished video, %.3fs of silence "
            "inserted at %d seam(s) so every timeline's audio matches its own picture "
            "(issue #66; padding rather than trimming keeps -c:v copy intact)",
            source,
            len(placements),
            placements[-1].end_seconds,
            total_pad,
            len(placements) - 1,
        )


def predicted_measurements(
    timelines: tuple[Timeline, ...],
    chunk_timeline_seconds: dict[str, float],
    track_seconds: dict[str, float],
) -> list[tuple[str, str | None, float, float]]:
    """Build :func:`place_timelines` input from Stage 2's own numbers, for
    the no-GPU report ``--prepare`` gives.

    A *prediction*: the chunk timeline's end is what the render is asked to
    produce, and assembly re-asks ffprobe what it actually produced. They
    should agree, and issue #22 exists because for eight months nothing
    checked."""
    return [
        (
            timeline.name,
            timeline.fingerprint_name,
            chunk_timeline_seconds[timeline.name],
            track_seconds[timeline.name],
        )
        for timeline in timelines
    ]
