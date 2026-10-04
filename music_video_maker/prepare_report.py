"""What ``--prepare`` found, written down (issue #36's "gate on the
pre-render checks", read-only half).

``--prepare`` is this project's 50-second, no-GPU check: it runs Stages 1-2
and reports, at INFO and WARNING, everything a render is about to be
committed to -- the #35 alignment-quality summary, the phrases this run's
``max_chunk_seconds`` cannot hold whole (#70), the chunk boundaries that cut
a sung phrase (#70), the voiced chunks whose first prompted word starts a
second or more after the chunk does (#79), and how far the chunk timeline
drifts from the master track (#22). Every one of those is a measured lesson
this project already paid for.

**And all of it went to a terminal that scrolled.** Nothing was persisted,
so nothing could be read later, by a second tool, or by anyone who was not
watching the run of ``--prepare`` that produced it. That is what this module
fixes: ``--prepare`` now also writes a small JSON *report* beside the run's
own state file (``RunConfig.prepare_report_file``), and the read-only
monitor (:mod:`music_video_maker.webui`) renders it.

A report, not a behaviour change
-----------------------------------
Nothing here decides anything. ``--prepare`` slices exactly the same
timeline, writes exactly the same skeleton and refuses exactly the same
things it did before; this file only writes down what it already said. The
report is derived data and is overwritten on every ``--prepare`` -- unlike
the shot-plan skeleton, which is real work and is refused without
``--force``.

Why the monitor reads a file instead of running the check
------------------------------------------------------------
``webui.py`` deliberately never imports ``alignment`` or ``slicing``:
Stage 2 *writes* ``chunk_NNN.wav`` into the live run's own ``chunks_dir``,
so a long-lived poller that recomputed the timeline would race a render for
that directory (and spend ~6 s of CPU per poll doing it). Alignment is a
step the operator invokes, not something a page does when it is opened. So
the split is: ``--prepare`` measures and writes; the monitor reads and
renders; if nothing has been prepared, the page says exactly that. This
module is the file format in the middle, and it imports nothing from
``alignment``, ``slicing`` or ``cli`` so that reading a report can never
drag Stage 1-2 in behind it.

Provenance is a field, not a filename (#93)
----------------------------------------------
A measurement artefact that cannot name what it was computed from can
assert a subject it never opened -- this project lost a week to exactly
that, a face scan of one render wearing another's filename. So a report
records every input it was built from by resolved path, size and mtime
(:class:`InputStamp`), and the page shows them. A report whose lyrics file
has changed since it was written is still readable; it is just visibly
about something else.

What this module does *not* do
---------------------------------
It does not re-check anything. The notices it carries are the records the
render's own Stage 1-2 code emitted, captured while they were emitted
(:func:`collect_stage_notices`) -- never a second implementation of a lint,
which is the rule ``authoring/plan.py`` and ``review.py`` both already
follow and which exists because a second copy drifts from the first inside
a month.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from music_video_maker.alignment_quality import AlignmentQualityReport, Severity, format_summary
from music_video_maker.contracts import AudioChunk

logger = logging.getLogger(__name__)


PREPARE_REPORT_SCHEMA_VERSION = 1
"""Stamped into every file this module writes.

Read the same way ``resilience.RUN_STATE_SCHEMA_VERSION`` is: a reader
refuses a version it does not know rather than guessing at a payload's
meaning. Unlike a run state, though, nothing is lost by refusing one -- the
remedy is to re-run ``--prepare``, which costs ~50 s and no GPU. So a future
change that alters what a field *means* should bump this; one that only adds
an optional field need not, and :meth:`PrepareReport.from_dict` defaults
every field that could be absent."""


class PrepareReportError(RuntimeError):
    """A prepare report could not be read.

    Every failure mode collapses here, naming the path -- missing file,
    torn JSON, unknown schema version -- because the one consumer
    (:mod:`music_video_maker.webui`) treats them identically: there is no
    usable report, so say so on the page. The same stance
    ``progress.read_run_state`` takes for ``run_state.json``."""


STAGE_LOGGER_NAMES = ("music_video_maker.slicing", "music_video_maker.cli")
"""The loggers whose WARNING-and-above records a report captures.

Narrow on purpose, the same reasoning ``review.py``'s ``_LINT_LOGGER_NAMES``
gives: ``slicing`` is where #70's untenable segments and mid-phrase cuts and
#79's leading vocal offsets are reported, and ``cli`` is where #22's
timeline-versus-track drift is. ``alignment_quality``'s findings are *not*
collected this way -- they arrive structured, through the report object
Stage 1 already builds, and re-reading them out of log text would be
strictly worse."""

_ISSUE_IN_MESSAGE = re.compile(r"issue #(\d+)")


# --------------------------------------------------------------------------- #
# Capturing what Stage 1-2 said, while it says it
# --------------------------------------------------------------------------- #


class _NoticeCollector(logging.Handler):
    """Collects WARNING-and-above records while attached."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def collect_stage_notices(
    logger_names: Sequence[str] = STAGE_LOGGER_NAMES,
) -> Iterator[list[logging.LogRecord]]:
    """Capture the WARNING-and-above records ``logger_names`` emit inside
    this block, without changing what the console shows.

    A logger's *effective level* gates a record before any handler sees it,
    so an operator running ``--prepare --log-level ERROR`` would otherwise
    get a report with every Stage-2 warning silently missing from it. Where
    that would happen, the level is lowered for the duration and propagation
    switched off, so the collector sees the record and the console still
    shows only what the operator asked for. Both are restored in ``finally``.

    (``review.py`` does the same nine lines for the shot-plan lints. It is
    not imported from here, and this module is not imported from there,
    because ``review.py`` imports ``cli`` and ``cli`` imports this module --
    consolidating the two means moving ``review.py``'s copy here, which is a
    safe direction and is left to whoever next has reason to touch it.)"""
    collector = _NoticeCollector()
    loggers = [logging.getLogger(name) for name in logger_names]
    saved = [(one.level, one.propagate) for one in loggers]
    for one in loggers:
        one.addHandler(collector)
        if not one.isEnabledFor(logging.WARNING):
            one.setLevel(logging.WARNING)
            one.propagate = False
    try:
        yield collector.records
    finally:
        for one, (level, propagate) in zip(loggers, saved, strict=True):
            one.removeHandler(collector)
            one.setLevel(level)
            one.propagate = propagate


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InputStamp:
    """One file this report was computed from, stamped (#93).

    ``exists`` is ``False`` rather than an error for a path that has gone
    away between the run and the read: a report about inputs that no longer
    exist is exactly the thing worth showing, not worth crashing on."""

    label: str
    path: str
    exists: bool
    size_bytes: int | None
    mtime_ns: int | None

    @classmethod
    def of(cls, label: str, path: Path | str | None) -> InputStamp | None:
        """``None`` when ``path`` is ``None`` -- an unset optional input
        (no ``shot_plan``, say) is not a missing file, and the difference
        matters to a reader."""
        if path is None:
            return None
        resolved = Path(path)
        try:
            stat = resolved.stat()
        except OSError:
            return cls(label=label, path=str(resolved), exists=False, size_bytes=None,
                       mtime_ns=None)
        return cls(
            label=label,
            path=str(resolved),
            exists=True,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path": self.path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> InputStamp:
        return cls(
            label=str(raw.get("label", "")),
            path=str(raw.get("path", "")),
            exists=bool(raw.get("exists", False)),
            size_bytes=_opt_int(raw.get("size_bytes")),
            mtime_ns=_opt_int(raw.get("mtime_ns")),
        )


@dataclass(frozen=True)
class StageNotice:
    """One WARNING-or-above record Stage 1-2 emitted during this prepare.

    ``issue`` is the issue number the message names itself, parsed out of
    the text (``"(issue #70)"``) purely so a page can group thirty mid-phrase
    cut warnings under one heading. It is a grouping key taken from prose,
    not a classification this module performs: ``None`` simply means the
    message did not name one, and nothing downstream may treat that as
    meaning anything else."""

    logger: str
    level: str
    message: str
    issue: str | None = None

    @classmethod
    def of(cls, record: logging.LogRecord) -> StageNotice:
        message = record.getMessage()
        found = _ISSUE_IN_MESSAGE.findall(message)
        return cls(
            logger=record.name,
            level=record.levelname,
            message=message,
            # The last mention: these messages cite background issues mid-
            # sentence and close with the one they are actually about.
            issue=found[-1] if found else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "logger": self.logger,
            "level": self.level,
            "message": self.message,
            "issue": self.issue,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> StageNotice:
        issue = raw.get("issue")
        return cls(
            logger=str(raw.get("logger", "")),
            level=str(raw.get("level", "WARNING")),
            message=str(raw.get("message", "")),
            issue=None if issue is None else str(issue),
        )


@dataclass(frozen=True)
class AlignmentFindingRow:
    """One :class:`~music_video_maker.alignment_quality.Finding`, flattened
    (``Severity`` by name, so no custom JSON encoder is needed)."""

    severity: str
    code: str
    message: str
    start: float
    end: float
    segment_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "segment_index": self.segment_index,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> AlignmentFindingRow:
        return cls(
            severity=str(raw.get("severity", "INFO")),
            code=str(raw.get("code", "")),
            message=str(raw.get("message", "")),
            start=float(raw.get("start", 0.0)),  # type: ignore[arg-type]
            end=float(raw.get("end", 0.0)),  # type: ignore[arg-type]
            segment_index=_opt_int(raw.get("segment_index")),
        )


@dataclass(frozen=True)
class PrepareReport:
    """Everything one ``--prepare`` found, as a file.

    Deliberately flat and small: this is read by a browser page and by a
    human running ``jq``, not by the render, which re-derives all of it from
    the audio every time it runs. **Nothing reads this back into a render**,
    the same boundary an authored ``shot_plan.toml`` sits on the other side
    of."""

    generated_at: str
    """ISO date (or datetime) the caller supplies -- never read from the
    clock here, so a test can produce a byte-identical file twice. Same
    reasoning as ``cli.prepare_shot_plan``'s own ``generated_at``."""
    config_path: str
    inputs: tuple[InputStamp, ...] = ()
    alignment_model_size: str | None = None
    strict_alignment: bool = False

    chunk_count: int = 0
    instrumental_chunk_count: int = 0
    voiced_chunk_count: int = 0
    total_frames: int | None = None
    max_chunk_frames: int | None = None
    """The largest ``frame_count`` in this timeline, and the one number the
    issue #98 render-envelope gate needs (``max_chunk_frames_chunk_id`` names
    the chunk it came from).

    Recorded so that something which is *not* running Stage 1-2 can evaluate
    that gate from published data instead of recomputing the timeline -- the
    web monitor's control half (:mod:`music_video_maker.control`) does
    exactly this before it will start a render. It is sufficient, not a
    convenience: ``envelope.EnvelopePoint.covers`` is per-axis, so at one
    fixed resolution the chunk with the most frames decides the whole run.
    Cover the largest and every shorter chunk is covered; miss it and the run
    is refused. (Which *other* chunks also miss is reporting, and only the
    per-chunk check inside the run can enumerate those.)

    ``None`` on a report written before this field existed, which is read as
    "this report cannot answer the question", never as "nothing is too
    long" -- the same distinction ``largest_rendered_frame_count`` draws."""
    max_chunk_frames_chunk_id: int | None = None
    timeline_start: float | None = None
    timeline_end: float | None = None
    track_duration_seconds: float | None = None
    timeline_drift_seconds: float | None = None
    """Stage 2's timeline minus the master track's own duration (#22).
    Positive is an overshoot (the mux's ``-shortest`` throws those frames
    away); negative is an undershoot, which cuts the song's own ending out
    of the finished file."""
    timeline_drift_frames: float | None = None
    duration_tolerance_seconds: float | None = None

    alignment_summary: str = ""
    """:func:`~music_video_maker.alignment_quality.format_summary`'s own
    line, unmodified -- one place decides how a report reads."""
    alignment_finding_counts: Mapping[str, int] = field(default_factory=dict)
    alignment_findings: tuple[AlignmentFindingRow, ...] = ()
    """Findings at WARNING and above. INFO findings are counted in
    :attr:`alignment_finding_counts` and not listed: the summary line already
    carries what they contribute, and a page that lists every one of them
    buries the twenty that matter."""
    notices: tuple[StageNotice, ...] = ()
    plan_errors: tuple[str, ...] = ()
    """Shot-plan failures found by resolving the plan against *this*
    timeline: a plan that will not load, or a ``ShotPlanDriftError`` naming a
    chunk whose start has moved since the plan was authored. The check a
    human review of the plan cannot perform, because the plan looks complete
    either way."""
    plan_checked: str | None = None
    """The plan file :attr:`plan_errors` is about, or ``None`` when no plan
    was checked."""
    plan_lengths_applied: bool = False
    """Whether the timeline was re-anchored against that plan's own
    ``length_seconds`` (``--prepare --from-plan``). ``False`` with a plan set
    means the two describe different timelines *by construction*, so drift
    findings below are expected rather than alarming -- see CLAUDE.md's "a
    plan that sets ``length_seconds`` has two timelines" bullet."""

    schema_version: int = PREPARE_REPORT_SCHEMA_VERSION

    @property
    def critical_finding_count(self) -> int:
        return int(dict(self.alignment_finding_counts).get(Severity.CRITICAL.name, 0))

    @property
    def warning_finding_count(self) -> int:
        return int(dict(self.alignment_finding_counts).get(Severity.WARNING.name, 0))

    def notices_by_issue(self) -> tuple[tuple[str, tuple[StageNotice, ...]], ...]:
        """Notices grouped by the issue number their own text names, ordered
        by descending group size then issue, with the ungrouped ones
        (``"other"``) last. Presentation only -- see :class:`StageNotice`."""
        groups: dict[str, list[StageNotice]] = {}
        for notice in self.notices:
            groups.setdefault(notice.issue or "other", []).append(notice)
        ordered = sorted(
            groups.items(),
            key=lambda item: (item[0] == "other", -len(item[1]), item[0]),
        )
        return tuple((key, tuple(value)) for key, value in ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "config_path": self.config_path,
            "inputs": [stamp.to_dict() for stamp in self.inputs],
            "alignment_model_size": self.alignment_model_size,
            "strict_alignment": self.strict_alignment,
            "chunk_count": self.chunk_count,
            "instrumental_chunk_count": self.instrumental_chunk_count,
            "voiced_chunk_count": self.voiced_chunk_count,
            "total_frames": self.total_frames,
            "max_chunk_frames": self.max_chunk_frames,
            "max_chunk_frames_chunk_id": self.max_chunk_frames_chunk_id,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "track_duration_seconds": self.track_duration_seconds,
            "timeline_drift_seconds": self.timeline_drift_seconds,
            "timeline_drift_frames": self.timeline_drift_frames,
            "duration_tolerance_seconds": self.duration_tolerance_seconds,
            "alignment_summary": self.alignment_summary,
            "alignment_finding_counts": dict(self.alignment_finding_counts),
            "alignment_findings": [row.to_dict() for row in self.alignment_findings],
            "notices": [notice.to_dict() for notice in self.notices],
            "plan_errors": list(self.plan_errors),
            "plan_checked": self.plan_checked,
            "plan_lengths_applied": self.plan_lengths_applied,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PrepareReport:
        version = raw.get("schema_version")
        if version != PREPARE_REPORT_SCHEMA_VERSION:
            raise PrepareReportError(
                f"prepare report schema_version is {version!r}, this build reads "
                f"{PREPARE_REPORT_SCHEMA_VERSION} -- re-run --prepare (it costs ~50s and no "
                "GPU) rather than reading a payload whose fields may mean something else"
            )
        counts = raw.get("alignment_finding_counts") or {}
        return cls(
            generated_at=str(raw.get("generated_at", "")),
            config_path=str(raw.get("config_path", "")),
            inputs=tuple(
                InputStamp.from_dict(one) for one in _sequence(raw.get("inputs"))
            ),
            alignment_model_size=_opt_str(raw.get("alignment_model_size")),
            strict_alignment=bool(raw.get("strict_alignment", False)),
            chunk_count=int(raw.get("chunk_count", 0)),  # type: ignore[arg-type]
            instrumental_chunk_count=int(raw.get("instrumental_chunk_count", 0)),  # type: ignore[arg-type]
            voiced_chunk_count=int(raw.get("voiced_chunk_count", 0)),  # type: ignore[arg-type]
            total_frames=_opt_int(raw.get("total_frames")),
            max_chunk_frames=_opt_int(raw.get("max_chunk_frames")),
            max_chunk_frames_chunk_id=_opt_int(raw.get("max_chunk_frames_chunk_id")),
            timeline_start=_opt_float(raw.get("timeline_start")),
            timeline_end=_opt_float(raw.get("timeline_end")),
            track_duration_seconds=_opt_float(raw.get("track_duration_seconds")),
            timeline_drift_seconds=_opt_float(raw.get("timeline_drift_seconds")),
            timeline_drift_frames=_opt_float(raw.get("timeline_drift_frames")),
            duration_tolerance_seconds=_opt_float(raw.get("duration_tolerance_seconds")),
            alignment_summary=str(raw.get("alignment_summary", "")),
            alignment_finding_counts={
                str(key): int(value) for key, value in dict(counts).items()  # type: ignore[arg-type]
            },
            alignment_findings=tuple(
                AlignmentFindingRow.from_dict(one)
                for one in _sequence(raw.get("alignment_findings"))
            ),
            notices=tuple(StageNotice.from_dict(one) for one in _sequence(raw.get("notices"))),
            plan_errors=tuple(str(one) for one in _sequence(raw.get("plan_errors"))),
            plan_checked=_opt_str(raw.get("plan_checked")),
            plan_lengths_applied=bool(raw.get("plan_lengths_applied", False)),
            schema_version=PREPARE_REPORT_SCHEMA_VERSION,
        )


def _sequence(value: object) -> Sequence[Mapping[str, object]]:
    return value if isinstance(value, list) else []  # type: ignore[return-value]


def _opt_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _opt_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def stale_inputs(report: PrepareReport) -> tuple[str, ...]:
    """The labels of inputs whose file on disk no longer matches the stamp
    this report recorded for it -- re-stat'ed now, at read time.

    This is what makes :class:`InputStamp` worth carrying rather than
    decorative. A report is a measurement of one set of files; the moment the
    lyrics file is edited, the timeline it describes is not the timeline a
    render would produce, and the artefact itself cannot tell (#93 -- a
    filename is not provenance). Comparing size *and* mtime, not content: a
    hash of a master track on every page load is not free, and an edit that
    preserved both would have to be deliberate.

    A file that has gone missing counts as changed. A file that was already
    missing when the report was written does not -- that was recorded then
    and is already visible as ``exists: false``."""
    changed: list[str] = []
    for stamp in report.inputs:
        try:
            stat = Path(stamp.path).stat()
        except OSError:
            if stamp.exists:
                changed.append(stamp.label)
            continue
        if not stamp.exists or stat.st_size != stamp.size_bytes or (
            stat.st_mtime_ns != stamp.mtime_ns
        ):
            changed.append(stamp.label)
    return tuple(changed)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_prepare_report(
    *,
    generated_at: str,
    config_path: Path | str,
    inputs: Sequence[InputStamp | None],
    alignment_model_size: str | None,
    strict_alignment: bool,
    chunks: Sequence[AudioChunk],
    quality_report: AlignmentQualityReport,
    track_duration_seconds: float | None,
    timeline_drift_seconds: float | None,
    fps: int | float | None,
    duration_tolerance_seconds: float | None,
    notice_records: Sequence[logging.LogRecord] = (),
    plan_errors: Sequence[str] = (),
    plan_checked: Path | str | None = None,
    plan_lengths_applied: bool = False,
) -> PrepareReport:
    """Assemble a :class:`PrepareReport` from what Stage 1-2 already
    produced. Pure: no file I/O, no clock, no re-measurement.

    Every argument is something the caller already has in hand from the one
    ``prepare_timeline`` run -- which is the point. A builder that went and
    fetched any of it itself could describe a different timeline than the one
    ``--prepare`` just wrote a skeleton for, and nothing would say so."""
    counts = {
        severity.name: count for severity, count in quality_report.counts_by_severity().items()
    }
    findings = tuple(
        AlignmentFindingRow(
            severity=finding.severity.name,
            code=finding.code,
            message=finding.message,
            start=finding.start,
            end=finding.end,
            segment_index=finding.segment_index,
        )
        for finding in quality_report.at_least(Severity.WARNING)
    )
    frames = [chunk.frame_count for chunk in chunks if chunk.frame_count is not None]
    longest = max(
        (chunk for chunk in chunks if chunk.frame_count is not None),
        key=lambda chunk: chunk.frame_count,
        default=None,
    )
    instrumental = sum(1 for chunk in chunks if chunk.is_instrumental)
    drift_frames = (
        timeline_drift_seconds * fps
        if timeline_drift_seconds is not None and fps is not None
        else None
    )
    return PrepareReport(
        generated_at=generated_at,
        config_path=str(config_path),
        inputs=tuple(stamp for stamp in inputs if stamp is not None),
        alignment_model_size=alignment_model_size,
        strict_alignment=strict_alignment,
        chunk_count=len(chunks),
        instrumental_chunk_count=instrumental,
        voiced_chunk_count=len(chunks) - instrumental,
        total_frames=sum(frames) if frames else None,
        max_chunk_frames=longest.frame_count if longest is not None else None,
        max_chunk_frames_chunk_id=longest.chunk_id if longest is not None else None,
        timeline_start=min((chunk.start for chunk in chunks), default=None),
        timeline_end=max((chunk.end for chunk in chunks), default=None),
        track_duration_seconds=track_duration_seconds,
        timeline_drift_seconds=timeline_drift_seconds,
        timeline_drift_frames=drift_frames,
        duration_tolerance_seconds=duration_tolerance_seconds,
        alignment_summary=format_summary(quality_report),
        alignment_finding_counts=counts,
        alignment_findings=findings,
        notices=tuple(StageNotice.of(record) for record in notice_records),
        plan_errors=tuple(plan_errors),
        plan_checked=None if plan_checked is None else str(plan_checked),
        plan_lengths_applied=plan_lengths_applied,
    )


def plan_resolution_errors(
    plan_path: Path | str,
    chunks: Sequence[AudioChunk],
    *,
    setting: str | None,
    cast_names: Sequence[str],
) -> tuple[str, ...]:
    """Resolve ``plan_path`` against ``chunks`` through the render's own
    loaders and return whatever refused, as text.

    This is the ``ShotPlanDriftError`` check, run where it is cheap. A plan
    authored against a timeline that has since moved reads perfectly well --
    on "Deathless" the refusal named a 1.4 s discrepancy no human review of
    the file could have seen -- and today the first thing that notices is the
    render, hours in.

    ``load_shot_plan`` / ``resolve_shot`` are the render's own functions,
    imported here rather than reimplemented: a second copy of "what counts as
    drift" would drift from the first. Nothing raises out of this -- a plan
    that cannot be read is a *finding*, which is what a report is for."""
    from music_video_maker.shot_plan import ShotPlanError, load_shot_plan, resolve_shot

    errors: list[str] = []
    try:
        plan = load_shot_plan(plan_path, setting=setting, cast_names=tuple(cast_names))
    except ShotPlanError as exc:
        return (str(exc),)
    for chunk in chunks:
        try:
            resolve_shot(plan, chunk)
        except ShotPlanError as exc:
            # One refusal per chunk, not one for the whole plan: a
            # re-anchored plan typically drifts from some known point
            # onward, and the chunk it starts at is the useful fact.
            errors.append(str(exc))
    return tuple(errors)


# --------------------------------------------------------------------------- #
# File I/O
# --------------------------------------------------------------------------- #


def write_prepare_report(report: PrepareReport, path: Path | str) -> Path:
    """Write ``report`` to ``path`` (parents created), pretty-printed and
    key-sorted so two prepares over unchanged inputs produce a diffable
    file. Overwrites unconditionally -- a report is derived data, unlike the
    shot-plan skeleton beside it."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Wrote the pre-render report to %s (issue #36)", resolved)
    return resolved


def read_prepare_report(path: Path | str) -> PrepareReport:
    """Read a report written by :func:`write_prepare_report`.

    Raises :class:`PrepareReportError` for every failure, naming ``path``:
    the monitor's page has one thing to say about all of them ("nothing has
    been prepared for this run yet"), and three exception types reaching it
    would be three ways to write the same sentence."""
    resolved = Path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepareReportError(f"could not read a prepare report from {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PrepareReportError(
            f"{resolved} does not contain a prepare report (top level is "
            f"{type(raw).__name__}, not an object)"
        )
    try:
        return PrepareReport.from_dict(raw)
    except PrepareReportError:
        raise
    except (TypeError, ValueError) as exc:
        raise PrepareReportError(f"could not read a prepare report from {resolved}: {exc}") from exc


__all__ = [
    "PREPARE_REPORT_SCHEMA_VERSION",
    "STAGE_LOGGER_NAMES",
    "AlignmentFindingRow",
    "InputStamp",
    "PrepareReport",
    "PrepareReportError",
    "StageNotice",
    "build_prepare_report",
    "collect_stage_notices",
    "plan_resolution_errors",
    "read_prepare_report",
    "stale_inputs",
    "write_prepare_report",
]
