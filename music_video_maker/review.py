"""The read-only pre-render review artifact (issue #36's "next slice,
specified", ``docs/design-web-ui.md``).

A run today means hand-editing ``run.toml``/``shot_plan.toml``, launching a
CLI, and reading ``render.log`` with ``grep``. The two expensive lessons this
project has paid for -- an alignment timeline that believed a song stopped
singing at 3:11 of 8:32 (#35), and a shot plan whose lyric named a printer
the video never showed (#37) -- were both visible in the *data* a
``--prepare``-style run already produces, before any GPU time was spent. This
module turns that data into one self-contained HTML page (and the same data
as JSON): no server, no forms, no start button, no socket imports. Building
it is a file write; opening it is the review.

Two halves, kept deliberately separate and independently testable:

* :func:`build_review` -- the pure builder. Inputs (a :class:`RunConfig` plus
  the injectable alignment-model seam every other Stage-1 caller already
  takes) to a :class:`ReviewData`, a plain, JSON-able tree. No file I/O, no
  HTML.
* :func:`render_review_html` / :func:`render_review_json` -- pure renderers.
  ``ReviewData`` to a string. No knowledge of where Stage 1-2's data came
  from.

Getting the data: reuse, never reimplement
-------------------------------------------
:func:`~music_video_maker.cli.prepare_timeline` (factored out of
``cli.prepare_shot_plan`` for this issue) runs Stages 1-2 exactly as a real
render would and hands back the chunk timeline, the raw alignment, and the
:class:`~music_video_maker.alignment_quality.AlignmentQualityReport` Stage 1
already computes and, before this issue, only logged.

The shot-plan lints are collected the same way
``authoring/plan.check_plan`` collects them on the authoring side: attach a
``logging.Handler`` around a call to the render's own lint functions and read
back what they said. That module's own docstring calls this out as the rule
("checked by the render's own loaders, never a copy of them") rather than
reimplementing what a lint decides -- and issue #36's design doc gives the
same instruction for this page. This file cannot import
``authoring/plan.py``'s ``_Collector`` to get it: ``tests/test_authoring_boundary.py``
forbids anything outside ``music_video_maker/authoring/`` from importing that
package (issue #54 design section 2), and this module lives in the render
half on purpose -- a review of a run's own inputs is exactly the render's
business, not the authoring layer's. So :class:`_LintCollector` below is a
small, independent reimplementation of *just* the handler (nine lines), not
of any lint -- the lints themselves are called via
:func:`~music_video_maker.cli.run_shot_plan_lints`, the exact function
``cli.run_pipeline`` runs, factored out for this issue so there is only ever
one copy of "which lints, in what order" to drift.
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path

from music_video_maker import cli
from music_video_maker.alignment_quality import (
    AlignmentQualityReport,
    Finding,
    Severity,
    format_summary,
)
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk
from music_video_maker.shot_plan import (
    ShotPlanEntry,
    ShotPlanError,
    load_shot_plan,
    resolve_camera,
    resolve_conditions,
    resolve_framing,
    resolve_location,
    resolve_present,
    resolve_shot,
    resolve_subject,
)

logger = logging.getLogger(__name__)

_LINT_LOGGER_NAMES = ("music_video_maker.shot_plan", "music_video_maker.cli")
"""Loggers to collect from while the plan is loaded, linted and resolved --
``shot_plan`` for every ``lint_*``/``resolve_*`` warning, ``cli`` for the one
lint (:func:`~music_video_maker.shot_plan.lint_mouth_direction_on_instrumental_chunks`)
whose findings ``cli.run_shot_plan_lints`` formats itself rather than logging
through ``shot_plan``'s own logger -- see that function's body. Narrower than
"everything at WARNING" on purpose, the same reasoning
``authoring/plan.py``'s ``_LINT_LOGGERS`` gives: config and slicing also warn
during this call, about things that are not this review's business."""

_CHUNK_ID_IN_MESSAGE = re.compile(r"chunk_id=(\d+)")


class _LintCollector(logging.Handler):
    """Collects the WARNING-and-above records the render's own shot-plan
    loaders emit while attached (issue #36).

    Deliberately minimal -- nine lines -- and *not* imported from
    ``authoring/plan.py``'s own ``_Collector``, which does the identical
    thing: see this module's docstring for why importing across that
    boundary is off the table rather than merely discouraged."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _chunk_id_from(message: str) -> int | None:
    match = _CHUNK_ID_IN_MESSAGE.search(message)
    return int(match.group(1)) if match else None


@contextmanager
def _collecting_lint_records():
    collector = _LintCollector()
    loggers = [logging.getLogger(name) for name in _LINT_LOGGER_NAMES]
    saved = [(one_logger.level, one_logger.propagate) for one_logger in loggers]
    for one_logger in loggers:
        one_logger.addHandler(collector)
        # `--log-level` sets the root logger's level, and a logger's effective
        # level gates a record before any handler sees it -- so at ERROR the
        # page would silently lose every lint warning. The console's verbosity
        # is the operator's choice; the page's contents are not. Where warnings
        # would have been dropped, let them through to the collector only.
        if not one_logger.isEnabledFor(logging.WARNING):
            one_logger.setLevel(logging.WARNING)
            one_logger.propagate = False
    try:
        yield collector
    finally:
        for one_logger, (level, propagate) in zip(loggers, saved, strict=True):
            one_logger.removeHandler(collector)
            one_logger.setLevel(level)
            one_logger.propagate = propagate


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlignmentFindingView:
    """One :class:`~music_video_maker.alignment_quality.Finding`, in the
    review's wire shape -- a plain dict rather than the
    :class:`~music_video_maker.alignment_quality.Severity` ``IntEnum``, so
    :func:`ReviewData.to_dict` needs no custom JSON encoder."""

    severity: str
    """``"INFO" | "WARNING" | "CRITICAL"`` -- :attr:`Severity.name`."""
    code: str
    message: str
    start: float
    end: float
    segment_index: int | None

    @classmethod
    def of(cls, finding: Finding) -> AlignmentFindingView:
        return cls(
            severity=finding.severity.name,
            code=finding.code,
            message=finding.message,
            start=finding.start,
            end=finding.end,
            segment_index=finding.segment_index,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "segment_index": self.segment_index,
        }


@dataclass(frozen=True)
class LintWarning:
    """One record a shot-plan loader logged while checking the plan (issue
    #36) -- a ``lint_*`` finding, an unfilled shot line, a missing entry, a
    drift refusal. ``chunk_id`` is ``None`` when the message does not name
    one (a file-level problem, not a per-chunk one)."""

    severity: str
    """``"warning" | "error"`` -- the record's own log level, not a lint's
    internal severity concept (shot-plan lints have none; they are WARNING or
    ERROR only, and ERROR here means the same "structural, revision-worthy"
    thing it means to ``authoring/plan.check_plan``)."""
    message: str
    chunk_id: int | None

    @classmethod
    def of(cls, record: logging.LogRecord) -> LintWarning:
        message = record.getMessage()
        return cls(
            severity="error" if record.levelno >= logging.ERROR else "warning",
            message=message,
            chunk_id=_chunk_id_from(message),
        )

    def to_dict(self) -> dict[str, object]:
        return {"severity": self.severity, "message": self.message, "chunk_id": self.chunk_id}


_EMPTY_PLAN_FIELDS: Mapping[str, object] = {
    "shot": None,
    "camera": None,
    "location": None,
    "present": (),
    "subject": None,
    "conditions": None,
    "framing": None,
}


@dataclass(frozen=True)
class ChunkReview:
    """One chunk's full review row: everything the design doc's "per chunk"
    list asks for."""

    chunk_id: int
    start: float
    end: float
    duration: float
    frame_count: int | None
    voiced: bool
    lyric: str
    shot: str | None
    camera: str | None
    location: str | None
    present: tuple[str, ...]
    subject: str | None
    conditions: str | None
    framing: str | None = None
    """This chunk's authored framing intent (issue #97) -- how much of the
    frame its focus member should fill. Defaulted, unlike the fields above
    it, so every construction site that predates the field keeps working;
    `_resolve_chunk_plan_fields` always supplies it."""
    findings: tuple[AlignmentFindingView, ...] = ()
    """Every alignment-quality finding whose span touches this chunk's --
    see :func:`_findings_for_chunk`."""
    lint_warnings: tuple[LintWarning, ...] = ()
    """Shot-plan warnings that named this ``chunk_id``."""

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "voiced": self.voiced,
            "lyric": self.lyric,
            "shot": self.shot,
            "camera": self.camera,
            "location": self.location,
            "present": list(self.present),
            "subject": self.subject,
            "conditions": self.conditions,
            "framing": self.framing,
            "findings": [finding.to_dict() for finding in self.findings],
            "lint_warnings": [warning.to_dict() for warning in self.lint_warnings],
        }


@dataclass(frozen=True)
class ReviewData:
    """Everything the review page shows: the chunk timeline, the run-level
    alignment-quality summary, and every shot-plan warning (issue #36).

    Deliberately carries no timestamp, hostname or run id -- see
    ``docs/design-web-ui.md``'s "next slice, specified" and the golden-file
    test in ``tests/test_review.py``: two runs of :func:`build_review` over
    the same inputs must produce byte-identical JSON and HTML, which a
    generated-at field would break for no reader's benefit."""

    chunks: tuple[ChunkReview, ...]
    alignment_summary: str
    """:func:`~music_video_maker.alignment_quality.format_summary`'s own
    one-line text, unmodified -- one place decides how a report reads."""
    alignment_finding_counts: Mapping[str, int]
    lint_warnings: tuple[LintWarning, ...]
    """Every warning the shot-plan loaders raised, attributed or not --
    :attr:`ChunkReview.lint_warnings` is the subset that named a chunk,
    duplicated there (not moved) so a reader of one chunk's row sees
    everything about it without cross-referencing this list."""
    plan_errors: tuple[str, ...] = ()
    """Structural shot-plan failures: the plan would not load, or a lint that
    raises (``lint_subject_on_voiced_chunk``) refused it, or an individual
    chunk's entry drifted (``ShotPlanDriftError``). Never silently dropped --
    a real render would have refused for the same reason."""
    would_refuse_render: str | None = None
    """Set when ``config.strict_alignment`` is ``True`` *and* the alignment
    report has a finding at or above :attr:`~music_video_maker.alignment_quality.Severity.CRITICAL`
    -- naming the count. A real render with this exact config would raise
    ``AlignmentQualityError`` and refuse before touching the GPU; the review
    itself never raises regardless of ``strict_alignment`` (see
    :func:`build_review`'s docstring), so this is the one place that fact
    would otherwise go missing. ``None`` when ``strict_alignment`` is unset,
    or set but nothing would trip it."""

    def to_dict(self) -> dict[str, object]:
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "alignment_summary": self.alignment_summary,
            "alignment_finding_counts": dict(self.alignment_finding_counts),
            "lint_warnings": [warning.to_dict() for warning in self.lint_warnings],
            "plan_errors": list(self.plan_errors),
            "would_refuse_render": self.would_refuse_render,
        }


# --------------------------------------------------------------------------- #
# The pure builder
# --------------------------------------------------------------------------- #


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Whether span ``[a_start, a_end]`` touches span ``[b_start, b_end]`` --
    inclusive of the endpoints, so a zero-length finding landing exactly on a
    chunk boundary still counts as touching it."""
    return a_start <= b_end and a_end >= b_start


def _findings_for_chunk(
    report: AlignmentQualityReport, chunk: AudioChunk
) -> tuple[AlignmentFindingView, ...]:
    return tuple(
        AlignmentFindingView.of(finding)
        for finding in report.findings
        if _overlaps(finding.start, finding.end, chunk.start, chunk.end)
    )


def _resolve_chunk_plan_fields(
    plan: Mapping[int, ShotPlanEntry] | None, chunk: AudioChunk, plan_errors: list[str]
) -> Mapping[str, object]:
    """The plan-derived fields for one chunk, via the render's own
    ``resolve_*`` functions -- never a second read of the raw entry.

    A per-chunk drift (:class:`~music_video_maker.shot_plan.ShotPlanDriftError`,
    a subclass of :class:`~music_video_maker.shot_plan.ShotPlanError`) is
    caught here rather than left to abort the whole review: one stale entry
    should not hide every other chunk's data from a reviewer trying to see
    what changed. ``resolve_shot`` runs the shared drift check first (see
    ``shot_plan._resolve_entry``), so a drifting chunk raises there and the
    other five calls never run -- this chunk's fields fall back to "no plan
    entry", exactly as an unauthored chunk's would."""
    try:
        return {
            "shot": resolve_shot(plan, chunk),
            "camera": resolve_camera(plan, chunk),
            "location": resolve_location(plan, chunk),
            "present": resolve_present(plan, chunk),
            "subject": resolve_subject(plan, chunk),
            "conditions": resolve_conditions(plan, chunk),
            "framing": resolve_framing(plan, chunk),
        }
    except ShotPlanError as exc:
        plan_errors.append(str(exc))
        return dict(_EMPTY_PLAN_FIELDS)


def build_review(
    config: RunConfig,
    *,
    align_model: object | None = None,
    from_plan: str | Path | None = None,
) -> ReviewData:
    """Build a :class:`ReviewData` from a ``--prepare``-style input: a
    :class:`RunConfig` plus (in tests) an injected alignment model.

    Runs Stages 1-2 through :func:`~music_video_maker.cli.prepare_timeline`
    -- the exact function ``--prepare`` uses -- so the chunk timeline and
    quality report describe precisely the run ``--prepare``/a real render
    would produce, not a second approximation of it. No GPU, no ComfyUI, no
    custody, no file I/O: this function returns data.

    When ``config.shot_plan`` is set, the plan is loaded and linted through
    :func:`~music_video_maker.cli.run_shot_plan_lints` -- the same function,
    same order, ``run_pipeline`` calls -- with a log handler attached (see
    the module docstring). A plan that fails to load, or that a raising lint
    refuses, is recorded in :attr:`ReviewData.plan_errors` rather than
    raised: the review's whole purpose is to surface exactly this kind of
    problem without needing a stack trace to do it.

    ``from_plan`` defaults to ``config.shot_plan`` when not given, not to
    "no editorial lengths": ``run_pipeline`` always slices with
    ``shot_length_requests(plan)`` from ``config.shot_plan`` directly, so a
    review that ignored it for any plan setting ``length_seconds`` would
    describe chunks the render never emits and report the resulting mismatch
    as plan drift instead of the merged timeline the render actually
    produces. An explicit ``from_plan`` (checking a *candidate* plan before
    it is wired into the config, the same case ``--prepare --from-plan``
    exists for) still overrides it, exactly as passing it explicitly always
    has.

    The alignment step itself always runs non-strict, whatever
    ``config.strict_alignment`` says: ``prepare_timeline`` -> ``align()``
    raises ``AlignmentQualityError`` once ``strict_alignment`` is set and a
    finding reaches CRITICAL, and that is exactly the run a reviewer most
    needs to see rather than a traceback for. ``--prepare`` itself is
    untouched -- this override is local to this function's own call into
    ``prepare_timeline``. When the *original* config was strict and the
    report does have a CRITICAL-or-above finding, that fact is not silently
    dropped: it survives as :attr:`ReviewData.would_refuse_render`, naming
    the count, because "this run would in fact refuse" is exactly the kind
    of thing the review exists to surface before the GPU is committed.
    """
    effective_from_plan = from_plan if from_plan is not None else config.shot_plan
    align_config = dc_replace(config, strict_alignment=False) if config.strict_alignment else config

    plan_errors: list[str] = []
    try:
        timeline = cli.prepare_timeline(
            align_config, align_model=align_model, from_plan=effective_from_plan
        )
    except ShotPlanError as exc:
        # A plan that cannot be read is a hard failure for --prepare (real
        # work depends on it landing correctly) but not for a review: the
        # whole point here is to surface a problem, never to crash instead
        # of showing one. Fall back to the natural (no editorial lengths)
        # timeline so the rest of the review still has something to show --
        # ``config.shot_plan``'s own load a few lines down will hit the same
        # error again and add it to ``plan_errors`` too if it names the same
        # file; a caller reads two lines naming one root cause rather than a
        # traceback naming none.
        plan_errors.append(str(exc))
        timeline = cli.prepare_timeline(align_config, align_model=align_model, from_plan=None)

    would_refuse_render: str | None = None
    if config.strict_alignment:
        blocking = timeline.quality_report.at_least(Severity.CRITICAL)
        if blocking:
            would_refuse_render = (
                f"strict_alignment is set on this config -- a real render would refuse "
                f"before touching the GPU: {len(blocking)} finding(s) at "
                f"{Severity.CRITICAL.name} severity or above (see the alignment findings "
                "below)"
            )

    plan: Mapping[int, ShotPlanEntry] = {}
    lint_records: tuple[logging.LogRecord, ...] = ()

    if config.shot_plan is not None:
        with _collecting_lint_records() as collector:
            try:
                plan = load_shot_plan(
                    config.shot_plan, setting=config.setting, cast_names=tuple(config.cast)
                )
            except ShotPlanError as exc:
                plan_errors.append(str(exc))
                plan = {}

            if plan:
                try:
                    cli.run_shot_plan_lints(plan, timeline.chunks, config)
                except ShotPlanError as exc:
                    plan_errors.append(str(exc))

            chunk_fields = {
                chunk.chunk_id: _resolve_chunk_plan_fields(plan, chunk, plan_errors)
                for chunk in timeline.chunks
            }
        lint_records = tuple(collector.records)
    else:
        chunk_fields = {chunk.chunk_id: _EMPTY_PLAN_FIELDS for chunk in timeline.chunks}

    lint_warnings = tuple(LintWarning.of(record) for record in lint_records)
    lint_warnings_by_chunk: dict[int, list[LintWarning]] = {}
    for warning in lint_warnings:
        if warning.chunk_id is not None:
            lint_warnings_by_chunk.setdefault(warning.chunk_id, []).append(warning)

    chunks = tuple(
        ChunkReview(
            chunk_id=chunk.chunk_id,
            start=chunk.start,
            end=chunk.end,
            duration=chunk.duration,
            frame_count=chunk.frame_count,
            voiced=not chunk.is_instrumental,
            lyric=chunk.text,
            findings=_findings_for_chunk(timeline.quality_report, chunk),
            lint_warnings=tuple(lint_warnings_by_chunk.get(chunk.chunk_id, ())),
            **chunk_fields[chunk.chunk_id],
        )
        for chunk in timeline.chunks
    )

    return ReviewData(
        chunks=chunks,
        alignment_summary=format_summary(timeline.quality_report),
        alignment_finding_counts={
            severity.name: count
            for severity, count in timeline.quality_report.counts_by_severity().items()
        },
        lint_warnings=lint_warnings,
        plan_errors=tuple(plan_errors),
        would_refuse_render=would_refuse_render,
    )


# --------------------------------------------------------------------------- #
# Pure renderers
# --------------------------------------------------------------------------- #


def render_review_json(data: ReviewData) -> str:
    """``ReviewData`` as pretty-printed, key-sorted JSON -- deterministic
    byte-for-byte for the same input, which is what makes the golden-file
    test in ``tests/test_review.py`` possible."""
    return json.dumps(data.to_dict(), indent=2, sort_keys=True) + "\n"


_SEVERITY_CLASS = {
    "CRITICAL": "sev-critical",
    "error": "sev-critical",
    "WARNING": "sev-warning",
    "warning": "sev-warning",
    "INFO": "sev-info",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _badge(severity: str) -> str:
    css_class = _SEVERITY_CLASS.get(severity, "sev-info")
    return f'<span class="badge {css_class}">{_esc(severity.upper())}</span>'


def _findings_cell(findings: Sequence[AlignmentFindingView]) -> str:
    if not findings:
        return "&mdash;"
    items = "".join(
        f"<li>{_badge(finding.severity)} <code>{_esc(finding.code)}</code> "
        f"{_esc(finding.message)}</li>"
        for finding in findings
    )
    return f'<ul class="findings">{items}</ul>'


def _lint_cell(warnings: Sequence[LintWarning]) -> str:
    if not warnings:
        return "&mdash;"
    items = "".join(
        f"<li>{_badge(warning.severity)} {_esc(warning.message)}</li>" for warning in warnings
    )
    return f'<ul class="findings">{items}</ul>'


def _list_cell(values: Sequence[str]) -> str:
    return _esc(", ".join(values)) if values else "&mdash;"


def _text_cell(value: str | None) -> str:
    return _esc(value) if value else "&mdash;"


_CHUNK_ROW_TEMPLATE = """
    <tr class="{row_class}">
      <td>{chunk_id}</td>
      <td>{span}</td>
      <td>{duration:.3f}s</td>
      <td>{frame_count}</td>
      <td>{voicing}</td>
      <td>{lyric}</td>
      <td>{shot}</td>
      <td>{camera}</td>
      <td>{location}</td>
      <td>{present}</td>
      <td>{subject}</td>
      <td>{conditions}</td>
      <td>{framing}</td>
      <td>{findings}</td>
      <td>{lint_warnings}</td>
    </tr>"""


def _chunk_row(chunk: ChunkReview) -> str:
    has_critical = any(f.severity == "CRITICAL" for f in chunk.findings) or any(
        w.severity == "error" for w in chunk.lint_warnings
    )
    has_warning = not has_critical and (chunk.findings or chunk.lint_warnings)
    row_class = "row-critical" if has_critical else "row-warning" if has_warning else ""
    return _CHUNK_ROW_TEMPLATE.format(
        row_class=row_class,
        chunk_id=chunk.chunk_id,
        span=f"{chunk.start:.3f}s&ndash;{chunk.end:.3f}s",
        duration=chunk.duration,
        frame_count=chunk.frame_count if chunk.frame_count is not None else "&mdash;",
        voicing="voiced" if chunk.voiced else "instrumental",
        lyric=_text_cell(chunk.lyric),
        shot=_text_cell(chunk.shot),
        camera=_text_cell(chunk.camera),
        location=_text_cell(chunk.location),
        present=_list_cell(chunk.present),
        subject=_text_cell(chunk.subject),
        conditions=_text_cell(chunk.conditions),
        framing=_text_cell(chunk.framing),
        findings=_findings_cell(chunk.findings),
        lint_warnings=_lint_cell(chunk.lint_warnings),
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Shot Plan Review</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left;
            vertical-align: top; }}
  th {{ background: #f0f0f0; position: sticky; top: 0; }}
  tr.row-critical {{ background: #fde8e8; }}
  tr.row-warning {{ background: #fff6db; }}
  ul.findings {{ margin: 0; padding-left: 1.1rem; }}
  ul.findings li {{ margin: 0.15rem 0; }}
  .badge {{ display: inline-block; padding: 0.05rem 0.4rem; border-radius: 0.2rem;
            font-size: 0.72rem; font-weight: 600; color: #fff; }}
  .sev-critical {{ background: #b42318; }}
  .sev-warning {{ background: #b7791f; }}
  .sev-info {{ background: #4a5568; }}
  .summary {{ background: #f7f7f7; border: 1px solid #ddd; padding: 0.75rem 1rem; }}
  .plan-errors {{ background: #fde8e8; border: 1px solid #b42318; padding: 0.75rem 1rem; }}
  .refusal {{ background: #b42318; color: #fff; border-radius: 0.25rem;
              padding: 0.75rem 1rem; font-weight: 600; }}
  code {{ font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Shot Plan Review</h1>
{refusal_block}
<div class="summary">
  <p><strong>Alignment quality:</strong> {alignment_summary}</p>
  <p><strong>Findings by severity:</strong> {finding_counts}</p>
</div>
{plan_errors_block}
<h2>Run-level shot-plan warnings</h2>
{unattributed_lint_warnings}

<h2>Chunks ({chunk_count})</h2>
<table>
  <thead>
    <tr>
      <th>chunk_id</th><th>span</th><th>duration</th><th>frames</th><th>voicing</th>
      <th>lyric</th><th>shot</th><th>camera</th><th>location</th><th>present</th>
      <th>subject</th><th>conditions</th><th>framing</th>
      <th>alignment findings</th><th>lint warnings</th>
    </tr>
  </thead>
  <tbody>{chunk_rows}
  </tbody>
</table>
</body>
</html>
"""


def render_review_html(data: ReviewData) -> str:
    """``ReviewData`` as one self-contained HTML page: inline CSS, no
    external resources, no JavaScript required. Every piece of user text
    (lyrics, shot lines, lint messages) is HTML-escaped -- see
    ``tests/test_review.py`` for the ``<script>``-in-a-shot-line case this
    guards against."""
    finding_counts = ", ".join(
        f"{severity}: {count}"
        for severity, count in sorted(data.alignment_finding_counts.items())
        if count
    ) or "none"

    refusal_block = ""
    if data.would_refuse_render:
        refusal_block = f'<div class="refusal">{_esc(data.would_refuse_render)}</div>'

    unattributed = [w for w in data.lint_warnings if w.chunk_id is None]
    unattributed_block = (
        '<ul class="findings">'
        + "".join(
            f"<li>{_badge(w.severity)} {_esc(w.message)}</li>" for w in unattributed
        )
        + "</ul>"
        if unattributed
        else "<p>None.</p>"
    )

    plan_errors_block = ""
    if data.plan_errors:
        items = "".join(f"<li>{_esc(err)}</li>" for err in data.plan_errors)
        plan_errors_block = (
            f'<div class="plan-errors"><p><strong>Shot plan errors '
            f"(a real render would refuse):</strong></p><ul>{items}</ul></div>"
        )

    chunk_rows = "".join(_chunk_row(chunk) for chunk in data.chunks)

    return _PAGE_TEMPLATE.format(
        refusal_block=refusal_block,
        alignment_summary=_esc(data.alignment_summary),
        finding_counts=_esc(finding_counts),
        plan_errors_block=plan_errors_block,
        unattributed_lint_warnings=unattributed_block,
        chunk_count=len(data.chunks),
        chunk_rows=chunk_rows,
    )


__all__ = [
    "AlignmentFindingView",
    "ChunkReview",
    "LintWarning",
    "ReviewData",
    "build_review",
    "render_review_html",
    "render_review_json",
]
