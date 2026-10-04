"""Rank a finished render's #79 leading-offset chunks by how *visible* the
desync is, using delivered face size (issue #97).

Issue #79 gave this project a real metric -- the leading vocal offset, the
gap between a chunk's own start and the first word it is actually prompted
with -- and ``slicing.LEADING_VOCAL_OFFSET_WARN_SECONDS`` names every chunk
over 1.0 s of it. What #79 could not say is which of those a viewer will
ever notice, and #97 measured that the ranking by offset is the wrong
ranking. On the "Deathless" v13 render, every chunk over 1.0 s of offset,
adjudicated by a viewer:

===========  ======  ==============  =====================  ======
noticed      offset  face fraction   not noticed            offset
===========  ======  ==============  =====================  ======
74           1.067   0.0778          73 (0.0474)            1.282
38           2.588   0.0887          58 (0.0460)            2.648
41           1.607   0.1656          20 (0.0120)            2.650
35           1.858   0.2101
===========  ======  ==============  =====================  ======

The two groups separate perfectly on face size with no overlap, and run
*backwards* on offset: the two largest offsets in the song are the two least
noticed. Chunk 74 was predicted from its face fraction before it was shown
to a viewer, and the viewer named it unprompted.

What this module does and does not claim
----------------------------------------
It reports a **band**, not a threshold. n=7, one song, one render: the data
says the boundary lies somewhere in 0.0474-0.0778 and says nothing whatever
about where. Publishing a midpoint (0.0626) would turn a gap between two
observations into a number that looks measured, which is the mistake #76
recorded one level up ("a lint shipped on a partial corpus is a hypothesis
that has stopped announcing itself"). ``scenecuts`` ships a single number
because *every* threshold in [0.15, 0.30] gave the identical answer on two
independent renders; there is no second render here, so the honest output is
three states with an explicit uncertain band in the middle -- and the
uncertain band is exactly the list worth watching by eye.

Face size alone is not the claim either. Six voiced chunks on the same
render carry faces of 0.235-0.356 with near-zero offsets and have never been
reported by anyone. It is the *interaction* that predicts, which is why this
module ranks only chunks that already have an offset worth reporting.

Provenance is a precondition, not a nicety (issue #93)
-------------------------------------------------------
:func:`read_facescan_report` **refuses** a CSV with no
``music_video_maker.facescan`` provenance header. That is not tidiness: #93
is the record of a face-presence CSV that was pointed at a week-old render
for a week, asserting a subject it had never opened, while every analysis
built on it stayed internally consistent and wrong. A join between a face
measurement and a timeline is precisely where that mistake gets laundered
into a finding, so the consumer checks, and every report this module writes
names the directory the faces were actually measured from.

The same issue supplies the other half. ``facescan``'s ``inconclusive``
column counts frames where nothing cleared the 0.9 gate but a candidate
survived the 0.70 inspection floor -- "the detector saw something
face-shaped and was not confident". A chunk whose face fraction is small
*and* whose frames are inconclusive is not evidence of a small face, so it
is reported as :data:`UNMEASURED` rather than ranked as hidden. A zero from
a detector is not evidence of absence unless something asked the second
question.

Where the offsets come from
---------------------------
From the render's own log, parsed out of the WARNING line
``slicing._log_leading_vocal_offset`` already emits -- never recomputed
here. Reading a decision back out of the log record that announced it is the
same technique ``review.py`` uses to collect shot-plan lints ("checked by
the render's own loaders, never a copy of them"), and it keeps this module
free of the alignment stack entirely: it needs no audio, no whisper model,
no GPU and no OpenCV. ``tests/test_desync_risk.py`` feeds this parser the
output of the real emitter rather than a copied string, so a change to that
message breaks the test rather than silently emptying the ranking.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


class DesyncRiskError(ValueError):
    """Raised when an input cannot be trusted to describe what it claims to.

    In practice that means one thing: a face-presence CSV with no provenance
    header (issue #93). Every other degradation in this module is partial --
    a chunk with no row is reported as :data:`UNMEASURED` -- because one
    missing chunk must not lose the other seventy-nine. A file that cannot
    say which render it read is different in kind: nothing downstream of it
    can be checked at all.
    """


VISIBLE_FACE_FRACTION = 0.0557
"""At or above this **median** face fraction, a chunk with a leading vocal
offset is predicted **visible** (issue #97).

The smallest median among the four chunks a viewer noticed on "Deathless" v13
(chunk 38). Deliberately the observed value rather than a rounded one: 0.06
would be a number nobody measured sitting where a measurement is.

**Re-derived on medians 2026-10-04**, when the max statistic was shown to
invert a framing A/B on one frame in twelve. Rescanning the same v13 render
with `facescan`'s new median column keeps F42's separation intact on all
seven adjudicated chunks, with a tighter band:

  ========= ======== ======== ==========
  chunk     label    max      median
  ========= ======== ======== ==========
  35        noticed  0.2101   0.1982
  41        noticed  0.1656   0.1405
  74        noticed  0.0778   0.0711
  38        noticed  0.0887   0.0557
  73        not      0.0474   0.0441
  58        not      0.0460   0.0109
  20        not      0.0120   0.0090
  ========= ======== ======== ==========

n=7, no overlap on either statistic. The median is used because it answers
how the shot is *framed*, which is what perceptibility was found to depend
on; chunk 58 is the case that moves most (0.0460 -> 0.0109), and it is the
one a viewer did not notice."""

HIDDEN_FACE_FRACTION = 0.0441
"""At or below this **median** face fraction, a chunk with a leading vocal
offset is predicted **hidden** -- the largest median among the three offset
chunks nobody has ever reported (chunk 73), two of which carry the two
largest offsets in the whole song. Re-derived on medians 2026-10-04; see
:data:`VISIBLE_FACE_FRACTION` for the table.

Between this and :data:`VISIBLE_FACE_FRACTION` is :data:`UNCERTAIN`: seven
observations cannot locate a boundary inside their own gap, and this module
does not pretend otherwise."""

VISIBLE = "visible"
UNCERTAIN = "uncertain"
HIDDEN = "hidden"
UNMEASURED = "unmeasured"

_VERDICT_ORDER: dict[str, int] = {VISIBLE: 0, UNCERTAIN: 1, UNMEASURED: 2, HIDDEN: 3}
"""Report order: what to go and look at first. ``UNMEASURED`` outranks
``HIDDEN`` on purpose -- "we could not tell" is a reason to open the file,
"the face is too small to read" is a reason not to."""

_PROVENANCE_PREFIX = "#"
_INPUT_DIR_KEY = "input_dir"

_OFFSET_LINE = re.compile(
    r"Chunk (?P<chunk>\d+) \((?P<start>-?\d+\.\d+)-(?P<end>-?\d+\.\d+)s\) is prompted to "
    r"sing starting (?P<offset>-?\d+\.\d+)s into its own span"
)
"""Matches ``slicing._log_leading_vocal_offset``'s per-chunk WARNING line.

Anchored on the sentence, not on a log format: a line keeps matching through
any prefix a handler adds (timestamp, level, logger name), and stops matching
if the emitter's own wording changes -- which is what the round-trip test
against the real emitter is there to catch.
"""


# --------------------------------------------------------------------------- #
# The facescan report: rows plus the provenance that makes them citable
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FaceRow:
    """One chunk's face measurement, as ``music_video_maker.facescan`` wrote it."""

    chunk_id: int
    face_pct: float
    max_face_fraction: float
    median_face_fraction: float | None
    """``None`` for a CSV written before #97's median column existed. The
    ranking falls back to the max there and says so, rather than refusing a
    scan that is otherwise valid -- but the two statistics disagree, and the
    fallback is the one that can be inverted by a single frame."""
    inconclusive: int
    source_path: str
    """The file that row actually scanned -- the field whose absence is the
    whole of issue #93."""


def _ranking_fraction(row: FaceRow) -> float:
    """The statistic perceptibility is judged on: the median where the scan
    recorded one, the max for scans written before #97's fix.

    Measured 2026-10-04: a max over sampled frames answers "was there ever a
    big face", and one frame in twelve inverted a framing A/B that the medians
    and the frames themselves agreed on. The thresholds in this module were
    re-derived on medians at the same time, so a pre-#97 CSV is being judged
    against a band that was not calibrated for its statistic -- which is why
    the fallback is logged rather than silent."""
    if row.median_face_fraction is None:
        logger.warning(
            "desync_risk: chunk %d's row has no median_face_fraction -- this scan predates "
            "issue #97's column, so it is ranked on max_face_fraction against thresholds "
            "re-derived on medians. Re-run `python -m music_video_maker.facescan <dir>` to "
            "remove the mismatch.",
            row.chunk_id,
        )
        return row.max_face_fraction
    return row.median_face_fraction


@dataclass(frozen=True)
class FaceReport:
    """A parsed ``facescan`` CSV: its provenance header and its rows."""

    input_dir: str
    header: Mapping[str, str]
    """Every ``# key=value`` pair the header carried, unmodified. Kept whole
    rather than picked apart into fields so a report written by a later
    version of ``facescan`` -- with a threshold or a model this module has
    never heard of -- still reaches the reader instead of being silently
    dropped on the way through."""
    rows: Mapping[int, FaceRow]


def _parse_header_line(line: str, header: dict[str, str]) -> None:
    """Fold one ``# key=value key2=value2`` comment line into ``header``.

    ``facescan`` writes both one-pair lines (``# input_dir=/...``) and
    multi-pair ones (``# score_threshold=0.9 inspection_floor=0.7 ...``). A
    value containing a space -- a path with a space in it, which this
    project's own checkout has -- would be split by a naive tokenizer, so a
    line holding exactly one ``=`` is taken whole and only a line with
    several is split on whitespace.
    """
    body = line.lstrip(_PROVENANCE_PREFIX).strip()
    if "=" not in body:
        return
    if body.count("=") == 1:
        key, _, value = body.partition("=")
        header[key.strip()] = value.strip()
        return
    for token in body.split():
        if "=" in token:
            key, _, value = token.partition("=")
            header[key.strip()] = value.strip()


def read_facescan_report(path: Path) -> FaceReport:
    """Parse a ``music_video_maker.facescan`` CSV, header included.

    Raises :class:`DesyncRiskError` if the file carries no ``input_dir``
    provenance header -- i.e. if it is a pre-#93 CSV, or a hand-made one, or
    anything else that cannot say which render it measured. That refusal is
    the point: #93 is the record of exactly such a file being joined against
    a different render's frames for a week with nothing able to tell.

    A row missing a column, or carrying an unparseable number, is logged and
    skipped rather than fatal: the chunk simply has no face measurement and
    is reported :data:`UNMEASURED`, which is a true statement about it.
    """
    text = Path(path).read_text()
    header: dict[str, str] = {}
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(_PROVENANCE_PREFIX):
            _parse_header_line(line, header)
        else:
            data_lines.append(line)

    input_dir = header.get(_INPUT_DIR_KEY, "").strip()
    if not input_dir:
        raise DesyncRiskError(
            f"{path} carries no '# {_INPUT_DIR_KEY}=' provenance header, so it cannot say "
            "which render it measured. Issue #93 is the record of a face-presence CSV that "
            "named one render and had read another for a week; a ranking built on an "
            "unattributable measurement is not checkable. Re-run "
            "'python -m music_video_maker.facescan <chunks_dir> --out <csv>'."
        )

    rows: dict[int, FaceRow] = {}
    for record in csv.DictReader(data_lines):
        try:
            chunk_id = int(record["chunk_id"])
            row = FaceRow(
                chunk_id=chunk_id,
                face_pct=float(record["face_pct"]),
                max_face_fraction=float(record["max_face_fraction"]),
                median_face_fraction=(
                    float(record["median_face_fraction"])
                    if (record.get("median_face_fraction") or "").strip()
                    else None
                ),
                inconclusive=int(record.get("inconclusive") or 0),
                source_path=(record.get("source_path") or "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "desync_risk: %s has a row that could not be read (%r) -- skipping it; the "
                "chunk it described will be reported as %s.",
                path,
                record,
                UNMEASURED,
            )
            continue
        rows[chunk_id] = row

    logger.info(
        "desync_risk: read %d face row(s) from %s, measured over %s.", len(rows), path, input_dir
    )
    return FaceReport(input_dir=input_dir, header=header, rows=rows)


# --------------------------------------------------------------------------- #
# The offsets: read back out of the render's own log
# --------------------------------------------------------------------------- #


def parse_leading_offsets(text: str) -> dict[int, float]:
    """Every chunk #79 reported over its warning threshold, from a render log.

    Only chunks *over* ``slicing.LEADING_VOCAL_OFFSET_WARN_SECONDS`` appear
    in the log at WARNING, which is exactly the set this module is for: #97's
    question is which of the already-flagged chunks a viewer will notice, not
    whether an unflagged one might. A chunk with a small offset and a huge
    face is not a desync anybody has reported -- six such chunks exist on
    "Deathless" and none has ever been named.

    A chunk appearing twice (a log holding a run and its ``--resume``) keeps
    the **last** occurrence: the later line describes the later timeline, and
    a resumed run is the one whose output is on disk to be scanned.
    """
    offsets: dict[int, float] = {}
    for match in _OFFSET_LINE.finditer(text):
        offsets[int(match.group("chunk"))] = float(match.group("offset"))
    logger.info("desync_risk: parsed %d leading-offset warning(s) from the log.", len(offsets))
    return offsets


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkDesyncRisk:
    """One chunk's predicted desync visibility."""

    chunk_id: int
    leading_offset: float
    face_fraction: float | None
    """``None`` when the facescan report has no row for this chunk at all."""
    inconclusive_frames: int
    verdict: str
    reason: str
    """Why this verdict, in one clause -- so a reader who disagrees can see
    which number they are disagreeing with."""


def classify(face_fraction: float | None, inconclusive_frames: int = 0) -> tuple[str, str]:
    """The verdict and its reason for one chunk's face measurement.

    Four outcomes, and the fourth is the one #93 paid for:

    * no row at all -> :data:`UNMEASURED`.
    * at/above :data:`VISIBLE_FACE_FRACTION` -> :data:`VISIBLE`.
    * at/below :data:`HIDDEN_FACE_FRACTION` **with no inconclusive frames**
      -> :data:`HIDDEN`.
    * at/below that, but the detector turned up candidates it could not
      clear the gate with -> :data:`UNMEASURED`, not hidden. "Small face" and
      "the detector could not tell" are different claims, and only the first
      one predicts anything.
    * anything between the two constants -> :data:`UNCERTAIN`. Seven
      observations do not locate a boundary inside their own gap.
    """
    if face_fraction is None:
        return UNMEASURED, "no facescan row for this chunk"
    if face_fraction >= VISIBLE_FACE_FRACTION:
        return VISIBLE, f"face {face_fraction:.4f} >= {VISIBLE_FACE_FRACTION:.4f}"
    if face_fraction <= HIDDEN_FACE_FRACTION:
        if inconclusive_frames:
            return (
                UNMEASURED,
                f"face {face_fraction:.4f} is small but {inconclusive_frames} frame(s) are "
                "inconclusive (#93) -- a low fraction here is not evidence of a small face",
            )
        return HIDDEN, f"face {face_fraction:.4f} <= {HIDDEN_FACE_FRACTION:.4f}"
    return (
        UNCERTAIN,
        f"face {face_fraction:.4f} is inside the unresolved "
        f"{HIDDEN_FACE_FRACTION:.4f}-{VISIBLE_FACE_FRACTION:.4f} band (n=7)",
    )


def rank_chunks(
    offsets: Mapping[int, float], report: FaceReport
) -> tuple[ChunkDesyncRisk, ...]:
    """Join #79's offsets with #93-stamped face sizes, most-visible first.

    Ordered by verdict (:data:`VISIBLE`, then :data:`UNCERTAIN`, then
    :data:`UNMEASURED`, then :data:`HIDDEN`) and within a verdict by face
    fraction descending, with chunk id as the final tiebreak so the output is
    deterministic. **Not** ordered by offset anywhere, which is the entire
    point: #97's measurement is that the offset ranking runs backwards
    against what a viewer reports.
    """
    ranked: list[ChunkDesyncRisk] = []
    for chunk_id, offset in offsets.items():
        row = report.rows.get(chunk_id)
        fraction = _ranking_fraction(row) if row is not None else None
        inconclusive = row.inconclusive if row is not None else 0
        verdict, reason = classify(fraction, inconclusive)
        ranked.append(
            ChunkDesyncRisk(
                chunk_id=chunk_id,
                leading_offset=offset,
                face_fraction=fraction,
                inconclusive_frames=inconclusive,
                verdict=verdict,
                reason=reason,
            )
        )
    ranked.sort(
        key=lambda risk: (
            _VERDICT_ORDER[risk.verdict],
            -(risk.face_fraction if risk.face_fraction is not None else -1.0),
            risk.chunk_id,
        )
    )
    return tuple(ranked)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def format_report(ranked: Sequence[ChunkDesyncRisk], report: FaceReport) -> str:
    """A plain-text table, headed by the provenance of the faces it ranks.

    The header is not decoration. A ranking is a claim about a particular
    render, and #93 is the record of exactly this kind of artefact outliving
    the render it described. Anyone quoting a row of this table can see, on
    the same screen, which directory the faces came from.
    """
    lines = [
        "# music_video_maker.desync_risk (issue #97)",
        f"# faces measured over: {report.input_dir}",
        f"# face band: hidden <= {HIDDEN_FACE_FRACTION:.4f} < uncertain < "
        f"{VISIBLE_FACE_FRACTION:.4f} <= visible",
        "# band from 7 adjudicated chunks of one render -- a band, not a calibrated "
        "threshold (#97)",
        "",
        f"{'chunk':>5}  {'offset':>7}  {'face':>7}  {'verdict':<11}  reason",
    ]
    for risk in ranked:
        face = "--" if risk.face_fraction is None else f"{risk.face_fraction:.4f}"
        lines.append(
            f"{risk.chunk_id:>5}  {risk.leading_offset:>+7.3f}  {face:>7}  "
            f"{risk.verdict:<11}  {risk.reason}"
        )
    if not ranked:
        lines.append("(no chunk over the #79 leading-offset warning threshold in this log)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.desync_risk",
        description=(
            "Rank a render's #79 leading-offset chunks by predicted desync visibility, "
            "using delivered face size from a provenance-stamped facescan CSV (issue #97)."
        ),
    )
    parser.add_argument(
        "facescan_csv", type=Path, help="a CSV written by python -m music_video_maker.facescan"
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="the render log to read #79 leading-offset warnings out of",
    )
    parser.add_argument("--out", type=Path, default=None, help="output path (default: stdout)")
    return parser


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Entry point for ``python -m music_video_maker.desync_risk``."""
    args = build_parser().parse_args(argv)
    out = stdout if stdout is not None else sys.stdout

    try:
        report = read_facescan_report(args.facescan_csv)
    except (OSError, DesyncRiskError) as exc:
        logger.error("desync_risk: %s", exc)
        return 1

    try:
        log_text = args.log.read_text()
    except OSError as exc:
        logger.error("desync_risk: could not read the render log %s: %s", args.log, exc)
        return 1

    ranked = rank_chunks(parse_leading_offsets(log_text), report)
    text = format_report(ranked, report)
    if args.out is not None:
        args.out.write_text(text)
        logger.info("desync_risk: wrote %d ranked chunk(s) to %s.", len(ranked), args.out)
    else:
        out.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via main(), not this guard
    raise SystemExit(main())


__all__ = [
    "HIDDEN",
    "HIDDEN_FACE_FRACTION",
    "UNCERTAIN",
    "UNMEASURED",
    "VISIBLE",
    "VISIBLE_FACE_FRACTION",
    "ChunkDesyncRisk",
    "DesyncRiskError",
    "FaceReport",
    "FaceRow",
    "build_parser",
    "classify",
    "format_report",
    "main",
    "parse_leading_offsets",
    "rank_chunks",
    "read_facescan_report",
]
