"""Tests for the face-weighted desync-visibility ranking (issue #97).

Two things are being pinned here and only one of them is arithmetic.

The first is the join itself: #79's leading vocal offset says *which* chunks
are out of phase, and #97 measured that it does not say which ones a viewer
notices -- face size does, and it runs the other way (the two largest offsets
in "Deathless" are the two least-noticed chunks). The band 0.0474-0.0778 is
seven observations from one render, so the module reports three states with
the gap left explicitly unresolved, and these tests pin the *band*, not a
threshold: a future test that asserts a single cutoff has quietly invented a
measurement.

The second is provenance (#93). A face CSV that cannot say which render it
read is refused rather than ranked, and a small face on a chunk whose frames
were *inconclusive* is reported unmeasured rather than hidden -- a zero from
a detector is not evidence of absence unless something asked the second
question.

The offset parser is tested against the output of the real emitter
(`slicing._log_leading_vocal_offset`, captured through `caplog`) rather than
a copied string, so a change to that message fails here instead of silently
emptying the ranking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from music_video_maker import desync_risk, faces, facescan
from music_video_maker import slicing as slicing_module
from music_video_maker.contracts import AlignedSegment, WordTiming

# --------------------------------------------------------------------------- #
# Fixtures: a facescan CSV written by the real writer, not a handmade string
# --------------------------------------------------------------------------- #


def _row(
    chunk_id: int,
    fraction: float,
    *,
    inconclusive: int = 0,
    median: float | None = None,
) -> facescan.ChunkFaceScan:
    return facescan.ChunkFaceScan(
        chunk_id=chunk_id,
        frames=141,
        sampled=12,
        with_face=12 if fraction > 0 else 0,
        face_pct=100.0 if fraction > 0 else 0.0,
        carries_identity=1 if fraction >= 0.02 else 0,
        inconclusive=inconclusive,
        max_face_fraction=fraction,
        median_face_fraction=fraction if median is None else median,
        source_path=Path(f"/renders/chunks_v13/chunk_{chunk_id:04d}.mp4"),
        source_size_bytes=1234,
        source_mtime="2026-09-03T00:00:00+00:00",
    )


def _write_facescan(tmp_path: Path, rows, *, input_dir: str = "/renders/chunks_v13") -> Path:
    """Write a report through `facescan.write_report` itself, so these tests
    cannot drift away from the format the instrument actually produces."""
    path = tmp_path / "faces.csv"
    with path.open("w", newline="") as handle:
        facescan.write_report(
            rows,
            handle,
            input_dir=Path(input_dir),
            score_threshold=faces.DEFAULT_SCORE_THRESHOLD,
            inspect_floor=faces.DEFAULT_INSPECTION_FLOOR,
            samples=12,
        )
    return path


# --------------------------------------------------------------------------- #
# Reading the facescan report: provenance is a precondition (#93)
# --------------------------------------------------------------------------- #


def test_reads_rows_and_the_input_directory_from_a_real_facescan_report(tmp_path):
    path = _write_facescan(tmp_path, [_row(74, 0.0778), _row(20, 0.0120)])

    report = desync_risk.read_facescan_report(path)

    assert Path(report.input_dir).name == "chunks_v13"
    assert report.rows[74].max_face_fraction == pytest.approx(0.0778)
    assert report.rows[20].source_path.endswith("chunk_0020.mp4")


def test_a_csv_with_no_provenance_header_is_refused(tmp_path):
    """The whole of #93: a face CSV that named one render and had read
    another for a week, with every analysis built on it internally
    consistent and wrong. A join is exactly where that gets laundered into a
    finding, so the consumer checks rather than assuming."""
    path = tmp_path / "legacy.csv"
    path.write_text("chunk_id,face_pct,max_face_fraction\n74,100.0,0.0778\n")

    with pytest.raises(desync_risk.DesyncRiskError, match="input_dir"):
        desync_risk.read_facescan_report(path)


def test_a_header_value_containing_spaces_survives_parsing(tmp_path):
    """This project's own checkout lives under a path with spaces in it, so
    a naive whitespace tokenizer would truncate the one field that matters."""
    path = _write_facescan(tmp_path, [_row(1, 0.2)], input_dir=str(tmp_path / "a dir with spaces"))

    report = desync_risk.read_facescan_report(path)

    assert report.input_dir.endswith("a dir with spaces")


def test_multi_pair_header_lines_are_all_captured(tmp_path):
    path = _write_facescan(tmp_path, [_row(1, 0.2)])

    header = desync_risk.read_facescan_report(path).header

    assert header["score_threshold"] == str(faces.DEFAULT_SCORE_THRESHOLD)
    assert header["inspection_floor"] == str(faces.DEFAULT_INSPECTION_FLOOR)
    assert header["samples_per_chunk"] == "12"


def test_a_header_comment_with_no_equals_sign_is_ignored(tmp_path):
    path = _write_facescan(tmp_path, [_row(1, 0.2)])
    path.write_text("# a free-form note about this scan\n" + path.read_text())

    assert desync_risk.read_facescan_report(path).rows[1].max_face_fraction == pytest.approx(0.2)


def test_an_unreadable_row_is_skipped_and_the_rest_of_the_report_survives(tmp_path, caplog):
    """One bad row must not lose seventy-nine good ones -- the project-wide
    'one failure must not kill the run' rule, scaled down to a CSV line. The
    chunk it described simply has no measurement, which is a true statement
    about it."""
    path = _write_facescan(tmp_path, [_row(74, 0.0778), _row(20, 0.0120)])
    text = path.read_text().replace("0.012", "not-a-number")
    path.write_text(text)

    with caplog.at_level(logging.WARNING):
        report = desync_risk.read_facescan_report(path)

    assert 74 in report.rows
    assert 20 not in report.rows
    assert "skipping it" in caplog.text


# --------------------------------------------------------------------------- #
# The offsets: parsed out of the real emitter's own WARNING line (#79)
# --------------------------------------------------------------------------- #


def _word_seg(index: int, text: str, start: float, end: float) -> AlignedSegment:
    return AlignedSegment(
        index=index,
        text=text,
        start=start,
        end=end,
        words=(WordTiming(word=text, start=start, end=end),),
        characters=("Dianne",),
    )


def test_parses_the_offsets_the_real_slicing_emitter_logs(caplog):
    """Round-trip against the emitter rather than a copied string: if
    `_log_leading_vocal_offset`'s wording changes, this fails loudly instead
    of the ranking silently going empty."""
    seg_a = _word_seg(0, "late", 6.5, 9.5)
    seg_b = _word_seg(1, "later", 22.0, 25.0)
    pieces = [
        slicing_module._Piece(
            members=(seg_a,), start=5.0, end=11.0, is_split_continuation=False, frame_count=144
        ),
        slicing_module._Piece(
            members=(seg_b,), start=19.0, end=26.0, is_split_continuation=False, frame_count=168
        ),
    ]

    with caplog.at_level(logging.WARNING):
        slicing_module._log_leading_vocal_offset(pieces, (seg_a, seg_b))

    offsets = desync_risk.parse_leading_offsets(caplog.text)

    assert offsets == {0: pytest.approx(1.5), 1: pytest.approx(3.0)}


def test_only_chunks_over_the_warning_threshold_appear(caplog):
    """The log carries a per-chunk line only above
    `LEADING_VOCAL_OFFSET_WARN_SECONDS`, which is exactly the set #97 is
    about: six chunks on the same render carry faces of 0.235-0.356 with
    near-zero offsets and none has ever been reported."""
    quiet = _word_seg(0, "soon", 5.3, 9.0)
    piece = slicing_module._Piece(
        members=(quiet,), start=5.0, end=11.0, is_split_continuation=False, frame_count=144
    )

    with caplog.at_level(logging.INFO):
        slicing_module._log_leading_vocal_offset([piece], (quiet,))

    assert desync_risk.parse_leading_offsets(caplog.text) == {}


def test_a_chunk_logged_twice_keeps_the_later_line():
    """A log holding a run and its `--resume` describes the same chunk
    twice; the later line describes the later timeline, and the later
    timeline is the one whose mp4 is on disk to be scanned."""
    text = (
        "WARNING Chunk 38 (100.000-108.000s) is prompted to sing starting 2.588s into its "
        "own span (first word onset 102.588s) -- ...\n"
        "WARNING Chunk 38 (100.000-108.000s) is prompted to sing starting 1.100s into its "
        "own span (first word onset 101.100s) -- ...\n"
    )

    assert desync_risk.parse_leading_offsets(text) == {38: pytest.approx(1.1)}


def test_a_log_with_no_offset_warnings_parses_to_nothing():
    assert desync_risk.parse_leading_offsets("nothing to see here\n") == {}


# --------------------------------------------------------------------------- #
# The classification: a band, never a threshold
# --------------------------------------------------------------------------- #


def test_the_band_is_the_two_observed_values_not_a_rounded_pair():
    """0.0557 is the smallest median a viewer noticed; 0.0441 is the largest
    they did not. Rounding either would put a number nobody measured where a
    measurement is. Re-derived on medians 2026-10-04 (#97) after the max was
    shown to invert a framing A/B on one frame in twelve; the same seven
    adjudicated chunks, rescanned."""
    assert pytest.approx(0.0557) == desync_risk.VISIBLE_FACE_FRACTION
    assert pytest.approx(0.0441) == desync_risk.HIDDEN_FACE_FRACTION
    assert desync_risk.HIDDEN_FACE_FRACTION < desync_risk.VISIBLE_FACE_FRACTION


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.1982, desync_risk.VISIBLE),  # chunk 35, noticed
        (0.1405, desync_risk.VISIBLE),  # chunk 41, noticed
        (0.0711, desync_risk.VISIBLE),  # chunk 74, noticed
        (0.0557, desync_risk.VISIBLE),  # chunk 38, noticed -- the boundary itself
        (0.0441, desync_risk.HIDDEN),  # chunk 73, not noticed -- the other boundary
        (0.0109, desync_risk.HIDDEN),  # chunk 58, not noticed
        (0.0090, desync_risk.HIDDEN),  # chunk 20, not noticed
    ],
)
def test_every_adjudicated_chunk_of_issue_97_classifies_as_the_viewer_reported(
    fraction, expected
):
    """The seven chunks the issue's own table records, with the labels a
    viewer gave them. This is the entire evidence base; if a change to the
    constants breaks it, the change is wrong."""
    verdict, _ = desync_risk.classify(fraction)
    assert verdict == expected


def test_a_face_inside_the_band_is_uncertain_rather_than_guessed():
    verdict, reason = desync_risk.classify(0.0500)

    assert verdict == desync_risk.UNCERTAIN
    assert "band" in reason


def test_a_missing_row_is_unmeasured_not_hidden():
    verdict, reason = desync_risk.classify(None)

    assert verdict == desync_risk.UNMEASURED
    assert "no facescan row" in reason


def test_a_small_face_with_inconclusive_frames_is_unmeasured_not_hidden():
    """#93's other half, at the point where it actually changes a finding: a
    bare zero has only ever meant 'nothing cleared 0.9'. Ranking such a chunk
    as 'the face is too small to notice' is the same mistake that CSV made,
    one level down."""
    verdict, reason = desync_risk.classify(0.0100, inconclusive_frames=4)

    assert verdict == desync_risk.UNMEASURED
    assert "#93" in reason


def test_inconclusive_frames_do_not_downgrade_a_clearly_visible_face():
    """The `inconclusive` override applies only where the claim depends on
    the detector having looked hard enough -- a large detected face is not
    made uncertain by some other frame's near-miss."""
    verdict, _ = desync_risk.classify(0.2101, inconclusive_frames=3)

    assert verdict == desync_risk.VISIBLE


# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #


def _issue_97_report(tmp_path) -> desync_risk.FaceReport:
    rows = [
        _row(74, 0.0778, median=0.0711),
        _row(38, 0.0887, median=0.0557),
        _row(41, 0.1656, median=0.1405),
        _row(35, 0.2101, median=0.1982),
        _row(73, 0.0474, median=0.0441),
        _row(58, 0.0460, median=0.0109),
        _row(20, 0.0120, median=0.0090),
    ]
    return desync_risk.read_facescan_report(_write_facescan(tmp_path, rows))


_ISSUE_97_OFFSETS = {74: 1.067, 38: 2.588, 41: 1.607, 35: 1.858, 73: 1.282, 58: 2.648, 20: 2.650}


def test_the_ranking_reproduces_the_viewers_two_groups(tmp_path):
    ranked = desync_risk.rank_chunks(_ISSUE_97_OFFSETS, _issue_97_report(tmp_path))

    visible = [risk.chunk_id for risk in ranked if risk.verdict == desync_risk.VISIBLE]
    hidden = [risk.chunk_id for risk in ranked if risk.verdict == desync_risk.HIDDEN]

    assert set(visible) == {35, 41, 38, 74}
    assert set(hidden) == {73, 58, 20}


def test_the_ranking_is_not_the_offset_ranking(tmp_path):
    """The point of the module. Ordered by offset, chunk 20 (+2.650s) comes
    first and has never been reported by anyone; ordered by face, it comes
    last."""
    ranked = desync_risk.rank_chunks(_ISSUE_97_OFFSETS, _issue_97_report(tmp_path))

    assert ranked[0].chunk_id == 35
    assert ranked[-1].chunk_id == 20
    assert max(_ISSUE_97_OFFSETS, key=_ISSUE_97_OFFSETS.get) == 20


def test_unmeasured_chunks_are_reported_above_hidden_ones(tmp_path):
    """'We could not tell' is a reason to open the file; 'the face is too
    small to read' is a reason not to."""
    report = desync_risk.read_facescan_report(
        _write_facescan(tmp_path, [_row(20, 0.0120), _row(58, 0.0100, inconclusive=3)])
    )

    ranked = desync_risk.rank_chunks({20: 2.650, 58: 2.648, 99: 1.5}, report)

    assert [risk.verdict for risk in ranked] == [
        desync_risk.UNMEASURED,
        desync_risk.UNMEASURED,
        desync_risk.HIDDEN,
    ]
    assert {risk.chunk_id for risk in ranked[:2]} == {58, 99}


def test_the_ranking_is_deterministic_on_ties(tmp_path):
    report = desync_risk.read_facescan_report(
        _write_facescan(tmp_path, [_row(9, 0.1), _row(4, 0.1)])
    )

    ranked = desync_risk.rank_chunks({9: 1.2, 4: 1.3}, report)

    assert [risk.chunk_id for risk in ranked] == [4, 9]


# --------------------------------------------------------------------------- #
# The report and the CLI
# --------------------------------------------------------------------------- #


def test_the_report_names_the_directory_the_faces_were_measured_from(tmp_path):
    report = _issue_97_report(tmp_path)

    text = desync_risk.format_report(desync_risk.rank_chunks(_ISSUE_97_OFFSETS, report), report)

    assert "chunks_v13" in text
    assert "band, not a calibrated threshold" in text
    assert "   74  " in text


def test_the_report_says_so_when_nothing_was_flagged(tmp_path):
    report = _issue_97_report(tmp_path)

    text = desync_risk.format_report((), report)

    assert "no chunk over the #79 leading-offset warning threshold" in text


def test_a_chunk_with_no_face_row_prints_a_dash_rather_than_a_number(tmp_path):
    report = desync_risk.read_facescan_report(_write_facescan(tmp_path, [_row(1, 0.2)]))

    text = desync_risk.format_report(desync_risk.rank_chunks({42: 1.9}, report), report)

    assert "--" in text
    assert "unmeasured" in text


def test_cli_writes_a_ranking_to_stdout(tmp_path, capsys):
    csv_path = _write_facescan(tmp_path, [_row(38, 0.0887), _row(20, 0.0120)])
    log_path = tmp_path / "render.log"
    log_path.write_text(
        "Chunk 38 (100.000-108.000s) is prompted to sing starting 2.588s into its own span "
        "(first word onset 102.588s) -- x\n"
        "Chunk 20 (50.000-58.000s) is prompted to sing starting 2.650s into its own span "
        "(first word onset 52.650s) -- x\n"
    )

    rc = desync_risk.main([str(csv_path), "--log", str(log_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.index("   38") < out.index("   20")


def test_cli_writes_to_a_file_when_asked(tmp_path):
    csv_path = _write_facescan(tmp_path, [_row(38, 0.0887)])
    log_path = tmp_path / "render.log"
    log_path.write_text(
        "Chunk 38 (1.000-9.000s) is prompted to sing starting 2.588s into its own span "
        "(first word onset 3.588s) -- x\n"
    )
    out_path = tmp_path / "risk.txt"

    assert desync_risk.main([str(csv_path), "--log", str(log_path), "--out", str(out_path)]) == 0
    assert "visible" in out_path.read_text()


def test_cli_refuses_a_facescan_csv_with_no_provenance(tmp_path, caplog):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text("chunk_id,face_pct,max_face_fraction\n38,100.0,0.0887\n")
    log_path = tmp_path / "render.log"
    log_path.write_text("")

    with caplog.at_level(logging.ERROR):
        rc = desync_risk.main([str(csv_path), "--log", str(log_path)])

    assert rc == 1
    assert "provenance header" in caplog.text


def test_cli_reports_a_missing_facescan_file_without_raising(tmp_path, caplog):
    log_path = tmp_path / "render.log"
    log_path.write_text("")

    with caplog.at_level(logging.ERROR):
        rc = desync_risk.main([str(tmp_path / "nope.csv"), "--log", str(log_path)])

    assert rc == 1


def test_cli_reports_a_missing_log_without_raising(tmp_path, caplog):
    csv_path = _write_facescan(tmp_path, [_row(38, 0.0887)])

    with caplog.at_level(logging.ERROR):
        rc = desync_risk.main([str(csv_path), "--log", str(tmp_path / "nope.log")])

    assert rc == 1
    assert "render log" in caplog.text


# --------------------------------------------------------------------------- #
# The statistic the ranking is built on (#97, measured 2026-10-04).
# --------------------------------------------------------------------------- #

# The seven chunks a viewer adjudicated on "Deathless" v13, rescanned with
# facescan's median column: (chunk, label, max, median).
_ADJUDICATED = [
    (35, "noticed", 0.2101, 0.1982),
    (41, "noticed", 0.1656, 0.1405),
    (74, "noticed", 0.0778, 0.0711),
    (38, "noticed", 0.0887, 0.0557),
    (73, "not", 0.0474, 0.0441),
    (58, "not", 0.0460, 0.0109),
    (20, "not", 0.0120, 0.0090),
]


def test_every_adjudicated_chunk_classifies_as_the_viewer_labelled_it_on_medians(tmp_path):
    """F42's separation survives the change of statistic: n=7, no overlap.
    Chunk 58 moves most (0.0460 max -> 0.0109 median) and is one a viewer did
    not notice, which is the direction that makes the median the better
    statistic rather than merely a different one."""
    path = _write_facescan(
        tmp_path, [_row(c, mx, median=med) for c, _, mx, med in _ADJUDICATED]
    )
    report = desync_risk.read_facescan_report(path)

    for chunk, label, _, _ in _ADJUDICATED:
        verdict, _ = desync_risk.classify(
            desync_risk._ranking_fraction(report.rows[chunk]),
            report.rows[chunk].inconclusive,
        )
        expected = desync_risk.VISIBLE if label == "noticed" else desync_risk.HIDDEN
        assert verdict == expected, f"chunk {chunk} ({label}) classified {verdict}"


def test_the_median_is_what_is_ranked_not_the_max(tmp_path):
    """The measured inversion: a wide shot that ends on a close-up scores a
    higher max than the close arm of the same chunk, while the medians -- and
    the frames -- say the opposite."""
    path = _write_facescan(tmp_path, [_row(20, 0.1060, median=0.0098)])
    report = desync_risk.read_facescan_report(path)

    assert desync_risk._ranking_fraction(report.rows[20]) == 0.0098


def test_a_scan_written_before_the_median_column_falls_back_and_says_so(tmp_path, caplog):
    """A pre-#97 CSV is still usable, but it is being judged against a band
    re-derived on medians, so the mismatch is logged rather than silent."""
    path = _write_facescan(tmp_path, [_row(41, 0.1656)])
    text = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(text) if line.startswith("chunk_id,"))
    columns = text[header_index].split(",")
    drop = columns.index("median_face_fraction")
    text[header_index] = ",".join(c for i, c in enumerate(columns) if i != drop)
    for i in range(header_index + 1, len(text)):
        if text[i].strip():
            cells = text[i].split(",")
            text[i] = ",".join(c for j, c in enumerate(cells) if j != drop)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")

    report = desync_risk.read_facescan_report(path)
    with caplog.at_level(logging.WARNING):
        assert desync_risk._ranking_fraction(report.rows[41]) == 0.1656

    assert any("predates issue #97" in r.getMessage() for r in caplog.records)
