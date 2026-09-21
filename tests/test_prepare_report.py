"""Tests for the ``--prepare`` pre-render report (issue #36).

Three layers, matching how the module is split:

* the **file format** -- round-trip, schema refusal, and the read contract
  (every failure is one exception type naming the path, because the one
  consumer has one sentence to say about all of them);
* the **capture** -- that what Stage 1-2 logs is what the report carries,
  including under a ``--log-level`` that would otherwise have gated it away;
* the **wiring** -- that a real ``--prepare`` over the offline ``Rig``
  ``tests/test_cli.py`` already uses writes a report describing the timeline
  it just wrote a skeleton for.

No GPU, no ComfyUI, no network, no real alignment model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from music_video_maker import cli
from music_video_maker.alignment_quality import AlignmentQualityReport, Finding, Severity
from music_video_maker.contracts import AudioChunk
from music_video_maker.prepare_report import (
    PREPARE_REPORT_SCHEMA_VERSION,
    InputStamp,
    PrepareReport,
    PrepareReportError,
    StageNotice,
    build_prepare_report,
    collect_stage_notices,
    plan_resolution_errors,
    read_prepare_report,
    stale_inputs,
    write_prepare_report,
)
from tests.test_cli import Rig


def _chunk(chunk_id: int, start: float, end: float, *, instrumental: bool = False) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id:04d}.wav"),
        start=start,
        end=end,
        text="" if instrumental else "a lyric line",
        frame_count=141,
        is_instrumental=instrumental,
    )


def _quality_report() -> AlignmentQualityReport:
    return AlignmentQualityReport(
        findings=(
            Finding(
                segment_index=0,
                start=0.0,
                end=0.02,
                severity=Severity.CRITICAL,
                code="zero_length_segment",
                message="segment 0 is 20ms long",
            ),
            Finding(
                segment_index=1,
                start=6.0,
                end=9.0,
                severity=Severity.WARNING,
                code="implausible_word_rate",
                message="segment 1 sings 14 words in 3.0s",
            ),
            Finding(
                segment_index=2,
                start=9.0,
                end=12.0,
                severity=Severity.INFO,
                code="noted",
                message="nothing to act on",
            ),
        ),
        segment_count=3,
    )


def _report(**overrides) -> PrepareReport:
    base = build_prepare_report(
        generated_at="2026-09-21",
        config_path="run.toml",
        inputs=[InputStamp.of("lyrics_file", None)],
        alignment_model_size="small",
        strict_alignment=False,
        chunks=[_chunk(0, 0.0, 6.0), _chunk(1, 6.0, 9.0, instrumental=True)],
        quality_report=_quality_report(),
        track_duration_seconds=10.0,
        timeline_drift_seconds=-1.0,
        fps=24,
        duration_tolerance_seconds=0.042,
    )
    return base if not overrides else type(base)(**{**base.__dict__, **overrides})


# --------------------------------------------------------------------------- #
# The file format
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_to_dict_from_dict_is_lossless(self) -> None:
        report = _report(
            notices=(
                StageNotice(logger="x", level="WARNING", message="m (issue #70)", issue="70"),
            ),
            plan_errors=("chunk 3 drifted",),
            plan_checked="shot_plan.toml",
            plan_lengths_applied=True,
        )
        assert PrepareReport.from_dict(report.to_dict()) == report

    def test_the_payload_is_plain_json(self) -> None:
        """No custom encoder anywhere: a Severity IntEnum or a Path leaking
        into the tree would serialize by accident today and break a reader
        tomorrow."""
        payload = json.dumps(_report().to_dict(), sort_keys=True)
        assert json.loads(payload)["alignment_finding_counts"]["CRITICAL"] == 1

    def test_an_unknown_schema_version_is_refused_not_guessed(self) -> None:
        raw = _report().to_dict()
        raw["schema_version"] = PREPARE_REPORT_SCHEMA_VERSION + 1
        with pytest.raises(PrepareReportError, match="schema_version"):
            PrepareReport.from_dict(raw)

    def test_a_payload_with_no_schema_version_is_refused(self) -> None:
        raw = _report().to_dict()
        del raw["schema_version"]
        with pytest.raises(PrepareReportError):
            PrepareReport.from_dict(raw)

    def test_optional_keys_may_be_absent(self) -> None:
        """Only ``schema_version`` is load-bearing on read; everything else
        defaults, so a field added later does not make today's files
        unreadable."""
        report = PrepareReport.from_dict({"schema_version": PREPARE_REPORT_SCHEMA_VERSION})
        assert report.chunk_count == 0
        assert report.notices == ()
        assert report.critical_finding_count == 0


class TestReadWrite:
    def test_write_creates_parents_and_reads_back_equal(self, tmp_path: Path) -> None:
        path = tmp_path / "output" / "chunks" / "prepare_report.json"
        report = _report()
        assert write_prepare_report(report, path) == path
        assert read_prepare_report(path) == report

    def test_write_overwrites_because_a_report_is_derived_data(self, tmp_path: Path) -> None:
        path = tmp_path / "prepare_report.json"
        write_prepare_report(_report(), path)
        write_prepare_report(_report(generated_at="2026-09-22"), path)
        assert read_prepare_report(path).generated_at == "2026-09-22"

    def test_written_json_is_sorted_and_newline_terminated(self, tmp_path: Path) -> None:
        path = tmp_path / "prepare_report.json"
        write_prepare_report(_report(), path)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"

    @pytest.mark.parametrize(
        "contents", [None, "{not json", '"a string, not an object"', "[1, 2]"]
    )
    def test_every_read_failure_is_one_exception_naming_the_path(
        self, tmp_path: Path, contents: str | None
    ) -> None:
        path = tmp_path / "prepare_report.json"
        if contents is not None:
            path.write_text(contents, encoding="utf-8")
        with pytest.raises(PrepareReportError) as excinfo:
            read_prepare_report(path)
        assert str(path) in str(excinfo.value)

    def test_a_file_from_an_unknown_schema_reads_as_the_same_error(self, tmp_path: Path) -> None:
        path = tmp_path / "prepare_report.json"
        raw = _report().to_dict()
        raw["schema_version"] = 99
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(PrepareReportError):
            read_prepare_report(path)


# --------------------------------------------------------------------------- #
# Provenance (#93)
# --------------------------------------------------------------------------- #


class TestInputStamps:
    def test_an_unset_optional_input_is_none_not_a_missing_file(self) -> None:
        assert InputStamp.of("shot_plan", None) is None

    def test_a_missing_file_is_recorded_as_missing_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        stamp = InputStamp.of("lyrics_file", tmp_path / "gone.txt")
        assert stamp is not None
        assert stamp.exists is False
        assert stamp.size_bytes is None

    def test_a_real_file_records_size_and_mtime(self, tmp_path: Path) -> None:
        path = tmp_path / "lyrics.txt"
        path.write_text("la la la\n", encoding="utf-8")
        stamp = InputStamp.of("lyrics_file", path)
        assert stamp is not None
        assert stamp.exists is True
        assert stamp.size_bytes == path.stat().st_size
        assert stamp.mtime_ns == path.stat().st_mtime_ns

    def test_stale_inputs_is_empty_while_nothing_has_moved(self, tmp_path: Path) -> None:
        path = tmp_path / "lyrics.txt"
        path.write_text("la la la\n", encoding="utf-8")
        report = _report(inputs=(InputStamp.of("lyrics_file", path),))
        assert stale_inputs(report) == ()

    def test_an_edited_input_is_reported_stale(self, tmp_path: Path) -> None:
        """The whole reason the stamps exist: a report is a measurement of
        one set of files, and editing the lyrics makes it a measurement of
        something else while the file on disk still looks like a report
        about this run (#93 -- a filename is not provenance)."""
        path = tmp_path / "lyrics.txt"
        path.write_text("la la la\n", encoding="utf-8")
        report = _report(inputs=(InputStamp.of("lyrics_file", path),))
        path.write_text("a different lyric entirely\n", encoding="utf-8")
        assert stale_inputs(report) == ("lyrics_file",)

    def test_an_input_deleted_since_the_report_is_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "lyrics.txt"
        path.write_text("la la la\n", encoding="utf-8")
        report = _report(inputs=(InputStamp.of("lyrics_file", path),))
        path.unlink()
        assert stale_inputs(report) == ("lyrics_file",)

    def test_an_input_missing_then_and_now_is_not_reported_as_changed(
        self, tmp_path: Path
    ) -> None:
        """It was already recorded as missing. Reporting it as *changed*
        would send an operator looking for an edit nobody made."""
        report = _report(inputs=(InputStamp.of("lyrics_file", tmp_path / "gone.txt"),))
        assert stale_inputs(report) == ()

    def test_a_file_that_appeared_since_the_report_is_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "shot_plan.toml"
        report = _report(inputs=(InputStamp.of("shot_plan", path),))
        path.write_text("# now it exists\n", encoding="utf-8")
        assert stale_inputs(report) == ("shot_plan",)


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #


class TestCollectStageNotices:
    def test_captures_warnings_and_above_from_the_named_loggers_only(self) -> None:
        with collect_stage_notices(("a.logger", "b.logger")) as records:
            logging.getLogger("a.logger").warning("a warning (issue #70)")
            logging.getLogger("b.logger").error("an error")
            logging.getLogger("a.logger").info("an info line nobody asked for")
            logging.getLogger("c.logger").warning("a different logger entirely")
        assert [r.getMessage() for r in records] == ["a warning (issue #70)", "an error"]

    def test_captures_even_when_the_console_level_would_have_gated_it(self) -> None:
        """``--prepare --log-level ERROR`` must not silently produce a report
        with every Stage-2 warning missing from it. The console's verbosity
        is the operator's choice; the report's contents are not."""
        one = logging.getLogger("gated.logger")
        one.setLevel(logging.ERROR)
        try:
            with collect_stage_notices(("gated.logger",)) as records:
                one.warning("a warning the console would not show")
            assert [r.getMessage() for r in records] == [
                "a warning the console would not show"
            ]
        finally:
            one.setLevel(logging.NOTSET)

    def test_restores_level_handlers_and_propagation_afterwards(self) -> None:
        one = logging.getLogger("restored.logger")
        one.setLevel(logging.ERROR)
        one.propagate = True
        try:
            handlers_before = list(one.handlers)
            with collect_stage_notices(("restored.logger",)):
                pass
            assert one.level == logging.ERROR
            assert one.propagate is True
            assert one.handlers == handlers_before
        finally:
            one.setLevel(logging.NOTSET)

    def test_restores_even_when_the_block_raises(self) -> None:
        one = logging.getLogger("raising.logger")
        handlers_before = list(one.handlers)
        with pytest.raises(RuntimeError), collect_stage_notices(("raising.logger",)):
            raise RuntimeError("boom")
        assert one.handlers == handlers_before


class TestStageNotice:
    def test_takes_the_last_issue_number_the_message_names(self) -> None:
        """These messages cite background issues mid-sentence and close with
        the one they are actually about -- see slicing's own #70 warning,
        which mentions #79's pass before naming itself."""
        record = logging.LogRecord(
            "music_video_maker.slicing", logging.WARNING, __file__, 1,
            "pass 2 (issue #79) moved it; this is issue #70.", None, None,
        )
        assert StageNotice.of(record).issue == "70"

    def test_a_message_naming_no_issue_has_none(self) -> None:
        record = logging.LogRecord(
            "music_video_maker.cli", logging.ERROR, __file__, 1,
            "Stage 2 timeline UNDERshoots the master track by 6.000s", None, None,
        )
        assert StageNotice.of(record).issue is None

    def test_notices_group_by_issue_with_the_unattributed_ones_last(self) -> None:
        report = _report(
            notices=(
                StageNotice("music_video_maker.slicing", "WARNING", "a (issue #70)", "70"),
                StageNotice("music_video_maker.cli", "ERROR", "b", None),
                StageNotice("music_video_maker.slicing", "WARNING", "c (issue #70)", "70"),
                StageNotice("music_video_maker.slicing", "WARNING", "d (issue #79)", "79"),
            )
        )
        assert [key for key, _ in report.notices_by_issue()] == ["70", "79", "other"]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


class TestBuildPrepareReport:
    def test_counts_chunks_voiced_and_instrumental(self) -> None:
        report = _report()
        assert (report.chunk_count, report.voiced_chunk_count, report.instrumental_chunk_count) == (
            2,
            1,
            1,
        )
        assert report.timeline_start == 0.0
        assert report.timeline_end == 9.0
        assert report.total_frames == 282

    def test_lists_findings_at_warning_and_above_but_counts_all_of_them(self) -> None:
        report = _report()
        assert [row.severity for row in report.alignment_findings] == ["CRITICAL", "WARNING"]
        assert report.alignment_finding_counts["INFO"] == 1
        assert report.critical_finding_count == 1
        assert report.warning_finding_count == 1

    def test_carries_the_summary_line_unmodified(self) -> None:
        assert _report().alignment_summary.startswith("Alignment quality: 3 segment(s)")

    def test_drift_is_converted_to_frames_at_this_runs_fps(self) -> None:
        assert _report().timeline_drift_frames == pytest.approx(-24.0)

    def test_an_empty_timeline_reports_none_rather_than_a_number(self) -> None:
        report = build_prepare_report(
            generated_at="2026-09-21",
            config_path="run.toml",
            inputs=[],
            alignment_model_size=None,
            strict_alignment=False,
            chunks=[],
            quality_report=AlignmentQualityReport(),
            track_duration_seconds=None,
            timeline_drift_seconds=None,
            fps=None,
            duration_tolerance_seconds=None,
        )
        assert report.timeline_start is None
        assert report.timeline_end is None
        assert report.total_frames is None
        assert report.timeline_drift_frames is None


# --------------------------------------------------------------------------- #
# Shot-plan drift, through the render's own loaders
# --------------------------------------------------------------------------- #


_PLAN_TOML = """
[[shot]]
chunk_id = 0
start = 0.0
shot = "a wide shot of the corridor, her hand trailing the wall"

[[shot]]
chunk_id = 1
start = 6.0
shot = "the lights flicker above the empty desk"
"""


class TestPlanResolutionErrors:
    def test_a_plan_matching_the_timeline_reports_nothing(self, tmp_path: Path) -> None:
        plan = tmp_path / "shot_plan.toml"
        plan.write_text(_PLAN_TOML, encoding="utf-8")
        chunks = [_chunk(0, 0.0, 6.0), _chunk(1, 6.0, 9.0)]
        assert plan_resolution_errors(plan, chunks, setting=None, cast_names=()) == ()

    def test_a_moved_chunk_start_is_reported_per_chunk(self, tmp_path: Path) -> None:
        """This is the ShotPlanDriftError a human review of the plan cannot
        perform: the file reads perfectly well either way, and the first
        thing that notices today is the render, hours in."""
        plan = tmp_path / "shot_plan.toml"
        plan.write_text(_PLAN_TOML, encoding="utf-8")
        chunks = [_chunk(0, 0.0, 6.0), _chunk(1, 7.4, 10.4)]
        errors = plan_resolution_errors(plan, chunks, setting=None, cast_names=())
        assert len(errors) == 1
        assert "chunk_id=1" in errors[0]

    def test_a_plan_that_cannot_be_read_is_a_finding_not_an_exception(
        self, tmp_path: Path
    ) -> None:
        errors = plan_resolution_errors(
            tmp_path / "nope.toml", [_chunk(0, 0.0, 6.0)], setting=None, cast_names=()
        )
        assert len(errors) == 1


# --------------------------------------------------------------------------- #
# Wiring: a real --prepare writes a report describing the timeline it just
# wrote a skeleton for
# --------------------------------------------------------------------------- #


class TestPrepareWritesTheReport:
    def _prepare(self, tmp_path: Path, **kwargs):
        rig = Rig(tmp_path)
        report_path = tmp_path / "output" / "chunks" / "prepare_report.json"
        cli.prepare_shot_plan(
            rig.config,
            tmp_path / "shot_plan.toml",
            source="run.toml",
            generated_at="2026-09-21",
            align_model=rig.align_model,
            force=True,
            report_path=report_path,
            **kwargs,
        )
        return rig, report_path

    def test_the_report_describes_the_same_timeline_as_the_skeleton(self, tmp_path: Path) -> None:
        rig, report_path = self._prepare(tmp_path)
        report = read_prepare_report(report_path)
        assert report.chunk_count == 3  # the rig's three lyric segments
        assert report.generated_at == "2026-09-21"
        assert report.config_path == "run.toml"
        assert rig.session.requests == [], "still no ComfyUI"

    def test_the_report_carries_the_stage_2_warnings_that_only_went_to_a_log(
        self, tmp_path: Path
    ) -> None:
        """The rig's three segments end at 19.0s against a 25.0s master, so
        Stage 2 reports a real undershoot (#22). Before this report existed
        that line went to a terminal and nowhere else."""
        _rig, report_path = self._prepare(tmp_path)
        report = read_prepare_report(report_path)
        assert any("UNDERshoots" in notice.message for notice in report.notices)
        assert report.timeline_drift_seconds is not None
        assert report.timeline_drift_seconds < 0

    def test_the_report_stamps_the_inputs_it_was_computed_from(self, tmp_path: Path) -> None:
        _rig, report_path = self._prepare(tmp_path)
        report = read_prepare_report(report_path)
        labels = {stamp.label for stamp in report.inputs}
        assert {"master_audio", "lyrics_file"} <= labels
        assert stale_inputs(report) == ()

    def test_editing_the_lyrics_after_the_prepare_makes_the_report_stale(
        self, tmp_path: Path
    ) -> None:
        rig, report_path = self._prepare(tmp_path)
        rig.config.lyrics_file.write_text("a completely different song\n", encoding="utf-8")
        assert stale_inputs(read_prepare_report(report_path)) == ("lyrics_file",)

    def test_no_report_is_written_when_the_prepare_itself_refuses(self, tmp_path: Path) -> None:
        """A report left behind by a prepare that refused would describe
        something other than what its reader assumes -- the #93 failure in
        miniature. The skeleton is written first, and the report only after
        it lands."""
        from music_video_maker.shot_plan import ShotPlanError

        rig = Rig(tmp_path)
        out_path = tmp_path / "shot_plan.toml"
        out_path.write_text("# hand-authored, must not be clobbered\n", encoding="utf-8")
        report_path = tmp_path / "prepare_report.json"

        with pytest.raises(ShotPlanError):
            cli.prepare_shot_plan(
                rig.config,
                out_path,
                source="run.toml",
                generated_at="2026-09-21",
                align_model=rig.align_model,
                report_path=report_path,
            )

        assert not report_path.exists()

    def test_an_unwritable_report_path_does_not_fail_a_good_prepare(
        self, tmp_path: Path, caplog
    ) -> None:
        """The skeleton is the real work and it is already on disk. A full
        disk must not turn a successful prepare into a failed one -- but it
        is logged loudly, because a report nobody can write is still worth
        knowing about."""
        rig = Rig(tmp_path)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory\n", encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            result = cli.prepare_shot_plan(
                rig.config,
                tmp_path / "shot_plan.toml",
                source="run.toml",
                generated_at="2026-09-21",
                align_model=rig.align_model,
                report_path=blocker / "prepare_report.json",
            )
        assert result.exists()
        assert "pre-render report could not be written" in caplog.text

    def test_no_report_path_means_no_report_and_unchanged_behaviour(
        self, tmp_path: Path
    ) -> None:
        rig = Rig(tmp_path)
        cli.prepare_shot_plan(
            rig.config,
            tmp_path / "shot_plan.toml",
            source="run.toml",
            generated_at="2026-09-21",
            align_model=rig.align_model,
        )
        assert not (tmp_path / "output" / "chunks" / "prepare_report.json").exists()
