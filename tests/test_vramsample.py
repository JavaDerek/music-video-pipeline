"""The attended VRAM sampler (issue #98) -- exercised with a faked subprocess.

Nothing here runs ``nvidia-smi``; the module's whole shape exists so that it
cannot be. The test that matters most is the flush one: the instrument's job
is to have written down what the card was doing *up to the moment the host
wedged*, and a buffered writer would lose exactly that.
"""

from __future__ import annotations

import csv
import logging

import pytest

from music_video_maker import vramsample
from music_video_maker.vramsample import NVIDIA_SMI_ARGV, build_parser, main, sample_vram


class FakeClock:
    """Monotonic seconds advanced only by the sleeper, so a duration-bounded
    run terminates deterministically with no real time passing."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        body = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(body))


def test_it_records_every_reading_with_a_timestamp_and_reports_the_peak(tmp_path):
    clock = FakeClock()
    readings = iter(["500\n", "19995\n", "21000\n", "800\n"])
    out = tmp_path / "peak.csv"

    result = sample_vram(
        out,
        interval_seconds=0.5,
        duration_seconds=1.5,
        runner=lambda argv: next(readings),
        clock=clock,
        sleeper=clock.sleep,
    )

    rows = _rows(out)
    assert [row["used_mib"] for row in rows] == ["500", "19995", "21000", "800"]
    assert all(row["timestamp_utc"] for row in rows)
    assert result.peak_mib == 21000
    assert result.peak_at_elapsed == pytest.approx(1.0)
    assert result.failures == 0
    assert "peak 21000 MiB" in result.describe()


def test_it_runs_the_exact_query_the_runbook_names(tmp_path):
    seen: list[tuple[str, ...]] = []

    def runner(argv):
        seen.append(tuple(argv))
        return "1\n"

    clock = FakeClock()
    sample_vram(
        tmp_path / "o.csv",
        duration_seconds=0.0,
        runner=runner,
        clock=clock,
        sleeper=clock.sleep,
    )
    assert seen == [NVIDIA_SMI_ARGV]


def test_every_reading_is_flushed_as_it_is_taken(tmp_path):
    """A wedge needs a power cycle, so whatever is on disk at that instant is
    the entire record. Asserted by reading the file from inside the runner."""
    out = tmp_path / "peak.csv"
    seen_midway: list[int] = []
    clock = FakeClock()

    def runner(argv):
        seen_midway.append(len(_rows(out)))
        return "123\n"

    sample_vram(
        out,
        interval_seconds=0.5,
        duration_seconds=1.0,
        runner=runner,
        clock=clock,
        sleeper=clock.sleep,
    )
    # The third call must already see the first two rows on disk.
    assert seen_midway == [0, 1, 2]


def test_a_failing_reading_is_recorded_as_a_gap_and_sampling_continues(tmp_path, caplog):
    clock = FakeClock()
    def boom():
        raise OSError("boom")

    outcomes = iter([lambda: "100\n", boom, lambda: "200\n"])
    out = tmp_path / "peak.csv"

    with caplog.at_level(logging.WARNING, logger="music_video_maker.vramsample"):
        result = sample_vram(
            out,
            interval_seconds=0.5,
            duration_seconds=1.0,
            runner=lambda argv: next(outcomes)(),
            clock=clock,
            sleeper=clock.sleep,
        )

    rows = _rows(out)
    assert [row["used_mib"] for row in rows] == ["100", "", "200"]
    assert "OSError: boom" in rows[1]["raw"]
    assert result.failures == 1
    assert result.peak_mib == 200
    assert "recording it as a gap" in caplog.text


def test_an_unparseable_line_counts_as_a_failure_but_keeps_the_raw_text(tmp_path):
    clock = FakeClock()
    out = tmp_path / "peak.csv"
    result = sample_vram(
        out,
        duration_seconds=0.0,
        runner=lambda argv: "[N/A]\n",
        clock=clock,
        sleeper=clock.sleep,
    )
    rows = _rows(out)
    assert rows[0]["used_mib"] == ""
    assert rows[0]["raw"] == "[N/A]"
    assert result.failures == 1
    assert result.peak_mib is None
    assert "no usable reading at all" in result.describe()


def test_a_second_gpus_line_is_selected_by_index(tmp_path):
    clock = FakeClock()
    out = tmp_path / "peak.csv"
    result = sample_vram(
        out,
        duration_seconds=0.0,
        gpu_index=1,
        runner=lambda argv: "500\n7000\n",
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.peak_mib == 7000


def test_an_absent_gpu_index_is_a_gap_not_a_crash(tmp_path):
    clock = FakeClock()
    result = sample_vram(
        tmp_path / "o.csv",
        duration_seconds=0.0,
        gpu_index=3,
        runner=lambda argv: "500\n",
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.peak_mib is None and result.failures == 1


def test_ctrl_c_stops_sampling_and_still_reports_what_it_has(tmp_path, caplog):
    clock = FakeClock()
    calls = {"n": 0}

    def runner(argv):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return "42\n"

    with caplog.at_level(logging.INFO, logger="music_video_maker.vramsample"):
        result = sample_vram(
            tmp_path / "o.csv",
            interval_seconds=0.5,
            duration_seconds=None,
            runner=runner,
            clock=clock,
            sleeper=clock.sleep,
        )
    assert result.samples == 2
    assert result.peak_mib == 42
    assert "Interrupted after 2 sample(s)" in caplog.text


def test_the_file_opens_with_its_own_provenance(tmp_path):
    """Issue #93: a filename is not provenance. The label, the exact query and
    the start time live inside the file."""
    clock = FakeClock()
    out = tmp_path / "o.csv"
    sample_vram(
        out,
        duration_seconds=0.0,
        label="277-frame proof",
        runner=lambda argv: "1\n",
        clock=clock,
        sleeper=clock.sleep,
    )
    header = [line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    text = "\n".join(header)
    assert "'277-frame proof'" in text
    assert "--query-gpu=memory.used" in text
    assert "started_utc=" in text
    assert "WHOLE CARD" in text


def test_it_creates_the_output_directory(tmp_path):
    clock = FakeClock()
    out = tmp_path / "nested" / "dir" / "o.csv"
    sample_vram(
        out, duration_seconds=0.0, runner=lambda argv: "1\n", clock=clock, sleeper=clock.sleep
    )
    assert out.exists()


def test_main_wires_the_parser_through_to_a_real_sample_run(tmp_path, monkeypatch):
    captured = {}

    def fake_sample(out_file, **kwargs):
        captured["out"] = out_file
        captured.update(kwargs)
        return vramsample.SampleRun(
            samples=3, peak_mib=1, peak_at_elapsed=0.0, failures=0, out_file=tmp_path / "o.csv"
        )

    monkeypatch.setattr(vramsample, "sample_vram", fake_sample)
    assert main(["--out", str(tmp_path / "o.csv"), "--duration", "2", "--label", "x"]) == 0
    assert captured["duration_seconds"] == 2.0
    assert captured["label"] == "x"
    assert captured["gpu_index"] == 0


def test_main_reports_failure_when_it_sampled_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vramsample,
        "sample_vram",
        lambda out_file, **kwargs: vramsample.SampleRun(0, None, None, 0, tmp_path / "o.csv"),
    )
    assert main(["--out", str(tmp_path / "o.csv")]) == 1


def test_the_parser_requires_an_output_path():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
