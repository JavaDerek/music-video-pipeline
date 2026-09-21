"""Stage 5 across more than one timeline, and the measured seam (issue #66).

Never invokes a real ffmpeg: the runner is a small in-memory model of the
three things assembly asks ffmpeg for -- a duration, a stream signature, and
an output file -- so every assertion here is about what the pipeline *does
with* those answers, which is the part that can be wrong.

The model is deliberately honest about one thing: ``apad=whole_dur=X``
produces a file of exactly X seconds, so the padded-audio probe reads back
what the pad asked for. A fake that returned a constant would make the seam
assertion vacuous, which is the failure mode of every duration check this
issue has already found once (``tests/test_cli.py``'s ``FakeFfmpegRunner``
docstring records the same trap from the other direction).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from music_video_maker.assembly import (
    DEFAULT_OUTPUT_FILENAME,
    ConcatSignatureMismatchError,
    DurationMismatchError,
    MissingChunksError,
    SeamMismatchError,
    TimelineAssembly,
    assemble_timelines,
    build_audio_concat_args,
    build_audio_pad_args,
    compare_video_signatures,
    probe_audio_layout,
    probe_video_signature,
)
from music_video_maker.contracts import AudioChunk, ChunkResult, ChunkStatus
from music_video_maker.timelines import SeamOverrunError

SIGNATURE = {
    "codec_name": "h264",
    "profile": "High",
    "level": "30",
    "width": "864",
    "height": "480",
    "pix_fmt": "yuv420p",
    "field_order": "progressive",
    "r_frame_rate": "24/1",
    "time_base": "1/12288",
}
"""The one signature all 80 chunks of the real ``chunks_v12`` render share
(``docs/design-prologue-timelines.md``). Used verbatim so the fixture is the
measured shape rather than an invented one."""


def _chunk(chunk_id: int, timeline: str | None, start: float, end: float) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"/tmp/{timeline}_{chunk_id}.wav"),
        start=start,
        end=end,
        text="",
        timeline=timeline,
    )


class FakeFfmpeg:
    """An in-memory model of ffmpeg/ffprobe: durations, signatures, files.

    ``durations`` maps a path to what ``format=duration`` reports. Outputs
    get a duration the same way real ffmpeg would give them one: a concat is
    the sum of its inputs, an ``apad`` is its ``whole_dur``, and a mux keeps
    its video input's length.
    """

    def __init__(self, durations: dict[Path, float], *, signatures=None):
        self.durations = dict(durations)
        self.signatures = signatures or {}
        self.calls: list[list[str]] = []

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = [str(a) for a in args]
        self.calls.append(args)
        if args[0] == "ffprobe":
            return self._probe(args)
        return self._run(args)

    def _probe(self, args) -> subprocess.CompletedProcess:
        target = Path(args[-1])
        if any("format=duration" in a for a in args):
            seconds = self.durations.get(target)
            if seconds is None:
                return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"no such file")
            return subprocess.CompletedProcess(args, 0, stdout=f"{seconds}\n".encode())
        if any("codec_name" in a for a in args):
            signature = self.signatures.get(target, SIGNATURE)
            body = "\n".join(f"{k}={v}" for k, v in signature.items())
            return subprocess.CompletedProcess(args, 0, stdout=body.encode())
        if any("sample_rate" in a for a in args):
            return subprocess.CompletedProcess(
                args, 0, stdout=b"sample_rate=44100\nchannels=2\n"
            )
        raise AssertionError(f"unexpected ffprobe call: {args}")

    def _run(self, args) -> subprocess.CompletedProcess:
        destination = Path(args[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake")

        if "-f" in args and args[args.index("-f") + 1] == "concat":
            listing = Path(args[args.index("-i") + 1])
            entries = [
                Path(line[len("file '") : -1])
                for line in listing.read_text().splitlines()
                if line.startswith("file '")
            ]
            self.durations[destination] = sum(self.durations[e] for e in entries)
        elif "-af" in args:
            whole_dur = re.search(r"whole_dur=([\d.]+)", args[args.index("-af") + 1])
            assert whole_dur is not None
            source = Path(args[args.index("-i") + 1])
            self.durations[destination] = max(
                float(whole_dur.group(1)), self.durations[source]
            )
        elif "-filter_complex" in args:
            inputs = [Path(args[i + 1]) for i, a in enumerate(args) if a == "-i"]
            self.durations[destination] = sum(self.durations[p] for p in inputs)
        else:  # the mux: video length governs
            self.durations[destination] = self.durations[Path(args[args.index("-i") + 1])]
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")


class Fixture:
    """A prologue of two chunks in front of a song of three."""

    def __init__(self, tmp_path: Path, *, silent: bool = False):
        self.tmp_path = tmp_path
        self.output_dir = tmp_path / "final"
        durations: dict[Path, float] = {}

        def clips(timeline: str, count: int, seconds: float) -> list[ChunkResult]:
            results = []
            for cid in range(count):
                path = tmp_path / timeline / f"chunk_{cid:04d}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake")
                durations[path] = seconds
                results.append(
                    ChunkResult(chunk_id=cid, status=ChunkStatus.RENDERED, video_file=path)
                )
            return results

        self.prologue_audio = tmp_path / "prologue.wav"
        self.master_audio = tmp_path / "master.wav"
        for path in (self.prologue_audio, self.master_audio):
            path.write_bytes(b"fake")
        # The prologue's chunk timeline overshoots its own recording by 1.0s
        # and the song's by 1.837s -- the measured "Deathless" figure.
        durations[self.prologue_audio] = 11.0
        durations[self.master_audio] = 16.163

        prologue_results = clips("prologue", 2, 6.0)
        song_results = clips("song", 3, 6.0)
        self.runner = FakeFfmpeg(durations)

        self.prologue = TimelineAssembly(
            name="prologue",
            chunks=[_chunk(i, "prologue", i * 6.0, (i + 1) * 6.0) for i in range(2)],
            results={r.chunk_id: r for r in prologue_results},
            audio=None if silent else self.prologue_audio,
            fingerprint_name="prologue",
        )
        self.song = TimelineAssembly(
            name="song",
            chunks=[_chunk(i, None, i * 6.0, (i + 1) * 6.0) for i in range(3)],
            results={r.chunk_id: r for r in song_results},
            audio=None if silent else self.master_audio,
            fingerprint_name=None,
        )

    def assemble(self, **kwargs):
        return assemble_timelines(
            [self.prologue, self.song],
            self.output_dir,
            runner=self.runner,
            check_luminance=False,
            check_scene_cuts=False,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_two_timelines_are_laid_end_to_end_from_measured_durations(tmp_path: Path):
    result = Fixture(tmp_path).assemble()

    prologue, song = result.placements
    assert prologue.offset_seconds == 0.0
    assert prologue.video_seconds == pytest.approx(12.0)
    assert song.offset_seconds == pytest.approx(12.0)
    assert song.video_seconds == pytest.approx(18.0)
    assert result.measured_duration == pytest.approx(30.0)
    assert result.output_video == tmp_path / "final" / DEFAULT_OUTPUT_FILENAME
    assert result.output_video.is_file()
    assert result.has_audio


def test_each_timelines_audio_is_padded_up_to_its_own_video_not_the_others(tmp_path: Path):
    """The seam reconciliation: pad audio, never trim picture. The prologue
    is 1.0s short of its chunk timeline and the song 1.837s -- each closes
    its own gap, so the song still starts exactly where the prologue's last
    frame ends."""
    result = Fixture(tmp_path).assemble()
    prologue, song = result.placements
    assert prologue.pad_seconds == pytest.approx(1.0)
    assert song.pad_seconds == pytest.approx(1.837)
    # The seam is exact: the song's first frame is the prologue's last + 1.
    assert song.offset_seconds == pytest.approx(prologue.end_seconds)


def test_the_pad_is_measured_back_with_ffprobe_not_just_computed(tmp_path: Path):
    """Computing a pad is not evidence it happened. Every padded file is
    probed and compared against the video it has to match -- the check that
    would have caught a silently-failed pad."""
    fixture = Fixture(tmp_path)
    fixture.assemble()

    pad_calls = [c for c in fixture.runner.calls if "-af" in c]
    assert len(pad_calls) == 2
    padded_paths = [Path(c[-1]) for c in pad_calls]
    probe_calls = [
        Path(c[-1])
        for c in fixture.runner.calls
        if c[0] == "ffprobe" and any("format=duration" in a for a in c)
    ]
    for path in padded_paths:
        assert path in probe_calls, f"{path.name} was padded but never measured back"


def test_a_pad_that_does_not_take_raises_rather_than_desyncing_everything_after_it(
    tmp_path: Path,
):
    """The failure this check exists for: at a seam, a short pad pushes the
    entire song out of sync while both halves still look perfect alone, and
    there is no ``-shortest`` in the middle of a file to save it."""
    fixture = Fixture(tmp_path)
    original = fixture.runner._run

    def broken_pad(args):
        result = original(args)
        if "-af" in args:  # a pad that silently did nothing
            fixture.runner.durations[Path(args[-1])] = 11.0
        return result

    fixture.runner._run = broken_pad  # type: ignore[method-assign]
    with pytest.raises(SeamMismatchError) as excinfo:
        fixture.assemble()
    assert excinfo.value.name == "prologue"
    assert excinfo.value.measured == pytest.approx(11.0)


def test_audio_longer_than_its_video_is_refused_before_any_pad_runs(tmp_path: Path):
    fixture = Fixture(tmp_path)
    fixture.runner.durations[fixture.prologue_audio] = 99.0
    with pytest.raises(SeamOverrunError, match="prologue"):
        fixture.assemble()
    assert not any("-af" in c for c in fixture.runner.calls)


def test_the_finished_file_is_checked_against_the_sum_of_measured_timelines(tmp_path: Path):
    """``-shortest`` is what made the single-timeline overshoot invisible.
    With a seam it cannot help, so the finished file is measured."""
    fixture = Fixture(tmp_path)
    original = fixture.runner._run

    def short_mux(args):
        result = original(args)
        if "-map" in args and "0:v" in args:
            fixture.runner.durations[Path(args[-1])] = 25.0
        return result

    fixture.runner._run = short_mux  # type: ignore[method-assign]
    with pytest.raises(DurationMismatchError) as excinfo:
        fixture.assemble()
    assert excinfo.value.measured_duration == pytest.approx(25.0)
    assert excinfo.value.expected_duration == pytest.approx(30.0)
    # Raised AFTER the file is written, deliberately -- it stays on disk.
    assert (tmp_path / "final" / DEFAULT_OUTPUT_FILENAME).is_file()


# --------------------------------------------------------------------------- #
# The concat signature
# --------------------------------------------------------------------------- #


def test_timelines_whose_chunks_disagree_on_the_copy_signature_are_refused(tmp_path: Path):
    """The concat demuxer does not fail on mismatched inputs -- it produces a
    file that plays wrong, and the first symptom is watching it."""
    fixture = Fixture(tmp_path)
    odd = fixture.prologue.results[0].video_file
    fixture.runner.signatures = {odd: {**SIGNATURE, "width": "1344", "height": "768"}}

    with pytest.raises(ConcatSignatureMismatchError) as excinfo:
        fixture.assemble()
    assert set(excinfo.value.differing) == {"width", "height"}
    # Refused before anything was concatenated.
    assert not any("-f" in c and "concat" in c for c in fixture.runner.calls)


def test_the_agreed_signature_is_reported_as_evidence_the_check_ran(tmp_path: Path):
    result = Fixture(tmp_path).assemble()
    assert dict(result.video_signature) == SIGNATURE


def test_a_property_missing_from_every_timeline_is_not_a_mismatch():
    """An older ffmpeg that omits ``field_order`` must not be read as "these
    two differ" -- an absent key is absent from both."""
    reduced = {k: v for k, v in SIGNATURE.items() if k != "field_order"}
    assert compare_video_signatures({"a": reduced, "b": dict(reduced)}) == reduced


def test_comparing_no_signatures_at_all_is_not_an_error():
    assert compare_video_signatures({}) == {}


def test_probe_video_signature_parses_ffprobes_key_value_output(tmp_path: Path):
    path = tmp_path / "chunk.mp4"
    path.write_bytes(b"fake")
    runner = FakeFfmpeg({path: 5.0})
    assert probe_video_signature(path, runner) == SIGNATURE


def test_probe_audio_layout_reads_sample_rate_and_channels(tmp_path: Path):
    path = tmp_path / "master.wav"
    path.write_bytes(b"fake")
    runner = FakeFfmpeg({path: 5.0})
    assert probe_audio_layout(path, runner) == {"sample_rate": "44100", "channels": "2"}


def test_a_failing_probe_raises_with_the_command_and_stderr(tmp_path: Path):
    from music_video_maker.assembly import FfmpegError

    with pytest.raises(FfmpegError):
        probe_video_signature(tmp_path / "missing.mp4", _AlwaysFailing())


class _AlwaysFailing:
    def __call__(self, args) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(list(args), 1, stdout=b"", stderr=b"boom")


# --------------------------------------------------------------------------- #
# Arg shapes
# --------------------------------------------------------------------------- #


def test_pad_args_target_the_songs_own_layout(tmp_path: Path):
    args = build_audio_pad_args(
        tmp_path / "prologue.wav",
        tmp_path / "padded.wav",
        pad_to_seconds=12.5,
        sample_rate="44100",
        channels="2",
    )
    assert "apad=whole_dur=12.500000" in args
    assert args[args.index("-ar") + 1] == "44100"
    assert args[args.index("-ac") + 1] == "2"
    assert args[args.index("-c:a") + 1] == "pcm_s16le"


def test_pad_args_omit_a_layout_ffprobe_could_not_report(tmp_path: Path):
    args = build_audio_pad_args(
        tmp_path / "a.wav", tmp_path / "b.wav", pad_to_seconds=1.0,
        sample_rate=None, channels=None,
    )
    assert "-ar" not in args and "-ac" not in args


def test_audio_concat_uses_the_filter_not_the_demuxer(tmp_path: Path):
    """The demuxer copies packets and would splice two differently-encoded
    files into something no decoder can read."""
    args = build_audio_concat_args(
        [tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "out.wav"
    )
    assert "-f" not in args
    assert "[0:a][1:a]concat=n=2:v=0:a=1[seam]" in args
    assert args[args.index("-map") + 1] == "[seam]"


def test_generated_audio_is_still_stripped_from_every_timelines_concat(tmp_path: Path):
    """The invariant issue #22 was careful not to move: H3's own per-chunk
    audio never survives, on either timeline."""
    fixture = Fixture(tmp_path)
    fixture.assemble()
    concat_calls = [c for c in fixture.runner.calls if "-f" in c and "concat" in c]
    assert concat_calls
    assert all("-an" in c and "copy" in c for c in concat_calls)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_missing_chunk_names_the_timeline_it_is_missing_from(tmp_path: Path):
    """Chunk ids are a separate space per timeline, so "chunk 1 is missing"
    is ambiguous the moment there are two."""
    fixture = Fixture(tmp_path)
    broken = dict(fixture.prologue.results)
    broken[1] = ChunkResult(chunk_id=1, status=ChunkStatus.DEAD_LETTERED)
    fixture.prologue = TimelineAssembly(
        name="prologue",
        chunks=fixture.prologue.chunks,
        results=broken,
        audio=fixture.prologue_audio,
        fingerprint_name="prologue",
    )
    with pytest.raises(MissingChunksError) as excinfo:
        fixture.assemble()
    assert excinfo.value.timeline == "prologue"
    assert "prologue" in str(excinfo.value)
    # Refused before any subprocess ran at all.
    assert fixture.runner.calls == []


def test_some_timelines_with_audio_and_some_without_is_refused(tmp_path: Path):
    fixture = Fixture(tmp_path)
    fixture.song = TimelineAssembly(
        name="song",
        chunks=fixture.song.chunks,
        results=fixture.song.results,
        audio=None,
        fingerprint_name=None,
    )
    with pytest.raises(ValueError, match="every timeline must have audio"):
        fixture.assemble()


def test_no_timelines_at_all_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one timeline"):
        assemble_timelines([], tmp_path)


# --------------------------------------------------------------------------- #
# The silent (issue #22) path
# --------------------------------------------------------------------------- #


def test_the_silent_path_writes_straight_to_the_deliverable_with_no_mux(tmp_path: Path):
    fixture = Fixture(tmp_path, silent=True)
    result = fixture.assemble()
    assert result.has_audio is False
    assert result.mux_args == ()
    assert result.output_video == result.intermediate_video
    assert not any("-af" in c for c in fixture.runner.calls)
    # Still measured against the timelines it was cut to -- the whole point
    # of issue #22 is that there is no -shortest to hide behind.
    assert result.measured_duration == pytest.approx(30.0)


def test_the_silent_path_still_reports_where_each_timeline_starts(tmp_path: Path):
    result = Fixture(tmp_path, silent=True).assemble()
    assert [p.offset_seconds for p in result.placements] == [0.0, 12.0]
    assert all(p.pad_seconds == 0.0 for p in result.placements)
