"""Tests for music_video_maker.syncscore.

Fully offline: SyncNet, ffmpeg and music-video-maker are all played by an
injected fake runner. Nothing here imports torch or reads a real video.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from music_video_maker import syncscore
from music_video_maker.syncscore import (
    ChunkSync,
    SyncNetError,
    SyncNetInstall,
    Track,
    auto_reseed,
    discover_pairs,
    judge,
    parse_syncnet_output,
    reseed_command,
    score_dir,
    write_report,
)

SYNCNET_OUT = (
    "INFO Model loaded\n"
    "INFO AV offset: \t{off}\nINFO Min dist: \t7.644\nINFO Confidence: \t{conf}\n"
)


def make_install(tmp_path: Path) -> Path:
    root = tmp_path / "syncnet_python"
    for rel in ("run_pipeline.py", "run_syncnet.py", "data/syncnet_v2.model",
                "detectors/s3fd/weights/sfd_face.pth"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


def make_chunks(tmp_path: Path, ids, *, stems=True) -> Path:
    d = tmp_path / "chunks"
    d.mkdir()
    for cid in ids:
        (d / f"chunk_{cid:04d}.mp4").write_bytes(b"g0")
        if stems:
            (d / f"chunk_{cid:03d}.wav").write_bytes(b"wav")
    return d


class FakeWorld:
    """Plays ffmpeg, SyncNet and music-video-maker.

    ``scores[(chunk_id, generation)] = (offset, confidence)`` or None (no face).
    The chunk's current generation is whatever was last rendered into its mp4.
    """

    def __init__(self, chunks_dir: Path, scores: dict) -> None:
        self.chunks_dir = chunks_dir
        self.scores = scores
        self.calls: list[list[str]] = []
        self.muxed: list[tuple[str, str]] = []

    def generation(self, cid: int) -> int:
        return int((self.chunks_dir / f"chunk_{cid:04d}.mp4").read_bytes()[1:])

    def __call__(self, argv, cwd):
        argv = list(argv)
        self.calls.append(argv)
        if argv[0] == "ffmpeg":
            first = argv.index("-i")
            self.muxed.append((argv[first + 1], argv[argv.index("-i", first + 1) + 1]))
            Path(argv[-1]).write_bytes(b"clip")
            return 0, ""
        if argv[1] == "run_pipeline.py":
            return 0, ""
        if argv[1] == "run_syncnet.py":
            cid = int(argv[argv.index("--reference") + 1][len("chunk"):])
            score = self.scores.get((cid, self.generation(cid)), (0, 7.0))
            if score is None:
                return 0, "INFO no face tracks\n"
            return 0, SYNCNET_OUT.format(off=score[0], conf=score[1])
        # music-video-maker
        if "--reseed" in argv:
            ids, gen = argv[argv.index("--reseed") + 1], int(argv[argv.index(
                "--reseed-generation") + 1])
        else:
            ids, gen = argv[argv.index("--only-chunks") + 1], 0
        for cid in map(int, ids.split(",")):
            (self.chunks_dir / f"chunk_{cid:04d}.mp4").write_bytes(f"g{gen}".encode())
        return 0, ""


# --- install ------------------------------------------------------------------


def test_locate_refuses_an_incomplete_checkout(tmp_path):
    root = make_install(tmp_path)
    (root / "data" / "syncnet_v2.model").unlink()
    with pytest.raises(SyncNetError, match="syncnet_v2.model"):
        SyncNetInstall.locate(root)


def test_locate_records_the_weights_hash(tmp_path):
    install = SyncNetInstall.locate(make_install(tmp_path), Path("/venv/bin/python"))
    assert install.weights_sha256 == (
        "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")
    assert install.python == Path("/venv/bin/python")


# --- pairing and parsing --------------------------------------------------------


def test_pairs_match_on_numeric_id_not_padding(tmp_path):
    d = make_chunks(tmp_path, [10, 2])
    (d / "chunk_0011.mp4").write_bytes(b"g0")  # no stem
    (d / "chunk_final.mp4").write_bytes(b"g0")  # no id
    pairs = discover_pairs(d)
    assert [(c, v.name, s.name if s else None) for c, v, s in pairs] == [
        (2, "chunk_0002.mp4", "chunk_002.wav"),
        (10, "chunk_0010.mp4", "chunk_010.wav"),
        (11, "chunk_0011.mp4", None),
    ]


def test_parse_every_track():
    text = SYNCNET_OUT.format(off=-1, conf=6.5) + "AV offset: 4\nMin dist: 9.1\nConfidence: 1.2\n"
    assert parse_syncnet_output(text) == [Track(-1, 6.5, 7.644), Track(4, 1.2, 9.1)]
    assert parse_syncnet_output("nothing") == []


def test_judge_rule():
    assert judge(None) == "no_face"
    assert judge(Track(0, 6.0, 7.0)) == "ok"
    assert judge(Track(-2, 3.0, 7.0)) == "ok"
    assert judge(Track(3, 9.0, 7.0)) == "flagged"
    assert judge(Track(-3, 9.0, 7.0)) == "flagged"
    assert judge(Track(0, 2.99, 7.0)) == "flagged"


# --- scoring --------------------------------------------------------------------


def test_scores_against_the_staged_audio_never_the_mp4_track(tmp_path):
    d = make_chunks(tmp_path, [0, 1])
    world = FakeWorld(d, {(1, 0): (5, 6.0)})
    rows = score_dir(d, SyncNetInstall.locate(make_install(tmp_path)), runner=world)
    assert [r.verdict for r in rows] == ["ok", "flagged"]
    assert world.muxed == [(str((d / "chunk_0000.mp4").resolve()),
                            str((d / "chunk_000.wav").resolve())),
                           (str((d / "chunk_0001.mp4").resolve()),
                            str((d / "chunk_001.wav").resolve()))]
    ffmpeg = next(c for c in world.calls if c[0] == "ffmpeg")
    assert ffmpeg[ffmpeg.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in ffmpeg


def test_best_track_is_the_most_confident(tmp_path):
    d = make_chunks(tmp_path, [0])
    install = SyncNetInstall.locate(make_install(tmp_path))

    def runner(argv, cwd):
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"c")
            return 0, ""
        if argv[1] == "run_syncnet.py":
            return 0, (SYNCNET_OUT.format(off=7, conf=1.1) + SYNCNET_OUT.format(off=0, conf=8.0))
        return 0, ""

    (row,) = score_dir(d, install, runner=runner)
    assert row.best == Track(0, 8.0, 7.644) and row.verdict == "ok"


def test_no_face_and_no_stem_are_reported_not_scored(tmp_path):
    d = make_chunks(tmp_path, [0])
    (d / "chunk_0001.mp4").write_bytes(b"g0")
    world = FakeWorld(d, {(0, 0): None})
    rows = score_dir(d, SyncNetInstall.locate(make_install(tmp_path)), runner=world)
    assert [r.verdict for r in rows] == ["no_face", "no_stem"]
    assert not any(r.flagged for r in rows)


def test_a_failing_chunk_is_skipped_not_fatal(tmp_path, caplog):
    d = make_chunks(tmp_path, [0, 1])

    def runner(argv, cwd):
        if argv[0] == "ffmpeg":
            if "chunk_0000.mp4" in argv[argv.index("-i") + 1]:
                return 1, "boom"
            Path(argv[-1]).write_bytes(b"c")
            return 0, ""
        if argv[1] == "run_syncnet.py":
            return 0, SYNCNET_OUT.format(off=0, conf=5.0)
        return 0, ""

    rows = score_dir(d, SyncNetInstall.locate(make_install(tmp_path)), runner=runner)
    assert [r.chunk_id for r in rows] == [1]
    assert "chunk 0 could not be scored" in caplog.text


@pytest.mark.parametrize("script", ["run_pipeline.py", "run_syncnet.py"])
def test_syncnet_failure_raises_for_that_chunk(tmp_path, script):
    d = make_chunks(tmp_path, [0])
    install = SyncNetInstall.locate(make_install(tmp_path))

    def runner(argv, cwd):
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"c")
            return 0, ""
        return (1, "trace") if argv[1] == script else (0, "")

    with pytest.raises(SyncNetError, match=script):
        syncscore.score_chunk(0, d / "chunk_0000.mp4", d / "chunk_000.wav", install,
                              runner=runner, work_dir=tmp_path)


def test_report_header_carries_provenance(tmp_path):
    d = make_chunks(tmp_path, [0])
    install = SyncNetInstall.locate(make_install(tmp_path))
    rows = score_dir(d, install, runner=FakeWorld(d, {}))
    buf = io.StringIO()
    write_report(rows, buf, input_dir=d, install=install)
    lines = buf.getvalue().splitlines()
    assert lines[1] == f"# input_dir={d.resolve()}"
    assert install.weights_sha256 in lines[2]
    assert "offset_limit=3 confidence_floor=3.0" in lines[3]
    table = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    assert table[0]["verdict"] == "ok"
    assert table[0]["video_path"] == str((d / "chunk_0000.mp4").resolve())
    assert table[0]["stem_size_bytes"] == "3"


# --- auto re-roll ---------------------------------------------------------------


def test_reseed_command():
    assert reseed_command("mvm", Path("run.toml"), [3, 5], 2, "body") == [
        "mvm", "--config", "run.toml", "--reseed", "3,5", "--reseed-generation", "2",
        "--timeline", "body"]


def test_auto_reseed_rerolls_until_each_passes(tmp_path):
    d = make_chunks(tmp_path, [0, 1, 2])
    world = FakeWorld(d, {(1, 0): (4, 6.0), (2, 0): (0, 1.0), (2, 1): (0, 2.0)})
    install = SyncNetInstall.locate(make_install(tmp_path))
    rows = score_dir(d, install, runner=world)
    final = auto_reseed(d, install, rows, config=tmp_path / "run.toml", generations=3,
                        mvm="mvm", runner=world)
    mvm_calls = [c for c in world.calls if c[0] == "mvm"]
    assert [c[c.index("--reseed") + 1] for c in mvm_calls] == ["1,2", "2"]
    assert [r.verdict for r in final] == ["ok", "ok", "ok"]
    assert world.generation(2) == 2


def test_auto_reseed_rerenders_the_best_generation_when_none_pass(tmp_path):
    d = make_chunks(tmp_path, [0])
    world = FakeWorld(d, {(0, 0): (10, 7.5), (0, 1): (4, 2.0), (0, 2): (9, 7.1)})
    install = SyncNetInstall.locate(make_install(tmp_path))
    rows = score_dir(d, install, runner=world)
    (final,) = auto_reseed(d, install, rows, config=tmp_path / "run.toml", generations=2,
                           mvm="mvm", runner=world)
    # The file on disk is generation 1 again -- rendered, not copied -- and the row
    # describes that file.
    assert world.generation(0) == 1
    assert final.best.offset == 4 and final.flagged  # nearest miss, not most confident
    assert world.calls[-4][:6] == ["mvm", "--config", str(tmp_path / "run.toml"), "--reseed",
                                   "0", "--reseed-generation"]


def test_auto_reseed_restores_generation_zero_with_only_chunks(tmp_path):
    d = make_chunks(tmp_path, [0])
    world = FakeWorld(d, {(0, 0): (4, 5.0), (0, 1): (6, 1.0)})
    install = SyncNetInstall.locate(make_install(tmp_path))
    rows = score_dir(d, install, runner=world)
    auto_reseed(d, install, rows, config=tmp_path / "run.toml", generations=1,
                timeline="body", mvm="mvm", runner=world)
    restore = [c for c in world.calls if c[0] == "mvm"][-1]
    assert restore == ["mvm", "--config", str(tmp_path / "run.toml"), "--only-chunks", "0",
                       "--timeline", "body"]
    assert world.generation(0) == 0


def test_auto_reseed_raises_when_the_render_fails(tmp_path):
    d = make_chunks(tmp_path, [0])
    install = SyncNetInstall.locate(make_install(tmp_path))
    world = FakeWorld(d, {(0, 0): (4, 1.0)})
    rows = score_dir(d, install, runner=world)

    def failing(argv, cwd):
        return (1, "render died") if argv[0] == "mvm" else world(argv, cwd)

    with pytest.raises(SyncNetError, match="generation 1"):
        auto_reseed(d, install, rows, config=tmp_path / "run.toml", generations=1,
                    mvm="mvm", runner=failing)


def test_auto_reseed_leaves_no_face_chunks_alone(tmp_path):
    d = make_chunks(tmp_path, [0])
    world = FakeWorld(d, {(0, 0): None})
    install = SyncNetInstall.locate(make_install(tmp_path))
    rows = score_dir(d, install, runner=world)
    assert auto_reseed(d, install, rows, config=tmp_path / "r.toml", generations=2,
                       mvm="mvm", runner=world) == rows
    assert not [c for c in world.calls if c[0] == "mvm"]


# --- CLI ------------------------------------------------------------------------


def test_cli_writes_report_and_exits_1_when_anything_is_flagged(tmp_path):
    d = make_chunks(tmp_path, [0, 1])
    root = make_install(tmp_path)
    out = tmp_path / "sync.csv"
    world = FakeWorld(d, {(1, 0): (5, 6.0)})
    assert syncscore.main([str(d), "--syncnet", str(root), "--out", str(out)],
                          runner=world) == 1
    assert "flagged" in out.read_text()
    assert syncscore.main([str(d), "--syncnet", str(root), "--only", "0"], runner=world) == 0


def test_cli_auto_reseed(tmp_path, capsys):
    d = make_chunks(tmp_path, [0])
    root = make_install(tmp_path)
    world = FakeWorld(d, {(0, 0): (5, 6.0)})
    code = syncscore.main([str(d), "--syncnet", str(root), "--auto-reseed", "2", "--config",
                           str(tmp_path / "run.toml"), "--music-video-maker", "mvm"],
                          runner=world)
    assert code == 0
    assert ",ok," in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "code"),
    [(["--auto-reseed", "1"], 2), ([], 1)],
)
def test_cli_refusals(tmp_path, args, code):
    root = make_install(tmp_path)
    missing = tmp_path / "nope"
    assert syncscore.main([str(missing), "--syncnet", str(root), *args],
                          runner=FakeWorld(tmp_path, {})) == code


def test_cli_refuses_incomplete_install_and_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    root = make_install(tmp_path)
    assert syncscore.main([str(d), "--syncnet", str(root)], runner=FakeWorld(d, {})) == 1
    assert syncscore.main([str(d), "--syncnet", str(tmp_path / "x")],
                          runner=FakeWorld(d, {})) == 1


def test_chunk_sync_flagged_property():
    row = ChunkSync(0, "flagged", (), "v", 1, "t", "s", 1, "t")
    assert row.flagged and ChunkSync(0, "ok", (), "v", 1, "t", "s", 1, "t").best is None
