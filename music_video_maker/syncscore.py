"""Lip-sync scoring of rendered chunks with SyncNet.

H3 generates its own soundtrack, so a rendered chunk's embedded audio is not
the audio it was conditioned on. Scoring a chunk against its own mp4 audio
measures the model against itself. This instrument pairs each
``chunk_NNNN.mp4`` with the chunk's **staged conditioning audio**
(``chunk_NNN.wav`` in the same directory -- exactly what was uploaded to
ComfyUI for that chunk), muxes the two, and runs SyncNet (Chung & Zisserman,
"Out of time", 2016) over the result.

SyncNet is not a pip package: it is a checkout of ``syncnet_python`` plus two
weight files (``data/syncnet_v2.model``, ``detectors/s3fd/weights/
sfd_face.pth``), usually in its own virtualenv. This module drives it as a
subprocess, through an injected runner, so it needs nothing beyond the
standard library to import or test.

The decision rule -- flag when ``|AV offset| >= 3`` frames or confidence
``< 3.0`` -- was calibrated on 19 hand-labelled close-up speech chunks (6 bad,
13 good; it matched all 19). It is a rule for **one frontal face speaking**.
On sung, multi-performer, wide shots it is uncalibrated, and the report says
which chunks had no usable face track rather than scoring them. See
``docs/syncscore.md`` for the calibration and the music-video caveats.

Provenance follows ``facescan`` (#93): every row records the resolved path,
size and mtime of both files it read; the header records the input
directory, the SyncNet weights' sha256 and the thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)

OFFSET_LIMIT = 3
CONFIDENCE_FLOOR = 3.0
SYNCNET_WEIGHTS = Path("data") / "syncnet_v2.model"
FACE_WEIGHTS = Path("detectors") / "s3fd" / "weights" / "sfd_face.pth"

FIELDNAMES = (
    "chunk_id",
    "verdict",
    "offset",
    "confidence",
    "min_dist",
    "tracks",
    "video_path",
    "video_size_bytes",
    "video_mtime",
    "stem_path",
    "stem_size_bytes",
    "stem_mtime",
)

#: ``runner(argv, cwd) -> (returncode, combined stdout+stderr)``
Runner = Callable[[Sequence[str], Path | None], tuple[int, str]]


def run_subprocess(argv: Sequence[str], cwd: Path | None) -> tuple[int, str]:
    done = subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


class SyncNetError(RuntimeError):
    """The SyncNet install is unusable, or a chunk could not be scored at all."""


# --------------------------------------------------------------------------- #
# The SyncNet install
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SyncNetInstall:
    root: Path
    python: Path
    weights_sha256: str

    @classmethod
    def locate(cls, root: Path, python: Path | None = None) -> SyncNetInstall:
        """Validate a ``syncnet_python`` checkout; refuse loudly if incomplete."""
        root = root.expanduser().resolve()
        missing = [
            str(root / rel)
            for rel in (Path("run_pipeline.py"), Path("run_syncnet.py"), SYNCNET_WEIGHTS,
                        FACE_WEIGHTS)
            if not (root / rel).is_file()
        ]
        if missing:
            raise SyncNetError(
                "syncnet_python checkout is incomplete -- missing: " + ", ".join(missing)
            )
        return cls(
            root=root,
            python=(python or Path(sys.executable)).expanduser(),
            weights_sha256=_sha256(root / SYNCNET_WEIGHTS),
        )


# --------------------------------------------------------------------------- #
# Pairing and scoring
# --------------------------------------------------------------------------- #


def _chunk_id(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def discover_pairs(chunks_dir: Path) -> list[tuple[int, Path, Path | None]]:
    """``(chunk_id, video, stem)`` for every ``chunk_*.mp4``, sorted by id.

    The video is zero-padded to four digits and the staged audio to three
    (``chunk_0007.mp4`` / ``chunk_007.wav``); they are matched on the numeric
    id, never on the string. ``stem`` is None when no staged audio survives.
    """
    stems = {
        cid: p for p in chunks_dir.glob("chunk_*.wav") if (cid := _chunk_id(p)) is not None
    }
    pairs = []
    for video in chunks_dir.glob("chunk_*.mp4"):
        cid = _chunk_id(video)
        if cid is None:
            logger.warning("syncscore: %s has no chunk id in its name -- skipping", video)
            continue
        pairs.append((cid, video, stems.get(cid)))
    return sorted(pairs, key=lambda p: p[0])


@dataclass(frozen=True)
class Track:
    offset: int
    confidence: float
    min_dist: float


def parse_syncnet_output(text: str) -> list[Track]:
    """Every face track ``run_syncnet.py`` reported, in its order."""
    offsets = re.findall(r"AV offset:\s*(-?\d+)", text)
    dists = re.findall(r"Min dist:\s*([\d.]+)", text)
    confs = re.findall(r"Confidence:\s*(-?[\d.]+)", text)
    dists = dists + ["nan"] * (len(offsets) - len(dists))
    rows = zip(offsets, confs, dists, strict=False)
    return [Track(int(o), float(c), float(d)) for o, c, d in rows]


@dataclass(frozen=True)
class ChunkSync:
    chunk_id: int
    verdict: str  # "ok" | "flagged" | "no_face" | "no_stem"
    tracks: tuple[Track, ...]
    video_path: str
    video_size_bytes: int
    video_mtime: str
    stem_path: str
    stem_size_bytes: int | None
    stem_mtime: str

    @property
    def best(self) -> Track | None:
        """The most confident track -- the face the audio most plausibly belongs to."""
        return max(self.tracks, key=lambda t: t.confidence, default=None)

    @property
    def flagged(self) -> bool:
        return self.verdict == "flagged"


def judge(track: Track | None) -> str:
    if track is None:
        return "no_face"
    if abs(track.offset) >= OFFSET_LIMIT or track.confidence < CONFIDENCE_FLOOR:
        return "flagged"
    return "ok"


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def score_chunk(
    chunk_id: int,
    video: Path,
    stem: Path | None,
    install: SyncNetInstall,
    *,
    runner: Runner = run_subprocess,
    work_dir: Path,
) -> ChunkSync:
    """Mux ``video`` with ``stem`` and score it. Never reads the mp4's own audio."""
    video = video.resolve()
    stem_info: tuple[str, int | None, str] = ("", None, "")
    if stem is None:
        return ChunkSync(chunk_id, "no_stem", (), str(video), video.stat().st_size,
                         _mtime(video), *stem_info)
    stem = stem.resolve()
    stem_info = (str(stem), stem.stat().st_size, _mtime(stem))
    clip = work_dir / f"chunk_{chunk_id:04d}_stem.mp4"
    code, out = runner(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(stem),
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "libx264", "-crf", "17",
         "-c:a", "aac", str(clip)],
        None,
    )
    if code != 0:
        raise SyncNetError(f"chunk {chunk_id}: ffmpeg mux failed: {out.strip()[-300:]}")
    data_dir = work_dir / f"chunk_{chunk_id:04d}"
    args = ["--videofile", str(clip), "--reference", f"chunk{chunk_id:04d}",
            "--data_dir", str(data_dir)]
    code, out = runner([str(install.python), "run_pipeline.py", *args], install.root)
    if code != 0:
        raise SyncNetError(f"chunk {chunk_id}: run_pipeline.py failed: {out.strip()[-300:]}")
    code, out = runner([str(install.python), "run_syncnet.py", *args], install.root)
    if code != 0:
        raise SyncNetError(f"chunk {chunk_id}: run_syncnet.py failed: {out.strip()[-300:]}")
    tracks = tuple(parse_syncnet_output(out))
    best = max(tracks, key=lambda t: t.confidence, default=None)
    return ChunkSync(chunk_id, judge(best), tracks, str(video), video.stat().st_size,
                     _mtime(video), *stem_info)


def score_dir(
    chunks_dir: Path,
    install: SyncNetInstall,
    *,
    only: set[int] | None = None,
    runner: Runner = run_subprocess,
) -> list[ChunkSync]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="mvm_syncscore_") as tmp:
        for cid, video, stem in discover_pairs(chunks_dir):
            if only is not None and cid not in only:
                continue
            try:
                row = score_chunk(cid, video, stem, install, runner=runner, work_dir=Path(tmp))
            except SyncNetError:
                logger.exception("syncscore: chunk %d could not be scored -- skipping", cid)
                continue
            best = row.best
            logger.info("syncscore: chunk %d %s%s", cid, row.verdict,
                        f" offset={best.offset} conf={best.confidence:.3f}" if best else "")
            rows.append(row)
    return rows


def write_report(rows: Sequence[ChunkSync], out: TextIO, *, input_dir: Path,
                 install: SyncNetInstall) -> None:
    out.write("# music_video_maker.syncscore report\n")
    out.write(f"# input_dir={input_dir.resolve()}\n")
    out.write(f"# syncnet_root={install.root} syncnet_v2.model sha256={install.weights_sha256}\n")
    out.write(f"# offset_limit={OFFSET_LIMIT} confidence_floor={CONFIDENCE_FLOOR} "
              "audio=staged chunk wav (never the mp4's own track)\n")
    writer = csv.writer(out)
    writer.writerow(FIELDNAMES)
    for row in rows:
        best = row.best
        writer.writerow([
            row.chunk_id, row.verdict,
            best.offset if best else "", f"{best.confidence:.3f}" if best else "",
            f"{best.min_dist:.3f}" if best else "", len(row.tracks),
            row.video_path, row.video_size_bytes, row.video_mtime,
            row.stem_path, "" if row.stem_size_bytes is None else row.stem_size_bytes,
            row.stem_mtime,
        ])


# --------------------------------------------------------------------------- #
# Auto re-roll
# --------------------------------------------------------------------------- #


def reseed_command(mvm: str, config: Path, ids: Sequence[int], generation: int,
                   timeline: str | None) -> list[str]:
    cmd = [mvm, "--config", str(config), "--reseed", ",".join(str(i) for i in ids),
           "--reseed-generation", str(generation)]
    if timeline:
        cmd += ["--timeline", timeline]
    return cmd


def auto_reseed(
    chunks_dir: Path,
    install: SyncNetInstall,
    rows: Sequence[ChunkSync],
    *,
    config: Path,
    generations: int,
    timeline: str | None = None,
    mvm: str = "music-video-maker",
    runner: Runner = run_subprocess,
) -> list[ChunkSync]:
    """Re-roll flagged chunks under successive seed generations until each passes.

    A chunk that never passes is re-rendered once more under its *best*
    generation (by confidence), so the file on disk is always the render that
    ``run_state.json`` records -- copying an older take back into place would
    leave a chunk whose pixels and fingerprint disagree, the #93 defect in a
    new place. Deterministic seeds make that final render reproduce the take.
    """
    current = {r.chunk_id: r for r in rows}
    history: dict[int, list[tuple[int, ChunkSync]]] = {
        r.chunk_id: [(0, r)] for r in rows if r.flagged
    }
    for generation in range(1, generations + 1):
        pending = sorted(cid for cid, hist in history.items() if hist[-1][1].flagged)
        if not pending:
            break
        code, out = runner(reseed_command(mvm, config, pending, generation, timeline),
                           config.parent)
        if code != 0:
            raise SyncNetError(f"reseed generation {generation} failed: {out.strip()[-300:]}")
        for row in score_dir(chunks_dir, install, only=set(pending), runner=runner):
            history[row.chunk_id].append((generation, row))
            current[row.chunk_id] = row
    restore: dict[int, list[int]] = {}
    for cid, hist in history.items():
        if not hist[-1][1].flagged:
            continue
        scored = [(g, r) for g, r in hist if r.best is not None]
        if not scored:
            continue
        # The nearest miss, not the most confident: a confident 10-frame offset
        # is confidently wrong.
        best_gen, _best_row = min(
            scored, key=lambda gr: (abs(gr[1].best.offset), -gr[1].best.confidence)
        )
        if best_gen != hist[-1][0]:
            restore.setdefault(best_gen, []).append(cid)
    for generation, ids in sorted(restore.items()):
        cmd = reseed_command(mvm, config, sorted(ids), generation, timeline)
        if generation == 0:
            # Generation 0 is the base seed, which --reseed never produces.
            cmd = [mvm, "--config", str(config), "--only-chunks",
                   ",".join(str(i) for i in sorted(ids))]
            if timeline:
                cmd += ["--timeline", timeline]
        code, out = runner(cmd, config.parent)
        if code != 0:
            raise SyncNetError(f"restoring generation {generation} failed: {out.strip()[-300:]}")
        # Re-score what is now on disk, so every row describes the file it names.
        for row in score_dir(chunks_dir, install, only=set(ids), runner=runner):
            current[row.chunk_id] = row
    return [current[cid] for cid in sorted(current)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvm-syncscore",
        description="Score rendered chunks' lip-sync with SyncNet against their own "
        "conditioning audio, and optionally re-roll the ones out of sync.",
    )
    parser.add_argument("chunks_dir", type=Path, help="directory with chunk_*.mp4 + chunk_*.wav")
    parser.add_argument("--syncnet", type=Path, required=True,
                        help="syncnet_python checkout (with its two weight files)")
    parser.add_argument("--syncnet-python", type=Path, default=None,
                        help="python of SyncNet's own venv (default: this interpreter)")
    parser.add_argument("--only", default=None, help="comma-separated chunk ids to score")
    parser.add_argument("--out", type=Path, default=None, help="CSV path (default: stdout)")
    parser.add_argument("--auto-reseed", type=int, default=0, metavar="N",
                        help="re-roll flagged chunks under up to N seed generations "
                        "(needs --config)")
    parser.add_argument("--config", type=Path, default=None, help="the run's run.toml")
    parser.add_argument("--timeline", default=None, help="segment name, if not the song")
    parser.add_argument("--music-video-maker", default="music-video-maker",
                        help="CLI used for re-rolls")
    return parser


def main(argv: list[str] | None = None, *, runner: Runner = run_subprocess) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.auto_reseed and args.config is None:
        logger.error("syncscore: --auto-reseed needs --config")
        return 2
    if not args.chunks_dir.is_dir():
        logger.error("syncscore: %s is not a directory", args.chunks_dir)
        return 1
    if shutil.which("ffmpeg") is None and runner is run_subprocess:
        logger.error("syncscore: ffmpeg is not on PATH")
        return 1
    try:
        install = SyncNetInstall.locate(args.syncnet, args.syncnet_python)
    except SyncNetError as exc:
        logger.error("syncscore: %s", exc)
        return 1
    only = {int(x) for x in args.only.split(",")} if args.only else None
    rows = score_dir(args.chunks_dir, install, only=only, runner=runner)
    if not rows:
        logger.error("syncscore: nothing scored in %s", args.chunks_dir)
        return 1
    if args.auto_reseed:
        rows = auto_reseed(args.chunks_dir, install, rows, config=args.config,
                           generations=args.auto_reseed, timeline=args.timeline,
                           mvm=args.music_video_maker, runner=runner)
    if args.out is not None:
        with args.out.open("w", newline="") as fh:
            write_report(rows, fh, input_dir=args.chunks_dir, install=install)
    else:
        write_report(rows, sys.stdout, input_dir=args.chunks_dir, install=install)
    return 1 if any(r.flagged for r in rows) else 0


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
