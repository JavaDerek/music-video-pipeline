"""Lay out synthesized speech so every chunk cut falls in silence.

**General form: when the voice is synthesized one line at a time, the
timeline is a choice, not a measurement.** A sung master is fixed: the
vocalist breathed where they breathed, and slicing (issue #70) can only pick
the least-bad place to cut a phrase. A text-to-speech master is assembled by
the caller from one clip per line, so the pauses between lines can be chosen
*after* the chunk boundaries are -- and chosen so that every boundary lands
in silence, on H3's ``5 + 17k`` frame grid, with every chunk starting at the
first sample of speech.

That matters for lip-sync, measured on a 98 s synthesized-speech monologue
rendered through this pipeline with ordinary slicing: 14 of 21 chunk cuts
landed mid-phrase, and two chunks began more than a second before their
first word. H3 starts the mouth at frame 0 regardless (issue #79), so those
chunks open on a mouth moving over silence. Neither defect can happen to a
timeline built here.

What this module does:

1. Trims each line's clip to its speech (leading and trailing silence off).
2. Groups consecutive lines into chunks, greedily, joined by a fixed
   ``inner_gap``, so that each chunk's speech plus a minimum ``min_tail`` of
   trailing silence fits under ``max_frames``.
3. Pads each chunk's tail with silence up to the smallest grid-valid length
   that holds it (and at least ``min_frames``).
4. Writes the chunks back to back as one master, plus the script and a shot
   plan whose ``length_seconds`` pin those exact lengths (issue #27).

Every chunk therefore starts on a word and ends in silence; the silence a
cut falls in is the tail padding, which the speaker's next pause absorbs.

What it deliberately does not do: split a line. A line whose speech alone
cannot fit under ``max_frames`` is refused by name, because only the caller
knows where its text can be divided -- this module has the audio, not an
alignment of the words inside it. Synthesize long sentences clause by clause.

Stdlib only (``wave`` + ``array``); 16-bit PCM mono in, 16-bit PCM mono out.
No model is called here: the speech arrives already synthesized.
"""

from __future__ import annotations

import argparse
import array
import json
import logging
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker.contracts import FrameGrid

logger = logging.getLogger(__name__)

DEFAULT_INNER_GAP = 0.30
"""Seconds of silence between two lines that share a chunk."""

DEFAULT_MIN_TAIL = 0.25
"""Seconds of silence a chunk must end on, at minimum, before its cut. The
grid's padding usually adds more; this guarantees the mouth has closed."""

DEFAULT_SILENCE_THRESHOLD = 0.02
"""Fraction of full scale below which a sample counts as silence when
trimming a clip's leading and trailing edges."""


class SpeechPlanError(ValueError):
    """The lines cannot be laid out as asked (bad audio, or a line too long)."""


@dataclass(frozen=True)
class SpeechLine:
    """One line of the script and the clip that speaks it."""

    text: str
    audio: Path


@dataclass(frozen=True)
class PlannedChunk:
    """One chunk of the laid-out master."""

    chunk_id: int
    start: float
    """Seconds into the master where this chunk -- and its first word -- begins."""
    frames: int
    """Grid-valid frame count; the chunk's exact length."""
    line_indices: tuple[int, ...]
    speech_seconds: float
    """Seconds of speech plus inner gaps; the rest of the chunk is tail silence."""

    def length_seconds(self, fps: int) -> float:
        return self.frames / fps


@dataclass(frozen=True)
class SpeechPlan:
    chunks: tuple[PlannedChunk, ...]
    lines: tuple[SpeechLine, ...]
    sample_rate: int
    fps: int
    samples: array.array
    """The laid-out master, 16-bit mono."""

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def _read_mono16(path: Path) -> tuple[array.array, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels, width, rate = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
            )
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise SpeechPlanError(f"cannot read {path} as WAV: {exc}") from exc
    if channels != 1 or width != 2:
        raise SpeechPlanError(
            f"{path} is {channels}-channel {8 * width}-bit; speechplan takes 16-bit mono PCM "
            "(convert with: ffmpeg -i in.wav -ac 1 -c:a pcm_s16le out.wav)"
        )
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":  # WAV is little-endian on disk
        samples.byteswap()
    return samples, rate


def trim_silence(
    samples: Sequence[int], threshold: float = DEFAULT_SILENCE_THRESHOLD
) -> tuple[int, int]:
    """``(first, last + 1)`` of the samples louder than ``threshold`` of full
    scale, or ``(0, 0)`` for a clip that is silence throughout."""
    level = int(threshold * 32767)
    first = next((i for i, s in enumerate(samples) if abs(s) > level), None)
    if first is None:
        return 0, 0
    last = next(i for i in range(len(samples) - 1, -1, -1) if abs(samples[i]) > level)
    return first, last + 1


def _grid_frames_for(seconds: float, grid: FrameGrid, min_frames: int) -> int:
    """Smallest grid-valid frame count that holds ``seconds`` and is at least
    ``min_frames``."""
    frames = max(min_frames, grid.base_frames)
    while frames < seconds * grid.fps - 1e-9 or not grid.is_valid(frames):
        frames += 1
    return frames


def plan_speech(
    lines: Sequence[SpeechLine],
    *,
    grid: FrameGrid | None = None,
    min_frames: int = 124,
    max_frames: int = 141,
    inner_gap: float = DEFAULT_INNER_GAP,
    min_tail: float = DEFAULT_MIN_TAIL,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
) -> SpeechPlan:
    """Lay ``lines`` out into grid-exact chunks that start on speech and end
    in silence. See the module docstring for the rules."""
    grid = grid or FrameGrid()
    if not lines:
        raise SpeechPlanError("no lines to lay out")
    if not (grid.is_valid(min_frames) and grid.is_valid(max_frames)):
        raise SpeechPlanError(
            f"min_frames={min_frames} and max_frames={max_frames} must both be on the "
            f"{grid.base_frames}+{grid.step_frames}k grid"
        )
    if min_frames > max_frames:
        raise SpeechPlanError(f"min_frames={min_frames} exceeds max_frames={max_frames}")

    clips: list[array.array] = []
    rate: int | None = None
    for index, line in enumerate(lines):
        samples, line_rate = _read_mono16(line.audio)
        if rate is None:
            rate = line_rate
        elif line_rate != rate:
            raise SpeechPlanError(
                f"line {index} ({line.audio}) is {line_rate} Hz; line 0 is {rate} Hz. "
                "Resample every clip to one rate first."
            )
        first, end = trim_silence(samples, silence_threshold)
        if end == 0:
            raise SpeechPlanError(f"line {index} ({line.audio}) is silent: {line.text!r}")
        clips.append(samples[first:end])
    assert rate is not None

    max_seconds = max_frames / grid.fps
    gap = array.array("h", bytes(2 * round(inner_gap * rate)))
    speech = [len(c) / rate for c in clips]
    for index, seconds in enumerate(speech):
        if seconds + min_tail > max_seconds:
            raise SpeechPlanError(
                f"line {index} speaks for {seconds:.3f}s; with a {min_tail:.2f}s tail it "
                f"cannot fit in {max_frames} frames ({max_seconds:.3f}s). Split it where "
                f"its text allows and synthesize the pieces separately: {lines[index].text!r}"
            )

    groups: list[list[int]] = []
    current: list[int] = []
    current_seconds = 0.0
    for index, seconds in enumerate(speech):
        joined = current_seconds + (inner_gap if current else 0.0) + seconds
        if current and joined + min_tail > max_seconds:
            groups.append(current)
            current, current_seconds = [index], seconds
        else:
            current.append(index)
            current_seconds = joined
    groups.append(current)

    out = array.array("h")
    chunks: list[PlannedChunk] = []
    total_frames = 0
    for chunk_id, group in enumerate(groups):
        body = array.array("h")
        for position, index in enumerate(group):
            if position:
                body.extend(gap)
            body.extend(clips[index])
        body_seconds = len(body) / rate
        frames = _grid_frames_for(body_seconds + min_tail, grid, min_frames)
        start_sample = round(total_frames * rate / grid.fps)
        total_frames += frames
        end_sample = round(total_frames * rate / grid.fps)
        body.extend(array.array("h", bytes(2 * (end_sample - start_sample - len(body)))))
        out.extend(body)
        chunks.append(
            PlannedChunk(
                chunk_id=chunk_id,
                start=start_sample / rate,
                frames=frames,
                line_indices=tuple(group),
                speech_seconds=body_seconds,
            )
        )

    logger.info(
        "speechplan: %d line(s) -> %d chunk(s), %d frames (%.3fs) at %d fps; every cut "
        "in silence, every chunk starting on speech",
        len(lines),
        len(chunks),
        total_frames,
        total_frames / grid.fps,
        grid.fps,
    )
    return SpeechPlan(
        chunks=tuple(chunks),
        lines=tuple(lines),
        sample_rate=rate,
        fps=grid.fps,
        samples=out,
    )


def write_master(plan: SpeechPlan, path: Path) -> Path:
    data = plan.samples
    if sys.byteorder == "big":
        data = array.array("h", data)
        data.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(plan.sample_rate)
        handle.writeframes(data.tobytes())
    return path


def write_script(plan: SpeechPlan, path: Path) -> Path:
    """One line per script line, in order: the lyrics-format text the run's
    alignment reads (untagged lines take ``default_lead_vocalist``)."""
    path.write_text("".join(f"{line.text.strip()}\n" for line in plan.lines), encoding="utf-8")
    return path


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_shot_plan(plan: SpeechPlan, path: Path, *, shot: str = "") -> Path:
    """A shot plan pinning every chunk's start and exact length (issue #27).

    ``shot`` is written into every entry; leave it empty to let the run's
    narrative concept stand, or fill entries in by hand afterwards."""
    parts = [
        "# Written by music_video_maker.speechplan. Every chunk starts on speech and",
        "# ends in silence; length_seconds pins the grid-exact length that guarantees it.",
        "",
    ]
    for chunk in plan.chunks:
        text = " ".join(plan.lines[i].text.strip() for i in chunk.line_indices)
        parts += [
            "[[shot]]",
            f"chunk_id = {chunk.chunk_id}",
            f"start = {chunk.start!r}",
            f"length_seconds = {chunk.length_seconds(plan.fps)!r}  "
            f"# {chunk.frames} frames, {chunk.speech_seconds:.3f}s of speech",
            f"# lyric: {_toml_string(text)}",
            f"shot = {_toml_string(shot)}",
            "",
        ]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def load_lines(path: Path) -> tuple[SpeechLine, ...]:
    """Read ``[{"text": ..., "audio": ...}, ...]``; relative audio paths
    resolve against the JSON file's own directory."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpeechPlanError(f"cannot read lines file {path}: {exc}") from exc
    if not isinstance(raw, list) or not all(
        isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("audio")
        for item in raw
    ):
        raise SpeechPlanError(
            f"{path} must be a JSON list of {{\"text\": str, \"audio\": path}} objects"
        )
    base = path.parent
    return tuple(
        SpeechLine(text=item["text"], audio=(base / item["audio"]).resolve()) for item in raw
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.speechplan",
        description="Lay out per-line speech clips so every chunk cut falls in silence.",
    )
    parser.add_argument("lines", type=Path, help='JSON list of {"text": ..., "audio": ...}')
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=124)
    parser.add_argument("--max-frames", type=int, default=141)
    parser.add_argument("--inner-gap", type=float, default=DEFAULT_INNER_GAP)
    parser.add_argument("--min-tail", type=float, default=DEFAULT_MIN_TAIL)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    try:
        plan = plan_speech(
            load_lines(args.lines),
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            inner_gap=args.inner_gap,
            min_tail=args.min_tail,
        )
    except SpeechPlanError as exc:
        logger.error("speechplan: %s", exc)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_master(plan, args.out_dir / "master.wav")
    write_script(plan, args.out_dir / "script.txt")
    write_shot_plan(plan, args.out_dir / "shot_plan.toml")
    logger.info("speechplan: %d chunks, %.3fs -> %s", len(plan.chunks), plan.duration, args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
