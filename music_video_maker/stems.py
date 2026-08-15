"""Stage 2a-stem: cut an isolated vocal stem at the spans Stage 2a already
computed from the master (issue #25).

MiniMax H3 lip-syncs to whatever waveform it is conditioned on. Today that is
a slice of the **full mix**, where the voice competes with drums, bass and
guitars -- and where a voice-like lead synth is indistinguishable from a
singer. Two defects observed on real renders are audio-conditioning problems
that no amount of prompt work can reach:

* the protagonist mouthing along to an instrumental passage carrying a squealy
  lead synth (3:15 of the v5 render), on a chunk prompted with ``text=""``;
* no usable lip-sync across the shared-words chorus (chunks 17-19), where both
  voices are in the mix -- a two-reference experiment already ruled out the
  reference *count* as the cause.

Conditioning on a vocal-only stem makes an instrumental passage near-silence
(nothing voice-shaped to sync to) and makes within-chunk timing explicit
(2.3 s of voice then 4.3 s of silence, rather than a continuous band).

**This does not touch the audio invariant.** The stem exists only as the
Stage 3 conditioning input. Generated audio is discarded at assembly exactly
as before and Stage 5 muxes the same pristine master track it always did --
nothing in :mod:`music_video_maker.assembly` changes. Equally, the stem never
gets an alignment of its own: sync comes from the timestamps Stage 1 computed
against the master, and this module cuts the stem at precisely the
``AudioChunk`` spans that produced, byte offset for byte offset (the same
millisecond rounding :mod:`music_video_maker.slicing` uses).

Separation is deliberately **off the render path**. The stem is a file the
operator produces once, inspects, and points the run at -- not an inference
call inside the pipeline::

    python -m demucs --two-stems=vocals -o stems/ audio/master.wav
    # -> stems/htdemucs/master/vocals.wav  and  no_vocals.wav

Keeping it explicit keeps a heavy ML dependency out of the render path and,
more importantly, keeps the artefact inspectable: htdemucs classifies a
heavily processed or vocoded voice ambiguously and can file it under ``other``
rather than ``vocals``. Conditioning on a stem that silently dropped a real
voice renders a closed mouth over a sung line -- a regression *caused* by this
feature. :class:`StemQualityReport` catches that mechanically: every chunk
that carries a lyric but whose stem slice is near-silent is named in a
warning. See ``docs/vocal-stem-workflow.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

from pydub import AudioSegment

from music_video_maker.contracts import AudioChunk, FrameGrid

logger = logging.getLogger(__name__)

DEFAULT_DURATION_TOLERANCE_SECONDS = 0.100
"""How far the stem's duration may differ from the master's before the run is
refused.

Real separation output is sample-identical to its input, so this is slack for
a container/round-trip difference, not for an edit. Anything larger means the
stem was made from a different cut of the song, and every chunk after the
divergence would be conditioned on audio from the wrong moment -- silently,
because each slice is still exactly the right *length*."""

DEFAULT_MAX_TAIL_PAD_SECONDS = 1.0
"""How far the chunk timeline may legitimately run past the end of the stem.

The outro chunk is grid-quantized, so it can land up to one grid step
(~0.708 s) beyond the final sample -- :mod:`music_video_maker.slicing` pads
the master slice with silence for exactly this reason. Beyond this grace the
stem simply does not cover the song and is refused."""

DEFAULT_SILENT_VOCAL_DBFS = -45.0
"""Peak level below which a stem slice counts as carrying no voice."""

DEFAULT_LEAKAGE_DBFS = -30.0
"""Peak level above which an *instrumental* chunk's stem slice counts as
still carrying band audio -- separation leakage, which weakens the
near-silence this issue is banking on."""

_DURATION_MATCH_TOLERANCE_SECONDS = 1e-3
"""How exactly an emitted slice must match its chunk's own duration."""


class StemError(Exception):
    """Base for every refusal in this module."""


class StemValidationError(StemError):
    """The stem file cannot be trusted as conditioning audio for this run.

    Missing, unreadable, or a different length than the master (or than the
    chunk timeline). Always a refusal, never a silent pad: a stem that is even
    slightly differently timed desyncs every chunk against the master while
    every individual slice still looks correct.
    """


class StemSliceError(StemError):
    """An emitted stem slice does not match the chunk it was cut for.

    Mirrors :class:`music_video_maker.slicing.ChunkFrameMismatchError`: if the
    conditioning audio's duration disagrees with the chunk's ``frame_count``,
    Stage 4a renders a video length its own audio does not match, which is the
    drift issue #20 exists to eliminate.
    """


@dataclass(frozen=True)
class StemQualityReport:
    """What the stem turned out to contain, per chunk, in the run's own terms.

    ``silent_voiced_chunk_ids`` is the important one. The failure mode of
    conditioning on a stem is not a noisy stem -- it is a stem that *omits* a
    voice the separator failed to recognise, which reads as "the fix improved
    instrumental behaviour" right up until the closed mouth over the last
    chorus.
    """

    stem_path: Path
    stem_duration: float
    chunk_count: int
    voiced_chunk_count: int
    silent_voiced_chunk_ids: tuple[int, ...] = ()
    leaky_instrumental_chunk_ids: tuple[int, ...] = ()
    padded_chunk_ids: tuple[int, ...] = ()

    def summary(self) -> str:
        return (
            f"{self.chunk_count} stem chunk(s) cut from {self.stem_path.name} "
            f"({self.stem_duration:.3f}s): {self.voiced_chunk_count} with lyrics, "
            f"{len(self.silent_voiced_chunk_ids)} of those silent in the stem, "
            f"{len(self.leaky_instrumental_chunk_ids)} instrumental chunk(s) still "
            f"carrying signal, {len(self.padded_chunk_ids)} padded at the tail"
        )

    def log_summary(self) -> None:
        """Log the one-line summary at INFO, plus a WARNING naming any sung
        chunk the stem is silent over."""
        logger.info("%s", self.summary())
        if self.silent_voiced_chunk_ids:
            logger.warning(
                "Vocal stem %s is silent (below %.1f dBFS) across %d chunk(s) that carry a "
                "lyric: %s. Conditioning on it would render a closed mouth over sung lines -- "
                "a processed or vocoded voice is often separated into 'other' rather than "
                "'vocals'. Listen to no_vocals.wav for the missing voice before rendering.",
                self.stem_path.name,
                DEFAULT_SILENT_VOCAL_DBFS,
                len(self.silent_voiced_chunk_ids),
                self.silent_voiced_chunk_ids,
            )
        if self.leaky_instrumental_chunk_ids:
            logger.info(
                "Vocal stem %s still carries audible signal on %d instrumental chunk(s): %s. "
                "Separation leakage weakens (but does not break) the near-silence this "
                "conditioning relies on.",
                self.stem_path.name,
                len(self.leaky_instrumental_chunk_ids),
                self.leaky_instrumental_chunk_ids,
            )


@dataclass(frozen=True)
class StemSliceResult:
    """``chunks`` re-pointed at their stem-sourced conditioning audio, plus
    the quality report for the stem they came from.

    Every field of every chunk other than ``audio_file`` is carried through
    untouched -- spans, frame counts, text, characters. Swapping the
    conditioning waveform must never re-plan the timeline.
    """

    chunks: tuple[AudioChunk, ...]
    report: StemQualityReport


def _audio_duration_seconds(path: Path, *, what: str) -> tuple[AudioSegment, float]:
    """Load ``path`` with pydub and return it alongside its duration in seconds."""
    if not path.exists() or not path.is_file():
        logger.error("Vocal stem refused: %s does not exist: %s", what, path)
        raise StemValidationError(f"{what} does not exist: {path}")

    try:
        audio = AudioSegment.from_file(str(path))
    except Exception as exc:
        logger.exception("Vocal stem refused: could not read %s at %s", what, path)
        raise StemValidationError(f"could not read {what} {path}: {exc}") from exc

    return audio, len(audio) / 1000.0


def validate_stem_against_master(
    stem_path: Path,
    master_path: Path,
    *,
    tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
) -> float:
    """Refuse a stem that is not the same length as the master; return its duration.

    Cheap and honest: the stem is cut at spans derived from the master, so a
    stem of a different length is a stem of a different edit, and every chunk
    after the divergence would be conditioned on the wrong moment of the song
    while still being exactly the right duration. Nothing downstream could
    notice. This is the one comparison that catches it, so it is a refusal
    with both durations named, never a pad or a stretch.
    """
    stem_path = Path(stem_path)
    master_path = Path(master_path)

    _, stem_duration = _audio_duration_seconds(stem_path, what="vocal stem")
    _, master_duration = _audio_duration_seconds(master_path, what="master audio")

    drift = abs(stem_duration - master_duration)
    if drift > tolerance_seconds:
        logger.error(
            "Vocal stem refused: %s is %.3fs but the master %s is %.3fs (differs by %.3fs, "
            "tolerance %.3fs). A stem of a different length silently desyncs every chunk "
            "against the master -- re-separate from this exact master rather than padding.",
            stem_path,
            stem_duration,
            master_path,
            master_duration,
            drift,
            tolerance_seconds,
        )
        raise StemValidationError(
            f"vocal stem {stem_path} duration {stem_duration:.3f}s does not match master "
            f"{master_path} duration {master_duration:.3f}s (differs by {drift:.3f}s, "
            f"tolerance {tolerance_seconds:.3f}s)"
        )

    logger.info(
        "Vocal stem %s validated against master %s: %.3fs vs %.3fs (%.3fs apart).",
        stem_path.name,
        master_path.name,
        stem_duration,
        master_duration,
        drift,
    )
    return stem_duration


def slice_stem_for_chunks(
    stem_path: Path,
    chunks: tuple[AudioChunk, ...],
    stems_dir: Path,
    *,
    master_path: Path | None = None,
    tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    max_tail_pad_seconds: float = DEFAULT_MAX_TAIL_PAD_SECONDS,
    silent_dbfs: float = DEFAULT_SILENT_VOCAL_DBFS,
    leakage_dbfs: float = DEFAULT_LEAKAGE_DBFS,
    grid: FrameGrid | None = None,
) -> StemSliceResult:
    """Cut ``stem_path`` at every chunk's span and return the chunks re-pointed
    at the resulting conditioning audio.

    ``chunks`` are Stage 2a's output, unmodified -- this function consumes
    their ``start``/``end``/``frame_count`` and never recomputes them. Slices
    are written as ``chunk_{chunk_id:03d}_vocal.wav`` into ``stems_dir``
    (created if missing), which must not be the mix's ``chunks_dir``: both
    sets are worth keeping for the mix-vs-stem A/B and for listening to
    afterwards.

    ``master_path`` enables the duration check in
    :func:`validate_stem_against_master` and should always be passed by the
    pipeline. Even without it, a stem too short to cover the chunk timeline is
    refused.

    ``grid`` is H3's frame grid -- a model constant, so the default is the
    right one; it is a parameter only so a test can exercise the guard. Every
    emitted slice is checked against both the chunk's own span and the
    duration its ``frame_count`` implies.
    """
    stem_path = Path(stem_path)
    stems_dir = Path(stems_dir)
    grid = grid if grid is not None else FrameGrid()

    stem, stem_duration = _audio_duration_seconds(stem_path, what="vocal stem")

    if master_path is not None:
        validate_stem_against_master(
            stem_path, Path(master_path), tolerance_seconds=tolerance_seconds
        )

    if not chunks:
        logger.warning(
            "slice_stem_for_chunks called with no chunks; nothing to condition on the stem %s",
            stem_path,
        )
        report = StemQualityReport(
            stem_path=stem_path,
            stem_duration=stem_duration,
            chunk_count=0,
            voiced_chunk_count=0,
        )
        report.log_summary()
        return StemSliceResult(chunks=(), report=report)

    timeline_end = max(chunk.end for chunk in chunks)
    if timeline_end - stem_duration > max_tail_pad_seconds:
        logger.error(
            "Vocal stem refused: %s is only %.3fs but the chunk timeline runs to %.3fs "
            "(%.3fs short, grace %.3fs). Padding that much silence would blank real audio; "
            "this is the wrong stem for this song.",
            stem_path,
            stem_duration,
            timeline_end,
            timeline_end - stem_duration,
            max_tail_pad_seconds,
        )
        raise StemValidationError(
            f"vocal stem {stem_path} duration {stem_duration:.3f}s cannot cover a chunk "
            f"timeline ending at {timeline_end:.3f}s"
        )

    stems_dir.mkdir(parents=True, exist_ok=True)

    conditioned: list[AudioChunk] = []
    silent_voiced: list[int] = []
    leaky_instrumental: list[int] = []
    padded: list[int] = []
    voiced_count = 0

    for chunk in chunks:
        start_ms = round(chunk.start * 1000)
        end_ms = round(chunk.end * 1000)
        sliced = stem[start_ms:end_ms]

        # Same tail case slicing.py handles on the master: a grid-quantized
        # outro can land past the final sample, and pydub truncates silently.
        # A short conditioning stem is an audio/video length mismatch, so pad
        # it out. The stem-too-short refusal above is what keeps this from
        # ever masking a genuinely wrong file.
        shortfall_ms = (end_ms - start_ms) - len(sliced)
        if shortfall_ms > 0:
            logger.info(
                "Chunk %d runs %dms past the end of the vocal stem; padding the conditioning "
                "slice with silence so its duration still matches the chunk exactly.",
                chunk.chunk_id,
                shortfall_ms,
            )
            padded.append(chunk.chunk_id)
            sliced = (
                sliced
                + AudioSegment.silent(duration=shortfall_ms, frame_rate=stem.frame_rate)
                .set_channels(stem.channels)
                .set_sample_width(stem.sample_width)
            )

        sliced_duration = len(sliced) / 1000.0
        if abs(sliced_duration - chunk.duration) > _DURATION_MATCH_TOLERANCE_SECONDS:
            logger.error(
                "Chunk %d: stem slice is %.6fs but the chunk spans %.6fs (%.3f-%.3f). The "
                "conditioning audio must be exactly as long as the chunk it renders, or "
                "Stage 4a's video drifts from its own audio.",
                chunk.chunk_id,
                sliced_duration,
                chunk.duration,
                chunk.start,
                chunk.end,
            )
            raise StemSliceError(
                f"chunk {chunk.chunk_id}: stem slice duration {sliced_duration!r} does not "
                f"match chunk duration {chunk.duration!r}"
            )

        if chunk.frame_count is not None:
            frame_duration = grid.frames_to_seconds(chunk.frame_count)
            if abs(sliced_duration - frame_duration) > _DURATION_MATCH_TOLERANCE_SECONDS:
                logger.error(
                    "Chunk %d: stem slice is %.6fs but frame_count=%d implies %.6fs. Stage 4a "
                    "injects that frame count as H3's `length`, so the conditioning audio and "
                    "the rendered video would be different lengths -- exactly the per-chunk "
                    "drift issue #20 eliminated for the mix.",
                    chunk.chunk_id,
                    sliced_duration,
                    chunk.frame_count,
                    frame_duration,
                )
                raise StemSliceError(
                    f"chunk {chunk.chunk_id}: stem slice duration {sliced_duration!r} does not "
                    f"match the {frame_duration!r} implied by frame_count {chunk.frame_count!r}"
                )

        out_path = stems_dir / f"chunk_{chunk.chunk_id:03d}_vocal.wav"
        try:
            sliced.export(str(out_path), format="wav")
        except OSError as exc:
            logger.exception(
                "Failed writing the conditioning stem slice for chunk %d to %s",
                chunk.chunk_id,
                out_path,
            )
            raise StemSliceError(
                f"chunk {chunk.chunk_id}: could not write stem slice {out_path}: {exc}"
            ) from exc

        peak = sliced.max_dBFS
        if chunk.is_instrumental or not chunk.text.strip():
            if peak > leakage_dbfs:
                leaky_instrumental.append(chunk.chunk_id)
        else:
            voiced_count += 1
            if peak < silent_dbfs:
                silent_voiced.append(chunk.chunk_id)

        conditioned.append(replace(chunk, audio_file=out_path))

    report = StemQualityReport(
        stem_path=stem_path,
        stem_duration=stem_duration,
        chunk_count=len(conditioned),
        voiced_chunk_count=voiced_count,
        silent_voiced_chunk_ids=tuple(silent_voiced),
        leaky_instrumental_chunk_ids=tuple(leaky_instrumental),
        padded_chunk_ids=tuple(padded),
    )
    report.log_summary()
    return StemSliceResult(chunks=tuple(conditioned), report=report)


__all__ = [
    "DEFAULT_DURATION_TOLERANCE_SECONDS",
    "DEFAULT_LEAKAGE_DBFS",
    "DEFAULT_MAX_TAIL_PAD_SECONDS",
    "DEFAULT_SILENT_VOCAL_DBFS",
    "StemError",
    "StemQualityReport",
    "StemSliceError",
    "StemSliceResult",
    "StemValidationError",
    "slice_stem_for_chunks",
    "validate_stem_against_master",
]
