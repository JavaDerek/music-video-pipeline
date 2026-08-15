"""The chunk skeleton every authoring stage plans against (issue #54).

Mirrors ``cli.prepare_shot_plan``'s Stage 1-2 call sequence exactly (parse
lyrics, force-align, slice) -- but this module cannot import ``cli.py``: the
authoring/render import boundary (design section 2,
``tests/test_authoring_boundary.py``) only allows ``authoring/`` to import
``config``, ``contracts``, ``shot_plan``, ``alignment``, ``slicing`` and
``lyrics``, never ``cli``. Duplicating four function calls here is a much
smaller cost than blurring that boundary.

No GPU, no ComfyUI, no custody handoff -- same as ``--prepare``, and for the
same reason: this is ~6s of CPU work with nothing to hand either of them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from music_video_maker.alignment import align
from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk
from music_video_maker.lyrics import parse_lyrics
from music_video_maker.shot_plan import ShotLength
from music_video_maker.slicing import slice_audio

logger = logging.getLogger(__name__)


class SkeletonError(RuntimeError):
    """Raised when Stage 1-2 produces no chunks to author against."""


def load_chunk_skeleton(
    config: RunConfig,
    *,
    align_model: object | None = None,
    shot_lengths: Sequence[ShotLength] = (),
) -> tuple[AudioChunk, ...]:
    """Run Stages 1-2 and return the resulting chunk timeline.

    Every authoring stage that needs to know the song's structure (span,
    duration, frame count, lyric text, instrumental flag, per chunk) calls
    this rather than re-deriving it -- same one source of truth
    ``run_pipeline`` and ``prepare_shot_plan`` already share.

    ``shot_lengths`` re-cuts the timeline the way a render with those
    editorial lengths will (issue #54 design section 5, and the render-side
    ``--prepare --from-plan`` this mirrors). Stage 2 calls this a second time
    with its own sheet's requests, then re-anchors onto the result -- see
    :mod:`~music_video_maker.authoring.reanchor`.
    """
    lines = parse_lyrics(config.lyrics_file, config.cast, config.default_lead_vocalist)
    alignment = align(
        config.master_audio,
        lines,
        model=align_model,
        # The third call site, and the one easiest to miss: a plan authored
        # against a different alignment than the render uses describes chunks
        # the render never emits. On "Deathless" that was 71 chunks (15 voiced)
        # against 80 (41 voiced) -- every anchor in the plan wrong.
        model_size=config.alignment_model_size,
        strict_alignment=config.strict_alignment,
        overrides=config.alignment_overrides,
    )
    chunks = slice_audio(
        config.master_audio,
        alignment,
        config.hardware,
        config.chunks_dir,
        cover_instrumentals=config.instrumental_coverage,
        instrumental_shot_seconds=config.instrumental_shot_seconds,
        shot_lengths=shot_lengths,
    )
    if not chunks:
        raise SkeletonError(
            f"no chunks produced by slicing {config.master_audio} against "
            f"{config.lyrics_file} -- nothing to author against"
        )
    logger.info("Chunk skeleton loaded: %d chunk(s) available to author", len(chunks))
    return chunks


def skeleton_table_text(chunks: Sequence[AudioChunk]) -> str:
    """A deterministic, tab-separated rendering of the chunk skeleton --
    one line per chunk: ``chunk_id, start, end, frame_count, tag, lyric``.

    Two uses, both wanting the same thing: the concept stage feeds this to
    the model as "the song's structure", and :mod:`~music_video_maker.authoring.session`
    hashes it to detect when a stage's *input* has moved (a re-alignment, an
    edited lyrics file) since it last ran. Deterministic and whitespace-exact
    on purpose -- a hash that moved because of formatting rather than
    content would report false staleness."""
    lines = [
        "\t".join(
            (
                str(chunk.chunk_id),
                f"{chunk.start:.3f}",
                f"{chunk.end:.3f}",
                str(chunk.frame_count),
                "INSTRUMENTAL" if chunk.is_instrumental else "LYRIC",
                # Who is singing this chunk. The alignment has always known
                # (the render reads it to pick which cast photo conditions
                # the shot) and the authoring layer never showed it to the
                # model, so a beat sheet could hand a sung chunk to the
                # character who is not singing it -- and every downstream
                # stage then did its job correctly on a wrong premise. On the
                # first machine-authored plan to reach a GPU, 25 of 41 sung
                # chunks ended up framed on whoever was not singing them.
                # Empty on an instrumental: nobody is singing it, and saying
                # otherwise would invite a beat about a voice that is silent.
                "/".join(chunk.characters) if not chunk.is_instrumental else "",
                chunk.text.strip() if chunk.text else "",
            )
        )
        for chunk in chunks
    ]
    return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["SkeletonError", "load_chunk_skeleton", "skeleton_table_text"]
