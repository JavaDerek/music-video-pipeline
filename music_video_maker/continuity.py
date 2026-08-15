"""Issue #12: seed-and-feed I2V latent continuity across chunks.

Each chunk render is otherwise an independent diffusion sampling event, so
consecutive shots visually re-layout themselves with every lyric line. This
module bridges chunk N-1 into chunk N by extracting chunk N-1's rendered last
frame, staging it back to ComfyUI, and mutating chunk N's workflow onto the
I2V (:data:`music_video_maker.workflow_graph.CLASS_TYPE_H3_IMAGE_TO_VIDEO` --
``MiniMaxH3ImageToVideo``) path instead of the base T2V/reference path, so the
model begins denoising from the exact visual state the previous clip ended
on.

Frame-grid reconciliation
--------------------------
Two facts about H3's temporal quantization float around this project in two
different phrasings. They are **the same rule**, not competing ones:

1. ``docs/h3-node-schema.md`` (ground truth, pulled from a live
   ``/object_info``): the ``length`` INT input is a raw frame count at 24
   fps, quantized by ComfyUI as ``min=5, step=17``. Valid values are
   ``5 + 17k``: 5, 22, 39, 56 ... 124 (trained-range floor) ... 362
   (trained-range ceiling).
2. Issue #12's body (blueprint language): "H3-Base maps latent tokens to
   pixel frames in a repeating ``1,4,4,4,4`` pattern -- a requested 20-frame
   context decodes to 22 actual frames."

Reading #2 through #1: 20 is not itself a valid grid point, so it quantizes
**up** to the next one -- 22 (``5 + 17*1``). The ``1,4,4,4,4`` pattern is
*why* the step is 17 rather than some other number: it is one repeating
group of five decoded latents contributing ``1 + 4 + 4 + 4 + 4 = 17`` pixel
frames. So "a requested 20-frame context decodes to 22" and "the grid step
is 17, quantized from a minimum of 5" describe the identical quantization
rule from two altitudes -- the raw grid arithmetic (#1) and the VAE
mechanism that produces that specific step size (#2). :func:`quantize_length_frames`
implements rule #1 directly; nothing in this module needs to separately
model latent counts.

This module does **not** own chunk-duration policy (issue #20 does) --
:func:`quantize_length_frames` is exposed as pure grid math for whatever
duration a chunk ends up with, not a slicing decision.

Design constraints (see the issue and its cross-lane contract)
----------------------------------------------------------------
- The I2V path is a **separate authored template**
  (``config.i2v_workflow_template``), never a rewrite of the reference-to-
  video graph -- ``MiniMaxH3ImageToVideo``'s real input schema has not been
  dumped from a live server yet (issue #18 owns that). Chunk 0, and any
  chunk whose predecessor did not succeed, renders through the base
  template.
- The seed-frame node is a second ``LoadImage``, disambiguated from the
  cast-reference ``LoadImage`` by ``_meta.title`` (never by node id --
  :func:`music_video_maker.workflow_graph.find_one_node` handles that).
- All of this is inert unless ``config.i2v_continuity`` is set --
  :class:`ContinuityWorkflowProvider` takes that as an explicit
  ``continuity_enabled`` constructor flag so a caller can wire it
  unconditionally and let the flag decide.

Chaining consecutive shots (issue #28)
---------------------------------------
Bridging one boundary is issue #12. Running that bridge down a whole song --
so shot B's *first* frame is shot A's *last* frame all the way through -- is
issue #28, and it needs two limits on top of the machinery above.

**Re-anchoring.** Every chained chunk conditions on the previous chunk's
output, so any drift the model introduces is fed back in and compounds.
:func:`is_reanchor_chunk` breaks the chain on a fixed ``chunk_id %
reanchor_interval`` grid, sending that chunk through the base reference path
where the cast photo and the vocal stem re-assert themselves. The rule is
modulo rather than a running counter deliberately: ``--resume`` calls the
provider only for the chunks it re-renders, so a counter would produce a
different chain on a resumed run than on the original one, and the two runs'
chunks would no longer be interchangeable.

**Eligibility.** ``MiniMaxH3ImageToVideo`` has **no audio-conditioning input
at all** -- no ``audio_vae``, no ``ref_audios`` (``docs/h3-node-schema.md``;
the base ``MiniMaxH3ReferenceToVideo`` has both, and its
``ref_audios.ref_audio_0`` slot carrying the sliced vocal stem is what makes
a chunk lip-sync). So a chained chunk buys visual continuity by giving up
lip sync on that chunk. That is a fine trade on an instrumental chunk and a
bad one on a lyric line, and which is which is knowledge this module does
not have -- ``AudioChunk.is_instrumental`` and the shot plan live upstream.
``chainable_chunk_ids`` is therefore an injected set: the orchestrator says
which chunks may take the chained path, and ``None`` (every chunk eligible)
logs a loud warning naming the consequence rather than quietly desyncing a
song's worth of mouths.

**Provenance.** :attr:`ContinuityWorkflowProvider.chain_sources` records what
each chunk's first frame actually came from -- the predecessor id, or ``None``
for an unchained chunk. Chained and unchained renders of the same span are
different content, so ``--resume`` has to be able to tell them apart;
:func:`planned_chain_source` is the same decision computed from configuration
alone (what a run can predict before rendering), and
:func:`chain_reuse_blocked` is the one-line rule that stops a cached chunk
being reused after the chunk it was seeded from has been re-rendered.

Index-space guard
------------------
The seed frame comes from chunk **N-1** while the prompt/audio for the
workflow being built come from chunk **N**. ``AlignedSegment`` indices,
``LyricLine`` indices, and chunk ids are all bare ``int`` and are not
interchangeable -- this project has been bitten by that twice already (see
``workflow_graph.ChunkMismatchError``, ``staging.StagingError``'s chunk-id
check). Every place two chunk-carrying objects meet in this module is
guarded and raises :class:`ContinuityIndexError`, naming both ids, rather
than silently building a workflow that seeds chunk N from the wrong video.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Collection, Container, Mapping, Sequence
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

from music_video_maker.contracts import (
    H3_FRAME_GRID,
    AssetStager,
    ChunkResult,
    ExpandedPrompt,
    RunState,
    StagedAssets,
    Workflow,
    WorkflowMutator,
)
from music_video_maker.workflow_graph import (
    CLASS_TYPE_H3_IMAGE_TO_VIDEO,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    WorkflowGraphMutator,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Frame-grid math (pure; issue #20 owns chunk-duration policy, not this)
# --------------------------------------------------------------------------- #

# The grid itself now lives on ``contracts.FrameGrid`` (issue #20), so Stage 2a
# slicing and this module quantize against the *same* arithmetic rather than two
# copies of it. These names stay as aliases -- they are how the rest of the
# project already refers to the grid.

FPS = H3_FRAME_GRID.fps
"""``MiniMaxH3*ToVideo.length`` is a frame count at this fixed frame rate."""

MIN_LENGTH_FRAMES = H3_FRAME_GRID.base_frames
LENGTH_STEP_FRAMES = H3_FRAME_GRID.step_frames
"""One repeating ``1,4,4,4,4`` decode group (5 latents -> 17 pixel frames)."""

TRAINED_MIN_FRAMES = H3_FRAME_GRID.trained_min_frames
"""~5.17 s. Below this the model is extrapolating outside its trained range."""

TRAINED_MAX_FRAMES = H3_FRAME_GRID.trained_max_frames
"""~15.08 s. Above this the model is extrapolating outside its trained range."""

DECODE_GROUP_PATTERN = (1, 4, 4, 4, 4)
"""The repeating per-latent pixel-frame contribution; ``sum() == LENGTH_STEP_FRAMES``."""

assert sum(DECODE_GROUP_PATTERN) == LENGTH_STEP_FRAMES, (
    "DECODE_GROUP_PATTERN must sum to LENGTH_STEP_FRAMES -- these encode the same fact"
)


def is_valid_length_frames(frames: int) -> bool:
    """True if ``frames`` lands exactly on H3's ``length`` grid (``5 + 17k``)."""
    return H3_FRAME_GRID.is_valid(frames)


def quantize_length_frames(requested_frames: int) -> int:
    """Round ``requested_frames`` UP to the next valid point on the ``length``
    grid (``min=5, step=17``: 5, 22, 39 ... 124 ... 362).

    This is issue #12's "a requested 20-frame context decodes to 22" example,
    generalized: ``quantize_length_frames(20) == 22``. Rounds up (never down)
    so the rendered clip is never shorter than what was asked for -- a chunk
    coming in short would desync against its audio stem duration.

    Does not clamp into the trained range (124-362) -- issue #20 owns
    deciding chunk durations; this is pure grid arithmetic only.
    """
    return H3_FRAME_GRID.quantize_up(requested_frames)


def frames_to_seconds(frames: int, fps: int = FPS) -> float:
    return frames / fps


# --------------------------------------------------------------------------- #
# Chain policy (pure; issue #28)
# --------------------------------------------------------------------------- #


def is_reanchor_chunk(chunk_id: int, reanchor_interval: int | None) -> bool:
    """True when ``chunk_id`` is a deliberate re-anchor point: a chunk that
    renders through the base reference path even though its predecessor
    succeeded, to stop chained drift compounding across a whole song.

    ``None`` (and any non-positive interval, which is meaningless) disables
    re-anchoring entirely and returns ``False`` -- never ``True``, so a
    nonsensical value can only ever cost continuity, never silently turn the
    whole feature off in the other direction.

    The grid is ``chunk_id % interval == 0``, not a running count of how many
    chunks have chained so far. A running count would be read differently by a
    resumed run, which only calls the provider for the chunks it re-renders:
    the same chunk would chain in one run and re-anchor in the next, and the
    two renders of that chunk would not be interchangeable. Modulo is a pure
    function of the chunk id, so every run agrees. Note the side effect that
    ``interval=1`` marks every chunk a re-anchor point, i.e. chaining off --
    which is the honest reading of "re-anchor every chunk".
    """
    if reanchor_interval is None or reanchor_interval < 1:
        return False
    return chunk_id % reanchor_interval == 0


def planned_chain_source(
    chunk_id: int,
    *,
    continuity_enabled: bool,
    reanchor_interval: int | None = None,
    chainable_chunk_ids: Collection[int] | None = None,
) -> int | None:
    """The chunk whose last frame is *configured* to supply ``chunk_id``'s
    first frame, or ``None`` if this chunk is configured to render unchained.

    Deliberately considers configuration only -- never whether the predecessor
    actually rendered. It answers "what does this run intend for this chunk",
    which is what a resumed run can compute up front and compare against a
    cached chunk. What actually happened, including any degradation, is
    :attr:`ContinuityWorkflowProvider.chain_sources`.

    ``chainable_chunk_ids=None`` means every chunk is eligible; an empty
    collection means none is (an all-voiced song, say) -- the two are
    different answers and are kept distinguishable.
    """
    if not continuity_enabled:
        return None
    predecessor_id = chunk_id - 1
    if predecessor_id < 0:
        return None
    if is_reanchor_chunk(chunk_id, reanchor_interval):
        return None
    if chainable_chunk_ids is not None and chunk_id not in chainable_chunk_ids:
        return None
    return predecessor_id


def chain_reuse_blocked(
    stored_chain_source: int | None, rerendered_chunk_ids: Container[int]
) -> bool:
    """True when a cached chunk must be re-rendered because the chunk it was
    chained off has itself been re-rendered in this run.

    A chained chunk's first frame is *pixels from another chunk's video file*,
    not a value any fingerprint field can restate. So when chunk N re-renders,
    chunk N+1's cached video was seeded from footage that no longer exists,
    and reusing it puts a hard visual cut exactly where the chain was supposed
    to remove one. Chunks are rendered in ascending id order and the chain
    only ever reaches back one chunk, so applying this rule as each chunk is
    considered propagates down the whole chain by itself.

    ``stored_chain_source is None`` (an unchained chunk) is never blocked --
    it does not depend on anyone else's pixels.
    """
    return stored_chain_source is not None and stored_chain_source in rerendered_chunk_ids


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ContinuityError(Exception):
    """Base class for every continuity-bridging failure in this module."""


class FrameExtractionError(ContinuityError):
    """ffprobe/ffmpeg failed to determine or extract a chunk's last frame."""


class ContinuityIndexError(ContinuityError):
    """Two chunk-carrying objects that should refer to the same chunk id
    disagree (a registry key vs. an embedded ``chunk_id`` field, or similar).

    Always names both ids involved -- this project's two prior index-space
    bugs (issue #5's lyric-line/segment mix-up, workflow_graph's
    ``ChunkMismatchError``) only ever surfaced as the wrong face lip-syncing
    the wrong line, long after the render cost was paid. This variant is
    nastier: the seed frame legitimately comes from a *different* chunk id
    (N-1) than the prompt/audio (N), so "the ids differ" is not itself an
    error -- only an internal inconsistency between what a container claims
    about its own chunk id and what the caller looked it up by extension.
    """

    def __init__(self, message: str, *, chunk_id: int, other_id: int) -> None:
        self.chunk_id = chunk_id
        self.other_id = other_id
        super().__init__(f"{message} (chunk_id={chunk_id}, other_id={other_id})")


# --------------------------------------------------------------------------- #
# Last-frame extraction (injectable subprocess seam; mirrors assembly.py)
# --------------------------------------------------------------------------- #

# Injectable subprocess seam: unit tests supply a fake that never touches a
# real shell or a real ffmpeg/ffprobe binary.
SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


def _default_subprocess_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Real ffmpeg/ffprobe invocation. Never used by unit tests -- injected out."""
    return subprocess.run(list(args), capture_output=True, check=False)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", errors="replace")


def probe_frame_count(video_path: Path, runner: SubprocessRunner) -> int:
    """Return the actual decoded frame count of ``video_path`` via ffprobe.

    Raises :class:`FrameExtractionError` (logged) if ffprobe exits non-zero
    or its output isn't parseable as an integer frame count.
    """
    video_path = Path(video_path)
    args = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = runner(args)
    if result.returncode != 0:
        stderr = _decode(result.stderr)
        logger.error(
            "ffprobe frame-count probe failed for %s (exit=%s): %s",
            video_path,
            result.returncode,
            stderr,
        )
        raise FrameExtractionError(
            f"ffprobe failed for {video_path} (exit={result.returncode}): {stderr}"
        )

    stdout = _decode(result.stdout).strip()
    try:
        count = int(stdout)
    except ValueError as exc:
        logger.error("ffprobe returned a non-integer frame count for %s: %r", video_path, stdout)
        raise FrameExtractionError(
            f"ffprobe returned a non-integer frame count for {video_path}: {stdout!r}"
        ) from exc

    return count


def extract_last_frame(
    video_path: Path,
    dest_path: Path,
    *,
    runner: SubprocessRunner,
    expected_length_frames: int | None = None,
) -> Path:
    """Extract the ACTUAL last decoded frame of ``video_path`` to ``dest_path``.

    Always probes the real frame count via :func:`probe_frame_count` and
    extracts that frame -- never an off-by-N guess derived from a requested
    ``length``. If ``expected_length_frames`` is given (the quantized
    ``length`` the chunk was rendered with), a mismatch against the probed
    count is logged as a loud warning but does **not** change which frame
    gets extracted: the file's real content is always the source of truth,
    exactly the bug issue #12 calls out.

    Raises :class:`FrameExtractionError` (logged) if the probe fails, the
    video reports zero/negative frames, or the ffmpeg extraction itself
    fails.
    """
    video_path = Path(video_path)
    dest_path = Path(dest_path)

    actual_count = probe_frame_count(video_path, runner)
    if actual_count <= 0:
        logger.error(
            "video %s reports %d frame(s) -- cannot extract a last frame", video_path, actual_count
        )
        raise FrameExtractionError(f"{video_path} reports {actual_count} frame(s)")

    if expected_length_frames is not None and actual_count != expected_length_frames:
        logger.warning(
            "frame count MISMATCH for %s: expected quantized length=%d frame(s) but ffprobe "
            "counted %d -- extracting the file's ACTUAL last frame (index=%d), not a guess "
            "derived from the requested length",
            video_path,
            expected_length_frames,
            actual_count,
            actual_count - 1,
        )

    last_index = actual_count - 1
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    extract_args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{last_index})",
        "-vframes",
        "1",
        str(dest_path),
    ]
    result = runner(extract_args)
    if result.returncode != 0:
        stderr = _decode(result.stderr)
        logger.error(
            "ffmpeg last-frame extraction failed for %s (exit=%s): %s",
            video_path,
            result.returncode,
            stderr,
        )
        raise FrameExtractionError(
            f"ffmpeg failed to extract last frame from {video_path} "
            f"(exit={result.returncode}): {stderr}"
        )

    logger.info(
        "Extracted last frame (index=%d of %d frames) from %s -> %s",
        last_index,
        actual_count,
        video_path,
        dest_path,
    )
    return dest_path


# --------------------------------------------------------------------------- #
# ContinuityWorkflowProvider
# --------------------------------------------------------------------------- #

DEFAULT_CAST_IMAGE_TITLE = "Load Cast Reference"
"""Matches the baseline/i2v fixtures' cast-photo ``LoadImage`` title exactly."""

DEFAULT_SEED_FRAME_TITLE = "Seed Frame"
"""``WorkflowGraphMutator.mutate``'s ``seed_frame_title`` disambiguator."""

# Cross-lane contract with the resilience lane (issue #10): a workflow
# provider is called immediately before every render attempt. Typed inline
# rather than imported from resilience.py -- that module is being built by
# another lane concurrently and this module must not depend on it.
WorkflowProvider = Callable[[int, RunState], Workflow]


def _hash_template(
    hasher: Callable[[Workflow], str] | None, template: Workflow | None, label: str
) -> str | None:
    """``hasher(template)``, or ``None`` when there is nothing to hash.

    A hashing failure must never kill a run that would otherwise render fine.
    The cost of not having the hash is a chunk that cannot prove which graph
    made it, which is exactly what ``None`` already means to
    :class:`~music_video_maker.contracts.ChunkFingerprint` -- a resumed run
    re-renders it rather than trusting it. So this is logged loudly and
    swallowed."""
    if hasher is None or template is None:
        return None
    try:
        return hasher(template)
    except Exception:
        logger.exception(
            "failed to hash the %s workflow template -- the run continues, but its chunks "
            "cannot prove which graph produced them and --resume will re-render rather than "
            "trust them (issue #45)",
            label,
        )
        return None


class ContinuityWorkflowProvider:
    """A :data:`WorkflowProvider` that bridges chunk N-1's last frame into
    chunk N's I2V render when continuity is enabled and the predecessor
    succeeded, falling back to the base T2V/reference template otherwise.

    Never raises out of a missing/failed/dead-lettered/vanished predecessor,
    or a frame-extraction/upload failure -- those are all degradations,
    logged loudly, that fall back to the base path so one bridging failure
    never kills the run. It *does* raise :class:`ContinuityIndexError` when
    the caller's own bookkeeping is internally inconsistent (a registry key
    disagreeing with the object's own ``chunk_id``) -- that is not a
    transient render failure, it is a sign this module would otherwise wire
    the wrong chunk's data together silently.

    Chaining that bridge down a whole song (issue #28) is bounded by two
    constructor knobs, both defaulting to the pre-#28 behaviour:

    - ``reanchor_interval`` -- break the chain every N chunks so accumulated
      drift is re-anchored against the cast photo and the vocal stem.
    - ``chainable_chunk_ids`` -- which chunks may take the chained path at
      all. The chained node has no audio conditioning, so a chained chunk
      cannot lip-sync; ``None`` means every chunk is eligible and says so in
      a warning.

    :attr:`chain_sources` records what each chunk was actually chained to.
    """

    def __init__(
        self,
        *,
        base_template: Workflow,
        chunk_prompts: Mapping[int, ExpandedPrompt],
        chunk_assets: Mapping[int, StagedAssets],
        asset_stager: AssetStager,
        frames_dir: Path,
        i2v_template: Workflow | None = None,
        continuity_enabled: bool = False,
        mutator: WorkflowMutator | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        cast_image_title: str = DEFAULT_CAST_IMAGE_TITLE,
        seed_frame_title: str = DEFAULT_SEED_FRAME_TITLE,
        expected_length_frames: int | None = None,
        chunk_frame_counts: Mapping[int, int | None] | None = None,
        render_width: int | None = None,
        render_height: int | None = None,
        reanchor_interval: int | None = None,
        chainable_chunk_ids: Collection[int] | None = None,
        noise_seed: int = 0,
        text_encoder: str | None = None,
        lora: str | None = None,
        lora_strength: float = 1.0,
        graph_hasher: Callable[[Workflow], str] | None = None,
        seed_face_gate: Callable[[Path], bool] | None = None,
    ) -> None:
        if reanchor_interval is not None and reanchor_interval < 1:
            logger.error(
                "ContinuityWorkflowProvider: reanchor_interval=%r is meaningless -- pass a "
                "positive number of chunks, or None to never re-anchor",
                reanchor_interval,
            )
            raise ValueError(
                f"reanchor_interval must be >= 1 or None, got {reanchor_interval!r}"
            )

        if continuity_enabled and i2v_template is None:
            logger.error(
                "ContinuityWorkflowProvider: continuity_enabled=True but no i2v_template "
                "given -- the I2V path is a separate authored template, not a mutation of "
                "the base one (see config.i2v_workflow_template)"
            )
            raise ValueError(
                "i2v_template is required when continuity_enabled=True "
                "(config.i2v_workflow_template must be set)"
            )

        self._base_template = base_template
        self._i2v_template = i2v_template
        self._chunk_prompts = chunk_prompts
        self._chunk_assets = chunk_assets
        self._asset_stager = asset_stager
        self._frames_dir = Path(frames_dir)
        self._continuity_enabled = continuity_enabled
        self._mutator = mutator if mutator is not None else WorkflowGraphMutator()
        self._subprocess_runner = subprocess_runner or _default_subprocess_runner
        self._cast_image_title = cast_image_title
        self._seed_frame_title = seed_frame_title
        self._render_width = render_width
        self._render_height = render_height
        self._expected_length_frames = expected_length_frames
        self._chunk_frame_counts = chunk_frame_counts or {}
        self._reanchor_interval = reanchor_interval
        self._chainable_chunk_ids = chainable_chunk_ids
        # Issue #38: this must be the same config value the orchestrator
        # records in each ChunkFingerprint, or run_state.json names a seed
        # the render did not use. The cli wiring test pins both paths.
        self._noise_seed = noise_seed
        # Issue #39: pinned once and injected into *both* templates' own
        # CLIPLoader, so a chained chunk cannot be encoded at a different
        # precision from the unchained one either side of it. None leaves each
        # authored template's own choice alone.
        self._text_encoder = text_encoder
        self._lora = lora
        self._lora_strength = lora_strength
        self._chain_sources: dict[int, int | None] = {}
        # Issue #46: the prompt text each chunk was really submitted with. The
        # two paths are told different things, so what was planned is not
        # necessarily what ran.
        self._prompt_texts: dict[int, str] = {}
        # Issue #47: may this seed frame be chained from at all? None keeps
        # the pre-#47 behaviour of chaining from whatever the last frame was.
        self._seed_face_gate = seed_face_gate
        # Issue #45: which authored graph each chunk rendered through.
        #
        # Hashed once per *template*, not once per mutated chunk graph. The
        # mutated graph is the more tempting thing to hash -- it is literally
        # what was submitted -- but its hash cannot be known before the chunk
        # renders, and ``--resume`` has to compare against something it can
        # compute up front. A per-chunk hash therefore matches nothing on the
        # next run and re-renders the whole song every time, which is the exact
        # opposite of what this field is for.
        #
        # Nothing is lost by hashing the template: the per-chunk inputs the
        # mutator injects are either masked out of the hash anyway (prompt,
        # filenames, seed, length) or already have fingerprint fields of their
        # own (resolution, encoder, character, reference photo).
        self._template_hash_base = _hash_template(graph_hasher, base_template, "base")
        self._template_hash_i2v = _hash_template(graph_hasher, i2v_template, "i2v")
        self._template_hashes: dict[int, str] = {}

        if continuity_enabled:
            logger.info(
                "I2V chaining enabled: reanchor_interval=%s, chainable_chunk_ids=%s",
                reanchor_interval if reanchor_interval is not None else "never",
                "all chunks" if chainable_chunk_ids is None else sorted(chainable_chunk_ids),
            )
        if continuity_enabled and chainable_chunk_ids is None:
            logger.warning(
                "I2V chaining is enabled for EVERY chunk (chainable_chunk_ids=None). The "
                "chained path renders through %s, which has no audio-conditioning input at "
                "all -- no audio_vae and no ref_audios -- so the sliced vocal stem cannot "
                "drive the mouth and a chained chunk will NOT lip-sync. Only the base %s "
                "path is audio-conditioned. Pass chainable_chunk_ids (e.g. the instrumental "
                "chunk ids, or the chunks whose shot plan asks for one continuous move) to "
                "keep lip-synced lines on the base path.",
                CLASS_TYPE_H3_IMAGE_TO_VIDEO,
                CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
            )

    def __call__(self, chunk_id: int, state: RunState) -> Workflow:
        prompt = self._require_prompt(chunk_id)
        assets = self._require_assets(chunk_id)

        if not self._continuity_enabled:
            logger.debug("chunk_id=%s: i2v_continuity disabled -- using base path", chunk_id)
            return self._render_unchained(chunk_id, prompt, assets)

        if is_reanchor_chunk(chunk_id, self._reanchor_interval):
            logger.info(
                "chunk_id=%s: deliberate re-anchor point (every %s chunk(s)) -- rendering "
                "through the base reference path so the cast photo and the vocal stem "
                "re-assert themselves and chained drift stops compounding",
                chunk_id,
                self._reanchor_interval,
            )
            return self._render_unchained(chunk_id, prompt, assets)

        if self._chainable_chunk_ids is not None and chunk_id not in self._chainable_chunk_ids:
            logger.info(
                "chunk_id=%s: not eligible for chaining (not in chainable_chunk_ids) -- "
                "rendering through the audio-conditioned base reference path",
                chunk_id,
            )
            return self._render_unchained(chunk_id, prompt, assets)

        predecessor_id = chunk_id - 1
        predecessor_result = self._resolve_predecessor(chunk_id, predecessor_id, state)
        if predecessor_result is None:
            return self._render_unchained(chunk_id, prompt, assets)

        seed_filename = self._stage_seed_frame(chunk_id, predecessor_id, predecessor_result)
        if seed_filename is None:
            return self._render_unchained(chunk_id, prompt, assets)

        bridged_assets = StagedAssets(
            chunk_id=assets.chunk_id,
            image_filename=assets.image_filename,
            audio_filename=assets.audio_filename,
            seed_frame_filename=seed_filename,
        )
        self._chain_sources[chunk_id] = predecessor_id
        logger.info(
            "chunk_id=%s: chaining from chunk_id=%s (its last frame becomes this chunk's "
            "first frame). This chunk renders on the fl2va path, which carries no audio "
            "conditioning -- it will not lip-sync",
            chunk_id,
            predecessor_id,
        )
        return self._render_i2v(prompt, bridged_assets)

    # -- chain provenance (issue #28) ---------------------------------------- #

    @property
    def chain_sources(self) -> dict[int, int | None]:
        """A copy of ``{chunk_id: source_chunk_id or None}`` for every chunk
        this provider has decided so far.

        This is what *actually happened*, degradations included, as opposed to
        :func:`planned_chain_source`'s configuration-only prediction. A run
        report can read it to say how many boundaries were really chained, and
        the resume path needs it to record on each chunk what its first frame
        came from -- chained and unchained renders of the same span are
        different content.

        A copy, not the live dict: the caller reading this must not be able to
        rewrite what the provider believes it did.
        """
        return dict(self._chain_sources)

    def template_hash(self, chunk_id: int) -> str | None:
        """A hash of the graph ``chunk_id`` actually rendered through (issue
        #45), or ``None`` if this provider has not decided that chunk yet or
        was given no ``graph_hasher``.

        A run holds two templates and a chunk renders through exactly one, so
        the run-level question "which template?" has no single answer and the
        per-chunk one is only answerable here. Recorded at the point the graph
        is built, from the graph itself, rather than re-derived later from the
        template that *would* have been used.
        """
        return self._template_hashes.get(chunk_id)

    def prompt_text(self, chunk_id: int) -> str | None:
        """The prompt text ``chunk_id`` was actually submitted with, or
        ``None`` if this provider has not decided that chunk yet (issue #46).

        Sibling of :meth:`chain_source`, and for the same reason: the chained
        and unchained paths are given different sentences, so a fingerprint
        that hashes the planned prompt would claim a prompt the render did not
        use -- and a chunk that fell back to the base path would be recorded
        under the chained variant, silently blocking or permitting reuse on the
        wrong evidence.
        """
        return self._prompt_texts.get(chunk_id)

    def chain_source(self, chunk_id: int) -> int | None:
        """The chunk ``chunk_id`` was actually chained off, or ``None``.

        ``None`` covers both "rendered unchained" and "not decided yet" --
        the caller for this is the run's own bookkeeping, which only asks
        about chunks the provider has just been called for. Use
        ``chunk_id in provider.chain_sources`` to tell the two apart.
        """
        return self._chain_sources.get(chunk_id)

    # -- lookups with the index-space guard --------------------------------- #

    def _require_prompt(self, chunk_id: int) -> ExpandedPrompt:
        prompt = self._chunk_prompts.get(chunk_id)
        if prompt is None:
            logger.error("no ExpandedPrompt registered for chunk_id=%s", chunk_id)
            raise KeyError(f"no ExpandedPrompt registered for chunk_id={chunk_id}")
        if prompt.chunk_id != chunk_id:
            logger.error(
                "chunk_prompts registry key=%s does not match ExpandedPrompt.chunk_id=%s",
                chunk_id,
                prompt.chunk_id,
            )
            raise ContinuityIndexError(
                "chunk_prompts registry key does not match ExpandedPrompt.chunk_id",
                chunk_id=chunk_id,
                other_id=prompt.chunk_id,
            )
        return prompt

    def _require_assets(self, chunk_id: int) -> StagedAssets:
        assets = self._chunk_assets.get(chunk_id)
        if assets is None:
            logger.error("no StagedAssets registered for chunk_id=%s", chunk_id)
            raise KeyError(f"no StagedAssets registered for chunk_id={chunk_id}")
        if assets.chunk_id != chunk_id:
            logger.error(
                "chunk_assets registry key=%s does not match StagedAssets.chunk_id=%s",
                chunk_id,
                assets.chunk_id,
            )
            raise ContinuityIndexError(
                "chunk_assets registry key does not match StagedAssets.chunk_id",
                chunk_id=chunk_id,
                other_id=assets.chunk_id,
            )
        return assets

    def _resolve_predecessor(
        self, chunk_id: int, predecessor_id: int, state: RunState
    ) -> ChunkResult | None:
        if predecessor_id < 0:
            logger.info(
                "chunk_id=%s: no predecessor chunk (first chunk of the run) -- base T2V path",
                chunk_id,
            )
            return None

        result = state.results.get(predecessor_id)
        if result is None:
            logger.info(
                "chunk_id=%s: predecessor chunk_id=%s has no recorded result yet -- "
                "degrading to base T2V path",
                chunk_id,
                predecessor_id,
            )
            return None

        if result.chunk_id != predecessor_id:
            logger.error(
                "RunState corruption: results[%s].chunk_id=%s -- refusing to trust this as "
                "chunk_id=%s's predecessor (would seed one chunk's video from another "
                "chunk's slot)",
                predecessor_id,
                result.chunk_id,
                chunk_id,
            )
            raise ContinuityIndexError(
                "RunState.results key does not match ChunkResult.chunk_id",
                chunk_id=chunk_id,
                other_id=predecessor_id,
            )

        if not result.succeeded:
            logger.warning(
                "chunk_id=%s: predecessor chunk_id=%s did not succeed (status=%s) -- "
                "degrading to base T2V path",
                chunk_id,
                predecessor_id,
                result.status,
            )
            return None

        if result.video_file is None:
            logger.warning(
                "chunk_id=%s: predecessor chunk_id=%s succeeded but has no video_file "
                "recorded -- degrading to base T2V path",
                chunk_id,
                predecessor_id,
            )
            return None

        if not Path(result.video_file).exists():
            logger.warning(
                "chunk_id=%s: predecessor chunk_id=%s video_file=%s is missing from disk -- "
                "degrading to base T2V path",
                chunk_id,
                predecessor_id,
                result.video_file,
            )
            return None

        return result

    def _seed_frame_carries_identity(
        self, chunk_id: int, predecessor_id: int, frame_path: Path
    ) -> bool:
        """Whether ``frame_path`` can actually convey who the performer is
        (issue #47), or ``True`` when no gate was configured.

        A gate that raises is treated as a refusal, not a crash: the whole
        point is that an *unverified* seed is the case that loses the likeness,
        and falling back to the base reference path costs a slightly different
        shot while a dead run costs hours of GPU custody."""
        if self._seed_face_gate is None:
            return True
        try:
            allowed = self._seed_face_gate(frame_path)
        except Exception:
            logger.exception(
                "chunk_id=%s: the seed-face gate raised on %s -- treating it as unchainable "
                "and rendering through the base reference path (issue #47)",
                chunk_id,
                frame_path,
            )
            return False

        if not allowed:
            logger.warning(
                "chunk_id=%s: predecessor chunk_id=%s ends on a frame with no usable face "
                "(%s) -- NOT chaining. The fl2va path has no ref_images, so that frame "
                "would have been this shot's entire identity conditioning and the "
                "performer's likeness would be invented from nothing. Rendering through "
                "the base reference path instead, which restores both the cast photo and "
                "audio conditioning (issue #47).",
                chunk_id,
                predecessor_id,
                frame_path,
            )
        return allowed

    def _stage_seed_frame(
        self, chunk_id: int, predecessor_id: int, predecessor_result: ChunkResult
    ) -> str | None:
        frame_path = self._frames_dir / f"seed_{predecessor_id:04d}_into_{chunk_id:04d}.png"
        try:
            extract_last_frame(
                Path(predecessor_result.video_file),  # type: ignore[arg-type]
                frame_path,
                runner=self._subprocess_runner,
                expected_length_frames=self._expected_length_frames,
            )
            # Issue #47: gate BEFORE uploading. The chained node has no
            # ref_images, so this frame is the whole identity conditioning --
            # if it does not show a face, the next shot has nothing to build a
            # likeness from and will invent one. Checked here rather than at
            # plan time because it is a fact about rendered pixels, which do
            # not exist until the predecessor has run.
            if not self._seed_frame_carries_identity(chunk_id, predecessor_id, frame_path):
                return None
            seed_filename = self._asset_stager.upload_image(frame_path)
        except Exception:
            # Broad by design: any extraction or upload failure here is a
            # continuity-bridging degradation, not a run-ending failure --
            # requirement #4 is "never crash the run". Logged with full
            # context so the degradation is visible without re-running.
            logger.exception(
                "chunk_id=%s: failed to extract/stage a seed frame from predecessor "
                "chunk_id=%s (video_file=%s) -- degrading to base T2V path",
                chunk_id,
                predecessor_id,
                predecessor_result.video_file,
            )
            return None

        logger.info(
            "chunk_id=%s: staged seed frame from predecessor chunk_id=%s -> %s",
            chunk_id,
            predecessor_id,
            seed_filename,
        )
        return seed_filename

    # -- rendering ------------------------------------------------------------ #

    def _length_frames(self, chunk_id: int) -> int | None:
        """Issue #20's quantized ``length`` for ``chunk_id``, or ``None`` to
        leave the template's own value alone.

        Keyed by chunk id -- the *same* index space as ``_chunk_prompts`` and
        ``_chunk_assets``, and deliberately not positional, since a chunk id
        is not an ``AlignedSegment`` index and this module already guards
        that confusion everywhere else (see ``ContinuityIndexError``). A
        chunk with no recorded frame count logs rather than silently
        rendering at whatever length the template happens to carry."""
        frames = self._chunk_frame_counts.get(chunk_id)
        if frames is None and self._chunk_frame_counts:
            logger.warning(
                "chunk_id=%s: no quantized frame_count recorded -- falling back to the "
                "template's own 'length'. The rendered clip may not match its audio stem "
                "(issue #20).",
                chunk_id,
            )
        return frames

    def _dimension_overrides(self, class_type: str) -> dict[str, dict[str, object]] | None:
        """Per-run ``width``/``height`` for the H3 node, or ``None`` to leave
        whatever the template was authored with.

        Keyed by ``class_type`` so it rides the mutator's existing
        ``node_input_overrides`` seam rather than adding another bespoke
        parameter -- and so the I2V template's differently-named H3 node gets
        the same resolution as the base one. Both dimensions are guaranteed
        present together by config validation."""
        if self._render_width is None or self._render_height is None:
            return None
        return {class_type: {"width": self._render_width, "height": self._render_height}}

    def _render_unchained(
        self, chunk_id: int, prompt: ExpandedPrompt, assets: StagedAssets
    ) -> Workflow:
        """Render through the base reference path and record that this chunk
        carried no seed frame, so the provenance map has an entry for every
        chunk it decided rather than only for the chained ones."""
        self._chain_sources[chunk_id] = None
        return self._render_base(prompt, assets)

    def _for_path(self, prompt: ExpandedPrompt, *, chained: bool) -> ExpandedPrompt:
        """``prompt`` carrying the text variant this path actually renders with
        (issue #46), recorded on the way past.

        The chained path is shown no cast photo -- ``MiniMaxH3ImageToVideo``
        has no ``ref_images`` input -- so its identity comes entirely from the
        seed frame, which is already the appearance clause's own output.
        Restating the clause applies it a second time to its own result, and a
        relative directive compounds once per chained chunk.

        Resolved here rather than in ``expand_prompt`` because only this class
        knows which path a chunk took, and only at render time: a chunk whose
        predecessor dead-lettered falls back to the base path. The text is
        recorded as it is chosen, so the fingerprint can hash what was really
        submitted instead of what was planned."""
        text = prompt.text_for(chained=chained)
        self._prompt_texts[prompt.chunk_id] = text
        if text == prompt.prompt:
            return prompt
        return dc_replace(prompt, prompt=text)

    def _record_graph(self, chunk_id: int, workflow: Workflow, *, chained: bool) -> Workflow:
        """Record which authored template ``chunk_id`` rendered through (issue
        #45), returning ``workflow`` unchanged so this can wrap the build.

        Recorded here, at the point the graph is actually built, rather than
        alongside the plan -- a chunk whose predecessor dead-lettered renders
        through the base template no matter what the plan said, and its
        fingerprint has to say so or ``--resume`` compares it against a graph
        it never used."""
        template_hash = self.planned_template_hash(chained=chained)
        if template_hash is not None:
            self._template_hashes[chunk_id] = template_hash
        return workflow

    def planned_template_hash(self, *, chained: bool) -> str | None:
        """The hash of whichever template the ``chained`` flag selects, or
        ``None`` if it was not hashed (no hasher given, or no I2V template).

        Public because the orchestrator needs the same answer *before* the run
        to build the fingerprint ``--resume`` compares against, and asking the
        provider is what keeps the two from drifting apart -- the mistake this
        field exists to catch is precisely two sides disagreeing about which
        graph produced a chunk."""
        return self._template_hash_i2v if chained else self._template_hash_base

    def _render_base(self, prompt: ExpandedPrompt, assets: StagedAssets) -> Workflow:
        prompt = self._for_path(prompt, chained=False)
        return self._record_graph(
            prompt.chunk_id,
            self._mutator.mutate(
                self._base_template,
                prompt,
                assets,
                length_frames=self._length_frames(prompt.chunk_id),
                noise_seed=self._noise_seed,
                text_encoder=self._text_encoder,
                lora=self._lora,
                lora_strength=self._lora_strength,
                node_input_overrides=self._dimension_overrides(
                    CLASS_TYPE_H3_REFERENCE_TO_VIDEO
                ),
            ),
            chained=False,
        )

    def _render_i2v(self, prompt: ExpandedPrompt, assets: StagedAssets) -> Workflow:
        assert self._i2v_template is not None  # guaranteed by __init__'s validation
        prompt = self._for_path(prompt, chained=True)
        return self._record_graph(
            prompt.chunk_id,
            self._mutator.mutate(
                self._i2v_template,
                prompt,
                assets,
                cast_image_title=self._cast_image_title,
                seed_frame_title=self._seed_frame_title,
                length_frames=self._length_frames(prompt.chunk_id),
                noise_seed=self._noise_seed,
                text_encoder=self._text_encoder,
                lora=self._lora,
                lora_strength=self._lora_strength,
                node_input_overrides=self._dimension_overrides(CLASS_TYPE_H3_IMAGE_TO_VIDEO),
            ),
            chained=True,
        )


__all__ = [
    "DECODE_GROUP_PATTERN",
    "DEFAULT_CAST_IMAGE_TITLE",
    "DEFAULT_SEED_FRAME_TITLE",
    "FPS",
    "LENGTH_STEP_FRAMES",
    "MIN_LENGTH_FRAMES",
    "TRAINED_MAX_FRAMES",
    "TRAINED_MIN_FRAMES",
    "ContinuityError",
    "ContinuityIndexError",
    "ContinuityWorkflowProvider",
    "FrameExtractionError",
    "SubprocessRunner",
    "WorkflowProvider",
    "chain_reuse_blocked",
    "extract_last_frame",
    "frames_to_seconds",
    "is_reanchor_chunk",
    "is_valid_length_frames",
    "planned_chain_source",
    "probe_frame_count",
    "quantize_length_frames",
]
