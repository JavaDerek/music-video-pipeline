"""Frozen data contracts and client protocols shared by every pipeline stage.

This module is the seam that lets the stages be built independently. It is
**append-mostly**: changing an existing field breaks other lanes, so add rather
than mutate, and raise it in review if a change is genuinely needed.

Nothing here may import a stage module, ComfyUI, torch, or pydub -- it must stay
dependency-free so any lane can import it without pulling the world in.

Timeline convention
-------------------
All ``start`` / ``end`` values are **seconds from the beginning of the master
track**, as floats. They originate in Stage 1 forced alignment and must survive
every merge, pad, and split in Stage 2 unchanged in meaning -- Stage 5's mux
correctness depends on chunk boundaries still referring to the original
timeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Cast and lyrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CastMember:
    """One entry in the Cast Dictionary (issue #2)."""

    name: str
    role: str
    """Who this character *is* in the story -- e.g. 'Lead Vocalist, smiling
    constantly, oblivious'. Never what they are doing vocally: the role is
    injected into every chunk, so vocal action here contradicts the
    instrumental clause on every unvoiced chunk (see :mod:`prompting`)."""
    image: Path
    """Local path to the reference photo. Staged to ComfyUI in Stage 3."""
    appearance: str | None = None
    """Deliberate presentation direction for this member (issue #31) -- how
    they should be lit, styled and graded, e.g. 'looking a few years younger,
    softly lit, flattering'.

    A separate field rather than more free text in ``role`` for exactly the
    reason vocal action had to leave ``role``: it applies to *every* chunk the
    member appears in, but it is a different kind of statement from who the
    character is, and mixing kinds in one string is what produced that bug.

    Note the weaker lever: H3 is conditioned primarily on the reference photo,
    so a retouched ``image`` moves the likeness far more than any adjective
    here. ``None`` means no direction, never a fabricated default."""


@dataclass(frozen=True)
class LyricLine:
    """A single line of the lyrics file after character-tag parsing (issue #6)."""

    index: int
    text: str
    """Tag-stripped text. This is what is handed to forced alignment -- tags must
    never reach stable-ts or they pollute timestamps and prompts."""
    characters: tuple[str, ...] = ()
    """Every character audible on this line (issue #33).

    Plural because real songs are: "The Lucky Ones" has Jan alone on some
    lines, Jan and Dianne on the same words on others, and a closing section
    where they sing *different* words at once. A single ``str`` could express
    none of that, and the first full render put the lead's face on every one
    of Jan's lines as a result.

    Empty means "inherit the previously active character", falling back to the
    configured default lead -- the same rule the singular field had.

    Order is significant: the first name is the primary vocalist, which is who
    a renderer falls back to anywhere it can only use one (see
    :attr:`character`). It states who is *audible*, never who is *visible* --
    staging two people on screen is the shot plan's call, not the transcript's.
    """
    raw: str = ""
    """The original line as written, tag included. Diagnostics only."""

    @property
    def character(self) -> str | None:
        """The primary vocalist, or ``None``. Migration seam (issue #33): lets
        every stage that still thinks in one singer keep working unchanged
        while the plural field spreads through the pipeline."""
        return self.characters[0] if self.characters else None


# --------------------------------------------------------------------------- #
# Stage 1: forced alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class AlignedSegment:
    """A logically regrouped segment with word-level timings (issue #3)."""

    index: int
    text: str
    start: float
    end: float
    words: tuple[WordTiming, ...] = ()
    characters: tuple[str, ...] = ()
    """Resolved active characters, carried through from the source LyricLine
    (issue #33). Plural for the same reason."""

    @property
    def character(self) -> str | None:
        """Primary vocalist. Migration seam -- see ``LyricLine.character``."""
        return self.characters[0] if self.characters else None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class AlignmentOverride:
    """An authored correction pinning one aligned segment's span (issue #42).

    Forced alignment is the pipeline's ground truth for *where* the immutable
    lyrics are sung -- but models fail, each differently: on "The Lucky Ones"
    the base model degenerately piled a whole phrase at one instant while
    bigger models placed the same phrase correctly and hallucinated the
    vocoded tail lines into the dead outro instead. When no single model is
    right everywhere, the operator adjudicates -- against stem energy, other
    models, and ears -- and writes the verdict down HERE, as committed run
    data with a mandatory ``reason``. Same philosophy as the shot plan:
    authored once, deterministic, no inference at render time.

    ``segment_index`` is the aligned segment's index (the order align() logs
    them), NOT a chunk id and NOT a lyric line index -- this project's
    recurring index-space bug. Applied before quality evaluation, so the
    quality report judges the corrected timeline. Words are redistributed
    proportionally across the new span: the honest cheap rule, since the
    override knows where the segment sits, not where each word falls.
    """

    segment_index: int
    start: float
    end: float
    reason: str
    """Why this override exists -- logged on every application. Required:
    an unexplained pin is indistinguishable from a stale one after the next
    lyrics edit."""


@dataclass(frozen=True)
class AlignmentResult:
    """Stage 1 output handed to Stage 2."""

    segments: tuple[AlignedSegment, ...]
    track_duration: float
    """Full duration of the master track in seconds."""

    @property
    def voiced_duration(self) -> float:
        return sum(s.duration for s in self.segments)


# --------------------------------------------------------------------------- #
# Stage 2: slicing and prompt expansion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AudioChunk:
    """One renderable unit: a sliced audio stem plus its timeline bookkeeping.

    ``source_segment_indices`` records which :class:`AlignedSegment` indices fed
    this chunk, so merges (several segments -> one chunk) and splits (one segment
    -> several chunks) stay auditable against the original alignment.
    """

    chunk_id: int
    audio_file: Path
    start: float
    end: float
    text: str
    characters: tuple[str, ...] = ()
    """Every character audible in this chunk (issue #33), primary first."""
    concurrent_texts: tuple[str, ...] = ()
    """Issue #33 level 3: one entry per counterpoint stream audible in this
    chunk's span -- text sung *at the same time as* ``text``, in different
    words. Parallel with :attr:`concurrent_characters`. Empty for every chunk
    outside a ``[simultaneously]`` block, and for every pre-#33 chunk."""
    concurrent_characters: tuple[tuple[str, ...], ...] = ()
    """Who sings each entry of :attr:`concurrent_texts` (parallel tuples).
    These voices are also folded into :attr:`characters` -- after the spine
    voice, which keeps the primary slot (#40's dominant-voice rule) -- so
    multi-reference staging picks their faces up without special-casing."""
    source_segment_indices: tuple[int, ...] = ()
    is_split_continuation: bool = False
    """True when this chunk is the 2nd+ piece of a max-duration split (issue #4)."""

    frame_count: int | None = None
    """The exact H3 ``length`` (frame count on :class:`FrameGrid`) this chunk's
    duration was snapped to (issue #20), so Stage 4a injects it directly rather
    than recomputing a frame count from ``duration`` and risking a different
    rounding. ``None`` means the chunk predates frame-grid quantization; a
    caller must then fall back to quantizing ``duration`` itself."""

    is_instrumental: bool = False
    """True for a chunk that covers an unvoiced span of the master track --
    an intro, an outro, or an instrumental break between lyric lines.

    These carry no lyric text and no ``source_segment_indices``; Stage 2b
    already renders them with its instrumental clause (the character on
    screen, performing but not singing) because ``text`` is empty. The flag
    exists so later stages and run reports can distinguish "no lyric here by
    design" from "a lyric chunk that somehow lost its text"."""

    @property
    def character(self) -> str | None:
        """Primary vocalist. Migration seam -- see ``LyricLine.character``."""
        return self.characters[0] if self.characters else None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ExpandedPrompt:
    """Deterministic Stage 2b output for one chunk (issue #5)."""

    chunk_id: int
    prompt: str
    image_ref: Path
    """Local path to the *primary* character's reference photo. Always set --
    H3 needs at least one reference, and every renderer can use this one."""
    image_refs: tuple[Path, ...] = ()
    """Every active character's reference photo, primary first (issue #33).

    ``MiniMaxH3ReferenceToVideo.ref_images`` is a ``COMFY_AUTOGROW_V3`` list,
    so more than one reference is expressible -- whether H3 actually holds two
    likenesses is measured, not assumed. Empty means "just ``image_ref``",
    which keeps every pre-#33 caller correct."""
    characters: tuple[str, ...] = ()
    """Who is *audible* on this chunk, primary first -- never who is visible.
    On-screen-but-silent cast is :attr:`present_cast` (issue #59)."""
    present_cast: tuple[str, ...] = ()
    """Cast staged in this shot without singing it (issue #59), in the order
    they were composed, excluding anyone already in :attr:`characters`.

    Their photos are in :attr:`image_refs` after the singers'; this names them
    separately so ``ChunkFingerprint`` can record who was staged without
    re-deriving it from a path list."""
    chained_prompt: str | None = None
    """The same chunk's prompt as composed for the *chained* I2V path (#46).

    The two render paths are shown different things, so they must be told
    different things. The base reference path attaches the cast photo, and the
    appearance clauses (``CastMember.appearance``, ``global_appearance``) tell
    the model how to read it. The chained path has no photo at all --
    ``MiniMaxH3ImageToVideo`` has no ``ref_images`` input -- and its identity
    comes from ``first_frame``, which is the predecessor's final frame. That
    frame is *already* the output of the appearance clause, so restating the
    clause applies it a second time to its own result, and a relative
    directive ("a few years younger") compounds once per chained chunk.

    Both variants are composed up front because ``expand_prompt`` cannot know
    which path a chunk will take: that is decided at render time by
    ``ContinuityWorkflowProvider``, and it is not final even then -- a chunk
    whose predecessor dead-lettered falls back to the base path. Carrying both
    and choosing at the point of use is what keeps ``prompt_hash`` honest, via
    :meth:`text_for`.

    ``None`` means no chained variant was composed, and :meth:`text_for` falls
    back to :attr:`prompt` -- which is exactly the pre-#46 behaviour, so every
    caller that never chains is unaffected."""

    @property
    def character(self) -> str | None:
        """Primary vocalist. Migration seam -- see ``LyricLine.character``."""
        return self.characters[0] if self.characters else None

    def text_for(self, *, chained: bool) -> str:
        """The prompt text for the path this chunk is actually rendering on.

        One function so that the graph the mutator builds and the
        ``prompt_hash`` the fingerprint records can never disagree about which
        variant was used -- a chunk that falls back to the base path must not
        claim the chained prompt it did not render with, and vice versa.
        """
        if chained and self.chained_prompt is not None:
            return self.chained_prompt
        return self.prompt


# --------------------------------------------------------------------------- #
# Stage 3: asset staging
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StagedAssets:
    """ComfyUI *server-returned* filenames -- never local paths (issue #7)."""

    chunk_id: int
    image_filename: str
    audio_filename: str
    seed_frame_filename: str | None = None
    """Set only when I2V bridging is active and chunk N-1 succeeded (issue #12)."""

    image_filenames: tuple[str, ...] = ()
    """Every staged reference photo's *server-returned* filename, primary first
    (issue #33) -- the staging-side mirror of ``ExpandedPrompt.image_refs``.

    Empty means "just :attr:`image_filename`", which keeps every pre-#33 caller
    correct and keeps the single-reference path -- the one every run to date
    has used -- independent of this field entirely.

    Server-returned, never a local path: ComfyUI renames on collision, and the
    name it hands back is the only one the workflow may reference."""


# --------------------------------------------------------------------------- #
# Stage 4: execution
# --------------------------------------------------------------------------- #


class ChunkStatus(str, Enum):
    PENDING = "pending"
    RENDERED = "rendered"
    CACHED = "cached"
    """Completed in an earlier run and reused via --resume (issue #10)."""
    DEAD_LETTERED = "dead_lettered"
    """Exhausted retries; skipped so the run can finish (issue #10)."""


@dataclass(frozen=True)
class ChunkFingerprint:
    """What a rendered chunk must be able to prove about itself (issue #34).

    ``--resume`` reuses a cached chunk on the strength of its ``chunk_id`` and
    the mp4 still being on disk. That says the chunk *exists*; it does not say
    it is the *right* chunk. Anything that moves the chunk timeline -- editing
    the lyrics file, changing the alignment model or the duration bounds,
    toggling ``instrumental_coverage``, re-trimming the master audio -- leaves
    every mp4 and the whole run state file superficially valid while the spans
    they were rendered for no longer exist. Assembling those in Stage 5 gives a
    silently desynced video with no error anywhere.

    So a chunk records what it was rendered *for*, and a resumed run compares.
    The fields split into two tiers, which deserve different treatment:

    * :data:`TIMELINE_FIELDS` -- **wrong place**. A different span, frame count
      or resolution means the clip plays at a moment it was not rendered for
      (or, for resolution, cannot be concat-copied at all). Never reusable.
    * :data:`CONTENT_FIELDS` -- **right place, wrong content**. The span holds
      but the shot plan, cast role, global style, reference photo or noise seed
      changed. The video stays in sync, it just no longer reflects the config,
      so this tier is escapable via ``resume_ignore_prompt_changes`` --
      re-rendering 39 chunks to pick up a typo fix in one shot line is hours of
      GPU time.

    The prompt is stored as a hash rather than verbatim: the state file is
    diagnostic output a human reads, and 39 full prompts would bury it.
    """

    start: float
    end: float
    frame_count: int | None = None
    render_width: int | None = None
    render_height: int | None = None
    prompt_hash: str | None = None
    character: str | None = None
    image_ref: str | None = None
    """The active cast member's reference photo, as a path string."""
    present_cast: tuple[str, ...] | None = None
    """Who else was staged in this shot without singing it (issue #59), from
    ``ShotPlanEntry.present``.

    Adding or removing a present member changes both the prompt text and the
    set of reference photos the graph conditions on, so a chunk rendered
    before somebody was added must not be reused as though it showed them.
    ``prompt_hash`` happens to move too, but only incidentally -- the photo
    set is the part that would otherwise go unrecorded, and this project has
    now had three separate bugs (#38 seed, #39 encoder, #45 graph) from
    exactly that shape: an input that decides the pixels with nothing writing
    it down.

    Content tier: a chunk with the wrong companion in it covers the right
    span and stays in sync, so it is a *wrong-content* chunk rather than a
    misplaced one, and stays escapable via ``resume_ignore_prompt_changes``
    like every other content edit.

    Computable before the chunk renders (it comes from the shot plan), which
    is the precondition for any field ``--resume`` compares -- otherwise
    every resume re-renders everything. ``None`` means unrecorded (a pre-#59
    state file), never "nobody was present"."""
    noise_seed: int | None = None
    """The ``RandomNoise.noise_seed`` this chunk's graph was submitted with
    (issue #38).

    The last input that decides the pixels, and the one this fingerprint used
    to omit: every render to date used the template's incidental ``0``, nothing
    wrote it down, and a template edit changing it would have sailed through
    ``--resume`` undetected -- a hole in the very contract this class exists to
    close.

    Content tier, deliberately. A re-seeded chunk covers exactly the span,
    frame count and resolution it was rendered for, so it plays where it
    belongs and stays in sync; it is a *different take* of that moment, not a
    misplaced one. That also keeps it escapable, which matters because the
    common workflow is re-rolling one bad chunk -- forcing an inescapable
    re-render of the other 38 would cost hours of GPU time to fix nothing.

    ``None`` means **unrecorded**, never "it was 0": a chunk written before the
    seed was recorded cannot prove which seed produced its pixels, so it must
    not compare equal to one that can."""
    conditioning_source: str | None = None
    """What audio H3 was conditioned on (issue #25): ``"mix"``, or
    ``"stem:<filename>"`` for an isolated vocal stem.

    Its own tier, deliberately -- neither timeline (the chunk is in the right
    place) nor escapable content. Swapping mix for stem is the experiment
    variable of the stem A/B: letting ``resume_ignore_prompt_changes`` reuse
    a mix-conditioned mp4 inside a stem run hands the A/B its control twice
    and calls it a result. ``None`` means unrecorded (a pre-#25 state file),
    which proves nothing and never compares equal to a named source."""
    text_encoder: str | None = None
    """Which text encoder turned the prompt into conditioning (issue #39):
    the ``CLIPLoader.clip_name`` the chunk's graph was submitted with, e.g.
    ``"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"``.

    Recorded on every run, pinned or not, because the encoder is the one
    pixel-deciding input that lives *entirely* in the template file: swapping
    the 4-bit AWQ encoder for the INT8 one changes what H3 understands the
    sentence to mean, and nothing else in the fingerprint moves. Two renders
    differing only in encoder compared equal before this field existed.

    Same tier as ``conditioning_source``, and for the same reason: it is #39's
    experiment variable. The whole point of the precision A/B is to hold the
    span, prompt, seed and stem fixed and change only this, so a resumed run
    that reuses NVFP4 chunks inside an INT8 run scores the control against
    itself. Outside an A/B, the mixed result is a video whose shots were
    comprehended by two different encoders. ``None`` means unrecorded (a
    pre-#39 state file), which proves nothing and never compares equal to a
    named encoder."""

    lora: str | None = None
    """Which adapter was spliced into this chunk's graph (issue #62), and at
    what :attr:`lora_strength`.

    Conditioning tier, with ``text_encoder`` and ``template_hash``, and for
    the same reason: it is an experiment variable. The whole point of an
    adapter A/B is to hold span, prompt, seed and stem fixed and change only
    this, so a resumed run that reuses un-adapted chunks inside an adapted
    run scores the control against itself. Outside an A/B, the mixed result
    is a video whose shots were rendered by two different models.

    It needs its own field because ``template_hash`` cannot see it: that hash
    is taken over the *template*, and the LoRA node is spliced in during
    mutation. Fourth occurrence of the #38/#39/#45 blind spot.

    ``None`` means unrecorded (a pre-#62 state file) or no adapter; both are
    "this chunk was rendered by the base model", which is the same pixels."""

    lora_strength: float | None = None
    """``strength_model`` on that node (issue #62). Recorded separately
    because the same adapter at 0.6 and at 1.0 are different renders, and a
    strength change with the filename unchanged would otherwise be
    invisible."""

    chained_from: int | None = None
    """The chunk id whose last frame supplied this chunk's ``first_frame``
    (issue #28); ``None`` = unchained.

    Content tier: a chained chunk covers exactly the span it was rendered for
    and stays in sync -- a different-looking take, not a misplaced one -- so a
    chained-vs-unchained flip stays escapable like any other content edit.
    What the tier machinery *cannot* express is the pixels themselves: when
    the predecessor re-renders, the footage this chunk was seeded from no
    longer exists, which is why ``ResilientRunner`` additionally refuses to
    reuse a chunk whose ``chained_from`` re-rendered this run (see
    :func:`music_video_maker.continuity.chain_reuse_blocked`)."""

    template_hash: str | None = None
    """A hash of the *graph this chunk actually rendered through* (issue #45),
    with the per-chunk values masked out.

    The fingerprint records every value injected into the graph and never
    recorded the graph itself, so editing a committed template changed the
    pixels and ``--resume`` reused the old chunks anyway. That is the third
    occurrence of one blind spot: the seed (#38) and the encoder (#39) were
    each fixed by naming the specific field; this is the general case
    underneath both, and it covers the next such input without a fourth fix.

    Masked, not whitelisted. The hash covers the graph's shape and constants --
    wiring, sampler, steps, scheduler, shift, VAEs, resolution defaults, which
    ``unet_name`` is loaded -- while the values that legitimately differ per
    chunk (prompt string, image and audio filenames, ``filename_prefix``,
    ``noise_seed``, ``length``) are masked so the hash does not churn. A
    whitelist of "inputs that matter" would reproduce the original bug the
    first time someone adds a node; the default for anything unrecognised must
    be "this counts".

    Conditioning tier, with ``conditioning_source`` and ``text_encoder``: a
    template edit is not a cosmetic prompt tweak, and
    ``resume_ignore_prompt_changes`` must not forgive it. A run uses two
    templates and a chunk renders through exactly one, so this is the hash of
    the one *that chunk* used -- which the provider knows and the orchestrator
    learns through the fingerprint amender.

    ``None`` means unrecorded (a pre-#45 state file), which proves nothing and
    never compares equal to a recorded hash."""

    fallback_template_hash: str | None = None
    """The hash of the template a *degraded* chunk would have used -- i.e. the
    base reference graph. Set only on the **expected** side of a comparison,
    never recorded as evidence about a rendered chunk.

    It exists because ``chained_from`` and ``template_hash`` are two views of
    one fact: which path the chunk took. A run predicts both from config, and
    a render can decline both together -- a dead-lettered predecessor, an
    unextractable frame, or #47's face gate sends the chunk down the base path,
    where it records no chain *and* the base graph's hash. Forgiving the first
    without the second just moves the false mismatch into the conditioning
    tier, which no resume flag can escape.

    Not part of any comparison tier: it is the evidence that makes the
    degradation rule in :meth:`content_differences` safe, not a property of the
    chunk."""

    TIMELINE_FIELDS: ClassVar[tuple[str, ...]] = (
        "start",
        "end",
        "frame_count",
        "render_width",
        "render_height",
    )
    CONTENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "prompt_hash",
        "character",
        "image_ref",
        "present_cast",
        "noise_seed",
        "chained_from",
    )
    CONDITIONING_FIELDS: ClassVar[tuple[str, ...]] = (
        "conditioning_source",
        "text_encoder",
        "template_hash",
        "lora",
        "lora_strength",
    )
    """Fields whose change is never escapable yet not a timeline move: the
    clip is in the right place, but reusing it would corrupt the comparison
    the run exists to make (issues #25, #39, #45). Each names a *component* of
    the conditioning -- the audio that drove the mouth, the encoder that read
    the sentence, and the graph they were fed into."""

    TIME_PRECISION: ClassVar[int] = 3
    """Chunk bounds are rounded to milliseconds before comparison. They are
    floats recomputed from scratch every run, and a shift of 1e-9 s is float
    noise, not a moved chunk -- one frame at 24 fps is 41.7 ms, so millisecond
    precision is still ~40x finer than anything that could desync."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", round(self.start, self.TIME_PRECISION))
        object.__setattr__(self, "end", round(self.end, self.TIME_PRECISION))

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def of(
        cls,
        chunk: AudioChunk,
        prompt: ExpandedPrompt | None = None,
        *,
        render_width: int | None = None,
        render_height: int | None = None,
        noise_seed: int | None = None,
        conditioning_source: str | None = None,
        text_encoder: str | None = None,
        template_hash: str | None = None,
        lora: str | None = None,
        lora_strength: float | None = None,
    ) -> ChunkFingerprint:
        """Build a fingerprint from what the orchestrator already holds.

        ``prompt`` is optional so a caller that has not run Stage 2b (a test
        double, a caller checking timeline identity alone) still gets the
        timeline tier; the content tier is then simply unset and never trips a
        mismatch.

        ``noise_seed`` is the *resolved* seed for this chunk (issue #38) --
        whatever value is actually going into ``RandomNoise.noise_seed`` for
        it, not a per-run base a reader would have to re-derive. Per-chunk
        rather than per-run so varying the seed across chunks or across takes
        stays expressible without touching this contract. Omitted (``None``)
        means the caller manages no seed this run; it is never defaulted to 0,
        which would fabricate the evidence this field exists to demand.
        """
        return cls(
            start=chunk.start,
            end=chunk.end,
            frame_count=chunk.frame_count,
            render_width=render_width,
            render_height=render_height,
            prompt_hash=cls.hash_prompt(prompt.prompt) if prompt is not None else None,
            character=prompt.character if prompt is not None else chunk.character,
            image_ref=str(prompt.image_ref) if prompt is not None else None,
            # Issue #59. Deliberately ``None`` rather than ``()`` when nobody
            # is staged: "no present cast" is the default for every run and
            # every pre-#59 state file, so collapsing the two keeps this field
            # free for anyone not using it. A run that *does* stage somebody
            # records the names, and removing them later reads as the content
            # change it is.
            present_cast=(
                tuple(prompt.present_cast) or None if prompt is not None else None
            ),
            noise_seed=noise_seed,
            conditioning_source=conditioning_source,
            text_encoder=text_encoder,
            lora=lora,
            lora_strength=lora_strength,
            # Issue #45: normally left unset here and amended in at render
            # time, since which of the two templates a chunk used is the
            # provider's decision, not something Stage 2b can predict.
            template_hash=template_hash,
        )

    def _differences(self, other: ChunkFingerprint, names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(name for name in names if getattr(self, name) != getattr(other, name))

    def timeline_differences(self, other: ChunkFingerprint) -> tuple[str, ...]:
        """Fields whose change means the cached video is in the wrong place."""
        return self._differences(other, self.TIMELINE_FIELDS)

    def content_differences(self, other: ChunkFingerprint) -> tuple[str, ...]:
        """Fields whose change means the right span no longer shows the right
        thing, where ``self`` is what this run *expects* and ``other`` is what
        the cached chunk actually recorded.

        One asymmetry, deliberately: **a degradation is not a config change.**
        ``chained_from`` is predicted from configuration before the run, but
        whether a chunk really chains depends on rendered pixels -- the
        predecessor may have dead-lettered, its final frame may not have
        extracted, or the #47 face gate may refuse the seed because that frame
        shows the back of the performer's head. All of those record
        ``chained_from = None`` against a plan that said otherwise.

        Treating that as a mismatch re-renders those chunks on *every* resume,
        forever, because the next run degrades exactly the same way and records
        the same ``None`` -- the run never converges. So "expected a chain, got
        none" is forgiven; the cached chunk is a real render of that span
        through the base path, which is if anything better conditioned.

        The reverse is *not* forgiven. Expecting no chain and finding one, or
        finding a chain from a different chunk, both mean the cached video was
        built from footage this run does not intend.

        The cost is that a chunk which degraded once stays unchained even if
        chaining would now succeed. That is what resuming means -- reuse valid
        work -- and a fresh ``chunks_dir`` gets the chain back.
        """
        differences = self._differences(other, self.CONTENT_FIELDS)
        if self._degraded_to_base(other):
            differences = tuple(name for name in differences if name != "chained_from")
        return differences

    def _degraded_to_base(self, other: ChunkFingerprint) -> bool:
        """``other`` is this chunk rendered through the base path when the plan
        (``self``) expected it to chain -- a degradation, not a config change.

        Requires the stored graph to actually *be* the base one when that is
        known, so this cannot excuse a chunk built from some third template."""
        if self.chained_from is None or other.chained_from is not None:
            return False
        if self.fallback_template_hash is None or other.template_hash is None:
            return True
        return other.template_hash == self.fallback_template_hash

    def conditioning_differences(self, other: ChunkFingerprint) -> tuple[str, ...]:
        """Fields whose change re-renders unconditionally without being a
        timeline move -- see :data:`CONDITIONING_FIELDS`.

        ``template_hash`` gets the same degradation exemption as
        ``chained_from`` (see :meth:`content_differences`), and for the same
        reason: a chunk the render sent down the base path recorded the base
        graph, which is not evidence that anyone edited a template. Only that
        one substitution is forgiven -- a stored hash matching *neither* of
        this run's templates still re-renders, which is the case #45 exists
        for."""
        differences = self._differences(other, self.CONDITIONING_FIELDS)
        if self._degraded_to_base(other):
            differences = tuple(name for name in differences if name != "template_hash")
        return differences


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: int
    status: ChunkStatus
    video_file: Path | None = None
    prompt_id: str | None = None
    attempts: int = 1
    errors: tuple[str, ...] = ()
    render_seconds: float | None = None

    fingerprint: ChunkFingerprint | None = None
    """What this chunk was rendered *for* (issue #34), so a resumed run can
    prove the cached video still belongs where it is about to be placed.
    ``None`` means unrecorded -- which is not evidence of a match, and the
    resilience layer treats it as un-reusable rather than assuming."""

    @property
    def succeeded(self) -> bool:
        return self.status in (ChunkStatus.RENDERED, ChunkStatus.CACHED)


@dataclass(frozen=True)
class FrameGrid:
    """H3's ``length`` quantization grid (issue #20).

    ``MiniMaxH3*ToVideo.length`` is a **frame count**, not a duration, and
    ComfyUI quantizes it as ``min=base_frames, step=step_frames`` -- valid
    values are ``base_frames + step_frames * k``. The step is 17 because the
    temporal VAE decodes each repeating group of five latents into
    ``1+4+4+4+4 = 17`` pixel frames (issue #12's ``1,4,4,4,4`` note and this
    grid are the same rule at two altitudes -- see ``continuity.py``).

    This is a property of the *model*, not of the user's config or their card,
    so it is a constant here rather than a config knob. It lives in
    ``contracts`` because it is the one place both Stage 2a slicing and the
    Level 2 continuity math can reach without importing each other.

    Ground truth: ``docs/h3-node-schema.md``, from a live ``/object_info``.
    """

    fps: int = 24
    base_frames: int = 5
    """``length``'s ``min``. The grid's origin, *not* a usable chunk length."""
    step_frames: int = 17
    trained_min_frames: int = 124
    """~5.167 s. Below this the model extrapolates outside its trained range."""
    trained_max_frames: int = 362
    """~15.083 s. Above this the model extrapolates outside its trained range."""

    def is_valid(self, frames: int) -> bool:
        """True if ``frames`` lands exactly on the grid (ignores trained range)."""
        if frames < self.base_frames:
            return False
        return (frames - self.base_frames) % self.step_frames == 0

    def quantize_up(self, frames: int) -> int:
        """Round ``frames`` UP to the next grid point (never down)."""
        if frames <= self.base_frames:
            return self.base_frames
        steps = -(-(frames - self.base_frames) // self.step_frames)  # ceil division
        return self.base_frames + steps * self.step_frames

    def quantize_nearest(self, frames: int) -> int:
        """Round ``frames`` to the *nearest* grid point, ties rounding up.

        Issue #20's slicer wants nearest, not up: snapping every chunk upward
        would systematically lengthen the video against the master track.
        """
        if frames <= self.base_frames:
            return self.base_frames
        offset = frames - self.base_frames
        steps = (2 * offset + self.step_frames) // (2 * self.step_frames)
        return self.base_frames + steps * self.step_frames

    def clamp_to_trained(self, frames: int) -> int:
        """Clamp ``frames`` into ``[trained_min_frames, trained_max_frames]``.

        Both bounds are themselves grid points, so a grid-valid input stays
        grid-valid.
        """
        return max(self.trained_min_frames, min(self.trained_max_frames, frames))

    def frames_to_seconds(self, frames: int) -> float:
        return frames / self.fps

    def seconds_to_frames(self, seconds: float) -> float:
        """Unrounded frame count for ``seconds``. Callers choose the rounding."""
        return seconds * self.fps


H3_FRAME_GRID = FrameGrid()
"""The MiniMax-H3 grid: 24 fps, ``5 + 17k``, trained range 124-362 frames."""


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware limits (issue #13).

    Deliberately static: a render run takes *exclusive* GPU custody (issue #19),
    so this assumes the full card. Do **not** derive these from live
    ``nvidia-smi`` free-VRAM readings -- that was considered and rejected.
    """

    name: str
    vram_gb: float
    min_chunk_seconds: float = 124 / 24
    """Defaults to H3's trained floor, 124 frames at 24 fps (~5.167s). This
    was 4.0 -- the blueprint's figure, from before the live node schema showed
    the model is only trained from 124 frames (issue #20). Slicing clamps into
    the trained range regardless, but leaving a stale default here made every
    config that simply omitted the field trip that clamp warning."""
    max_chunk_seconds: float = 362 / 24
    """H3's trained ceiling, 362 frames at 24 fps (~15.083s)."""
    recommended_nodes: tuple[str, ...] = ()
    """Optimization node class_types this profile wants present in the workflow."""

    frame_grid: FrameGrid = H3_FRAME_GRID
    """The model's ``length`` grid (issue #20). Not user-configurable -- a
    profile may narrow the *duration* window via ``min_chunk_seconds`` /
    ``max_chunk_seconds``, but the grid itself is fixed by the model."""


@dataclass
class RunState:
    """Resumable per-run state persisted by the resilience layer (issue #10)."""

    run_id: str
    results: dict[int, ChunkResult] = field(default_factory=dict)

    @property
    def dead_lettered(self) -> tuple[int, ...]:
        return tuple(
            cid for cid, r in sorted(self.results.items())
            if r.status is ChunkStatus.DEAD_LETTERED
        )


# --------------------------------------------------------------------------- #
# Client protocols -- the injectable seams that keep every lane offline-testable
# --------------------------------------------------------------------------- #

# A ComfyUI API-format workflow: {node_id: {"class_type": str, "inputs": {...}}}
Workflow = dict[str, dict]


@runtime_checkable
class CastResolver(Protocol):
    """Resolves which cast member is active for a given lyric line (issue #6)."""

    def resolve(self, line_index: int) -> CastMember: ...


@runtime_checkable
class AssetStager(Protocol):
    """Stage 3 upload client (issue #7). Implementations must de-dupe uploads."""

    def upload_image(self, path: Path) -> str: ...
    def upload_audio(self, path: Path) -> str: ...


@runtime_checkable
class WorkflowMutator(Protocol):
    """Stage 4a graph introspection (issue #8).

    Implementations locate nodes by ``class_type``, never by hardcoded node id,
    and deep-copy the template before mutating.
    """

    def mutate(
        self, template: Workflow, prompt: ExpandedPrompt, assets: StagedAssets
    ) -> Workflow: ...


@runtime_checkable
class ExecutionClient(Protocol):
    """Stage 4b submit-and-monitor (issue #9). Event-driven; no sleep-polling."""

    def execute(self, workflow: Workflow, chunk_id: int, output_dir: Path) -> ChunkResult: ...
    def interrupt(self) -> None: ...
    def free(self, *, unload_models: bool = True) -> None: ...


@runtime_checkable
class CustodyManager(Protocol):
    """GPU custody seam (issue #19).

    Acquire asserts the card is actually free; release is unconditional and
    returns the card (``POST /free``). Stopping and restarting whatever else
    uses the GPU is a manual step outside this codebase -- see
    ``custody``'s module docstring.
    """

    def __enter__(self) -> CustodyManager: ...
    def __exit__(self, exc_type, exc, tb) -> bool: ...
