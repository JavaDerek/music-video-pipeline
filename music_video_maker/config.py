"""Run configuration model (issue #2).

Loads the TOML file that parameterizes an end-to-end pipeline run -- master
audio, lyrics, narrative framing, the cast dictionary, ComfyUI connection
details, the workflow template, and output locations -- and validates it
fast and loudly before any stage touches it.

Schema (all paths may be absolute or relative; relative paths resolve
against the *config file's* directory, not the process cwd)::

    master_audio       = "audio/master.wav"
    lyrics_file         = "lyrics.txt"
    global_style        = "Refestramus progressive rock music video, gently comic tone"
    cinematography      = "35mm film, shallow depth of field, warm natural light"  # issue #53
    narrative_concept   = "Wandering through a surgery, kicking a life support plug out"
    setting             = "London, UK -- contemporary, overcast winter"   # issue #32
    global_appearance   = "everyone slim, trim and healthy looking"       # issue #31
    global_demeanour    = "nobody smiles; this is a war"                  # issue #74
    avoid               = ["bass guitar", "microphone", "stage"]          # issue #73
    default_lead_vocalist = "Dianne"
    comfyui_url         = "http://127.0.0.1:8188"     # optional, this is the default
    workflow_template   = "workflow_api.json"
    chunks_dir          = "output/chunks"
    final_video_dir     = "output/final"

    # Wave 3, all optional (defaults shown):
    watchdog_timeout_seconds = 900.0     # issue #10
    max_render_attempts      = 3         # issue #10
    retry_backoff_seconds    = 5.0       # issue #10
    min_free_disk_gb         = 20.0      # issue #10 pre-flight
    min_free_vram_gb         = 16.0      # lowest free-VRAM figure ever proven; see RunConfig
    run_state_file           = "output/chunks/run_state.json"   # issue #10
    render_width             = 864       # optional; both or neither, multiples of 32
    render_height            = 480       # biggest lever on run time -- see RunConfig
    instrumental_coverage    = true      # render the unvoiced spans too
    i2v_continuity           = false     # issue #12
    i2v_workflow_template    = "workflow_i2v_api.json"          # issue #12
    resume_ignore_prompt_changes = false # issue #34; --ignore-prompt-changes overrides

    strict_alignment         = false     # issue #35; --strict-alignment overrides

    # Issue #39, optional. A file in ComfyUI's own models/text_encoders/ on the
    # render host -- NOT a local path. Unset = whatever the template names.
    text_encoder = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

    [cast.Dianne]
    role  = "Lead Vocalist, smiling constantly, oblivious"
    image = "cast/dianne_ref_01.jpg"
    appearance = "looking a few years younger, softly lit"   # optional, issue #31
    demeanour  = "grave and unsmiling, exhausted"             # optional, issue #74

    [cast.Rex]
    role  = "Drummer, background, never sings"
    image = "cast/rex_ref.jpg"

    [hardware]
    name = "RTX 4090"
    vram_gb = 24.0
    min_chunk_seconds = 4.0      # optional, defaults per contracts.HardwareProfile
    max_chunk_seconds = 15.0     # optional
    recommended_nodes = ["SageAttention"]   # optional

A cast member is not required to ever be the active vocalist -- background
members (e.g. a drummer who appears in shots but never sings) are simply
entries in ``cast`` that are never ``default_lead_vocalist`` and never named
as a chunk's active character downstream. Nothing here requires every cast
entry to be "used" as a vocalist.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NoReturn
from urllib.parse import urlparse

from music_video_maker.contracts import AlignmentOverride, CastMember, HardwareProfile
from music_video_maker.faces import (
    DEFAULT_MIN_FACE_FRACTION,
    DEFAULT_MIN_FACE_SIMILARITY,
    RECOGNITION_MODEL_SHA256,
    RECOGNITION_MODEL_SOURCE,
    resolve_recognition_model_path,
)

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

MAX_NOISE_SEED = 0xFFFFFFFFFFFFFFFF
"""``RandomNoise.noise_seed``'s upper bound in ComfyUI (issue #38). Mirrors
:data:`music_video_maker.workflow_graph.MAX_NOISE_SEED` (a test asserts the
two stay equal) -- config must not import the mutator just for a constant."""

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
"""Co-located default (issue #50): the orchestrator and ComfyUI on the same
host, no network hop required. A remote ComfyUI host -- e.g. reached over
Tailscale, never a raw LAN IP -- is a config override, not the assumption."""

ALIGNMENT_MODEL_SIZES = frozenset(
    {
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
    }
)
"""Whisper model names stable-ts will load. Validated at config load time
because the alternative is a stack trace from inside stable-ts *after* the
weights download, which on a first run is several minutes in."""

DEFAULT_ALIGNMENT_MODEL_SIZE = "base"
"""Mirrors :data:`music_video_maker.alignment.DEFAULT_MODEL_SIZE` (a test
asserts the two stay equal) -- config must not import the alignment module
just for a constant, the same rule :data:`MAX_NOISE_SEED` follows.

"base" and not the better-measured "small" on purpose: this key exists so a
run can *choose*, and changing the default would silently re-cut the timeline
of every config already committed."""

_PATH_FIELDS = ("master_audio", "lyrics_file", "workflow_template", "chunks_dir", "final_video_dir")
_SCALAR_FIELDS = ("global_style", "narrative_concept", "default_lead_vocalist")


class ConfigError(Exception):
    """Raised when a run config file fails to parse or validate."""


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved and validated parameterization of one pipeline run."""

    master_audio: Path
    lyrics_file: Path
    global_style: str
    narrative_concept: str
    cast: dict[str, CastMember]
    default_lead_vocalist: str
    comfyui_url: str
    workflow_template: Path
    chunks_dir: Path
    final_video_dir: Path
    hardware: HardwareProfile

    # -- Wave 3 seams. All optional with defaults, so configs written before
    #    Wave 3 keep loading unchanged. ----------------------------------- #

    watchdog_timeout_seconds: float = 900.0
    """Issue #10: no ``progress``/``executing`` WebSocket signal within this
    window means the backend is assumed dead and recovery begins."""

    max_render_attempts: int = 3
    """Issue #10: total attempts per chunk, including the first. Exhausting
    them dead-letters the chunk so the run can still finish."""

    retry_backoff_seconds: float = 5.0
    """Issue #10: base backoff between render attempts."""

    min_free_disk_gb: float = 20.0
    """Issue #10 pre-flight: doris runs high-90s %% full, so a run that would
    start with less than this much free space is refused up front rather than
    hitting ENOSPC dozens of chunks in."""

    run_state_file: Path | None = None
    """Issue #10: where the resumable :class:`~contracts.RunState` is
    persisted. :func:`load_config` always resolves this (defaulting to
    ``chunks_dir/run_state.json``); it is ``None`` only on a hand-constructed
    :class:`RunConfig`."""

    shot_plan: Path | None = None
    """Optional authored per-chunk narrative direction (see
    :mod:`music_video_maker.shot_plan`).

    Without it every chunk is prompted with the same ``narrative_concept``,
    which gives a video with no progression. The plan is authored once and
    committed as data -- the render loop never calls an LLM."""

    vocal_stem: Path | None = None
    """Issue #25: optional isolated vocal stem used as H3's *conditioning*
    audio instead of the full mix.

    Conditioning only: Stage 5 still muxes the pristine master, and the stem
    is cut at exactly the spans Stage 2a computed from the master (see
    :mod:`music_video_maker.stems`). The file is produced by the operator off
    the render path (e.g. ``python -m demucs --two-stems=vocals``) and
    inspected before use -- never an inference call inside the pipeline.
    ``None`` means today's behaviour: condition on the mix."""

    text_encoder: str | None = None
    """Issue #39: the ``CLIPLoader.clip_name`` every chunk of this run is
    encoded with -- e.g. ``"qwen3vl_32b_minimax_h3_int8_convrot.safetensors"``.

    A *server-side* name, resolved by ComfyUI inside its own
    ``models/text_encoders/`` on doris; never a path on the machine running
    this pipeline, and so never resolved against the config file's directory
    the way every real path field is.

    ``None`` (the default) leaves whatever the authored templates name, which
    is what every run to date has done. Naming one makes the encoder a
    run-level choice instead of a template constant -- the precision A/B this
    issue exists for swaps 4-bit AWQ for INT8 with the span, prompt, seed and
    conditioning stem all held fixed, and hand-editing a committed template to
    do that leaves the run with no record of which encoder made the pixels.
    Applies to both templates: each carries its own ``CLIPLoader``, and a pin
    that reached only the base path would give a mixed-encoder video."""

    face_treatment: str = "flattering"
    """Issue #62: whether this video wants faces **flattered** or **real**.

    One switch, because it is one decision, and it is a decision per *song*
    rather than per project. Derek, settling it after the v10 A/B: "for this
    song I want flattery, because I know that's what Dianne will want. For
    Storms, where the lead character is me, I want realistic."

    ``"flattering"`` (the default) is every render to date, including
    ``final_v10``, the current keeper: no adapter, and whatever
    :attr:`global_appearance` and ``CastMember.appearance`` ask for.

    ``"realistic"`` selects the realism adapter, its measured strength and
    its trigger word -- :data:`REALISM_LORA` and friends -- so the flag is
    genuinely one line rather than a label beside three others. Every one of
    those is still overridable explicitly: the preset is a default, never a
    lock, so a future adapter needs no code change to try.

    It deliberately does **not** silently strip the appearance fields when
    set to ``"realistic"``. Discarding config a human wrote is worse than the
    conflict it would avoid, and the fields are not always flattering --
    "weathered, unslept" is appearance direction that a realism adapter
    agrees with. What it does instead is *warn*, once, where the decision is
    made. See :func:`_warn_on_face_treatment_conflict`."""

    lora: str | None = None
    """Issue #62: a ``LoraLoaderModelOnly.lora_name`` spliced into both
    templates -- e.g. ``"h3-realism-people-t2v-i2v-r2v.safetensors"``.

    A *server-side* name resolved by ComfyUI inside its own ``models/loras/``
    on doris, exactly like :attr:`text_encoder` and for the same reason;
    never a path on the machine running this pipeline.

    ``None`` (the default) inserts no node at all, so the graph POSTed is
    byte-identical to every run before this field existed."""

    lora_strength: float = 1.0
    """Issue #62: ``strength_model`` on that node. The fal adapter documents
    1.0 as intended and 0.6-0.8 as "a lighter touch"; the lighter end is
    worth trying first here because a *realism* adapter is a plausible fight
    with a ``global_style`` that asks for a gently comic tone.

    Ignored when :attr:`lora` is ``None``, and recorded in the fingerprint
    either way -- strength decides the pixels as surely as the filename
    does."""

    lora_trigger: str | None = None
    """Issue #62: the adapter's trigger word (``"r34l1sm"`` for the fal
    realism LoRA), composed into every prompt as a leading style token.

    Its own field rather than something to bury in ``global_style``, so it
    cannot be set without a LoRA to trigger or left behind when one is
    removed -- the same "a property that must hold needs its own field" rule
    the setting and appearance fields exist under. Composed at the *front*,
    in the style position, deliberately: a trigger word is a tag, and giving
    it a sentence of its own would put it in the grammatical subject slot
    that ``docs/shot-writing-guide.md`` reserves for what the shot is about."""

    noise_seed: int = 0
    """Issue #38: the run-wide base seed each chunk of this run derives its
    own ``RandomNoise.noise_seed`` from -- see
    :func:`music_video_maker.workflow_graph.resolve_chunk_seed`, which adds
    the chunk id (and, for ``cli.py``'s ``--reseed``, a generation offset) so
    chunks differ deterministically instead of all rendering under this one
    value. The *derived* per-chunk value, not this field verbatim, is what
    reaches ``RandomNoise`` and what each chunk's fingerprint records.

    0 is what both committed templates have always incidentally carried, so
    chunk 0 of the default run renders byte-for-byte what this project has
    always rendered -- but the value now comes from the run and is written
    down, instead of being whatever the template file happened to hold.
    Bounded to ComfyUI's ``0 .. 2**64-1`` at load: an out-of-range value
    would be clamped server-side and the recorded seed would be a lie.

    A seed does not transport a composition across a change in precision,
    GPU, attention kernel, or resolution -- diffusion sampling is chaotic
    with respect to all four, so the same derived number on different
    numerics is a different video, not a higher-fidelity version of one."""

    alignment_overrides: tuple[AlignmentOverride, ...] = ()
    """Issue #42: authored corrections pinning specific aligned segments'
    spans, written as ``[[alignment_override]]`` tables with a mandatory
    ``reason``. The escape hatch for a model that provably mis-places a
    phrase when no single model is right everywhere -- adjudicated by the
    operator against stem energy, other models, and ears, then committed as
    run data. See :class:`~music_video_maker.contracts.AlignmentOverride`."""

    instrumental_shot_seconds: float | None = None
    """Issue #27: the longest an *instrumental filler* chunk may run.

    ``None`` means the same ceiling sung chunks get (``max_chunk_seconds``)
    -- today's behaviour. Given its own value it becomes the editorial
    lever: sung shots want to stay short and break near lyric boundaries,
    while a 30s solo becomes four scene changes when it should be one
    continuous take. Slicing clamps it into H3's trained range with a
    warning; above 141 frames is unmeasured territory on this card and is
    warned about, not gated."""

    render_width: int | None = None
    render_height: int | None = None
    """Override the H3 node's ``width``/``height`` for this run.

    Resolution is a run-level choice, not a template constant: the same
    authored graph is rendered small to preview a shot plan cheaply and large
    for the finished video. It is also the single biggest lever on run time,
    because cost scales with the latent volume -- 1344x768 measured at
    ~9.25 min/chunk on the 4090 against ~3 min at 864x480, a 3.1x difference
    that tracks the 2.8x pixel-volume ratio almost exactly.

    ``None`` (both) leaves whatever the template was authored with. Both must
    be set together -- see :func:`_render_dimensions`. Values must land on
    H3's declared 32-pixel grid (``docs/h3-node-schema.md``: min 32, max
    16384, step 32)."""

    setting: str | None = None
    """Issue #32: where and when the whole video takes place, e.g.
    ``"London, UK -- contemporary, overcast winter"``.

    Composed into *every* prompt, so geography cannot drift between shots. It
    lives alongside ``global_style`` rather than inside it for the same reason
    appearance does not live in ``role``: it is a distinct kind of statement
    and deserves to be set and validated on its own.

    ``None`` is legal but warned about at load. The first full render walked
    out of a British front door into an American park -- terraced street and
    wheelie bins in one shot, Central Park and US emergency vehicles in the
    next -- because nothing named a place and H3 picked per shot from whatever
    each description implied. Silence is what produced the drift."""

    global_appearance: str | None = None
    """Issue #31: presentation direction applying to the whole cast, e.g.
    ``"everyone slim, trim and healthy looking"``.

    The per-member equivalent is :attr:`contracts.CastMember.appearance`. Both
    are separate from ``global_style`` (which is about the film, not the
    people) and from ``role`` (which is about who a character is)."""

    global_demeanour: str | None = None
    """Issue #74: expression/mood direction applying to the whole cast, e.g.
    ``"nobody smiles; this is a war"``.

    Follows the ``global_appearance``/``CastMember.appearance`` precedent
    (#31) exactly, one field over: two of the three cast-facing fields
    (``appearance``, ``global_appearance``) push toward a pleasant, relaxed
    face and none of them says anything about mood, so on a real render
    ("Deathless") the reference photo's own smile silently carried across
    eight and a half minutes of a video about massacre. This is the fix --
    composed into every prompt exactly where ``global_appearance`` is (see
    ``prompting.py``'s ``_single_character_clause``/``_multi_character_clause``).

    The per-member equivalent is
    :attr:`~music_video_maker.contracts.CastMember.demeanour`, matching
    :attr:`~music_video_maker.contracts.CastMember.appearance`'s own shape
    (#31's precedent, followed exactly).

    **State an endpoint, never a prohibition or a displacement.** Two
    findings drive this, both hard-won on this project:

    - *Prohibition does not hold.* #73 measured this directly: the corrected
      role ``"...never holding anything"`` rendered a bass guitar anyway,
      twice, because the tokens a diffusion prompt actually sees are
      "holding" and "anything" -- the negation itself carries no weight.
      ``"never amused"`` is the same shape and is expected to fail the same
      way. Say what IS true instead: ``"grave and unsmiling"``, not
      ``"never amused"``.
    - *Displacement compounds on the chained path* (#46). ``"looking a few
      years younger"`` reapplies to a frame that is already its own output,
      once per chained chunk, and drifts further each time. ``"increasingly
      grim"`` is the same shape. ``"grave, unsmiling"`` is a fixed point:
      restating it against a frame that already looks that way changes
      nothing, which is what makes it safe to restate every chunk.

    :func:`load_config` warns (never refuses -- a false positive here costs
    nothing but a log line) when a demeanour value matches either an
    unenforceable-prohibition pattern or a compounding-displacement pattern.

    Unlike ``appearance``, demeanour is **not** stripped from the chained I2V
    prompt variant (issue #46's ``include_appearance=False``): appearance
    describes how to read a *photo*, which the chained path has none of, but
    demeanour is behavioural direction the same shape as ``role`` -- it says
    what the character is like, not how to interpret a reference image -- so
    it stays composed on both paths, restated every chunk exactly like
    ``role`` always has been. ``None`` means no direction, never a
    fabricated default: the reference photo's own expression decides, as it
    always has."""

    avoid: tuple[str, ...] = ()
    """Issue #73: things this video must never depict. **Read by the
    authoring layer, never composed into a render prompt.**

    It was composed into every render prompt for exactly one full render of
    "Deathless", and it manufactured what it forbids. Measured as a
    single-variable A/B at an identical seed, one clause different:

    * chunk 72, whose shot has no climbing in it and whose character's
      ``role`` has no climbing in it -- WITH ``avoid = [..., "modern climbing
      equipment, ropes, harnesses or safety gear"]`` H3 rendered a full
      climbing harness, carabiners and a trailing rope on him. WITHOUT it,
      plain trousers, with the framing, background and both characters
      otherwise unchanged.
    * chunk 53, where the shot already has her climbing on a rope, was
      unaffected by the same clause.

    ``MiniMaxH3ReferenceToVideo`` exposes one ``prompt`` input into a single
    ``BasicGuider``. There is no negative-conditioning channel, so a
    prohibition cannot subtract; it can only add its own nouns to the only
    channel there is. Where the scene supplies no other source for those
    nouns, the prohibition becomes the source -- which is why it does damage
    precisely where it has nothing to suppress, and why one chunk alone reads
    as "inert".

    The authoring-side ``avoid`` (concept -> beats/prose) is untouched and
    still worth setting: a model reading English does treat it as an
    instruction. This field stays so a run can carry that intent to the
    authoring layer -- but nothing downstream of Stage 2b may compose it.
    """

    cinematography: str | None = None
    """Issue #53: the whole-video film-direction language -- stock, lens,
    depth of field, lighting philosophy, grade -- e.g. ``"35mm film, shallow
    depth of field, warm natural light"``.

    Before this field existed, that language had nowhere to live except
    ``global_style``, which quietly became a junk drawer for six different
    kinds of statement (band/genre, film stock, lighting, lens, tone, and
    editing continuity, all in one string). ``global_style`` is meant to
    narrow back to genre and tone; this is the film's *look*.

    Composed into every prompt at a fixed position, right after
    ``global_style`` -- see ``prompting.py``'s ``_compose_prompt``. Both
    fields are optional and independent, so a config written before this
    field existed keeps loading and rendering unchanged; ``None`` composes
    nothing. Camera *movement and framing* is a different, per-shot kind of
    statement and belongs in :attr:`music_video_maker.shot_plan.ShotPlanEntry.camera`
    instead -- see that field's docstring for why sentence position matters
    there in a way it does not here."""

    between_chunk_min_free_vram_gb: float | None = None
    """Issue #23: optional free-VRAM floor re-checked *between* chunks.

    ``None`` (the default) means the between-chunk reading is logged and never
    gates the run. This is deliberately NOT ``min_free_vram_gb``: that floor
    describes a cold card before anything is loaded, while mid-render ComfyUI
    holds H3's 19995 MB staged and does not unload between prompts, so a
    healthy reading on the 4090 is 1.5-5 GB. Measured on doris 2026-08-08:
    20.72 GB free before chunk 0, 1.54 GB before chunk 1 of the same run.
    Equating the two aborted the run after its first chunk.

    Set it only to a figure you have watched your own card hold steadily
    mid-render. Even then it is a blunt instrument: our own workload swings
    several GB between sampling and VAE decode, so a threshold tight enough to
    catch a 2 GB intruder will fire on us instead."""

    alignment_model_size: str = DEFAULT_ALIGNMENT_MODEL_SIZE
    """Which whisper model Stage 1's forced alignment loads.

    ``align()`` always took a ``model_size``, but nothing ever passed it, so
    every run was pinned to ``"base"`` and the only remedy for a bad alignment
    was hand-authored ``[[alignment_override]]`` tables (issue #42) -- one per
    segment, which stops scaling the moment more than a handful are wrong.

    Measured on "Deathless" (8:32, heavily instrumented), same audio and
    lyrics, only this value differing:

    ==========  ========  ==============  =================
    model       segments  voiced total    last vocal placed
    ==========  ========  ==============  =================
    ``base``    49        77.2s (15.1%)   3:11
    ``small``   57        187.2s (36.6%)  8:21
    ``medium``  --        see below       --
    ==========  ========  ==============  =================

    ``base`` believed the singing stopped at 3:11 of an 8:32 track, which would
    have rendered 56 of 71 chunks as "performs silently, no lyric to sing"
    against a master that has her audibly singing -- a video-destroying result
    that raises no error and is only visible after hours of GPU time.

    Bigger is **not** automatically better, which is why this is a per-run
    value and not a new default. On "The Lucky Ones" both ``small`` and
    ``medium`` hallucinated the vocoded tail lines into the dead outro where
    ``base`` was correct (issue #42); on "Deathless" ``medium`` placed a lyric
    17s before any vocal energy exists and dumped two closing-refrain segments
    into the trailing silence. Measure per song; the alignment quality report
    (issue #35) is what to measure with.

    Defaults to ``"base"`` so every config written before this key existed
    aligns exactly as it always did."""

    strict_alignment: bool = False
    """Issue #35: fail the run when the alignment-quality check finds problems
    above its thresholds, instead of only reporting them.

    Off by default, deliberately: a real song always has some odd segments, so
    refusing by default would be wrong. Turn it on for an unattended long run,
    where the alternative is spending ~2.9 h of GPU time on a timeline that
    was already known to be broken in the ~6 s alignment takes."""

    resume_ignore_prompt_changes: bool = False
    """Issue #34: on ``--resume``, reuse a cached chunk whose span is unchanged
    but whose *prompt* is not -- an edited shot plan, cast role, or global
    style.

    Off by default, so a resumed run reflects the config it was given. Turn it
    on when the change is cosmetic and a full re-render is not worth the GPU
    hours: at ~3.7 min/chunk, picking up a typo fix in one shot line costs
    ~2.4 h across a 39-chunk song.

    It covers the *content* tier only. A chunk whose span, frame count or
    render resolution moved is re-rendered regardless -- that is a desync (or,
    for resolution, a concat that Stage 5 cannot copy), not a cosmetic
    difference, and no flag makes it reusable."""

    instrumental_coverage: bool = True
    """Render the unvoiced spans too -- intro, outro, and every instrumental
    break -- so the chunk timeline covers the whole track contiguously.

    On by default because the alternative is silently broken: Stage 5 concats
    chunks end to end, so any unrendered span is squeezed out of the video
    while the muxed master audio keeps it. A typical song is under half
    vocals, which means a short video whose every chunk after the first
    instrumental plays against the wrong moment of the song. Turn it off only
    to reproduce the old voiced-spans-only behaviour."""

    i2v_continuity: bool = False
    """Issue #12: enable seed-and-feed I2V bridging between chunks."""

    i2v_workflow_template: Path | None = None
    """Issue #12: the separate I2V-path template used for every chunk after a
    successful predecessor. Required when ``i2v_continuity`` is set -- the I2V
    graph is a different node topology (``MiniMaxH3ImageToVideo``), not a
    tweak of the reference-to-video one, so it is a second authored template
    rather than a rewrite of the first."""

    i2v_reanchor_interval: int | None = None
    """Issue #28: break the chain every N chunks (``chunk_id %% N == 0``) so
    accumulated drift is re-anchored against the cast photo and the vocal
    stem. ``None`` never re-anchors deliberately; the chain still breaks
    wherever a chunk is ineligible or a predecessor degrades."""

    i2v_require_seed_face: bool = True
    """Issue #47: refuse to chain from a seed frame with no usable face in it.

    The chained node has no ``ref_images``, so the seed frame is the *entire*
    identity conditioning. When the previous shot ends on the back of the
    performer's head, the next shot has nothing to build a likeness from and
    invents one -- which is what made the lead visibly become a different
    person mid-video. Such a chunk falls back to the base reference path, where
    the cast photo is a real reference and the vocal stem is back.

    On by default: chaining from a faceless frame has no upside, and the cost
    of the check is one small local detection per boundary. Set ``false`` to
    restore the pre-#47 behaviour (or to run without OpenCV installed)."""

    i2v_min_seed_face_fraction: float = DEFAULT_MIN_FACE_FRACTION
    """Issue #47: minimum face area, as a fraction of the frame, for a seed
    frame to count as carrying identity. See
    :data:`music_video_maker.faces.DEFAULT_MIN_FACE_FRACTION` for how the
    default was calibrated against real seed frames."""

    i2v_min_seed_face_similarity: float | None = None
    """Issue #49: minimum SFace cosine similarity between a seed frame's
    largest face and the active cast member's reference photo, for that frame
    to be trusted as *her* face rather than merely *a* face -- a third floor
    on top of :attr:`i2v_min_seed_face_fraction`'s area check and
    :func:`music_video_maker.faces.detect_faces`'s own confidence floor.

    ``None`` (the default) means recognition is OFF: the gate behaves exactly
    as issue #47 shipped it, detection-only. This is deliberately **not**
    defaulted to the calibrated
    :data:`music_video_maker.faces.DEFAULT_MIN_FACE_SIMILARITY`, even though
    that value is ready to use: doing so would silently change the behaviour
    of every config written before this key existed. The SFace weights this
    check needs are 38.7 MB and deliberately not committed to this repository
    (issue #51, ``docs/seed-face-recognition.md``), so defaulting it on would
    make every pre-existing config with ``i2v_continuity`` enabled start
    refusing every chain boundary the next time it runs, until its operator
    fetches a model they never asked for. Same "a new knob leaves every
    pre-existing config byte-identical" rule :attr:`alignment_model_size`
    follows.

    An operator opts in explicitly by setting this to
    :data:`~music_video_maker.faces.DEFAULT_MIN_FACE_SIMILARITY` (0.34, the
    measured recommendation) or their own value -- see that constant's
    docstring and ``docs/seed-face-recognition.md`` for the calibration.
    Once set, :func:`load_config` refuses to start a run that cannot honour
    it: see the SFace pre-flight check inside :func:`_validate`. A checkout
    with no model on disk must fail loudly before any GPU time is spent, not
    hand the operator a silently weaker detection-only gate while reporting
    success -- the run-time per-frame failure path (`degrade, never raise`)
    is for *unpredictable* things like an unreadable frame or a reference
    photo with no detectable face, not for a model file the operator could
    have checked at load time."""

    i2v_chain_scope: str = "instrumental"
    """Issue #28: which chunks may take the chained (fl2va) path when
    ``i2v_continuity`` is on -- ``"instrumental"``, ``"all"`` or ``"none"``.

    The chained node has NO audio-conditioning input, so a chained chunk
    cannot lip-sync. ``"instrumental"`` (the default) chains exactly where
    that trade is free: seamless boundaries through solos and intros, every
    lyric line still on the audio-conditioned base path. ``"all"`` chains
    everything and warns about the consequence; ``"none"`` scopes chaining
    off entirely."""

    # -- Issue #19 GPU-custody seam. All optional with defaults. --- #

    min_free_vram_gb: float = 16.0
    """Issue #19 custody pre-flight: the absolute free-VRAM floor, in GB, a
    run must clear before rendering starts.

    Set to **the lowest free-VRAM figure at which a render has ever
    demonstrably succeeded** -- ~16.4 GB, measured repeatedly on doris with
    the desktop session and other GPU services resident. Everything below that
    is untested; everything at or above it is proven. See
    ``custody.DEFAULT_MIN_FREE_VRAM_GB`` for the full reasoning, including why
    H3's 19995 MB staging figure is *not* the floor.

    This was 12.0 until 2026-08-07 -- below anything ever demonstrated -- and
    it green-lit a run onto a contended card that then went silent rather than
    raising CUDA OOM, wedging the host.

    Note this is a *point-in-time* check. It cannot see another process
    claiming the card mid-run, which is exactly how that incident started:
    doris shares one 4090 between four independent workloads. Confirm the card
    is genuinely clear before a long run; stopping the one tenant you know
    about is not the same as freeing the card."""

    i2v_continuity: bool = False
    """Issue #12: enable seed-and-feed I2V bridging between chunks."""

    i2v_workflow_template: Path | None = None
    """Issue #12: the separate I2V-path template used for every chunk after a
    successful predecessor. Required when ``i2v_continuity`` is set -- the I2V
    graph is a different node topology (``MiniMaxH3ImageToVideo``), not a
    tweak of the reference-to-video one, so it is a second authored template
    rather than a rewrite of the first."""

    # -- Issue #19 GPU-custody seam. All optional with defaults. --- #

    min_free_vram_gb: float = 16.0
    """Issue #19 custody pre-flight: the absolute free-VRAM floor, in GB, a
    run must clear before rendering starts. An absolute number, deliberately
    mirroring ``min_free_disk_gb`` -- an earlier version used a *fraction of
    the profile's nominal VRAM* (90% of 24 GB = 21.6 GB), which encoded an
    assumption rather than a measurement.

    **20 GB is what MiniMax H3 actually stages**, measured from ComfyUI's own
    load lines: ``MiniMaxH3`` 19995 MB for the diffusion model alone, plus a
    14956 MB text encoder, a 4965 MB video VAE and a 576 MB audio VAE. Those
    are loaded and offloaded around each other rather than being co-resident,
    which is why the blueprint's "~42.5 GB appetite" never materialized -- but
    the diffusion model's own 19995 MB is a hard floor for any run.

    This was 12.0 until 2026-08-07, chosen against an observed *working*
    configuration (~16.4 GB free) rather than against what the model stages.
    That was too low in the dangerous direction: it green-lit a run on a card
    that could not hold the model, and rather than raising CUDA OOM the load
    went **silent** mid-stage and wedged the host hard enough to need a power
    cycle. A pre-flight that passes a run which cannot possibly fit is worse
    than no pre-flight, because it converts a fast, legible failure into a
    hung machine.

    Note this is a *point-in-time* check. It cannot see another process
    claiming the card mid-run, which is exactly how that incident started --
    doris shares one 4090 between four independent workloads. Confirm the card
    is genuinely clear before a long run; stopping the one tenant you know
    about is not the same as freeing the card."""

_ALLOWED_OVERRIDE_KEYS = {f.name for f in dc_fields(RunConfig)}


def _fail(field_name: str, message: str) -> NoReturn:
    """Log and raise a :class:`ConfigError` for a validation failure."""
    logger.error("config validation failed for field=%s: %s", field_name, message)
    raise ConfigError(f"{field_name}: {message}")


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        logger.exception("Config file not found: %s", path)
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        logger.exception("Failed to read config file: %s", path)
        raise ConfigError(f"config file unreadable: {path} ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        logger.exception("Failed to parse TOML config file: %s", path)
        raise ConfigError(f"malformed TOML in config file {path}: {exc}") from exc


def _require(merged: dict, key: str) -> object:
    if key not in merged or merged[key] is None:
        _fail(key, "missing required field")
    return merged[key]


def _resolve_path(value: object, base_dir: Path) -> Path:
    candidate = value if isinstance(value, Path) else Path(str(value))
    return candidate if candidate.is_absolute() else base_dir / candidate


CAST_KEYS = frozenset({"role", "image", "appearance", "demeanour"})
"""Every key a ``[cast.<Name>]`` table may contain. Closed set, for the same
reason :data:`HARDWARE_KEYS` is: a misspelled ``appearence`` that is silently
ignored is config that reads as applied but never reaches a prompt."""

_PROHIBITION_PATTERN = re.compile(r"\b(never|without|no)\b", re.IGNORECASE)
"""Issue #73's own list of prohibition markers, applied to ``role`` and
``demeanour`` text. The corrected ``role`` that still rendered a bass guitar
read '...never holding anything' -- 'never'/'no'/'without' are the words a
human reaches for to write a prohibition, and none of them survive as tokens
a diffusion prompt actually conditions on."""

_VOCAL_EXCEPTION_PATTERN = re.compile(
    r"\b(sing|sings|singing|sung|vocal|vocalist)\b", re.IGNORECASE
)
"""Exempts the one prohibition pattern this project already relies on and
that already works for a completely different reason: 'never sings' /
'background, never sings' describes who is *audible*, which is decided by
``chunk.characters`` (issue #6/#33), never by the literal text of ``role`` --
so the #73 finding (a prohibition's tokens carry no weight in the rendered
prompt) does not apply to it and warning here would be a false alarm on the
project's own shipped example."""

_DISPLACEMENT_PATTERN = re.compile(
    r"\b(less|more|increasingly|gradually|growing|fading|slowly)\b", re.IGNORECASE
)
"""Issue #46's endpoint-not-displacement lesson, applied to demeanour text.
'looking a few years younger' is the appearance-field precedent for what a
displacement looks like and how it compounds on the chained I2V path, where
each chunk conditions on its own predecessor's output; 'increasingly grim' is
the same shape for demeanour."""


def _warn_if_prohibition(owner: str, field_label: str, text: str) -> None:
    """Issue #73: warn (never refuse -- false positives here cost nothing but
    a log line) when ``role`` or ``demeanour`` text reads as a prohibition.

    Measured, not guessed: the corrected ``role`` "...weathered, patient and
    unhurried, never holding anything" was in force for the entire "Deathless"
    render and rendered a bass guitar into Jan's hands twice anyway, because
    the only tokens a diffusion prompt actually sees in that phrase are
    'holding' and 'anything'. Nothing warned that the fix did not work: this
    closes that gap at the next config load, not after the next full render.
    """
    if not _PROHIBITION_PATTERN.search(text) or _VOCAL_EXCEPTION_PATTERN.search(text):
        return
    logger.warning(
        "%s (%s) reads as a prohibition: %r. This project measured that a prohibition "
        "is not enforceable in a diffusion prompt (issue #73) -- the corrected role "
        "'...never holding anything' rendered a bass guitar into the same hands twice "
        "anyway, because 'holding'/'anything' are the only tokens actually present. "
        "Say what IS true instead of what is not (e.g. 'empty-handed, hands at his "
        "sides' rather than 'never holding anything'). For a whole-video exclusion, "
        "the render-side `avoid` list is the supported lever, though it shares the "
        "same textual limitation; a shot-plan lint catching a contradicting shot line "
        "is the durable fix (see shot_plan.py).",
        owner,
        field_label,
        text,
    )


def _warn_if_displacement(owner: str, field_label: str, text: str) -> None:
    """Issue #74/#46: warn when demeanour text names a displacement rather
    than an endpoint -- the same compounding risk ``appearance`` already
    documents, applied to the new field before a render ever exercises it."""
    if not _DISPLACEMENT_PATTERN.search(text):
        return
    logger.warning(
        "%s (%s) reads as a displacement, not an endpoint: %r. On the chained I2V "
        "path each chunk conditions on the PREVIOUS chunk's own output, so a "
        "relative directive re-applies to a frame that already embodies it and "
        "drifts further each time (issue #46, measured on 'looking a few years "
        "younger' compounding across chained chunks). State a fixed point instead "
        "(e.g. 'grave and unsmiling' rather than 'increasingly grim') -- restating "
        "an endpoint against a frame that already looks that way changes nothing, "
        "which is what makes it safe to restate every chunk.",
        owner,
        field_label,
        text,
    )


def _optional_text(entry: dict, key: str) -> str | None:
    """Read an optional free-text field, treating blank as absent.

    An empty string is not direction -- it is an unfilled template -- and
    composing it into every prompt would add a stray separator for nothing.
    """
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(key, f"must be a string, got {value!r}")
    return value.strip() or None


def _build_cast(raw_cast: object, base_dir: Path) -> dict[str, CastMember]:
    """Build the cast dictionary from the parsed ``[cast.<Name>]`` tables."""
    if isinstance(raw_cast, dict) and raw_cast and all(
        isinstance(v, CastMember) for v in raw_cast.values()
    ):
        # Already-built cast dict, e.g. supplied wholesale via an override.
        return dict(raw_cast)

    if not isinstance(raw_cast, dict) or not raw_cast:
        _fail("cast", "config must define at least one [cast.<Name>] entry")

    cast: dict[str, CastMember] = {}
    for name, entry in raw_cast.items():  # type: ignore[union-attr]
        if not isinstance(entry, dict):
            _fail(f"cast.{name}", "must be a table with 'role' and 'image' keys")
        unknown = sorted(set(entry) - CAST_KEYS)
        if unknown:
            logger.error(
                "[cast.%s] contains unknown key(s): %s. Valid keys are %s.",
                name,
                ", ".join(unknown),
                ", ".join(sorted(CAST_KEYS)),
            )
            _fail(
                f"cast.{name}",
                f"unknown key(s): {', '.join(unknown)} (typo?). "
                f"Valid keys are: {', '.join(sorted(CAST_KEYS))}",
            )
        role = entry.get("role")
        image = entry.get("image")
        if not role:
            _fail(f"cast.{name}.role", "missing required 'role'")
        if not image:
            _fail(f"cast.{name}.image", "missing required 'image'")
        _warn_if_prohibition(f"cast.{name}", "role", role)
        member_demeanour = _optional_text(entry, "demeanour")
        if member_demeanour:
            _warn_if_prohibition(f"cast.{name}", "demeanour", member_demeanour)
            _warn_if_displacement(f"cast.{name}", "demeanour", member_demeanour)
        cast[name] = CastMember(
            name=name,
            role=role,
            image=_resolve_path(image, base_dir),
            appearance=_optional_text(entry, "appearance"),
            demeanour=member_demeanour,
        )
    return cast


_TOP_LEVEL_KEYS = frozenset(f.name for f in dc_fields(RunConfig))
"""Valid top-level config keys, derived from :class:`RunConfig` so it cannot
drift as fields are added. Used only to tell a *misplaced* setting apart from
a plain typo when reporting an unknown ``[hardware]`` key."""

FACE_TREATMENTS = frozenset({"flattering", "realistic"})
DEFAULT_FACE_TREATMENT = "flattering"
"""Issue #62. The default is what ``final_v10`` -- the current keeper -- was
rendered with, so an existing config keeps rendering what it always did."""

REALISM_LORA = "h3-realism-people-t2v-i2v-r2v.safetensors"
REALISM_LORA_STRENGTH = 0.8
REALISM_LORA_TRIGGER = "r34l1sm"
"""What ``face_treatment = "realistic"`` selects, all overridable.

0.8 rather than the model card's intended 1.0: measured on the v10 A/B, this
is the strength that showed a clear improvement in faces at ~1.5% wall clock.
1.0 is untried here. Provenance for the file itself -- source, licence,
sha256 and its training caveat -- lives in
``workflow_graph.KNOWN_LORAS``."""

ALIGNMENT_OVERRIDE_KEYS = frozenset({"segment_index", "start", "end", "reason"})
"""Every key an ``[[alignment_override]]`` table may contain. Closed for the
same reason :data:`HARDWARE_KEYS` is, and with a sharper edge: these tables
conventionally sit at the *end* of a run config, so they are what a bare key
appended to the file binds to."""

HARDWARE_KEYS = frozenset(
    {"name", "vram_gb", "min_chunk_seconds", "max_chunk_seconds", "recommended_nodes"}
)
"""Every key a ``[hardware]`` table may contain. Closed set, deliberately.

TOML assigns a bare key to whichever table precedes it, so a *top-level*
setting written below ``[hardware]`` silently becomes ``hardware.<key>``. That
is not hypothetical: ``examples/first-run.toml`` shipped with six settings
below the table -- including the 864x480 that was the entire point of the file
-- so every copy of it rendered at 1344x768 and took 3x as long as advertised,
with nothing to indicate anything was wrong. Validating against a closed set
turns that into a loud failure naming the key.
"""


def _warn_on_face_treatment_conflict(
    face_treatment: str, values: dict, merged: dict
) -> None:
    """Say once, where the decision is made, that a realism adapter and an
    appearance direction can pull against each other (issue #62).

    Measured on the v10 A/B rather than reasoned about. Derek, watching it:
    "The LoRA is definitely making faces more realistic, but in this
    PARTICULAR case, it is fighting against the AI age reductions."

    A *warning*, and deliberately not a keyword check on the adjectives.
    Appearance direction is not inherently flattering -- "weathered,
    unslept, badly lit" is appearance direction a realism adapter agrees
    with -- so scanning for "flattering" words would be exactly the kind of
    guess this project has already learned loses to structure. What is
    knowable here is only that both signals are present and that they
    *can* conflict; which way this song wants it is the author's call, which
    is the whole point of the flag.
    """
    if face_treatment != "realistic":
        return
    sources = []
    if merged.get("global_appearance"):
        sources.append("global_appearance")
    if any(
        isinstance(entry, dict) and entry.get("appearance")
        for entry in (merged.get("cast") or {}).values()
    ):
        sources.append("cast appearance")
    if not sources:
        return
    logger.warning(
        'face_treatment = "realistic" and %s are both set, and they can pull in opposite '
        "directions: the adapter renders real skin and real age, while appearance "
        "direction asking for a flattering or younger look argues against it, and the "
        "stronger signal wins without saying so. Measured on a real A/B (issue #62). "
        "Fine if the direction is *not* flattering -- 'weathered, unslept' agrees with "
        "the adapter -- but worth a look before spending a full render on it.",
        " and ".join(sources),
    )


def _build_hardware(raw_hardware: object) -> HardwareProfile:
    if isinstance(raw_hardware, HardwareProfile):
        return raw_hardware
    if not isinstance(raw_hardware, dict):
        _fail("hardware", "must be a [hardware] table")

    unknown = sorted(set(raw_hardware) - HARDWARE_KEYS)  # type: ignore[arg-type]
    if unknown:
        misplaced = [k for k in unknown if k in _TOP_LEVEL_KEYS]
        logger.error(
            "[hardware] contains unknown key(s): %s. Valid keys are %s.%s",
            ", ".join(unknown),
            ", ".join(sorted(HARDWARE_KEYS)),
            (
                f" {', '.join(misplaced)} is a top-level setting -- in TOML a bare key "
                "belongs to whichever table precedes it, so it must be moved ABOVE the "
                "[hardware] header to take effect."
                if misplaced
                else ""
            ),
        )
        hint = (
            f"; {', '.join(misplaced)} is a top-level setting and must be moved ABOVE "
            "the [hardware] table header to take effect (in TOML a bare key belongs to "
            "whichever table precedes it)"
            if misplaced
            else " (typo?)"
        )
        _fail(
            "hardware",
            f"unknown key(s) in the [hardware] table: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(HARDWARE_KEYS))}{hint}",
        )

    name = raw_hardware.get("name")  # type: ignore[union-attr]
    vram_gb = raw_hardware.get("vram_gb")  # type: ignore[union-attr]
    if not name:
        _fail("hardware.name", "missing required 'name'")
    if vram_gb is None:
        _fail("hardware.vram_gb", "missing required 'vram_gb'")

    kwargs: dict[str, object] = {"name": name, "vram_gb": float(vram_gb)}  # type: ignore[arg-type]
    if "min_chunk_seconds" in raw_hardware:  # type: ignore[operator]
        kwargs["min_chunk_seconds"] = float(raw_hardware["min_chunk_seconds"])  # type: ignore[index]
    if "max_chunk_seconds" in raw_hardware:  # type: ignore[operator]
        kwargs["max_chunk_seconds"] = float(raw_hardware["max_chunk_seconds"])  # type: ignore[index]
    if "recommended_nodes" in raw_hardware:  # type: ignore[operator]
        kwargs["recommended_nodes"] = tuple(raw_hardware["recommended_nodes"])  # type: ignore[index]
    return HardwareProfile(**kwargs)  # type: ignore[arg-type]


def _positive_number(merged: dict, key: str, default: float) -> float:
    """Read an optional strictly-positive numeric knob."""
    if key not in merged or merged[key] is None:
        return default
    try:
        value = float(merged[key])  # type: ignore[arg-type]
    except (TypeError, ValueError):
        _fail(key, f"must be a number, got {merged[key]!r}")
    if value <= 0:
        _fail(key, f"must be greater than zero, got {value}")
    return value


def _positive_int(merged: dict, key: str, default: int) -> int:
    if key not in merged or merged[key] is None:
        return default
    if isinstance(merged[key], bool) or not isinstance(merged[key], int):
        _fail(key, f"must be an integer, got {merged[key]!r}")
    value = int(merged[key])  # type: ignore[arg-type]
    if value < 1:
        _fail(key, f"must be at least 1, got {value}")
    return value


H3_DIMENSION_STEP = 32
"""H3's declared ``width``/``height`` grid (``docs/h3-node-schema.md``:
min 32, max 16384, step 32, from a live ``/object_info`` dump)."""

H3_DIMENSION_MIN = 32
H3_DIMENSION_MAX = 16384


def _render_dimensions(merged: dict) -> tuple[int | None, int | None]:
    """Resolve the optional per-run render resolution.

    Both fields must be given together. A width without a height is not a
    partial override -- it silently changes the aspect ratio of every shot in
    the run against whatever height the template happens to carry, which is
    the kind of thing nobody notices until the whole render is done.
    """
    width = merged.get("render_width")
    height = merged.get("render_height")

    if width is None and height is None:
        return None, None
    if width is None or height is None:
        missing = "render_height" if width is not None else "render_width"
        _fail(
            missing,
            "render_width and render_height must be set together -- setting only one "
            "changes the aspect ratio of every shot against the template's other "
            f"dimension (got render_width={width!r}, render_height={height!r})",
        )

    return _h3_dimension(merged, "render_width"), _h3_dimension(merged, "render_height")


def _h3_dimension(merged: dict, key: str) -> int:
    value = merged[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(key, f"must be an integer, got {value!r}")
    if not (H3_DIMENSION_MIN <= value <= H3_DIMENSION_MAX):
        _fail(
            key,
            f"must be between {H3_DIMENSION_MIN} and {H3_DIMENSION_MAX}, got {value}",
        )
    if value % H3_DIMENSION_STEP:
        _fail(
            key,
            f"must be a multiple of {H3_DIMENSION_STEP} (H3's declared dimension grid); "
            f"got {value}. The nearest valid values are "
            f"{value - value % H3_DIMENSION_STEP} and "
            f"{value - value % H3_DIMENSION_STEP + H3_DIMENSION_STEP}",
        )
    return int(value)


def _flag(merged: dict, key: str, default: bool) -> bool:
    if key not in merged or merged[key] is None:
        return default
    if not isinstance(merged[key], bool):
        _fail(key, f"must be a boolean, got {merged[key]!r}")
    return bool(merged[key])


def _validate_file(field_name: str, path: Path) -> None:
    if not path.exists():
        _fail(field_name, f"file does not exist: {path}")
    if not path.is_file():
        _fail(field_name, f"path exists but is not a file: {path}")
    if not os.access(path, os.R_OK):
        _fail(field_name, f"file is not readable: {path}")


def _validate_url(field_name: str, url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        _fail(field_name, f"not a well-formed http(s) URL: {url!r}")


def _ensure_dir(field_name: str, path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.exception("Failed to create output directory for field=%s: %s", field_name, path)
        raise ConfigError(f"{field_name}: could not create directory {path} ({exc})") from exc


def _validate(config: RunConfig) -> None:
    _validate_file("master_audio", config.master_audio)
    _validate_file("lyrics_file", config.lyrics_file)

    for name, member in config.cast.items():
        _validate_file(f"cast.{name}.image", member.image)

    if config.default_lead_vocalist not in config.cast:
        _fail(
            "default_lead_vocalist",
            f"{config.default_lead_vocalist!r} is not a key in the cast dictionary "
            f"(known cast members: {sorted(config.cast)})",
        )

    _validate_url("comfyui_url", config.comfyui_url)
    _validate_file("workflow_template", config.workflow_template)

    _ensure_dir("chunks_dir", config.chunks_dir)
    _ensure_dir("final_video_dir", config.final_video_dir)

    if config.i2v_continuity and config.i2v_workflow_template is None:
        _fail(
            "i2v_workflow_template",
            "required when i2v_continuity is enabled: the I2V path is a separate "
            "authored workflow template (MiniMaxH3ImageToVideo), not a mutation "
            "of the reference-to-video one",
        )
    if config.i2v_workflow_template is not None:
        _validate_file("i2v_workflow_template", config.i2v_workflow_template)

    if config.i2v_min_seed_face_similarity is not None:
        # Issue #49: recognition was explicitly requested. Refuse loudly and
        # early rather than silently degrade to detection-only at render time
        # -- that would hand the operator a weaker guard than they asked for
        # while reporting success, hours into a run this same stat() call
        # could have caught for free.
        recognition_model = resolve_recognition_model_path()
        if not recognition_model.exists():
            _fail(
                "i2v_min_seed_face_similarity",
                f"is set to {config.i2v_min_seed_face_similarity!r}, which requires the "
                f"SFace face-recognition model, but it is not present at "
                f"{recognition_model}. It is deliberately NOT committed to this "
                f"repository (issue #49, ~38.7 MB -- see docs/seed-face-recognition.md): "
                f"fetch it from {RECOGNITION_MODEL_SOURCE} (sha256 "
                f"{RECOGNITION_MODEL_SHA256}) and place it at that path, or unset "
                "i2v_min_seed_face_similarity to render with the detection-only gate "
                "instead",
            )


def load_config(path: Path, **overrides: object) -> RunConfig:
    """Load, merge overrides onto, and validate a run config TOML file.

    ``overrides`` beat file values and must be named after :class:`RunConfig`
    fields (this is how a CLI's ``--comfyui-url``-style flags plug in).
    Raises :class:`ConfigError` -- logged via ``logger.error``/``logger.exception``
    before raising -- for any parse or validation failure.
    """
    path = Path(path)
    raw = _read_toml(path)
    base_dir = path.parent.resolve()

    unknown = set(overrides) - _ALLOWED_OVERRIDE_KEYS
    if unknown:
        _fail("overrides", f"unknown override key(s): {sorted(unknown)}")

    merged: dict[str, object] = dict(raw)
    merged.update(overrides)

    values: dict[str, object] = {}
    for key in _SCALAR_FIELDS:
        values[key] = _require(merged, key)

    values["comfyui_url"] = merged.get("comfyui_url") or DEFAULT_COMFYUI_URL

    for key in _PATH_FIELDS:
        values[key] = _resolve_path(_require(merged, key), base_dir)

    values["cast"] = _build_cast(merged.get("cast"), base_dir)
    values["hardware"] = _build_hardware(_require(merged, "hardware"))

    values["watchdog_timeout_seconds"] = _positive_number(merged, "watchdog_timeout_seconds", 900.0)
    values["max_render_attempts"] = _positive_int(merged, "max_render_attempts", 3)
    values["retry_backoff_seconds"] = _positive_number(merged, "retry_backoff_seconds", 5.0)
    values["min_free_disk_gb"] = _positive_number(merged, "min_free_disk_gb", 20.0)
    values["min_free_vram_gb"] = _positive_number(merged, "min_free_vram_gb", 16.0)
    values["render_width"], values["render_height"] = _render_dimensions(merged)
    values["instrumental_coverage"] = _flag(merged, "instrumental_coverage", True)
    values["i2v_continuity"] = _flag(merged, "i2v_continuity", False)
    values["resume_ignore_prompt_changes"] = _flag(merged, "resume_ignore_prompt_changes", False)
    values["strict_alignment"] = _flag(merged, "strict_alignment", False)

    model_size = merged.get("alignment_model_size", DEFAULT_ALIGNMENT_MODEL_SIZE)
    if model_size not in ALIGNMENT_MODEL_SIZES:
        raise ConfigError(
            f"alignment_model_size must be one of {sorted(ALIGNMENT_MODEL_SIZES)}; got "
            f"{model_size!r}. Caught here because the alternative is a failure from "
            "inside stable-ts after the weights download, minutes into Stage 1"
        )
    values["alignment_model_size"] = model_size
    between_chunk_floor = merged.get("between_chunk_min_free_vram_gb")
    values["between_chunk_min_free_vram_gb"] = (
        _positive_number(merged, "between_chunk_min_free_vram_gb", 0.0)
        if between_chunk_floor is not None
        else None
    )
    raw_overrides = merged.get("alignment_override", [])
    if not isinstance(raw_overrides, list):
        raise ConfigError(
            f"alignment_override must be [[alignment_override]] tables, got "
            f"{type(raw_overrides).__name__}"
        )
    parsed_overrides = []
    for i, raw in enumerate(raw_overrides):
        # Issue #63: closed set, for exactly the reason HARDWARE_KEYS is one.
        # TOML binds a bare key to whichever table precedes it, and these
        # tables sit at the END of every real run config -- so a top-level
        # setting appended to the file lands in here and is silently dropped.
        # Not hypothetical: issue #62's A/B was built with `lora` appended
        # below them, and both arms of the experiment loaded identically.
        if isinstance(raw, dict):
            unknown = sorted(set(raw) - ALIGNMENT_OVERRIDE_KEYS)
            if unknown:
                misplaced = [k for k in unknown if k in _TOP_LEVEL_KEYS]
                logger.error(
                    "[[alignment_override]] #%d contains unknown key(s): %s. Valid keys "
                    "are %s.%s",
                    i,
                    ", ".join(unknown),
                    ", ".join(sorted(ALIGNMENT_OVERRIDE_KEYS)),
                    (
                        f" {', '.join(misplaced)} is a top-level setting -- in TOML a bare "
                        "key belongs to whichever table precedes it, so it must be moved "
                        "ABOVE the first table, not appended to the end of the file."
                        if misplaced
                        else ""
                    ),
                )
                raise ConfigError(
                    f"[[alignment_override]] #{i} contains unknown key(s): "
                    f"{', '.join(unknown)}"
                    + (
                        f" -- {', '.join(misplaced)} is a top-level setting that TOML has "
                        "bound to this table; move it above the first table"
                        if misplaced
                        else " (typo?)"
                    )
                )
        try:
            segment_index = raw["segment_index"]
            start = raw["start"]
            end = raw["end"]
            reason = raw["reason"]
        except (KeyError, TypeError) as exc:
            raise ConfigError(
                f"alignment_override #{i} needs segment_index, start, end and reason "
                f"(reason is mandatory: an unexplained pin is indistinguishable from a "
                f"stale one after the next lyrics edit): {exc}"
            ) from exc
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index < 0
        ):
            raise ConfigError(
                f"alignment_override #{i}: segment_index must be a non-negative integer, "
                f"got {segment_index!r}"
            )
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
            raise ConfigError(
                f"alignment_override #{i}: start/end must be numbers with start < end, "
                f"got {start!r}..{end!r}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"alignment_override #{i}: reason must be a non-empty string")
        parsed_overrides.append(
            AlignmentOverride(
                segment_index=segment_index, start=float(start), end=float(end),
                reason=reason.strip(),
            )
        )
    values["alignment_overrides"] = tuple(parsed_overrides)

    instrumental_shot = merged.get("instrumental_shot_seconds")
    values["instrumental_shot_seconds"] = (
        _positive_number(merged, "instrumental_shot_seconds", 0.0)
        if instrumental_shot is not None
        else None
    )

    values["setting"] = _optional_text(merged, "setting")
    values["global_appearance"] = _optional_text(merged, "global_appearance")
    values["cinematography"] = _optional_text(merged, "cinematography")

    global_demeanour = _optional_text(merged, "global_demeanour")
    if global_demeanour:
        _warn_if_prohibition("run config", "global_demeanour", global_demeanour)
        _warn_if_displacement("run config", "global_demeanour", global_demeanour)
    values["global_demeanour"] = global_demeanour

    raw_avoid = merged.get("avoid", [])
    if raw_avoid is None:
        raw_avoid = []
    if not isinstance(raw_avoid, list):
        raise ConfigError(
            f"avoid must be a list of strings, got {type(raw_avoid).__name__} (issue #73)"
        )
    avoid_items = []
    for i, item in enumerate(raw_avoid):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"avoid[{i}] must be a non-empty string, got {item!r} (issue #73)"
            )
        avoid_items.append(item.strip())
    values["avoid"] = tuple(avoid_items)

    shot_plan_path = merged.get("shot_plan")
    if shot_plan_path:
        resolved_plan = _resolve_path(shot_plan_path, base_dir)
        _validate_file("shot_plan", resolved_plan)
        values["shot_plan"] = resolved_plan
    else:
        values["shot_plan"] = None

    noise_seed = merged.get("noise_seed", 0)
    if (
        isinstance(noise_seed, bool)
        or not isinstance(noise_seed, int)
        or not (0 <= noise_seed <= MAX_NOISE_SEED)
    ):
        raise ConfigError(
            f"noise_seed must be an integer in 0..{MAX_NOISE_SEED}, got {noise_seed!r} -- "
            "ComfyUI clamps anything else server-side, and a clamped seed means the run "
            "records a seed the render did not use (issue #38)"
        )
    values["noise_seed"] = noise_seed

    # Issue #39: deliberately NOT a path field. clip_name is resolved by
    # ComfyUI inside its own models/text_encoders/ on doris, so resolving it
    # against this config file's directory (as _PATH_FIELDS does) would
    # produce a local path the render host has never seen -- and validating
    # its existence here would refuse every correct value.
    text_encoder = merged.get("text_encoder")
    if text_encoder is not None:
        if not isinstance(text_encoder, str) or not text_encoder.strip():
            raise ConfigError(
                f"text_encoder must be a non-empty string naming a weights file in "
                f"ComfyUI's models/text_encoders/, got {text_encoder!r} (issue #39)"
            )
        if PurePosixPath(text_encoder).is_absolute() or PureWindowsPath(text_encoder).is_absolute():
            raise ConfigError(
                f"text_encoder must be relative to ComfyUI's models/text_encoders/, got the "
                f"absolute path {text_encoder!r} -- ComfyUI joins it under that directory, so "
                "the load would fail on the render host after custody was taken (issue #39)"
            )
        text_encoder = text_encoder.strip()
        logger.info(
            "text_encoder pinned to %s -- overriding whatever CLIPLoader.clip_name the "
            "workflow template(s) name (issue #39)",
            text_encoder,
        )
    values["text_encoder"] = text_encoder

    face_treatment = merged.get("face_treatment", DEFAULT_FACE_TREATMENT)
    if face_treatment not in FACE_TREATMENTS:
        raise ConfigError(
            f"face_treatment must be one of {sorted(FACE_TREATMENTS)}, got "
            f"{face_treatment!r} (issue #62)"
        )
    values["face_treatment"] = face_treatment

    # Issue #62, and a path field for exactly the same non-reason as
    # text_encoder above: lora_name is resolved by ComfyUI inside its own
    # models/loras/ on doris.
    lora = merged.get("lora")
    if lora is None and face_treatment == "realistic":
        # The preset that makes the flag one line instead of four. A default,
        # never a lock -- an explicit `lora` below overrides it untouched.
        lora = REALISM_LORA
        merged.setdefault("lora_strength", REALISM_LORA_STRENGTH)
        merged.setdefault("lora_trigger", REALISM_LORA_TRIGGER)
        logger.info(
            "face_treatment = \"realistic\": using %s at strength %s, trigger %r "
            "(issue #62). Set lora/lora_strength/lora_trigger explicitly to override.",
            REALISM_LORA,
            merged["lora_strength"],
            merged["lora_trigger"],
        )
    if lora is not None:
        if not isinstance(lora, str) or not lora.strip():
            raise ConfigError(
                f"lora must be a non-empty string naming a weights file in ComfyUI's "
                f"models/loras/, got {lora!r} (issue #62)"
            )
        if PurePosixPath(lora).is_absolute() or PureWindowsPath(lora).is_absolute():
            raise ConfigError(
                f"lora must be relative to ComfyUI's models/loras/, got the absolute path "
                f"{lora!r} -- ComfyUI joins it under that directory, so the load would fail "
                "on the render host after custody was taken (issue #62)"
            )
        lora = lora.strip()
    values["lora"] = lora

    lora_strength = merged.get("lora_strength", 1.0)
    if isinstance(lora_strength, bool) or not isinstance(lora_strength, (int, float)):
        raise ConfigError(
            f"lora_strength must be a number, got {lora_strength!r} (issue #62)"
        )
    values["lora_strength"] = float(lora_strength)

    lora_trigger = merged.get("lora_trigger")
    if lora_trigger is not None and (
        not isinstance(lora_trigger, str) or not lora_trigger.strip()
    ):
        raise ConfigError(
            f"lora_trigger must be a non-empty string, got {lora_trigger!r} (issue #62)"
        )
    values["lora_trigger"] = lora_trigger.strip() if lora_trigger else None

    # A trigger word with no adapter behind it is a stray token in all 36
    # prompts that nothing acts on -- and, being first, in the position that
    # most shapes the style. Refuse rather than warn: it is free to catch here
    # and costs a whole render to notice.
    if lora_trigger and not lora:
        raise ConfigError(
            f"lora_trigger={lora_trigger!r} is set but lora is not, so the trigger word "
            "would be composed into every prompt with no adapter to trigger (issue #62)"
        )
    if lora:
        logger.info(
            "LoRA %s pinned at strength %s%s -- spliced into both templates (issue #62)",
            lora,
            values["lora_strength"],
            f", trigger word {values['lora_trigger']!r}" if values["lora_trigger"] else "",
        )
    _warn_on_face_treatment_conflict(face_treatment, values, merged)

    vocal_stem = merged.get("vocal_stem")
    if vocal_stem:
        resolved_stem = _resolve_path(vocal_stem, base_dir)
        _validate_file("vocal_stem", resolved_stem)
        values["vocal_stem"] = resolved_stem
    else:
        values["vocal_stem"] = None

    i2v_template = merged.get("i2v_workflow_template")
    values["i2v_workflow_template"] = (
        _resolve_path(i2v_template, base_dir) if i2v_template else None
    )

    reanchor = merged.get("i2v_reanchor_interval")
    if reanchor is not None and (
        isinstance(reanchor, bool) or not isinstance(reanchor, int) or reanchor < 1
    ):
        raise ConfigError(
            f"i2v_reanchor_interval must be an integer >= 1 (or absent to never "
            f"re-anchor), got {reanchor!r}"
        )
    values["i2v_reanchor_interval"] = reanchor

    require_face = merged.get("i2v_require_seed_face", True)
    if not isinstance(require_face, bool):
        raise ConfigError(
            f"i2v_require_seed_face must be true or false, got {require_face!r}"
        )
    values["i2v_require_seed_face"] = require_face

    face_fraction = merged.get("i2v_min_seed_face_fraction", DEFAULT_MIN_FACE_FRACTION)
    if isinstance(face_fraction, bool) or not isinstance(face_fraction, (int, float)):
        raise ConfigError(
            f"i2v_min_seed_face_fraction must be a number, got {face_fraction!r}"
        )
    if not 0.0 < float(face_fraction) < 1.0:
        raise ConfigError(
            f"i2v_min_seed_face_fraction is a FRACTION of the frame area, so it must be "
            f"between 0 and 1 (the calibrated default is "
            f"{DEFAULT_MIN_FACE_FRACTION} = {DEFAULT_MIN_FACE_FRACTION * 100:.2f}%); "
            f"got {face_fraction!r}"
        )
    values["i2v_min_seed_face_fraction"] = float(face_fraction)

    # Issue #49: unset (None) means recognition is OFF -- see the field's own
    # docstring for why that default cannot be the calibrated
    # DEFAULT_MIN_FACE_SIMILARITY. Validated the same shape as the fraction
    # above it, but against a cosine similarity's -1.0..1.0 range rather than
    # a fraction's 0..1.
    face_similarity = merged.get("i2v_min_seed_face_similarity")
    if face_similarity is not None:
        if isinstance(face_similarity, bool) or not isinstance(face_similarity, (int, float)):
            raise ConfigError(
                f"i2v_min_seed_face_similarity must be a number, got {face_similarity!r}"
            )
        if not -1.0 <= float(face_similarity) <= 1.0:
            raise ConfigError(
                f"i2v_min_seed_face_similarity is a cosine similarity, so it must be "
                f"between -1.0 and 1.0 (the calibrated recommendation is "
                f"faces.DEFAULT_MIN_FACE_SIMILARITY = {DEFAULT_MIN_FACE_SIMILARITY} -- see "
                f"docs/seed-face-recognition.md); got {face_similarity!r}"
            )
        face_similarity = float(face_similarity)
    values["i2v_min_seed_face_similarity"] = face_similarity

    chain_scope = merged.get("i2v_chain_scope", "instrumental")
    if chain_scope not in ("instrumental", "all", "none"):
        raise ConfigError(
            f"i2v_chain_scope must be one of 'instrumental', 'all', 'none'; got "
            f"{chain_scope!r}. 'instrumental' chains only where lip-sync is not given up "
            "-- the fl2va path has no audio conditioning (issue #28)"
        )
    values["i2v_chain_scope"] = chain_scope


    run_state_file = merged.get("run_state_file")
    values["run_state_file"] = (
        _resolve_path(run_state_file, base_dir)
        if run_state_file
        else Path(values["chunks_dir"]) / "run_state.json"  # type: ignore[arg-type]
    )

    config = RunConfig(**values)  # type: ignore[arg-type]
    _validate(config)

    if config.setting is None:
        logger.warning(
            "No 'setting' in %s: nothing anchors where the video takes place, so each shot's "
            "geography is whatever its own description implies. The first full render walked "
            "out of a British terraced street into a US city park this way. Set e.g. "
            'setting = "London, UK -- contemporary, overcast winter"',
            path,
        )

    logger.info(
        "Loaded run config from %s (cast=%s, default_lead_vocalist=%s, comfyui_url=%s)",
        path,
        sorted(config.cast),
        config.default_lead_vocalist,
        config.comfyui_url,
    )
    return config
