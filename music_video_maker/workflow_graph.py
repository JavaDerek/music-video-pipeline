"""Stage 4a: workflow template loading + graph introspection by ``class_type``
(issue #8).

Ground truth for the graph shape is ``docs/h3-node-schema.md`` (pulled from a
live ``GET /object_info`` on the real ComfyUI on doris) and
``tests/fixtures/workflows/README.md``. Two corrections to the earlier,
guessed graph shape matter here:

- **There is no text-encode node.** The prompt is a plain ``STRING`` input at
  ``MiniMaxH3ReferenceToVideo.inputs.prompt``. The classic ComfyUI
  positive/negative ``CLIPTextEncode`` disambiguation problem does not exist
  in this graph.
- **The real ambiguity is the two ``VAELoader`` nodes** (video VAE vs audio
  VAE, two different weight files loaded through two separate node
  instances). When both are left at ComfyUI's un-customized default title,
  they can only be told apart by tracing which input socket on
  ``MiniMaxH3ReferenceToVideo`` each one's output feeds (``vae`` vs
  ``audio_vae``) -- see :func:`resolve_upstream_node`.

This module has two layers:

1. **General graph introspection** (:func:`load_workflow_template`,
   :func:`find_nodes_by_class_type`, :func:`find_one_node`,
   :func:`resolve_upstream_node`, :func:`assert_no_cloud_api_nodes`) --
   reusable by later stages (issues #10 resume/dead-letter, #12 I2V
   continuity, #14 the CLI) that also need to locate or validate nodes by
   ``class_type`` rather than by node id, since user edits on the ComfyUI
   canvas renumber node ids freely.
2. **The mutator** (:class:`WorkflowGraphMutator`), which implements
   ``contracts.WorkflowMutator`` and performs the actual per-chunk
   injection Stage 4b (issue #9) needs before submitting a render.

Node-id independence is a hard invariant throughout: nothing in this module
may key off a hardcoded node id. Everything is located by ``class_type``,
optionally disambiguated by ``_meta.title`` or by wiring.

``ref_images`` / ``ref_audios`` on ``MiniMaxH3ReferenceToVideo`` are
``COMFY_AUTOGROW_V3`` dynamic multi-slot inputs. Issue #18 confirmed their
serialized shape against a real exported ``workflow_api.json`` (and a real
render): each populated slot is an ordinary flat, dotted key directly in the
node's ``inputs`` dict -- e.g. ``"ref_images.ref_image_0": ["50", 0]`` --
holding a single plain ``[node_id, output_index]`` link, exactly like any
other input. There is no nesting and no list-of-links wrapper. This module
does not depend on that shape at all: the mutator never writes into
``ref_images``/``ref_audios`` itself, it only injects filenames into the
``LoadImage``/``LoadAudio`` nodes those slots reference, and
:func:`resolve_upstream_node` needs no AUTOGROW-specific handling since a
populated slot is just a plain reference under its own key.

**More than one reference photo (issue #33 levels 1-2).**
:attr:`~music_video_maker.contracts.StagedAssets.image_filenames` carries
every staged reference filename, primary first; the primary always renders
through the existing, unchanged single-``LoadImage`` path
(:attr:`~music_video_maker.contracts.StagedAssets.image_filename`). Anything
beyond the primary is wired in by :func:`WorkflowGraphMutator.mutate`
synthesizing one new ``LoadImage`` node per extra photo and adding its own
``ref_images.ref_image_1``, ``ref_image_2``, ... dotted key to the H3
node -- never by adding a second ``LoadImage`` node to the *template*
itself. The committed ``workflow_api.json`` deliberately keeps exactly one
``LoadImage`` node: ``continuity.py``'s ``_render_base`` calls ``mutate()``
with no ``cast_image_title`` at all, relying on ``find_one_node`` resolving
that single node unambiguously (see ``docs/workflow-template-guide.md``'s
"hint vs. identity claim" section) -- a second node committed to the
template would make that lookup ambiguous for every single-reference chunk,
not just multi-reference ones. Synthesizing the extra node only when there
is a second filename to inject keeps the single-reference path -- the one
every run to date has used -- untouched in both the template file and the
returned graph.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from music_video_maker.contracts import H3_FRAME_GRID, Workflow

if TYPE_CHECKING:
    from music_video_maker.contracts import ExpandedPrompt, StagedAssets, WorkflowMutator

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# class_type constants
# --------------------------------------------------------------------------- #
#
# Defined here, independently of tests/harness/factories.py's identical
# constants -- production code must not import from the test tree. Values are
# confirmed from a live /object_info dump; see docs/h3-node-schema.md.

CLASS_TYPE_H3_REFERENCE_TO_VIDEO = "MiniMaxH3ReferenceToVideo"
CLASS_TYPE_H3_IMAGE_TO_VIDEO = "MiniMaxH3ImageToVideo"
"""The I2V continuity path (issue #12) -- a separate authored template from
``CLASS_TYPE_H3_REFERENCE_TO_VIDEO``'s, never a rewrite of it. Both nodes own
the ``prompt`` STRING input in their respective graphs; see
:func:`_find_h3_conditioning_node`."""
CLASS_TYPE_VAE_LOADER = "VAELoader"
CLASS_TYPE_IMAGE_LOAD = "LoadImage"
CLASS_TYPE_AUDIO_LOAD = "LoadAudio"
CLASS_TYPE_VIDEO_SAVE = "SaveVideo"
CLASS_TYPE_RANDOM_NOISE = "RandomNoise"
"""The node owning ``noise_seed`` (issue #38). Present exactly once in *both*
committed templates -- the base and the I2V one -- so it resolves through
:func:`find_one_node` with no title disambiguator on either."""

CLASS_TYPE_CLIP_LOADER = "CLIPLoader"
"""The node owning ``clip_name`` -- which text encoder reads the prompt
(issue #39). Present exactly once in both committed templates, with
``type="minimax"`` naming the H3 tokenizer family (which is *not* what a
precision swap changes; only the weights file is)."""

CLASS_TYPE_LORA_LOADER = "LoraLoaderModelOnly"
"""Issue #62. ``LoraLoaderModelOnly`` rather than ``LoraLoader`` because the
adapter is a DiT LoRA -- its 208 tensors are all ``diffusion_model.blocks.*``,
so there is no CLIP side to patch and the two-output loader would need a
text-encoder edge that does not exist in this graph."""

CLASS_TYPE_UNET_LOADER_NAME = "UNETLoader"
"""Where the MODEL edge starts, and so where a LoRA is spliced in (#62).
Neither committed template carries a LoRA node, so unlike ``clip_name`` this
cannot be an input pin -- see :func:`insert_lora`."""

LORA_NAME_INPUT = "lora_name"
LORA_STRENGTH_INPUT = "strength_model"

KNOWN_LORAS = {
    "h3-realism-people-t2v-i2v-r2v.safetensors": {
        "source": "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA",
        "sha256": "acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e",
        "bytes": 131229656,
        "licence": "MiniMax H3 Community License (adapter follows the base model's)",
        "trigger": "r34l1sm",
        "rank": 32,
        # From the file's own safetensors metadata, which does NOT match the
        # filename's promise, and the difference decides where this is safe:
        #   base_model: minimax-h3-fl2va
        #   training_strategy: text_to_video
        #   first_frame_conditioning_p: 0.0
        # Trained as pure t2v with *zero* first-frame conditioning, so despite
        # the "t2v-i2v-r2v" in the name it has never seen a first frame during
        # training -- and very likely never seen reference-image conditioning
        # either, which is the whole identity mechanism on our base path. Treat
        # the effect on likeness as unmeasured until a slice says otherwise
        # (issue #62).
        "caveat": "trained text_to_video, first_frame_conditioning_p=0.0",
    },
}
"""Provenance for third-party weights, recorded beside the code that loads
them the way ``faces.py`` does for YuNet -- source, licence and sha256, so
"it downloaded fine" is never the licence check. Descriptive only: nothing
here restricts which name a config may set, since which adapters are
installed on the ComfyUI host is not this repo's business."""

NOISE_SEED_INPUT = "noise_seed"

TEXT_ENCODER_INPUT = "clip_name"
"""``CLIPLoader``'s weights-file input. A server-side name resolved by
ComfyUI inside its own ``models/text_encoders/``, never a path on the machine
running this pipeline."""

DEFAULT_NOISE_SEED = 0
"""The seed used when a caller names none.

Deliberately the value both committed templates already carry, so a run that
does not ask for a seed renders byte-for-byte what this project has always
rendered. It is a *default*, not a floor: the point of issue #38 is that the
value reaching ComfyUI comes from the run, not from whatever the template file
happens to hold at the time."""

MAX_NOISE_SEED = 0xFFFFFFFFFFFFFFFF
"""``RandomNoise.noise_seed``'s ``max`` in ComfyUI (``0 .. 2**64-1``)."""

RESEED_GENERATION_STRIDE = 2**32
"""Per-generation offset :func:`resolve_chunk_seed` adds for ``cli.py``'s
``--reseed`` (issue #38 CLI). Chosen to sit comfortably between the two
things it must never collide with: a real song's chunk_id range (H3's
minimum chunk length is ~5s, so even an hours-long track is only in the low
thousands of chunks) on one side, and :data:`MAX_NOISE_SEED`'s ~1.8e19 range
on the other. A reseed's generation bump and another chunk's ordinary
chunk_id offset can therefore never land on the same value by coincidence --
see :func:`resolve_chunk_seed`."""


def resolve_chunk_seed(base_seed: int, chunk_id: int, *, reseed_generation: int = 0) -> int:
    """Derive one chunk's ``RandomNoise.noise_seed`` from a run-wide base
    (issue #38's variation half).

    ``seed = (base_seed + chunk_id + reseed_generation * RESEED_GENERATION_STRIDE)
    mod (MAX_NOISE_SEED + 1)``

    Deliberately a pure function of three plain integers, with no run state
    to read: a resumed run recomposes the exact same value for the exact
    same chunk without consulting anything, which is what lets ``--resume``
    keep 38 of 39 chunks as cache hits when only one was reseeded --
    ``resilience.py``'s content-tier fingerprint comparison is what actually
    decides that, and it can only work if this function always answers the
    same question the same way.

    ``chunk_id`` is added, not hashed or multiplied, so neighbouring chunks
    get close-but-different numbers -- legible while debugging a run, and
    harmless: nothing about H3's sampling makes nearby seeds produce
    nearby-looking frames, so this function only has to be *different* per
    chunk, not *dispersed*. The noise generator's own mixing supplies the
    dispersion.

    ``reseed_generation`` (default 0 -- what every chunk in an ordinary run
    gets) is ``cli.py``'s ``--reseed``/``--reseed-generation``: naming a
    chunk in that mapping bumps its generation, which shifts the result by a
    multiple of :data:`RESEED_GENERATION_STRIDE` -- far outside the range any
    chunk_id offset reaches, so a reseeded chunk's new take can never
    reproduce its own previous take, a neighbour's ordinary seed, or another
    generation's value for the same chunk.

    Wraps modulo ``MAX_NOISE_SEED + 1`` rather than raising on overflow: the
    result is always a value :class:`WorkflowGraphMutator` (and ComfyUI)
    would accept outright, so a caller near the top of a large base seed's
    range never has to special-case anything. Every input this project
    validates (``config.py``'s ``noise_seed``, a chunk_id, a
    ``--reseed-generation``) is non-negative, so the result is too --
    Python's ``%`` returns a non-negative value for a non-negative modulus
    regardless.

    A seed is not a promise about *pixels* across anything but this exact
    numeric identity: it does not transport a composition across a change in
    precision, GPU, attention kernel, or resolution (diffusion sampling is
    chaotic with respect to all four) -- only across a re-run of the same
    graph, on the same hardware and settings, under the same derived number.
    """
    return (base_seed + chunk_id + reseed_generation * RESEED_GENERATION_STRIDE) % (
        MAX_NOISE_SEED + 1
    )

REF_IMAGES_AUTOGROW_FIELD = "ref_images"
"""The ``COMFY_AUTOGROW_V3`` field name on ``MiniMaxH3ReferenceToVideo``
(only) that ``ref_image_N`` slots dot into -- see
``docs/h3-node-schema.md`` finding 3. Never present, and never needed, on
``MiniMaxH3ImageToVideo`` (the I2V path), which has no ``ref_images`` input
at all."""
REF_IMAGE_SLOT_PREFIX = "ref_image_"
"""Confirmed slot-name prefix (issue #18): a populated slot's key is
``f"{REF_IMAGES_AUTOGROW_FIELD}.{REF_IMAGE_SLOT_PREFIX}{index}"``, e.g.
``"ref_images.ref_image_1"``."""

# Billed, remote cloud-API variants (comfy_api_nodes/nodes_minimax.py).
# Spelled "Minimax..." (lowercase m) vs the local "MiniMaxH3..." nodes above
# -- an easy misgrab when introspecting the node registry by substring match.
# Must never appear in a graph this project builds or executes.
CLOUD_API_CLASS_TYPES = (
    "MinimaxHailuo03FirstLastFrameNode",
    "MinimaxHailuo03ReferenceNode",
    "MinimaxHailuo03TextToVideoNode",
    "MinimaxHailuoVideoNode",
    "MinimaxImageToVideoNode",
    "MinimaxTextToVideoNode",
)

# Per-chunk SaveVideo.inputs.filename_prefix format. Cross-lane contract with
# Stage 4b (issue #9): output filenames are prefixed this way so execution
# can tell chunk outputs apart on disk / in ComfyUI history, e.g. chunk_id=7
# -> "mvm_chunk_0007".
CHUNK_FILENAME_PREFIX_FORMAT = "mvm_chunk_{chunk_id:04d}"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class WorkflowGraphError(Exception):
    """Base class for every graph-introspection/mutation failure in this module."""


class WorkflowTemplateError(WorkflowGraphError):
    """The template file is missing, not valid JSON, or not shaped like a
    ComfyUI API-format workflow (flat dict of node objects with ``class_type``
    and ``inputs``)."""


class NodeNotFoundError(WorkflowGraphError):
    """No node with the requested ``class_type`` exists in the workflow."""


class AmbiguousNodeError(WorkflowGraphError):
    """Multiple nodes match the requested ``class_type`` and no disambiguator
    (title or wiring) resolved it to exactly one."""


class DanglingNodeReferenceError(WorkflowGraphError):
    """A node-reference input (a ``[node_id, output_index]`` link, whether a
    plain input or a populated AUTOGROW_V3 slot's dotted key) points at a
    node id absent from the workflow."""


class InvalidLengthError(WorkflowGraphError):
    """An injected ``length`` is not a value the H3 node can honor exactly
    (issue #20).

    ``length`` is quantized by ComfyUI itself (``min=5, step=17``), so a
    value off that grid is silently rounded server-side while the chunk's
    audio stem keeps its original duration -- the per-chunk mismatch that
    accumulates into lip-sync drift across a song. A value outside the
    trained range (124-362 frames) is a different failure: it renders, but
    out of distribution. Both are caught here rather than after the GPU
    hours are spent."""


class InvalidNoiseSeedError(WorkflowGraphError):
    """An injected ``noise_seed`` is not a value ComfyUI would honor exactly
    (issue #38).

    ``RandomNoise.noise_seed`` is an ``INT`` bounded to ``0 ..
    0xffffffffffffffff``; anything outside that is clamped server-side, so the
    render happens under a seed that differs from the one the run recorded --
    which is the whole defect this issue closes, reintroduced one layer down.
    A ``bool`` is rejected too: ``True`` is a valid ``int`` in Python and a
    wiring bug everywhere else."""


class InvalidTextEncoderError(WorkflowGraphError):
    """An injected ``clip_name`` is not a name ComfyUI could resolve (issue
    #39).

    Blank, non-string, or an absolute path -- the last being the local-path
    misreading of a *server-side* name: ComfyUI joins it under its own
    ``models/text_encoders/``, so an absolute path fails at model-load time,
    which on a render run is after custody of the card has been taken."""


class ChunkMismatchError(WorkflowGraphError):
    """The :class:`ExpandedPrompt` and :class:`StagedAssets` handed to
    :meth:`WorkflowGraphMutator.mutate` describe different chunks.

    Stage 2's regrouping means several plain-``int`` index spaces coexist
    (lyric line, aligned segment, chunk). Mixing them here would build a graph
    that renders one chunk's prompt over another chunk's audio stem and
    reference photo -- a defect visible only as the wrong face lip-syncing the
    wrong line in the finished video, long after the render cost was paid.
    """


class CloudApiNodeDetectedError(WorkflowGraphError):
    """A billed, remote cloud-API MiniMax node (lowercase ``m`` in "minimax")
    is present in the workflow. This graph must only use the local
    ``MiniMaxH3*`` nodes (core ComfyUI, no billing, no network)."""


# --------------------------------------------------------------------------- #
# Layer 1: general graph introspection
# --------------------------------------------------------------------------- #


def load_workflow_template(path: str | Path) -> Workflow:
    """Load and validate an API-format ComfyUI workflow JSON.

    Validates that the payload is a flat dict mapping node id -> node object,
    and that every node object has at least ``class_type`` and ``inputs``.
    Raises :class:`WorkflowTemplateError`, logged with context, for a missing
    file, malformed JSON, or a payload that isn't shaped like a workflow.
    """
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("failed to read workflow template %s: %s", path, exc)
        raise WorkflowTemplateError(
            f"workflow template not found or unreadable: {path} ({exc})"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("workflow template %s is not valid JSON: %s", path, exc)
        raise WorkflowTemplateError(
            f"workflow template {path} is not valid JSON: {exc}"
        ) from exc

    _validate_workflow_shape(payload, path)
    return payload


def _validate_workflow_shape(payload: object, path: Path) -> None:
    if not isinstance(payload, dict):
        logger.error(
            "workflow template %s must be a JSON object mapping node id -> node, got %s",
            path,
            type(payload).__name__,
        )
        raise WorkflowTemplateError(
            f"workflow template {path} must be a JSON object mapping node id -> "
            f"node, got {type(payload).__name__}"
        )
    for node_id, node in payload.items():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            logger.error(
                "workflow template %s: node %r is malformed (must have "
                "'class_type' and 'inputs'): %r",
                path,
                node_id,
                node,
            )
            raise WorkflowTemplateError(
                f"workflow template {path}: node {node_id!r} must be an object "
                "with 'class_type' and 'inputs'"
            )


def find_nodes_by_class_type(workflow: Workflow, class_type: str) -> list[tuple[str, dict]]:
    """Return every ``(node_id, node)`` pair whose ``class_type`` matches,
    in the workflow dict's iteration order (deterministic for a given
    workflow -- never relies on node id sort order, which shifts freely
    across canvas edits)."""
    return [
        (node_id, node)
        for node_id, node in workflow.items()
        if node.get("class_type") == class_type
    ]


def find_one_node(
    workflow: Workflow, class_type: str, *, title: str | None = None
) -> tuple[str, dict]:
    """Locate exactly one node of ``class_type``.

    Raises :class:`NodeNotFoundError` (logged) if no node of that
    ``class_type`` exists at all, and :class:`AmbiguousNodeError` (logged) if
    more than one exists and ``title`` either wasn't given or didn't narrow
    the match down to exactly one. Both errors name the ``class_type`` and
    the candidate node ids.
    """
    matches = find_nodes_by_class_type(workflow, class_type)

    if not matches:
        known = sorted({node.get("class_type") for node in workflow.values()})
        logger.error(
            "no node with class_type=%r found in workflow (known class_types: %s)",
            class_type,
            known,
        )
        raise NodeNotFoundError(f"no node with class_type {class_type!r} found in workflow")

    if len(matches) == 1:
        return matches[0]

    if title is not None:
        titled = [(nid, n) for nid, n in matches if n.get("_meta", {}).get("title") == title]
        if len(titled) == 1:
            return titled[0]

    candidate_ids = [node_id for node_id, _ in matches]
    logger.error(
        "ambiguous class_type=%r: %d candidate nodes %s (title disambiguator=%r did not "
        "resolve to exactly one)",
        class_type,
        len(matches),
        candidate_ids,
        title,
    )
    raise AmbiguousNodeError(
        f"ambiguous class_type {class_type!r}: candidate node ids {candidate_ids}; "
        "provide a title or resolve by wiring (see resolve_upstream_node)"
    )


def _is_link_pair(value: object) -> bool:
    """True if ``value`` is a plain ``[node_id, output_index]`` reference."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _first_referenced_node_id(value: object) -> str | None:
    """Extract the upstream node id from an input value that is a plain
    ``[node_id, output_index]`` link. Returns ``None`` if ``value`` isn't a
    recognizable node-reference shape at all.

    A populated AUTOGROW_V3 slot (``ref_images.ref_image_0`` etc.) needs no
    special handling here -- confirmed (issue #18) to serialize as its own
    flat, dotted key holding exactly this plain-link shape, not a nested
    list.
    """
    if _is_link_pair(value):
        return value[0]
    return None


def resolve_upstream_node(
    workflow: Workflow,
    consumer_class_type: str,
    input_name: str,
    *,
    consumer_title: str | None = None,
) -> tuple[str, dict]:
    """Resolve which upstream node feeds a given input socket on the (single)
    node of ``consumer_class_type``.

    This is the wiring-based disambiguator: when two nodes share a
    ``class_type`` and an un-renamed default title (e.g. two ``VAELoader``
    nodes), the only reliable way to tell them apart is which downstream
    socket their output is wired into -- e.g. "the ``VAELoader`` feeding
    ``MiniMaxH3ReferenceToVideo.inputs.vae``" vs "... ``.inputs.audio_vae``".

    ``input_name`` can be a plain input (``"vae"``) or a populated
    AUTOGROW_V3 slot's dotted key (``"ref_images.ref_image_0"``) -- both hold
    the same plain ``[node_id, output_index]`` link shape.

    Raises :class:`NodeNotFoundError`/:class:`AmbiguousNodeError` (via
    :func:`find_one_node`) if the consumer itself can't be located uniquely,
    or :class:`DanglingNodeReferenceError` (logged) if the referenced input
    is absent, not a recognizable node reference, or points at a node id not
    present in the workflow.
    """
    consumer_id, consumer_node = find_one_node(workflow, consumer_class_type, title=consumer_title)

    if input_name not in consumer_node["inputs"]:
        logger.error(
            "%s node (id=%s) has no input %r to resolve wiring for",
            consumer_class_type,
            consumer_id,
            input_name,
        )
        raise DanglingNodeReferenceError(
            f"{consumer_class_type} (id={consumer_id}) has no input {input_name!r}"
        )

    raw_value = consumer_node["inputs"][input_name]
    upstream_id = _first_referenced_node_id(raw_value)

    if upstream_id is None:
        logger.error(
            "%s node (id=%s) input %r is not a node reference: %r",
            consumer_class_type,
            consumer_id,
            input_name,
            raw_value,
        )
        raise DanglingNodeReferenceError(
            f"{consumer_class_type} (id={consumer_id}) input {input_name!r} is not a "
            f"node reference: {raw_value!r}"
        )

    if upstream_id not in workflow:
        logger.error(
            "%s node (id=%s) input %r references missing node id %r -- dangling reference "
            "(a node was likely deleted from the template without rewiring its consumers)",
            consumer_class_type,
            consumer_id,
            input_name,
            upstream_id,
        )
        raise DanglingNodeReferenceError(
            f"{consumer_class_type} (id={consumer_id}) input {input_name!r} references "
            f"missing node id {upstream_id!r}"
        )

    return upstream_id, workflow[upstream_id]


def assert_no_cloud_api_nodes(workflow: Workflow) -> None:
    """Raise :class:`CloudApiNodeDetectedError` (logged) if any billed,
    remote cloud-API MiniMax node is present in ``workflow``.

    These spell "Minimax" with a lowercase ``m`` (vs the local ``MiniMaxH3``
    nodes' capital ``M``) -- an easy misgrab that would otherwise silently
    route a render through a billed cloud API instead of the local GPU.
    """
    found = [
        (node_id, node.get("class_type"))
        for node_id, node in workflow.items()
        if node.get("class_type") in CLOUD_API_CLASS_TYPES
    ]
    if found:
        logger.error(
            "cloud-API MiniMax node(s) present in workflow -- billed, remote, forbidden: %s",
            found,
        )
        raise CloudApiNodeDetectedError(
            f"cloud-API MiniMax node(s) present in workflow: {found}; use the local "
            "MiniMaxH3* nodes instead"
        )


def _find_h3_conditioning_node(workflow: Workflow) -> tuple[str, dict]:
    """Locate the single H3 node that owns the ``prompt`` STRING input --
    either ``MiniMaxH3ReferenceToVideo`` (the base T2V/reference path) or
    ``MiniMaxH3ImageToVideo`` (the I2V continuity path, issue #12).

    Exactly one of the two must be present in a well-formed template -- a
    template mixing both, or neither, is malformed. Raises
    :class:`NodeNotFoundError` (logged, naming both candidate class_types) if
    neither is present, or :class:`AmbiguousNodeError` (logged) if both are.
    """
    ref_matches = find_nodes_by_class_type(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    i2v_matches = find_nodes_by_class_type(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO)

    if ref_matches and i2v_matches:
        logger.error(
            "workflow has both %s and %s nodes -- ambiguous which one owns the prompt input",
            CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
            CLASS_TYPE_H3_IMAGE_TO_VIDEO,
        )
        raise AmbiguousNodeError(
            f"workflow has both {CLASS_TYPE_H3_REFERENCE_TO_VIDEO!r} and "
            f"{CLASS_TYPE_H3_IMAGE_TO_VIDEO!r} nodes -- exactly one is expected"
        )
    if ref_matches:
        return find_one_node(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    if i2v_matches:
        return find_one_node(workflow, CLASS_TYPE_H3_IMAGE_TO_VIDEO)

    known = sorted({node.get("class_type") for node in workflow.values()})
    logger.error(
        "no node with class_type %r or %r found in workflow (known class_types: %s)",
        CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
        CLASS_TYPE_H3_IMAGE_TO_VIDEO,
        known,
    )
    raise NodeNotFoundError(
        f"no node with class_type {CLASS_TYPE_H3_REFERENCE_TO_VIDEO!r} or "
        f"{CLASS_TYPE_H3_IMAGE_TO_VIDEO!r} found in workflow"
    )


def read_text_encoder(workflow: Workflow) -> str | None:
    """Report the ``CLIPLoader.clip_name`` this graph would load (issue #39),
    or ``None`` when the graph names no encoder at all.

    Read, never write: this is how a run records which encoder produced its
    conditioning even when it pins nothing, so a template edited on the canvas
    to a different precision is caught by ``--resume`` instead of quietly
    producing footage nothing can distinguish from the previous take. ``None``
    (no ``CLIPLoader``, or a ``clip_name`` that is not a string) is
    *unrecorded* and, per :class:`~contracts.ChunkFingerprint`, never compares
    equal to a named encoder.

    Introspective, so it never raises: a caller asking "what does this graph
    use?" of a graph that uses nothing gets an answer, not an exception. The
    mutator's *injection* path is the one that refuses (see
    :meth:`WorkflowGraphMutator.mutate`) -- there, a missing ``CLIPLoader``
    means the pin the operator asked for silently did not happen.
    """
    matches = find_nodes_by_class_type(workflow, CLASS_TYPE_CLIP_LOADER)
    if len(matches) != 1:
        logger.debug(
            "workflow has %d %s node(s) -- no single text encoder identity to record",
            len(matches),
            CLASS_TYPE_CLIP_LOADER,
        )
        return None
    name = matches[0][1].get("inputs", {}).get(TEXT_ENCODER_INPUT)
    return name if isinstance(name, str) else None


def find_titled_node(workflow: Workflow, class_type: str, title: str) -> tuple[str, dict]:
    """Locate the one node of ``class_type`` whose ``_meta.title`` is exactly
    ``title``, requiring the title to *match*.

    This is deliberately stricter than :func:`find_one_node`, where ``title``
    is only a tiebreaker consulted when several candidates exist -- a sole
    candidate is returned there whatever it is called. That is right when the
    title is a hint for disambiguating, and wrong when the title is an
    *identity* claim: the I2V template (issue #12) carries two ``LoadImage``
    nodes, the cast reference photo and the seed frame, and they are told
    apart by nothing but their titles. Resolve those by hint on a template
    that happens to hold only one ``LoadImage`` and both lookups land on the
    same node, so the seed frame silently overwrites the cast photo and the
    render carries no cast face -- visible only after the GPU hours are
    spent. ComfyUI does not enforce node titles, and the real I2V template is
    still unauthored (issue #18), so this must fail loudly instead.

    Raises :class:`NodeNotFoundError` (logged, naming the title and the
    titles actually present) if nothing matches, and
    :class:`AmbiguousNodeError` if several nodes share the title.
    """
    matches = find_nodes_by_class_type(workflow, class_type)
    titled = [(nid, n) for nid, n in matches if n.get("_meta", {}).get("title") == title]

    if not titled:
        present = [n.get("_meta", {}).get("title") for _, n in matches]
        logger.error(
            "no %r node titled %r in workflow (titles present on %d %r node(s): %s)",
            class_type,
            title,
            len(matches),
            class_type,
            present,
        )
        raise NodeNotFoundError(
            f"no {class_type!r} node titled {title!r} found in workflow "
            f"(titles present: {present})"
        )

    if len(titled) > 1:
        candidate_ids = [nid for nid, _ in titled]
        logger.error(
            "several %r nodes share the title %r: %s", class_type, title, candidate_ids
        )
        raise AmbiguousNodeError(
            f"several {class_type!r} nodes share the title {title!r}: {candidate_ids}"
        )

    return titled[0]


# --------------------------------------------------------------------------- #
# graph_fingerprint (issue #45): recording the graph itself, not just what
# was injected into it
# --------------------------------------------------------------------------- #

FINGERPRINT_MASKED_INPUT_KEYS: frozenset[str] = frozenset(
    {"prompt", "image", "audio", "filename_prefix", "noise_seed", "length"}
)
"""The six inputs :meth:`WorkflowGraphMutator.mutate` legitimately varies per
chunk (issue #45's Proposal, verbatim): the prompt string, the cast/extra
reference-photo and audio-stem filenames (``LoadImage.image`` /
``LoadAudio.audio``), ``SaveVideo.filename_prefix``,
``RandomNoise.noise_seed``, and the H3 node's ``length``. These are masked
out of :func:`graph_fingerprint` by *input key name*, wherever that key
occurs in the graph -- not scoped to a particular ``class_type`` -- because
the masking side of this function is the one place a whitelist is actually
correct: it is a short, closed, spec-named list of *values* the mutator
itself owns, not a list of *nodes* or *node kinds*. Node identity -- which
class_types exist, how they're wired, every other input -- is never
whitelisted; see :func:`graph_fingerprint`'s docstring for that half of the
design.

Known, narrow tradeoff: a future node that happens to reuse one of these six
names for something that is *not* a per-chunk value would be masked too,
incorrectly. That is the cost of matching the input-name list issue #45
specifies exactly, rather than inventing a per-``class_type`` mask table
this module would then have to keep hand-in-sync with the mutator."""

_FINGERPRINT_MASK_PLACEHOLDER = "<mvm:masked-per-chunk-value>"
"""A fixed marker, distinguishable from any real ComfyUI input value (a
string, int, float, bool, or a ``[node_id, output_index]`` link pair), so
masking never accidentally collides with -- and thereby fails to distinguish
from -- a legitimate unmasked value elsewhere in the graph."""

FINGERPRINT_DIGEST_LENGTH = 16
"""Matches :meth:`~music_video_maker.contracts.ChunkFingerprint.hash_prompt`'s
truncation -- both are SHA-256 digests trimmed for a state file a human
reads, not a security boundary."""


class GraphFingerprintError(WorkflowGraphError):
    """:func:`graph_fingerprint` was given something that isn't a non-empty
    workflow graph. A fingerprint of nothing proves nothing, and silently
    hashing an empty/malformed input would make every such caller collide on
    the same digest instead of surfacing the caller's bug."""


def graph_fingerprint(workflow: Workflow) -> str:
    """Return a short, stable hex digest of the *mutated* graph's shape and
    constants (issue #45), with the per-chunk values masked out.

    This is the general fix underneath two narrower ones: the noise seed
    (#38) and the text encoder (#39) were each recorded by naming the one
    field that could silently change a render. The template surrounding both
    of them was the actual blind spot -- editing ``workflow_api.json`` or
    ``workflow_i2v_api.json`` (a sampler, a step count, a VAE file, the
    resolution default, a rewired socket) changes every pixel and nothing in
    ``ChunkFingerprint`` recorded it, so ``--resume`` reused the old chunks
    anyway. #44 is the concrete instance: removing ``last_frame`` from the
    I2V template changed every chained chunk, and there was no field to
    catch it -- forcing the re-render meant hand-deleting 12 results from
    ``run_state.json``.

    **Mask, don't whitelist.** Only :data:`FINGERPRINT_MASKED_INPUT_KEYS` --
    the six inputs the mutator injects per chunk -- are replaced with a fixed
    placeholder before hashing. Every other input, on every node this module
    has ever seen *and every node it hasn't*, counts by default. A whitelist
    of "the inputs that matter" would reproduce the #45 defect itself the
    first time someone adds a node with an input nobody thought to name.

    **Structure, not bytes.** ``workflow`` is re-serialized to JSON with
    sorted keys before hashing, so reformatting, reindenting, or reordering
    keys in the template file never moves the digest -- only the parsed
    values do.

    **Node ids are not identity.** This module's hard, standing invariant
    (see the module docstring) is that a ComfyUI canvas edit renumbers node
    ids freely without changing what renders, which is why every lookup
    above resolves by ``class_type`` (plus title or wiring), never by id.
    ``graph_fingerprint`` extends that invariant to the fingerprint itself:
    before hashing, every node is relabeled by a *structural* signature --
    its own class_type/title/inputs, plus, transitively, the same signature
    of whatever it is wired to -- instead of its id string, and every
    ``[node_id, output_index]`` link is rewritten to point at the relabeled
    id (see :func:`_canonicalize_for_fingerprint`). A canvas round-trip that
    renumbers every node but changes nothing about the render (a routine
    save-and-reopen) therefore does not force a whole song's worth of chunks
    to re-render.

    This is a deliberate choice, not the only defensible one: hashing raw ids
    is simpler and would also be correct per the acceptance criteria. It was
    rejected here because it makes the *common* case (an export that
    renumbers for no semantic reason) indistinguishable from the *rare* one
    (an actual rewire) -- the wrong default for an input this expensive to
    get wrong, on a codebase that already treats "id ordering never means
    anything" as load-bearing everywhere else. The known cost of the choice
    made here: two nodes that are genuinely structurally identical in every
    way (same class_type, title, scalar inputs, and wiring, all the way down
    the graph) are not distinguishable by this signature and fall back to
    their position in the source dict to stay deterministic -- not a shape
    any committed H3 template has, or is likely to grow.

    **Expect noisy invalidation.** Any actual structural edit to a committed
    template -- a changed sampler, a bumped step count, a different VAE file,
    a rewired socket -- re-renders every chunk that used it. That is the
    correct default per issue #45's acceptance criteria, not a bug to work
    around; producing that noise on a real template edit is the entire point
    of this function.

    Raises :class:`GraphFingerprintError` (logged) if ``workflow`` is not a
    non-empty dict -- there is nothing to fingerprint.
    """
    if not isinstance(workflow, dict) or not workflow:
        logger.error("graph_fingerprint: workflow must be a non-empty dict, got %r", workflow)
        raise GraphFingerprintError(
            f"cannot fingerprint an empty or non-dict workflow: {workflow!r}"
        )

    masked = _mask_fingerprint_inputs(workflow)
    canonical = _canonicalize_for_fingerprint(masked)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_DIGEST_LENGTH]


def insert_lora(workflow: Workflow, *, lora: str | None, strength: float) -> Workflow:
    """Splice a ``LoraLoaderModelOnly`` between the ``UNETLoader`` and every
    consumer of its MODEL output (issue #62), in place, returning the same
    dict.

    This is deliberately *surgery*, not an input pin. ``text_encoder`` (#39)
    could be a pin because both templates already carry a ``CLIPLoader``;
    neither carries a LoRA node, and there is no honest way to author one
    into the committed templates that is inert when no LoRA is configured --
    ComfyUI's API format has no bypass, and a zero-strength loader still
    demands a real filename and still loads it.

    ``lora=None`` returns the graph untouched, so a run without one POSTs
    byte-identically to every run before this existed.

    **This is invisible to ``template_hash``.** That hash is taken over the
    *template*, before any mutation (see ``continuity._hash_template``), on
    the stated grounds that everything the mutator injects is either masked
    or has a fingerprint field of its own. A LoRA is the latter: it needs
    ``ChunkFingerprint.lora``/``lora_strength``, and it is the fourth
    occurrence of the blind spot behind #38 (seed), #39 (encoder) and #45
    (graph) -- an input that decides the pixels with nothing writing it down.
    """
    if lora is None:
        return workflow
    if not isinstance(lora, str) or not lora.strip():
        raise WorkflowGraphError(f"lora must be a non-empty string, got {lora!r}")
    if PurePosixPath(lora).is_absolute() or PureWindowsPath(lora).is_absolute():
        raise WorkflowGraphError(
            f"lora must be relative to ComfyUI's models/loras/, got the absolute path "
            f"{lora!r} -- ComfyUI joins it under that directory, so the load would fail "
            "on the render host after custody of the GPU was taken (issue #62)"
        )
    if find_nodes_by_class_type(workflow, CLASS_TYPE_LORA_LOADER):
        raise WorkflowGraphError(
            "this graph already carries a "
            f"{CLASS_TYPE_LORA_LOADER}; inserting a second one would stack two adapters "
            "and silently double the strength (issue #62)"
        )

    unet_id, _ = find_one_node(workflow, CLASS_TYPE_UNET_LOADER_NAME)
    lora_id = _free_node_id(workflow)

    # Rewire first, then add -- so the new node's own `model` input, written
    # below, cannot be caught by the rewrite of everyone else's.
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and value[0] == unet_id:
                inputs[key] = [lora_id, value[1]]

    workflow[lora_id] = {
        "class_type": CLASS_TYPE_LORA_LOADER,
        "inputs": {
            "model": [unet_id, 0],
            LORA_NAME_INPUT: lora.strip(),
            LORA_STRENGTH_INPUT: float(strength),
        },
    }
    logger.info(
        "LoRA %s inserted at strength %s between %s node %s and %d consumer(s) (issue #62)",
        lora.strip(),
        strength,
        CLASS_TYPE_UNET_LOADER_NAME,
        unet_id,
        sum(
            1
            for node in workflow.values()
            for value in node.get("inputs", {}).values()
            if isinstance(value, list) and len(value) == 2 and value[0] == lora_id
        ),
    )
    return workflow


def _free_node_id(workflow: Workflow) -> str:
    """A node id no existing node uses. Derived from the graph rather than
    hardcoded -- user edits on the ComfyUI canvas renumber everything, which
    is why nothing in this module hardcodes an id."""
    numeric = [int(nid) for nid in workflow if str(nid).isdigit()]
    return str(max(numeric, default=0) + 1)


def _mask_fingerprint_inputs(workflow: Workflow) -> Workflow:
    """Return a new workflow dict with every
    :data:`FINGERPRINT_MASKED_INPUT_KEYS` input replaced by a fixed
    placeholder.

    Builds a fresh dict/inputs-dict at every level rather than mutating in
    place: ``workflow`` is the graph about to be (or that already was)
    POSTed to ComfyUI, and the caller must get its own object back completely
    unchanged.
    """
    masked: Workflow = {}
    for node_id, node in workflow.items():
        new_inputs = {}
        for key, value in node.get("inputs", {}).items():
            new_inputs[key] = (
                _FINGERPRINT_MASK_PLACEHOLDER if key in FINGERPRINT_MASKED_INPUT_KEYS else value
            )
        new_node = {k: v for k, v in node.items() if k != "inputs"}
        new_node["inputs"] = new_inputs
        masked[node_id] = new_node
    return masked


def _fingerprint_node_local_signature(node: dict) -> str:
    """The part of a node's identity that doesn't depend on what it is wired
    to: every field on the node other than ``inputs`` (``class_type``,
    ``_meta`` -- and anything else a future node happens to carry, per "mask,
    don't whitelist"), plus its own non-link (scalar) inputs."""
    scalar_inputs = {
        key: value for key, value in node.get("inputs", {}).items() if not _is_link_pair(value)
    }
    other_fields = {key: value for key, value in node.items() if key != "inputs"}
    payload = {"other_fields": other_fields, "scalar_inputs": scalar_inputs}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _canonicalize_for_fingerprint(workflow: Workflow) -> dict[str, dict]:
    """Relabel every node by a structural signature instead of its id, and
    rewrite every ``[node_id, output_index]`` link to match, so the result
    of :func:`graph_fingerprint` is independent of node ids (see its
    docstring for why that is the deliberate choice made here).

    Uses iterative signature refinement, bounded at ``len(workflow) + 1``
    rounds -- the standard bound for this kind of color-refinement graph
    canonicalization to reach a fixed point on a graph this small. Round 0 is
    each node's local signature (:func:`_fingerprint_node_local_signature`);
    round N+1 folds in the round-N signatures of whatever each node's link
    inputs point to, so two nodes only end up with the same final signature
    if their whole upstream shape matches too. Nodes are then ordered by
    their final signature, and that order -- never the original id string --
    becomes the canonical id.
    """
    node_ids = list(workflow.keys())
    signatures = {
        node_id: _fingerprint_node_local_signature(workflow[node_id]) for node_id in node_ids
    }

    for _ in range(len(node_ids) + 1):
        next_signatures = {}
        for node_id in node_ids:
            node = workflow[node_id]
            link_signatures = []
            for key, value in node.get("inputs", {}).items():
                if _is_link_pair(value):
                    ref_id, output_index = value
                    ref_signature = signatures.get(ref_id, f"<dangling:{ref_id}>")
                    link_signatures.append((key, ref_signature, output_index))
            payload = {"self": signatures[node_id], "links": sorted(link_signatures)}
            combined = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            next_signatures[node_id] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if next_signatures == signatures:
            break
        signatures = next_signatures

    # Deterministic even in the (unseen, in any committed template) case of
    # two nodes sharing an identical final signature: Python's sort is
    # stable, so a genuine tie falls back to node_ids' original relative
    # order rather than raising or picking arbitrarily.
    ordered_ids = sorted(node_ids, key=lambda node_id: signatures[node_id])
    width = max(1, len(str(len(ordered_ids) - 1))) if ordered_ids else 1
    canonical_id_of = {
        node_id: f"n{index:0{width}d}" for index, node_id in enumerate(ordered_ids)
    }

    canonical: dict[str, dict] = {}
    for node_id in node_ids:
        node = workflow[node_id]
        new_inputs = {}
        for key, value in node.get("inputs", {}).items():
            if _is_link_pair(value):
                ref_id, output_index = value
                new_inputs[key] = [
                    canonical_id_of.get(ref_id, f"<dangling:{ref_id}>"),
                    output_index,
                ]
            else:
                new_inputs[key] = value
        other_fields = {key: value for key, value in node.items() if key != "inputs"}
        canonical[canonical_id_of[node_id]] = {**other_fields, "inputs": new_inputs}
    return canonical


# --------------------------------------------------------------------------- #
# Layer 2: the mutator
# --------------------------------------------------------------------------- #


class WorkflowGraphMutator:
    """Implements ``contracts.WorkflowMutator`` (issue #8).

    Locates every node it injects into by ``class_type`` -- never by
    hardcoded node id -- and always deep-copies the template first, so the
    caller's in-memory template object is never mutated (Stage 4b reuses one
    loaded template across every chunk in a run).
    """

    def mutate(
        self,
        template: Workflow,
        prompt: ExpandedPrompt,
        assets: StagedAssets,
        *,
        node_input_overrides: dict[str, dict[str, object]] | None = None,
        cast_image_title: str | None = None,
        seed_frame_title: str | None = None,
        length_frames: int | None = None,
        noise_seed: int = DEFAULT_NOISE_SEED,
        text_encoder: str | None = None,
        lora: str | None = None,
        lora_strength: float = 1.0,
    ) -> Workflow:
        """Produce a per-chunk-ready workflow from ``template``.

        Injects:

        - ``prompt.prompt`` into whichever H3 node owns the ``prompt``
          STRING input -- ``MiniMaxH3ReferenceToVideo`` on the base
          template, or ``MiniMaxH3ImageToVideo`` on the I2V continuity
          template (issue #12); see :func:`_find_h3_conditioning_node`.
        - ``LoadImage.inputs.image = assets.image_filename`` (the cast
          reference photo). ``cast_image_title`` disambiguates which
          ``LoadImage`` this is when the template has more than one -- the
          I2V template does, since it also carries a second, seed-frame
          ``LoadImage``. ``None`` (the default) works unchanged for a
          template with exactly one ``LoadImage``, which is every template
          this method handled before issue #12. When a title *is* given it
          is an identity claim, not a hint: it is resolved through
          :func:`find_titled_node` and must actually match, so a template
          whose nodes are titled differently raises
          :class:`NodeNotFoundError` rather than quietly injecting into
          whatever single ``LoadImage`` it happens to contain.
        - ``LoadAudio.inputs.audio = assets.audio_filename``
        - ``SaveVideo.inputs.filename_prefix`` set to a deterministic
          per-chunk prefix derived from ``prompt.chunk_id``
          (:data:`CHUNK_FILENAME_PREFIX_FORMAT`, e.g. ``"mvm_chunk_0007"``).
          **Cross-lane contract**: Stage 4b (issue #9) relies on this exact
          format to tell chunk outputs apart in ComfyUI's history / on disk.

        ``assets.image_filenames[1:]`` (issue #33 levels 1-2) -- every staged
        reference photo beyond the primary -- is wired into
        ``ref_images.ref_image_1``, ``ref_image_2``, ... on the H3 node via
        :func:`_inject_extra_reference_images`, one newly synthesized
        ``LoadImage`` node per extra photo. ``assets.image_filenames`` is
        ``()`` by default, so for every caller that predates multiple
        references -- and for every ordinary single-reference chunk today --
        this is a complete no-op: no node is added, no input on the H3 node
        changes beyond what this method already did before issue #33. On the
        I2V template (no ``ref_images`` input at all) this logs a warning and
        renders the primary reference only rather than raising.

        ``length_frames`` is issue #20's frame-quantized ``length``, injected
        into the same H3 node this method already located -- so no caller has
        to know which of the two H3 class_types the template uses. It is a
        value, not a policy: this method still never *computes* a frame count
        or quantizes anything (issue #20's slicer owns that, and puts the
        answer on ``AudioChunk.frame_count``); it only refuses an obviously
        invalid one. A value off the ``length`` grid, or outside the model's
        trained range, raises :class:`InvalidLengthError` rather than being
        sent to a server that would silently round it -- silent rounding at
        execution time is the precise defect #20 exists to eliminate, so
        letting one through here would reintroduce it. ``None`` (the default)
        leaves whatever ``length`` the authored template carries.

        ``noise_seed`` is issue #38's ``RandomNoise.noise_seed``, injected into
        the single ``RandomNoise`` node -- located by ``class_type``, like
        everything else here, since both committed templates happen to number
        it ``10`` and a canvas edit would move it. It is written on **every**
        mutation, never conditionally: a seed left to whatever the template
        file carries is a render nobody can reproduce and a template edit
        nobody would notice, which is the defect. The default is
        :data:`DEFAULT_NOISE_SEED`, the value both templates already hold, so
        an unaware caller renders exactly what it always did.

        It is a per-call parameter rather than per-run state on purpose: the
        seed a chunk (or a re-roll of one chunk) renders with is a policy
        decision that belongs above this module, and this keeps varying it
        across chunks or takes a one-line change for the caller. A value
        outside ComfyUI's ``0 .. 2**64-1`` raises :class:`InvalidNoiseSeedError`
        rather than being silently clamped server-side -- the same reasoning as
        ``length_frames``: a value the server quietly alters means the seed the
        run recorded is not the seed that made the pixels. A template with no
        ``RandomNoise`` node raises :class:`NodeNotFoundError`; such a graph
        cannot be seeded, and cannot sample either.

        ``text_encoder`` is issue #39's ``CLIPLoader.clip_name``, injected into
        the single ``CLIPLoader`` -- located by ``class_type``, like everything
        else here. ``None`` (the default) leaves whatever encoder the authored
        template names, which is the behaviour of every run to date; a caller
        that names one is running the precision A/B, where the encoder is the
        only variable and every other input must stay fixed. Doing it here
        rather than by hand-editing the template means the swap is a run-level
        choice that the fingerprint can record, applies identically to the base
        and I2V templates (each carries its own ``CLIPLoader``), and leaves the
        committed artefacts untouched.

        Unlike ``noise_seed`` this is *not* written on every mutation: there is
        no project-wide default encoder to write, and inventing one here would
        mean this module, rather than the authored template, decides which
        weights the ComfyUI host loads. A blank or non-``str`` value raises
        :class:`InvalidTextEncoderError`; a template with no ``CLIPLoader``
        raises :class:`NodeNotFoundError` rather than rendering the whole run
        through the encoder the caller was trying to replace.

        ``node_input_overrides`` is an optional seam -- ``{class_type:
        {input_name: value}}`` -- so later issues can push additional inputs
        (e.g. the CLI's ``width``/``height``) through without this module
        owning that policy. It is applied *after* every injection above,
        including the seed, so a caller reaching for the generic seam is never
        silently overruled by a default.
        Overrides are applied via :func:`find_one_node`, so an unknown
        ``class_type`` in ``node_input_overrides`` raises the same
        :class:`NodeNotFoundError` naming it, and an ambiguous ``class_type``
        (e.g. ``VAELoader``, which legitimately appears twice) raises
        :class:`AmbiguousNodeError` -- overrides do not currently support a
        title disambiguator.

        ``seed_frame_title`` is issue #12's I2V continuity bridge. When
        ``None`` (the default), behavior is unchanged from Stage 4a: if
        ``assets.seed_frame_filename`` happens to be set anyway, this logs an
        info note naming it rather than silently dropping it, but does not
        otherwise touch the graph. When a title *is* given and
        ``assets.seed_frame_filename`` is set, the second, seed-frame
        ``LoadImage`` (located by that title -- never by node id) gets
        ``assets.seed_frame_filename`` injected into its ``image`` input.
        Giving a ``seed_frame_title`` with no ``assets.seed_frame_filename``
        logs a warning and skips injection rather than raising -- that
        combination means a caller reused the I2V template for a chunk that
        ended up on the fallback (non-bridged) path.

        Raises :class:`CloudApiNodeDetectedError` if ``template`` contains a
        billed cloud-API MiniMax node, and :class:`NodeNotFoundError` /
        :class:`AmbiguousNodeError` if a required node can't be located
        uniquely by ``class_type``. Every failure path logs before raising.
        """
        if prompt.chunk_id != assets.chunk_id:
            logger.error(
                "refusing to mutate mismatched chunks: prompt.chunk_id=%s != assets.chunk_id=%s",
                prompt.chunk_id,
                assets.chunk_id,
            )
            raise ChunkMismatchError(
                f"chunk id mismatch: prompt.chunk_id={prompt.chunk_id} != "
                f"assets.chunk_id={assets.chunk_id}"
            )

        workflow = copy.deepcopy(template)

        assert_no_cloud_api_nodes(workflow)

        _, h3_node = _find_h3_conditioning_node(workflow)
        h3_node["inputs"]["prompt"] = prompt.prompt

        if length_frames is not None:
            _validate_length_frames(length_frames, prompt.chunk_id)
            h3_node["inputs"]["length"] = length_frames

        _validate_noise_seed(noise_seed, prompt.chunk_id)
        if text_encoder is not None:
            _validate_text_encoder(text_encoder, prompt.chunk_id)

        if cast_image_title is None:
            _, image_node = find_one_node(workflow, CLASS_TYPE_IMAGE_LOAD)
        else:
            _, image_node = find_titled_node(workflow, CLASS_TYPE_IMAGE_LOAD, cast_image_title)
        image_node["inputs"]["image"] = assets.image_filename

        _inject_extra_reference_images(
            workflow,
            h3_node,
            assets.image_filenames[1:] if len(assets.image_filenames) > 1 else (),
            chunk_id=prompt.chunk_id,
        )

        _, audio_node = find_one_node(workflow, CLASS_TYPE_AUDIO_LOAD)
        audio_node["inputs"]["audio"] = assets.audio_filename

        _, save_node = find_one_node(workflow, CLASS_TYPE_VIDEO_SAVE)
        save_node["inputs"]["filename_prefix"] = CHUNK_FILENAME_PREFIX_FORMAT.format(
            chunk_id=prompt.chunk_id
        )

        if seed_frame_title is not None:
            if assets.seed_frame_filename:
                _, seed_node = find_titled_node(
                    workflow, CLASS_TYPE_IMAGE_LOAD, seed_frame_title
                )
                seed_node["inputs"]["image"] = assets.seed_frame_filename
                logger.info(
                    "chunk_id=%s: injected seed_frame_filename=%s into LoadImage titled %r",
                    prompt.chunk_id,
                    assets.seed_frame_filename,
                    seed_frame_title,
                )
            else:
                logger.warning(
                    "chunk_id=%s: seed_frame_title=%r given but assets.seed_frame_filename is "
                    "not set -- skipping seed-frame injection",
                    prompt.chunk_id,
                    seed_frame_title,
                )
        elif assets.seed_frame_filename:
            logger.info(
                "chunk_id=%s has seed_frame_filename=%s set, but I2V continuity bridging "
                "is out of scope for Stage 4a (issue #12) -- ignoring it",
                prompt.chunk_id,
                assets.seed_frame_filename,
            )

        # Last of the injections, immediately before the generic override
        # seam: the seed is the one input written on every mutation whatever
        # the caller asked for, so a template that fails an *identity* claim
        # above (a mistitled cast or seed-frame node) still reports that
        # failure rather than "no RandomNoise node".
        _, noise_node = find_one_node(workflow, CLASS_TYPE_RANDOM_NOISE)
        noise_node["inputs"][NOISE_SEED_INPUT] = noise_seed

        if text_encoder is not None:
            _, clip_node = find_one_node(workflow, CLASS_TYPE_CLIP_LOADER)
            previous = clip_node["inputs"].get(TEXT_ENCODER_INPUT)
            clip_node["inputs"][TEXT_ENCODER_INPUT] = text_encoder
            if previous != text_encoder:
                logger.info(
                    "chunk_id=%s: text encoder pinned to %s (template names %s)",
                    prompt.chunk_id,
                    text_encoder,
                    previous,
                )

        # Issue #62, last of all: graph *surgery*, not an input pin, so it
        # runs after every node-locating injection above -- a template that
        # fails an identity claim should report that failure rather than
        # "no UNETLoader node" from a graph this call has already rewired.
        insert_lora(workflow, lora=lora, strength=lora_strength)

        if node_input_overrides:
            _apply_node_input_overrides(workflow, node_input_overrides)

        logger.debug(
            "mutated workflow for chunk_id=%s (image=%s, audio=%s, filename_prefix=%s, "
            "noise_seed=%s)",
            prompt.chunk_id,
            assets.image_filename,
            assets.audio_filename,
            save_node["inputs"]["filename_prefix"],
            noise_seed,
        )

        return workflow


class PerChunkSeedMutator:
    """Decorates any :class:`~music_video_maker.contracts.WorkflowMutator`,
    replacing whatever ``noise_seed`` its caller passes with the per-chunk
    value :func:`resolve_chunk_seed` derives (issue #38, variation half).

    Why this exists rather than a change to the caller: the one place a run
    actually threads a seed into a chunk's graph today is
    ``continuity.py``'s ``ContinuityWorkflowProvider``, which -- by design,
    the same way it treats ``text_encoder`` and ``lora`` -- takes a single
    flat ``noise_seed`` for the whole run and passes it unchanged to every
    ``mutate()`` call (see its own constructor docstring). That module is
    outside this file's ownership for this change. Its constructor already
    exposes exactly the seam this needs, though: ``mutator: WorkflowMutator
    | None``. This class sits at that seam -- ``cli.py`` hands the provider a
    ``PerChunkSeedMutator`` wrapping the usual :class:`WorkflowGraphMutator`
    instead of the plain one, and every ``mutate()`` call the provider makes
    is intercepted here first. Nothing about ``continuity.py`` has to know
    this exists; it keeps calling ``mutate(..., noise_seed=self._noise_seed)``
    exactly as before, and gets a *different* number back in the graph than
    the one it asked for.

    ``reseed_generations`` (default empty) is how ``cli.py``'s ``--reseed``
    reaches this class: a chunk id present in the mapping renders at that
    generation instead of 0, so :func:`resolve_chunk_seed` guarantees it a
    seed that differs from every ordinary chunk's -- while every chunk *not*
    named stays on generation 0, the exact value it would have gotten
    without ``--reseed`` at all. That is what keeps the rest of a resumed
    run's chunks cache hits: their expected fingerprint doesn't move, only
    the reseeded chunk's does.
    """

    def __init__(
        self,
        inner: WorkflowMutator,
        *,
        base_seed: int,
        reseed_generations: Mapping[int, int] | None = None,
    ) -> None:
        self._inner = inner
        self._base_seed = base_seed
        self._reseed_generations = dict(reseed_generations or {})

    def mutate(
        self,
        template: Workflow,
        prompt: ExpandedPrompt,
        assets: StagedAssets,
        **kwargs: object,
    ) -> Workflow:
        """Delegate to the wrapped mutator with ``noise_seed`` replaced.

        Every other keyword argument -- ``length_frames``, ``text_encoder``,
        ``lora``, ``cast_image_title``, and anything a future caller adds --
        passes through untouched; this class only ever looks at
        ``prompt.chunk_id`` and the ``noise_seed`` key."""
        generation = self._reseed_generations.get(prompt.chunk_id, 0)
        kwargs["noise_seed"] = resolve_chunk_seed(
            self._base_seed, prompt.chunk_id, reseed_generation=generation
        )
        return self._inner.mutate(template, prompt, assets, **kwargs)


def _validate_length_frames(length_frames: int, chunk_id: int) -> None:
    """Refuse a ``length`` the H3 node cannot honor exactly. See
    :class:`InvalidLengthError`."""
    grid = H3_FRAME_GRID
    if not grid.is_valid(length_frames):
        logger.error(
            "chunk_id=%s: length=%d is not on H3's frame grid (%d + %dk); ComfyUI would "
            "silently round it while the audio stem keeps its own duration",
            chunk_id,
            length_frames,
            grid.base_frames,
            grid.step_frames,
        )
        raise InvalidLengthError(
            f"chunk_id={chunk_id}: length={length_frames} is not a valid H3 frame count "
            f"({grid.base_frames} + {grid.step_frames}k)"
        )
    if not (grid.trained_min_frames <= length_frames <= grid.trained_max_frames):
        logger.error(
            "chunk_id=%s: length=%d is outside H3's trained range (%d-%d frames); the model "
            "would be extrapolating",
            chunk_id,
            length_frames,
            grid.trained_min_frames,
            grid.trained_max_frames,
        )
        raise InvalidLengthError(
            f"chunk_id={chunk_id}: length={length_frames} is outside H3's trained range "
            f"({grid.trained_min_frames}-{grid.trained_max_frames} frames)"
        )


def _validate_noise_seed(noise_seed: object, chunk_id: int) -> None:
    """Refuse a ``noise_seed`` ComfyUI would clamp or reject. See
    :class:`InvalidNoiseSeedError`."""
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        logger.error(
            "chunk_id=%s: noise_seed=%r is %s, not an int -- RandomNoise.noise_seed is an "
            "INT input and the run must record the exact value it rendered under",
            chunk_id,
            noise_seed,
            type(noise_seed).__name__,
        )
        raise InvalidNoiseSeedError(
            f"chunk_id={chunk_id}: noise_seed={noise_seed!r} is not an int"
        )
    if not (0 <= noise_seed <= MAX_NOISE_SEED):
        logger.error(
            "chunk_id=%s: noise_seed=%d is outside ComfyUI's RandomNoise range (0-%d); it "
            "would be clamped server-side and the render would not match the recorded seed",
            chunk_id,
            noise_seed,
            MAX_NOISE_SEED,
        )
        raise InvalidNoiseSeedError(
            f"chunk_id={chunk_id}: noise_seed={noise_seed} is outside the valid range "
            f"(0-{MAX_NOISE_SEED})"
        )


def _validate_text_encoder(text_encoder: object, chunk_id: int) -> None:
    """Refuse a ``clip_name`` ComfyUI could not resolve. See
    :class:`InvalidTextEncoderError`."""
    if not isinstance(text_encoder, str) or not text_encoder.strip():
        logger.error(
            "chunk_id=%s: text_encoder=%r is not a usable CLIPLoader.clip_name -- it names a "
            "weights file inside ComfyUI's own models/text_encoders/",
            chunk_id,
            text_encoder,
        )
        raise InvalidTextEncoderError(
            f"chunk_id={chunk_id}: text_encoder={text_encoder!r} is not a non-empty string"
        )
    if Path(text_encoder).is_absolute():
        logger.error(
            "chunk_id=%s: text_encoder=%r is an absolute path; ComfyUI resolves clip_name "
            "*inside* its own models/text_encoders/, so this would fail at model-load time "
            "-- on a render run, after custody of the card was taken",
            chunk_id,
            text_encoder,
        )
        raise InvalidTextEncoderError(
            f"chunk_id={chunk_id}: text_encoder={text_encoder!r} is an absolute path, not a "
            "name relative to ComfyUI's models/text_encoders/"
        )


def _apply_node_input_overrides(
    workflow: Workflow, overrides: dict[str, dict[str, object]]
) -> None:
    for class_type, input_map in overrides.items():
        _, node = find_one_node(workflow, class_type)
        node["inputs"].update(input_map)


def _allocate_node_id(workflow: Workflow, hint: str) -> str:
    """Return a node id guaranteed absent from ``workflow``, based on
    ``hint``.

    Used only to synthesize brand-new ``LoadImage`` nodes for extra
    reference photos (issue #33) -- every other node in this module is
    located, never created, by ``class_type``. Node ids elsewhere in this
    codebase are opaque strings (not necessarily int-parseable, e.g. a
    canvas re-export), so this never assumes numeric ids; it just avoids
    colliding with whatever is already present.
    """
    candidate = hint
    suffix = 1
    while candidate in workflow:
        suffix += 1
        candidate = f"{hint}_{suffix}"
    return candidate


def _inject_extra_reference_images(
    workflow: Workflow,
    h3_node: dict,
    extra_filenames: tuple[str, ...],
    *,
    chunk_id: int,
) -> None:
    """Wire every reference photo beyond the primary into a freshly
    synthesized ``LoadImage`` node on ``ref_images.ref_image_1``,
    ``ref_image_2``, ... (issue #33 levels 1-2).

    ``extra_filenames`` is ``assets.image_filenames[1:]`` -- empty for
    every caller that predates multiple references, and for the ordinary
    single-reference case, so this is a no-op and touches nothing: the
    graph :meth:`WorkflowGraphMutator.mutate` returns for those callers is
    unaffected node-for-node.

    Only ``MiniMaxH3ReferenceToVideo`` has a ``ref_images`` input at all --
    ``MiniMaxH3ImageToVideo`` (the I2V continuity path, issue #12) does not.
    A chunk that reaches here with extra filenames while rendering through
    the I2V template logs a warning and renders with the primary reference
    only, rather than raising: multi-reference support is additive, and per
    issue #33 the design must degrade to today's known-good single-reference
    behavior on any card/path that can't (or doesn't yet) support it, never
    crash a chunk that would otherwise render fine.

    Follows the existing pattern from ``docs/h3-node-schema.md`` finding 3:
    the mutator never assigns a scalar value into ``ref_images`` itself,
    only a flat ``ref_images.ref_image_N`` dotted key holding a plain
    ``[node_id, output_index]`` link, exactly like the template-authored
    ``ref_image_0`` slot. Each extra slot's ``LoadImage`` node is newly
    synthesized here -- deliberately never added to the committed template
    itself, since the base template's single, title-free ``LoadImage`` node
    is relied on by every existing single-reference caller via
    ``find_one_node(..., title=None)``; permanently adding a second
    ``LoadImage`` node to the template would make that lookup ambiguous for
    every chunk, single-reference or not (confirmed against
    ``continuity.py``'s ``_render_base``, which calls ``mutate()`` with no
    ``cast_image_title`` at all).
    """
    if not extra_filenames:
        return

    if h3_node.get("class_type") != CLASS_TYPE_H3_REFERENCE_TO_VIDEO:
        logger.warning(
            "chunk_id=%s: %d extra reference filename(s) given but the H3 node is "
            "class_type=%r, which has no ref_images input -- rendering with the "
            "primary reference only",
            chunk_id,
            len(extra_filenames),
            h3_node.get("class_type"),
        )
        return

    for offset, filename in enumerate(extra_filenames, start=1):
        node_id = _allocate_node_id(workflow, f"mvm_ref_image_{offset}")
        workflow[node_id] = {
            "class_type": CLASS_TYPE_IMAGE_LOAD,
            "inputs": {"image": filename},
            "_meta": {"title": f"Load Extra Cast Reference {offset}"},
        }
        slot_key = f"{REF_IMAGES_AUTOGROW_FIELD}.{REF_IMAGE_SLOT_PREFIX}{offset}"
        h3_node["inputs"][slot_key] = [node_id, 0]
        logger.info(
            "chunk_id=%s: injected extra reference filename=%s into new LoadImage "
            "node id=%s -> %s",
            chunk_id,
            filename,
            node_id,
            slot_key,
        )
