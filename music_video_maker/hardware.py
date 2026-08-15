"""Static hardware profiles and workflow optimization-node awareness (issue #13).

Deliberately **static**. A render run takes *exclusive* GPU custody (issue
#19) -- anything else using the card is stopped before rendering starts,
freeing the whole card -- so a profile can safely assume its full nominal
VRAM. Do
**not** query ``nvidia-smi`` or ComfyUI's ``/system_stats`` here to derive
anything from live free VRAM; that was proposed and explicitly rejected in
issue #13's second comment. A live free-VRAM pre-flight belongs to the
custody handoff in issue #19, not this module.

Two things live here:

1. Named :class:`~music_video_maker.contracts.HardwareProfile` constants --
   VRAM size -> safe chunk-duration window (the bound issue #4's slicing
   enforces) + recommended optimization nodes.
2. A minimal workflow scan that reports which of a profile's
   ``recommended_nodes`` are absent from a loaded workflow template, logging
   a warning that names the exact custom-node repo to install.

Scope guard: the scan below only collects the set of ``class_type`` values
present in a :data:`~music_video_maker.contracts.Workflow` dict -- it is
intentionally *not* general graph introspection (locating nodes by title,
resolving links, disambiguating duplicates, etc). That machinery is issue
#8's, built by another lane. When #8 lands, this scan should delegate node
discovery to it rather than re-implement any part of it here.
"""

from __future__ import annotations

import logging

from music_video_maker.contracts import H3_FRAME_GRID, HardwareProfile, Workflow

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Optimization node -> custom-node repo, from issue #13's table
# --------------------------------------------------------------------------- #

NODE_SAGE_ATTENTION = "SageAttention"
"""Patch Sage/Sol-Attn attention node from ComfyUI-KJNodes -- drastically
faster generation than native PyTorch attention."""

NODE_SPECTRUM_MINIMAX_H3 = "SpectrumMiniMaxH3"
"""Chebyshev ridge fit on post-transformer features; skips expensive
transformer evaluations. Recommended for <24 GB VRAM hardware."""

NODE_EASY_CACHE = "EasyCache"
"""Caches intermediate diffusion steps. Fallback option if OOM errors
persist -- carries a quality-degradation risk, so it is a last resort."""

OPTIMIZATION_NODE_REPOS: dict[str, str] = {
    NODE_SAGE_ATTENTION: "ComfyUI-KJNodes",
    NODE_SPECTRUM_MINIMAX_H3: "ComfyUI-Spectrum-MiniMax-H3",
    NODE_EASY_CACHE: "Block/EasyCache",
}
"""Maps a recommended optimization node identifier to the exact custom-node
repo that provides it, for the install instructions in scan warnings."""


# --------------------------------------------------------------------------- #
# Named static profiles
# --------------------------------------------------------------------------- #
#
# min_chunk_seconds / max_chunk_seconds (issue #20)
# --------------------------------------------------
# These are duration *windows*, not the frame grid itself (that lives on
# ``HardwareProfile.frame_grid``, fixed at ``H3_FRAME_GRID`` for every
# profile -- it is a property of the model weights, not the card). Both
# bounds below are pinned to H3's *trained* frame range (124-362 frames),
# derived from the grid rather than hardcoded, so they can never silently
# drift out of sync with ``docs/h3-node-schema.md``:
#
#   min = H3_FRAME_GRID.frames_to_seconds(124) = ~5.167s
#   max = H3_FRAME_GRID.frames_to_seconds(362) = ~15.083s
#
# More VRAM does **not** raise the safe max duration -- that was this
# module's assumption before issue #20 (see the old PROFILE_48GB
# max_chunk_seconds=24.0), but it was wrong: 24s is *above* the range H3 was
# trained on, and generating past it means the model is extrapolating, not a
# VRAM tradeoff. VRAM headroom still buys something (fewer/no optimization
# nodes needed -- see ``recommended_nodes`` below), just not a longer window.
# ``slicing._effective_bounds`` clamps defensively into this same trained
# range at the point of use for any ``HardwareProfile`` (built here or by a
# caller's own config) that asks for something outside it, logging a warning
# naming both the requested and clamped values -- belt-and-suspenders with
# the profiles already being correct here.

_TRAINED_MIN_CHUNK_SECONDS = H3_FRAME_GRID.frames_to_seconds(H3_FRAME_GRID.trained_min_frames)
_TRAINED_MAX_CHUNK_SECONDS = H3_FRAME_GRID.frames_to_seconds(H3_FRAME_GRID.trained_max_frames)

PROFILE_RTX_4090_24GB = HardwareProfile(
    name="RTX 4090 24GB (doris)",
    vram_gb=24.0,
    min_chunk_seconds=_TRAINED_MIN_CHUNK_SECONDS,
    max_chunk_seconds=_TRAINED_MAX_CHUNK_SECONDS,
    recommended_nodes=(NODE_SAGE_ATTENTION, NODE_SPECTRUM_MINIMAX_H3, NODE_EASY_CACHE),
)
"""doris's card. Issue #13 put H3's quantized appetite at ~42.5 GB, which
would make 24 GB hopeless -- but that figure sums the weight *files*, and
ComfyUI loads them sequentially and offloads between stages, so the ~15.7 GB
text encoder and the ~21 GB diffusion model are never co-resident. Measured
reality on doris: 5-second clips render fast and at good quality with only
~16.4 GB free, other GPU services still resident. The optimization nodes are
still recommended here (they buy speed and headroom), but they are not the
difference between working and not working at this clip length.

Untested above ~124 frames: temporal VAE decode memory scales non-linearly
with frame count, and nothing longer than a 5s clip has been rendered on this
card yet. Chunks may reach 362 frames (issue #20), roughly 3x that."""

PROFILE_48GB = HardwareProfile(
    name="48GB (e.g. RTX 6000 Ada / A6000)",
    vram_gb=48.0,
    min_chunk_seconds=_TRAINED_MIN_CHUNK_SECONDS,
    max_chunk_seconds=_TRAINED_MAX_CHUNK_SECONDS,
    recommended_nodes=(NODE_SAGE_ATTENTION,),
)
"""Above H3's quantized VRAM appetite, but the chunk-duration window is
identical to the 24 GB profile -- the trained ceiling wins regardless of
available VRAM (issue #20). The only thing the extra headroom changes is
which optimization nodes are still worth recommending: only the pure-speed
one (Sage Attention), since Spectrum/EasyCache exist to survive tight VRAM,
which this profile does not have."""

PROFILES: dict[str, HardwareProfile] = {
    PROFILE_RTX_4090_24GB.name: PROFILE_RTX_4090_24GB,
    PROFILE_48GB.name: PROFILE_48GB,
}
"""Registry of every named static profile, keyed by :attr:`HardwareProfile.name`."""


class UnknownProfileError(KeyError):
    """Raised by :func:`get_profile` for a name not in :data:`PROFILES`."""


def get_profile(name: str) -> HardwareProfile:
    """Look up a named static profile from :data:`PROFILES`.

    Raises :class:`UnknownProfileError` (logged) for an unrecognized name --
    config-driven callers (``RunConfig.hardware``) build their own
    ``HardwareProfile`` directly and never need this lookup; it exists for
    code that wants one of the pre-canned profiles by name.
    """
    try:
        return PROFILES[name]
    except KeyError:
        logger.error(
            "Unknown hardware profile name=%r; known profiles: %s", name, sorted(PROFILES)
        )
        raise UnknownProfileError(name) from None


# --------------------------------------------------------------------------- #
# Workflow-template scan
# --------------------------------------------------------------------------- #


def _class_types_present(workflow: Workflow) -> set[str]:
    """Collect the ``class_type`` of every node dict in ``workflow``.

    Minimal by design -- see the scope-guard note in the module docstring.
    """
    present: set[str] = set()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            logger.warning(
                "Workflow node %r is not a dict (got %s); skipping during hardware scan",
                node_id,
                type(node).__name__,
            )
            continue
        class_type = node.get("class_type")
        if class_type:
            present.add(class_type)
    return present


def scan_workflow_for_missing_optimizations(
    workflow: Workflow, profile: HardwareProfile
) -> tuple[str, ...]:
    """Report which of ``profile.recommended_nodes`` are absent from ``workflow``.

    For every missing node, logs a warning naming the exact custom-node repo
    to install (from :data:`OPTIMIZATION_NODE_REPOS`). Returns the missing
    node identifiers in the order they appear on the profile, so callers can
    act on (or just log) them without re-deriving anything.
    """
    present = _class_types_present(workflow)
    missing = tuple(node for node in profile.recommended_nodes if node not in present)

    for node in missing:
        repo = OPTIMIZATION_NODE_REPOS.get(node)
        if repo is None:
            logger.warning(
                "Recommended optimization node %r (profile=%r) is absent from the workflow "
                "template, but no known custom-node repo is registered for it.",
                node,
                profile.name,
            )
        else:
            logger.warning(
                "Recommended optimization node %r (profile=%r) is absent from the workflow "
                "template. Install it from %r to enable this optimization.",
                node,
                profile.name,
                repo,
            )

    if not missing:
        logger.info(
            "All %d recommended optimization node(s) for profile=%r are present.",
            len(profile.recommended_nodes),
            profile.name,
        )

    return missing
