"""Generate a synthetic cast member's reference imagery (issue #56, part 2).

Part 1 of #56 shipped the *queryable* half -- ``CastMember.synthetic`` /
``[cast.<name>.origin]`` in ``contracts.py``/``config.py`` -- and the offline
acceptance check, :mod:`music_video_maker.castcheck`. What it deliberately
deferred was the model decision. That decision was made on 2026-10-04:
**Krea 2**, which is already on the render host, so no new model family and no
new disk is spent (``docs/design-synthetic-cast.md``'s first constraint).

This module is the generator. It never scores a set -- ``castcheck`` already
does, and it is called from here rather than reimplemented -- and it never
decides whether a character is good enough: it produces images, records how,
and hands the numbers back.

Where this lives, and why it is not in ``authoring/``
------------------------------------------------------
This is an **authoring-time** tool: a character is generated once, by hand,
and its images are committed as *run assets* under ``~/mvm-runs/<song>/cast/``
-- never into this repo, and never during a render. The render path still just
loads a file off disk (``ref_images`` receives a photo, of nobody).

It is a module CLI (``python -m music_video_maker.castgen``), the same shape
``castcheck``/``facescan``/``calibrate_voicing``/``vramsample`` already use,
rather than a fourth ``[project.scripts]`` console script, because it is not
load-bearing for any run.

It sits at the top level rather than inside ``music_video_maker/authoring/``
on purpose. ``authoring/`` may only reach into the render half through a small
named allowlist (``tests/test_authoring_boundary.py``), and this module needs
:mod:`music_video_maker.workflow_graph`, :mod:`music_video_maker.staging` and
:mod:`music_video_maker.custody`. Adding three render-side modules to that
allowlist to host one authoring tool would blur the exact boundary the list
exists to keep sharp. Instead the boundary that matters here is enforced from
the other side: ``test_authoring_boundary.py`` asserts that **nothing in the
package imports this module**, so "does the render binary ever call an image
model?" stays answerable by reading an import graph.

The identity problem, and the method actually available on doris
----------------------------------------------------------------
The hard part of #56 is not a portrait, it is a character who reads as the
same person across eighty shots. The design doc is explicit that a set must
**not** be built by re-rolling one prompt at different seeds -- that is four
different people -- and names the three methods that hold identity:
img2img/inpaint from the first image, IP-Adapter-style conditioning on it, or
a small LoRA trained on it.

Of those three, exactly one is available with core ComfyUI nodes and the
weights installed on the render host, which is why this module implements
**an anchor plus img2img views**:

* ComfyUI's own shipped Krea 2 blueprints were read for the graph shape
  (``blueprints/Text to Image (Krea-2 Turbo).json``). The reference-conditioning
  route in its sibling blueprint (``Image Style Reference``) needs
  ``krea2_style_reference.safetensors``, which is **not** installed; the only
  Krea 2 LoRA on the host is ``krea2_darkbrush.safetensors``, whose own
  trigger phrase in that blueprint is "muted minimalist sketch style" -- a
  style adapter, exactly the wrong thing for a photoreal cast portrait, and
  issue #62 already measured what happens when a LoRA and the prompt want
  different pictures (the stronger signal wins silently).
* So: view 1 is text-to-image (the anchor), and every other view is
  ``LoadImage`` -> ``VAEEncode`` -> ``KSampler(denoise < 1)`` from an earlier
  view's *rendered file*. Two committed templates, never one rewritten:
  ``workflow_cast_api.json`` and ``workflow_cast_view_api.json``, the same
  base-vs-I2V split ``docs/workflow-template-guide.md`` describes.

The re-roll prohibition is **mechanical here, not advisory**: a spec whose
second view has no ``from``, or whose derived view sets ``denoise >= 1.0``
(which destroys the anchor latent completely, i.e. re-rolls the prompt), is
refused by :func:`load_character_spec` before any GPU time is spent. There is
no default ``denoise``: what value holds identity for Krea 2 is unmeasured,
and inventing a calibrated-looking default would be this project's
most-repeated bug in a new place.

Krea 2 Community License -- recorded, and what it obliges
---------------------------------------------------------
See :data:`KREA2_COMMUNITY_LICENCE` and :data:`KNOWN_IMAGE_MODELS`, kept
beside the code that loads these weights exactly the way
``workflow_graph.KNOWN_LORAS`` and ``faces.py`` record theirs. Three terms
reach this code rather than only the README's rights table:

1. **Commercial use is free under $1M annual revenue and under 50 seats.**
   Beyond either, the deployer needs Krea's commercial terms. Recorded in
   every manifest this module writes, so a published asset carries the
   condition it was produced under.
2. **A derivative model's name must begin with "Krea".** Nothing here trains
   or merges a model, so nothing here can breach it -- but if a cast LoRA is
   ever trained on this output, it must be named ``krea2-...``.
3. **No classifiers ship with the open weights, so filtering is explicitly
   the deployer's responsibility.** This module ships no classifier and does
   not pretend to: every manifest records ``filtering = "none"`` with that
   reason, and every generate run logs it, so no provenance record can be
   read as "something checked these images". A human looks at every image
   before it is used. That is also why the one thing ``castcheck``'s docstring
   says it cannot tell you -- mode collapse -- is printed beside its numbers
   here rather than inferred from them.

Does the 0.34 SFace floor transfer to generated faces?
------------------------------------------------------
**Unknown, and this module does not assume it does.** 0.34 was calibrated on
16 pairs of photographs of real people (``docs/seed-face-recognition.md``);
generated faces can sit *closer* to each other than two photographs of one
person do, which is mode collapse wearing identity's clothes. So a ``pass``
from ``castcheck`` here is necessary and not sufficient, the report says so
every time, and :class:`CharacterResult` additionally reports the pairwise
**spread** (min/mean/max) as numbers with no verdict attached: a set of views
that were authored to differ in angle and light, all scoring near-identical,
is the signature to go look at. Re-deriving the floor on synthetic pairs is
still open work, and it needs generated material to measure -- which is what
this module finally produces.

Everything is offline-testable
------------------------------
Every I/O seam is injected, so the whole module is exercised against
``tests/harness/comfyui_mock.FakeComfyUISession`` with no GPU, no weights and
no network: ``session`` (a ``requests.Session``-shaped object), ``stager``
(``staging.ComfyUIAssetStager``, reused rather than a second uploader),
``sleep``/``now`` clocks, and ``checker`` (``castcheck.check_consistency``).

Completion is witnessed by polling ``GET /history/{prompt_id}``, not over the
WebSocket the render path uses. That is a deliberate difference, not an
oversight: ``execution.py``'s socket exists because a render runs for hours
and a lost socket must not be mistaken for a lost render (issue #43). A 1024px
portrait is seconds, this tool is attended, and a bounded poll with an
injected ``sleep`` is both testable and incapable of mistaking a dead socket
for a failure -- it only ever believes ``/history``, which is what #43's
reconciliation concluded was the only trustworthy source anyway.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

import requests

from music_video_maker import castcheck, config, custody, execution, faces, hardware, staging
from music_video_maker import workflow_graph as wg
from music_video_maker.contracts import HardwareProfile, Workflow

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib


# --------------------------------------------------------------------------- #
# Licence and model provenance -- recorded beside the code that loads the
# weights, the way workflow_graph.KNOWN_LORAS and faces.py do.
# --------------------------------------------------------------------------- #

KREA2_COMMUNITY_LICENCE = (
    "Krea 2 Community License: commercial use free under $1M annual revenue and "
    "under 50 seats; a derivative model's name must begin with 'Krea'; no "
    "classifiers ship with the open weights, so content filtering is explicitly "
    "the deployer's responsibility"
)
"""The licence the generated imagery is produced under, in one string, so it
can be written verbatim into every manifest instead of summarized differently
each time. The full text travels with the weights; this is the part that
decides what a *user of this pipeline* may do with the output."""

FILTERING_DISCLOSURE = (
    "none -- no classifier ships with the Krea 2 open weights and none is shipped "
    "here, so filtering is the deployer's responsibility (Krea 2 Community "
    "License); a human must look at every image before it is used"
)
"""Recorded in every manifest. A provenance record that is silent about
filtering can be read as "something checked this"; this one cannot."""

KNOWN_IMAGE_MODELS: dict[str, dict[str, object]] = {
    "krea2-raw": {
        "source": "https://huggingface.co/krea/Krea-2-Raw",
        "weights": "raw.safetensors",
        "bytes": 26283332608,
        "licence": KREA2_COMMUNITY_LICENCE,
        "architecture": "krea2 SingleStreamDiT (comfy/ldm/krea2/model.py), bf16",
        "needs_text_encoder": "Qwen3-VL-4B, loaded with CLIPLoader type='krea2'",
        "needs_vae": "qwen_image_vae.safetensors (latent_format Wan21)",
        # Measured by reading the safetensors header on doris, 2026-10-04: the
        # file is the DiT ONLY -- 364 'blocks.*' tensors at 24.32 GB, plus
        # txtfusion/tproj/tmlp. No 'vae.' and no 'text_encoders.' keys, despite
        # comfy.supported_models.Krea2 declaring both prefixes, so this cannot
        # be loaded with CheckpointLoaderSimple: it is a UNETLoader file and
        # the text encoder and VAE are separate installs.
        "caveat": (
            "24.32 GB of bf16 DiT blocks alone, against a 24 GB card -- ComfyUI "
            "will offload rather than hold it resident, and no Krea 2 generation "
            "has been measured on doris yet"
        ),
    },
    "krea2-turbo": {
        "source": "https://huggingface.co/krea/Krea-2-Turbo",
        "weights": "turbo.safetensors",
        "bytes": 26283332608,
        "licence": KREA2_COMMUNITY_LICENCE,
        "architecture": "krea2 SingleStreamDiT, bf16 (few-step distillation of Raw)",
        "needs_text_encoder": "Qwen3-VL-4B, loaded with CLIPLoader type='krea2'",
        "needs_vae": "qwen_image_vae.safetensors (latent_format Wan21)",
        "caveat": (
            "ComfyUI's own blueprint samples Turbo at steps=8, cfg=1.0 -- those "
            "are Turbo's numbers and are wrong for Raw; set them per spec"
        ),
    },
}
"""Provenance for the image models #56's decision picked from. Descriptive
only: nothing here restricts which file a spec may name, because which weights
are installed on the ComfyUI host is not this repo's business (the same stance
``KNOWN_LORAS`` takes). A spec naming a model that is *not* in this table is
generated anyway, with ``licence = "UNRECORDED"`` in its manifest and a
WARNING -- loud, because "it downloaded fine" is not a licence."""

HOSTED_API_CLASS_TYPES = (
    "Krea2ImageNode",
    "Krea2StyleReferenceNode",
)
"""Billed, remote Krea nodes (``comfy_api_nodes/nodes_krea.py``), forbidden
here for exactly the reason ``workflow_graph.CLOUD_API_CLASS_TYPES`` forbids
the hosted MiniMax ones: this project's cost story is GPU hours and
electricity, and a template that quietly metered a hosted API would look
identical in every log. The local path is ``UNETLoader`` + ``CLIPLoader`` +
``KSampler``; see the committed templates."""


# --------------------------------------------------------------------------- #
# Templates and defaults
# --------------------------------------------------------------------------- #

ANCHOR_TEMPLATE_FILENAME = "workflow_cast_api.json"
VIEW_TEMPLATE_FILENAME = "workflow_cast_view_api.json"

DEFAULT_SAMPLER_NAME = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
"""Defaults taken from ComfyUI's own shipped Krea 2 blueprint rather than
chosen here -- a citable source for a value that decides pixels. ``steps`` and
``cfg`` get no default on purpose: the blueprint's (8 / 1.0) are *Turbo's*,
and silently applying them to Raw would be a generation nobody authored."""

WEIGHT_INPUTS: tuple[tuple[str, str], ...] = (
    ("UNETLoader", "unet_name"),
    ("CLIPLoader", "clip_name"),
    ("VAELoader", "vae_name"),
    ("LoraLoaderModelOnly", "lora_name"),
)
"""Which ``(class_type, input)`` pairs name a *server-side weights file*, and
so can be checked against what the server actually has installed before
custody is spent -- see :func:`preflight_installed_weights`."""

OBJECT_INFO_PATH = "/object_info"
PROMPT_PATH = "/prompt"
HISTORY_PATH = "/history"
VIEW_PATH = "/view"

DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_POLL_ATTEMPTS = 300
"""Ten minutes at the default interval. Generous for one 1024px portrait and
bounded, so an attended tool cannot sit forever against a wedged server."""

FILENAME_PREFIX_FORMAT = "mvm_cast_{character}_{view}"
"""``SaveImage.filename_prefix`` per view, so a human can tell outputs apart
in ComfyUI's own output directory and history -- the same reasoning as
``workflow_graph.CHUNK_FILENAME_PREFIX_FORMAT``."""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class CastGenError(RuntimeError):
    """Base class for every failure in this module."""


class CharacterSpecError(CastGenError):
    """The character spec file is missing, malformed, or describes a set that
    cannot hold identity (see :func:`load_character_spec`)."""


class HostedApiNodeError(CastGenError):
    """A billed, remote Krea API node is present in a template."""


class MissingWeightsError(CastGenError):
    """A weights file a template names is not installed on the ComfyUI host.
    Raised *before* submission, so the failure names the file and where it
    goes instead of arriving as a graph-validation blob after custody was
    taken."""


class GenerationTimeoutError(CastGenError):
    """``GET /history/{prompt_id}`` never reported a finished image within the
    poll budget."""


# --------------------------------------------------------------------------- #
# The character spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ViewSpec:
    """One image in a character's reference set.

    ``from_view`` is what makes the set a *person* rather than four people:
    every view but the first is generated from an earlier view's rendered
    file, at ``denoise`` below 1.0. The anchor (``from_view is None``) must be
    the first view and is the only text-to-image generation in the set.
    """

    name: str
    prompt: str
    """The full prompt actually submitted: the spec's base ``prompt`` plus
    this view's own clause. Recorded per image, because a reference set whose
    images were generated from different prompts and does not say so is the
    provenance bug #56 exists to close."""
    from_view: str | None = None
    denoise: float | None = None
    """Required when ``from_view`` is set, forbidden otherwise. No default --
    see the module docstring."""


@dataclass(frozen=True)
class CharacterSpec:
    """A whole invented character, as authored in one small TOML file."""

    character: str
    prompt: str
    seed: int
    steps: int
    cfg: float
    views: tuple[ViewSpec, ...]
    model: str | None = None
    """A :data:`KNOWN_IMAGE_MODELS` key, for the licence record."""
    sampler_name: str = DEFAULT_SAMPLER_NAME
    scheduler: str = DEFAULT_SCHEDULER
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    unet_name: str | None = None
    clip_name: str | None = None
    vae_name: str | None = None
    """Server-side weights filenames, overriding whatever the committed
    template carries. ``None`` leaves the template's own value, which is the
    only value this repo ever *states*; what is installed on a given host is
    the host's business."""
    source_path: Path | None = None

    @property
    def anchor(self) -> ViewSpec:
        return self.views[0]

    @property
    def licence_record(self) -> dict[str, object]:
        """The :data:`KNOWN_IMAGE_MODELS` entry for :attr:`model`, or an
        explicit "unrecorded" record -- never an empty dict, which would read
        in a manifest as "no licence question here"."""
        if self.model is not None and self.model in KNOWN_IMAGE_MODELS:
            return dict(KNOWN_IMAGE_MODELS[self.model])
        return {
            "source": "unrecorded",
            "licence": "UNRECORDED -- 'it downloaded fine' is not a licence",
            "model": self.model,
        }


_SPEC_TOP_LEVEL_KEYS = frozenset(
    {
        "character",
        "prompt",
        "seed",
        "model",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "width",
        "height",
        "unet_name",
        "clip_name",
        "vae_name",
        "views",
    }
)
_VIEW_KEYS = frozenset({"name", "prompt", "from", "denoise"})


def _spec_fail(field_name: str, message: str) -> NoReturn:
    logger.error("character spec invalid: %s: %s", field_name, message)
    raise CharacterSpecError(f"{field_name}: {message}")


def _require_str(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        _spec_fail(f"{where}{key}", f"must be a non-empty string, got {value!r}")
    return str(value).strip()


def _require_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _spec_fail(f"{where}{key}", f"must be an integer, got {value!r}")
    return int(value)  # type: ignore[arg-type]


def _require_number(raw: Mapping[str, Any], key: str, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _spec_fail(f"{where}{key}", f"must be a number, got {value!r}")
    return float(value)  # type: ignore[arg-type]


def load_character_spec(path: Path | str) -> CharacterSpec:
    """Read and validate a character spec TOML file.

    The validations that matter are the ones that refuse a set which cannot
    hold identity, all of them *before* any GPU time:

    * **At least two views.** ``castcheck`` reports a set of one as
      ``insufficient`` rather than a vacuous pass; refusing it here moves that
      discovery from after the generation to before it.
    * **Exactly one anchor, and it is first.** A second view with no ``from``
      is the re-roll the design doc forbids -- "that is four different people"
      -- and is refused by name.
    * **``from`` must name an earlier view**, so the set is a chain from one
      face and never a cycle or a forward reference.
    * **A derived view needs ``0 < denoise < 1``.** At 1.0 the anchor's latent
      is entirely replaced, which is a re-roll with extra steps.
    * Unknown keys are refused rather than ignored, so a typo
      (``denoize = 0.4``) cannot silently become a different generation.
    """
    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        logger.error("character spec not found: %s", path)
        raise CharacterSpecError(f"character spec not found: {path}") from exc
    except OSError as exc:
        logger.error("character spec unreadable: %s (%s)", path, exc)
        raise CharacterSpecError(f"character spec unreadable: {path} ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        logger.error("character spec %s is not valid TOML: %s", path, exc)
        raise CharacterSpecError(f"malformed TOML in character spec {path}: {exc}") from exc

    unknown = sorted(set(raw) - _SPEC_TOP_LEVEL_KEYS)
    if unknown:
        _spec_fail(
            "spec",
            f"unknown key(s): {', '.join(unknown)} (known: "
            f"{', '.join(sorted(_SPEC_TOP_LEVEL_KEYS))})",
        )

    character = _require_str(raw, "character", "")
    base_prompt = _require_str(raw, "prompt", "")
    seed = _require_int(raw, "seed", "")
    steps = _require_int(raw, "steps", "")
    cfg = _require_number(raw, "cfg", "")

    raw_views = raw.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        _spec_fail("views", "must be a non-empty array of [[views]] tables")
    if len(raw_views) < 2:
        _spec_fail(
            "views",
            "a reference set needs at least two views -- castcheck reports a set of "
            "one as 'insufficient' (no pair to score), so a one-view character cannot "
            "be shown to read as the same person at all (issue #56)",
        )

    views: list[ViewSpec] = []
    seen: list[str] = []
    for index, raw_view in enumerate(raw_views):
        where = f"views[{index}]."
        if not isinstance(raw_view, dict):
            _spec_fail(f"views[{index}]", f"must be a table, got {raw_view!r}")
        unknown_view = sorted(set(raw_view) - _VIEW_KEYS)
        if unknown_view:
            _spec_fail(
                f"views[{index}]",
                f"unknown key(s): {', '.join(unknown_view)} (known: "
                f"{', '.join(sorted(_VIEW_KEYS))})",
            )

        name = _require_str(raw_view, "name", where)
        if name in seen:
            _spec_fail(f"{where}name", f"duplicate view name {name!r}")

        clause = raw_view.get("prompt")
        if clause is not None and (not isinstance(clause, str) or not clause.strip()):
            _spec_fail(f"{where}prompt", f"must be a non-empty string when given, got {clause!r}")
        prompt = base_prompt if clause is None else f"{base_prompt}, {str(clause).strip()}"

        from_view = raw_view.get("from")
        if index == 0:
            if from_view is not None:
                _spec_fail(
                    f"{where}from",
                    "the first view is the anchor and is generated from the prompt alone, "
                    "so it cannot be derived from another view",
                )
            if raw_view.get("denoise") is not None:
                _spec_fail(
                    f"{where}denoise",
                    "the anchor is a text-to-image generation at denoise 1.0; a denoise "
                    "here would be ignored",
                )
            views.append(ViewSpec(name=name, prompt=prompt))
            seen.append(name)
            continue

        if from_view is None:
            _spec_fail(
                f"{where}from",
                "every view after the anchor must name an earlier view to generate from. "
                "A set built by re-rolling one prompt at different seeds is four "
                "different people, not four views of one (docs/design-synthetic-cast.md, "
                "'One reference image, or a set?')",
            )
        if not isinstance(from_view, str) or from_view not in seen:
            _spec_fail(
                f"{where}from",
                f"must name an earlier view (one of {seen}), got {from_view!r}",
            )

        denoise_raw = raw_view.get("denoise")
        if denoise_raw is None:
            _spec_fail(
                f"{where}denoise",
                "a derived view must state its own denoise -- there is no calibrated "
                "default for Krea 2, and a default that looks measured is worse than "
                "asking",
            )
        denoise = _require_number(raw_view, "denoise", where)
        if not 0.0 < denoise < 1.0:
            _spec_fail(
                f"{where}denoise",
                f"must be between 0 and 1 (exclusive), got {denoise} -- at 1.0 the "
                "anchor's latent is replaced entirely, which is a re-roll of the prompt, "
                "and at 0 nothing is generated at all",
            )
        views.append(
            ViewSpec(name=name, prompt=prompt, from_view=str(from_view), denoise=denoise)
        )
        seen.append(name)

    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        _spec_fail("model", f"must be a non-empty string when given, got {model!r}")
    if isinstance(model, str) and model.strip() not in KNOWN_IMAGE_MODELS:
        logger.warning(
            "character spec names model %r, which is not in KNOWN_IMAGE_MODELS (%s) -- "
            "its licence will be recorded as UNRECORDED; add it to the table in "
            "castgen.py with source, licence and size before publishing anything made "
            "with it",
            model,
            sorted(KNOWN_IMAGE_MODELS),
        )

    spec = CharacterSpec(
        character=character,
        prompt=base_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        views=tuple(views),
        model=model.strip() if isinstance(model, str) else None,
        sampler_name=str(raw.get("sampler_name", DEFAULT_SAMPLER_NAME)),
        scheduler=str(raw.get("scheduler", DEFAULT_SCHEDULER)),
        width=int(raw.get("width", DEFAULT_WIDTH)),
        height=int(raw.get("height", DEFAULT_HEIGHT)),
        unet_name=raw.get("unet_name"),
        clip_name=raw.get("clip_name"),
        vae_name=raw.get("vae_name"),
        source_path=path,
    )
    logger.info(
        "loaded character spec %s: %s, %d views (anchor=%s), seed=%d",
        path,
        spec.character,
        len(spec.views),
        spec.anchor.name,
        spec.seed,
    )
    return spec


# --------------------------------------------------------------------------- #
# Planning: spec + templates -> submittable workflows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ViewPlan:
    """One view's submittable workflow, plus everything about it that will be
    recorded as provenance whether or not the generation succeeds."""

    view: ViewSpec
    seed: int
    filename_prefix: str
    workflow: Workflow
    derived: bool

    @property
    def name(self) -> str:
        return self.view.name


def slugify(text: str) -> str:
    """Filesystem- and ComfyUI-safe token (``[A-Za-z0-9_-]``)."""
    cleaned = "".join(char if (char.isalnum() or char in "-_") else "_" for char in text)
    return cleaned.strip("_") or "unnamed"


def assert_no_hosted_api_nodes(workflow: Workflow) -> None:
    """Refuse a template carrying a billed, remote image-API node.

    Delegates the MiniMax half to ``workflow_graph.assert_no_cloud_api_nodes``
    rather than restating its table, then applies the same rule to Krea's own
    hosted nodes (:data:`HOSTED_API_CLASS_TYPES`). Both exist in the ComfyUI
    install this runs against, and a template that misgrabbed one would render
    a plausible portrait off a metered endpoint with nothing in any log to say
    so.
    """
    wg.assert_no_cloud_api_nodes(workflow)
    found = [
        (node_id, node.get("class_type"))
        for node_id, node in workflow.items()
        if node.get("class_type") in HOSTED_API_CLASS_TYPES
    ]
    if found:
        logger.error("hosted Krea API node(s) present in template -- billed, remote: %s", found)
        raise HostedApiNodeError(
            f"hosted Krea API node(s) present in template: {found}; this module only "
            "drives the local weights (UNETLoader + CLIPLoader type='krea2')"
        )


def _set_input(workflow: Workflow, class_type: str, input_name: str, value: object) -> None:
    """Write one input on the single node of ``class_type``, located by
    ``class_type`` and never by node id (``workflow_graph``'s hard invariant:
    a canvas edit renumbers ids freely)."""
    _node_id, node = wg.find_one_node(workflow, class_type)
    node["inputs"][input_name] = value


def _apply_weight_overrides(workflow: Workflow, spec: CharacterSpec) -> None:
    for class_type, input_name, value in (
        ("UNETLoader", "unet_name", spec.unet_name),
        ("CLIPLoader", "clip_name", spec.clip_name),
        ("VAELoader", "vae_name", spec.vae_name),
    ):
        if value is None:
            continue
        _set_input(workflow, class_type, input_name, value)


def plan_views(
    spec: CharacterSpec,
    *,
    anchor_template: Workflow,
    view_template: Workflow,
) -> tuple[ViewPlan, ...]:
    """Turn a spec plus the two committed templates into one submittable
    workflow per view. Pure: no network, no disk, no GPU.

    Both templates are deep-copied per view -- a template is loaded once and
    reused across a whole character, exactly like the render path's, and must
    never be mutated in place.

    Each template is also checked for the *shape* its role requires: the
    anchor template must have an ``EmptyLatentImage`` and no ``LoadImage``, and
    the view template must have both a ``LoadImage`` and a ``VAEEncode``.
    Swapping the two arguments is then a loud refusal rather than a set of
    four portraits that silently ignored the anchor -- the same class of
    mistake as the I2V seed frame overwriting the cast photo
    (``docs/workflow-template-guide.md``).
    """
    assert_no_hosted_api_nodes(anchor_template)
    # LoadImage first, because the swapped-templates case is the one with a
    # useful message: the img2img template has no EmptyLatentImage either, and
    # "no node with class_type 'EmptyLatentImage'" would name the symptom
    # while this names the mistake.
    if wg.find_nodes_by_class_type(anchor_template, "LoadImage"):
        raise CharacterSpecError(
            "the anchor template carries a LoadImage node -- it is the text-to-image "
            f"template ({ANCHOR_TEMPLATE_FILENAME}); the img2img one "
            f"({VIEW_TEMPLATE_FILENAME}) was probably passed in its place"
        )
    wg.find_one_node(anchor_template, "EmptyLatentImage")

    plans: list[ViewPlan] = []
    for index, view in enumerate(spec.views):
        seed = spec.seed + index
        prefix = FILENAME_PREFIX_FORMAT.format(
            character=slugify(spec.character), view=slugify(view.name)
        )
        if view.from_view is None:
            workflow = copy.deepcopy(anchor_template)
            _set_input(workflow, "EmptyLatentImage", "width", spec.width)
            _set_input(workflow, "EmptyLatentImage", "height", spec.height)
            _set_input(workflow, "EmptyLatentImage", "batch_size", 1)
            denoise = 1.0
        else:
            assert_no_hosted_api_nodes(view_template)
            wg.find_one_node(view_template, "LoadImage")
            wg.find_one_node(view_template, "VAEEncode")
            workflow = copy.deepcopy(view_template)
            denoise = float(view.denoise or 0.0)

        _set_input(workflow, "CLIPTextEncode", "text", view.prompt)
        _set_input(workflow, "KSampler", "seed", seed)
        _set_input(workflow, "KSampler", "steps", spec.steps)
        _set_input(workflow, "KSampler", "cfg", spec.cfg)
        _set_input(workflow, "KSampler", "sampler_name", spec.sampler_name)
        _set_input(workflow, "KSampler", "scheduler", spec.scheduler)
        _set_input(workflow, "KSampler", "denoise", denoise)
        _set_input(workflow, "SaveImage", "filename_prefix", prefix)
        _apply_weight_overrides(workflow, spec)

        plans.append(
            ViewPlan(
                view=view,
                seed=seed,
                filename_prefix=prefix,
                workflow=workflow,
                derived=view.from_view is not None,
            )
        )
    return tuple(plans)


# --------------------------------------------------------------------------- #
# Pre-flight: are the weights this template names actually installed?
# --------------------------------------------------------------------------- #


def read_installed_weights(
    session: Any, base_url: str, class_type: str, input_name: str
) -> tuple[str, ...] | None:
    """What files ``class_type``'s ``input_name`` enum offers on this server.

    ``GET /object_info/{class_type}`` is metadata: it loads no weights and
    touches no GPU. Returns ``None`` -- logged -- whenever the answer cannot
    be trusted (unreachable, non-200, unfamiliar body, or the 200-with-``{}``
    real ComfyUI returns for a class it does not have). ``None`` means "cannot
    check", never "nothing installed"; callers degrade rather than refuse, the
    same decision ``custody._fetch_free_vram_gb`` makes for an unreadable VRAM
    reading.
    """
    url = f"{base_url.rstrip('/')}{OBJECT_INFO_PATH}/{class_type}"
    try:
        response = session.get(url, timeout=execution.DEFAULT_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        logger.warning("weights pre-flight could not reach %s (%s)", url, exc)
        return None
    if getattr(response, "status_code", None) != 200:
        logger.warning(
            "weights pre-flight: %s returned status=%s", url, getattr(response, "status_code", "?")
        )
        return None
    try:
        payload = response.json()
    except ValueError:
        logger.warning("weights pre-flight: %s returned a non-JSON body", url)
        return None

    node = payload.get(class_type) if isinstance(payload, dict) else None
    if not isinstance(node, dict):
        logger.warning("weights pre-flight: %s does not describe %s", url, class_type)
        return None
    spec = node.get("input", {})
    for section in ("required", "optional"):
        entry = spec.get(section, {}) if isinstance(spec, dict) else {}
        if isinstance(entry, dict) and input_name in entry:
            candidates = entry[input_name]
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
                return tuple(str(value) for value in candidates[0])
    logger.warning("weights pre-flight: %s has no %r enum to read", class_type, input_name)
    return None


def preflight_installed_weights(
    session: Any, base_url: str, workflows: Sequence[Workflow]
) -> dict[str, str]:
    """Refuse before submission if a template names a weights file the host
    does not have.

    ComfyUI would reject the prompt anyway -- as a ``node_errors`` blob, after
    the operator has already taken custody of the card and stopped whatever
    else was using it. This turns that into a message naming the missing file
    and the directory it belongs in, which matters most for exactly the state
    doris is in: the Krea 2 DiT lives in the HuggingFace cache and is not
    visible to ``UNETLoader`` until it is linked into
    ``models/diffusion_models/``, and the Krea 2 text encoder
    (``CLIPLoader type='krea2'``, Qwen3-VL-4B) is a separate download.

    Returns the ``{class_type.input: filename}`` map it checked, for the
    manifest.
    """
    checked: dict[str, str] = {}
    missing: list[str] = []
    for class_type, input_name in WEIGHT_INPUTS:
        wanted = {
            str(node["inputs"][input_name])
            for workflow in workflows
            for node in workflow.values()
            if node.get("class_type") == class_type and input_name in node.get("inputs", {})
        }
        if not wanted:
            continue
        installed = read_installed_weights(session, base_url, class_type, input_name)
        for filename in sorted(wanted):
            checked[f"{class_type}.{input_name}"] = filename
            if installed is None:
                continue
            if filename not in installed:
                missing.append(
                    f"{class_type}.{input_name}={filename!r} (installed: "
                    f"{sorted(installed) if installed else 'none'})"
                )

    if missing:
        logger.error("weights pre-flight FAILED: %s", "; ".join(missing))
        raise MissingWeightsError(
            "the ComfyUI host does not have every weights file these templates name: "
            + "; ".join(missing)
            + ". Put the file in the matching models/ directory (a symlink is enough) "
            "and restart or refresh ComfyUI so its loader enum picks it up."
        )
    logger.info("weights pre-flight OK: %s", checked or "nothing to check")
    return checked


# --------------------------------------------------------------------------- #
# Results and provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratedImage:
    """One generated reference image and everything needed to make it again."""

    view: str
    path: Path
    sha256: str
    bytes_written: int
    seed: int
    denoise: float
    prompt: str
    from_view: str | None
    prompt_id: str
    server_filename: str
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        return {
            "view": self.view,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes_written,
            "seed": self.seed,
            "denoise": self.denoise,
            "prompt": self.prompt,
            "from_view": self.from_view,
            "prompt_id": self.prompt_id,
            "server_filename": self.server_filename,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class CharacterResult:
    """A generated character: its images, its provenance, and what the
    consistency check made of it."""

    spec: CharacterSpec
    images: tuple[GeneratedImage, ...]
    consistency: castcheck.ConsistencyReport | None
    consistency_detail: str | None
    free_vram_gb: float | None
    render_stack: custody.RenderStack
    weights_checked: dict[str, str]
    generated_at: str

    @property
    def status(self) -> str:
        """``"pass"``/``"fail"``/``"needs_review"``/``"insufficient"`` from
        ``castcheck``, or ``"unchecked"`` when the check could not run at all
        (no SFace weights, no OpenCV). ``"unchecked"`` is never a pass."""
        if self.consistency is None:
            return "unchecked"
        return self.consistency.status

    @property
    def similarity_spread(self) -> tuple[float, float, float] | None:
        """``(min, mean, max)`` over every scored pair, or ``None``.

        Numbers with no verdict attached, on purpose. ``castcheck``'s own
        docstring says what it cannot tell you: generated faces are often
        *more* similar to each other than two photographs of one person are,
        so a set of views authored to differ in angle and light that scores
        near-identical everywhere is the mode-collapse signature -- and no
        threshold here could separate it from success. Printing the spread
        puts the thing a human should look at in front of them without
        inventing a measurement.
        """
        if self.consistency is None or not self.consistency.pairs:
            return None
        scores = [pair.similarity for pair in self.consistency.pairs]
        return (min(scores), sum(scores) / len(scores), max(scores))

    def origin_toml(self) -> str:
        """The ``[cast.<name>]`` block to paste into a run config.

        Emits exactly what ``config.py`` already requires of a synthetic cast
        member (issue #56 part 1): ``synthetic = true`` plus an ``origin``
        table with ``model``, ``prompt``, ``seed`` and ``created``, and the
        rest of this generation's settings as the free-form scalar extras that
        table accepts. The anchor is the ``image``, because it is the identity
        anchor every other view was generated from; the whole set goes into
        ``ref_images`` by the usual config means, which is not this block's
        business.
        """
        spec = self.spec
        anchor = self.images[0] if self.images else None
        lines = [f"[cast.{spec.character}]"]
        lines.append('role = "<describe what they are doing, not how they look>"')
        if anchor is not None:
            lines.append(f'image = "cast/{anchor.path.name}"')
        lines.append("synthetic = true")
        lines.append("")
        lines.append(f"[cast.{spec.character}.origin]")
        model_name = spec.model or (spec.unet_name or "krea2")
        lines.append(f'model = "{model_name}"')
        lines.append(f'prompt = "{_toml_escape(spec.prompt)}"')
        lines.append(f"seed = {spec.seed}")
        lines.append(f"created = {self.generated_at[:10]}")
        lines.append(f'sampler = "{spec.sampler_name}"')
        lines.append(f'scheduler = "{spec.scheduler}"')
        lines.append(f"steps = {spec.steps}")
        lines.append(f"cfg = {spec.cfg}")
        lines.append(f"width = {spec.width}")
        lines.append(f"height = {spec.height}")
        lines.append(f'views = "{",".join(image.view for image in self.images)}"')
        lines.append('method = "anchor + img2img views (castgen.py, issue #56)"')
        lines.append(f'licence = "{_toml_escape(str(spec.licence_record.get("licence")))}"')
        lines.append(f'filtering = "{_toml_escape(FILTERING_DISCLOSURE)}"')
        lines.append(f'consistency = "{self.status}"')
        return "\n".join(lines) + "\n"

    def manifest(self) -> dict[str, object]:
        """The full machine-readable record, written beside the images."""
        spread = self.similarity_spread
        return {
            "tool": "music_video_maker.castgen (issue #56)",
            "character": self.spec.character,
            "generated_at": self.generated_at,
            "spec_path": str(self.spec.source_path) if self.spec.source_path else None,
            "model": self.spec.model,
            "model_record": self.spec.licence_record,
            "licence": self.spec.licence_record.get("licence"),
            "filtering": FILTERING_DISCLOSURE,
            "method": "anchor (text-to-image) + img2img views at denoise < 1",
            "sampler": {
                "sampler_name": self.spec.sampler_name,
                "scheduler": self.spec.scheduler,
                "steps": self.spec.steps,
                "cfg": self.spec.cfg,
                "width": self.spec.width,
                "height": self.spec.height,
                "base_seed": self.spec.seed,
            },
            "weights_checked": dict(self.weights_checked),
            "free_vram_gb_at_start": self.free_vram_gb,
            "render_stack": {
                "comfyui_version": self.render_stack.comfyui_version,
                "torch_version": self.render_stack.torch_version,
            },
            "images": [image.as_dict() for image in self.images],
            "consistency": {
                "status": self.status,
                "detail": self.consistency_detail,
                "floor": self.consistency.floor if self.consistency else None,
                "floor_calibration": (
                    "real photographs of real people, 16 pairs "
                    "(docs/seed-face-recognition.md) -- NOT re-derived on synthetic "
                    "pairs; whether it transfers to generated faces is unknown"
                ),
                "pairs_scored": len(self.consistency.pairs) if self.consistency else 0,
                "similarity_min": spread[0] if spread else None,
                "similarity_mean": spread[1] if spread else None,
                "similarity_max": spread[2] if spread else None,
                "mode_collapse": (
                    "castcheck cannot detect it: generated faces can sit closer to each "
                    "other than two photographs of one person do. Near-identical scores "
                    "across views authored to differ are a reason to look, not a pass"
                ),
            },
        }

    def render(self) -> str:
        """A human-readable report, carrying every caveat that must travel
        with these numbers rather than leaving them to be remembered."""
        lines = [
            f"# music_video_maker.castgen report (issue #56): {self.spec.character}",
            f"# generated_at={self.generated_at}",
            f"# model={self.spec.model} licence={self.spec.licence_record.get('licence')}",
            f"# filtering={FILTERING_DISCLOSURE}",
            "#",
        ]
        lines.append(
            "# CAVEAT: the 0.34 SFace floor below was calibrated on photographs of REAL "
            "people (docs/seed-face-recognition.md). Whether it transfers to generated "
            "faces is unknown and has not been re-derived."
        )
        lines.append(
            "# CAVEAT: castcheck cannot see mode collapse. Four near-identical scores "
            "across views authored to differ in angle and light may be one face four "
            "times. Look at the images."
        )
        lines.append("")
        lines.append(f"images: {len(self.images)}")
        for image in self.images:
            origin = f"from={image.from_view} denoise={image.denoise}" if image.from_view else (
                "anchor (text-to-image)"
            )
            lines.append(
                f"  {image.view}: {image.path} seed={image.seed} {origin} "
                f"sha256={image.sha256[:12]}... ({image.bytes_written}B)"
            )
        spread = self.similarity_spread
        lines.append("")
        if spread is not None:
            lines.append(
                f"pairwise similarity: min={spread[0]:.4f} mean={spread[1]:.4f} "
                f"max={spread[2]:.4f}"
            )
        else:
            lines.append("pairwise similarity: not scored")
        if self.consistency_detail:
            lines.append(f"consistency check: {self.consistency_detail}")
        lines.append(f"RESULT: {self.status}")
        return "\n".join(lines)


def _toml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

Checker = Callable[[str, Sequence[Path]], castcheck.ConsistencyReport]
"""Injected consistency check -- ``castcheck.check_consistency`` with real
detectors in production, a fake in tests. This module never scores a set
itself."""

Sleep = Callable[[float], None]


def _submit(session: Any, base_url: str, workflow: Workflow, client_id: str) -> str:
    """``POST /prompt``, returning the ``prompt_id``.

    Reuses ``execution.PromptSubmissionError`` rather than defining a parallel
    error for the same failure: a rejected graph is a rejected graph, and two
    exception types for it would make a caller choose."""
    url = f"{base_url.rstrip('/')}{PROMPT_PATH}"
    payload = {"prompt": workflow, "client_id": client_id}
    try:
        response = session.post(
            url, json=payload, timeout=execution.DEFAULT_HTTP_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        logger.error("castgen: POST %s failed: %s", url, exc)
        raise execution.PromptSubmissionError(f"POST {url} failed: {exc}") from exc

    if response.status_code != 200:
        logger.error("castgen: POST %s returned status=%s", url, response.status_code)
        raise execution.PromptSubmissionError(
            f"POST {url} returned status {response.status_code}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise execution.PromptSubmissionError(f"POST {url} returned a non-JSON body") from exc

    prompt_id = body.get("prompt_id")
    if not prompt_id:
        node_errors = body.get("node_errors") or {}
        logger.error("castgen: ComfyUI rejected the workflow: %s", node_errors)
        raise execution.PromptSubmissionError(
            f"ComfyUI rejected the workflow: {node_errors}", node_errors=node_errors
        )
    return str(prompt_id)


def _await_image(
    session: Any,
    base_url: str,
    prompt_id: str,
    *,
    sleep: Sleep,
    poll_interval: float,
    attempts: int,
) -> dict[str, Any]:
    """Poll ``GET /history/{prompt_id}`` until an image output appears.

    An entry that completes with no image output is a failure, not a wait:
    that is a graph whose ``SaveImage`` never ran, and no amount of further
    polling changes it."""
    url = f"{base_url.rstrip('/')}{HISTORY_PATH}/{prompt_id}"
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=execution.DEFAULT_HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            logger.error("castgen: GET %s failed: %s", url, exc)
            raise execution.HistoryError(f"GET {url} failed: {exc}") from exc
        if response.status_code != 200:
            raise execution.HistoryError(f"GET {url} returned status {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise execution.HistoryError(f"GET {url} returned a non-JSON body") from exc

        entry = body.get(prompt_id) if isinstance(body, dict) else None
        if isinstance(entry, dict):
            image = _find_image_output(entry.get("outputs") or {})
            if image is not None:
                return image
            status = entry.get("status") or {}
            if status.get("completed") or status.get("status_str") in ("success", "error"):
                logger.error(
                    "castgen: prompt %s finished with no image output (status=%s)",
                    prompt_id,
                    status,
                )
                raise execution.OutputRetrievalError(
                    f"prompt {prompt_id} finished with no saved image (status={status}) -- "
                    "does the template's SaveImage node actually run? A PreviewImage "
                    "writes a temp file ComfyUI may delete and does not count"
                )
        if attempt + 1 < attempts:
            sleep(poll_interval)

    raise GenerationTimeoutError(
        f"prompt {prompt_id} produced no image within "
        f"{attempts * poll_interval:.0f}s ({attempts} polls) -- the card may still be "
        "working; check ComfyUI's own queue before resubmitting"
    )


def _find_image_output(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
    """The first *saved* image in a history entry's outputs.

    ``type`` matters: ``SaveImage`` writes ``"output"``, while
    ``PreviewImage`` writes ``"temp"`` -- a file ComfyUI is free to delete,
    and which a tool whose whole job is to keep the picture must not accept
    as the deliverable. A graph with only a preview node therefore reads as
    "finished with no image", which is what it is.
    """
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if isinstance(images, list):
            for image in images:
                if (
                    isinstance(image, dict)
                    and image.get("filename")
                    and str(image.get("type", "output")) == "output"
                ):
                    return dict(image)
    return None


def _fetch_image_bytes(session: Any, base_url: str, image: Mapping[str, Any]) -> bytes:
    url = f"{base_url.rstrip('/')}{VIEW_PATH}"
    params = {
        "filename": image.get("filename", ""),
        "subfolder": image.get("subfolder", ""),
        "type": image.get("type", "output"),
    }
    try:
        response = session.get(url, params=params, timeout=execution.DEFAULT_HTTP_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise execution.OutputRetrievalError(f"GET {url} failed: {exc}") from exc
    if response.status_code != 200:
        raise execution.OutputRetrievalError(
            f"GET {url} for {params['filename']!r} returned status {response.status_code}"
        )
    content = response.content
    if not content:
        raise execution.OutputRetrievalError(
            f"GET {url} for {params['filename']!r} returned an empty body"
        )
    return content


def _default_checker(character: str, images: Sequence[Path]) -> castcheck.ConsistencyReport:
    return castcheck.check_consistency(
        character,
        list(images),
        detector=castcheck.build_default_detector(),
        recognizer=castcheck.build_default_recognizer(),
    )


def generate_character(
    spec: CharacterSpec,
    *,
    output_dir: Path | str,
    anchor_template: Workflow,
    view_template: Workflow,
    session: Any,
    base_url: str = config.DEFAULT_COMFYUI_URL,
    stager: staging.ComfyUIAssetStager | None = None,
    checker: Checker | None = _default_checker,
    sleep: Sleep = lambda _seconds: None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_attempts: int = DEFAULT_POLL_ATTEMPTS,
    min_free_vram_gb: float = custody.DEFAULT_MIN_FREE_VRAM_GB,
    hardware_profile: HardwareProfile = hardware.PROFILE_RTX_4090_24GB,
    client_id: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CharacterResult:
    """Generate every view of ``spec`` and return the whole record.

    Order of operations, and why:

    1. **Plan everything first** (:func:`plan_views`), so a spec that cannot
       hold identity, or a template mix-up, fails before the card is touched.
    2. **Weights pre-flight** (:func:`preflight_installed_weights`) -- a
       missing file named precisely, rather than a validation blob after
       custody is taken.
    3. **Free-VRAM pre-flight**, reusing ``custody.preflight_free_vram`` so
       there is one copy of that comparison, one refusal message and one
       "unreadable is degraded, not fatal" decision in this project. Note what
       the floor means here: it is the *is the card actually free* check, from
       H3's measurements. **No Krea 2 generation has been measured on doris**,
       so there is no Krea-2-specific floor to apply and inventing one would
       be a number nobody measured (``envelope.py``: no evidence means no
       gate, not a refusal).
    4. **Generate view by view, in spec order**, uploading each derived view's
       source image (the *rendered file* of an earlier view) through the
       existing ``staging.ComfyUIAssetStager``.
    5. **Release the card unconditionally** with ``POST /free`` in a
       ``finally``, exactly as a render does -- a half-finished character must
       not leave 24 GB of DiT resident on a shared 4090.
    6. **Score the set with ``castcheck``**, never with a second scorer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")
    plans = plan_views(spec, anchor_template=anchor_template, view_template=view_template)
    stager = stager or staging.ComfyUIAssetStager(base_url=base_url, session=session)
    client_id = client_id or str(uuid.uuid4())

    weights_checked = preflight_installed_weights(
        session, base_url, [plan.workflow for plan in plans]
    )
    free_vram_gb = custody.preflight_free_vram(
        session,
        base_url,
        min_free_vram_gb=min_free_vram_gb,
        hardware=hardware_profile,
    )
    render_stack = custody.read_render_stack(session, base_url)

    logger.warning(
        "castgen: generating %d views of %r with %s. No classifier ships with these "
        "weights and none runs here -- filtering is the deployer's responsibility "
        "(Krea 2 Community License); look at every image before using it.",
        len(plans),
        spec.character,
        spec.model or spec.unet_name or "the template's declared weights",
    )

    images: list[GeneratedImage] = []
    by_view: dict[str, Path] = {}
    try:
        for plan in plans:
            workflow = plan.workflow
            if plan.view.from_view is not None:
                source = by_view.get(plan.view.from_view)
                if source is None:  # pragma: no cover - plan_views orders the chain
                    raise CastGenError(
                        f"view {plan.name!r} is derived from {plan.view.from_view!r}, "
                        "which has not been generated"
                    )
                server_name = stager.upload_image(source)
                _set_input(workflow, "LoadImage", "image", server_name)
                logger.info(
                    "castgen: view %s is img2img from %s (uploaded as %s) at denoise %s",
                    plan.name,
                    plan.view.from_view,
                    server_name,
                    plan.view.denoise,
                )

            prompt_id = _submit(session, base_url, workflow, client_id)
            logger.info("castgen: submitted view %s as prompt %s", plan.name, prompt_id)
            output = _await_image(
                session,
                base_url,
                prompt_id,
                sleep=sleep,
                poll_interval=poll_interval,
                attempts=poll_attempts,
            )
            content = _fetch_image_bytes(session, base_url, output)
            destination = output_dir / (
                f"{slugify(spec.character)}_{slugify(plan.name)}"
                f"{Path(str(output.get('filename'))).suffix or '.png'}"
            )
            destination.write_bytes(content)
            by_view[plan.name] = destination
            images.append(
                GeneratedImage(
                    view=plan.name,
                    path=destination,
                    sha256=hashlib.sha256(content).hexdigest(),
                    bytes_written=len(content),
                    seed=plan.seed,
                    denoise=1.0 if plan.view.denoise is None else plan.view.denoise,
                    prompt=plan.view.prompt,
                    from_view=plan.view.from_view,
                    prompt_id=prompt_id,
                    server_filename=str(output.get("filename")),
                    width=spec.width,
                    height=spec.height,
                )
            )
            logger.info("castgen: wrote %s (%d bytes)", destination, len(content))
    finally:
        # Custody release, unconditional -- a failure partway through a
        # character must not leave the model resident on a shared card.
        custody.build_vram_releaser(session, base_url)()

    report: castcheck.ConsistencyReport | None = None
    detail: str | None = None
    if checker is None:
        detail = "skipped (no checker supplied)"
    else:
        try:
            report = checker(spec.character, [image.path for image in images])
        except faces.FaceDetectionError as exc:
            detail = f"could not run: {exc}"
            logger.warning("castgen: consistency check could not run: %s", exc)
    if report is not None and report.status != "pass":
        logger.warning(
            "castgen: %r is NOT ready to condition a render on (castcheck status=%s)",
            spec.character,
            report.status,
        )

    return CharacterResult(
        spec=spec,
        images=tuple(images),
        consistency=report,
        consistency_detail=detail,
        free_vram_gb=free_vram_gb,
        render_stack=render_stack,
        weights_checked=weights_checked,
        generated_at=now().isoformat(),
    )


def write_character_record(result: CharacterResult, output_dir: Path | str) -> dict[str, Path]:
    """Write the three files that make a generated character re-creatable:
    the report, the JSON manifest, and the ``[cast.<name>]`` block.

    Separate from :func:`generate_character` so a caller can inspect a result
    before anything lands on disk, and so the files are written in one place
    rather than three."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(result.spec.character)
    paths = {
        "report": output_dir / f"{stem}_castgen_report.txt",
        "manifest": output_dir / f"{stem}_castgen.json",
        "origin": output_dir / f"{stem}_origin.toml",
    }
    paths["report"].write_text(result.render() + "\n", encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(result.manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["origin"].write_text(result.origin_toml(), encoding="utf-8")
    logger.info("castgen: wrote %s", ", ".join(str(path) for path in paths.values()))
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.castgen",
        description=(
            "Generate an invented cast member's reference image set with Krea 2 "
            "(issue #56). Authoring-time, outside the render path: a character is "
            "generated once by hand and its images are committed as run assets."
        ),
    )
    parser.add_argument(
        "command",
        choices=("plan", "generate"),
        help=(
            "'plan' validates the spec and builds every view's workflow offline "
            "(no server, no GPU); 'generate' submits them to ComfyUI and downloads "
            "the set"
        ),
    )
    parser.add_argument("spec", type=Path, help="path to the character spec TOML")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=(
            "where the images and provenance go -- a run's own cast directory "
            "(~/mvm-runs/<song>/cast), never this repository"
        ),
    )
    parser.add_argument(
        "--comfyui-url",
        default=config.DEFAULT_COMFYUI_URL,
        help="ComfyUI base URL (generate only)",
    )
    parser.add_argument(
        "--anchor-template",
        type=Path,
        default=Path(ANCHOR_TEMPLATE_FILENAME),
        help="text-to-image template for the anchor view",
    )
    parser.add_argument(
        "--view-template",
        type=Path,
        default=Path(VIEW_TEMPLATE_FILENAME),
        help="img2img template for every derived view",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="seconds between /history polls (generate only)",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=custody.DEFAULT_MIN_FREE_VRAM_GB,
        help=(
            "free-VRAM floor for the pre-flight. The default is H3's measured "
            "custody floor reused as an 'is the card free' check -- no Krea 2 "
            "generation has been measured on this host"
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help=(
            "do not run castcheck after generating (it needs the SFace weights); "
            "the set's status is then recorded as 'unchecked', which is not a pass"
        ),
    )
    return parser


def main(argv: list[str] | None = None, *, session: Any = None) -> int:
    """Entry point for ``python -m music_video_maker.castgen``.

    ``session`` is injected by tests; production builds a real
    ``requests.Session``. Exit code 0 only when the whole set generated **and**
    ``castcheck`` passed -- every other outcome means this character is not
    ready to condition a render on, which is the honest failure mode
    ``docs/design-synthetic-cast.md`` asks for.
    """
    args = build_parser().parse_args(argv)
    try:
        spec = load_character_spec(args.spec)
        anchor_template = wg.load_workflow_template(args.anchor_template)
        view_template = wg.load_workflow_template(args.view_template)
    except (CharacterSpecError, wg.WorkflowTemplateError) as exc:
        sys.stdout.write(f"castgen: {exc}\n")
        return 2

    if args.command == "plan":
        try:
            plans = plan_views(
                spec, anchor_template=anchor_template, view_template=view_template
            )
        except (CastGenError, wg.WorkflowGraphError) as exc:
            sys.stdout.write(f"castgen: {exc}\n")
            return 2
        args.out_dir.mkdir(parents=True, exist_ok=True)
        sys.stdout.write(f"character: {spec.character} ({len(plans)} views)\n")
        for plan in plans:
            destination = args.out_dir / f"{plan.filename_prefix}_workflow.json"
            destination.write_text(
                json.dumps(plan.workflow, indent=2) + "\n", encoding="utf-8"
            )
            origin = (
                f"img2img from {plan.view.from_view} at denoise {plan.view.denoise}"
                if plan.derived
                else "anchor (text-to-image)"
            )
            sys.stdout.write(
                f"  {plan.name}: seed={plan.seed} {origin}\n"
                f"    prompt: {plan.view.prompt}\n"
                f"    workflow: {destination}\n"
            )
        sys.stdout.write(
            "\nNothing was generated: 'plan' is offline. Before 'generate', take GPU "
            "custody by hand (stop whatever else holds the card), and remember that no "
            "classifier runs here -- filtering is the deployer's responsibility under "
            "the Krea 2 Community License.\n"
        )
        return 0

    http = session if session is not None else requests.Session()
    try:
        result = generate_character(
            spec,
            output_dir=args.out_dir,
            anchor_template=anchor_template,
            view_template=view_template,
            session=http,
            base_url=args.comfyui_url,
            checker=None if args.skip_check else _default_checker,
            sleep=_real_sleep,
            poll_interval=args.poll_interval,
            min_free_vram_gb=args.min_free_vram_gb,
        )
    except (CastGenError, execution.ExecutionError, custody.CustodyError) as exc:
        sys.stdout.write(f"castgen: {exc}\n")
        return 1

    written = write_character_record(result, args.out_dir)
    sys.stdout.write(result.render() + "\n")
    sys.stdout.write(
        "\nwrote:\n" + "".join(f"  {path}\n" for path in written.values())
    )
    sys.stdout.write(
        f"\nPaste {written['origin'].name} into your run config, and re-score the set "
        f"any time with:\n  python -m music_video_maker.castcheck {spec.character} "
        + " ".join(str(image.path) for image in result.images)
        + "\n"
    )
    return 0 if result.status == "pass" else 1


def _real_sleep(seconds: float) -> None:  # pragma: no cover - trivial, and sleeps
    time.sleep(seconds)


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())


__all__ = [
    "ANCHOR_TEMPLATE_FILENAME",
    "FILTERING_DISCLOSURE",
    "HOSTED_API_CLASS_TYPES",
    "KNOWN_IMAGE_MODELS",
    "KREA2_COMMUNITY_LICENCE",
    "VIEW_TEMPLATE_FILENAME",
    "CastGenError",
    "CharacterResult",
    "CharacterSpec",
    "CharacterSpecError",
    "GeneratedImage",
    "GenerationTimeoutError",
    "HostedApiNodeError",
    "MissingWeightsError",
    "ViewPlan",
    "ViewSpec",
    "assert_no_hosted_api_nodes",
    "build_parser",
    "generate_character",
    "load_character_spec",
    "main",
    "plan_views",
    "preflight_installed_weights",
    "read_installed_weights",
    "slugify",
    "write_character_record",
]
