"""Shared test-data factories for the offline test suite (issue #16).

Plain functions -- deliberately **not** pytest fixtures -- so any test module
in any lane can ``from tests.harness.factories import make_...`` without
needing to be wired into a fixture graph. Every factory returns a real
``music_video_maker.contracts`` object (or, where the consumer needs to
duck-type an external library's return shape -- e.g. a mocked stable-ts
``model.align()`` call -- a minimal stand-in built from stdlib types only).

Nothing here imports torch, pydub, ComfyUI, or the network. WAV files are
synthesized with the stdlib ``wave`` module; nothing binary is committed to
the repo.

See ``tests/fixtures/workflows/README.md`` for the MiniMax H3 ``class_type``
names used below. They match the real, committed ``workflow_api.json`` /
``workflow_i2v_api.json`` at the repo root -- issue #18 authored those
against a live ComfyUI on doris and validated each by actually rendering a
5s clip end-to-end, so every node kind and wiring here is confirmed, not
guessed.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

from music_video_maker import contracts
from music_video_maker.continuity import DEFAULT_CAST_IMAGE_TITLE, DEFAULT_SEED_FRAME_TITLE

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
WORKFLOWS_DIR = FIXTURES_DIR / "workflows"
LYRICS_DIR = FIXTURES_DIR / "lyrics"

# --------------------------------------------------------------------------- #
# Stage 4a: workflow / graph-introspection fixtures (issue #8)
# --------------------------------------------------------------------------- #
#
# Reconciled against docs/h3-node-schema.md (live GET /object_info on doris,
# ComfyUI v0.30.2) and issue #18's real, rendered workflow_api.json /
# workflow_i2v_api.json. class_type names and wiring below are CONFIRMED --
# every AUTOGROW_V3 dotted key, the sampler chain, and both H3 node schemas
# were validated by an actual render on doris, not guessed.

# Local H3 nodes (core ComfyUI, comfy_extras/nodes_minimax_h3.py). Note the
# capitalization: "MiniMaxH3..." (capital M in "Max").
CLASS_TYPE_H3_REFERENCE_TO_VIDEO = "MiniMaxH3ReferenceToVideo"
CLASS_TYPE_H3_IMAGE_TO_VIDEO = "MiniMaxH3ImageToVideo"  # I2V path (#12); see make_workflow_i2v()
CLASS_TYPE_H3_EMPTY_LATENT_AV = "EmptyMiniMaxH3LatentAV"  # unused: both H3 nodes emit own LATENT
# Optional quality-tuning patch node, not used by the canonical committed
# templates -- the official ComfyUI-shipped MiniMax H3 templates omit it
# too (UNETLoader's MODEL feeds BasicScheduler/BasicGuider directly).
CLASS_TYPE_H3_SIGMA_SHIFT = "MiniMaxH3SigmaShift"

# Cloud-API variants (comfy_api_nodes/nodes_minimax.py) -- billed, remote,
# DO NOT USE. Spelled "Minimax..." (lowercase m) vs the local "MiniMaxH3...";
# an easy misgrab when introspecting the node registry by substring match.
CLOUD_API_CLASS_TYPES = (
    "MinimaxHailuo03FirstLastFrameNode",
    "MinimaxHailuo03ReferenceNode",
    "MinimaxHailuo03TextToVideoNode",
    "MinimaxHailuoVideoNode",
    "MinimaxImageToVideoNode",
    "MinimaxTextToVideoNode",
)

# Stock ComfyUI nodes the H3 graph also uses. There is no H3-specific
# sampler -- the real pipeline is the same generic ComfyUI "advanced
# sampling" chain (KSamplerSelect + BasicScheduler + RandomNoise +
# BasicGuider + SamplerCustomAdvanced) any other model uses, confirmed
# against ComfyUI's own official MiniMax H3 template and a real render.
# There is no `KSampler` node and no negative CONDITIONING anywhere in this
# graph -- `BasicGuider` takes only a single (positive) conditioning input.
CLASS_TYPE_UNET_LOADER = "UNETLoader"
CLASS_TYPE_CLIP_LOADER = "CLIPLoader"  # used with type="minimax"
CLASS_TYPE_VAE_LOADER = "VAELoader"  # used twice: video VAE and audio VAE
CLASS_TYPE_IMAGE_LOAD = "LoadImage"
CLASS_TYPE_AUDIO_LOAD = "LoadAudio"
CLASS_TYPE_SAMPLER_SELECT = "KSamplerSelect"
CLASS_TYPE_SCHEDULER = "BasicScheduler"
CLASS_TYPE_NOISE = "RandomNoise"
CLASS_TYPE_GUIDER = "BasicGuider"
CLASS_TYPE_SAMPLER_ADVANCED = "SamplerCustomAdvanced"
CLASS_TYPE_VAE_DECODE = "VAEDecode"
CLASS_TYPE_VAE_DECODE_AUDIO = "VAEDecodeAudio"
CLASS_TYPE_VIDEO_CREATE = "CreateVideo"
CLASS_TYPE_VIDEO_SAVE = "SaveVideo"

# Expected occurrence count of each required class_type in the baseline
# graph. VAELoader is the one node kind that legitimately appears twice --
# the video VAE and the audio VAE are two different weight files loaded
# through two separate node instances (see the "ambiguous" fixture below).
REQUIRED_CLASS_TYPE_COUNTS = {
    CLASS_TYPE_UNET_LOADER: 1,
    CLASS_TYPE_CLIP_LOADER: 1,
    CLASS_TYPE_VAE_LOADER: 2,
    CLASS_TYPE_IMAGE_LOAD: 1,
    CLASS_TYPE_AUDIO_LOAD: 1,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO: 1,
    CLASS_TYPE_SAMPLER_SELECT: 1,
    CLASS_TYPE_SCHEDULER: 1,
    CLASS_TYPE_NOISE: 1,
    CLASS_TYPE_GUIDER: 1,
    CLASS_TYPE_SAMPLER_ADVANCED: 1,
    CLASS_TYPE_VAE_DECODE: 1,
    CLASS_TYPE_VAE_DECODE_AUDIO: 1,
    CLASS_TYPE_VIDEO_CREATE: 1,
    CLASS_TYPE_VIDEO_SAVE: 1,
}
REQUIRED_CLASS_TYPES = tuple(REQUIRED_CLASS_TYPE_COUNTS)

# Declared output slot count per class_type, used by the connectivity tests
# to check that a [node_id, output_index] reference names an output the
# target node actually has. Confirmed from a live /object_info dump.
NODE_OUTPUT_SLOT_COUNTS = {
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO: 2,  # positive CONDITIONING, LATENT
    CLASS_TYPE_H3_IMAGE_TO_VIDEO: 2,  # positive CONDITIONING, LATENT
    CLASS_TYPE_H3_EMPTY_LATENT_AV: 1,  # LATENT
    CLASS_TYPE_H3_SIGMA_SHIFT: 1,  # MODEL
    CLASS_TYPE_UNET_LOADER: 1,  # MODEL
    CLASS_TYPE_CLIP_LOADER: 1,  # CLIP
    CLASS_TYPE_VAE_LOADER: 1,  # VAE
    CLASS_TYPE_IMAGE_LOAD: 2,  # IMAGE, MASK
    CLASS_TYPE_AUDIO_LOAD: 1,  # AUDIO
    CLASS_TYPE_SAMPLER_SELECT: 1,  # SAMPLER
    CLASS_TYPE_SCHEDULER: 1,  # SIGMAS
    CLASS_TYPE_NOISE: 1,  # NOISE
    CLASS_TYPE_GUIDER: 1,  # GUIDER
    CLASS_TYPE_SAMPLER_ADVANCED: 2,  # output, denoised_output (both LATENT)
    CLASS_TYPE_VAE_DECODE: 1,  # IMAGE
    CLASS_TYPE_VAE_DECODE_AUDIO: 1,  # AUDIO
    CLASS_TYPE_VIDEO_CREATE: 1,  # VIDEO
    CLASS_TYPE_VIDEO_SAVE: 0,  # terminal node, no outputs
}

# Real weight filenames confirmed on doris (comfyui-setup-summary.md). These
# were already correct and are unaffected by this reconciliation.
WEIGHT_DIFFUSION_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
WEIGHT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
WEIGHT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
WEIGHT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

WORKFLOW_BASELINE_PATH = WORKFLOWS_DIR / "baseline_h3.json"
WORKFLOW_SHIFTED_IDS_PATH = WORKFLOWS_DIR / "shifted_ids_h3.json"
WORKFLOW_AMBIGUOUS_VAE_LOADERS_PATH = WORKFLOWS_DIR / "ambiguous_vae_h3.json"
WORKFLOW_MISSING_NODE_PATH = WORKFLOWS_DIR / "missing_node_h3.json"
WORKFLOW_I2V_PATH = WORKFLOWS_DIR / "i2v_h3.json"


def _node(class_type: str, title: str, inputs: dict) -> dict:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _autogrow_key(field: str, prefix: str, index: int = 0) -> str:
    """Build the confirmed key for one slot of a ``COMFY_AUTOGROW_V3``
    dynamic multi-input (``ref_images`` / ``ref_audios`` on
    ``MiniMaxH3ReferenceToVideo``).

    CONFIRMED (issue #18): ComfyUI's dynamic-input expansion
    (``comfy_api/latest/_io.py``'s ``finalize_prefix``) joins the field name
    and the generated slot name with a dot, e.g. ``"ref_images.ref_image_0"``
    -- and that dotted string is an ordinary flat key in the node's
    ``inputs`` dict, holding a single plain ``[node_id, output_index]`` link,
    exactly like any other input. There is no nesting and no list-of-links
    wrapper; verified both by reading ComfyUI's own dynamic-input expansion
    code and by a real render on doris using this exact key shape.
    """
    return f"{field}.{prefix}{index}"


def _looks_like_node_ref(value: object) -> bool:
    """True if ``value`` is a plain ``[node_id, output_index]`` reference."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _remap_node_refs(inputs: dict, id_map: dict[str, str]) -> dict:
    """Copy ``inputs`` with every node-reference value (including the
    AUTOGROW_V3 dotted-key entries, which hold a plain link like any other
    input) rewritten through ``id_map``."""

    def remap_value(value):
        if _looks_like_node_ref(value):
            node_id, out_idx = value
            return [id_map.get(node_id, node_id), out_idx]
        return value

    return {key: remap_value(value) for key, value in inputs.items()}


def _shift_workflow_ids(workflow: contracts.Workflow, id_map: dict[str, str]) -> contracts.Workflow:
    """Copy ``workflow`` with every node id -- dict keys and node-reference
    input values alike -- rewritten through ``id_map``."""
    return {
        id_map[node_id]: {**node, "inputs": _remap_node_refs(node["inputs"], id_map)}
        for node_id, node in workflow.items()
    }


def make_workflow_baseline() -> contracts.Workflow:
    """Baseline MiniMax H3 (ref2va) API-format graph -- the same shape as the
    real, committed ``workflow_api.json`` at the repo root, which issue #18
    authored against a live ComfyUI on doris and validated by actually
    rendering a 5s clip end-to-end.

    Wiring: ``UNETLoader`` -> (``BasicScheduler``, ``BasicGuider``);
    ``CLIPLoader`` + two ``VAELoader``\\ s (video, audio) + ``LoadImage`` +
    ``LoadAudio`` -> ``MiniMaxH3ReferenceToVideo`` -> positive
    ``CONDITIONING`` -> ``BasicGuider`` -> ``GUIDER``; ``RandomNoise`` +
    ``GUIDER`` + ``KSamplerSelect`` + ``SIGMAS`` + the H3 node's own
    ``LATENT`` output -> ``SamplerCustomAdvanced`` -> (``VAEDecode``,
    ``VAEDecodeAudio``) -> ``CreateVideo`` -> ``SaveVideo``.

    ``prompt`` is a plain ``STRING`` input directly on
    ``MiniMaxH3ReferenceToVideo`` -- there is no text-encode node in this
    graph at all.

    There is no ``KSampler`` and no negative conditioning anywhere in this
    graph -- confirmed against ComfyUI's own official MiniMax H3 template
    (``video_minimax_h3_r2v.json``, shipped in
    ``comfyui_workflow_templates_json``) and a real render: ``BasicGuider``
    takes a single (positive) ``CONDITIONING`` input, full stop.

    ``ref_images`` / ``ref_audios`` use the confirmed AUTOGROW_V3 dotted-key
    shape -- see ``_autogrow_key()``.
    """
    return {
        "10": _node(
            CLASS_TYPE_UNET_LOADER,
            "Load H3 UNET (ref2va)",
            {"unet_name": WEIGHT_DIFFUSION_MODEL, "weight_dtype": "default"},
        ),
        "30": _node(
            CLASS_TYPE_CLIP_LOADER,
            "Load H3 CLIP",
            {"clip_name": WEIGHT_TEXT_ENCODER, "type": "minimax"},
        ),
        "40": _node(
            CLASS_TYPE_VAE_LOADER,
            "Load Video VAE",
            {"vae_name": WEIGHT_VIDEO_VAE},
        ),
        "41": _node(
            CLASS_TYPE_VAE_LOADER,
            "Load Audio VAE",
            {"vae_name": WEIGHT_AUDIO_VAE},
        ),
        "50": _node(
            CLASS_TYPE_IMAGE_LOAD,
            "Load Cast Reference",
            {"image": "cast_reference.png"},
        ),
        "51": _node(
            CLASS_TYPE_AUDIO_LOAD,
            "Load Vocal Stem",
            {"audio": "chunk_0000.wav"},
        ),
        "60": _node(
            CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
            "H3 Reference To Video",
            {
                "clip": ["30", 0],
                "vae": ["40", 0],
                "audio_vae": ["41", 0],
                "prompt": "placeholder prompt",
                "width": 1344,
                "height": 768,
                "length": 124,
                "ref_image_size": "match",
                _autogrow_key("ref_images", "ref_image_", 0): ["50", 0],
                _autogrow_key("ref_audios", "ref_audio_", 0): ["51", 0],
            },
        ),
        "71": _node(
            CLASS_TYPE_SAMPLER_SELECT,
            "H3 Sampler Select",
            {"sampler_name": "res_multistep"},
        ),
        "72": _node(
            CLASS_TYPE_SCHEDULER,
            "H3 Scheduler",
            {"model": ["10", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0},
        ),
        "73": _node(
            CLASS_TYPE_NOISE,
            "H3 Noise Seed",
            {"noise_seed": 0},
        ),
        "74": _node(
            CLASS_TYPE_GUIDER,
            "H3 Guider",
            {"model": ["10", 0], "conditioning": ["60", 0]},
        ),
        "75": _node(
            CLASS_TYPE_SAMPLER_ADVANCED,
            "H3 Sample",
            {
                "noise": ["73", 0],
                "guider": ["74", 0],
                "sampler": ["71", 0],
                "sigmas": ["72", 0],
                "latent_image": ["60", 1],
            },
        ),
        "80": _node(
            CLASS_TYPE_VAE_DECODE,
            "Decode H3 Video Latent",
            {"samples": ["75", 0], "vae": ["40", 0]},
        ),
        "81": _node(
            CLASS_TYPE_VAE_DECODE_AUDIO,
            "Decode H3 Audio Latent",
            {"samples": ["75", 0], "vae": ["41", 0]},
        ),
        "90": _node(
            CLASS_TYPE_VIDEO_CREATE,
            "Create H3 Video",
            {"images": ["80", 0], "audio": ["81", 0], "fps": 24, "bit_depth": 8},
        ),
        "100": _node(
            CLASS_TYPE_VIDEO_SAVE,
            "Save Output Video",
            {
                "video": ["90", 0],
                "filename_prefix": "mvm_chunk_0000",
                "format": "auto",
                "codec": "auto",
            },
        ),
    }


def make_workflow_shifted_ids() -> contracts.Workflow:
    """Same graph as :func:`make_workflow_baseline`, renumbered node ids.

    Proves that graph introspection (#8) locates nodes by ``class_type``,
    never by hardcoded or positional node id -- user edits on the ComfyUI
    canvas renumber ids freely. Built by remapping the baseline output
    (rather than a hand-duplicated literal) so the two can never drift.
    """
    baseline = make_workflow_baseline()
    id_map = {node_id: f"9{int(node_id):03d}" for node_id in baseline}
    return _shift_workflow_ids(baseline, id_map)


def make_workflow_ambiguous_vae_loaders() -> contracts.Workflow:
    """Baseline graph with both ``VAELoader`` nodes left at ComfyUI's
    un-customized default title -- when a node is never renamed on the
    canvas, its ``_meta.title`` equals its ``class_type`` verbatim. That
    makes title text alone unable to distinguish the video-VAE loader from
    the audio-VAE loader; #8 must instead trace which downstream input on
    ``MiniMaxH3ReferenceToVideo`` each output feeds (``vae`` vs
    ``audio_vae``).

    Replaces the old positive/negative ``CLIPTextEncode`` ambiguity fixture:
    that ambiguity no longer exists now that this graph has no text-encode
    node at all (see docs/h3-node-schema.md). The two same-``class_type``
    ``VAELoader`` nodes are the real disambiguation problem in this graph.
    """
    wf = copy.deepcopy(make_workflow_baseline())
    for node in wf.values():
        if node["class_type"] == CLASS_TYPE_VAE_LOADER:
            node["_meta"]["title"] = CLASS_TYPE_VAE_LOADER
    return wf


def make_workflow_missing_node(*, omit: str = CLASS_TYPE_AUDIO_LOAD) -> contracts.Workflow:
    """Baseline graph with one required ``class_type`` removed entirely.

    Lets #8 prove it fails loudly -- not silently -- when a required node is
    absent from the template. Any node-reference inputs elsewhere that
    pointed at the removed node are deliberately left dangling (simulating a
    template author who deleted a node without rewiring its consumers) --
    e.g. omitting ``LoadAudio`` leaves the H3 node's
    ``ref_audios.ref_audio_0`` pointing at a node id no longer present.
    """
    wf = make_workflow_baseline()
    return {node_id: node for node_id, node in wf.items() if node["class_type"] != omit}


def make_workflow_i2v() -> contracts.Workflow:
    """I2V continuity-path graph (issue #12), matching the real, committed
    ``workflow_i2v_api.json`` at the repo root -- issue #18 authored and
    validated it by actually rendering a 5s clip through
    ``MiniMaxH3ImageToVideo`` on doris.

    Differences from :func:`make_workflow_baseline`, all confirmed against a
    live ``/object_info`` dump:

    - The H3 node is ``MiniMaxH3ImageToVideo`` (the ``fl2va`` weights), not
      ``MiniMaxH3ReferenceToVideo``. Its real required inputs are just
      ``clip, vae, prompt, width, height, length`` plus optional
      ``first_frame`` / ``last_frame`` ``IMAGE`` sockets -- no
      ``ref_image_size``, no ``ref_images``/``ref_audios`` AUTOGROW inputs,
      and no ``audio_vae`` (unlike the reference-to-video node).
    - The seed frame from the previous chunk
      (``continuity.DEFAULT_SEED_FRAME_TITLE``, "Seed Frame") is wired to
      ``first_frame`` so this chunk's motion continues from where the last
      one left off. ``last_frame`` is deliberately **left unwired** (#44):
      it was once connected to the cast photo in the belief that it acted as
      an identity anchor, but ``comfy_extras/nodes_minimax_h3.py`` resolves
      it to a keyframe pinned at ``frame_count - 1`` and re-injects it every
      sampling step -- so it does not *influence* identity, it *dictates the
      final frame*. Every chained chunk ended on a cover-cropped portrait,
      and since the next chained chunk is seeded from that frame, it began
      on one too.
    - The cast reference ``LoadImage`` stays in the graph, titled as before,
      so the orchestrator's title-based injection is unchanged -- but with
      nothing consuming it, ComfyUI never executes it. Identity on this path
      comes from the seed frame, and drift is answered by re-anchoring
      (``i2v_reanchor_interval``), which routes a chunk back through the
      audio-conditioned reference template where the cast photo is a real
      reference rather than a mandated frame.
    - ``MiniMaxH3ImageToVideo`` has no audio-conditioning input at all, so
      there is no VAE-decode-audio stage to throw away (Stage 5 always
      discards generated audio regardless). Instead ``LoadAudio`` (the
      sliced vocal stem, which issue #18's body explicitly requires the
      orchestrator be able to inject on every template) is wired directly
      into ``CreateVideo.audio``, so per-chunk I2V clips at least carry the
      real vocal stem for manual review even though final assembly replaces
      it with the pristine master track either way.
    """
    wf = copy.deepcopy(make_workflow_baseline())

    h3_id, h3_node = next(
        (nid, n) for nid, n in wf.items() if n["class_type"] == CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    )
    h3_node["class_type"] = CLASS_TYPE_H3_IMAGE_TO_VIDEO
    h3_node["_meta"]["title"] = "H3 Image To Video"

    _cast_id, cast_node = next(
        (nid, n) for nid, n in wf.items() if n["class_type"] == CLASS_TYPE_IMAGE_LOAD
    )
    assert cast_node["_meta"]["title"] == DEFAULT_CAST_IMAGE_TITLE, (
        "baseline's cast LoadImage title drifted from continuity.DEFAULT_CAST_IMAGE_TITLE"
    )

    wf["55"] = _node(
        CLASS_TYPE_IMAGE_LOAD,
        DEFAULT_SEED_FRAME_TITLE,
        {"image": "seed_placeholder.png"},
    )

    h3_node["inputs"] = {
        "clip": h3_node["inputs"]["clip"],
        "vae": h3_node["inputs"]["vae"],
        "first_frame": ["55", 0],
        "prompt": h3_node["inputs"]["prompt"],
        "width": h3_node["inputs"]["width"],
        "height": h3_node["inputs"]["height"],
        "length": h3_node["inputs"]["length"],
    }

    # MiniMaxH3ImageToVideo has no audio-conditioning input, so this template
    # has no VAEDecodeAudio/audio-VAELoader stage at all -- CreateVideo.audio
    # is wired straight to the vocal-stem LoadAudio instead of a decoded
    # generated-audio latent.
    vae_decode_audio_id, _ = next(
        (nid, n) for nid, n in wf.items() if n["class_type"] == CLASS_TYPE_VAE_DECODE_AUDIO
    )
    audio_vae_id, _ = next(
        (nid, n)
        for nid, n in wf.items()
        if n["class_type"] == CLASS_TYPE_VAE_LOADER and n["inputs"]["vae_name"] == WEIGHT_AUDIO_VAE
    )
    audio_load_id, _ = next(
        (nid, n) for nid, n in wf.items() if n["class_type"] == CLASS_TYPE_AUDIO_LOAD
    )
    del wf[vae_decode_audio_id]
    del wf[audio_vae_id]
    create_video_id, create_video_node = next(
        (nid, n) for nid, n in wf.items() if n["class_type"] == CLASS_TYPE_VIDEO_CREATE
    )
    create_video_node["inputs"]["audio"] = [audio_load_id, 0]

    return wf


def load_workflow_fixture(path: Path) -> contracts.Workflow:
    """Load one of the committed JSON fixtures under ``tests/fixtures/workflows/``."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Lyrics fixtures (issues #3, #6)
# --------------------------------------------------------------------------- #

LYRICS_PLAIN_PATH = LYRICS_DIR / "plain.txt"
LYRICS_TAGGED_PATH = LYRICS_DIR / "tagged.txt"
LYRICS_MIXED_INHERIT_PATH = LYRICS_DIR / "mixed_inherit.txt"
LYRICS_GAPS_PATH = LYRICS_DIR / "gaps.txt"
LYRICS_UNKNOWN_CHARACTER_PATH = LYRICS_DIR / "unknown_character.txt"


def make_lyric_line(
    index: int, text: str, character: str | None = None, raw: str | None = None
) -> contracts.LyricLine:
    """Build a single :class:`contracts.LyricLine`.

    If ``raw`` is not given, it is synthesized to look like a ``[Name: Role]``
    tagged source line when ``character`` is set, or the bare text otherwise.
    """
    if raw is None:
        raw = f"[{character}: Role] {text}" if character else text
    return contracts.LyricLine(
        index=index,
        text=text,
        characters=(character,) if character else (),
        raw=raw,
    )


def make_lyric_lines(lines: list[tuple[str, str | None]]) -> tuple[contracts.LyricLine, ...]:
    """Build an auto-indexed tuple of :class:`contracts.LyricLine` from
    ``(text, character)`` pairs."""
    return tuple(
        make_lyric_line(i, text, character) for i, (text, character) in enumerate(lines)
    )


# --------------------------------------------------------------------------- #
# Synthetic audio (issues #3, #4, #7)
# --------------------------------------------------------------------------- #


def write_silent_wav(
    path: Path, seconds: float, sample_rate: int = 44100, channels: int = 1
) -> Path:
    """Write a silent 16-bit PCM WAV of the given duration using stdlib ``wave``.

    No ffmpeg, no pydub, nothing binary committed to the repo -- generated
    fresh in each test run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(0, round(seconds * sample_rate))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames * channels)
    return path


def write_tone_wav(
    path: Path,
    seconds: float,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = 44100,
    channels: int = 1,
) -> Path:
    """Write an audible sine-tone 16-bit PCM WAV using stdlib ``wave`` + ``struct``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(0, round(seconds * sample_rate))
    amplitude = max(0.0, min(1.0, amplitude))
    peak = int(amplitude * 32767)
    samples = [
        int(peak * math.sin(2 * math.pi * frequency * (i / sample_rate)))
        for i in range(n_frames)
    ]
    frame_data = struct.pack(f"<{n_frames}h", *samples)
    if channels > 1:
        # Duplicate the mono signal across channels rather than silently
        # dropping the requested channel count.
        interleaved = bytearray()
        for i in range(n_frames):
            frame_data_i = struct.pack("<h", samples[i])
            interleaved += frame_data_i * channels
        frame_data = bytes(interleaved)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frame_data)
    return path


def make_oversized_file_stub(path: Path, size_bytes: int = 50 * 1024 * 1024 + 1) -> Path:
    """Create a sparse file that reports ``size_bytes`` via ``stat()`` without
    writing that many actual bytes to disk.

    For issue #7's 50 MB upload-limit check: default is one byte over the
    50 MB boundary. Pass a smaller ``size_bytes`` to build an under-the-limit
    stand-in instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        if size_bytes > 0:
            f.truncate(size_bytes)
    return path


# --------------------------------------------------------------------------- #
# Stage 1: canned alignment results (issues #3, #4)
# --------------------------------------------------------------------------- #


def make_word_timings(
    words: list[str], start: float, end: float
) -> tuple[contracts.WordTiming, ...]:
    """Evenly distribute ``words`` across ``[start, end]`` as :class:`contracts.WordTiming`."""
    if not words:
        return ()
    span = (end - start) / len(words)
    timings = []
    for i, word in enumerate(words):
        w_start = start + i * span
        w_end = start + (i + 1) * span
        timings.append(contracts.WordTiming(word=word, start=w_start, end=w_end))
    return tuple(timings)


def make_aligned_segment(
    index: int,
    text: str,
    start: float,
    end: float,
    character: str | None = None,
    words: tuple[contracts.WordTiming, ...] | None = None,
) -> contracts.AlignedSegment:
    """Build a single :class:`contracts.AlignedSegment`.

    Auto-generates word timings from ``text`` if ``words`` is omitted.
    """
    if words is None:
        words = make_word_timings(text.split(), start, end)
    return contracts.AlignedSegment(
        index=index,
        text=text,
        start=start,
        end=end,
        words=words,
        characters=(character,) if character else (),
    )


def make_alignment_result_normal_song() -> contracts.AlignmentResult:
    """A well-behaved alignment: monotonic, non-overlapping segments within the
    standard 4-15s chunk window, small natural gaps between them."""
    specs = [
        (0.0, 6.5, "walking through the empty halls tonight", "Dianne"),
        (7.2, 12.0, "nobody's watching nobody cares", "Dianne"),
        (13.0, 18.0, "the lights flicker but i don't mind", "Dianne"),
        (19.5, 26.0, "i've been here a thousand times before", "Dianne"),
    ]
    segments = tuple(
        make_aligned_segment(i, text, start, end, character)
        for i, (start, end, text, character) in enumerate(specs)
    )
    return contracts.AlignmentResult(segments=segments, track_duration=32.0)


def make_alignment_result_short_segments() -> contracts.AlignmentResult:
    """Segments shorter than the 4 s Stage 2 minimum -- exercises pad/merge logic."""
    specs = [
        (0.0, 1.5, "oh", "Dianne"),
        (2.0, 3.2, "yeah", "Dianne"),
        (3.5, 5.8, "come on", "Dianne"),
    ]
    segments = tuple(
        make_aligned_segment(i, text, start, end, character)
        for i, (start, end, text, character) in enumerate(specs)
    )
    return contracts.AlignmentResult(segments=segments, track_duration=10.0)


def make_alignment_result_long_segment() -> contracts.AlignmentResult:
    """One segment longer than the 15 s Stage 2 maximum -- exercises split logic."""
    segments = (
        make_aligned_segment(
            0,
            "we started this together side by side and we'll carry on despite the noise "
            "no matter how far we wander we will always find our way back home again",
            0.0,
            18.5,
            "Dianne",
        ),
    )
    return contracts.AlignmentResult(segments=segments, track_duration=20.0)


def make_alignment_result_with_gaps() -> contracts.AlignmentResult:
    """Segments separated by long instrumental gaps (guitar solo / bridge)."""
    specs = [
        (0.0, 5.0, "count the days until the sun comes back", "Dianne"),
        (35.0, 40.0, "nothing but the echo of a drum", "Dianne"),  # 30s instrumental gap
        (41.0, 47.0, "give the silence room to breathe", "Marcus"),
    ]
    segments = tuple(
        make_aligned_segment(i, text, start, end, character)
        for i, (start, end, text, character) in enumerate(specs)
    )
    return contracts.AlignmentResult(segments=segments, track_duration=55.0)


def _stablets_word(word: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=word, start=start, end=end, probability=0.99)


def _stablets_segment(text: str, start: float, end: float) -> SimpleNamespace:
    words = text.split()
    span = (end - start) / max(len(words), 1)
    word_objs = [
        _stablets_word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)
    ]
    return SimpleNamespace(text=text, start=start, end=end, words=word_objs)


def make_raw_stablets_result() -> SimpleNamespace:
    """Stand-in for what a mocked ``stable_ts`` ``model.align()`` call returns.

    Stage 1 (issue #3) mocks the whole stable-ts model, so its tests need
    something that duck-types stable-ts's ``WhisperResult``: attribute
    access (``result.segments[i].text/.start/.end``, and
    ``segment.words[j].word/.start/.end``), not a plain dict. Built with
    ``SimpleNamespace`` so this stays torch- and stable-ts-free.
    """
    segments = [
        _stablets_segment("walking through the empty halls tonight", 0.0, 6.5),
        _stablets_segment("nobody's watching nobody cares", 7.2, 12.0),
        _stablets_segment("the lights flicker but i don't mind", 13.0, 18.0),
    ]
    return SimpleNamespace(segments=segments, language="en")


# --------------------------------------------------------------------------- #
# Cast factories (issue #2's contracts.CastMember; not the config model)
# --------------------------------------------------------------------------- #


def make_cast_member(
    name: str = "Dianne",
    role: str = "Lead Vocalist, smiling constantly, oblivious",
    image: Path | None = None,
) -> contracts.CastMember:
    """Build a single :class:`contracts.CastMember`. Image paths are placeholders
    (not required to exist on disk -- Stage 3 upload code owns that check)."""
    if image is None:
        image = Path(f"cast/{name.lower()}_ref.png")
    return contracts.CastMember(name=name, role=role, image=image)


def make_cast_dict() -> dict[str, contracts.CastMember]:
    """A small cast dictionary including a non-singing background member.

    Mirrors the example cast in ``music_video_maker/config.py``'s docstring
    (Dianne the lead, Rex the drummer) plus a second vocalist (Marcus) so
    character-switching fixtures (issue #6) have someone to switch to.
    """
    return {
        "Dianne": make_cast_member("Dianne", "Lead Vocalist, smiling constantly, oblivious"),
        "Marcus": make_cast_member("Marcus", "Backup Vocalist, watching from the wings"),
        "Rex": make_cast_member("Rex", "Drummer, background, never sings"),
    }
