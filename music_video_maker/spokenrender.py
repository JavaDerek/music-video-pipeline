"""Render a spoken script one shot at a time, each started from its camera setup's still (#119).

**General form: a lesson captured as a linter and a guide, while the code that exercises it lives
in a private scratch driver, is half captured.** :mod:`music_video_maker.spokenscript` refuses a
dialogue script's known failures before any GPU time; this module is the path the script then
takes to become a video, so the next script gets the render as well as the refusals.

Each shot is one generation on the local graph (the committed ``workflow_api.json``, nodes found
by ``class_type``):

* the speaker's ``portrait`` as reference image 1, anyone else in frame after it;
* the finished line (synthesized elsewhere, one WAV per shot, ``<lines>/<shot id>.wav``) as the
  reference audio, padded to the shot's grid length plus a small margin;
* **the setup's still and the same line pinned at frame 0** with ``MiniMaxH3AddGuide`` between
  the reference node and the guider (:func:`anchor_setup`). Measured on three cut-backs to two
  setups, 2 seeds: started from the still, a shot's first frame scored 3.7-5.0 from it against
  27.6-44.3 without (:mod:`music_video_maker.setupmatch`); with the line pinned as well, SyncNet
  passed 3 of 6 where the line given only as a reference passed 0 of 15 (#120, #121);
* a frame count on H3's ``5 + 17k`` grid covering the line plus :data:`TAIL_SECONDS`, never under
  the 124-frame trained floor: the render is quantized, the cut is not (#100). Assembly trims
  each take to the last word plus the tail and realises ``pause_before`` as a hold on the first
  frame, because the model starts the mouth at frame 0 and leading silence in the audio is a
  mouth moving over nothing (#103, #107).

The finished lines are the only audio in the cut; everything the model generates is discarded,
the same rule as the master track on the sung path. Every take is checked with ``setupmatch``
and recorded in ``provenance.json`` as it lands, so a run that dies on shot 7 still proves 1-6.

Local graph only, so ``local_h3`` and its authorization lock apply (:mod:`lipsync`): the anchor
is ``MiniMaxH3AddGuide``, and the hosted API has not been tried this way. Not here yet (#119):
line synthesis, take scoring and retakes, an ambience bed and loudness, and a check of the
speaker's face size on the still (crop it to roughly 200 px of 480 tall; the prompt does not set
size). The camera may still drift inside a shot; ``setupmatch`` checks the first frame only.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
import subprocess
import sys
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from music_video_maker import lipsync, setupmatch
from music_video_maker import spokenscript as ss
from music_video_maker.contracts import H3_FRAME_GRID
from music_video_maker.provenance import ShotProvenance, record_shot, sha256_file, utc_now_iso
from music_video_maker.workflow_graph import (
    _free_node_id,  # one allocator for new node ids, shared with insert_lora
    assert_no_cloud_api_nodes,
    find_nodes_by_class_type,
    find_one_node,
    resolve_chunk_seed,
)

logger = logging.getLogger(__name__)

Workflow = dict[str, Any]
SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]

TAIL_SECONDS = 0.5
"""Face kept after the last word."""
PAD_MARGIN_SECONDS = 0.2
"""The reference line is padded a little past the shot's own length, as measured."""
MAX_REFERENCE_IMAGES = 4
SPEAKER_TAG = " <Picture 1> is the person speaking, and <Audio 1> is their voice."

CLASS_REFERENCE = "MiniMaxH3ReferenceToVideo"
CLASS_ADD_GUIDE = "MiniMaxH3AddGuide"
CLASS_GUIDER = "BasicGuider"
CLASS_LOAD_IMAGE = "LoadImage"
CLASS_LOAD_AUDIO = "LoadAudio"
CLASS_NOISE = "RandomNoise"
CLASS_SAVE = "SaveVideo"
CLASS_UNET = "UNETLoader"
REF_IMAGE_KEY = "ref_images.ref_image_{}"
REF_AUDIO_KEY = "ref_audios.ref_audio_0"


class SpokenRenderError(RuntimeError):
    """A shot cannot be rendered as asked."""


# --------------------------------------------------------------------------- sizes


def shot_frames(seconds: float) -> int:
    """Frames to render for ``seconds`` of shot: on the grid, at least the trained floor."""
    needed = math.ceil(seconds * H3_FRAME_GRID.fps - 1e-9)
    frames = max(H3_FRAME_GRID.quantize_up(needed), H3_FRAME_GRID.trained_min_frames)
    if frames > H3_FRAME_GRID.trained_max_frames:
        raise SpokenRenderError(
            f"{seconds:.2f} s needs {frames} frames, past H3's trained "
            f"{H3_FRAME_GRID.trained_max_frames}; split the line into two shots")
    return frames


def speech_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------- the graph


def anchor_setup(workflow: Workflow, *, still: str, line: str | None) -> Workflow:
    """Pin ``still`` (and ``line``, when given) at frame 0, in place: a ``MiniMaxH3AddGuide``
    between the reference node's conditioning and the guider. The sampler keeps taking its latent
    from the reference node, which is what the measured graph did."""
    if find_nodes_by_class_type(workflow, CLASS_ADD_GUIDE):
        raise SpokenRenderError(f"this graph already carries a {CLASS_ADD_GUIDE}")
    ref_id, ref = find_one_node(workflow, CLASS_REFERENCE)
    _, guider = find_one_node(workflow, CLASS_GUIDER)
    if guider["inputs"].get("conditioning") != [ref_id, 0]:
        raise SpokenRenderError(
            f"{CLASS_GUIDER} does not take its conditioning from {CLASS_REFERENCE} {ref_id}; "
            "the anchor has nowhere to go in this graph")
    guide = {"positive": [ref_id, 0], "latent": [ref_id, 1], "frame_idx": 0,
             "vae": ref["inputs"]["vae"]}
    image_id = _free_node_id(workflow)
    workflow[image_id] = {"class_type": CLASS_LOAD_IMAGE, "inputs": {"image": still}}
    guide["image"] = [image_id, 0]
    if line is not None:
        if "audio_vae" not in ref["inputs"]:
            raise SpokenRenderError(f"{CLASS_REFERENCE} has no audio_vae to pin a line with")
        audio_id = _free_node_id(workflow)
        workflow[audio_id] = {"class_type": CLASS_LOAD_AUDIO, "inputs": {"audio": line}}
        guide["audio_vae"] = ref["inputs"]["audio_vae"]
        guide["audio"] = [audio_id, 0]
    guide_id = _free_node_id(workflow)
    workflow[guide_id] = {"class_type": CLASS_ADD_GUIDE, "inputs": guide}
    guider["inputs"]["conditioning"] = [guide_id, 0]
    return workflow


def build_shot_workflow(template: Workflow, *, prompt: str, width: int, height: int, frames: int,
                        seed: int, images: Sequence[str], line: str | None, still: str,
                        prefix: str) -> Workflow:
    """One shot's graph from the reference-to-video template; ``template`` is not modified."""
    if not images:
        raise SpokenRenderError("a shot needs at least one reference image")
    workflow = copy.deepcopy(template)
    assert_no_cloud_api_nodes(workflow)
    _, ref = find_one_node(workflow, CLASS_REFERENCE)
    ri = ref["inputs"]
    ri.update(prompt=prompt, width=width, height=height, length=frames)
    first = ri.get(REF_IMAGE_KEY.format(0))
    if not first:
        raise SpokenRenderError(f"{CLASS_REFERENCE} has no {REF_IMAGE_KEY.format(0)} to fill")
    workflow[first[0]]["inputs"]["image"] = images[0]
    for i, name in enumerate(images[1:], 1):
        nid = _free_node_id(workflow)
        workflow[nid] = {"class_type": CLASS_LOAD_IMAGE, "inputs": {"image": name}}
        ri[REF_IMAGE_KEY.format(i)] = [nid, 0]
    audio_link = ri.get(REF_AUDIO_KEY)
    if line is not None:
        if not audio_link:
            raise SpokenRenderError(f"{CLASS_REFERENCE} has no {REF_AUDIO_KEY} to fill")
        workflow[audio_link[0]]["inputs"]["audio"] = line
    else:
        for key in [k for k in ri if k.startswith("ref_audios.")]:
            del ri[key]
        if audio_link and not _referenced(workflow, audio_link[0]):
            del workflow[audio_link[0]]
    find_one_node(workflow, CLASS_NOISE)[1]["inputs"]["noise_seed"] = int(seed)
    find_one_node(workflow, CLASS_SAVE)[1]["inputs"]["filename_prefix"] = prefix
    return anchor_setup(workflow, still=still, line=line)


def _referenced(workflow: Workflow, node_id: str) -> bool:
    return any(isinstance(v, list) and len(v) == 2 and v[0] == node_id
               for node in workflow.values() for v in node.get("inputs", {}).values())


def reference_cast(shot: ss.Shot) -> tuple[str, ...]:
    """Who the shot's reference images show, speaker first (the prompt calls them Picture 1)."""
    speaker = [shot.speaker] if shot.speaker else []
    order = speaker + [c for c in shot.characters if c != shot.speaker]
    return tuple(order[:MAX_REFERENCE_IMAGES])


def shot_prompt(script: ss.Script, shot: ss.Shot) -> str:
    """Description only (``compose_generation_prompt``), plus which picture and audio speak."""
    prompt = ss.compose_generation_prompt(script, shot)
    return prompt + SPEAKER_TAG if shot.line else prompt


# --------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class PlannedShot:
    shot: ss.Shot
    index: int            # position in the script; the seed derives from it, so --only reproduces
    frames: int
    keep_seconds: float   # what the edit keeps: the line plus the tail, or the shot's seconds
    line: Path | None


def plan_shots(script: ss.Script, lines_dir: Path, *, only: Iterable[str] | None = None
               ) -> list[PlannedShot]:
    """Every shot's size and line, checked before anything renders (all or nothing)."""
    wanted = set(only) if only else None
    if wanted:
        unknown = wanted - {s.id for s in script.shots}
        if unknown:
            raise SpokenRenderError(f"no such shot(s): {sorted(unknown)}")
    plans, missing = [], []
    for index, shot in enumerate(script.shots):
        if wanted and shot.id not in wanted:
            continue
        if shot.line:
            wav = Path(lines_dir) / f"{shot.id}.wav"
            if not wav.is_file():
                missing.append(shot.id)
                continue
            keep = speech_seconds(wav) + TAIL_SECONDS
            plans.append(PlannedShot(shot, index, shot_frames(keep), keep, wav))
        else:
            plans.append(PlannedShot(shot, index, shot_frames(shot.seconds), shot.seconds, None))
    if missing:
        raise SpokenRenderError(f"no finished line in {lines_dir} for shot(s) {missing} "
                                "(expected <shot id>.wav)")
    return plans


# --------------------------------------------------------------------------- rendering


@dataclass(frozen=True)
class RenderedShot:
    shot_id: str
    index: int
    take: Path
    setup: str
    setup_match: setupmatch.SetupMatch
    keep_seconds: float
    pause_before: float
    line: Path | None


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, check=False)


def _run(runner: SubprocessRunner, args: Sequence[str]) -> None:
    result = runner(args)
    if result.returncode != 0:
        err = result.stderr
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        raise SpokenRenderError(f"ffmpeg failed ({args[-1]}): {str(err).strip()[-300:]}")


def render_shots(script: ss.Script, plans: Sequence[PlannedShot], *, template: Workflow,
                 client: Any, stager: Any, out_dir: Path, seed: int, width: int, height: int,
                 runner: SubprocessRunner | None = None,
                 check: Callable[[Path, Path], setupmatch.SetupMatch] = setupmatch.check_shot,
                 provenance_path: Path | None = None) -> list[RenderedShot]:
    """Render each planned shot in order. The card is released (``free``) however this ends."""
    runner = runner or _default_runner
    out_dir = Path(out_dir)
    takes, padded = out_dir / "takes", out_dir / "lines"
    takes.mkdir(parents=True, exist_ok=True)
    padded.mkdir(parents=True, exist_ok=True)
    provenance_path = provenance_path or out_dir / "provenance.json"
    model = find_one_node(template, CLASS_UNET)[1]["inputs"].get("unet_name", "unknown")
    done: list[RenderedShot] = []
    try:
        for plan in plans:
            shot = plan.shot
            setup = script.setups[shot.setup]  # type: ignore[index]
            still = ss.setup_still_path(script, setup)
            portraits = [ss.cast_portrait_path(script, script.cast[k])
                         for k in reference_cast(shot)]
            images = [stager.upload_image(p) for p in portraits]
            still_name = stager.upload_image(still)
            line_name, line_pad = None, None
            if plan.line is not None:
                line_pad = padded / f"{shot.id}_pad.wav"
                pad_to = plan.frames / H3_FRAME_GRID.fps + PAD_MARGIN_SECONDS
                _run(runner, lipsync.build_pad_audio_args(plan.line, line_pad, seconds=pad_to))
                line_name = stager.upload_audio(line_pad)
            shot_seed = resolve_chunk_seed(seed, plan.index)
            workflow = build_shot_workflow(
                template, prompt=shot_prompt(script, shot), width=width, height=height,
                frames=plan.frames, seed=shot_seed, images=images, line=line_name,
                still=still_name, prefix=f"spoken_{shot.id}")
            requested = utc_now_iso()
            result = client.execute(workflow, chunk_id=plan.index, output_dir=takes / "raw")
            take = takes / f"{shot.id}.mp4"
            shutil.move(str(result.video_file), take)
            match = check(still, take)
            log = logger.info if match.matches else logger.warning
            log("shot %s: %s", shot.id, match.describe())
            inputs = {"reference_image": sha256_file(portraits[0]),
                      "setup_still": sha256_file(still)}
            if line_pad is not None:
                inputs["reference_audio"] = sha256_file(line_pad)
            record_shot(provenance_path, ShotProvenance(
                shot_id=shot.id, backend=lipsync.BACKEND_LOCAL_H3, model=model,
                tier=lipsync.TIER_DRAFT, resolution=f"{width}x{height}", frames=plan.frames,
                fps=H3_FRAME_GRID.fps, duration_seconds=plan.frames / H3_FRAME_GRID.fps,
                task_id=getattr(result, "prompt_id", None), requested_at_utc=requested,
                completed_at_utc=utc_now_iso(), inputs=inputs, output_file=str(take),
                output_sha256=sha256_file(take),
                notes={"setup": setup.key, "setupmatch_distance": match.distance,
                       "seed": shot_seed}))
            pause = shot.delivery.pause_before if shot.delivery else 0.0
            done.append(RenderedShot(shot.id, plan.index, take, setup.key, match,
                                     plan.keep_seconds, pause, plan.line))
    finally:
        client.free()
    return done


# --------------------------------------------------------------------------- assembly


def build_segment_args(take: Path, line: Path | None, dst: Path, *, keep_seconds: float,
                       pause_before: float, width: int, height: int) -> list[str]:
    """One shot of the cut: trimmed to ``keep_seconds``, ``pause_before`` held on the first frame,
    and the finished line (or silence) as its only audio."""
    vf = (f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={H3_FRAME_GRID.fps},format=yuv420p")
    if pause_before:
        vf += f",tpad=start_mode=clone:start_duration={pause_before:g}"
    total = keep_seconds + pause_before
    if line is not None:
        ms = int(round(pause_before * 1000))
        audio_in = ["-i", str(line)]
        af = (f"adelay={ms}|{ms},apad=whole_dur={total:.3f},aresample=48000,"
              "aformat=channel_layouts=stereo")
    else:
        audio_in = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        af = "aresample=48000,aformat=channel_layouts=stereo"
    return ["ffmpeg", "-y", "-v", "error", "-i", str(take), *audio_in,
            "-map", "0:v", "-map", "1:a", "-vf", vf, "-af", af, "-t", f"{total:.3f}",
            "-c:v", "libx264", "-crf", "17", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(dst)]


def assemble(script: ss.Script, rendered: Sequence[RenderedShot], out_dir: Path, *, width: int,
             height: int, runner: SubprocessRunner | None = None) -> Path:
    """The cut, in script order: every rendered shot trimmed and concatenated."""
    runner = runner or _default_runner
    segments = Path(out_dir) / "segments"
    segments.mkdir(parents=True, exist_ok=True)
    parts = []
    for shot in sorted(rendered, key=lambda r: r.index):
        dst = segments / f"{shot.shot_id}.mp4"
        _run(runner, build_segment_args(shot.take, shot.line, dst, keep_seconds=shot.keep_seconds,
                                        pause_before=shot.pause_before, width=width, height=height))
        parts.append(dst)
    listing = segments / "concat.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    cut = Path(out_dir) / "spoken_cut.mp4"
    _run(runner, ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                  "-c", "copy", str(cut)])
    return cut


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """``python -m music_video_maker.spokenrender SCRIPT --lines DIR --out DIR --h3-authorization
    FILE``: exit 0 rendered and every shot opens on its still, 1 rendered with a setup mismatch,
    2 refused before rendering."""
    from music_video_maker import config
    from music_video_maker.custody import CustodyError, preflight_free_vram
    from music_video_maker.execution import ComfyUIExecutionClient
    from music_video_maker.hardware import PROFILE_RTX_4090_24GB
    from music_video_maker.staging import ComfyUIAssetStager
    from music_video_maker.workflow_graph import WorkflowGraphError, load_workflow_template

    parser = argparse.ArgumentParser(prog="python -m music_video_maker.spokenrender",
                                     description="Render a spoken script shot by shot.")
    parser.add_argument("script", type=Path)
    parser.add_argument("--lines", type=Path, required=True, help="<shot id>.wav per spoken shot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h3-authorization", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("workflow_api.json"))
    parser.add_argument("--comfyui", default=config.DEFAULT_COMFYUI_URL)
    parser.add_argument("--only", default="", help="comma-separated shot ids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-free-vram-gb", type=float, default=16.0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    def refuse(message: str) -> int:
        sys.stderr.write(f"{message}\n")
        return 2

    try:
        script = ss.load_script(args.script)
        lipsync.validate_settings(lipsync.LipSyncSettings(
            backend=lipsync.BACKEND_LOCAL_H3, tier=lipsync.TIER_DRAFT,
            h3_authorization_file=args.h3_authorization))
        ss.require_clean(ss.lint_script(script, for_render=True))
        only = {s for s in args.only.split(",") if s} or None
        plans = plan_shots(script, args.lines, only=only)
        template = load_workflow_template(args.template)
    except (ss.SpokenScriptError, lipsync.LipSyncError, SpokenRenderError,
            WorkflowGraphError) as exc:
        return refuse(str(exc))
    stager = ComfyUIAssetStager(args.comfyui)
    try:
        preflight_free_vram(stager.session, args.comfyui, min_free_vram_gb=args.min_free_vram_gb,
                            hardware=PROFILE_RTX_4090_24GB)
    except CustodyError as exc:
        return refuse(str(exc))
    client = ComfyUIExecutionClient(args.comfyui)
    rendered = render_shots(script, plans, template=template, client=client, stager=stager,
                            out_dir=args.out, seed=args.seed, width=args.width, height=args.height)
    cut = assemble(script, rendered, args.out, width=args.width, height=args.height)
    for shot in rendered:
        sys.stdout.write(shot.setup_match.describe() + "\n")
    sys.stdout.write(f"cut: {cut}\n")
    return 0 if all(s.setup_match.matches for s in rendered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
