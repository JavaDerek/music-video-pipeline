"""Lip-sync / image-to-video backends: which model makes the talking shots.

Three backends, chosen by config (``lipsync_backend``) and never by editing
code:

``minimax_api``
    MiniMax's hosted API (``https://api.minimax.io``, model ``MiniMax-H3`` /
    ``MiniMax-H3-Max``) in reference-to-video mode: a reference image, a
    reference audio clip and a prompt in, a talking video out. The default,
    and the only backend a final render may ship. The API is governed by the
    MiniMax Open Platform Terms of Service (:data:`MINIMAX_TERMS`), under
    which the client keeps ownership of generated content -- unlike the open
    weights, whose Community License names the United States an Excluded
    Territory. The key comes from the process environment
    (``MINIMAX_API_KEY``) and is never written to disk, a log or a
    provenance record.

``local_draft``
    A locally run open model for free iteration on a workstation GPU. Never
    MiniMax H3. No model is integrated yet: the config refuses the backend
    until ``local_draft_model`` names one this module knows how to drive
    (:data:`LOCAL_DRAFT_MODELS`, empty until a model is chosen and its licence
    recorded), so a draft config cannot silently fall through to anything.

``local_h3``
    The ComfyUI path this project was built on -- the H3 open weights on a
    local card. Kept, and **hard-locked**: it refuses to run unless
    ``h3_authorization_file`` names a file that exists -- the written
    authorization MiniMax grants for use inside an Excluded Territory
    (``https://platform.minimax.io/h3-license``). A missing file is a
    config-load failure, before any alignment or GPU time.

Orthogonal to the backend is the **quality tier** (``quality_tier``):
``draft`` is low resolution for iteration (``MiniMax-H3-Max`` at 480P, or a
local draft model); ``final`` is 2K through ``minimax_api`` only. Every shot
records backend, model, resolution and tier in ``provenance.json``
(:mod:`music_video_maker.provenance`), and a final-grade render, publish or
package refuses to carry a shot from a backend that may not be exhibited.

Money is a first-class output here, not a surprise on an invoice: every API
run is costed from the published per-second prices before anything is
submitted (:func:`estimate_cost`), the estimate is logged, and a run over
:data:`CONFIRM_THRESHOLD_USD` needs an explicit confirmation (``--yes`` on
the CLI) or it does not start.

Everything that talks to the network goes through an injectable
``requests``-shaped session and an injectable sleeper/clock, so the whole
backend is tested offline against a scripted fake, the same way the ComfyUI
client is.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from music_video_maker import workflow_graph
from music_video_maker.contracts import (
    AudioChunk,
    ChunkResult,
    ChunkStatus,
    ExpandedPrompt,
    StagedAssets,
    Workflow,
)
from music_video_maker.provenance import (
    BACKEND_LOCAL_DRAFT,
    BACKEND_LOCAL_H3,
    BACKEND_MINIMAX_API,
    BACKENDS,
    PROVENANCE_FILENAME,
    TIER_DRAFT,
    TIER_FINAL,
    TIERS,
    ShotProvenance,
    record_shot,
    sha256_file,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = BACKEND_MINIMAX_API
DEFAULT_TIER = TIER_DRAFT
"""``draft`` by default: a run that forgot to say spends $0.05/s, not $0.13/s,
and a final render has to be asked for by name."""

MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_API_BASE_URL = "https://api.minimax.io"
MINIMAX_CREATE_PATH = "/v2/video_generation"
MINIMAX_QUERY_PATH = "/v2/query/video_generation/{task_id}"
MINIMAX_UPLOAD_PATH = "/v1/files/upload"
MINIMAX_UPLOAD_PURPOSE = "video_generation_input"

MINIMAX_MODEL_H3 = "MiniMax-H3"
MINIMAX_MODEL_H3_MAX = "MiniMax-H3-Max"

MINIMAX_MODEL_LIMITS: Mapping[str, Mapping[str, Any]] = {
    # API reference, read 2026-10-13: duration is an integer number of
    # seconds; H3 does 768P/2K and 4-15 s, H3-Max does 480P/768P and 5-15 s.
    MINIMAX_MODEL_H3: {"resolutions": ("768P", "2K"), "min_seconds": 4, "max_seconds": 15},
    MINIMAX_MODEL_H3_MAX: {"resolutions": ("480P", "768P"), "min_seconds": 5, "max_seconds": 15},
}
MINIMAX_AUDIO_MIN_SECONDS = 2.0
MINIMAX_AUDIO_MAX_SECONDS = 15.0
"""A reference audio clip is 2-15 s, and the clips together at most 15 s."""

MINIMAX_RATIOS: Mapping[str, float] = {
    "21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16,
}

TIER_MODELS: Mapping[str, tuple[str, str]] = {
    TIER_DRAFT: (MINIMAX_MODEL_H3_MAX, "480P"),
    TIER_FINAL: (MINIMAX_MODEL_H3, "2K"),
}
"""Which hosted model and resolution each tier means."""

PRICE_USD_PER_SECOND: Mapping[tuple[str, str], float] = {
    (MINIMAX_MODEL_H3, "768P"): 0.08,
    (MINIMAX_MODEL_H3, "2K"): 0.13,
    (MINIMAX_MODEL_H3_MAX, "480P"): 0.05,
    (MINIMAX_MODEL_H3_MAX, "768P"): 0.08,
}
PRICE_SOURCE = "https://platform.minimax.io/docs/pricing/overview"
PRICE_READ_ON = "2026-10-13"
"""Billed per second of output; audio input is free. A price table is a
snapshot -- the date is recorded in every estimate so a stale one is visible."""

CONFIRM_THRESHOLD_USD = 10.0
"""An estimated spend above this needs an explicit yes before submission."""

MINIMAX_TERMS: Mapping[str, str] = {
    "name": "MiniMax Open Platform Terms of Service",
    "url": "https://platform.minimax.io/docs/guides/terms-of-service",
    "effective_date": "2026-03-30",
    "provider": "Nanonoble Pte. Ltd.",
    "ownership_clause": (
        "Intellectual Property (Non-Transfer) and Indemnity Obligations, clause 3: "
        "\"As between you and us, and to the extent permitted by applicable laws, you "
        "retain your ownership rights in Client input and generated content. We may use "
        "the input and generated content to provide, maintain, develop, and improve our "
        "Services, comply with applicable law, enforce our terms and policies, and keep "
        "our Services safe.\""
    ),
}
"""Recorded verbatim into every provenance file an API run writes, so a
package carries the terms it relies on. Read from the platform's own page on
2026-10-13; re-check the effective date when the page changes."""

LOCAL_DRAFT_MODELS: Mapping[str, str] = {}
"""Local draft models this module can drive, name -> licence note. Empty on
purpose: a model enters here only once it has been chosen and its licence
(commercial use, no territorial exclusion) recorded beside it."""

AUDIO_BINDING_SENTENCE = (
    "The speech in the video is reference audio 1, kept exactly as given; the "
    "lips follow it precisely for the whole clip."
)
"""Appended to every API prompt. H3's prompt grammar names inputs by index;
this binds the audio clip to the mouth rather than leaving it a timbre hint."""

OUTPUT_FPS = 24
"""H3's native rate and the one every chunk in this project is cut at."""

SubprocessRunner = Callable[[Sequence[str]], Any]
Sleeper = Callable[[float], None]
Clock = Callable[[], float]
SpendConfirmer = Callable[["CostEstimate"], bool]


class LipSyncError(RuntimeError):
    """Base class for everything this module refuses or fails at."""


class LocalH3LockedError(LipSyncError):
    """``local_h3`` was asked for without an authorization file on disk."""


class LocalDraftUnavailableError(LipSyncError):
    """``local_draft`` was asked for before any local model was integrated."""


class SpendNotConfirmedError(LipSyncError):
    """The estimate crossed the threshold and nobody said yes."""


class MiniMaxApiError(LipSyncError):
    """The hosted API refused, failed, or answered with something unusable."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LipSyncSettings:
    """The backend/tier decision for one run, as read off the config."""

    backend: str = DEFAULT_BACKEND
    tier: str = DEFAULT_TIER
    h3_authorization_file: Path | None = None
    local_draft_model: str | None = None
    api_key_env: str = MINIMAX_API_KEY_ENV

    @property
    def minimax_model(self) -> str:
        return TIER_MODELS[self.tier][0]

    @property
    def minimax_resolution(self) -> str:
        return TIER_MODELS[self.tier][1]

    @property
    def uses_comfyui(self) -> bool:
        """Only the open-weights path drives a ComfyUI server."""
        return self.backend == BACKEND_LOCAL_H3


def validate_settings(settings: LipSyncSettings) -> None:
    """Refuse an impossible combination before anything else runs.

    * unknown backend or tier;
    * ``final`` on anything but ``minimax_api`` (a final render is 2K through
      the hosted API, by definition);
    * ``local_h3`` with no authorization file on disk (the lock);
    * ``local_draft`` with no integrated model.
    """
    if settings.backend not in BACKENDS:
        raise LipSyncError(
            f"lipsync_backend must be one of {BACKENDS}, got {settings.backend!r}"
        )
    if settings.tier not in TIERS:
        raise LipSyncError(f"quality_tier must be one of {TIERS}, got {settings.tier!r}")
    if settings.tier == TIER_FINAL and settings.backend != BACKEND_MINIMAX_API:
        raise LipSyncError(
            f"quality_tier = 'final' is only available through lipsync_backend = "
            f"'{BACKEND_MINIMAX_API}' (2K via the hosted API); got {settings.backend!r}. "
            f"A {settings.backend!r} shot can never be part of a final render"
        )
    if settings.backend == BACKEND_LOCAL_H3:
        path = settings.h3_authorization_file
        if path is None or not Path(path).is_file():
            raise LocalH3LockedError(
                "lipsync_backend = 'local_h3' is locked: running the MiniMax H3 open "
                "weights needs MiniMax's written authorization for this territory "
                "(https://platform.minimax.io/h3-license). Set h3_authorization_file to "
                "that document's path; "
                + ("it is unset" if path is None else f"{path} does not exist")
                + ". Until then use lipsync_backend = 'minimax_api' (the default)"
            )
    if settings.backend == BACKEND_LOCAL_DRAFT:
        name = settings.local_draft_model
        if not name or name not in LOCAL_DRAFT_MODELS:
            known = ", ".join(sorted(LOCAL_DRAFT_MODELS)) or "none integrated yet"
            raise LocalDraftUnavailableError(
                f"lipsync_backend = 'local_draft' needs local_draft_model to name an "
                f"integrated model (known: {known}); got {name!r}. A local draft model "
                f"must not be MiniMax H3 and must carry a licence allowing commercial use "
                f"in this territory -- pick one and integrate it before using this backend"
            )


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostEstimate:
    model: str
    resolution: str
    shots: int
    billable_seconds: int
    usd_per_second: float
    price_read_on: str = PRICE_READ_ON
    price_source: str = PRICE_SOURCE

    @property
    def usd(self) -> float:
        return round(self.billable_seconds * self.usd_per_second, 2)

    @property
    def needs_confirmation(self) -> bool:
        return self.usd > CONFIRM_THRESHOLD_USD

    def describe(self) -> str:
        return (
            f"{self.shots} shot(s), {self.billable_seconds} billable second(s) of "
            f"{self.model} at {self.resolution}: ${self.usd:.2f} "
            f"(${self.usd_per_second:.2f}/s, prices read {self.price_read_on} from "
            f"{self.price_source})"
        )


def billable_seconds(frames: int, *, model: str, fps: int = OUTPUT_FPS) -> int:
    """The integer ``duration`` the API bills for a shot of ``frames`` frames:
    rounded *up* to whole seconds, then clamped into the model's range."""
    limits = MINIMAX_MODEL_LIMITS[model]
    seconds = math.ceil(frames / fps - 1e-9)
    return int(min(max(seconds, limits["min_seconds"]), limits["max_seconds"]))


def estimate_cost(frame_counts: Iterable[int], *, model: str, resolution: str,
                  fps: int = OUTPUT_FPS) -> CostEstimate:
    key = (model, resolution)
    if key not in PRICE_USD_PER_SECOND:
        raise LipSyncError(
            f"no published price for {model} at {resolution}; known: "
            f"{sorted(PRICE_USD_PER_SECOND)}"
        )
    counts = list(frame_counts)
    total = sum(billable_seconds(n, model=model, fps=fps) for n in counts)
    return CostEstimate(model=model, resolution=resolution, shots=len(counts),
                        billable_seconds=total, usd_per_second=PRICE_USD_PER_SECOND[key])


def confirm_spend(estimate: CostEstimate, *, yes: bool,
                  confirmer: SpendConfirmer | None = None) -> None:
    """Log the estimate; above the threshold, require ``yes`` (or a confirmer
    that returns True) or refuse before a single request is made."""
    logger.info("Estimated MiniMax API cost: %s", estimate.describe())
    if not estimate.needs_confirmation:
        return
    if yes or (confirmer is not None and confirmer(estimate)):
        logger.info("Spend of $%.2f confirmed", estimate.usd)
        return
    raise SpendNotConfirmedError(
        f"estimated cost ${estimate.usd:.2f} is over the ${CONFIRM_THRESHOLD_USD:.0f} "
        f"confirmation threshold ({estimate.describe()}); re-run with --yes to proceed"
    )


# --------------------------------------------------------------------------- #
# Hosted API client
# --------------------------------------------------------------------------- #


def read_api_key(env: str = MINIMAX_API_KEY_ENV, *,
                 environ: Mapping[str, str] | None = None) -> str:
    """The key, from the environment and nowhere else. Never logged."""
    value = (environ if environ is not None else os.environ).get(env, "").strip()
    if not value:
        raise MiniMaxApiError(
            f"no MiniMax API key: set the {env} environment variable (Account Management "
            f"> API Keys on https://platform.minimax.io). The key is read from the "
            f"environment only and is never written to disk or logs"
        )
    return value


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if secret else text


def ratio_for(width: int, height: int) -> str:
    """The API's nearest aspect-ratio label for a frame size."""
    aspect = width / height
    return min(MINIMAX_RATIOS, key=lambda name: abs(MINIMAX_RATIOS[name] - aspect))


def build_reference_content(*, prompt: str, image_url: str, audio_url: str,
                            extra_image_urls: Sequence[str] = ()) -> list[dict]:
    """The ``content`` array for a reference-to-video request: one text item,
    the reference image(s), and the reference audio."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in (image_url, *extra_image_urls):
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "reference_image"})
    content.append({"type": "audio_url", "audio_url": {"url": audio_url},
                    "role": "reference_audio"})
    return content


def build_pad_audio_args(src: Path, dst: Path, *, seconds: float) -> list[str]:
    """ffmpeg argv: ``src`` cut/padded with silence to exactly ``seconds``,
    mono 48 kHz 16-bit WAV, the shape the API accepts and the length the
    output will have."""
    return [
        "ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(src),
        "-af", f"apad=whole_dur={seconds:.6f}", "-t", f"{seconds:.6f}",
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(dst),
    ]


def build_conform_args(api_video: Path, audio: Path, dst: Path, *, frames: int,
                       fps: int = OUTPUT_FPS) -> list[str]:
    """ffmpeg argv: the API's mp4 conformed to exactly ``frames`` frames at
    ``fps`` (whatever rate the service returned), with the conditioning audio
    muxed in place of the service's own track -- the sync reference is the
    audio the shot was asked to follow, not a re-synthesis of it. Short
    outputs are extended by cloning the last frame; long ones are cut."""
    return [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-i", str(api_video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"fps={fps},tpad=stop_mode=clone:stop={frames}",
        "-frames:v", str(frames),
        "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{frames / fps:.6f}",
        "-movflags", "+faststart", str(dst),
    ]


def _default_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), check=True, capture_output=True)


@dataclass
class ShotRequest:
    """One talking shot to ask the hosted API for."""

    shot_id: str
    image: Path
    audio: Path
    prompt: str
    frames: int
    extra_images: tuple[Path, ...] = ()
    width: int | None = None
    height: int | None = None
    fps: int = OUTPUT_FPS

    @property
    def ratio(self) -> str:
        if self.width and self.height:
            return ratio_for(self.width, self.height)
        return "adaptive"


class MiniMaxVideoClient:
    """A thin, injectable client for the video-generation API: upload inputs,
    create a task, poll it, download the result, conform it, record it."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: Any = None,
        base_url: str = MINIMAX_API_BASE_URL,
        sleeper: Sleeper = time.sleep,
        clock: Clock = time.monotonic,
        runner: SubprocessRunner = _default_runner,
        poll_interval_seconds: float = 10.0,
        max_wait_seconds: float = 1800.0,
        http_timeout: float | None = 120.0,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else read_api_key(environ=environ)
        self.session = session if session is not None else requests.Session()
        self.base_url = base_url.rstrip("/")
        self.sleeper = sleeper
        self.clock = clock
        self.runner = runner
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_wait_seconds
        self.http_timeout = http_timeout

    # -- transport --------------------------------------------------------- #

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _fail(self, what: str, response: Any) -> MiniMaxApiError:
        status = getattr(response, "status_code", "?")
        try:
            body = response.text
        except Exception:  # noqa: BLE001 - a fake may not have .text
            body = "<no body>"
        body = _redact(str(body)[:800], self._api_key)
        logger.error("MiniMax API %s failed: HTTP %s %s", what, status, body)
        return MiniMaxApiError(f"MiniMax API {what} failed: HTTP {status} {body}")

    def _json(self, what: str, response: Any) -> dict:
        if getattr(response, "status_code", 200) >= 400:
            raise self._fail(what, response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise MiniMaxApiError(f"MiniMax API {what} returned a non-JSON body") from exc
        if not isinstance(payload, dict):
            raise MiniMaxApiError(f"MiniMax API {what} returned {type(payload).__name__}")
        base = payload.get("base_resp")
        if isinstance(base, dict) and base.get("status_code") not in (None, 0):
            logger.error("MiniMax API %s refused: %s", what, base)
            raise MiniMaxApiError(f"MiniMax API {what} refused: {base}")
        return payload

    def upload(self, path: Path, *, purpose: str = MINIMAX_UPLOAD_PURPOSE) -> str:
        """``POST /v1/files/upload``; returns the ``mm_file://`` URL to use in a
        request. Uploaded inputs are kept by the service for 7 days."""
        path = Path(path)
        with open(path, "rb") as handle:
            try:
                response = self.session.post(
                    f"{self.base_url}{MINIMAX_UPLOAD_PATH}", headers=self._headers(),
                    data={"purpose": purpose}, files={"file": (path.name, handle)},
                    timeout=self.http_timeout,
                )
            except requests.RequestException as exc:
                raise MiniMaxApiError(f"upload of {path.name} failed: {exc}") from exc
        payload = self._json(f"upload {path.name}", response)
        file_id = (payload.get("file") or {}).get("file_id")
        if file_id in (None, ""):
            raise MiniMaxApiError(f"upload of {path.name} returned no file_id: {payload}")
        logger.info("Uploaded %s to MiniMax as file %s", path.name, file_id)
        return f"mm_file://{file_id}"

    def create_task(self, *, model: str, content: list[dict], resolution: str,
                    duration: int, ratio: str | None = None) -> str:
        body: dict[str, Any] = {"model": model, "content": content,
                                "resolution": resolution, "duration": duration}
        if ratio:
            body["ratio"] = ratio
        try:
            response = self.session.post(
                f"{self.base_url}{MINIMAX_CREATE_PATH}", headers=self._headers(),
                json=body, timeout=self.http_timeout,
            )
        except requests.RequestException as exc:
            raise MiniMaxApiError(f"create task failed: {exc}") from exc
        payload = self._json("create task", response)
        task_id = payload.get("task_id")
        if not task_id:
            raise MiniMaxApiError(f"create task returned no task_id: {payload}")
        logger.info("MiniMax task %s created: %s %s %ds ratio=%s",
                    task_id, model, resolution, duration, ratio or "adaptive")
        return str(task_id)

    def query(self, task_id: str) -> dict:
        try:
            response = self.session.get(
                f"{self.base_url}{MINIMAX_QUERY_PATH.format(task_id=task_id)}",
                headers=self._headers(), timeout=self.http_timeout,
            )
        except requests.RequestException as exc:
            raise MiniMaxApiError(f"query of task {task_id} failed: {exc}") from exc
        payload = self._json(f"query {task_id}", response)
        task = payload.get("task")
        if not isinstance(task, dict):
            raise MiniMaxApiError(f"query of task {task_id} returned no task object: {payload}")
        return task

    def wait(self, task_id: str) -> dict:
        """Poll until the task succeeds; raise on failure/cancellation or
        after ``max_wait_seconds``. Polling is the API's own contract here
        (a callback URL needs a public endpoint), with the interval injectable."""
        started = self.clock()
        while True:
            task = self.query(task_id)
            status = task.get("status")
            if status == "succeeded":
                return task
            if status in ("failed", "cancelled"):
                error = task.get("error") or {}
                logger.error("MiniMax task %s %s: %s", task_id, status, error)
                raise MiniMaxApiError(f"task {task_id} {status}: {error}")
            if self.clock() - started > self.max_wait_seconds:
                raise MiniMaxApiError(
                    f"task {task_id} still {status!r} after {self.max_wait_seconds:.0f} s"
                )
            self.sleeper(self.poll_interval_seconds)

    def download(self, url: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self.session.get(url, stream=True, timeout=self.http_timeout)
        except requests.RequestException as exc:
            raise MiniMaxApiError(f"download failed: {exc}") from exc
        if getattr(response, "status_code", 200) >= 400:
            raise self._fail("download", response)
        with open(dest, "wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                if block:
                    handle.write(block)
        return dest

    # -- one shot, end to end ---------------------------------------------- #

    def generate_shot(
        self,
        request: ShotRequest,
        *,
        model: str,
        resolution: str,
        tier: str,
        out_path: Path,
        provenance_path: Path | None = None,
    ) -> ShotProvenance:
        """Pad the audio to the billed duration, upload, generate, download,
        conform to the requested frame count, hash everything, and record it.

        The raw API file is kept beside the conformed one (``<stem>.api.mp4``)
        so the hash in the provenance record is of bytes the service produced.
        """
        limits = MINIMAX_MODEL_LIMITS[model]
        if resolution not in limits["resolutions"]:
            raise LipSyncError(f"{model} does not offer {resolution}; "
                               f"offers {limits['resolutions']}")
        duration = billable_seconds(request.frames, model=model, fps=request.fps)
        wanted = request.frames / request.fps
        if wanted > duration + 1e-6:
            raise LipSyncError(
                f"shot {request.shot_id}: {request.frames} frames at {request.fps} fps is "
                f"{wanted:.3f} s, over the {model} maximum of {limits['max_seconds']} s"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = out_path.with_name(out_path.stem + ".api.mp4")
        requested_at = utc_now_iso()
        started = self.clock()

        with tempfile.TemporaryDirectory(prefix="mvm-lipsync-") as tmp:
            padded = Path(tmp) / f"{request.shot_id}_audio.wav"
            self.runner(build_pad_audio_args(request.audio, padded, seconds=float(duration)))
            image_url = self.upload(request.image)
            extra_urls = [self.upload(p) for p in request.extra_images]
            audio_url = self.upload(padded)
            prompt = request.prompt.rstrip() + " " + AUDIO_BINDING_SENTENCE
            content = build_reference_content(prompt=prompt, image_url=image_url,
                                              audio_url=audio_url,
                                              extra_image_urls=extra_urls)
            task_id = self.create_task(model=model, content=content, resolution=resolution,
                                       duration=duration, ratio=request.ratio)
            task = self.wait(task_id)
            url = (task.get("content") or {}).get("url")
            if not url:
                raise MiniMaxApiError(f"task {task_id} succeeded with no content.url: {task}")
            self.download(url, raw_path)
            self.runner(build_conform_args(raw_path, padded, out_path,
                                           frames=request.frames, fps=request.fps))
            inputs = {
                "reference_image": sha256_file(request.image),
                "reference_audio": sha256_file(request.audio),
                "reference_audio_padded": sha256_file(padded),
            }
            for i, extra in enumerate(request.extra_images, start=1):
                inputs[f"reference_image_{i}"] = sha256_file(extra)

        usage = task.get("usage") or {}
        shot = ShotProvenance(
            shot_id=request.shot_id, backend=BACKEND_MINIMAX_API, model=model, tier=tier,
            resolution=resolution, frames=request.frames, fps=request.fps,
            duration_seconds=float(duration), task_id=task_id,
            requested_at_utc=requested_at, completed_at_utc=utc_now_iso(),
            inputs=inputs, output_file=str(out_path), output_sha256=sha256_file(out_path),
            cost_usd=round(duration * PRICE_USD_PER_SECOND.get((model, resolution), 0.0), 2),
            notes={
                "api_output_file": str(raw_path), "api_output_sha256": sha256_file(raw_path),
                "ratio": request.ratio, "prompt": prompt, "usage": usage,
                "wall_seconds": round(self.clock() - started, 1),
            },
        )
        if provenance_path is not None:
            record_shot(provenance_path, shot, terms=MINIMAX_TERMS)
        logger.info("Shot %s: task %s -> %s (%d frames, $%.2f)",
                    request.shot_id, task_id, out_path, request.frames, shot.cost_usd or 0.0)
        return shot


# --------------------------------------------------------------------------- #
# Pipeline seams: an AssetStager and an ExecutionClient for the hosted API
# --------------------------------------------------------------------------- #


class LocalAssetStager:
    """``contracts.AssetStager`` for a backend that reads files itself: the
    "staged filename" is the file's own absolute path. Mirrors
    ``ComfyUIAssetStager.stage_chunk``'s shape so the workflow provider and
    mutator run unchanged; the mutated graph then carries paths where the
    ComfyUI path carries server filenames, and :class:`MiniMaxExecutionClient`
    reads them back out of the graph."""

    def upload_image(self, path: Path) -> str:
        return self._stage(path)

    def upload_audio(self, path: Path) -> str:
        return self._stage(path)

    @staticmethod
    def _stage(path: Path) -> str:
        path = Path(path)
        if not path.is_file():
            raise LipSyncError(f"asset not found: {path}")
        return str(path.resolve())

    def stage_chunk(self, prompt: ExpandedPrompt, chunk: AudioChunk) -> StagedAssets:
        if prompt.chunk_id != chunk.chunk_id:
            raise LipSyncError(
                f"chunk id mismatch: prompt.chunk_id={prompt.chunk_id} != "
                f"chunk.chunk_id={chunk.chunk_id}"
            )
        return StagedAssets(
            chunk_id=chunk.chunk_id,
            image_filename=self.upload_image(prompt.image_ref),
            audio_filename=self.upload_audio(chunk.audio_file),
            image_filenames=tuple(self.upload_image(p) for p in prompt.image_refs),
        )


def _read_shot_request(workflow: Workflow, chunk_id: int, *, fps: int) -> ShotRequest:
    """Read what the mutated graph asked H3 for -- by node class and wiring,
    never by node id -- and turn it into an API request."""
    node_id, h3 = workflow_graph._find_h3_conditioning_node(workflow)
    if h3.get("class_type") != workflow_graph.CLASS_TYPE_H3_REFERENCE_TO_VIDEO:
        raise LipSyncError(
            f"the hosted API backend drives reference-to-video only; the workflow's H3 node "
            f"is {h3.get('class_type')!r} (i2v_continuity is not available on this backend)"
        )
    inputs = h3.get("inputs", {})
    prompt = str(inputs.get("prompt", ""))
    try:
        frames = int(inputs["length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LipSyncError(f"H3 node {node_id} has no usable 'length': {inputs}") from exc
    width, height = workflow_graph.read_render_dimensions(workflow)

    def upstream_file(input_name: str, loader_class: str, key: str) -> Path:
        _, node = workflow_graph.resolve_upstream_node(
            workflow, workflow_graph.CLASS_TYPE_H3_REFERENCE_TO_VIDEO, input_name
        )
        if node.get("class_type") != loader_class:
            raise LipSyncError(f"{input_name} is fed by {node.get('class_type')!r}, "
                               f"expected {loader_class!r}")
        return Path(str(node["inputs"][key]))

    image = upstream_file("ref_images.ref_image_0", workflow_graph.CLASS_TYPE_IMAGE_LOAD, "image")
    audio = upstream_file("ref_audios.ref_audio_0", workflow_graph.CLASS_TYPE_AUDIO_LOAD, "audio")
    extras: list[Path] = []
    for i in range(1, 10):
        name = f"ref_images.ref_image_{i}"
        if name not in inputs:
            break
        extras.append(upstream_file(name, workflow_graph.CLASS_TYPE_IMAGE_LOAD, "image"))
    return ShotRequest(shot_id=f"chunk_{chunk_id:04d}", image=image, audio=audio,
                       prompt=prompt, frames=frames, extra_images=tuple(extras),
                       width=width, height=height, fps=fps)


class MiniMaxExecutionClient:
    """``contracts.ExecutionClient`` over the hosted API. ``interrupt`` and
    ``free`` are no-ops: there is no card to release."""

    def __init__(self, client: MiniMaxVideoClient, *, settings: LipSyncSettings,
                 clock: Clock = time.monotonic, fps: int = OUTPUT_FPS) -> None:
        self.client = client
        self.settings = settings
        self.clock = clock
        self.fps = fps

    def execute(self, workflow: Workflow, chunk_id: int, output_dir: Path) -> ChunkResult:
        output_dir = Path(output_dir)
        request = _read_shot_request(workflow, chunk_id, fps=self.fps)
        started = self.clock()
        shot = self.client.generate_shot(
            request, model=self.settings.minimax_model,
            resolution=self.settings.minimax_resolution, tier=self.settings.tier,
            out_path=output_dir / f"chunk_{chunk_id:04d}.mp4",
            provenance_path=output_dir / PROVENANCE_FILENAME,
        )
        return ChunkResult(
            chunk_id=chunk_id, status=ChunkStatus.RENDERED,
            video_file=Path(shot.output_file or ""), prompt_id=shot.task_id, attempts=1,
            render_seconds=self.clock() - started,
        )

    def interrupt(self) -> None:  # pragma: no cover - nothing to interrupt
        logger.info("MiniMax API backend: interrupt is a no-op")

    def free(self, *, unload_models: bool = True) -> None:  # pragma: no cover
        logger.info("MiniMax API backend: free is a no-op (no local card)")


@dataclass(frozen=True)
class LipSyncBackend:
    """What ``run_pipeline`` needs from a backend: the stager and client, and
    a label for the fingerprint's stack tier so a resumed run can tell a
    hosted chunk from a local one."""

    settings: LipSyncSettings
    stager: Any
    execution_client: Any
    stack_label: str

    @property
    def uses_comfyui(self) -> bool:
        return self.settings.uses_comfyui


def build_minimax_backend(settings: LipSyncSettings, *, session: Any = None,
                          sleeper: Sleeper = time.sleep, clock: Clock = time.monotonic,
                          runner: SubprocessRunner = _default_runner,
                          environ: Mapping[str, str] | None = None) -> LipSyncBackend:
    validate_settings(settings)
    if settings.backend != BACKEND_MINIMAX_API:
        raise LipSyncError(f"not the hosted backend: {settings.backend!r}")
    client = MiniMaxVideoClient(session=session, sleeper=sleeper, clock=clock, runner=runner,
                                environ=environ)
    return LipSyncBackend(
        settings=settings, stager=LocalAssetStager(),
        execution_client=MiniMaxExecutionClient(client, settings=settings, clock=clock),
        stack_label=f"{BACKEND_MINIMAX_API}:{settings.minimax_model}@{settings.minimax_resolution}",
    )


__all__ = [
    "AUDIO_BINDING_SENTENCE", "CONFIRM_THRESHOLD_USD", "CostEstimate", "DEFAULT_BACKEND",
    "DEFAULT_TIER", "LOCAL_DRAFT_MODELS", "LipSyncBackend", "LipSyncError",
    "LipSyncSettings", "LocalAssetStager", "LocalDraftUnavailableError",
    "LocalH3LockedError", "MINIMAX_API_KEY_ENV", "MINIMAX_TERMS", "MiniMaxApiError",
    "MiniMaxExecutionClient", "MiniMaxVideoClient", "PRICE_USD_PER_SECOND",
    "ShotRequest", "SpendNotConfirmedError", "TIER_MODELS", "billable_seconds",
    "build_conform_args", "build_minimax_backend", "build_pad_audio_args",
    "build_reference_content", "confirm_spend", "estimate_cost", "ratio_for",
    "read_api_key", "validate_settings",
]
