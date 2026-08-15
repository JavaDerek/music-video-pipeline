"""Fake ComfyUI HTTP surface for offline tests (issue #16).

:class:`FakeComfyUISession` duck-types the surface of ``requests.Session`` that
the Stage 3 asset stager (issue #7), Stage 4b execution client (issue #9), and
the resilience state machine (issue #10) drive: ``.post(url, ...)`` and
``.get(url, ...)``, both returning a :class:`FakeResponse` shaped like
``requests.Response`` (``.status_code``, ``.json()``, ``.content``, ``.text``,
``.raise_for_status()``, ``.iter_content()``).

It fakes ``POST /upload/image`` (the universal image+audio ingest point),
``POST /prompt``, ``GET /history/{prompt_id}``, ``GET /view``,
``POST /interrupt``, ``POST /free``, and ``GET /system_stats``.

Every request is recorded (method, url, kwargs, and derived metadata) on
``.requests`` so tests can assert call sequences -- including that a client's
upload cache actually prevented a redundant ``/upload/image`` call, which
issue #7 requires. The fake does **not** dedupe uploads itself: like real
ComfyUI, re-uploading the same original filename without ``overwrite`` gets a
disambiguated server filename (``photo.png`` -> ``photo (1).png``), so a client
that fails to cache is observable in the response, not silently papered over.

Nothing here talks to a real socket or filesystem.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import requests

logger = logging.getLogger(__name__)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class FakeResponse:
    """Duck-types ``requests.Response`` for the fields the clients use."""

    def __init__(
        self,
        status_code: int,
        *,
        json_data: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        if content is not None:
            self.content = content
        elif json_data is not None:
            self.content = json.dumps(json_data).encode("utf-8")
        else:
            self.content = b""
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Error for fake ComfyUI response", response=self  # type: ignore[arg-type]
            )

    def iter_content(self, chunk_size: int = 8192):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]


@dataclass
class RecordedRequest:
    """One call made against :class:`FakeComfyUISession`."""

    method: str
    url: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadRecord:
    """One successful upload, for easy dedup/collision assertions."""

    original_filename: str
    server_filename: str
    upload_type: str
    subfolder: str
    size: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_fake_png_bytes(width: int, height: int, *, padding: int = 0) -> bytes:
    """Minimal PNG-header bytes carrying real width/height in the IHDR chunk.

    Not a decodable image -- just enough for this module's megapixel sniffing
    to read real dimensions in tests, without pulling in a Pillow dependency
    the project doesn't otherwise need.
    """
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_len = struct.pack(">I", len(ihdr_data))
    body = ihdr_len + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"  # fake CRC
    return _PNG_SIGNATURE + body + (b"\x00" * padding)


def _sniff_megapixels(content: bytes) -> float | None:
    """Best-effort dimension sniff. Returns ``None`` when not a recognized format."""
    if content[:8] == _PNG_SIGNATURE and len(content) >= 24:
        width, height = struct.unpack(">II", content[16:24])
        return (width * height) / 1_000_000
    return None


def _save_video_prefix(workflow: dict[str, Any]) -> str | None:
    """The ``filename_prefix`` of a submitted workflow's ``SaveVideo`` node.

    Located by ``class_type`` rather than node id -- the project's hard
    invariant is that user edits on the ComfyUI canvas renumber ids, so a
    fixture that keyed off ``"16"`` would pass here and lie about the real
    server. Returns ``None`` when the workflow has no ``SaveVideo`` node or no
    usable prefix.
    """
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "SaveVideo":
            continue
        prefix = (node.get("inputs") or {}).get("filename_prefix")
        if isinstance(prefix, str) and prefix:
            return prefix
    return None


def _extract_file(value: Any) -> tuple[str, bytes]:
    if isinstance(value, tuple):
        filename = value[0]
        payload = value[1]
    else:
        payload = value
        filename = os.path.basename(getattr(value, "name", "upload.bin"))
    if hasattr(payload, "read"):
        content = payload.read()
    elif isinstance(payload, bytearray):
        content = bytes(payload)
    elif isinstance(payload, bytes):
        content = payload
    elif isinstance(payload, str):
        content = payload.encode("utf-8")
    else:
        raise TypeError(f"FakeComfyUISession: unsupported file payload type {type(payload)!r}")
    return filename, content


def default_system_stats(
    *, vram_total: int = 25_757_220_864, vram_free: int = 24_000_000_000
) -> dict[str, Any]:
    """A realistic ``/system_stats`` body for an RTX 4090 (~24 GB VRAM)."""
    return {
        "system": {
            "os": "posix",
            "python_version": "3.12.8",
            "comfyui_version": "0.30.2",
        },
        "devices": [
            {
                "name": "cuda:0 NVIDIA GeForce RTX 4090",
                "type": "cuda",
                "index": 0,
                "vram_total": vram_total,
                "vram_free": vram_free,
                "torch_vram_total": vram_total,
                "torch_vram_free": vram_free,
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


class FakeComfyUISession:
    """Fake of ``requests.Session`` bound to a fake ComfyUI backend."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8188",
        max_upload_bytes: int = 50 * 1024 * 1024,
        max_megapixels: float = 64.0,
        system_stats: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_upload_bytes = max_upload_bytes
        self.max_megapixels = max_megapixels

        self.requests: list[RecordedRequest] = []
        self.uploads: list[UploadRecord] = []
        self.submitted_prompts: list[dict[str, Any]] = []
        self.interrupt_calls = 0
        self.free_calls: list[dict[str, Any]] = []

        self._filenames_by_bucket: dict[tuple[str, str], set[str]] = {}
        self._stored_outputs: dict[tuple[str, str, str], bytes] = {}
        self._view_overrides: dict[tuple[str, str, str], str] = {}
        self._history: dict[str, dict[str, Any]] = {}
        self._node_errors_queue: deque[dict[str, Any]] = deque()
        self._prompt_counter = 0
        self._system_stats = system_stats or default_system_stats()
        self._auto_history: dict[str, Any] | None = None

    # -- requests.Session surface -------------------------------------------- #
    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        path = self._path(url)
        if path == "/upload/image":
            return self._handle_upload(url, **kwargs)
        self._record("POST", url, kwargs)
        if path == "/prompt":
            return self._handle_prompt(**kwargs)
        if path == "/interrupt":
            return self._handle_interrupt(**kwargs)
        if path == "/free":
            return self._handle_free(**kwargs)
        logger.error("FakeComfyUISession: unhandled POST path %s", path)
        return FakeResponse(404, json_data={"error": f"unhandled path {path}"})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self._record("GET", url, kwargs)
        path = self._path(url)
        if path.startswith("/history/"):
            return self._handle_history(path)
        if path == "/view":
            return self._handle_view(url, kwargs.get("params"))
        if path == "/system_stats":
            return self._handle_system_stats()
        logger.error("FakeComfyUISession: unhandled GET path %s", path)
        return FakeResponse(404, json_data={"error": f"unhandled path {path}"})

    # -- seeding / configuration ------------------------------------------------ #
    def seed_history_success(
        self,
        prompt_id: str,
        *,
        video_filename: str,
        video_bytes: bytes = b"\x00\x00\x00\x18ftypisommock-mp4-bytes",
        node_id: str = "9",
        subfolder: str = "",
        type_: str = "output",
        output_key: str = "videos",
    ) -> None:
        """Seed a realistic successful ``/history/{prompt_id}`` entry plus the
        bytes ``/view`` will serve for it."""
        self._history[prompt_id] = {
            "prompt": [self._prompt_counter, prompt_id, {}, {}, []],
            "outputs": {
                node_id: {
                    output_key: [
                        {
                            "filename": video_filename,
                            "subfolder": subfolder,
                            "type": type_,
                            "format": "video/h264-mp4",
                        }
                    ]
                }
            },
            "status": {"status_str": "success", "completed": True, "messages": []},
        }
        self._stored_outputs[(video_filename, subfolder, type_)] = video_bytes

    def seed_history_without_video(self, prompt_id: str, *, node_id: str = "9") -> None:
        """Seed a completed history entry whose outputs contain no MP4 at all
        (e.g. a workflow whose video-save node never ran)."""
        self._history[prompt_id] = {
            "prompt": [self._prompt_counter, prompt_id, {}, {}, []],
            "outputs": {
                node_id: {
                    "images": [
                        {"filename": "preview_00001_.png", "subfolder": "", "type": "temp"}
                    ]
                }
            },
            "status": {"status_str": "success", "completed": True, "messages": []},
        }

    def corrupt_view_output(
        self, filename: str, subfolder: str = "", type_: str = "output", *, mode: str = "truncate"
    ) -> None:
        """Simulate the disk-full failure mode noted on issue #10: ``mode`` is
        one of ``"truncate"`` (partial bytes), ``"empty"`` (zero bytes), or
        ``"error"`` (HTTP 500)."""
        if mode not in ("truncate", "empty", "error"):
            raise ValueError(f"unknown corruption mode: {mode!r}")
        self._view_overrides[(filename, subfolder, type_)] = mode

    def enable_auto_history(
        self,
        *,
        video_bytes: bytes = b"\x00\x00\x00\x18ftypisommock-mp4-bytes",
        node_id: str = "9",
        suffix: str = "_00001.mp4",
        fallback_prefix: str = "ComfyUI",
    ) -> None:
        """Auto-seed a successful ``/history`` entry for every accepted ``/prompt``.

        Additive and off by default -- a session that never calls this behaves
        exactly as before, and :meth:`seed_history_success` keeps working
        alongside it (an explicitly seeded ``prompt_id`` is never overwritten).

        Without this, a test that submits several prompts has to know each
        generated ``prompt_id`` in advance and pre-seed it, which couples the
        test to this class's id-numbering. A multi-chunk run (issue #28 chains
        chunk N off chunk N-1's *rendered file*, so it genuinely needs several
        real submissions in one test) is the case that makes that painful.

        The output filename mirrors real ComfyUI: it is derived from the
        submitted workflow's ``SaveVideo.filename_prefix`` -- located by
        ``class_type``, never by node id, since the orchestrator's own
        invariant is that node ids renumber freely -- so a workflow carrying
        ``mvm_chunk_0003`` produces ``mvm_chunk_0003_00001.mp4``, which is what
        the execution client keys chunk outputs on.
        """
        self._auto_history = {
            "video_bytes": video_bytes,
            "node_id": node_id,
            "suffix": suffix,
            "fallback_prefix": fallback_prefix,
        }

    def queue_node_errors(self, node_errors: dict[str, Any]) -> None:
        """Make the *next* ``/prompt`` submission come back with validation
        errors instead of a ``prompt_id`` (ComfyUI's graph-validation-failed shape)."""
        self._node_errors_queue.append(node_errors)

    def set_vram_free(self, vram_free: int, *, vram_total: int | None = None) -> None:
        stats = default_system_stats(
            vram_total=vram_total or self._system_stats["devices"][0]["vram_total"],
            vram_free=vram_free,
        )
        self._system_stats = stats

    # -- upload ---------------------------------------------------------------- #
    def _handle_upload(
        self,
        url: str,
        *,
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> FakeResponse:
        files = files or {}
        data = data or {}
        if not files:
            self._record("POST", url, {"files": files, "data": data}, extra={"error": "no file"})
            return FakeResponse(400, json_data={"error": "no file provided"})

        field_name, field_value = next(iter(files.items()))
        filename, content = _extract_file(field_value)
        size = len(content)
        self._record(
            "POST",
            url,
            {"data": data},
            extra={"field": field_name, "filename": filename, "size": size},
        )

        if size > self.max_upload_bytes:
            logger.info(
                "FakeComfyUISession: rejecting upload %s (%d bytes > limit %d)",
                filename,
                size,
                self.max_upload_bytes,
            )
            return FakeResponse(400, json_data={"error": f"file too large ({size} bytes)"})

        megapixels = _sniff_megapixels(content)
        if megapixels is not None and megapixels > self.max_megapixels:
            logger.info(
                "FakeComfyUISession: rejecting upload %s (%.1f MP > limit %.1f MP)",
                filename,
                megapixels,
                self.max_megapixels,
            )
            return FakeResponse(400, json_data={"error": f"image too large ({megapixels:.1f} MP)"})

        upload_type = str(data.get("type", "input"))
        subfolder = str(data.get("subfolder", ""))
        overwrite = str(data.get("overwrite", "false")).lower() in ("1", "true", "yes")

        server_filename = self._finalize_filename(
            filename, (upload_type, subfolder), overwrite=overwrite
        )
        self._stored_outputs[(server_filename, subfolder, upload_type)] = content
        self.uploads.append(
            UploadRecord(
                original_filename=filename,
                server_filename=server_filename,
                upload_type=upload_type,
                subfolder=subfolder,
                size=size,
            )
        )
        return FakeResponse(
            200, json_data={"name": server_filename, "subfolder": subfolder, "type": upload_type}
        )

    def _finalize_filename(self, original: str, bucket: tuple[str, str], *, overwrite: bool) -> str:
        used = self._filenames_by_bucket.setdefault(bucket, set())
        if overwrite or original not in used:
            used.add(original)
            return original
        stem, ext = os.path.splitext(original)
        n = 1
        candidate = f"{stem} ({n}){ext}"
        while candidate in used:
            n += 1
            candidate = f"{stem} ({n}){ext}"
        used.add(candidate)
        return candidate

    # -- prompt ------------------------------------------------------------------ #
    def _handle_prompt(
        self, *, json: dict[str, Any] | None = None, **_ignored: Any
    ) -> FakeResponse:
        payload = json or {}
        workflow = payload.get("prompt", {})
        client_id = payload.get("client_id")

        if self._node_errors_queue:
            node_errors = self._node_errors_queue.popleft()
            self.submitted_prompts.append(
                {"client_id": client_id, "workflow": workflow, "rejected": True}
            )
            logger.info("FakeComfyUISession: rejecting /prompt submission with node_errors")
            return FakeResponse(
                200,
                json_data={
                    "prompt_id": None,
                    "number": self._prompt_counter,
                    "node_errors": node_errors,
                },
            )

        self._prompt_counter += 1
        prompt_id = f"prompt-{self._prompt_counter:04d}"
        self.submitted_prompts.append(
            {"prompt_id": prompt_id, "client_id": client_id, "workflow": workflow}
        )
        self._maybe_auto_seed_history(prompt_id, workflow)
        return FakeResponse(
            200,
            json_data={
                "prompt_id": prompt_id,
                "number": self._prompt_counter,
                "node_errors": {},
            },
        )

    def _maybe_auto_seed_history(self, prompt_id: str, workflow: dict[str, Any]) -> None:
        """Seed a success entry for ``prompt_id`` when auto-history is on.

        Never overwrites an entry a test seeded explicitly, so the two seeding
        styles can coexist in one session.
        """
        settings = self._auto_history
        if settings is None or prompt_id in self._history:
            return
        prefix = _save_video_prefix(workflow) or settings["fallback_prefix"]
        self.seed_history_success(
            prompt_id,
            video_filename=f"{prefix}{settings['suffix']}",
            video_bytes=settings["video_bytes"],
            node_id=settings["node_id"],
        )

    # -- interrupt / free ---------------------------------------------------------- #
    def _handle_interrupt(self, **_ignored: Any) -> FakeResponse:
        self.interrupt_calls += 1
        return FakeResponse(200, json_data={})

    def _handle_free(self, *, json: dict[str, Any] | None = None, **_ignored: Any) -> FakeResponse:
        self.free_calls.append(json or {})
        return FakeResponse(200, json_data={})

    # -- history / view / system_stats ---------------------------------------------- #
    def _handle_history(self, path: str) -> FakeResponse:
        prompt_id = path[len("/history/") :].strip("/")
        entry = self._history.get(prompt_id)
        if entry is None:
            return FakeResponse(200, json_data={})
        return FakeResponse(200, json_data={prompt_id: entry})

    def _handle_view(self, url: str, params: dict[str, Any] | None) -> FakeResponse:
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        if params:
            query.update({k: str(v) for k, v in params.items()})
        key = (query.get("filename", ""), query.get("subfolder", ""), query.get("type", "output"))
        content = self._stored_outputs.get(key)
        mode = self._view_overrides.get(key)

        if content is None:
            return FakeResponse(404, json_data={"error": "file not found"})
        if mode == "error":
            logger.info("FakeComfyUISession: simulating disk-full on /view for %s", key)
            return FakeResponse(500, json_data={"error": "disk full"})
        if mode == "empty":
            return FakeResponse(200, content=b"")
        if mode == "truncate":
            return FakeResponse(200, content=content[: max(1, len(content) // 4)])
        return FakeResponse(200, content=content)

    def _handle_system_stats(self) -> FakeResponse:
        return FakeResponse(200, json_data=self._system_stats)

    # -- bookkeeping ------------------------------------------------------------------ #
    def _path(self, url: str) -> str:
        if url.startswith(self.base_url):
            url = url[len(self.base_url) :]
        return url.split("?", 1)[0]

    def _record(
        self, method: str, url: str, kwargs: dict[str, Any], extra: dict[str, Any] | None = None
    ) -> None:
        self.requests.append(
            RecordedRequest(method=method, url=url, kwargs=kwargs, extra=extra or {})
        )
        logger.debug("FakeComfyUISession: recorded %s %s", method, url)


__all__ = [
    "FakeComfyUISession",
    "FakeResponse",
    "RecordedRequest",
    "UploadRecord",
    "default_system_stats",
    "make_fake_png_bytes",
]
