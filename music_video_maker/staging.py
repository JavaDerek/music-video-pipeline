"""Stage 3: ComfyUI asset staging client (issue #7).

Uploads cast reference images and per-chunk audio stems to a running ComfyUI
instance ahead of Stage 4 execution. Despite the endpoint's name,
``POST /upload/image`` is ComfyUI's universal ingest point for *both* images
and audio -- there is no separate ``/upload/audio``.

Two things this module owns that the rest of the pipeline depends on:

1. **De-dupe.** The same cast reference photo is reused by every chunk that
   character appears in. ComfyUI does not dedupe uploads server-side -- a
   second upload of the same original filename comes back disambiguated
   (``photo.png`` -> ``photo (1).png``), a *different* server filename that
   would silently break the whole point of caching. So this client caches
   local path -> server filename for the run and only uploads a given local
   file once.
2. **Local pre-validation.** File existence/readability, the 50 MB size
   limit, and (best-effort) the 64 MP image-dimension limit are all checked
   *before* any HTTP call, so a bad reference photo fails fast with a clear
   error instead of burning a round trip to the ComfyUI host.

Nothing here talks to a real socket in tests -- the ``session`` constructor
argument duck-types the subset of ``requests.Session`` this module calls
(``.post``), so tests inject
:class:`tests.harness.comfyui_mock.FakeComfyUISession`.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any

import requests

from music_video_maker import config
from music_video_maker.contracts import AudioChunk, ExpandedPrompt, StagedAssets

logger = logging.getLogger(__name__)

UPLOAD_PATH = "/upload/image"
"""ComfyUI's single ingest endpoint for both images and audio."""

DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_MEGAPIXELS = 64.0

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class StagingError(RuntimeError):
    """Raised when a Stage 3 upload fails.

    Covers local pre-validation failures, non-200 HTTP responses (including
    ComfyUI's 400s), network errors, and malformed/missing ``name`` in the
    JSON response. Carries whatever of ``path``/``size``/``endpoint``/
    ``status_code`` is known at the failure point so a caller (or the log
    line emitted just before this is raised) has everything needed to
    diagnose without re-running anything.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        size: int | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.path = path
        self.size = size
        self.endpoint = endpoint
        self.status_code = status_code

        details = []
        if path is not None:
            details.append(f"path={path}")
        if size is not None:
            details.append(f"size={size}")
        if endpoint is not None:
            details.append(f"endpoint={endpoint}")
        if status_code is not None:
            details.append(f"status_code={status_code}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")


def _sniff_png_megapixels(content: bytes) -> float | None:
    """Best-effort megapixel sniff from a PNG's IHDR chunk.

    Deliberately minimal -- reads only the signature + IHDR width/height,
    no full PNG decode and no Pillow dependency. Returns ``None`` for any
    content this can't confidently read as a PNG, so callers skip the
    megapixel check rather than guess.
    """
    if content[:8] == _PNG_SIGNATURE and len(content) >= 24:
        width, height = struct.unpack(">II", content[16:24])
        return (width * height) / 1_000_000
    return None


def _response_detail(response: Any) -> str:
    """Best-effort human-readable detail from a failed response, for logging."""
    try:
        return str(response.json())
    except (ValueError, AttributeError):
        return str(getattr(response, "text", ""))


class ComfyUIAssetStager:
    """Uploads images and audio to ComfyUI, with local validation and caching.

    Implements the :class:`music_video_maker.contracts.AssetStager` Protocol.
    """

    def __init__(
        self,
        base_url: str = config.DEFAULT_COMFYUI_URL,
        session: Any = None,
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_megapixels: float = DEFAULT_MAX_MEGAPIXELS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.max_upload_bytes = max_upload_bytes
        self.max_megapixels = max_megapixels
        self._cache: dict[str, str] = {}

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{UPLOAD_PATH}"

    # -- cache inspection -------------------------------------------------- #

    def cache_snapshot(self) -> dict[str, str]:
        """A copy of the current local-path -> server-filename cache."""
        return dict(self._cache)

    def clear_cache(self) -> None:
        """Drop all cached uploads. The next upload of any file re-hits ComfyUI."""
        self._cache.clear()

    # -- AssetStager protocol ------------------------------------------------ #

    def upload_image(self, path: Path) -> str:
        return self._upload(Path(path), check_megapixels=True)

    def upload_audio(self, path: Path) -> str:
        return self._upload(Path(path), check_megapixels=False)

    # -- convenience -------------------------------------------------------- #

    def stage_chunk(self, prompt: ExpandedPrompt, chunk: AudioChunk) -> StagedAssets:
        """Upload a chunk's active-character image(s) and audio stem, and
        assemble the resulting :class:`StagedAssets`.

        ``seed_frame_filename`` is always left ``None`` here -- I2V
        continuity bridging is issue #12's, out of scope for Stage 3.

        The two arguments must describe the *same* chunk. Mismatched ids are
        rejected loudly rather than silently staging one chunk's audio against
        another chunk's reference photo -- Stage 2's regrouping means several
        index spaces coexist, and a silent mix-up here surfaces only as the
        wrong face lip-syncing the wrong line in the finished video.

        ``prompt.image_refs`` (issue #33 levels 1-2) -- every active
        character's reference photo, primary first -- is staged into
        ``StagedAssets.image_filenames`` alongside the unchanged
        ``image_filename`` upload of ``prompt.image_ref``. Each upload still
        goes through :meth:`upload_image`, so this module's existing local
        path -> server filename cache (see the module docstring) de-dupes
        for free: a photo shared by ``prompt.image_ref`` and
        ``prompt.image_refs`` (or by two different chunks) is only ever
        uploaded once. ``prompt.image_refs`` defaults to ``()`` for every
        caller that predates multiple references, so
        ``StagedAssets.image_filenames`` then stays ``()`` too and exactly
        one upload happens -- identical to this method's behavior before
        issue #33.
        """
        if prompt.chunk_id != chunk.chunk_id:
            logger.error(
                "Stage 3 refusing to stage mismatched chunks: prompt.chunk_id=%s != "
                "chunk.chunk_id=%s",
                prompt.chunk_id,
                chunk.chunk_id,
            )
            raise StagingError(
                f"chunk id mismatch: prompt.chunk_id={prompt.chunk_id} != "
                f"chunk.chunk_id={chunk.chunk_id}",
                endpoint=self.endpoint,
            )

        image_filename = self.upload_image(prompt.image_ref)
        image_filenames = tuple(self.upload_image(ref) for ref in prompt.image_refs)
        audio_filename = self.upload_audio(chunk.audio_file)
        return StagedAssets(
            chunk_id=chunk.chunk_id,
            image_filename=image_filename,
            image_filenames=image_filenames,
            audio_filename=audio_filename,
        )

    # -- internals ------------------------------------------------------------ #

    def _upload(self, path: Path, *, check_megapixels: bool) -> str:
        cache_key = str(path.resolve())
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Stage 3 cache hit: %s -> %s (skipping upload)", path, cached)
            return cached

        content = self._validate_and_read(path, check_megapixels=check_megapixels)

        try:
            response = self.session.post(
                self.endpoint,
                files={"image": (path.name, content)},
                data={"type": "input", "subfolder": "", "overwrite": "false"},
            )
        except requests.RequestException as exc:
            logger.exception(
                "Network error uploading %s (%d bytes) to %s", path, len(content), self.endpoint
            )
            raise StagingError(
                f"network error uploading {path.name} to ComfyUI",
                path=path,
                size=len(content),
                endpoint=self.endpoint,
            ) from exc

        if response.status_code != 200:
            logger.error(
                "ComfyUI rejected upload of %s (%d bytes, status=%s, endpoint=%s): %s",
                path,
                len(content),
                response.status_code,
                self.endpoint,
                _response_detail(response),
            )
            raise StagingError(
                f"ComfyUI rejected upload of {path.name}",
                path=path,
                size=len(content),
                endpoint=self.endpoint,
                status_code=response.status_code,
            )

        name = self._extract_name(response, path=path, size=len(content))
        self._cache[cache_key] = name
        logger.info("Staged %s -> %s (%d bytes) via %s", path, name, len(content), self.endpoint)
        return name

    def _extract_name(self, response: Any, *, path: Path, size: int) -> str:
        try:
            payload = response.json()
            name = payload["name"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception(
                "Malformed upload response for %s from %s (status=%s)",
                path,
                self.endpoint,
                response.status_code,
            )
            raise StagingError(
                f"malformed upload response for {path.name}: missing or invalid 'name'",
                path=path,
                size=size,
                endpoint=self.endpoint,
                status_code=response.status_code,
            ) from exc

        if not isinstance(name, str) or not name:
            logger.error(
                "Malformed upload response for %s from %s: 'name' was %r (status=%s)",
                path,
                self.endpoint,
                name,
                response.status_code,
            )
            raise StagingError(
                f"malformed upload response for {path.name}: 'name' was {name!r}",
                path=path,
                size=size,
                endpoint=self.endpoint,
                status_code=response.status_code,
            )
        return name

    def _validate_and_read(self, path: Path, *, check_megapixels: bool) -> bytes:
        if not path.exists() or not path.is_file():
            logger.error(
                "Staging failed: file does not exist (endpoint=%s): %s", self.endpoint, path
            )
            raise StagingError(f"file does not exist: {path}", path=path, endpoint=self.endpoint)

        if not os.access(path, os.R_OK):
            logger.error(
                "Staging failed: file is not readable (endpoint=%s): %s", self.endpoint, path
            )
            raise StagingError(f"file is not readable: {path}", path=path, endpoint=self.endpoint)

        size = path.stat().st_size
        if size > self.max_upload_bytes:
            logger.error(
                "Staging failed: %s is %d bytes, exceeds limit %d bytes (endpoint=%s)",
                path,
                size,
                self.max_upload_bytes,
                self.endpoint,
            )
            raise StagingError(
                f"{path.name} ({size} bytes) exceeds the {self.max_upload_bytes}-byte upload limit",
                path=path,
                size=size,
                endpoint=self.endpoint,
            )

        try:
            content = path.read_bytes()
        except OSError as exc:
            logger.exception("Staging failed: could not read %s (endpoint=%s)", path, self.endpoint)
            raise StagingError(
                f"could not read {path.name}: {exc}",
                path=path,
                size=size,
                endpoint=self.endpoint,
            ) from exc

        if check_megapixels:
            megapixels = _sniff_png_megapixels(content)
            if megapixels is not None and megapixels > self.max_megapixels:
                logger.error(
                    "Staging failed: %s is %.1f MP, exceeds limit %.1f MP (endpoint=%s)",
                    path,
                    megapixels,
                    self.max_megapixels,
                    self.endpoint,
                )
                raise StagingError(
                    f"{path.name} ({megapixels:.1f} MP) exceeds the "
                    f"{self.max_megapixels}-MP local limit",
                    path=path,
                    size=size,
                    endpoint=self.endpoint,
                )

        return content


__all__ = ["ComfyUIAssetStager", "StagingError"]
