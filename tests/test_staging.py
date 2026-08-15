"""Tests for Stage 3 ComfyUI asset staging (issue #7).

Everything is driven through :class:`tests.harness.comfyui_mock.FakeComfyUISession`
-- no real network, no GPU, no live ComfyUI. The fake deliberately does *not*
dedupe uploads itself (a re-upload of the same original filename comes back
disambiguated, e.g. ``photo (1).png``), which is what makes the stager's own
local upload cache observable: a client that fails to cache would leak a
second ``/upload/image`` call and get back the disambiguated name.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import requests

from music_video_maker import contracts
from music_video_maker.staging import ComfyUIAssetStager, StagingError
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import make_oversized_file_stub, write_silent_wav

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_png(path: Path, width: int = 64, height: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_fake_png_bytes(width, height))
    return path


class _NetworkErrorSession:
    """Minimal stand-in whose ``post`` always raises a network error."""

    def post(self, url: str, **kwargs):
        raise requests.ConnectionError("simulated network failure")


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_upload_image_returns_server_filename(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")

    name = stager.upload_image(image)

    assert name == "dianne_ref.png"
    assert len(session.requests) == 1
    assert session.requests[0].method == "POST"
    assert session.requests[0].url.endswith("/upload/image")
    assert len(session.uploads) == 1
    assert session.uploads[0].original_filename == "dianne_ref.png"
    assert session.uploads[0].server_filename == "dianne_ref.png"


def test_upload_audio_returns_server_filename(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    audio = write_silent_wav(tmp_path / "chunk_0000.wav", seconds=1.0)

    name = stager.upload_audio(audio)

    assert name == "chunk_0000.wav"
    assert len(session.requests) == 1
    assert session.requests[0].url.endswith("/upload/image")
    assert session.uploads[0].original_filename == "chunk_0000.wav"


# --------------------------------------------------------------------------- #
# Cache / de-dupe
# --------------------------------------------------------------------------- #


def test_cache_dedupes_repeated_image_upload(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "rex_ref.png")

    first = stager.upload_image(image)
    second = stager.upload_image(image)

    assert first == second == "rex_ref.png"
    assert " (1)" not in second
    assert len(session.uploads) == 1
    assert len(session.requests) == 1
    assert stager.cache_snapshot() == {str(image.resolve()): "rex_ref.png"}


def test_cache_can_be_cleared(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "rex_ref.png")

    stager.upload_image(image)
    stager.clear_cache()
    stager.upload_image(image)

    assert stager.cache_snapshot() == {str(image.resolve()): "rex_ref (1).png"}
    assert len(session.uploads) == 2


def test_distinct_files_each_upload_independently(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    dianne = _write_png(tmp_path / "dianne_ref.png")
    rex = _write_png(tmp_path / "rex_ref.png")

    name_a = stager.upload_image(dianne)
    name_b = stager.upload_image(rex)

    assert name_a == "dianne_ref.png"
    assert name_b == "rex_ref.png"
    assert len(session.uploads) == 2


# --------------------------------------------------------------------------- #
# Local pre-validation
# --------------------------------------------------------------------------- #


def test_oversized_file_rejected_locally_without_http_call(tmp_path, caplog):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    stub = make_oversized_file_stub(tmp_path / "huge.png")

    with caplog.at_level(logging.ERROR), pytest.raises(StagingError) as excinfo:
        stager.upload_image(stub)

    assert excinfo.value.path == stub
    assert excinfo.value.size == stub.stat().st_size
    assert len(session.requests) == 0
    assert len(session.uploads) == 0
    assert "huge.png" in caplog.text


def test_over_megapixel_image_rejected_locally_without_http_call(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    # 9000x9000 = 81 MP, over the default 64 MP local limit.
    huge_image = _write_png(tmp_path / "too_big.png", width=9000, height=9000)

    with pytest.raises(StagingError) as excinfo:
        stager.upload_image(huge_image)

    assert excinfo.value.path == huge_image
    assert len(session.requests) == 0


def test_missing_file_raises_staging_error(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    missing = tmp_path / "does_not_exist.png"

    with pytest.raises(StagingError) as excinfo:
        stager.upload_image(missing)

    assert excinfo.value.path == missing
    assert len(session.requests) == 0


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission bits are not meaningfully enforceable as root or on Windows",
)
def test_unreadable_file_raises_staging_error(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    unreadable = _write_png(tmp_path / "locked.png")
    unreadable.chmod(0o000)
    try:
        with pytest.raises(StagingError) as excinfo:
            stager.upload_image(unreadable)
        assert excinfo.value.path == unreadable
        assert len(session.requests) == 0
    finally:
        unreadable.chmod(0o644)


# --------------------------------------------------------------------------- #
# Server / network failures
# --------------------------------------------------------------------------- #


def test_server_side_400_surfaces_as_structured_error_and_is_logged(tmp_path, caplog):
    # Server has a strict 64 MP limit (the fake's default); the stager is
    # configured more permissively so the 400 comes from ComfyUI, not local
    # pre-validation -- proving the HTTP-error path, not the local-check path.
    session = FakeComfyUISession(max_megapixels=64.0)
    stager = ComfyUIAssetStager(session=session, max_megapixels=1000.0)
    huge_image = _write_png(tmp_path / "over_server_limit.png", width=9000, height=9000)

    with caplog.at_level(logging.ERROR), pytest.raises(StagingError) as excinfo:
        stager.upload_image(huge_image)

    assert excinfo.value.status_code == 400
    assert excinfo.value.path == huge_image
    assert "over_server_limit.png" in caplog.text
    assert len(session.requests) == 1
    assert len(session.uploads) == 0


def test_network_error_surfaces_as_structured_error(tmp_path, caplog):
    stager = ComfyUIAssetStager(session=_NetworkErrorSession())
    image = _write_png(tmp_path / "dianne_ref.png")

    with caplog.at_level(logging.ERROR), pytest.raises(StagingError) as excinfo:
        stager.upload_image(image)

    assert excinfo.value.path == image
    assert "dianne_ref.png" in caplog.text


def test_malformed_response_missing_name_surfaces_as_structured_error(tmp_path, caplog):
    class _MalformedSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, **kwargs):
            self.calls += 1
            from tests.harness.comfyui_mock import FakeResponse

            return FakeResponse(200, json_data={"subfolder": "", "type": "input"})

    session = _MalformedSession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")

    with caplog.at_level(logging.ERROR), pytest.raises(StagingError) as excinfo:
        stager.upload_image(image)

    assert session.calls == 1
    assert excinfo.value.path == image


# --------------------------------------------------------------------------- #
# StagedAssets assembly
# --------------------------------------------------------------------------- #


def test_stage_chunk_assembles_staged_assets_with_server_filenames(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")
    audio = write_silent_wav(tmp_path / "chunk_0003.wav", seconds=5.0)

    prompt = contracts.ExpandedPrompt(
        chunk_id=3, prompt="a lead vocalist singing", image_ref=image, characters=("Dianne",)
    )
    chunk = contracts.AudioChunk(
        chunk_id=3, audio_file=audio, start=10.0, end=15.0, text="line", characters=("Dianne",)
    )

    staged = stager.stage_chunk(prompt, chunk)

    assert isinstance(staged, contracts.StagedAssets)
    assert staged.chunk_id == 3
    assert staged.image_filename == "dianne_ref.png"
    assert staged.audio_filename == "chunk_0003.wav"
    assert staged.seed_frame_filename is None
    # Never leaks a local path into the contract.
    assert str(image) not in staged.image_filename
    assert str(audio) not in staged.audio_filename


def test_stage_chunk_rejects_mismatched_chunk_ids(tmp_path, caplog):
    """Guards the index-space confusion that produced a real bug in #5: several
    chunk/segment/line index spaces coexist and all of them are plain ints."""
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")
    audio = write_silent_wav(tmp_path / "chunk_0004.wav", seconds=5.0)

    prompt = contracts.ExpandedPrompt(
        chunk_id=3, prompt="a lead vocalist singing", image_ref=image, characters=("Dianne",)
    )
    chunk = contracts.AudioChunk(
        chunk_id=4, audio_file=audio, start=15.0, end=20.0, text="line", characters=("Dianne",)
    )

    with caplog.at_level(logging.ERROR), pytest.raises(StagingError) as excinfo:
        stager.stage_chunk(prompt, chunk)

    assert "chunk id mismatch" in str(excinfo.value)
    assert "mismatched chunks" in caplog.text
    # Refused before any upload happened -- no half-staged state on the server.
    assert session.uploads == []


# --------------------------------------------------------------------------- #
# Multiple reference photos (issue #33 levels 1-2)
# --------------------------------------------------------------------------- #


def test_stage_chunk_with_no_image_refs_leaves_image_filenames_empty(tmp_path):
    """The default, pre-#33 shape: `prompt.image_refs` is `()`, so
    `StagedAssets.image_filenames` must stay `()` too -- one upload, exactly
    as before this feature existed."""
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")
    audio = write_silent_wav(tmp_path / "chunk_0003.wav", seconds=5.0)
    prompt = contracts.ExpandedPrompt(
        chunk_id=3, prompt="a lead vocalist singing", image_ref=image, characters=("Dianne",)
    )
    chunk = contracts.AudioChunk(
        chunk_id=3, audio_file=audio, start=10.0, end=15.0, text="line", characters=("Dianne",)
    )

    staged = stager.stage_chunk(prompt, chunk)

    assert staged.image_filenames == ()
    # One image upload (the primary) plus the audio stem -- exactly as
    # before this feature existed.
    assert len(session.uploads) == 2
    assert len([u for u in session.uploads if u.original_filename.endswith(".png")]) == 1


def test_stage_chunk_stages_every_reference_photo_in_order(tmp_path):
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    dianne = _write_png(tmp_path / "dianne_ref.png")
    jan = _write_png(tmp_path / "jan_ref.png")
    audio = write_silent_wav(tmp_path / "chunk_0005.wav", seconds=5.0)
    prompt = contracts.ExpandedPrompt(
        chunk_id=5,
        prompt="two voices",
        image_ref=dianne,
        image_refs=(dianne, jan),
        characters=("Dianne", "Jan"),
    )
    chunk = contracts.AudioChunk(
        chunk_id=5,
        audio_file=audio,
        start=0.0,
        end=5.0,
        text="line",
        characters=("Dianne", "Jan"),
    )

    staged = stager.stage_chunk(prompt, chunk)

    assert staged.image_filename == "dianne_ref.png"
    assert staged.image_filenames == ("dianne_ref.png", "jan_ref.png")
    # dianne_ref.png is the primary AND the first entry in image_refs -- the
    # cache must dedupe that overlap into a single upload. 2 image uploads
    # (dianne, jan) plus the audio stem = 3 total.
    assert len(session.uploads) == 3
    image_uploads = [
        u.original_filename for u in session.uploads if u.original_filename.endswith(".png")
    ]
    assert image_uploads == ["dianne_ref.png", "jan_ref.png"]


def test_stage_chunk_dedupes_a_reference_photo_shared_across_two_chunks(tmp_path):
    """The same cast reference photo staged for two different chunks (e.g.
    Jan appears both solo and alongside Dianne) must only cross the wire
    once -- the whole point of the stager's upload cache."""
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    dianne = _write_png(tmp_path / "dianne_ref.png")
    jan = _write_png(tmp_path / "jan_ref.png")
    audio_a = write_silent_wav(tmp_path / "chunk_0006.wav", seconds=5.0)
    audio_b = write_silent_wav(tmp_path / "chunk_0007.wav", seconds=5.0)

    prompt_a = contracts.ExpandedPrompt(
        chunk_id=6,
        prompt="duet",
        image_ref=dianne,
        image_refs=(dianne, jan),
        characters=("Dianne", "Jan"),
    )
    chunk_a = contracts.AudioChunk(
        chunk_id=6,
        audio_file=audio_a,
        start=0.0,
        end=5.0,
        text="line a",
        characters=("Dianne", "Jan"),
    )
    prompt_b = contracts.ExpandedPrompt(
        chunk_id=7, prompt="Jan solo", image_ref=jan, characters=("Jan",)
    )
    chunk_b = contracts.AudioChunk(
        chunk_id=7, audio_file=audio_b, start=5.0, end=10.0, text="line b", characters=("Jan",)
    )

    staged_a = stager.stage_chunk(prompt_a, chunk_a)
    staged_b = stager.stage_chunk(prompt_b, chunk_b)

    assert staged_a.image_filenames == ("dianne_ref.png", "jan_ref.png")
    assert staged_b.image_filename == "jan_ref.png"
    # 2 photos total (dianne, jan) uploaded once each, plus 2 distinct audio
    # stems -- never a re-upload of jan_ref.png for chunk_b.
    image_uploads = [u for u in session.uploads if u.original_filename.endswith(".png")]
    assert len(image_uploads) == 2
    assert " (1)" not in staged_b.image_filename


def test_stage_chunk_uploads_stem_sourced_conditioning_audio_unchanged(tmp_path):
    """Issue #25: conditioning H3 on an isolated vocal stem changes *which*
    file a chunk's ``audio_file`` points at, and nothing else about Stage 3.

    The stem slice is an ordinary wav produced by
    :mod:`music_video_maker.stems` in its own directory, so it goes over the
    same ``/upload/image`` ingest with the same validation and the same cache.
    This test exists to keep that true: if staging ever grows a special case
    for mix slices (a filename assumption, a path assumption), the stem path
    must not silently fall out of it.
    """
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(session=session)
    image = _write_png(tmp_path / "dianne_ref.png")
    stem_slice = write_silent_wav(tmp_path / "stem_chunks" / "chunk_017_vocal.wav", seconds=5.0)

    prompt = contracts.ExpandedPrompt(
        chunk_id=17, prompt="the chorus", image_ref=image, characters=("Dianne",)
    )
    chunk = contracts.AudioChunk(
        chunk_id=17,
        audio_file=stem_slice,
        start=103.0,
        end=108.0,
        text="shared words",
        characters=("Dianne",),
    )

    staged = stager.stage_chunk(prompt, chunk)

    assert staged.audio_filename == "chunk_017_vocal.wav"
    audio_uploads = [u for u in session.uploads if u.original_filename.endswith(".wav")]
    assert len(audio_uploads) == 1


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #


def test_stager_satisfies_asset_stager_protocol():
    stager = ComfyUIAssetStager(session=FakeComfyUISession())
    assert isinstance(stager, contracts.AssetStager)
