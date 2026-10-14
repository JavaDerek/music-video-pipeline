"""Tests for the lip-sync backends (``lipsync.py``) and shot provenance
(``provenance.py``). Everything runs offline: the hosted API is a scripted
fake session, ffmpeg is a recording runner that writes the files it is asked
for, and no key is read from the real environment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_video_maker import lipsync, provenance
from music_video_maker.lipsync import (
    AUDIO_BINDING_SENTENCE,
    CostEstimate,
    LipSyncError,
    LipSyncSettings,
    LocalAssetStager,
    LocalDraftUnavailableError,
    LocalH3LockedError,
    MiniMaxApiError,
    MiniMaxExecutionClient,
    MiniMaxVideoClient,
    ShotRequest,
    SpendNotConfirmedError,
    billable_seconds,
    build_conform_args,
    build_minimax_backend,
    build_pad_audio_args,
    build_reference_content,
    confirm_spend,
    estimate_cost,
    ratio_for,
    read_api_key,
    validate_settings,
)
from music_video_maker.provenance import (
    FinalRenderGuardError,
    Provenance,
    ProvenanceError,
    ShotProvenance,
    assert_final_grade,
    load_provenance,
    merge_provenance,
    record_shot,
    save_provenance,
    sha256_file,
)
from tests.harness.factories import make_workflow_baseline

KEY = "sk-test-not-a-real-key"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, status_code: int = 200, payload=None, body: bytes = b""):
        self.status_code = status_code
        self._payload = payload
        self._body = body
        self.text = json.dumps(payload) if payload is not None else body.decode("latin-1")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_content(self, chunk_size=1 << 20):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class FakeMiniMaxSession:
    """Scripted ``requests.Session`` stand-in for the three endpoints and the
    download URL. ``statuses`` is the sequence of task statuses the query
    endpoint walks through."""

    def __init__(self, *, statuses=("queued", "running", "succeeded"),
                 video_bytes: bytes = b"\x00\x00\x00\x18ftypmp42-fake-video",
                 upload_fail: bool = False, create_status: int = 200):
        self.statuses = list(statuses)
        self.video_bytes = video_bytes
        self.upload_fail = upload_fail
        self.create_status = create_status
        self.calls: list[tuple[str, str, dict]] = []
        self.uploads: list[str] = []
        self.created: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        assert kwargs["headers"]["Authorization"] == f"Bearer {KEY}"
        if url.endswith(lipsync.MINIMAX_UPLOAD_PATH):
            if self.upload_fail:
                return _Response(400, {"base_resp": {"status_code": 2013, "status_msg": "bad"}})
            name = kwargs["files"]["file"][0]
            self.uploads.append(name)
            return _Response(200, {"file": {"file_id": 1000 + len(self.uploads),
                                            "filename": name},
                                   "base_resp": {"status_code": 0}})
        if url.endswith(lipsync.MINIMAX_CREATE_PATH):
            self.created.append(kwargs["json"])
            if self.create_status != 200:
                return _Response(self.create_status,
                                 {"type": "error", "error": {"message": "nope"}})
            return _Response(200, {"task_id": "424010985738629"})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if "/v2/query/video_generation/" in url:
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            task = {"id": url.rsplit("/", 1)[1], "model": "MiniMax-H3", "status": status,
                    "resolution": "2K", "duration": 6}
            if status == "succeeded":
                task["content"] = {"url": "https://cdn.example/out.mp4"}
                task["usage"] = {"total_seconds": 6}
            if status == "failed":
                task["error"] = {"code": "1026", "message": "sensitive"}
            return _Response(200, {"task": task})
        if url == "https://cdn.example/out.mp4":
            return _Response(200, body=self.video_bytes)
        raise AssertionError(f"unexpected GET {url}")


class RecordingRunner:
    """Pretends to be ffmpeg: records argv and writes the output path named
    last, so hashes and existence checks have something to read."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        Path(args[-1]).write_bytes(b"out:" + " ".join(args).encode()[:64])
        return None


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "ref.png"
    audio = tmp_path / "line.wav"
    image.write_bytes(b"\x89PNG fake")
    audio.write_bytes(b"RIFF fake wav")
    return image, audio


def _client(session, runner, **kwargs) -> MiniMaxVideoClient:
    sleeps: list[float] = []
    ticks = iter(range(0, 100000))
    client = MiniMaxVideoClient(
        api_key=KEY, session=session, runner=runner, sleeper=sleeps.append,
        clock=lambda: float(next(ticks)), poll_interval_seconds=7.0, **kwargs,
    )
    client._sleeps = sleeps  # type: ignore[attr-defined]
    return client


# --------------------------------------------------------------------------- #
# Settings and the locks
# --------------------------------------------------------------------------- #


def test_defaults_are_the_hosted_api_at_draft_tier():
    settings = LipSyncSettings()
    validate_settings(settings)
    assert settings.backend == "minimax_api"
    assert settings.tier == "draft"
    assert settings.minimax_model == "MiniMax-H3-Max"
    assert settings.minimax_resolution == "480P"
    assert not settings.uses_comfyui


def test_final_tier_is_two_k_through_the_api():
    settings = LipSyncSettings(tier="final")
    validate_settings(settings)
    assert (settings.minimax_model, settings.minimax_resolution) == ("MiniMax-H3", "2K")


@pytest.mark.parametrize("kwargs", [{"backend": "comfy"}, {"tier": "best"}])
def test_unknown_backend_or_tier_is_refused(kwargs):
    with pytest.raises(LipSyncError):
        validate_settings(LipSyncSettings(**kwargs))


def test_local_h3_is_locked_without_an_authorization_file(tmp_path):
    with pytest.raises(LocalH3LockedError, match="h3-license"):
        validate_settings(LipSyncSettings(backend="local_h3"))
    with pytest.raises(LocalH3LockedError, match="does not exist"):
        validate_settings(LipSyncSettings(backend="local_h3",
                                          h3_authorization_file=tmp_path / "missing.pdf"))


def test_local_h3_unlocks_only_with_the_file_on_disk(tmp_path):
    auth = tmp_path / "minimax_h3_authorization.pdf"
    auth.write_bytes(b"%PDF-1.4 fake")
    settings = LipSyncSettings(backend="local_h3", h3_authorization_file=auth)
    validate_settings(settings)
    assert settings.uses_comfyui


def test_final_tier_is_refused_on_a_local_backend(tmp_path):
    auth = tmp_path / "auth.pdf"
    auth.write_bytes(b"x")
    with pytest.raises(LipSyncError, match="only available through"):
        validate_settings(LipSyncSettings(backend="local_h3", tier="final",
                                          h3_authorization_file=auth))


def test_local_draft_refuses_until_a_model_is_integrated():
    with pytest.raises(LocalDraftUnavailableError, match="none integrated yet"):
        validate_settings(LipSyncSettings(backend="local_draft"))
    with pytest.raises(LocalDraftUnavailableError):
        validate_settings(LipSyncSettings(backend="local_draft", local_draft_model="wav2lip"))


def test_no_local_draft_model_is_integrated_yet():
    assert lipsync.LOCAL_DRAFT_MODELS == {}


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "frames, model, expected",
    [
        (124, "MiniMax-H3", 6),      # 5.167 s rounds up
        (192, "MiniMax-H3", 8),      # exactly 8.0 s stays 8
        (48, "MiniMax-H3", 4),       # 2 s clamps to the H3 floor
        (48, "MiniMax-H3-Max", 5),   # ... and to the Max floor
        (362, "MiniMax-H3", 15),     # 15.083 s clamps to the ceiling
    ],
)
def test_billable_seconds_round_up_and_clamp(frames, model, expected):
    assert billable_seconds(frames, model=model) == expected


def test_estimate_cost_uses_the_published_per_second_prices():
    estimate = estimate_cost([243, 141, 209, 124, 226, 209], model="MiniMax-H3", resolution="2K")
    assert estimate.billable_seconds == 11 + 6 + 9 + 6 + 10 + 9
    assert estimate.usd == pytest.approx(51 * 0.13)
    assert not estimate.needs_confirmation
    big = estimate_cost([243] * 12, model="MiniMax-H3", resolution="2K")
    assert big.usd == pytest.approx(12 * 11 * 0.13) and big.needs_confirmation
    assert "prices read" in estimate.describe()
    cheap = estimate_cost([124], model="MiniMax-H3-Max", resolution="480P")
    assert cheap.usd == pytest.approx(0.30)
    assert not cheap.needs_confirmation


def test_estimate_cost_refuses_an_unpriced_combination():
    with pytest.raises(LipSyncError, match="no published price"):
        estimate_cost([124], model="MiniMax-H3-Max", resolution="2K")


def test_confirm_spend_needs_a_yes_over_the_threshold():
    big = CostEstimate(model="MiniMax-H3", resolution="2K", shots=1,
                       billable_seconds=100, usd_per_second=0.13)
    with pytest.raises(SpendNotConfirmedError, match="--yes"):
        confirm_spend(big, yes=False)
    confirm_spend(big, yes=True)
    confirm_spend(big, yes=False, confirmer=lambda e: True)
    small = CostEstimate(model="MiniMax-H3-Max", resolution="480P", shots=1,
                         billable_seconds=10, usd_per_second=0.05)
    confirm_spend(small, yes=False)


# --------------------------------------------------------------------------- #
# The hosted client
# --------------------------------------------------------------------------- #


def test_api_key_comes_from_the_environment_only():
    assert read_api_key(environ={"MINIMAX_API_KEY": " k "}) == "k"
    with pytest.raises(MiniMaxApiError, match="MINIMAX_API_KEY"):
        read_api_key(environ={})
    with pytest.raises(MiniMaxApiError):
        MiniMaxVideoClient(session=FakeMiniMaxSession(), environ={})


@pytest.mark.parametrize("size, ratio", [((864, 480), "16:9"), ((768, 1344), "9:16"),
                                         ((1024, 1024), "1:1"), ((2560, 1080), "21:9")])
def test_ratio_for_picks_the_nearest_label(size, ratio):
    assert ratio_for(*size) == ratio


def test_reference_content_binds_image_and_audio_by_role():
    content = build_reference_content(prompt="p", image_url="mm_file://1",
                                      audio_url="mm_file://2", extra_image_urls=["mm_file://3"])
    assert content[0] == {"type": "text", "text": "p"}
    assert [c["role"] for c in content[1:]] == ["reference_image", "reference_image",
                                                "reference_audio"]
    assert content[-1]["audio_url"] == {"url": "mm_file://2"}


def test_pad_and_conform_argv(tmp_path):
    pad = build_pad_audio_args(tmp_path / "a.wav", tmp_path / "p.wav", seconds=6)
    assert "apad=whole_dur=6.000000" in pad and pad[pad.index("-t") + 1] == "6.000000"
    assert "-nostdin" in pad
    conform = build_conform_args(tmp_path / "api.mp4", tmp_path / "p.wav", tmp_path / "c.mp4",
                                 frames=141)
    assert "fps=24,tpad=stop_mode=clone:stop=141" in conform
    assert conform[conform.index("-frames:v") + 1] == "141"
    assert conform[conform.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in conform


def test_generate_shot_uploads_creates_polls_downloads_conforms_and_records(tmp_path):
    image, audio = _inputs(tmp_path)
    session = FakeMiniMaxSession()
    runner = RecordingRunner()
    client = _client(session, runner)
    out = tmp_path / "chunks" / "chunk_0007.mp4"
    prov = tmp_path / "chunks" / "provenance.json"
    request = ShotRequest(shot_id="chunk_0007", image=image, audio=audio,
                          prompt="She sings to the camera.", frames=175, width=864, height=480)

    shot = client.generate_shot(request, model="MiniMax-H3", resolution="2K", tier="final",
                                out_path=out, provenance_path=prov)

    # three uploads: image, then the padded audio (the raw audio is never sent)
    assert session.uploads == ["ref.png", "chunk_0007_audio.wav"]
    created = session.created[0]
    assert created["model"] == "MiniMax-H3" and created["resolution"] == "2K"
    assert created["duration"] == 8 and created["ratio"] == "16:9"
    assert created["content"][0]["text"].endswith(AUDIO_BINDING_SENTENCE)
    assert created["content"][1] == {"type": "image_url", "image_url": {"url": "mm_file://1001"},
                                     "role": "reference_image"}
    assert created["content"][2]["role"] == "reference_audio"
    # polled through queued/running to succeeded, sleeping between reads
    assert client._sleeps == [7.0, 7.0]
    # ffmpeg: pad first, conform last, to the requested frame count
    assert runner.calls[0][0] == "ffmpeg" and "apad=whole_dur=8.000000" in runner.calls[0]
    assert runner.calls[-1][-1] == str(out) and "-frames:v" in runner.calls[-1]
    raw = out.with_name("chunk_0007.api.mp4")
    assert raw.read_bytes() == session.video_bytes
    assert out.exists()
    # the record
    assert shot.backend == "minimax_api" and shot.tier == "final"
    assert shot.task_id == "424010985738629" and shot.frames == 175
    assert shot.duration_seconds == 8.0 and shot.cost_usd == pytest.approx(1.04)
    assert shot.inputs["reference_image"] == sha256_file(image)
    assert shot.inputs["reference_audio"] == sha256_file(audio)
    assert shot.output_sha256 == sha256_file(out)
    assert shot.notes["api_output_sha256"] == sha256_file(raw)
    saved = load_provenance(prov)
    assert saved.find("chunk_0007") == shot
    assert saved.terms["effective_date"] == "2026-03-30"
    assert "retain your ownership rights" in saved.terms["ownership_clause"]
    # the key is in headers only -- never in the record
    assert KEY not in prov.read_text()


def test_generate_shot_refuses_a_shot_too_long_for_the_model(tmp_path):
    image, audio = _inputs(tmp_path)
    client = _client(FakeMiniMaxSession(), RecordingRunner())
    with pytest.raises(LipSyncError, match="over the MiniMax-H3 maximum"):
        client.generate_shot(ShotRequest("s", image, audio, "p", frames=24 * 16),
                             model="MiniMax-H3", resolution="2K", tier="final",
                             out_path=tmp_path / "s.mp4")
    with pytest.raises(LipSyncError, match="does not offer"):
        client.generate_shot(ShotRequest("s", image, audio, "p", frames=124),
                             model="MiniMax-H3-Max", resolution="2K", tier="final",
                             out_path=tmp_path / "s.mp4")


def test_failed_task_and_rejected_create_raise_with_the_reason(tmp_path):
    image, audio = _inputs(tmp_path)
    failing = _client(FakeMiniMaxSession(statuses=("queued", "failed")), RecordingRunner())
    with pytest.raises(MiniMaxApiError, match="sensitive"):
        failing.generate_shot(ShotRequest("s", image, audio, "p", frames=124),
                              model="MiniMax-H3", resolution="2K", tier="final",
                              out_path=tmp_path / "s.mp4")
    rejected = _client(FakeMiniMaxSession(create_status=401), RecordingRunner())
    with pytest.raises(MiniMaxApiError, match="HTTP 401"):
        rejected.generate_shot(ShotRequest("s", image, audio, "p", frames=124),
                               model="MiniMax-H3", resolution="2K", tier="final",
                               out_path=tmp_path / "s.mp4")
    upload_refused = _client(FakeMiniMaxSession(upload_fail=True), RecordingRunner())
    with pytest.raises(MiniMaxApiError, match="upload"):
        upload_refused.upload(image)


def test_wait_gives_up_after_the_deadline(tmp_path):
    client = _client(FakeMiniMaxSession(statuses=("running",)), RecordingRunner(),
                     max_wait_seconds=3)
    with pytest.raises(MiniMaxApiError, match="still 'running'"):
        client.wait("t")


def test_error_bodies_never_echo_the_key():
    session = FakeMiniMaxSession(create_status=500)
    client = _client(session, RecordingRunner())
    session.post = lambda url, **kw: _Response(500, {"error": {"message": f"leak {KEY}"}})
    with pytest.raises(MiniMaxApiError) as excinfo:
        client.create_task(model="MiniMax-H3", content=[], resolution="2K", duration=5)
    assert KEY not in str(excinfo.value) and "<redacted>" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# The pipeline seams
# --------------------------------------------------------------------------- #


def test_local_stager_returns_absolute_paths_and_checks_existence(tmp_path):
    image, audio = _inputs(tmp_path)
    stager = LocalAssetStager()
    assert stager.upload_image(image) == str(image.resolve())
    with pytest.raises(LipSyncError, match="asset not found"):
        stager.upload_audio(tmp_path / "nope.wav")
    from music_video_maker.contracts import AudioChunk, ExpandedPrompt
    prompt = ExpandedPrompt(chunk_id=3, prompt="p", image_ref=image, image_refs=(image,))
    chunk = AudioChunk(chunk_id=3, audio_file=audio, start=0.0, end=5.2, text="la")
    staged = stager.stage_chunk(prompt, chunk)
    assert staged.audio_filename == str(audio.resolve())
    assert staged.image_filenames == (str(image.resolve()),)
    with pytest.raises(LipSyncError, match="mismatch"):
        stager.stage_chunk(prompt, AudioChunk(chunk_id=4, audio_file=audio, start=0, end=5,
                                              text=""))


def _mutated_workflow(image: Path, audio: Path) -> dict:
    """The committed template's shape, after the mutator: local paths staged
    in, a prompt, a length and a size on the H3 node."""
    workflow = make_workflow_baseline()
    for node in workflow.values():
        if node["class_type"] == "LoadImage":
            node["inputs"]["image"] = str(image)
        if node["class_type"] == "LoadAudio":
            node["inputs"]["audio"] = str(audio)
        if node["class_type"] == "MiniMaxH3ReferenceToVideo":
            node["inputs"].update(prompt="The singer, grave.", length=141, width=864, height=480)
    return workflow


def test_execution_client_reads_the_graph_by_class_and_wiring(tmp_path):
    image, audio = _inputs(tmp_path)
    session = FakeMiniMaxSession()
    client = _client(session, RecordingRunner())
    settings = LipSyncSettings(tier="final")
    execution = MiniMaxExecutionClient(client, settings=settings, clock=lambda: 0.0)
    result = execution.execute(_mutated_workflow(image, audio), 12, tmp_path / "out")
    assert result.chunk_id == 12 and result.status.value == "rendered"
    assert result.video_file == tmp_path / "out" / "chunk_0012.mp4"
    assert result.prompt_id == "424010985738629"
    created = session.created[0]
    assert created["duration"] == 6 and created["ratio"] == "16:9"
    assert created["content"][0]["text"].startswith("The singer, grave.")
    assert load_provenance(tmp_path / "out" / "provenance.json").find("chunk_0012") is not None


def test_execution_client_refuses_the_chained_i2v_graph(tmp_path):
    from tests.harness.factories import make_workflow_i2v
    client = _client(FakeMiniMaxSession(), RecordingRunner())
    execution = MiniMaxExecutionClient(client, settings=LipSyncSettings())
    with pytest.raises(LipSyncError, match="reference-to-video only"):
        execution.execute(make_workflow_i2v(), 1, tmp_path)


def test_build_minimax_backend_validates_and_labels_the_stack():
    backend = build_minimax_backend(LipSyncSettings(tier="final"), session=FakeMiniMaxSession(),
                                    environ={"MINIMAX_API_KEY": KEY})
    assert isinstance(backend.stager, LocalAssetStager)
    assert isinstance(backend.execution_client, MiniMaxExecutionClient)
    assert backend.stack_label == "minimax_api:MiniMax-H3@2K"
    assert not backend.uses_comfyui
    with pytest.raises(LipSyncError):
        build_minimax_backend(LipSyncSettings(backend="local_h3"), environ={"MINIMAX_API_KEY": KEY})


# --------------------------------------------------------------------------- #
# Provenance and the final-render guard
# --------------------------------------------------------------------------- #


def _shot(shot_id: str, backend: str = "minimax_api", tier: str = "final",
          model: str = "MiniMax-H3", resolution: str = "2K") -> ShotProvenance:
    return ShotProvenance(shot_id=shot_id, backend=backend, model=model, tier=tier,
                          resolution=resolution)


def test_shot_record_refuses_unknown_backend_or_tier():
    with pytest.raises(ProvenanceError):
        ShotProvenance(shot_id="a", backend="fal", model="m", tier="final")
    with pytest.raises(ProvenanceError):
        ShotProvenance(shot_id="a", backend="minimax_api", model="m", tier="best")


def test_provenance_round_trips_and_replaces_by_shot_id(tmp_path):
    path = tmp_path / "provenance.json"
    record_shot(path, _shot("a"), terms={"url": "u"})
    record_shot(path, _shot("b"))
    record_shot(path, _shot("a", tier="draft", model="MiniMax-H3-Max", resolution="480P"))
    loaded = load_provenance(path)
    assert [s.shot_id for s in loaded.shots] == ["b", "a"]
    assert loaded.find("a").tier == "draft"
    assert loaded.terms == {"url": "u"}
    assert loaded.schema_version == provenance.PROVENANCE_SCHEMA_VERSION
    with pytest.raises(ProvenanceError, match="not found"):
        load_provenance(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ProvenanceError, match="not valid JSON"):
        load_provenance(tmp_path / "bad.json")
    (tmp_path / "shape.json").write_text(json.dumps({"shots": {"a": 1}}))
    with pytest.raises(ProvenanceError, match="must be a list"):
        load_provenance(tmp_path / "shape.json")
    (tmp_path / "field.json").write_text(json.dumps({"shots": [{"shot_id": "a", "x": 1}]}))
    with pytest.raises(ProvenanceError):
        load_provenance(tmp_path / "field.json")


def test_merge_provenance_keeps_the_first_terms_and_the_last_shot(tmp_path):
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    save_provenance(Provenance(shots=[_shot("a", tier="draft")], terms={"url": "one"}), one)
    save_provenance(Provenance(shots=[_shot("a"), _shot("b")], terms={"url": "two"}), two)
    merged = merge_provenance([one, two])
    assert merged.terms == {"url": "one"}
    assert {s.shot_id: s.tier for s in merged.shots} == {"a": "final", "b": "final"}


def test_final_grade_guard_refuses_local_backends_and_draft_tiers():
    assert_final_grade(Provenance(shots=[_shot("a"), _shot("b")]))
    with pytest.raises(FinalRenderGuardError, match="local_h3"):
        assert_final_grade(Provenance(shots=[_shot("a"), _shot("b", backend="local_h3")]))
    with pytest.raises(FinalRenderGuardError, match="local_draft"):
        assert_final_grade(Provenance(shots=[_shot("b", backend="local_draft", tier="draft")]))
    with pytest.raises(FinalRenderGuardError, match="draft'-tier"):
        assert_final_grade(Provenance(shots=[_shot("a", tier="draft", resolution="480P")]))
    # a draft step may carry draft API shots, but never a local one
    assert_final_grade(Provenance(shots=[_shot("a", tier="draft")]), tier="draft")
    with pytest.raises(FinalRenderGuardError):
        assert_final_grade(Provenance(shots=[_shot("a", backend="local_h3", tier="draft")]),
                           tier="draft")


def test_final_grade_guard_needs_every_expected_shot_and_at_least_one():
    with pytest.raises(FinalRenderGuardError, match="no shots at all"):
        assert_final_grade(Provenance())
    with pytest.raises(FinalRenderGuardError, match="no provenance record: \\['c'\\]"):
        assert_final_grade(Provenance(shots=[_shot("a")]), expected_shot_ids=["a", "c"])
    with pytest.raises(ProvenanceError):
        assert_final_grade(Provenance(shots=[_shot("a")]), tier="best")


def test_sha256_file_streams(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    import hashlib
    assert sha256_file(path, chunk_bytes=1024) == hashlib.sha256(path.read_bytes()).hexdigest()
