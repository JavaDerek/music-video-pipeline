"""Exemplar tests proving the mock-ComfyUI harness works (issue #16).

These tests exercise the harness itself -- one endpoint / one failure mode at a
time -- so Wave 2 lanes (#7, #9, #10, #12, #14) can pattern-match against them
when they wire the harness into their own clients. Fully offline: no sockets,
no filesystem, no ``time.sleep()``.
"""

from __future__ import annotations

import json
import logging

import pytest
import requests
from websocket import WebSocketConnectionClosedException, WebSocketTimeoutException

from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.ws import (
    ScriptedWebSocket,
    build_disconnect_sequence,
    build_hang_sequence,
    build_noise_messages,
    build_oom_sequence,
    build_success_sequence,
    executing_message,
    execution_error_message,
    make_ws_factory,
)

# --------------------------------------------------------------------------- #
# /upload/image
# --------------------------------------------------------------------------- #


def test_upload_image_returns_server_filename_and_records_request():
    session = FakeComfyUISession()
    resp = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("lead.png", make_fake_png_bytes(64, 64), "image/png")},
        data={"type": "input", "subfolder": "", "overwrite": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"name": "lead.png", "subfolder": "", "type": "input"}
    assert len(session.uploads) == 1
    assert session.uploads[0].original_filename == "lead.png"
    assert len(session.requests) == 1
    assert session.requests[0].extra["filename"] == "lead.png"


def test_upload_audio_stem_via_same_endpoint():
    session = FakeComfyUISession()
    resp = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("chunk_0001.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        data={"type": "input"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "chunk_0001.wav"


def test_upload_rejects_oversized_file_with_400():
    session = FakeComfyUISession(max_upload_bytes=1024)
    big = b"0" * 2048
    resp = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("big.png", big, "image/png")},
        data={"type": "input"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()
    assert session.uploads == []


def test_upload_rejects_over_megapixel_image_with_400():
    session = FakeComfyUISession(max_megapixels=1.0)
    huge = make_fake_png_bytes(10_000, 10_000)  # 100 MP, tiny byte size
    resp = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("huge.png", huge, "image/png")},
        data={"type": "input"},
    )
    assert resp.status_code == 400
    assert session.uploads == []


def test_repeated_upload_without_client_dedup_gets_disambiguated_filename():
    """Models real ComfyUI collision behavior: a client that fails to cache
    uploads gets a visibly different filename back on the 2nd call -- the
    signal issue #7's dedup requirement is tested against."""
    session = FakeComfyUISession()
    first = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("lead.png", b"same-bytes", "image/png")},
        data={"type": "input"},
    )
    second = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("lead.png", b"same-bytes", "image/png")},
        data={"type": "input"},
    )
    assert first.json()["name"] == "lead.png"
    assert second.json()["name"] == "lead (1).png"
    assert len(session.requests) == 2


def test_upload_with_overwrite_keeps_same_filename():
    session = FakeComfyUISession()
    session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("lead.png", b"v1", "image/png")},
        data={"type": "input"},
    )
    resp = session.post(
        f"{session.base_url}/upload/image",
        files={"image": ("lead.png", b"v2", "image/png")},
        data={"type": "input", "overwrite": "true"},
    )
    assert resp.json()["name"] == "lead.png"


def test_upload_with_no_files_returns_400():
    session = FakeComfyUISession()
    resp = session.post(f"{session.base_url}/upload/image", files={}, data={})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# /prompt
# --------------------------------------------------------------------------- #


def test_prompt_submission_returns_prompt_id_and_number():
    session = FakeComfyUISession()
    resp = session.post(
        f"{session.base_url}/prompt",
        json={"prompt": {"3": {"class_type": "KSampler", "inputs": {}}}, "client_id": "abc-123"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt_id"]
    assert body["number"] == 1
    assert body["node_errors"] == {}
    assert session.submitted_prompts[0]["client_id"] == "abc-123"


def test_prompt_ids_increment_across_submissions():
    session = FakeComfyUISession()
    first = session.post(f"{session.base_url}/prompt", json={"prompt": {}, "client_id": "c1"})
    second = session.post(f"{session.base_url}/prompt", json={"prompt": {}, "client_id": "c1"})
    assert first.json()["prompt_id"] != second.json()["prompt_id"]
    assert second.json()["number"] == 2


def test_prompt_can_be_scripted_to_return_node_errors():
    session = FakeComfyUISession()
    session.queue_node_errors({"3": {"errors": [{"message": "required input missing"}]}})
    resp = session.post(f"{session.base_url}/prompt", json={"prompt": {}, "client_id": "c1"})
    body = resp.json()
    assert body["prompt_id"] is None
    assert "3" in body["node_errors"]


# --------------------------------------------------------------------------- #
# /history/{prompt_id}
# --------------------------------------------------------------------------- #


def test_history_contains_mp4_output():
    session = FakeComfyUISession()
    session.seed_history_success(
        "prompt-0001", video_filename="chunk_0001.mp4", video_bytes=b"fake-mp4-bytes"
    )
    resp = session.get(f"{session.base_url}/history/prompt-0001")
    assert resp.status_code == 200
    entry = resp.json()["prompt-0001"]
    outputs = entry["outputs"]
    node = next(iter(outputs.values()))
    video = node["videos"][0]
    assert video["filename"] == "chunk_0001.mp4"
    assert video["type"] == "output"


def test_history_with_no_video_output_has_no_mp4():
    session = FakeComfyUISession()
    session.seed_history_without_video("prompt-0002")
    resp = session.get(f"{session.base_url}/history/prompt-0002")
    entry = resp.json()["prompt-0002"]
    for node_output in entry["outputs"].values():
        for items in node_output.values():
            assert all(not str(i).endswith(".mp4") for i in items)


def test_history_for_unknown_prompt_id_is_empty():
    session = FakeComfyUISession()
    resp = session.get(f"{session.base_url}/history/does-not-exist")
    assert resp.status_code == 200
    assert resp.json() == {}


# --------------------------------------------------------------------------- #
# /view
# --------------------------------------------------------------------------- #


def test_view_returns_binary_bytes_for_seeded_output():
    session = FakeComfyUISession()
    session.seed_history_success(
        "prompt-0001", video_filename="chunk_0001.mp4", video_bytes=b"\x00\x01mp4-payload"
    )
    resp = session.get(
        f"{session.base_url}/view",
        params={"filename": "chunk_0001.mp4", "subfolder": "", "type": "output"},
    )
    assert resp.status_code == 200
    assert resp.content == b"\x00\x01mp4-payload"
    assert b"".join(resp.iter_content(chunk_size=4)) == resp.content


def test_view_disk_full_truncates_content():
    session = FakeComfyUISession()
    session.seed_history_success(
        "prompt-0001", video_filename="chunk_0001.mp4", video_bytes=b"0123456789" * 100
    )
    session.corrupt_view_output("chunk_0001.mp4", mode="truncate")
    resp = session.get(
        f"{session.base_url}/view",
        params={"filename": "chunk_0001.mp4", "subfolder": "", "type": "output"},
    )
    assert resp.status_code == 200
    assert 0 < len(resp.content) < 1000


def test_view_disk_full_returns_empty_content():
    session = FakeComfyUISession()
    session.seed_history_success(
        "prompt-0001", video_filename="chunk_0001.mp4", video_bytes=b"data"
    )
    session.corrupt_view_output("chunk_0001.mp4", mode="empty")
    resp = session.get(
        f"{session.base_url}/view",
        params={"filename": "chunk_0001.mp4", "subfolder": "", "type": "output"},
    )
    assert resp.status_code == 200
    assert resp.content == b""


def test_view_for_missing_file_is_404():
    session = FakeComfyUISession()
    resp = session.get(
        f"{session.base_url}/view",
        params={"filename": "nope.mp4", "subfolder": "", "type": "output"},
    )
    assert resp.status_code == 404
    with pytest.raises(requests.HTTPError):
        resp.raise_for_status()


# --------------------------------------------------------------------------- #
# /interrupt, /free, /system_stats
# --------------------------------------------------------------------------- #


def test_interrupt_records_call():
    session = FakeComfyUISession()
    resp = session.post(f"{session.base_url}/interrupt")
    assert resp.status_code == 200
    assert session.interrupt_calls == 1


def test_free_records_unload_models_payload():
    session = FakeComfyUISession()
    resp = session.post(
        f"{session.base_url}/free", json={"unload_models": True, "free_memory": True}
    )
    assert resp.status_code == 200
    assert session.free_calls == [{"unload_models": True, "free_memory": True}]


def test_system_stats_reports_vram():
    session = FakeComfyUISession()
    resp = session.get(f"{session.base_url}/system_stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["devices"][0]["vram_total"] > 0


def test_system_stats_vram_free_is_configurable():
    session = FakeComfyUISession()
    session.set_vram_free(0)
    resp = session.get(f"{session.base_url}/system_stats")
    assert resp.json()["devices"][0]["vram_free"] == 0


# --------------------------------------------------------------------------- #
# ScriptedWebSocket -- happy path
# --------------------------------------------------------------------------- #


def test_scripted_ws_delivers_success_sequence_ending_in_completion_signal():
    prompt_id = "prompt-0001"
    messages = build_success_sequence(prompt_id, node_ids=("3", "7"), progress_steps=2)
    ws = ScriptedWebSocket(messages, client_id="client-abc")
    ws.connect("ws://doris:8188/ws?clientId=client-abc")

    seen_types = []
    completed = False
    while not completed:
        raw = ws.recv()
        msg = json.loads(raw)
        seen_types.append(msg["type"])
        if msg["type"] == "executing" and msg["data"]["node"] is None:
            assert msg["data"]["prompt_id"] == prompt_id
            completed = True

    assert "execution_start" in seen_types
    assert "progress" in seen_types
    assert ws.connect_calls == ["ws://doris:8188/ws?clientId=client-abc"]


def test_scripted_ws_includes_cached_node_message_when_requested():
    messages = build_success_sequence("p1", include_cached=True, cached_nodes=("4",))
    ws = ScriptedWebSocket(messages)
    types = []
    for _ in range(len(messages)):
        types.append(json.loads(ws.recv())["type"])
    assert "execution_cached" in types


def test_ws_factory_connects_and_returns_scripted_ws():
    factory = make_ws_factory(build_success_sequence("p1"), client_id="c1")
    ws = factory("ws://doris:8188/ws?clientId=c1")
    assert isinstance(ws, ScriptedWebSocket)
    assert ws.connect_calls == ["ws://doris:8188/ws?clientId=c1"]


# --------------------------------------------------------------------------- #
# ScriptedWebSocket -- failure modes (issue #10)
# --------------------------------------------------------------------------- #


def test_ws_hang_raises_timeout_without_sleeping():
    ws = ScriptedWebSocket(build_hang_sequence("prompt-0001"))
    # Drain the real messages first.
    ws.recv()
    ws.recv()
    ws.recv()
    with pytest.raises(WebSocketTimeoutException):
        ws.recv()


def test_ws_hang_helper_method_queues_timeout():
    ws = ScriptedWebSocket([executing_message("p1", "3")])
    ws.hang()
    ws.recv()
    with pytest.raises(WebSocketTimeoutException):
        ws.recv()


def test_ws_oom_sequence_carries_realistic_traceback():
    messages = build_oom_sequence("prompt-0001", node_id="7")
    ws = ScriptedWebSocket(messages)
    last = None
    for _ in range(len(messages)):
        last = json.loads(ws.recv())
    assert last["type"] == "execution_error"
    assert "CUDA out of memory" in last["data"]["exception_message"]
    assert last["data"]["node_id"] == "7"


def test_ws_execution_error_message_builder_is_independently_usable():
    msg = json.loads(execution_error_message("p1", node_id="9"))
    assert msg["type"] == "execution_error"
    assert msg["data"]["prompt_id"] == "p1"
    assert "traceback" in msg["data"]


def test_ws_disconnect_mid_stream_raises_connection_closed():
    messages = build_disconnect_sequence("prompt-0001", progress_steps=1)
    ws = ScriptedWebSocket(messages)
    ws.recv()  # status
    ws.recv()  # execution_start
    ws.recv()  # executing
    ws.recv()  # progress
    with pytest.raises(WebSocketConnectionClosedException):
        ws.recv()


def test_ws_disconnect_helper_method():
    ws = ScriptedWebSocket([])
    ws.disconnect("bye")
    with pytest.raises(WebSocketConnectionClosedException):
        ws.recv()


def test_ws_send_after_close_raises():
    ws = ScriptedWebSocket([])
    ws.close()
    assert ws.close_calls == 1
    with pytest.raises(WebSocketConnectionClosedException):
        ws.send("hello")


def test_ws_noise_messages_carry_foreign_prompt_id_for_client_to_ignore():
    real_prompt_id = "prompt-0001"
    noise = build_noise_messages(prompt_id="someone-elses-job", node_id="3")
    real = build_success_sequence(real_prompt_id, node_ids=("3",), progress_steps=1)
    ws = ScriptedWebSocket([*noise, *real])

    own_messages = []
    completed = False
    while not completed:
        msg = json.loads(ws.recv())
        if msg.get("data", {}).get("prompt_id") == real_prompt_id:
            own_messages.append(msg)
            if msg["type"] == "executing" and msg["data"]["node"] is None:
                completed = True

    assert all(m["data"]["prompt_id"] == real_prompt_id for m in own_messages)
    assert len(own_messages) < len(noise) + len(real)


# --------------------------------------------------------------------------- #
# Logging sanity (global standard: every module has a logger)
# --------------------------------------------------------------------------- #


def test_harness_modules_have_loggers():
    from tests.harness import comfyui_mock, ws

    assert isinstance(comfyui_mock.logger, logging.Logger)
    assert isinstance(ws.logger, logging.Logger)
