"""Cross-lane integration: Stage 3 -> Stage 4a -> Stage 4b (issues #7, #8, #9).

Each of those stages was built in its own lane against the mock harness, and
each is well covered on its own. This module covers what none of them can:
the invariants that only exist in the *seams between* them, across more than
one chunk. Those seams are where the expensive failures live -- a graph that
renders the wrong chunk's audio, or a per-chunk mutation that leaks into the
next chunk, costs a full GPU render to discover and looks like a model problem
rather than a plumbing problem.

Still fully offline: :class:`FakeComfyUISession` and :class:`ScriptedWebSocket`
stand in for ComfyUI, so this runs in CI with no GPU, no network, no server.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.execution import ComfyUIExecutionClient
from music_video_maker.staging import ComfyUIAssetStager
from music_video_maker.workflow_graph import WorkflowGraphMutator, find_one_node
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import (
    CLASS_TYPE_AUDIO_LOAD,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    CLASS_TYPE_IMAGE_LOAD,
    CLASS_TYPE_VIDEO_SAVE,
    make_workflow_baseline,
    write_silent_wav,
)
from tests.harness.ws import build_success_sequence, make_ws_factory

CHUNK_COUNT = 2


@pytest.fixture
def staged_run(tmp_path: Path):
    """Two chunks sharing one cast reference photo, run end to end.

    Sharing the photo is the point: it is what makes the Stage 3 upload cache
    observable from the far end of the chain.
    """
    image = tmp_path / "dianne_ref.png"
    image.write_bytes(make_fake_png_bytes(1344, 768))

    session = FakeComfyUISession()
    template = make_workflow_baseline()
    template_before = copy.deepcopy(template)

    stager = ComfyUIAssetStager(base_url=session.base_url, session=session)
    mutator = WorkflowGraphMutator()

    submitted: list[contracts.Workflow] = []
    results: list[contracts.ChunkResult] = []

    for chunk_id in range(CHUNK_COUNT):
        audio = write_silent_wav(tmp_path / f"chunk_{chunk_id:04d}.wav", seconds=5.2)
        chunk = contracts.AudioChunk(
            chunk_id=chunk_id,
            audio_file=audio,
            start=chunk_id * 5.2,
            end=(chunk_id + 1) * 5.2,
            text=f"line {chunk_id}",
            characters=("Dianne",),
        )
        prompt = contracts.ExpandedPrompt(
            chunk_id=chunk_id,
            prompt=f"expanded prompt {chunk_id}",
            image_ref=image,
            characters=("Dianne",),
        )

        assets = stager.stage_chunk(prompt, chunk)
        workflow = mutator.mutate(template, prompt, assets)
        submitted.append(workflow)

        prompt_id = f"prompt-{chunk_id + 1:04d}"
        session.seed_history_success(
            prompt_id, video_filename=f"mvm_chunk_{chunk_id:04d}_00001.mp4"
        )
        client = ComfyUIExecutionClient(
            base_url=session.base_url,
            session=session,
            ws_factory=make_ws_factory(build_success_sequence(prompt_id)),
            client_id=f"client-{chunk_id}",
        )
        results.append(client.execute(workflow, chunk_id=chunk_id, output_dir=tmp_path / "out"))

    return {
        "session": session,
        "template": template,
        "template_before": template_before,
        "submitted": submitted,
        "results": results,
    }


def test_staged_server_filenames_reach_the_workflow(staged_run):
    """Stage 3 returns *server* filenames; Stage 4a must inject those exact
    strings. ComfyUI is sandboxed and cannot resolve a local path, so a local
    path leaking through here fails only at render time, on the GPU."""
    for chunk_id, workflow in enumerate(staged_run["submitted"]):
        _, image_node = find_one_node(workflow, CLASS_TYPE_IMAGE_LOAD)
        _, audio_node = find_one_node(workflow, CLASS_TYPE_AUDIO_LOAD)

        assert image_node["inputs"]["image"] == "dianne_ref.png"
        assert audio_node["inputs"]["audio"] == f"chunk_{chunk_id:04d}.wav"
        assert "/" not in image_node["inputs"]["image"]
        assert "/" not in audio_node["inputs"]["audio"]


def test_shared_cast_photo_uploaded_once_across_chunks(staged_run):
    """The Stage 3 cache, observed from the end of the chain. The mock does not
    dedupe server-side (like real ComfyUI), so a cache miss would surface as a
    second upload and a ``dianne_ref (1).png`` filename in a later chunk's
    graph -- pointing that chunk at a redundant copy."""
    uploads = staged_run["session"].uploads
    originals = [u.original_filename for u in uploads]

    assert originals.count("dianne_ref.png") == 1
    assert all(" (1)" not in u.server_filename for u in uploads)
    # Audio stems are genuinely distinct per chunk and must each upload.
    assert sorted(o for o in originals if o.endswith(".wav")) == [
        f"chunk_{i:04d}.wav" for i in range(CHUNK_COUNT)
    ]


def test_each_chunk_submits_its_own_prompt_and_audio(staged_run):
    """The whole pipeline exists to keep a chunk's audio, prompt and reference
    photo together. Verified on what actually reached ``/prompt``, not on the
    local object, so an in-place mutation or a stale reuse would show up."""
    for chunk_id, submission in enumerate(staged_run["session"].submitted_prompts):
        workflow = submission["workflow"]
        _, h3_node = find_one_node(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
        _, audio_node = find_one_node(workflow, CLASS_TYPE_AUDIO_LOAD)

        assert h3_node["inputs"]["prompt"] == f"expanded prompt {chunk_id}"
        assert audio_node["inputs"]["audio"] == f"chunk_{chunk_id:04d}.wav"
        assert submission["client_id"] == f"client-{chunk_id}"


def test_template_is_never_mutated_in_place(staged_run):
    """Stage 4b reuses one loaded template for every chunk in a run. If Stage
    4a mutated it in place, chunk N's prompt would persist into chunk N+1 and
    every later chunk would render the first chunk's line."""
    assert staged_run["template"] == staged_run["template_before"]


def test_filename_prefix_contract_holds_between_stages(staged_run):
    """The cross-lane contract between #8 and #9: #8 stamps a per-chunk
    SaveVideo prefix so chunk outputs stay distinguishable in ComfyUI's output
    namespace, while #9 names the local file from chunk_id independently."""
    for chunk_id, workflow in enumerate(staged_run["submitted"]):
        _, save_node = find_one_node(workflow, CLASS_TYPE_VIDEO_SAVE)
        assert save_node["inputs"]["filename_prefix"] == f"mvm_chunk_{chunk_id:04d}"


def test_every_chunk_lands_as_its_own_video_file(staged_run):
    """Stage 5 concatenates these in chunk order, so a collision or a
    zero-byte file would silently shorten the finished video."""
    results = staged_run["results"]

    assert [r.status for r in results] == [contracts.ChunkStatus.RENDERED] * CHUNK_COUNT
    assert [r.chunk_id for r in results] == list(range(CHUNK_COUNT))
    assert [r.video_file.name for r in results] == [
        f"chunk_{i:04d}.mp4" for i in range(CHUNK_COUNT)
    ]
    assert len({r.video_file for r in results}) == CHUNK_COUNT
    for result in results:
        assert result.video_file.read_bytes()
        assert result.succeeded
