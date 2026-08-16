"""Tests for Stage 4a graph introspection + mutation (issue #8).

Ground truth for the graph shape is ``docs/h3-node-schema.md`` and
``tests/fixtures/workflows/README.md``: there is no text-encode node (the
prompt is a plain ``STRING`` input on ``MiniMaxH3ReferenceToVideo``), and the
real disambiguation problem in this graph is the two same-``class_type``
``VAELoader`` nodes (video vs audio VAE), not a positive/negative
conditioning pair.

All ``class_type`` strings and workflow shapes come from
``tests.harness.factories`` -- never hardcoded here -- so this suite tracks
the fixtures if the schema is ever reconciled again.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from music_video_maker import contracts, workflow_graph
from music_video_maker.continuity import DEFAULT_CAST_IMAGE_TITLE, DEFAULT_SEED_FRAME_TITLE
from music_video_maker.execution import ComfyUIExecutionClient
from music_video_maker.workflow_graph import (
    CLASS_TYPE_LORA_LOADER,
    WorkflowGraphError,
    find_one_node,
    insert_lora,
)
from tests.harness.comfyui_mock import FakeComfyUISession
from tests.harness.factories import (
    CLASS_TYPE_AUDIO_LOAD,
    CLASS_TYPE_CLIP_LOADER,
    CLASS_TYPE_H3_IMAGE_TO_VIDEO,
    CLASS_TYPE_H3_REFERENCE_TO_VIDEO,
    CLASS_TYPE_IMAGE_LOAD,
    CLASS_TYPE_NOISE,
    CLASS_TYPE_SAMPLER_SELECT,
    CLASS_TYPE_SCHEDULER,
    CLASS_TYPE_UNET_LOADER,
    CLASS_TYPE_VAE_LOADER,
    CLASS_TYPE_VIDEO_SAVE,
    CLOUD_API_CLASS_TYPES,
    WEIGHT_TEXT_ENCODER,
    WORKFLOW_AMBIGUOUS_VAE_LOADERS_PATH,
    WORKFLOW_BASELINE_PATH,
    WORKFLOW_I2V_PATH,
    WORKFLOW_SHIFTED_IDS_PATH,
    make_workflow_ambiguous_vae_loaders,
    make_workflow_baseline,
    make_workflow_i2v,
    make_workflow_missing_node,
    make_workflow_shifted_ids,
)
from tests.harness.ws import build_success_sequence, make_ws_factory


def _prompt(
    chunk_id: int = 7, text: str = "a fully expanded test prompt"
) -> contracts.ExpandedPrompt:
    return contracts.ExpandedPrompt(
        chunk_id=chunk_id,
        prompt=text,
        image_ref=Path("cast/dianne_ref.png"),
        characters=("Dianne",),
    )


def _assets(chunk_id: int = 7) -> contracts.StagedAssets:
    return contracts.StagedAssets(
        chunk_id=chunk_id,
        image_filename="staged_image_0007.png",
        audio_filename="staged_audio_0007.wav",
    )


def _assets_multi(
    chunk_id: int = 7, image_filenames: tuple[str, ...] = ()
) -> contracts.StagedAssets:
    """A :class:`~music_video_maker.contracts.StagedAssets` carrying more
    than one staged reference filename (issue #33 levels 1-2).
    ``image_filenames[0]`` is expected to equal ``image_filename`` (the
    staging-side mirror of ``ExpandedPrompt.image_ref``/``image_refs``)."""
    return contracts.StagedAssets(
        chunk_id=chunk_id,
        image_filename=image_filenames[0] if image_filenames else "staged_image_0007.png",
        audio_filename="staged_audio_0007.wav",
        image_filenames=image_filenames,
    )


# --------------------------------------------------------------------------- #
# load_workflow_template
# --------------------------------------------------------------------------- #


def test_load_workflow_template_reads_committed_baseline_fixture():
    workflow = workflow_graph.load_workflow_template(WORKFLOW_BASELINE_PATH)
    assert workflow == make_workflow_baseline()


def test_load_workflow_template_missing_file_raises_and_logs(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.json"
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.WorkflowTemplateError, match=str(missing)),
    ):
        workflow_graph.load_workflow_template(missing)
    assert any("does_not_exist.json" in r.message for r in caplog.records)


def test_load_workflow_template_malformed_json_raises_and_logs(tmp_path, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.WorkflowTemplateError):
        workflow_graph.load_workflow_template(bad)
    assert any("bad.json" in r.message for r in caplog.records)


def test_load_workflow_template_non_dict_payload_raises_and_logs(tmp_path, caplog):
    not_a_dict = tmp_path / "list_payload.json"
    not_a_dict.write_text(json.dumps(["not", "a", "workflow"]), encoding="utf-8")
    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.WorkflowTemplateError):
        workflow_graph.load_workflow_template(not_a_dict)
    assert any("list_payload.json" in r.message for r in caplog.records)


def test_load_workflow_template_node_missing_class_type_raises(tmp_path):
    malformed = tmp_path / "malformed_node.json"
    malformed.write_text(json.dumps({"10": {"inputs": {}}}), encoding="utf-8")
    with pytest.raises(workflow_graph.WorkflowTemplateError, match="10"):
        workflow_graph.load_workflow_template(malformed)


# --------------------------------------------------------------------------- #
# find_nodes_by_class_type / find_one_node
# --------------------------------------------------------------------------- #


def test_find_nodes_by_class_type_returns_all_matches_deterministically():
    workflow = make_workflow_ambiguous_vae_loaders()
    matches = workflow_graph.find_nodes_by_class_type(workflow, CLASS_TYPE_VAE_LOADER)
    assert len(matches) == 2
    # Deterministic: repeated calls return the identical ordering.
    matches_again = workflow_graph.find_nodes_by_class_type(workflow, CLASS_TYPE_VAE_LOADER)
    assert [nid for nid, _ in matches] == [nid for nid, _ in matches_again]


def test_find_nodes_by_class_type_no_match_returns_empty_list():
    workflow = make_workflow_baseline()
    assert workflow_graph.find_nodes_by_class_type(workflow, "SomeNonexistentClassType") == []


def test_find_one_node_locates_single_match():
    workflow = make_workflow_baseline()
    node_id, node = workflow_graph.find_one_node(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert node["class_type"] == CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    assert workflow[node_id] is node


def test_find_one_node_absent_class_type_raises_and_logs(caplog):
    workflow = make_workflow_baseline()
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.NodeNotFoundError, match="NoSuchNode"),
    ):
        workflow_graph.find_one_node(workflow, "NoSuchNode")
    assert any("NoSuchNode" in r.message for r in caplog.records)


def test_find_one_node_ambiguous_without_disambiguator_raises_with_candidate_ids(caplog):
    workflow = make_workflow_ambiguous_vae_loaders()
    matches = workflow_graph.find_nodes_by_class_type(workflow, CLASS_TYPE_VAE_LOADER)
    expected_ids = sorted(nid for nid, _ in matches)

    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.AmbiguousNodeError) as excinfo:
        workflow_graph.find_one_node(workflow, CLASS_TYPE_VAE_LOADER)

    assert CLASS_TYPE_VAE_LOADER in str(excinfo.value)
    for node_id in expected_ids:
        assert node_id in str(excinfo.value)
    assert any(CLASS_TYPE_VAE_LOADER in r.message for r in caplog.records)


def test_find_one_node_disambiguated_by_title_on_baseline():
    workflow = make_workflow_baseline()
    node_id, node = workflow_graph.find_one_node(
        workflow, CLASS_TYPE_VAE_LOADER, title="Load Audio VAE"
    )
    assert node["_meta"]["title"] == "Load Audio VAE"
    assert workflow[node_id] is node


def test_find_one_node_title_disambiguator_does_not_help_when_titles_collide(caplog):
    # ambiguous_vae_h3.json gives both VAELoader nodes the same default title,
    # so passing that title as a disambiguator must still raise ambiguity.
    workflow = make_workflow_ambiguous_vae_loaders()
    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.AmbiguousNodeError):
        workflow_graph.find_one_node(workflow, CLASS_TYPE_VAE_LOADER, title=CLASS_TYPE_VAE_LOADER)


# --------------------------------------------------------------------------- #
# wiring-based disambiguation
# --------------------------------------------------------------------------- #


def test_resolve_upstream_by_wiring_finds_video_vae_via_ambiguous_titles():
    workflow = make_workflow_ambiguous_vae_loaders()
    node_id, node = workflow_graph.resolve_upstream_node(
        workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "vae"
    )
    assert node["class_type"] == CLASS_TYPE_VAE_LOADER
    assert node["inputs"]["vae_name"] == "minimax_h3_video_vae_fp16.safetensors"


def test_resolve_upstream_by_wiring_finds_audio_vae_via_ambiguous_titles():
    workflow = make_workflow_ambiguous_vae_loaders()
    node_id, node = workflow_graph.resolve_upstream_node(
        workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "audio_vae"
    )
    assert node["class_type"] == CLASS_TYPE_VAE_LOADER
    assert node["inputs"]["vae_name"] == "minimax_h3_audio_vae_fp32.safetensors"


def test_resolve_upstream_by_wiring_video_and_audio_vae_are_distinct_nodes():
    workflow = make_workflow_ambiguous_vae_loaders()
    video_id, _ = workflow_graph.resolve_upstream_node(
        workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "vae"
    )
    audio_id, _ = workflow_graph.resolve_upstream_node(
        workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "audio_vae"
    )
    assert video_id != audio_id


def test_resolve_upstream_by_wiring_handles_autogrow_dotted_key():
    # ref_images is a COMFY_AUTOGROW_V3 input; each populated slot serializes
    # as its own flat dotted key ("ref_images.ref_image_0") holding a plain
    # [node_id, output_index] link, confirmed against ComfyUI's own dynamic-
    # input expansion code and a real render (issue #18).
    workflow = make_workflow_baseline()
    node_id, node = workflow_graph.resolve_upstream_node(
        workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "ref_images.ref_image_0"
    )
    assert node["class_type"] == CLASS_TYPE_IMAGE_LOAD


def test_resolve_upstream_node_unknown_input_name_raises_and_logs(caplog):
    workflow = make_workflow_baseline()
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.DanglingNodeReferenceError, match="no_such_input"),
    ):
        workflow_graph.resolve_upstream_node(
            workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "no_such_input"
        )
    assert any("no_such_input" in r.message for r in caplog.records)


def test_resolve_upstream_node_scalar_input_is_not_a_reference_raises_and_logs(caplog):
    workflow = make_workflow_baseline()
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.DanglingNodeReferenceError, match="width"),
    ):
        # "width" is a plain INT input (1344 by default), not a node reference.
        workflow_graph.resolve_upstream_node(workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "width")
    assert any("width" in r.message for r in caplog.records)


def test_resolve_upstream_by_wiring_dangling_reference_raises_and_logs(caplog):
    # LoadAudio is removed entirely; ref_audios.ref_audio_0 on the H3 node
    # still points at it (the fixture leaves the reference dangling on
    # purpose).
    workflow = make_workflow_missing_node(omit=CLASS_TYPE_AUDIO_LOAD)
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.DanglingNodeReferenceError),
    ):
        workflow_graph.resolve_upstream_node(
            workflow, CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "ref_audios.ref_audio_0"
        )


# --------------------------------------------------------------------------- #
# cloud-API validation
# --------------------------------------------------------------------------- #


def test_assert_no_cloud_api_nodes_passes_on_clean_baseline():
    workflow_graph.assert_no_cloud_api_nodes(make_workflow_baseline())  # must not raise


@pytest.mark.parametrize("cloud_class_type", CLOUD_API_CLASS_TYPES)
def test_assert_no_cloud_api_nodes_rejects_each_cloud_variant(cloud_class_type, caplog):
    workflow = make_workflow_baseline()
    workflow["999"] = {
        "class_type": cloud_class_type,
        "inputs": {},
        "_meta": {"title": "sneaky cloud node"},
    }
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.CloudApiNodeDetectedError, match=cloud_class_type),
    ):
        workflow_graph.assert_no_cloud_api_nodes(workflow)
    assert any(cloud_class_type in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# WorkflowGraphMutator.mutate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make_workflow",
    [make_workflow_baseline, make_workflow_shifted_ids],
    ids=["baseline_ids", "shifted_ids"],
)
def test_mutate_injects_prompt_image_audio_and_prefix_regardless_of_node_ids(make_workflow):
    template = make_workflow()
    mutator = workflow_graph.WorkflowGraphMutator()
    prompt = _prompt(chunk_id=7, text="a fully expanded test prompt")
    assets = _assets(chunk_id=7)

    result = mutator.mutate(template, prompt, assets)

    _, h3_node = workflow_graph.find_one_node(result, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    _, image_node = workflow_graph.find_one_node(result, CLASS_TYPE_IMAGE_LOAD)
    _, audio_node = workflow_graph.find_one_node(result, CLASS_TYPE_AUDIO_LOAD)
    _, save_node = workflow_graph.find_one_node(result, CLASS_TYPE_VIDEO_SAVE)

    assert h3_node["inputs"]["prompt"] == "a fully expanded test prompt"
    assert image_node["inputs"]["image"] == "staged_image_0007.png"
    assert audio_node["inputs"]["audio"] == "staged_audio_0007.wav"
    assert save_node["inputs"]["filename_prefix"] == "mvm_chunk_0007"


def test_mutate_deep_copies_template_leaving_original_untouched():
    template = make_workflow_baseline()
    original = json.loads(json.dumps(template))  # independent, structurally-equal snapshot

    mutator = workflow_graph.WorkflowGraphMutator()
    result = mutator.mutate(template, _prompt(), _assets())

    assert template == original
    assert result != template  # the mutated output actually differs
    # Provably a different object graph, not just equal-by-luck.
    h3_id, _ = workflow_graph.find_one_node(template, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert result[h3_id] is not template[h3_id]
    assert result[h3_id]["inputs"] is not template[h3_id]["inputs"]


def test_mutate_missing_required_node_raises_loudly_and_logs(caplog):
    template = make_workflow_missing_node(omit=CLASS_TYPE_AUDIO_LOAD)
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.NodeNotFoundError, match=CLASS_TYPE_AUDIO_LOAD),
    ):
        mutator.mutate(template, _prompt(), _assets())

    assert any(CLASS_TYPE_AUDIO_LOAD in r.message for r in caplog.records)


def test_mutate_rejects_template_containing_a_cloud_api_node(caplog):
    template = make_workflow_baseline()
    template["999"] = {
        "class_type": "MinimaxTextToVideoNode",
        "inputs": {},
        "_meta": {"title": "sneaky cloud node"},
    }
    mutator = workflow_graph.WorkflowGraphMutator()

    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.CloudApiNodeDetectedError):
        mutator.mutate(template, _prompt(), _assets())


def test_mutate_seed_frame_filename_logs_and_is_ignored(caplog):
    template = make_workflow_baseline()
    assets = contracts.StagedAssets(
        chunk_id=7,
        image_filename="staged_image_0007.png",
        audio_filename="staged_audio_0007.wav",
        seed_frame_filename="seed_0006_last_frame.png",
    )
    mutator = workflow_graph.WorkflowGraphMutator()

    with caplog.at_level("DEBUG"):
        result = mutator.mutate(template, _prompt(chunk_id=7), assets)

    # Out of scope for #8 (I2V bridging is issue #12) -- must not silently
    # vanish without a trace, and must not touch the graph.
    assert any("seed_0006_last_frame.png" in r.message for r in caplog.records)
    _, image_node = workflow_graph.find_one_node(result, CLASS_TYPE_IMAGE_LOAD)
    assert image_node["inputs"]["image"] == "staged_image_0007.png"


def test_mutate_without_seed_frame_filename_does_not_log_about_it(caplog):
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    with caplog.at_level("DEBUG"):
        mutator.mutate(template, _prompt(), _assets())

    assert not any("seed_frame" in r.message for r in caplog.records)


def test_mutate_applies_node_input_overrides():
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(
        template,
        _prompt(),
        _assets(),
        node_input_overrides={
            CLASS_TYPE_H3_REFERENCE_TO_VIDEO: {"width": 1920, "height": 1080, "seed": 42},
        },
    )

    _, h3_node = workflow_graph.find_one_node(result, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert h3_node["inputs"]["width"] == 1920
    assert h3_node["inputs"]["height"] == 1080
    assert h3_node["inputs"]["seed"] == 42
    # The mandatory injections still happened alongside the overrides.
    assert h3_node["inputs"]["prompt"] == _prompt().prompt


def test_mutate_node_input_overrides_unknown_class_type_raises_clearly(caplog):
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.NodeNotFoundError, match="NoSuchClassType"),
    ):
        mutator.mutate(
            template,
            _prompt(),
            _assets(),
            node_input_overrides={"NoSuchClassType": {"foo": "bar"}},
        )

    assert any("NoSuchClassType" in r.message for r in caplog.records)


def test_mutate_does_not_implement_frame_quantization():
    # #8 must not silently compute `length` -- that policy belongs to #20.
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(template, _prompt(), _assets())

    _, h3_node = workflow_graph.find_one_node(result, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    _, original_h3_node = workflow_graph.find_one_node(template, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert h3_node["inputs"]["length"] == original_h3_node["inputs"]["length"]


def test_mutator_satisfies_workflow_mutator_protocol():
    assert isinstance(workflow_graph.WorkflowGraphMutator(), contracts.WorkflowMutator)


# --------------------------------------------------------------------------- #
# Committed fixture round-trip through the mutator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [WORKFLOW_BASELINE_PATH, WORKFLOW_SHIFTED_IDS_PATH],
    ids=["baseline_h3.json", "shifted_ids_h3.json"],
)
def test_mutate_end_to_end_from_disk_fixture(path):
    template = workflow_graph.load_workflow_template(path)
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(template, _prompt(chunk_id=3), _assets(chunk_id=3))

    _, save_node = workflow_graph.find_one_node(result, CLASS_TYPE_VIDEO_SAVE)
    assert save_node["inputs"]["filename_prefix"] == "mvm_chunk_0003"


def test_mutate_on_ambiguous_vae_fixture_still_succeeds_since_mutate_never_touches_vae():
    # mutate() only injects prompt/image/audio/filename_prefix -- none of
    # which are ambiguous class_types -- so the VAELoader ambiguity must not
    # block a full mutate() call even though find_one_node(VAELoader) alone
    # would raise.
    template = load_ambiguous_vae_template()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(template, _prompt(), _assets())

    _, h3_node = workflow_graph.find_one_node(result, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert h3_node["inputs"]["prompt"] == _prompt().prompt


def test_mutate_rejects_mismatched_chunk_ids(caplog):
    """Guards the index-space confusion that produced a real bug in #5: lyric
    line, aligned segment, and chunk indices are all plain ints, so a mix-up
    type-checks fine and surfaces only as the wrong face lip-syncing the wrong
    line in the finished video."""
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.ChunkMismatchError, match="chunk id mismatch"),
    ):
        mutator.mutate(template, _prompt(chunk_id=7), _assets(chunk_id=8))

    assert "mismatched chunks" in caplog.text


def load_ambiguous_vae_template():
    return workflow_graph.load_workflow_template(WORKFLOW_AMBIGUOUS_VAE_LOADERS_PATH)


# --------------------------------------------------------------------------- #
# I2V continuity additions (issue #12) -- additive only, existing behavior
# above is unchanged.
# --------------------------------------------------------------------------- #


def test_mutate_on_i2v_template_injects_prompt_into_image_to_video_node():
    # The I2V template carries MiniMaxH3ImageToVideo, not
    # MiniMaxH3ReferenceToVideo -- mutate() must still find the node that
    # owns `prompt` (issue #12's dual-node lookup, _find_h3_conditioning_node).
    template = make_workflow_i2v()
    mutator = workflow_graph.WorkflowGraphMutator()
    prompt = _prompt(chunk_id=3, text="a bridged i2v prompt")
    assets = contracts.StagedAssets(
        chunk_id=3,
        image_filename="staged_image_0003.png",
        audio_filename="staged_audio_0003.wav",
        seed_frame_filename="seed_0002_into_0003.png",
    )

    result = mutator.mutate(
        template,
        prompt,
        assets,
        cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
        seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
    )

    _, i2v_node = workflow_graph.find_one_node(result, CLASS_TYPE_H3_IMAGE_TO_VIDEO)
    assert i2v_node["inputs"]["prompt"] == "a bridged i2v prompt"

    _, cast_node = workflow_graph.find_one_node(
        result, CLASS_TYPE_IMAGE_LOAD, title=DEFAULT_CAST_IMAGE_TITLE
    )
    assert cast_node["inputs"]["image"] == "staged_image_0003.png"

    _, seed_node = workflow_graph.find_one_node(
        result, CLASS_TYPE_IMAGE_LOAD, title=DEFAULT_SEED_FRAME_TITLE
    )
    assert seed_node["inputs"]["image"] == "seed_0002_into_0003.png"

    # The two LoadImage nodes must remain distinct.
    assert cast_node is not seed_node


def test_mutate_on_i2v_template_without_cast_title_raises_ambiguous(caplog):
    # The I2V template has two LoadImage nodes; omitting cast_image_title
    # must not silently pick one -- it has to raise, same as any other
    # unresolved ambiguity in this module.
    template = make_workflow_i2v()
    mutator = workflow_graph.WorkflowGraphMutator()
    assets = contracts.StagedAssets(
        chunk_id=3,
        image_filename="staged_image_0003.png",
        audio_filename="staged_audio_0003.wav",
        seed_frame_filename="seed_0002_into_0003.png",
    )

    with caplog.at_level("ERROR"), pytest.raises(workflow_graph.AmbiguousNodeError):
        mutator.mutate(
            template,
            _prompt(chunk_id=3),
            assets,
            seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
        )


def test_mutate_seed_frame_title_given_but_no_seed_filename_logs_warning_and_skips(caplog):
    template = make_workflow_i2v()
    mutator = workflow_graph.WorkflowGraphMutator()
    assets = contracts.StagedAssets(
        chunk_id=3,
        image_filename="staged_image_0003.png",
        audio_filename="staged_audio_0003.wav",
        # seed_frame_filename intentionally left unset
    )

    with caplog.at_level("WARNING"):
        result = mutator.mutate(
            template,
            _prompt(chunk_id=3),
            assets,
            cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
            seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
        )

    assert any("seed_frame_title" in r.message for r in caplog.records)
    _, seed_node = workflow_graph.find_one_node(
        result, CLASS_TYPE_IMAGE_LOAD, title=DEFAULT_SEED_FRAME_TITLE
    )
    # Untouched -- still the fixture's placeholder, not overwritten with junk.
    assert seed_node["inputs"]["image"] == "seed_placeholder.png"


def test_mutate_both_h3_conditioning_nodes_present_raises_ambiguous(caplog):
    # A malformed template that somehow carries both the T2V/reference node
    # and the I2V node -- exactly one is ever expected.
    template = make_workflow_baseline()
    ref_id, ref_node = workflow_graph.find_one_node(template, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    stray_i2v = dict(ref_node)
    stray_i2v["class_type"] = CLASS_TYPE_H3_IMAGE_TO_VIDEO
    stray_i2v["_meta"] = {"title": "stray I2V node"}
    template["999"] = stray_i2v
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.AmbiguousNodeError) as excinfo,
    ):
        mutator.mutate(template, _prompt(), _assets())

    assert CLASS_TYPE_H3_REFERENCE_TO_VIDEO in str(excinfo.value)
    assert CLASS_TYPE_H3_IMAGE_TO_VIDEO in str(excinfo.value)


def test_mutate_missing_both_h3_conditioning_nodes_raises_naming_both(caplog):
    template = make_workflow_baseline()
    h3_id, _ = workflow_graph.find_one_node(template, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    template = {nid: n for nid, n in template.items() if nid != h3_id}
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.NodeNotFoundError) as excinfo,
    ):
        mutator.mutate(template, _prompt(), _assets())

    assert CLASS_TYPE_H3_REFERENCE_TO_VIDEO in str(excinfo.value)
    assert CLASS_TYPE_H3_IMAGE_TO_VIDEO in str(excinfo.value)


def test_mutate_baseline_template_unaffected_by_new_optional_kwargs():
    # Regression guard: calling mutate() on the pre-existing baseline
    # template with the new kwargs at their defaults must produce the exact
    # same result as the pre-#12 call signature.
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    old_style = mutator.mutate(template, _prompt(chunk_id=5), _assets(chunk_id=5))
    new_style = mutator.mutate(
        template,
        _prompt(chunk_id=5),
        _assets(chunk_id=5),
        cast_image_title=None,
        seed_frame_title=None,
    )

    assert old_style == new_style


def test_i2v_fixture_on_disk_matches_factory_output():
    assert WORKFLOW_I2V_PATH.is_file(), f"missing fixture file {WORKFLOW_I2V_PATH}"
    from tests.harness.factories import load_workflow_fixture

    on_disk = load_workflow_fixture(WORKFLOW_I2V_PATH)
    assert on_disk == make_workflow_i2v()


def test_mutate_end_to_end_from_i2v_disk_fixture():
    template = workflow_graph.load_workflow_template(WORKFLOW_I2V_PATH)
    mutator = workflow_graph.WorkflowGraphMutator()
    assets = contracts.StagedAssets(
        chunk_id=9,
        image_filename="staged_image_0009.png",
        audio_filename="staged_audio_0009.wav",
        seed_frame_filename="seed_0008_into_0009.png",
    )

    result = mutator.mutate(
        template,
        _prompt(chunk_id=9),
        assets,
        cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
        seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
    )

    _, save_node = workflow_graph.find_one_node(result, CLASS_TYPE_VIDEO_SAVE)
    assert save_node["inputs"]["filename_prefix"] == "mvm_chunk_0009"


def test_explicit_title_is_enforced_not_merely_a_tiebreaker(caplog):
    """An explicitly-requested title must actually match the node found.

    ``find_one_node`` returns the sole candidate without consulting ``title``
    when only one node of that ``class_type`` exists -- correct as a
    *disambiguator*, but the mutator uses the title as an *identity* claim.
    On an I2V template authored with only one ``LoadImage`` (the real one is
    still unauthored -- issue #18 -- and the titles are a convention, not
    something ComfyUI enforces), both the cast-photo and the seed-frame
    lookups resolve to that same node, so the seed frame silently overwrites
    the cast reference photo and the render carries no cast face at all.
    Silent, and only visible after the GPU hours are spent.
    """
    template = {
        "1": {"class_type": workflow_graph.CLASS_TYPE_H3_IMAGE_TO_VIDEO, "inputs": {"prompt": ""}},
        "2": {
            "class_type": CLASS_TYPE_IMAGE_LOAD,
            "inputs": {"image": ""},
            "_meta": {"title": DEFAULT_CAST_IMAGE_TITLE},
        },
        "3": {"class_type": CLASS_TYPE_AUDIO_LOAD, "inputs": {"audio": ""}},
        "4": {"class_type": CLASS_TYPE_VIDEO_SAVE, "inputs": {"filename_prefix": ""}},
    }
    assets = contracts.StagedAssets(
        chunk_id=3,
        image_filename="cast_0003.png",
        audio_filename="audio_0003.wav",
        seed_frame_filename="seed_0002_into_0003.png",
    )

    with pytest.raises(workflow_graph.NodeNotFoundError, match=DEFAULT_SEED_FRAME_TITLE):
        workflow_graph.WorkflowGraphMutator().mutate(
            template,
            _prompt(chunk_id=3),
            assets,
            cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
            seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
        )


def test_explicit_cast_title_that_matches_nothing_raises():
    """Same identity claim, checked on the cast-photo side."""
    template = {
        "1": {"class_type": workflow_graph.CLASS_TYPE_H3_IMAGE_TO_VIDEO, "inputs": {"prompt": ""}},
        "2": {
            "class_type": CLASS_TYPE_IMAGE_LOAD,
            "inputs": {"image": ""},
            "_meta": {"title": "Something Else Entirely"},
        },
        "3": {"class_type": CLASS_TYPE_AUDIO_LOAD, "inputs": {"audio": ""}},
        "4": {"class_type": CLASS_TYPE_VIDEO_SAVE, "inputs": {"filename_prefix": ""}},
    }
    assets = contracts.StagedAssets(
        chunk_id=3, image_filename="cast_0003.png", audio_filename="audio_0003.wav"
    )

    with pytest.raises(workflow_graph.NodeNotFoundError, match=DEFAULT_CAST_IMAGE_TITLE):
        workflow_graph.WorkflowGraphMutator().mutate(
            template, _prompt(chunk_id=3), assets, cast_image_title=DEFAULT_CAST_IMAGE_TITLE
        )


# --------------------------------------------------------------------------- #
# Issue #20's quantized `length` injection.
#
# mutate() injects a frame count it is *given*; it never computes one. What it
# does own is refusing a value the H3 node cannot honor exactly -- a value off
# the grid would be silently rounded server-side while the chunk's audio stem
# kept its own duration, which is the drift #20 exists to eliminate.
# --------------------------------------------------------------------------- #


def _h3_inputs(workflow):
    _, node = workflow_graph._find_h3_conditioning_node(workflow)
    return node["inputs"]


def test_length_frames_is_injected_into_the_h3_node():
    template = make_workflow_baseline()
    mutated = workflow_graph.WorkflowGraphMutator().mutate(
        template, _prompt(chunk_id=3), _assets(chunk_id=3), length_frames=141
    )
    assert _h3_inputs(mutated)["length"] == 141
    # The caller's template is never mutated in place.
    assert _h3_inputs(template)["length"] == 124


def test_length_frames_none_leaves_the_templates_own_length_alone():
    template = make_workflow_baseline()
    mutated = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), _assets())
    assert _h3_inputs(mutated)["length"] == 124


@pytest.mark.parametrize("bad", [125, 130, 200, 363])
def test_length_frames_off_the_grid_is_refused(bad, caplog):
    """ComfyUI quantizes `length` itself, so an off-grid value is not an
    error there -- it is silently rounded. That silent rounding is exactly
    the defect, so it must fail here instead."""
    with caplog.at_level(logging.ERROR), pytest.raises(workflow_graph.InvalidLengthError):
        workflow_graph.WorkflowGraphMutator().mutate(
            make_workflow_baseline(),
            _prompt(chunk_id=7),
            _assets(chunk_id=7),
            length_frames=bad,
        )
    assert "7" in caplog.text


@pytest.mark.parametrize("out_of_range", [5, 22, 107, 379, 396])
def test_length_frames_outside_the_trained_range_is_refused(out_of_range):
    """These are all valid grid points -- ComfyUI would accept every one of
    them. They are refused because the model was only trained on 124-362."""
    assert contracts.H3_FRAME_GRID.is_valid(out_of_range)
    with pytest.raises(workflow_graph.InvalidLengthError, match="trained range"):
        workflow_graph.WorkflowGraphMutator().mutate(
            make_workflow_baseline(), _prompt(), _assets(), length_frames=out_of_range
        )


# --------------------------------------------------------------------------- #
# More than one reference photo (issue #33 levels 1-2)
#
# H3's `ref_images` input is COMFY_AUTOGROW_V3 -- a growable list of flat,
# dotted-key slots (`ref_images.ref_image_0`, `ref_image_1`, ...), each
# holding a plain [node_id, output_index] link to its own LoadImage node
# (docs/h3-node-schema.md finding 3). The mutator never writes into
# `ref_images` itself; extra slots get a freshly synthesized LoadImage node
# each, never a second node added to the committed template -- see
# workflow_graph.py's module docstring for why (continuity.py's
# `_render_base` relies on exactly one LoadImage node existing in the base
# template to resolve unambiguously with no title at all).
# --------------------------------------------------------------------------- #


def test_single_reference_is_a_complete_no_op_for_the_new_code_path():
    """`image_filenames=()` (the default StagedAssets shape) must produce a
    graph identical, node-for-node, to what this mutator returned before
    issue #33 -- the whole point of the additive design."""
    template = make_workflow_baseline()
    mutated = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), _assets())

    assert set(mutated) == set(template)
    _, h3_node = workflow_graph.find_one_node(mutated, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    ref_keys = [k for k in h3_node["inputs"] if k.startswith("ref_images.")]
    assert ref_keys == ["ref_images.ref_image_0"]


def test_single_entry_image_filenames_behaves_identically_to_empty():
    """A one-element `image_filenames` (primary only, no extra characters)
    must not be treated differently from the default `()` -- both mean "just
    the primary"."""
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    empty = mutator.mutate(template, _prompt(), _assets(chunk_id=7))
    single = mutator.mutate(
        template, _prompt(), _assets_multi(chunk_id=7, image_filenames=("staged_image_0007.png",))
    )

    assert empty == single


def test_second_reference_photo_gets_its_own_load_image_node_and_slot():
    template = make_workflow_baseline()
    orig_image_id, _ = workflow_graph.find_one_node(template, CLASS_TYPE_IMAGE_LOAD)
    assets = _assets_multi(
        chunk_id=7, image_filenames=("staged_image_0007.png", "staged_jan_ref.png")
    )

    mutated = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), assets)

    new_ids = set(mutated) - set(template)
    assert len(new_ids) == 1
    new_id = new_ids.pop()
    assert mutated[new_id]["class_type"] == CLASS_TYPE_IMAGE_LOAD
    assert mutated[new_id]["inputs"]["image"] == "staged_jan_ref.png"

    _, h3_node = workflow_graph.find_one_node(mutated, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert h3_node["inputs"]["ref_images.ref_image_0"] == [orig_image_id, 0]
    assert h3_node["inputs"]["ref_images.ref_image_1"] == [new_id, 0]
    # The primary photo's own node and slot are completely untouched.
    assert mutated[orig_image_id]["inputs"]["image"] == "staged_image_0007.png"


def test_three_or_more_references_get_one_new_node_and_slot_each():
    template = make_workflow_baseline()
    assets = _assets_multi(
        chunk_id=7,
        image_filenames=("staged_image_0007.png", "staged_jan_ref.png", "staged_rex_ref.png"),
    )

    mutated = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), assets)

    new_ids = set(mutated) - set(template)
    assert len(new_ids) == 2
    _, h3_node = workflow_graph.find_one_node(mutated, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    slot_1_id, _ = h3_node["inputs"]["ref_images.ref_image_1"]
    slot_2_id, _ = h3_node["inputs"]["ref_images.ref_image_2"]
    assert {slot_1_id, slot_2_id} == new_ids
    assert mutated[slot_1_id]["inputs"]["image"] == "staged_jan_ref.png"
    assert mutated[slot_2_id]["inputs"]["image"] == "staged_rex_ref.png"


def test_extra_reference_node_ids_never_collide_with_renumbered_template_ids():
    """Node lookup and the new node's freshly allocated id both stay correct
    when a user has renumbered every id on the canvas -- never hardcoded."""
    template = make_workflow_shifted_ids()
    assets = _assets_multi(
        chunk_id=7, image_filenames=("staged_image_0007.png", "staged_jan_ref.png")
    )

    mutated = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), assets)

    new_ids = set(mutated) - set(template)
    assert len(new_ids) == 1
    new_id = new_ids.pop()
    assert new_id not in template
    _, h3_node = workflow_graph.find_one_node(mutated, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    assert h3_node["inputs"]["ref_images.ref_image_1"] == [new_id, 0]
    assert mutated[new_id]["inputs"]["image"] == "staged_jan_ref.png"


def test_second_reference_on_baseline_leaves_the_template_object_untouched():
    template = make_workflow_baseline()
    original = json.loads(json.dumps(template))
    assets = _assets_multi(
        chunk_id=7, image_filenames=("staged_image_0007.png", "staged_jan_ref.png")
    )

    workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), assets)

    assert template == original


def test_extra_reference_on_i2v_template_logs_warning_and_renders_primary_only(caplog):
    """MiniMaxH3ImageToVideo has no `ref_images` input at all -- a chunk that
    reaches mutate() with extra filenames while rendering through the I2V
    template must not crash; it degrades to primary-only, exactly as if
    issue #33's second reference had never been requested."""
    template = make_workflow_i2v()
    original_node_ids = set(template)
    assets = _assets_multi(
        chunk_id=7, image_filenames=("staged_image_0007.png", "staged_jan_ref.png")
    )

    with caplog.at_level(logging.WARNING):
        mutated = workflow_graph.WorkflowGraphMutator().mutate(
            template,
            _prompt(),
            assets,
            cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
            seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
        )

    assert set(mutated) == original_node_ids
    assert "ref_images" in caplog.text
    assert "MiniMaxH3ImageToVideo" in caplog.text
    _, cast_node = workflow_graph.find_titled_node(
        mutated, CLASS_TYPE_IMAGE_LOAD, DEFAULT_CAST_IMAGE_TITLE
    )
    assert cast_node["inputs"]["image"] == "staged_image_0007.png"


# --------------------------------------------------------------------------- #
# The noise seed (issue #38, recording half)
#
# ``RandomNoise.noise_seed`` is the last input that decides the pixels, and it
# was the one input nothing in the pipeline ever set, recorded or checked. The
# mutator now writes it on every chunk from an explicit parameter, so a take is
# reproducible and a template edit cannot silently change what renders.
# Located by ``class_type`` like everything else here -- the committed
# templates happen to number it "10", which is exactly the sort of fact a
# canvas edit invalidates.
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]


def _noise_inputs(workflow: contracts.Workflow) -> dict:
    _, node = workflow_graph.find_one_node(workflow, CLASS_TYPE_NOISE)
    return node["inputs"]


def test_mutate_injects_the_requested_noise_seed():
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(template, _prompt(), _assets(), noise_seed=1234567)

    assert _noise_inputs(result)["noise_seed"] == 1234567


def test_mutate_finds_the_noise_node_after_a_canvas_renumbering():
    template = make_workflow_shifted_ids()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(template, _prompt(), _assets(), noise_seed=99)

    assert _noise_inputs(result)["noise_seed"] == 99


def test_mutate_seeds_the_i2v_template_too():
    """Both authored templates carry a ``RandomNoise`` node, so the I2V
    continuity path must be seeded identically -- a chunk that renders through
    the bridge is no less a take than one that does not."""
    template = make_workflow_i2v()
    mutator = workflow_graph.WorkflowGraphMutator()

    result = mutator.mutate(
        template,
        _prompt(),
        _assets(),
        cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
        seed_frame_title=DEFAULT_SEED_FRAME_TITLE,
        noise_seed=555,
    )

    assert _noise_inputs(result)["noise_seed"] == 555


def test_mutate_overwrites_whatever_seed_the_template_happened_to_carry():
    """The defect in one test: the seed must come from the run, not from the
    template file. An edit on the canvas that changes it must not reach the
    GPU unnoticed."""
    template = make_workflow_baseline()
    _noise_inputs(template)["noise_seed"] = 4242

    result = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), _assets())

    assert _noise_inputs(result)["noise_seed"] == workflow_graph.DEFAULT_NOISE_SEED
    assert _noise_inputs(template)["noise_seed"] == 4242  # template itself untouched


def test_default_noise_seed_preserves_todays_renders_exactly():
    """Every render this project has produced used the template's 0. The
    default must keep producing that byte-for-byte, or the first run after
    this change silently becomes a different video."""
    assert workflow_graph.DEFAULT_NOISE_SEED == 0
    template = make_workflow_baseline()
    assert _noise_inputs(template)["noise_seed"] == 0

    result = workflow_graph.WorkflowGraphMutator().mutate(template, _prompt(), _assets())

    assert _noise_inputs(result)["noise_seed"] == 0


def test_each_chunk_can_carry_its_own_seed():
    """A per-call parameter, not a per-run constant: varying takes (the other
    half of #38) must be a policy decision made above this module."""
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    seeds = [
        _noise_inputs(
            mutator.mutate(template, _prompt(chunk_id=cid), _assets(chunk_id=cid), noise_seed=seed)
        )["noise_seed"]
        for cid, seed in ((0, 7), (1, 8), (2, 9))
    ]

    assert seeds == [7, 8, 9]


def test_mutate_refuses_a_template_with_no_noise_node(caplog):
    """A graph with no ``RandomNoise`` cannot be seeded, and cannot sample
    either -- fail here, not after the upload."""
    template = make_workflow_missing_node(omit=CLASS_TYPE_NOISE)
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.NodeNotFoundError, match=CLASS_TYPE_NOISE),
    ):
        mutator.mutate(template, _prompt(), _assets())

    assert any(CLASS_TYPE_NOISE in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "bad_seed",
    [-1, workflow_graph.MAX_NOISE_SEED + 1, 1.5, "42", True],
    ids=["negative", "above_max", "float", "string", "bool"],
)
def test_mutate_rejects_a_seed_comfyui_would_not_honor_exactly(bad_seed, caplog):
    """Same reasoning as ``length`` (#20): ComfyUI clamps ``noise_seed`` into
    ``0..2**64-1`` server-side, so an out-of-range value renders under a seed
    nobody recorded -- which is the whole defect."""
    template = make_workflow_baseline()
    mutator = workflow_graph.WorkflowGraphMutator()

    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.InvalidNoiseSeedError),
    ):
        mutator.mutate(template, _prompt(), _assets(), noise_seed=bad_seed)


def test_node_input_overrides_still_win_over_the_seed_parameter():
    """The generic seam is applied last by design, so a caller that reaches for
    it is not silently overruled."""
    template = make_workflow_baseline()

    result = workflow_graph.WorkflowGraphMutator().mutate(
        template,
        _prompt(),
        _assets(),
        noise_seed=1,
        node_input_overrides={CLASS_TYPE_NOISE: {"noise_seed": 2}},
    )

    assert _noise_inputs(result)["noise_seed"] == 2


@pytest.mark.parametrize(
    "template_name", ["workflow_api.json", "workflow_i2v_api.json"], ids=["base", "i2v"]
)
def test_the_committed_templates_expose_exactly_one_seedable_noise_node(template_name):
    """Against the real committed artifacts, not a fixture: if either template
    ever grew a second ``RandomNoise`` node the lookup would become ambiguous
    and every render would fail -- better to learn that here."""
    workflow = workflow_graph.load_workflow_template(REPO_ROOT / template_name)

    matches = workflow_graph.find_nodes_by_class_type(workflow, CLASS_TYPE_NOISE)

    assert len(matches) == 1
    assert "noise_seed" in matches[0][1]["inputs"]


def test_the_seed_reaches_the_payload_actually_submitted_to_comfyui(tmp_path):
    """End to end through the mock server: what matters is the seed in the
    JSON that lands on ``/prompt``, not the seed in a local variable."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="mvm_chunk_0007_00001.mp4")
    workflow = workflow_graph.WorkflowGraphMutator().mutate(
        make_workflow_baseline(), _prompt(chunk_id=7), _assets(chunk_id=7), noise_seed=8675309
    )

    client = ComfyUIExecutionClient(
        base_url=session.base_url,
        session=session,
        ws_factory=make_ws_factory(build_success_sequence("prompt-0001")),
        client_id="client-0",
    )
    client.execute(workflow, chunk_id=7, output_dir=tmp_path / "out")

    submitted = session.submitted_prompts[0]["workflow"]
    assert _noise_inputs(submitted)["noise_seed"] == 8675309


# --------------------------------------------------------------------------- #
# Issue #38 (variation half): per-chunk seed derivation and re-rolling
#
# The mutator's ``noise_seed`` is a per-call parameter by design (see
# ``test_each_chunk_can_carry_its_own_seed`` above) -- deriving *what* value
# each chunk gets from a single run-wide base is a policy decision that
# belongs above this module. ``resolve_chunk_seed`` is that policy, and
# ``PerChunkSeedMutator`` is the seam that applies it without a caller having
# to compute anything itself.
# --------------------------------------------------------------------------- #


def test_resolve_chunk_seed_varies_by_chunk_id():
    """The whole point: ten chunks under one base seed must not render as
    ten copies of the same take."""
    seeds = {
        workflow_graph.resolve_chunk_seed(base_seed=1000, chunk_id=cid) for cid in range(10)
    }
    assert len(seeds) == 10


def test_resolve_chunk_seed_is_a_pure_function_of_its_inputs():
    """A resumed run must recompose the exact same value for the exact same
    chunk with no state to read back -- see ``resilience.py``'s content-tier
    comparison, which is what actually makes a stale chunk re-render."""
    first = workflow_graph.resolve_chunk_seed(base_seed=777, chunk_id=12)
    second = workflow_graph.resolve_chunk_seed(base_seed=777, chunk_id=12)
    assert first == second


def test_resolve_chunk_seed_matches_todays_base_seed_at_chunk_zero():
    """``base_seed + 0 == base_seed``, so a single-chunk run (or chunk 0 of
    any run) is the identity case -- nothing exotic happens at the origin."""
    assert workflow_graph.resolve_chunk_seed(base_seed=42, chunk_id=0) == 42


def test_resolve_chunk_seed_wraps_within_comfyuis_valid_range():
    """A base seed near the top of ComfyUI's range plus a chunk_id offset
    must still land inside 0..MAX_NOISE_SEED, or the mutator's own validation
    (``InvalidNoiseSeedError``) would refuse the very value this function
    derived."""
    seed = workflow_graph.resolve_chunk_seed(
        base_seed=workflow_graph.MAX_NOISE_SEED, chunk_id=5
    )
    assert 0 <= seed <= workflow_graph.MAX_NOISE_SEED
    assert seed == 4  # wraps: MAX + 5 == (MAX + 1) + 4


def test_resolve_chunk_seed_reseed_generation_differs_from_generation_zero():
    """``--reseed``'s whole job: the same chunk must get a seed that provably
    differs from the one already on disk."""
    generation_zero = workflow_graph.resolve_chunk_seed(base_seed=99, chunk_id=12)
    generation_one = workflow_graph.resolve_chunk_seed(
        base_seed=99, chunk_id=12, reseed_generation=1
    )
    assert generation_one != generation_zero


def test_resolve_chunk_seed_reseed_generation_cannot_alias_a_neighbours_seed():
    """A reseeded chunk 12 must not happen to land on chunk 11 or chunk 13's
    ordinary (generation 0) seed -- that would look like a coincidence, not a
    deliberate different take. RESEED_GENERATION_STRIDE is chosen far larger
    than any real chunk_id range specifically to rule this out."""
    reseeded_12 = workflow_graph.resolve_chunk_seed(base_seed=99, chunk_id=12, reseed_generation=1)
    neighbours = {
        workflow_graph.resolve_chunk_seed(base_seed=99, chunk_id=cid) for cid in range(0, 1000)
    }
    assert reseeded_12 not in neighbours


def test_resolve_chunk_seed_successive_generations_differ_too():
    """A chunk re-rolled twice (``--reseed-generation 2`` after generation 1
    still wasn't right) must get a third distinct value, not flip back to
    generation 0's or generation 1's."""
    seeds = {
        workflow_graph.resolve_chunk_seed(base_seed=5, chunk_id=3, reseed_generation=g)
        for g in range(4)
    }
    assert len(seeds) == 4


class _RecordingMutator:
    """A minimal ``WorkflowMutator`` double that records exactly the kwargs
    it was called with, so ``PerChunkSeedMutator`` tests can assert what it
    delegates without depending on ``WorkflowGraphMutator``'s own behaviour."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def mutate(self, template, prompt, assets, **kwargs):
        self.calls.append({"template": template, "prompt": prompt, "assets": assets, **kwargs})
        return {"recorded": True}


def test_per_chunk_seed_mutator_overrides_whatever_seed_the_caller_passed():
    """``ContinuityWorkflowProvider`` (continuity.py) passes one flat
    ``noise_seed`` for the whole run to every ``mutate()`` call -- this is the
    seam that turns that into a per-chunk value without editing that module,
    by intercepting the call and substituting its own before delegating."""
    inner = _RecordingMutator()
    mutator = workflow_graph.PerChunkSeedMutator(inner, base_seed=1000)

    mutator.mutate(make_workflow_baseline(), _prompt(chunk_id=3), _assets(chunk_id=3), noise_seed=0)

    assert inner.calls[0]["noise_seed"] == workflow_graph.resolve_chunk_seed(
        base_seed=1000, chunk_id=3
    )
    assert inner.calls[0]["noise_seed"] != 0


def test_per_chunk_seed_mutator_varies_across_chunks_from_one_flat_caller_value():
    inner = _RecordingMutator()
    mutator = workflow_graph.PerChunkSeedMutator(inner, base_seed=2024)

    for cid in range(5):
        mutator.mutate(
            make_workflow_baseline(), _prompt(chunk_id=cid), _assets(chunk_id=cid), noise_seed=2024
        )

    seeds = [call["noise_seed"] for call in inner.calls]
    assert len(set(seeds)) == 5


def test_per_chunk_seed_mutator_passes_through_every_other_kwarg_unchanged():
    inner = _RecordingMutator()
    mutator = workflow_graph.PerChunkSeedMutator(inner, base_seed=1)

    mutator.mutate(
        make_workflow_baseline(),
        _prompt(chunk_id=1),
        _assets(chunk_id=1),
        noise_seed=0,
        length_frames=141,
        text_encoder="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        lora="h3-realism-people-t2v-i2v-r2v.safetensors",
        lora_strength=0.8,
        cast_image_title="Cast Reference",
    )

    call = inner.calls[0]
    assert call["length_frames"] == 141
    assert call["text_encoder"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert call["lora"] == "h3-realism-people-t2v-i2v-r2v.safetensors"
    assert call["lora_strength"] == 0.8
    assert call["cast_image_title"] == "Cast Reference"


def test_per_chunk_seed_mutator_reseed_generation_only_touches_the_named_chunk():
    """The heart of ``--reseed``'s composability with ``--resume``: chunks
    absent from ``reseed_generations`` must derive *exactly* the ordinary
    generation-0 value, so their fingerprint still matches what's on disk and
    they stay cache hits."""
    inner = _RecordingMutator()
    mutator = workflow_graph.PerChunkSeedMutator(
        inner, base_seed=500, reseed_generations={12: 1}
    )

    for cid in (11, 12, 13):
        mutator.mutate(
            make_workflow_baseline(), _prompt(chunk_id=cid), _assets(chunk_id=cid), noise_seed=0
        )

    seeds = {call["prompt"].chunk_id: call["noise_seed"] for call in inner.calls}
    assert seeds[11] == workflow_graph.resolve_chunk_seed(base_seed=500, chunk_id=11)
    assert seeds[13] == workflow_graph.resolve_chunk_seed(base_seed=500, chunk_id=13)
    assert seeds[12] == workflow_graph.resolve_chunk_seed(
        base_seed=500, chunk_id=12, reseed_generation=1
    )
    assert seeds[12] != workflow_graph.resolve_chunk_seed(base_seed=500, chunk_id=12)


def test_per_chunk_seed_mutator_delegates_to_a_real_mutator_end_to_end():
    """Not just a recording double: wired to the real ``WorkflowGraphMutator``,
    the seed that actually lands in the graph must be the derived one."""
    mutator = workflow_graph.PerChunkSeedMutator(
        workflow_graph.WorkflowGraphMutator(), base_seed=10_000
    )

    result = mutator.mutate(make_workflow_baseline(), _prompt(chunk_id=9), _assets(chunk_id=9))

    assert _noise_inputs(result)["noise_seed"] == workflow_graph.resolve_chunk_seed(
        base_seed=10_000, chunk_id=9
    )


# --------------------------------------------------------------------------- #
# Issue #39: which text encoder encoded the prompt
# --------------------------------------------------------------------------- #


def _clip_inputs(workflow: contracts.Workflow) -> dict:
    matches = workflow_graph.find_nodes_by_class_type(workflow, CLASS_TYPE_CLIP_LOADER)
    assert len(matches) == 1
    return matches[0][1]["inputs"]


def test_read_text_encoder_reports_what_the_template_names():
    """The run must be able to write down which encoder it used even when it
    pins nothing -- otherwise a template edit swapping NVFP4 for INT8 sails
    past --resume exactly as an unrecorded seed once did (issue #38)."""
    assert workflow_graph.read_text_encoder(make_workflow_baseline()) == WEIGHT_TEXT_ENCODER


def test_read_text_encoder_is_none_when_the_graph_has_no_clip_loader():
    """A graph with no CLIPLoader has no encoder identity to report. None
    means unrecorded, and never compares equal to a named encoder."""
    workflow = {
        node_id: node
        for node_id, node in make_workflow_baseline().items()
        if node["class_type"] != CLASS_TYPE_CLIP_LOADER
    }
    assert workflow_graph.read_text_encoder(workflow) is None


def test_mutate_leaves_the_template_encoder_alone_by_default():
    workflow = workflow_graph.WorkflowGraphMutator().mutate(
        make_workflow_baseline(), _prompt(), _assets()
    )
    assert _clip_inputs(workflow)["clip_name"] == WEIGHT_TEXT_ENCODER


def test_mutate_injects_the_pinned_text_encoder_into_the_clip_loader():
    """Issue #39's A/B swaps one component and holds everything else fixed.
    Hand-editing the template to do it changes a committed artefact and
    leaves no record in the run."""
    workflow = workflow_graph.WorkflowGraphMutator().mutate(
        make_workflow_baseline(),
        _prompt(),
        _assets(),
        text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    )
    assert _clip_inputs(workflow)["clip_name"] == (
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    # ... and the encoder *type* is untouched: it names the H3 tokenizer
    # family, not the weights file.
    assert _clip_inputs(workflow)["type"] == "minimax"


def test_mutate_injects_the_pinned_text_encoder_on_the_i2v_template():
    """Both authored templates carry their own CLIPLoader. A run that pinned
    the encoder for the base path only would render a mixed-encoder video."""
    workflow = workflow_graph.WorkflowGraphMutator().mutate(
        make_workflow_i2v(),
        _prompt(),
        _assets(),
        cast_image_title=DEFAULT_CAST_IMAGE_TITLE,
        text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    )
    assert _clip_inputs(workflow)["clip_name"] == (
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )


def test_mutate_refuses_a_blank_text_encoder():
    with pytest.raises(workflow_graph.InvalidTextEncoderError):
        workflow_graph.WorkflowGraphMutator().mutate(
            make_workflow_baseline(), _prompt(), _assets(), text_encoder="   "
        )


def test_mutate_refuses_a_non_string_text_encoder():
    with pytest.raises(workflow_graph.InvalidTextEncoderError):
        workflow_graph.WorkflowGraphMutator().mutate(
            make_workflow_baseline(), _prompt(), _assets(), text_encoder=7
        )


def test_pinning_an_encoder_on_a_graph_with_no_clip_loader_raises():
    """Refuse loudly rather than render the whole run through the encoder the
    operator was trying to replace."""
    workflow = {
        node_id: node
        for node_id, node in make_workflow_baseline().items()
        if node["class_type"] != CLASS_TYPE_CLIP_LOADER
    }
    with pytest.raises(workflow_graph.NodeNotFoundError):
        workflow_graph.WorkflowGraphMutator().mutate(
            workflow, _prompt(), _assets(), text_encoder="anything.safetensors"
        )


def test_the_pinned_encoder_reaches_the_payload_submitted_to_comfyui(tmp_path):
    """What matters is the clip_name in the JSON that lands on /prompt."""
    session = FakeComfyUISession()
    session.seed_history_success("prompt-0001", video_filename="mvm_chunk_0007_00001.mp4")
    workflow = workflow_graph.WorkflowGraphMutator().mutate(
        make_workflow_baseline(),
        _prompt(chunk_id=7),
        _assets(chunk_id=7),
        text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    )

    client = ComfyUIExecutionClient(
        base_url=session.base_url,
        session=session,
        ws_factory=make_ws_factory(build_success_sequence("prompt-0001")),
        client_id="client-0",
    )
    client.execute(workflow, chunk_id=7, output_dir=tmp_path / "out")

    submitted = session.submitted_prompts[0]["workflow"]
    assert _clip_inputs(submitted)["clip_name"] == (
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )


# --------------------------------------------------------------------------- #
# graph_fingerprint (issue #45): the graph a chunk actually rendered through
#
# ChunkFingerprint recorded every value injected into the graph and never the
# graph itself, so editing a committed template (#44 removing `last_frame`
# from the I2V template) changed the pixels while --resume reused the old
# chunks anyway -- the same blind spot the seed (#38) and the encoder (#39)
# were each fixed by naming one field. graph_fingerprint is the general case:
# a hash of the mutated graph's shape and constants, with only the six
# per-chunk-varying inputs masked out.
# --------------------------------------------------------------------------- #


def test_graph_fingerprint_returns_a_stable_short_hex_digest():
    digest = workflow_graph.graph_fingerprint(make_workflow_baseline())
    assert isinstance(digest, str)
    assert len(digest) == workflow_graph.FINGERPRINT_DIGEST_LENGTH
    int(digest, 16)  # raises ValueError if not valid hex


def test_graph_fingerprint_is_deterministic_for_the_same_workflow():
    workflow = make_workflow_baseline()
    assert workflow_graph.graph_fingerprint(workflow) == workflow_graph.graph_fingerprint(workflow)


def test_graph_fingerprint_does_not_mutate_the_caller_workflow():
    # The graph handed in is about to be (or already was) POSTed to ComfyUI;
    # fingerprinting it must never be visible to the caller as a side effect.
    workflow = make_workflow_baseline()
    original = json.loads(json.dumps(workflow))

    workflow_graph.graph_fingerprint(workflow)

    assert workflow == original


@pytest.mark.parametrize(
    "class_type,input_key,new_value",
    [
        (CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "prompt", "an entirely different prompt"),
        (CLASS_TYPE_IMAGE_LOAD, "image", "a_different_staged_image.png"),
        (CLASS_TYPE_AUDIO_LOAD, "audio", "a_different_staged_audio.wav"),
        (CLASS_TYPE_VIDEO_SAVE, "filename_prefix", "mvm_chunk_9999"),
        (CLASS_TYPE_NOISE, "noise_seed", 999999),
        (CLASS_TYPE_H3_REFERENCE_TO_VIDEO, "length", 141),
    ],
    ids=["prompt", "image", "audio", "filename_prefix", "noise_seed", "length"],
)
def test_graph_fingerprint_ignores_each_masked_per_chunk_field(class_type, input_key, new_value):
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, class_type)
    node["inputs"][input_key] = new_value

    assert workflow_graph.graph_fingerprint(changed) == baseline_digest


def test_graph_fingerprint_changes_when_the_sampler_changes():
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, CLASS_TYPE_SAMPLER_SELECT)
    node["inputs"]["sampler_name"] = "euler"

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_changes_when_step_count_changes():
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, CLASS_TYPE_SCHEDULER)
    node["inputs"]["steps"] = 30

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_changes_when_scheduler_changes():
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, CLASS_TYPE_SCHEDULER)
    node["inputs"]["scheduler"] = "karras"

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_changes_when_unet_name_changes():
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, CLASS_TYPE_UNET_LOADER)
    node["inputs"]["unet_name"] = "a_different_unet.safetensors"

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_changes_when_a_vae_filename_changes():
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    matches = workflow_graph.find_nodes_by_class_type(changed, CLASS_TYPE_VAE_LOADER)
    _, video_vae_node = next(
        (nid, n) for nid, n in matches if n["_meta"]["title"] == "Load Video VAE"
    )
    video_vae_node["inputs"]["vae_name"] = "a_different_video_vae.safetensors"

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


@pytest.mark.parametrize(("dimension", "value"), [("width", 864), ("height", 480)])
def test_graph_fingerprint_changes_when_a_resolution_default_changes(dimension: str, value: int):
    """Each dimension on its own, not both together.

    Changing the pair at once passes even if only one of them is counted, so
    it cannot tell "resolution is fingerprinted" from "half of resolution is".
    That distinction is load-bearing here: a run that leaves ``render_width``/
    ``render_height`` unset renders at whatever the template carries, and the
    fingerprint's own resolution fields then record ``None`` -- so the
    template's dimensions are the *only* evidence of the size a chunk was
    rendered at, and resolution is this project's dominant cost."""
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, node = workflow_graph.find_one_node(changed, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    node["inputs"][dimension] = value

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_changes_when_wiring_changes():
    """The video and audio VAELoader outputs feed different sockets on the H3
    node. Swapping which node feeds which socket changes nothing scalar --
    every node's own inputs are untouched -- but is exactly the wiring bug
    this hash exists to catch."""
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    _, h3_node = workflow_graph.find_one_node(changed, CLASS_TYPE_H3_REFERENCE_TO_VIDEO)
    h3_node["inputs"]["vae"], h3_node["inputs"]["audio_vae"] = (
        h3_node["inputs"]["audio_vae"],
        h3_node["inputs"]["vae"],
    )

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_counts_a_node_and_input_this_module_has_never_seen():
    """The explicit acceptance test for 'mask, don't whitelist': a whitelist
    of "inputs that matter" reproduces the original defect the first time
    someone adds a node. The default for anything unrecognised -- a made-up
    node with a made-up input -- must be 'this counts'."""
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())

    changed = make_workflow_baseline()
    changed["999"] = {
        "class_type": "SomeFutureNodeThisModuleHasNeverSeen",
        "inputs": {"some_future_tuning_knob": 3.14159},
        "_meta": {"title": "a node from a future issue"},
    }

    assert workflow_graph.graph_fingerprint(changed) != baseline_digest


def test_graph_fingerprint_ignores_json_key_ordering_and_reformatting():
    """Hash the parsed structure, not file bytes: reformatting, reindenting,
    or reordering keys in the template file must not invalidate a whole
    song's worth of renders."""
    baseline = make_workflow_baseline()
    baseline_digest = workflow_graph.graph_fingerprint(baseline)

    reversed_workflow = {}
    for node_id in reversed(list(baseline.keys())):
        node = dict(baseline[node_id])
        node["inputs"] = dict(reversed(list(node.get("inputs", {}).items())))
        reversed_workflow[node_id] = node

    assert workflow_graph.graph_fingerprint(reversed_workflow) == baseline_digest


def test_graph_fingerprint_is_unchanged_by_a_canvas_renumbering():
    """Node ids are not identity anywhere else in this module -- every lookup
    resolves by class_type, plus title or wiring, never by id, because a
    ComfyUI canvas edit renumbers ids freely without changing what renders.
    shifted_ids_h3.json is baseline_h3.json with every id offset by a
    constant: structurally identical, differently labeled. A routine
    save-and-reopen on the canvas must not force a whole song to re-render."""
    baseline_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())
    shifted_digest = workflow_graph.graph_fingerprint(make_workflow_shifted_ids())

    assert shifted_digest == baseline_digest


def test_graph_fingerprint_differs_between_base_and_i2v_templates():
    base_digest = workflow_graph.graph_fingerprint(make_workflow_baseline())
    i2v_digest = workflow_graph.graph_fingerprint(make_workflow_i2v())

    assert base_digest != i2v_digest


def test_graph_fingerprint_rejects_an_empty_workflow(caplog):
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.GraphFingerprintError),
    ):
        workflow_graph.graph_fingerprint({})


def test_graph_fingerprint_rejects_a_non_dict_workflow(caplog):
    with (
        caplog.at_level("ERROR"),
        pytest.raises(workflow_graph.GraphFingerprintError),
    ):
        workflow_graph.graph_fingerprint(["not", "a", "workflow"])  # type: ignore[arg-type]


def test_graph_fingerprint_is_byte_identical_across_processes_and_hash_seeds():
    """No hash(), no set iteration order, no dict-insertion-order dependence,
    no float-repr instability -- issue #45's acceptance criteria name this
    explicitly. Computes the real digest in two subprocesses under two
    different PYTHONHASHSEED values and compares the printed digests."""
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from music_video_maker import workflow_graph; "
        "from tests.harness.factories import make_workflow_baseline; "
        "print(workflow_graph.graph_fingerprint(make_workflow_baseline()))"
    )

    digests = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(result.stdout.strip())

    assert digests[0] == digests[1]
    assert len(digests[0]) == workflow_graph.FINGERPRINT_DIGEST_LENGTH


# --------------------------------------------------------------------------- #
# LoRA insertion (issue #62)
#
# Unlike `text_encoder` (#39), which pins an input on a node the templates
# already carry, a LoRA needs a node that is not in either template at all.
# So this is graph surgery: insert, then rewire every consumer of the model
# edge. Which also means `template_hash` cannot see it -- the hash is taken
# over the *template*, before mutation -- and it needs fingerprint fields of
# its own. Fourth occurrence of the #38/#39/#45 blind spot.
# --------------------------------------------------------------------------- #


def _model_graph() -> dict:
    """The MODEL edge as both committed templates actually wire it."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "steps": 20}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0]}},
    }


def test_no_lora_leaves_the_graph_untouched():
    """A run without a LoRA must produce byte-identical bytes to today."""
    graph = _model_graph()
    assert insert_lora(graph, lora=None, strength=1.0) == _model_graph()


def test_lora_is_inserted_between_the_loader_and_every_consumer():
    graph = insert_lora(_model_graph(), lora="realism.safetensors", strength=0.8)

    lora_id, lora_node = find_one_node(graph, CLASS_TYPE_LORA_LOADER)
    assert lora_node["inputs"]["lora_name"] == "realism.safetensors"
    assert lora_node["inputs"]["strength_model"] == 0.8
    # It takes the loader's output...
    assert lora_node["inputs"]["model"] == ["1", 0]
    # ...and *every* consumer now takes the LoRA's, not the loader's. A
    # rewire that missed one would silently render half the graph un-LoRA'd.
    assert graph["9"]["inputs"]["model"] == [lora_id, 0]
    assert graph["11"]["inputs"]["model"] == [lora_id, 0]


def test_the_new_node_id_does_not_collide():
    graph = _model_graph()
    lora_id, _ = find_one_node(insert_lora(graph, lora="x.safetensors", strength=1.0),
                               CLASS_TYPE_LORA_LOADER)
    assert lora_id not in _model_graph()


def test_an_absolute_lora_path_is_refused():
    """Same rule as text_encoder: ComfyUI resolves lora_name inside its own
    models/loras/, so an absolute path fails at load time on the render host
    -- after custody of the GPU has been taken."""
    with pytest.raises(WorkflowGraphError, match="absolute"):
        insert_lora(_model_graph(), lora="/home/derek/x.safetensors", strength=1.0)


def test_inserting_twice_is_refused_rather_than_stacking():
    once = insert_lora(_model_graph(), lora="x.safetensors", strength=1.0)
    with pytest.raises(WorkflowGraphError, match="already"):
        insert_lora(once, lora="x.safetensors", strength=1.0)
