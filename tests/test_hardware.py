"""Tests for the static hardware profile + workflow optimization scan (issue #13).

Fully offline: no nvidia-smi, no ComfyUI, no network. The whole point of this
module is that it does *not* touch live GPU state -- see the module
docstring in ``music_video_maker/hardware.py`` for why.
"""

from __future__ import annotations

import logging

import pytest

from music_video_maker import hardware
from music_video_maker.contracts import HardwareProfile
from tests.harness.factories import make_workflow_baseline

# --------------------------------------------------------------------------- #
# Named profiles
# --------------------------------------------------------------------------- #


def test_profiles_registry_has_a_doris_24gb_profile():
    profile = hardware.PROFILE_RTX_4090_24GB
    assert isinstance(profile, HardwareProfile)
    assert profile.vram_gb == pytest.approx(24.0)
    assert profile.name in hardware.PROFILES
    assert hardware.PROFILES[profile.name] is profile


def test_profiles_registry_has_a_larger_profile():
    profile = hardware.PROFILE_48GB
    assert isinstance(profile, HardwareProfile)
    assert profile.vram_gb == pytest.approx(48.0)
    assert profile.name in hardware.PROFILES


def test_larger_vram_profile_does_not_extend_the_trained_chunk_window():
    # issue #20: more VRAM does NOT raise the safe max duration -- H3's
    # trained range (124-362 frames) wins regardless of the card. This
    # corrects the pre-#20 assumption (more VRAM -> looser max-duration
    # bound) that PROFILE_48GB's old max_chunk_seconds=24.0 encoded.
    larger, doris = hardware.PROFILE_48GB, hardware.PROFILE_RTX_4090_24GB
    assert larger.max_chunk_seconds == pytest.approx(doris.max_chunk_seconds)
    assert larger.min_chunk_seconds == pytest.approx(doris.min_chunk_seconds)


def test_profiles_pin_the_chunk_window_to_h3s_trained_frame_range():
    grid = hardware.H3_FRAME_GRID
    trained_min_s = grid.frames_to_seconds(grid.trained_min_frames)
    trained_max_s = grid.frames_to_seconds(grid.trained_max_frames)
    for profile in hardware.PROFILES.values():
        assert profile.min_chunk_seconds == pytest.approx(trained_min_s)
        assert profile.max_chunk_seconds == pytest.approx(trained_max_s)


def test_profiles_have_sane_chunk_windows():
    for profile in hardware.PROFILES.values():
        assert profile.min_chunk_seconds > 0
        assert profile.max_chunk_seconds > profile.min_chunk_seconds


def test_24gb_profile_recommends_every_optimization_node():
    # 24 GB is below H3's quantized VRAM appetite per issue #13 -- every
    # optimization in the table is warranted.
    profile = hardware.PROFILE_RTX_4090_24GB
    assert hardware.NODE_SAGE_ATTENTION in profile.recommended_nodes
    assert hardware.NODE_SPECTRUM_MINIMAX_H3 in profile.recommended_nodes
    assert hardware.NODE_EASY_CACHE in profile.recommended_nodes


def test_get_profile_returns_named_profile():
    profile = hardware.get_profile(hardware.PROFILE_RTX_4090_24GB.name)
    assert profile is hardware.PROFILE_RTX_4090_24GB


def test_get_profile_raises_and_logs_for_unknown_name(caplog):
    with caplog.at_level(logging.ERROR), pytest.raises(hardware.UnknownProfileError):
        hardware.get_profile("nonexistent-gpu")
    assert "nonexistent-gpu" in caplog.text


# --------------------------------------------------------------------------- #
# Workflow scan
# --------------------------------------------------------------------------- #


def _node(class_type: str) -> dict:
    return {"class_type": class_type, "inputs": {}, "_meta": {"title": class_type}}


def test_scan_reports_all_recommended_nodes_missing_from_baseline_workflow():
    # The shared baseline fixture only contains the core H3 pipeline nodes --
    # none of the optimization nodes are wired in.
    workflow = make_workflow_baseline()
    missing = hardware.scan_workflow_for_missing_optimizations(
        workflow, hardware.PROFILE_RTX_4090_24GB
    )
    assert set(missing) == set(hardware.PROFILE_RTX_4090_24GB.recommended_nodes)


def test_scan_reports_nothing_missing_when_all_nodes_present():
    workflow = make_workflow_baseline()
    for i, node_type in enumerate(hardware.PROFILE_RTX_4090_24GB.recommended_nodes):
        workflow[f"opt-{i}"] = _node(node_type)
    missing = hardware.scan_workflow_for_missing_optimizations(
        workflow, hardware.PROFILE_RTX_4090_24GB
    )
    assert missing == ()


def test_scan_reports_partial_when_some_nodes_present():
    workflow = make_workflow_baseline()
    workflow["opt-0"] = _node(hardware.NODE_SAGE_ATTENTION)
    missing = hardware.scan_workflow_for_missing_optimizations(
        workflow, hardware.PROFILE_RTX_4090_24GB
    )
    assert hardware.NODE_SAGE_ATTENTION not in missing
    assert hardware.NODE_SPECTRUM_MINIMAX_H3 in missing
    assert hardware.NODE_EASY_CACHE in missing


def test_scan_logs_warning_naming_the_exact_repo_to_install(caplog):
    workflow = make_workflow_baseline()
    with caplog.at_level(logging.WARNING):
        hardware.scan_workflow_for_missing_optimizations(workflow, hardware.PROFILE_RTX_4090_24GB)
    assert "ComfyUI-KJNodes" in caplog.text
    assert "ComfyUI-Spectrum-MiniMax-H3" in caplog.text
    assert "Block/EasyCache" in caplog.text


def test_scan_produces_different_recommendations_for_different_profiles():
    workflow = make_workflow_baseline()
    missing_24gb = hardware.scan_workflow_for_missing_optimizations(
        workflow, hardware.PROFILE_RTX_4090_24GB
    )
    missing_48gb = hardware.scan_workflow_for_missing_optimizations(workflow, hardware.PROFILE_48GB)
    assert set(missing_24gb) != set(missing_48gb)
    assert set(missing_48gb) <= set(missing_24gb)


def test_scan_ignores_non_dict_workflow_values_without_raising(caplog):
    workflow = make_workflow_baseline()
    workflow["weird"] = "not-a-node"
    with caplog.at_level(logging.WARNING):
        missing = hardware.scan_workflow_for_missing_optimizations(
            workflow, hardware.PROFILE_RTX_4090_24GB
        )
    assert set(missing) == set(hardware.PROFILE_RTX_4090_24GB.recommended_nodes)
