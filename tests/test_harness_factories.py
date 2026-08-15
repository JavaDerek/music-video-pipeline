"""Tests proving tests/harness/factories.py produces well-formed contract
objects and valid fixture files (issue #16).

Fully offline: no network, no GPU, no real ComfyUI, no torch.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import pytest

from music_video_maker import contracts
from tests.harness import factories

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared timeline assertions
# --------------------------------------------------------------------------- #


def _assert_monotonic_non_overlapping(segments: tuple[contracts.AlignedSegment, ...]) -> None:
    assert len(segments) > 0
    prev_end = None
    for seg in segments:
        assert seg.end > seg.start, f"segment {seg.index} has end <= start"
        if prev_end is not None:
            assert seg.start >= prev_end, f"segment {seg.index} overlaps the previous one"
        prev_end = seg.end


def _assert_words_within_segment(seg: contracts.AlignedSegment) -> None:
    prev_end = None
    for w in seg.words:
        assert w.end > w.start
        assert w.start >= seg.start - 1e-9
        assert w.end <= seg.end + 1e-9
        if prev_end is not None:
            assert w.start >= prev_end - 1e-9
        prev_end = w.end


# --------------------------------------------------------------------------- #
# Workflow fixtures (issue #8)
# --------------------------------------------------------------------------- #


def _is_node_ref(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _each_node_ref(inputs: dict):
    """Yield every ``[node_id, output_index]`` reference in a node's
    ``inputs``. AUTOGROW_V3 slots (``ref_images.ref_image_0`` etc.) are
    ordinary flat keys holding a single plain reference, like any other
    input -- confirmed, no special-casing needed."""
    for value in inputs.values():
        if _is_node_ref(value):
            yield value


def _assert_graph_connectivity_valid(wf: dict) -> None:
    """Every node-reference input must point at a node that exists, and at
    an output index that node's class_type actually declares."""
    for node_id, node in wf.items():
        for target_id, out_idx in _each_node_ref(node["inputs"]):
            assert target_id in wf, f"node {node_id} references missing node {target_id}"
            target_type = wf[target_id]["class_type"]
            max_outputs = factories.NODE_OUTPUT_SLOT_COUNTS[target_type]
            assert 0 <= out_idx < max_outputs, (
                f"node {node_id} references out-of-range output {out_idx} "
                f"on {target_id} ({target_type}, {max_outputs} output(s))"
            )


def test_workflow_baseline_has_every_required_class_type_expected_number_of_times():
    wf = factories.make_workflow_baseline()
    class_types = [node["class_type"] for node in wf.values()]
    for required, expected_count in factories.REQUIRED_CLASS_TYPE_COUNTS.items():
        assert class_types.count(required) == expected_count, required
    # every node has a title for disambiguation
    assert all("_meta" in node and "title" in node["_meta"] for node in wf.values())


def test_workflow_baseline_contains_no_cloud_api_node():
    wf = factories.make_workflow_baseline()
    class_types = {node["class_type"] for node in wf.values()}
    for cloud_type in factories.CLOUD_API_CLASS_TYPES:
        assert cloud_type not in class_types


def test_workflow_baseline_prompt_is_a_plain_string_with_no_text_encode_node():
    wf = factories.make_workflow_baseline()
    ref_to_video = next(
        node
        for node in wf.values()
        if node["class_type"] == factories.CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    )
    assert isinstance(ref_to_video["inputs"]["prompt"], str)
    assert ref_to_video["inputs"]["prompt"]
    # no text-encode node of any kind exists anywhere in the graph
    for node in wf.values():
        assert "textencode" not in node["class_type"].lower()


def test_workflow_baseline_vae_loaders_distinguishable_by_title():
    wf = factories.make_workflow_baseline()
    vae_nodes = [
        node for node in wf.values() if node["class_type"] == factories.CLASS_TYPE_VAE_LOADER
    ]
    assert len(vae_nodes) == 2
    titles = {node["_meta"]["title"] for node in vae_nodes}
    assert len(titles) == 2  # distinct titles disambiguate video VAE from audio VAE


def test_workflow_baseline_connectivity_is_valid():
    _assert_graph_connectivity_valid(factories.make_workflow_baseline())


def test_workflow_shifted_ids_has_same_required_kinds_different_ids():
    baseline = factories.make_workflow_baseline()
    shifted = factories.make_workflow_shifted_ids()

    assert set(baseline.keys()).isdisjoint(shifted.keys())

    baseline_types = sorted(node["class_type"] for node in baseline.values())
    shifted_types = sorted(node["class_type"] for node in shifted.values())
    assert baseline_types == shifted_types
    for required in factories.REQUIRED_CLASS_TYPES:
        assert required in shifted_types


def test_workflow_shifted_ids_connectivity_is_valid():
    _assert_graph_connectivity_valid(factories.make_workflow_shifted_ids())


def test_workflow_ambiguous_vae_loaders_share_identical_generic_title():
    wf = factories.make_workflow_ambiguous_vae_loaders()
    vae_nodes = {
        node_id: node
        for node_id, node in wf.items()
        if node["class_type"] == factories.CLASS_TYPE_VAE_LOADER
    }
    assert len(vae_nodes) == 2
    titles = {node["_meta"]["title"] for node in vae_nodes.values()}
    # both left at ComfyUI's un-renamed default title -- title text alone
    # can no longer disambiguate them
    assert titles == {factories.CLASS_TYPE_VAE_LOADER}


def test_workflow_ambiguous_vae_loaders_disambiguated_by_tracing_wiring():
    wf = factories.make_workflow_ambiguous_vae_loaders()
    vae_node_ids = {
        node_id
        for node_id, node in wf.items()
        if node["class_type"] == factories.CLASS_TYPE_VAE_LOADER
    }
    ref_to_video = next(
        node
        for node in wf.values()
        if node["class_type"] == factories.CLASS_TYPE_H3_REFERENCE_TO_VIDEO
    )
    vae_id, _ = ref_to_video["inputs"]["vae"]
    audio_vae_id, _ = ref_to_video["inputs"]["audio_vae"]
    # not the same node, and together they account for both VAELoader nodes
    assert vae_id != audio_vae_id
    assert {vae_id, audio_vae_id} == vae_node_ids


def test_workflow_ambiguous_vae_loaders_connectivity_is_valid():
    _assert_graph_connectivity_valid(factories.make_workflow_ambiguous_vae_loaders())


def test_workflow_missing_node_omits_the_required_class_type():
    wf = factories.make_workflow_missing_node(omit=factories.CLASS_TYPE_AUDIO_LOAD)
    class_types = {node["class_type"] for node in wf.values()}
    assert factories.CLASS_TYPE_AUDIO_LOAD not in class_types
    # everything else required is still present
    for required in factories.REQUIRED_CLASS_TYPES:
        if required != factories.CLASS_TYPE_AUDIO_LOAD:
            assert required in class_types


def test_workflow_missing_node_fails_connectivity_check():
    # proves the connectivity checker itself detects the dangling references
    # left behind when the required audio-load node is removed
    wf = factories.make_workflow_missing_node(omit=factories.CLASS_TYPE_AUDIO_LOAD)
    with pytest.raises(AssertionError):
        _assert_graph_connectivity_valid(wf)


@pytest.mark.parametrize(
    ("path_attr", "factory_name"),
    [
        ("WORKFLOW_BASELINE_PATH", "make_workflow_baseline"),
        ("WORKFLOW_SHIFTED_IDS_PATH", "make_workflow_shifted_ids"),
        ("WORKFLOW_AMBIGUOUS_VAE_LOADERS_PATH", "make_workflow_ambiguous_vae_loaders"),
        ("WORKFLOW_MISSING_NODE_PATH", "make_workflow_missing_node"),
    ],
)
def test_workflow_json_fixture_on_disk_matches_factory_output(path_attr, factory_name):
    path: Path = getattr(factories, path_attr)
    assert path.is_file(), f"missing fixture file {path}"
    on_disk = factories.load_workflow_fixture(path)
    expected = getattr(factories, factory_name)()
    assert on_disk == expected


def test_workflow_fixtures_readme_documents_the_confirmed_schema():
    readme = factories.WORKFLOWS_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "#18" in text
    assert "autogrow" in text.lower()
    for required in factories.REQUIRED_CLASS_TYPES:
        assert required in text
    for cloud_type in factories.CLOUD_API_CLASS_TYPES:
        assert cloud_type in text


# --------------------------------------------------------------------------- #
# Lyrics fixtures (issues #3, #6)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path_attr",
    [
        "LYRICS_PLAIN_PATH",
        "LYRICS_TAGGED_PATH",
        "LYRICS_MIXED_INHERIT_PATH",
        "LYRICS_GAPS_PATH",
        "LYRICS_UNKNOWN_CHARACTER_PATH",
    ],
)
def test_lyrics_fixture_exists_and_nonempty(path_attr):
    path: Path = getattr(factories, path_attr)
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip() != ""


def test_plain_lyrics_has_no_character_tags():
    text = factories.LYRICS_PLAIN_PATH.read_text(encoding="utf-8")
    assert "[" not in text and "]" not in text


def test_tagged_lyrics_uses_bracket_tag_syntax():
    text = factories.LYRICS_TAGGED_PATH.read_text(encoding="utf-8")
    assert "[Dianne: Lead]" in text
    assert "[Marcus: Backup]" in text


def test_mixed_inherit_lyrics_has_untagged_lines_after_a_tag():
    lines = factories.LYRICS_MIXED_INHERIT_PATH.read_text(encoding="utf-8").splitlines()
    tagged_indices = [i for i, line in enumerate(lines) if line.startswith("[")]
    assert tagged_indices
    # at least one tag is followed by more than one untagged line before the
    # next tag (or EOF) -- that's the "inherit across multiple lines" case
    tagged_indices_with_end = [*tagged_indices, len(lines)]
    gaps = [
        tagged_indices_with_end[i + 1] - tagged_indices_with_end[i]
        for i in range(len(tagged_indices))
    ]
    assert any(gap > 2 for gap in gaps)


def test_gaps_lyrics_has_blank_line_runs():
    lines = factories.LYRICS_GAPS_PATH.read_text(encoding="utf-8").splitlines()
    blank_run = 0
    max_blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            max_blank_run = max(max_blank_run, blank_run)
        else:
            blank_run = 0
    assert max_blank_run >= 2


def test_unknown_character_lyrics_references_character_not_in_cast():
    text = factories.LYRICS_UNKNOWN_CHARACTER_PATH.read_text(encoding="utf-8")
    assert "[Ghost:" in text
    cast = factories.make_cast_dict()
    assert "Ghost" not in cast


# --------------------------------------------------------------------------- #
# Synthetic audio (issues #3, #4, #7)
# --------------------------------------------------------------------------- #


def test_write_silent_wav_has_requested_duration_and_is_silent(tmp_path):
    path = tmp_path / "silence.wav"
    factories.write_silent_wav(path, seconds=2.0, sample_rate=8000)

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 8000
        assert wf.getnchannels() == 1
        n_frames = wf.getnframes()
        duration = n_frames / wf.getframerate()
        assert duration == pytest.approx(2.0, abs=1e-3)
        frames = wf.readframes(n_frames)
        assert frames == b"\x00\x00" * n_frames


def test_write_tone_wav_has_requested_duration_and_is_audible(tmp_path):
    path = tmp_path / "tone.wav"
    factories.write_tone_wav(path, seconds=1.0, frequency=440.0, sample_rate=8000)

    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 8000
        n_frames = wf.getnframes()
        duration = n_frames / wf.getframerate()
        assert duration == pytest.approx(1.0, abs=1e-3)
        frames = wf.readframes(n_frames)
        assert any(b != 0 for b in frames)


def test_write_tone_wav_supports_stereo(tmp_path):
    path = tmp_path / "tone_stereo.wav"
    factories.write_tone_wav(path, seconds=0.5, sample_rate=8000, channels=2)

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2
        n_frames = wf.getnframes()
        assert n_frames / wf.getframerate() == pytest.approx(0.5, abs=1e-3)


def test_make_oversized_file_stub_reports_size_over_50mb_limit(tmp_path):
    path = tmp_path / "oversized.bin"
    size = 50 * 1024 * 1024 + 1
    factories.make_oversized_file_stub(path, size_bytes=size)
    assert path.stat().st_size == size


def test_make_oversized_file_stub_can_report_under_limit_size(tmp_path):
    path = tmp_path / "just_fine.bin"
    size = 10 * 1024 * 1024
    factories.make_oversized_file_stub(path, size_bytes=size)
    assert path.stat().st_size == size


# --------------------------------------------------------------------------- #
# Alignment result factories (issues #3, #4)
# --------------------------------------------------------------------------- #


def test_alignment_result_normal_song_is_well_formed():
    result = factories.make_alignment_result_normal_song()
    assert isinstance(result, contracts.AlignmentResult)
    _assert_monotonic_non_overlapping(result.segments)
    for seg in result.segments:
        assert 4.0 <= seg.duration <= 15.0
        _assert_words_within_segment(seg)
    assert result.voiced_duration == pytest.approx(sum(s.duration for s in result.segments))
    assert result.track_duration >= result.segments[-1].end


def test_alignment_result_short_segments_below_stage2_minimum():
    result = factories.make_alignment_result_short_segments()
    _assert_monotonic_non_overlapping(result.segments)
    assert any(seg.duration < 4.0 for seg in result.segments)


def test_alignment_result_long_segment_above_stage2_maximum():
    result = factories.make_alignment_result_long_segment()
    _assert_monotonic_non_overlapping(result.segments)
    assert any(seg.duration > 15.0 for seg in result.segments)


def test_alignment_result_with_gaps_has_a_long_instrumental_gap():
    result = factories.make_alignment_result_with_gaps()
    _assert_monotonic_non_overlapping(result.segments)
    gaps = [
        result.segments[i + 1].start - result.segments[i].end
        for i in range(len(result.segments) - 1)
    ]
    assert any(gap > 15.0 for gap in gaps)
    # exercises character switching too
    characters = {seg.character for seg in result.segments}
    assert "Dianne" in characters and "Marcus" in characters


def test_make_word_timings_evenly_spans_segment():
    words = factories.make_word_timings(["a", "b", "c", "d"], start=10.0, end=14.0)
    assert len(words) == 4
    assert words[0].start == pytest.approx(10.0)
    assert words[-1].end == pytest.approx(14.0)
    for w in words:
        assert isinstance(w, contracts.WordTiming)


def test_make_word_timings_empty_words_returns_empty_tuple():
    assert factories.make_word_timings([], start=0.0, end=5.0) == ()


def test_raw_stablets_result_duck_types_expected_attribute_access():
    result = factories.make_raw_stablets_result()
    assert hasattr(result, "segments")
    assert len(result.segments) > 0
    for seg in result.segments:
        assert isinstance(seg.text, str)
        assert seg.end > seg.start
        assert len(seg.words) > 0
        prev_end = None
        for w in seg.words:
            assert isinstance(w.word, str)
            assert w.end > w.start
            assert w.start >= seg.start - 1e-9
            assert w.end <= seg.end + 1e-9
            if prev_end is not None:
                assert w.start >= prev_end - 1e-9
            prev_end = w.end


# --------------------------------------------------------------------------- #
# Cast factories
# --------------------------------------------------------------------------- #


def test_make_cast_member_defaults_produce_valid_cast_member():
    member = factories.make_cast_member()
    assert isinstance(member, contracts.CastMember)
    assert member.name == "Dianne"
    assert isinstance(member.image, Path)
    with pytest.raises(AttributeError):
        member.name = "mutated"  # type: ignore[misc]


def test_make_cast_member_accepts_overrides():
    member = factories.make_cast_member(name="Rex", role="Drummer", image=Path("x/y.png"))
    assert member.name == "Rex"
    assert member.role == "Drummer"
    assert member.image == Path("x/y.png")


def test_make_cast_dict_includes_a_non_singing_background_member():
    cast = factories.make_cast_dict()
    assert set(cast.keys()) == {"Dianne", "Marcus", "Rex"}
    for name, member in cast.items():
        assert isinstance(member, contracts.CastMember)
        assert member.name == name
    assert "never sings" in cast["Rex"].role


# --------------------------------------------------------------------------- #
# Lyric line factories
# --------------------------------------------------------------------------- #


def test_make_lyric_line_defaults_and_overrides():
    untagged = factories.make_lyric_line(0, "hello world")
    assert untagged.character is None
    assert untagged.raw == "hello world"

    tagged = factories.make_lyric_line(1, "hello world", character="Dianne")
    assert tagged.character == "Dianne"
    assert "Dianne" in tagged.raw


def test_make_lyric_lines_auto_indexes_and_preserves_order():
    lines = factories.make_lyric_lines(
        [("first", "Dianne"), ("second", None), ("third", "Marcus")]
    )
    assert [line.index for line in lines] == [0, 1, 2]
    assert [line.text for line in lines] == ["first", "second", "third"]
    assert [line.character for line in lines] == ["Dianne", None, "Marcus"]
    assert all(isinstance(line, contracts.LyricLine) for line in lines)


# --------------------------------------------------------------------------- #
# Fixture directory sanity
# --------------------------------------------------------------------------- #


def test_fixture_directories_resolve_under_tests_fixtures():
    assert factories.WORKFLOWS_DIR.name == "workflows"
    assert factories.LYRICS_DIR.name == "lyrics"
    assert factories.FIXTURES_DIR.is_dir()
