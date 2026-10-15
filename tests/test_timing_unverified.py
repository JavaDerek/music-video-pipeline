"""Issue #104 (a): a sung chunk whose timing the aligner could not verify is
*recorded* -- in the alignment report, in ``run_state.json`` and on the review
page -- and the operator is pointed at ``[[alignment_override]]`` (#42).

Recorded, never acted on: nothing here changes what slices or renders (#104's
(b) strict refusal and (c) proposed spans are separate, undecided questions).

The word fixtures for chunk 63 are the real per-word probabilities from
"Deathless" v16 (CPU re-alignment, 2026-10-14, which reproduces the issue's
numbers exactly: "won" 0.02, "but" 0.08).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from music_video_maker.alignment_quality import (
    LOW_WORD_CONFIDENCE,
    TIMING_UNVERIFIED_SHARE,
    AlignmentQualityReport,
    LowConfidenceWord,
    evaluate_alignment_quality,
    log_timing_unverified,
    timing_unverified_chunks,
)
from music_video_maker.config import ConfigError, load_config
from music_video_maker.contracts import AudioChunk, RunState, TimingUnverified
from music_video_maker.resilience import dump_run_state, load_run_state
from tests.harness.factories import make_alignment_result_normal_song


def _chunk(
    chunk_id: int, start: float, end: float, *, instrumental: bool = False
) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"c{chunk_id}.wav"),
        start=start,
        end=end,
        text="" if instrumental else "la",
        is_instrumental=instrumental,
    )


def _word(word: str, start: float, end: float, p: float, segment: int = 0) -> LowConfidenceWord:
    return LowConfidenceWord(
        segment_index=segment, word=word, start=start, end=end, probability=p
    )


def _report(*words: LowConfidenceWord) -> AlignmentQualityReport:
    return AlignmentQualityReport(word_confidences=tuple(words))


# chunk 63 of v16: 438.83-446.83 s, "life forevermore. He won the final battle,"
CHUNK_63_WORDS = (
    _word(" life", 439.35, 439.39, 0.00, 49),
    _word(" forevermore.", 439.39, 442.23, 0.33, 49),
    _word(" He", 442.75, 443.95, 0.16, 50),
    _word(" won", 443.95, 444.31, 0.02, 50),
    _word(" the", 444.31, 444.67, 0.55, 50),
    _word(" final", 444.67, 445.15, 0.79, 50),
    _word(" battle,", 445.15, 446.35, 0.48, 50),
)


def test_default_floor_is_the_existing_low_word_floor_and_a_majority_share():
    assert LOW_WORD_CONFIDENCE == 0.35
    assert TIMING_UNVERIFIED_SHARE == 0.5


def test_the_issues_chunk_63_is_flagged_with_its_worst_word():
    flagged = timing_unverified_chunks(
        [_chunk(63, 438.833, 446.833)], _report(*CHUNK_63_WORDS)
    )

    entry = flagged[63]
    assert entry.chunk_id == 63
    assert (entry.word_count, entry.low_word_count) == (7, 4)
    assert entry.worst_word == "life"
    assert entry.worst_probability == pytest.approx(0.0)
    assert entry.start == pytest.approx(438.833)
    assert entry.end == pytest.approx(446.833)
    assert entry.share == pytest.approx(4 / 7)


def test_a_mostly_confident_chunk_is_not_flagged_even_with_one_bad_word():
    words = (
        _word("a", 0.0, 0.5, 0.01),
        _word("b", 0.5, 1.0, 0.9),
        _word("c", 1.0, 1.5, 0.8),
    )
    assert timing_unverified_chunks([_chunk(0, 0.0, 2.0)], _report(*words)) == {}


def test_exactly_at_the_share_is_flagged_and_exactly_at_the_floor_is_not_low():
    at_share = (_word("a", 0.0, 0.5, 0.1), _word("b", 0.5, 1.0, 0.9))
    at_floor = (_word("a", 0.0, 0.5, 0.35), _word("b", 0.5, 1.0, 0.35))

    assert 0 in timing_unverified_chunks([_chunk(0, 0.0, 2.0)], _report(*at_share))
    assert timing_unverified_chunks([_chunk(0, 0.0, 2.0)], _report(*at_floor)) == {}


def test_a_word_straddling_a_boundary_counts_in_one_chunk_only():
    words = (_word("a", 1.9, 2.3, 0.0), _word("b", 2.3, 2.6, 0.0))
    first = timing_unverified_chunks([_chunk(0, 0.0, 2.0)], _report(*words))
    both = timing_unverified_chunks(
        [_chunk(0, 0.0, 2.0), _chunk(1, 2.0, 4.0)], _report(*words)
    )

    assert first == {}  # midpoint 2.1 belongs to the next chunk
    assert both[1].word_count == 2
    assert 0 not in both


def test_instrumental_and_wordless_chunks_are_never_flagged():
    instrumental = _chunk(0, 0.0, 2.0, instrumental=True)
    wordless = _chunk(1, 2.0, 4.0)
    words = (_word("a", 0.5, 1.0, 0.0),)

    assert timing_unverified_chunks([instrumental, wordless], _report(*words)) == {}


def test_words_of_a_segment_an_override_already_pins_are_not_counted():
    words = (
        _word("a", 0.0, 0.5, 0.0, segment=7),
        _word("b", 0.5, 1.0, 0.0, segment=7),
        _word("c", 1.0, 1.5, 0.9, segment=8),
    )
    chunks = [_chunk(0, 0.0, 2.0)]

    assert 0 in timing_unverified_chunks(chunks, _report(*words))
    assert timing_unverified_chunks(chunks, _report(*words), pinned_segments={7}) == {}


def test_floor_and_share_are_parameters():
    words = (_word("a", 0.0, 0.5, 0.4), _word("b", 0.5, 1.0, 0.9))
    chunks = [_chunk(0, 0.0, 2.0)]

    assert timing_unverified_chunks(chunks, _report(*words)) == {}
    assert 0 in timing_unverified_chunks(chunks, _report(*words), word_floor=0.5)
    assert timing_unverified_chunks(chunks, _report(*words), word_floor=0.5, share=0.75) == {}


def test_evaluate_alignment_quality_keeps_every_word_it_was_given():
    words = (_word("a", 0.0, 0.5, 0.9), _word("b", 0.5, 1.0, 0.1))
    report = evaluate_alignment_quality(
        make_alignment_result_normal_song(), low_confidence_words=words
    )

    assert report.word_confidences == words
    assert evaluate_alignment_quality(make_alignment_result_normal_song()).word_confidences == ()


def test_the_log_names_the_chunks_and_points_at_alignment_override(caplog):
    flagged = timing_unverified_chunks([_chunk(63, 438.833, 446.833)], _report(*CHUNK_63_WORDS))

    with caplog.at_level(logging.WARNING):
        log_timing_unverified(flagged, can_override=True)

    text = caplog.text
    assert "63" in text and "timing unverified" in text
    assert "[[alignment_override]]" in text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_the_log_does_not_point_a_prologue_at_overrides_it_would_ignore(caplog):
    flagged = timing_unverified_chunks([_chunk(1, 0.0, 2.0)], _report(_word("a", 0.0, 1.0, 0.0)))

    with caplog.at_level(logging.WARNING):
        log_timing_unverified(flagged, can_override=False)

    assert "timing unverified" in caplog.text
    assert "[[alignment_override]]" not in caplog.text


def test_nothing_flagged_logs_nothing(caplog):
    with caplog.at_level(logging.DEBUG):
        log_timing_unverified({}, can_override=True)
    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# run_state.json
# --------------------------------------------------------------------------- #


def _entry(chunk_id: int = 63) -> TimingUnverified:
    return TimingUnverified(
        chunk_id=chunk_id,
        start=438.833,
        end=446.833,
        word_count=7,
        low_word_count=4,
        worst_word="life",
        worst_probability=0.0,
        word_floor=0.35,
        share_threshold=0.5,
    )


def test_run_state_round_trips_timing_unverified(tmp_path: Path):
    state = RunState(run_id="r", timing_unverified={63: _entry()})
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps(dump_run_state(state)))

    loaded = load_run_state(path)

    assert loaded.timing_unverified == {63: _entry()}
    assert json.loads(path.read_text())["timing_unverified"]["63"]["chunk_id"] == 63


def test_a_run_state_written_before_the_field_loads_as_empty(tmp_path: Path):
    payload = dump_run_state(RunState(run_id="r"))
    payload.pop("timing_unverified")
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps(payload))

    assert load_run_state(path).timing_unverified == {}


def test_adding_the_field_did_not_bump_the_schema():
    from music_video_maker.resilience import RUN_STATE_SCHEMA_VERSION

    assert dump_run_state(RunState(run_id="r"))["schema_version"] == RUN_STATE_SCHEMA_VERSION


def test_render_run_records_the_flagged_chunks_and_replaces_a_resumed_runs(tmp_path: Path):
    from tests.test_resilience import (
        StubExecutionClient,
        _make_runner,
        _provider_returning,
        _rendered,
    )

    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    runner = _make_runner(client, tmp_path)
    state = runner.render_run(
        [1], _provider_returning(), tmp_path / "chunks", timing_unverified={63: _entry()}
    )
    assert state.timing_unverified == {63: _entry()}
    assert load_run_state(tmp_path / "run_state.json").timing_unverified == {63: _entry()}

    # A resume re-aligns: the new alignment's verdict replaces the old one.
    client2 = StubExecutionClient({})
    resumed = _make_runner(client2, tmp_path).render_run(
        [1], _provider_returning(), tmp_path / "chunks", resume=True, timing_unverified={}
    )
    assert resumed.timing_unverified == {}


def test_a_caller_that_does_not_pass_it_leaves_the_state_empty(tmp_path: Path):
    from tests.test_resilience import (
        StubExecutionClient,
        _make_runner,
        _provider_returning,
        _rendered,
    )

    client = StubExecutionClient({1: [_rendered(1, tmp_path)]})
    state = _make_runner(client, tmp_path).render_run(
        [1], _provider_returning(), tmp_path / "chunks"
    )
    assert state.timing_unverified == {}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_key_defaults_to_the_module_default_and_is_read(tmp_path: Path):
    from tests.test_config import _create_default_assets, _write_config

    _create_default_assets(tmp_path)

    assert load_config(_write_config(tmp_path)).timing_unverified_share == TIMING_UNVERIFIED_SHARE
    cfg = load_config(_write_config(tmp_path, extra_toml="timing_unverified_share = 0.75"))
    assert cfg.timing_unverified_share == 0.75


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", '"half"'])
def test_config_key_outside_zero_exclusive_one_inclusive_is_refused(tmp_path: Path, bad: str):
    from tests.test_config import _create_default_assets, _write_config

    _create_default_assets(tmp_path)

    with pytest.raises(ConfigError, match="timing_unverified_share"):
        load_config(_write_config(tmp_path, extra_toml=f"timing_unverified_share = {bad}"))


# --------------------------------------------------------------------------- #
# End to end through the real pipeline wiring (the stable-ts model is faked)
# --------------------------------------------------------------------------- #


def _make_first_segment_unconfident(rig, probability: float = 0.05) -> None:
    for word in rig.align_model._raw_result.segments[0].words:
        word.probability = probability


def test_a_real_run_records_the_unverified_chunk_in_run_state_and_the_log(
    tmp_path: Path, caplog
):
    from tests.test_cli import Rig, build_success_sequence

    rig = Rig(tmp_path)
    _make_first_segment_unconfident(rig)
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]

    with caplog.at_level(logging.WARNING):
        report = rig.run(sequences)

    assert report.rendered == 3
    saved = load_run_state(rig.config.run_state_file)
    assert list(saved.timing_unverified) == [0]
    assert saved.timing_unverified[0].low_word_count == 6
    assert "timing unverified" in caplog.text
    assert "[[alignment_override]]" in caplog.text


def test_flagging_a_chunk_changes_nothing_that_renders(tmp_path: Path):
    from tests.test_cli import Rig, build_success_sequence

    def render(root: Path, unconfident: bool):
        rig = Rig(root)
        if unconfident:
            _make_first_segment_unconfident(rig)
        sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
        rig.run(sequences)
        state = load_run_state(rig.config.run_state_file)
        return [
            (cid, r.status, r.fingerprint.start, r.fingerprint.end, r.fingerprint.prompt_hash)
            for cid, r in sorted(state.results.items())
        ]

    assert render(tmp_path / "a", False) == render(tmp_path / "b", True)


def test_prepare_carries_the_verdict_for_the_review_page(tmp_path: Path):
    from music_video_maker import cli
    from tests.test_cli import Rig

    rig = Rig(tmp_path)
    _make_first_segment_unconfident(rig)

    prepared = cli.prepare_timeline(rig.config, align_model=rig.align_model)

    assert list(prepared.timing_unverified) == [0]


def test_review_shows_the_chunk_and_the_override_pointer(tmp_path: Path):
    from music_video_maker.review import build_review, render_review_html, render_review_json
    from tests.test_cli import Rig

    rig = Rig(tmp_path)
    _make_first_segment_unconfident(rig)

    data = build_review(rig.config, align_model=rig.align_model)

    first, second = data.chunks[0], data.chunks[1]
    assert first.timing_unverified is not None
    assert first.timing_unverified.low_word_count == 6
    assert second.timing_unverified is None
    as_json = json.loads(render_review_json(data))
    assert as_json["chunks"][0]["timing_unverified"]["worst_word"] == "walking"
    assert as_json["chunks"][1]["timing_unverified"] is None
    html = render_review_html(data)
    assert "timing unverified" in html
    assert "[[alignment_override]]" in html


def test_a_span_the_operator_already_pinned_is_not_flagged_again(tmp_path: Path):
    from dataclasses import replace

    from music_video_maker import cli
    from music_video_maker.contracts import AlignmentOverride
    from tests.test_cli import Rig

    rig = Rig(tmp_path)
    _make_first_segment_unconfident(rig)
    pinned = replace(
        rig.config,
        alignment_overrides=(
            AlignmentOverride(segment_index=0, start=0.0, end=6.5, reason="checked by ear"),
        ),
    )

    prepared = cli.prepare_timeline(pinned, align_model=rig.align_model)

    assert prepared.timing_unverified == {}
