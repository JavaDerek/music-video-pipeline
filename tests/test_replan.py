"""Carrying a human-reviewed shot plan onto a changed chunk timeline.

General form: a shot plan anchors each entry to a chunk's *start*, and the
render refuses a chunk whose start drifted (``ShotPlanDriftError``). Anything
that moves boundaries -- an alignment override, ``phrase_aware_slicing``,
``boundary_overrun`` -- leaves a reviewed plan unloadable, and the only path
that existed (``--prepare``) emits a *blank* skeleton. Re-authoring through
``mvm-author`` would re-generate the prose a human already approved.

``authoring.replan`` carries each entry onto the new chunk that holds the same
sung words (identified by segment index and word position, so a phrase an
override moved twelve seconds still finds its own direction) or the same
instrumental span, copies its fields verbatim, and **flags** -- never fills
in -- every new chunk without one clear source. Pure mapping on hand-built
timelines here; the CLI is exercised with Stage 1-2 stubbed out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from music_video_maker.authoring import replan
from music_video_maker.contracts import AlignedSegment, AlignmentResult, AudioChunk, WordTiming
from music_video_maker.shot_plan import load_shot_plan, resolve_shot

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def _segment(index: int, start: float, end: float, words: list[str]) -> AlignedSegment:
    span = (end - start) / len(words)
    timings = tuple(
        WordTiming(word=w, start=start + i * span, end=start + (i + 1) * span)
        for i, w in enumerate(words)
    )
    return AlignedSegment(
        index=index, text=" ".join(words), start=start, end=end, words=timings,
        characters=("Dianne",),
    )


def _alignment(*segments: AlignedSegment, track: float = 60.0) -> AlignmentResult:
    return AlignmentResult(segments=segments, track_duration=track)


def _chunks(edges: list[float], alignment: AlignmentResult) -> tuple[AudioChunk, ...]:
    chunks = []
    for cid, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        words = [
            w.word for s in alignment.segments for w in s.words
            if start <= (w.start + w.end) / 2 < end
        ]
        sources = tuple(
            s.index for s in alignment.segments if s.start < end and s.end > start
        )
        chunks.append(
            AudioChunk(
                chunk_id=cid,
                audio_file=Path(f"chunk_{cid:03d}.wav"),
                start=start,
                end=end,
                text=" ".join(words),
                characters=("Dianne",) if words else (),
                source_segment_indices=sources if words else (),
                is_instrumental=not words,
            )
        )
    return tuple(chunks)


def _plan(chunks, extra: dict[int, dict] | None = None) -> dict[int, dict]:
    extra = extra or {}
    return {
        c.chunk_id: {
            "chunk_id": c.chunk_id,
            "start": c.start,
            "shot": f"direction for old chunk {c.chunk_id}",
            "camera": f"camera {c.chunk_id}",
            **({"present": ["Jan"]} if c.chunk_id % 2 else {}),
            **extra.get(c.chunk_id, {}),
        }
        for c in chunks
    }


SEG_A = _segment(0, 12.2, 17.6, ["out", "of", "the", "heat", "and", "into", "the", "fire"])
SEG_B = _segment(1, 18.1, 24.9, ["the", "souls", "falling", "down", "and", "flames"])


# --------------------------------------------------------------------------- #
# The mapping
# --------------------------------------------------------------------------- #


def test_an_unchanged_timeline_carries_every_entry_verbatim():
    alignment = _alignment(SEG_A, SEG_B, track=36.0)
    chunks = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], alignment)
    plan = _plan(chunks)

    result = replan.map_plan(chunks, alignment, chunks, alignment, plan)

    assert result.flagged == ()
    assert [m.source_chunk_id for m in result.mapped] == [c.chunk_id for c in chunks]
    for mapping in result.mapped:
        carried = plan[mapping.source_chunk_id]
        anchors = ("chunk_id", "start")
        assert mapping.fields == {k: v for k, v in carried.items() if k not in anchors}


def test_a_moved_boundary_carries_by_where_the_words_went():
    alignment = _alignment(SEG_A, SEG_B, track=36.0)
    old = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], alignment)
    # Old chunk 2 ended at 18.0 inside nothing; new boundaries shifted by 0.4 s.
    new = _chunks([0.0, 5.8, 11.6, 17.8, 25.1, 30.7, 36.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    assert result.flagged == ()
    assert [m.source_chunk_id for m in result.mapped] == [0, 1, 2, 3, 4, 5]


def test_a_phrase_reunited_from_two_old_chunks_is_flagged_with_both_candidates():
    alignment = _alignment(SEG_A, SEG_B, track=36.0)
    # Old timeline cut SEG_A in half (the defect); the new one holds it whole.
    old = _chunks([0.0, 9.6, 14.9, 21.0, 26.2, 31.0, 36.0], alignment)
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))

    flagged = {f.new_chunk_id: f for f in result.flagged}
    assert 2 in flagged, "new chunk 2 holds SEG_A whole: no single old entry owns it"
    candidate_ids = [cid for cid, _share in flagged[2].candidates]
    assert set(candidate_ids) >= {1, 2}
    assert all(m.new_chunk_id != 2 for m in result.mapped)


def test_a_phrase_moved_by_an_override_follows_its_own_words():
    # The override moves segment 1 twelve seconds earlier. The old plan's
    # direction for it lives on old chunk 4 (where the words USED to be); by
    # time, the new chunk now holding them overlaps old chunk 2 instead.
    seg_b_late = _segment(1, 30.4, 36.6, ["deathless", "forevermore"])
    seg_b_early = _segment(1, 18.4, 24.6, ["deathless", "forevermore"])
    old_alignment = _alignment(SEG_A, seg_b_late, track=48.0)
    new_alignment = _alignment(SEG_A, seg_b_early, track=48.0)
    old = _chunks([0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 37.0, 42.5, 48.0], old_alignment)
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.0, 31.0, 37.0, 42.5, 48.0], new_alignment)

    result = replan.map_plan(old, old_alignment, new, new_alignment, _plan(old))

    by_new = {m.new_chunk_id: m.source_chunk_id for m in result.mapped}
    assert by_new[3] == 5, "the words' own old chunk, not the one at the same time"


def test_an_instrumental_run_with_the_same_count_maps_in_order():
    alignment = _alignment(_segment(0, 30.3, 36.1, ["only", "line", "here"]), track=50.0)
    old = _chunks([0.0, 8.0, 16.0, 24.0, 30.0, 37.0, 43.5, 50.0], alignment)
    new = _chunks([0.0, 7.3, 14.6, 21.9, 30.2, 37.5, 43.5, 50.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    assert result.flagged == ()
    assert [m.source_chunk_id for m in result.mapped] == [0, 1, 2, 3, 4, 5, 6]


def test_an_instrumental_chunk_straddling_two_old_shots_is_flagged():
    alignment = _alignment(_segment(0, 30.3, 36.1, ["only", "line", "here"]), track=50.0)
    old = _chunks([0.0, 7.5, 15.0, 22.5, 30.0, 37.0, 43.5, 50.0], alignment)
    # Three old intro shots, now four -- no count-preserving correspondence.
    new = _chunks([0.0, 5.5, 11.5, 17.5, 23.6, 30.0, 37.0, 43.5, 50.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    flagged = {f.new_chunk_id for f in result.flagged}
    assert flagged & {1, 2, 3}
    for f in result.flagged:
        assert f.reason


def test_one_old_entry_is_never_carried_onto_two_new_chunks():
    alignment = _alignment(SEG_A, SEG_B, track=40.0)
    # Old chunk 2 held both phrases; the new timeline gives each its own.
    old = _chunks([0.0, 6.0, 12.0, 25.0, 31.0, 40.0], alignment)
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.0, 31.0, 40.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    sources = [m.source_chunk_id for m in result.mapped]
    assert len(sources) == len(set(sources))
    assert any("already carried" in f.reason for f in result.flagged)


def test_a_source_of_the_other_kind_is_flagged_not_carried():
    # New chunk 2 is voiced, but the only old entry it could take was authored
    # as INSTRUMENTAL (the phrase sat demoted inside a long filler) -- an
    # instrumental shot's subject/focus is refused on a sung chunk.
    alignment = _alignment(_segment(0, 13.0, 13.4, ["hey"]), track=30.0)
    old = _chunks([0.0, 6.0, 12.0, 18.0, 24.0, 30.0], alignment)
    old = tuple(c if c.chunk_id != 2 else _instrumental(c) for c in old)
    new = _chunks([0.0, 6.0, 12.5, 18.0, 24.0, 30.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    flagged = {f.new_chunk_id: f for f in result.flagged}
    assert 2 in flagged and "instrumental" in flagged[2].reason


def _instrumental(chunk: AudioChunk) -> AudioChunk:
    from dataclasses import replace

    return replace(chunk, text="", characters=(), source_segment_indices=(), is_instrumental=True)


def test_an_old_chunk_with_no_plan_entry_is_never_a_source():
    alignment = _alignment(SEG_A, SEG_B, track=36.0)
    chunks = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], alignment)
    plan = _plan(chunks)
    del plan[3]
    result = replan.map_plan(chunks, alignment, chunks, alignment, plan)
    flagged = {f.new_chunk_id: f for f in result.flagged}
    assert 3 in flagged and "no entry" in flagged[3].reason


# --------------------------------------------------------------------------- #
# The written file is checked by the render's own loader
# --------------------------------------------------------------------------- #


def test_the_written_plan_loads_and_resolves_against_the_new_chunks(tmp_path):
    alignment = _alignment(SEG_A, SEG_B, track=36.0)
    old = _chunks([0.0, 9.6, 14.9, 21.0, 26.2, 31.0, 36.0], alignment)
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], alignment)
    plan = _plan(old, {0: {"framing": "wide", "focus": "action", "shot": 'a "quoted" line'}})
    result = replan.map_plan(old, alignment, new, alignment, plan)

    text = replan.render_plan_toml(
        result, new, provenance={"generated_by": "mvm-author 0.1.0"}, header="# test\n"
    )
    out = tmp_path / "replanned.toml"
    out.write_text(text)

    payload = tomllib.loads(text)
    assert payload["provenance"] == {"generated_by": "mvm-author 0.1.0"}
    loaded = load_shot_plan(out)
    assert sorted(loaded) == [c.chunk_id for c in new]
    for chunk in new:
        resolve_shot(loaded, chunk)  # raises ShotPlanDriftError on any drift
    by_new = {m.new_chunk_id: m for m in result.mapped}
    raw = {entry["chunk_id"]: entry for entry in payload["shot"]}
    for cid, mapping in by_new.items():
        for key, value in mapping.fields.items():
            assert raw[cid][key] == value
    for flag in result.flagged:
        assert raw[flag.new_chunk_id]["shot"] == ""
        assert f"REANCHOR FLAG chunk_id={flag.new_chunk_id}" in text


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def _write_plan(path: Path, chunks, plan) -> Path:
    path.write_text(
        replan.render_plan_toml(
            replan.map_plan(chunks, _ALIGN, chunks, _ALIGN, plan), chunks, provenance=None,
            header="",
        )
    )
    return path


_ALIGN = _alignment(SEG_A, SEG_B, track=36.0)


@pytest.fixture
def stub_stage_1_2(monkeypatch, tmp_path):
    old = _chunks([0.0, 9.6, 14.9, 21.0, 26.2, 31.0, 36.0], _ALIGN)
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], _ALIGN)
    timelines = {"old.toml": (_ALIGN, old), "new.toml": (_ALIGN, new)}
    calls: list[Path] = []

    def fake_load_config(path):
        return Path(path)

    def fake_timeline(config, scratch):
        calls.append(scratch)
        return timelines[config.name]

    monkeypatch.setattr(replan, "load_config", fake_load_config)
    monkeypatch.setattr(replan, "_timeline", fake_timeline)
    plan_path = _write_plan(tmp_path / "plan.toml", old, _plan(old))
    return tmp_path, plan_path, calls


def test_cli_writes_the_carried_plan_and_says_how_many_were_flagged(stub_stage_1_2, capsys):
    tmp_path, plan_path, _calls = stub_stage_1_2
    out = tmp_path / "out.toml"
    code = replan.main(
        ["--old-config", "old.toml", "--config", "new.toml", "--plan", str(plan_path),
         "--out", str(out)]
    )
    assert code == replan.EXIT_FLAGGED
    assert out.exists()
    report = capsys.readouterr().out
    assert "carried" in report and "flagged" in report


def test_cli_refuses_to_overwrite(stub_stage_1_2):
    tmp_path, plan_path, _calls = stub_stage_1_2
    out = tmp_path / "out.toml"
    out.write_text("keep me")
    code = replan.main(
        ["--old-config", "old.toml", "--config", "new.toml", "--plan", str(plan_path),
         "--out", str(out)]
    )
    assert code == replan.EXIT_ERROR
    assert out.read_text() == "keep me"


def test_cli_refuses_a_plan_that_was_not_authored_against_the_old_timeline(stub_stage_1_2):
    tmp_path, _plan_path, _calls = stub_stage_1_2
    new = _chunks([0.0, 6.0, 12.0, 18.0, 25.5, 30.7, 36.0], _ALIGN)
    wrong = _write_plan(tmp_path / "wrong.toml", new, _plan(new))
    code = replan.main(
        ["--old-config", "old.toml", "--config", "new.toml", "--plan", str(wrong),
         "--out", str(tmp_path / "out.toml")]
    )
    assert code == replan.EXIT_ERROR
    assert not (tmp_path / "out.toml").exists()


def test_cli_refuses_a_plan_with_length_requests(stub_stage_1_2):
    tmp_path, plan_path, _calls = stub_stage_1_2
    line = 'shot = "direction for old chunk 1"'
    text = plan_path.read_text().replace(line, line + "\nlength_seconds = 9.0", 1)
    plan_path.write_text(text)
    code = replan.main(
        ["--old-config", "old.toml", "--config", "new.toml", "--plan", str(plan_path),
         "--out", str(tmp_path / "out.toml")]
    )
    assert code == replan.EXIT_ERROR


def test_stage_1_2_never_writes_into_a_configs_own_chunks_dir(stub_stage_1_2):
    tmp_path, plan_path, calls = stub_stage_1_2
    replan.main(
        ["--old-config", "old.toml", "--config", "new.toml", "--plan", str(plan_path),
         "--out", str(tmp_path / "out.toml")]
    )
    assert len(calls) == 2
    for scratch in calls:
        assert not scratch.exists(), "stems belong in a temporary directory that is removed"


def test_stage_1_2_slices_into_the_scratch_dir_not_the_configs(tmp_path):
    from tests.test_authoring_concept import _config

    config = _config(tmp_path)  # empty lyrics: a wholly instrumental timeline
    scratch = tmp_path / "scratch"
    alignment, chunks = replan._timeline(config, scratch)
    assert chunks and all(c.is_instrumental for c in chunks)
    assert alignment.segments == ()
    assert list(scratch.glob("chunk_*.wav"))
    assert not config.chunks_dir.exists() or not list(config.chunks_dir.glob("chunk_*.wav"))


def test_toml_values_round_trip_and_refuse_what_a_plan_cannot_hold():
    text = "\n".join(
        f"{k} = {replan._toml_value(v)}"
        for k, v in {"a": True, "b": 3, "c": 1.5, "d": ["x", 'y "q"'], "e": "é\n"}.items()
    )
    assert tomllib.loads(text) == {"a": True, "b": 3, "c": 1.5, "d": ["x", 'y "q"'], "e": "é\n"}
    with pytest.raises(TypeError):
        replan._toml_value({"nested": "table"})


def test_a_chunk_overlapping_nothing_old_is_flagged():
    alignment = _alignment(SEG_A, track=30.0)
    old = _chunks([0.0, 6.0, 12.0, 18.0, 24.0], alignment)  # ends short of the new timeline
    new = _chunks([0.0, 6.0, 12.0, 18.0, 24.0, 30.0], alignment)
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    flagged = {f.new_chunk_id: f for f in result.flagged}
    assert 4 in flagged and "no old chunk" in flagged[4].reason


def test_an_instrumental_passage_retiled_to_fewer_shots_carries_in_order():
    alignment = _alignment(_segment(0, 30.3, 36.1, ["only", "line", "here"]), track=50.0)
    old = _chunks([0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 37.0, 43.5, 50.0], alignment)  # 5 intro shots
    new = _chunks([0.0, 7.5, 15.0, 22.5, 30.0, 37.0, 43.5, 50.0], alignment)  # now 4
    result = replan.map_plan(old, alignment, new, alignment, _plan(old))
    intro = [m for m in result.mapped if m.new_chunk_id < 4]
    assert [m.source_chunk_id for m in intro] == sorted(m.source_chunk_id for m in intro)
    assert len(intro) == 4 and all(m.retiled for m in intro)
    assert len(result.dropped) == 1
