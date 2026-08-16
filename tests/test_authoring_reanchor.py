"""Re-anchoring beats onto the timeline their own lengths produced
(issue #54 design section 5).

The design calls this "the piece most likely to be subtly wrong", and it is:
a beat authored against chunk 12 of one timeline has to find chunk 9 of
another, by song time, without ever consulting a chunk id. Getting it wrong
does not raise -- it silently attaches shot 12's direction to shot 9's audio,
which is exactly the failure ``ShotPlanDriftError`` exists to make loud.

Everything here is a pure function of two lists, so no audio, no model and no
slicing is involved: the chunk timelines are built by hand precisely so the
degenerate cases the design names (a length on the first chunk, on the last,
two adjacent, one that gets clamped, one that swallows an instrumental gap
whole) can be constructed exactly rather than coaxed out of the slicer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.reanchor import reanchor_beats
from music_video_maker.contracts import AudioChunk


def _chunks(spans: list[tuple[float, float]], *, instrumental_from: int | None = None):
    """A contiguous chunk timeline from ``(start, end)`` spans, numbered from 1."""
    return tuple(
        AudioChunk(
            chunk_id=index + 1,
            audio_file=Path(f"chunk_{index + 1:04d}.wav"),
            start=start,
            end=end,
            text="" if instrumental_from is not None and index >= instrumental_from else "a lyric",
            is_instrumental=instrumental_from is not None and index >= instrumental_from,
        )
        for index, (start, end) in enumerate(spans)
    )


def _beat(
    chunk_id, start, end, *,
    role="transition", group=1, focus="subject", length=None, location="the room",
    act="the story", subject=None,
):
    return Beat(
        chunk_id=chunk_id,
        start=start,
        end=end,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=group,
        location=location,
        act=act,
        focus=focus,
        length_seconds=length,
        subject=subject,
    )


def _uniform(n: int, width: float = 6.0):
    return [(i * width, (i + 1) * width) for i in range(n)]


# --------------------------------------------------------------------------- #
# The identity case: nothing moved
# --------------------------------------------------------------------------- #


def test_an_unchanged_timeline_returns_the_beats_untouched():
    chunks = _chunks(_uniform(4))
    beats = tuple(_beat(c.chunk_id, c.start, c.end) for c in chunks)

    result = reanchor_beats(beats, chunks)

    assert result.beats == beats
    assert result.unmapped_chunk_ids == ()
    assert result.merges == ()


# --------------------------------------------------------------------------- #
# Mapping is by song time, never by chunk id
# --------------------------------------------------------------------------- #


def test_a_beat_follows_its_midpoint_not_its_chunk_id():
    """v0 chunk 3 (12-18 s) becomes v1 chunk 2 once a long take swallows the
    two chunks before it. The direction has to follow the audio."""
    v0 = _chunks(_uniform(4))
    v1 = _chunks([(0.0, 12.0), (12.0, 18.0), (18.0, 24.0)])
    beats = tuple(_beat(c.chunk_id, c.start, c.end) for c in v0)

    result = reanchor_beats(beats, v1)

    third = next(b for b in result.beats if "beat 3" in b.beat)
    assert third.chunk_id == 2
    assert (third.start, third.end) == (12.0, 18.0)


def test_a_beat_whose_midpoint_lands_exactly_on_a_boundary_goes_to_the_later_chunk():
    """Half-open spans, the same convention every other timeline consumer in
    this project uses: ``start <= t < end``. Stated in a test because the
    alternative is a silent off-by-one on exactly the beats a long take
    produced."""
    v1 = _chunks([(0.0, 6.0), (6.0, 12.0)])
    beats = (_beat(1, 3.0, 9.0),)  # midpoint exactly 6.0

    result = reanchor_beats(beats, v1)

    assert result.beats[0].chunk_id == 2


# --------------------------------------------------------------------------- #
# Many-to-one: what a long take is for
# --------------------------------------------------------------------------- #


def test_several_beats_swallowed_by_one_long_take_merge_in_chunk_order():
    v0 = _chunks(_uniform(4))
    v1 = _chunks([(0.0, 18.0), (18.0, 24.0)])
    beats = tuple(_beat(c.chunk_id, c.start, c.end) for c in v0)

    result = reanchor_beats(beats, v1)

    assert len(result.beats) == 2
    merged = result.beats[0]
    assert merged.chunk_id == 1
    assert merged.merged_from == (1, 2, 3)
    assert merged.beat == "beat 1; beat 2; beat 3"
    assert result.merges == ((1, (1, 2, 3)),)


def test_a_merge_that_swallowed_a_consequence_keeps_focus_on_the_action():
    """The one merge rule the design states outright. ``focus`` is per-chunk
    state; losing it in a merge would put the performer back in the subject
    slot for a shot whose whole job is the state change -- the bug
    ``focus = "action"`` exists to fix."""
    v1 = _chunks([(0.0, 18.0)])
    beats = (
        _beat(1, 0.0, 6.0, role="plant"),
        _beat(2, 6.0, 12.0, role="contact"),
        _beat(3, 12.0, 18.0, role="consequence", focus="action"),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].focus == "action"
    assert result.beats[0].beat_role == "consequence"


def test_a_merge_keeps_the_first_members_group():
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, group=2, role="plant"),
        _beat(2, 6.0, 12.0, group=3, role="contact"),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].beat_group == 2


def test_a_merge_warns_when_it_joins_two_different_beat_groups(caplog):
    """Not an error -- the timeline really did merge them and the prose stage
    still needs one line for that chunk -- but a group boundary vanishing is
    the thing that turns a three-beat gag into one compressed shot, which is
    the whole defect ``docs/shot-writing-guide.md`` exists to prevent."""
    v1 = _chunks([(0.0, 12.0)])
    beats = (_beat(1, 0.0, 6.0, group=1), _beat(2, 6.0, 12.0, group=2))

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "beat group" in caplog.text.lower()


def test_a_merge_keeps_the_first_members_location():
    """`location` follows the same "first member wins" rule as `beat_group`
    -- issue #78's field is a beat-derived anchor like any other, and gets
    the same merge treatment."""
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, location="the mill"),
        _beat(2, 6.0, 12.0, location="the watch-post"),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].location == "the mill"


def test_a_merge_warns_when_it_joins_two_different_locations(caplog):
    """Not an error, same reason as the beat-group warning above: the
    timeline really did merge them into one chunk. But two different
    `location` values folding into one chunk silently is exactly the kind of
    thing issue #78 exists to make loud instead."""
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, location="the mill"),
        _beat(2, 6.0, 12.0, location="the watch-post"),
    )

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "location" in caplog.text.lower()
    assert "the mill" in caplog.text
    assert "the watch-post" in caplog.text


def test_a_merge_with_matching_locations_does_not_warn(caplog):
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, location="the mill"),
        _beat(2, 6.0, 12.0, location="the mill"),
    )

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "location" not in caplog.text.lower()


# --------------------------------------------------------------------------- #
# `act` -- issue #84 -- follows the exact same merge treatment as `location`:
# first member wins, a mismatch is warned about rather than silently dropped.
# --------------------------------------------------------------------------- #


def test_a_merge_keeps_the_first_members_act():
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, act="situation"),
        _beat(2, 6.0, 12.0, act="resolution"),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].act == "situation"


def test_a_merge_warns_when_it_joins_two_different_acts(caplog):
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, act="situation"),
        _beat(2, 6.0, 12.0, act="resolution"),
    )

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "act" in caplog.text.lower()
    assert "situation" in caplog.text
    assert "resolution" in caplog.text


def test_a_merge_with_matching_acts_does_not_warn(caplog):
    v1 = _chunks([(0.0, 12.0)])
    beats = (
        _beat(1, 0.0, 6.0, act="situation"),
        _beat(2, 6.0, 12.0, act="situation"),
    )

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "different act" not in caplog.text.lower()


def test_a_single_member_chunk_keeps_its_own_act():
    """The identity path -- `replace()`, not `_merge`'s multi-member branch
    -- must carry `act` through exactly like every other field."""
    v0 = _chunks(_uniform(2))
    beats = (_beat(1, 0.0, 6.0, act="situation"), _beat(2, 6.0, 12.0, act="resolution"))

    result = reanchor_beats(beats, v0)

    assert [b.act for b in result.beats] == ["situation", "resolution"]


# --------------------------------------------------------------------------- #
# `subject` -- issue #82. Rides along like `location`/`act`, BUT a beat
# re-anchored onto a chunk that is now VOICED must have it dropped, not
# carried: the field is illegal there (issues #58, #59, #60), and a
# `length_seconds` re-cut can change which chunks are voiced.
# --------------------------------------------------------------------------- #


def test_a_single_beats_subject_survives_reanchoring_onto_a_still_instrumental_chunk():
    v0 = _chunks(_uniform(2), instrumental_from=0)
    beats = (_beat(1, 0.0, 6.0, subject="Jan"), _beat(2, 6.0, 12.0))

    result = reanchor_beats(beats, v0)

    assert result.beats[0].subject == "Jan"


def test_a_single_beats_subject_is_dropped_when_its_chunk_is_now_voiced(caplog):
    """The identity path (one beat, one chunk) must apply the same drop rule
    as a merge -- a `length_seconds` re-cut is not the only way a beat's
    target chunk can turn out to be voiced."""
    v1 = _chunks([(0.0, 6.0)])  # voiced: instrumental_from unset
    beats = (_beat(1, 0.0, 6.0, subject="Jan"),)

    with caplog.at_level(logging.WARNING):
        result = reanchor_beats(beats, v1)

    assert result.beats[0].subject is None
    assert "subject" in caplog.text.lower()


def test_a_merge_keeps_the_only_set_subject():
    v1 = _chunks([(0.0, 12.0)], instrumental_from=0)
    beats = (
        _beat(1, 0.0, 6.0, subject=None),
        _beat(2, 6.0, 12.0, subject="Jan"),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].subject == "Jan"


def test_a_merge_warns_when_it_joins_two_different_subjects(caplog):
    v1 = _chunks([(0.0, 12.0)], instrumental_from=0)
    beats = (
        _beat(1, 0.0, 6.0, subject="Jan"),
        _beat(2, 6.0, 12.0, subject="Dianne"),
    )

    with caplog.at_level(logging.WARNING):
        result = reanchor_beats(beats, v1)

    assert result.beats[0].subject == "Jan"
    assert "subject" in caplog.text.lower()
    assert "Jan" in caplog.text
    assert "Dianne" in caplog.text


def test_a_merge_with_matching_subjects_does_not_warn(caplog):
    v1 = _chunks([(0.0, 12.0)], instrumental_from=0)
    beats = (
        _beat(1, 0.0, 6.0, subject="Jan"),
        _beat(2, 6.0, 12.0, subject="Jan"),
    )

    with caplog.at_level(logging.WARNING):
        reanchor_beats(beats, v1)

    assert "different subject" not in caplog.text.lower()


def test_a_merge_drops_subject_when_the_merged_chunk_is_now_voiced(caplog):
    """The case the issue exists for: honouring a `length_seconds` re-cut
    swallowed an instrumental beat's subject into a chunk that is voiced in
    the retiled timeline."""
    v1 = _chunks([(0.0, 12.0)])  # voiced: instrumental_from unset
    beats = (
        _beat(1, 0.0, 6.0, subject="Jan"),
        _beat(2, 6.0, 12.0),
    )

    with caplog.at_level(logging.WARNING):
        result = reanchor_beats(beats, v1)

    assert result.beats[0].subject is None
    assert "subject" in caplog.text.lower()


def test_a_merge_onto_a_still_instrumental_chunk_keeps_the_subject():
    v1 = _chunks([(0.0, 12.0)], instrumental_from=0)
    beats = (
        _beat(1, 0.0, 6.0, subject="Jan"),
        _beat(2, 6.0, 12.0),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].subject == "Jan"


# --------------------------------------------------------------------------- #
# A v1 chunk nothing maps into
# --------------------------------------------------------------------------- #


def test_a_v1_chunk_no_beat_maps_into_is_reported_never_left_blank():
    """Design section 5: "flagged and sent back to stage 2 ... not silently
    left blank". A blank direction renders as the global narrative_concept,
    which looks deliberate and is not."""
    v0 = _chunks([(0.0, 6.0), (6.0, 12.0)])
    v1 = _chunks([(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
    beats = tuple(_beat(c.chunk_id, c.start, c.end) for c in v0)

    result = reanchor_beats(beats, v1)

    # Midpoints 3.0 and 9.0 land in v1 chunks 1 and 3; chunk 2 gets nothing.
    assert result.unmapped_chunk_ids == (2,)


def test_unmapped_chunks_do_not_suppress_the_beats_that_did_map():
    v1 = _chunks([(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
    beats = (_beat(1, 0.0, 6.0), _beat(2, 6.0, 12.0))

    result = reanchor_beats(beats, v1)

    assert [b.chunk_id for b in result.beats] == [1, 3]


# --------------------------------------------------------------------------- #
# The degenerate cases design section 5 names by hand
# --------------------------------------------------------------------------- #


def test_a_length_on_the_very_first_chunk():
    v1 = _chunks([(0.0, 12.0), (12.0, 18.0), (18.0, 24.0)])
    beats = (
        _beat(1, 0.0, 6.0, length=12.0),
        _beat(2, 6.0, 12.0),
        _beat(3, 12.0, 18.0),
        _beat(4, 18.0, 24.0),
    )

    result = reanchor_beats(beats, v1)

    assert result.beats[0].chunk_id == 1
    assert result.beats[0].start == 0.0
    assert result.beats[0].length_seconds == 12.0
    assert [b.chunk_id for b in result.beats] == [1, 2, 3]


def test_a_length_on_the_last_authored_take_swallows_the_tail():
    v1 = _chunks([(0.0, 6.0), (6.0, 12.0), (12.0, 18.0), (18.0, 30.0)])
    beats = (
        _beat(1, 0.0, 6.0),
        _beat(2, 6.0, 12.0),
        _beat(3, 12.0, 18.0),
        _beat(4, 18.0, 24.0, length=12.0),
        _beat(5, 24.0, 30.0),
    )

    result = reanchor_beats(beats, v1)

    last = result.beats[-1]
    assert last.chunk_id == 4
    assert last.merged_from == (4, 5)
    assert last.length_seconds == 12.0


def test_two_adjacent_lengths_each_keep_their_own_chunk():
    v1 = _chunks([(0.0, 12.0), (12.0, 24.0), (24.0, 36.0)])
    beats = (
        _beat(1, 0.0, 6.0, length=12.0),
        _beat(2, 6.0, 12.0),
        _beat(3, 12.0, 18.0, length=12.0),
        _beat(4, 18.0, 24.0),
        _beat(5, 24.0, 30.0),
        _beat(6, 30.0, 36.0),
    )

    result = reanchor_beats(beats, v1)

    assert [b.chunk_id for b in result.beats] == [1, 2, 3]
    assert [b.length_seconds for b in result.beats] == [12.0, 12.0, None]
    assert [b.merged_from for b in result.beats] == [(1, 2), (3, 4), (5, 6)]


def test_a_clamped_length_keeps_the_authors_request_not_the_achieved_duration():
    """H3's trained ceiling is 15.083 s, so a 22 s take is clamped by the
    slicer. The beat keeps saying 22: re-quantizing the same request against
    the same alignment yields the same frames, and rewriting it to the clamped
    value would quietly launder a refusal into an agreement."""
    v1 = _chunks([(0.0, 15.083), (15.083, 21.083), (21.083, 27.083), (27.083, 36.0)])
    beats = tuple(_beat(i + 1, i * 6.0, (i + 1) * 6.0) for i in range(6))
    beats = (_beat(1, 0.0, 6.0, length=22.0),) + beats[1:]

    result = reanchor_beats(beats, v1)

    assert result.beats[0].length_seconds == 22.0
    assert result.beats[0].end == 15.083
    assert result.beats[0].merged_from == (1, 2, 3)
    assert result.unmapped_chunk_ids == ()


def test_a_length_that_swallows_an_instrumental_gap_whole():
    """The case a long take is most often authored for: one unbroken shot
    across a solo. Every instrumental beat inside it merges, and the merged
    entry is still anchored on the chunk the render will actually produce."""
    v0 = _chunks(
        [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0), (18.0, 24.0), (24.0, 30.0)],
        instrumental_from=1,
    )
    v1 = _chunks([(0.0, 6.0), (6.0, 24.0), (24.0, 30.0)], instrumental_from=1)
    beats = tuple(
        _beat(c.chunk_id, c.start, c.end, role="instrumental" if c.is_instrumental else "plant")
        for c in v0
    )

    result = reanchor_beats(beats, v1)

    solo = result.beats[1]
    assert solo.chunk_id == 2
    assert solo.merged_from == (2, 3, 4)
    assert solo.beat_role == "instrumental"
    assert result.unmapped_chunk_ids == ()


def test_a_second_length_swallowed_by_the_first_is_warned_about(caplog):
    """Two takes cannot both be on screen. ``slicing`` already warns when it
    drops the swallowed request; re-anchoring says the same thing in the
    authoring layer's own vocabulary, where the author can still act on it."""
    v1 = _chunks([(0.0, 18.0), (18.0, 24.0)])
    beats = (
        _beat(1, 0.0, 6.0, length=18.0),
        _beat(2, 6.0, 12.0, length=9.0),
        _beat(3, 12.0, 18.0),
        _beat(4, 18.0, 24.0),
    )

    with caplog.at_level(logging.WARNING):
        result = reanchor_beats(beats, v1)

    assert result.beats[0].length_seconds == 18.0
    assert "swallow" in caplog.text.lower()


# --------------------------------------------------------------------------- #
# Beats that fall off the end
# --------------------------------------------------------------------------- #


def test_a_beat_past_the_end_of_the_new_timeline_clamps_to_the_last_chunk():
    """The retiled timeline can be a fraction shorter than the one the beats
    were authored against (the outro tail deficit this project already
    records). Dropping the last beat there would lose authored direction for a
    rounding error."""
    v1 = _chunks([(0.0, 6.0), (6.0, 11.5)])
    beats = (_beat(1, 0.0, 6.0), _beat(2, 6.0, 12.0), _beat(3, 12.0, 18.0))

    result = reanchor_beats(beats, v1)

    assert [b.chunk_id for b in result.beats] == [1, 2]
    assert result.beats[-1].merged_from == (2, 3)
    assert result.unmapped_chunk_ids == ()


def test_no_beats_at_all_reports_every_chunk_as_unmapped():
    v1 = _chunks(_uniform(3))

    result = reanchor_beats((), v1)

    assert result.beats == ()
    assert result.unmapped_chunk_ids == (1, 2, 3)


def test_re_anchoring_onto_an_empty_timeline_raises_rather_than_losing_beats():
    with pytest.raises(ValueError):
        reanchor_beats((_beat(1, 0.0, 6.0),), ())


# --------------------------------------------------------------------------- #
# plan_beats: generate -> re-cut -> re-anchor, with the coverage loop
# --------------------------------------------------------------------------- #


def _reply(*entries: dict) -> dict:
    return {"beats": list(entries)}


def _entry(chunk_id, *, length=None) -> dict:
    entry = {
        "chunk_id": chunk_id,
        "beat": f"beat {chunk_id}",
        "beat_role": "transition",
        "beat_group": 1,
        "focus": "subject",
        "location": "the room",
        "act": "the story",
    }
    if length is not None:
        entry["length_seconds"] = length
    return entry


def _run_config(tmp_path: Path):
    from music_video_maker import contracts
    from music_video_maker.config import RunConfig
    from tests.harness.factories import make_cast_dict, write_silent_wav

    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("", encoding="utf-8")
    return RunConfig(
        master_audio=write_silent_wav(tmp_path / "master.wav", seconds=24.0),
        lyrics_file=lyrics_file,
        global_style="style",
        narrative_concept="placeholder",
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
    )


CONCEPT = {"logline": "a walk", "setting": "a town", "tone": "quiet", "motifs": [], "avoid": []}


def test_plan_beats_leaves_the_timeline_alone_when_no_length_is_asked_for(tmp_path):
    from music_video_maker.authoring.driver import ScriptedDriver
    from music_video_maker.authoring.reanchor import plan_beats

    chunks = _chunks(_uniform(4))
    driver = ScriptedDriver([_reply(*(_entry(c.chunk_id) for c in chunks))])

    def reslice(requests):
        pytest.fail("must not re-slice when the sheet asks for no lengths")

    plan = plan_beats(_run_config(tmp_path), chunks, CONCEPT, driver, reslice=reslice)

    assert plan.reanchored is False
    assert plan.chunks == chunks
    assert [b.chunk_id for b in plan.beats] == [1, 2, 3, 4]


def test_plan_beats_re_cuts_and_re_anchors_when_a_length_is_asked_for(tmp_path):
    from music_video_maker.authoring.driver import ScriptedDriver
    from music_video_maker.authoring.reanchor import plan_beats

    chunks = _chunks(_uniform(4))
    recut = _chunks([(0.0, 12.0), (12.0, 18.0), (18.0, 24.0)])
    driver = ScriptedDriver(
        [_reply(_entry(1, length=12.0), _entry(2), _entry(3), _entry(4))]
    )
    seen = {}

    def reslice(requests):
        seen["requests"] = tuple(requests)
        return recut

    plan = plan_beats(_run_config(tmp_path), chunks, CONCEPT, driver, reslice=reslice)

    assert [(r.start, r.length_seconds) for r in seen["requests"]] == [(0.0, 12.0)]
    assert plan.reanchored is True
    assert plan.chunks == recut
    assert [b.chunk_id for b in plan.beats] == [1, 2, 3]
    assert plan.merges == ((1, (1, 2)),)


def test_plan_beats_sends_an_uncovered_span_back_to_the_model(tmp_path):
    """Design section 5, step 3. A chunk with no beat only exists *after* the
    sheet's own lengths have re-tiled the timeline, so the validator cannot
    see it -- it has to come back here and go out again as work to redo."""
    from music_video_maker.authoring.driver import ScriptedDriver
    from music_video_maker.authoring.reanchor import plan_beats

    chunks = _chunks(_uniform(2))
    gappy = _chunks([(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
    driver = ScriptedDriver(
        [
            _reply(_entry(1, length=4.0), _entry(2)),
            _reply(_entry(1), _entry(2)),
        ]
    )

    plan = plan_beats(
        _run_config(tmp_path),
        chunks,
        CONCEPT,
        driver,
        reslice=lambda requests: gappy,
    )

    assert len(driver.calls) == 2
    assert "no direction" in driver.calls[1]["prompt"]
    assert "4.0-8.0s" in driver.calls[1]["prompt"]
    # The second sheet asks for no lengths, so the original timeline stands.
    assert plan.reanchored is False
    assert plan.chunks == chunks


def test_plan_beats_gives_up_rather_than_shipping_an_undirected_span(tmp_path):
    from music_video_maker.authoring.beats import BeatsValidationError
    from music_video_maker.authoring.driver import ScriptedDriver
    from music_video_maker.authoring.reanchor import plan_beats

    chunks = _chunks(_uniform(2))
    gappy = _chunks([(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)])
    sheet = _reply(_entry(1, length=4.0), _entry(2))
    driver = ScriptedDriver([sheet, sheet])

    with pytest.raises(BeatsValidationError, match="no direction"):
        plan_beats(
            _run_config(tmp_path),
            chunks,
            CONCEPT,
            driver,
            reslice=lambda requests: gappy,
        )

    assert len(driver.calls) == 2
