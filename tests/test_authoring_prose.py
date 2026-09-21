"""Tests for Stage 4 -- prose (issue #54 design section 4).

Prose is the stage that finally writes ``shot`` lines, so it is also the
stage with the most ways to write something that passes every structural
check and is still wrong. What is testable here is the *prohibitions*: the
design says prose is told not to write camera direction, not to restate
identity or the lyric, and "all four are then checked rather than trusted".

Two of those four are checked as errors and two as warnings, and the reason
is the guide itself -- see :mod:`music_video_maker.authoring.prose`'s
docstring. It is the sort of thing worth a test each, because the tiers are
a judgement call that a later reader will otherwise assume was an oversight.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from music_video_maker import contracts
from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.driver import MODEL_SONNET, DriverError, ScriptedDriver
from music_video_maker.authoring.prose import (
    ProseIssue,
    ProseValidationError,
    beat_windows,
    build_prose_prompt,
    generate_prose,
    plant_end_state_issues,
    pop_distant_staging_issues,
    pop_object_named_in_shot_issues,
    prose_input_hashes,
    validate_prose,
)
from music_video_maker.config import RunConfig
from tests.harness.factories import make_cast_dict, write_silent_wav

CONCEPT = {
    "logline": "A drummer walks through gathering weather.",
    "setting": "a coastal town, contemporary",
    "tone": "elegiac",
    "motifs": ["gathering clouds"],
    "avoid": ["literal lightning strikes"],
}


def _config(tmp_path: Path) -> RunConfig:
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("", encoding="utf-8")
    return RunConfig(
        master_audio=write_silent_wav(tmp_path / "master.wav", seconds=20.0),
        lyrics_file=lyrics_file,
        global_style="Refestramus progressive rock music video",
        narrative_concept="placeholder",
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
        setting="a coastal town, contemporary",
    )


def _chunks(texts: list[str], width: float = 6.0):
    return tuple(
        contracts.AudioChunk(
            chunk_id=i + 1,
            audio_file=Path(f"chunk_{i + 1:04d}.wav"),
            start=i * width,
            end=(i + 1) * width,
            text=text,
            is_instrumental=not text,
        )
        for i, text in enumerate(texts)
    )


def _beat(
    chunk_id, *, group=1, role="transition", start=None, focus="subject", location="the room",
    pop_object=None,
):
    start = (chunk_id - 1) * 6.0 if start is None else start
    return Beat(
        chunk_id=chunk_id,
        start=start,
        end=start + 6.0,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=group,
        location=location,
        focus=focus,
        pop_object=pop_object,
    )


def _shots(*pairs) -> dict:
    return {"shots": [{"chunk_id": cid, "shot": text} for cid, text in pairs]}


GOOD = "The printer at the end of the aisle erupts in a fireball, papers catching light"


# --------------------------------------------------------------------------- #
# Windowing on beat groups
# --------------------------------------------------------------------------- #


def test_windows_are_beat_groups_not_fixed_chunk_counts():
    """Design section 4: "a window boundary that falls between a contact and
    its consequence produces exactly the defect the three-beat rule exists to
    prevent"."""
    beats = (
        _beat(1, group=1),
        _beat(2, group=1),
        _beat(3, group=1),
        _beat(4, group=2),
        _beat(5, group=2),
    )

    windows = beat_windows(beats)

    assert [[b.chunk_id for b in w] for w in windows] == [[1, 2, 3], [4, 5]]


def test_windows_come_out_in_song_order_whatever_the_group_numbers_are():
    beats = (_beat(1, group=7), _beat(2, group=3), _beat(3, group=7))

    windows = beat_windows(beats)

    assert [w[0].beat_group for w in windows] == [7, 3]
    assert [[b.chunk_id for b in w] for w in windows] == [[1, 3], [2]]


def test_a_group_split_across_the_song_stays_one_window():
    """The guide allows a plant to sit a chunk or two before its contact, so
    a group is not necessarily contiguous -- and it must still be written in
    one call, or the payoff is written by a call that never saw the plant."""
    beats = (_beat(1, group=1), _beat(2, group=2), _beat(3, group=1))

    windows = beat_windows(beats)

    assert [b.chunk_id for b in windows[0]] == [1, 3]


def test_only_the_requested_groups_are_windowed():
    beats = (_beat(1, group=1), _beat(2, group=2), _beat(3, group=3))

    windows = beat_windows(beats, groups={2, 3})

    assert [w[0].beat_group for w in windows] == [2, 3]


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #


def test_a_well_formed_reply_validates():
    window = (_beat(1), _beat(2))
    chunks = _chunks(["", ""])

    shots, _present, issues = validate_prose(
        _shots((1, GOOD), (2, "Glass across the floor in the foreground")),
        window,
        chunks=chunks,
        cast_names=("Dianne",),
        camera={},
    )

    assert set(shots) == {1, 2}
    assert issues == ()


def test_a_chunk_in_the_window_with_no_shot_is_an_error():
    with pytest.raises(ProseValidationError, match="no shot"):
        validate_prose(
            _shots((1, GOOD)),
            (_beat(1), _beat(2)),
            chunks=_chunks(["", ""]),
            cast_names=("Dianne",),
            camera={},
        )


def test_a_chunk_outside_the_window_is_dropped(caplog):
    """The two chunks either side are read-only context; a reply that
    rewrites them would silently overwrite prose another window approved."""
    with caplog.at_level("WARNING"):
        shots, _present, _ = validate_prose(
            _shots((1, GOOD), (9, "not in this window")),
            (_beat(1),),
            chunks=_chunks(["", ""]),
            cast_names=("Dianne",),
            camera={},
        )

    assert set(shots) == {1}
    assert "9" in caplog.text


def test_a_blank_shot_is_an_error():
    with pytest.raises(ProseValidationError):
        validate_prose(
            _shots((1, "   ")),
            (_beat(1),),
            chunks=_chunks([""]),
            cast_names=("Dianne",),
            camera={},
        )


def test_a_reply_with_no_shots_array_is_rejected():
    with pytest.raises(ProseValidationError, match="shots"):
        validate_prose(
            {"lines": []}, (_beat(1),), chunks=_chunks([""]), cast_names=(), camera={}
        )


# --------------------------------------------------------------------------- #
# The prohibitions checked as ERRORS
# --------------------------------------------------------------------------- #


def test_naming_a_cast_member_is_an_error():
    """``prompting.py`` composes the character's name and role into every
    prompt already. A shot line that names her spends its words on something
    the render loop supplies and, worse, competes with it."""
    with pytest.raises(ProseValidationError, match="Dianne"):
        validate_prose(
            _shots((1, "Dianne walks the length of the empty street")),
            (_beat(1),),
            chunks=_chunks([""]),
            cast_names=("Dianne", "Rex"),
            camera={},
        )


def test_quoting_the_chunks_own_lyric_verbatim_is_an_error():
    with pytest.raises(ProseValidationError, match="lyric"):
        validate_prose(
            _shots((1, "Walking through the empty halls tonight, the lights behind her fail")),
            (_beat(1),),
            chunks=_chunks(["walking through the empty halls tonight"]),
            cast_names=("Dianne",),
            camera={},
        )


def test_a_short_overlap_with_the_lyric_is_not_quoting():
    """Three words in common is coincidence; the check has to be about
    verbatim quoting, not about vocabulary the shot and the lyric share."""
    shots, _present, issues = validate_prose(
        _shots((1, "The empty halls stretch away behind a failing strip light")),
        (_beat(1),),
        chunks=_chunks(["walking through the empty halls tonight"]),
        cast_names=("Dianne",),
        camera={},
    )

    assert shots[1]
    assert issues == ()


def test_camera_direction_is_an_error_when_that_chunk_already_has_a_camera_field():
    """Once photography owns ``camera`` (phase 4), a camera phrase in ``shot``
    composes twice -- the exact double-composition ``shot_plan``'s own lint
    catches on the render side."""
    with pytest.raises(ProseValidationError, match="camera"):
        validate_prose(
            _shots((1, "The street empties out, camera tracking backwards ahead of her")),
            (_beat(1),),
            chunks=_chunks([""]),
            cast_names=("Dianne",),
            camera={1: "tracking backwards ahead of her"},
        )


# --------------------------------------------------------------------------- #
# The prohibitions reported as WARNINGS, and why they are not errors
# --------------------------------------------------------------------------- #


def test_camera_language_with_no_camera_field_is_only_a_warning():
    """``docs/shot-writing-guide.md`` is fed to this stage as its system
    prompt and *every worked example in it* ends with a trailing camera
    phrase. Rejecting one while handing the model that document would be
    telling it to disobey its own reference."""
    shots, _present, issues = validate_prose(
        _shots((1, "The street empties out, camera tracking backwards ahead of her")),
        (_beat(1),),
        chunks=_chunks([""]),
        cast_names=("Dianne",),
        camera={},
    )

    assert shots[1]
    assert [i.severity for i in issues] == ["warning"]
    assert "camera" in issues[0].message


def test_saying_the_performer_is_singing_is_only_a_warning():
    """The guide's checklist says not to restate that the character is
    singing; the guide's own worked examples say "still singing to camera",
    and its grammatical-subject rule warns in as many words that writing her
    out of the line costs the lip-sync for that whole chunk. Rejecting the
    phrase would push the model toward the more expensive mistake."""
    shots, _present, issues = validate_prose(
        _shots((1, "The monitors go black as she passes in the foreground, still singing")),
        (_beat(1),),
        chunks=_chunks([""]),
        cast_names=("Dianne",),
        camera={},
    )

    assert shots[1]
    assert [i.severity for i in issues] == ["warning"]


def test_a_clean_line_produces_no_issues_at_all():
    shots, _present, issues = validate_prose(
        _shots((1, GOOD)),
        (_beat(1),),
        chunks=_chunks(["the printer would explode somehow"]),
        cast_names=("Dianne",),
        camera={},
    )

    assert shots[1] == GOOD
    assert issues == ()


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_the_prompt_carries_the_window_its_context_and_the_concept(tmp_path):
    beats = (_beat(1, group=1), _beat(2, group=2), _beat(3, group=2), _beat(4, group=3))
    chunks = _chunks(["first line", "", "second line", "third line"])

    prompt = build_prose_prompt(
        _config(tmp_path), CONCEPT, beats[1:3], beats, chunks, camera={}, notes=None
    )

    assert "beat 2" in prompt and "beat 3" in prompt
    assert CONCEPT["logline"] in prompt
    assert "second line" in prompt  # the window's own lyric
    assert "beat 1" in prompt and "beat 4" in prompt  # read-only context either side
    assert "context" in prompt.lower()


def test_the_prompt_marks_a_consequence_beats_focus(tmp_path):
    beats = (_beat(1, role="consequence", focus="action"),)
    prompt = build_prose_prompt(
        _config(tmp_path), CONCEPT, beats, beats, _chunks([""]), camera={}, notes=None
    )

    assert "consequence" in prompt
    assert "action" in prompt


def test_the_prompt_marks_a_pop_beats_object(tmp_path):
    """Issue #68: `pop_object` has to reach the model the same way
    `beat_role`/`focus`/`camera` already do, via `_beat_line`."""
    beats = (_beat(1, role="contact", pop_object="the printer"),)
    prompt = build_prose_prompt(
        _config(tmp_path), CONCEPT, beats, beats, _chunks([""]), camera={}, notes=None
    )

    assert "POP" in prompt
    assert "the printer" in prompt


def test_a_beat_with_no_pop_object_carries_no_pop_marker(tmp_path):
    beats = (_beat(1, role="contact"),)
    prompt = build_prose_prompt(
        _config(tmp_path), CONCEPT, beats, beats, _chunks([""]), camera={}, notes=None
    )

    assert "POP" not in prompt


def test_notes_reach_the_prompt(tmp_path):
    beats = (_beat(1),)
    prompt = build_prose_prompt(
        _config(tmp_path),
        CONCEPT,
        beats,
        beats,
        _chunks([""]),
        camera={},
        notes="stop putting her in doorways",
    )

    assert "stop putting her in doorways" in prompt


def test_input_hashes_cover_the_beats_so_re_running_beats_makes_prose_stale(tmp_path):
    chunks = _chunks(["", ""])
    beats = (_beat(1), _beat(2))

    before = prose_input_hashes(_config(tmp_path), chunks, CONCEPT, beats)
    after = prose_input_hashes(
        _config(tmp_path), chunks, CONCEPT, (beats[0], _beat(2, group=9))
    )

    assert before != after
    assert set(before) == {"skeleton", "concept", "beats", "shot_writing_guide"}


# --------------------------------------------------------------------------- #
# generate_prose: one call per window, retry with feedback
# --------------------------------------------------------------------------- #


def test_generate_prose_calls_the_model_once_per_beat_group(tmp_path):
    beats = (_beat(1, group=1), _beat(2, group=1), _beat(3, group=2))
    chunks = _chunks(["", "", ""])
    driver = ScriptedDriver([_shots((1, GOOD), (2, GOOD)), _shots((3, GOOD))])

    result = generate_prose(_config(tmp_path), CONCEPT, beats, chunks, driver)

    assert len(driver.calls) == 2
    assert set(result.shots) == {1, 2, 3}
    assert driver.calls[0]["model"] == MODEL_SONNET


def test_generate_prose_threads_lyric_literalness_into_the_system_prompt(tmp_path):
    """Issue #67: the directorial choice has to reach the model, not just
    live in config."""
    from music_video_maker.authoring.prompts import LITERALNESS_BLOCKS

    config = replace(_config(tmp_path), lyric_literalness="thematic")
    driver = ScriptedDriver([_shots((1, GOOD))])

    generate_prose(config, CONCEPT, (_beat(1),), _chunks([""]), driver)

    assert LITERALNESS_BLOCKS["thematic"] in driver.calls[0]["system"]


def test_a_prohibited_line_feeds_its_own_error_back_into_the_next_prompt(tmp_path):
    beats = (_beat(1),)
    chunks = _chunks([""])
    driver = ScriptedDriver(
        [_shots((1, "Dianne walks the length of the street")), _shots((1, GOOD))]
    )

    result = generate_prose(_config(tmp_path), CONCEPT, beats, chunks, driver)

    assert len(driver.calls) == 2
    assert "Dianne" in driver.calls[1]["prompt"]
    assert result.shots[1] == GOOD


def test_generate_prose_gives_up_after_the_bounded_number_of_attempts(tmp_path):
    beats = (_beat(1),)
    bad = _shots((1, "Dianne walks the length of the street"))
    driver = ScriptedDriver([bad, bad, bad])

    with pytest.raises(ProseValidationError):
        generate_prose(_config(tmp_path), CONCEPT, beats, _chunks([""]), driver)

    assert len(driver.calls) == 3


def test_warnings_are_collected_rather_than_retried(tmp_path):
    """Design section 6: warnings get one revision round *later*, at the plan
    level -- never a retry here. Every one of these lints fires on prose a
    human wrote deliberately, and a loop that retries until they are silent
    will happily rewrite a correct shot to please a heuristic."""
    beats = (_beat(1),)
    driver = ScriptedDriver([_shots((1, "The street empties, camera pushing in slowly"))])

    result = generate_prose(_config(tmp_path), CONCEPT, beats, _chunks([""]), driver)

    assert len(driver.calls) == 1
    assert [i.severity for i in result.issues] == ["warning"]


def test_a_driver_error_propagates_and_writes_nothing(tmp_path):
    with pytest.raises(DriverError):
        generate_prose(
            _config(tmp_path), CONCEPT, (_beat(1),), _chunks([""]), ScriptedDriver([None])
        )


def test_only_the_requested_groups_are_generated(tmp_path):
    """"regenerate group 3" is a sentence a human will actually say, and the
    other groups' approved prose must not be touched to satisfy it."""
    beats = (_beat(1, group=1), _beat(2, group=2))
    driver = ScriptedDriver([_shots((2, GOOD))])

    result = generate_prose(
        _config(tmp_path), CONCEPT, beats, _chunks(["", ""]), driver, groups={2}
    )

    assert len(driver.calls) == 1
    assert set(result.shots) == {2}


# --------------------------------------------------------------------------- #
# revise_prose: targeted means targeted
# --------------------------------------------------------------------------- #


def test_a_revision_returns_only_the_objected_chunks(tmp_path):
    """The failure this guards: a retry that quietly rewords thirty approved
    shots destroys the value of the approval that came before it. The model is
    shown the whole beat group so its rewrite stays coherent with the plant
    and the payoff -- but it may only *write* the chunks under objection, and
    anything else in the reply is dropped rather than merged."""
    from music_video_maker.authoring.prose import revise_prose

    beats = (_beat(1, group=1), _beat(2, group=1), _beat(3, group=1))
    chunks = _chunks(["", "", ""])
    shots = {1: GOOD, 2: "the line under objection", 3: GOOD}
    driver = ScriptedDriver(
        [_shots((1, "REWRITTEN"), (2, "A payphone swings loose on its cord"), (3, "REWRITTEN"))]
    )

    result = revise_prose(
        _config(tmp_path),
        CONCEPT,
        beats,
        chunks,
        driver,
        shots=shots,
        objections={2: ["too bleak"]},
    )

    assert set(result.shots) == {2}
    assert result.shots[2] == "A payphone swings loose on its cord"


def test_a_revision_shows_the_model_its_own_line_and_the_objection(tmp_path):
    from music_video_maker.authoring.prose import revise_prose

    driver = ScriptedDriver([_shots((1, GOOD))])

    revise_prose(
        _config(tmp_path),
        CONCEPT,
        (_beat(1),),
        _chunks([""]),
        driver,
        shots={1: "the line under objection"},
        objections={1: ["names a landmark outside the setting"]},
    )

    prompt = driver.calls[0]["prompt"]
    assert "the line under objection" in prompt
    assert "names a landmark outside the setting" in prompt


def test_a_revision_threads_lyric_literalness_into_the_system_prompt(tmp_path):
    from music_video_maker.authoring.prompts import LITERALNESS_BLOCKS
    from music_video_maker.authoring.prose import revise_prose

    config = replace(_config(tmp_path), lyric_literalness="literal")
    driver = ScriptedDriver([_shots((1, GOOD))])

    revise_prose(
        config,
        CONCEPT,
        (_beat(1),),
        _chunks([""]),
        driver,
        shots={1: "the line under objection"},
        objections={1: ["too bleak"]},
    )

    assert LITERALNESS_BLOCKS["literal"] in driver.calls[0]["system"]


def test_a_revision_skips_groups_nothing_was_objected_to(tmp_path):
    from music_video_maker.authoring.prose import revise_prose

    beats = (_beat(1, group=1), _beat(2, group=2))
    driver = ScriptedDriver([_shots((2, GOOD))])

    result = revise_prose(
        _config(tmp_path),
        CONCEPT,
        beats,
        _chunks(["", ""]),
        driver,
        shots={1: GOOD, 2: GOOD},
        objections={2: ["too bleak"]},
    )

    assert len(driver.calls) == 1
    assert set(result.shots) == {2}


def test_a_revision_that_keeps_failing_raises_rather_than_returning_the_old_line(tmp_path):
    from music_video_maker.authoring.prose import revise_prose

    bad = _shots((1, "Dianne walks the length of the street"))
    driver = ScriptedDriver([bad, bad, bad])

    with pytest.raises(ProseValidationError):
        revise_prose(
            _config(tmp_path),
            CONCEPT,
            (_beat(1),),
            _chunks([""]),
            driver,
            shots={1: GOOD},
            objections={1: ["too bleak"]},
        )


def test_the_global_style_reaches_the_prose_prompt(tmp_path):
    """Same reason photography gets it: a shot line can contradict "one
    continuous unbroken take, no cuts" outright, and the render composes that
    constraint into every prompt whether this stage has seen it or not."""
    config = _config(tmp_path)

    prompt = build_prose_prompt(
        config, CONCEPT, (_beat(1),), (_beat(1),), _chunks([""]), camera={}, notes=None
    )

    assert config.global_style in prompt


def test_the_system_prompt_warns_against_turning_the_guide_into_a_template():
    """Measured on a real re-authoring of a whole song: "in the foreground"
    landed in 24 of 36 generated shots where the human author used it in 5.
    The guide's foreground/background rule is real, but applied to every line
    it stops carrying information and starts reading as a template -- which is
    exactly the "passes every lint and is still boring" failure the design
    predicted and no lint can catch."""
    from music_video_maker.authoring.prompts import prose_system_prompt

    system = prose_system_prompt()

    assert "in the foreground" in system
    assert "24 of 36" in system  # the measurement, not a vague exhortation


def test_the_prompt_distinguishes_a_constant_trait_from_a_changing_one():
    """A property that holds across the whole video belongs in the field that
    composes into every chunk -- this project's oldest structural lesson. The
    lead's "smiling constantly" is in ``cast.Dianne.role`` and fires in all 36
    prompts, so a shot line saying "still smiling" spends itself on something
    it is already given.

    The escalation is the opposite case and the prose stage is the only place
    it can live: ``role`` is one string reused in every chunk, so it cannot
    say "her smile widens as the disaster compounds". State the change, never
    the constant.
    """
    from music_video_maker.authoring.prompts import prose_system_prompt

    system = prose_system_prompt()

    assert "standing demeanour" in system
    assert "State the change, never the constant" in system


# --------------------------------------------------------------------------- #
# `present`: binding the pronoun to a cast member (issue #59)
#
# Rule 2 of PROSE_PREAMBLE forbids naming a cast member in the shot line, so
# a second character can only be "him"/"her" there. On a real 36-chunk render
# nine lines said "him", seven of those chunks had nobody bound to it, and H3
# invented a different man each time.
# --------------------------------------------------------------------------- #


def test_present_is_returned_per_chunk():
    reply = {
        "shots": [
            {"chunk_id": 1, "shot": GOOD, "present": ["Jan"]},
            {"chunk_id": 2, "shot": "Glass across the floor"},
        ]
    }

    _shots_out, present, _issues = validate_prose(
        reply,
        (_beat(1), _beat(2)),
        chunks=_chunks(["", ""]),
        cast_names=("Dianne", "Jan"),
        camera={},
    )

    assert present[1] == ("Jan",)
    # A chunk nobody was named for is absent rather than empty: the renderer's
    # default is "she is alone", and an empty list would say the same thing
    # more loudly in every one of 36 blocks.
    assert 2 not in present


def test_an_unknown_present_name_is_an_error_not_a_dropped_entry():
    """A misspelt name stages nobody, which is precisely the silent failure
    the field exists to end -- so it must be loud."""
    reply = {"shots": [{"chunk_id": 1, "shot": GOOD, "present": ["Jann"]}]}

    with pytest.raises(ProseValidationError, match="Jann"):
        validate_prose(
            reply,
            (_beat(1),),
            chunks=_chunks([""]),
            cast_names=("Dianne", "Jan"),
            camera={},
        )


def test_present_that_is_not_a_list_of_names_is_an_error():
    reply = {"shots": [{"chunk_id": 1, "shot": GOOD, "present": "Jan"}]}

    with pytest.raises(ProseValidationError, match="present"):
        validate_prose(
            reply,
            (_beat(1),),
            chunks=_chunks([""]),
            cast_names=("Dianne", "Jan"),
            camera={},
        )


def test_duplicate_present_names_are_collapsed():
    reply = {"shots": [{"chunk_id": 1, "shot": GOOD, "present": ["Jan", "Jan"]}]}

    _shots_out, present, _issues = validate_prose(
        reply,
        (_beat(1),),
        chunks=_chunks([""]),
        cast_names=("Dianne", "Jan"),
        camera={},
    )

    assert present[1] == ("Jan",)


def test_the_preamble_tells_the_model_about_present_and_anaphora():
    """Both rules are measured findings, so they belong in the preamble the
    model actually reads, not only in a validator that rejects afterwards."""
    from music_video_maker.authoring.prompts import PROSE_PREAMBLE

    assert "present" in PROSE_PREAMBLE.lower()
    assert "that same window" in PROSE_PREAMBLE


# --------------------------------------------------------------------------- #
# A sung chunk's action belongs to the singer.
#
# `present` (above) binds a pronoun so H3 stops inventing a stranger -- and it
# works. What it does not do is decide WHOSE shot it is. Measured on
# "Deathless" chunk 7, a chunk Dianne sings whose line gave the action to the
# other character ("...as he stands high above it on the mountain's edge,
# watching"):
#
#   present = ["Jan"]  -> Jan renders correctly, holds frames 0-101 of 175,
#                         and the singer gets the last 41% of the chunk
#   no present         -> the singer INHERITS the watching, turns away, and
#                         0 of 35 sampled frames contain a detectable face
#
# So both settings lose the lip-sync and the field is not the lever: whatever
# action the sentence describes is performed by whoever is on screen. The
# lever is the sentence, which is #60's lesson one level up.
# --------------------------------------------------------------------------- #


def test_the_preamble_says_a_sung_shot_belongs_to_the_singer():
    """A rule the validator cannot enforce -- "is this verb the singer's?" is
    not decidable from the text -- so the preamble is the only place it can
    live."""
    from music_video_maker.authoring.prompts import PROSE_PREAMBLE

    lowered = PROSE_PREAMBLE.lower()
    assert "carries a lyric" in lowered or "sung shot" in lowered
    # The measured numbers, so a future editor cannot mistake it for taste.
    assert "0 of 35" in PROSE_PREAMBLE


# --------------------------------------------------------------------------- #
# An idiom that names a physical object gets rendered as that object
# (issue #75).
#
# H3 has no idiom dictionary -- every noun in a shot line is a candidate for
# the frame. Measured on "Deathless": "as she holds her line above the
# crumbling rock" is climbing idiom for a route or a rope, and H3 rendered an
# actual rope, plus the harness and hardware that go with it, into a video
# whose `role` names no equipment at all. Whether a *lint* should also catch
# this is a separate question (issue #75's own corpus is n=1 -- one measured
# hit, several measured non-hits) -- the preamble rule is the fix that works
# at the stage that writes the sentence, and is shipped regardless.
# --------------------------------------------------------------------------- #


def test_the_preamble_warns_that_every_noun_renders_literally():
    """A rule the validator cannot enforce -- "does this idiom happen to name
    an object" is not mechanically decidable -- so, like the singer/action
    rule above, the preamble is the only place it can live."""
    from music_video_maker.authoring.prompts import PROSE_PREAMBLE

    lowered = PROSE_PREAMBLE.lower()
    assert "rendered literally" in lowered or "idiom" in lowered
    # The measured example, so a future editor cannot mistake it for taste.
    assert "holds her line" in PROSE_PREAMBLE


# --------------------------------------------------------------------------- #
# A plant's prose can state the end state its own consequence delivers
# (issue #85).
#
# "Describe the end state, not the motion" (rule 5) is right for a `contact`
# or a `consequence` and wrong for a `plant`, whose whole job is the BEFORE.
# On "Deathless" `shot_plan_v6.toml.before_g5fix`, beat group 5 put an island
# GONE at 6:26 (the plant), PRESENT at 6:31 (the contact) and GONE again at
# 6:38 (the consequence) -- each line internally consistent, so nothing that
# compares a shot line to itself could ever see the contradiction.
# --------------------------------------------------------------------------- #

PLANT_ISLAND_GONE = (
    "She casts a glance out across the flat, grey sea toward the empty "
    "horizon where the island once stood, no silhouette left to mark it."
)
PLANT_ISLAND_PRESENT = (
    "A low dark island holds firm along the horizon across the flat grey "
    "sea, filling the middle distance in the pale light, as she glances "
    "out toward it."
)
CONSEQUENCE_ISLAND_GONE = (
    "The seabed lies bare and drained, the island fully gone, no needle "
    "glinting anywhere in it."
)


def test_fires_on_the_real_island_plant_and_consequence_pair():
    """The exact defect #85 was filed about: a plant that already states the
    end state its own group's consequence delivers."""
    beats = (
        _beat(59, group=5, role="plant"),
        _beat(60, group=5, role="contact"),
        _beat(61, group=5, role="consequence"),
    )
    shots = {59: PLANT_ISLAND_GONE, 60: "the light sweeps the sea", 61: CONSEQUENCE_ISLAND_GONE}

    issues = plant_end_state_issues(shots, beats)

    assert [i.chunk_id for i in issues] == [59]
    assert issues[0].severity == "warning"
    assert "island" in issues[0].message
    assert "61" in issues[0].message


def test_silent_when_the_plant_states_the_before_state():
    """The restored line -- the island holding firm -- is exactly what a
    plant is supposed to say, and must not be flagged."""
    beats = (
        _beat(59, group=5, role="plant"),
        _beat(60, group=5, role="contact"),
        _beat(61, group=5, role="consequence"),
    )
    shots = {
        59: PLANT_ISLAND_PRESENT,
        60: "the light sweeps the sea",
        61: CONSEQUENCE_ISLAND_GONE,
    }

    assert plant_end_state_issues(shots, beats) == ()


def test_silent_when_the_two_lines_share_no_noun():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    shots = {
        1: "A battered kettle sits cold on the shelf near the door.",
        2: "The lantern gutters out completely, its flame gone for good.",
    }

    assert plant_end_state_issues(shots, beats) == ()


def test_silent_when_the_group_has_no_consequence():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="contact"))
    shots = {1: PLANT_ISLAND_GONE, 2: CONSEQUENCE_ISLAND_GONE}

    assert plant_end_state_issues(shots, beats) == ()


def test_silent_when_the_plant_comes_after_the_consequence():
    """Only a plant whose chunk_id precedes the consequence's counts -- a
    'plant' authored later in the timeline is not this defect."""
    beats = (
        _beat(61, group=5, role="plant"),
        _beat(59, group=5, role="consequence"),
    )
    shots = {61: PLANT_ISLAND_GONE, 59: CONSEQUENCE_ISLAND_GONE}

    assert plant_end_state_issues(shots, beats) == ()


def test_one_issue_per_plant_chunk_even_when_several_nouns_match():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    shots = {
        1: (
            "No needle left where the island once stood, nothing of the "
            "shrine remains either."
        ),
        2: "The island sinks as the needle vanishes and the shrine crumbles into the sea.",
    }

    issues = plant_end_state_issues(shots, beats)

    assert len(issues) == 1
    assert issues[0].chunk_id == 1


def test_pattern_where_it_once_stood():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    shots = {
        1: "The old bridge holds firm where it has always crossed the ravine below.",
        2: "The bridge finally gives way and falls into the ravine.",
    }
    # This one shares "bridge" but has no absence phrasing -- silent.
    assert plant_end_state_issues(shots, beats) == ()

    shots_gone = {
        1: "She looks out toward the far bank, where the bridge once stood.",
        2: "The bridge finally gives way and falls into the ravine.",
    }
    issues = plant_end_state_issues(shots_gone, beats)
    assert len(issues) == 1
    assert "bridge" in issues[0].message


def test_pattern_no_noun_left():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    shots = {
        1: "The valley holds no bridge left, its ruins hidden in the mist below.",
        2: "The bridge finally gives way and falls into the ravine.",
    }

    issues = plant_end_state_issues(shots, beats)

    assert len(issues) == 1
    assert issues[0].chunk_id == 1


def test_pattern_nothing_of_noun_remains():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    shots = {
        1: (
            "Nothing of the tower remains against the skyline, a bare field "
            "stretching where the walls once framed the square."
        ),
        2: "The tower crumbles into rubble as dust rises over the square.",
    }

    issues = plant_end_state_issues(shots, beats)

    assert len(issues) == 1
    assert issues[0].chunk_id == 1


def test_the_returned_severity_is_warning():
    beats = (_beat(1, group=1, role="plant"), _beat(2, group=1, role="consequence"))
    issues = plant_end_state_issues(
        {1: PLANT_ISLAND_GONE, 2: CONSEQUENCE_ISLAND_GONE}, beats
    )

    assert issues
    assert all(issue.severity == "warning" for issue in issues)


# --------------------------------------------------------------------------- #
# pop_distant_staging_issues -- issue #68: a pop beat's own shot line trips
# the render-side distant-staging vocabulary (issue #58, re-scored #76).
# Reuses `shot_plan._distant_staging_match` rather than a second keyword
# list -- the design doc's own build order: a pop-specific keyword lint
# scores zero against the only real corpus, which has no pop beats in it.
# --------------------------------------------------------------------------- #


def test_fires_when_a_pop_beats_shot_stages_the_object_distant():
    beats = (
        _beat(1, group=1, role="plant"),
        _beat(2, group=1, role="contact", pop_object="the printer"),
    )
    shots = {
        1: "A beige printer sits blinking on the office windowsill beside her.",
        2: "The printer hangs small against the tower far behind her, tumbling toward the lens.",
    }

    issues = pop_distant_staging_issues(shots, beats)

    assert [i.chunk_id for i in issues] == [2]
    assert issues[0].severity == "warning"
    assert "printer" in issues[0].message
    assert issues[0].revisable is True


def test_silent_when_a_pop_beats_shot_stages_the_object_near():
    beats = (
        _beat(1, group=1, role="plant"),
        _beat(2, group=1, role="contact", pop_object="the printer"),
    )
    shots = {
        1: "A beige printer sits blinking on the office windowsill beside her.",
        2: "The printer bursts toward the lens, filling the frame as it tumbles past her shoulder.",
    }

    assert pop_distant_staging_issues(shots, beats) == ()


def test_silent_on_a_non_pop_beat_even_when_staged_distant():
    """The same distant-staging text on an ordinary beat is not this
    function's business -- `_lint_distant_staging` already covers every
    beat via `load_shot_plan`; this one is scoped to pop beats only, per its
    own docstring."""
    beats = (_beat(1, group=1, role="transition"),)
    shots = {1: "The printer sits tiny in the distance, far behind on the street."}

    assert pop_distant_staging_issues(shots, beats) == ()


def test_silent_when_the_pop_beat_has_no_shot_line_yet():
    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)

    assert pop_distant_staging_issues({}, beats) == ()


def test_reuses_the_render_sides_own_vocabulary_not_a_copy(monkeypatch):
    """Proof this is the SAME predicate, not a second keyword list: patching
    the render-side function that `pop_distant_staging_issues` imports
    changes this function's answer too."""
    from music_video_maker.authoring import prose as prose_module

    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)
    shots = {1: "an ordinary sentence, staged near, with nothing flagged in it"}

    assert pop_distant_staging_issues(shots, beats) == ()

    monkeypatch.setattr(prose_module, "_distant_staging_match", lambda shot: "patched")
    issues = pop_distant_staging_issues(shots, beats)

    assert len(issues) == 1
    assert "patched" in issues[0].message


def test_the_preamble_tells_a_plant_to_show_the_before_state():
    """Issue #85: rule 5 ("describe the end state, not the motion") is right
    for a contact or a consequence and wrong for a plant, and nothing
    arbitrated that until this rule existed."""
    from music_video_maker.authoring.prompts import PROSE_PREAMBLE

    lowered = PROSE_PREAMBLE.lower()
    assert "plant" in lowered and "before" in lowered
    # The measured numbers, so a future editor cannot mistake it for taste.
    assert "6:26" in PROSE_PREAMBLE
    assert "6:31" in PROSE_PREAMBLE
    assert "6:38" in PROSE_PREAMBLE


def test_the_preamble_tells_the_model_to_compose_a_pop_toward_the_lens():
    """Issue #68: near, large, held long enough to read, and clear of the
    frame edges -- the design doc's own vocabulary for what makes an object
    read as crossing the screen plane rather than clipped by a window
    violation."""
    from music_video_maker.authoring.prompts import PROSE_PREAMBLE

    lowered = PROSE_PREAMBLE.lower()
    assert "pop" in lowered and "lens" in lowered
    assert "edge" in lowered  # the window-violation warning
    assert "#68" in PROSE_PREAMBLE or "68" in PROSE_PREAMBLE


# --------------------------------------------------------------------------- #
# Issue #86: song_facts, composed the same way and in the same position as
# concept.build_concept_prompt (see that module's tests for the direct unit
# tests of prompts.song_facts_block itself).
# --------------------------------------------------------------------------- #


def test_prompt_has_no_established_facts_section_by_default(tmp_path):
    beats = (_beat(1),)
    prompt = build_prose_prompt(
        _config(tmp_path), CONCEPT, beats, beats, _chunks([""]), camera={}, notes=None
    )
    assert "Established facts about this song" not in prompt


def test_prompt_puts_song_facts_first_when_the_config_has_them(tmp_path):
    config = replace(_config(tmp_path), song_facts=("the mill was destroyed in act 5",))
    beats = (_beat(1),)

    prompt = build_prose_prompt(config, CONCEPT, beats, beats, _chunks([""]), camera={}, notes=None)

    assert prompt.startswith("## Established facts about this song")
    assert "the mill was destroyed in act 5" in prompt
    assert prompt.index("Established facts") < prompt.index("The approved concept")


def test_input_hashes_have_no_song_facts_key_by_default(tmp_path):
    chunks = _chunks(["", ""])
    beats = (_beat(1), _beat(2))

    hashes = prose_input_hashes(_config(tmp_path), chunks, CONCEPT, beats)

    assert "song_facts" not in hashes
    assert set(hashes) == {"skeleton", "concept", "beats", "shot_writing_guide"}


def test_input_hashes_include_song_facts_when_set_and_change_with_it(tmp_path):
    chunks = _chunks(["", ""])
    beats = (_beat(1), _beat(2))
    config = replace(_config(tmp_path), song_facts=("fact one",))

    before = prose_input_hashes(config, chunks, CONCEPT, beats)
    after = prose_input_hashes(
        replace(config, song_facts=("fact two",)), chunks, CONCEPT, beats
    )

    assert "song_facts" in before
    assert before["song_facts"] != after["song_facts"]


# --------------------------------------------------------------------------- #
# ProseIssue.revisable -- some findings name a defect prose cannot fix (a
# world-state continuity error belongs to the beat sheet, per `mvm-author
# beats --notes`), and #87 is the standing evidence that a revision round
# will happily rewrite approved prose to satisfy anything it is handed. This
# field is the switch; nothing in this module sets it False yet.
# --------------------------------------------------------------------------- #


def test_prose_issue_revisable_defaults_to_true():
    issue = ProseIssue(chunk_id=1, severity="warning", message="a camera phrase in the line")

    assert issue.revisable is True


def test_a_prose_issue_can_be_marked_not_revisable():
    issue = ProseIssue(
        chunk_id=1, severity="warning", message="a world-state continuity error", revisable=False
    )

    assert issue.revisable is False


# --------------------------------------------------------------------------- #
# pop_object_named_in_shot_issues -- issue #68 step 3, the structural half.
# `pop_object` is composed into NO prompt (it rides in the plan's `# beat:`
# comment like `act`), so a pop beat whose shot line never names its object
# is marked for something the render is never told about. No vocabulary, so
# nothing to be scored wrong -- see docs/pop-beat-corpus.md for what n=3 can
# and cannot claim.
# --------------------------------------------------------------------------- #


def test_fires_when_a_pop_beats_shot_line_never_names_its_object():
    beats = (
        _beat(1, group=1, role="plant"),
        _beat(2, group=1, role="contact", pop_object="the printer"),
    )
    shots = {
        1: "A beige printer sits blinking on the office windowsill beside her.",
        2: "Paper bursts toward the lens in a torn white sheet, filling the frame.",
    }

    issues = pop_object_named_in_shot_issues(shots, beats)

    assert [i.chunk_id for i in issues] == [2]
    assert issues[0].severity == "warning"
    assert "printer" in issues[0].message
    assert issues[0].revisable is True


def test_silent_when_the_shot_line_names_the_pop_object():
    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)
    shots = {1: "The printer tumbles at the lens, filling the frame as it comes."}

    assert pop_object_named_in_shot_issues(shots, beats) == ()


def test_a_plural_in_the_line_still_matches_a_singular_pop_object():
    """Same `stageable_noun_stems` normalisation `lint_shots_against_lyrics`
    uses -- never a second copy of it, which would disagree within a month."""
    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)
    shots = {1: "Two printers tumble at the lens, filling the frame as they come."}

    assert pop_object_named_in_shot_issues(shots, beats) == ()


def test_one_word_of_a_multi_word_pop_object_is_enough():
    """Fires only when NONE of the object's stems appears. "mushroom cap"
    staged as "the mushroom" is authored, not a defect, and a rule demanding
    every word would fire on it."""
    beats = (_beat(1, group=1, role="contact", pop_object="the mushroom cap"),)
    shots = {1: "The mushroom swells at the lens, its underside filling the near frame."}

    assert pop_object_named_in_shot_issues(shots, beats) == ()


def test_silent_on_a_beat_with_no_pop_object_at_all():
    beats = (_beat(1, group=1, role="transition"),)
    shots = {1: "She walks the ridge line with nothing in her hands."}

    assert pop_object_named_in_shot_issues(shots, beats) == ()


def test_silent_when_the_pop_beat_has_no_shot_line_written_yet():
    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)

    assert pop_object_named_in_shot_issues({}, beats) == ()


def test_an_unmatchable_pop_object_is_unexamined_not_passed(caplog):
    """#93: a zero from a detector is not evidence of absence unless
    something asked the second question. "the axe" normalises to no stems at
    all (every word is a stopword or under the lint's minimum length), so the
    beat is UNEXAMINED and the log says so -- it does not silently pass."""
    beats = (_beat(1, group=1, role="contact", pop_object="the axe"),)
    shots = {1: "Something swings at the lens, filling the near frame."}

    with caplog.at_level(logging.INFO, logger="music_video_maker.authoring.prose"):
        assert pop_object_named_in_shot_issues(shots, beats) == ()

    assert "UNEXAMINED" in caplog.text


def test_the_three_rendered_pop_lines_do_not_fire():
    """The corpus, such as it is (docs/pop-beat-corpus.md): the three
    hand-written pop lines that rendered as intended on 2026-09-20 -- chunk
    45 the needle, 46 the ember, 66 the mushroom cap.

    The lines themselves live in a run directory that is not part of this
    repo, so these are reconstructions from the phrases issue #68's own
    comment publishes, NOT the rendered text. They can prove one thing and
    only one: a line that names its object does not fire. 0 of 3, at n=3."""
    beats = (
        _beat(45, group=1, role="contact", pop_object="the needle"),
        _beat(46, group=1, role="contact", pop_object="the ember"),
        _beat(66, group=2, role="contact", pop_object="the mushroom cap"),
    )
    shots = {
        45: "The needle comes at the lens, held large and steady in the near frame.",
        46: "An ember blooms out of the black straight toward the lens, filling the centre.",
        66: "The mushroom cap fills the foreground, swelling toward the lens.",
    }

    assert pop_object_named_in_shot_issues(shots, beats) == ()


def test_the_two_pop_checks_are_independent():
    """A line can name its object and still stage it distant, and it can
    stage something near and never name the object: the two #68 checks
    answer different questions and neither subsumes the other."""
    beats = (_beat(1, group=1, role="contact", pop_object="the printer"),)
    distant_but_named = {1: "The printer sits far behind her on the street."}
    near_but_unnamed = {1: "A white sheet of paper tumbles at the lens, filling the frame."}

    assert len(pop_distant_staging_issues(distant_but_named, beats)) == 1
    assert pop_object_named_in_shot_issues(distant_but_named, beats) == ()
    assert pop_distant_staging_issues(near_but_unnamed, beats) == ()
    assert len(pop_object_named_in_shot_issues(near_but_unnamed, beats)) == 1
