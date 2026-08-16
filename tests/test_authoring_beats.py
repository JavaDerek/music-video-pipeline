"""Tests for Stage 2 -- beats (issue #54 design section 4).

No real model call anywhere in this file: :class:`ScriptedDriver` stands in
for the Claude CLI throughout, and the re-slice that a ``length_seconds``
triggers is an injected callable rather than real audio.

The interesting behaviour is the structural half. ``beat_role`` and
``beat_group`` exist so three of ``docs/shot-writing-guide.md``'s checklist
items become machine-checkable *before* a word of prose is written -- every
consequence needs an earlier plant and contact in its own group, every
consequence needs ``focus = "action"``, and a one-member consequence group is
a cause and its effect compressed into a single shot, which is the defect the
whole guide exists to fix.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from music_video_maker import contracts
from music_video_maker.authoring.beats import (
    BEAT_ROLES,
    Beat,
    BeatsValidationError,
    beat_length_requests,
    beats_input_hashes,
    build_beats_prompt,
    check_act_structure,
    generate_beats,
    validate_beats,
)
from music_video_maker.authoring.driver import MODEL_OPUS, DriverError, ScriptedDriver
from music_video_maker.authoring.prompts import SHOT_WRITING_GUIDE_DOC, beats_system_prompt
from music_video_maker.config import RunConfig
from tests.harness.factories import make_cast_dict, make_raw_stablets_result, write_silent_wav

CONCEPT = {
    "logline": "A drummer walks through gathering weather as a storm builds and clears.",
    "setting": "a coastal town, contemporary",
    "tone": "elegiac, quiet",
    "motifs": ["gathering clouds", "an empty chair"],
    "avoid": ["literal lightning strikes"],
}


class _FakeAlignModel:
    def align(self, audio: str, text: str, **kwargs) -> SimpleNamespace:
        return make_raw_stablets_result()


def _config(tmp_path: Path, *, lyrics_text: str = "") -> RunConfig:
    master_audio = write_silent_wav(tmp_path / "audio" / "master.wav", seconds=20.0)
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text(lyrics_text, encoding="utf-8")
    return RunConfig(
        master_audio=master_audio,
        lyrics_file=lyrics_file,
        global_style="Refestramus progressive rock music video",
        narrative_concept="placeholder",
        cast=make_cast_dict(),
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "output" / "chunks",
        final_video_dir=tmp_path / "output" / "final",
        hardware=contracts.HardwareProfile(name="RTX 4090", vram_gb=24.0),
    )



def _chunk(chunk_id, text, *, instrumental=False, characters=()):
    """One chunk, with the singer attached -- the field the skeleton table
    now has to surface."""
    return contracts.AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id:04d}.wav"),
        start=float(chunk_id) * 6.0,
        end=float(chunk_id) * 6.0 + 6.0,
        text=text,
        is_instrumental=instrumental,
        characters=characters,
    )


def _chunks(n: int = 3, width: float = 6.0):
    return tuple(
        contracts.AudioChunk(
            chunk_id=i + 1,
            audio_file=Path(f"chunk_{i + 1:04d}.wav"),
            start=i * width,
            end=(i + 1) * width,
            text=f"lyric line {i + 1}",
        )
        for i in range(n)
    )


def _reply(*entries: dict) -> dict:
    return {"beats": list(entries)}


def _entry(
    chunk_id, *, role="transition", group=1, focus="subject", location="the room",
    act="the story", **extra,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "beat": f"beat {chunk_id}",
        "beat_role": role,
        "beat_group": group,
        "focus": focus,
        "location": location,
        "act": act,
        **extra,
    }


def _three_beat_gag() -> dict:
    return _reply(
        _entry(1, role="plant", group=1),
        _entry(2, role="contact", group=1),
        _entry(3, role="consequence", group=1, focus="action"),
    )


# --------------------------------------------------------------------------- #
# Shape validation
# --------------------------------------------------------------------------- #


def test_a_well_formed_beat_sheet_validates():
    beats = validate_beats(_three_beat_gag(), _chunks())

    assert [b.chunk_id for b in beats] == [1, 2, 3]
    assert [b.beat_role for b in beats] == ["plant", "contact", "consequence"]


def test_start_and_end_are_re_emitted_from_the_skeleton_never_from_the_model():
    """Design section 4: "we re-emit ``start`` from the skeleton every time".
    An invented anchor cannot reach disk, which is what makes
    ``ShotPlanDriftError`` structurally unreachable for a generated plan
    rather than merely unlikely."""
    reply = _three_beat_gag()
    for entry in reply["beats"]:
        entry["start"] = 999.0
        entry["end"] = 1000.0

    beats = validate_beats(reply, _chunks())

    assert [(b.start, b.end) for b in beats] == [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0)]


def test_an_invented_chunk_id_is_dropped(caplog):
    reply = _three_beat_gag()
    reply["beats"].append(_entry(99))

    with caplog.at_level("WARNING"):
        beats = validate_beats(reply, _chunks())

    assert [b.chunk_id for b in beats] == [1, 2, 3]
    assert "99" in caplog.text


def test_a_chunk_with_no_beat_is_an_error_not_a_blank():
    """Sent back to the model rather than left to fall through to the global
    ``narrative_concept``, which renders as something deliberate-looking that
    nobody authored."""
    with pytest.raises(BeatsValidationError, match="no beat"):
        validate_beats(_reply(_entry(1), _entry(2)), _chunks(3))


def test_a_duplicate_chunk_id_is_an_error():
    with pytest.raises(BeatsValidationError, match="more than once"):
        validate_beats(_reply(_entry(1), _entry(1), _entry(2), _entry(3)), _chunks())


def test_a_reply_that_is_not_an_object_is_rejected():
    with pytest.raises(BeatsValidationError):
        validate_beats(["not", "a", "dict"], _chunks())


def test_a_missing_beats_array_is_rejected():
    with pytest.raises(BeatsValidationError, match="beats"):
        validate_beats({"shots": []}, _chunks())


@pytest.mark.parametrize(
    "bad",
    [
        {"beat": ""},
        {"beat_role": "denouement"},
        {"beat_group": "three"},
        {"focus": "the performer"},
        {"length_seconds": 0},
        {"length_seconds": "nine"},
        {"location": ""},
        {"location": None},
        {"location": 5},
        {"act": ""},
        {"act": None},
        {"act": 5},
    ],
)
def test_malformed_fields_are_rejected(bad):
    entry = _entry(1)
    entry.update(bad)
    with pytest.raises(BeatsValidationError):
        validate_beats(_reply(entry, _entry(2), _entry(3)), _chunks())


# --------------------------------------------------------------------------- #
# `location` -- issue #78's closed vocabulary. `beat_role` is checked
# against a fixed enum; `location` is checked against the concept's own
# `locations` list, which is why it takes a separate `locations=` argument
# rather than living in BEAT_ROLES-style module constant.
# --------------------------------------------------------------------------- #

_APPROVED_LOCATIONS = ("the valley floor", "the switchback", "the mill", "the watch-post")


def test_a_location_outside_the_approved_list_is_rejected():
    reply = _three_beat_gag()
    reply["beats"][0]["location"] = "a shopping mall"

    with pytest.raises(BeatsValidationError, match="approved locations"):
        validate_beats(reply, _chunks(), _APPROVED_LOCATIONS)


def test_a_location_on_the_approved_list_is_accepted():
    reply = _three_beat_gag()
    for entry in reply["beats"]:
        entry["location"] = "the mill"

    beats = validate_beats(reply, _chunks(), _APPROVED_LOCATIONS)

    assert [b.location for b in beats] == ["the mill"] * 3


def test_location_matching_is_case_and_whitespace_insensitive_and_canonicalizes():
    """A model that varies capitalization across beats must not fragment one
    place into two in the written plan -- every match resolves to the
    concept's own exact spelling."""
    reply = _three_beat_gag()
    reply["beats"][0]["location"] = "  The Mill  "
    reply["beats"][1]["location"] = "the mill"
    reply["beats"][2]["location"] = "THE MILL"

    beats = validate_beats(reply, _chunks(), _APPROVED_LOCATIONS)

    assert [b.location for b in beats] == ["the mill"] * 3


def test_an_empty_locations_vocabulary_accepts_anything_non_empty():
    """No vocabulary supplied (a pre-#78 concept, or a caller not using the
    field) means no closed-set check at all -- only "is this a non-empty
    string", same as every other required field's presence check."""
    reply = _three_beat_gag()
    for entry in reply["beats"]:
        entry["location"] = "anywhere at all"

    beats = validate_beats(reply, _chunks(), ())

    assert [b.location for b in beats] == ["anywhere at all"] * 3


def test_generate_beats_threads_the_concepts_locations_into_validation(tmp_path):
    """The concept dict IS the source of the vocabulary -- `generate_beats`
    must derive it from `concept["locations"]` without a caller having to
    pass it separately."""
    concept_with_locations = {**CONCEPT, "locations": list(_APPROVED_LOCATIONS)}
    bad = _three_beat_gag()
    bad["beats"][0]["location"] = "somewhere not on the list"
    good = _three_beat_gag()
    for entry in good["beats"]:
        entry["location"] = "the mill"
    driver = ScriptedDriver([bad, good])

    result = generate_beats(_config(tmp_path), _chunks(), concept_with_locations, driver)

    assert len(driver.calls) == 2
    assert "approved locations" in driver.calls[1]["prompt"]
    assert len(result.beats) == 3


# --------------------------------------------------------------------------- #
# `act` -- issue #84's closed, ORDERED vocabulary. Membership works exactly
# like `location` (issue #78); the arc structure itself (order, contiguity,
# payoff) is `check_act_structure`'s own section below. These membership
# tests deliberately use a SINGLE-act vocabulary so `check_act_structure`'s
# coverage/payoff checks (exercised on their own, below) cannot fail a sheet
# whose whole point is testing membership resolution in isolation.
# --------------------------------------------------------------------------- #

_APPROVED_ACTS = ("situation", "complication", "resolution")
_ONE_ACT = ("the whole song",)


def test_an_act_outside_the_approved_list_is_rejected():
    reply = _three_beat_gag()
    reply["beats"][0]["act"] = "an act nobody declared"

    with pytest.raises(BeatsValidationError, match="approved acts"):
        validate_beats(reply, _chunks(), (), _APPROVED_ACTS)


def test_an_act_on_the_approved_list_is_accepted():
    reply = _three_beat_gag()
    for entry in reply["beats"]:
        entry["act"] = "the whole song"

    beats = validate_beats(reply, _chunks(), (), _ONE_ACT)

    assert [b.act for b in beats] == ["the whole song"] * 3


def test_act_matching_is_case_and_whitespace_insensitive_and_canonicalizes():
    reply = _three_beat_gag()
    reply["beats"][0]["act"] = "  The Whole Song  "
    reply["beats"][1]["act"] = "the whole song"
    reply["beats"][2]["act"] = "THE WHOLE SONG"

    beats = validate_beats(reply, _chunks(), (), _ONE_ACT)

    assert [b.act for b in beats] == ["the whole song"] * 3


def test_an_empty_acts_vocabulary_accepts_anything_non_empty():
    """No vocabulary supplied (a pre-#84 concept, or a caller not using the
    field) means no closed-set check and no arc-structure check at all --
    only "is this a non-empty string"."""
    reply = _three_beat_gag()
    for entry in reply["beats"]:
        entry["act"] = "whatever, no vocabulary was given"

    beats = validate_beats(reply, _chunks(), (), ())

    assert [b.act for b in beats] == ["whatever, no vocabulary was given"] * 3


def test_generate_beats_threads_the_concepts_acts_into_validation(tmp_path):
    concept_with_acts = {**CONCEPT, "acts": [{"name": n, "function": "x"} for n in _ONE_ACT]}
    bad = _three_beat_gag()
    bad["beats"][0]["act"] = "not on the list"
    good = _three_beat_gag()
    for entry in good["beats"]:
        entry["act"] = "the whole song"
    driver = ScriptedDriver([bad, good])

    result = generate_beats(_config(tmp_path), _chunks(), concept_with_acts, driver)

    assert len(driver.calls) == 2
    assert "approved acts" in driver.calls[1]["prompt"]
    assert len(result.beats) == 3


def test_a_concept_with_no_acts_logs_a_warning_but_still_generates(tmp_path, caplog):
    """Backward compatibility, deliberately chosen: a pre-#84 persisted
    concept degrades gracefully (no arc to check) rather than hard-failing,
    the same treatment `locations` already gets -- but it is reported, not
    silent."""
    driver = ScriptedDriver([_three_beat_gag()])
    with caplog.at_level("WARNING"):
        result = generate_beats(_config(tmp_path), _chunks(), CONCEPT, driver)

    assert len(result.beats) == 3
    assert "no `acts` structure" in caplog.text


# --------------------------------------------------------------------------- #
# check_act_structure -- issue #84's four mechanical arc checks. Each is
# independently testable and independently failing, and a sheet violating
# more than one must report all of them in a single message (one retry round
# costs a whole model call).
# --------------------------------------------------------------------------- #


def _act_beat(chunk_id, act, *, role="transition", group=1, focus="subject", start=None):
    return Beat(
        chunk_id=chunk_id,
        start=start if start is not None else float(chunk_id) * 6.0,
        end=(start if start is not None else float(chunk_id) * 6.0) + 6.0,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=group,
        location="the room",
        act=act,
        focus=focus,
    )


_THREE_ACTS = ("situation", "complication", "resolution")


def test_an_empty_acts_vocabulary_skips_every_check():
    beats = (_act_beat(1, "anything"), _act_beat(2, "anything else"))
    problems: list[str] = []

    check_act_structure(beats, (), problems)

    assert problems == []


def test_a_valid_arc_passes_every_check():
    beats = (
        _act_beat(1, "situation", role="plant", group=1),
        _act_beat(2, "complication", role="contact", group=1),
        _act_beat(3, "resolution", role="consequence", group=1, focus="action"),
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert problems == []


def test_an_act_with_no_beat_at_all_is_reported():
    beats = (
        _act_beat(1, "situation", role="plant", group=1),
        _act_beat(2, "resolution", role="consequence", group=1, focus="action"),
        # "complication" never appears
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert any("complication" in p and "no beat" in p for p in problems)


def test_acts_visited_out_of_the_concepts_stated_order_are_reported():
    """Order can fail even when every act's own beats stay contiguous: here
    the concept declares situation/complication/resolution, but the beats
    visit resolution, then situation, then complication -- three clean
    blocks, wrong sequence."""
    beats = (
        _act_beat(1, "resolution", start=0.0),
        _act_beat(2, "situation", start=6.0),
        _act_beat(3, "complication", start=12.0),
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert any("does not follow the concept's stated order" in p for p in problems)
    # Contiguity is NOT violated by this sheet -- each act's beats still form
    # one unbroken run -- so this must be the only failure of its kind.
    assert not any("reappear" in p for p in problems)


def test_an_act_that_reappears_after_a_later_one_started_is_reported():
    """Contiguity can fail even when the overall first-appearance order is
    correct: situation is seen first, complication second (correct order),
    but situation then comes BACK after complication has already started."""
    beats = (
        _act_beat(1, "situation", start=0.0),
        _act_beat(2, "complication", start=6.0),
        _act_beat(3, "situation", start=12.0),
        _act_beat(4, "resolution", start=18.0),
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert any("reappear" in p and "situation" in p for p in problems)
    # Order of FIRST appearances (situation, complication, resolution) is
    # still correct, so this must be the only failure of its kind.
    assert not any("does not follow the concept's stated order" in p for p in problems)


def test_a_final_act_with_no_payoff_of_an_earlier_plant_is_reported():
    """The only real test of an arc, and the issue says so: a sequence of
    events with no earlier plant paid off at the end is not a story."""
    beats = (
        _act_beat(1, "situation", role="transition", start=0.0),
        _act_beat(2, "complication", role="transition", start=6.0),
        _act_beat(3, "resolution", role="transition", start=12.0),
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert any("no consequence beat" in p for p in problems)


def test_a_final_act_consequence_whose_plant_is_in_the_same_act_does_not_count():
    """The plant has to be in an EARLIER act -- a plant and its consequence
    both inside the final act is not an arc, it is one more local gag."""
    beats = (
        _act_beat(1, "situation", role="transition", start=0.0),
        _act_beat(2, "complication", role="transition", start=6.0),
        _act_beat(3, "resolution", role="plant", group=5, start=12.0),
        _act_beat(4, "resolution", role="contact", group=5, start=18.0),
        _act_beat(5, "resolution", role="consequence", group=5, focus="action", start=24.0),
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert any("no consequence beat" in p for p in problems)
    assert not any("no beat" in p for p in problems)  # every act IS used; isolate the payoff fault


def test_a_single_act_skips_the_payoff_check():
    """With only one act there is no EARLIER act to plant from, so the
    payoff check would be unsatisfiable by construction -- deliberately
    skipped rather than an always-failing trap for a legitimately single-act
    song."""
    beats = (_act_beat(1, "the whole song", role="transition"),)
    problems: list[str] = []

    check_act_structure(beats, ("the whole song",), problems)

    assert problems == []


def test_every_act_problem_is_reported_at_once():
    """One retry round costs a whole model call: a sheet with several arc
    faults must come back with several complaints, not the first one."""
    beats = (
        _act_beat(1, "resolution", role="transition", start=0.0),
        _act_beat(2, "situation", role="transition", start=6.0),
        # "complication" never appears, order is wrong, and there is no
        # payoff -- three distinct faults in one small sheet.
    )
    problems: list[str] = []

    check_act_structure(beats, _THREE_ACTS, problems)

    assert len(problems) >= 3


def test_every_named_beat_role_is_accepted():
    chunks = _chunks(len(BEAT_ROLES))
    reply = _reply(
        *(
            _entry(i + 1, role=role, focus="action" if role == "consequence" else "subject")
            for i, role in enumerate(BEAT_ROLES)
        )
    )
    # The consequence in this sheet needs a plant and a contact before it in
    # its own group; BEAT_ROLES happens to list them in that order.
    beats = validate_beats(reply, chunks)

    assert [b.beat_role for b in beats] == list(BEAT_ROLES)


# --------------------------------------------------------------------------- #
# Structural checks: the reason this stage is worth its own call
# --------------------------------------------------------------------------- #


def test_a_consequence_without_focus_action_is_rejected():
    reply = _three_beat_gag()
    reply["beats"][2]["focus"] = "subject"

    with pytest.raises(BeatsValidationError, match='focus = "action"'):
        validate_beats(reply, _chunks())


def test_a_consequence_with_no_earlier_plant_in_its_group_is_rejected():
    reply = _reply(
        _entry(1, role="transition", group=1),
        _entry(2, role="contact", group=1),
        _entry(3, role="consequence", group=1, focus="action"),
    )

    with pytest.raises(BeatsValidationError, match="plant"):
        validate_beats(reply, _chunks())


def test_a_consequence_with_no_earlier_contact_in_its_group_is_rejected():
    reply = _reply(
        _entry(1, role="plant", group=1),
        _entry(2, role="transition", group=1),
        _entry(3, role="consequence", group=1, focus="action"),
    )

    with pytest.raises(BeatsValidationError, match="contact"):
        validate_beats(reply, _chunks())


def test_a_plant_and_contact_in_a_different_group_do_not_count():
    """Groups are what make "earlier in the same sequence" mean anything. A
    plant belonging to a different gag is not this consequence's plant, and
    counting it would let the checker bless exactly the compression it exists
    to catch."""
    reply = _reply(
        _entry(1, role="plant", group=1),
        _entry(2, role="contact", group=1),
        _entry(3, role="consequence", group=2, focus="action"),
    )

    with pytest.raises(BeatsValidationError):
        validate_beats(reply, _chunks())


def test_a_plant_after_its_consequence_does_not_count():
    reply = _reply(
        _entry(1, role="contact", group=1),
        _entry(2, role="consequence", group=1, focus="action"),
        _entry(3, role="plant", group=1),
    )

    with pytest.raises(BeatsValidationError):
        validate_beats(reply, _chunks())


def test_a_one_member_consequence_group_is_rejected_by_name():
    """The design names this shape separately from the plant/contact rule
    because it is the *original* defect: one shot asked to carry both a cause
    and its effect."""
    reply = _reply(
        _entry(1, role="plant", group=1),
        _entry(2, role="contact", group=1),
        _entry(3, role="consequence", group=9, focus="action"),
    )

    with pytest.raises(BeatsValidationError, match="on its own"):
        validate_beats(reply, _chunks())


def test_every_structural_problem_is_reported_at_once():
    """One retry round costs a whole model call, so a sheet with three faults
    must come back with three complaints, not the first one."""
    reply = _reply(
        _entry(1, role="consequence", group=1),
        _entry(2, role="consequence", group=2),
        _entry(3, role="transition", group=3),
    )

    with pytest.raises(BeatsValidationError) as excinfo:
        validate_beats(reply, _chunks())

    assert str(excinfo.value).count("chunk_id=") >= 2


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_the_system_prompt_carries_the_shot_writing_guide_from_disk():
    """Design section 9: the docs *are* the system prompt. A copy pasted into
    ``prompts.py`` would be wrong within two commits of the next render."""
    system = beats_system_prompt()

    assert "The three-beat rule" in system
    assert SHOT_WRITING_GUIDE_DOC.read_text(encoding="utf-8") in system


def test_the_user_prompt_carries_the_concept_and_the_skeleton(tmp_path):
    config = _config(tmp_path)
    chunks = _chunks()

    prompt = build_beats_prompt(config, chunks, CONCEPT)

    assert CONCEPT["logline"] in prompt
    assert CONCEPT["tone"] in prompt
    assert "gathering clouds" in prompt
    assert "literal lightning strikes" in prompt
    assert "lyric line 2" in prompt
    assert "12.000" in prompt  # chunk 3's start, straight from the skeleton


def test_notes_reach_the_prompt(tmp_path):
    prompt = build_beats_prompt(_config(tmp_path), _chunks(), CONCEPT, notes="stop using doorways")

    assert "stop using doorways" in prompt


def test_the_concepts_locations_reach_the_prompt(tmp_path):
    """The model can only assign a `location` it has been shown -- the
    approved list has to be composed into the prompt, not just checked
    after the fact."""
    concept_with_locations = {**CONCEPT, "locations": ["the pier", "the boardwalk"]}

    prompt = build_beats_prompt(_config(tmp_path), _chunks(), concept_with_locations)

    assert "the pier" in prompt
    assert "the boardwalk" in prompt


def test_input_hashes_cover_the_concept_so_editing_it_makes_beats_stale(tmp_path):
    config = _config(tmp_path)
    chunks = _chunks()

    before = beats_input_hashes(config, chunks, CONCEPT)
    after = beats_input_hashes(config, chunks, {**CONCEPT, "tone": "comic"})

    assert before != after
    assert set(before) == {"skeleton", "concept", "shot_writing_guide"}


# --------------------------------------------------------------------------- #
# generate_beats: the retry-with-feedback loop
# --------------------------------------------------------------------------- #


def test_generate_beats_returns_a_validated_sheet(tmp_path):
    driver = ScriptedDriver([_three_beat_gag()])

    result = generate_beats(_config(tmp_path), _chunks(), CONCEPT, driver)

    assert [b.chunk_id for b in result.beats] == [1, 2, 3]
    assert driver.calls[0]["model"] == MODEL_OPUS


def test_a_structural_failure_feeds_its_own_error_back_into_the_next_prompt(tmp_path):
    bad = _three_beat_gag()
    bad["beats"][2]["focus"] = "subject"
    driver = ScriptedDriver([bad, _three_beat_gag()])

    result = generate_beats(_config(tmp_path), _chunks(), CONCEPT, driver)

    assert len(driver.calls) == 2
    assert "failed validation" in driver.calls[1]["prompt"]
    assert 'focus = "action"' in driver.calls[1]["prompt"]
    assert len(result.beats) == 3


def test_generate_beats_gives_up_after_the_bounded_number_of_attempts(tmp_path):
    bad = _three_beat_gag()
    bad["beats"][2]["focus"] = "subject"
    driver = ScriptedDriver([bad, bad, bad])

    with pytest.raises(BeatsValidationError):
        generate_beats(_config(tmp_path), _chunks(), CONCEPT, driver)

    assert len(driver.calls) == 3


def test_a_driver_error_propagates_and_writes_nothing(tmp_path):
    driver = ScriptedDriver([None])

    with pytest.raises(DriverError):
        generate_beats(_config(tmp_path), _chunks(), CONCEPT, driver)


# --------------------------------------------------------------------------- #
# Length requests
# --------------------------------------------------------------------------- #


def test_beats_with_no_lengths_yield_no_requests():
    beats = validate_beats(_three_beat_gag(), _chunks())

    assert beat_length_requests(beats) == ()


def test_a_length_becomes_an_anchored_request_in_chunk_order():
    reply = _three_beat_gag()
    reply["beats"][2]["length_seconds"] = 12.0
    reply["beats"][0]["length_seconds"] = 9.0
    beats = validate_beats(reply, _chunks())

    requests = beat_length_requests(beats)

    assert [(r.start, r.length_seconds, r.source_chunk_id) for r in requests] == [
        (0.0, 9.0, 1),
        (12.0, 12.0, 3),
    ]


def test_a_beat_is_serialisable_and_round_trips():
    """``.authoring/beats.json`` is what phase 3 reads; a field that does not
    survive the round trip is a field the prose stage silently never sees."""
    beats = validate_beats(_three_beat_gag(), _chunks())

    assert tuple(Beat.from_dict(b.to_dict()) for b in beats) == beats


# --------------------------------------------------------------------------- #
# `subject` -- issue #82: whose shot an INSTRUMENTAL beat is. Emitted here
# rather than inferred from pronouns later, per the issue's own instruction:
# "the beats stage already knows whose beat it is". Legal only on a chunk
# nobody sings; the stage has `chunks` in scope, so that is checkable here.
# --------------------------------------------------------------------------- #

_CAST_NAMES = ("Dianne", "Jan")


def _mixed_chunks():
    """One instrumental chunk (1) and two voiced chunks (2, 3), all in the
    same beat_group=1/transition shape so no consequence plumbing is needed
    to isolate `subject` behaviour."""
    return (
        _chunk(1, "", instrumental=True),
        _chunk(2, "lyric line two", characters=("Dianne",)),
        _chunk(3, "lyric line three", characters=("Jan",)),
    )


def test_a_subject_on_an_instrumental_chunk_is_accepted():
    reply = _reply(
        _entry(1, subject="Jan"),
        _entry(2),
        _entry(3),
    )

    beats = validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)

    assert next(b for b in beats if b.chunk_id == 1).subject == "Jan"
    assert next(b for b in beats if b.chunk_id == 2).subject is None


def test_subject_is_absent_by_default():
    beats = validate_beats(_three_beat_gag(), _chunks())

    assert all(b.subject is None for b in beats)


@pytest.mark.parametrize("bad", [123, [], {}, True])
def test_a_non_string_subject_is_rejected(bad):
    reply = _reply(_entry(1, subject=bad), _entry(2), _entry(3))

    with pytest.raises(BeatsValidationError):
        validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)


def test_an_empty_string_subject_is_rejected():
    reply = _reply(_entry(1, subject="   "), _entry(2), _entry(3))

    with pytest.raises(BeatsValidationError):
        validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)


def test_a_subject_not_in_the_known_cast_is_rejected():
    reply = _reply(_entry(1, subject="Marcus"), _entry(2), _entry(3))

    with pytest.raises(BeatsValidationError, match="cast"):
        validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)


def test_an_empty_cast_names_vocabulary_accepts_any_non_empty_subject():
    """Same "no vocabulary supplied" degradation `locations`/`acts` already
    get -- a caller that has not wired the cast through must not be blocked
    from parsing at all."""
    reply = _reply(_entry(1, subject="Anybody"), _entry(2), _entry(3))

    beats = validate_beats(reply, _mixed_chunks(), cast_names=())

    assert next(b for b in beats if b.chunk_id == 1).subject == "Anybody"


def test_a_subject_on_a_voiced_chunk_is_rejected():
    """The beats stage has `chunks` in scope, so this is checkable here --
    "a lint that guesses loses to a stage that knows"."""
    reply = _reply(_entry(1), _entry(2, subject="Jan"), _entry(3))

    with pytest.raises(BeatsValidationError, match="voiced"):
        validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)


def test_a_subject_round_trips():
    reply = _reply(_entry(1, subject="Jan"), _entry(2), _entry(3))
    beats = validate_beats(reply, _mixed_chunks(), cast_names=_CAST_NAMES)

    assert tuple(Beat.from_dict(b.to_dict()) for b in beats) == beats


def test_from_dict_tolerates_a_persisted_beat_sheet_with_no_subject_key():
    """A pre-#82 persisted beat sheet has no `subject` key at all --
    `payload.get`, exactly as `location` tolerates its own pre-#78 absence."""
    beats = validate_beats(_three_beat_gag(), _chunks())
    payload = beats[0].to_dict()
    del payload["subject"]

    restored = Beat.from_dict(payload)

    assert restored.subject is None


def test_generate_beats_threads_the_cast_into_validation(tmp_path):
    """`config.cast` is already in scope in `generate_beats` -- the cast
    membership check must use it without a caller threading anything extra."""
    bad = _reply(
        _entry(1, subject="Not A Cast Member"),
        _entry(2),
        _entry(3),
    )
    good = _reply(_entry(1, subject="Marcus"), _entry(2), _entry(3))
    driver = ScriptedDriver([bad, good])

    result = generate_beats(_config(tmp_path), _mixed_chunks(), CONCEPT, driver)

    assert len(driver.calls) == 2
    assert "cast members" in driver.calls[1]["prompt"]
    assert next(b for b in result.beats if b.chunk_id == 1).subject == "Marcus"


def test_the_beats_preamble_documents_subject():
    from music_video_maker.authoring.prompts import BEATS_PREAMBLE

    lowered = BEATS_PREAMBLE.lower()
    assert "subject" in lowered
    assert "instrumental" in lowered
    assert "#82" in BEATS_PREAMBLE or "82" in BEATS_PREAMBLE


def test_the_user_prompt_names_the_default_lead_vocalist(tmp_path):
    """The model can only apply "someone other than the default lead
    vocalist" if it is told who that is."""
    config = _config(tmp_path)

    prompt = build_beats_prompt(config, _chunks(), CONCEPT)

    assert config.default_lead_vocalist in prompt


# --------------------------------------------------------------------------- #
# A sung chunk's beat belongs to the person singing it.
#
# The alignment already knows who sings each chunk -- the render uses
# `chunk.characters` to pick which cast photo conditions the shot. The
# authoring layer never showed it to the model, so the beat sheet assigned
# the OTHER character to chunks the singer carries, and every downstream
# stage then did its job correctly on a wrong premise: prose wrote him into
# the sentence, photography framed him close, `present` supplied his
# photograph. Measured on the first machine-authored plan to reach a GPU:
# 25 of 41 sung chunks ended up framed on whoever was not singing them.
# --------------------------------------------------------------------------- #


def test_the_skeleton_table_names_who_sings_each_chunk():
    """Unenforceable otherwise: the model cannot honour a rule about the
    singer if the skeleton never says who that is."""
    from music_video_maker.authoring.chunks import skeleton_table_text

    chunks = [
        _chunk(1, "Walking the empty road tonight", characters=("Dianne",)),
        _chunk(2, "", instrumental=True),
        _chunk(3, "Nobody was counting then", characters=("Jan",)),
    ]

    table = skeleton_table_text(chunks)
    rows = [r.split("\t") for r in table.strip().split("\n")]

    assert "Dianne" in rows[0]
    assert "Jan" in rows[2]
    # An instrumental has nobody singing it, and must not claim otherwise.
    assert "Dianne" not in rows[1] and "Jan" not in rows[1]


def test_the_beats_preamble_puts_a_sung_beat_on_its_own_singer():
    from music_video_maker.authoring.prompts import BEATS_PREAMBLE

    lowered = BEATS_PREAMBLE.lower()
    assert "singer" in lowered
    assert "25 of 41" in BEATS_PREAMBLE
