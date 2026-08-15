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


def _entry(chunk_id, *, role="transition", group=1, focus="subject", **extra) -> dict:
    return {
        "chunk_id": chunk_id,
        "beat": f"beat {chunk_id}",
        "beat_role": role,
        "beat_group": group,
        "focus": focus,
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
    ],
)
def test_malformed_fields_are_rejected(bad):
    entry = _entry(1)
    entry.update(bad)
    with pytest.raises(BeatsValidationError):
        validate_beats(_reply(entry, _entry(2), _entry(3)), _chunks())


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
