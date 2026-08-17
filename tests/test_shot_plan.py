"""Tests for the authored shot plan (per-chunk narrative direction).

`narrative_concept` is one global string, so every chunk in a run gets the
same scene description. That produces a video with no progression: forty
renders of the same sentence, and -- if the sentence names several settings --
forty identical montages cycling through all of them.

A shot plan replaces that with one authored line per chunk, so shot 12 can
pay off a gag planted in shot 4. The planning happens ONCE, at authoring
time, and its output is committed data. Nothing here calls an LLM: the render
loop stays pure string composition, deterministic and resumable, exactly as
prompting.py's docstring promises.

The interesting failure mode is drift. A plan is written against one
alignment; re-running alignment with different lyrics or a different model
renumbers the chunks, and shot 12's direction would then silently land on
chunk 14's audio. Every entry therefore carries the chunk start time it was
authored against, and a mismatch is a hard error rather than a warning --
a misaligned plan produces a video that looks intentional and is wrong.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import music_video_maker.shot_plan as shot_plan_module
from music_video_maker.contracts import AudioChunk, CastMember
from music_video_maker.shot_plan import (
    ShotPlanDriftError,
    ShotPlanEntry,
    ShotPlanError,
    lint_camera_face_away_on_voiced_chunks,
    lint_instrumental_focus_mismatch,
    lint_present_location_mismatch,
    lint_role_prohibition_contradiction,
    lint_shots_against_lyrics,
    lint_subject_on_voiced_chunk,
    lint_unbound_companion_referent,
    lint_voiced_framing,
    load_shot_plan,
    render_shot_plan_skeleton,
    resolve_camera,
    resolve_location,
    resolve_shot,
    resolve_subject,
    stageable_noun_stems,
    write_shot_plan_skeleton,
)

TOLERANCE_EXCEEDED = 5.0


def _chunk(chunk_id: int, start: float, end: float, text: str = "a lyric") -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=None,  # type: ignore[arg-type]  -- unused by shot-plan resolution
        start=start,
        end=end,
        text=text,
        characters=("Dianne",),
        frame_count=141,
    )


def _write_plan(tmp_path, body: str):
    path = tmp_path / "shot_plan.toml"
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_loads_entries_keyed_by_chunk_id(tmp_path):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
shot = "She steps out of her front door into an ordinary morning"

[[shot]]
chunk_id = 1
start = 8.0
shot = "A bin lorry reverses into a parked car behind her"
""",
    )

    plan = load_shot_plan(path)

    assert set(plan) == {0, 1}
    assert plan[1].shot == "A bin lorry reverses into a parked car behind her"
    assert plan[1].start == pytest.approx(8.0)


def test_duplicate_chunk_id_is_rejected(tmp_path):
    """Two directions for one chunk means one of them is silently discarded."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 0
start = 0.0
shot = "first"

[[shot]]
chunk_id = 0
start = 0.0
shot = "second"
""",
    )

    with pytest.raises(ShotPlanError) as excinfo:
        load_shot_plan(path)
    assert "0" in str(excinfo.value)


def test_entry_missing_shot_text_is_rejected(tmp_path):
    path = _write_plan(tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\n')
    with pytest.raises(ShotPlanError):
        load_shot_plan(path)


def test_non_string_shot_text_is_rejected(tmp_path):
    path = _write_plan(tmp_path, "[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = 5\n")
    with pytest.raises(ShotPlanError):
        load_shot_plan(path)


def test_blank_shot_text_loads_and_falls_back_like_no_entry(tmp_path, caplog):
    """Issue #52: a blank shot (what --prepare's skeleton emits) is an
    unfilled line, not a malformed one -- it must load, and resolve_shot must
    fall back to the global concept with a warning, exactly like a chunk with
    no entry at all."""
    path = _write_plan(tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "   "\n')

    plan = load_shot_plan(path)

    assert plan[0].shot == ""
    with caplog.at_level(logging.WARNING):
        assert resolve_shot(plan, _chunk(0, 0.0, 5.167)) is None
    assert "0" in caplog.text


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ShotPlanError):
        load_shot_plan(tmp_path / "nope.toml")


# --------------------------------------------------------------------------- #
# Resolution against real chunks
# --------------------------------------------------------------------------- #


def test_resolve_returns_the_authored_shot_for_a_matching_chunk(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "the authored direction"\n',
    )
    plan = load_shot_plan(path)

    assert resolve_shot(plan, _chunk(0, 0.0, 5.167)) == "the authored direction"


def test_resolve_tolerates_sub_frame_start_jitter(tmp_path):
    """Chunk starts are floats accumulated across the timeline; a few
    microseconds of drift is arithmetic, not a stale plan."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 3\nstart = 21.166667\nshot = "direction"\n',
    )
    plan = load_shot_plan(path)

    assert resolve_shot(plan, _chunk(3, 21.166669, 26.33)) == "direction"


def test_resolve_raises_when_the_plan_has_drifted(tmp_path):
    """The whole point: a plan written against a different alignment must
    fail loudly instead of pairing shot 3 with chunk 3's new audio."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 3\nstart = 21.17\nshot = "direction"\n',
    )
    plan = load_shot_plan(path)

    with pytest.raises(ShotPlanDriftError) as excinfo:
        resolve_shot(plan, _chunk(3, 21.17 + TOLERANCE_EXCEEDED, 32.0))
    message = str(excinfo.value)
    assert "3" in message
    assert "re-author" in message.lower() or "realign" in message.lower()


def test_resolve_falls_back_when_a_chunk_has_no_entry(tmp_path, caplog):
    """A partially authored plan is a normal state while writing one, so a
    missing entry warns and falls back rather than aborting the run."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "only chunk zero"\n',
    )
    plan = load_shot_plan(path)

    with caplog.at_level(logging.WARNING):
        assert resolve_shot(plan, _chunk(7, 40.0, 46.0)) is None
    assert "7" in caplog.text


def test_resolve_with_no_plan_at_all_returns_none():
    assert resolve_shot(None, _chunk(0, 0.0, 5.167)) is None


# --------------------------------------------------------------------------- #
# Geography lint against `setting` (issue #32's "Interaction with the shot plan")
# --------------------------------------------------------------------------- #

LONDON_SETTING = "London, UK -- contemporary, overcast winter"
NEW_YORK_SETTING = "New York City, USA -- gritty modern day"


def test_shot_naming_a_landmark_from_a_different_city_warns(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "She wanders through Central Park at dusk"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=LONDON_SETTING)

    assert len(caplog.records) == 1
    assert "central park" in caplog.text.lower()
    assert str(0) in caplog.text


def test_ordinary_prose_does_not_warn(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "She walks down a quiet residential street"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=LONDON_SETTING)

    assert caplog.records == []


def test_a_location_that_matches_the_setting_does_not_warn(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "She wanders through Central Park at dusk"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=NEW_YORK_SETTING)

    assert caplog.records == []


def test_a_cast_member_name_that_matches_a_landmark_word_does_not_warn(tmp_path, caplog):
    """A performer coincidentally named after a place (e.g. 'Manhattan') must
    never be mistaken for a geography slip."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "Manhattan steps into frame and looks around"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=LONDON_SETTING, cast_names=["Manhattan"])

    assert caplog.records == []


def test_no_setting_never_warns_even_with_a_landmark_mismatch(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "She wanders through Central Park at dusk"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Shot-vs-lyric lint (issue #37)
#
# "The Lucky Ones" sings "your printer would explode somehow" twice, at
# 0:26-0:40 and again at 3:03-3:17. The plan staged a printer only for the
# second, so over the first the video showed a hospital gag while the lyric
# named a printer. Both strings were in hand at prompt-composition time and
# nothing compared them.
#
# The lint only fires on a word the author staged SOMEWHERE ELSE in the same
# plan. Requiring shots to echo lyric words in general would fire on nearly
# every chunk and be rightly ignored; a word already in the author's own shot
# vocabulary is demonstrably an object they meant to stage.
# --------------------------------------------------------------------------- #

def _chunk_stub(
    chunk_id: int,
    text: str,
    *,
    instrumental: bool = False,
    characters: tuple[str, ...] = (),
):
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id}.wav"),
        start=float(chunk_id),
        end=float(chunk_id) + 5.0,
        text=text,
        is_instrumental=instrumental,
        characters=characters,
    )


def _plan_from(pairs: dict[int, str]) -> dict[int, ShotPlanEntry]:
    return {
        cid: ShotPlanEntry(chunk_id=cid, start=float(cid), shot=shot)
        for cid, shot in pairs.items()
    }


def test_an_object_named_in_the_lyric_but_staged_only_elsewhere_warns(caplog):
    plan = _plan_from({
        4: "The corridor: the plug lies loose and every monitor has flatlined",
        29: "The printer at the end of the desk aisle jammed and juddering",
    })
    chunks = [
        _chunk_stub(4, "There was a time when I hoped your printer"),
        _chunk_stub(29, "I hoped your printer would explode"),
    ]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "printer" in messages
    assert "chunk_id=4" in messages
    assert "29" in messages  # names where it IS staged, so the fix is obvious


def test_an_object_staged_where_its_lyric_falls_does_not_warn(caplog):
    """The snow plow and the muted mic are staged where they are sung. Silence
    on these is what makes the printer warning worth reading."""
    plan = _plan_from({
        12: "A snow plow pulls out behind her as a jogger runs into frame",
        24: "Close on her finger pressing a glowing red mute button",
    })
    chunks = [
        _chunk_stub(12, "Central Park That you'd run in front of a snow plow."),
        _chunk_stub(24, "I mute your mic when you're talking"),
    ]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    assert caplog.records == []


def test_ordinary_prose_that_shares_no_words_with_the_lyric_does_not_warn(caplog):
    """A shot is not required to echo its lyric -- most legitimately do not."""
    plan = _plan_from({0: "She walks toward camera along a grey sidewalk"})
    chunks = [_chunk_stub(0, "There was a time I felt the world was breaking")]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    assert caplog.records == []


def test_instrumental_chunks_are_never_linted(caplog):
    plan = _plan_from({0: "A printer sits at the end of the aisle", 1: "She walks on"})
    chunks = [_chunk_stub(0, "A printer somewhere"), _chunk_stub(1, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    assert caplog.records == []


def test_the_lint_never_raises_whatever_it_finds():
    """A false positive must not be able to block a run."""
    plan = _plan_from({0: "The printer erupts", 1: "She walks past a window"})
    chunks = [_chunk_stub(0, "no match here"), _chunk_stub(1, "your printer explodes")]

    lint_shots_against_lyrics(plan, chunks)  # must simply return


def test_an_empty_plan_or_no_chunks_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics({}, [_chunk_stub(0, "your printer")])
        lint_shots_against_lyrics(_plan_from({0: "a printer"}), [])

    assert caplog.records == []


def test_an_object_staged_in_an_adjacent_chunk_does_not_warn(caplog):
    """Three-beat staging spreads a gag across consecutive chunks, so an
    object established in the neighbouring shot is part of the same visual
    moment. Measured: without this, the lint warned that chunk 12's lyric
    names a park while chunk 11 -- the shot immediately before it --
    establishes the park she is standing in."""
    plan = _plan_from({
        11: "She crosses into a wide downtown park, bare trees behind her",
        12: "A snow plow pulls out behind her as a jogger runs into frame",
        30: "A park bench sits in the foreground",
    })
    chunks = [_chunk_stub(12, "Central Park That you'd run in front of a snow plow.")]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    assert caplog.records == []


def test_generic_people_words_never_warn(caplog):
    """'people' produced half the noise on the first real run: it is sung
    constantly and written into shot prose constantly, and is never a
    stageable prop."""
    plan = _plan_from({
        0: "Jan stands among the moving crowd with his bass",
        9: "Several people rise out of their chairs",
    })
    chunks = [_chunk_stub(0, "I know when it's people like you Hating people like me")]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)

    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Per-shot focus (issue #26)
#
# "Dianne, <role>, is the focus of this shot" is composed into EVERY chunk,
# including the ones whose entire job is to show a consequence she has already
# walked away from. The prompt then says both "the flatlining monitors are the
# beat" and "she is the focus", and the character-attached instruction wins --
# the same shape as the bug that made her sing through the instrumentals.
# --------------------------------------------------------------------------- #


def test_focus_defaults_to_the_subject(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
    )
    assert load_shot_plan(path)[0].subject_is_focus is True


def test_a_shot_can_declare_the_action_is_the_focus(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "The printer erupts in a fireball behind her"\nfocus = "action"\n',
    )
    assert load_shot_plan(path)[0].subject_is_focus is False


def test_focus_subject_is_accepted_explicitly(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\nfocus = "subject"\n',
    )
    assert load_shot_plan(path)[0].subject_is_focus is True


def test_an_unknown_focus_value_fails_loudly(tmp_path):
    """A typo'd focus that silently meant "subject" would be config that reads
    as applied and never reaches a prompt."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\nfocus = "the printer"\n',
    )
    with pytest.raises(ShotPlanError) as excinfo:
        load_shot_plan(path)
    assert "focus" in str(excinfo.value)
    assert "action" in str(excinfo.value)  # names the valid values


# --------------------------------------------------------------------------- #
# Consequence-vs-focus lint (issue #41)
#
# focus = "action" (issue #26) exists precisely so a consequence beat -- the
# printer that erupts, the monitors that flatline -- can hand the shot's
# subject to the thing that actually happened instead of the performer who
# already walked away. Nothing told an author when they forgot to reach for
# it, so a plan could describe the world changing while still implicitly
# keeping the singer as the subject. This lint closes that gap: a warning,
# never an error, when a shot reads as a consequence and does not set
# focus = "action".
# --------------------------------------------------------------------------- #


def test_a_consequence_shot_without_focus_warns(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 5\nstart = 12.0\n'
        'shot = "The printer erupts in a fireball behind her"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert len(caplog.records) == 1
    assert "chunk_id=5" in caplog.text
    assert 'focus = "action"' in caplog.text


def test_the_same_shot_with_focus_action_does_not_warn(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 5\nstart = 12.0\n'
        'shot = "The printer erupts in a fireball behind her"\nfocus = "action"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert caplog.records == []


def test_an_ordinary_performance_shot_does_not_warn(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "She walks along the corridor, glancing back over her shoulder"\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert caplog.records == []


def test_the_consequence_lint_never_raises_whatever_it_finds(tmp_path):
    """A false positive here must not be able to block a run."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\n'
        'shot = "Monitors flatline, the printer shatters and spills sparks '
        'as the lights flicker and short-circuit"\n',
    )

    load_shot_plan(path)  # must simply return, not raise


# The keywords below (`flat line`, `flood`, `smoke`, `jammed`, `window dark`)
# were added after measuring this lint against the real 38-entry plan
# authored for "The Lucky Ones" (issue #41 review): the original list caught
# only 1 of 5 genuine consequence beats in that file. These fixtures are
# original prose exercising the same categories, not the real plan's text,
# so the suite stays self-contained and offline.


@pytest.mark.parametrize(
    "shot_text",
    [
        "Every monitor on the cart behind her has dropped to a flat line",
        "Every monitor on the cart behind her has dropped to a flat-line",
        "Coffee floods across the keyboard while she is already leaving",
        "Smoke starts curling from the printer's seams as she rounds the corner",
        "The printer at the end of the aisle sits jammed, paper crumpled in its tray",
        "The control room's window dark, an engineer staring at the muted desk",
    ],
)
def test_additional_consequence_variants_found_by_measuring_a_real_plan_warn(
    tmp_path, caplog, shot_text
):
    path = _write_plan(
        tmp_path, f'[[shot]]\nchunk_id = 9\nstart = 3.0\nshot = "{shot_text}"\n'
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert len(caplog.records) == 1


@pytest.mark.parametrize(
    "shot_text",
    [
        # A person can black out; monitors "blacking out" is not distinguishable
        # from that on text alone, so this stays unflagged (already covered by
        # `flood`/`spill` when the shot also names the spill itself).
        "The monitors above the desk black out one after another",
        # Present-tense/gerund "jam(s)/jamming" reads as a band jamming, which
        # is ordinary music-video prose -- only the mechanical "jammed" is kept.
        "Jan and the band start jamming as the crowd presses in",
        # Bare "dark" describes ordinary scene lighting constantly; only the
        # specific "window dark" phrase is curated in.
        "She walks down a dark alley toward the waiting car",
        # "smoking" is at least as likely to be a person with a cigarette.
        "He leans on the doorway smoking, watching her go",
    ],
)
def test_ambiguous_variants_considered_and_rejected_do_not_warn(
    tmp_path, caplog, shot_text
):
    path = _write_plan(
        tmp_path, f'[[shot]]\nchunk_id = 9\nstart = 3.0\nshot = "{shot_text}"\n'
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    # Asserted against *this* lint's message rather than "no warnings at all":
    # the last case ("He leans on the doorway...") legitimately trips the
    # implied-companion lint (issue #64), which is a different check with its
    # own reasons. A blanket record count made this test quietly own every
    # lint in the module.
    assert "reads as a consequence beat" not in caplog.text


# --------------------------------------------------------------------------- #
# Camera direction (issue #53)
# --------------------------------------------------------------------------- #


def test_camera_is_none_by_default(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
    )
    assert load_shot_plan(path)[0].camera is None


def test_camera_is_read_from_the_entry(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
        'camera = "tracking backwards ahead of her"\n',
    )
    assert load_shot_plan(path)[0].camera == "tracking backwards ahead of her"


def test_a_non_string_camera_is_rejected(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\ncamera = 5\n'
    )
    with pytest.raises(ShotPlanError):
        load_shot_plan(path)


def test_a_blank_camera_is_treated_as_unset(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\ncamera = "   "\n'
    )
    assert load_shot_plan(path)[0].camera is None


def test_resolve_camera_returns_the_authored_direction(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
        'camera = "tracking backwards ahead of her"\n',
    )
    plan = load_shot_plan(path)

    assert resolve_camera(plan, _chunk(0, 0.0, 5.167)) == "tracking backwards ahead of her"


def test_resolve_camera_applies_even_when_shot_is_blank(tmp_path):
    """Camera direction is independent of shot: an unfilled shot line still
    lets camera direction pair with the global narrative_concept fallback."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = ""\n'
        'camera = "pushing in slowly on her hands"\n',
    )
    plan = load_shot_plan(path)

    assert resolve_shot(plan, _chunk(0, 0.0, 5.167)) is None
    assert resolve_camera(plan, _chunk(0, 0.0, 5.167)) == "pushing in slowly on her hands"


def test_resolve_camera_with_no_entry_returns_none(tmp_path, caplog):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "only chunk zero"\n'
    )
    plan = load_shot_plan(path)

    with caplog.at_level(logging.WARNING):
        assert resolve_camera(plan, _chunk(7, 40.0, 46.0)) is None


def test_resolve_camera_with_no_plan_returns_none():
    assert resolve_camera(None, _chunk(0, 0.0, 5.167)) is None


def test_resolve_camera_raises_on_drift_same_as_resolve_shot(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 3\nstart = 21.17\nshot = "direction"\n'
        'camera = "tracking backwards"\n',
    )
    plan = load_shot_plan(path)

    with pytest.raises(ShotPlanDriftError):
        resolve_camera(plan, _chunk(3, 21.17 + TOLERANCE_EXCEEDED, 32.0))


# --------------------------------------------------------------------------- #
# --prepare skeleton generation (issue #52)
# --------------------------------------------------------------------------- #


def _instrumental_chunk(
    chunk_id: int, start: float, end: float, frame_count: int = 141
) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=None,  # type: ignore[arg-type]
        start=start,
        end=end,
        text="",
        characters=("Dianne",),
        frame_count=frame_count,
        is_instrumental=True,
    )


def test_render_shot_plan_skeleton_emits_blank_shots_for_every_chunk():
    chunks = [
        _chunk(0, 0.0, 5.875, text="and the lights flicker but i do not mind"),
        _instrumental_chunk(1, 5.875, 16.0),
    ]

    text = render_shot_plan_skeleton(chunks, source="run.toml", generated_at="2026-08-12")

    assert text.count('shot = ""') == 2
    assert "chunk_id = 0" in text
    assert "chunk_id = 1" in text
    assert "and the lights flicker but i do not mind" in text
    assert "INSTRUMENTAL" in text
    assert "run.toml" in text
    assert "2026-08-12" in text


def test_render_shot_plan_skeleton_escapes_quotes_in_the_lyric_comment():
    chunks = [_chunk(0, 0.0, 5.875, text='she said "hello" to no one')]

    text = render_shot_plan_skeleton(chunks, source="run.toml", generated_at="2026-08-12")

    assert '"hello"' not in text
    assert "'hello'" in text


def test_render_shot_plan_skeleton_loads_cleanly_with_no_drift(tmp_path):
    """The whole point of #52: chunk_id/start come straight from the chunks
    that produced them, so loading the skeleton back against those same
    chunks can never raise ShotPlanDriftError."""
    chunks = [
        _chunk(0, 0.0, 5.875, text="a lyric"),
        _instrumental_chunk(1, 5.875, 16.0),
    ]
    text = render_shot_plan_skeleton(chunks, source="run.toml", generated_at="2026-08-12")

    path = tmp_path / "shot_plan.toml"
    path.write_text(text)
    plan = load_shot_plan(path)

    for chunk in chunks:
        assert resolve_shot(plan, chunk) is None  # blank -- falls back, never raises


def test_write_shot_plan_skeleton_writes_the_file(tmp_path):
    chunks = [_chunk(0, 0.0, 5.875, text="a lyric")]

    out = write_shot_plan_skeleton(
        chunks, tmp_path / "shot_plan.toml", source="run.toml", generated_at="2026-08-12"
    )

    assert out == tmp_path / "shot_plan.toml"
    assert out.exists()
    assert 'shot = ""' in out.read_text()


def test_write_shot_plan_skeleton_refuses_to_overwrite_an_existing_file(tmp_path):
    out_path = tmp_path / "shot_plan.toml"
    out_path.write_text(
        '# hand-authored, real work\n[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "x"\n'
    )
    chunks = [_chunk(0, 0.0, 5.875, text="a lyric")]

    with pytest.raises(ShotPlanError):
        write_shot_plan_skeleton(
            chunks, out_path, source="run.toml", generated_at="2026-08-12"
        )

    assert "hand-authored" in out_path.read_text()


def test_write_shot_plan_skeleton_wraps_an_os_error(tmp_path):
    # A directory at the output path can never be written_text()'d -- forces
    # the OSError branch without touching real filesystem permissions.
    out_path = tmp_path / "shot_plan.toml"
    out_path.mkdir()
    chunks = [_chunk(0, 0.0, 5.875, text="a lyric")]

    with pytest.raises(ShotPlanError):
        write_shot_plan_skeleton(
            chunks, out_path, source="run.toml", generated_at="2026-08-12", force=True
        )


def test_write_shot_plan_skeleton_force_overwrites(tmp_path):
    out_path = tmp_path / "shot_plan.toml"
    out_path.write_text("stale content\n")
    chunks = [_chunk(0, 0.0, 5.875, text="a lyric")]

    write_shot_plan_skeleton(
        chunks, out_path, source="run.toml", generated_at="2026-08-12", force=True
    )

    assert "stale content" not in out_path.read_text()
    assert 'shot = ""' in out_path.read_text()


# --------------------------------------------------------------------------- #
# Camera double-composition, and unknown keys (issue #54 design section 6)
# --------------------------------------------------------------------------- #


def test_camera_language_in_shot_alongside_a_camera_field_warns(tmp_path, caplog):
    """``prompting`` composes ``camera`` as a trailing "..., camera <x>"
    clause, so a shot line that already carries its own camera phrase
    composes twice -- "..., camera tracking backwards, camera pushing in".
    Issue #53 opened this door by giving camera a field of its own while
    ``shot`` stayed free text; nothing was checking that an author had moved
    the direction rather than duplicated it."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
camera = "pushing in slowly on her hands"
shot = "She rounds the end of the office aisle, camera tracking backwards ahead of her"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "composes twice" in caplog.text
    assert "chunk_id=4" in caplog.text


def test_camera_language_in_shot_with_no_camera_field_is_silent(tmp_path, caplog):
    """The pre-#53 style -- a trailing camera phrase inside ``shot`` -- is
    still exactly right when there is no ``camera`` field to collide with,
    and ``docs/shot-writing-guide.md`` is full of worked examples in it."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
shot = "She rounds the end of the office aisle, camera tracking backwards ahead of her"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "composes twice" not in caplog.text


def test_a_camera_field_with_no_camera_word_in_the_shot_is_silent(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
camera = "pushing in slowly on her hands"
shot = "She rounds the end of the office aisle, still singing"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "composes twice" not in caplog.text


def test_an_unknown_key_in_a_shot_entry_warns(tmp_path, caplog):
    """``_parse_entry`` reads only the keys it knows, so a typo used to be
    dropped in silence. That was tolerable when every key was typed by the
    person who would notice; it is not once a model is emitting them."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
lenght_seconds = 9.0
camara = "pushing in"
shot = "The printer erupts in a fireball"
""",
    )

    with caplog.at_level(logging.WARNING):
        plan = load_shot_plan(path)

    assert "lenght_seconds" in caplog.text
    assert "camara" in caplog.text
    assert plan[4].length_seconds is None  # really was dropped -- that is the point


def test_provenance_keys_are_not_reported_as_unknown(tmp_path, caplog):
    """``generated_by``/``content_sha256`` are unknown to the renderer by
    design (issue #54 design section 8): provenance lives in the file itself
    rather than a sidecar, precisely because the renderer ignores it."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
generated_by = "prose"
content_sha256 = "e11a"
shot = "The printer erupts in a fireball"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "generated_by" not in caplog.text
    assert "content_sha256" not in caplog.text


def test_the_lyric_lint_ignores_the_preposition_through(tmp_path, caplog):
    """Same part of speech as the five prepositions already beside it in the
    stopword list, and simply missed. It fired on a real generated plan
    against the lyric "As you ran Through Central Park"."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 1
start = 0.0
shot = "The plow's blade bites into the snowbank as she walks on"

[[shot]]
chunk_id = 5
start = 30.0
shot = "Paper pours down through the falling snow behind her"
""",
    )
    chunks = [
        _chunk(1, 0.0, 5.875, text="as you ran through central park"),
        _chunk(5, 30.0, 35.875, text="another line entirely"),
    ]

    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(load_shot_plan(path), chunks)

    assert "'through'" not in caplog.text


# --------------------------------------------------------------------------- #
# Distant-staging lint (issue #58)
# --------------------------------------------------------------------------- #


def test_a_subject_staged_far_behind_warns(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
shot = "A beige printer sits blinking on the lit sill, small against the tower far behind her"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "chunk_id=4" in caplog.text
    assert "small or far away" in caplog.text


def test_a_subject_staged_near_is_silent(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
shot = "The printer at the end of the office aisle erupts in a fireball"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "small or far away" not in caplog.text


def test_bare_far_and_small_do_not_warn(tmp_path, caplog):
    """Bare 'far' and 'small' are ordinary prose far more often than they are
    staging distance -- only the curated phrases/words fire."""
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
shot = "She has come so far tonight, standing in a small office by the window"
""",
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "small or far away" not in caplog.text


def test_the_distant_staging_lint_never_raises_whatever_it_finds(tmp_path):
    path = _write_plan(
        tmp_path,
        """
[[shot]]
chunk_id = 4
start = 26.0
shot = "The printer lies smashed in a snowbank far behind on the street, tiny in the distance"
""",
    )

    load_shot_plan(path)  # must simply return


# --------------------------------------------------------------------------- #
# Camera-vs-lip-sync lint (issue #58)
# --------------------------------------------------------------------------- #


def test_a_camera_direction_that_turns_her_away_on_a_voiced_chunk_warns(caplog):
    plan = {
        26: ShotPlanEntry(
            chunk_id=26,
            start=26.0,
            shot="An empty jagged pane, an overturned chair in the snow",
            camera="craning up the building face, leaving her back to camera",
        )
    }
    chunks = [_chunk_stub(26, "the world was breaking, always taking me")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert "chunk_id=26" in caplog.text
    assert "away from the lens" in caplog.text


def test_the_same_camera_direction_on_an_instrumental_chunk_is_silent(caplog):
    plan = {
        26: ShotPlanEntry(
            chunk_id=26,
            start=26.0,
            shot="An empty jagged pane, an overturned chair in the snow",
            camera="craning up the building face, leaving her back to camera",
        )
    }
    chunks = [_chunk_stub(26, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_a_camera_direction_that_keeps_her_face_available_is_silent(caplog):
    plan = {
        26: ShotPlanEntry(
            chunk_id=26,
            start=26.0,
            shot="An empty jagged pane, an overturned chair in the snow",
            camera="pushing in past her shoulder to the empty pane, then holding on both",
        )
    }
    chunks = [_chunk_stub(26, "the world was breaking, always taking me")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_a_voiced_chunk_with_no_camera_direction_is_silent(caplog):
    plan = {
        26: ShotPlanEntry(chunk_id=26, start=26.0, shot="She walks on"),
    }
    chunks = [_chunk_stub(26, "the world was breaking, always taking me")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_focus_action_alone_on_a_voiced_chunk_no_longer_warns(caplog):
    """RETIRED signal (issue #60, retiring #58's primary check).

    #58 flagged `focus = "action"` on a voiced chunk, generalising from one
    chunk re-rendered three times. The first full 36-chunk render tested it
    on 7 voiced chunks at once and the generalisation did not hold: all 7
    turned away as predicted, and all 7 kept their lip-sync. The predicted
    *cost* never materialised. 7 false positives is worse than no lint --
    it trains the reader to skip the warning that matters."""
    plan = {
        26: ShotPlanEntry(
            chunk_id=26,
            start=26.0,
            shot="The pane hangs empty and jagged",
            subject_is_focus=False,
        )
    }
    chunks = [_chunk_stub(26, "the world was breaking, always taking me")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_a_shot_line_walking_her_away_on_a_voiced_chunk_warns(caplog):
    """The REPLACEMENT signal (issue #60), taken from the one chunk in that
    render that actually lost sync -- chunk 13, "There was a time / But
    that's the past". Its line put the camera behind them and had them
    walking on, while `focus` was unset and the old lint said nothing. The
    predictor was in the prose, not the field."""
    plan = {
        13: ShotPlanEntry(
            chunk_id=13,
            start=83.5,
            shot=(
                "Behind them, slush slides slowly off the buried car's roof as they "
                "keep walking, his bass still setting the pace beside her"
            ),
        )
    }
    chunks = [_chunk_stub(13, "There was a time But that's the past")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert "chunk_id=13" in caplog.text
    assert "walking away from the lens" in caplog.text


def test_the_same_shot_line_on_an_instrumental_chunk_is_silent(caplog):
    plan = {
        13: ShotPlanEntry(
            chunk_id=13, start=83.5, shot="Behind them, slush slides as they keep walking"
        )
    }
    chunks = [_chunk_stub(13, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_a_shot_line_that_merely_stages_something_behind_her_is_silent(caplog):
    """"Behind her" is about narrative obliviousness, not the camera -- it is
    in 20 of the 36 lines of a real plan whose lip-sync was fine. Flagging it
    would make this lint useless."""
    plan = {
        16: ShotPlanEntry(
            chunk_id=16,
            start=103.2,
            shot=(
                "Office paper pours down the face of the tower, settling into the "
                "falling snow behind her as she walks on unbothered in the foreground"
            ),
        )
    }
    chunks = [_chunk_stub(16, "people like you hating people like me")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_focus_action_on_an_instrumental_chunk_is_silent(caplog):
    plan = {
        26: ShotPlanEntry(
            chunk_id=26, start=26.0, shot="The pane hangs empty", subject_is_focus=False
        )
    }
    chunks = [_chunk_stub(26, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_receding_is_deliberately_not_a_keyword(caplog):
    """An evidence-based exclusion, not an oversight. "already receding
    behind her unhurried stride" is chunk 7 of the same real render: it
    turned away and kept its lip-sync. Including the word would have cost a
    false positive for no true one."""
    plan = {
        7: ShotPlanEntry(
            chunk_id=7,
            start=46.8,
            shot=(
                "The stapler lies half-sunk in gray slush, already receding behind "
                "her unhurried stride, its keys dark with wet"
            ),
        )
    }
    chunks = [_chunk_stub(7, "wondering when it's ending")]

    with caplog.at_level(logging.WARNING):
        lint_camera_face_away_on_voiced_chunks(plan, chunks)

    assert caplog.records == []


def test_the_camera_lint_never_raises_whatever_it_finds():
    plan = {
        26: ShotPlanEntry(
            chunk_id=26, start=26.0, shot="She walks on", camera="turning away from her"
        )
    }
    chunks = [_chunk_stub(26, "a lyric")]

    lint_camera_face_away_on_voiced_chunks(plan, chunks)  # must simply return


# --------------------------------------------------------------------------- #
# `present`: who is on screen, as opposed to who is singing (issue #59)
# --------------------------------------------------------------------------- #


def test_present_is_parsed_as_a_tuple_of_names(tmp_path: Path):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "She walks past him"\n'
        'present = ["Jan"]\n',
    )
    assert load_shot_plan(plan)[4].present == ("Jan",)


def test_present_defaults_to_empty_not_none(tmp_path: Path):
    """Empty means 'nobody else in shot', which is the historical behaviour
    and must compose byte-identically -- never a None to branch on."""
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "She walks"\n'
    )
    assert load_shot_plan(plan)[4].present == ()


def test_present_that_is_not_a_list_is_an_error(tmp_path: Path):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\npresent = "Jan"\n',
    )
    with pytest.raises(ShotPlanError, match="present"):
        load_shot_plan(plan)


def test_present_with_a_non_string_entry_is_an_error(tmp_path: Path):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\npresent = ["Jan", 3]\n',
    )
    with pytest.raises(ShotPlanError, match="present"):
        load_shot_plan(plan)


def test_present_is_a_known_key(tmp_path: Path, caplog):
    """It must not be reported by the unknown-key lint."""
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\npresent = ["Jan"]\n',
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "nothing reads" not in caplog.text


# --------------------------------------------------------------------------- #
# `subject`: whose shot this is, distinct from `present` (issue #82)
#
# `present` (issue #59) answers who a pronoun binds to; it never answered
# whose shot this IS. On an instrumental chunk `chunk.characters` is empty,
# so `prompting._resolve_active_members` falls back to
# `config.default_lead_vocalist` and composes THAT person as "the focus of
# this shot" -- a config default silently answering a question the shot line
# already answers differently. `subject` names the actual focus; it is a
# single name, not a list (a shot has exactly one subject), and it is legal
# only on an instrumental chunk -- three measured findings (#58, #59, #60)
# say the singer owns the frame on a voiced one, so `subject` there is an
# error (see the lint section below), never a silently-honoured override.
# --------------------------------------------------------------------------- #


def test_subject_is_parsed_as_a_single_name(tmp_path: Path):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\n'
        'shot = "His boot settles into a hollow"\nsubject = "Jan"\n',
    )
    assert load_shot_plan(plan, cast_names=("Jan", "Dianne"))[4].subject == "Jan"


def test_subject_defaults_to_none(tmp_path: Path):
    plan = _write_plan(tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\n')
    assert load_shot_plan(plan)[4].subject is None


def test_subject_wrong_type_is_an_error(tmp_path: Path):
    """Unlike `present`, this is a single name -- a list is the wrong shape."""
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = ["Jan"]\n'
    )
    with pytest.raises(ShotPlanError, match="subject"):
        load_shot_plan(plan)


def test_subject_non_string_is_an_error(tmp_path: Path):
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = 3\n'
    )
    with pytest.raises(ShotPlanError, match="subject"):
        load_shot_plan(plan)


def test_subject_empty_string_is_an_error(tmp_path: Path):
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = "   "\n'
    )
    with pytest.raises(ShotPlanError, match="subject"):
        load_shot_plan(plan)


def test_subject_unknown_cast_name_is_an_error(tmp_path: Path):
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = "Nobody"\n'
    )
    with pytest.raises(ShotPlanError, match="Nobody"):
        load_shot_plan(plan, cast_names=("Jan", "Dianne"))


def test_subject_skips_the_membership_check_when_cast_names_is_empty(tmp_path: Path):
    """The "cast unknown" call sites -- most tests, and any caller that has
    not wired the cast dict through -- must not be forced to supply one just
    to parse a plan, the same rule every other cast-valued field follows."""
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = "Whoever"\n'
    )
    assert load_shot_plan(plan)[4].subject == "Whoever"


def test_subject_is_a_known_key(tmp_path: Path, caplog):
    """It must not be reported by the unknown-key lint."""
    plan = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "x"\nsubject = "Jan"\n'
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan, cast_names=("Jan",))
    assert "nothing reads" not in caplog.text


def test_resolve_subject_returns_none_with_no_plan():
    chunk = _chunk(4, 26.3, 34.3)
    assert resolve_subject(None, chunk) is None


def test_resolve_subject_returns_the_entry_subject():
    plan = {4: ShotPlanEntry(chunk_id=4, start=26.3, shot="x", subject="Jan")}
    chunk = _chunk(4, 26.3, 34.3)
    assert resolve_subject(plan, chunk) == "Jan"


def test_resolve_subject_is_none_when_the_entry_never_set_it():
    plan = {4: ShotPlanEntry(chunk_id=4, start=26.3, shot="x")}
    chunk = _chunk(4, 26.3, 34.3)
    assert resolve_subject(plan, chunk) is None


def test_resolve_subject_raises_on_drift_same_as_resolve_shot():
    """Shares `_resolve_entry`'s drift check -- a stale plan must refuse every
    field the same way, not just the ones the render loop always consumed."""
    plan = {4: ShotPlanEntry(chunk_id=4, start=26.3, shot="x", subject="Jan")}
    chunk = _chunk(4, 26.3 + TOLERANCE_EXCEEDED, 34.3 + TOLERANCE_EXCEEDED)
    with pytest.raises(ShotPlanDriftError):
        resolve_subject(plan, chunk)


# --------------------------------------------------------------------------- #
# The `subject`-on-a-voiced-chunk lint: the one lint in this module that
# RAISES rather than warns (issue #82). The singer owns the frame on a voiced
# chunk (#58, #59, #60); honouring `subject` there would reintroduce the
# desync the field exists to avoid, so it is refused before any GPU time is
# spent rather than silently mis-composed.
# --------------------------------------------------------------------------- #


def test_subject_on_a_voiced_chunk_raises(caplog):
    plan = {4: ShotPlanEntry(chunk_id=4, start=4.0, shot="x", subject="Jan")}
    chunks = [_chunk_stub(4, "a lyric", instrumental=False)]

    with caplog.at_level(logging.ERROR), pytest.raises(ShotPlanError, match="Jan"):
        lint_subject_on_voiced_chunk(plan, chunks)
    assert "chunk_id=4" in caplog.text


def test_subject_on_an_instrumental_chunk_does_not_raise():
    plan = {4: ShotPlanEntry(chunk_id=4, start=4.0, shot="x", subject="Jan")}
    chunks = [_chunk_stub(4, "", instrumental=True)]

    lint_subject_on_voiced_chunk(plan, chunks)  # must simply return


def test_no_subject_set_never_raises_even_on_a_voiced_chunk():
    plan = {4: ShotPlanEntry(chunk_id=4, start=4.0, shot="x")}
    chunks = [_chunk_stub(4, "a lyric", instrumental=False)]

    lint_subject_on_voiced_chunk(plan, chunks)  # must simply return


def test_subject_lint_is_silent_with_no_plan_or_no_chunks():
    lint_subject_on_voiced_chunk({}, [_chunk_stub(4, "a lyric")])
    lint_subject_on_voiced_chunk(
        {4: ShotPlanEntry(chunk_id=4, start=4.0, shot="x", subject="Jan")}, []
    )


def test_subject_lint_ignores_a_chunk_missing_from_the_plan():
    lint_subject_on_voiced_chunk({}, [_chunk_stub(4, "a lyric", instrumental=False)])


# --------------------------------------------------------------------------- #
# Distant staging: the phrasings the first full render actually used (#58)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "phrase",
    ["far below", "far above", "deep background", "high in the background"],
)
def test_distant_staging_catches_the_v9_phrasings(tmp_path: Path, caplog, phrase: str):
    """`far below` staged chunk 27's printer and it did not render. The
    keyword list had "far behind" but not the other four directions."""
    plan = _write_plan(
        tmp_path,
        f'[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "The printer teeters, '
        f'she walking on {phrase}"\n',
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "small or far away" in caplog.text


# --------------------------------------------------------------------------- #
# Anaphora: a shot line cannot refer to a shot H3 has never seen (issue #61)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "phrase",
    [
        "that same empty frame",
        "those same memos",
        "the aforementioned printer",
        "the window from the previous shot",
        "the chair from the earlier shot",
    ],
)
def test_anaphora_is_flagged(tmp_path: Path, caplog, phrase: str):
    plan = _write_plan(
        tmp_path, f'[[shot]]\nchunk_id = 4\nstart = 26.3\nshot = "The printer sits on {phrase}"\n'
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "no memory of any other shot" in caplog.text


def test_a_self_contained_line_is_not_flagged_as_anaphora(tmp_path: Path, caplog):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 4\nstart = 26.3\n'
        'shot = "A beige printer sits blinking on the lit sill beside her"\n',
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "no memory of any other shot" not in caplog.text


# --------------------------------------------------------------------------- #
# An implied companion with nobody bound to it (issue #64)
#
# #59 gave a shot a `present` list; it did not make anyone notice when one is
# MISSING. Found by eye in final_v10: "she's walking with 3 different people,
# none of whom are Jan" across 2:07-2:36. Those lines name no pronoun at all
# -- they refer to the companion by his instrument and by a plural -- so the
# pronoun-shaped audit that populated `present` skipped them.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line",
    [
        "A snowplow rumbles past, while she walks on to the pulse of a bass carried alongside her",
        "The gates come into view, the speaker still squawking beside them as she approaches",
        "A long unbroken bassline paces the walk beside her along the gray water",
        "The stapler sits abandoned in the slush as the walk continues beside a steady bassline",
        "She walks the sidewalk with him keeping pace just behind",
    ],
)
def test_a_shot_implying_a_companion_with_no_present_warns(tmp_path: Path, caplog, line: str):
    plan = _write_plan(
        tmp_path, f'[[shot]]\nchunk_id = 20\nstart = 126.7\nshot = "{line}"\n'
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "second person on screen" in caplog.text


@pytest.mark.parametrize(
    "line",
    [
        "A snowplow rumbles past, while she walks on alone through the grey afternoon",
        "The printer lies smashed in a snowbank right at the curb beside her",
    ],
)
def test_a_shot_with_nobody_else_in_it_is_silent(tmp_path: Path, caplog, line: str):
    plan = _write_plan(tmp_path, f'[[shot]]\nchunk_id = 20\nstart = 126.7\nshot = "{line}"\n')
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "second person on screen" not in caplog.text


def test_the_warning_stops_once_present_binds_somebody(tmp_path: Path, caplog):
    plan = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 20\nstart = 126.7\npresent = ["Jan"]\n'
        'shot = "She walks on to the pulse of a bass carried alongside her"\n',
    )
    with caplog.at_level(logging.WARNING):
        load_shot_plan(plan)
    assert "second person on screen" not in caplog.text


# --------------------------------------------------------------------------- #
# A sung chunk needs the singer close enough to read a mouth.
#
# The first machine-authored plan to reach a GPU put 41 voiced chunks on
# screen with a close or medium framing on **11** of them: 1 explicitly wide
# and 29 with no `camera` value at all. Across the chunks rendered from it,
# a face was detectable in 0-33% of sampled frames, voiced or not, because
# the plan is composed of landscape shots -- mills, valleys, massed armies --
# and the singer is small inside them. No amount of `present` or shot-line
# rewording fixed it; four separate variants of one chunk were rendered and
# every one lost the face.
#
# The generator has no reason to know this: it optimises the image, and a
# wide valley IS the better image. So the constraint has to be stated.
# --------------------------------------------------------------------------- #


def test_a_voiced_chunk_with_no_camera_direction_warns(caplog):
    plan = _plan_from({3: "She stands on the ridge as the valley burns below"})
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert "chunk_id=3" in caplog.text
    assert "camera" in caplog.text.lower()


def test_a_voiced_chunk_framed_wide_warns(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="She stands on the ridge as the valley burns below",
            camera="extreme wide, static, her figure small on the bare hill",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert "chunk_id=3" in caplog.text


def test_a_voiced_chunk_framed_close_is_quiet(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="She stands on the ridge as the valley burns below",
            camera="close on her face, the valley soft behind her",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_an_instrumental_chunk_may_be_as_wide_as_it_likes(caplog):
    """The whole point of the rule is that it applies to sung chunks only --
    an instrumental has no mouth to match, and the wide landscape shots are
    where a music video earns its scale."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="The valley lies churned into craters under a low sky",
            camera="extreme wide, static, the ridge small against the horizon",
        )
    }
    chunks = [_chunk_stub(3, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_a_voiced_chunk_framed_at_foot_level_no_longer_warns(caplog):
    """`_FOOT_LEVEL_KEYWORDS` is retired (issue #76). The underlying idea it
    encoded was real -- this exact scenario (a shot whose line read "pushing
    up between his boots") once rendered legs, boots and mushrooms perfectly,
    with no face anywhere -- but the keyword LIST built from it did not
    generalise: on the full 80-chunk "Deathless" render, the same keyword set
    scored 0.99x against the rest of the voiced corpus, indistinguishable
    from noise ("the ground" alone ranged 33%-92% across its two hits). A
    weak lint costs nothing at runtime and a great deal the moment somebody
    acts on it, so it is retired rather than kept as false confidence.

    This camera/shot pair deliberately avoids any other keyword this lint
    checks (no gaze verb, no "travelling with", no wide/close framing word,
    no "in profile"), so a silent result here isolates the foot-level
    retirement specifically rather than being silenced by some other check."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="Pale mushroom caps push up between his boots where he stands",
            camera="static, the mushrooms breaking up around his boots",
        )
    }
    chunks = [_chunk_stub(3, "Nobody was counting then")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_an_instrumental_chunk_may_be_framed_at_foot_level(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="Mushrooms push up through the churned soil",
            camera="macro on the soil, caps breaking the crust",
        )
    }
    chunks = [_chunk_stub(3, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_a_voiced_chunk_sending_the_singer_gaze_away_warns(caplog):
    """The camera can ask for her face and lose it anyway: the sentence
    outranks the field. Measured -- chunk 9's camera read "medium close,
    slightly above her, her face centre" and its shot line read "as she looks
    back down at them"; she rendered back-to-camera for the whole chunk."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="Armies mass across the valley below, as she looks back down at them",
            camera="medium close, her face centre with the valley out of focus",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert "chunk_id=3" in caplog.text
    assert "looks back" in caplog.text


def test_a_voiced_chunk_glancing_at_something_is_quiet(caplog):
    """`glancing` is deliberately NOT a keyword. The chunk that scored highest
    all run -- 80-89% face presence -- reads "glancing up at a motionless
    figure". A glance returns; a gaze settles."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="She climbs the trail, glancing up at a motionless figure on the ridge",
            camera="medium close on her face as she climbs",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_a_voiced_chunk_shot_from_behind_warns(caplog):
    """"Travelling with her" up a climb is a following shot: the back of a
    head. Measured 0% face presence. Distinct from "ahead of her", which
    faces her and measured 89% on the same run -- the direction of travel is
    not the point, where the lens is relative to the face is."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0, shot="The mill turns slower overhead as she climbs on",
            camera="medium close, travelling with her, the sails passing behind",
        )
    }
    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, [_chunk_stub(3, "Walking the empty road tonight")])
    assert "chunk_id=3" in caplog.text


def test_a_voiced_chunk_shot_from_ahead_is_quiet(caplog):
    """The control that stops this becoming a lint against movement."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0, shot="She climbs the switchback with the needle in her fist",
            camera="medium close, tracking backwards ahead of her, her face held",
        )
    }
    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, [_chunk_stub(3, "Walking the empty road tonight")])
    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# Re-score against the full render (issue #76).
#
# `_GAZE_AWAY_KEYWORDS`, `_BEHIND_CAMERA_KEYWORDS` and `_FOOT_LEVEL_KEYWORDS`
# above were all scored on a partial "Deathless" render, mid-render. The
# finished 80-chunk render (41 voiced chunks, corpus mean face presence
# 53.3%) is now available, and does not support all three as shipped:
#
#   | keyword set              | shipped n | avg face | vs rest |
#   |---------------------------|-----------|----------|---------|
#   | `_GAZE_AWAY_KEYWORDS`      | 9         | 40.7%    | 0.72x   |
#   | `_BEHIND_CAMERA_KEYWORDS`  | 2         | 41.7%    | 0.77x   |
#   | `_FOOT_LEVEL_KEYWORDS`     | 3         | 52.8%    | 0.99x   |
#   | `in profile` (not shipped) | 5         | 23.3%    | 0.41x   |
#
# Full numbers and per-chunk detail: docs/deathless-render-corpus.md Part 3.
# Foot-level's retirement is covered by the test just above this block;
# behind-camera is intentionally left unchanged (flagged in the docs for a
# future pass, out of scope for what issue #76 measured and asked for).
# --------------------------------------------------------------------------- #


def test_gaze_away_keywords_excluded_by_the_full_render_no_longer_warn(caplog):
    """`_GAZE_AWAY_KEYWORDS` was re-scored against all 41 voiced chunks of
    the finished render (issue #76), keeping only words whose one measured
    occurrence landed below the 53.3% corpus mean. "gaze drops", "gaze
    lifts" and "lifts her gaze" all measured ABOVE the mean (100%, 83% and
    92% face presence respectively) -- the gaze verb fired, but the camera
    clause in the same entry also named the face, and the camera won. All
    three are excluded now; this is the "gaze drops" case (chunk 42, 100%
    face, camera "close, straight on, static" in the real plan)."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="Ash drifts past her still form as her gaze drops to the fire below",
            camera="close, straight on, static",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


def test_a_voiced_chunk_in_profile_warns(caplog):
    """`in profile` ships now (issue #76), superseding the earlier n=3
    decision to leave it out. That decision was reversed by a larger sample,
    not silently changed: on the finished render it is the one predictor
    with NO counter-example across all 5 occurrences (18, 21, 37, 43, 59),
    worst case 8%, best case 42%, corpus mean 53.3%. It has to be checked
    before the close-framing check below it in this function, since every
    one of those 5 real camera values also contains "close" or "medium
    close" -- a profile shot IS a close shot, and a close shot in profile
    still loses the face far more often than a close shot generally does."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="She holds the ridge line as the wind pulls at her coat",
            camera="medium close in profile, the fire raking one side of her face",
        )
    }
    chunks = [_chunk_stub(3, "Walking the empty road tonight")]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert "chunk_id=3" in caplog.text
    assert "in profile" in caplog.text


def test_an_instrumental_chunk_may_be_in_profile(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="The ridge line holds steady against the wind",
            camera="medium close in profile, the fire raking the rock",
        )
    }
    chunks = [_chunk_stub(3, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_voiced_framing(plan, chunks)

    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# Referent-based companion lint (issue #72)
#
# Chunk 65 of a real "Deathless" render: Dianne is singing, `present =
# ["Dianne"]` -- so issue #64's own lint (one section up) is satisfied, since
# some cast member's name is in the prompt. But naming the singer again binds
# nothing new, and the shot's own "he"/"his" points at Jan, who is nowhere in
# any field. H3 invented a stranger and a viewer asked what the statue was.
# --------------------------------------------------------------------------- #


def test_the_real_chunk_65_case_warns(caplog):
    """Dianne singing, present=["Dianne"] (so #64 is silent), but "a few
    paces off...he keeps his gaze" names a second, unbound person."""
    plan = {
        65: ShotPlanEntry(
            chunk_id=65,
            start=422.958,
            shot=(
                "A few paces off, an ash-crusted figure stands motionless, staring down "
                "into the same valley, as he keeps his gaze fixed on the drop from the "
                "watch-post stone."
            ),
            present=("Dianne",),
        )
    }
    chunks = [_chunk_stub(65, "carrying the needle", characters=("Dianne",))]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert "chunk_id=65" in caplog.text
    assert "Dianne" in caplog.text
    assert "a few paces off" in caplog.text.lower()


def test_the_real_chunk_66_mirror_case_warns(caplog):
    """The reverse of 65: Jan's own "He" is unbound (present=["Dianne"]
    only), while the "figure...just behind him" is Dianne, correctly bound.
    Structurally the same defect -- a single bound name plus a shot line that
    insists on a second person."""
    plan = {
        66: ShotPlanEntry(
            chunk_id=66,
            start=428.0,
            shot=(
                "He stays fixed on the empty valley below, not so much as a glance "
                "toward the ash-crusted figure now standing just behind him at the "
                "stone."
            ),
            present=("Dianne",),
        )
    }
    chunks = [_chunk_stub(66, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert "chunk_id=66" in caplog.text
    assert "just behind him" in caplog.text.lower()


@pytest.mark.parametrize(
    "chunk_id,shot",
    [
        (
            42,
            "His gaze drops back to the valley below without a blink, the watch-post "
            "stone unyielding beneath him.",
        ),
        (
            43,
            "A glow over the plain dims and finally goes dark, the last light of a "
            "scattered battle fading low across the ground just beneath the watch-post, "
            "while he stands motionless, the plain grown hard to see below him.",
        ),
        (
            44,
            "The last banner shapes below dissolve into dust, their outlines breaking "
            "apart over the churned valley floor, as he keeps his post above the valley.",
        ),
    ],
)
def test_self_referential_pronouns_on_the_singer_do_not_warn(caplog, chunk_id, shot):
    """The real chunks 42-44: Jan sings, present=["Jan"], and every pronoun
    points back at Jan himself -- harmless, and #72 is explicit that these
    must stay quiet."""
    plan = {
        chunk_id: ShotPlanEntry(
            chunk_id=chunk_id, start=float(chunk_id), shot=shot, present=("Jan",)
        )
    }
    chunks = [_chunk_stub(chunk_id, "his ancient watch", characters=("Jan",))]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert caplog.records == []


@pytest.mark.parametrize(
    "chunk_id,shot",
    [
        (
            18,
            "Firelight throws her shadow sprawling up the rock face beside her as she "
            "presses on along the narrow trail.",
        ),
        (
            24,
            "She climbs up onto the mill's terrace, its roof beams collapsed and "
            "blackened close beside her, the mountain's slope rising steeply beyond "
            "into the grey air.",
        ),
        (
            30,
            "She picks her way across a long scree traverse, the sheer summit wall "
            "rising close beside her.",
        ),
    ],
)
def test_beside_her_naming_an_inanimate_object_does_not_warn(caplog, chunk_id, shot):
    """"Beside her" was one of issue #72's own suggested phrases and is
    deliberately NOT in the shipped keyword set: measured on the real plan it
    fires 3 for 3 on an inanimate noun (a shadow, roof beams, a rock face)
    with nobody there at all -- the same lesson issue #64's own test suite
    already recorded for this phrase."""
    plan = {chunk_id: ShotPlanEntry(chunk_id=chunk_id, start=float(chunk_id), shot=shot)}
    singer = ("Dianne",) if chunk_id == 18 else ()
    chunks = [_chunk_stub(chunk_id, "", instrumental=chunk_id != 18, characters=singer)]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert caplog.records == []


def test_two_bound_names_is_silent_even_with_a_distance_phrase(caplog):
    """Chunk 69 of the real plan: both Dianne (singing) and Jan (present) are
    already bound, so a pronoun has somewhere real to land even though it is
    ambiguous which -- both identities are in the prompt regardless."""
    plan = {
        69: ShotPlanEntry(
            chunk_id=69,
            start=460.0,
            shot=(
                "Her closed fist rises into the space between them, the needle inside "
                "it that once anchored to the island now lost beneath the horizon, as "
                "he stands motionless at the cliff's edge facing her."
            ),
            present=("Jan",),
        )
    }
    chunks = [_chunk_stub(69, "the needle", characters=("Dianne",))]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert caplog.records == []


def test_no_pronoun_at_all_is_silent(caplog):
    plan = {
        5: ShotPlanEntry(
            chunk_id=5, start=5.0,
            shot="The mountain path continues across the burning valley floor, smoke drifts past.",
        )
    }
    chunks = [_chunk_stub(5, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent(plan, chunks)

    assert caplog.records == []


def test_a_chunk_missing_from_the_plan_is_silent(caplog):
    with caplog.at_level(logging.WARNING):
        lint_unbound_companion_referent({}, [_chunk_stub(1, "")])
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Role-prohibition-vs-shot lint (issue #73)
#
# The corrected role "...never holding anything" was in force for an entire
# real render and a bass guitar still appeared in the same hands twice.
# Chunk 72's own line never uses the word "hold" -- "close his fingers over
# it" -- which is exactly why the shipped check is a small curated
# physical-contact synonym set, not a literal search for "hold".
# --------------------------------------------------------------------------- #

_JAN_PROHIBITED_ROLE = (
    "Kashay Besmertny the Deathless, an immortal watchman, weathered, patient and "
    "unhurried, never holding anything"
)


def test_the_real_chunk_72_case_warns(caplog):
    plan = {
        72: ShotPlanEntry(
            chunk_id=72,
            start=466.208,
            shot=(
                "Her fingers press the needle into his open palm and close his fingers "
                "over it, the ash-grey summit bare around them."
            ),
            present=("Jan",),
        )
    }
    chunks = [_chunk_stub(72, "the needle", characters=("Dianne",))]
    cast = {"Jan": CastMember(name="Jan", role=_JAN_PROHIBITED_ROLE, image=Path("jan.jpg"))}

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert "chunk_id=72" in caplog.text
    assert "Jan" in caplog.text
    assert "never holding anything" in caplog.text


@pytest.mark.parametrize(
    "chunk_id,shot,camera",
    [
        # "holds his stance"/"holds his ground" -- the plain verb "hold" used
        # idiomatically, never touching an object.
        (38, "He holds his stance on the same stretch of ancient stone he has never left, "
             "the valley's armies rotating through centuries in the haze below him.", None),
        (49, "The valley below empties out at last, the final stragglers of some ancient "
             "army dissolving into drifting dust, as he holds his stance at the rock's lip.",
             None),
        # "holding" as a CAMERA direction (cinematography jargon for framing,
        # not a hand) -- this is why `camera` is excluded from what this lint
        # reads at all.
        (37, "A fresh column grinds across the churned valley floor below, engines and "
             "dust replacing the line that broke there, while he remains a still figure "
             "on the ridge above.", "medium close in profile, holding steady"),
    ],
)
def test_ordinary_uses_of_hold_do_not_warn(caplog, chunk_id, shot, camera):
    plan = {
        chunk_id: ShotPlanEntry(
            chunk_id=chunk_id, start=float(chunk_id), shot=shot, camera=camera,
            present=("Jan",),
        )
    }
    chunks = [_chunk_stub(chunk_id, "", instrumental=True)]
    cast = {"Jan": CastMember(name="Jan", role=_JAN_PROHIBITED_ROLE, image=Path("jan.jpg"))}

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert caplog.records == []


def test_the_prohibited_member_not_present_is_silent(caplog):
    """The same shot line as chunk 72, but Jan is not bound to this chunk at
    all -- nothing to contradict his role with."""
    plan = {
        72: ShotPlanEntry(
            chunk_id=72,
            start=466.208,
            shot="Her fingers press the needle into her own palm and close her fingers over it.",
        )
    }
    chunks = [_chunk_stub(72, "the needle", characters=("Dianne",))]
    cast = {"Jan": CastMember(name="Jan", role=_JAN_PROHIBITED_ROLE, image=Path("jan.jpg"))}

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert caplog.records == []


def test_a_role_with_no_prohibition_never_warns(caplog):
    plan = {
        72: ShotPlanEntry(
            chunk_id=72,
            start=466.208,
            shot="Her fingers press the needle into his open palm and close his fingers over it.",
            present=("Jan",),
        )
    }
    chunks = [_chunk_stub(72, "the needle", characters=("Dianne",))]
    cast = {
        "Jan": CastMember(
            name="Jan",
            role="Kashay Besmertny the Deathless, an immortal watchman, weathered and unhurried",
            image=Path("jan.jpg"),
        )
    }

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert caplog.records == []


def test_a_literal_noun_prohibition_still_catches_a_literal_echo(caplog):
    """The general (non-"hold") path: a plain word right after the trigger,
    searched for literally, singularized. This is the honestly narrow case
    this lint can actually generalise to."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3,
            start=3.0,
            shot="He tunes the strings of a battered old instrument, ignoring the war below.",
        )
    }
    chunks = [_chunk_stub(3, "", characters=("Jan",))]
    cast = {
        "Jan": CastMember(
            name="Jan",
            role="An immortal watchman, weathered and unhurried, no instruments allowed",
            image=Path("jan.jpg"),
        )
    }

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert "chunk_id=3" in caplog.text
    assert "instrument" in caplog.text.lower()


def test_a_chunk_missing_from_the_plan_is_silent_for_the_prohibition_lint(caplog):
    cast = {"Jan": CastMember(name="Jan", role=_JAN_PROHIBITED_ROLE, image=Path("jan.jpg"))}
    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction({}, [_chunk_stub(1, "", characters=("Jan",))], cast)
    assert caplog.records == []


def test_a_chunk_nobody_is_bound_to_is_silent(caplog):
    """No `chunk.characters` and no `present` at all -- nothing to check any
    role against, whatever the cast says."""
    plan = {
        9: ShotPlanEntry(
            chunk_id=9, start=9.0,
            shot="He holds nothing, an empty stretch of watch-post stone in the wind.",
        )
    }
    chunks = [_chunk_stub(9, "", instrumental=True)]
    cast = {"Jan": CastMember(name="Jan", role=_JAN_PROHIBITED_ROLE, image=Path("jan.jpg"))}

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert caplog.records == []


def test_a_role_with_only_stopword_prohibitions_is_silent(caplog):
    """`_extract_prohibited_terms` must skip a trigger word followed only by
    a stopword ("without a care") rather than treat "a" as the forbidden
    term -- exercised here via a role with no OTHER prohibition to fall back
    on, so this also proves an empty extraction stays silent."""
    plan = {
        72: ShotPlanEntry(
            chunk_id=72,
            start=466.208,
            shot="Her fingers press the needle into his open palm and close his fingers over it.",
            present=("Jan",),
        )
    }
    chunks = [_chunk_stub(72, "", characters=("Dianne",))]
    cast = {
        "Jan": CastMember(
            name="Jan",
            role="An immortal watchman who moves without a care, unhurried",
            image=Path("jan.jpg"),
        )
    }

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, cast)

    assert caplog.records == []


def test_an_unknown_cast_member_in_present_is_silent_not_a_crash(caplog):
    """`present` names are validated against the cast elsewhere (prompting's
    UnknownCastMemberError); this lint must not itself explode on a name the
    `cast` mapping it was given does not have."""
    plan = {
        72: ShotPlanEntry(
            chunk_id=72, start=466.208, shot="Something happens.", present=("Nobody",),
        )
    }
    chunks = [_chunk_stub(72, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_role_prohibition_contradiction(plan, chunks, {})

    assert caplog.records == []


# --------------------------------------------------------------------------- #
# `location` field (issue #78)
#
# `setting` (#32) anchors the world's contents; it says nothing about where
# each character is inside that world at a given moment. `location` is the
# per-chunk answer -- a tag drawn from a small enumerated set the concept
# stage defines, assigned by the beats stage (issue #78's authoring-side
# half) and re-emitted here from the frozen timeline like every other
# anchor. Purely a checkable field: it is never composed into a prompt (that
# would need `prompting.py`, outside this module's reach), only used by the
# two lints below.
# --------------------------------------------------------------------------- #


def test_location_is_none_by_default(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
    )
    assert load_shot_plan(path)[0].location is None


def test_location_is_read_from_the_entry(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\n'
        'location = "switchback"\n',
    )
    assert load_shot_plan(path)[0].location == "switchback"


def test_a_non_string_location_is_rejected(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\nlocation = 5\n'
    )
    with pytest.raises(ShotPlanError):
        load_shot_plan(path)


def test_a_blank_location_is_treated_as_unset(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\nlocation = "   "\n'
    )
    assert load_shot_plan(path)[0].location is None


def test_resolve_location_returns_the_authored_tag(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "She walks"\nlocation = "mill"\n',
    )
    plan = load_shot_plan(path)

    assert resolve_location(plan, _chunk(0, 0.0, 5.167)) == "mill"


def test_resolve_location_applies_even_when_shot_is_blank(tmp_path):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = ""\nlocation = "valley floor"\n'
    )
    plan = load_shot_plan(path)

    assert resolve_shot(plan, _chunk(0, 0.0, 5.167)) is None
    assert resolve_location(plan, _chunk(0, 0.0, 5.167)) == "valley floor"


def test_resolve_location_with_no_entry_returns_none(tmp_path, caplog):
    path = _write_plan(
        tmp_path, '[[shot]]\nchunk_id = 0\nstart = 0.0\nshot = "only chunk zero"\n'
    )
    plan = load_shot_plan(path)

    with caplog.at_level(logging.WARNING):
        assert resolve_location(plan, _chunk(7, 40.0, 46.0)) is None


def test_resolve_location_with_no_plan_returns_none():
    assert resolve_location(None, _chunk(0, 0.0, 5.167)) is None


def test_resolve_location_raises_on_drift_same_as_resolve_shot(tmp_path):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 3\nstart = 21.17\nshot = "direction"\nlocation = "mill"\n',
    )
    plan = load_shot_plan(path)

    with pytest.raises(ShotPlanDriftError):
        resolve_location(plan, _chunk(3, 21.17 + TOLERANCE_EXCEEDED, 30.0))


# --------------------------------------------------------------------------- #
# Landmark position-contradiction lint (issue #78)
#
# The defect: chunk 7 of the real "Deathless" plan (0:54) puts "Volokov's
# ruined mill and its slow-turning sails standing in the valley below";
# chunk 13 (1:32), 37.4s later with nothing but continued climbing in
# between, has "her palm trailing along its worn wood grain" as she crosses
# "the mill's turning wheel below the watch-post" -- at shoulder height. She
# cannot be climbing away from the mill and touching it a minute later.
#
# This is deliberately narrow, and the module docstring/CLAUDE.md record why:
# an early version matched any repeated word plus a bare "below" against any
# other occurrence of it anywhere in the plan and fired on ~30 pairs across a
# third of a real 80-chunk plan, almost all false -- "below" and "at shoulder
# height" routinely describe two DIFFERENT nouns in the same sentence (the
# valley below vs. her fingertips guiding a needle), which a keyword search
# cannot tell apart. What survives is: (a) candidate "landmark" words are
# restricted to content words that also appear in `setting` -- the vocabulary
# of things the run has already named as fixed and significant, not any
# repeated word; (b) FAR/NEAR keyword sets are curated down to phrases
# unambiguous enough that they are very rarely about some OTHER noun in the
# same sentence (no bare "below", no generic "her palm"/"her fingertips");
# and (c) the two mentions must fall within
# `LANDMARK_CONTRADICTION_WINDOW_SECONDS` of each other in song time, since a
# landmark legitimately looks far in one shot and close in another an hour
# of story time later -- that is the character closing the distance, not a
# contradiction. Measured against the real 80-chunk plan with these three
# restrictions: exactly 1 pair fires, chunks 7 and 13, and it is the true
# positive above.
# --------------------------------------------------------------------------- #

_DEATHLESS_SETTING = (
    "A blasted Slavic mountaintop above a ruined mill (Volokov's) and a "
    "battlefield valley, where time is smeared: medieval hosts, industrial "
    "armies, and nuclear glow all visible from the same watch-post, ending "
    "in a post-war, wind-still near-future where the mountain has eroded to "
    "a hill."
)


def test_the_real_chunk_7_and_13_mill_contradiction_warns(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 7\nstart = 54.583\n'
        'shot = "She climbs a rocky switchback path with the needle glinting tight in '
        "her closed fist, Volokov's ruined mill and its slow-turning sails standing in "
        'the valley below, the mountain\'s blasted summit rising ahead of her."\n\n'
        '[[shot]]\nchunk_id = 13\nstart = 91.958\n'
        'shot = "A sail vane sweeps low at shoulder height as it turns, her palm '
        "trailing along its worn wood grain as she crosses past the mill's turning "
        'wheel below the watch-post."\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=_DEATHLESS_SETTING)

    assert "chunk_id=7" in caplog.text
    assert "chunk_id=13" in caplog.text
    assert "mill" in caplog.text.lower()


def test_the_full_deathless_plan_fires_on_exactly_the_one_true_positive(tmp_path, caplog):
    """The measurement claim itself, not just the two-chunk excerpt: replay
    every `mill`-adjacent shot the real 80-chunk plan carries (chunks 2, 7,
    13, 24, 25, 26, 55, 67, 79) and confirm only the 7/13 pair fires."""
    entries = [
        (2, 8.0, "The mill's four sails turn steadily above the ruined slate roof "
                 "below, catching what pre-dawn light there is, the watch-post's "
                 "broken stone standing on the ridge beyond."),
        (7, 54.583, "She climbs a rocky switchback path with the needle glinting "
                    "tight in her closed fist, Volokov's ruined mill and its "
                    "slow-turning sails standing in the valley below, the "
                    "mountain's blasted summit rising ahead of her."),
        (13, 91.958, "A sail vane sweeps low at shoulder height as it turns, her "
                     "palm trailing along its worn wood grain as she crosses past "
                     "the mill's turning wheel below the watch-post."),
        (24, 157.0, "She climbs up onto the mill's terrace, its roof beams "
                    "collapsed and blackened close beside her, the mountain's "
                    "slope rising steeply beyond into the grey air."),
        (25, 164.0, "Volokov's weathered mill turns its sails a perceptible notch "
                    "slower overhead, timber wheel groaning beneath them, as she "
                    "climbs on past its shadow toward the ridge."),
        (26, 171.0, "She looks up at the lone watch-post stone, still perched high "
                    "above the ruined mill on the ridge she has yet to climb."),
        (55, 360.0, "The mill's four sails stand locked motionless against the "
                    "sky, each vane furred thick with ice, the wheel beneath "
                    "rusted still in its housing."),
        (67, 430.0, "The valley below lies motionless and silent, no armies, no "
                    "dust, not even wind in the ash, the mill's sails hanging "
                    "still over the ruin below, as she listens from the mountain "
                    "path, breath held."),
        (79, 500.0, "Flat daylight lies across the silent valley and the dead, "
                    "sail-still mill below the hill, the whole battlefield "
                    "motionless under the open sky."),
    ]
    body = "\n\n".join(
        f'[[shot]]\nchunk_id = {cid}\nstart = {start}\nshot = {shot!r}\n'
        for cid, start, shot in entries
    )
    path = _write_plan(tmp_path, body)

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=_DEATHLESS_SETTING)

    fired_pairs = {
        (r.args[1], r.args[2])
        for r in caplog.records
        if "contradictory distances" in r.getMessage()
    }
    assert fired_pairs == {(7, 13)}


def test_landmark_contradiction_is_silent_beyond_the_time_window(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 7\nstart = 54.583\n'
        'shot = "Volokov\'s ruined mill stands in the valley below."\n\n'
        '[[shot]]\nchunk_id = 24\nstart = 300.0\n'
        'shot = "She climbs up onto the mill\'s terrace, close beside her."\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=_DEATHLESS_SETTING)

    assert "contradictory distances" not in caplog.text


def test_landmark_contradiction_is_silent_with_no_setting(tmp_path, caplog):
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 7\nstart = 54.583\n'
        'shot = "Volokov\'s ruined mill stands in the valley below."\n\n'
        '[[shot]]\nchunk_id = 13\nstart = 91.958\n'
        'shot = "Her palm trailing along the mill\'s sail at shoulder height."\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "contradictory distances" not in caplog.text


def test_landmark_contradiction_is_silent_when_only_one_side_is_curated_far_or_near(
    tmp_path, caplog
):
    """Both chunks name the mill, both are close in time, but neither carries
    a curated FAR or NEAR phrase -- ordinary variation in wording, not a
    measured contradiction signal."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 7\nstart = 54.583\n'
        'shot = "Volokov\'s ruined mill turns its old sails."\n\n'
        '[[shot]]\nchunk_id = 13\nstart = 91.958\n'
        'shot = "The mill\'s sails groan as she passes."\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=_DEATHLESS_SETTING)

    assert "contradictory distances" not in caplog.text


def test_landmark_contradiction_ignores_a_word_not_in_the_setting(tmp_path, caplog):
    """"below"/"at shoulder height" on a repeated word that is NOT part of
    `setting`'s own vocabulary must not be treated as a landmark -- the
    restriction that keeps this lint from re-exploding into the noisy
    any-repeated-word version measured and rejected above."""
    path = _write_plan(
        tmp_path,
        '[[shot]]\nchunk_id = 7\nstart = 54.583\n'
        'shot = "Her fingertips guide a needle far below the ridge line."\n\n'
        '[[shot]]\nchunk_id = 13\nstart = 91.958\n'
        'shot = "Her fingertips press the needle, trailing along its point."\n',
    )

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path, setting=_DEATHLESS_SETTING)

    assert "contradictory distances" not in caplog.text


# --------------------------------------------------------------------------- #
# present-vs-location mismatch lint (issue #78)
#
# `present` (issue #59) stages a companion without them singing; nothing
# checked whether the run even knows the two characters are in the same
# place. Chunk 7 of the real plan sets `present = ["Jan"]` on a hillside
# scene that never mentions him -- the "hill where Jan is" a viewer's first
# note complained about. This lint operates on the STRUCTURED `location` tag
# alone, never on prose: it compares `entry.location` against the set of
# locations a name's own singing chunks establish, and warns only on an
# outright contradiction, never on the (very common) case of a companion who
# simply has no singing chunk of their own to compare against.
# --------------------------------------------------------------------------- #


def test_present_at_a_location_that_contradicts_their_own_singing_chunk_warns(caplog):
    plan = {
        7: ShotPlanEntry(
            chunk_id=7, start=54.583, shot="She climbs alone.",
            location="valley approach", present=("Jan",),
        ),
        40: ShotPlanEntry(
            chunk_id=40, start=250.0, shot="He keeps his watch.",
            location="watch-post",
        ),
    }
    chunks = [
        _chunk_stub(7, "", characters=("Dianne",)),
        _chunk_stub(40, "his ancient watch", characters=("Jan",)),
    ]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert "chunk_id=7" in caplog.text
    assert "Jan" in caplog.text
    assert "valley approach" in caplog.text
    assert "watch-post" in caplog.text


def test_present_at_the_same_location_as_their_own_singing_chunk_is_silent(caplog):
    plan = {
        24: ShotPlanEntry(
            chunk_id=24, start=157.0, shot="She climbs onto the terrace.",
            location="mill", present=("Jan",),
        ),
        25: ShotPlanEntry(
            chunk_id=25, start=164.0, shot="He watches the sails turn.",
            location="mill",
        ),
    }
    chunks = [
        _chunk_stub(24, "", characters=("Dianne",)),
        _chunk_stub(25, "", characters=("Jan",)),
    ]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert caplog.records == []


def test_present_companion_with_no_singing_chunk_of_their_own_is_silent(caplog):
    """The common case: a companion who never sings solo has no established
    location to contradict -- silence here is deliberate, not a gap. Warning
    on absence rather than contradiction would fire on nearly every use of
    `present` for a non-singing character."""
    plan = {
        7: ShotPlanEntry(
            chunk_id=7, start=54.583, shot="She climbs alone.",
            location="valley approach", present=("Jan",),
        ),
    }
    chunks = [_chunk_stub(7, "", characters=("Dianne",))]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert caplog.records == []


def test_present_location_mismatch_is_silent_when_location_unset(caplog):
    plan = {
        7: ShotPlanEntry(chunk_id=7, start=54.583, shot="She climbs alone.", present=("Jan",)),
        40: ShotPlanEntry(
            chunk_id=40, start=250.0, shot="He keeps his watch.", location="watch-post"
        ),
    }
    chunks = [
        _chunk_stub(7, "", characters=("Dianne",)),
        _chunk_stub(40, "", characters=("Jan",)),
    ]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert caplog.records == []


def test_present_location_mismatch_is_silent_with_no_present(caplog):
    plan = {
        7: ShotPlanEntry(chunk_id=7, start=54.583, shot="She climbs alone.", location="valley"),
    }
    chunks = [_chunk_stub(7, "", characters=("Dianne",))]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert caplog.records == []


def test_present_location_mismatch_on_the_real_80_chunk_plan_is_silent(caplog):
    """The real "Deathless" plan predates `location` entirely, so every entry
    resolves it to `None` -- this lint must be silent, exactly like every
    other optional-field lint in this module, not fail closed on a field
    nothing has populated yet."""
    plan = {
        7: ShotPlanEntry(
            chunk_id=7, start=54.583, shot="She climbs a rocky switchback path.",
            present=("Jan",),
        ),
        8: ShotPlanEntry(chunk_id=8, start=61.875, shot="She halts on the trail."),
        9: ShotPlanEntry(chunk_id=9, start=67.042, shot="She looks back down."),
    }
    chunks = [
        _chunk_stub(7, "", characters=("Dianne",)),
        _chunk_stub(8, "his ancient watch", characters=("Jan",)),
        _chunk_stub(9, "", characters=("Dianne",)),
    ]

    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch(plan, chunks)

    assert caplog.records == []


def test_present_location_mismatch_ignores_a_chunk_missing_from_the_plan(caplog):
    with caplog.at_level(logging.WARNING):
        lint_present_location_mismatch({}, [_chunk_stub(1, "", characters=("Jan",))])
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Instrumental focus mismatch: the general shape of the chunk 29 bug (issue
# #82). `chunk.characters` is empty on an instrumental chunk, so the render
# falls back to `default_lead_vocalist` and composes THEM as the focus, even
# when the shot line is grammatically, entirely about somebody named in
# `present` instead. `subject` is the fix an author reaches for; this lint
# notices a plan that needed it and does not have it, before a render finds
# out.
#
# Which name a pronoun "belongs to" is measured from THIS plan's own
# solo-voiced shot lines, never guessed from the name (issue #72 already
# rejected that): on a solo voiced chunk the sentence's subject is already
# the singer (#58, #59, #60), so a solo singer's own third-person pronouns
# are self-reference by construction.
# --------------------------------------------------------------------------- #


def test_instrumental_focus_mismatch_warns_when_the_line_is_entirely_about_present(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="He keeps his own steady watch, his rifle close at hand.",
        ),
        29: ShotPlanEntry(
            chunk_id=29, start=29.0,
            shot="His boot settles into a hollow already worn into the stone.",
            present=("Jan",),
        ),
    }
    chunks = [
        _chunk_stub(3, "a lyric", characters=("Jan",)),
        _chunk_stub(29, "", instrumental=True),
    ]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert "chunk_id=29" in caplog.text
    assert "Jan" in caplog.text


def test_instrumental_focus_mismatch_is_silent_when_the_line_also_names_the_default(caplog):
    """Chunk 70 of the real plan: 'She steps up onto the ridge beside him,
    ...his eyes stay fixed...' -- Jan is present and named by a pronoun, but
    so is Dianne (the default focus), so the composed focus has real textual
    ground and nothing is wrong."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="He keeps his own steady watch, his rifle close at hand.",
        ),
        5: ShotPlanEntry(
            chunk_id=5, start=5.0, shot="She waits there too, she never leaves.",
        ),
        70: ShotPlanEntry(
            chunk_id=70, start=70.0,
            shot="She steps up onto the ridge beside him, his eyes stay fixed below.",
            present=("Jan",),
        ),
    }
    chunks = [
        _chunk_stub(3, "a lyric", characters=("Jan",)),
        _chunk_stub(5, "a lyric", characters=("Dianne",)),
        _chunk_stub(70, "", instrumental=True),
    ]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert caplog.records == []


def test_instrumental_focus_mismatch_is_silent_with_no_pronoun_evidence_yet(caplog):
    """No solo-voiced chunk anywhere in the plan establishes which name owns
    which pronoun -- silent on absence of evidence, the same discipline every
    other lint in this module follows, rather than firing on a guess."""
    plan = {
        29: ShotPlanEntry(
            chunk_id=29, start=29.0,
            shot="His boot settles into a hollow already worn into the stone.",
            present=("Jan",),
        ),
    }
    chunks = [_chunk_stub(29, "", instrumental=True)]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert caplog.records == []


def test_instrumental_focus_mismatch_is_silent_when_subject_is_already_set(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="He keeps his own steady watch, his rifle close at hand.",
        ),
        29: ShotPlanEntry(
            chunk_id=29, start=29.0,
            shot="His boot settles into a hollow already worn into the stone.",
            present=("Jan",), subject="Jan",
        ),
    }
    chunks = [
        _chunk_stub(3, "a lyric", characters=("Jan",)),
        _chunk_stub(29, "", instrumental=True),
    ]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert caplog.records == []


def test_instrumental_focus_mismatch_is_silent_with_no_present(caplog):
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="He keeps his own steady watch, his rifle close at hand.",
        ),
        29: ShotPlanEntry(
            chunk_id=29, start=29.0,
            shot="His boot settles into a hollow already worn into the stone.",
        ),
    }
    chunks = [
        _chunk_stub(3, "a lyric", characters=("Jan",)),
        _chunk_stub(29, "", instrumental=True),
    ]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert caplog.records == []


def test_instrumental_focus_mismatch_is_silent_on_a_voiced_chunk(caplog):
    """This lint's whole premise is the instrumental fallback -- a voiced
    chunk's focus is the singer, not a config default, so there is nothing
    for `subject` to fix there in the first place (that case is
    `lint_subject_on_voiced_chunk`'s, and it raises, not warns)."""
    plan = {
        3: ShotPlanEntry(
            chunk_id=3, start=3.0,
            shot="He keeps his own steady watch, his rifle close at hand.",
        ),
        29: ShotPlanEntry(
            chunk_id=29, start=29.0,
            shot="His boot settles into a hollow already worn into the stone.",
            present=("Jan",),
        ),
    }
    chunks = [
        _chunk_stub(3, "a lyric", characters=("Jan",)),
        _chunk_stub(29, "a lyric", characters=("Dianne",), instrumental=False),
    ]

    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(plan, chunks, "Dianne")

    assert caplog.records == []


def test_instrumental_focus_mismatch_ignores_a_chunk_missing_from_the_plan(caplog):
    with caplog.at_level(logging.WARNING):
        lint_instrumental_focus_mismatch(
            {}, [_chunk_stub(29, "", instrumental=True)], "Dianne"
        )
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Issue #87: the #37 lint's precision, and what the warning round pays for it.
#
# Measured on "Deathless" shot_plan_v6.toml: 21 of the plan's 45 advisory
# lints were `lint_shots_against_lyrics` firing on function words -- 'upon',
# 'alway' (which is 'always' after _singularish), 'never', 'left', 'free',
# 'high', 'higher', 'little' -- and `write`'s one warning round then spent a
# model call rewriting approved prose to satisfy them, changing 37 of 80 shot
# lines. A precision problem in a lint became a content problem in the plan.
# --------------------------------------------------------------------------- #


def _lyric_chunk(chunk_id: int, text: str) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"/tmp/chunk_{chunk_id}.wav"),
        start=float(chunk_id) * 5.0,
        end=float(chunk_id) * 5.0 + 5.0,
        text=text,
        characters=("Dianne",),
        source_segment_indices=(chunk_id,),
        frame_count=124,
    )


def _lyric_plan(**shots: str) -> dict[int, ShotPlanEntry]:
    return {
        int(cid): ShotPlanEntry(chunk_id=int(cid), start=float(cid) * 5.0, shot=shot)
        for cid, shot in shots.items()
    }


@pytest.mark.parametrize("word", ["always", "upon", "never", "free", "little", "higher"])
def test_function_words_in_a_lyric_never_fire_the_staged_elsewhere_lint(caplog, word):
    """A function word is not a prop. 'Show <upon> on screen' is not a thing
    that can be done, and every one of these fired on a real generated plan."""
    plan = _lyric_plan(
        **{
            "0": f"She climbs the switchback {word} the bare rock face.",
            "9": "A needle glints in the stone.",
        }
    )
    chunks = [_lyric_chunk(9, f"High {word} a mountain")]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)
    assert f"names {word!r}" not in caplog.text


def test_always_does_not_reach_the_lint_as_the_non_word_alway(caplog):
    """`_singularish` strips a trailing 's' from any word over four letters,
    so 'always' became 'alway' and was reported as a missing object by that
    name. Filtering happens before stemming, so the stopword entry is enough."""
    plan = _lyric_plan(
        **{"0": "He stands always at the watch-post stone.", "9": "A bare ledge."}
    )
    chunks = [_lyric_chunk(9, "and he always will")]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)
    assert "alway" not in caplog.text


def test_a_real_prop_staged_elsewhere_still_fires(caplog):
    """The lint's whole reason for existing (issue #37) must survive the
    precision work: a printer sung here and staged only over there."""
    plan = _lyric_plan(
        **{
            "0": "A beige printer sits blinking on the sill.",
            "9": "She walks past a bare wall.",
        }
    )
    chunks = [_lyric_chunk(9, "I hoped your printer would explode")]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)
    assert "names 'printer'" in caplog.text


def test_stageable_nouns_restricts_the_lint_to_named_objects(caplog):
    """Issue #87: #69's `reading.nouns` is the list of concrete objects the
    lyrics actually put on the table, which is what this lint was always
    approximating. Given it, nothing outside it may fire."""
    plan = _lyric_plan(
        **{
            "0": "A needle glints in a seam of the stone, a banner burning beside it.",
            "9": "Bare grey water.",
        }
    )
    chunks = [_lyric_chunk(9, "the needle and the banner")]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks, stageable_nouns=("needle",))
    assert "names 'needle'" in caplog.text
    assert "names 'banner'" not in caplog.text


def test_stageable_nouns_empty_leaves_todays_behaviour_untouched(caplog):
    """Absent or empty means 'no vocabulary supplied' -- the same convention
    `locations` and `acts` use -- never 'nothing may fire'."""
    plan = _lyric_plan(
        **{
            "0": "A beige printer sits blinking on the sill.",
            "9": "She walks past a bare wall.",
        }
    )
    chunks = [_lyric_chunk(9, "I hoped your printer would explode")]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks, stageable_nouns=())
    assert "names 'printer'" in caplog.text


def test_stageable_noun_stems_normalises_multiword_phrases():
    """The concept writes phrases ("Volokov's mill", "mushroom glow /
    mushroom cloud"), the lint compares single stemmed words. One shared
    normaliser, because a second implementation drifts within a month."""
    stems = stageable_noun_stems(
        ["Volokov's mill", "mushroom glow / mushroom cloud", "armies"]
    )
    assert {"volokov", "mill", "mushroom", "glow", "cloud", "armie"} <= stems
    assert "with" not in stems and "" not in stems


def test_subject_binds_a_pronoun_for_the_unbound_companion_lint(caplog):
    """Issue #82 added `subject`; #64's lint only knew about `present`, so an
    instrumental chunk whose pronoun is bound by `subject` was reported as
    unbound. Measured on "Deathless" chunk 79."""
    plan = {
        79: ShotPlanEntry(
            chunk_id=79,
            start=508.75,
            shot="The hilltop stone stands hollowed smooth beside him, worn deep by centuries.",
            subject="Jan",
        )
    }
    chunks = [_chunk_stub(79, "", instrumental=True)]
    with caplog.at_level(logging.WARNING):
        lint_shots_against_lyrics(plan, chunks)
        shot_plan_module._lint_implied_companion_without_present(plan, Path("plan.toml"))
    assert "implies a second person" not in caplog.text
