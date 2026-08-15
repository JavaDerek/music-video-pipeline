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

from music_video_maker.contracts import AudioChunk
from music_video_maker.shot_plan import (
    ShotPlanDriftError,
    ShotPlanEntry,
    ShotPlanError,
    lint_camera_face_away_on_voiced_chunks,
    lint_shots_against_lyrics,
    load_shot_plan,
    render_shot_plan_skeleton,
    resolve_camera,
    resolve_shot,
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

def _chunk_stub(chunk_id: int, text: str, *, instrumental: bool = False):
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"chunk_{chunk_id}.wav"),
        start=float(chunk_id),
        end=float(chunk_id) + 5.0,
        text=text,
        is_instrumental=instrumental,
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
