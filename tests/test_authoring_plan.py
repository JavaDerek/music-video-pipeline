"""Composing, checking and writing the generated shot plan
(issue #54 design sections 6 and 8).

The property that matters most here is that a generated plan **round-trips**:
``load_shot_plan`` plus ``resolve_shot`` over the very chunks it was
generated from raises nothing. That is what makes drift structurally
impossible for a generated plan rather than merely unlikely, and it is
asserted against the real loaders rather than a reimplementation of them --
which is also the design's own reason for running the real ones in
``check_plan``.

The two revision tiers are asymmetric on purpose and each has its own test,
because "warnings get exactly one round" is the kind of deliberate limit a
later reader will otherwise assume was an oversight.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

from music_video_maker import contracts
from music_video_maker.authoring.beats import Beat
from music_video_maker.authoring.plan import (
    MAX_ERROR_ROUNDS,
    PlanError,
    PlanIssue,
    Provenance,
    build_plan,
    check_plan,
    human_edited_chunks,
    lint_comments_for,
    objections_by_chunk,
    render_plan_toml,
    write_plan,
)
from music_video_maker.config import RunConfig
from music_video_maker.shot_plan import load_shot_plan, resolve_shot, shot_length_requests
from tests.harness.factories import make_cast_dict, write_silent_wav

PROVENANCE = Provenance(
    generated_by="mvm-author 0.1.0",
    generated_at="2026-08-13",
    models={"concept": "claude-fable-5", "beats": "claude-opus-5", "prose": "claude-sonnet-5"},
    hashes={"concept": "3f1c", "guide": "a90b", "lyrics": "77de", "skeleton": "c204"},
)


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
        setting="London, UK -- contemporary",
    )


def _chunks(texts: list[str], width: float = 5.875):
    return tuple(
        contracts.AudioChunk(
            chunk_id=i + 1,
            audio_file=Path(f"chunk_{i + 1:04d}.wav"),
            start=i * width,
            end=(i + 1) * width,
            text=text,
            is_instrumental=not text,
            frame_count=141,
        )
        for i, text in enumerate(texts)
    )


def _beat(
    chunk_id, *, group=1, role="transition", focus="subject", length=None, width=5.875,
    location="the street", act="the story", subject=None,
):
    start = (chunk_id - 1) * width
    return Beat(
        chunk_id=chunk_id,
        start=start,
        end=start + width,
        beat=f"beat {chunk_id}",
        beat_role=role,
        beat_group=group,
        location=location,
        act=act,
        focus=focus,
        length_seconds=length,
        subject=subject,
    )


LINES = {
    1: "The street empties out ahead of her, shutters coming down one after another",
    2: "A payphone receiver swings loose on its cord in the foreground",
    3: "Rain crosses the whole length of the pavement, gutters filling",
}

# A sung chunk needs a framing close enough to read a mouth, or
# `lint_voiced_framing` rightly objects -- these fixtures stand in for a
# *clean* plan, so they have to satisfy the rule like a real one would.
CAMERAS = {
    1: "medium on her as the shutters come down behind her",
    2: "close on the swinging receiver, her face held behind it",
    3: "medium on her, rain crossing the frame",
}


class _Reviser:
    """Stands in for ``prose.revise_prose``: records what it was asked to fix
    and returns whatever it was scripted to say."""

    def __init__(self, replies: list[dict[int, str]] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[dict[int, list[str]]] = []

    def __call__(self, shots, objections):
        self.calls.append({k: list(v) for k, v in objections.items()})
        reply = self._replies.pop(0) if self._replies else {}
        return type("R", (), {"shots": dict(reply), "driver_results": ()})()


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def test_a_generated_plan_round_trips_through_the_real_loaders(tmp_path):
    """The property that makes drift structurally impossible: load the plan
    back against the very chunks it was generated from, and nothing raises."""
    chunks = _chunks(["a first line", "", "a third line"])
    beats = (_beat(1), _beat(2), _beat(3))

    text = render_plan_toml(chunks, beats, LINES, provenance=PROVENANCE, camera=CAMERAS)
    path = tmp_path / "shot_plan.toml"
    path.write_text(text)

    plan = load_shot_plan(path, setting="London, UK", cast_names=("Dianne",))

    assert set(plan) == {1, 2, 3}
    for chunk in chunks:
        assert resolve_shot(plan, chunk) == LINES[chunk.chunk_id]


def test_anchors_come_from_the_chunks_not_from_the_beats(tmp_path):
    """A beat's own span is where it *was*; the chunk's is where the render
    will put it. Only one of those can be the drift anchor."""
    chunks = _chunks(["a line"])
    stale = (
        Beat(
            chunk_id=1, start=999.0, end=1005.0, beat="b", beat_role="plant", beat_group=1,
            location="the street",
        ),
    )

    text = render_plan_toml(chunks, stale, {1: LINES[1]}, provenance=PROVENANCE, camera=CAMERAS)
    path = tmp_path / "shot_plan.toml"
    path.write_text(text)

    assert resolve_shot(load_shot_plan(path), chunks[0]) == LINES[1]


def test_shot_text_with_quotes_and_backslashes_survives_verbatim(tmp_path):
    chunks = _chunks(["a line"])
    tricky = 'A sign reading "CLOSED" hangs at 45\\60 degrees, glass starring outward'

    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(chunks, (_beat(1),), {1: tricky}, provenance=PROVENANCE))

    assert load_shot_plan(path)[1].shot == tricky


def test_focus_and_length_come_from_the_beat(tmp_path):
    chunks = _chunks(["a line", ""])
    beats = (_beat(1, role="consequence", focus="action"), _beat(2, length=9.0))

    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(chunks, beats, LINES, provenance=PROVENANCE, camera=CAMERAS))
    plan = load_shot_plan(path)

    assert plan[1].subject_is_focus is False
    assert plan[2].length_seconds == 9.0
    assert [r.source_chunk_id for r in shot_length_requests(plan)] == [2]


def test_a_beat_with_no_opinion_emits_no_focus_or_length(tmp_path):
    """Absent means "no editorial opinion", never a default -- the same
    convention the hand-authored format already uses."""
    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE))

    text = path.read_text()
    assert "focus" not in text
    assert "length_seconds" not in text


def test_location_comes_from_the_beat(tmp_path):
    """Issue #78: `location` is re-emitted from the frozen beat, the same
    "anchors are copied from the chunks/beats" rule every other beat-derived
    field here follows -- never something a later stage (prose) could
    inject."""
    chunks = _chunks(["a line", ""])
    beats = (_beat(1, location="the mill"), _beat(2, location="the watch-post"))

    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(chunks, beats, LINES, provenance=PROVENANCE, camera=CAMERAS))
    plan = load_shot_plan(path)

    assert plan[1].location == "the mill"
    assert plan[2].location == "the watch-post"


def test_act_is_surfaced_in_the_beat_comment_line(tmp_path):
    """Issue #84: a human reviewing the plan as text has to be able to see
    the arc, so `act` rides beside `beat_role`/`beat_group` in the same
    `# beat: ...` comment line rather than a separate one."""
    chunks = _chunks(["a line", ""])
    beats = (
        _beat(1, role="plant", group=1, act="situation"),
        _beat(2, role="consequence", group=1, focus="action", act="resolution"),
    )

    text = render_plan_toml(chunks, beats, LINES, provenance=PROVENANCE, camera=CAMERAS)

    assert '[plant, group 1, act "situation"]' in text
    assert '[consequence, group 1, act "resolution"]' in text


def test_an_absent_act_is_omitted_from_the_comment_line(tmp_path):
    """A hand-built `Beat` or a pre-#84 persisted sheet has no `act` at all
    (``""`` -- the same "not authored" default `location` follows). The
    comment must not claim an act that was never assigned."""
    chunks = _chunks(["a line"])
    beats = (_beat(1, role="plant", group=1, act=""),)

    text = render_plan_toml(chunks, beats, LINES, provenance=PROVENANCE, camera=CAMERAS)

    assert "act" not in text.split("# beat:")[1].split("\n")[0]


def test_subject_is_emitted_next_to_present(tmp_path):
    """Issue #82: whose shot this is, re-emitted from the beat -- the same
    "anchors are copied from the chunks/beats" rule `location`/`act` follow."""
    chunks = _chunks(["", "a second line"])
    beats = (_beat(1, subject="Jan"), _beat(2))
    shots = {1: "An instrumental line about Jan", 2: LINES[2]}

    text = render_plan_toml(
        chunks, beats, shots, provenance=PROVENANCE, camera=CAMERAS, present={1: ["Jan"]},
    )

    assert 'subject = "Jan"' in text
    lines = text.split("[[shot]]")[1].splitlines()
    present_at = next(i for i, line in enumerate(lines) if line.startswith("present"))
    subject_at = next(i for i, line in enumerate(lines) if line.startswith("subject"))
    assert subject_at == present_at + 1


def test_a_beat_with_no_subject_emits_no_subject_key(tmp_path):
    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE))

    assert "subject" not in path.read_text()


def test_provenance_records_hashes_never_content(tmp_path):
    """Design section 8: a committed plan must not drag the whole lyric sheet
    into git -- and per #51, anything committed ships."""
    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE))

    provenance = tomllib.loads(path.read_text())["provenance"]
    assert provenance["prose_model"] == "claude-sonnet-5"
    assert provenance["lyrics_sha256"] == "77de"
    assert provenance["generated_at"] == "2026-08-13"


def test_every_entry_carries_its_own_provenance(tmp_path):
    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE))

    entry = tomllib.loads(path.read_text())["shot"][0]
    assert entry["generated_by"] == "prose"
    assert len(entry["content_sha256"]) == 64


def test_provenance_keys_do_not_trip_the_unknown_key_lint(tmp_path, caplog):
    path = tmp_path / "shot_plan.toml"
    path.write_text(render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE))

    with caplog.at_level(logging.WARNING):
        load_shot_plan(path)

    assert "nothing reads" not in caplog.text


def test_lint_comments_land_above_their_entry(tmp_path):
    chunks = _chunks(["x", "y"])
    text = render_plan_toml(
        chunks,
        (_beat(1), _beat(2)),
        LINES,
        provenance=PROVENANCE,
        lint_comments={2: ["the lyric names 'printer' but this shot does not show it"]},
    )

    lines = text.splitlines()
    comment_at = next(i for i, line in enumerate(lines) if line.startswith("# lint:"))
    assert "chunk_id = 2" in lines[comment_at + 2]
    # ...and it is still valid TOML that loads.
    (tmp_path / "p.toml").write_text(text)
    assert set(load_shot_plan(tmp_path / "p.toml")) == {1, 2}


# --------------------------------------------------------------------------- #
# check_plan, through the real loaders
# --------------------------------------------------------------------------- #


def test_a_clean_plan_checks_clean(tmp_path):
    chunks = _chunks(["a first line", "", "a third line"])
    text = render_plan_toml(
        chunks, (_beat(1), _beat(2), _beat(3)), LINES,
        provenance=PROVENANCE, camera=CAMERAS,
    )

    check = check_plan(text, _config(tmp_path), chunks)

    assert check.ok
    assert check.warnings == ()


def test_a_blank_shot_is_an_error_not_a_warning(tmp_path):
    """``resolve_shot`` only warns about a blank line, because a half-filled
    skeleton is a normal state to hand-author through. A *generated* plan has
    no such excuse: the chunk would fall back to narrative_concept and render
    as something deliberate-looking nobody authored."""
    chunks = _chunks(["a line", ""])
    text = render_plan_toml(chunks, (_beat(1), _beat(2)), {1: LINES[1]}, provenance=PROVENANCE)

    check = check_plan(text, _config(tmp_path), chunks)

    assert not check.ok
    assert [issue.chunk_id for issue in check.errors] == [2]


def test_a_real_lint_warning_is_captured_and_attributed(tmp_path):
    """The geography lint, run for real: "Central Park" under a London
    setting. Captured from the loader's own logger rather than re-derived."""
    chunks = _chunks(["a first line", "a second line"])
    shots = {
        1: "She crosses Central Park as the shutters come down behind her",
        2: LINES[2],
    }
    text = render_plan_toml(chunks, (_beat(1), _beat(2)), shots, provenance=PROVENANCE)

    check = check_plan(text, _config(tmp_path), chunks)

    assert check.ok  # a warning must never block
    assert any("central park" in issue.message.lower() for issue in check.warnings)
    assert 1 in {issue.chunk_id for issue in check.warnings}


# --------------------------------------------------------------------------- #
# Two lints wired into check_plan by this change (issue #78's hand-off): both
# were built and tested in isolation by an earlier pass but never reached the
# render's own loaders. "A generated plan is checked by the render's own
# loaders, never a copy of them" only holds if these are actually in the
# loop, so each gets a real trigger here, not a mock.
# --------------------------------------------------------------------------- #


def test_the_unbound_companion_referent_lint_is_wired_into_check_plan(tmp_path):
    """Issue #72: exactly one bound name (Dianne, via `present`), but the
    shot's own prose insists on a second, distinct person."""
    chunks = _chunks(["a first line"])
    shots = {
        1: (
            "A few paces off, an ash-crusted figure stands motionless, staring down "
            "into the valley, as he keeps his gaze fixed on the drop."
        ),
    }
    text = render_plan_toml(
        chunks, (_beat(1),), shots, provenance=PROVENANCE, present={1: ["Dianne"]},
    )

    check = check_plan(text, _config(tmp_path), chunks)

    assert check.ok  # a warning must never block
    assert any("issue #72" in issue.message for issue in check.warnings)


def test_the_role_prohibition_contradiction_lint_is_wired_into_check_plan(tmp_path):
    """Issue #73: a `role` written as a prohibition, contradicted by the shot
    line actually describing the forbidden thing."""
    config = _config(tmp_path)
    prohibited = replace(
        config.cast["Dianne"],
        role="Lead Vocalist, smiling constantly, oblivious, never holding anything",
    )
    config = replace(config, cast={**config.cast, "Dianne": prohibited})

    chunks = _chunks(["a first line"])
    shots = {1: "Her fingers close his fingers over the needle, ash drifting past."}
    text = render_plan_toml(
        chunks, (_beat(1),), shots, provenance=PROVENANCE, present={1: ["Dianne"]},
    )

    check = check_plan(text, config, chunks)

    assert check.ok  # a warning must never block
    assert any("issue #73" in issue.message for issue in check.warnings)


def test_the_present_location_mismatch_lint_is_wired_into_check_plan(tmp_path):
    """Issue #78: `present` stages a companion at a location that contradicts
    where their own singing chunks place them elsewhere in the same plan."""
    config = _config(tmp_path)
    chunks = (
        contracts.AudioChunk(
            chunk_id=1, audio_file=Path("c1.wav"), start=0.0, end=5.875,
            text="a first line", characters=("Dianne",),
        ),
        contracts.AudioChunk(
            chunk_id=2, audio_file=Path("c2.wav"), start=5.875, end=11.75,
            text="a second line", characters=("Rex",),
        ),
    )
    beats = (
        _beat(1, location="the valley approach"),
        _beat(2, location="the watch-post"),
    )
    shots = {1: LINES[1], 2: LINES[2]}
    text = render_plan_toml(
        chunks, beats, shots, provenance=PROVENANCE, present={1: ["Rex"]},
    )

    check = check_plan(text, config, chunks)

    assert check.ok  # a warning must never block
    assert any("issue #78" in issue.message for issue in check.warnings)


def test_malformed_toml_is_reported_as_an_unattributed_error(tmp_path):
    check = check_plan("this is not toml at all {{{", _config(tmp_path), _chunks(["x"]))

    assert not check.ok
    assert check.errors[0].chunk_id is None


def test_objections_ignore_unattributable_issues():
    issues = [
        PlanIssue(chunk_id=None, severity="error", message="whole file is broken"),
        PlanIssue(chunk_id=4, severity="error", message="blank"),
    ]

    assert objections_by_chunk(issues) == {4: ["blank"]}
    assert set(lint_comments_for(issues)) == {None, 4}


# --------------------------------------------------------------------------- #
# The two revision tiers
# --------------------------------------------------------------------------- #


def test_an_error_is_revised_and_the_fixed_plan_is_returned(tmp_path):
    chunks = _chunks(["a line", ""])
    beats = (_beat(1), _beat(2))
    reviser = _Reviser([{2: LINES[2]}])

    built = build_plan(
        _config(tmp_path),
        chunks,
        beats,
        {1: LINES[1]},  # chunk 2 blank -> an error
        provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
    )

    assert [sorted(call) for call in reviser.calls] == [[2]]
    assert built.shots[2] == LINES[2]
    assert check_plan(built.text, _config(tmp_path), chunks).ok


def test_an_unfixable_error_aborts_after_the_bound_and_writes_nothing(tmp_path):
    chunks = _chunks(["a line", ""])
    reviser = _Reviser([{}, {}, {}])

    with pytest.raises(PlanError, match="Nothing was written"):
        build_plan(
            _config(tmp_path),
            chunks,
            (_beat(1), _beat(2)),
            {1: LINES[1]},
            provenance=PROVENANCE,
            reviser=reviser,
            camera=CAMERAS,
        )

    assert len(reviser.calls) == MAX_ERROR_ROUNDS


def test_a_warning_gets_exactly_one_revision_round_then_is_annotated(tmp_path):
    """Design section 6: "one revision round, then stop. Do not grind them to
    zero." A loop that retried until the heuristics were silent would happily
    rewrite a correct shot to please one."""
    chunks = _chunks(["a first line", "a second line"])
    shots = {1: "She crosses Central Park as the shutters come down", 2: LINES[2]}
    # The reviser is unhelpful: it hands back a line that still trips the lint.
    reviser = _Reviser([{1: "She crosses Central Park again, slower this time"}])

    built = build_plan(
        _config(tmp_path), chunks, (_beat(1), _beat(2)), shots, provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
    )

    assert len(reviser.calls) == 1  # exactly one, never a second
    assert built.surviving_warnings
    assert "# lint:" in built.text
    assert "central park" in built.text.lower()


def test_a_revision_that_breaks_the_plan_is_discarded(tmp_path):
    """A rewrite that fixes a heuristic and breaks the loader is strictly
    worse than the warning it was chasing."""
    chunks = _chunks(["a first line", "a second line"])
    shots = {1: "She crosses Central Park as the shutters come down", 2: LINES[2]}
    reviser = _Reviser([{1: "   "}])  # blank -> an error

    built = build_plan(
        _config(tmp_path), chunks, (_beat(1), _beat(2)), shots, provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
    )

    # The prose is rolled back too, not just the verdict: keeping the new line
    # while reporting the old warnings would write a plan neither check passed.
    assert built.shots[1] == shots[1]
    assert check_plan(built.text, _config(tmp_path), chunks).ok
    assert built.surviving_warnings


def test_a_clean_plan_calls_the_reviser_not_at_all(tmp_path):
    chunks = _chunks(["a first line", "", "a third line"])
    reviser = _Reviser()

    built = build_plan(
        _config(tmp_path),
        chunks,
        (_beat(1), _beat(2), _beat(3)),
        LINES,
        provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
    )

    assert reviser.calls == []
    assert built.surviving_warnings == ()


def test_a_revision_leaves_every_out_of_scope_line_byte_identical(tmp_path):
    """The failure mode this guards: a retry that quietly rewords thirty
    approved shots destroys the value of the approval that came before it."""
    chunks = _chunks(["a line", "", "another line"])
    shots = {1: LINES[1], 3: LINES[3]}  # chunk 2 blank -> one error
    reviser = _Reviser([{2: LINES[2]}])

    built = build_plan(
        _config(tmp_path),
        chunks,
        (_beat(1), _beat(2), _beat(3)),
        shots,
        provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
    )

    assert built.shots[1] == LINES[1]
    assert built.shots[3] == LINES[3]


# --------------------------------------------------------------------------- #
# Writing, and detecting a human edit afterwards
# --------------------------------------------------------------------------- #


def test_write_plan_refuses_to_clobber_without_force(tmp_path):
    out = tmp_path / "shot_plan.toml"
    out.write_text("# hand-authored\n")

    with pytest.raises(PlanError):
        write_plan("# generated\n", out)

    assert out.read_text() == "# hand-authored\n"


def test_write_plan_force_overwrites(tmp_path):
    out = tmp_path / "shot_plan.toml"
    out.write_text("# hand-authored\n")

    write_plan("# generated\n", out, force=True)

    assert out.read_text() == "# generated\n"


def test_a_freshly_written_plan_reports_no_human_edits(tmp_path):
    out = tmp_path / "shot_plan.toml"
    write_plan(
        render_plan_toml(_chunks(["x", "y"]), (_beat(1), _beat(2)), LINES, provenance=PROVENANCE),
        out,
    )

    assert human_edited_chunks(out) == ()


def test_an_edited_shot_line_is_detected(tmp_path):
    """Design section 8: "was this shot written or generated" is exactly the
    question #34/#38/#39/#45 each needed answered after the fact."""
    out = tmp_path / "shot_plan.toml"
    write_plan(
        render_plan_toml(_chunks(["x", "y"]), (_beat(1), _beat(2)), LINES, provenance=PROVENANCE),
        out,
    )
    out.write_text(out.read_text().replace(LINES[2], "A human wrote this line instead"))

    assert human_edited_chunks(out) == (2,)


def test_a_hand_written_plan_with_no_hashes_reports_nothing(tmp_path):
    """Calling every line of a hand-authored plan "edited" would make the
    signal useless."""
    out = tmp_path / "shot_plan.toml"
    out.write_text('[[shot]]\nchunk_id = 1\nstart = 0.0\nshot = "hand written"\n')

    assert human_edited_chunks(out) == ()


def test_human_edited_chunks_wraps_an_unreadable_file(tmp_path):
    out = tmp_path / "shot_plan.toml"
    out.write_text("not toml {{{")

    with pytest.raises(PlanError):
        human_edited_chunks(out)


def test_extra_checks_reach_the_written_file_as_lint_comments(tmp_path):
    """The prose stage's own advisory prohibitions have to land in the file
    too -- design section 13 asks for surviving warnings "in context", and a
    warning only in a log is a warning nobody reads."""
    from music_video_maker.authoring.plan import PlanIssue as Issue

    chunks = _chunks(["a first line", "a second line"])
    reviser = _Reviser([{}])

    built = build_plan(
        _config(tmp_path),
        chunks,
        (_beat(1), _beat(2)),
        LINES,
        provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
        extra_checks=lambda shots: [
            Issue(chunk_id=2, severity="warning", message="carries its own camera direction")
        ],
    )

    assert "# lint: carries its own camera direction" in built.text
    assert [issue.chunk_id for issue in built.surviving_warnings] == [2]


def test_extra_checks_are_re_run_on_the_revised_text_not_the_original(tmp_path):
    """A revision can rewrite the very sentence a warning was about.
    Annotating the file with a complaint about a sentence that no longer
    exists is worse than not annotating it."""
    from music_video_maker.authoring.plan import PlanIssue as Issue

    chunks = _chunks(["a first line", "a second line"])
    reviser = _Reviser([{2: "A clean line with nothing to object to"}])
    seen: list[dict] = []

    def extra_checks(shots):
        seen.append(dict(shots))
        if shots.get(2) == LINES[2]:
            return [Issue(chunk_id=2, severity="warning", message="carries a camera direction")]
        return []

    built = build_plan(
        _config(tmp_path),
        chunks,
        (_beat(1), _beat(2)),
        LINES,
        provenance=PROVENANCE,
        reviser=reviser,
        camera=CAMERAS,
        extra_checks=extra_checks,
    )

    assert len(seen) > 1  # re-run, not carried over
    assert built.surviving_warnings == ()
    assert "carries a camera direction" not in built.text


# --------------------------------------------------------------------------- #
# Found by re-authoring a real song (issue #54 design section 12's honest test)
# --------------------------------------------------------------------------- #


def test_a_lint_comment_does_not_carry_the_temp_path_it_was_checked_in(tmp_path):
    """``check_plan`` writes the candidate to a temp file, so every loader
    message is prefixed "Shot plan /var/folders/.../tmpXXXX/shot_plan.toml:".
    Writing that into the plan puts a dead local path in a file that is meant
    to be read by a human and, per #51, may ship."""
    chunks = _chunks(["a first line", "a second line"])
    shots = {1: "She crosses Central Park as the shutters come down", 2: LINES[2]}

    check = check_plan(
        render_plan_toml(chunks, (_beat(1), _beat(2)), shots, provenance=PROVENANCE),
        _config(tmp_path),
        chunks,
    )

    assert check.warnings
    for issue in check.warnings:
        assert "/tmp" not in issue.message
        assert "shot_plan.toml" not in issue.message


def test_a_keyword_consequence_warning_is_dropped_when_the_beat_says_otherwise(tmp_path):
    """``_lint_consequence_focus`` guesses "this reads like a consequence"
    from a keyword list. The beat sheet *knows* -- and on the first real run
    it fired on four chunks the beats stage had already classified as plants
    and transitions, purely because they mentioned smoke.

    Worse than noise: those four went into the revision round, and the
    rewrite it produced put "She" back in the subject slot of a line that had
    correctly led with the object. That is exactly the failure design section
    6 predicts from grinding warnings -- "a loop that keeps retrying will
    happily rewrite a correct shot to please a heuristic" -- so the structural
    signal has to win before the objection is ever raised.
    """
    chunks = _chunks(["a line"])
    shots = {1: "Smoke leaks from an upper window while she walks past in the foreground"}
    plant = _beat(1, role="plant")

    check = check_plan(
        render_plan_toml(chunks, (plant,), shots, provenance=PROVENANCE),
        _config(tmp_path),
        chunks,
        beats=(plant,),
    )

    assert not any("consequence beat" in issue.message for issue in check.warnings)


def test_the_keyword_consequence_warning_survives_when_the_beat_agrees(tmp_path):
    """The lint is only redundant where the structure contradicts it. A beat
    the model *did* call a consequence, written without focus = "action", is
    a real finding and must still be reported."""
    chunks = _chunks(["a line"])
    shots = {1: "Smoke leaks from an upper window while she walks past in the foreground"}
    # A consequence whose focus was never set to "action" -- validate_beats
    # would reject that, but a hand-edited beats.json can still produce it.
    consequence = _beat(1, role="consequence", focus="subject")

    check = check_plan(
        render_plan_toml(chunks, (consequence,), shots, provenance=PROVENANCE),
        _config(tmp_path),
        chunks,
        beats=(consequence,),
    )

    assert any("consequence beat" in issue.message for issue in check.warnings)


def test_without_beats_the_keyword_warning_is_left_alone(tmp_path):
    """No structural signal means no grounds to suppress anything."""
    chunks = _chunks(["a line"])
    shots = {1: "Smoke leaks from an upper window while she walks past in the foreground"}

    check = check_plan(
        render_plan_toml(chunks, (_beat(1),), shots, provenance=PROVENANCE),
        _config(tmp_path),
        chunks,
    )

    assert any("consequence beat" in issue.message for issue in check.warnings)


# --------------------------------------------------------------------------- #
# `present` round-trips from the prose stage into the render's own loader (#59)
# --------------------------------------------------------------------------- #


def test_present_round_trips_through_the_real_loader(tmp_path):
    """The two halves of #59 have to agree on the key's name and shape, and
    the only way to prove that is to write it with one and read it with the
    other -- the same discipline `plan.check_plan` applies to everything else."""
    chunks = _chunks(["a line", ""])
    path = tmp_path / "shot_plan.toml"
    path.write_text(
        render_plan_toml(
            chunks,
            (_beat(1), _beat(2)),
            LINES,
            provenance=PROVENANCE,
            present={1: ["Jan"]},
        )
    )
    plan = load_shot_plan(path)

    assert plan[1].present == ("Jan",)
    assert plan[2].present == ()


def test_no_present_emits_no_key_at_all(tmp_path):
    """Same convention as focus/length: absent means "she is alone in shot"."""
    path = tmp_path / "shot_plan.toml"
    path.write_text(
        render_plan_toml(_chunks(["x"]), (_beat(1),), LINES, provenance=PROVENANCE)
    )

    assert "present" not in path.read_text()


# --------------------------------------------------------------------------- #
# `subject` round-trips through the real loader, and `check_plan` refuses a
# generated plan that sets it on a voiced chunk through the RENDER'S OWN rule
# (`lint_subject_on_voiced_chunk`), never a copy of it (issue #82).
# --------------------------------------------------------------------------- #


def test_subject_round_trips_through_the_real_loader(tmp_path):
    chunks = _chunks(["", "a second line"])
    path = tmp_path / "shot_plan.toml"
    path.write_text(
        render_plan_toml(
            chunks,
            (_beat(1, subject="Jan"), _beat(2)),
            {1: "An instrumental line about Jan", 2: LINES[2]},
            provenance=PROVENANCE,
            camera=CAMERAS,
            present={1: ["Jan"]},
        )
    )
    plan = load_shot_plan(path, cast_names=("Dianne", "Jan"))

    assert plan[1].subject == "Jan"
    assert plan[2].subject is None


def test_check_plan_refuses_a_generated_plan_with_a_voiced_chunk_subject(tmp_path):
    """The whole point of wiring `lint_subject_on_voiced_chunk` into
    `check_plan`: a plan that sets `subject` on a voiced chunk is refused by
    the render's own rule (issues #58, #59, #60), not a reimplementation of
    it -- and if this module ever drifted from the render's rule, this test
    would still pass on a stale copy while the render refused the same plan.
    """
    chunks = _chunks(["a voiced line", ""])
    beats = (_beat(1, subject="Dianne"), _beat(2))
    shots = {1: LINES[1], 2: "An instrumental line"}
    text = render_plan_toml(
        chunks, beats, shots, provenance=PROVENANCE, camera=CAMERAS,
    )

    check = check_plan(text, _config(tmp_path), chunks)

    assert not check.ok


def test_check_plan_accepts_a_subject_on_an_instrumental_chunk(tmp_path):
    chunks = _chunks(["", "a second line"])
    beats = (_beat(1, subject="Marcus"), _beat(2))
    shots = {1: "An instrumental line about Marcus", 2: LINES[2]}
    text = render_plan_toml(
        chunks, beats, shots, provenance=PROVENANCE, camera=CAMERAS, present={1: ["Marcus"]},
    )

    check = check_plan(text, _config(tmp_path), chunks)

    assert check.ok
