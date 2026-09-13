"""Tests for the read-only pre-render review page (issue #36's "next slice,
specified", ``docs/design-web-ui.md``).

Two layers, matching how ``music_video_maker.review`` itself is split:

* **Golden-file tests** exercise the pure renderers (:func:`render_review_json`
  / :func:`render_review_html`) against a small, hand-built :class:`ReviewData`
  -- no alignment, no slicing, no shot plan file, so the fixture is exactly
  the data the design doc asks for (one voiced chunk, one instrumental, one
  with an alignment finding, one with a lint warning) and nothing about
  ``slice_audio``'s or ``align()``'s own behaviour can make this fixture
  drift. Set ``MVM_UPDATE_GOLDEN=1`` to regenerate the committed files under
  ``tests/fixtures/review/`` after a deliberate output-format change.
* **Wiring tests** exercise :func:`build_review` against the same offline
  ``Rig`` ``tests/test_cli.py`` uses for ``--prepare`` (imported, not
  copied -- see ``tests/test_cli_roadmap_wiring.py`` for the same pattern),
  with a real ``FakeAlignModel`` injected exactly as every other Stage 1
  caller's tests inject one. These assert on shape (a finding exists, a lint
  warning exists, an instrumental chunk exists), not on exact byte output --
  that precision belongs to the golden-file tests above.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from music_video_maker import cli
from music_video_maker.review import (
    AlignmentFindingView,
    ChunkReview,
    LintWarning,
    ReviewData,
    build_review,
    render_review_html,
    render_review_json,
)
from tests.test_cli import Rig, _anchors, _make_config_file, _prepare

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "review"
GOLDEN_JSON = FIXTURES_DIR / "review_golden.json"
GOLDEN_HTML = FIXTURES_DIR / "review_golden.html"


# --------------------------------------------------------------------------- #
# The golden fixture: hand-built, not run through alignment/slicing at all --
# see the module docstring for why.
# --------------------------------------------------------------------------- #


def _golden_review_data() -> ReviewData:
    voiced_chunk = ChunkReview(
        chunk_id=0,
        start=0.0,
        end=6.5,
        duration=6.5,
        frame_count=141,
        voiced=True,
        lyric="walking through the empty halls tonight",
        shot="wide shot of the corridor, her hand trailing the wall",
        camera="slow push in",
        location="hallway",
        present=(),
        subject=None,
        conditions=None,
    )
    instrumental_chunk = ChunkReview(
        chunk_id=1,
        start=6.5,
        end=8.0,
        duration=1.5,
        frame_count=124,
        voiced=False,
        lyric="",
        shot=None,
        camera=None,
        location=None,
        present=(),
        subject="Dianne",
        conditions="dust settling",
    )
    finding_chunk = ChunkReview(
        chunk_id=2,
        start=8.0,
        end=8.02,
        duration=0.02,
        frame_count=None,
        voiced=True,
        lyric="gone",
        # Deliberately includes a `<script>` tag: the design doc requires
        # every piece of user text (lyrics, shot lines) to be HTML-escaped,
        # and a shot line is exactly the kind of free text an author (or,
        # via mvm-author, a model) writes.
        shot='<script>alert("staged")</script> she vanishes mid-step',
        camera=None,
        location="hallway",
        present=(),
        subject=None,
        conditions=None,
        findings=(
            AlignmentFindingView(
                severity="CRITICAL",
                code="zero_length_segment",
                message=(
                    "segment 1 (8.000s -> 8.020s) is only 20ms long -- too short to "
                    "contain real audio"
                ),
                start=8.0,
                end=8.02,
                segment_index=1,
            ),
        ),
    )
    lint_warning = LintWarning(
        severity="warning",
        message=(
            'chunk_id=3: lyric names "printer", staged only in chunk(s) [5] -- consider '
            "staging it here too (or removing it from the lyric-vs-shot check)"
        ),
        chunk_id=3,
    )
    lint_chunk = ChunkReview(
        chunk_id=3,
        start=14.0,
        end=19.0,
        duration=5.0,
        frame_count=141,
        voiced=True,
        lyric="the printer sparks and dies behind her",
        shot="she walks on, not looking back",
        camera=None,
        location="hallway",
        present=(),
        subject=None,
        conditions=None,
        lint_warnings=(lint_warning,),
    )

    unattributed_warning = LintWarning(
        severity="warning",
        message="shot plan has 1 chunk(s) with no camera direction set",
        chunk_id=None,
    )

    return ReviewData(
        chunks=(voiced_chunk, instrumental_chunk, finding_chunk, lint_chunk),
        alignment_summary="Alignment quality: 4 segment(s), 1 critical, 0 warning finding(s)",
        alignment_finding_counts={"INFO": 0, "WARNING": 0, "CRITICAL": 1},
        lint_warnings=(lint_warning, unattributed_warning),
        plan_errors=(),
    )


def _write_golden(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_golden_json(tmp_path: Path):
    data = _golden_review_data()
    actual = render_review_json(data)

    if os.environ.get("MVM_UPDATE_GOLDEN"):
        _write_golden(GOLDEN_JSON, actual)
        pytest.skip("MVM_UPDATE_GOLDEN=1: wrote tests/fixtures/review/review_golden.json")

    expected = GOLDEN_JSON.read_text(encoding="utf-8")
    assert actual == expected, (
        "review JSON output changed -- if this is a deliberate format change, "
        "regenerate with: MVM_UPDATE_GOLDEN=1 pytest tests/test_review.py -k golden"
    )


def test_golden_json_round_trips_to_equivalent_data(tmp_path: Path):
    """The golden file is not just bytes to diff -- it must actually decode
    back into the same structure, so a future change to key names or nesting
    is caught even if some other edit happened to keep the byte diff small."""
    data = _golden_review_data()
    payload = json.loads(render_review_json(data))
    assert payload == data.to_dict()


def test_golden_html(tmp_path: Path):
    data = _golden_review_data()
    actual = render_review_html(data)

    if os.environ.get("MVM_UPDATE_GOLDEN"):
        _write_golden(GOLDEN_HTML, actual)
        pytest.skip("MVM_UPDATE_GOLDEN=1: wrote tests/fixtures/review/review_golden.html")

    expected = GOLDEN_HTML.read_text(encoding="utf-8")
    assert actual == expected, (
        "review HTML output changed -- if this is a deliberate format change, "
        "regenerate with: MVM_UPDATE_GOLDEN=1 pytest tests/test_review.py -k golden"
    )


def test_html_escapes_script_tag_in_shot_line():
    """The design doc's own acceptance bar: "escape ALL text (lyrics and
    shot lines are user text -- test that `<script>` in a shot line is
    escaped)". Independent of the golden file so this specific guarantee
    cannot be lost inside an otherwise-passing byte diff."""
    data = _golden_review_data()
    output = render_review_html(data)

    assert "<script>alert(" not in output
    assert "&lt;script&gt;" in output
    # The rest of that line is real content and must still be readable.
    assert "she vanishes mid-step" in output


def test_html_marks_critical_and_warning_distinguishably():
    data = _golden_review_data()
    output = render_review_html(data)

    assert "sev-critical" in output
    assert "sev-warning" in output
    assert "CRITICAL" in output
    assert "WARNING" in output


def test_json_has_no_timestamp_or_other_nondeterministic_field():
    """Design doc: "Make output deterministic (no timestamps unless
    injected, sorted keys)". A loose regex rather than a hardcoded key list,
    so a future field named `generated_at` etc. is caught even if this test
    is not updated in the same commit that adds it."""
    payload = render_review_json(_golden_review_data())
    for banned in ("generated_at", "timestamp", "run_id", "hostname"):
        assert banned not in payload


def test_json_is_byte_identical_across_two_builds():
    first = render_review_json(_golden_review_data())
    second = render_review_json(_golden_review_data())
    assert first == second


# --------------------------------------------------------------------------- #
# Wiring: build_review against the real Stage 1-2 functions, offline.
# --------------------------------------------------------------------------- #


def test_build_review_with_no_shot_plan(tmp_path: Path):
    rig = Rig(tmp_path)

    data = build_review(rig.config, align_model=rig.align_model)

    assert len(data.chunks) == 3  # DEFAULT_SEGMENT_SPECS, instrumental_coverage off
    assert all(chunk.shot is None for chunk in data.chunks)
    assert all(chunk.voiced for chunk in data.chunks)
    assert data.plan_errors == ()
    assert "Alignment quality" in data.alignment_summary


def test_build_review_finds_instrumental_and_voiced_chunks(tmp_path: Path):
    # A 10s lead-in before the first lyric is well past instrumental_coverage's
    # merge threshold, so it lands as its own instrumental chunk rather than
    # being absorbed into a neighbour's span (the small gaps in
    # DEFAULT_SEGMENT_SPECS are exactly that threshold and do get absorbed --
    # not this test's concern; see slicing's own tests for the merge rule).
    rig = Rig(
        tmp_path,
        instrumental_coverage=True,
        segment_specs=[
            ("walking through the empty halls tonight", 10.0, 16.5),
            ("nobody is watching nobody cares", 17.5, 22.0),
        ],
    )

    data = build_review(rig.config, align_model=rig.align_model)

    assert any(chunk.voiced for chunk in data.chunks)
    assert any(not chunk.voiced for chunk in data.chunks)
    # instrumental_coverage tiles from the very start of the track.
    assert data.chunks[0].start == 0.0
    assert not data.chunks[0].voiced


def test_build_review_surfaces_an_alignment_finding(tmp_path: Path):
    """A segment under alignment_quality.NEAR_ZERO_DURATION_S is a CRITICAL
    finding (issue #35) -- the review must show it on the chunk(s) whose span
    it touches, not just log it."""
    rig = Rig(
        tmp_path,
        segment_specs=[
            ("walking through the empty halls tonight", 0.0, 6.5),
            ("gone", 8.0, 8.02),
            ("the lights flicker but i do not mind", 14.0, 19.0),
        ],
    )

    data = build_review(rig.config, align_model=rig.align_model)

    assert "1 critical" in data.alignment_summary
    touching = [chunk for chunk in data.chunks if chunk.findings]
    assert touching, "the near-zero-duration segment's finding must land on some chunk"
    assert any(f.severity == "CRITICAL" for chunk in touching for f in chunk.findings)


def test_build_review_surfaces_shot_plan_lint_warnings(tmp_path: Path):
    """An unfilled shot line (``--prepare``'s own skeleton, before a human
    authors it) is exactly the kind of thing a reviewer needs called out
    before spending GPU time -- ``resolve_shot`` warns through the real
    loader, and the review must show it, not swallow it."""
    rig = Rig(tmp_path)
    skeleton_path = tmp_path / "shot_plan.toml"
    cli.prepare_shot_plan(
        rig.config,
        skeleton_path,
        source="run.toml",
        generated_at="2026-09-13",
        align_model=rig.align_model,
    )
    config = dc_replace(rig.config, shot_plan=skeleton_path)

    data = build_review(config, align_model=rig.align_model)

    assert data.plan_errors == ()
    assert data.lint_warnings, "every entry in the skeleton has a blank shot line"
    assert any(chunk.lint_warnings for chunk in data.chunks)
    assert all(chunk.shot is None for chunk in data.chunks)  # blank -> falls back


def test_build_review_reports_a_plan_that_fails_to_load(tmp_path: Path):
    rig = Rig(tmp_path)
    bad_plan = tmp_path / "shot_plan.toml"
    bad_plan.write_text("this is not [valid toml\n")
    config = dc_replace(rig.config, shot_plan=bad_plan)

    data = build_review(config, align_model=rig.align_model)

    assert data.plan_errors, "a plan that cannot be parsed must be reported, not silently skipped"
    assert len(data.chunks) == 3  # the timeline itself is unaffected


def test_build_review_reports_subject_on_voiced_chunk_as_a_plan_error(tmp_path: Path):
    """``lint_subject_on_voiced_chunk`` (issue #82) raises rather than warns
    -- a real render would refuse outright. The review must say so via
    ``plan_errors``, never crash, and never silently drop the rest of the
    review."""
    rig = Rig(tmp_path)
    skeleton_path = tmp_path / "shot_plan.toml"
    cli.prepare_shot_plan(
        rig.config,
        skeleton_path,
        source="run.toml",
        generated_at="2026-09-13",
        align_model=rig.align_model,
    )
    text = skeleton_path.read_text()
    # chunk_id=0 is the first voiced chunk in DEFAULT_SEGMENT_SPECS; `subject`
    # is legal only on an instrumental chunk (issue #82).
    text = text.replace(
        'shot = ""\n\n[[shot]]\nchunk_id = 1',
        'subject = "Dianne"\nshot = ""\n\n[[shot]]\nchunk_id = 1',
        1,
    )
    skeleton_path.write_text(text)
    config = dc_replace(rig.config, shot_plan=skeleton_path)

    data = build_review(config, align_model=rig.align_model)

    assert data.plan_errors
    assert len(data.chunks) == 3


def test_build_review_touches_no_comfyui(tmp_path: Path):
    """The design doc: "No server, no forms, no start button" -- ``build_review``
    runs Stages 1-2 only (alignment + slicing, which does write chunk audio
    stems to ``chunks_dir``, exactly like ``--prepare``) and never reaches
    ComfyUI, GPU custody, or Stage 4/5. Mirrors
    ``test_prepare_shot_plan_writes_a_skeleton_touching_no_comfyui`` in
    ``tests/test_cli.py``, which makes the same claim about ``--prepare``."""
    rig = Rig(tmp_path)

    build_review(rig.config, align_model=rig.align_model)

    assert rig.session.requests == []


# --------------------------------------------------------------------------- #
# Bug: build_review ignored config.shot_plan's own editorial lengths, so a
# plan with a `length_seconds` merging chunks got reviewed against the
# NATURAL (unmerged) timeline -- describing chunks a real render never
# emits, and reporting the resulting drift as plan_errors rather than the
# merged timeline the render actually produces. `run_pipeline` always slices
# with `shot_length_requests(plan)` from `config.shot_plan` directly; there
# is no separate "from_plan" concept at render time -- that flag exists only
# because `--prepare` typically runs *before* `config.shot_plan` exists, to
# preview a re-anchor against some other candidate plan.
# --------------------------------------------------------------------------- #


def _merged_plan_fixture(rig: Rig, tmp_path: Path) -> tuple[Path, list[tuple[int, float]]]:
    """A self-consistent shot plan -- anchored against the very timeline its
    own `length_seconds` produces (the "round trips onto its own anchors"
    property `test_prepare_from_plan_round_trips_onto_its_own_anchors` in
    ``tests/test_cli.py`` proves) -- with a long take on chunk index 1 that
    merges what would otherwise be two natural chunks into one.

    Returns the plan path and the merged timeline's own ``(chunk_id, start)``
    anchors, so a test can assert against them without re-deriving.
    """
    plain_anchors = _anchors(_prepare(rig, tmp_path / "plain.toml"))

    draft_path = tmp_path / "draft.toml"
    draft_path.write_text(
        "\n".join(
            f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "beat {cid}"'
            + ("\nlength_seconds = 12.0" if index == 1 else "")
            for index, (cid, start) in enumerate(plain_anchors)
        )
    )
    merged_anchors = _anchors(_prepare(rig, tmp_path / "reanchored.toml", from_plan=draft_path))
    assert len(merged_anchors) < len(plain_anchors), "fixture must actually merge chunks"

    final_path = tmp_path / "shot_plan.toml"
    final_path.write_text(
        "\n".join(
            f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "beat {cid} final"'
            + ("\nlength_seconds = 12.0" if index == 1 else "")
            for index, (cid, start) in enumerate(merged_anchors)
        )
    )
    return final_path, merged_anchors


def test_build_review_uses_config_shot_plan_lengths_with_no_from_plan(tmp_path: Path):
    """The defect: without the fix, ``build_review`` slices with no
    ``shot_lengths`` (natural timeline) while the plan is anchored against
    the merged one, so every chunk after the long take drifts."""
    rig = Rig(tmp_path, instrumental_coverage=True)
    plan_path, merged_anchors = _merged_plan_fixture(rig, tmp_path)
    config = dc_replace(rig.config, shot_plan=plan_path)

    data = build_review(config, align_model=rig.align_model)

    assert data.plan_errors == (), (
        "config.shot_plan's own length_seconds must be used to slice the timeline "
        "this review describes, the same way run_pipeline does"
    )
    assert len(data.chunks) == len(merged_anchors)
    assert [c.chunk_id for c in data.chunks] == [cid for cid, _ in merged_anchors]
    merged_chunk = data.chunks[1]
    assert merged_chunk.shot == f"beat {merged_chunk.chunk_id} final"
    assert merged_chunk.duration == pytest.approx(12.0, abs=0.5)


def test_build_review_explicit_from_plan_still_wins(tmp_path: Path):
    """An explicit ``--from-plan`` overrides ``config.shot_plan`` for the
    *lengths* -- the same precedence ``--prepare --from-plan`` already
    documents -- even though ``config.shot_plan`` is set to something else
    entirely."""
    rig = Rig(tmp_path, instrumental_coverage=True)
    plan_path, merged_anchors = _merged_plan_fixture(rig, tmp_path)
    natural_anchors = _anchors(_prepare(rig, tmp_path / "natural_source.toml"))
    # A second, natural-length plan `config.shot_plan` points at -- distinct
    # from `plan_path`, which is passed explicitly as `from_plan`.
    unmerged_plan_path = tmp_path / "unmerged.toml"
    unmerged_plan_path.write_text(
        "\n".join(
            f'[[shot]]\nchunk_id = {cid}\nstart = {start!r}\nshot = "natural {cid}"'
            for cid, start in natural_anchors
        )
    )
    config = dc_replace(rig.config, shot_plan=unmerged_plan_path)

    data = build_review(config, align_model=rig.align_model, from_plan=plan_path)

    # The timeline followed from_plan's lengths (merged), not
    # config.shot_plan's (natural) -- so config.shot_plan's own entries
    # (anchored at the natural timeline) now drift against it.
    assert len(data.chunks) == len(merged_anchors)
    assert data.plan_errors, (
        "config.shot_plan is anchored against the natural timeline; slicing with "
        "from_plan's merged lengths instead must surface that as drift, proving "
        "from_plan -- not config.shot_plan -- decided the timeline"
    )


def test_build_review_reports_unreadable_config_shot_plan_instead_of_crashing(tmp_path: Path):
    """``build_review`` now feeds ``config.shot_plan`` into
    ``prepare_timeline``'s own ``from_plan`` (to read its lengths), and that
    function does its own ``load_shot_plan`` call for them -- a malformed
    ``config.shot_plan`` must not crash there, before this function's own
    (already-tested) graceful load a few lines later ever gets a chance to
    catch it."""
    rig = Rig(tmp_path)
    bad_plan = tmp_path / "shot_plan.toml"
    bad_plan.write_text("this is not [valid toml\n")
    config = dc_replace(rig.config, shot_plan=bad_plan)

    data = build_review(config, align_model=rig.align_model)  # must not raise

    assert data.plan_errors
    assert len(data.chunks) == 3  # falls back to the natural timeline


# --------------------------------------------------------------------------- #
# CLI wiring: --review writes <stem>.html and <stem>.json
# --------------------------------------------------------------------------- #


def test_main_review_writes_html_and_json(tmp_path: Path, monkeypatch):
    """``cli.main``'s ``--review`` wiring, at the argv/exit-code layer -- the
    same layer ``test_main_prepare_flag_calls_prepare_shot_plan_and_returns_success``
    tests ``--prepare`` at (``tests/test_cli.py``). ``music_video_maker.review.build_review``
    is monkeypatched so this test is about the CLI's file-writing, not a
    second exercise of Stage 1-2 (that is ``test_build_review_*`` above and
    ``test_configured_alignment_model_size_reaches_align`` in
    ``tests/test_cli.py``)."""
    config_path = _make_config_file(tmp_path)
    out_stem = tmp_path / "out" / "review"
    data = _golden_review_data()
    captured = {}

    def fake_build_review(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return data

    monkeypatch.setattr("music_video_maker.review.build_review", fake_build_review)

    exit_code = cli.main(["--config", str(config_path), "--review", str(out_stem)])

    assert exit_code == cli.EXIT_SUCCESS
    html_path = out_stem.with_suffix(".html")
    json_path = out_stem.with_suffix(".json")
    assert html_path.read_text(encoding="utf-8") == render_review_html(data)
    assert json_path.read_text(encoding="utf-8") == render_review_json(data)


def test_main_review_accepts_a_path_without_html_suffix(tmp_path: Path, monkeypatch):
    config_path = _make_config_file(tmp_path)
    out_stem = tmp_path / "out" / "review.html"  # passed WITH .html this time
    monkeypatch.setattr(
        "music_video_maker.review.build_review", lambda config, **kwargs: _golden_review_data()
    )

    exit_code = cli.main(["--config", str(config_path), "--review", str(out_stem)])

    assert exit_code == cli.EXIT_SUCCESS
    assert (tmp_path / "out" / "review.html").is_file()
    assert (tmp_path / "out" / "review.json").is_file()


def test_main_review_returns_error_code_when_it_raises(tmp_path: Path, monkeypatch, caplog):
    config_path = _make_config_file(tmp_path)

    def fake_build_review(config, **kwargs):
        raise cli.PipelineError("no chunks")

    monkeypatch.setattr("music_video_maker.review.build_review", fake_build_review)

    with caplog.at_level(logging.ERROR):
        exit_code = cli.main(
            ["--config", str(config_path), "--review", str(tmp_path / "review")]
        )

    assert exit_code == cli.EXIT_ERROR
    assert any("Failed to build review" in r.message for r in caplog.records)


def test_main_review_and_prepare_together_is_an_error(tmp_path: Path):
    config_path = tmp_path / "run.toml"
    config_path.write_text("# not read -- rejected before load_config\n")

    exit_code = cli.main(
        ["--config", str(config_path), "--prepare", "--review", str(tmp_path / "out")]
    )

    assert exit_code == cli.EXIT_ERROR
