"""Tests for automatic vocalist diarization (issue #101).

Fully offline, like everything else here: no network, no GPU, no weights, and
``pyannote.audio`` is **not installed in this environment and must never need
to be**. Three injection points make that true rather than aspirational --
:type:`Diarizer` (audio in, speaker turns out, the same shape as ``align``'s
``model``), and ``build_pyannote_diarizer``'s ``importer`` and ``env``, which
let every one of the four refusal messages be asserted on a machine where the
package does not exist and no token is set.

The weights are gated and had not been accepted by any account when this was
written, so the one thing these tests cannot do is prove that the real pipeline
produces useful clusters on real singing. What they *can* prove, and do, is
that each way of not having the weights produces its own actionable message
rather than a stack trace, that an authored tag is never overwritten, and that
a run which does not ask for diarization is untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_video_maker import diarization
from music_video_maker.alignment import ConcurrentSegment, CounterpointAlignmentResult
from music_video_maker.contracts import AlignedSegment, AlignmentResult
from music_video_maker.diarization import (
    AGREED,
    ASSIGNED,
    ATTRIBUTION,
    CONTESTED,
    DISAGREED,
    PYANNOTE_GATED_PAGES,
    PYANNOTE_PIPELINE,
    PYANNOTE_TOKEN_ENV_VARS,
    UNCONFIDENT,
    UNCOVERED,
    UNMAPPED,
    DiarizerAccessError,
    DiarizerTokenError,
    DiarizerUnavailableError,
    DiarizerWeightsError,
    SpeakerSpan,
    assign_characters,
    build_pyannote_diarizer,
    lazy_pyannote_diarizer,
    spans_from_annotation,
    summarise_speakers,
)

STEM = Path("stems/htdemucs/master/vocals.wav")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _segment(
    index: int,
    start: float,
    end: float,
    *,
    characters: tuple[str, ...] = (),
    authored: bool = False,
    text: str = "a line of the song",
) -> AlignedSegment:
    return AlignedSegment(
        index=index,
        text=text,
        start=start,
        end=end,
        characters=characters,
        characters_authored=authored,
    )


def _alignment(*segments: AlignedSegment, duration: float = 60.0) -> AlignmentResult:
    return AlignmentResult(segments=tuple(segments), track_duration=duration)


def _diarizer(spans: list[SpeakerSpan], *, seen: list[Path] | None = None):
    def _run(audio_path: Path):
        if seen is not None:
            seen.append(audio_path)
        return spans

    return _run


def _raising_diarizer(exc: Exception):
    def _run(_audio_path: Path):
        raise exc

    return _run


class _FakePipeline:
    """Duck-types enough of a pyannote pipeline: callable, returns an annotation."""

    def __init__(self, spans: list[tuple[float, float, str]]) -> None:
        self.spans = spans
        self.calls: list[str] = []

    def __call__(self, audio: str):
        self.calls.append(audio)
        return _FakeAnnotation(self.spans)


class _FakeAnnotation:
    def __init__(self, spans: list[tuple[float, float, str]]) -> None:
        self._spans = spans

    def itertracks(self, yield_label: bool = False):
        assert yield_label, "this module must always ask for labels"
        for i, (start, end, label) in enumerate(self._spans):
            yield SimpleNamespace(start=start, end=end), f"track{i}", label


def _fake_importer(*, pipeline=None, raises: Exception | None = None, returns_none: bool = False):
    """A stand-in for ``pyannote.audio``, exposing only ``Pipeline.from_pretrained``."""

    def _from_pretrained(name: str, use_auth_token: str | None = None):
        _from_pretrained.calls.append((name, use_auth_token))
        if raises is not None:
            raise raises
        if returns_none:
            return None
        return pipeline

    _from_pretrained.calls = []  # type: ignore[attr-defined]

    def _import() -> object:
        return SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=_from_pretrained))

    _import.from_pretrained = _from_pretrained  # type: ignore[attr-defined]
    return _import


# --------------------------------------------------------------------------- #
# The seam: turns in, summaries out
# --------------------------------------------------------------------------- #


def test_spans_from_annotation_flattens_itertracks_in_time_order():
    """The only part of pyannote's API this module touches, duck-typed.

    Sorted on the way out so a log line reads in song order -- the turns come
    out of a clustering step and are not guaranteed to."""
    annotation = _FakeAnnotation([(4.0, 6.0, "SPEAKER_01"), (0.0, 3.0, "SPEAKER_00")])

    spans = spans_from_annotation(annotation)

    assert [(s.start, s.end, s.speaker) for s in spans] == [
        (0.0, 3.0, "SPEAKER_00"),
        (4.0, 6.0, "SPEAKER_01"),
    ]
    assert all(s.confidence is None for s in spans), (
        "a real pyannote Annotation carries no per-turn probability; inventing one here "
        "would be a number with nothing behind it"
    )


def test_summarise_speakers_orders_clusters_longest_first_with_an_example_lyric():
    """This table IS the feature's first run: diarization never produces a
    name, so an operator labels the clusters from it. Longest first because
    the lead is almost always the biggest cluster."""
    spans = [
        SpeakerSpan(0.0, 2.0, "SPEAKER_01"),
        SpeakerSpan(10.0, 20.0, "SPEAKER_00"),
        SpeakerSpan(30.0, 33.0, "SPEAKER_01"),
    ]
    segments = [_segment(0, 10.0, 20.0, text="the lead's own line")]

    summaries = summarise_speakers(spans, segments, {"SPEAKER_00": "Dianne"})

    assert [s.speaker for s in summaries] == ["SPEAKER_00", "SPEAKER_01"]
    assert summaries[0].total_seconds == pytest.approx(10.0)
    assert summaries[0].mapped_to == "Dianne"
    assert summaries[0].example_text == "the lead's own line"
    assert summaries[1].span_count == 2
    assert summaries[1].first_onset == pytest.approx(0.0)
    assert summaries[1].mapped_to is None


# --------------------------------------------------------------------------- #
# Assignment: the untagged lines are the ones this feature is for
# --------------------------------------------------------------------------- #


def test_an_untagged_segment_gets_the_detected_character_in_the_same_field():
    """The whole design in one assertion: diarization writes
    ``AlignedSegment.characters``, the field a ``[Name: Role]`` tag writes, so
    nothing downstream can tell where the value came from."""
    alignment = _alignment(
        _segment(0, 0.0, 5.0, characters=("Dianne",)),
        _segment(1, 10.0, 15.0, characters=("Dianne",)),
    )
    spans = [SpeakerSpan(0.0, 5.0, "SPEAKER_00"), SpeakerSpan(10.0, 15.0, "SPEAKER_01")]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    assert [s.characters for s in result.alignment.segments] == [("Dianne",), ("Marcus",)]
    assert [a.decision for a in result.report.assignments] == [ASSIGNED, ASSIGNED]
    assert result.report.disagreements == ()


def test_a_diarized_segment_is_not_marked_authored():
    """A classifier's guess must not come back out claiming to be a tag.

    If it did, a second pass (or any future consumer of the flag) would treat
    an inference as ground truth -- which is the exact confusion the flag was
    added to remove."""
    alignment = _alignment(_segment(0, 0.0, 5.0, characters=("Dianne",)))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 5.0, "SPEAKER_01")]),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
    )

    assert result.alignment.segments[0].characters == ("Marcus",)
    assert result.alignment.segments[0].characters_authored is False


def test_the_dominant_voice_wins_a_shared_segment_by_voiced_duration():
    """Issue #40's rule as #92 re-measured it: overlapping seconds inside the
    span, not word count and not whole-turn length. Deliberately the same
    measure ``slicing._dominant_character_member`` uses."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))
    spans = [
        SpeakerSpan(0.0, 7.5, "SPEAKER_01"),  # 7.5s inside, 75% share
        SpeakerSpan(7.5, 10.0, "SPEAKER_00"),  # 2.5s inside
    ]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    assert result.alignment.segments[0].characters == ("Marcus",)
    assert result.report.assignments[0].share == pytest.approx(0.75)


def test_only_the_overlap_inside_the_segment_counts_not_the_whole_turn():
    """A long turn that barely touches the segment must not win it.

    The old slicing rule compared whole-segment duration and credited a singer
    with 9.030s of a 5.577s chunk; this is that mistake's mirror image on the
    audio side, and the guard against it is that every total is clipped to the
    segment's own span before anything is compared."""
    alignment = _alignment(_segment(0, 9.0, 10.0, characters=("Dianne",)))
    spans = [
        SpeakerSpan(0.0, 9.2, "SPEAKER_00"),  # 60s of turn, 0.2s inside
        SpeakerSpan(9.2, 10.0, "SPEAKER_01"),  # 0.8s inside
    ]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    assert result.alignment.segments[0].characters == ("Marcus",)


def test_a_half_covered_segment_is_left_alone_rather_than_guessed_at():
    """Below ``min_coverage`` the diarizer effectively heard nothing there."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 2.0, "SPEAKER_01")]),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
    )

    assert result.alignment.segments[0].characters == ("Dianne",)
    assert result.report.assignments[0].decision == UNCOVERED
    assert result.report.fallback_segment_indices == (0,)


def test_a_contested_segment_falls_back_instead_of_flipping_a_coin():
    """A 50/50 harmony is where this module deliberately parts company with
    ``slicing``: slicing must pick someone (it has a chunk to render), and this
    can decline, because declining leaves the authored or default value --
    something a viewer can be warned about -- rather than a guess."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))
    spans = [SpeakerSpan(0.0, 5.0, "SPEAKER_00"), SpeakerSpan(5.0, 10.0, "SPEAKER_01")]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    assert result.report.assignments[0].decision == CONTESTED
    assert result.alignment.segments[0].characters == ("Dianne",)


def test_an_unmapped_cluster_names_nobody_and_says_which_label_to_map(caplog):
    """Diarization yields clusters, never names. With no mapping for a label,
    the honest outcome is "nothing assigned, here is the label"."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    with caplog.at_level(logging.WARNING, logger="music_video_maker.diarization"):
        result = assign_characters(
            alignment,
            diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_07")]),
            audio_path=STEM,
            speakers={},
        )

    assert result.report.assignments[0].decision == UNMAPPED
    assert result.alignment.segments[0].characters == ("Dianne",)
    assert any("SPEAKER_07" in r.getMessage() for r in caplog.records)
    assert any("diarization_speakers" in r.getMessage() for r in caplog.records)


def test_a_first_run_with_no_mapping_still_logs_the_cluster_table(caplog):
    """The designed first pass: no mapping can exist before the clusters do,
    so the run's job is to hand the operator the table to write one from."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",), text="hold the line"))

    with caplog.at_level(logging.INFO, logger="music_video_maker.diarization"):
        report = assign_characters(
            alignment,
            diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_00")]),
            audio_path=STEM,
        ).report

    table = report.speaker_table()
    assert "SPEAKER_00" in table
    assert "UNMAPPED" in table
    assert "hold the line" in table
    assert any("Detected speaker clusters" in r.getMessage() for r in caplog.records)


def test_a_confidence_bar_is_honoured_when_the_diarizer_actually_supplies_one():
    """``min_confidence`` is off by default because pyannote exposes nothing to
    gate on. The field exists so a diarizer that *does* can be gated, rather
    than this project inventing a probability to put in its place."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))
    spans = [SpeakerSpan(0.0, 10.0, "SPEAKER_01", confidence=0.2)]

    confident = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
    )
    gated = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
        min_confidence=0.5,
    )

    assert confident.report.assignments[0].decision == ASSIGNED
    assert gated.report.assignments[0].decision == UNCONFIDENT
    assert gated.alignment.segments[0].characters == ("Dianne",)


def test_a_zero_length_segment_is_uncovered_rather_than_a_division_by_zero():
    alignment = _alignment(_segment(0, 5.0, 5.0, characters=("Dianne",)))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_00")]),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Marcus"},
    )

    assert result.report.assignments[0].decision == UNCOVERED


def test_an_alignment_with_no_segments_reports_nothing_and_does_not_raise():
    result = assign_characters(
        _alignment(),
        diarizer=_diarizer([SpeakerSpan(0.0, 1.0, "SPEAKER_00")]),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne"},
    )

    assert result.report.assignments == ()
    assert result.alignment.segments == ()
    assert "1 speaker cluster(s)" in result.report.summary()


def test_the_stem_is_what_gets_diarized_not_the_master():
    """Diarizing the full mix is what made this feature not worth attempting
    (issue #101): a voice-like lead synth clusters as a singer. The path this
    module is handed is the one it reads."""
    seen: list[Path] = []
    assign_characters(
        _alignment(_segment(0, 0.0, 1.0)),
        diarizer=_diarizer([], seen=seen),
        audio_path=STEM,
        speakers={},
    )
    assert seen == [STEM]


# --------------------------------------------------------------------------- #
# The disagreement rule: the tag wins, and the clash is REPORTED
# --------------------------------------------------------------------------- #


def test_an_authored_tag_is_never_overwritten_and_the_clash_is_reported(caplog):
    """The rule this feature turns on. A silent overwrite of an authored fact
    is the defect, not the fix -- so the authored value stays, and the
    disagreement is a WARNING naming both sides and the numbers behind the
    detection, so whoever reads the log can fix the tag if the tag is wrong."""
    alignment = _alignment(
        _segment(0, 0.0, 10.0, characters=("Dianne",), authored=True, text="her tagged line")
    )

    with caplog.at_level(logging.WARNING, logger="music_video_maker.diarization"):
        result = assign_characters(
            alignment,
            diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_01")]),
            audio_path=STEM,
            speakers={"SPEAKER_01": "Marcus"},
        )

    segment = result.alignment.segments[0]
    assert segment.characters == ("Dianne",), "the authored tag must survive untouched"
    assert segment.characters_authored is True

    assert len(result.report.disagreements) == 1
    clash = result.report.disagreements[0]
    assert clash.authored == ("Dianne",)
    assert clash.detected == ("Marcus",)
    assert clash.speaker == "SPEAKER_01"
    assert result.report.assignments[0].decision == DISAGREED

    messages = [r.getMessage() for r in caplog.records]
    assert any("TAG WINS" in m and "Dianne" in m and "Marcus" in m for m in messages)
    assert any("her tagged line" in m for m in messages)


def test_a_tag_the_detection_agrees_with_is_counted_but_not_warned_about(caplog):
    """Agreement is the expected case; a line per segment would bury the
    clashes, which are the only thing anybody has to act on."""
    alignment = _alignment(
        _segment(0, 0.0, 10.0, characters=("Dianne",), authored=True),
        _segment(1, 20.0, 30.0, characters=("Dianne",), authored=True),
    )
    spans = [SpeakerSpan(0.0, 10.0, "SPEAKER_00"), SpeakerSpan(20.0, 30.0, "SPEAKER_00")]

    with caplog.at_level(logging.WARNING, logger="music_video_maker.diarization"):
        result = assign_characters(
            alignment,
            diarizer=_diarizer(spans),
            audio_path=STEM,
            speakers={"SPEAKER_00": "Dianne"},
        )

    assert result.report.agreed_count == 2
    assert result.report.disagreements == ()
    assert not any("TAG WINS" in r.getMessage() for r in caplog.records)


def test_an_untagged_default_and_an_explicit_tag_of_the_same_name_are_told_apart():
    """The reason ``characters_authored`` exists at all.

    Both segments carry ``("Dianne",)``. One is an authored ``[Dianne]`` tag
    and one is ``default_lead_vocalist`` showing through because nobody tagged
    that part of the song, and without the flag they are indistinguishable --
    so either every default would be protected from detection (making the
    feature inert) or every tag would be overwritable (making it dangerous)."""
    alignment = _alignment(
        _segment(0, 0.0, 10.0, characters=("Dianne",), authored=True),
        _segment(1, 20.0, 30.0, characters=("Dianne",), authored=False),
    )
    spans = [SpeakerSpan(0.0, 10.0, "SPEAKER_01"), SpeakerSpan(20.0, 30.0, "SPEAKER_01")]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
    )

    assert [s.characters for s in result.alignment.segments] == [("Dianne",), ("Marcus",)]
    assert [a.decision for a in result.report.assignments] == [DISAGREED, ASSIGNED]


def test_an_undecided_detection_over_a_tagged_segment_is_not_a_fallback():
    """A fallback is a thing that happened to an *untagged* segment. Where a
    tag is in force there is nothing to fall back to: the tag was the answer
    before the diarizer ran and still is."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Marcus",), authored=True))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 1.0, "SPEAKER_00")]),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne"},
        default_lead_vocalist="Dianne",
    )

    assert result.report.assignments[0].decision == UNCOVERED
    assert result.report.fallback_segment_indices == ()
    assert result.alignment.segments[0].characters == ("Marcus",)


def test_the_fallback_is_announced_loudly_and_names_the_default_lead(caplog):
    """"The wrong face on screen is worse than the default face" only holds if
    somebody is told which spans took the default."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    with caplog.at_level(logging.INFO, logger="music_video_maker.diarization"):
        assign_characters(
            alignment,
            diarizer=_diarizer([SpeakerSpan(0.0, 1.0, "SPEAKER_00")]),
            audio_path=STEM,
            speakers={"SPEAKER_00": "Marcus"},
            default_lead_vocalist="Dianne",
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any("default_lead_vocalist" in m for m in messages)
    assert any("'Dianne'" in m for m in messages)


def test_the_unmeasured_thresholds_announce_themselves_on_every_run(caplog):
    """A threshold picked without a measurement is a guess wearing a number.
    These two have never been scored against a hand-tagged multi-vocalist song,
    so the report says so every time rather than once in a doc."""
    with caplog.at_level(logging.WARNING, logger="music_video_maker.diarization"):
        assign_characters(
            _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",))),
            diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_00")]),
            audio_path=STEM,
            speakers={"SPEAKER_00": "Dianne"},
        )

    assert any("UNMEASURED" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Counterpoint: a level-3 file must not lose its type passing through here
# --------------------------------------------------------------------------- #


def test_a_counterpoint_result_keeps_its_class_and_its_concurrent_segments():
    """``alignment``'s own docstring says a level-3 file whose streams were
    dropped must still read as a counterpoint file. Diarization must not be
    the thing that quietly re-types it -- and concurrent segments are not
    diarized at all: the aligner never heard those voices separately, and their
    characters came from an explicit sub-block tag either way."""
    concurrent = ConcurrentSegment(
        index=1,
        text="the other voice",
        start=0.0,
        end=10.0,
        characters=("Marcus",),
        characters_authored=True,
    )
    alignment = CounterpointAlignmentResult(
        segments=(_segment(0, 0.0, 10.0, characters=("Dianne",)),),
        track_duration=60.0,
        concurrent_segments=(concurrent,),
    )

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_00")]),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne"},
    )

    assert isinstance(result.alignment, CounterpointAlignmentResult)
    assert result.alignment.concurrent_segments == (concurrent,)


# --------------------------------------------------------------------------- #
# Degrading: four ways to not have the weights, four messages
# --------------------------------------------------------------------------- #


def test_a_diarizer_failure_degrades_to_the_manual_tags_rather_than_ending_the_run(caplog):
    """A render is hours of GPU custody; ending one over an unset environment
    variable is the expensive mistake. The alignment comes back untouched --
    which is exactly the ``diarize = false`` behaviour -- and the message
    naming the remedy is on the log at ERROR."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",), authored=True))
    exc = DiarizerTokenError("no Hugging Face token found: set HF_TOKEN")

    with caplog.at_level(logging.ERROR, logger="music_video_maker.diarization"):
        result = assign_characters(
            alignment, diarizer=_raising_diarizer(exc), audio_path=STEM, speakers={}
        )

    assert result.alignment is alignment
    assert result.report.failure is not None
    assert "HF_TOKEN" in result.report.failure
    assert result.report.assignments == ()
    assert "did not run" in result.report.summary()
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("falls back to the manual" in m and "HF_TOKEN" in m for m in errors)


@pytest.mark.parametrize(
    "exc",
    [
        DiarizerUnavailableError("not installed"),
        DiarizerTokenError("no token"),
        DiarizerAccessError("gated"),
        DiarizerWeightsError("no weights"),
    ],
)
def test_every_refusal_type_degrades_the_same_way(exc):
    """All four are ``DiarizationError``s, so one ``except`` covers them and a
    fifth cause added later cannot sneak past it as an uncaught exception."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    result = assign_characters(
        alignment, diarizer=_raising_diarizer(exc), audio_path=STEM, speakers={}
    )

    assert result.alignment is alignment
    assert result.report.failure == str(exc)


def test_a_non_diarization_exception_is_not_swallowed():
    """Degrading is for "we could not get the weights", not for a bug.

    An unexpected exception must still reach the operator as a failure, or the
    next defect in this module is a run that silently stopped diarizing."""
    with pytest.raises(ValueError):
        assign_characters(
            _alignment(_segment(0, 0.0, 10.0)),
            diarizer=_raising_diarizer(ValueError("a real bug")),
            audio_path=STEM,
        )


def test_no_token_in_the_environment_names_the_variables_and_the_gated_pages():
    """The message has to carry both halves: a token is necessary and not
    sufficient, because the account that owns it must also have accepted the
    terms -- and finding that out one error at a time wastes a download."""
    with pytest.raises(DiarizerTokenError) as excinfo:
        build_pyannote_diarizer(env={}, importer=_fake_importer(pipeline=object()))

    message = str(excinfo.value)
    for name in PYANNOTE_TOKEN_ENV_VARS:
        assert name in message
    for page in PYANNOTE_GATED_PAGES:
        assert page in message
    assert "huggingface.co/settings/tokens" in message
    assert "run.toml" in message, "a token must never be asked for in committed run data"


@pytest.mark.parametrize("env_var", PYANNOTE_TOKEN_ENV_VARS)
def test_the_token_is_read_from_any_of_the_accepted_environment_variables(env_var):
    importer = _fake_importer(pipeline=_FakePipeline([]))

    build_pyannote_diarizer(env={env_var: "hf_secret"}, importer=importer)

    assert importer.from_pretrained.calls == [(PYANNOTE_PIPELINE, "hf_secret")]


def test_a_blank_token_counts_as_no_token():
    """An exported-but-empty variable is the commonest way to think you have
    set a credential, and it must not reach the hub as one."""
    with pytest.raises(DiarizerTokenError):
        build_pyannote_diarizer(env={"HF_TOKEN": "   "}, importer=_fake_importer())


def test_pyannote_not_being_installed_names_the_extra_to_install():
    """The real import path, run on a machine where the package is genuinely
    absent -- which is every machine in this project's CI, by design."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - only on a machine that installed the extra
        pytest.skip("pyannote.audio is installed here, so the absent-package path cannot run")

    with pytest.raises(DiarizerUnavailableError) as excinfo:
        build_pyannote_diarizer(env={"HF_TOKEN": "hf_secret"})

    message = str(excinfo.value)
    assert "music-video-maker[diarize]" in message
    assert "pyannote.audio>=3.1" in message
    assert "torch" in message, "the reason it is an extra and not a dependency"


@pytest.mark.parametrize(
    "raised",
    [
        RuntimeError("403 Client Error: Forbidden for url: https://huggingface.co/..."),
        RuntimeError("Access to model pyannote/segmentation-3.0 is restricted"),
        RuntimeError("You are awaiting approval to access this gated repo"),
        RuntimeError("You must accept the conditions to access this repo"),
    ],
)
def test_a_gated_403_says_which_pages_to_accept_and_with_which_account(raised):
    """The state this feature was built in: metadata answers 200 and the files
    answer 403, because the terms have not been accepted. The remedy is not a
    new token, so it must not be reported as one."""
    with pytest.raises(DiarizerAccessError) as excinfo:
        build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_secret"}, importer=_fake_importer(raises=raised)
        )

    message = str(excinfo.value)
    assert "GATED" in message
    for page in PYANNOTE_GATED_PAGES:
        assert page in message
    assert "owns the token" in message


def test_all_three_gated_repositories_are_named_not_just_the_pipeline():
    """``speaker-diarization-3.1`` loads a segmentation model and an embedding
    model, each gated separately. Accepting only the pipeline produces a 403
    naming a repository the operator never asked for, which is the difference
    between a fixable error and a confusing one."""
    assert len(PYANNOTE_GATED_PAGES) == 3
    assert any("segmentation-3.0" in page for page in PYANNOTE_GATED_PAGES)
    assert any("wespeaker" in page for page in PYANNOTE_GATED_PAGES)


def test_a_pipeline_that_loads_as_none_is_the_access_case():
    """pyannote's documented soft failure: ``from_pretrained`` returns ``None``
    instead of raising when the hub refuses it. Unhandled, that surfaces as
    ``TypeError: 'NoneType' object is not callable`` on the first chunk."""
    with pytest.raises(DiarizerAccessError) as excinfo:
        build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_secret"}, importer=_fake_importer(returns_none=True)
        )

    assert "None" in str(excinfo.value)
    assert PYANNOTE_GATED_PAGES[0] in str(excinfo.value)


@pytest.mark.parametrize(
    "raised",
    [
        RuntimeError("401 Client Error: Unauthorized"),
        RuntimeError("Invalid user token."),
    ],
)
def test_a_rejected_token_is_reported_as_a_token_problem_not_a_licence_one(raised):
    with pytest.raises(DiarizerTokenError) as excinfo:
        build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_bad"}, importer=_fake_importer(raises=raised)
        )

    assert "rejected" in str(excinfo.value)
    assert "huggingface.co/settings/tokens" in str(excinfo.value)


def test_missing_weights_talk_about_the_cache_and_the_network():
    """The fourth case, and the one whose remedy is neither a token nor a
    licence: nothing is committed to this repository, so the weights are
    fetched once per machine and the failure is about where they are not."""
    with pytest.raises(DiarizerWeightsError) as excinfo:
        build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_secret"},
            importer=_fake_importer(raises=OSError("Connection error, offline mode is enabled")),
        )

    message = str(excinfo.value)
    assert "HF_HOME" in message
    assert "~/.cache/huggingface" in message
    assert "Nothing is committed to this repository" in message


def test_a_successful_load_logs_the_cc_by_attribution_and_diarizes(caplog):
    """CC-BY-4.0 asks for credit from whoever *uses* the work, so the credit is
    emitted where the use happens, not only in a repository file nobody running
    a render will open."""
    pipeline = _FakePipeline([(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")])

    with caplog.at_level(logging.INFO, logger="music_video_maker.diarization"):
        diarizer = build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_secret"}, importer=_fake_importer(pipeline=pipeline)
        )
        spans = diarizer(STEM)

    assert [s.speaker for s in spans] == ["SPEAKER_00", "SPEAKER_01"]
    assert pipeline.calls == [str(STEM)]
    assert any(ATTRIBUTION in r.getMessage() for r in caplog.records)


def test_the_attribution_names_the_licence_the_author_and_the_pipeline():
    """What CC-BY actually requires, checked rather than assumed."""
    assert "CC-BY-4.0" in ATTRIBUTION
    assert "MIT" in ATTRIBUTION
    assert "pyannote" in ATTRIBUTION
    assert "Bredin" in ATTRIBUTION
    assert PYANNOTE_PIPELINE in ATTRIBUTION
    assert "No weights are redistributed" in ATTRIBUTION


def test_the_lazy_diarizer_defers_every_refusal_into_the_degrade_path():
    """Building the seam must not raise, or a loader failure escapes
    ``assign_characters``'s ``try`` and ends the run -- the thing the degrade
    path exists to prevent. So construction is deferred to first use."""
    built = lazy_pyannote_diarizer(env={}, importer=_fake_importer())

    result = assign_characters(
        _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",), authored=True)),
        diarizer=built,
        audio_path=STEM,
    )

    assert result.report.failure is not None
    assert "HF_TOKEN" in result.report.failure
    assert result.alignment.segments[0].characters == ("Dianne",)


def test_importing_this_module_does_not_import_pyannote():
    """The module is imported by ``cli`` unconditionally, so it must cost a run
    that never asks for diarization exactly nothing -- the same rule
    ``alignment`` follows for stable-ts."""
    import sys

    assert "pyannote" not in sys.modules
    assert diarization.PYANNOTE_PIPELINE  # the module is genuinely loaded


def test_iter_decisions_filters_assignments_by_outcome():
    result = assign_characters(
        _alignment(
            _segment(0, 0.0, 10.0, characters=("Dianne",)),
            _segment(1, 20.0, 30.0, characters=("Dianne",)),
        ),
        diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_00")]),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Marcus"},
    )

    assigned = list(diarization.iter_decisions(result.report, ASSIGNED))
    uncovered = list(diarization.iter_decisions(result.report, UNCOVERED))
    assert [a.segment_index for a in assigned] == [0]
    assert [a.segment_index for a in uncovered] == [1]
    assert AGREED not in {a.decision for a in result.report.assignments}


def test_the_report_summary_counts_every_outcome_it_claims_to():
    alignment = _alignment(
        _segment(0, 0.0, 10.0, characters=("Dianne",), authored=True),
        _segment(1, 10.0, 20.0, characters=("Dianne",), authored=True),
        _segment(2, 20.0, 30.0, characters=("Dianne",)),
        _segment(3, 40.0, 50.0, characters=("Dianne",)),
    )
    spans = [
        SpeakerSpan(0.0, 10.0, "SPEAKER_00"),
        SpeakerSpan(10.0, 20.0, "SPEAKER_01"),
        SpeakerSpan(20.0, 30.0, "SPEAKER_01"),
    ]

    report = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    ).report

    assert report.agreed_count == 1
    assert len(report.disagreements) == 1
    assert report.assigned_count == 1
    assert report.fallback_segment_indices == (3,)
    summary = report.summary()
    assert "1 assigned" in summary
    assert "1 agreed" in summary
    assert "1 disagreed" in summary


def test_a_refusal_raised_by_the_loader_itself_is_not_re_wrapped():
    """A specific message must not be replaced by the generic weights one.

    ``_classify_load_failure`` is for exceptions from *someone else's* code; a
    ``DiarizationError`` that reaches here already knows what it is, and
    re-classifying it would turn "accept these three pages" into "the weights
    are not available"."""
    raised = DiarizerAccessError("the weights are GATED: accept page one")

    with pytest.raises(DiarizerAccessError) as excinfo:
        build_pyannote_diarizer(
            env={"HF_TOKEN": "hf_secret"}, importer=_fake_importer(raises=raised)
        )

    assert excinfo.value is raised


def test_a_confidence_bar_with_no_confidences_to_read_does_not_block_an_assignment():
    """``min_confidence`` can only gate what the diarizer actually reports.

    Every real pyannote turn has ``confidence=None``, so a bar set against one
    must be a no-op rather than a silent refusal of every segment -- otherwise
    setting the option disables the feature."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_01")]),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
        min_confidence=0.99,
    )

    assert result.report.assignments[0].decision == ASSIGNED
    assert result.alignment.segments[0].characters == ("Marcus",)


def test_a_reported_confidence_above_the_bar_is_assigned():
    """The other side of the confidence gate: a diarizer that reports a
    confidence clearing the bar must be acted on, not merely not-refused."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))

    result = assign_characters(
        alignment,
        diarizer=_diarizer([SpeakerSpan(0.0, 10.0, "SPEAKER_01", confidence=0.9)]),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
        min_confidence=0.5,
    )

    assert result.report.assignments[0].decision == ASSIGNED
    assert result.alignment.segments[0].characters == ("Marcus",)


def test_confidence_is_weighted_by_how_long_each_turn_lasts():
    """A 0.1-second turn reported at 0.1 confidence must not sink a nine-second
    one reported at 0.95. The mean is duration-weighted for the same reason the
    dominant-voice rule counts seconds: a long turn is more of the evidence."""
    alignment = _alignment(_segment(0, 0.0, 10.0, characters=("Dianne",)))
    spans = [
        SpeakerSpan(0.0, 9.5, "SPEAKER_01", confidence=0.95),
        SpeakerSpan(9.5, 10.0, "SPEAKER_01", confidence=0.10),
    ]

    result = assign_characters(
        alignment,
        diarizer=_diarizer(spans),
        audio_path=STEM,
        speakers={"SPEAKER_01": "Marcus"},
        min_confidence=0.8,
    )

    assert result.report.assignments[0].decision == ASSIGNED
