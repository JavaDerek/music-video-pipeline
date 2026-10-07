"""Phrase-aware slicing: keep a sung phrase whole inside one chunk whenever the
chunk window allows it (opt-in, ``phrase_aware_slicing``).

General form: a chunk boundary that lands inside a sung phrase makes the
singer restart mid-word across two independently rendered shots. #70 measured
that most such cuts are made by instrumental coverage's grid re-anchoring, not
by the max-duration split, and that a local, one-boundary preference cannot
pay for a move (4-6 grid steps against a neighbour at the 124-frame floor).
The fix tested here is a global one: choose every boundary at once, over every
grid-valid tiling of the track, so the slack a move needs can come from
anywhere in the passage -- an instrumental gap seconds away -- rather than
only from the two chunks either side of it.

What is tested, by layer:

* **Opt-in and byte-identical when off** -- pinned against the timeline the
  pre-change code produced for the same fixture.
* **The guarantee** -- every phrase that fits the window is held whole when a
  tiling exists that does it; the timeline stays contiguous, on the grid,
  inside the window, and covers the track.
* **The fallback** -- a phrase longer than the window is cut at an inter-word
  gap (never inside a word), and the cut is logged with the words either side.
* **What it cannot do is said** -- a fitting phrase the window genuinely
  cannot hold is named at WARNING.
* **Interactions** -- refused alongside shot-plan ``length_seconds`` (#27),
  ignored loudly without instrumental coverage, composed with issue #79's
  onset preference (which can never re-introduce a cut) and with issue #100's
  overrun (which may only remove more).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from music_video_maker.contracts import (
    H3_FRAME_GRID,
    AlignmentResult,
    HardwareProfile,
    WordTiming,
)
from music_video_maker.shot_plan import ShotLength
from music_video_maker.slicing import _segment_containing, slice_audio
from tests.harness.factories import make_aligned_segment, write_silent_wav

GRID = H3_FRAME_GRID
HALF_FRAME = 0.5 / GRID.fps

# Deathless's own window: 8.0 s == 192 frames, the trained floor at 124.
PROFILE = HardwareProfile(name="8s", vram_gb=24.0, min_chunk_seconds=5.0, max_chunk_seconds=8.0)

# Shaped on the dense passage of "Deathless" (106.63-143.50 s): phrases of
# 5-7 s separated by gaps far shorter than one grid step, which is where the
# default tiling cuts three of the five phrases below.
_OFFSET = 10.0
_SPECS = [
    (_OFFSET + 2.13, _OFFSET + 7.61, "out of the heat and into the fire"),
    (_OFFSET + 7.95, _OFFSET + 15.15, "the souls falling down and flames"),
    (_OFFSET + 16.03, _OFFSET + 18.27, "growing higher"),
    (_OFFSET + 24.4, _OFFSET + 30.46, "and as the armies fight tonight"),
    (_OFFSET + 31.2, _OFFSET + 36.9, "kashay prays for an endless night"),
]
_TRACK = _OFFSET + 45.0

# What the pre-change code emitted for this fixture, frame count per chunk
# (starts are their running sum). Recorded before phrase_aware_slicing existed.
_DEFAULT_FRAMES = [141, 141, 141, 175, 158, 124, 141, 158, 141]


def _alignment(specs=_SPECS, track=_TRACK, segments=None) -> AlignmentResult:
    if segments is None:
        segments = tuple(
            make_aligned_segment(i, text, start, end, "Dianne")
            for i, (start, end, text) in enumerate(specs)
        )
    return AlignmentResult(segments=segments, track_duration=track)


def _slice(tmp_path: Path, alignment: AlignmentResult | None = None, *, name="run", **kwargs):
    alignment = alignment if alignment is not None else _alignment()
    root = tmp_path / name
    master = write_silent_wav(root / "master.wav", alignment.track_duration)
    return slice_audio(
        master,
        alignment,
        kwargs.pop("profile", PROFILE),
        root / "chunks",
        cover_instrumentals=kwargs.pop("cover_instrumentals", True),
        **kwargs,
    )


def _cuts(chunks, segments):
    """Every interior boundary strictly inside an aligned segment."""
    return [
        (chunk.end, _segment_containing(chunk.end, segments))
        for chunk in chunks[:-1]
        if _segment_containing(chunk.end, segments) is not None
    ]


def _assert_timeline_is_sound(chunks, track, *, max_frames=192):
    assert chunks[0].start == 0.0
    for earlier, later in zip(chunks[:-1], chunks[1:], strict=True):
        assert later.start == pytest.approx(earlier.end, abs=1e-9)
    for chunk in chunks:
        assert GRID.is_valid(chunk.frame_count)
        assert 124 <= chunk.frame_count <= max_frames
        assert chunk.end - chunk.start == pytest.approx(chunk.frame_count / GRID.fps, abs=1e-6)
    assert chunks[-1].end >= track - 1.0 / GRID.fps


# --------------------------------------------------------------------------- #
# Opt-in, and the default path untouched
# --------------------------------------------------------------------------- #


def test_default_timeline_is_unchanged_from_before_the_mode_existed(tmp_path):
    chunks = _slice(tmp_path)
    assert [c.frame_count for c in chunks] == _DEFAULT_FRAMES
    # The fixture is a real test of the mode: the default cuts three phrases.
    assert len(_cuts(chunks, _alignment().segments)) == 3


def test_flag_off_is_identical_to_not_passing_it(tmp_path):
    implicit = _slice(tmp_path, name="implicit")
    explicit = _slice(tmp_path, name="explicit", phrase_aware_slicing=False)
    strip = [
        (c.chunk_id, c.start, c.end, c.frame_count, c.render_frames, c.text,
         c.characters, c.source_segment_indices, c.is_instrumental, c.is_split_continuation)
        for c in implicit
    ]
    assert strip == [
        (c.chunk_id, c.start, c.end, c.frame_count, c.render_frames, c.text,
         c.characters, c.source_segment_indices, c.is_instrumental, c.is_split_continuation)
        for c in explicit
    ]
    for a, b in zip(implicit, explicit, strict=True):
        assert a.audio_file.read_bytes() == b.audio_file.read_bytes()


# --------------------------------------------------------------------------- #
# The guarantee
# --------------------------------------------------------------------------- #


def test_every_fitting_phrase_is_held_whole_when_a_tiling_exists(tmp_path):
    alignment = _alignment()
    chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)

    assert _cuts(chunks, alignment.segments) == []
    _assert_timeline_is_sound(chunks, _TRACK)
    # Every word is still prompted exactly once, in order.
    sung = " ".join(c.text for c in chunks if not c.is_instrumental).split()
    assert sung == " ".join(text for _s, _e, text in _SPECS).split()


def test_each_phrase_lands_in_exactly_one_chunk(tmp_path):
    alignment = _alignment()
    chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)
    for segment in alignment.segments:
        holders = [
            c for c in chunks
            if c.start < segment.end - 1e-6 and c.end > segment.start + 1e-6
        ]
        assert len(holders) == 1, (segment.index, [(c.start, c.end) for c in holders])


def test_the_summary_reports_how_many_phrases_were_kept_whole(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="music_video_maker.slicing"):
        _slice(tmp_path, phrase_aware_slicing=True)
    summary = [r.getMessage() for r in caplog.records if "Phrase-aware slicing:" in r.getMessage()]
    assert summary, "no phrase-aware summary line was logged"
    assert "5 of 5" in summary[-1]


# --------------------------------------------------------------------------- #
# The fallback: a phrase that cannot fit is cut between words, never inside one
# --------------------------------------------------------------------------- #


def _long_phrase_alignment() -> AlignmentResult:
    # 10.4 s phrase: no 8 s chunk can hold it. One real pause (0.30 s) between
    # "river" and "and"; every other word abuts the next.
    start = 20.13
    spec = [
        ("down", 0.0, 1.1), ("by", 1.1, 1.9), ("the", 1.9, 2.4), ("river", 2.4, 4.6),
        ("and", 4.9, 5.7), ("over", 5.7, 7.0), ("the", 7.0, 7.6), ("hill", 7.6, 10.4),
    ]
    words = tuple(WordTiming(word=w, start=start + a, end=start + b) for w, a, b in spec)
    long_phrase = make_aligned_segment(
        0, " ".join(w for w, _a, _b in spec), start, start + 10.4, "Dianne", words=words
    )
    after = make_aligned_segment(1, "and back again", 39.07, 43.5, "Dianne")
    return _alignment(segments=(long_phrase, after), track=60.0)


def test_a_phrase_longer_than_the_window_is_cut_between_words(tmp_path, caplog):
    alignment = _long_phrase_alignment()
    long_phrase = alignment.segments[0]
    with caplog.at_level(logging.INFO, logger="music_video_maker.slicing"):
        chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)

    cuts = _cuts(chunks, alignment.segments)
    assert [segment.index for _t, segment in cuts] == [0], "cut exactly once, and only there"
    boundary = cuts[0][0]
    for word in long_phrase.words:
        assert not (word.start + HALF_FRAME < boundary < word.end - HALF_FRAME), (
            f"boundary {boundary:.3f}s is inside the word {word.word!r}"
        )
    _assert_timeline_is_sound(chunks, alignment.track_duration)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "longer than" in messages and "between" in messages


# --------------------------------------------------------------------------- #
# What it cannot do is said, by name
# --------------------------------------------------------------------------- #


def test_a_fitting_phrase_the_window_cannot_hold_is_named(tmp_path, caplog):
    # Two 7.4 s phrases 2.0 s apart. Holding both whole needs a boundary in the
    # 2.0 s gap whose chunk on each side runs >= 7.4 s and <= 8.0 s -- and the
    # chunk before the first phrase starts at 0, too early to give the first
    # one a legal start. Something has to be cut, and the log must say what.
    alignment = _alignment(
        specs=[(0.4, 7.8, "first phrase goes on and on"), (9.8, 17.2, "second phrase goes on too")],
        track=30.0,
    )
    with caplog.at_level(logging.WARNING, logger="music_video_maker.slicing"):
        chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)

    cuts = _cuts(chunks, alignment.segments)
    assert cuts, "the fixture should be infeasible"
    named = [
        r.getMessage() for r in caplog.records
        if "Phrase-aware slicing" in r.getMessage() and "could not be held whole" in r.getMessage()
    ]
    assert len(named) == len(cuts)
    _assert_timeline_is_sound(chunks, alignment.track_duration)


# --------------------------------------------------------------------------- #
# Interactions with #27, #21, #79, #100
# --------------------------------------------------------------------------- #


def test_refused_alongside_shot_length_requests(tmp_path):
    with pytest.raises(ValueError, match="phrase_aware_slicing"):
        _slice(
            tmp_path,
            phrase_aware_slicing=True,
            shot_lengths=(ShotLength(start=0.0, length_seconds=10.0, source_chunk_id=0),),
        )


def test_ignored_loudly_without_instrumental_coverage(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="music_video_maker.slicing"):
        with_flag = _slice(tmp_path, name="a", phrase_aware_slicing=True, cover_instrumentals=False)
    without = _slice(tmp_path, name="b", cover_instrumentals=False)
    assert [(c.start, c.frame_count) for c in with_flag] == [
        (c.start, c.frame_count) for c in without
    ]
    assert any("Ignoring phrase_aware_slicing" in r.getMessage() for r in caplog.records)


def test_issue_79_onset_preference_never_reintroduces_a_cut(tmp_path):
    # A long instrumental gap before each phrase gives #79 room to move every
    # boundary toward the onset; whatever it does, no phrase may end up cut.
    alignment = _alignment(
        specs=[
            (12.9, 18.7, "first line of the verse here"),
            (24.7, 30.2, "second line of the verse here"),
            (41.9, 47.0, "third line of the verse here"),
        ],
        track=58.0,
    )
    chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)
    assert _cuts(chunks, alignment.segments) == []
    for chunk in chunks:
        if not chunk.is_instrumental:
            first = min(
                w.start
                for s in alignment.segments
                if s.index in chunk.source_segment_indices
                for w in s.words
            )
            # Starts at or before its first word, never after it -- and, with
            # a gap this long to stand in, opens within #79's 1 s warning line.
            assert chunk.start <= first + 1e-6
            assert first - chunk.start <= 1.0


def _leading_offsets(chunks, segments):
    offsets = []
    for chunk in chunks:
        if chunk.is_instrumental:
            continue
        words = [
            w for s in segments if s.index in chunk.source_segment_indices for w in s.words
            if chunk.start <= (w.start + w.end) / 2 < chunk.end
        ]
        offsets.append(min(w.start for w in words) - chunk.start)
    return offsets


def test_issue_79_leading_offset_is_not_traded_away_for_phrases(tmp_path):
    """The plan must not buy a whole phrase with a chunk that opens seconds
    before anyone sings: that is #79's defect, which a viewer named twice. On
    this fixture the default's worst offset is the bar."""
    alignment = _alignment()
    default = _leading_offsets(_slice(tmp_path, alignment, name="d"), alignment.segments)
    planned = _leading_offsets(
        _slice(tmp_path, alignment, name="p", phrase_aware_slicing=True), alignment.segments
    )
    assert sum(1 for o in planned if o > 1.0) <= sum(1 for o in default if o > 1.0)
    assert max(planned) <= max(max(default), 1.0)


def test_composes_with_boundary_overrun(tmp_path):
    alignment = _alignment()
    phrase_only = _slice(tmp_path, alignment, name="p", phrase_aware_slicing=True)
    both = _slice(
        tmp_path, alignment, name="both", phrase_aware_slicing=True, boundary_overrun=True
    )
    assert len(_cuts(both, alignment.segments)) <= len(_cuts(phrase_only, alignment.segments))
    # #100's own invariant survives the composition: kept frames telescope.
    for chunk in both:
        assert chunk.frame_count == GRID.frames_between(chunk.start, chunk.end)
        assert GRID.is_valid(chunk.rendered_frame_count)


# --------------------------------------------------------------------------- #
# Edges of the mechanism
# --------------------------------------------------------------------------- #


def test_abutting_phrases_split_within_half_a_frame_are_reported_as_such(tmp_path, caplog):
    # Two phrases with no gap at all (stable-ts often emits that), together
    # longer than the window: the only clean boundary is an instant off the
    # frame grid, so the plan lands within half a frame of it and says so
    # rather than calling it a cut. The shared edge sits 0.010 s after frame
    # 598, a position three grid chunks from 0 can reach -- a fixture this
    # short has too few chunks before it to reach every phase.
    edge = 598 / 24 + 0.010
    alignment = _alignment(
        specs=[(20.13, edge, "the first of two lines"), (edge, 30.6, "the second of two lines")],
        track=42.0,
    )
    with caplog.at_level(logging.INFO, logger="music_video_maker.slicing"):
        chunks = _slice(tmp_path, alignment, phrase_aware_slicing=True)
    for t, segment in _cuts(chunks, alignment.segments):
        assert min(t - segment.start, segment.end - t) <= HALF_FRAME
    messages = [r.getMessage() for r in caplog.records]
    assert any("within half a frame" in m and "index=0" in m for m in messages)
    assert any("1 more split within half a frame" in m for m in messages)


def test_long_instrumental_ceiling_applies_to_filler_only(tmp_path):
    # instrumental_shot_seconds lets filler run long; a chunk overlapping a
    # phrase is still held to max_chunk_seconds.
    alignment = _alignment(specs=[(30.13, 36.2, "one line in the middle")], track=70.0)
    chunks = _slice(
        tmp_path, alignment, phrase_aware_slicing=True, instrumental_shot_seconds=12.0
    )
    assert _cuts(chunks, alignment.segments) == []
    for chunk in chunks:
        assert GRID.is_valid(chunk.frame_count)
        assert chunk.frame_count <= (192 if not chunk.is_instrumental else 294)
    assert any(c.is_instrumental and c.frame_count > 192 for c in chunks)


def test_cut_detail_treats_a_phrase_without_word_timings_as_one_word():
    from music_video_maker.contracts import AlignedSegment
    from music_video_maker.slicing import _cut_detail

    bare = AlignedSegment(index=3, text="hum", start=1.0, end=3.0, words=())
    in_word, where = _cut_detail(2.0, bare, HALF_FRAME)
    assert in_word is True
    assert "no word timings" in where
