"""Issue #100: quantize the work, not the boundary.

A field report from another pipeline built on this one: H3 only renders
``5 + 17k`` frame lengths, and #70 closed on the finding that quantizing a
chunk *moves its boundary*, which is what cuts phrases mid-utterance. Their
proposal is to choose the boundary where the content needs it, render the next
valid length **past** it, and discard the overrun at assembly.

What is tested here, by layer:

* ``FrameGrid.cover_frames`` / ``frames_between`` -- the two pieces of
  arithmetic, including the telescoping property the whole design rests on.
* ``slice_audio(boundary_overrun=...)`` -- that it is genuinely opt-in (the
  default timeline is identical), that it clears mid-phrase cuts the grid used
  to force, that every chunk's stem is cut to its *rendered* length, and that
  **video offset == audio offset by construction** for every chunk including
  the last one and the instrumental filler. That last one is proved with
  integer frame arithmetic over a whole synthetic song, not asserted: this
  project has been bitten twice at exactly that seam (the ``-shortest``
  bullet and the short-outro fix).
* ``ChunkFingerprint`` -- that a rendered-and-trimmed chunk cannot be
  confused with one rendered to length, in the tier that cannot be escaped,
  and that a pre-#100 state file still compares equal to a present-day
  default run so no existing resume churns.
* ``assembly.trim_overruns`` -- that the overrun comes off before concat,
  with a stream copy and no re-encode, that a run with no overrun makes
  exactly the ffmpeg calls it always did, and that a chunk claiming an
  overrun it cannot describe is refused before any subprocess.
* ``load_config`` -- the flag, its default, and the one combination that is
  refused outright.

Fully offline: synthesized WAVs via stdlib ``wave``, a fake subprocess runner
for every ffmpeg/ffprobe call, no GPU and no network.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import wave
from pathlib import Path

import pytest

from music_video_maker import hardware
from music_video_maker.assembly import (
    CONCAT_LIST_FILENAME,
    AssemblyResult,
    UntrimmableOverrunError,
    assemble_final_video,
    build_trim_args,
    trim_overruns,
)
from music_video_maker.contracts import (
    H3_FRAME_GRID,
    AlignmentResult,
    AudioChunk,
    ChunkFingerprint,
    ChunkResult,
    ChunkStatus,
    FrameGrid,
)
from music_video_maker.slicing import ChunkFrameMismatchError, slice_audio
from tests.harness.factories import make_aligned_segment, write_silent_wav

GRID = H3_FRAME_GRID
PROFILE = hardware.PROFILE_RTX_4090_24GB


# --------------------------------------------------------------------------- #
# Fixtures: one synthetic song with phrases the grid cannot help cutting
# --------------------------------------------------------------------------- #

# Phrase starts and ends chosen deliberately OFF the 1/24s frame grid and off
# any 5+17k multiple, which is what real forced-alignment output looks like --
# a fixture whose edges land on frames would make the outward-rounding rule
# below vacuously true.
_SPECS = [
    (2.13, 9.07, "the first long phrase that runs on and on and on", "Dianne"),
    (11.53, 19.01, "a second phrase that also keeps going for a while", "Dianne"),
    (24.07, 33.11, "third phrase here with plenty of words to carry", "Dianne"),
    (40.03, 52.29, "a fourth very long phrase indeed running past the ceiling", "Dianne"),
]
_TRACK_SECONDS = 61.3


def _alignment() -> AlignmentResult:
    segments = tuple(
        make_aligned_segment(i, text, start, end, character)
        for i, (start, end, text, character) in enumerate(_SPECS)
    )
    return AlignmentResult(segments=segments, track_duration=_TRACK_SECONDS)


def _slice(tmp_path: Path, *, overrun: bool, name: str = "run", **kwargs):
    root = tmp_path / name
    master = write_silent_wav(root / "master.wav", _TRACK_SECONDS)
    return slice_audio(
        master,
        _alignment(),
        kwargs.pop("profile", PROFILE),
        root / "chunks",
        cover_instrumentals=kwargs.pop("cover_instrumentals", True),
        boundary_overrun=overrun,
        **kwargs,
    )


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


# --------------------------------------------------------------------------- #
# The arithmetic (FrameGrid)
# --------------------------------------------------------------------------- #


def test_cover_frames_is_the_smallest_valid_length_that_covers():
    for frames in range(GRID.trained_min_frames, GRID.trained_max_frames + 1):
        covered = GRID.cover_frames(frames)
        assert GRID.is_valid(covered)
        assert covered >= frames
        assert covered - frames < GRID.step_frames
        assert GRID.trained_min_frames <= covered <= GRID.trained_max_frames


def test_cover_frames_matches_the_issues_own_closed_form():
    # The issue states it as max(124, 5 + 17*ceil((d*24 - 5)/17)); this is the
    # same number, and pinning the agreement is what makes a cost figure
    # computed here comparable with the 3.7% they measured on their clip.
    import math

    for frames in (1, 50, 124, 125, 141, 142, 200, 345, 346, 362):
        expected = max(124, 5 + 17 * math.ceil((frames - 5) / 17))
        assert GRID.cover_frames(frames) == min(expected, GRID.trained_max_frames)


def test_cover_frames_never_exceeds_the_trained_ceiling_for_a_legal_span():
    # 362 is itself a grid point (5 + 17*21), which is the reason a kept count
    # inside the trained range can always be covered inside it.
    assert GRID.is_valid(GRID.trained_max_frames)
    assert GRID.cover_frames(GRID.trained_max_frames) == GRID.trained_max_frames
    assert GRID.cover_frames(GRID.trained_max_frames - 1) == GRID.trained_max_frames


def test_frames_between_telescopes_so_no_boundary_choice_can_drift():
    # The load-bearing property: frames are measured from the track's start,
    # so the kept counts of a contiguous run always sum to the difference of
    # its two outer positions -- whatever the boundaries in between are, and
    # whether or not they land on a frame.
    boundaries = [0.0, 5.1739, 11.0001, 11.5, 29.74999, 40.0, 61.2999]
    kept = [
        GRID.frames_between(boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
    assert sum(kept) == GRID.frames_between(boundaries[0], boundaries[-1])
    # And a partial sum is exactly the absolute position of that boundary,
    # which is what makes "video offset == audio offset" a construction
    # rather than a check.
    for i in range(len(boundaries)):
        assert sum(kept[:i]) == round(boundaries[i] * GRID.fps)


def test_frames_between_is_not_the_rounded_duration():
    # Rounding the DIFFERENCE instead would let each chunk keep its own
    # residue, and a few hundred of those are the drift issue #20 exists to
    # eliminate. Pinned on a boundary pair where the two genuinely disagree,
    # so a future "simplification" to round(duration * fps) fails here.
    grid = FrameGrid()
    start, end = 0.979, 2.021
    assert grid.frames_between(start, end) == 49 - 23
    assert round((end - start) * grid.fps) == 25
    assert grid.frames_between(start, end) != round((end - start) * grid.fps)


# --------------------------------------------------------------------------- #
# Opt-in: the default timeline must not move at all
# --------------------------------------------------------------------------- #


def test_off_by_default_the_timeline_is_identical_and_declares_no_overrun(tmp_path):
    default = _slice(tmp_path, overrun=False, name="default")
    explicit_off = slice_audio(
        write_silent_wav(tmp_path / "b" / "master.wav", _TRACK_SECONDS),
        _alignment(),
        PROFILE,
        tmp_path / "b" / "chunks",
        cover_instrumentals=True,
    )
    assert [(c.start, c.end, c.frame_count) for c in default] == [
        (c.start, c.end, c.frame_count) for c in explicit_off
    ]
    assert all(c.render_frames is None for c in default)
    assert all(c.overrun_frames == 0 for c in default)
    assert all(c.rendered_frame_count == c.frame_count for c in default)
    # And every chunk's length is still grid-valid, which is the pre-#100
    # contract Stage 4a's `length` injection depends on.
    assert all(GRID.is_valid(c.frame_count) for c in default)


def test_on_it_moves_boundaries_the_grid_used_to_decide(tmp_path):
    off = _slice(tmp_path, overrun=False, name="off")
    on = _slice(tmp_path, overrun=True, name="on")
    assert [c.start for c in on] != [c.start for c in off]
    assert any(c.render_frames is not None for c in on)
    # Same number of chunks: this pass moves boundaries, it does not add or
    # remove any.
    assert len(on) == len(off)


def test_on_a_moved_boundary_lands_outside_the_phrase_not_at_its_far_end(tmp_path):
    # #70's own first mistake, which this pass could repeat for free: snapping
    # to a phrase edge and then rounding to the nearest frame lands back
    # inside the phrase. Measured on "Deathless" before the fix: 24 mid-phrase
    # cuts went to 23 while the deepest went from 86.0% to 99.9% through its
    # phrase. Every boundary this pass produces must be at or outside an edge.
    on = _slice(tmp_path, overrun=True, name="outward")
    segments = _alignment().segments
    off_starts = {round(c.start, 6) for c in _slice(tmp_path, overrun=False, name="base")}
    for chunk in on:
        if round(chunk.start, 6) in off_starts:
            continue  # not moved by this pass
        for segment in segments:
            assert not (segment.start < chunk.start < segment.end), (
                f"moved boundary {chunk.start} landed inside segment {segment.index} "
                f"({segment.start}-{segment.end})"
            )


def test_on_the_mid_phrase_cut_count_does_not_go_up(tmp_path, caplog):
    def cuts(overrun: bool, name: str) -> int:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="music_video_maker.slicing"):
            _slice(tmp_path, overrun=overrun, name=name)
        summaries = [
            record.getMessage()
            for record in caplog.records
            if "Mid-phrase boundary cuts:" in record.getMessage()
        ]
        assert summaries, "the #70 summary must be logged either way"
        message = summaries[-1]
        if "none --" in message:
            return 0
        return int(message.split("Mid-phrase boundary cuts: ")[1].split(" of ")[0])

    assert cuts(True, "cuts_on") <= cuts(False, "cuts_off")


def test_requires_instrumental_coverage_and_says_so(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="music_video_maker.slicing"):
        chunks = _slice(tmp_path, overrun=True, name="nocover", cover_instrumentals=False)
    assert any("Ignoring boundary_overrun" in r.getMessage() for r in caplog.records)
    assert all(c.render_frames is None for c in chunks)


# --------------------------------------------------------------------------- #
# The invariant: video offset == audio offset, by construction
# --------------------------------------------------------------------------- #


def test_every_chunk_starts_exactly_where_the_kept_frames_before_it_end(tmp_path):
    """The one invariant this must not break, proved with integer arithmetic.

    Stage 5 lays the *kept* frames end to end and muxes the pristine master
    over the result, so chunk N's position in the finished video is the sum of
    the kept frame counts of chunks 0..N-1 -- while its audio comes from
    ``chunk.start`` seconds into the master. Those two have to be the same
    instant for every chunk, and the only honest way to say so is in frames:
    a tolerance in seconds would hide exactly the sub-frame residue that
    accumulates into desync.
    """
    chunks = _slice(tmp_path, overrun=True, name="invariant")
    cumulative = 0
    for chunk in chunks:
        assert chunk.frame_count is not None
        # Where Stage 5 will place this chunk == where its audio was cut from.
        assert round(chunk.start * GRID.fps) == cumulative
        # And the frames it keeps are exactly the frames its span contains.
        assert chunk.frame_count == GRID.frames_between(chunk.start, chunk.end)
        cumulative += chunk.frame_count
    # Including the last one: the timeline ends on the track, to the frame.
    assert cumulative == round(_TRACK_SECONDS * GRID.fps)
    assert chunks[-1].end == pytest.approx(cumulative / GRID.fps)


def test_the_instrumental_filler_obeys_the_same_arithmetic(tmp_path):
    chunks = _slice(tmp_path, overrun=True, name="filler")
    filler = [c for c in chunks if c.is_instrumental]
    assert filler, "this fixture has unvoiced spans; they must be covered"
    cumulative = {
        chunk.chunk_id: sum(c.frame_count or 0 for c in chunks[: chunk.chunk_id])
        for chunk in chunks
    }
    for chunk in filler:
        assert round(chunk.start * GRID.fps) == cumulative[chunk.chunk_id]
        assert chunk.frame_count == GRID.frames_between(chunk.start, chunk.end)


def test_the_timeline_stops_overshooting_the_track(tmp_path):
    # CLAUDE.md's "-shortest" invariant: the grid-tiled timeline runs past the
    # master by up to one trained-floor chunk and the mux silently throws
    # those frames away. A free final boundary just stops where the song does,
    # and the frames become a recorded overrun instead of an invisible one.
    off = _slice(tmp_path, overrun=False, name="drift_off")
    on = _slice(tmp_path, overrun=True, name="drift_on")
    assert off[-1].end - _TRACK_SECONDS > 0.3
    assert abs(on[-1].end - _TRACK_SECONDS) <= 1.0 / GRID.fps
    assert on[-1].overrun_frames > 0


def test_a_timeline_that_already_ends_on_the_track_gives_nothing_back(tmp_path):
    # The mirror of the bullet above, and the branch it is easy to leave
    # untested: 260 frames is itself a grid point (5 + 17*15), so a 10.833s
    # unvoiced track tiles exactly and there is no tail to reclaim. Nothing
    # may be "given back" here -- taking a frame off a timeline that already
    # ends on the song would make the video short, which is the worse defect
    # of the two (the song's own ending cut out of the file).
    root = tmp_path / "exact"
    seconds = 260 / GRID.fps
    master = write_silent_wav(root / "master.wav", seconds)
    chunks = slice_audio(
        master,
        AlignmentResult(segments=(), track_duration=seconds),
        PROFILE,
        root / "chunks",
        cover_instrumentals=True,
        boundary_overrun=True,
    )
    assert sum(c.frame_count for c in chunks) == 260
    assert all(c.render_frames is None for c in chunks)
    assert chunks[-1].end == pytest.approx(seconds)


def test_a_zero_segment_track_is_unaffected_and_still_covered(tmp_path):
    # The wholly unvoiced case: nothing to move onto, so the only thing this
    # pass can do is give the tail back, and the covering must survive it.
    root = tmp_path / "silent"
    master = write_silent_wav(root / "master.wav", 40.0)
    chunks = slice_audio(
        master,
        AlignmentResult(segments=(), track_duration=40.0),
        PROFILE,
        root / "chunks",
        cover_instrumentals=True,
        boundary_overrun=True,
    )
    assert chunks
    cumulative = 0
    for chunk in chunks:
        assert round(chunk.start * GRID.fps) == cumulative
        cumulative += chunk.frame_count
    assert cumulative == round(40.0 * GRID.fps)


# --------------------------------------------------------------------------- #
# The rendered length, and the stem that conditions it
# --------------------------------------------------------------------------- #


def test_the_rendered_length_is_grid_valid_inside_the_trained_range(tmp_path):
    chunks = _slice(tmp_path, overrun=True, name="lengths")
    for chunk in chunks:
        rendered = chunk.rendered_frame_count
        assert rendered is not None
        assert GRID.is_valid(rendered), rendered
        assert GRID.trained_min_frames <= rendered <= GRID.trained_max_frames
        assert rendered >= chunk.frame_count
    # Every chunk but the last pays less than one grid step. The LAST one can
    # pay more, and legitimately: a tail with only a second of song left still
    # has to be rendered at H3's 124-frame trained floor, so the overrun there
    # is the floor minus the tail. That is the same arithmetic that used to
    # overshoot the track and let -shortest discard the difference in silence
    # (CLAUDE.md, issue #22); it is now a recorded number.
    for chunk in chunks[:-1]:
        assert chunk.overrun_frames < GRID.step_frames
    assert chunks[-1].rendered_frame_count >= GRID.trained_min_frames


def test_the_stem_is_cut_to_the_rendered_length_not_the_kept_one(tmp_path):
    # H3's conditioning has to cover what H3 renders -- this is the cost the
    # issue names, and it must be real in the file rather than implied.
    chunks = _slice(tmp_path, overrun=True, name="stems")
    trimmed = [c for c in chunks if c.render_frames is not None]
    assert trimmed, "this fixture must produce at least one overrun chunk"
    for chunk in chunks:
        expected = GRID.frames_to_seconds(chunk.rendered_frame_count)
        assert _wav_seconds(chunk.audio_file) == pytest.approx(expected, abs=1e-3)
        if chunk.render_frames is not None:
            # ...and it is strictly longer than the chunk's own span.
            assert _wav_seconds(chunk.audio_file) > chunk.duration


def test_an_overrun_is_normalised_away_when_nothing_was_actually_trimmed(tmp_path):
    # render_frames == frame_count is "rendered to length", and recording it
    # as a number would make an identical chunk re-render on resume.
    for chunk in _slice(tmp_path, overrun=True, name="normalise"):
        assert chunk.render_frames != chunk.frame_count


def test_emission_refuses_a_kept_count_that_disagrees_with_its_own_span(tmp_path, caplog):
    """The guard that makes the invariant above a refusal rather than a hope.

    Today ``_cover_instrumentals`` derives each piece's ``end`` from its own
    kept count, so the two agree by construction and this cannot fire -- which
    is exactly why it is worth pinning. A chunk whose kept count is one frame
    off its own span desyncs every chunk after it with no error anywhere, and
    Stage 2a is the last place that can still notice; the next edit to this
    file must not be able to introduce that silently. Injected at the pass
    boundary rather than faked further in, so what is tested is the real
    emission path.
    """
    import music_video_maker.slicing as slicing_module

    real = slicing_module._cover_instrumentals

    def desync_one_piece(*args, **kwargs):
        pieces = real(*args, **kwargs)
        # frame_count up by one, `end` untouched: the chunk now claims a frame
        # its own span does not contain. Contiguity is undisturbed, so this
        # reaches the #100 guard rather than the older overlap check.
        broken = dataclasses.replace(pieces[0], frame_count=pieces[0].frame_count + 1)
        return [broken, *pieces[1:]]

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(slicing_module, "_cover_instrumentals", desync_one_piece)
    try:
        with (
            caplog.at_level(logging.ERROR, logger="music_video_maker.slicing"),
            pytest.raises(ChunkFrameMismatchError, match="span containing"),
        ):
            _slice(tmp_path, overrun=True, name="corrupt")
    finally:
        monkeypatched.undo()
    assert any("issue #100" in record.getMessage() for record in caplog.records)


def test_emission_refuses_a_rendered_length_off_the_grid(tmp_path, caplog):
    import music_video_maker.slicing as slicing_module

    real = slicing_module._cover_instrumentals

    def off_grid(*args, **kwargs):
        pieces = real(*args, **kwargs)
        broken = dataclasses.replace(pieces[0], render_frames=(pieces[0].frame_count or 0) + 1)
        return [broken, *pieces[1:]]

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(slicing_module, "_cover_instrumentals", off_grid)
    try:
        with (
            caplog.at_level(logging.ERROR, logger="music_video_maker.slicing"),
            pytest.raises(ChunkFrameMismatchError, match="not a valid H3 grid point"),
        ):
            _slice(tmp_path, overrun=True, name="offgrid")
    finally:
        monkeypatched.undo()


# --------------------------------------------------------------------------- #
# ChunkFingerprint: a trimmed chunk must not pass as one rendered to length
# --------------------------------------------------------------------------- #


def test_render_frames_is_in_the_inescapable_conditioning_tier():
    assert "render_frames" in ChunkFingerprint.CONDITIONING_FIELDS
    assert "render_frames" not in ChunkFingerprint.TIMELINE_FIELDS
    assert "render_frames" not in ChunkFingerprint.CONTENT_FIELDS
    assert "render_frames" not in ChunkFingerprint.STACK_FIELDS


def test_a_trimmed_chunk_does_not_match_one_rendered_to_length():
    planned = ChunkFingerprint(start=0.0, end=5.5, frame_count=132, render_frames=141)
    cached = ChunkFingerprint(start=0.0, end=5.5, frame_count=132)
    # Same place, same span, same kept count -- different conditioning.
    assert planned.timeline_differences(cached) == ()
    assert planned.content_differences(cached) == ()
    assert planned.conditioning_differences(cached) == ("render_frames",)
    assert cached.conditioning_differences(planned) == ("render_frames",)


def test_two_different_overruns_do_not_match_each_other():
    a = ChunkFingerprint(start=0.0, end=5.5, frame_count=132, render_frames=141)
    b = ChunkFingerprint(start=0.0, end=5.5, frame_count=132, render_frames=158)
    assert b.conditioning_differences(a) == ("render_frames",)


def test_a_pre_100_state_file_still_matches_a_default_run():
    # No schema bump: `None` means "rendered to length", which is what every
    # old file and every default run both record, so nothing re-renders.
    old = ChunkFingerprint(start=0.0, end=5.5, frame_count=132)
    today = ChunkFingerprint(start=0.0, end=5.5, frame_count=132)
    assert today.conditioning_differences(old) == ()
    assert today.timeline_differences(old) == ()


def test_of_reads_the_overrun_off_the_chunk():
    chunk = AudioChunk(
        chunk_id=3,
        audio_file=Path("/tmp/c.wav"),
        start=1.0,
        end=6.5,
        text="x",
        frame_count=132,
        render_frames=141,
    )
    assert ChunkFingerprint.of(chunk).render_frames == 141
    plain = dataclasses.replace(chunk, render_frames=None)
    assert ChunkFingerprint.of(plain).render_frames is None


def test_the_fingerprint_survives_run_state_json(tmp_path):
    # Recording the overrun is only useful if a *resumed* run can read it
    # back. The three fields #38/#39/#45 were each fixed for were inputs that
    # decided the pixels with nothing writing them down; a field written to a
    # dataclass and dropped on serialisation is the same blind spot one layer
    # out.
    from music_video_maker.resilience import _deserialize_fingerprint, _serialize_fingerprint

    original = ChunkFingerprint(start=0.0, end=5.5, frame_count=132, render_frames=141)
    raw = _serialize_fingerprint(original)
    assert raw["render_frames"] == 141
    assert _deserialize_fingerprint(raw) == original
    # A pre-#100 file has no such key at all, and must read as "to length".
    del raw["render_frames"]
    assert _deserialize_fingerprint(raw).render_frames is None


def test_h3_is_asked_for_the_rendered_length_not_the_kept_one():
    # The number that actually reaches the graph. Handing H3 the kept count
    # would render exactly what the trim exists to avoid having to do, and
    # leave the stem (cut to the rendered length) longer than the video --
    # issue #20's drift, reintroduced from the other side.
    chunk = _chunk(0, frame_count=132, render_frames=141)
    assert chunk.rendered_frame_count == 141
    assert GRID.is_valid(chunk.rendered_frame_count)
    plain = _chunk(1, frame_count=141)
    assert plain.rendered_frame_count == 141


def test_the_isolated_vocal_stem_also_covers_the_rendered_length(tmp_path):
    """Issue #25's stem path, which cuts its own conditioning audio.

    ``slice_stem_for_chunks`` cut at the chunk's span and refused anything
    that disagreed with ``frame_count``. Under an overrun that produces a stem
    SHORTER than the `length` Stage 4a injects -- issue #20's drift arriving
    from the other side, on the one path whose whole purpose is to be the
    conditioning signal. It now cuts to the rendered end, like the mix does.
    """
    from music_video_maker.stems import slice_stem_for_chunks

    chunks = (
        _chunk(0, frame_count=124),
        _chunk(1, frame_count=132, render_frames=141),
    )
    stem = write_silent_wav(tmp_path / "vocal.wav", 60.0)
    result = slice_stem_for_chunks(stem, chunks, tmp_path / "stem_chunks", grid=GRID)

    assert len(result.chunks) == 2
    assert _wav_seconds(result.chunks[0].audio_file) == pytest.approx(
        GRID.frames_to_seconds(124), abs=1e-3
    )
    assert _wav_seconds(result.chunks[1].audio_file) == pytest.approx(
        GRID.frames_to_seconds(141), abs=1e-3
    )
    # ...and the overrun survives onto the re-pointed chunk, so the
    # fingerprint and Stage 5's trim still see it.
    assert result.chunks[1].render_frames == 141


# --------------------------------------------------------------------------- #
# Stage 5: the overrun is discarded, with a stream copy
# --------------------------------------------------------------------------- #


class _FakeRunner:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append(args)
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")


def _chunk(chunk_id: int, *, frame_count: int, render_frames: int | None = None) -> AudioChunk:
    start = chunk_id * 5.0
    return AudioChunk(
        chunk_id=chunk_id,
        audio_file=Path(f"/tmp/chunk_{chunk_id}.wav"),
        start=start,
        end=start + frame_count / GRID.fps,
        text=f"line {chunk_id}",
        frame_count=frame_count,
        render_frames=render_frames,
    )


def _results(chunk_ids) -> dict[int, ChunkResult]:
    return {
        cid: ChunkResult(
            chunk_id=cid,
            status=ChunkStatus.RENDERED,
            video_file=Path(f"/tmp/chunk_{cid}.mp4"),
        )
        for cid in chunk_ids
    }


def test_build_trim_args_copies_without_re_encoding():
    args = build_trim_args(Path("/tmp/in.mp4"), Path("/tmp/out.mp4"), 132)
    assert args[:3] == ["ffmpeg", "-y", "-i"]
    assert "-c:v" in args and args[args.index("-c:v") + 1] == "copy"
    assert "-frames:v" in args and args[args.index("-frames:v") + 1] == "132"
    assert "-an" in args
    # Invariant 1: nothing in this call may re-encode.
    assert "libx264" not in args
    assert args[-1] == "/tmp/out.mp4"


def test_trim_overruns_is_a_no_op_when_nothing_declares_one(tmp_path):
    chunks = [_chunk(0, frame_count=124), _chunk(1, frame_count=141)]
    runner = _FakeRunner()
    paths = [Path("/tmp/chunk_0.mp4"), Path("/tmp/chunk_1.mp4")]
    resolved, trims = trim_overruns(chunks, paths, [0, 1], tmp_path, runner)
    assert resolved == paths
    assert trims == ()
    assert runner.calls == []


def test_trim_overruns_replaces_only_the_chunks_that_need_it(tmp_path):
    chunks = [
        _chunk(0, frame_count=124),
        _chunk(1, frame_count=132, render_frames=141),
        _chunk(2, frame_count=141),
    ]
    runner = _FakeRunner()
    paths = [Path(f"/tmp/chunk_{i}.mp4") for i in range(3)]
    resolved, trims = trim_overruns(chunks, paths, [0, 1, 2], tmp_path, runner)
    assert len(runner.calls) == 1
    assert resolved[0] == paths[0] and resolved[2] == paths[2]
    assert resolved[1] != paths[1]
    assert len(trims) == 1
    trim = trims[0]
    assert (trim.chunk_id, trim.rendered_frames, trim.kept_frames) == (1, 141, 132)
    assert trim.discarded_frames == 9
    assert trim.trimmed == resolved[1]
    assert runner.calls[0] == list(trim.args)
    assert runner.calls[0][runner.calls[0].index("-frames:v") + 1] == "132"


def test_trim_overruns_keeps_separate_files_per_timeline(tmp_path):
    runner = _FakeRunner()
    song, _ = trim_overruns(
        [_chunk(3, frame_count=132, render_frames=141)],
        [Path("/tmp/song_3.mp4")],
        [3],
        tmp_path,
        runner,
    )
    prologue, _ = trim_overruns(
        [_chunk(3, frame_count=130, render_frames=141)],
        [Path("/tmp/prologue_3.mp4")],
        [3],
        tmp_path,
        runner,
        label="prologue",
    )
    # Chunk ids are a separate space per timeline (issue #66): a shared
    # filename would have one trim overwrite the other's frames.
    assert song[0] != prologue[0]


def test_trim_overruns_refuses_an_overrun_it_cannot_describe(tmp_path):
    runner = _FakeRunner()
    broken = dataclasses.replace(_chunk(0, frame_count=132), frame_count=None, render_frames=141)
    with pytest.raises(UntrimmableOverrunError):
        trim_overruns([broken], [Path("/tmp/c.mp4")], [0], tmp_path, runner)
    assert runner.calls == [], "nothing may run before the refusal"


def test_trim_overruns_refuses_a_render_shorter_than_what_is_kept(tmp_path):
    runner = _FakeRunner()
    impossible = _chunk(0, frame_count=141, render_frames=124)
    with pytest.raises(UntrimmableOverrunError):
        trim_overruns([impossible], [Path("/tmp/c.mp4")], [0], tmp_path, runner)
    assert runner.calls == []


def test_assembly_trims_before_concat_and_names_the_trimmed_files(tmp_path):
    chunks = [
        _chunk(0, frame_count=124),
        _chunk(1, frame_count=132, render_frames=141),
    ]
    runner = _FakeRunner()
    result = assemble_final_video(
        chunks,
        _results([0, 1]),
        tmp_path / "master.wav",
        tmp_path / "out",
        runner=runner,
        check_luminance=False,
        check_scene_cuts=False,
    )
    assert isinstance(result, AssemblyResult)
    assert len(result.overrun_trims) == 1
    concat_text = (tmp_path / "out" / CONCAT_LIST_FILENAME).read_text(encoding="utf-8")
    assert str(result.overrun_trims[0].trimmed.resolve()) in concat_text
    assert "chunk_0.mp4" in concat_text
    assert "chunk_1.mp4'" not in concat_text
    # trim, then concat, then mux -- in that order, and the trim is first
    # because the concat demuxer must never see the untrimmed file.
    steps = ["trim" if "-frames:v" in call else call[call.index("-i") + 1] for call in runner.calls]
    assert steps[0] == "trim"
    assert len(runner.calls) == 3


def test_assembly_without_any_overrun_makes_exactly_the_calls_it_always_did(tmp_path):
    chunks = [_chunk(0, frame_count=124), _chunk(1, frame_count=141)]
    runner = _FakeRunner()
    result = assemble_final_video(
        chunks,
        _results([0, 1]),
        tmp_path / "master.wav",
        tmp_path / "out",
        runner=runner,
        check_luminance=False,
        check_scene_cuts=False,
    )
    assert result.overrun_trims == ()
    assert len(runner.calls) == 2  # concat + mux, nothing else
    concat_text = (tmp_path / "out" / CONCAT_LIST_FILENAME).read_text(encoding="utf-8")
    assert "_trimmed_" not in concat_text
