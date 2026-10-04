"""End-to-end chain test for issue #29: a second vocalist's ``[Name]`` tag.

#29's own claim was that manual character denotation "already works end to
end" and the real gap was documentation/workflow, not capability -- but that
claim was an assertion, never a measurement, and #29 says so itself: "worth
verifying end to end with a genuinely multi-vocalist song before building
anything." This test is that verification. It walks the exact chain #29
describes -- lyrics text -> ``parse_lyrics_text`` -> ``align()`` -> Stage 1
-> ``slice_audio`` -> Stage 2b's ``expand_prompt`` -> Stage 3's
``ComfyUIAssetStager`` -- in a single function, offline, with no network, no
GPU and no real whisper model (the fake ``model.align()`` seam
``tests/test_alignment.py`` already uses).

Individual links here already have their own coverage, and this test is not
trying to duplicate it: ``tests/test_alignment.py::
test_align_character_switches_with_source_lyric_lines`` already shows a
tagged file's characters survive ``align()``, and ``tests/test_slicing.py``'s
``test_merged_chunk_with_differing_characters_attributes_to_dominant_voice``
family already covers what happens when two different singers' segments get
merged into *one* chunk. What none of those touch is the far end of the
chain: does the composed *prompt* actually carry the second vocalist's own
``role`` text, and does the *reference image path Stage 3 would upload* for
that chunk actually point at the second vocalist's photo rather than the
lead's. If either link were silently broken, "the tags already work end to
end" would be false in exactly the way #29 describes as the motivating bug:
a mis-tagged (or, here, a mis-*wired*) line puts the wrong face on screen for
that chunk, with no error anywhere.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from music_video_maker.alignment import align
from music_video_maker.config import RunConfig
from music_video_maker.diarization import SpeakerSpan, assign_characters
from music_video_maker.hardware import PROFILE_RTX_4090_24GB
from music_video_maker.lyrics import parse_lyrics_text
from music_video_maker.prompting import expand_prompt
from music_video_maker.slicing import slice_audio
from music_video_maker.staging import ComfyUIAssetStager
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes
from tests.harness.factories import make_cast_member, write_silent_wav


def _word(word: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=word, start=start, end=end, probability=0.99)


def _fake_raw_segment(text: str, start: float, end: float) -> SimpleNamespace:
    """Duck-types one entry of stable-ts's ``WhisperResult.segments``, the
    same shape ``tests/test_alignment.py``'s ``_segment`` helper builds.

    Built from the already-parsed lyric line's own text (word for word), the
    same trick that test uses to keep the fake model's word count in exact
    positional agreement with ``align()``'s word-owner walk -- character
    attribution is by word position, not by text matching, so the raw
    segment's word count must equal the source line's.
    """
    words = text.split()
    span = (end - start) / max(len(words), 1)
    word_objs = [_word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    return SimpleNamespace(text=text, start=start, end=end, words=word_objs)


def test_second_vocalist_tag_reaches_prompt_and_staged_reference_image(tmp_path):
    """A ``[Marcus: Backup]`` tag must reach Marcus's own ``role`` text in the
    composed prompt and Marcus's own reference photo in the file Stage 3
    would stage -- not Dianne's, the lead and ``default_lead_vocalist``.
    """
    dianne_image = tmp_path / "dianne_ref.png"
    dianne_image.write_bytes(make_fake_png_bytes(64, 64))
    marcus_image = tmp_path / "marcus_ref.png"
    marcus_image.write_bytes(make_fake_png_bytes(64, 64))

    cast = {
        "Dianne": make_cast_member(
            "Dianne", "Lead Vocalist, smiling constantly, oblivious", dianne_image
        ),
        "Marcus": make_cast_member(
            "Marcus", "Backup Vocalist, watching from the wings", marcus_image
        ),
    }

    # Step 1: lyrics text with two [Name] tags -> parse_lyrics_text.
    lyrics_text = (
        "[Dianne: Lead]\n"
        "The lucky ones dont ever have to try\n"
        "[Marcus: Backup]\n"
        "Were watching from the wings tonight\n"
    )
    doc = parse_lyrics_text(lyrics_text, cast, default_lead_vocalist="Dianne")
    assert [line.character for line in doc] == ["Dianne", "Marcus"]

    # Step 2: align() against a fake stable-ts model -- the exact seam
    # tests/test_alignment.py uses, never a real model or GPU.
    raw_segments = [
        _fake_raw_segment(doc[0].text, 0.0, 6.5),
        _fake_raw_segment(doc[1].text, 8.0, 14.0),
    ]
    model = MagicMock()
    model.align.return_value = SimpleNamespace(segments=raw_segments, language="en")

    master = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    alignment = align(master, doc, model=model)
    assert [segment.character for segment in alignment.segments] == ["Dianne", "Marcus"]

    # Step 3: slice_audio. Both segments (6.5s, 6.0s) already clear H3's
    # ~5.17s trained floor, so neither needs padding or merging and each
    # keeps its own segment's character -- this is deliberately the plain
    # case, not the merge-attribution case test_slicing.py's
    # test_merged_chunk_* family already covers.
    chunks = slice_audio(master, alignment, PROFILE_RTX_4090_24GB, tmp_path / "chunks")
    assert [chunk.characters for chunk in chunks] == [("Dianne",), ("Marcus",)]
    marcus_chunk = chunks[1]

    config = RunConfig(
        master_audio=master,
        lyrics_file=tmp_path / "lyrics.txt",
        global_style="Refestramus progressive rock music video",
        narrative_concept="Wandering through an empty theatre",
        cast=cast,
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=PROFILE_RTX_4090_24GB,
    )

    # Step 4: expand_prompt (Stage 2b's real entry point) -- does the
    # prompt carry Marcus's own role text, not Dianne's?
    prompt = expand_prompt(config, marcus_chunk)
    assert "Backup Vocalist, watching from the wings" in prompt.prompt
    assert "Lead Vocalist, smiling constantly, oblivious" not in prompt.prompt
    assert prompt.characters == ("Marcus",)
    assert prompt.image_ref == marcus_image

    # Step 5: Stage 3 staging (ComfyUIAssetStager against the offline fake
    # ComfyUI session, issue #16's harness -- no network). The file that
    # would actually be uploaded for this chunk must be Marcus's photo.
    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(base_url=session.base_url, session=session)
    assets = stager.stage_chunk(prompt, marcus_chunk)
    assert assets.image_filename == "marcus_ref.png"
    assert assets.image_filename != "dianne_ref.png"


# --------------------------------------------------------------------------- #
# Issue #101: the same chain, with nobody tagged and a diarizer instead
# --------------------------------------------------------------------------- #


def test_diarization_reaches_prompt_and_staged_reference_image_on_an_untagged_file(tmp_path):
    """The acceptance test for issue #101's shape, walked end to end offline.

    Same chain as the test above -- ``parse_lyrics_text`` -> ``align()`` ->
    ``slice_audio`` -> ``expand_prompt`` -> ``ComfyUIAssetStager.stage_chunk``
    -- with **one** difference: the lyrics file carries no ``[Name]`` tags at
    all, so every line comes out of the parser attributed to
    ``default_lead_vocalist``, which is exactly the bug that put the lead's
    face on every line of "The Lucky Ones".

    Diarization is inserted between ``align()`` and ``slice_audio``, where the
    real pipeline puts it, and writes the **same**
    ``AlignedSegment.characters`` field the tags write. Nothing after Stage 2a
    is changed or configured for this test, which is the claim being checked:
    the prompt carries Marcus's own ``role`` text and Stage 3 would upload
    Marcus's photograph, with no code downstream knowing a classifier was
    involved.

    The diarizer is the injected seam, so this runs with no ``pyannote``, no
    weights, no network and no GPU.
    """
    dianne_image = tmp_path / "dianne_ref.png"
    dianne_image.write_bytes(make_fake_png_bytes(64, 64))
    marcus_image = tmp_path / "marcus_ref.png"
    marcus_image.write_bytes(make_fake_png_bytes(64, 64))

    cast = {
        "Dianne": make_cast_member(
            "Dianne", "Lead Vocalist, smiling constantly, oblivious", dianne_image
        ),
        "Marcus": make_cast_member(
            "Marcus", "Backup Vocalist, watching from the wings", marcus_image
        ),
    }

    # Step 1: NO tags anywhere. Both lines inherit the default lead.
    lyrics_text = (
        "The lucky ones dont ever have to try\nWere watching from the wings tonight\n"
    )
    doc = parse_lyrics_text(lyrics_text, cast, default_lead_vocalist="Dianne")
    assert [line.character for line in doc] == ["Dianne", "Dianne"]
    assert not any(line.characters_authored for line in doc), (
        "an untagged file must not claim its characters were authored, or diarization "
        "would be forbidden from touching the very lines it exists for"
    )

    raw_segments = [
        _fake_raw_segment(doc[0].text, 0.0, 6.5),
        _fake_raw_segment(doc[1].text, 8.0, 14.0),
    ]
    model = MagicMock()
    model.align.return_value = SimpleNamespace(segments=raw_segments, language="en")

    master = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    alignment = align(master, doc, model=model)
    assert [s.character for s in alignment.segments] == ["Dianne", "Dianne"]

    # Step 2: diarization, on the isolated vocal stem (issue #25's artefact),
    # with an authored cluster -> cast mapping. Two clusters, one per line.
    stem = write_silent_wav(tmp_path / "vocals.wav", seconds=20.0)
    spans = [
        SpeakerSpan(start=0.0, end=6.5, speaker="SPEAKER_00"),
        SpeakerSpan(start=8.0, end=14.0, speaker="SPEAKER_01"),
    ]
    diarized = assign_characters(
        alignment,
        diarizer=lambda _path: spans,
        audio_path=stem,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
        default_lead_vocalist="Dianne",
    )
    assert [s.character for s in diarized.alignment.segments] == ["Dianne", "Marcus"]
    assert diarized.report.disagreements == (), "nothing was tagged, so nothing can clash"

    # Step 3 onwards: the untouched render path.
    chunks = slice_audio(
        master, diarized.alignment, PROFILE_RTX_4090_24GB, tmp_path / "chunks"
    )
    assert [chunk.characters for chunk in chunks] == [("Dianne",), ("Marcus",)]
    marcus_chunk = chunks[1]

    config = RunConfig(
        master_audio=master,
        lyrics_file=tmp_path / "lyrics.txt",
        global_style="Refestramus progressive rock music video",
        narrative_concept="Wandering through an empty theatre",
        cast=cast,
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=PROFILE_RTX_4090_24GB,
        vocal_stem=stem,
        diarize=True,
        diarization_speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    prompt = expand_prompt(config, marcus_chunk)
    assert "Backup Vocalist, watching from the wings" in prompt.prompt
    assert "Lead Vocalist, smiling constantly, oblivious" not in prompt.prompt
    assert prompt.characters == ("Marcus",)
    assert prompt.image_ref == marcus_image

    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(base_url=session.base_url, session=session)
    assets = stager.stage_chunk(prompt, marcus_chunk)
    assert assets.image_filename == "marcus_ref.png"
    assert assets.image_filename != "dianne_ref.png"


def test_a_tag_beats_diarization_all_the_way_to_the_staged_photograph(tmp_path):
    """The disagreement rule, checked at the far end of the chain rather than
    at the report.

    The lyrics tag the second line as Dianne's; the diarizer says Marcus. The
    photograph Stage 3 would upload is **Dianne's**, and the clash is in the
    report for someone to act on. A test that only asserted
    ``report.disagreements`` would pass even if the authored value had been
    overwritten one layer down."""
    dianne_image = tmp_path / "dianne_ref.png"
    dianne_image.write_bytes(make_fake_png_bytes(64, 64))
    marcus_image = tmp_path / "marcus_ref.png"
    marcus_image.write_bytes(make_fake_png_bytes(64, 64))
    cast = {
        "Dianne": make_cast_member("Dianne", "Lead Vocalist, oblivious", dianne_image),
        "Marcus": make_cast_member("Marcus", "Backup Vocalist, in the wings", marcus_image),
    }

    lyrics_text = (
        "[Dianne: Lead]\n"
        "The lucky ones dont ever have to try\n"
        "Were watching from the wings tonight\n"
    )
    doc = parse_lyrics_text(lyrics_text, cast, default_lead_vocalist="Dianne")
    assert all(line.characters_authored for line in doc)

    raw_segments = [
        _fake_raw_segment(doc[0].text, 0.0, 6.5),
        _fake_raw_segment(doc[1].text, 8.0, 14.0),
    ]
    model = MagicMock()
    model.align.return_value = SimpleNamespace(segments=raw_segments, language="en")

    master = write_silent_wav(tmp_path / "master.wav", seconds=20.0)
    alignment = align(master, doc, model=model)

    stem = write_silent_wav(tmp_path / "vocals.wav", seconds=20.0)
    diarized = assign_characters(
        alignment,
        diarizer=lambda _path: [
            SpeakerSpan(start=0.0, end=6.5, speaker="SPEAKER_00"),
            SpeakerSpan(start=8.0, end=14.0, speaker="SPEAKER_01"),
        ],
        audio_path=stem,
        speakers={"SPEAKER_00": "Dianne", "SPEAKER_01": "Marcus"},
    )

    assert len(diarized.report.disagreements) == 1
    clash = diarized.report.disagreements[0]
    assert clash.authored == ("Dianne",)
    assert clash.detected == ("Marcus",)

    chunks = slice_audio(
        master, diarized.alignment, PROFILE_RTX_4090_24GB, tmp_path / "chunks"
    )
    config = RunConfig(
        master_audio=master,
        lyrics_file=tmp_path / "lyrics.txt",
        global_style="Refestramus progressive rock music video",
        narrative_concept="Wandering through an empty theatre",
        cast=cast,
        default_lead_vocalist="Dianne",
        comfyui_url="http://doris:8188",
        workflow_template=Path("workflow_api.json"),
        chunks_dir=tmp_path / "chunks",
        final_video_dir=tmp_path / "final",
        hardware=PROFILE_RTX_4090_24GB,
    )
    prompt = expand_prompt(config, chunks[1])

    session = FakeComfyUISession()
    stager = ComfyUIAssetStager(base_url=session.base_url, session=session)
    assets = stager.stage_chunk(prompt, chunks[1])
    assert assets.image_filename == "dianne_ref.png", (
        "diarization overwrote an authored tag all the way to the staged photograph -- "
        "a silent overwrite of an authored fact is the defect, not the fix"
    )
