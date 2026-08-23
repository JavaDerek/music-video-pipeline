"""The two places ``run_pipeline`` reaches into the roadmap modules.

Both are one-line hooks, and both are the kind of wiring that is easy to
believe is present because the module underneath is well tested. It is not
the same claim: ``assembly`` accepting ``master_audio=None`` says nothing
about whether any run ever passes ``None``, and ``profiles`` being able to
write a record says nothing about whether a render writes one. These tests
close that gap and nothing else.

The end-to-end rig lives in ``tests/test_cli.py`` -- real config, real
stager, mutator, execution client, resilient runner and assembly, with only
ComfyUI, ffmpeg, sleep, the stable-ts model and the clock faked. It is
imported rather than copied: a second rig would drift from the first, and
these tests are only meaningful against the one the rest of the CLI suite
already trusts.
"""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from pathlib import Path

from music_video_maker.assembly import DEFAULT_OUTPUT_FILENAME
from music_video_maker.config import load_config
from music_video_maker.profiles import PROFILE_RECORD_FILENAME, load_profile
from tests.test_cli import Rig, build_success_sequence

# --------------------------------------------------------------------------- #
# Issue #22: silent_output
# --------------------------------------------------------------------------- #


def _run_three_chunks(rig: Rig):
    sequences = [build_success_sequence(rig.seed_success(n, n - 1)) for n in (1, 2, 3)]
    return rig.run(sequences)


def _ffmpeg_calls(rig: Rig) -> list[list[str]]:
    """Only the calls that write a video -- not the ffprobe frame counts or
    the raw-frame luminance pipes issue #77 makes on the way past."""
    return [
        call
        for call in rig.ffmpeg.calls
        if call and call[0] == "ffmpeg" and call[-1] != "-" and "-vf" not in call
    ]


def test_default_run_muxes_the_master_audio_over_the_video(tmp_path: Path):
    """The control for the test below. Two ffmpeg passes, the second mapping
    audio in from a second input."""
    rig = Rig(tmp_path)
    report = _run_three_chunks(rig)

    calls = _ffmpeg_calls(rig)
    assert len(calls) == 2, calls
    assert [calls[1][i] for i in (calls[1].index("-map"), calls[1].index("1:a"))] == ["-map", "1:a"]
    assert report.output_video.is_file()


def test_silent_output_produces_one_pass_and_no_audio_mapping(tmp_path: Path):
    """Issue #22: the band is the audio. A file with an audio track risks
    double-audio if a playback rig un-mutes it, so there must be no mux at
    all -- not a mux of silence."""
    rig = Rig(tmp_path)
    rig.config = dc_replace(rig.config, silent_output=True)
    report = _run_three_chunks(rig)

    calls = _ffmpeg_calls(rig)
    assert len(calls) == 1, calls
    concat = calls[0]
    assert "-map" not in concat
    # -an survives: "generated audio is always discarded" is not the invariant
    # this flag suspends, and H3's own audio must not leak through.
    assert "-an" in concat
    assert concat[-1] == str(rig.config.final_video_dir / DEFAULT_OUTPUT_FILENAME)
    assert report.output_video == rig.config.final_video_dir / DEFAULT_OUTPUT_FILENAME
    assert report.output_video.is_file()


def _minimal_config_toml(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "audio").mkdir(exist_ok=True)
    (tmp_path / "audio" / "master.wav").write_bytes(b"RIFF")
    (tmp_path / "lyrics.txt").write_text("a line\n")
    (tmp_path / "cast").mkdir(exist_ok=True)
    (tmp_path / "cast" / "ref.png").write_bytes(b"\x89PNG")
    (tmp_path / "workflow_api.json").write_text("{}")
    path = tmp_path / "run.toml"
    path.write_text(
        'master_audio = "audio/master.wav"\n'
        'lyrics_file = "lyrics.txt"\n'
        'global_style = "s"\n'
        'narrative_concept = "n"\n'
        'default_lead_vocalist = "Dianne"\n'
        'workflow_template = "workflow_api.json"\n'
        'chunks_dir = "output/chunks"\n'
        'final_video_dir = "output/final"\n'
        f"{extra}"
        "\n[cast.Dianne]\n"
        'role = "Lead Vocalist"\n'
        'image = "cast/ref.png"\n'
        "\n[hardware]\n"
        'name = "RTX 4090"\n'
        "vram_gb = 24.0\n"
    )
    return path


def test_silent_output_defaults_false(tmp_path: Path):
    assert load_config(_minimal_config_toml(tmp_path)).silent_output is False


def test_silent_output_reads_true_from_the_config_file(tmp_path: Path):
    config = load_config(_minimal_config_toml(tmp_path, "silent_output = true\n"))
    assert config.silent_output is True


# --------------------------------------------------------------------------- #
# Issue #55: the resolved profile is recorded beside the run's outputs
# --------------------------------------------------------------------------- #


def _write_profile(tmp_path: Path) -> Path:
    path = tmp_path / "house.toml"
    path.write_text(
        'version = 3\n'
        'name = "house"\n'
        'cinematography = "35mm anamorphic, cold slate grade"\n'
        'lora_strength = 0.8\n'
        "\n[provenance]\n"
        'source = "authored"\n'
    )
    return path


def test_a_run_with_a_profile_records_it_verbatim_beside_run_state(tmp_path: Path):
    """A hash proves a look *changed*; six months on, with the profile at v4,
    only this file can say what the look *was*."""
    rig = Rig(tmp_path)
    profile = load_profile(_write_profile(tmp_path))
    # The rig builds a RunConfig directly, so it has to stand in for what
    # load_config would have done: inherit the profile's fields onto the
    # config. That inheritance is tested against the real loader in
    # tests/test_profiles.py; what is being tested here is only that a *run*
    # writes the record.
    rig.config = dc_replace(
        rig.config,
        cinematography_profile=profile,
        cinematography=profile.values["cinematography"],
        lora_strength=profile.values["lora_strength"],
    )

    _run_three_chunks(rig)

    record_path = rig.config.chunks_dir / PROFILE_RECORD_FILENAME
    assert record_path.is_file()
    record = json.loads(record_path.read_text())
    assert record["profile"]["name"] == "house"
    assert record["profile"]["version"] == 3
    assert record["profile"]["values"]["cinematography"] == "35mm anamorphic, cold slate grade"
    assert record["profile_sha256"] == profile.sha256
    assert record["effective"]["cinematography"] == "35mm anamorphic, cold slate grade"
    assert record["overridden_by_run_config"] == []


def test_the_record_names_the_fields_the_run_config_took_back(tmp_path: Path):
    rig = Rig(tmp_path)
    profile = load_profile(_write_profile(tmp_path))
    rig.config = dc_replace(
        rig.config,
        cinematography_profile=profile,
        cinematography="something else",
        # `overridden` is not recomputable from a resolved RunConfig -- see
        # RunConfig.cinematography_profile_overrides. load_config records it
        # while the raw TOML is still visible; the rig builds a RunConfig by
        # hand, so it states it by hand too.
        cinematography_profile_overrides=("cinematography",),
    )

    _run_three_chunks(rig)

    record = json.loads((rig.config.chunks_dir / PROFILE_RECORD_FILENAME).read_text())
    assert record["overridden_by_run_config"] == ["cinematography"]
    assert record["effective"]["cinematography"] == "something else"
    # The profile's own text is still there verbatim, which is the point: the
    # record has to show what was inherited AND what beat it.
    assert record["profile"]["values"]["cinematography"] == "35mm anamorphic, cold slate grade"


def test_no_profile_writes_no_record(tmp_path: Path):
    rig = Rig(tmp_path)
    _run_three_chunks(rig)
    assert not (rig.config.chunks_dir / PROFILE_RECORD_FILENAME).exists()


def test_a_failing_record_write_never_aborts_the_run(tmp_path: Path, monkeypatch, caplog):
    """Provenance is worth a log line, never a lost render: at ~3.7 min/chunk
    an aborted run is hours, and the look is still recoverable from the
    profile file that ``prompt_hash`` pins."""
    rig = Rig(tmp_path)
    profile = load_profile(_write_profile(tmp_path))
    rig.config = dc_replace(rig.config, cinematography_profile=profile)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("music_video_maker.cli.write_profile_record", _boom)

    with caplog.at_level("ERROR"):
        report = _run_three_chunks(rig)

    assert report.rendered == 3
    assert report.output_video.is_file()
    assert "cinematography profile record" in caplog.text
