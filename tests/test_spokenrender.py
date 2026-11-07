"""Tests for rendering a spoken script one shot at a time (#119).

Each shot is the speaker's reference image plus the finished line, started from its camera
setup's still with the line pinned at frame 0 (#120, #121): measured, a cut-back opened 3.7-5.0
from its still against 27.6-44.3 without it, and pinning the line was the first change that
moved lip-sync. Everything here runs offline: the graph is the committed template, the client,
stager and ffmpeg are fakes.
"""

from __future__ import annotations

import copy
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from music_video_maker import lipsync
from music_video_maker import spokenrender as sr
from music_video_maker import spokenscript as ss
from music_video_maker.workflow_graph import find_nodes_by_class_type, find_one_node

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "workflow_api.json"

SCRIPT = """
[script]
style = "Late afternoon, a dry hillside."

[cast.mara]
name = "Mara"
appearance = "weathered woman in her forties"
voice = "low, dry"
portrait = "cast/mara.png"

[cast.teo]
name = "Teo"
appearance = "thin young man"
voice = "high, quick"
portrait = "cast/teo.png"

[setup.mara_single]
characters = ["mara"]
posture = { mara = "seated" }
still = "stills/mara_single.png"
framing = "close"

[setup.teo_single]
characters = ["teo"]
posture = { teo = "standing" }
still = "stills/teo_single.png"
framing = "close"

[[shot]]
id = "s1"
seconds = 3
characters = ["mara"]
speaker = "mara"
line = "Quick question."
text = "Medium close-up of Mara seated, raising a hand."
posture = { mara = "seated" }
setup = "mara_single"
[shot.delivery]
pace = "normal"
pitch = "low"
volume = "normal"

[[shot]]
id = "s2"
seconds = 4
characters = ["teo"]
text = "Teo looks out over the hill, saying nothing."
posture = { teo = "standing" }
setup = "teo_single"

[[shot]]
id = "s3"
seconds = 3
characters = ["teo"]
speaker = "teo"
line = "Which one?"
text = "Medium close-up of Teo, patient."
posture = { teo = "standing" }
setup = "teo_single"
[shot.delivery]
pace = "brisk"
pitch = "mid"
volume = "normal"
pause_before = 0.3
"""


def write_wav(path: Path, seconds: float, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "script.toml").write_text(SCRIPT)
    for rel in ("cast/mara.png", "cast/teo.png", "stills/mara_single.png", "stills/teo_single.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"png:" + rel.encode())
    write_wav(tmp_path / "lines" / "s1.wav", 1.29)
    write_wav(tmp_path / "lines" / "s3.wav", 1.39)
    return tmp_path


def template() -> dict:
    return json.loads(TEMPLATE.read_text())


def renumbered(workflow: dict, offset: int = 100) -> dict:
    """The same graph with every node id moved -- what a canvas edit does."""
    def remap(value):
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
            return [str(int(value[0]) + offset), value[1]]
        return value

    return {str(int(k) + offset): {**n, "inputs": {a: remap(b) for a, b in n["inputs"].items()}}
            for k, n in workflow.items()}


# --------------------------------------------------------------------------- frames


def test_a_short_line_renders_at_the_trained_floor():
    assert sr.shot_frames(1.29 + sr.TAIL_SECONDS) == 124


def test_a_longer_shot_rounds_up_onto_the_grid():
    assert sr.shot_frames(5.5) == 141          # 132 frames needed -> next 5+17k


def test_a_shot_past_the_trained_ceiling_is_refused():
    with pytest.raises(sr.SpokenRenderError, match="362"):
        sr.shot_frames(16.0)


def test_speech_seconds_reads_the_wav(tmp_path):
    assert sr.speech_seconds(write_wav(tmp_path / "a.wav", 1.5)) == pytest.approx(1.5)


# --------------------------------------------------------------------------- the anchor


def _check_anchor(workflow: dict, still: str, line: str | None) -> None:
    ref_id, ref = find_one_node(workflow, "MiniMaxH3ReferenceToVideo")
    guide_id, guide = find_one_node(workflow, "MiniMaxH3AddGuide")
    _, guider = find_one_node(workflow, "BasicGuider")
    assert guider["inputs"]["conditioning"] == [guide_id, 0]
    gi = guide["inputs"]
    assert gi["positive"] == [ref_id, 0] and gi["latent"] == [ref_id, 1]
    assert gi["frame_idx"] == 0 and gi["vae"] == ref["inputs"]["vae"]
    assert workflow[gi["image"][0]] == {"class_type": "LoadImage", "inputs": {"image": still}}
    if line is None:
        assert "audio" not in gi and "audio_vae" not in gi
    else:
        assert gi["audio_vae"] == ref["inputs"]["audio_vae"]
        assert workflow[gi["audio"][0]] == {"class_type": "LoadAudio", "inputs": {"audio": line}}
    # the sampler still takes its latent from the reference node
    _, sampler = find_one_node(workflow, "SamplerCustomAdvanced")
    assert sampler["inputs"]["latent_image"] == [ref_id, 1]


def test_anchor_pins_the_still_and_the_line_at_frame_zero():
    wf = sr.anchor_setup(template(), still="still.png", line="line.wav")
    _check_anchor(wf, "still.png", "line.wav")


def test_anchor_without_a_line_pins_only_the_still():
    _check_anchor(sr.anchor_setup(template(), still="still.png", line=None), "still.png", None)


def test_anchor_finds_nodes_by_class_type_not_id():
    wf = sr.anchor_setup(renumbered(template()), still="s.png", line="l.wav")
    _check_anchor(wf, "s.png", "l.wav")


def test_anchor_refuses_a_second_anchor():
    wf = sr.anchor_setup(template(), still="a.png", line=None)
    with pytest.raises(sr.SpokenRenderError, match="already"):
        sr.anchor_setup(wf, still="b.png", line=None)


# --------------------------------------------------------------------------- the shot's graph


def _inputs_of(workflow: dict, class_type: str) -> dict:
    return find_one_node(workflow, class_type)[1]["inputs"]


def test_shot_workflow_sets_prompt_size_length_seed_and_prefix():
    wf = sr.build_shot_workflow(template(), prompt="P", width=864, height=480, frames=124, seed=7,
                                images=["mara.png"], line="s1.wav", still="still.png",
                                prefix="spoken_s1")
    ref = _inputs_of(wf, "MiniMaxH3ReferenceToVideo")
    assert (ref["prompt"], ref["width"], ref["height"], ref["length"]) == ("P", 864, 480, 124)
    assert _inputs_of(wf, "RandomNoise")["noise_seed"] == 7
    assert _inputs_of(wf, "SaveVideo")["filename_prefix"] == "spoken_s1"
    assert wf[ref["ref_images.ref_image_0"][0]]["inputs"]["image"] == "mara.png"
    ref_audio = wf[ref["ref_audios.ref_audio_0"][0]]
    assert (ref_audio["class_type"], ref_audio["inputs"]) == ("LoadAudio", {"audio": "s1.wav"})
    _check_anchor(wf, "still.png", "s1.wav")


def test_shot_workflow_does_not_touch_the_template():
    base = template()
    before = copy.deepcopy(base)
    sr.build_shot_workflow(base, prompt="P", width=864, height=480, frames=124, seed=1,
                           images=["a.png"], line="l.wav", still="s.png", prefix="x")
    assert base == before


def test_a_silent_shot_has_no_reference_audio_and_no_pinned_line():
    wf = sr.build_shot_workflow(template(), prompt="P", width=864, height=480, frames=124, seed=1,
                                images=["teo.png"], line=None, still="still.png", prefix="x")
    ref = _inputs_of(wf, "MiniMaxH3ReferenceToVideo")
    assert not any(k.startswith("ref_audios.") for k in ref)
    loads = [n["inputs"]["audio"] for _, n in find_nodes_by_class_type(wf, "LoadAudio")]
    assert loads == []
    _check_anchor(wf, "still.png", None)


def test_extra_people_in_frame_become_further_reference_images():
    wf = sr.build_shot_workflow(template(), prompt="P", width=864, height=480, frames=124, seed=1,
                                images=["a.png", "b.png"], line=None, still="s.png", prefix="x")
    ref = _inputs_of(wf, "MiniMaxH3ReferenceToVideo")
    assert wf[ref["ref_images.ref_image_1"][0]]["inputs"]["image"] == "b.png"


def test_shot_prompt_names_the_speaker_first_and_never_carries_the_line(project):
    script = ss.load_script(project / "script.toml")
    shot = script.shots[0]
    prompt = sr.shot_prompt(script, shot)
    assert prompt.startswith(ss.compose_generation_prompt(script, shot))
    assert prompt.endswith("<Picture 1> is the person speaking, and <Audio 1> is their voice.")
    assert shot.line not in prompt
    assert "<Audio 1>" not in sr.shot_prompt(script, script.shots[1])
    assert ss.lint_generation_prompt(prompt) == []


def test_reference_images_put_the_speaker_first(project):
    script = ss.load_script(project / "script.toml")
    assert sr.reference_cast(script.shots[0]) == ("mara",)


# --------------------------------------------------------------------------- planning


def test_plan_reads_each_line_and_sizes_each_shot(project):
    script = ss.load_script(project / "script.toml")
    plans = sr.plan_shots(script, project / "lines")
    assert [(p.shot.id, p.frames) for p in plans] == [("s1", 124), ("s2", 124), ("s3", 124)]
    assert plans[0].keep_seconds == pytest.approx(1.29 + sr.TAIL_SECONDS)
    assert plans[1].line is None and plans[1].keep_seconds == 4
    assert plans[2].line == project / "lines" / "s3.wav"


def test_plan_refuses_a_missing_line_before_anything_renders(project):
    (project / "lines" / "s3.wav").unlink()
    script = ss.load_script(project / "script.toml")
    with pytest.raises(sr.SpokenRenderError, match="s3"):
        sr.plan_shots(script, project / "lines")


def test_plan_can_be_limited_to_some_shots(project):
    script = ss.load_script(project / "script.toml")
    assert [p.shot.id for p in sr.plan_shots(script, project / "lines", only={"s3"})] == ["s3"]
    with pytest.raises(sr.SpokenRenderError, match="zz"):
        sr.plan_shots(script, project / "lines", only={"zz"})


# --------------------------------------------------------------------------- assembly argv


def test_segment_trims_to_the_line_holds_the_pause_and_carries_only_the_line(tmp_path):
    args = sr.build_segment_args(Path("t.mp4"), Path("l.wav"), Path("o.mp4"), keep_seconds=1.79,
                                 pause_before=0.3, width=864, height=480)
    vf = args[args.index("-vf") + 1]
    af = args[args.index("-af") + 1]
    assert "tpad=start_mode=clone:start_duration=0.3" in vf
    assert af.startswith("adelay=300|300,apad=whole_dur=2.090")
    assert args[args.index("-t") + 1] == "2.090"
    assert args[args.index("-map"):args.index("-map") + 4] == ["-map", "0:v", "-map", "1:a"]
    assert args[-1] == "o.mp4"


def test_a_silent_segment_gets_silence_not_the_models_audio(tmp_path):
    args = sr.build_segment_args(Path("t.mp4"), None, Path("o.mp4"), keep_seconds=4.0,
                                 pause_before=0.0, width=864, height=480)
    assert "anullsrc=r=48000:cl=stereo" in args
    assert "tpad" not in args[args.index("-vf") + 1]


# --------------------------------------------------------------------------- the render loop


@dataclass
class FakeResult:
    video_file: Path
    prompt_id: str


class FakeClient:
    def __init__(self, fail_on: str | None = None) -> None:
        self.workflows: list[dict] = []
        self.freed = 0
        self.fail_on = fail_on

    def execute(self, workflow, chunk_id, output_dir):
        self.workflows.append(workflow)
        prefix = find_one_node(workflow, "SaveVideo")[1]["inputs"]["filename_prefix"]
        if self.fail_on and prefix.endswith(self.fail_on):
            raise RuntimeError("render failed")
        out = Path(output_dir) / f"chunk_{chunk_id:04d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4:" + prefix.encode())
        return FakeResult(out, f"pid-{chunk_id}")

    def free(self, *, unload_models=True):
        self.freed += 1


class FakeStager:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload_image(self, path):
        self.uploads.append(Path(path).name)
        return f"up_{Path(path).name}"

    def upload_audio(self, path):
        self.uploads.append(Path(path).name)
        return f"up_{Path(path).name}"


def fake_runner(args):
    Path(args[-1]).write_bytes(b"ffmpeg")
    return subprocess.CompletedProcess(args, 0, b"", b"")


def fake_check(still, video):
    return sr.setupmatch.SetupMatch(Path(still), Path(video), 0, 4.2, 10.0)


def run(project: Path, client: FakeClient, **kw):
    script = ss.load_script(project / "script.toml")
    plans = sr.plan_shots(script, project / "lines", only=kw.pop("only", None))
    return sr.render_shots(script, plans, template=template(), client=client, stager=FakeStager(),
                           out_dir=project / "out", seed=11, width=864, height=480,
                           runner=fake_runner, check=fake_check, **kw)


def test_render_writes_each_take_records_provenance_and_frees_the_card(project):
    client = FakeClient()
    done = run(project, client)
    assert [r.shot_id for r in done] == ["s1", "s2", "s3"]
    assert all((project / "out" / "takes" / f"{r.shot_id}.mp4").is_file() for r in done)
    assert all(r.setup_match.matches for r in done)
    assert client.freed == 1
    prov = json.loads((project / "out" / "provenance.json").read_text())
    s1 = next(s for s in prov["shots"] if s["shot_id"] == "s1")
    assert s1["backend"] == lipsync.BACKEND_LOCAL_H3 and s1["tier"] == "draft"
    assert s1["frames"] == 124 and s1["resolution"] == "864x480"
    assert set(s1["inputs"]) == {"reference_image", "reference_audio", "setup_still"}
    assert s1["notes"]["setup"] == "mara_single" and s1["notes"]["setupmatch_distance"] == 4.2


def test_render_uses_the_portrait_the_still_and_the_padded_line(project):
    client = FakeClient()
    run(project, client, only={"s1"})
    wf = client.workflows[0]
    ref = _inputs_of(wf, "MiniMaxH3ReferenceToVideo")
    assert wf[ref["ref_images.ref_image_0"][0]]["inputs"]["image"] == "up_mara.png"
    assert wf[ref["ref_audios.ref_audio_0"][0]]["inputs"]["audio"] == "up_s1_pad.wav"
    _check_anchor(wf, "up_mara_single.png", "up_s1_pad.wav")


def test_seeds_differ_per_shot_and_are_reproducible(project):
    a, b = FakeClient(), FakeClient()
    run(project, a)
    run(project, b)
    seeds = [_inputs_of(w, "RandomNoise")["noise_seed"] for w in a.workflows]
    assert len(set(seeds)) == 3
    assert seeds == [_inputs_of(w, "RandomNoise")["noise_seed"] for w in b.workflows]


def test_a_failed_shot_still_frees_the_card_and_keeps_earlier_provenance(project):
    client = FakeClient(fail_on="s2")
    with pytest.raises(RuntimeError):
        run(project, client)
    assert client.freed == 1
    prov = json.loads((project / "out" / "provenance.json").read_text())
    assert [s["shot_id"] for s in prov["shots"]] == ["s1"]


def test_assemble_concats_segments_in_script_order(project):
    done = run(project, FakeClient())
    calls: list[list[str]] = []

    def runner(args):
        calls.append(list(args))
        return fake_runner(args)

    script = ss.load_script(project / "script.toml")
    out = sr.assemble(script, done, project / "out", width=864, height=480, runner=runner)
    assert out == project / "out" / "spoken_cut.mp4"
    concat = (project / "out" / "segments" / "concat.txt").read_text().splitlines()
    assert [Path(line.split("'")[1]).stem for line in concat] == ["s1", "s2", "s3"]
    assert calls[-1][-1] == str(out)


# --------------------------------------------------------------------------- the CLI gates


def test_cli_refuses_without_the_h3_authorization(project, capsys):
    code = sr.main([str(project / "script.toml"), "--lines", str(project / "lines"),
                    "--out", str(project / "out"), "--h3-authorization", str(project / "nope.pdf")])
    assert code == 2
    assert "locked" in capsys.readouterr().err


def test_cli_refuses_a_script_that_is_not_render_ready(project, capsys):
    (project / "stills" / "teo_single.png").unlink()
    auth = project / "auth.pdf"
    auth.write_bytes(b"x")
    code = sr.main([str(project / "script.toml"), "--lines", str(project / "lines"),
                    "--out", str(project / "out"), "--h3-authorization", str(auth)])
    assert code == 2
    assert "setup-still-missing" in capsys.readouterr().err


# --------------------------------------------------------------------------- framing


def test_the_script_framing_vocabulary_is_the_render_path_one():
    from music_video_maker import prompting
    assert ss.FRAMINGS == prompting.FRAMING_LEVELS


def test_the_setups_framing_sentence_follows_the_description_and_precedes_the_tag(project):
    script = ss.load_script(project / "script.toml")
    prompt = sr.shot_prompt(script, script.shots[0])
    assert prompt.endswith("Medium close-up of Mara seated, raising a hand. Mara is framed head "
                           "and shoulders, the face taking up most of the frame." + sr.SPEAKER_TAG)


def test_a_description_without_a_full_stop_gets_one_before_the_framing(project):
    script_file = project / "script.toml"
    text = script_file.read_text().replace("Teo, patient.", "Teo, patient")
    script_file.write_text(text)
    script = ss.load_script(script_file)
    prompt = sr.shot_prompt(script, script.shots[2])
    assert "Teo, patient. Teo is framed head and shoulders" in prompt


def test_a_silent_shot_frames_whoever_its_setup_shows(project):
    script = ss.load_script(project / "script.toml")
    prompt = sr.shot_prompt(script, script.shots[1])
    assert prompt.endswith("Teo is framed head and shoulders, the face taking up most of "
                           "the frame.")


def test_a_setup_without_framing_composes_none(project):
    text = (project / "script.toml").read_text().replace('framing = "close"\n', "")
    (project / "script.toml").write_text(text)
    script = ss.load_script(project / "script.toml")
    assert "framed" not in sr.shot_prompt(script, script.shots[1])


EMPTY_SHOT = """
[setup.empty_hill]
characters = []
posture = {}
still = "stills/empty_hill.png"
framing = "wide"

[[shot]]
id = "s4"
seconds = 3
characters = []
text = "The empty hillside at dusk, grass moving."
posture = {}
setup = "empty_hill"
"""


def test_a_shot_with_nobody_in_it_references_its_own_still(project):
    with (project / "script.toml").open("a") as f:
        f.write(EMPTY_SHOT)
    (project / "stills" / "empty_hill.png").write_bytes(b"png:empty")
    client = FakeClient()
    run(project, client, only={"s4"})
    wf = client.workflows[0]
    ref = _inputs_of(wf, "MiniMaxH3ReferenceToVideo")
    assert wf[ref["ref_images.ref_image_0"][0]]["inputs"]["image"] == "up_empty_hill.png"
    _check_anchor(wf, "up_empty_hill.png", None)
    prov = json.loads((project / "out" / "provenance.json").read_text())
    s4 = prov["shots"][0]["inputs"]
    assert s4["reference_image"] == s4["setup_still"]
