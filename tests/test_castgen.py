"""Tests for the synthetic-cast *generator* (issue #56, part 2).

Four layers, and every one of them runs with no GPU, no weights, no network
and no ComfyUI:

* **the spec**, which is where the design doc's "do not build the set by
  re-rolling one prompt at different seeds" stops being advice and becomes a
  refusal;
* **planning**, spec + the two committed templates -> one submittable
  workflow per view, located by ``class_type`` and never by node id (there is
  a test that renumbers every node in both templates and expects identical
  output);
* **the committed templates themselves**, treated as data under test -- a
  Krea 2 graph that has lost its ``type = "krea2"`` encoder, or grown a
  second ``LoadImage``, or acquired a billed hosted-API node, is a defect
  this suite catches rather than one a human discovers after taking custody
  of the card;
* **generation**, against ``tests/harness/comfyui_mock.FakeComfyUISession``,
  including the unhappy paths that matter most for a tool that holds a shared
  4090: a missing weights file, an insufficient VRAM reading, a graph ComfyUI
  rejects, a prompt that finishes with no image, and a poll budget that runs
  out. ``POST /free`` is asserted on the failure paths, not just the happy
  one.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest
import requests

from music_video_maker import castcheck, castgen, config, custody, execution, faces
from music_video_maker import workflow_graph as wg
from tests.harness.comfyui_mock import FakeComfyUISession, make_fake_png_bytes

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_TEMPLATE_PATH = REPO_ROOT / castgen.ANCHOR_TEMPLATE_FILENAME
VIEW_TEMPLATE_PATH = REPO_ROOT / castgen.VIEW_TEMPLATE_FILENAME

INSTALLED = {
    "UNETLoader": {"unet_name": ["krea2_raw_bf16.safetensors"]},
    "CLIPLoader": {"clip_name": ["qwen3vl_4b_fp8_scaled.safetensors"]},
    "VAELoader": {"vae_name": ["qwen_image_vae.safetensors"]},
}


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def anchor_template() -> dict:
    return wg.load_workflow_template(ANCHOR_TEMPLATE_PATH)


@pytest.fixture
def view_template() -> dict:
    return wg.load_workflow_template(VIEW_TEMPLATE_PATH)


SPEC_TOML = """
character = "Nobody"
prompt = "studio portrait photograph of a woman in her thirties, short dark hair"
seed = 4242
model = "krea2-raw"
steps = 28
cfg = 4.0

[[views]]
name = "frontal"
prompt = "looking straight into the lens, neutral expression, even soft light"

[[views]]
name = "three_quarter"
from = "frontal"
denoise = 0.45
prompt = "head turned thirty degrees to camera left, same person, same face"
"""


def write_spec(tmp_path: Path, body: str = SPEC_TOML) -> Path:
    path = tmp_path / "nobody.castgen.toml"
    path.write_text(body, encoding="utf-8")
    return path


def seed_installed(session: FakeComfyUISession) -> None:
    for class_type, enums in INSTALLED.items():
        session.seed_object_info(class_type, enums)


def seed_image_history(session: FakeComfyUISession, prompt_id: str, filename: str) -> None:
    """Seed a finished ``/history`` entry whose output is an *image*.

    ``seed_history_success``'s parameters are named for the render path's
    videos; ``output_key="images"`` is the same seam real ComfyUI uses for a
    ``SaveImage`` node, so this needs no second harness method.
    """
    session.seed_history_success(
        prompt_id,
        video_filename=filename,
        video_bytes=make_fake_png_bytes(1024, 1024),
        node_id="9",
        output_key="images",
    )


def pass_checker(character: str, images) -> castcheck.ConsistencyReport:
    """An injected ``castcheck`` run with fakes -- the real report logic, no
    cv2, no SFace weights."""
    paths = list(images)
    return castcheck.check_consistency(
        character,
        paths,
        detector=lambda _path: faces.FaceObservation(
            face_count=1, largest_fraction=0.2, score=0.95
        ),
        recognizer=lambda _a, _b: 0.61,
    )


# --------------------------------------------------------------------------- #
# The spec: what it accepts, and what it refuses before any GPU time
# --------------------------------------------------------------------------- #


def test_spec_loads_and_composes_each_views_own_prompt(tmp_path):
    spec = castgen.load_character_spec(write_spec(tmp_path))

    assert spec.character == "Nobody"
    assert spec.seed == 4242
    assert spec.steps == 28
    assert spec.cfg == pytest.approx(4.0)
    assert [view.name for view in spec.views] == ["frontal", "three_quarter"]
    assert spec.anchor.from_view is None
    assert spec.views[1].from_view == "frontal"
    assert spec.views[1].denoise == pytest.approx(0.45)
    # Every view carries the base prompt plus its own clause, in full, because
    # that is what the per-image provenance record has to be able to replay.
    assert spec.views[0].prompt.startswith("studio portrait photograph")
    assert "even soft light" in spec.views[0].prompt
    assert "thirty degrees" in spec.views[1].prompt
    # Defaults come from ComfyUI's own Krea 2 blueprint, not from here.
    assert (spec.sampler_name, spec.scheduler) == ("euler", "simple")
    assert (spec.width, spec.height) == (1024, 1024)


def test_a_single_view_set_is_refused_because_castcheck_cannot_score_it(tmp_path):
    body = """
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "only"
"""
    with pytest.raises(castgen.CharacterSpecError, match="at least two views"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_a_second_view_with_no_from_is_refused_as_a_re_roll(tmp_path, caplog):
    body = """
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "frontal"
[[views]]
name = "three_quarter"
"""
    with caplog.at_level(logging.ERROR), pytest.raises(castgen.CharacterSpecError) as excinfo:
        castgen.load_character_spec(write_spec(tmp_path, body))

    # The refusal has to say why, because "it worked for me" is what a seed
    # re-roll looks like until someone compares the faces.
    assert "four different people" in str(excinfo.value)
    assert "re-roll" in caplog.text or "four different people" in caplog.text


@pytest.mark.parametrize(
    ("denoise_line", "expected"),
    [
        ("", "must state its own denoise"),
        ("denoise = 1.0", "between 0 and 1"),
        ("denoise = 1.5", "between 0 and 1"),
        ("denoise = 0.0", "between 0 and 1"),
    ],
)
def test_a_derived_view_needs_a_denoise_strictly_inside_zero_and_one(
    tmp_path, denoise_line, expected
):
    body = f"""
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "frontal"
[[views]]
name = "three_quarter"
from = "frontal"
{denoise_line}
"""
    with pytest.raises(castgen.CharacterSpecError, match=expected):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_from_must_name_an_earlier_view(tmp_path):
    body = """
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "frontal"
[[views]]
name = "three_quarter"
from = "graded"
denoise = 0.4
[[views]]
name = "graded"
from = "frontal"
denoise = 0.4
"""
    with pytest.raises(castgen.CharacterSpecError, match="must name an earlier view"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_the_anchor_may_not_be_derived_or_carry_a_denoise(tmp_path):
    body = """
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "frontal"
denoise = 0.5
[[views]]
name = "other"
from = "frontal"
denoise = 0.4
"""
    with pytest.raises(castgen.CharacterSpecError, match="views\\[0\\].denoise"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_duplicate_view_names_are_refused(tmp_path):
    body = """
character = "Nobody"
prompt = "a portrait"
seed = 1
steps = 20
cfg = 4.0
[[views]]
name = "frontal"
[[views]]
name = "frontal"
from = "frontal"
denoise = 0.4
"""
    with pytest.raises(castgen.CharacterSpecError, match="duplicate view name"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    body = SPEC_TOML + "\nsteeps = 40\n"
    with pytest.raises(castgen.CharacterSpecError, match="unknown key"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_an_unknown_view_key_is_refused_so_a_typo_cannot_become_a_generation(tmp_path):
    body = SPEC_TOML.replace("denoise = 0.45", "denoise = 0.45\ndenoize = 0.9")
    with pytest.raises(castgen.CharacterSpecError, match="unknown key"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_a_missing_or_malformed_spec_is_a_spec_error_not_a_traceback(tmp_path):
    with pytest.raises(castgen.CharacterSpecError, match="not found"):
        castgen.load_character_spec(tmp_path / "nope.toml")

    broken = tmp_path / "broken.toml"
    broken.write_text("character = \n", encoding="utf-8")
    with pytest.raises(castgen.CharacterSpecError, match="malformed TOML"):
        castgen.load_character_spec(broken)


def test_required_scalars_are_type_checked(tmp_path):
    body = SPEC_TOML.replace("seed = 4242", 'seed = "4242"')
    with pytest.raises(castgen.CharacterSpecError, match="seed: must be an integer"):
        castgen.load_character_spec(write_spec(tmp_path, body))


def test_an_unrecorded_model_warns_and_records_its_licence_as_unrecorded(tmp_path, caplog):
    body = SPEC_TOML.replace('model = "krea2-raw"', 'model = "something-else"')
    with caplog.at_level(logging.WARNING):
        spec = castgen.load_character_spec(write_spec(tmp_path, body))

    assert "KNOWN_IMAGE_MODELS" in caplog.text
    assert "UNRECORDED" in str(spec.licence_record["licence"])


def test_the_krea2_licence_record_carries_the_three_terms_that_reach_the_code():
    record = castgen.KNOWN_IMAGE_MODELS["krea2-raw"]
    licence = str(record["licence"])
    assert "$1M" in licence and "50 seats" in licence
    assert "Krea" in licence  # derivative naming rule
    assert "deployer" in licence  # filtering responsibility
    assert record["source"].startswith("https://huggingface.co/krea/")
    # The disclosure that goes into every manifest must not be mistakable for
    # "a filter ran".
    assert castgen.FILTERING_DISCLOSURE.startswith("none")


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_plan_views_injects_everything_the_render_must_not_decide(
    tmp_path, anchor_template, view_template
):
    spec = castgen.load_character_spec(write_spec(tmp_path))
    plans = castgen.plan_views(
        spec, anchor_template=anchor_template, view_template=view_template
    )

    assert [plan.name for plan in plans] == ["frontal", "three_quarter"]
    assert [plan.derived for plan in plans] == [False, True]
    # Distinct, derived, recorded seeds -- a view left on whatever the
    # template carried is an image nobody can make again.
    assert [plan.seed for plan in plans] == [4242, 4243]

    anchor = plans[0].workflow
    _, sampler = wg.find_one_node(anchor, "KSampler")
    assert sampler["inputs"]["seed"] == 4242
    assert sampler["inputs"]["steps"] == 28
    assert sampler["inputs"]["cfg"] == pytest.approx(4.0)
    assert sampler["inputs"]["denoise"] == pytest.approx(1.0)
    _, latent = wg.find_one_node(anchor, "EmptyLatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (1024, 1024)
    _, encode = wg.find_one_node(anchor, "CLIPTextEncode")
    assert "even soft light" in encode["inputs"]["text"]
    _, save = wg.find_one_node(anchor, "SaveImage")
    assert save["inputs"]["filename_prefix"] == "mvm_cast_Nobody_frontal"

    view = plans[1].workflow
    _, view_sampler = wg.find_one_node(view, "KSampler")
    assert view_sampler["inputs"]["denoise"] == pytest.approx(0.45)
    assert view_sampler["inputs"]["seed"] == 4243
    wg.find_one_node(view, "LoadImage")  # the identity anchor's own socket
    wg.find_one_node(view, "VAEEncode")


def test_plan_views_never_mutates_the_templates_it_was_given(
    tmp_path, anchor_template, view_template
):
    spec = castgen.load_character_spec(write_spec(tmp_path))
    before = (
        json.dumps(anchor_template, sort_keys=True),
        json.dumps(view_template, sort_keys=True),
    )

    castgen.plan_views(spec, anchor_template=anchor_template, view_template=view_template)

    after = (json.dumps(anchor_template, sort_keys=True), json.dumps(view_template, sort_keys=True))
    assert before == after


def _renumber(workflow: dict) -> dict:
    """Renumber every node id (and every link) -- what a canvas edit does."""
    mapping = {old: f"9{index}00" for index, old in enumerate(workflow)}
    renumbered: dict = {}
    for old, node in workflow.items():
        clone = json.loads(json.dumps(node))
        for key, value in clone["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and value[0] in mapping:
                clone["inputs"][key] = [mapping[value[0]], value[1]]
        renumbered[mapping[old]] = clone
    return renumbered


def test_planning_is_independent_of_node_ids(tmp_path, anchor_template, view_template):
    spec = castgen.load_character_spec(write_spec(tmp_path))
    original = castgen.plan_views(
        spec, anchor_template=anchor_template, view_template=view_template
    )
    shifted = castgen.plan_views(
        spec,
        anchor_template=_renumber(anchor_template),
        view_template=_renumber(view_template),
    )

    def inputs_by_class(workflow: dict) -> dict:
        return {
            node["class_type"]: {
                key: value for key, value in node["inputs"].items() if not isinstance(value, list)
            }
            for node in workflow.values()
        }

    for left, right in zip(original, shifted, strict=True):
        assert inputs_by_class(left.workflow) == inputs_by_class(right.workflow)


def test_swapping_the_two_templates_is_refused_loudly(tmp_path, view_template):
    spec = castgen.load_character_spec(write_spec(tmp_path))
    with pytest.raises(castgen.CharacterSpecError, match="LoadImage"):
        castgen.plan_views(
            spec, anchor_template=view_template, view_template=view_template
        )


def test_a_hosted_krea_api_node_is_refused(tmp_path, anchor_template, view_template):
    spec = castgen.load_character_spec(write_spec(tmp_path))
    anchor_template["99"] = {"class_type": "Krea2ImageNode", "inputs": {"prompt": "x"}}
    with pytest.raises(castgen.HostedApiNodeError, match="Krea2ImageNode"):
        castgen.plan_views(
            spec, anchor_template=anchor_template, view_template=view_template
        )


def test_the_hosted_minimax_table_is_reused_not_restated(anchor_template):
    anchor_template["99"] = {"class_type": "MinimaxTextToVideoNode", "inputs": {}}
    with pytest.raises(wg.CloudApiNodeDetectedError):
        castgen.assert_no_hosted_api_nodes(anchor_template)


def test_weights_overrides_from_the_spec_win_over_the_template(
    tmp_path, anchor_template, view_template
):
    # A top-level key has to precede the [[views]] tables -- after them TOML
    # reads it as a key of the last view, which the spec loader then refuses
    # as an unknown view key (a real and useful refusal, tested above).
    body = SPEC_TOML.replace(
        "cfg = 4.0", 'cfg = 4.0\nunet_name = "krea2_turbo_fp8.safetensors"', 1
    )
    spec = castgen.load_character_spec(write_spec(tmp_path, body))
    plans = castgen.plan_views(
        spec, anchor_template=anchor_template, view_template=view_template
    )
    for plan in plans:
        _, loader = wg.find_one_node(plan.workflow, "UNETLoader")
        assert loader["inputs"]["unet_name"] == "krea2_turbo_fp8.safetensors"


def test_slugify_keeps_a_filename_safe(tmp_path):
    assert castgen.slugify("Kashay the Deathless") == "Kashay_the_Deathless"
    assert castgen.slugify("///") == "unnamed"


# --------------------------------------------------------------------------- #
# The committed templates, as data under test
# --------------------------------------------------------------------------- #


def test_both_committed_templates_are_a_valid_local_krea2_graph(
    anchor_template, view_template
):
    for workflow in (anchor_template, view_template):
        castgen.assert_no_hosted_api_nodes(workflow)
        for class_type in ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
                           "KSampler", "VAEDecode", "SaveImage"):
            wg.find_one_node(workflow, class_type)
        _, clip = wg.find_one_node(workflow, "CLIPLoader")
        # Without type='krea2' the DiT is handed 4096-wide conditioning it
        # cannot consume -- comfy/ldm/krea2/model.py raises and says exactly
        # this. Cheap to assert, expensive to discover on the card.
        assert clip["inputs"]["type"] == "krea2"

    # The anchor is text-to-image; the view template is img2img from a file.
    wg.find_one_node(anchor_template, "EmptyLatentImage")
    assert not wg.find_nodes_by_class_type(anchor_template, "LoadImage")
    wg.find_one_node(view_template, "LoadImage")
    wg.find_one_node(view_template, "VAEEncode")
    assert not wg.find_nodes_by_class_type(view_template, "EmptyLatentImage")


def test_the_two_templates_declare_the_same_weights(anchor_template, view_template):
    """A human editing one template and not the other would otherwise
    generate the anchor with one model and its views with another -- a set of
    two different people, with nothing in either file looking wrong."""
    for class_type, input_name in (
        ("UNETLoader", "unet_name"),
        ("CLIPLoader", "clip_name"),
        ("VAELoader", "vae_name"),
    ):
        _, anchor_node = wg.find_one_node(anchor_template, class_type)
        _, view_node = wg.find_one_node(view_template, class_type)
        assert anchor_node["inputs"][input_name] == view_node["inputs"][input_name]


def test_no_lora_is_committed_into_either_template(anchor_template, view_template):
    """The only Krea 2 LoRA installed on the render host is a *sketch style*
    adapter (``krea2_darkbrush.safetensors``; ComfyUI's own blueprint gives
    its trigger as "muted minimalist sketch style"), and #62 measured a LoRA
    and the prompt fighting over a photoreal face. The reference-conditioning
    adapter that would actually help identity is not installed. So neither
    template carries a LoRA node, and this test is the record of that being a
    decision rather than an omission."""
    for workflow in (anchor_template, view_template):
        assert not wg.find_nodes_by_class_type(workflow, "LoraLoaderModelOnly")
        assert not wg.find_nodes_by_class_type(workflow, "LoraLoader")


# --------------------------------------------------------------------------- #
# Pre-flight: installed weights
# --------------------------------------------------------------------------- #


def test_read_installed_weights_reads_the_servers_own_enum():
    session = FakeComfyUISession()
    seed_installed(session)
    assert castgen.read_installed_weights(
        session, session.base_url, "UNETLoader", "unet_name"
    ) == ("krea2_raw_bf16.safetensors",)


def test_read_installed_weights_degrades_on_an_unknown_class(caplog):
    session = FakeComfyUISession()
    with caplog.at_level(logging.WARNING):
        assert (
            castgen.read_installed_weights(
                session, session.base_url, "UNETLoader", "unet_name"
            )
            is None
        )
    assert "does not describe" in caplog.text


def test_preflight_refuses_a_weights_file_the_host_does_not_have(
    anchor_template, view_template, tmp_path
):
    session = FakeComfyUISession()
    seed_installed(session)
    session.seed_object_info("UNETLoader", {"unet_name": ["something_else.safetensors"]})
    spec = castgen.load_character_spec(write_spec(tmp_path))
    plans = castgen.plan_views(
        spec, anchor_template=anchor_template, view_template=view_template
    )

    with pytest.raises(castgen.MissingWeightsError) as excinfo:
        castgen.preflight_installed_weights(
            session, session.base_url, [plan.workflow for plan in plans]
        )

    message = str(excinfo.value)
    assert "krea2_raw_bf16.safetensors" in message
    assert "models/" in message  # names the remedy, not just the symptom


def test_preflight_degrades_when_the_server_will_not_say(
    anchor_template, view_template, tmp_path
):
    session = FakeComfyUISession()  # nothing seeded: 200 {} like the real server
    spec = castgen.load_character_spec(write_spec(tmp_path))
    plans = castgen.plan_views(
        spec, anchor_template=anchor_template, view_template=view_template
    )

    checked = castgen.preflight_installed_weights(
        session, session.base_url, [plan.workflow for plan in plans]
    )
    assert checked["UNETLoader.unet_name"] == "krea2_raw_bf16.safetensors"


# --------------------------------------------------------------------------- #
# Generation, end to end against the fake ComfyUI
# --------------------------------------------------------------------------- #


def seed_happy_path(session: FakeComfyUISession) -> FakeComfyUISession:
    seed_installed(session)
    seed_image_history(session, "prompt-0001", "mvm_cast_Nobody_frontal_00001_.png")
    seed_image_history(session, "prompt-0002", "mvm_cast_Nobody_three_quarter_00001_.png")
    return session


def generate(tmp_path, anchor_template, view_template, session=None, **overrides):
    """Drive a whole character against the fake ComfyUI.

    A session passed in is used *exactly as the test seeded it* -- the helper
    only seeds the happy path when it creates the session itself, because
    every unhappy-path test here is defined by what it deliberately did not
    seed.
    """
    session = session if session is not None else seed_happy_path(FakeComfyUISession())
    spec = overrides.pop("spec", None) or castgen.load_character_spec(write_spec(tmp_path))
    kwargs = {
        "output_dir": tmp_path / "cast",
        "anchor_template": anchor_template,
        "view_template": view_template,
        "session": session,
        "base_url": session.base_url,
        "checker": pass_checker,
    }
    kwargs.update(overrides)
    return session, castgen.generate_character(spec, **kwargs)


def test_generate_writes_the_set_and_its_provenance(tmp_path, anchor_template, view_template):
    session, result = generate(tmp_path, anchor_template, view_template)

    assert [image.view for image in result.images] == ["frontal", "three_quarter"]
    for image in result.images:
        assert image.path.exists()
        assert image.bytes_written > 0
        assert len(image.sha256) == 64
    assert result.images[0].from_view is None
    assert result.images[1].from_view == "frontal"
    assert result.images[1].denoise == pytest.approx(0.45)
    assert result.status == "pass"
    assert result.free_vram_gb is not None
    assert result.render_stack.comfyui_version is not None

    # The card is handed back, always.
    assert session.free_calls, "POST /free must release the card after a character"


def test_a_derived_view_is_img2img_from_the_previous_views_rendered_file(
    tmp_path, anchor_template, view_template
):
    session, result = generate(tmp_path, anchor_template, view_template)

    # The anchor's downloaded file is uploaded once, and the second
    # submission's LoadImage points at the server filename it came back as.
    assert [upload.original_filename for upload in session.uploads] == [
        result.images[0].path.name
    ]
    second = session.submitted_prompts[1]["workflow"]
    _, load_image = wg.find_one_node(second, "LoadImage")
    assert load_image["inputs"]["image"] == session.uploads[0].server_filename
    # ... and it really is the img2img graph, at the authored denoise.
    _, sampler = wg.find_one_node(second, "KSampler")
    assert sampler["inputs"]["denoise"] == pytest.approx(0.45)


def test_the_manifest_records_the_licence_the_filtering_and_the_floors_calibration(
    tmp_path, anchor_template, view_template
):
    _session, result = generate(tmp_path, anchor_template, view_template)
    manifest = result.manifest()

    assert "$1M" in str(manifest["licence"])
    assert manifest["filtering"] == castgen.FILTERING_DISCLOSURE
    assert manifest["model_record"]["source"].startswith("https://huggingface.co/krea/")
    assert manifest["consistency"]["floor"] == pytest.approx(faces.DEFAULT_MIN_FACE_SIMILARITY)
    assert "NOT re-derived" in manifest["consistency"]["floor_calibration"]
    assert "cannot detect it" in manifest["consistency"]["mode_collapse"]
    assert manifest["consistency"]["pairs_scored"] == 1
    assert manifest["method"].startswith("anchor")
    assert len(manifest["images"]) == 2
    assert manifest["weights_checked"]["CLIPLoader.clip_name"] == (
        "qwen3vl_4b_fp8_scaled.safetensors"
    )


def test_the_report_carries_both_caveats_every_time(tmp_path, anchor_template, view_template):
    _session, result = generate(tmp_path, anchor_template, view_template)
    text = result.render()

    assert "calibrated on photographs of REAL" in text
    assert "mode collapse" in text.lower()
    assert "RESULT: pass" in text
    assert "pairwise similarity: min=0.6100" in text


def test_the_similarity_spread_is_reported_as_numbers_with_no_verdict(
    tmp_path, anchor_template, view_template
):
    scores = iter([0.9991])

    def checker(character, images):
        return castcheck.check_consistency(
            character,
            list(images),
            detector=lambda _p: faces.FaceObservation(
                face_count=1, largest_fraction=0.2, score=0.95
            ),
            recognizer=lambda _a, _b: next(scores),
        )

    _session, result = generate(tmp_path, anchor_template, view_template, checker=checker)

    # A near-1.0 pair over two views authored to differ is the mode-collapse
    # signature -- and it still reports "pass", which is exactly why the
    # spread and the caveat are printed instead of a verdict.
    assert result.similarity_spread == pytest.approx((0.9991, 0.9991, 0.9991))
    assert result.status == "pass"


def test_the_origin_block_is_what_configpy_already_requires(
    tmp_path, anchor_template, view_template
):
    """The emitted ``[cast.<name>.origin]`` must load, not merely look right.

    Validated through ``config``'s own origin builder rather than a copy of
    its rules -- a second implementation of "what a synthetic cast member
    needs" drifts from the first within a month.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10, which this project still supports
        import tomli as tomllib

    _session, result = generate(tmp_path, anchor_template, view_template)
    parsed = tomllib.loads(result.origin_toml())

    entry = parsed["cast"]["Nobody"]
    assert entry["synthetic"] is True
    assert entry["image"] == result.images[0].path.as_posix()
    origin = config._build_cast_origin("Nobody", entry["origin"])
    assert origin.seed == 4242
    assert origin.created
    extras = dict(origin.extra)
    assert extras["steps"] == 28
    assert "img2img" in str(extras["method"])
    assert "$1M" in str(extras["licence"])


@pytest.mark.parametrize("character", ["The Dead", 'Odd "Quoted" One', "Nobody.Else"])
def test_the_origin_block_quotes_a_name_that_is_not_a_bare_key(
    tmp_path, anchor_template, view_template, character
):
    """``[cast.The Dead]`` is not TOML. A name outside ``[A-Za-z0-9_-]`` must
    be a quoted key, or the block the operator pastes refuses to load
    (seen for real on 2026-10-05). Round-tripped through the TOML parser *and*
    config's own origin builder, not checked by string shape."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    spec = dataclasses.replace(
        castgen.load_character_spec(write_spec(tmp_path)), character=character
    )
    _session, result = generate(tmp_path, anchor_template, view_template, spec=spec)
    text = result.origin_toml()
    parsed = tomllib.loads(text)

    assert list(parsed["cast"]) == [character]
    entry = parsed["cast"][character]
    assert entry["synthetic"] is True
    origin = config._build_cast_origin(character, entry["origin"])
    assert origin.seed == 4242


def test_a_bare_key_name_stays_unquoted(tmp_path, anchor_template, view_template):
    _session, result = generate(tmp_path, anchor_template, view_template)
    text = result.origin_toml()
    assert "[cast.Nobody]\n" in text
    assert "[cast.Nobody.origin]\n" in text


def test_the_origin_image_is_the_anchor_as_actually_written(
    tmp_path, anchor_template, view_template
):
    """``image`` was hard-coded to ``cast/<file>`` while the files went to
    ``--out-dir`` (``cast/the_dead_turbo_v2/`` for real) -- a path to nothing.
    It must be the written anchor's path, as the output directory was given."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    out_dir = tmp_path / "cast" / "the_dead_turbo_v2"
    out_dir.mkdir(parents=True)
    _session, result = generate(tmp_path, anchor_template, view_template, output_dir=out_dir)
    entry = tomllib.loads(result.origin_toml())["cast"]["Nobody"]

    anchor = result.images[0]
    assert anchor.from_view is None
    assert entry["image"] == anchor.path.as_posix()
    assert Path(entry["image"]).exists()
    assert Path(entry["image"]).parent == out_dir


def test_write_character_record_writes_three_files(tmp_path, anchor_template, view_template):
    _session, result = generate(tmp_path, anchor_template, view_template)
    written = castgen.write_character_record(result, tmp_path / "cast")

    assert set(written) == {"report", "manifest", "origin"}
    for path in written.values():
        assert path.exists() and path.read_text(encoding="utf-8").strip()
    assert json.loads(written["manifest"].read_text(encoding="utf-8"))["character"] == "Nobody"


def test_an_unchecked_set_is_never_a_pass(tmp_path, anchor_template, view_template):
    _session, result = generate(tmp_path, anchor_template, view_template, checker=None)
    assert result.consistency is None
    assert result.status == "unchecked"
    assert result.similarity_spread is None


def test_a_check_that_cannot_run_is_recorded_rather_than_swallowed(
    tmp_path, anchor_template, view_template, caplog
):
    def checker(_character, _images):
        raise faces.FaceDetectionError("SFace weights not found")

    with caplog.at_level(logging.WARNING):
        _session, result = generate(
            tmp_path, anchor_template, view_template, checker=checker
        )

    assert result.status == "unchecked"
    assert "SFace weights not found" in str(result.consistency_detail)
    assert result.manifest()["consistency"]["status"] == "unchecked"


def test_a_rejected_graph_surfaces_as_a_prompt_submission_error_and_frees_the_card(
    tmp_path, anchor_template, view_template
):
    session = FakeComfyUISession()
    session.queue_node_errors({"7": {"errors": [{"message": "value not in list"}]}})

    with pytest.raises(execution.PromptSubmissionError, match="value not in list"):
        generate(tmp_path, anchor_template, view_template, session=session)

    assert session.free_calls, "a failed character must still release the card"


def test_a_prompt_that_finishes_with_no_image_is_a_failure_not_a_wait(
    tmp_path, anchor_template, view_template
):
    session = FakeComfyUISession()
    seed_installed(session)
    session.seed_history_without_video("prompt-0001", node_id="9")

    with pytest.raises(execution.OutputRetrievalError, match="SaveImage"):
        generate(tmp_path, anchor_template, view_template, session=session)


def test_the_poll_budget_is_bounded(tmp_path, anchor_template, view_template):
    session = FakeComfyUISession()
    seed_installed(session)  # no history ever seeded: the prompt never finishes
    slept: list[float] = []

    with pytest.raises(castgen.GenerationTimeoutError, match="no image within"):
        generate(
            tmp_path,
            anchor_template,
            view_template,
            session=session,
            sleep=slept.append,
            poll_attempts=3,
            poll_interval=1.5,
        )

    assert slept == [1.5, 1.5]
    assert session.free_calls


def test_an_insufficient_vram_reading_refuses_before_anything_is_submitted(
    tmp_path, anchor_template, view_template
):
    session = FakeComfyUISession()
    seed_installed(session)
    session.set_vram_free(2 * 1024**3)

    with pytest.raises(custody.CustodyError):
        generate(tmp_path, anchor_template, view_template, session=session)

    assert not session.submitted_prompts


def test_an_unreadable_vram_reading_degrades_rather_than_refusing(
    tmp_path, anchor_template, view_template
):
    session = seed_happy_path(FakeComfyUISession(system_stats={"system": {}, "devices": []}))
    _session, result = generate(tmp_path, anchor_template, view_template, session=session)
    assert result.free_vram_gb is None
    assert result.status == "pass"


def test_generation_warns_that_no_classifier_ran(
    tmp_path, anchor_template, view_template, caplog
):
    with caplog.at_level(logging.WARNING):
        generate(tmp_path, anchor_template, view_template)
    assert "filtering is the deployer's responsibility" in caplog.text


# --------------------------------------------------------------------------- #
# Transport failures: every one of them names itself
# --------------------------------------------------------------------------- #


class _Stub:
    """A ``requests.Session``-shaped object that answers every call the same
    way -- the cheapest way to drive each transport failure branch."""

    def __init__(self, *, raises=None, status=200, body=None, text="not json"):
        self.raises = raises
        self.status_code = status
        self.body = body
        self.text = text

    def _respond(self):
        if self.raises is not None:
            raise self.raises
        return self

    def get(self, *_args, **_kwargs):
        return self._respond()

    def post(self, *_args, **_kwargs):
        return self._respond()

    def json(self):
        if self.body is None:
            raise ValueError("no json")
        return self.body

    @property
    def content(self):
        return b"" if self.body is None else b"bytes"


def test_read_installed_weights_degrades_on_every_unusable_answer(caplog):


    unreachable = _Stub(raises=requests.ConnectionError("no route"))
    with caplog.at_level(logging.WARNING):
        assert (
            castgen.read_installed_weights(unreachable, "http://x", "UNETLoader", "unet_name")
            is None
        )
        assert (
            castgen.read_installed_weights(
                _Stub(status=503, body={}), "http://x", "UNETLoader", "unet_name"
            )
            is None
        )
        assert (
            castgen.read_installed_weights(_Stub(), "http://x", "UNETLoader", "unet_name")
            is None
        )
        # A class that exists but has no such enum: readable, and still not an
        # answer about installed files.
        assert (
            castgen.read_installed_weights(
                _Stub(body={"UNETLoader": {"input": {"required": {}}}}),
                "http://x",
                "UNETLoader",
                "unet_name",
            )
            is None
        )
    assert "could not reach" in caplog.text
    assert "status=503" in caplog.text
    assert "non-JSON body" in caplog.text
    assert "no 'unet_name' enum" in caplog.text


@pytest.mark.parametrize(
    ("stub_kwargs", "expected"),
    [
        ({"raises": requests.ConnectionError("down")}, "failed"),
        ({"status": 500, "body": {}}, "status 500"),
        ({"body": None}, "non-JSON body"),
        ({"body": {"prompt_id": None, "node_errors": {"1": "bad"}}}, "rejected the workflow"),
    ],
)
def test_submission_failures_are_typed_and_named(anchor_template, stub_kwargs, expected):
    with pytest.raises(execution.PromptSubmissionError, match=expected):
        castgen._submit(_Stub(**stub_kwargs), "http://x", anchor_template, "client")


@pytest.mark.parametrize(
    ("stub_kwargs", "expected"),
    [
        ({"raises": requests.ConnectionError("down")}, "failed"),
        ({"status": 500, "body": {}}, "status 500"),
        ({"body": None}, "non-JSON body"),
    ],
)
def test_history_failures_are_typed_and_named(stub_kwargs, expected):
    with pytest.raises(execution.HistoryError, match=expected):
        castgen._await_image(
            _Stub(**stub_kwargs),
            "http://x",
            "prompt-0001",
            sleep=lambda _s: None,
            poll_interval=0.0,
            attempts=1,
        )


@pytest.mark.parametrize(
    ("stub_kwargs", "expected"),
    [
        ({"raises": requests.ConnectionError("down")}, "failed"),
        ({"status": 404, "body": {}}, "status 404"),
        ({"body": None}, "empty body"),
    ],
)
def test_a_download_that_does_not_arrive_is_never_written(stub_kwargs, expected):
    with pytest.raises(execution.OutputRetrievalError, match=expected):
        castgen._fetch_image_bytes(
            _Stub(**stub_kwargs), "http://x", {"filename": "a.png", "type": "output"}
        )


def test_a_temp_preview_is_not_a_saved_image():
    preview = {"9": {"images": [{"filename": "p.png", "type": "temp"}]}}
    assert castgen._find_image_output(preview) is None
    found = castgen._find_image_output(
        {"9": {"images": [{"filename": "p.png", "type": "output"}]}}
    )
    assert found["filename"] == "p.png"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_plan_command_is_offline_and_writes_one_workflow_per_view(tmp_path, capsys):
    spec_path = write_spec(tmp_path)
    out = tmp_path / "plan"

    code = castgen.main(
        [
            "plan",
            str(spec_path),
            "--out-dir",
            str(out),
            "--anchor-template",
            str(ANCHOR_TEMPLATE_PATH),
            "--view-template",
            str(VIEW_TEMPLATE_PATH),
        ],
        session=None,  # never touched on this path
    )

    assert code == 0
    written = sorted(path.name for path in out.glob("*_workflow.json"))
    assert written == [
        "mvm_cast_Nobody_frontal_workflow.json",
        "mvm_cast_Nobody_three_quarter_workflow.json",
    ]
    printed = capsys.readouterr().out
    assert "Nothing was generated" in printed
    assert "filtering is the deployer's responsibility" in printed


def test_generate_command_exits_zero_only_on_a_pass(tmp_path, capsys, monkeypatch):
    session = FakeComfyUISession()
    seed_installed(session)
    seed_image_history(session, "prompt-0001", "a_00001_.png")
    seed_image_history(session, "prompt-0002", "b_00001_.png")
    monkeypatch.setattr(castgen, "_default_checker", pass_checker)

    argv = [
        "generate",
        str(write_spec(tmp_path)),
        "--out-dir",
        str(tmp_path / "cast"),
        "--comfyui-url",
        session.base_url,
        "--anchor-template",
        str(ANCHOR_TEMPLATE_PATH),
        "--view-template",
        str(VIEW_TEMPLATE_PATH),
    ]
    assert castgen.main(argv, session=session) == 0
    printed = capsys.readouterr().out
    assert "RESULT: pass" in printed
    assert "python -m music_video_maker.castcheck Nobody" in printed


def test_generate_command_exits_one_when_the_set_is_not_ready(tmp_path, capsys):
    session = FakeComfyUISession()
    seed_installed(session)
    seed_image_history(session, "prompt-0001", "a_00001_.png")
    seed_image_history(session, "prompt-0002", "b_00001_.png")

    argv = [
        "generate",
        str(write_spec(tmp_path)),
        "--out-dir",
        str(tmp_path / "cast"),
        "--comfyui-url",
        session.base_url,
        "--anchor-template",
        str(ANCHOR_TEMPLATE_PATH),
        "--view-template",
        str(VIEW_TEMPLATE_PATH),
        "--skip-check",
    ]
    assert castgen.main(argv, session=session) == 1
    assert "RESULT: unchecked" in capsys.readouterr().out


def test_a_bad_spec_exits_two_without_touching_the_server(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("character = 1\n", encoding="utf-8")
    code = castgen.main(["generate", str(bad), "--out-dir", str(tmp_path)], session=None)
    assert code == 2
    assert "castgen:" in capsys.readouterr().out


def test_a_generation_failure_exits_one(tmp_path, capsys):
    session = FakeComfyUISession()
    seed_installed(session)
    session.set_vram_free(1 * 1024**3)
    argv = [
        "generate",
        str(write_spec(tmp_path)),
        "--out-dir",
        str(tmp_path / "cast"),
        "--comfyui-url",
        session.base_url,
        "--anchor-template",
        str(ANCHOR_TEMPLATE_PATH),
        "--view-template",
        str(VIEW_TEMPLATE_PATH),
    ]
    assert castgen.main(argv, session=session) == 1
    assert "castgen:" in capsys.readouterr().out
