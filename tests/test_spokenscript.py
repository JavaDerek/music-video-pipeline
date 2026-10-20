"""Tests for the spoken-script guardrails.

A *spoken script* is the data a dialogue-led video is authored in: a cast, and
an ordered list of single-speaker shots, each carrying its line, how the line
is delivered, who is in frame and in what posture. Every rule below exists
because a first full render of a hand-written 19-clip dialogue piece failed in
that exact way, and the failure was visible only after ~8 GPU-hours:

* two characters' lines in one generated clip -> the model gave one character's
  line to the other's face;
* voices bootstrapped from the model itself -> every voice identical;
* prose delivery directions ("flat", "annoyed") -> the same intonation on
  almost every line;
* a garment never stated per character -> a modern button-down on a period cast;
* posture never stated -> people standing, sitting, standing between cuts;
* a character named in a shot's prose but not listed in frame -> that person
  walking into a scene they had no part in;
* labels in the prompt ("Scene, 12 seconds:") spoken aloud by the model.

Each is a refusal at authoring time, before any GPU is spent.
"""

from __future__ import annotations

import textwrap

import pytest

from music_video_maker import spokenscript as ss

GOOD = textwrap.dedent(
    """
    [script]
    era = "pre-industrial"
    style = "Warm natural light, handheld, a market square at dusk."

    [cast.mara]
    name = "Mara"
    garment = "undyed linen tunic, round neck, rope belt"
    appearance = "weathered woman in her forties, grey-streaked braid"
    voice = "low, dry, unhurried"

    [cast.teo]
    name = "Teo"
    garment = "wool vest over a patched linen tunic, round neck"
    appearance = "thin young man, nervous eyes"
    voice = "high, quick, eager"

    [[shot]]
    id = "s1"
    seconds = 6
    characters = ["mara"]
    speaker = "mara"
    line = "We open at dawn."
    text = "Close on Mara at her stall, hands on the counter."
    posture = { mara = "standing" }
    [shot.delivery]
    pace = "measured"
    pitch = "low"
    volume = "normal"
    pause_before = 0.4

    [[shot]]
    id = "s2"
    seconds = 6
    characters = ["teo"]
    speaker = "teo"
    line = "Dawn? Which dawn?"
    text = "Close on Teo, leaning in over a crate."
    posture = { teo = "standing" }
    [shot.delivery]
    pace = "fast"
    pitch = "high"
    volume = "loud"
    pause_before = 0.2
    """
)


def load(text: str = GOOD) -> ss.Script:
    return ss.load_script_text(text)


def rules(findings: list[ss.Finding]) -> set[str]:
    return {f.rule for f in findings}


def errors(findings: list[ss.Finding]) -> set[str]:
    return {f.rule for f in findings if f.severity == "error"}


def test_a_clean_script_has_no_findings():
    assert ss.lint_script(load()) == []


def test_require_clean_raises_on_errors_and_lists_them():
    script = load(GOOD.replace('speaker = "mara"', 'speaker = "teo"'))
    with pytest.raises(ss.SpokenScriptError) as exc:
        ss.require_clean(ss.lint_script(script))
    assert "speaker-in-shot" in str(exc.value)


def test_warnings_alone_do_not_raise():
    ss.require_clean([ss.Finding("x", "s1", "warning", "just a warning")])


# --- one voice per generated shot --------------------------------------------------


def test_speaker_must_be_in_frame():
    findings = ss.lint_script(load(GOOD.replace('speaker = "mara"', 'speaker = "teo"')))
    assert "speaker-in-shot" in errors(findings)


def test_unknown_character_is_an_error():
    findings = ss.lint_script(load(GOOD.replace('characters = ["mara"]', 'characters = ["zed"]')))
    assert "unknown-character" in errors(findings)


def test_quoted_speech_in_prose_is_an_error():
    bad = GOOD.replace(
        "Close on Mara at her stall, hands on the counter.",
        'Close on Mara, who says \\"we open at dawn\\" to the crowd.',
    )
    assert "dialogue-in-prose" in errors(ss.lint_script(load(bad)))


# --- delivery is data, not prose -----------------------------------------------


def test_a_line_without_delivery_is_an_error():
    bad = GOOD.replace(
        '[shot.delivery]\npace = "measured"\npitch = "low"\nvolume = "normal"\n'
        "pause_before = 0.4\n",
        "",
    )
    assert "delivery-missing" in errors(ss.lint_script(load(bad)))


def test_delivery_values_must_come_from_the_vocabulary():
    bad = GOOD.replace('pace = "measured"', 'pace = "flatly annoyed"')
    with pytest.raises(ss.SpokenScriptError, match="pace"):
        load(bad)


def test_identical_delivery_on_adjacent_speakers_is_flagged():
    same = GOOD.replace('pace = "fast"', 'pace = "measured"').replace(
        'pitch = "high"', 'pitch = "low"'
    ).replace('volume = "loud"', 'volume = "normal"')
    findings = ss.lint_script(load(same))
    assert "delivery-flat" in rules(findings)
    assert "delivery-flat" not in errors(findings)  # a smell, not a refusal


def test_one_delivery_across_most_of_a_long_script_is_flagged():
    shots = []
    for i in range(6):
        who = "mara" if i % 2 == 0 else "teo"
        shots.append(
            textwrap.dedent(
                f"""
                [[shot]]
                id = "x{i}"
                seconds = 6
                characters = ["{who}"]
                speaker = "{who}"
                line = "Line {i}."
                text = "A plain shot of {who}."
                posture = {{ {who} = "standing" }}
                [shot.delivery]
                pace = "normal"
                pitch = "mid"
                volume = "normal"
                """
            )
        )
    head = GOOD.split("[[shot]]")[0]
    findings = ss.lint_script(load(head + "".join(shots)))
    assert any(f.rule == "delivery-flat" and f.shot_id is None for f in findings)


def test_a_line_too_long_for_its_shot_is_an_error():
    long_line = " ".join(["word"] * 40)
    bad = GOOD.replace("We open at dawn.", long_line)
    assert "line-too-long" in errors(ss.lint_script(load(bad)))


# --- wardrobe and posture are stated, not hoped for ------------------------------


def test_every_cast_member_must_state_a_garment_when_an_era_is_set():
    bad = GOOD.replace('garment = "undyed linen tunic, round neck, rope belt"\n', "")
    assert "garment-missing" in errors(ss.lint_script(load(bad)))


def test_modern_garment_terms_are_refused_for_a_pre_industrial_era():
    bad = GOOD.replace(
        "wool vest over a patched linen tunic, round neck", "button-down shirt, round neck"
    )
    findings = ss.lint_script(load(bad))
    assert "wardrobe-term" in errors(findings)
    assert any("button-down" in f.message for f in findings)


def test_wardrobe_terms_in_shot_prose_are_refused_too():
    bad = GOOD.replace("hands on the counter", "hands on the counter, collar open")
    assert "wardrobe-term" in errors(ss.lint_script(load(bad)))


def test_extra_banned_terms_can_be_declared_by_the_script():
    bad = GOOD.replace('era = "pre-industrial"', 'era = "pre-industrial"\nbanned_terms = ["vest"]')
    assert "wardrobe-term" in errors(ss.lint_script(load(bad)))


def test_no_era_means_no_wardrobe_lint():
    no_era = GOOD.replace('era = "pre-industrial"\n', "").replace(
        "wool vest over a patched linen tunic", "button-down shirt"
    )
    assert "wardrobe-term" not in rules(ss.lint_script(load(no_era)))


def test_a_character_in_frame_must_have_a_posture():
    bad = GOOD.replace('posture = { mara = "standing" }', "")
    assert "posture-missing" in errors(ss.lint_script(load(bad)))


def test_posture_must_come_from_the_vocabulary():
    with pytest.raises(ss.SpokenScriptError, match="posture"):
        load(GOOD.replace('mara = "standing"', 'mara = "levitating"'))


def test_an_unmarked_posture_change_between_shots_is_an_error():
    bad = GOOD.replace('characters = ["teo"]', 'characters = ["mara", "teo"]').replace(
        'posture = { teo = "standing" }', 'posture = { teo = "standing", mara = "seated" }'
    )
    assert "posture-change-unmarked" in errors(ss.lint_script(load(bad)))


def test_a_marked_posture_change_is_allowed():
    ok = (
        GOOD.replace('characters = ["teo"]', 'characters = ["mara", "teo"]')
        .replace(
            'posture = { teo = "standing" }', 'posture = { teo = "standing", mara = "seated" }'
        )
        .replace(
            'text = "Close on Teo, leaning in over a crate."',
            'text = "Close on Teo, leaning in over a crate."\n'
            'transition = "Mara sits down on a stool"',
        )
    )
    assert "posture-change-unmarked" not in rules(ss.lint_script(load(ok)))


def test_an_unknown_key_is_refused_rather_than_silently_ignored():
    with pytest.raises(ss.SpokenScriptError, match="postur"):
        load(GOOD.replace("posture = { teo", "postur = { teo"))


# --- nobody appears who was not cast in the shot ---------------------------------


def test_naming_an_unlisted_character_in_prose_is_an_error():
    bad = GOOD.replace(
        "Close on Teo, leaning in over a crate.",
        "Close on Teo, leaning in over a crate while Mara walks in behind him.",
    )
    findings = ss.lint_script(load(bad))
    assert "unlisted-character-in-text" in errors(findings)


def test_too_many_people_in_one_shot_is_a_warning():
    people = "".join(
        f'\n    [cast.p{i}]\n    name = "P{i}"\n    garment = "linen tunic"\n'
        f'    appearance = "a villager"\n    voice = "plain"\n'
        for i in range(6)
    )
    head, _, tail = GOOD.partition("[[shot]]")
    crowd = (
        head
        + people
        + '\n    [[shot]]\n    id = "c"\n    seconds = 4\n'
        + '    characters = ["p0","p1","p2","p3","p4","p5"]\n'
        + '    text = "A wide shot of the villagers."\n'
        + "    posture = { "
        + ", ".join(f'p{i}="standing"' for i in range(6))
        + " }\n"
    )
    findings = ss.lint_script(load(crowd))
    assert "too-many-visible" in rules(findings)


# --- the prompt that reaches the model has nothing to read aloud ---------------


def test_the_composed_prompt_passes_its_own_scaffold_lint():
    script = load()
    for shot in script.shots:
        prompt = ss.compose_generation_prompt(script, shot)
        assert ss.lint_generation_prompt(prompt) == []
        assert shot.line not in prompt  # the line is TTS, never prose to the model


def test_the_composed_prompt_names_each_character_with_posture_and_garment():
    script = load()
    prompt = ss.compose_generation_prompt(script, script.shots[0])
    assert "Mara" in prompt and "standing" in prompt and "linen tunic" in prompt
    assert "Teo" not in prompt  # not in frame


@pytest.mark.parametrize(
    "scaffold",
    [
        "Scene, 12 seconds: two people talk.",
        "Shot, 6 seconds: a close-up.",
        "Use the voice exactly as it is for his lines.",
        "Use it exactly for his lines.",
    ],
)
def test_scaffolding_known_to_be_spoken_aloud_is_flagged(scaffold):
    assert ss.lint_generation_prompt(f"Warm light. {scaffold}")


def test_plain_description_is_not_flagged():
    plain = "Warm light. Mara stands at her stall, hands on the counter."
    assert ss.lint_generation_prompt(plain) == []


# --- loading -------------------------------------------------------------------


def test_load_script_reads_a_file(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text(GOOD)
    assert len(ss.load_script(p).shots) == 2


def test_malformed_toml_raises_a_named_error(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text("[[shot")
    with pytest.raises(ss.SpokenScriptError, match="s.toml"):
        ss.load_script(p)


def test_a_shot_without_an_id_is_refused():
    with pytest.raises(ss.SpokenScriptError, match="id"):
        load(GOOD.replace('id = "s1"\n', "", 1))


def test_duplicate_shot_ids_are_refused():
    with pytest.raises(ss.SpokenScriptError, match="duplicate"):
        load(GOOD.replace('id = "s2"', 'id = "s1"'))


# --- the command an author runs ------------------------------------------------


def test_cli_exits_zero_and_says_clean_for_a_good_script(tmp_path, capsys):
    p = tmp_path / "s.toml"
    p.write_text(GOOD)
    assert ss.main([str(p)]) == 0
    assert "clean" in capsys.readouterr().out


def test_cli_exits_nonzero_and_lists_every_error(tmp_path, capsys):
    p = tmp_path / "s.toml"
    p.write_text(GOOD.replace('speaker = "mara"', 'speaker = "teo"'))
    assert ss.main([str(p)]) == 1
    assert "speaker-in-shot" in capsys.readouterr().out


def test_cli_reports_a_load_failure_without_a_traceback(tmp_path, capsys):
    p = tmp_path / "s.toml"
    p.write_text("[[shot")
    assert ss.main([str(p)]) == 2
    assert "not valid TOML" in capsys.readouterr().err


def test_cli_prints_warnings_but_still_exits_zero(tmp_path, capsys):
    p = tmp_path / "s.toml"
    p.write_text(
        GOOD.replace('pace = "fast"', 'pace = "measured"')
        .replace('pitch = "high"', 'pitch = "low"')
        .replace('volume = "loud"', 'volume = "normal"')
    )
    assert ss.main([str(p)]) == 0
    assert "delivery-flat" in capsys.readouterr().out


# --- the second render's lessons: openings, portraits, voices -----------------------


def test_a_placket_or_toggle_is_refused_for_a_pre_industrial_era():
    for term in ("placket", "toggles"):
        bad = GOOD.replace("linen tunic, round neck, rope belt", f"linen tunic with a {term}")
        assert "wardrobe-term" in errors(ss.lint_script(load(bad))), term


def test_a_garment_that_never_states_its_opening_is_a_warning():
    vague = GOOD.replace(
        "undyed linen tunic, round neck, rope belt", "undyed linen tunic, rope belt"
    )
    findings = ss.lint_script(load(vague))
    assert "garment-opening-unstated" in rules(findings)
    assert "garment-opening-unstated" not in errors(findings)


def test_the_portrait_prompt_carries_the_era_lock_garment_and_style():
    script = load()
    prompt = ss.compose_portrait_prompt(script, script.cast["mara"], "Head-and-shoulders portrait.")
    assert "undyed linen tunic" in prompt and "market square" in prompt
    assert "no placket" in prompt and "no buttons" in prompt
    assert "no front opening" in prompt.lower()
    assert "Head-and-shoulders portrait." in prompt


def test_a_cast_member_can_override_the_era_lock():
    custom = GOOD.replace(
        'voice = "low, dry, unhurried"',
        'voice = "low, dry, unhurried"\nwardrobe_lock = "Leather jerkin, laced."'
    )
    script = load(custom)
    prompt = ss.compose_portrait_prompt(script, script.cast["mara"], "")
    assert "Leather jerkin, laced." in prompt and "no placket" not in prompt


def test_no_era_means_no_lock_in_the_portrait_prompt():
    script = load(GOOD.replace('era = "pre-industrial"\n', ""))
    assert "no placket" not in ss.compose_portrait_prompt(script, script.cast["mara"], "")


def _with_voices(text, **voices):
    descriptions = {"mara": "low, dry, unhurried", "teo": "high, quick, eager"}
    for who, tts in voices.items():
        d = descriptions[who]
        text = text.replace(f'voice = "{d}"', f'voice = "{d}"\ntts_voice = "{tts}"')
    return text


def test_two_characters_sharing_a_synthesis_voice_is_an_error():
    same = _with_voices(GOOD, teo="Aaa", mara="aaa")
    assert "voice-shared" in errors(ss.lint_script(load(same)))


def test_distinct_synthesis_voices_pass_and_a_missing_one_warns():
    ok = _with_voices(GOOD, teo="Bbb", mara="Aaa")
    assert "voice-shared" not in rules(ss.lint_script(load(ok)))
    one = _with_voices(GOOD, mara="Aaa")
    assert "voice-missing" in rules(ss.lint_script(load(one)))
