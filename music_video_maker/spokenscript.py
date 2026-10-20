"""Guardrails for a dialogue-led video: the script is data; bad scripts are refused before GPU time.

A *spoken script* is a cast plus an ordered list of single-speaker shots. Each
shot carries the line (spoken by a separate voice track, never by the video
model), how that line is delivered, who is in frame and in what posture. It is
the speech-led sibling of ``shot_plan.toml``: authored once, reviewed by a
human, committed as data. Authoring-time only -- nothing in the render path
imports it, and it imports nothing that renders.

**General form: every property a first full render got wrong must have a
field and a refusal, because the failure is only visible after the GPU hours
are spent.** A first end-to-end render of a hand-written 19-clip dialogue piece
(about eight GPU-hours, every clip two to four people at once) failed in six
ways, each reproducible from the script text alone:

* **Two speakers in one generated clip** -- one character's line came out of
  another's face. The model binds a voice to a face by guessing when several
  faces share a clip. A shot has exactly one ``speaker``, who must be in frame,
  and a line spoken by anyone else is another shot.
* **Delivery written as prose** ("flat", "annoyed", "hopeful") -- the same
  intonation on most lines and comedy that did not land. Prose adjectives are
  not a channel any speech stage reads. ``delivery`` is a closed vocabulary
  (pace, pitch, volume, plus a pause before the line) that a synthesis stage can
  act on and a linter can compare: identical delivery on adjacent speakers, or
  one delivery across most of a script, is reported.
* **A garment never stated** -- a modern button-down on a period cast. The
  portrait prompt said "tunic" in places and nothing anywhere said "no
  buttons". With an ``era`` set, every cast member states a ``garment`` and a
  list of terms that cannot exist in that era is refused wherever it appears:
  the cast, and the prose of every shot.
* **Posture never stated** -- people stood, sat, stood and sat between cuts,
  because a model re-decides posture for every clip it is not told about. Every
  character in frame carries a ``posture`` from a closed list, and a change
  from one shot to the next must be marked by the later shot's ``transition``.
* **A person in prose who is not in the frame list** -- someone walked into a
  scene they were never cast in, because the sentence named them and the shot
  did not. A cast member named in a shot's prose must be in that shot's
  ``characters``; a shot with a crowd is warned.
* **Prompt scaffolding spoken aloud** -- "Scene, 12 seconds:" and "use it
  exactly for his lines" were read out as dialogue. Any label or instruction
  placed ahead of the description is text the model may say. The prompt this
  module composes contains description only and never a line of dialogue, and
  :func:`lint_generation_prompt` refuses the two measured patterns so a future
  composer cannot reintroduce them.

Two further rules are arithmetic, not lessons: the quoted speech must fit the
shot (words over the pace's words-per-second, minus the pause), and dialogue
quoted inside a shot's prose is refused because the line belongs on the voice
track -- prose that quotes speech invites the model to speak it itself.

What this module deliberately does not do: judge whether a delivery is funny,
check the *rendered* portraits for anachronism (that needs a vision model; the
check here is on the text that produced them), or decide which synthesis engine
speaks the lines. A finding is ``error`` (refuse) or ``warning`` (report); only
errors raise from :func:`require_clean`. Every vocabulary below is the smallest
set the measured failures needed, and is a starting point, not a standard.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PACES = ("slow", "measured", "normal", "brisk", "fast")
PITCHES = ("low", "mid", "high")
VOLUMES = ("whisper", "soft", "normal", "loud")
POSTURES = ("standing", "seated", "kneeling", "walking", "lying", "crouching")

# Spoken words per second at each pace. Starting values (ordinary conversation is
# ~2.5 wps); a synthesis engine's real rate replaces these once one is chosen.
WORDS_PER_SECOND = {"slow": 1.8, "measured": 2.2, "normal": 2.6, "brisk": 3.1, "fast": 3.6}

MAX_VISIBLE = 4
FLAT_SHARE = 0.6
FLAT_MIN_LINES = 5

# Terms that cannot exist in the named era. Deliberately short and unambiguous: a
# lint that forbids ordinary words gets disabled. Whole-word matches only.
ERA_BANNED: dict[str, tuple[str, ...]] = {
    "pre-industrial": (
        "button", "buttons", "buttoned", "button-down", "button-up", "collar", "collared",
        "shirt", "t-shirt", "polo", "henley", "jeans", "denim", "zipper", "velcro",
        "sneakers", "sunglasses", "wristwatch", "hoodie", "blazer", "necktie", "lapel",
    ),
}

_QUOTED = re.compile(r"[\"“][^\"”]{3,}[\"”]")

# Scaffolding measured to be read aloud by the video model (see module docstring).
_SCAFFOLD_PATTERNS = (
    (re.compile(r"\b\d+(?:\.\d+)?\s*seconds?\s*:", re.I), "a duration label ('N seconds:')"),
    (
        re.compile(r"^\s*(?:scene|shot|clip)\b[^.:\n]{0,24}:", re.I | re.M),
        "a 'Scene:'/'Shot:' label",
    ),
    (
        re.compile(
            r"\buse\s+(?:it|this|that|the\s+(?:voice|audio|reference))\b[^.]*\bexactly\b", re.I
        ),
        "an instruction to use the audio 'exactly'",
    ),
)


class SpokenScriptError(ValueError):
    """The script could not be loaded, or has errors (see ``findings``)."""

    def __init__(self, message: str, findings: tuple[Finding, ...] = ()) -> None:
        super().__init__(message)
        self.findings = findings


@dataclass(frozen=True)
class Finding:
    rule: str
    shot_id: str | None
    severity: str  # "error" | "warning"
    message: str


@dataclass(frozen=True)
class Delivery:
    pace: str
    pitch: str
    volume: str
    pause_before: float = 0.0

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.pace, self.pitch, self.volume)


@dataclass(frozen=True)
class CastMember:
    key: str
    name: str
    appearance: str
    voice: str
    garment: str = ""


@dataclass(frozen=True)
class Shot:
    id: str
    seconds: float
    characters: tuple[str, ...]
    text: str
    posture: dict[str, str] = field(default_factory=dict)
    speaker: str | None = None
    line: str | None = None
    delivery: Delivery | None = None
    transition: str = ""


@dataclass(frozen=True)
class Script:
    cast: dict[str, CastMember]
    shots: tuple[Shot, ...]
    style: str = ""
    era: str | None = None
    banned_terms: tuple[str, ...] = ()


# --------------------------------------------------------------------- loading
_SCRIPT_KEYS = {"era", "style", "banned_terms"}
_CAST_KEYS = {"name", "garment", "appearance", "voice"}
_SHOT_KEYS = {
    "id", "seconds", "characters", "speaker", "line", "text", "posture", "transition", "delivery",
}
_DELIVERY_KEYS = {"pace", "pitch", "volume", "pause_before"}


def _unknown(where: str, got: Iterable[str], allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise SpokenScriptError(f"{where}: unknown key(s) {extra} (allowed: {sorted(allowed)})")


def _choice(where: str, what: str, value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise SpokenScriptError(f"{where}: {what} {value!r} is not one of {list(allowed)}")
    return value


def _parse_delivery(where: str, raw: dict[str, Any]) -> Delivery:
    _unknown(f"{where} delivery", raw, _DELIVERY_KEYS)
    for key in ("pace", "pitch", "volume"):
        if key not in raw:
            raise SpokenScriptError(f"{where}: delivery needs {key}")
    return Delivery(
        pace=_choice(where, "delivery.pace", raw["pace"], PACES),
        pitch=_choice(where, "delivery.pitch", raw["pitch"], PITCHES),
        volume=_choice(where, "delivery.volume", raw["volume"], VOLUMES),
        pause_before=float(raw.get("pause_before", 0.0)),
    )


def load_script_text(text: str, source: str = "<script>") -> Script:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SpokenScriptError(f"{source}: not valid TOML: {exc}") from exc
    return _build(raw)


def load_script(path: str | Path) -> Script:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpokenScriptError(f"{path}: cannot read: {exc}") from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SpokenScriptError(f"{path}: not valid TOML: {exc}") from exc
    return _build(raw)


def _build(raw: dict[str, Any]) -> Script:
    head = raw.get("script", {})
    _unknown("[script]", head, _SCRIPT_KEYS)
    cast: dict[str, CastMember] = {}
    for key, c in raw.get("cast", {}).items():
        _unknown(f"[cast.{key}]", c, _CAST_KEYS)
        cast[key] = CastMember(
            key=key,
            name=c.get("name", key),
            appearance=c.get("appearance", ""),
            voice=c.get("voice", ""),
            garment=c.get("garment", ""),
        )
    shots: list[Shot] = []
    seen: set[str] = set()
    for i, s in enumerate(raw.get("shot", []), 1):
        sid = s.get("id")
        if not sid:
            raise SpokenScriptError(f"shot #{i}: missing id")
        if sid in seen:
            raise SpokenScriptError(f"shot {sid}: duplicate id")
        seen.add(sid)
        _unknown(f"shot {sid}", s, _SHOT_KEYS)
        posture = {
            who: _choice(f"shot {sid}", f"posture for {who!r}", p, POSTURES)
            for who, p in s.get("posture", {}).items()
        }
        delivery = _parse_delivery(f"shot {sid}", s["delivery"]) if "delivery" in s else None
        shots.append(
            Shot(
                id=sid,
                seconds=float(s.get("seconds", 0)),
                characters=tuple(s.get("characters", ())),
                text=s.get("text", ""),
                posture=posture,
                speaker=s.get("speaker"),
                line=s.get("line"),
                delivery=delivery,
                transition=s.get("transition", ""),
            )
        )
    return Script(
        cast=cast,
        shots=tuple(shots),
        style=head.get("style", ""),
        era=head.get("era"),
        banned_terms=tuple(head.get("banned_terms", ())),
    )


# --------------------------------------------------------------------- linting
def _word_re(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.I)


def _banned_for(script: Script) -> tuple[str, ...]:
    return tuple(ERA_BANNED.get(script.era or "", ())) + script.banned_terms


def lint_script(script: Script) -> list[Finding]:
    out: list[Finding] = []

    def add(rule: str, shot: Shot | None, severity: str, message: str) -> None:
        out.append(Finding(rule, shot.id if shot else None, severity, message))

    if script.era and script.era not in ERA_BANNED:
        add("era-unknown", None, "warning",
            f"era {script.era!r} has no built-in wardrobe list; only banned_terms apply "
            f"(known: {sorted(ERA_BANNED)})")

    banned = _banned_for(script)
    if script.era:
        for member in script.cast.values():
            if not member.garment:
                add("garment-missing", None, "error",
                    f"cast {member.key!r} states no garment; with era {script.era!r} every "
                    "character's clothing must be written down, or the model picks it")
    for member in script.cast.values():
        for term in banned:
            if _word_re(term).search(f"{member.garment} {member.appearance}"):
                add("wardrobe-term", None, "error",
                    f"cast {member.key!r}: {term!r} does not belong in era {script.era!r}")

    last_posture: dict[str, str] = {}
    spoken: list[Shot] = []
    for shot in script.shots:
        for who in shot.characters:
            if who not in script.cast:
                add("unknown-character", shot, "error", f"{who!r} is not in the cast")
        if shot.speaker is not None and shot.speaker not in shot.characters:
            add("speaker-in-shot", shot, "error",
                f"speaker {shot.speaker!r} must be in this shot's characters")
        if (shot.speaker is None) != (shot.line is None):
            add("speaker-line-pair", shot, "error", "speaker and line must be given together")
        if _QUOTED.search(shot.text):
            add("dialogue-in-prose", shot, "error",
                "quoted speech in the shot text; put the words in `line` (the voice track), "
                "or the video model will speak them itself")
        for term in banned:
            if _word_re(term).search(shot.text):
                add("wardrobe-term", shot, "error",
                    f"{term!r} in shot text does not belong in era {script.era!r}")
        for key, member in script.cast.items():
            if key in shot.characters:
                continue
            if _word_re(member.name).search(shot.text) or _word_re(key).search(shot.text):
                add("unlisted-character-in-text", shot, "error",
                    f"{member.name!r} is named in the prose but not in this shot's characters; "
                    "list them (with a posture) or stop naming them")
        if len(shot.characters) > MAX_VISIBLE:
            add("too-many-visible", shot, "warning",
                f"{len(shot.characters)} characters in one shot (> {MAX_VISIBLE}); "
                "faces, voices and postures blur with a crowd")
        for who in shot.characters:
            if who not in shot.posture:
                add("posture-missing", shot, "error", f"{who!r} is in frame with no posture")
                continue
            before = last_posture.get(who)
            if before and before != shot.posture[who] and not shot.transition:
                add("posture-change-unmarked", shot, "error",
                    f"{who!r} goes {before} -> {shot.posture[who]} with no `transition` "
                    "saying how; an unexplained change reads as a continuity error")
            last_posture[who] = shot.posture[who]
        if shot.line is not None:
            spoken.append(shot)
            if shot.delivery is None:
                add("delivery-missing", shot, "error",
                    "a spoken line needs delivery (pace, pitch, volume); prose adjectives are "
                    "not read by anything")
            else:
                words = len(re.findall(r"[\w']+", shot.line))
                needed = words / WORDS_PER_SECOND[shot.delivery.pace]
                available = shot.seconds - shot.delivery.pause_before
                if needed > available:
                    add("line-too-long", shot, "error",
                        f"{words} words at pace {shot.delivery.pace!r} need ~{needed:.1f}s but the "
                        f"shot has {available:.1f}s after its pause")

    out.extend(_delivery_variety(spoken))
    return out


def _delivery_variety(spoken: list[Shot]) -> list[Finding]:
    found: list[Finding] = []
    timed = [s for s in spoken if s.delivery is not None]
    for prev, cur in zip(timed, timed[1:], strict=False):
        if prev.speaker != cur.speaker and prev.delivery.signature == cur.delivery.signature:  # type: ignore[union-attr]
            found.append(Finding("delivery-flat", cur.id, "warning",
                                 f"{prev.speaker!r} and {cur.speaker!r} are delivered identically "
                                 f"{cur.delivery.signature}; a comic exchange needs contrast"))  # type: ignore[union-attr]
    if len(timed) >= FLAT_MIN_LINES:
        counts: dict[tuple[str, str, str], int] = {}
        for s in timed:
            counts[s.delivery.signature] = counts.get(s.delivery.signature, 0) + 1  # type: ignore[union-attr]
        sig, n = max(counts.items(), key=lambda kv: kv[1])
        if n / len(timed) > FLAT_SHARE:
            found.append(Finding("delivery-flat", None, "warning",
                                 f"{n} of {len(timed)} lines share one delivery {sig}"))
    return found


def require_clean(findings: list[Finding]) -> None:
    errs = tuple(f for f in findings if f.severity == "error")
    if errs:
        listing = "; ".join(f"{f.rule} ({f.shot_id or 'script'}): {f.message}" for f in errs)
        raise SpokenScriptError(f"{len(errs)} error(s): {listing}", errs)


# ------------------------------------------------------------------ the prompt
def compose_generation_prompt(script: Script, shot: Shot) -> str:
    """Description only: who is in frame, in what posture and garment, and what happens.

    Never the shot's ``line`` (that is the voice track) and never a label or an
    instruction, because the video model reads aloud what it is given.
    """
    parts = [script.style.strip()] if script.style.strip() else []
    for who in shot.characters:
        member = script.cast.get(who)
        if member is None:
            continue
        posture = shot.posture.get(who, "")
        bits = [member.name]
        if posture:
            bits.append(posture)
        sentence = ", ".join(bits)
        if member.garment:
            sentence += f", in {member.garment}"
        if member.appearance:
            sentence += f", {member.appearance}"
        parts.append(sentence + ".")
    if shot.text.strip():
        parts.append(shot.text.strip())
    return " ".join(parts)


def lint_generation_prompt(prompt: str) -> list[str]:
    """Problems with a prompt's wording; empty when it is description only."""
    return [
        f"prompt contains {label}"
        for pattern, label in _SCAFFOLD_PATTERNS
        if pattern.search(prompt)
    ]


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    """``python -m music_video_maker.spokenscript s.toml``: exit 0 clean, 1 errors, 2 unreadable."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m music_video_maker.spokenscript <script.toml>\n")
        return 2
    try:
        script = load_script(args[0])
    except SpokenScriptError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    findings = lint_script(script)
    for f in findings:
        sys.stdout.write(f"{f.severity.upper():7} {f.rule} [{f.shot_id or 'script'}] {f.message}\n")
    errors = sum(1 for f in findings if f.severity == "error")
    if not findings:
        sys.stdout.write(f"clean: {len(script.shots)} shots, {len(script.cast)} cast\n")
    elif not errors:
        sys.stdout.write(f"{len(findings)} warning(s), no errors\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
