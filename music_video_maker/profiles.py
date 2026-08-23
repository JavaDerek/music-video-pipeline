"""Lock a house cinematography style into a versioned, reusable profile
(issue #55) -- the natural endpoint of #53 (cinematography as its own field)
and #54 (a model proposing candidates for it): once a look works, freeze it
so a catalogue of videos shares a recognisable signature instead of every
run re-rolling the look from scratch.

**Reference, not copy (D1/D2).** A run config points at a profile file by
path; :func:`~music_video_maker.config.load_config` reads it and fills in
whatever look fields the run config itself left unset. The run config always
wins on a field it does set -- the same precedence ``overrides`` already has
over file values in ``load_config``, and the only alternative is a run that
cannot deviate from a house style without editing the file every other video
shares, which is exactly the coupling that makes a house style dangerous in
the first place. Every inherited field and every field the run config
overrode is logged at INFO: a silent inheritance is how a "locked" look
drifts without anyone noticing which run changed it.

**The field set is closed and small (D3).** :data:`LOOK_FIELDS` is
whole-video and about the *look*, not the story. Three related fields were
excluded on purpose:

* ``global_style`` -- issue #53 split cinematography out of it precisely so
  it would stop being a junk drawer for genre/tone/band, which is per-song,
  not house style.
* ``render_width``/``render_height`` -- measured to be the dominant cost
  lever (864x480 = 3.7 min/chunk vs 1344x768 = 9 m 15 s on the same 4090,
  see the project's own CLAUDE.md). A look profile that silently triples a
  run's GPU hours is the wrong kind of inheritance to make invisible. A
  future revision may add resolution here; it must be loud if it does, not
  slipped into this tuple.
* per-shot ``camera`` -- issue #55's whole point is that shot-level framing
  and movement stays per-video, or every video becomes the same video.

**Every field a profile may set must already be recorded in a
``ChunkFingerprint`` (D4).** This is a constraint on the schema, not a
coincidence: a look field admitted here that nothing fingerprints would let
a profile edit change the pixels while ``--resume`` reuses the old chunks and
reports a clean match. :data:`FINGERPRINT_EVIDENCE` names, for every entry
of :data:`LOOK_FIELDS`, which fingerprint field would move if that look field
changed; ``tests/test_profiles.py`` checks every value against
``contracts.ChunkFingerprint``'s own field tuples so this cannot drift
silently as either module grows.

**A hash proves change, not content (D5).** ``prompt_hash`` tells a resumed
run that *something* changed; six months later, with the profile edited to
v3, it cannot say what look produced a chunk rendered under v1.
:func:`write_profile_record` writes a JSON sidecar recording the resolved
profile verbatim, so a finished video can prove what look actually made it.

**Validate shape here, never domain (D6).** This module checks types and
non-emptiness only. It must never know that ``face_treatment`` is one of
``{"flattering", "realistic"}`` -- ``config.py`` already owns that closed
set, and encoding it twice is how the two definitions drift apart. Because a
profile's values are filled into ``merged`` *before* ``config.load_config``
validates, an invalid profile value (e.g. a bogus ``face_treatment``) is
caught by config's own existing validation, which is exactly where it should
be caught.

**No imports from ``music_video_maker`` (D7).** ``config.py`` imports this
module, so the reverse would be circular. :class:`ProfileError` is defined
here rather than reusing ``config.ConfigError``; ``config.py`` catches
:class:`ProfileError` at its one call site and re-raises its own
``ConfigError``. The single narrow exception is a *deferred* import inside
:func:`main`, see that function's docstring.

**TOML's bare-key hazard, twice over (D8).** This repository has been bitten
by this before: ``config.HARDWARE_KEYS`` and ``ALIGNMENT_OVERRIDE_KEYS``
both exist because TOML binds a bare key to whichever table precedes it, so
a top-level setting written below a table silently becomes a key of that
table instead. A profile file has exactly one table (``[provenance]``), so a
look field appended below it would vanish into ``provenance`` and never
reach ``LOOK_FIELDS`` at all. :data:`TOP_LEVEL_KEYS` and
:data:`PROVENANCE_KEYS` are both closed sets, and :func:`load_profile`
recognises when an unknown ``[provenance]`` key is actually a real top-level
key and says so explicitly, naming the rule, rather than reporting a plain
"unknown key" that leaves the reader to rediscover TOML's own semantics.

``PROFILE_FORMAT_VERSION`` names the file *format* this build reads and
writes -- it would only need to change if the shape of a profile file
itself changed (new required key, renamed table). It is deliberately
independent of a profile's own ``version`` field, which is the house style's
*content* version (v1, v2, v3 of the same look) and is never touched by this
module except to read and re-emit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

PROFILE_FORMAT_VERSION = 1
"""The profile *file format* this build reads and writes -- distinct from a
profile's own ``version`` field (the house style's content version). See the
module docstring's closing paragraph. Written into every record
:func:`write_profile_record` produces, so a reader of an old sidecar knows
which shape it is reading rather than guessing from the keys present."""

PROFILE_RECORD_FILENAME = "cinematography_profile.json"
"""Where :func:`write_profile_record`'s sidecar lands, by convention, inside a
run's ``chunks_dir`` -- beside ``run_state.json``, so everything that proves
what a finished video is made of lives in one directory."""

LOOK_FIELDS: tuple[str, ...] = (
    "cinematography",
    "face_treatment",
    "lora",
    "lora_strength",
    "lora_trigger",
)
"""The closed, deliberately small set of fields a profile may lock (D3).
Whole-video and about the look, never the story and never a per-shot
concern -- see the module docstring for what was excluded and why."""

FINGERPRINT_EVIDENCE: dict[str, str] = {
    "cinematography": "prompt_hash",  # composed into every prompt (prompting._compose_prompt)
    "face_treatment": "lora",  # a preset that resolves to lora/lora_strength/lora_trigger
    "lora": "lora",
    "lora_strength": "lora_strength",
    "lora_trigger": "prompt_hash",  # composed into every prompt when a lora is set
}
"""D4's contract, made checkable: for every :data:`LOOK_FIELDS` entry, the
``ChunkFingerprint`` field (or ``"prompt_hash"``) that would move if this
look field changed. ``tests/test_profiles.py`` asserts every value here is
either ``"prompt_hash"`` or a name in one of ``ChunkFingerprint``'s own
``CONDITIONING_FIELDS``/``CONTENT_FIELDS``/``TIMELINE_FIELDS`` tuples, so a
look field admitted to :data:`LOOK_FIELDS` with no fingerprint evidence is a
test failure, not a silent gap."""

TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"version", "name", "description", "provenance"}
) | frozenset(LOOK_FIELDS)
"""Closed set of keys a profile file may set at the top level (D8)."""

PROVENANCE_KEYS: frozenset[str] = frozenset(
    {
        "source",
        "song",
        "promoted_from",
        "model",
        "generated_at",
        "promoted_at",
        "concept_hash",
        "stance_index",
        "notes",
    }
)
"""Closed set of keys the optional ``[provenance]`` table may contain (D8).

``generated_at`` and ``promoted_at`` are two different dates and conflating
them loses the one that matters. ``generated_at`` is when the photography
stage produced this look (its ``completed_at`` in ``session.json``);
``promoted_at`` is when a human said "that one, keep it". A look generated in
August and promoted in November is the normal case -- the gap *is* the
review -- and a single timestamp cannot say which end of it you are reading.
"""

_PROVENANCE_KEY_ORDER: tuple[str, ...] = (
    "source",
    "song",
    "promoted_from",
    "model",
    "generated_at",
    "promoted_at",
    "concept_hash",
    "stance_index",
    "notes",
)
"""Stable rendering order for :func:`render_profile_toml` -- cosmetic only;
:data:`PROVENANCE_KEYS` is the actual closed set."""


class ProfileError(ValueError):
    """Raised when a cinematography profile file fails to parse or validate,
    or when :func:`promote_photography` cannot build one. ``config.py``
    catches this at its one call site and re-raises ``ConfigError`` (D7)."""


@dataclass(frozen=True)
class Profile:
    """One loaded, validated cinematography profile."""

    path: Path
    """Resolved path the profile was read from."""
    sha256: str
    """sha256 of the file's raw bytes, so a later edit is provable (D5)."""
    version: int
    name: str
    description: str | None
    values: Mapping[str, object]
    """Only the look fields actually present in the file, in
    :data:`LOOK_FIELDS` order regardless of the order they appeared in the
    file itself."""
    provenance: Mapping[str, object]
    """The ``[provenance]`` table, verbatim. Possibly empty."""

    def to_record(self) -> dict:
        """JSON-safe representation, for :func:`write_profile_record`."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "values": dict(self.values),
            "provenance": dict(self.provenance),
        }


def _misplaced_provenance_message(key: str) -> str:
    kind = "a top-level look field" if key in LOOK_FIELDS else "a top-level field"
    return (
        f"{key!r} is {kind}; TOML has bound it to [provenance] because it was written "
        "below that table. Move it above the first table."
    )


def _validated_look_value(field: str, value: object) -> object:
    """Type/non-emptiness check only (D6) -- domain validation (e.g. that
    ``face_treatment`` is one of a closed set) is ``config.py``'s job alone,
    and duplicating it here would give the project two sources of truth that
    can drift. A blank string is rejected rather than treated as absent
    (unlike ``config._optional_text``'s run-config convention): a profile's
    whole purpose is to lock an explicit value, so a field present but empty
    reads as an authoring mistake, not "not specified"."""
    if field == "lora_strength":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileError(f"{field} must be a number, got {value!r}")
        return float(value)
    if not isinstance(value, str):
        raise ProfileError(f"{field} must be a string, got {value!r}")
    stripped = value.strip()
    if not stripped:
        raise ProfileError(
            f"{field} is present but blank -- a profile locks explicit values; an "
            "unfilled look field should be omitted, not set to an empty string"
        )
    return stripped


def load_profile(path: Path | str) -> Profile:
    """Read and validate one cinematography profile TOML file.

    Raises :class:`ProfileError` for a missing/unreadable file, malformed
    TOML, an unknown top-level or ``[provenance]`` key (with D8's misplaced-
    key hint when applicable), a missing/invalid ``version`` or ``name``, a
    wrongly-typed look field, or a profile that sets none of
    :data:`LOOK_FIELDS` at all -- a profile that locks nothing is a mistake,
    not a no-op.
    """
    resolved = Path(path).resolve()
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        logger.exception("Cannot read cinematography profile %s", resolved)
        raise ProfileError(f"cannot read profile {resolved}: {exc}") from exc

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        logger.exception("Cinematography profile %s is not valid UTF-8", resolved)
        raise ProfileError(f"profile {resolved} is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        logger.exception("Malformed TOML in cinematography profile %s", resolved)
        raise ProfileError(f"malformed TOML in profile {resolved}: {exc}") from exc

    unknown_top = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown_top:
        raise ProfileError(
            f"profile {resolved} has unknown top-level key(s): {', '.join(unknown_top)}. "
            f"Valid keys are: {', '.join(sorted(TOP_LEVEL_KEYS))}"
        )

    provenance_raw = raw.get("provenance", {})
    if provenance_raw is None:
        provenance_raw = {}
    if not isinstance(provenance_raw, dict):
        raise ProfileError(f"profile {resolved}: [provenance] must be a table")
    unknown_provenance = sorted(set(provenance_raw) - PROVENANCE_KEYS)
    if unknown_provenance:
        misplaced = [k for k in unknown_provenance if k in TOP_LEVEL_KEYS]
        if misplaced:
            raise ProfileError(
                f"profile {resolved}: {_misplaced_provenance_message(misplaced[0])}"
            )
        raise ProfileError(
            f"profile {resolved}: [provenance] has unknown key(s): "
            f"{', '.join(unknown_provenance)}. Valid keys are: "
            f"{', '.join(sorted(PROVENANCE_KEYS))}"
        )

    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProfileError(
            f"profile {resolved}: version must be an int >= 1 (a deliberate bump, "
            f"never a side effect), got {version!r}"
        )

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProfileError(f"profile {resolved}: name must be a non-empty string, got {name!r}")
    name = name.strip()

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ProfileError(f"profile {resolved}: description must be a string, got {description!r}")
    description = description.strip() if isinstance(description, str) else None
    description = description or None

    values: dict[str, object] = {}
    for field in LOOK_FIELDS:
        if field in raw and raw[field] is not None:
            values[field] = _validated_look_value(field, raw[field])

    if not values:
        raise ProfileError(
            f"profile {resolved} sets none of {LOOK_FIELDS} -- a profile that locks "
            "nothing is a mistake, not a no-op"
        )

    profile = Profile(
        path=resolved,
        sha256=sha256,
        version=version,
        name=name,
        description=description,
        values=values,
        provenance=dict(provenance_raw),
    )
    logger.info(
        "Loaded cinematography profile %s v%d from %s (locks: %s)",
        profile.name,
        profile.version,
        resolved,
        sorted(values),
    )
    return profile


def resolve_look(
    profile: Profile | None, run_values: Mapping[str, object]
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Effective look values, and the names of the fields the run config
    overrode (D2). ``run_values`` is what the run config (plus CLI overrides)
    supplied -- a key whose value is ``None`` or absent counts as unset.

    With no profile there is nothing to inherit or override, so this returns
    ``({}, ())`` regardless of what ``run_values`` sets: the run's own values
    already stand on their own and this function's job is only to describe
    what a *profile* contributes.

    Pure; logs nothing -- the caller (``config.load_config``) is the one
    place that knows enough (the profile's name, version and path) to log a
    message worth reading, and this function may be called from tests with
    no such context.
    """
    if profile is None:
        return {}, ()

    effective: dict[str, object] = dict(profile.values)
    overridden: list[str] = []
    for field in LOOK_FIELDS:
        run_value = run_values.get(field)
        if run_value is not None:
            effective[field] = run_value
            if field in profile.values:
                overridden.append(field)
    return effective, tuple(overridden)


def write_profile_record(
    profile: Profile,
    effective: Mapping[str, object],
    overridden: tuple[str, ...],
    dest: Path | str,
) -> Path:
    """Write the JSON sidecar D5 exists for: a hash proves *change*, this
    proves *content*. Six months after a profile has moved to v3, this file
    is what lets a finished video still say what look produced it.

    Atomic-ish (temp file + replace) so a crash mid-write never leaves a
    half-written sidecar next to a run's other outputs; creates ``dest``'s
    parent directories.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": PROFILE_FORMAT_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(profile.path),
        "profile_sha256": profile.sha256,
        "profile": profile.to_record(),
        "effective": dict(effective),
        "overridden_by_run_config": list(overridden),
    }
    tmp = dest.with_name(dest.name + ".tmp")
    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(dest)
    logger.info(
        "Wrote cinematography profile record to %s (profile=%s v%d)",
        dest,
        profile.name,
        profile.version,
    )
    return dest


def _toml_string(value: str) -> str:
    """Render a Python string as a TOML basic string.

    ``json.dumps(..., ensure_ascii=False)`` escapes exactly the two
    characters a TOML basic string cares about (``"`` and ``\\``), using the
    same backslash escapes TOML itself defines, and leaves every other
    character -- including an em-dash or any other non-ASCII text -- as
    literal UTF-8 rather than a ``\\uXXXX`` escape.
    """
    return json.dumps(value, ensure_ascii=False)


def render_profile_toml(
    *,
    version: int,
    name: str,
    description: str | None = None,
    values: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> str:
    """Pure text renderer for a profile TOML file -- no filesystem access, so
    a test can assert on the exact bytes produced and :func:`load_profile`
    can re-parse the result to check the round trip.
    """
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProfileError(f"version must be an int >= 1 to render, got {version!r}")
    if not isinstance(name, str) or not name.strip():
        raise ProfileError(f"name must be a non-empty string to render, got {name!r}")

    lines: list[str] = [f"version = {version}", f"name = {_toml_string(name.strip())}"]
    if description:
        lines.append(f"description = {_toml_string(description)}")

    look_lines: list[str] = []
    for field in LOOK_FIELDS:
        if field not in values or values[field] is None:
            continue
        value = values[field]
        if field == "lora_strength":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProfileError(f"{field} must be a number to render, got {value!r}")
            look_lines.append(f"{field} = {float(value)!r}")
        else:
            if not isinstance(value, str):
                raise ProfileError(f"{field} must be a string to render, got {value!r}")
            look_lines.append(f"{field} = {_toml_string(value)}")
    if look_lines:
        lines.append("")
        lines.extend(look_lines)

    if provenance:
        unknown = sorted(set(provenance) - PROVENANCE_KEYS)
        if unknown:
            raise ProfileError(
                f"provenance contains unknown key(s): {', '.join(unknown)}. Valid keys "
                f"are: {', '.join(sorted(PROVENANCE_KEYS))}"
            )
        lines.append("")
        lines.append("[provenance]")
        for key in _PROVENANCE_KEY_ORDER:
            if key not in provenance or provenance[key] is None:
                continue
            pvalue = provenance[key]
            if isinstance(pvalue, bool):
                lines.append(f"{key} = {'true' if pvalue else 'false'}")
            elif isinstance(pvalue, (int, float)):
                lines.append(f"{key} = {pvalue!r}")
            elif isinstance(pvalue, str):
                lines.append(f"{key} = {_toml_string(pvalue)}")
            else:
                raise ProfileError(
                    f"provenance.{key} must be a string, number or bool to render, "
                    f"got {pvalue!r}"
                )

    return "\n".join(lines) + "\n"


def promote_photography(
    run_dir: Path | str,
    *,
    name: str,
    version: int = 1,
    out_path: Path | str,
    description: str | None = None,
    overwrite: bool = False,
    provenance_extra: Mapping[str, object] | None = None,
) -> Path:
    """Promote one run's approved ``photography.json`` into a reusable,
    versioned cinematography profile -- issue #55's "that one, keep it"
    (design step 3).

    Reads plain JSON from ``<run_dir>/.authoring/`` via stdlib only, and
    **never imports ``music_video_maker.authoring``**: that boundary is
    mechanically enforced for the render path by
    ``tests/test_authoring_boundary.py``, and this module cannot import
    anything under ``music_video_maker`` at all (D7), so the two constraints
    reinforce each other.

    Refuses to clobber an existing ``out_path`` unless ``overwrite=True`` --
    a locked house style is real work, the same reasoning ``--prepare`` uses
    to refuse overwriting an existing shot plan.
    """
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise ProfileError(
            f"{out_path} already exists -- a locked house style is real work; pass "
            "overwrite=True to replace it deliberately"
        )
    if not isinstance(name, str) or not name.strip():
        raise ProfileError(f"name must be a non-empty string, got {name!r}")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProfileError(f"version must be an int >= 1, got {version!r}")

    authoring_dir = run_dir / ".authoring"
    photography_path = authoring_dir / "photography.json"
    try:
        photography_raw = json.loads(photography_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.exception("Cannot read %s to promote a cinematography profile", photography_path)
        raise ProfileError(f"cannot read {photography_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        logger.exception("Malformed JSON in %s", photography_path)
        raise ProfileError(f"malformed JSON in {photography_path}: {exc}") from exc

    cinematography = photography_raw.get("cinematography")
    if not isinstance(cinematography, str) or not cinematography.strip():
        raise ProfileError(
            f"{photography_path} has no usable 'cinematography' string (got "
            f"{cinematography!r}). The photography stage returns nothing for it when "
            "the run config's own `cinematography` already fixes the look (see "
            "authoring/photography.py's Photography.cinematography docstring) -- "
            "promote from the run config's own `cinematography` value by hand instead"
        )
    cinematography = cinematography.strip()

    session_file = authoring_dir / "session.json"
    model: str | None = None
    completed_at: str | None = None
    concept_hash: str | None = None
    if session_file.exists():
        try:
            session_raw = json.loads(session_file.read_text(encoding="utf-8"))
            photography_stage = session_raw.get("stages", {}).get("photography", {}) or {}
            model = photography_stage.get("model")
            completed_at = photography_stage.get("completed_at")
            concept_hash = (photography_stage.get("input_hashes", {}) or {}).get("concept")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read %s for provenance (%s) -- promoting with partial "
                "provenance instead of failing the promotion over optional metadata",
                session_file,
                exc,
            )
    else:
        logger.warning(
            "%s not found -- promoting %s with no model/completed_at/concept_hash "
            "provenance recorded",
            session_file,
            photography_path,
        )

    provenance: dict[str, object] = {
        "source": "promoted",
        "promoted_from": str(photography_path),
        "promoted_at": datetime.now(timezone.utc).date().isoformat(),
    }
    if model:
        provenance["model"] = model
    if completed_at:
        provenance["generated_at"] = completed_at
    if concept_hash:
        provenance["concept_hash"] = concept_hash
    if provenance_extra:
        unknown_extra = sorted(set(provenance_extra) - PROVENANCE_KEYS)
        if unknown_extra:
            raise ProfileError(
                f"provenance_extra contains unknown key(s): {', '.join(unknown_extra)}. "
                f"Valid keys are: {', '.join(sorted(PROVENANCE_KEYS))}"
            )
        provenance.update(provenance_extra)

    text = render_profile_toml(
        version=version,
        name=name.strip(),
        description=description,
        values={"cinematography": cinematography},
        provenance=provenance,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info(
        "Promoted %s to cinematography profile %r v%d at %s",
        photography_path,
        name,
        version,
        out_path,
    )
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.profiles",
        description="Lock a house cinematography style into a versioned profile (issue #55).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser(
        "promote",
        help="Promote one run's approved .authoring/photography.json into a profile file.",
    )
    promote.add_argument(
        "--run-dir", required=True, type=Path, help="Run directory containing .authoring/"
    )
    promote.add_argument("--name", required=True, help="Profile identity, e.g. 'refestramus-house'")
    promote.add_argument("--out", required=True, type=Path, help="Where to write the profile TOML")
    promote.add_argument("--version", type=int, default=1, help="Profile version (default: 1)")
    promote.add_argument("--description", default=None)
    promote.add_argument(
        "--overwrite", action="store_true", help="Replace an existing file at --out"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """``python -m music_video_maker.profiles promote ...``

    Deferred, narrowly-scoped exception to D7's "no imports from
    ``music_video_maker``": ``logging_setup`` has no dependency on
    ``config`` or this module, so importing it here -- inside the CLI
    entrypoint only, never at module scope -- carries none of the
    circularity risk that rule exists to avoid, while still honouring the
    global logging standard (stderr, configured before anything runs).
    """
    from music_video_maker.logging_setup import configure_logging

    configure_logging()

    args = _build_arg_parser().parse_args(argv)
    try:
        if args.command == "promote":
            path = promote_photography(
                args.run_dir,
                name=args.name,
                version=args.version,
                out_path=args.out,
                description=args.description,
                overwrite=args.overwrite,
            )
            logger.info("Wrote cinematography profile to %s", path)
        return 0
    except ProfileError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
