"""Shot provenance: which backend made each lip-synced shot, and the proof.

A generated shot is only as usable as the licence of whatever generated it.
The MiniMax H3 open weights, for one, carry a Community License that names
the United States an Excluded Territory, so a shot made by running those
weights on a machine in Chicago cannot be exhibited -- while the same model
behind MiniMax's hosted API is governed by the platform's Terms of Service
instead, under which the client keeps ownership of generated content. The
pixels look the same. Nothing about an mp4 on disk says which path made it.

So every shot a lip-sync backend produces is recorded here, by shot id, with
the backend, model, quality tier, task id, timestamps, and SHA-256 hashes of
every input and of the output. A finished package ships this file alongside
its conformance report, and :func:`assert_final_grade` is the one function a
renderer, a publisher or a packager calls to refuse work that contains a shot
from a backend that may not be exhibited.

Stdlib-only on purpose, like ``dome.py``: the guard runs on the render host,
which has ffmpeg and nothing else installed, and ``lipsync.py`` (which needs
``requests``) imports *this*, never the other way round.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 1

BACKEND_MINIMAX_API = "minimax_api"
BACKEND_LOCAL_DRAFT = "local_draft"
BACKEND_LOCAL_H3 = "local_h3"
BACKENDS = (BACKEND_MINIMAX_API, BACKEND_LOCAL_DRAFT, BACKEND_LOCAL_H3)
"""Every lip-sync / image-to-video backend a shot can name. Closed set."""

EXHIBITABLE_BACKENDS = frozenset({BACKEND_MINIMAX_API})
"""The backends whose output may appear in a final render, a publish or a
package. ``local_h3`` is the open weights (territorially restricted);
``local_draft`` is whatever free local model iterates on a workstation --
cheap, and never shipped."""

TIER_DRAFT = "draft"
TIER_FINAL = "final"
TIERS = (TIER_DRAFT, TIER_FINAL)
"""Quality tiers, independent of backend: ``draft`` is low resolution for
iteration, ``final`` is full resolution through an exhibitable backend."""


class ProvenanceError(RuntimeError):
    """A provenance file is missing, malformed, or names an unknown backend."""


class FinalRenderGuardError(ProvenanceError):
    """A final-grade step (render, publish, package) was asked to carry a shot
    that may not be exhibited. The message names every offending shot."""


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed (inputs are megabytes, outputs can be
    hundreds)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ShotProvenance:
    """One generated shot. ``inputs`` maps a role (``reference_image``,
    ``reference_audio``, ...) to the SHA-256 of the file that filled it."""

    shot_id: str
    backend: str
    model: str
    tier: str
    resolution: str | None = None
    frames: int | None = None
    fps: int | None = None
    duration_seconds: float | None = None
    task_id: str | None = None
    requested_at_utc: str | None = None
    completed_at_utc: str | None = None
    inputs: Mapping[str, str] = field(default_factory=dict)
    output_file: str | None = None
    output_sha256: str | None = None
    cost_usd: float | None = None
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ProvenanceError(
                f"shot {self.shot_id!r}: backend must be one of {BACKENDS}, got {self.backend!r}"
            )
        if self.tier not in TIERS:
            raise ProvenanceError(
                f"shot {self.shot_id!r}: tier must be one of {TIERS}, got {self.tier!r}"
            )

    @property
    def exhibitable(self) -> bool:
        return self.backend in EXHIBITABLE_BACKENDS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["inputs"] = dict(self.inputs)
        data["notes"] = dict(self.notes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ShotProvenance:
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ProvenanceError(f"shot record has unknown field(s): {unknown}")
        try:
            return cls(**data)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ProvenanceError(f"malformed shot record: {exc}") from exc


@dataclass
class Provenance:
    """The file: every shot, plus the terms the exhibitable backend relies on
    (URL, effective date, and the ownership clause quoted verbatim), so the
    package carries its own justification."""

    shots: list[ShotProvenance] = field(default_factory=list)
    terms: dict[str, object] = field(default_factory=dict)
    generator: str = "music_video_maker.provenance"
    created_at_utc: str = field(default_factory=utc_now_iso)
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    def add(self, shot: ShotProvenance) -> None:
        """Insert or replace by ``shot_id`` -- a re-run of one shot overwrites
        its own record and nobody else's."""
        self.shots = [s for s in self.shots if s.shot_id != shot.shot_id]
        self.shots.append(shot)

    def find(self, shot_id: str) -> ShotProvenance | None:
        for shot in self.shots:
            if shot.shot_id == shot_id:
                return shot
        return None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generator": self.generator,
            "created_at_utc": self.created_at_utc,
            "terms": dict(self.terms),
            "shots": [shot.to_dict() for shot in self.shots],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Provenance:
        if not isinstance(data, Mapping) or "shots" not in data:
            raise ProvenanceError("provenance must be an object with a 'shots' list")
        shots_raw = data["shots"]
        if not isinstance(shots_raw, list):
            raise ProvenanceError("'shots' must be a list")
        return cls(
            shots=[ShotProvenance.from_dict(s) for s in shots_raw],
            terms=dict(data.get("terms") or {}),  # type: ignore[arg-type]
            generator=str(data.get("generator") or "unknown"),
            created_at_utc=str(data.get("created_at_utc") or utc_now_iso()),
            schema_version=int(data.get("schema_version") or PROVENANCE_SCHEMA_VERSION),
        )


def load_provenance(path: Path | str) -> Provenance:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceError(f"provenance file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"provenance file {path} is not valid JSON: {exc}") from exc
    return Provenance.from_dict(payload)


def save_provenance(provenance: Provenance, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(provenance.to_dict(), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def record_shot(path: Path | str, shot: ShotProvenance, *,
                terms: Mapping[str, object] | None = None) -> Provenance:
    """Append-or-replace one shot in the provenance file at ``path``, creating
    the file if needed. Written after *every* shot, not at the end of a run:
    a run that dies on shot 7 must still prove shots 1-6."""
    path = Path(path)
    provenance = load_provenance(path) if path.exists() else Provenance()
    if terms:
        provenance.terms = dict(terms)
    provenance.add(shot)
    save_provenance(provenance, path)
    return provenance


def merge_provenance(paths: Iterable[Path | str]) -> Provenance:
    """One file from several -- a package made of shots from more than one
    run directory. Later files win on a shared ``shot_id``; the first non-empty
    ``terms`` is kept."""
    merged = Provenance()
    for path in paths:
        part = load_provenance(path)
        if part.terms and not merged.terms:
            merged.terms = dict(part.terms)
        for shot in part.shots:
            merged.add(shot)
    return merged


def assert_final_grade(provenance: Provenance, *, tier: str = TIER_FINAL,
                       expected_shot_ids: Iterable[str] | None = None) -> None:
    """Refuse a final-grade step that would carry a non-exhibitable shot.

    Two checks, both hard:

    * **licence** -- every shot's backend is in :data:`EXHIBITABLE_BACKENDS`.
      This is the one that matters legally, and it applies whatever ``tier``
      is asked for.
    * **tier** -- when ``tier`` is ``final``, every shot was itself made at the
      ``final`` tier. A 480P draft through the hosted API is lawful but is not
      the deliverable, and silently shipping one inside a final package is
      the kind of mistake this file exists to make loud.

    ``expected_shot_ids``, when given, must all be present: a package that
    *forgot* to record a shot proves nothing about it.
    """
    if tier not in TIERS:
        raise ProvenanceError(f"tier must be one of {TIERS}, got {tier!r}")
    if not provenance.shots:
        raise FinalRenderGuardError(
            "provenance lists no shots at all -- nothing is proven; record every "
            "generated shot before a final render, publish or package"
        )
    problems: list[str] = []
    for shot in provenance.shots:
        if not shot.exhibitable:
            problems.append(
                f"shot {shot.shot_id!r} was made by backend {shot.backend!r} "
                f"(model {shot.model!r}), which may not be exhibited"
            )
        elif tier == TIER_FINAL and shot.tier != TIER_FINAL:
            problems.append(
                f"shot {shot.shot_id!r} is a {shot.tier!r}-tier shot "
                f"({shot.model!r} at {shot.resolution or 'unknown resolution'}); "
                f"a final render needs final-tier shots"
            )
    if expected_shot_ids is not None:
        present = {shot.shot_id for shot in provenance.shots}
        missing = sorted(set(expected_shot_ids) - present)
        if missing:
            problems.append(f"shot(s) with no provenance record: {missing}")
    if problems:
        raise FinalRenderGuardError(
            f"refusing the {tier} step: " + "; ".join(problems)
        )
