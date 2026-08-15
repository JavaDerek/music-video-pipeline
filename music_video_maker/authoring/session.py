"""Session state (issue #54 design section 7): what has been generated, from
what inputs, and whether it is stale.

Lives beside the run config -- ``~/mvm-runs/<song>/.authoring/session.json``
-- never in the repo (same convention as every other run asset; see project
``CLAUDE.md``'s "Run assets live outside the repo").

**Never auto-heals.** A stage whose recorded inputs no longer match what is
on disk now is reported as stale; nothing here regenerates it automatically.
Auto-regenerating would throw away a human's approval of the previous output
to satisfy a hash -- the same reasoning ``contracts.ChunkFingerprint``
already uses for render-time resume: detect, report, refuse to guess.

Phases 1-2 (as landed) write records for ``concept`` and ``beats``;
``photography``/``prose`` don't exist as runnable code yet. :data:`STAGES`
names all four, in pipeline order, so ``mvm-author status`` has a stable
shape to report against and a stage that cannot run yet is reported from
what is persisted rather than given a guessed staleness verdict.

**Cascading falls out of the input hashes rather than being its own
mechanism.** ``beats`` records a hash of the concept it descended from, so
re-rolling the concept reports ``beats`` as stale by exactly the comparison
that catches an edited lyrics file -- no separate dependency graph, and
nothing that could disagree with what a stage actually consumed. Each new
stage gets the cascade by hashing its upstream's output, the way
:func:`~music_video_maker.authoring.beats.beats_input_hashes` does.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = ("concept", "beats", "photography", "prose")
"""Every stage the design names (section 4), in pipeline order."""

SESSION_DIRNAME = ".authoring"
SESSION_FILENAME = "session.json"


class SessionError(RuntimeError):
    """Raised when an authoring session file is malformed, unreadable, or
    references an unknown stage."""


@dataclass(frozen=True)
class StageRecord:
    """One completed stage run, as persisted in ``session.json``.

    ``input_hashes`` is sha256 of everything this run actually consumed --
    e.g. for ``concept``: the lyrics file, the chunk skeleton, and
    ``docs/lyrics-format.md``. Compared against freshly recomputed hashes to
    detect staleness (:func:`stage_staleness`) -- never against content.
    """

    stage: str
    model: str
    completed_at: str
    cost_usd: float | None
    input_hashes: dict[str, str]


def session_path(run_dir: Path | str) -> Path:
    """Where a run's session file lives, given the run's own directory
    (i.e. the run config's parent -- the same directory ``--prepare``
    defaults ``shot_plan.toml`` into)."""
    return Path(run_dir) / SESSION_DIRNAME / SESSION_FILENAME


@dataclass
class AuthoringSession:
    """Reads and writes one run's ``.authoring/session.json``."""

    path: Path
    records: dict[str, StageRecord] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> AuthoringSession:
        path = Path(path)
        if not path.exists():
            return cls(path=path, records={})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to read authoring session %s", path)
            raise SessionError(f"could not read authoring session {path}: {exc}") from exc

        records: dict[str, StageRecord] = {}
        for stage, entry in payload.get("stages", {}).items():
            if stage not in STAGES:
                logger.warning(
                    "Authoring session %s names unknown stage %r; ignoring its record",
                    path,
                    stage,
                )
                continue
            records[stage] = StageRecord(
                stage=stage,
                model=entry["model"],
                completed_at=entry["completed_at"],
                cost_usd=entry.get("cost_usd"),
                input_hashes=dict(entry.get("input_hashes", {})),
            )
        return cls(path=path, records=records)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stages": {
                stage: {
                    "model": record.model,
                    "completed_at": record.completed_at,
                    "cost_usd": record.cost_usd,
                    "input_hashes": record.input_hashes,
                }
                for stage, record in self.records.items()
            }
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("Wrote authoring session state to %s", self.path)

    def record(self, stage_record: StageRecord) -> None:
        if stage_record.stage not in STAGES:
            raise SessionError(
                f"unknown stage {stage_record.stage!r}; known stages: {STAGES}"
            )
        self.records[stage_record.stage] = stage_record

    def get(self, stage: str) -> StageRecord | None:
        return self.records.get(stage)


@dataclass(frozen=True)
class StalenessCheck:
    stale: bool
    reason: str | None
    """``None`` exactly when ``stale`` is ``False``."""


def stage_staleness(
    record: StageRecord | None, current_hashes: Mapping[str, str]
) -> StalenessCheck:
    """Whether a stage's recorded run (if any) is stale against
    ``current_hashes`` -- its inputs, recomputed right now.

    A pure function of the record and the current hashes, deliberately: a
    caller supplies whichever hashes are relevant to the stage in question
    (:mod:`~music_video_maker.authoring.concept` knows what *its* stage
    consumes; this module does not need to)."""
    if record is None:
        return StalenessCheck(stale=True, reason="not started")

    current = dict(current_hashes)
    if record.input_hashes == current:
        return StalenessCheck(stale=False, reason=None)

    changed = sorted(
        key
        for key in set(record.input_hashes) | set(current)
        if record.input_hashes.get(key) != current.get(key)
    )
    return StalenessCheck(
        stale=True, reason=f"input(s) changed since this ran: {', '.join(changed)}"
    )


__all__ = [
    "STAGES",
    "AuthoringSession",
    "SessionError",
    "StageRecord",
    "StalenessCheck",
    "session_path",
    "stage_staleness",
]
