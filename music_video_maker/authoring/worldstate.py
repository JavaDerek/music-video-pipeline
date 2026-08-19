"""World-state core timeline (run-dmcp design, FINAL DRAFT v1.0, spec §5.1).

The music-video side's committed prototype of that cross-project engine's
core: a small, working implementation of the same three-type model, built to
be swappable for the real engine behind the same seam later. Zero imports
from the rest of ``music_video_maker`` -- ``t`` is a bare float, not an
``AudioChunk`` -- which is also what keeps this module invisible to
``tests/test_authoring_boundary.py``: there is nothing here to wire into the
render path.

**The three types (spec §5.1, adopted verbatim plus one field, see below):**

* :class:`Entity` -- ``id``, ``kind``, ``name``, ``created_at_t``,
  ``destroyed_at_t | None``.
* :class:`Fact` -- ``id``, ``entity_id``, ``key``, ``value``,
  ``valid_from_t``, ``valid_to_t | None``, ``irreversible``. Plus
  ``opened_by_event_id``, a deliberate addition beyond the literal spec
  wording: it is what makes "one hop of causality" (see
  :class:`IrreversibleFactViolation`) O(1) without repurposing
  ``Event.causes``, which is left free for the client's own event-to-event
  narrative annotation instead of fact provenance.
* :class:`Event` -- ``id``, ``at_t``, ``kind``, ``description``, ``causes``
  (an opaque tuple of ids the engine never validates or reads back).

**``t`` is story time.** For this client that is song time in seconds, a
float. The rule that decided this, and it is testable: *t is the axis that
remains invariant under re-segmentation of the client's own units.* Chunk
index is the wrong answer -- nothing in this module is indexed by chunk id,
and nothing here imports anything that could tempt it to be.

**Facts are interval-versioned. Nothing is ever overwritten.** Setting a key
closes the previous row's ``valid_to_t`` and opens a new one; the old row
stays in the log exactly as it was written, forever. :meth:`WorldState.facts`
is append-only.

**There is exactly one place a new fact version can be created:**
:meth:`WorldState.set_fact`. Irreversibility is enforced there, and nowhere
else: before opening a new ``(entity_id, key)`` version it looks up the
currently-open row for that pair and rejects the write if that row is
``irreversible`` and the new value differs. Because that is the only legal
path to a new version, "an irreversible fact is never contradicted" holds
for the whole log by construction -- there is no history to re-check,
because a contradicting write could never have been accepted in the first
place.

:meth:`WorldState.destroy_entity` is sugar over that SAME choke point, not a
second, parallel irreversibility mechanism: it writes an irreversible fact
under the module-reserved :data:`EXISTENCE_KEY`, which ordinary
:meth:`WorldState.set_fact` calls are refused from touching directly.

**The two queries are pure reads over the log, with no verdict baked in.**
:meth:`WorldState.replay` is the load-bearing snapshot query.
:meth:`WorldState.changes_within` returns raw transitions in a half-open
interval -- events, entities created/destroyed, facts opened -- and nothing
resembling a boolean or a severity. *The engine provides the query; the
client declares the policy*: a continuous-take renderer reads a mid-interval
change as a defect, a turn-based consumer reads the same query to build a
briefing. Both are legitimate readings of the identical output.

**No English-understanding checks live here.** This module ships zero
domain vocabulary -- no noun lists, no trouble-word lists, nothing.
:func:`check_claim` is the legitimate version of that check: it matches
free text against tokens the CALLER supplied, per call (``nouns``) and per
fact (``Fact.contradicted_by``) -- never words this module invented.
Verifiable mechanically: ``worldstate.py``'s own source contains no
song-specific noun, checked by
``tests/test_authoring_worldstate.py::test_worldstate_module_source_contains_no_client_specific_vocabulary``,
the same "a machine re-checks the rule on every commit" discipline
``tests/test_repo_assets.py`` and ``tests/test_authoring_boundary.py``
apply to their own invariants. Inferring meaning from arbitrary English is
not legitimate here, and has failed every time it was tried elsewhere in
this codebase.

**Export/import (spec §6).** :meth:`WorldState.save`/:meth:`WorldState.load`
serialize the whole timeline -- every fact version ever opened, not just
what is currently valid -- to a JSON file a consumer depends on, matching
``.authoring/``'s existing conventions (``session.json``, ``beats.json``):
``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing newline, and a
``schema_version`` int (:data:`WORLDSTATE_SCHEMA_VERSION`) read the same
"detect, report, refuse to guess" way ``resilience.py``'s
``RUN_STATE_SCHEMA_VERSION`` is. The round-trip is lossless: a restored
:class:`WorldState` answers :meth:`replay`/:meth:`changes_within`
identically to the one it was written from, and still enforces
irreversibility at the same one choke point -- it is not a read-only copy.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

EXISTENCE_KEY = "__destroyed__"
"""Module-reserved fact key :meth:`WorldState.destroy_entity` uses to record
destruction through the same choke point every other fact goes through.
:meth:`WorldState.set_fact` refuses this key (:class:`ReservedFactKeyError`)
so a caller can never collide with it -- entity destruction has exactly one
legitimate entry point, :meth:`WorldState.destroy_entity`. Public (not
underscore-prefixed) precisely so a caller can recognize and avoid it."""

WORLDSTATE_SCHEMA_VERSION = 1
"""Bumped whenever :meth:`WorldState.to_dict`'s shape changes in a way an
older file can no longer be read as. :meth:`WorldState.from_dict` refuses
any other value outright (:class:`WorldStateFileError`) -- the same
"detect, report, refuse to guess" discipline ``resilience.py``'s
``RUN_STATE_SCHEMA_VERSION`` already uses for ``run_state.json``."""


class WorldStateError(ValueError):
    """Base class for every error this module raises. Rooted in
    ``ValueError``, matching this package's own convention
    (``ConceptValidationError``, ``BeatsValidationError``,
    ``ProseValidationError`` are all ``ValueError`` subclasses, not
    ``RuntimeError``)."""


class UnknownEntityError(WorldStateError):
    """Raised when an operation names an ``entity_id`` this ``WorldState``
    has no record of."""


class ReservedFactKeyError(WorldStateError):
    """Raised when a caller passes :data:`EXISTENCE_KEY` to
    :meth:`WorldState.set_fact` directly. Use
    :meth:`WorldState.destroy_entity` instead -- it writes that key through
    the same choke point, with the irreversibility this module always gives
    destruction."""


class NonMonotonicFactWriteError(WorldStateError):
    """Raised when a new fact version's ``valid_from_t`` does not strictly
    advance past the currently-open version for the same
    ``(entity_id, key)``. Covers both a tie (two writes to the same key at
    the same instant, which is ambiguous ordering) and time moving backward
    -- neither is allowed to silently last-write-win."""


class EntityAlreadyDestroyedError(WorldStateError):
    """Raised by :meth:`WorldState.destroy_entity` when the entity already
    has a ``destroyed_at_t`` -- destruction, like any irreversible fact,
    happens once."""


class WorldStateFileError(WorldStateError):
    """Raised by :meth:`WorldState.save`/:meth:`WorldState.load` for I/O
    failures, malformed JSON, or a ``schema_version`` this build does not
    write -- the same "detect, report, refuse to guess" treatment
    ``resilience.RunStateSchemaError`` gives ``run_state.json``: a file from
    an incompatible shape must never be silently misread."""


class IrreversibleFactViolation(WorldStateError):
    """Raised at the one choke point (:meth:`WorldState.set_fact`, via
    :meth:`WorldState._open_fact_version`) when a write would contradict a
    fact marked ``irreversible``.

    Carries exactly **one hop of causality** -- the contradicted fact
    (``contradicted_fact``, whose own ``valid_from_t`` is on it), the id of
    the event that opened it (``opened_by_event_id``), and the value that
    was rejected (``attempted_value``) -- never a trace of this checker's
    own reasoning. A reviewer's only decision at a fired check is "is the
    fact wrong, or is the claim wrong", and that is undecidable without
    knowing what made the fact true; it is decidable with exactly this much.
    """

    def __init__(self, *, contradicted_fact: Fact, attempted_value: Any) -> None:
        self.contradicted_fact = contradicted_fact
        self.opened_by_event_id = contradicted_fact.opened_by_event_id
        self.attempted_value = attempted_value
        super().__init__(
            f"fact {contradicted_fact.key!r} on entity {contradicted_fact.entity_id!r} "
            f"was set to {contradicted_fact.value!r} at valid_from_t="
            f"{contradicted_fact.valid_from_t} (opened by event "
            f"{contradicted_fact.opened_by_event_id!r}) and marked irreversible; "
            f"cannot set it to {attempted_value!r}"
        )


# ---------------------------------------------------------------------------
# The three core types (spec §5.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """A thing that exists in the story world from ``created_at_t`` until
    ``destroyed_at_t`` (or forever, if ``None``). Never mutated after
    creation -- destruction produces a *replacement* row in
    ``WorldState.entities``, via :meth:`WorldState.destroy_entity`, never an
    in-place edit of this one."""

    id: str
    kind: str
    name: str
    created_at_t: float
    destroyed_at_t: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation, spec §6. Round-trips losslessly
        through :meth:`from_dict`."""
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "created_at_t": self.created_at_t,
            "destroyed_at_t": self.destroyed_at_t,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Entity:
        return cls(
            id=str(payload["id"]),
            kind=str(payload["kind"]),
            name=str(payload["name"]),
            created_at_t=float(payload["created_at_t"]),
            destroyed_at_t=(
                None
                if payload.get("destroyed_at_t") is None
                else float(payload["destroyed_at_t"])
            ),
        )


@dataclass(frozen=True)
class Fact:
    """One interval-versioned statement about an entity: ``key`` had
    ``value`` from ``valid_from_t`` until ``valid_to_t`` (or still holds, if
    ``None``). Once written, never mutated or deleted -- superseding it
    closes this exact row (a new ``Fact`` with the same id, `key`, `value`
    and `valid_from_t` but a filled-in `valid_to_t`) and appends a new one;
    see :meth:`WorldState.set_fact`."""

    id: str
    entity_id: str
    key: str
    value: Any
    valid_from_t: float
    valid_to_t: float | None = None
    irreversible: bool = False
    opened_by_event_id: str | None = None
    contradicted_by: tuple[str, ...] = ()
    """Caller-injected vocabulary (Approach B), not a module default --
    empty unless the caller passes it to :meth:`WorldState.set_fact`. What
    this fact's value being contradicted reads like in generated text (e.g.
    for a ``state="destroyed"`` fact, the words that would describe it as
    intact instead). Matched by :func:`check_claim` against caller-supplied
    ``nouns``, never inferred from English; ships no words of its own."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation, spec §6. Round-trips losslessly
        through :meth:`from_dict`. ``value`` must itself be JSON-safe
        (``str``/``int``/``float``/``bool``/``None``/``list``/``dict``) --
        :meth:`WorldState.save` raises :class:`WorldStateFileError` if it
        is not."""
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "key": self.key,
            "value": self.value,
            "valid_from_t": self.valid_from_t,
            "valid_to_t": self.valid_to_t,
            "irreversible": self.irreversible,
            "opened_by_event_id": self.opened_by_event_id,
            "contradicted_by": list(self.contradicted_by),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Fact:
        return cls(
            id=str(payload["id"]),
            entity_id=str(payload["entity_id"]),
            key=str(payload["key"]),
            value=payload["value"],
            valid_from_t=float(payload["valid_from_t"]),
            valid_to_t=(
                None if payload.get("valid_to_t") is None else float(payload["valid_to_t"])
            ),
            irreversible=bool(payload.get("irreversible", False)),
            opened_by_event_id=payload.get("opened_by_event_id"),
            contradicted_by=tuple(payload.get("contradicted_by", ())),
        )


@dataclass(frozen=True)
class Event:
    """Something that happened at ``at_t``. ``causes`` is an opaque tuple of
    ids -- this engine never validates or reads it back; it exists for the
    client's own event-to-event narrative annotation."""

    id: str
    at_t: float
    kind: str
    description: str
    causes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation, spec §6. Round-trips losslessly
        through :meth:`from_dict`."""
        return {
            "id": self.id,
            "at_t": self.at_t,
            "kind": self.kind,
            "description": self.description,
            "causes": list(self.causes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Event:
        return cls(
            id=str(payload["id"]),
            at_t=float(payload["at_t"]),
            kind=str(payload["kind"]),
            description=str(payload["description"]),
            causes=tuple(payload.get("causes", ())),
        )


# ---------------------------------------------------------------------------
# Query result types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySnapshot:
    """One entity as it stood at a given ``t``: the entity row itself, plus
    the single currently-valid :class:`Fact` for each key that has ever been
    set on it (never the reserved :data:`EXISTENCE_KEY` -- destruction is
    already expressed by the entity's own absence from the snapshot)."""

    entity: Entity
    facts: Mapping[str, Fact]


@dataclass(frozen=True)
class Snapshot:
    """The return of :meth:`WorldState.replay`: every entity alive at ``t``,
    each with the facts valid at ``t``."""

    t: float
    entities: tuple[EntitySnapshot, ...]


@dataclass(frozen=True)
class ChangeSet:
    """The return of :meth:`WorldState.changes_within`: raw transitions in
    the half-open interval ``[t0, t1)`` -- events, entities created,
    entities destroyed, facts opened. **Never a verdict.** No ``is_clean``,
    no ``severity``, nothing resembling one. The engine provides the query;
    the client declares the policy."""

    t0: float
    t1: float
    events: tuple[Event, ...]
    entities_created: tuple[Entity, ...]
    entities_destroyed: tuple[Entity, ...]
    facts_opened: tuple[Fact, ...]


# ---------------------------------------------------------------------------
# The timeline itself.
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    """An append-only log of entities, facts and events, plus the two
    spec-required queries over it.

    ``entities`` maps id -> the entity's *current* row (replaced, never
    mutated in place, by :meth:`destroy_entity`). ``facts`` and ``events``
    are plain lists in insertion order -- every version ever opened stays in
    ``facts`` forever, which both gives :meth:`changes_within` a
    deterministic same-``at_t`` tie-break for free (insertion order) and is
    what "nothing is ever overwritten" means operationally.
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    # -- writes ------------------------------------------------------

    def create_entity(
        self,
        *,
        kind: str,
        name: str,
        created_at_t: float,
        entity_id: str | None = None,
    ) -> Entity:
        """Register a new entity. Raises :class:`WorldStateError` if
        ``entity_id`` is already in use."""
        eid = entity_id if entity_id is not None else f"entity-{uuid4().hex}"
        if eid in self.entities:
            raise WorldStateError(f"entity id {eid!r} already exists")
        entity = Entity(id=eid, kind=kind, name=name, created_at_t=created_at_t)
        self.entities[eid] = entity
        return entity

    def record_event(
        self,
        *,
        at_t: float,
        kind: str,
        description: str,
        causes: Sequence[str] = (),
        event_id: str | None = None,
    ) -> Event:
        """Append a new event. Raises :class:`WorldStateError` if
        ``event_id`` is already in use."""
        evid = event_id if event_id is not None else f"event-{uuid4().hex}"
        if any(e.id == evid for e in self.events):
            raise WorldStateError(f"event id {evid!r} already exists")
        event = Event(
            id=evid, at_t=at_t, kind=kind, description=description, causes=tuple(causes)
        )
        self.events.append(event)
        return event

    def set_fact(
        self,
        *,
        entity_id: str,
        key: str,
        value: Any,
        valid_from_t: float,
        opened_by_event_id: str | None = None,
        irreversible: bool = False,
        contradicted_by: Sequence[str] = (),
        fact_id: str | None = None,
    ) -> Fact:
        """The public write path for every fact except entity existence.

        Closes the currently-open ``(entity_id, key)`` row (if any) and
        opens a new one, unless the currently-open row is ``irreversible``
        and ``value`` differs from it, in which case this raises
        :class:`IrreversibleFactViolation` and writes nothing. Raises
        :class:`ReservedFactKeyError` for ``key == EXISTENCE_KEY`` -- use
        :meth:`destroy_entity`. Raises :class:`UnknownEntityError` if
        ``entity_id`` has no registered entity, and
        :class:`NonMonotonicFactWriteError` if ``valid_from_t`` does not
        strictly advance past the currently-open row for this key.

        ``contradicted_by`` is caller-injected vocabulary for
        :func:`check_claim` (Approach B) -- what this fact's value being
        contradicted reads like in generated text. Empty by default; this
        module never supplies its own.
        """
        if key == EXISTENCE_KEY:
            raise ReservedFactKeyError(
                f"{EXISTENCE_KEY!r} is reserved for entity destruction; "
                "call destroy_entity() instead of set_fact()"
            )
        return self._open_fact_version(
            entity_id=entity_id,
            key=key,
            value=value,
            valid_from_t=valid_from_t,
            opened_by_event_id=opened_by_event_id,
            irreversible=irreversible,
            contradicted_by=contradicted_by,
            fact_id=fact_id,
        )

    def destroy_entity(
        self,
        entity_id: str,
        at_t: float,
        *,
        opened_by_event_id: str | None = None,
    ) -> Entity:
        """Destroy an entity at ``at_t``.

        Sugar over the exact same choke point :meth:`set_fact` uses
        (:meth:`_open_fact_version`), writing an irreversible fact under the
        reserved :data:`EXISTENCE_KEY` -- not a second, independent
        mechanism living outside the fact log. Raises
        :class:`UnknownEntityError` if the entity does not exist, and
        :class:`EntityAlreadyDestroyedError` if it was already destroyed.
        """
        entity = self._require_entity(entity_id)
        if entity.destroyed_at_t is not None:
            raise EntityAlreadyDestroyedError(
                f"entity {entity_id!r} was already destroyed at "
                f"t={entity.destroyed_at_t}"
            )
        self._open_fact_version(
            entity_id=entity_id,
            key=EXISTENCE_KEY,
            value="destroyed",
            valid_from_t=at_t,
            opened_by_event_id=opened_by_event_id,
            irreversible=True,
            contradicted_by=(),
            fact_id=None,
        )
        destroyed = replace(entity, destroyed_at_t=at_t)
        self.entities[entity_id] = destroyed
        return destroyed

    def _require_entity(self, entity_id: str) -> Entity:
        try:
            return self.entities[entity_id]
        except KeyError:
            raise UnknownEntityError(f"no entity with id {entity_id!r}") from None

    def _current_open_fact(self, entity_id: str, key: str) -> Fact | None:
        # At most one row can legally be open per (entity_id, key) at a
        # time -- _open_fact_version is the only writer and it always closes
        # the previous one before appending a new one. Scanning to the end
        # rather than short-circuiting keeps this correct even if that
        # invariant were ever violated, at no real cost for a single song's
        # worth of facts.
        open_fact: Fact | None = None
        for f in self.facts:
            if f.entity_id == entity_id and f.key == key and f.valid_to_t is None:
                open_fact = f
        return open_fact

    def _open_fact_version(
        self,
        *,
        entity_id: str,
        key: str,
        value: Any,
        valid_from_t: float,
        opened_by_event_id: str | None,
        irreversible: bool,
        contradicted_by: Sequence[str],
        fact_id: str | None,
    ) -> Fact:
        """THE choke point. Every legal write to the fact log, including
        entity destruction, funnels through here -- which is what makes
        irreversibility a property of the whole log by construction rather
        than something re-checked after the fact."""
        self._require_entity(entity_id)
        if opened_by_event_id is not None and not any(
            e.id == opened_by_event_id for e in self.events
        ):
            raise WorldStateError(
                f"opened_by_event_id {opened_by_event_id!r} does not name a recorded event"
            )

        current = self._current_open_fact(entity_id, key)
        if current is not None:
            if valid_from_t <= current.valid_from_t:
                raise NonMonotonicFactWriteError(
                    f"cannot open a new version of ({entity_id!r}, {key!r}) at "
                    f"valid_from_t={valid_from_t}: the currently open version was "
                    f"opened at valid_from_t={current.valid_from_t}; a new write "
                    "must strictly advance past it"
                )
            if current.irreversible and current.value != value:
                raise IrreversibleFactViolation(contradicted_fact=current, attempted_value=value)
            # Irreversibility is STICKY. It was previously read off `current`
            # to reject a contradiction and then thrown away: the superseding
            # row carried whatever flag THIS call passed, which defaults to
            # False. So a legal same-value restatement silently un-froze the
            # key, and the next write could contradict it -- the exact defect
            # this module exists to catch, through its own public API. Once
            # true for an (entity_id, key), it is true for every later row.
            irreversible = irreversible or current.irreversible
            self._close_fact(current, valid_to_t=valid_from_t)

        fid = fact_id if fact_id is not None else f"fact-{uuid4().hex}"
        if any(f.id == fid for f in self.facts):
            raise WorldStateError(f"fact id {fid!r} already exists")

        new_fact = Fact(
            id=fid,
            entity_id=entity_id,
            key=key,
            value=value,
            valid_from_t=valid_from_t,
            valid_to_t=None,
            irreversible=irreversible,
            opened_by_event_id=opened_by_event_id,
            contradicted_by=tuple(contradicted_by),
        )
        self.facts.append(new_fact)
        return new_fact

    def _close_fact(self, fact: Fact, *, valid_to_t: float) -> None:
        """Replace ``fact``'s row in the log with a copy carrying
        ``valid_to_t`` -- the ONLY mutation-shaped operation in this module,
        and it never touches ``value``, ``valid_from_t`` or any other field:
        the row's history is preserved exactly, only its open-endedness
        changes. Located by id, not by dataclass equality, since equality
        would (harmlessly, but confusingly) also match were two rows ever
        identical in every field."""
        for i, existing in enumerate(self.facts):
            if existing.id == fact.id:
                self.facts[i] = replace(existing, valid_to_t=valid_to_t)
                return
        raise WorldStateError(f"fact id {fact.id!r} not found in the log")  # pragma: no cover

    # -- reads ---------------------------------------------------------

    def replay(self, t: float) -> Snapshot:
        """Full snapshot: every entity alive at ``t``, each with the facts
        valid at ``t``. The load-bearing query.

        An entity is alive at ``t`` when ``created_at_t <= t`` and
        (``destroyed_at_t is None`` or ``t < destroyed_at_t``) -- half-open,
        the same convention fact validity uses. A fact is valid at ``t``
        when ``valid_from_t <= t`` and (``valid_to_t is None`` or
        ``t < valid_to_t``).
        """
        entity_snapshots: list[EntitySnapshot] = []
        for entity in self.entities.values():
            if entity.created_at_t > t:
                continue
            if entity.destroyed_at_t is not None and t >= entity.destroyed_at_t:
                continue
            facts_by_key: dict[str, Fact] = {}
            for f in self.facts:
                if f.entity_id != entity.id or f.key == EXISTENCE_KEY:
                    continue
                if f.valid_from_t <= t and (f.valid_to_t is None or t < f.valid_to_t):
                    facts_by_key[f.key] = f
            entity_snapshots.append(EntitySnapshot(entity=entity, facts=facts_by_key))
        return Snapshot(t=t, entities=tuple(entity_snapshots))

    def changes_within(self, t0: float, t1: float) -> ChangeSet:
        """Events and fact transitions in the half-open interval
        ``[t0, t1)``. Returns transitions, never a verdict -- no boolean, no
        severity. The engine provides the query; the client declares the
        policy: a continuous-take renderer reads a mid-interval change as a
        defect, a turn-based consumer reads the same query to build a
        briefing.

        Same-``at_t``/``valid_from_t`` ties resolve by insertion order,
        since ``events``/``facts`` are plain append-only lists and this
        method filters without reordering.
        """
        events = tuple(e for e in self.events if t0 <= e.at_t < t1)
        entities_created = tuple(
            e for e in self.entities.values() if t0 <= e.created_at_t < t1
        )
        entities_destroyed = tuple(
            e
            for e in self.entities.values()
            if e.destroyed_at_t is not None and t0 <= e.destroyed_at_t < t1
        )
        facts_opened = tuple(
            f
            for f in self.facts
            if f.key != EXISTENCE_KEY and t0 <= f.valid_from_t < t1
        )
        return ChangeSet(
            t0=t0,
            t1=t1,
            events=events,
            entities_created=entities_created,
            entities_destroyed=entities_destroyed,
            facts_opened=facts_opened,
        )

    # -- export / import (spec §6) --------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The whole timeline, losslessly -- every fact version ever
        opened, not just what is currently valid. A consumer depends on
        this file, never on a live :class:`WorldState` object (spec §6)."""
        return {
            "schema_version": WORLDSTATE_SCHEMA_VERSION,
            "entities": [e.to_dict() for e in self.entities.values()],
            "facts": [f.to_dict() for f in self.facts],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorldState:
        """Inverse of :meth:`to_dict`. Raises :class:`WorldStateFileError`
        if ``schema_version`` is anything other than
        :data:`WORLDSTATE_SCHEMA_VERSION` -- detect, report, refuse to
        guess, rather than misreading an incompatible shape. The restored
        object is a live :class:`WorldState`, not a read-only copy: its
        choke point still enforces irreversibility on every subsequent
        write, and :meth:`_validate_fact_log` re-establishes on the loaded
        rows what that choke point guarantees on written ones -- so a
        restored timeline is as trustworthy as a built one, rather than
        merely being handled by the same code afterwards."""
        version = payload.get("schema_version")
        if version != WORLDSTATE_SCHEMA_VERSION:
            raise WorldStateFileError(
                f"world state schema_version={version!r}, this build reads and "
                f"writes {WORLDSTATE_SCHEMA_VERSION}"
            )
        entities: dict[str, Entity] = {}
        for raw_entity in payload.get("entities", ()):
            entity = Entity.from_dict(raw_entity)
            entities[entity.id] = entity
        facts = [Fact.from_dict(raw_fact) for raw_fact in payload.get("facts", ())]
        events = [Event.from_dict(raw_event) for raw_event in payload.get("events", ())]
        cls._validate_fact_log(facts)
        cls._validate_entity_existence(entities, facts)
        return cls(entities=entities, facts=facts, events=events)

    @staticmethod
    def _validate_entity_existence(
        entities: Mapping[str, Entity], facts: Sequence[Fact]
    ) -> None:
        """Cross-check the two records of one truth.

        ``destroy_entity`` writes ``Entity.destroyed_at_t`` *and* an
        irreversible fact under the reserved existence key, through the same
        choke point. They agree by construction on write and by nothing at
        all on load, so a payload editing only the entity row brought a
        destroyed entity back to life without touching the fact log that
        still said otherwise.
        """
        destroyed_by_fact = {
            fact.entity_id: fact.valid_from_t
            for fact in facts
            if fact.key == EXISTENCE_KEY and fact.value == "destroyed"
        }
        for entity_id, entity in entities.items():
            fact_t = destroyed_by_fact.get(entity_id)
            if entity.destroyed_at_t is None and fact_t is not None:
                raise WorldStateFileError(
                    f"entity {entity_id!r} has destroyed_at_t=None but the fact log "
                    f"records it destroyed at t={fact_t}"
                )
            if entity.destroyed_at_t is not None and fact_t is None:
                raise WorldStateFileError(
                    f"entity {entity_id!r} records destroyed_at_t="
                    f"{entity.destroyed_at_t} with no matching destruction in the "
                    "fact log"
                )
            if fact_t is not None and entity.destroyed_at_t != fact_t:
                raise WorldStateFileError(
                    f"entity {entity_id!r} records destroyed_at_t="
                    f"{entity.destroyed_at_t} but the fact log destroys it at t={fact_t}"
                )

    @staticmethod
    def _validate_fact_log(facts: Sequence[Fact]) -> None:
        """Re-establish on load what the choke point guarantees on write.

        Without this, persistence was a hole straight through the module's
        one real guarantee: a hand-edited or merged payload carrying two open
        rows for one key, or a closed irreversible row followed by a
        different value, loaded silently and ``replay`` answered from it. A
        restored timeline has to be exactly as trustworthy as a built one,
        because the whole point of the frozen-artifact boundary is that
        consumers depend on the file.

        Checked per ``(entity_id, key)`` chain, in ``valid_from_t`` order:
        at most one open row, no overlaps, and no value change after a row
        marked irreversible.
        """
        chains: dict[tuple[str, str], list[Fact]] = {}
        for fact in facts:
            chains.setdefault((fact.entity_id, fact.key), []).append(fact)

        for (entity_id, key), chain in chains.items():
            chain.sort(key=lambda f: f.valid_from_t)
            open_rows = [f for f in chain if f.valid_to_t is None]
            if len(open_rows) > 1:
                raise WorldStateFileError(
                    f"({entity_id!r}, {key!r}) has {len(open_rows)} simultaneously open "
                    f"fact versions; at most one may be open at a time"
                )
            frozen_value: Any = None
            frozen = False
            for earlier, later in zip(chain, chain[1:], strict=False):
                if earlier.valid_to_t is None:
                    raise WorldStateFileError(
                        f"({entity_id!r}, {key!r}) has an open version at "
                        f"valid_from_t={earlier.valid_from_t} followed by another at "
                        f"{later.valid_from_t}; an open version must be the last"
                    )
                if earlier.valid_to_t > later.valid_from_t:
                    raise WorldStateFileError(
                        f"({entity_id!r}, {key!r}) has overlapping versions: one closes at "
                        f"{earlier.valid_to_t}, the next opens at {later.valid_from_t}"
                    )
            for fact in chain:
                if frozen and fact.value != frozen_value:
                    raise WorldStateFileError(
                        f"({entity_id!r}, {key!r}) is marked irreversible with value "
                        f"{frozen_value!r} but a later version at "
                        f"valid_from_t={fact.valid_from_t} carries {fact.value!r}; "
                        "this log was already inconsistent when it was written"
                    )
                if fact.irreversible:
                    frozen = True
                    frozen_value = fact.value

    def save(self, path: Path | str) -> None:
        """Write the whole timeline to ``path`` as JSON -- the same
        convention ``.authoring/session.json``/``beats.json`` already use:
        ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
        newline. Creates parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        except TypeError as exc:
            logger.exception("World state contains a non-JSON-safe fact value")
            raise WorldStateFileError(
                f"could not serialize world state to JSON: {exc}"
            ) from exc
        path.write_text(text, encoding="utf-8")
        logger.info("Wrote world state to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> WorldState:
        """Read a timeline written by :meth:`save`. Raises
        :class:`WorldStateFileError` for a missing file, unreadable file,
        malformed JSON, or an incompatible ``schema_version``."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to read world state %s", path)
            raise WorldStateFileError(f"could not read world state {path}: {exc}") from exc
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# check_claim: Approach B. Built ONLY on the two public queries above --
# replay()'s Snapshot -- never given privileged internal access to the log.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contradiction:
    """One finding from :func:`check_claim`: a caller's free-text claim
    disagreed with a fact this :class:`WorldState` already holds.

    Carries exactly **one hop of causality** -- ``fact`` (whose own
    ``valid_from_t`` is on it) and ``opened_by_event_id`` -- never a trace
    of how this function matched the text. A reviewer's only decision at a
    fired :class:`Contradiction` is "is the fact wrong, or is the claim
    wrong", and that is undecidable without knowing what made the fact
    true; it is decidable with exactly this much -- the same rationale
    :class:`IrreversibleFactViolation` documents.
    """

    entity_id: str
    entity_name: str
    matched_nouns: tuple[str, ...]
    fact: Fact
    matched_tokens: tuple[str, ...]
    opened_by_event_id: str | None


def _phrase_in_text(phrase: str, lowered_text: str) -> bool:
    """Whole-word (well, whole-phrase), case-insensitive containment --
    the exact idiom ``shot_plan.py``'s own #78 landmark-contradiction lint
    and #73 role-prohibition lint already use
    (``re.search(rf"\\b{re.escape(x)}\\b", text)``), not a new matching
    strategy."""
    return re.search(rf"\b{re.escape(phrase.lower())}\b", lowered_text) is not None


def check_claim(
    snapshot: Snapshot,
    text: str,
    *,
    nouns: Mapping[str, Sequence[str]],
) -> tuple[Contradiction, ...]:
    """Match a free-text claim -- generated for the story-time ``snapshot``
    already reflects (``snapshot.t``) -- against ``snapshot``. Read-only,
    never raises, never invents a verdict about English.

    Ships zero vocabulary of its own. It matches exactly two things the
    CALLER supplied: ``nouns``, a caller-owned ``{entity_id: (phrase, ...)}``
    mapping naming each entity in generated text, and each currently-valid
    :class:`Fact`'s own ``contradicted_by`` tuple (see
    :meth:`WorldState.set_fact`). For each entity in ``snapshot`` whose
    declared nouns appear in ``text``, this scans that entity's
    currently-valid facts; a fact contributes one :class:`Contradiction` for
    every one of ITS ``contradicted_by`` tokens that also appears in
    ``text``.

    **An empty return means "found nothing to object to", never "this text
    is consistent."** It is only ever as complete as the vocabulary the
    caller injected via ``nouns`` and ``contradicted_by`` -- this must never
    be read as a completeness guarantee the mechanism cannot give.

    A destroyed entity is already absent from ``snapshot`` (see
    :meth:`WorldState.replay`), so describing it afterwards -- as a ruin or
    anything else -- can never match here; it is simply not there to check.
    """
    hits: list[Contradiction] = []
    lowered = text.lower()
    for entity_snapshot in snapshot.entities:
        entity_nouns = tuple(nouns.get(entity_snapshot.entity.id, ()))
        matched_nouns = tuple(n for n in entity_nouns if _phrase_in_text(n, lowered))
        if not matched_nouns:
            continue
        for fact in entity_snapshot.facts.values():
            matched_tokens = tuple(
                token for token in fact.contradicted_by if _phrase_in_text(token, lowered)
            )
            if not matched_tokens:
                continue
            hits.append(
                Contradiction(
                    entity_id=entity_snapshot.entity.id,
                    entity_name=entity_snapshot.entity.name,
                    matched_nouns=matched_nouns,
                    fact=fact,
                    matched_tokens=matched_tokens,
                    opened_by_event_id=fact.opened_by_event_id,
                )
            )
    return tuple(hits)


__all__ = [
    "EXISTENCE_KEY",
    "WORLDSTATE_SCHEMA_VERSION",
    "WorldStateError",
    "UnknownEntityError",
    "ReservedFactKeyError",
    "NonMonotonicFactWriteError",
    "EntityAlreadyDestroyedError",
    "IrreversibleFactViolation",
    "WorldStateFileError",
    "Entity",
    "Fact",
    "Event",
    "EntitySnapshot",
    "Snapshot",
    "ChangeSet",
    "Contradiction",
    "WorldState",
    "check_claim",
]
