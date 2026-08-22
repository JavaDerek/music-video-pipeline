"""Tests for the world-state core timeline (run-dmcp design v1.0 spec §5.1).

Covers the core model -- the three dataclasses, interval-versioned fact
writes through the single ``set_fact``/``destroy_entity`` choke point,
``replay(t)``, and ``changes_within(t0, t1)`` -- plus the two additions built
on top of it in the same module: lossless JSON export/import (spec §6) and
``check_claim``, the caller-vocabulary-only contradiction check (Approach B).

No model calls, no GPU, no ComfyUI, no chunk ids anywhere: ``t`` is bare
story-time seconds, a float, exactly as the module docstring requires. Every
song-specific word used below (the mill wheel, the flood, "lush", "turning",
...) lives ONLY in this test file's fixtures -- ``worldstate.py`` itself must
never contain one, and that is asserted mechanically near the bottom of this
file, the same "a machine re-checks the rule on every commit" discipline
``tests/test_repo_assets.py`` and ``tests/test_authoring_boundary.py`` apply
to their own invariants.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from music_video_maker.authoring import worldstate
from music_video_maker.authoring.worldstate import (
    WORLDSTATE_SCHEMA_VERSION,
    ChangeSet,
    Entity,
    EntityAlreadyDestroyedError,
    Event,
    Fact,
    IrreversibleFactViolation,
    LocatedSpan,
    NonMonotonicFactWriteError,
    ReservedFactKeyError,
    Snapshot,
    StaleLocationTag,
    UnknownEntityError,
    WorldState,
    WorldStateError,
    WorldStateFileError,
    check_claim,
    check_location_tags,
)

# ---------------------------------------------------------------------------
# Dataclass shape (spec §5.1, adopted verbatim) and the "t is a float, never
# a chunk id" rule.
# ---------------------------------------------------------------------------


def _field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_entity_fields_match_spec():
    assert _field_names(Entity) == {"id", "kind", "name", "created_at_t", "destroyed_at_t"}


def test_fact_fields_are_spec_plus_the_one_hop_causality_field():
    # spec §5.1 plus opened_by_event_id -- the field that makes "one hop of
    # causality" O(1) without repurposing Event.causes (documented deviation).
    assert _field_names(Fact) >= {
        "id",
        "entity_id",
        "key",
        "value",
        "valid_from_t",
        "valid_to_t",
        "irreversible",
    }
    assert "opened_by_event_id" in _field_names(Fact)


def test_event_fields_match_spec():
    assert _field_names(Event) == {"id", "at_t", "kind", "description", "causes"}


def test_no_chunk_id_concept_anywhere_in_the_core_dataclasses():
    """t is story time in seconds -- the axis invariant under re-segmentation
    of the client's own units. Chunk index is the wrong answer by design."""
    forbidden = {"chunk_id", "chunk_index", "chunk"}
    for cls in (Entity, Fact, Event, Snapshot, ChangeSet):
        assert not (_field_names(cls) & forbidden), cls


def test_core_dataclasses_are_frozen_append_only_rows():
    entity = Entity(id="e1", kind="prop", name="a wheel", created_at_t=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entity.name = "renamed"  # type: ignore[misc]

    fact = Fact(id="f1", entity_id="e1", key="state", value="ok", valid_from_t=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fact.value = "broken"  # type: ignore[misc]

    event = Event(id="ev1", at_t=0.0, kind="x", description="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.description = "z"  # type: ignore[misc]


def test_error_hierarchy_is_rooted_in_valueerror():
    # Matches this package's own convention: ConceptValidationError(ValueError),
    # BeatsValidationError(ValueError), ProseValidationError(ValueError) --
    # not RuntimeError.
    assert issubclass(WorldStateError, ValueError)
    assert issubclass(IrreversibleFactViolation, WorldStateError)
    assert issubclass(UnknownEntityError, WorldStateError)
    assert issubclass(ReservedFactKeyError, WorldStateError)
    assert issubclass(NonMonotonicFactWriteError, WorldStateError)
    assert issubclass(EntityAlreadyDestroyedError, WorldStateError)


# ---------------------------------------------------------------------------
# create_entity / record_event
# ---------------------------------------------------------------------------


def test_create_entity_registers_and_returns_entity():
    ws = WorldState()
    e = ws.create_entity(kind="prop", name="the mill wheel", created_at_t=0.0, entity_id="mill")
    assert e.id == "mill"
    assert e.kind == "prop"
    assert e.name == "the mill wheel"
    assert e.created_at_t == 0.0
    assert e.destroyed_at_t is None
    assert ws.entities["mill"] == e


def test_create_entity_auto_generates_id_when_omitted():
    ws = WorldState()
    e = ws.create_entity(kind="prop", name="a lantern", created_at_t=0.0)
    assert e.id
    assert ws.entities[e.id] == e


def test_create_entity_duplicate_id_raises():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="dup")
    with pytest.raises(WorldStateError):
        ws.create_entity(kind="prop", name="b", created_at_t=1.0, entity_id="dup")


def test_record_event_returns_event_and_appends():
    ws = WorldState()
    ev = ws.record_event(at_t=10.0, kind="destruction", description="flood", event_id="flood")
    assert ev.id == "flood"
    assert ev.at_t == 10.0
    assert ev in ws.events


def test_record_event_duplicate_id_raises():
    ws = WorldState()
    ws.record_event(at_t=0.0, kind="x", description="y", event_id="dup")
    with pytest.raises(WorldStateError):
        ws.record_event(at_t=1.0, kind="x", description="z", event_id="dup")


def test_event_causes_field_is_an_opaque_tuple_never_validated():
    ws = WorldState()
    # Arbitrary strings, including ids that reference nothing real -- the
    # engine never validates or reads causes back.
    ev = ws.record_event(
        at_t=0.0, kind="x", description="y", causes=("nonexistent-1", "nonexistent-2")
    )
    assert ev.causes == ("nonexistent-1", "nonexistent-2")


# ---------------------------------------------------------------------------
# set_fact: interval-versioning, the one choke point, "nothing overwritten"
# ---------------------------------------------------------------------------


def test_set_fact_on_unknown_entity_raises():
    ws = WorldState()
    with pytest.raises(UnknownEntityError):
        ws.set_fact(entity_id="nope", key="state", value="x", valid_from_t=0.0)


def test_set_fact_first_write_opens_an_unbounded_interval():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    f = ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    assert f.value == "intact"
    assert f.valid_from_t == 0.0
    assert f.valid_to_t is None
    assert f in ws.facts


def test_set_fact_duplicate_fact_id_raises():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="v0", valid_from_t=0.0, fact_id="dup")
    with pytest.raises(WorldStateError):
        ws.set_fact(entity_id="e1", key="mood", value="calm", valid_from_t=0.0, fact_id="dup")


def test_set_fact_versioning_closes_previous_and_opens_new_without_deleting_history():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    first = ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    second = ws.set_fact(entity_id="e1", key="state", value="broken", valid_from_t=50.0)

    # Nothing is ever overwritten: both rows are still present in the log.
    assert len(ws.facts) == 2
    ids = {f.id for f in ws.facts}
    assert {first.id, second.id} == ids

    # The first row's own historical value is untouched; only its
    # valid_to_t reflects that it was superseded.
    closed_first = next(f for f in ws.facts if f.id == first.id)
    assert closed_first.value == "intact"
    assert closed_first.valid_from_t == 0.0
    assert closed_first.valid_to_t == 50.0

    open_second = next(f for f in ws.facts if f.id == second.id)
    assert open_second.value == "broken"
    assert open_second.valid_to_t is None


def test_set_fact_history_survives_three_versions():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="v0", valid_from_t=0.0)
    ws.set_fact(entity_id="e1", key="state", value="v1", valid_from_t=10.0)
    ws.set_fact(entity_id="e1", key="state", value="v2", valid_from_t=20.0)
    assert len(ws.facts) == 3
    values_in_order = [f.value for f in sorted(ws.facts, key=lambda f: f.valid_from_t)]
    assert values_in_order == ["v0", "v1", "v2"]


def test_set_fact_independent_keys_do_not_interfere():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    ws.set_fact(entity_id="e1", key="location", value="valley", valid_from_t=0.0)
    # Two different keys opened at the same t must not collide with each
    # other -- only same-key writes are ordered against one another.
    assert len(ws.facts) == 2


def test_set_fact_non_monotonic_write_raises_and_writes_nothing():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=50.0)
    before = list(ws.facts)
    with pytest.raises(NonMonotonicFactWriteError):
        # Same valid_from_t as the currently open row -- ambiguous ordering,
        # must never silently last-write-win.
        ws.set_fact(entity_id="e1", key="state", value="broken", valid_from_t=50.0)
    assert ws.facts == before  # no partial write happened


def test_set_fact_time_moving_backward_raises():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=50.0)
    with pytest.raises(NonMonotonicFactWriteError):
        ws.set_fact(entity_id="e1", key="state", value="broken", valid_from_t=10.0)


def test_set_fact_reserved_existence_key_is_rejected():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    with pytest.raises(ReservedFactKeyError):
        ws.set_fact(
            entity_id="e1", key=worldstate.EXISTENCE_KEY, value="destroyed", valid_from_t=0.0
        )


def test_set_fact_unknown_opened_by_event_id_raises():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    with pytest.raises(WorldStateError):
        ws.set_fact(
            entity_id="e1",
            key="state",
            value="intact",
            valid_from_t=0.0,
            opened_by_event_id="no-such-event",
        )


# ---------------------------------------------------------------------------
# Irreversibility: enforced at the one choke point.
# ---------------------------------------------------------------------------


def test_irreversible_fact_contradicted_by_later_write_raises_with_one_hop():
    ws = WorldState()
    ws.create_entity(kind="prop", name="the mill wheel", created_at_t=0.0, entity_id="mill")
    flood = ws.record_event(at_t=386.0, kind="destruction", description="the flood tears it apart")
    ws.set_fact(
        entity_id="mill",
        key="state",
        value="destroyed",
        valid_from_t=386.0,
        irreversible=True,
        opened_by_event_id=flood.id,
    )

    with pytest.raises(IrreversibleFactViolation) as excinfo:
        ws.set_fact(entity_id="mill", key="state", value="intact", valid_from_t=408.0)

    err = excinfo.value
    # One hop, never a reasoning trace: the contradicted fact, its
    # valid_from_t, and the id of the event that opened it.
    assert err.contradicted_fact.value == "destroyed"
    assert err.contradicted_fact.valid_from_t == 386.0
    assert err.contradicted_fact.entity_id == "mill"
    assert err.opened_by_event_id == flood.id
    assert err.attempted_value == "intact"


def test_irreversible_fact_violation_does_not_mutate_the_log():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(
        entity_id="e1", key="state", value="destroyed", valid_from_t=10.0, irreversible=True
    )
    before = list(ws.facts)
    with pytest.raises(IrreversibleFactViolation):
        ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=20.0)
    assert ws.facts == before


def test_irreversible_fact_same_value_restated_is_not_a_contradiction():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(
        entity_id="e1", key="state", value="destroyed", valid_from_t=10.0, irreversible=True
    )
    # Restating the SAME value later is not a contradiction -- only a
    # differing value trips the choke point.
    restated = ws.set_fact(entity_id="e1", key="state", value="destroyed", valid_from_t=20.0)
    assert restated.value == "destroyed"
    assert len(ws.facts) == 2


def test_reversible_fact_may_be_contradicted_freely():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="mood", value="calm", valid_from_t=0.0, irreversible=False)
    changed = ws.set_fact(entity_id="e1", key="mood", value="angry", valid_from_t=10.0)
    assert changed.value == "angry"


# ---------------------------------------------------------------------------
# destroy_entity: sugar over the SAME choke point as set_fact, not a second
# parallel irreversibility mechanism.
# ---------------------------------------------------------------------------


def test_destroy_entity_sets_destroyed_at_t_and_updates_registry():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    destroyed = ws.destroy_entity("e1", at_t=50.0)
    assert destroyed.destroyed_at_t == 50.0
    assert ws.entities["e1"].destroyed_at_t == 50.0


def test_destroy_entity_unknown_entity_raises():
    ws = WorldState()
    with pytest.raises(UnknownEntityError):
        ws.destroy_entity("nope", at_t=0.0)


def test_destroy_entity_twice_raises():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.destroy_entity("e1", at_t=50.0)
    with pytest.raises(EntityAlreadyDestroyedError):
        ws.destroy_entity("e1", at_t=60.0)


def test_destroy_entity_routes_through_the_same_fact_log_not_a_second_mechanism():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.destroy_entity("e1", at_t=50.0)
    # The destruction is recorded as an (irreversible) row in the very same
    # append-only fact log set_fact writes to -- not a bare mutated field
    # living outside the fact machinery.
    existence_facts = [
        f for f in ws.facts if f.entity_id == "e1" and f.key == worldstate.EXISTENCE_KEY
    ]
    assert len(existence_facts) == 1
    assert existence_facts[0].irreversible is True
    assert existence_facts[0].valid_from_t == 50.0


def test_destroy_entity_with_opened_by_event_id_is_recorded():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    flood = ws.record_event(at_t=50.0, kind="destruction", description="flood")
    ws.destroy_entity("e1", at_t=50.0, opened_by_event_id=flood.id)
    existence_fact = next(f for f in ws.facts if f.key == worldstate.EXISTENCE_KEY)
    assert existence_fact.opened_by_event_id == flood.id


# ---------------------------------------------------------------------------
# replay(t): full snapshot, half-open validity, half-open entity lifetime.
# ---------------------------------------------------------------------------


def test_replay_excludes_entity_not_yet_created():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=100.0, entity_id="e1")
    snap = ws.replay(50.0)
    assert snap.t == 50.0
    assert snap.entities == ()


def test_replay_includes_entity_at_exact_creation_instant():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=100.0, entity_id="e1")
    snap = ws.replay(100.0)
    assert [es.entity.id for es in snap.entities] == ["e1"]


def test_replay_excludes_destroyed_entity_at_and_after_destruction_half_open():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.destroy_entity("e1", at_t=50.0)
    assert [es.entity.id for es in ws.replay(49.9).entities] == ["e1"]
    # Half-open: gone exactly AT the destruction instant, same convention as
    # fact validity intervals.
    assert ws.replay(50.0).entities == ()
    assert ws.replay(50.1).entities == ()


def test_replay_returns_the_fact_version_valid_at_t():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    ws.set_fact(entity_id="e1", key="state", value="broken", valid_from_t=50.0)

    snap_before = ws.replay(49.0)
    entity_snap = next(es for es in snap_before.entities if es.entity.id == "e1")
    assert entity_snap.facts["state"].value == "intact"

    snap_after = ws.replay(50.0)
    entity_snap = next(es for es in snap_after.entities if es.entity.id == "e1")
    assert entity_snap.facts["state"].value == "broken"


def test_replay_fact_validity_is_half_open_at_the_boundary():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    ws.set_fact(entity_id="e1", key="state", value="broken", valid_from_t=50.0)
    # Exactly one microsecond before the boundary still reads the old value;
    # exactly at it reads the new one (already covered above) -- and never
    # both / neither.
    just_before = ws.replay(49.999999)
    es = next(e for e in just_before.entities if e.entity.id == "e1")
    assert es.facts["state"].value == "intact"


def test_replay_omits_a_fact_key_never_set_before_t():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=100.0)
    snap = ws.replay(50.0)
    es = next(e for e in snap.entities if e.entity.id == "e1")
    assert "state" not in es.facts


def test_replay_omits_the_reserved_existence_key_from_facts():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    snap = ws.replay(0.0)
    es = next(e for e in snap.entities if e.entity.id == "e1")
    assert worldstate.EXISTENCE_KEY not in es.facts


def test_replay_multiple_entities_are_independent():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.create_entity(kind="prop", name="b", created_at_t=0.0, entity_id="e2")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=0.0)
    ws.set_fact(entity_id="e2", key="state", value="broken", valid_from_t=0.0)
    snap = ws.replay(10.0)
    by_id = {es.entity.id: es for es in snap.entities}
    assert by_id["e1"].facts["state"].value == "intact"
    assert by_id["e2"].facts["state"].value == "broken"


def test_replay_entity_order_is_deterministic_creation_order():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.create_entity(kind="prop", name="b", created_at_t=0.0, entity_id="e2")
    ws.create_entity(kind="prop", name="c", created_at_t=0.0, entity_id="e3")
    snap = ws.replay(0.0)
    assert [es.entity.id for es in snap.entities] == ["e1", "e2", "e3"]


# ---------------------------------------------------------------------------
# changes_within(t0, t1): half-open, transitions only -- never a verdict.
# ---------------------------------------------------------------------------


def test_changes_within_is_half_open_on_events():
    ws = WorldState()
    ws.record_event(at_t=10.0, kind="x", description="at t0", event_id="a")
    ws.record_event(at_t=20.0, kind="x", description="at t1", event_id="b")
    ws.record_event(at_t=15.0, kind="x", description="inside", event_id="c")
    cs = ws.changes_within(10.0, 20.0)
    ids = {e.id for e in cs.events}
    assert ids == {"a", "c"}  # t0 included, t1 excluded


def test_changes_within_reports_entities_created():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=15.0, entity_id="e1")
    ws.create_entity(kind="prop", name="b", created_at_t=99.0, entity_id="e2")
    cs = ws.changes_within(10.0, 20.0)
    assert [e.id for e in cs.entities_created] == ["e1"]


def test_changes_within_reports_entities_destroyed():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.destroy_entity("e1", at_t=15.0)
    cs = ws.changes_within(10.0, 20.0)
    assert [e.id for e in cs.entities_destroyed] == ["e1"]


def test_changes_within_reports_facts_opened_and_excludes_existence_key():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="intact", valid_from_t=15.0)
    ws.destroy_entity("e1", at_t=16.0)
    cs = ws.changes_within(10.0, 20.0)
    assert [f.key for f in cs.facts_opened] == ["state"]
    # The destruction shows up as entities_destroyed, not as a raw
    # reserved-key fact leaking the internal mechanism into a public query.
    assert all(f.key != worldstate.EXISTENCE_KEY for f in cs.facts_opened)


def test_changes_within_never_returns_a_boolean_or_severity_verdict():
    """The engine provides the query; the client declares the policy. No
    is_clean, no severity -- transitions only."""
    ws = WorldState()
    cs = ws.changes_within(0.0, 100.0)
    field_names = _field_names(ChangeSet)
    forbidden = {"is_clean", "severity", "ok", "valid", "passed", "verdict"}
    assert not (field_names & forbidden)
    for name in forbidden:
        assert not hasattr(cs, name)


def test_changes_within_same_at_t_events_preserve_insertion_order():
    ws = WorldState()
    ev1 = ws.record_event(at_t=5.0, kind="x", description="first", event_id="first")
    ev2 = ws.record_event(at_t=5.0, kind="x", description="second", event_id="second")
    cs = ws.changes_within(0.0, 10.0)
    assert [e.id for e in cs.events] == [ev1.id, ev2.id]


def test_changes_within_serves_a_continuous_take_reader_and_a_briefing_builder_alike():
    """One primitive, two opposite verdicts drawn by the CLIENT -- the query
    itself must not have baked in either reading."""
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value="mid-change", valid_from_t=5.0)
    cs = ws.changes_within(0.0, 10.0)
    # A continuous-take client reads this as a defect (something changed
    # mid-interval); a turn-based client reads it as material to build a
    # briefing. changes_within must hand back only the transition, letting
    # either policy be layered on top by the caller.
    assert len(cs.facts_opened) == 1
    assert cs.facts_opened[0].value == "mid-change"


# ---------------------------------------------------------------------------
# The defect this exists to catch, worked end-to-end at the core-model layer
# (no check_claim yet -- that lands in a later change to this same module).
# ---------------------------------------------------------------------------


def test_end_to_end_irreversible_destruction_rejects_a_later_contradiction():
    ws = WorldState()
    mill = ws.create_entity(
        kind="prop", name="the mill wheel", created_at_t=0.0, entity_id="mill"
    )
    flood = ws.record_event(
        at_t=386.0, kind="destruction", description="the flood tears the wheel apart"
    )
    ws.set_fact(
        entity_id=mill.id,
        key="state",
        value="destroyed",
        valid_from_t=386.0,
        opened_by_event_id=flood.id,
        irreversible=True,
    )

    # 6:38 -- a barren island, consistent with destruction: fine, no write
    # attempted here at all in this minimal core-model test.
    snap_638 = ws.replay(398.0)
    es = next(e for e in snap_638.entities if e.entity.id == "mill")
    assert es.facts["state"].value == "destroyed"

    # 6:48 -- "a lush, turning water wheel": whoever is authoring this beat
    # tries to assert the wheel is intact again. The choke point refuses.
    with pytest.raises(IrreversibleFactViolation) as excinfo:
        ws.set_fact(entity_id=mill.id, key="state", value="intact", valid_from_t=408.0)
    assert excinfo.value.contradicted_fact.valid_from_t == 386.0
    assert excinfo.value.opened_by_event_id == flood.id


# ---------------------------------------------------------------------------
# contradicted_by on Fact -- caller-injected vocabulary (Approach B), never
# a module default. Threads through the same set_fact() choke point every
# other fact field does.
# ---------------------------------------------------------------------------


def test_set_fact_contradicted_by_defaults_to_empty_tuple():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    f = ws.set_fact(entity_id="e1", key="state", value="ok", valid_from_t=0.0)
    assert f.contradicted_by == ()


def test_set_fact_stores_caller_supplied_contradicted_by_tokens():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    f = ws.set_fact(
        entity_id="e1",
        key="state",
        value="destroyed",
        valid_from_t=0.0,
        contradicted_by=("lush", "turning"),
    )
    assert f.contradicted_by == ("lush", "turning")


def test_worldstate_error_hierarchy_includes_file_error():
    assert issubclass(WorldStateFileError, WorldStateError)


# ---------------------------------------------------------------------------
# check_claim: Approach B. Matches ONLY caller-injected vocabulary -- a
# per-call `nouns` mapping and each fact's own `contradicted_by` -- never
# English. All song-specific words below live in this fixture, never in
# worldstate.py; see the source-scan test at the bottom of this file.
# ---------------------------------------------------------------------------


def _mill_world() -> tuple[WorldState, str, str]:
    """The module docstring's own defect example, built entirely in a TEST
    fixture. ``irreversible=True`` guards the underlying fact (any later
    ``set_fact`` call trying to un-destroy it raises, per the tests above);
    ``contradicted_by`` is the separate, softer vocabulary that lets
    ``check_claim`` flag a shot line that merely *describes* the wheel as
    though it weren't destroyed, without ever attempting to write that
    claim into the world model itself."""
    ws = WorldState()
    mill = ws.create_entity(kind="prop", name="the mill wheel", created_at_t=0.0, entity_id="mill")
    flood = ws.record_event(
        at_t=386.0, kind="destruction", description="the flood tears the wheel apart"
    )
    ws.set_fact(
        entity_id=mill.id,
        key="state",
        value="destroyed",
        valid_from_t=386.0,
        opened_by_event_id=flood.id,
        irreversible=True,
        contradicted_by=(
            "lush",
            "turning",
            "spinning",
            "intact",
            "whole",
            "working",
            "churning",
        ),
    )
    return ws, mill.id, flood.id


def test_check_claim_fires_when_text_matches_a_noun_and_a_contradicted_by_token():
    ws, mill_id, flood_id = _mill_world()
    snap = ws.replay(408.0)  # 6:48 -- "a lush, turning water wheel"
    hits = check_claim(
        snap,
        "a lush, turning water wheel spins gently in the valley below",
        nouns={mill_id: ("water wheel", "mill wheel")},
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit.entity_id == mill_id
    assert hit.entity_name == "the mill wheel"
    assert set(hit.matched_tokens) == {"lush", "turning"}
    assert "water wheel" in hit.matched_nouns
    # One hop of causality, never a reasoning trace: the contradicted fact
    # (with its own valid_from_t) and the event id that opened it.
    assert hit.fact.value == "destroyed"
    assert hit.fact.valid_from_t == 386.0
    assert hit.opened_by_event_id == flood_id


def test_check_claim_does_not_fire_when_no_contradicted_by_token_is_present():
    """The legitimate-ruin case (task item 3): the SAME destroyed entity,
    the SAME fact, described in words nobody declared to contradict it."""
    ws, mill_id, _flood_id = _mill_world()
    snap = ws.replay(491.0)  # 8:11 -- "the mill wheel lies split in two"
    hits = check_claim(
        snap,
        "the mill wheel lies split in two",
        nouns={mill_id: ("water wheel", "mill wheel")},
    )
    assert hits == ()


def test_check_claim_a_hard_destroyed_entity_is_absent_from_the_snapshot_so_nothing_fires():
    """The structural half of the legitimate-ruin guarantee: a fully
    destroy_entity()'d entity is already gone from replay() (covered above
    in the replay() section), so describing it afterwards -- as a ruin or
    in ANY words, including ones that would otherwise be flagged -- has
    nothing left in the snapshot to match against."""
    ws = WorldState()
    statue = ws.create_entity(kind="prop", name="a statue", created_at_t=0.0, entity_id="statue")
    ws.set_fact(
        entity_id="statue",
        key="state",
        value="standing",
        valid_from_t=0.0,
        irreversible=True,
        contradicted_by=("shattered", "toppled", "ruined"),
    )
    ws.destroy_entity("statue", at_t=100.0)
    snap = ws.replay(200.0)
    hits = check_claim(
        snap,
        "the shattered, toppled ruins of the statue litter the ground",
        nouns={statue.id: ("statue",)},
    )
    assert hits == ()


def test_check_claim_ignores_entity_whose_noun_never_appears_in_the_text():
    ws, mill_id, _flood_id = _mill_world()
    snap = ws.replay(408.0)
    hits = check_claim(
        snap,
        "a lush green valley stretches to the horizon",
        nouns={mill_id: ("water wheel", "mill wheel")},
    )
    assert hits == ()


def test_check_claim_returns_empty_tuple_and_never_raises_with_no_nouns_supplied():
    ws, _mill_id, _flood_id = _mill_world()
    snap = ws.replay(408.0)
    hits = check_claim(snap, "a lush, turning water wheel", nouns={})
    assert hits == ()


def test_check_claim_matching_is_whole_word_not_substring():
    ws, mill_id, _flood_id = _mill_world()
    snap = ws.replay(408.0)
    # "millstream" contains "mill" but is not the phrase "mill wheel" --
    # the same \b<phrase>\b idiom shot_plan.py's own lints already use.
    hits = check_claim(
        snap,
        "a peaceful millstream trickles through the reeds",
        nouns={mill_id: ("mill wheel",)},
    )
    assert hits == ()


def test_check_claim_matching_is_case_insensitive():
    ws, mill_id, _flood_id = _mill_world()
    snap = ws.replay(408.0)
    hits = check_claim(
        snap,
        "The LUSH, Turning WATER WHEEL spins in the valley",
        nouns={mill_id: ("water wheel",)},
    )
    assert len(hits) == 1


def test_check_claim_scores_each_fact_independently_across_multiple_keys():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a device", created_at_t=0.0, entity_id="e1")
    ws.set_fact(
        entity_id="e1",
        key="state",
        value="broken",
        valid_from_t=0.0,
        irreversible=True,
        contradicted_by=("intact", "whole"),
    )
    ws.set_fact(
        entity_id="e1",
        key="location",
        value="valley",
        valid_from_t=0.0,
        irreversible=True,
        contradicted_by=("mountain", "summit"),
    )
    snap = ws.replay(10.0)
    hits = check_claim(
        snap,
        "the whole thing sits atop the summit",
        nouns={"e1": ("thing",)},
    )
    matched_keys = {h.fact.key for h in hits}
    assert matched_keys == {"state", "location"}


def test_check_claim_result_type_is_frozen():
    ws, mill_id, _flood_id = _mill_world()
    snap = ws.replay(408.0)
    hits = check_claim(
        snap, "a lush water wheel", nouns={mill_id: ("water wheel",)}
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hits[0].entity_id = "changed"  # type: ignore[misc]


def test_worldstate_module_source_contains_no_client_specific_vocabulary():
    """The mechanism belongs in the module; the vocabulary belongs to the
    caller. worldstate.py must never itself name a song-specific noun or
    contradiction word -- every one used across this test file lives in a
    fixture, never in the module under test."""
    source = Path(worldstate.__file__).read_text(encoding="utf-8").lower()
    forbidden_words = (
        "mill",
        "wheel",
        "island",
        "lush",
        "turning",
        "spinning",
        "churning",
        "flood",
        "statue",
        "millstream",
        "shattered",
        "toppled",
    )
    for word in forbidden_words:
        assert word not in source, f"worldstate.py must not contain {word!r}"


# ---------------------------------------------------------------------------
# Export / import (spec §6): lossless JSON round-trip. The file is what a
# consumer depends on, never a live WorldState object.
# ---------------------------------------------------------------------------


def _populated_world() -> WorldState:
    ws = WorldState()
    mill = ws.create_entity(kind="prop", name="the mill wheel", created_at_t=0.0, entity_id="mill")
    ws.create_entity(kind="character", name="Dianne", created_at_t=0.0, entity_id="dianne")
    flood = ws.record_event(
        at_t=386.0,
        kind="destruction",
        description="the flood tears the wheel apart",
        causes=("some-earlier-event",),
    )
    ws.set_fact(entity_id=mill.id, key="location", value="valley", valid_from_t=0.0)
    ws.set_fact(
        entity_id=mill.id,
        key="state",
        value="destroyed",
        valid_from_t=386.0,
        opened_by_event_id=flood.id,
        irreversible=True,
        contradicted_by=("lush", "turning"),
    )
    ws.set_fact(entity_id="dianne", key="mood", value="grieving", valid_from_t=386.0)
    ws.destroy_entity("dianne", at_t=500.0)
    return ws


def test_entity_to_dict_from_dict_round_trip():
    e = Entity(id="e1", kind="prop", name="a wheel", created_at_t=0.0, destroyed_at_t=50.0)
    assert Entity.from_dict(e.to_dict()) == e


def test_entity_to_dict_from_dict_round_trip_with_none_destroyed_at_t():
    e = Entity(id="e1", kind="prop", name="a wheel", created_at_t=0.0)
    assert Entity.from_dict(e.to_dict()) == e


def test_fact_to_dict_from_dict_round_trip():
    f = Fact(
        id="f1",
        entity_id="e1",
        key="state",
        value="destroyed",
        valid_from_t=386.0,
        valid_to_t=408.0,
        irreversible=True,
        contradicted_by=("lush", "turning"),
        opened_by_event_id="flood",
    )
    assert Fact.from_dict(f.to_dict()) == f


def test_fact_to_dict_from_dict_round_trip_with_defaults():
    f = Fact(id="f1", entity_id="e1", key="mood", value="calm", valid_from_t=0.0)
    assert Fact.from_dict(f.to_dict()) == f


def test_event_to_dict_from_dict_round_trip():
    ev = Event(id="ev1", at_t=10.0, kind="x", description="y", causes=("a", "b"))
    assert Event.from_dict(ev.to_dict()) == ev


def test_worldstate_to_dict_round_trip_is_lossless():
    ws = _populated_world()
    restored = WorldState.from_dict(ws.to_dict())

    assert restored.entities == ws.entities
    assert restored.facts == ws.facts
    assert restored.events == ws.events

    # And not merely field-equal -- the two queries agree at every t that
    # matters, including across the destroyed dianne entity and the
    # multi-version mill facts.
    for t in (0.0, 200.0, 386.0, 408.0, 499.9, 500.0, 600.0):
        assert restored.replay(t) == ws.replay(t)
    assert restored.changes_within(0.0, 1000.0) == ws.changes_within(0.0, 1000.0)


def test_worldstate_to_dict_includes_schema_version():
    ws = _populated_world()
    payload = ws.to_dict()
    assert payload["schema_version"] == WORLDSTATE_SCHEMA_VERSION


def test_worldstate_from_dict_rejects_unknown_schema_version():
    ws = _populated_world()
    payload = ws.to_dict()
    payload["schema_version"] = 999
    with pytest.raises(WorldStateFileError):
        WorldState.from_dict(payload)


def test_worldstate_save_load_round_trip_via_file(tmp_path):
    ws = _populated_world()
    path = tmp_path / ".authoring" / "worldstate.json"
    ws.save(path)
    restored = WorldState.load(path)

    assert restored.entities == ws.entities
    assert restored.facts == ws.facts
    assert restored.events == ws.events
    assert restored.replay(408.0) == ws.replay(408.0)


def test_worldstate_save_matches_dot_authoring_json_conventions(tmp_path):
    """Same convention session.json/beats.json already use: indent=2,
    sorted keys, a single trailing newline -- so this file diffs cleanly in
    a PR the way every other .authoring/ artifact does."""
    ws = _populated_world()
    path = tmp_path / "worldstate.json"
    ws.save(path)
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    # Round-trips through the exact same json module with the exact same
    # formatting a hand run of `json.dumps(ws.to_dict(), indent=2,
    # sort_keys=True) + "\n"` would produce.
    expected = json.dumps(ws.to_dict(), indent=2, sort_keys=True) + "\n"
    assert text == expected


def test_worldstate_save_creates_parent_directories(tmp_path):
    ws = _populated_world()
    path = tmp_path / "nested" / "dir" / "worldstate.json"
    ws.save(path)
    assert path.exists()


def test_worldstate_save_raises_worldstate_file_error_for_non_json_safe_value(tmp_path):
    """`Fact.value` must stay JSON-safe for the round-trip to be lossless
    (documented on `Fact.to_dict`) -- a caller that violates this gets a
    named, typed error at save() time, not a bare TypeError from `json`."""
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.set_fact(entity_id="e1", key="state", value={1, 2, 3}, valid_from_t=0.0)
    with pytest.raises(WorldStateFileError):
        ws.save(tmp_path / "worldstate.json")


def test_worldstate_load_missing_file_raises_worldstate_file_error(tmp_path):
    with pytest.raises(WorldStateFileError):
        WorldState.load(tmp_path / "nope.json")


def test_worldstate_load_malformed_json_raises_worldstate_file_error(tmp_path):
    path = tmp_path / "worldstate.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    with pytest.raises(WorldStateFileError):
        WorldState.load(path)


def test_worldstate_from_dict_preserves_entity_creation_order():
    ws = WorldState()
    ws.create_entity(kind="prop", name="a", created_at_t=0.0, entity_id="e1")
    ws.create_entity(kind="prop", name="b", created_at_t=0.0, entity_id="e2")
    ws.create_entity(kind="prop", name="c", created_at_t=0.0, entity_id="e3")
    restored = WorldState.from_dict(ws.to_dict())
    assert list(restored.entities.keys()) == ["e1", "e2", "e3"]


def test_worldstate_round_trip_preserves_irreversibility_enforcement():
    """A restored WorldState is not a read-only copy -- the choke point
    still enforces irreversibility exactly as the original did."""
    ws = _populated_world()
    restored = WorldState.from_dict(ws.to_dict())
    with pytest.raises(IrreversibleFactViolation):
        restored.set_fact(entity_id="mill", key="state", value="intact", valid_from_t=999.0)


# --- irreversibility must actually be durable (adversarial review, 2026-08-19) ---


def test_irreversible_survives_a_same_value_restatement():
    """The defect the module exists to catch, reachable through its own API.

    ``_open_fact_version`` compared ``current.irreversible`` but wrote the
    *caller's* flag onto the superseding row, and that flag defaults to False.
    So restating a value -- legal, and exactly what
    ``test_irreversible_fact_same_value_restated_is_not_a_contradiction``
    does -- silently dropped irreversibility, and the next write put the
    island back. 100% line coverage did not catch it because no test ran the
    three-call sequence.

    Once true for an ``(entity_id, key)``, irreversible is sticky.
    """
    ws = WorldState()
    ws.create_entity(entity_id="island", kind="place", name="the island", created_at_t=0.0)
    ws.set_fact(
        entity_id="island", key="state", value="destroyed",
        valid_from_t=386.0, irreversible=True,
    )
    restated = ws.set_fact(
        entity_id="island", key="state", value="destroyed", valid_from_t=400.0
    )

    assert restated.irreversible is True, "irreversibility must not be dropped by a restatement"

    with pytest.raises(IrreversibleFactViolation):
        ws.set_fact(entity_id="island", key="state", value="intact", valid_from_t=420.0)


def test_loading_refuses_two_simultaneously_open_versions_of_one_key():
    """A restored WorldState must be as trustworthy as a built one.

    ``from_dict`` did no structural validation, so a payload carrying two open
    rows for the same ``(entity_id, key)`` -- which ``set_fact`` itself would
    refuse -- loaded without complaint and ``replay`` returned one of them.
    That reproduces the destroyed-entity-returns defect through the module's
    own sanctioned persistence path.
    """
    ws = WorldState()
    ws.create_entity(entity_id="island", kind="place", name="the island", created_at_t=0.0)
    ws.set_fact(
        entity_id="island", key="state", value="destroyed",
        valid_from_t=386.0, irreversible=True,
    )
    payload = ws.to_dict()
    twin = dict(next(f for f in payload["facts"] if f["key"] == "state"))
    twin["id"] = twin["id"] + "-twin"
    twin["value"] = "intact"
    twin["irreversible"] = False
    payload["facts"].append(twin)

    with pytest.raises(WorldStateFileError, match="open"):
        WorldState.from_dict(payload)


def test_loading_refuses_a_log_that_already_contradicts_an_irreversible_fact():
    """Irreversibility is claimed to hold for the whole log. A closed
    irreversible row followed by a different value is a log where it never
    did, and loading it silently would make the guarantee a documentation
    claim rather than a property."""
    ws = WorldState()
    ws.create_entity(entity_id="island", kind="place", name="the island", created_at_t=0.0)
    ws.set_fact(
        entity_id="island", key="state", value="destroyed",
        valid_from_t=386.0, irreversible=True,
    )
    payload = ws.to_dict()
    for row in payload["facts"]:
        if row["key"] == "state":
            row["valid_to_t"] = 400.0
    payload["facts"].append(
        {
            "id": "fact-resurrection",
            "entity_id": "island",
            "key": "state",
            "value": "intact",
            "valid_from_t": 400.0,
            "valid_to_t": None,
            "irreversible": False,
            "opened_by_event_id": None,
            "contradicted_by": [],
        }
    )

    with pytest.raises(WorldStateFileError, match="irreversible"):
        WorldState.from_dict(payload)


def test_loading_refuses_an_entity_whose_destruction_disagrees_with_its_fact():
    """``destroyed_at_t`` and the ``EXISTENCE_KEY`` fact are two records of one
    truth, kept in step only by the choke point at write time. On load they
    were never cross-checked, so editing the entity row alone silently
    resurrected an entity the fact log still says is destroyed -- the same
    hole as two open versions, one table over.
    """
    ws = WorldState()
    ws.create_entity(entity_id="x", kind="place", name="x", created_at_t=0.0)
    ws.destroy_entity(entity_id="x", at_t=100.0)
    payload = ws.to_dict()
    for row in payload["entities"]:
        if row["id"] == "x":
            row["destroyed_at_t"] = None

    with pytest.raises(WorldStateFileError, match="destro"):
        WorldState.from_dict(payload)


# ---------------------------------------------------------------------------
# check_location_tags -- a place label reused across a change to the world.
# ---------------------------------------------------------------------------


def _ended_world() -> tuple[WorldState, str]:
    """A world with one event the caller classifies as ending it. Every
    song-specific word stays in this fixture, as everywhere else in this file.
    """
    ws = WorldState()
    ws.create_entity(entity_id="e", kind="place", name="a place", created_at_t=0.0)
    ending = ws.record_event(
        at_t=100.0, kind="ending", description="the blast finishes sweeping the valley"
    )
    ws.set_fact(
        entity_id="e",
        key="state",
        value="emptied",
        valid_from_t=100.0,
        opened_by_event_id=ending.id,
        irreversible=True,
    )
    return ws, ending.id


def test_check_location_tags_fires_on_a_tag_used_either_side_of_an_ending():
    ws, ending_id = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit.location == "the yard"
    assert hit.event_id == ending_id
    assert hit.event_at_t == 100.0
    assert hit.before_refs == ("a",)
    assert hit.after_refs == ("b",)


def test_check_location_tags_is_silent_when_each_side_uses_its_own_tag():
    """The discriminating property, and the whole reason this needs no
    vocabulary: correct authoring and incorrect authoring have different
    shapes. One label spanning the change fires; two labels do not."""
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard, after", from_t=150.0, to_t=160.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert hits == ()


def test_check_location_tags_is_silent_for_a_tag_used_only_before():
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=20.0, to_t=30.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert hits == ()


def test_check_location_tags_is_silent_for_a_tag_used_only_after():
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref="a"),
            LocatedSpan(location="the yard", from_t=170.0, to_t=180.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert hits == ()


def test_check_location_tags_a_span_containing_the_event_narrates_it_and_counts_as_before():
    """The span the change happens *inside* is the one describing the change.
    Its label is legitimately the old one, so it must never be reported --
    and on its own it must not make the tag look like it spans the event."""
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=90.0, to_t=110.0, ref="during"),
        ],
        event_kinds=("ending",),
    )
    assert hits == ()


def test_check_location_tags_ignores_event_kinds_the_caller_did_not_name():
    """Zero vocabulary of its own: which kinds end a world is the caller's
    classification, exactly as ``nouns`` is in :func:`check_claim`."""
    ws, _ = _ended_world()
    spans = [
        LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
        LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref="b"),
    ]
    assert check_location_tags(ws, spans, event_kinds=("destruction",)) == ()
    assert check_location_tags(ws, spans, event_kinds=()) == ()


def test_check_location_tags_reports_every_offending_tag_independently():
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref="b"),
            LocatedSpan(location="the shed", from_t=5.0, to_t=15.0, ref="c"),
            LocatedSpan(location="the shed", from_t=155.0, to_t=165.0, ref="d"),
        ],
        event_kinds=("ending",),
    )
    assert {h.location for h in hits} == {"the yard", "the shed"}
    assert len(hits) == 2


def test_check_location_tags_reports_one_finding_per_tag_and_event():
    ws, _ = _ended_world()
    second = ws.record_event(at_t=200.0, kind="ending", description="a later ending")
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=250.0, to_t=260.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert len(hits) == 2
    assert {h.event_at_t for h in hits} == {100.0, 200.0}
    assert second.id in {h.event_id for h in hits}


def test_check_location_tags_returns_empty_and_never_raises_on_empty_input():
    ws, _ = _ended_world()
    assert check_location_tags(ws, [], event_kinds=("ending",)) == ()
    assert check_location_tags(WorldState(), [], event_kinds=("ending",)) == ()


def test_check_location_tags_ignores_spans_with_no_location():
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location=None, from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location=None, from_t=150.0, to_t=160.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    assert hits == ()


def test_check_location_tags_result_types_are_frozen():
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref="a"),
            LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref="b"),
        ],
        event_kinds=("ending",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hits[0].location = "changed"  # type: ignore[misc]
    span = LocatedSpan(location="the yard", from_t=0.0, to_t=1.0, ref="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.location = "changed"  # type: ignore[misc]


def test_located_span_ref_is_opaque_and_never_interpreted():
    """``ref`` is the caller's own annotation, the same contract
    ``Event.causes`` documents -- carried through to the finding untouched so
    a reporting caller can name its own units, and never parsed here."""
    ws, _ = _ended_world()
    hits = check_location_tags(
        ws,
        [
            LocatedSpan(location="the yard", from_t=0.0, to_t=10.0, ref={"anything": 1}),
            LocatedSpan(location="the yard", from_t=150.0, to_t=160.0, ref=object),
        ],
        event_kinds=("ending",),
    )
    assert len(hits) == 1
    assert hits[0].before_refs == ({"anything": 1},)
    assert hits[0].after_refs == (object,)


def test_check_location_tags_keeps_the_story_time_axis():
    """The module's axis invariant: findings are in seconds, never in the
    caller's segmentation units."""
    assert not (_field_names(StaleLocationTag) & {"chunk_id", "chunk_index", "chunk"})
    assert not (_field_names(LocatedSpan) & {"chunk_id", "chunk_index", "chunk"})
