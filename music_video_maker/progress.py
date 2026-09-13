"""Turn what a run already knows into a JSON event stream (issue #36).

There is no live progress signal anywhere a browser could reach: today the
only evidence a run is healthy rather than wedged is INFO logging to stderr.
Issue #36 calls this "a plumbing problem, not a measurement problem" -- the
data already exists in ``run_state.json``, persisted atomically after every
chunk by ``resilience.ResilientRunner._persist``, and already trusted by
``--resume``. This module is the reader-plus-differ that sits on top of it.

No server lives here
---------------------
This module imports no ``http``, no ``socket``, opens nothing, and runs no
``asyncio`` event loop -- deliberately. Issue #36's UI is bound by a hard
constraint: anything that can start a render can write files and load custom
nodes by proxy through an unauthenticated ComfyUI, so a server for this must
bind to loopback + tailnet only, never ``0.0.0.0``, and that decision needs
the owner's answer to "where does this run" (the driving Mac, or doris next
to ComfyUI). Both are out of scope for this file on purpose: a module that
cannot listen on a socket cannot get that constraint wrong. Whoever writes
the server imports :func:`format_sse` and :func:`events_between` and owns
the bind address; nothing here decides it for them.

Design decisions (issue #36, settled before this module was written)
----------------------------------------------------------------------
* **D1 -- poll ``run_state.json``, don't add an observer callback.**
  ``ResilientRunner`` has no seam today for a caller to be notified per
  chunk, and adding one is a change to another lane's file. ``run_state.json``
  is already the thing ``--resume`` trusts, and issue #36 names it directly
  as "a resumable, pollable source of truth" -- so this module is a reader
  (:func:`read_run_state`) plus a pure differ (:func:`events_between`), not a
  hook into the render loop.
* **D2 -- snapshot first, diffs after.** A client connecting to a run already
  40 chunks in must not be handed 40 replayed "chunk finished" events as
  though they just happened. :func:`events_between` therefore treats
  ``previous is None`` as "this poller has no prior state" and returns
  exactly one ``run_snapshot`` carrying the whole :class:`RunProgress`; every
  later call against a real ``previous`` emits only what changed. This is
  what makes a resumed run report correctly to a client that only just
  opened the page, instead of narrating history that already happened.
* **D3 -- project finish from RENDERED chunks only, never CACHED ones.** A
  cached chunk was reused from a prior run via ``--resume`` and cost this
  run's GPU nothing; its stored ``render_seconds`` is whatever a *previous*
  render actually took (see ``resilience._reusable_cached_result``, which
  carries the field through unchanged via ``dataclasses.replace``). Folding
  that number into "how fast is this run going" is the averaging bug that
  looks right in the happy case and lies on exactly the run where a human
  most needs the projection: a --resume with 36 of 40 chunks already cached
  would report a near-instant mean and tell a human a 100-minute run has 4
  minutes left. :attr:`RunProgress.mean_render_seconds` and
  :attr:`RunProgress.projected_remaining_seconds` both exclude ``"cached"``
  chunks for this reason -- see the comment on
  :attr:`RunProgress.mean_render_seconds` and the test named for it in
  ``tests/test_progress.py``.
* **D4 -- within-chunk step progress is out of scope.** ComfyUI's WebSocket
  ``progress`` event (step N of 20 *inside* the chunk currently rendering)
  lives on the live connection inside ``execution.ComfyUIExecutionClient``
  and is never written to ``run_state.json`` -- there is nothing here for
  this module to read. If ``RunState`` does not know it, this module does
  not report it: a real server wanting that number needs to additionally
  listen on that WebSocket itself, which is a live per-render seam this
  offline, polling module cannot reach and should not fake.
* **D5 -- free VRAM and the resume reason are recorded per chunk, so this
  module just carries them.** Two of the three gaps this file used to list
  below are closed: ``ChunkResult.free_vram_gb_before`` (issue #23) and
  ``ChunkResult.rerender_reason`` / ``.rerender_reason_fields`` (issue #34's
  fingerprint comparison, finally written down instead of only logged) are
  now real fields on the ``run_state.json`` schema, and :class:`ChunkProgress`
  exposes both unchanged -- this module does no interpretation of either, the
  same "reader, not a measurement" stance as everything else here. The
  between-chunk stop-below-floor path additionally gets one run-level field,
  ``RunProgress.vram_stop_chunk_id`` / ``.vram_stop_free_vram_gb`` /
  ``.vram_stop_floor_gb`` (from ``contracts.VramStopEvent``), because the
  chunk that tripped the floor never gets a ``ChunkResult`` at all -- see
  that class's docstring. :func:`events_between` emits a one-time
  ``run_stopped`` event the first poll after ``vram_stop`` appears, mirroring
  how ``run_finished`` fires once when :attr:`RunProgress.finished` flips.
  Within-chunk step progress (D4) remains the one gap nothing here can close.

Reuse, don't reimplement, the schema
-------------------------------------
``resilience.load_run_state`` (and ``dump_run_state``, its serialize
counterpart) own ``RUN_STATE_SCHEMA_VERSION`` and every backward-compatibility
rule recorded beside it -- re-implementing that parsing here would be a second
copy of the schema that silently drifts from the first. Both are public
(issue #36 closed the reach this docstring used to apologise for: it named
``resilience._deserialize_run_state`` as a private function this module
called directly, with a public loader "proposed to that file's owner").
:func:`read_run_state` now calls :func:`resilience.load_run_state`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from music_video_maker import resilience
from music_video_maker.contracts import ChunkResult, ChunkStatus, RunState

logger = logging.getLogger(__name__)


class ProgressError(RuntimeError):
    """``run_state.json`` could not be turned into a :class:`RunState`.

    A UI polls this file every few seconds while a chunk renders. The
    atomic temp-file-plus-``os.replace`` write in
    ``resilience.ResilientRunner._persist`` makes a torn read unlikely but
    not impossible on every filesystem, and a stale build reading a file
    written by a newer schema (or vice versa, right after an upgrade) is a
    real scenario, not a hypothetical. All of it should look the same to a
    poller -- "not right now, try again" -- rather than three different
    exception types (``OSError``, ``json.JSONDecodeError``,
    ``resilience.ResilienceError``) leaking through to three different
    callers. The message always names the path.
    """


# --------------------------------------------------------------------------- #
# Wire-format chunk status
# --------------------------------------------------------------------------- #

_STATUS_WIRE_NAMES: Mapping[ChunkStatus, str] = {
    ChunkStatus.RENDERED: "rendered",
    ChunkStatus.CACHED: "cached",
    ChunkStatus.DEAD_LETTERED: "dead_lettered",
}
"""Every ``ChunkStatus`` member that represents a resolved outcome. Deliberately
does not include :attr:`ChunkStatus.PENDING`: no code path today persists a
``ChunkResult`` with that status into ``run_state.json`` (a pending chunk
simply has no ``ChunkResult`` yet), but the type allows it -- a hand-edited
state file, or a future build recording an in-flight attempt -- and a chunk
in that shape is not "waiting to start", it is "something recorded a result
that resolved to nothing". :func:`_wire_status` maps it to ``"failed"``."""


def _wire_status(status: ChunkStatus) -> str:
    """A ``ChunkResult`` that is neither succeeded nor dead-lettered is
    ``"failed"`` -- see :data:`_STATUS_WIRE_NAMES`. ``"pending"`` is reserved
    for a ``chunk_id`` with *no* ``ChunkResult`` at all; see
    :meth:`RunProgress.from_run_state`."""
    return _STATUS_WIRE_NAMES.get(status, "failed")


_TERMINAL_STATUSES = frozenset({"rendered", "cached", "dead_lettered"})
_COMPLETED_STATUSES = frozenset({"rendered", "cached"})


@dataclass(frozen=True)
class ChunkProgress:
    """One chunk's progress, in the wire vocabulary rather than
    :class:`~music_video_maker.contracts.ChunkStatus`."""

    chunk_id: int
    status: str
    """``"pending" | "rendered" | "cached" | "dead_lettered" | "failed"``."""
    attempts: int
    errors: tuple[str, ...]
    render_seconds: float | None
    video_file: str | None
    """A path string, not a ``Path`` -- this is a wire format, not an
    in-process value; ``None`` means the chunk has not produced a file
    (pending, failed, or dead-lettered with no output)."""
    free_vram_gb_before: float | None = None
    """Straight passthrough of ``ChunkResult.free_vram_gb_before`` (issue
    #23). ``None`` for a pending chunk, a cached chunk (never probed), or any
    chunk rendered without a ``vram_probe`` injected -- see that field's
    docstring for the full list; this module draws no distinction between
    them, the same way it never has for ``render_seconds``."""
    rerender_reason: str | None = None
    """Straight passthrough of ``ChunkResult.rerender_reason`` (issue #36).
    ``None`` for a pending chunk, a reused (``"cached"``) chunk, or a chunk
    rendered fresh with nothing to reject -- see that field's docstring for
    why the last two are deliberately not distinguished."""
    rerender_reason_fields: tuple[str, ...] = ()
    """Straight passthrough of ``ChunkResult.rerender_reason_fields``."""

    @property
    def terminal(self) -> bool:
        """No further processing is coming for this chunk in this run.
        ``True`` for ``rendered``/``cached``/``dead_lettered``; ``False`` for
        ``pending`` and ``failed`` (a failed attempt may still be retried)."""
        return self.status in _TERMINAL_STATUSES

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "status": self.status,
            "attempts": self.attempts,
            "errors": list(self.errors),
            "render_seconds": self.render_seconds,
            "video_file": self.video_file,
            "free_vram_gb_before": self.free_vram_gb_before,
            "rerender_reason": self.rerender_reason,
            "rerender_reason_fields": list(self.rerender_reason_fields),
        }


def _chunk_progress(chunk_id: int, result: ChunkResult | None) -> ChunkProgress:
    if result is None:
        return ChunkProgress(
            chunk_id=chunk_id,
            status="pending",
            attempts=0,
            errors=(),
            render_seconds=None,
            video_file=None,
        )
    return ChunkProgress(
        chunk_id=chunk_id,
        status=_wire_status(result.status),
        attempts=result.attempts,
        errors=tuple(result.errors),
        render_seconds=result.render_seconds,
        video_file=str(result.video_file) if result.video_file is not None else None,
        free_vram_gb_before=result.free_vram_gb_before,
        rerender_reason=result.rerender_reason,
        rerender_reason_fields=tuple(result.rerender_reason_fields),
    )


@dataclass(frozen=True)
class RunProgress:
    """A whole run's progress: one :class:`ChunkProgress` per chunk this run
    *intends* to produce, ordered by ``chunk_id``."""

    run_id: str
    chunks: tuple[ChunkProgress, ...]
    vram_stop_chunk_id: int | None = None
    """From ``RunState.vram_stop`` (issue #23/#36): the chunk id the run
    refused to submit because a between-chunk VRAM reading dropped below the
    floor. ``None`` for every run that never hit it. This is run-level, not a
    per-chunk field, because that chunk id never got a ``ChunkResult`` at all
    -- see ``contracts.VramStopEvent``'s docstring."""
    vram_stop_free_vram_gb: float | None = None
    """The reading that tripped the stop, in GB. ``None`` exactly when
    :attr:`vram_stop_chunk_id` is ``None``."""
    vram_stop_floor_gb: float | None = None
    """The floor it dropped below. ``None`` exactly when
    :attr:`vram_stop_chunk_id` is ``None``."""

    @property
    def total(self) -> int:
        return len(self.chunks)

    @property
    def completed(self) -> int:
        """Chunks needing no further GPU work: rendered, cached, or
        dead-lettered. Deliberately includes dead-letters -- the run has
        moved on from them -- so a progress bar reaches 100% exactly when
        :attr:`finished` is ``True``, even on a run with dead-letters."""
        return sum(1 for chunk in self.chunks if chunk.terminal)

    @property
    def rendered(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.status == "rendered")

    @property
    def cached(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.status == "cached")

    @property
    def dead_lettered(self) -> tuple[int, ...]:
        """Chunk ids, not a count -- issue #36 asks for dead-letters "called
        out with their error history", which needs the id to look the chunk
        up. Mirrors ``RunState.dead_lettered`` in ``contracts.py``."""
        return tuple(chunk.chunk_id for chunk in self.chunks if chunk.status == "dead_lettered")

    @property
    def pending(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.status == "pending")

    @property
    def mean_render_seconds(self) -> float | None:
        """Mean ``render_seconds`` over ``"rendered"`` chunks only -- see the
        module docstring's D3. Do **not** widen this to include ``"cached"``
        chunks: a cached chunk's ``render_seconds`` is a *prior* run's GPU
        time, not this run's, and folding it in understates the true
        per-chunk cost on exactly the runs (``--resume``) where the number
        matters most. ``None`` when no chunk has rendered yet in this run --
        there is nothing to project from."""
        rendered_seconds = [
            chunk.render_seconds
            for chunk in self.chunks
            if chunk.status == "rendered" and chunk.render_seconds is not None
        ]
        if not rendered_seconds:
            return None
        return sum(rendered_seconds) / len(rendered_seconds)

    @property
    def projected_remaining_seconds(self) -> float | None:
        """``mean_render_seconds`` times the count of non-terminal chunks
        (pending + failed). ``None`` when :attr:`mean_render_seconds` is
        ``None`` -- projecting from zero real samples is a guess wearing a
        number, not an estimate."""
        mean = self.mean_render_seconds
        if mean is None:
            return None
        remaining = sum(1 for chunk in self.chunks if not chunk.terminal)
        return mean * remaining

    @property
    def finished(self) -> bool:
        """Every expected chunk has reached a terminal status -- including
        dead-lettered ones. A run that dead-letters its last chunk and moves
        on is finished, even though nothing in it "succeeded" in that slot."""
        return all(chunk.terminal for chunk in self.chunks)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "total": self.total,
            "completed": self.completed,
            "rendered": self.rendered,
            "cached": self.cached,
            "dead_lettered": list(self.dead_lettered),
            "pending": self.pending,
            "mean_render_seconds": self.mean_render_seconds,
            "projected_remaining_seconds": self.projected_remaining_seconds,
            "finished": self.finished,
            "vram_stop_chunk_id": self.vram_stop_chunk_id,
            "vram_stop_free_vram_gb": self.vram_stop_free_vram_gb,
            "vram_stop_floor_gb": self.vram_stop_floor_gb,
        }

    @classmethod
    def from_run_state(
        cls, run_state: RunState, expected_chunk_ids: Sequence[int]
    ) -> RunProgress:
        """Build a :class:`RunProgress` naming exactly the chunks this run
        intends to produce (Stage 2's chunk ids), one entry per id, ordered.

        ``expected_chunk_ids`` is the authority on what this run is about --
        a ``ChunkResult`` in ``run_state`` for some other id is dropped with
        a warning rather than folded in, because silently widening the set
        would inflate the denominator every progress fraction in this module
        is computed against (e.g. a stale state file left over from a
        differently-sliced run).
        """
        expected = set(expected_chunk_ids)
        for chunk_id in sorted(run_state.results):
            if chunk_id not in expected:
                logger.warning(
                    "run_state.json has a ChunkResult for chunk_id=%s, which is not in this "
                    "run's expected_chunk_ids -- dropping it rather than letting a stale or "
                    "differently-sliced state file inflate this run's progress denominator",
                    chunk_id,
                )
        chunks = tuple(
            _chunk_progress(chunk_id, run_state.results.get(chunk_id))
            for chunk_id in sorted(expected)
        )
        vram_stop = run_state.vram_stop
        return cls(
            run_id=run_state.run_id,
            chunks=chunks,
            vram_stop_chunk_id=vram_stop.chunk_id if vram_stop is not None else None,
            vram_stop_free_vram_gb=vram_stop.free_vram_gb if vram_stop is not None else None,
            vram_stop_floor_gb=vram_stop.floor_gb if vram_stop is not None else None,
        )


@dataclass(frozen=True)
class ProgressEvent:
    """One SSE-shaped event: a name from issue #36's vocabulary plus a JSON
    payload. See :func:`format_sse` for the wire encoding."""

    event: str
    """``"run_snapshot" | "chunk_completed" | "chunk_dead_lettered" |
    "chunk_retried" | "run_stopped" | "run_finished"``."""
    data: Mapping[str, object]

    def to_json(self) -> str:
        """A single JSON line: no ``indent`` (would introduce real
        newlines), ``sort_keys=True`` for a stable, testable string. A bare
        newline inside an SSE ``data:`` field silently truncates the frame
        at the client -- ``json.dumps`` already escapes any ``\\n`` found
        inside a string value as the two characters ``\\`` and ``n`` rather
        than a literal newline, which is what keeps this always one line."""
        return json.dumps(self.data, sort_keys=True)


def events_between(previous: RunProgress | None, current: RunProgress) -> tuple[ProgressEvent, ...]:
    """The events a poller should emit moving from ``previous`` to ``current``.

    ``previous is None`` means "no prior snapshot has been sent to this
    client" (D2): the whole state goes out as one ``run_snapshot``, never a
    replay of every chunk that happened before the client connected.

    Otherwise this is a pure diff, chunk-level events ordered by
    ``chunk_id`` within each event kind so the stream is reproducible: every
    newly-completed chunk (``rendered``/``cached``), then every newly
    dead-lettered chunk (with its full error history), then every chunk
    whose ``attempts`` rose without reaching a terminal status this poll,
    then at most one ``run_stopped`` when ``current.vram_stop_chunk_id``
    appears where ``previous`` had none (issue #23/#36 -- the between-chunk
    VRAM floor stopped the run; see ``RunProgress.vram_stop_chunk_id``), and
    finally at most one ``run_finished`` when ``current.finished`` flips
    from ``False`` to ``True``. The two are mutually exclusive in practice --
    a stopped run has an unsubmitted chunk still pending, so ``finished`` is
    ``False`` -- but nothing here assumes that; both checks are independent.
    """
    if previous is None:
        return (ProgressEvent(event="run_snapshot", data=current.to_dict()),)

    if previous.run_id != current.run_id:
        # A different run, not a continuation of the one this poller was
        # tracking -- diffing their chunk histories against each other would
        # attribute one run's completions to another run's chunk ids.
        logger.error(
            "run_id changed between polls (%s -> %s) -- treating this as a different run and "
            "re-snapshotting rather than diffing across two runs' chunk histories",
            previous.run_id,
            current.run_id,
        )
        return (ProgressEvent(event="run_snapshot", data=current.to_dict()),)

    prev_by_id = {chunk.chunk_id: chunk for chunk in previous.chunks}
    events: list[ProgressEvent] = []
    ordered_current = sorted(current.chunks, key=lambda chunk: chunk.chunk_id)

    for chunk in ordered_current:
        prior = prev_by_id.get(chunk.chunk_id)
        prior_status = prior.status if prior is not None else "pending"
        if chunk.status in _COMPLETED_STATUSES and prior_status not in _COMPLETED_STATUSES:
            data: dict[str, object] = {
                **chunk.to_dict(),
                "completed": current.completed,
                "total": current.total,
                "projected_remaining_seconds": current.projected_remaining_seconds,
            }
            events.append(ProgressEvent(event="chunk_completed", data=data))

    for chunk in ordered_current:
        prior = prev_by_id.get(chunk.chunk_id)
        prior_status = prior.status if prior is not None else "pending"
        if chunk.status == "dead_lettered" and prior_status != "dead_lettered":
            events.append(ProgressEvent(event="chunk_dead_lettered", data=chunk.to_dict()))

    for chunk in ordered_current:
        prior = prev_by_id.get(chunk.chunk_id)
        if prior is None:
            continue
        if chunk.attempts > prior.attempts and not chunk.terminal:
            events.append(ProgressEvent(event="chunk_retried", data=chunk.to_dict()))

    if current.vram_stop_chunk_id is not None and previous.vram_stop_chunk_id is None:
        events.append(ProgressEvent(event="run_stopped", data=current.to_dict()))

    if current.finished and not previous.finished:
        events.append(ProgressEvent(event="run_finished", data=current.to_dict()))

    return tuple(events)


def format_sse(event: ProgressEvent) -> str:
    """Render ``event`` as one Server-Sent-Events frame.

    ``event.to_json()`` is guaranteed single-line (see its docstring), which
    is what makes this safe: an SSE ``data:`` field ends at the first bare
    newline, so a multi-line payload here would silently truncate at the
    client with no error on either side.
    """
    return f"event: {event.event}\ndata: {event.to_json()}\n\n"


def read_run_state(path: Path | str) -> RunState:
    """Read and deserialize a ``run_state.json``.

    Delegates to :func:`resilience.load_run_state` rather than
    re-implementing the schema -- that function (and
    ``RUN_STATE_SCHEMA_VERSION`` beside it) is the one place that knows
    every backward-compatibility rule for this file, and a second
    implementation here would drift from it within a release. Issue #36
    proposed exactly this public loader for exactly this call site, which
    used to reach into ``resilience._deserialize_run_state`` directly; it now
    does not.

    Every failure mode collapses to :class:`ProgressError` naming ``path``:
    a missing file (``OSError``), a half-written file caught mid atomic
    write (``json.JSONDecodeError``), or a file from a schema this build
    does not read (``resilience.ResilienceError`` -- covers both
    ``RunStateSchemaError`` and the index-space guard's
    ``ChunkIdMismatchError``). A poller hitting any of these should treat it
    as "not ready yet, try again shortly", not crash on three different
    exception shapes.
    """
    resolved = Path(path)
    try:
        return resilience.load_run_state(resolved)
    except (OSError, json.JSONDecodeError, resilience.ResilienceError) as exc:
        logger.exception("Could not read run state from %s", resolved)
        raise ProgressError(f"could not read run state from {resolved}: {exc}") from exc


__all__ = [
    "ChunkProgress",
    "ProgressError",
    "ProgressEvent",
    "RunProgress",
    "events_between",
    "format_sse",
    "read_run_state",
]
