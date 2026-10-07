"""Carry a human-reviewed shot plan onto a changed chunk timeline.

General form: a shot plan anchors every entry to a chunk's ``start``, and the
render refuses a chunk whose start drifted (``ShotPlanDriftError``) -- which is
right, because a stale anchor silently attaches one shot's direction to another
shot's audio. But anything that legitimately moves boundaries (an
``[[alignment_override]]``, ``phrase_aware_slicing``, ``boundary_overrun``, a
different whisper model) then leaves a reviewed plan unloadable, and nothing
carried it across: ``--prepare`` emits a *blank* skeleton, ``--from-plan`` reads
only ``length_seconds``, and ``mvm-author`` would regenerate the prose a human
already approved.

So, run by hand, no model anywhere::

    python -m music_video_maker.authoring.replan \\
        --old-config run_v15.toml --config run_v16.toml \\
        --plan shot_plan_v13.toml --out shot_plan_v16.toml

**How an entry finds its chunk.** Never by chunk id (ids renumber the moment
the count changes) and, for sung chunks, not by time either: by the *words*.
Each word is identified by ``(segment index, position in segment)``, which an
override does not change, so a phrase an override moved twelve seconds still
finds the direction that was written for it. A new sung chunk's source is the
old chunk its words came from, weighted by word duration; a new instrumental
chunk's source is the old instrumental chunk at the same moment -- in order,
when an instrumental passage kept its shot count, by overlap otherwise.

**What it refuses to do is the point.** A new chunk whose words came from two
old shots (exactly what phrase-aware slicing does when it reunites a phrase
the old timeline cut), an instrumental chunk straddling two old shots, a source
of the other kind (an instrumental shot's ``subject`` is refused on a sung
chunk), an old entry already carried onto another chunk, or a source chunk the
plan never directed: each is **flagged** -- written with ``shot = ""`` and the
candidate entries as commented-out TOML beside it -- never filled in. A blank
shot falls back to ``narrative_concept``, which renders as something
deliberate-looking that nobody authored, so the command exits non-zero when
anything is flagged and the file says so at the top.

Carried entries keep every field verbatim (``shot``, ``camera``, ``framing``,
``present``, ``location``, ... and the ``generated_by``/``content_sha256``
provenance, since the content is unchanged); only ``chunk_id`` and ``start``
are re-emitted, from the new timeline. The written file is then loaded with
the render's own ``load_shot_plan`` and every entry resolved against the new
chunks, so a drifted anchor fails here rather than hours into a render.

Both timelines come from :func:`~music_video_maker.authoring.chunks.load_alignment_and_skeleton`
-- the authoring layer's own Stage 1-2 -- sliced into a temporary directory
that is removed afterwards, because slicing writes a WAV per chunk and a tool
that only needs the timeline must never overwrite a render's stems.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from music_video_maker.authoring.chunks import load_alignment_and_skeleton
from music_video_maker.config import load_config
from music_video_maker.contracts import AlignmentResult, AudioChunk
from music_video_maker.logging_setup import configure_logging
from music_video_maker.shot_plan import (
    START_TOLERANCE_SECONDS,
    ShotPlanError,
    load_shot_plan,
    resolve_shot,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 2
EXIT_FLAGGED = 3
"""The file was written, but some chunks have no clear source and carry a
blank shot -- a human must resolve them before the plan is rendered."""

CLEAR_SHARE = 2.0 / 3.0
"""How much of a new chunk one old entry must account for to be its source:
two thirds of its sung words (by duration) or of its instrumental span. A
chunk split evenly between two old shots has no clear source, and picking
the larger half silently discards the other half's direction."""

_MIN_WORD_WEIGHT = 0.01
"""Weight for a word the aligner gave zero duration ("So" at 230.15-230.15 on
"Deathless"), so it still counts as being somewhere."""

_ANCHOR_KEYS = ("chunk_id", "start")


@dataclass(frozen=True)
class Carried:
    """One new chunk that carries an old entry."""

    new_chunk_id: int
    source_chunk_id: int
    share: float
    fields: dict = field(default_factory=dict)
    """The old entry's every key but its anchors, verbatim."""
    retiled: bool = False
    """Carried by position across an instrumental passage whose shot count
    changed -- in order, but worth a look."""


@dataclass(frozen=True)
class Flag:
    """One new chunk with no clear source, and why."""

    new_chunk_id: int
    reason: str
    candidates: tuple[tuple[int, float], ...]
    """``(old chunk id, share)``, largest first -- what a human chooses from."""


@dataclass(frozen=True)
class ReplanResult:
    mapped: tuple[Carried, ...]
    flagged: tuple[Flag, ...]
    dropped: tuple[int, ...]
    """Old entries carried onto no chunk at all. Reported: their direction
    is in the old file and nowhere in the new one."""
    plan: Mapping[int, dict]


def _chunk_index(chunks: Sequence[AudioChunk], t: float) -> int | None:
    starts = [c.start for c in chunks]
    index = bisect.bisect_right(starts, t) - 1
    if index < 0 or t >= chunks[index].end:
        return None
    return index


def _word_sources(
    alignment: AlignmentResult, chunks: Sequence[AudioChunk]
) -> dict[tuple[int, int], int]:
    """``(segment index, word position) -> chunk id`` by word midpoint -- the
    same rule slicing uses to decide which chunk sings a word."""
    owner: dict[tuple[int, int], int] = {}
    for segment in alignment.segments:
        for position, word in enumerate(segment.words):
            index = _chunk_index(chunks, (word.start + word.end) / 2.0)
            if index is not None:
                owner[(segment.index, position)] = chunks[index].chunk_id
    return owner


def _overlap(a: AudioChunk, b: AudioChunk) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _candidates_by_words(
    new_chunk: AudioChunk,
    new_alignment: AlignmentResult,
    old_owner: Mapping[tuple[int, int], int],
) -> tuple[tuple[int, float], ...]:
    weights: dict[int, float] = {}
    for segment in new_alignment.segments:
        for position, word in enumerate(segment.words):
            if not new_chunk.start <= (word.start + word.end) / 2.0 < new_chunk.end:
                continue
            source = old_owner.get((segment.index, position))
            if source is None:
                continue
            weight = max(word.end - word.start, _MIN_WORD_WEIGHT)
            weights[source] = weights.get(source, 0.0) + weight
    total = sum(weights.values())
    if not total:
        return ()
    return tuple(sorted(((cid, w / total) for cid, w in weights.items()), key=lambda x: -x[1]))


def _candidates_by_time(
    new_chunk: AudioChunk, old_chunks: Sequence[AudioChunk]
) -> tuple[tuple[int, float], ...]:
    span = new_chunk.end - new_chunk.start
    shares = [
        (old.chunk_id, _overlap(new_chunk, old) / span)
        for old in old_chunks
        if _overlap(new_chunk, old) > 0.0
    ]
    return tuple(sorted(shares, key=lambda x: -x[1]))


def _instrumental_runs(chunks: Sequence[AudioChunk]) -> list[list[AudioChunk]]:
    runs: list[list[AudioChunk]] = []
    for chunk in chunks:
        if chunk.is_instrumental and runs and runs[-1][-1].is_instrumental and (
            runs[-1][-1].chunk_id == chunk.chunk_id - 1
        ):
            runs[-1].append(chunk)
        elif chunk.is_instrumental:
            runs.append([chunk])
    return runs


def _ordered_pairs(
    new_run: Sequence[AudioChunk], olds: Sequence[AudioChunk]
) -> list[tuple[AudioChunk, AudioChunk]]:
    """The order-preserving one-to-one pairing of ``new_run`` with ``olds``
    that maximises total overlap, never pairing chunks that do not overlap."""
    n, m = len(new_run), len(olds)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            overlap = _overlap(new_run[i - 1], olds[j - 1])
            paired = best[i - 1][j - 1] + overlap if overlap > 0.0 else -1.0
            best[i][j] = max(best[i - 1][j], best[i][j - 1], paired)
    pairs: list[tuple[AudioChunk, AudioChunk]] = []
    i, j = n, m
    while i and j:
        overlap = _overlap(new_run[i - 1], olds[j - 1])
        if overlap > 0.0 and best[i][j] == best[i - 1][j - 1] + overlap:
            pairs.append((new_run[i - 1], olds[j - 1]))
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def map_plan(
    old_chunks: Sequence[AudioChunk],
    old_alignment: AlignmentResult,
    new_chunks: Sequence[AudioChunk],
    new_alignment: AlignmentResult,
    plan: Mapping[int, dict],
) -> ReplanResult:
    """Decide, for every new chunk, which old plan entry it carries -- or
    flag it. Pure: no I/O, no slicing. ``plan`` is the old file's raw
    ``[[shot]]`` tables keyed by ``chunk_id``."""
    old_by_id = {c.chunk_id: c for c in old_chunks}
    old_owner = _word_sources(old_alignment, old_chunks)

    # An instrumental passage maps in order: its shots are a sequence (the
    # beats of a solo, an intro's establishing shots), and re-tiling it moves
    # where each shot starts without changing which comes first. Which old
    # shots belong to a new passage is decided by midpoint, so a sliver of
    # overlap with a neighbouring passage does not count; the pairing is the
    # order-preserving one that overlaps most. A passage that now has MORE
    # shots than before leaves the extras to the per-chunk rule below, where
    # they are flagged -- there is no old shot left for them to be.
    in_order: dict[int, tuple[int, float]] = {}
    retiled: set[int] = set()
    for run in _instrumental_runs(new_chunks):
        lo, hi = run[0].start, run[-1].end
        olds = [o for o in old_chunks if lo <= (o.start + o.end) / 2.0 < hi]
        if not olds or not all(o.is_instrumental for o in olds):
            continue
        for n, o in _ordered_pairs(run, olds):
            in_order[n.chunk_id] = (o.chunk_id, _overlap(n, o) / (n.end - n.start))
            if len(olds) != len(run):
                retiled.add(n.chunk_id)

    chosen: dict[int, tuple[int, float]] = {}
    flags: dict[int, Flag] = {}
    for new_chunk in new_chunks:
        cid = new_chunk.chunk_id
        if cid in in_order:
            chosen[cid] = in_order[cid]
            continue
        if new_chunk.is_instrumental:
            candidates = _candidates_by_time(new_chunk, old_chunks)
            basis = "of its span"
        else:
            candidates = _candidates_by_words(new_chunk, new_alignment, old_owner)
            basis = "of its sung words"
        if not candidates:
            flags[cid] = Flag(cid, "no old chunk overlaps it", ())
            continue
        best_id, best_share = candidates[0]
        source = old_by_id[best_id]
        if best_share < CLEAR_SHARE - 1e-9:
            flags[cid] = Flag(
                cid,
                f"no single old shot holds two thirds {basis} (best: old chunk {best_id} "
                f"at {best_share:.0%})",
                candidates,
            )
        elif source.is_instrumental != new_chunk.is_instrumental:
            kind = "instrumental" if source.is_instrumental else "sung"
            flags[cid] = Flag(
                cid,
                f"its source, old chunk {best_id}, was a {kind} shot and this chunk is "
                f"{'instrumental' if new_chunk.is_instrumental else 'sung'} -- fields like "
                "subject/focus do not carry across that line",
                candidates,
            )
        elif best_id not in plan:
            flags[cid] = Flag(
                cid, f"its source, old chunk {best_id}, has no entry in the plan", candidates
            )
        else:
            chosen[cid] = (best_id, best_share)

    # One old entry, one new chunk: the strongest claim keeps it.
    claims: dict[int, list[tuple[float, int]]] = {}
    for cid, (source, share) in chosen.items():
        claims.setdefault(source, []).append((share, cid))
    mapped: list[Carried] = []
    for source, claimants in claims.items():
        claimants.sort(key=lambda x: (-x[0], x[1]))
        share, winner = claimants[0]
        mapped.append(
            Carried(
                new_chunk_id=winner,
                source_chunk_id=source,
                share=share,
                fields={k: v for k, v in plan[source].items() if k not in _ANCHOR_KEYS},
                retiled=winner in retiled,
            )
        )
        for loser_share, loser in claimants[1:]:
            flags[loser] = Flag(
                loser,
                f"its source, old chunk {source}, is already carried onto new chunk {winner}",
                ((source, loser_share),),
            )

    mapped.sort(key=lambda m: m.new_chunk_id)
    used = {m.source_chunk_id for m in mapped}
    dropped = tuple(sorted(cid for cid in plan if cid not in used))
    return ReplanResult(
        mapped=tuple(mapped),
        flagged=tuple(flags[cid] for cid in sorted(flags)),
        dropped=dropped,
        plan=plan,
    )


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} value {value!r} into a shot plan")


def _entry_lines(fields: Mapping[str, object]) -> list[str]:
    return [f"{key} = {_toml_value(value)}" for key, value in fields.items()]


def render_plan_toml(
    result: ReplanResult,
    new_chunks: Sequence[AudioChunk],
    *,
    provenance: Mapping[str, object] | None,
    header: str,
) -> str:
    """The re-anchored plan as TOML text: every new chunk gets an entry, a
    carried one with its source's fields verbatim and a flagged one with a
    blank shot and its candidates commented out beside it."""
    by_new = {m.new_chunk_id: m for m in result.mapped}
    flags = {f.new_chunk_id: f for f in result.flagged}
    blocks = [header.rstrip("\n")] if header.strip() else []
    if provenance:
        blocks.append("\n".join(["[provenance]", *_entry_lines(provenance)]))
    for chunk in new_chunks:
        span = f"{chunk.start:.3f} - {chunk.end:.3f}  ({chunk.end - chunk.start:.3f}s)"
        lines = ["[[shot]]", f"chunk_id = {chunk.chunk_id}", f"start = {chunk.start!r}   # {span}"]
        lyric = " ".join(chunk.text.split()).replace('"', "'")
        lines.append(f'# lyric: "{lyric}"' if lyric else "# INSTRUMENTAL -- no lyric to sing")
        mapping = by_new.get(chunk.chunk_id)
        if mapping is not None:
            lines.insert(
                1,
                f"# carried from old chunk_id={mapping.source_chunk_id} "
                f"(start {result.plan[mapping.source_chunk_id].get('start', 0.0):.3f}s, "
                f"{mapping.share:.0%} of this chunk"
                + (
                    "; in order across an instrumental passage re-tiled to a different "
                    "shot count -- check it still reads)"
                    if mapping.retiled
                    else ")"
                ),
            )
            lines.extend(_entry_lines(mapping.fields))
        else:
            flag = flags[chunk.chunk_id]
            lines.insert(1, f"# REANCHOR FLAG chunk_id={chunk.chunk_id}: {flag.reason}.")
            lines.append(
                "# Pick one candidate below (uncomment its lines and delete the blank "
                "shot line), write a new shot, or leave it blank on purpose -- a blank shot "
                "renders narrative_concept."
            )
            for source, share in flag.candidates:
                entry = result.plan.get(source)
                if entry is None:
                    lines.append(f"#   candidate old chunk_id={source} ({share:.0%}): no entry")
                    continue
                lines.append(f"#   candidate old chunk_id={source} ({share:.0%}):")
                fields = {k: v for k, v in entry.items() if k not in _ANCHOR_KEYS}
                lines.extend(f"#   {line}" for line in _entry_lines(fields))
            lines.append('shot = ""')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _timeline(config, scratch: Path) -> tuple[AlignmentResult, tuple[AudioChunk, ...]]:
    """Stage 1-2 for ``config``, stems written under ``scratch``."""
    return load_alignment_and_skeleton(config, chunks_dir=scratch)


def _check_authored_against(
    plan: Mapping[int, dict], chunks: Sequence[AudioChunk], plan_path: Path
) -> None:
    """The old plan must describe the old timeline -- or the mapping would
    be measured against chunks the plan was never written for."""
    by_id = {c.chunk_id: c for c in chunks}
    for cid, entry in sorted(plan.items()):
        chunk = by_id.get(cid)
        start = float(entry.get("start", -1.0))
        if chunk is None or abs(chunk.start - start) > START_TOLERANCE_SECONDS:
            where = (
                "has no such chunk"
                if chunk is None
                else f"starts chunk {cid} at {chunk.start:.3f}s"
            )
            raise ShotPlanError(
                f"{plan_path}: chunk_id={cid} was authored against start={start:.3f}s but the "
                f"old config's timeline {where} -- --old-config is not the timeline this plan "
                "was written for"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m music_video_maker.authoring.replan",
        description="Carry a reviewed shot plan onto a changed chunk timeline, flagging "
        "every chunk without one clear source.",
    )
    parser.add_argument(
        "--old-config", required=True, help="Run config the plan was authored against."
    )
    parser.add_argument(
        "--config", required=True, help="Run config whose timeline to carry it onto."
    )
    parser.add_argument("--plan", required=True, help="The reviewed shot plan (never modified).")
    parser.add_argument(
        "--out", required=True, help="Where to write the carried plan (must not exist)."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    plan_path, out = Path(args.plan), Path(args.out)
    if out.exists() or out.resolve() == plan_path.resolve():
        logger.error("Refusing to overwrite %s -- write the carried plan to a new file.", out)
        return EXIT_ERROR
    try:
        payload = tomllib.loads(plan_path.read_text(encoding="utf-8"))
        load_shot_plan(plan_path)  # the render's own validation, before any work
        plan = {int(e["chunk_id"]): dict(e) for e in payload.get("shot", [])}
        if any("length_seconds" in e for e in plan.values()):
            raise ShotPlanError(
                f"{plan_path} sets length_seconds; carrying editorial lengths across a "
                "re-cut timeline is not supported (their anchors are the old timeline's)"
            )
        with tempfile.TemporaryDirectory(prefix="replan-") as scratch:
            old_alignment, old_chunks = _timeline(
                load_config(Path(args.old_config)), Path(scratch) / "old"
            )
            new_alignment, new_chunks = _timeline(
                load_config(Path(args.config)), Path(scratch) / "new"
            )
        _check_authored_against(plan, old_chunks, plan_path)
    except (ShotPlanError, OSError, tomllib.TOMLDecodeError, KeyError, ValueError) as exc:
        logger.error("Cannot carry the plan: %s", exc)
        return EXIT_ERROR

    result = map_plan(old_chunks, old_alignment, new_chunks, new_alignment, plan)
    header = (
        f"# Carried by `python -m music_video_maker.authoring.replan` from {plan_path.name}\n"
        f"# (authored against {args.old_config}) onto the timeline of {args.config}.\n"
        f"# {len(result.mapped)} of {len(new_chunks)} chunk(s) carry an entry verbatim; "
        f"{len(result.flagged)} are FLAGGED (blank shot, candidates beside them) and must be\n"
        "# resolved by a human before this plan is rendered. Search: REANCHOR FLAG.\n"
    )
    if result.dropped:
        header += f"# Old entries carried nowhere: {list(result.dropped)}.\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_plan_toml(result, new_chunks, provenance=payload.get("provenance"), header=header),
        encoding="utf-8",
    )
    written = load_shot_plan(out)
    for chunk in new_chunks:
        resolve_shot(written, chunk)  # ShotPlanDriftError here, not hours into a render

    retiled_count = sum(1 for m in result.mapped if m.retiled)
    sys.stdout.write(
        f"{out}: {len(new_chunks)} chunk(s); {len(result.mapped)} carried "
        f"({retiled_count} by position across a re-tiled instrumental passage), "
        f"{len(result.flagged)} flagged, {len(result.dropped)} old entr(ies) carried nowhere.\n"
    )
    for flag in result.flagged:
        sys.stdout.write(f"  FLAG chunk {flag.new_chunk_id}: {flag.reason}\n")
    return EXIT_FLAGGED if result.flagged else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
