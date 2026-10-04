"""Stage 1a: who is singing, read off the isolated vocal stem (issue #101).

An **alternative front-end** to a tested path, never a parallel mechanism.
``AlignedSegment.characters`` is the one place "who is singing this" lives
downstream of Stage 1 -- manual ``[Name: Role]`` tags populate it today, and
everything after that (``AudioChunk.characters``, ``prompting``'s active-member
resolution, the staged reference photo, #92's dominant-voice attribution) is
unchanged by construction because this module writes **the same field** and
nothing else. If a change here needed a special case anywhere downstream, the
design would be wrong; see ``docs/design-multi-vocalist.md``.

Three things it is not::

    it is not a speaker *recogniser*   -- diarization yields clusters
                                         ("SPEAKER_00"), never names;
    it is not a transcriber            -- the lyrics stay immutable truth and
                                         the timeline stays the one Stage 1
                                         computed against the master;
    it is not an authority             -- an authored tag outranks it always.

**Opt in with ``diarize = true``, and it needs ``vocal_stem``.** Diarizing the
full mix is what made this not worth attempting (issue #101); the input is the
separated vocal stem issue #25 already produces off the render path, so this
feature costs the operator no new artefact. A run that does not ask for
diarization never imports anything heavy, never reads an environment variable,
and is byte-identical to today's.

The disagreement rule, in one line
----------------------------------

**The tag wins, and the disagreement is reported.** Detection writes
``characters`` only where ``AlignedSegment.characters_authored`` is ``False`` --
that is, where nobody tagged that part of the song and the field is carrying
``default_lead_vocalist`` by fallback. Where a tag *is* in force and the
dominant detected speaker maps to a different cast member, the segment is left
exactly as authored and the clash is recorded in
:attr:`DiarizationReport.disagreements` and logged at WARNING with the segment
index, its span, both names and the detected share. Three reasons, in
increasing order of how much they decide:

1. It is this project's existing hierarchy one layer up. Forced alignment fits
   timestamps to human-supplied text and never renegotiates the text; a
   ``[Name: Role]`` tag is the same kind of human-supplied fact about the same
   audio.
2. The two mistakes cost differently. Tag right / detection wrong, resolved
   the detector's way, puts the wrong face on screen and **reports nothing** --
   the silent-overwrite defect. Tag wrong / detection right, resolved the
   tag's way, puts the *authored* face on screen and names the clash in the
   report, so the cheap edit (fix the tag) is the one the operator is handed.
3. Detection has no name of its own. The name comes from
   ``diarization_speakers``, which is also authored, so "detection overrides
   the tag" really means "one authored mapping overrides another authored
   tag" -- and the tag is the more specific, more local statement.

A tagged segment the detection agrees with is counted, not logged: agreement
is the expected case and a line per segment would bury the clashes.

Confidence, coverage and harmonies
----------------------------------

``pyannote``'s pretrained pipelines return an ``Annotation`` -- speaker-labelled
time ranges with **no per-turn probability** -- so this module does not invent
one. What it measures instead are two quantities it can compute from the turns
themselves, both per aligned segment:

``coverage``
    how much of the segment any speaker turn covers at all. Below
    :data:`DEFAULT_MIN_COVERAGE` the diarizer effectively heard nothing there
    and the segment is left alone.
``share``
    the winning speaker's fraction of the covered time. Below
    :data:`DEFAULT_MIN_SHARE` the segment is *contested* and left alone.

The winner is chosen by **overlapping duration**, which is issue #40's
dominant-voice rule as #92 re-measured it: voiced seconds inside the span, not
word count and not whole-segment length. Deliberately the same measure
``slicing._dominant_character_member`` uses, so the two cannot disagree about
who a passage belongs to.

Where this and ``slicing`` part company is what happens when nobody dominates.
Slicing *must* pick someone -- it has a chunk to render and one reference photo
to stage. Diarization can decline, because declining leaves a known, named
state behind (the tag, or ``default_lead_vocalist``) rather than a coin flip.
So a 50/50 harmony is reported as ``contested`` and falls back loudly. Emitting
*both* names was considered and rejected for now: ``characters`` is plural and
"both audible" is exactly what a harmony is, but a thrashing diarizer produces
spurious two-character segments, and plural semantics invented from a
classifier's noise is a worse failure than a named fallback. Revisit it with a
real overlapping-vocal passage measured, which is the design document's third
unmet precondition.

**The two thresholds are shapes, not measurements.** No multi-vocalist song has
been diarized here, so nothing has scored them; :meth:`DiarizationReport.log_summary`
says so at WARNING on every run that uses this module, and
``docs/vocalist-diarization.md`` records what would have to be measured to
replace them. A threshold picked without that measurement is a guess wearing a
number, and this one announces that it is.

Degrading rather than crashing
------------------------------

The weights are gated: using them at all requires accepting the terms on
Hugging Face with the account whose token is in the environment. Four distinct
things can therefore go wrong, and each has its own exception type and its own
message naming exactly what to do about it:
:class:`DiarizerUnavailableError` (the package is not installed),
:class:`DiarizerTokenError` (no token in the environment),
:class:`DiarizerAccessError` (a token that has not accepted the gated pages),
:class:`DiarizerWeightsError` (anything else about loading the weights --
offline, no cache, a renamed pipeline).

:func:`assign_characters` **catches all four and degrades**: it logs the
message at ERROR, records it in :attr:`DiarizationReport.failure`, and returns
the alignment untouched, so the run proceeds exactly as it would without
``diarize = true`` -- manual tags and the default lead, a known failure mode a
viewer can be warned about. A render is hours of GPU custody and killing one
over an unset environment variable is the expensive mistake; the place to find
this out is ``--prepare``, which runs Stage 1-2 with no GPU and surfaces the
same ERROR in about a minute.

Attribution
-----------

``pyannote.audio`` is MIT; its pretrained pipelines are **CC-BY-4.0**, which
requires attribution from anyone who uses them. :data:`ATTRIBUTION` is logged
once per run that loads them -- so the credit travels with the use, not only
with the repository -- and is carried in the README's own License section and
in ``docs/vocalist-diarization.md``. Nothing here downloads, vendors or
redistributes any weights: the operator accepts the terms and fetches them with
their own account, exactly as with the SFace weights ``faces.py`` points at.

Offline testability
-------------------

Everything in this module is injectable. :type:`Diarizer` is the seam --
``Callable[[Path], Sequence[SpeakerSpan]]``, the same shape as ``align``'s
``model``, ``resilience``'s ``vram_probe`` and ``alignment_quality``'s
``ffmpeg_runner`` -- so the test suite passes a fake and never installs
``pyannote``, downloads a model, opens a socket or touches a GPU.
:func:`build_pyannote_diarizer` additionally takes its importer and its
environment, so all four refusal messages are exercised offline on a machine
where the package does not exist.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from music_video_maker.contracts import AlignedSegment, AlignmentResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Provenance and attribution (CC-BY-4.0 -- see this module's docstring)
# --------------------------------------------------------------------------- #

PYANNOTE_PIPELINE = "pyannote/speaker-diarization-3.1"
"""The pretrained pipeline this module is written against.

Chosen over 2.x because 3.1 runs its segmentation in pure PyTorch (no
``onnxruntime`` pin) and because it is the version whose gated pages the
messages below name. A different pipeline name is a parameter, not an edit."""

PYANNOTE_GATED_PAGES = (
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
)
"""Every page whose terms must be accepted, by the account owning the token.

Two, not one: ``speaker-diarization-3.1`` is a *pipeline* that also loads a
segmentation model, gated separately, and accepting only the first produces a
403 on the second naming a repository the operator never asked for.

Checked against the live Hub on 2026-10-04 with a real token rather than
inferred from the model cards: both of these answer 403 on their files until
the account accepts, while the third model the pipeline loads --
``wespeaker-voxceleb-resnet34-LM``, the embedding model -- reports
``gated: False`` and serves its files, so naming it here would send an
operator to click a button that does not exist. If a future pyannote release
gates it, this list is where that goes.

``pyannote/speaker-diarization-community-1`` is pyannote's newer recommended
pipeline and is gated the same way; it is not the default here only because
nothing in this project has run it. Switching is this constant plus
:data:`PYANNOTE_PIPELINE`."""

PYANNOTE_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")
"""Environment variables consulted for the Hugging Face token, in order.

Read from the environment and never from the run config: a token is a
credential, and CLAUDE.md's rule is that credentials live in a gitignored
``.env`` and never in anything committed -- a ``run.toml`` is committed run
data."""

PYANNOTE_CODE_LICENCE = "MIT"
PYANNOTE_MODEL_LICENCE = "CC-BY-4.0"

ATTRIBUTION = (
    "Speaker diarization by pyannote.audio (code: MIT) using the pretrained pipeline "
    "pyannote/speaker-diarization-3.1 (models: CC-BY-4.0), by Herve Bredin and "
    "contributors -- https://github.com/pyannote/pyannote-audio. Cite: H. Bredin, "
    "'pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe', "
    "Interspeech 2023. No weights are redistributed by this project; they are fetched "
    "by the operator under the terms they accepted."
)
"""The CC-BY-4.0 attribution, logged once per run that loads the weights.

A licence recorded only in a repository file is attribution the *reader of the
repository* sees. CC-BY asks for credit from whoever uses the work, so the
credit is emitted where the use happens -- in the run log, beside the report
whose numbers came out of those models."""


# --------------------------------------------------------------------------- #
# Thresholds (unmeasured -- see this module's docstring)
# --------------------------------------------------------------------------- #

DEFAULT_MIN_COVERAGE = 0.5
"""Fraction of an aligned segment's span that speaker turns must cover before
the segment is assignable at all.

Half is a shape: a segment the diarizer covered less than halfway is one it
mostly did not hear a voice in, and a sung lyric line it mostly did not hear a
voice in is a segment whose attribution should not move. Not scored against any
song -- see this module's docstring."""

DEFAULT_MIN_SHARE = 0.6
"""Fraction of the *covered* time the winning speaker must hold.

Above it, one voice dominates and the dominant-voice rule applies. Below it the
segment is contested -- a harmony, or a thrashing diarizer -- and is left as
authored or defaulted. Not scored against any song."""

_EPS = 1e-9
"""Float slack for duration comparisons, matching ``slicing``'s own."""


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpeakerSpan:
    """One speaker turn: a cluster label over a half-open time range.

    ``speaker`` is a *cluster* label, not a name -- ``"SPEAKER_00"`` out of
    pyannote. Mapping it to a cast member is authored data
    (``diarization_speakers``), never inference; see
    :func:`summarise_speakers`, whose table exists so an operator can write
    that mapping after one listen.

    ``confidence`` is ``None`` for every real pyannote turn, because its
    ``Annotation`` carries no per-turn probability. The field exists so a
    diarizer that *does* expose one can hand it over and
    :func:`assign_characters` can gate on it, rather than this project
    inventing a number to put in its place.
    """

    start: float
    end: float
    speaker: str
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


Diarizer = Callable[[Path], Sequence[SpeakerSpan]]
"""The injected seam: audio file in, speaker turns out.

A plain callable on purpose, exactly like ``align``'s ``model`` and
``resilience``'s ``vram_probe``. Tests pass a list-returning lambda; a real run
passes :func:`build_pyannote_diarizer`'s return value. Nothing in this module
knows that pyannote exists except that function."""


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


class DiarizationError(RuntimeError):
    """Base for every reason diarization could not run.

    Always carries a message naming the next action, because the operator who
    sees it is mid-run and the alternative is a stack trace from inside a
    dependency they did not know they had.
    """


class DiarizerUnavailableError(DiarizationError):
    """``pyannote.audio`` is not installed."""


class DiarizerTokenError(DiarizationError):
    """No Hugging Face token in the environment, or the one there was rejected."""


class DiarizerAccessError(DiarizationError):
    """The token is valid but has not accepted the gated models' terms.

    The distinctive case, and the reason this type is separate from
    :class:`DiarizerTokenError`: the remedy is not a new token, it is clicking
    accept on :data:`PYANNOTE_GATED_PAGES` with that same account. A 403 on the
    *files* with a 200 on the metadata is exactly this.
    """


class DiarizerWeightsError(DiarizationError):
    """The weights could not be loaded for any other reason.

    Offline with a cold cache, a renamed pipeline, a corrupt download. Kept
    distinct from the access and token cases so its message can talk about the
    cache and the network instead of about terms.
    """


def _install_hint() -> str:
    return (
        "install it with: pip install 'music-video-maker[diarize]' (or pip install "
        "'pyannote.audio>=3.1'). It pulls torch, so install it on the machine that "
        "actually diarizes -- the rest of this pipeline does not need it"
    )


def _token_hint() -> str:
    return (
        f"set one of {', '.join(PYANNOTE_TOKEN_ENV_VARS)} in your environment (this "
        "project reads .env, which is gitignored -- never put a token in run.toml). "
        "Create a read-scoped token at https://huggingface.co/settings/tokens"
    )


def _accept_hint() -> str:
    pages = "\n  ".join(PYANNOTE_GATED_PAGES)
    return (
        "the weights are GATED: sign in as the account that owns the token and click "
        "accept on every one of these pages, then retry --\n  " + pages + "\n"
        "Accepting only the first gives a 403 on the others. Metadata answering 200 "
        "while the files answer 403 is this exact state"
    )


# --------------------------------------------------------------------------- #
# The real diarizer
# --------------------------------------------------------------------------- #


def _import_pyannote() -> object:
    """Import ``pyannote.audio`` lazily, or refuse with the install line.

    Lazy for the same reason ``alignment._load_model`` is: the package pulls
    torch, every other lane's tests do not install it, and a module-level
    import would break them all. Separated out as a function so a test can
    inject one and exercise the failure messages without the package.
    """
    try:
        from pyannote import audio as pyannote_audio
    except ImportError as exc:
        raise DiarizerUnavailableError(
            f"pyannote.audio is not installed, so diarize = true cannot run: {_install_hint()}"
        ) from exc
    return pyannote_audio  # pragma: no cover - only on a machine with the extra installed


def _token_from_env(env: Mapping[str, str]) -> str | None:
    for name in PYANNOTE_TOKEN_ENV_VARS:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


_ACCESS_MARKERS = ("403", "gated", "awaiting", "accept the conditions", "access to model")
_TOKEN_MARKERS = ("401", "unauthorized", "invalid user token", "invalid credentials")


def _classify_load_failure(exc: BaseException, pipeline_name: str) -> DiarizationError:
    """Turn whatever ``from_pretrained`` raised into one of our four refusals.

    Matched on the exception's own text rather than on a huggingface_hub
    exception class, deliberately: the hub's exception taxonomy has moved more
    than once and this module must not acquire an import of it just to catch
    it. The classification is a *message* decision -- every branch still
    refuses -- so a miss costs the operator a less specific hint, never a wrong
    outcome.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _ACCESS_MARKERS):
        return DiarizerAccessError(
            f"could not load {pipeline_name}: {_accept_hint()} (underlying error: {exc})"
        )
    if any(marker in text for marker in _TOKEN_MARKERS):
        return DiarizerTokenError(
            f"could not load {pipeline_name}: the Hugging Face token was rejected -- "
            f"{_token_hint()} (underlying error: {exc})"
        )
    return DiarizerWeightsError(
        f"could not load {pipeline_name}: the weights are not available. Nothing is "
        "committed to this repository, so they are fetched once per machine and cached "
        "under HF_HOME (default ~/.cache/huggingface) -- run it online once, with the "
        f"terms already accepted ({PYANNOTE_GATED_PAGES[0]}). Underlying error: {exc}"
    )


def spans_from_annotation(annotation: object) -> tuple[SpeakerSpan, ...]:
    """Flatten a pyannote ``Annotation`` into :class:`SpeakerSpan`s.

    Duck-typed against ``itertracks(yield_label=True)``, which is the only part
    of pyannote's API this module depends on -- so a fake annotation in a test
    is five lines and no dependency. Turns are returned in time order, since
    everything downstream of here intersects them against aligned segments and
    an ordered list makes a failure readable in the log.
    """
    tracks = annotation.itertracks(yield_label=True)  # type: ignore[attr-defined]
    spans = [
        SpeakerSpan(start=float(turn.start), end=float(turn.end), speaker=str(label))
        for turn, _track, label in tracks
    ]
    spans.sort(key=lambda span: (span.start, span.end, span.speaker))
    return tuple(spans)


def build_pyannote_diarizer(
    *,
    pipeline_name: str = PYANNOTE_PIPELINE,
    token: str | None = None,
    env: Mapping[str, str] | None = None,
    importer: Callable[[], object] | None = None,
) -> Diarizer:
    """Load the gated pyannote pipeline and return it as a :type:`Diarizer`.

    Raises one of the four :class:`DiarizationError` subclasses, each naming
    its own remedy, before any audio is read -- the token is checked first
    because it is the cheapest thing to be wrong and the most common.

    ``env`` and ``importer`` exist so every refusal is reachable in a fully
    offline test on a machine with no ``pyannote`` and no token: the same
    reason ``align`` takes a ``model``. Defaults are the real environment and
    the real lazy import.

    Logs :data:`ATTRIBUTION` at INFO on success. That is not decoration: these
    weights are CC-BY-4.0 and the credit belongs where the work is used.
    """
    env = os.environ if env is None else env

    resolved_token = token if token and token.strip() else _token_from_env(env)
    if not resolved_token:
        raise DiarizerTokenError(
            f"no Hugging Face token found, so the gated {pipeline_name} weights cannot be "
            f"fetched: {_token_hint()}. The terms must also be accepted -- {_accept_hint()}"
        )

    module = (importer or _import_pyannote)()

    try:
        pipeline = module.Pipeline.from_pretrained(  # type: ignore[attr-defined]
            pipeline_name, use_auth_token=resolved_token
        )
    except DiarizationError:
        # Already one of ours (an injected importer, or a future loader that
        # refuses for its own reason) -- re-raised rather than re-wrapped, so a
        # specific message is not replaced by the generic weights one.
        raise
    except Exception as exc:
        raise _classify_load_failure(exc, pipeline_name) from exc

    if pipeline is None:
        # pyannote's documented soft failure: from_pretrained returns None
        # rather than raising when the hub refuses it. Treated as the access
        # case because that is what it is in practice -- a valid token whose
        # account has not accepted the terms.
        raise DiarizerAccessError(
            f"{pipeline_name} loaded as None, which is how pyannote reports that the hub "
            f"refused it rather than raising: {_accept_hint()}"
        )

    logger.info("%s", ATTRIBUTION)

    def _diarize(audio_path: Path) -> Sequence[SpeakerSpan]:
        logger.info("Diarizing %s with %s", audio_path, pipeline_name)
        return spans_from_annotation(pipeline(str(audio_path)))

    return _diarize


def lazy_pyannote_diarizer(**kwargs: object) -> Diarizer:
    """A :type:`Diarizer` that builds the real pipeline on first *use*.

    The default seam the pipeline injects, and the reason it is deferred is the
    degrade path: :func:`assign_characters` catches
    :class:`DiarizationError` raised while *diarizing*, so a loader failure
    raised at wiring time would escape it and end the run over a missing
    environment variable. Deferring construction puts all four refusals inside
    the one ``try`` that knows how to degrade from them.

    Costs nothing when diarization is not asked for: this function builds a
    closure, and the closure is what imports anything.
    """

    def _diarize(audio_path: Path) -> Sequence[SpeakerSpan]:
        return build_pyannote_diarizer(**kwargs)(audio_path)  # type: ignore[arg-type]

    return _diarize


# --------------------------------------------------------------------------- #
# Report shapes
# --------------------------------------------------------------------------- #

ASSIGNED = "assigned"
"""Detection wrote this segment's ``characters``; nobody had tagged it."""
AGREED = "agreed"
"""A tag was in force and detection named the same character."""
DISAGREED = "disagreed"
"""A tag was in force and detection named someone else. Authored value kept."""
UNCOVERED = "uncovered"
"""Speaker turns covered too little of the segment to say anything."""
CONTESTED = "contested"
"""No speaker held :data:`DEFAULT_MIN_SHARE` of the covered time."""
UNMAPPED = "unmapped"
"""The winning cluster label has no entry in ``diarization_speakers``."""
UNCONFIDENT = "unconfident"
"""The winner's own confidence, where the diarizer supplies one, was too low."""

UNDECIDED_DECISIONS = frozenset({UNCOVERED, CONTESTED, UNMAPPED, UNCONFIDENT})
"""Decisions in which detection declined to name anybody.

On an untagged segment each of these is a *loud fallback* to
``default_lead_vocalist``: the field already holds that name, so no data moves
-- what the run gains is a log line and a report entry saying the wrong face
here would be the default face, not a guess. On a tagged segment they are a
no-op, because the tag was already the answer."""


@dataclass(frozen=True)
class SpeakerSummary:
    """One detected cluster, in terms an operator can act on.

    This is the table that makes the feature usable at all. Diarization never
    produces a name, so the first run on a song is *expected* to assign
    nothing: you read this summary, listen at ``first_onset``, and write the
    mapping into ``diarization_speakers``. Hence ``example_text`` -- the lyric
    line the cluster's longest turn lands on, which is usually enough to
    recognise a voice without opening the audio at all.
    """

    speaker: str
    total_seconds: float
    span_count: int
    first_onset: float
    mapped_to: str | None = None
    example_text: str = ""


@dataclass(frozen=True)
class SegmentAssignment:
    """What diarization decided about one aligned segment, and on what numbers.

    ``characters`` is the value the segment carries *after* this step, so a
    reader never has to re-derive the outcome from ``decision``. ``authored``
    records whether a tag was in force, which is what made the decision what it
    is.
    """

    segment_index: int
    start: float
    end: float
    decision: str
    authored: bool
    characters: tuple[str, ...]
    speaker: str | None = None
    detected_characters: tuple[str, ...] = ()
    coverage: float = 0.0
    share: float = 0.0


@dataclass(frozen=True)
class Disagreement:
    """A tagged segment whose detected voice is somebody else.

    Reported, never resolved. The authored value stayed; this is the receipt
    that says so, and it carries both names plus the numbers behind the
    detection so the operator can judge which side is wrong.
    """

    segment_index: int
    start: float
    end: float
    authored: tuple[str, ...]
    detected: tuple[str, ...]
    speaker: str
    share: float
    coverage: float
    text: str

    def describe(self) -> str:
        return (
            f"segment {self.segment_index} [{self.start:.3f}-{self.end:.3f}s] is tagged "
            f"{list(self.authored)} but {self.speaker} (mapped to {list(self.detected)}) holds "
            f"{self.share * 100:.0f}% of its detected voice (coverage {self.coverage * 100:.0f}%)"
            f": {self.text[:48]!r}"
        )


@dataclass(frozen=True)
class DiarizationReport:
    """What diarization did to a run, in the run's own terms.

    ``failure`` is the degrade path: non-``None`` means the diarizer could not
    run at all, every other field is empty, and the alignment came back
    untouched. A caller that wants to refuse on that can -- nothing here
    decides for it -- but the default behaviour is to carry on with the manual
    tags, because a dead run costs hours and this costs a known fallback.
    """

    audio_path: Path | None
    speakers: tuple[SpeakerSummary, ...] = ()
    assignments: tuple[SegmentAssignment, ...] = ()
    disagreements: tuple[Disagreement, ...] = ()
    min_coverage: float = DEFAULT_MIN_COVERAGE
    min_share: float = DEFAULT_MIN_SHARE
    failure: str | None = None

    @property
    def assigned_count(self) -> int:
        return sum(1 for a in self.assignments if a.decision == ASSIGNED)

    @property
    def agreed_count(self) -> int:
        return sum(1 for a in self.assignments if a.decision == AGREED)

    @property
    def fallback_segment_indices(self) -> tuple[int, ...]:
        """Untagged segments detection declined to name, so the default lead stands.

        Only the untagged ones: an undecided detection over a *tagged* segment
        changes nothing and warns about nothing, because the tag was already
        the answer.
        """
        return tuple(
            a.segment_index
            for a in self.assignments
            if a.decision in UNDECIDED_DECISIONS and not a.authored
        )

    def summary(self) -> str:
        if self.failure is not None:
            return f"Diarization did not run: {self.failure}"
        return (
            f"Diarization of {self.audio_path.name if self.audio_path else '(no audio)'}: "
            f"{len(self.speakers)} speaker cluster(s) over {len(self.assignments)} aligned "
            f"segment(s) -- {self.assigned_count} assigned, {self.agreed_count} agreed with a "
            f"tag, {len(self.disagreements)} disagreed with a tag (authored value kept), "
            f"{len(self.fallback_segment_indices)} untagged segment(s) left on the default lead"
        )

    def speaker_table(self) -> str:
        """The cluster table, as an operator-readable block.

        Returned as a string rather than printed: everything diagnostic in this
        project goes through ``logger``, and ``T20`` forbids ``print`` outside
        the two module CLIs whose table *is* the product.
        """
        if not self.speakers:
            return "  (no speaker clusters detected)"
        rows = [
            f"  {s.speaker:<16} {s.total_seconds:8.2f}s  {s.span_count:4d} turn(s)  "
            f"first at {s.first_onset:8.3f}s  -> "
            f"{s.mapped_to or 'UNMAPPED -- add it to [diarization_speakers]'}"
            + (f"  e.g. {s.example_text[:40]!r}" if s.example_text else "")
            for s in self.speakers
        ]
        return "\n".join(rows)

    def log_summary(self) -> None:
        """Log the summary, the cluster table, every disagreement, and the
        standing caveat about the thresholds.

        Ordered so the thing an operator must act on is last: a first run on a
        new song ends with "here are the clusters, nothing was assigned, write
        the mapping", which is the next step rather than a complaint.
        """
        if self.failure is not None:
            logger.error(
                "Diarization was requested but could not run, so this run falls back to the "
                "manual [Name: Role] tags and default_lead_vocalist exactly as if diarize "
                "were false. %s",
                self.failure,
            )
            return

        logger.info("%s", self.summary())
        logger.info("Detected speaker clusters:\n%s", self.speaker_table())

        for clash in self.disagreements:
            logger.warning(
                "Diarization disagrees with an authored tag and the TAG WINS: %s. Nothing was "
                "overwritten -- a silent overwrite of an authored fact is the defect, not the "
                "fix. If the detection is right, fix the tag in the lyrics file (issue #101).",
                clash.describe(),
            )

        fallbacks = self.fallback_segment_indices
        if fallbacks:
            undecided = {
                a.segment_index: a
                for a in self.assignments
                if a.segment_index in set(fallbacks)
            }
            logger.warning(
                "Diarization could not name a voice on %d untagged segment(s): %s. Each keeps "
                "default_lead_vocalist, which is a known and nameable failure mode -- the wrong "
                "face on screen is worse than the default face. Reasons: %s",
                len(fallbacks),
                fallbacks,
                sorted({a.decision for a in undecided.values()}),
            )

        unmapped = [s.speaker for s in self.speakers if s.mapped_to is None]
        if unmapped:
            logger.warning(
                "%d detected speaker cluster(s) have no entry in diarization_speakers and so "
                "named nobody: %s. Diarization yields clusters, never names: listen at each "
                "cluster's first onset above and write the mapping into the run config. Until "
                "then those spans keep whatever the lyrics file says.",
                len(unmapped),
                unmapped,
            )

        logger.warning(
            "Diarization thresholds are UNMEASURED: min_coverage=%.2f and min_share=%.2f have "
            "never been scored against a hand-tagged multi-vocalist song, so they are shapes, "
            "not measurements (issue #101, docs/vocalist-diarization.md). Check the agreed/"
            "disagreed split above against what you can hear before trusting an assignment.",
            self.min_coverage,
            self.min_share,
        )


@dataclass(frozen=True)
class DiarizationResult:
    """The alignment after diarization, beside the report that explains it.

    ``alignment`` is a new object of the *same class* as the one handed in --
    ``dataclasses.replace`` preserves ``CounterpointAlignmentResult``, so a
    level-3 lyrics file does not silently become a plain one by passing through
    here (``alignment``'s own docstring explains why that distinction matters).
    """

    alignment: AlignmentResult
    report: DiarizationReport


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def summarise_speakers(
    spans: Sequence[SpeakerSpan],
    segments: Sequence[AlignedSegment] = (),
    speakers: Mapping[str, str] | None = None,
) -> tuple[SpeakerSummary, ...]:
    """One :class:`SpeakerSummary` per detected cluster, longest first.

    Longest first because that is the order an operator wants to label them in:
    the lead is almost always the biggest cluster, so the first row is the one
    whose name they already know.

    ``segments`` is optional and used only for ``example_text`` -- the lyric a
    cluster's single longest turn overlaps most, which is what makes a cluster
    identifiable without opening the audio.
    """
    mapping = dict(speakers or {})
    by_speaker: dict[str, list[SpeakerSpan]] = {}
    for span in spans:
        by_speaker.setdefault(span.speaker, []).append(span)

    summaries = []
    for speaker, owned in by_speaker.items():
        longest = max(owned, key=lambda s: s.duration)
        example = ""
        best_overlap = 0.0
        for segment in segments:
            overlap = _overlap(longest.start, longest.end, segment.start, segment.end)
            if overlap > best_overlap + _EPS:
                best_overlap = overlap
                example = segment.text
        summaries.append(
            SpeakerSummary(
                speaker=speaker,
                total_seconds=sum(s.duration for s in owned),
                span_count=len(owned),
                first_onset=min(s.start for s in owned),
                mapped_to=mapping.get(speaker),
                example_text=example,
            )
        )
    summaries.sort(key=lambda s: (-s.total_seconds, s.speaker))
    return tuple(summaries)


@dataclass(frozen=True)
class _Detection:
    """What the turns say about one segment, before any authored tag is consulted."""

    speaker: str | None
    coverage: float
    share: float
    refusal: str | None


def _detect(
    segment: AlignedSegment,
    spans: Sequence[SpeakerSpan],
    *,
    speakers: Mapping[str, str],
    min_coverage: float,
    min_share: float,
    min_confidence: float | None,
) -> _Detection:
    """Intersect ``spans`` with one segment and name its dominant speaker.

    The winner is decided by overlapping **duration** inside the segment --
    issue #40's dominant-voice rule as #92 re-measured it, and deliberately the
    same measure ``slicing._dominant_character_member`` uses. Ties keep the
    earliest-starting speaker, matching that function's strict ``>``.
    """
    duration = segment.duration
    if duration <= _EPS:
        return _Detection(None, 0.0, 0.0, UNCOVERED)

    totals: dict[str, float] = {}
    weighted_confidence: dict[str, float] = {}
    confident_seconds: dict[str, float] = {}
    first_seen: dict[str, float] = {}
    for span in spans:
        overlap = _overlap(span.start, span.end, segment.start, segment.end)
        if overlap <= _EPS:
            continue
        totals[span.speaker] = totals.get(span.speaker, 0.0) + overlap
        first_seen.setdefault(span.speaker, span.start)
        if span.confidence is not None:
            weighted_confidence[span.speaker] = (
                weighted_confidence.get(span.speaker, 0.0) + span.confidence * overlap
            )
            confident_seconds[span.speaker] = confident_seconds.get(span.speaker, 0.0) + overlap

    covered = sum(totals.values())
    coverage = min(covered / duration, 1.0)
    if coverage < min_coverage:
        return _Detection(None, coverage, 0.0, UNCOVERED)

    # Earliest onset breaks a tie, so the result does not depend on dict order.
    winner = min(totals, key=lambda s: (-totals[s], first_seen[s], s))
    share = totals[winner] / covered if covered > _EPS else 0.0
    if share < min_share:
        return _Detection(winner, coverage, share, CONTESTED)

    if min_confidence is not None and confident_seconds.get(winner, 0.0) > _EPS:
        mean_confidence = weighted_confidence[winner] / confident_seconds[winner]
        if mean_confidence < min_confidence:
            return _Detection(winner, coverage, share, UNCONFIDENT)

    if winner not in speakers:
        return _Detection(winner, coverage, share, UNMAPPED)

    return _Detection(winner, coverage, share, None)


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #


def assign_characters(
    alignment: AlignmentResult,
    *,
    diarizer: Diarizer,
    audio_path: Path,
    speakers: Mapping[str, str] | None = None,
    default_lead_vocalist: str | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_share: float = DEFAULT_MIN_SHARE,
    min_confidence: float | None = None,
) -> DiarizationResult:
    """Fill ``characters`` on the segments nobody tagged, and report the rest.

    ``audio_path`` should be the **isolated vocal stem** (``vocal_stem``, issue
    #25), not the master: diarizing a full mix is what made this not worth
    attempting, and a voice-like lead synth clusters as a singer. Nothing here
    enforces that -- the config loader does, because it is the layer that knows
    what the run has.

    ``speakers`` maps cluster labels to cast names and is authored data. Absent
    or empty is a **legitimate first run**: nothing is assigned, the cluster
    table is logged, and the operator writes the mapping from it. That is the
    designed workflow, not a degraded one.

    ``default_lead_vocalist`` is used only in log messages, to name what an
    untagged segment is falling back to. The data needs no change for that
    fallback -- the parser already put that name there -- which is the point:
    declining to assign is not a second guess, it is leaving the known value
    alone.

    Every :class:`DiarizationError` is caught here and turned into
    ``report.failure``, with the alignment returned untouched. See this
    module's docstring for why a missing token must not end a render.
    """
    speakers = dict(speakers or {})

    try:
        spans = tuple(diarizer(Path(audio_path)))
    except DiarizationError as exc:
        report = DiarizationReport(
            audio_path=Path(audio_path),
            min_coverage=min_coverage,
            min_share=min_share,
            failure=str(exc),
        )
        report.log_summary()
        return DiarizationResult(alignment=alignment, report=report)

    segments = alignment.segments
    summaries = summarise_speakers(spans, segments, speakers)

    assignments: list[SegmentAssignment] = []
    disagreements: list[Disagreement] = []
    updated: list[AlignedSegment] = []

    for segment in segments:
        detection = _detect(
            segment,
            spans,
            speakers=speakers,
            min_coverage=min_coverage,
            min_share=min_share,
            min_confidence=min_confidence,
        )
        detected: tuple[str, ...] = ()
        if detection.refusal is None and detection.speaker is not None:
            detected = (speakers[detection.speaker],)

        if detection.refusal is not None:
            decision = detection.refusal
            kept = segment
        elif segment.characters_authored:
            # The whole rule, in one branch: an authored tag is never
            # overwritten, and the clash is recorded instead.
            if detected and detected[0] == segment.character:
                decision = AGREED
            else:
                decision = DISAGREED
                disagreements.append(
                    Disagreement(
                        segment_index=segment.index,
                        start=segment.start,
                        end=segment.end,
                        authored=segment.characters,
                        detected=detected,
                        speaker=detection.speaker or "",
                        share=detection.share,
                        coverage=detection.coverage,
                        text=segment.text,
                    )
                )
            kept = segment
        else:
            decision = ASSIGNED
            kept = replace(segment, characters=detected, characters_authored=False)

        updated.append(kept)
        assignments.append(
            SegmentAssignment(
                segment_index=segment.index,
                start=segment.start,
                end=segment.end,
                decision=decision,
                authored=segment.characters_authored,
                characters=kept.characters,
                speaker=detection.speaker,
                detected_characters=detected,
                coverage=detection.coverage,
                share=detection.share,
            )
        )

    report = DiarizationReport(
        audio_path=Path(audio_path),
        speakers=summaries,
        assignments=tuple(assignments),
        disagreements=tuple(disagreements),
        min_coverage=min_coverage,
        min_share=min_share,
    )
    report.log_summary()
    if default_lead_vocalist and report.fallback_segment_indices:
        logger.info(
            "The %d untagged segment(s) diarization declined to name keep %r, the configured "
            "default lead vocalist.",
            len(report.fallback_segment_indices),
            default_lead_vocalist,
        )

    # replace() rather than a constructor call: it preserves the concrete class,
    # so a CounterpointAlignmentResult stays one (and keeps its concurrent
    # segments, which are NOT diarized -- the aligner never heard those voices
    # separately, so their timings are derived and their characters came from an
    # explicit sub-block tag either way).
    return DiarizationResult(
        alignment=replace(alignment, segments=tuple(updated)), report=report
    )


def iter_decisions(report: DiarizationReport, decision: str) -> Iterable[SegmentAssignment]:
    """Every assignment with one decision -- a reading convenience for callers
    and tests, so neither has to re-filter the tuple by hand."""
    return (a for a in report.assignments if a.decision == decision)


__all__ = [
    "AGREED",
    "ASSIGNED",
    "ATTRIBUTION",
    "CONTESTED",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MIN_SHARE",
    "DISAGREED",
    "Diarizer",
    "DiarizationError",
    "DiarizationReport",
    "DiarizationResult",
    "DiarizerAccessError",
    "DiarizerTokenError",
    "DiarizerUnavailableError",
    "DiarizerWeightsError",
    "Disagreement",
    "PYANNOTE_CODE_LICENCE",
    "PYANNOTE_GATED_PAGES",
    "PYANNOTE_MODEL_LICENCE",
    "PYANNOTE_PIPELINE",
    "PYANNOTE_TOKEN_ENV_VARS",
    "SegmentAssignment",
    "SpeakerSpan",
    "SpeakerSummary",
    "UNCONFIDENT",
    "UNCOVERED",
    "UNDECIDED_DECISIONS",
    "UNMAPPED",
    "assign_characters",
    "build_pyannote_diarizer",
    "iter_decisions",
    "lazy_pyannote_diarizer",
    "spans_from_annotation",
    "summarise_speakers",
]
