"""Stage 2b: deterministic multimodal prompt expansion (issue #5).

This is **pure string composition** -- no LLM, no API call, no inference of
any kind. `expand_prompt` is a plain, deterministic function that turns one
:class:`~music_video_maker.contracts.AudioChunk` into the exact text prompt
that will condition MiniMax H3, plus the reference-image path Stage 3 will
upload for it.

Per the blueprint (extended by issues #31, #32 and #53), the prompt is the
additive concatenation of, in order:

1. **Global Style** -- ``config.global_style``.
2. **Cinematography** -- ``config.cinematography`` (issue #53), e.g. ``"35mm
   film, shallow depth of field, warm natural light"``: the whole-video film
   look -- stock, lens, grade, lighting philosophy. Composed unconditionally
   right after Global Style, at the same fixed position on every chunk --
   this is film-level language, not shot-level, and giving it a stable
   position is what stops it drifting back into being free text inside
   ``global_style`` or ``shot`` the way it used to. ``None`` omits the
   sentence entirely.
3. **Narrative Concept** -- ``config.narrative_concept``, or this chunk's
   authored ``shot`` line when a shot plan supplies one, with this chunk's
   authored ``camera`` direction (issue #53) appended as a **trailing
   clause** on that same sentence, never as a separate one -- see "Camera
   direction" below.
4. **Setting** -- ``config.setting`` (issue #32), e.g. ``"London, UK --
   contemporary, overcast winter"``. Composed unconditionally, after the
   concept/shot sentence -- see ``_setting_clause``'s own docstring for why
   position matters here too -- so it lands on *every* chunk regardless of
   whether the chunk carries a lyric, a shot-plan line, or neither. That is
   the entire point of #32: geography is a whole-video property and must not
   be able to drift shot to shot. ``None`` omits the sentence entirely; it
   never fabricates a place.
5. **Active character state** -- name + role + visual descriptors of
   whichever cast member(s) are active for this chunk (lead vocalist, a
   background member such as the drummer injected for a specific shot, or --
   issue #33 levels 1-2 -- more than one name at once, e.g. two vocalists
   singing the same words). One active member composes exactly as before:
   name, role, then that member's **appearance direction** (issue #31) and
   **demeanour direction** (issue #74) in the same clause -- appearance first
   (``config.global_appearance`` then ``member.appearance``), demeanour
   second (``config.global_demeanour`` then this member's own
   ``member.demeanour``), all optional, comma-joined, any or all silently
   omitted if ``None``.

   **More than one active member** composes as one shared clause, never one
   clause per person: each member contributes its own ``name, role[,
   appearance][, demeanour]`` fragment (that member's own appearance and
   demeanour stay attached to that member, per #33's decision that
   resolution is incremental but per-character data must not blur together),
   the fragments are joined into a natural list ("A and B are the focus of
   this shot" / "A, B, and C are the focus of this shot"), and
   ``config.global_appearance``/``config.global_demeanour`` -- being
   whole-cast properties, not per-member ones -- are appended exactly once at
   the end of the shared clause rather than once per person.

   This clause **never asserts staging**. Per #33's recorded decision, `&`
   (or any multi-name chunk) states who is *audible*, not who is *visible*
   or how they are framed -- composing "both appear side by side" or "split
   screen" here would move an editorial decision the shot plan owns into
   what is supposed to read as a transcript. How multiple active characters
   are actually framed together is the shot line's job, already composed
   in part 3.
6. **The literal lyric line** -- phrased for lip-sync/contextual meaning, or
   an instrumental-passage clause when the chunk carries no lyric text.
   Grammatical number agrees with how many characters are active (singular
   "the character is..." for one, plural "the characters are..." for more
   than one) -- this is the only change multiple active characters make to
   part 6, which otherwise stays exactly what it always was: the sole
   authority on whether anyone is singing (see the rule below).
7. **The avoid list** (issue #73) -- ``config.avoid`` is *not* composed
   into a prompt. It was for exactly one render, and it put a climbing
   harness on a character who had none. H3 has one prompt input and no
   negative-conditioning channel, so a prohibition can only add its own
   nouns to the only channel there is. See ``RunConfig.avoid``.

"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from music_video_maker.config import RunConfig
from music_video_maker.contracts import AudioChunk, CastMember, ExpandedPrompt

logger = logging.getLogger(__name__)

_INSTRUMENTAL_CLAUSE_SINGULAR = (
    "Instrumental passage: the character performs silently, no lyric to sing"
)
_INSTRUMENTAL_CLAUSE_PLURAL = (
    "Instrumental passage: the characters perform silently, no lyric to sing"
)


class UnknownCastMemberError(ValueError):
    """Raised when a chunk names an active character absent from the cast dictionary.

    A typo'd or stale character tag must fail loudly, never quietly become
    the lead vocalist -- that would silently put the wrong face in frame.
    """


def expand_prompt(
    config: RunConfig,
    chunk: AudioChunk,
    shot: str | None = None,
    subject_is_focus: bool = True,
    camera: str | None = None,
    present: Sequence[str] = (),
) -> ExpandedPrompt:
    """Compose the deterministic Stage 2b prompt for one audio chunk.

    The active cast members are ``config.cast[name]`` for every ``name`` in
    ``chunk.characters``, in order (issue #33 levels 1-2 -- one name is the
    original solo case, more than one is two or more vocalists active on the
    same chunk). An empty ``chunk.characters`` falls back to
    ``[config.default_lead_vocalist]``, same as the pre-#33 single-name rule.
    Raises :class:`UnknownCastMemberError` naming the specific offending name
    if any entry in ``chunk.characters`` is absent from ``config.cast`` --
    even when other names in the same tuple resolve fine, because a typo must
    never quietly become "just the members we recognised".

    ``shot`` is this chunk's authored direction from the shot plan (see
    :mod:`music_video_maker.shot_plan`). When given it takes the place of
    ``config.narrative_concept`` -- and only that; the global style, the
    active character(s) and the lyric line are still composed around it. When
    ``None`` the global concept is used, which is every chunk in a run with
    no shot plan.

    ``present`` (issue #59) is this chunk's on-screen-but-not-singing cast,
    from ``ShotPlanEntry.present``. Those members contribute their name, role
    and appearance to a clause of their own and their reference photos to
    ``image_refs``, but never the lyric: ``chunk.characters`` remains the
    sole authority on who is *audible*, and ``ExpandedPrompt.characters``
    still reports only the singers. A name that is already singing this chunk
    is dropped rather than composed twice. Empty (the default) composes
    byte-identically to the pre-#59 prompt.

    ``camera`` (issue #53) is this chunk's authored camera direction, from
    ``ShotPlanEntry.camera``. It is independent of ``shot``: passing one
    without the other is fine, and ``camera`` still composes even when
    ``shot`` is ``None`` and the concept falls back to
    ``config.narrative_concept``. See :func:`_apply_camera_clause` for why it
    is appended to that sentence rather than composed as its own.

    Passing the text in rather than looking it up keeps this module free of
    file I/O and keeps it a pure function of its arguments -- resolution
    (including the drift check) happens once, upstream, in ``cli``.
    """
    members = _resolve_active_members(config, chunk)
    present_members = _resolve_present_members(config, chunk, present, members)
    prompt = _compose_prompt(
        config, members, chunk, shot, subject_is_focus, camera=camera, present=present_members
    )
    # Issue #46: the chained I2V path has no reference photo, so the seed
    # frame (the predecessor's own output) is already the output of the
    # appearance clause -- restating it applies it a second time. Same
    # composition function, appearance omitted, so the two variants cannot
    # drift out of step with each other. Camera direction is about framing,
    # not identity, so it stays on the chained variant unchanged.
    chained_prompt = _compose_prompt(
        config,
        members,
        chunk,
        shot,
        subject_is_focus,
        include_appearance=False,
        camera=camera,
        present=present_members,
    )

    logger.debug(
        "expanded prompt for chunk_id=%s characters=%s prompt_len=%d chained_prompt_len=%d",
        chunk.chunk_id,
        tuple(member.name for member in members),
        len(prompt),
        len(chained_prompt),
    )

    return ExpandedPrompt(
        chunk_id=chunk.chunk_id,
        prompt=prompt,
        chained_prompt=chained_prompt,
        image_ref=members[0].image,
        image_refs=tuple(
            member.image for member in (*members, *present_members)
        ),
        characters=tuple(member.name for member in members),
        present_cast=tuple(member.name for member in present_members),
    )


def _resolve_active_members(config: RunConfig, chunk: AudioChunk) -> tuple[CastMember, ...]:
    names = chunk.characters or (config.default_lead_vocalist,)
    members = []
    for name in names:
        try:
            members.append(config.cast[name])
        except KeyError:
            logger.error(
                "chunk_id=%s references unknown cast character %r (known cast: %s)",
                chunk.chunk_id,
                name,
                sorted(config.cast),
            )
            raise UnknownCastMemberError(
                f"chunk {chunk.chunk_id} references unknown cast character {name!r}; "
                f"known cast members: {sorted(config.cast)}"
            ) from None
    return tuple(members)


def _resolve_present_members(
    config: RunConfig,
    chunk: AudioChunk,
    present: Sequence[str],
    active: tuple[CastMember, ...],
) -> tuple[CastMember, ...]:
    """Resolve ``ShotPlanEntry.present`` names to cast members (issue #59).

    Anyone already singing this chunk is dropped: naming Jan in ``present``
    on a chunk he also sings is a reasonable thing for an author to write,
    and it must not compose him twice or stage his photo twice. An unknown
    name raises, exactly as a singing name does -- see
    :func:`_resolve_active_members`."""
    singing = {member.name for member in active}
    members = []
    for name in present:
        if name in singing:
            continue
        try:
            member = config.cast[name]
        except KeyError:
            logger.error(
                "chunk_id=%s lists unknown cast character %r as present in shot "
                "(known cast: %s)",
                chunk.chunk_id,
                name,
                sorted(config.cast),
            )
            raise UnknownCastMemberError(
                f"chunk {chunk.chunk_id} lists unknown cast character {name!r} as present "
                f"in shot; known cast members: {sorted(config.cast)}"
            ) from None
        if member not in members:
            members.append(member)
    return tuple(members)


def _present_clause(
    present: tuple[CastMember, ...],
    include_appearance: bool = True,
) -> str | None:
    """Name, role and appearance for everyone on screen who is not singing
    (issue #59), as a single shared clause.

    Says "silent" positively rather than "not singing": this module's hardest
    lesson is that a *character-attached* instruction beats the global lyric
    clause (it is what made a performer sing through every instrumental), so
    the vocal state of a present member has to be attached to that member,
    and a negation is the weaker way to say it.

    ``config.global_appearance`` is whole-cast and is already composed once
    in the character clause, so it is not repeated here -- the same reasoning
    :func:`_multi_character_clause` applies to its own members. No
    ``RunConfig`` is needed here at all: everything this function composes is
    a plain ``CastMember`` attribute.

    ``include_appearance=False`` is issue #46's chained-path switch, and it
    applies to everyone in frame, not just the lead: on the chained path
    there is no photo for anybody, so a present member's appearance would
    compound against the model's own previous output just as the lead's did.
    """
    if not present:
        return None
    descriptors = tuple(_member_descriptor(member, include_appearance) for member in present)
    return f"Also in shot, silent: {_join_member_list(descriptors)}"


def _compose_prompt(
    config: RunConfig,
    members: tuple[CastMember, ...],
    chunk: AudioChunk,
    shot: str | None = None,
    subject_is_focus: bool = True,
    include_appearance: bool = True,
    camera: str | None = None,
    present: tuple[CastMember, ...] = (),
) -> str:
    character_clause = _character_clause(config, members, subject_is_focus, include_appearance)
    lyric_clause = _lyric_clause(chunk.text, len(members), singers=members if present else ())
    concept = shot if shot and shot.strip() else config.narrative_concept
    concept = _apply_camera_clause(concept, camera)
    parts = (
        # Issue #62: the adapter's trigger word, first and bare. A trigger is
        # a tag, not a statement -- giving it a sentence would put it in the
        # grammatical subject slot the shot line needs.
        config.lora_trigger if config.lora else None,
        config.global_style,
        config.cinematography,
        concept,
        None if subject_is_focus else REFOCUS_SENTENCE,
        _setting_clause(config.setting),
        character_clause,
        _present_clause(present, include_appearance),
        lyric_clause,
        *_counterpoint_clauses(chunk),
        # Issue #73: `config.avoid` is deliberately NOT composed here. It was,
        # for exactly one render, and it manufactured what it forbids -- see
        # `RunConfig.avoid`'s docstring for the measurement.
    )
    return _join_sentences(parts)


def _apply_camera_clause(concept: str, camera: str | None) -> str:
    """Append this chunk's camera direction (issue #53) to ``concept`` as a
    **trailing clause on the same sentence**, never as a sentence of its own.

    H3 renders the grammatical subject of a sentence (see
    ``docs/shot-writing-guide.md``'s "Make the effect the grammatical
    subject"), so giving camera direction its own sentence would put "camera"
    in subject position, competing with whatever the shot is actually about.
    Attaching it as a trailing clause -- ``"<concept>, camera <direction>"``,
    the same shape as a hand-written line like "she walks toward camera...,
    camera tracking backwards ahead of her" -- keeps the concept's own
    subject in the subject slot.

    ``camera`` is expected to read naturally after the word "camera" (e.g.
    ``"tracking backwards ahead of her"``), matching
    ``ShotPlanEntry.camera``'s docstring. ``None``/blank leaves ``concept``
    untouched."""
    if not camera or not camera.strip():
        return concept
    return f"{concept.strip().rstrip('.')}, camera {camera.strip().rstrip('.')}"


def _counterpoint_clauses(chunk: AudioChunk) -> tuple[str, ...]:
    """One sentence per counterpoint stream (issue #33 level 3).

    A counterpoint chunk carries two simultaneous lyric texts; saying only
    the spine's leaves the second voice as a silent extra on screen. Each
    stream gets its own transcript-style sentence, parallel in shape to
    :func:`_lyric_clause` and attributed by name, so the model knows which
    face sings which words at the same moment.
    """
    if not chunk.concurrent_texts:
        return ()
    clauses = []
    for text, voices in zip(chunk.concurrent_texts, chunk.concurrent_characters, strict=True):
        names = " and ".join(voices) if voices else "another voice"
        verb = "sing" if len(voices) > 1 else "sings"
        clauses.append(
            f"Simultaneously, over the same seconds, {names} {verb} a different "
            f"counterpoint lyric: '{text.strip()}'"
        )
    return tuple(clauses)


def _setting_clause(setting: str | None) -> str | None:
    """Compose ``setting`` as a *constraint on* the shot, never as a subject
    *of* it (issue #32, corrected after the first Chicago render).

    The original implementation dropped the raw setting in as its own
    declarative sentence high in the prompt. A bare noun phrase about a place
    reads as something to depict, and being both concrete and visually loaded
    it out-competed the shot's own location: chunks written for an intensive
    care corridor and a surgical observation gallery rendered as Loop
    sidewalks, with a medical cart and a mixing console standing out on the
    pavement, because the model moved the action outdoors to get the skyline
    and the elevated tracks into frame. Geography stopped drifting, which is
    what #32 asked for, but the video stopped making sense, which it did not.

    What the field is actually for: *if* a shot shows something
    location-identifying, that location is this one. It must not argue for
    showing a location at all. Hence the conditional phrasing, the explicit
    instruction not to relocate or add landmarks, and the position after the
    shot line so the shot leads and this qualifies.
    """
    if not setting or not setting.strip():
        return None
    return (
        "Location continuity: wherever the location is identifiable it is "
        f"{setting.strip().rstrip('.')}, and never anywhere else; do not relocate the "
        "action or add landmarks to establish it, this shot's own described location "
        "is what is on screen"
    )


REFOCUS_SENTENCE = (
    "What this shot is about is the action described above, not the performer"
)
"""Issue #26. Its own sentence, deliberately, rather than more subordinate
clause on the character text: the phrase it counteracts sits mid-clause with
appearance text after it, and a first attempt that matched only at the end of
the string silently did nothing -- a prompt that reads as applied and never
reaches the model."""

_NOT_THE_SUBJECT = {
    "is the focus of this shot": "is present in this shot but is not its subject",
    "are the focus of this shot": "are present in this shot but are not its subject",
}


def _refocus_on_the_action(clause: str) -> str:
    """Say the performer is present without saying the shot is about them.

    A consequence beat -- a flatlining monitor, a burning printer -- had the
    shot line saying the consequence was the beat while the character clause
    said the performer was the focus. Across four renders the
    character-attached instruction won and the consequence was dropped or
    relegated. Same shape as the bug that made a performer sing through every
    instrumental: per-chunk state asserted globally.

    The performer stays named, described and in frame; only which of the two
    the prompt calls the subject changes. Adds no framing ("at the edge of
    frame") on purpose -- where they stand is the shot line's business, and
    this module must not smuggle staging into character text.
    """
    for assertion, replacement in _NOT_THE_SUBJECT.items():
        if assertion in clause:
            return clause.replace(assertion, replacement, 1)
    return clause


def _character_clause(
    config: RunConfig,
    members: tuple[CastMember, ...],
    subject_is_focus: bool = True,
    include_appearance: bool = True,
) -> str:
    """Name + role + appearance, attached to the active character(s) (issue
    #31, widened by #33). One member composes exactly as before -- this
    branch is untouched so the solo case stays byte-identical. More than one
    member composes as a single shared clause (never one clause per person)
    via :func:`_multi_character_clause`.

    ``include_appearance`` is issue #46's chained-path switch: ``False`` omits
    both ``CastMember.appearance`` and ``config.global_appearance`` while
    leaving name and role untouched -- see the module docstring's "Chained
    I2V prompt variant" section for why."""
    base = (
        _single_character_clause(config, members[0], include_appearance)
        if len(members) == 1
        else _multi_character_clause(config, members, include_appearance)
    )
    return base if subject_is_focus else _refocus_on_the_action(base)


def _single_character_clause(
    config: RunConfig, member: CastMember, include_appearance: bool = True
) -> str:
    """Appearance and demeanour ride in the same clause as the role rather
    than as their own sentences: each is a different *kind* of statement than
    "who this character is", but each is still a statement about this same
    character, not the scene or the shot.

    ``include_appearance=False`` (issue #46) skips the appearance fragment
    entirely -- the chained path has no photo for it to describe, and the
    seed frame already embodies whatever it would have said. Demeanour is
    NOT gated by this flag (issue #74): it is behavioural direction, the same
    kind of statement as ``role``, not a photo description, and it is safe to
    restate every chunk because it is required to be phrased as an endpoint
    -- see the module docstring's "Demeanour is deliberately NOT gated"
    paragraph."""
    base = f"{member.name}, {member.role}, is the focus of this shot"
    fragments = []
    if include_appearance:
        appearance = _appearance_text(config, member)
        if appearance:
            fragments.append(appearance)
    demeanour = _demeanour_text(config, member)
    if demeanour:
        fragments.append(demeanour)
    if not fragments:
        return base
    return f"{base}, {', '.join(fragments)}"


def _appearance_text(config: RunConfig, member: CastMember) -> str | None:
    """Whole-cast direction followed by this member's own, comma-joined.

    Either or both may be ``None`` -- silence is the correct output for a
    field nobody set, never a fabricated default."""
    fragments = [
        p.strip().rstrip(".")
        for p in (config.global_appearance, member.appearance)
        if p and p.strip()
    ]
    if not fragments:
        return None
    return ", ".join(fragments)


def _demeanour_text(config: RunConfig, member: CastMember) -> str | None:
    """Whole-cast demeanour direction followed by this member's own,
    comma-joined -- the demeanour counterpart of :func:`_appearance_text`
    (issue #74). Either or both may be ``None``.

    ``member.demeanour`` is a plain ``CastMember`` attribute, matching
    ``member.appearance``'s own shape. Unlike appearance, this function is
    never skipped for the chained path; see the module docstring."""
    fragments = [
        p.strip().rstrip(".")
        for p in (config.global_demeanour, member.demeanour)
        if p and p.strip()
    ]
    if not fragments:
        return None
    return ", ".join(fragments)


def _multi_character_clause(
    config: RunConfig, members: tuple[CastMember, ...], include_appearance: bool = True
) -> str:
    """Two or more active characters, composed as one shared clause.

    Each member contributes its own ``name, role[, appearance][, demeanour]``
    fragment -- that member's *own* appearance and demeanour stay attached to
    it, never blurred into a combined block -- and the fragments are joined
    into a natural list ("A and B are the focus of this shot" / "A, B, and C
    are the focus of this shot"). ``config.global_appearance`` and
    ``config.global_demeanour`` are whole-cast, not per-member, so each is
    appended exactly once at the end of the shared clause rather than once
    per person.

    Deliberately does **not** say anything about how these characters are
    framed together (no "side by side", no "split screen") -- per #33's
    recorded decision, the transcript states who is audible, not who is
    visible or how they are staged; that is the shot plan's call and is
    already composed separately as the narrative-concept/shot part of the
    prompt.

    ``include_appearance=False`` (issue #46) strips both the per-member
    fragments' appearance text AND the trailing ``global_appearance``
    fragment -- every appearance source, not just the primary member's.
    Demeanour is unaffected by this flag (issue #74) -- see the module
    docstring."""
    descriptors = tuple(_member_descriptor(m, include_appearance) for m in members)
    clause = f"{_join_member_list(descriptors)} are the focus of this shot"
    global_fragments = []
    if include_appearance:
        global_appearance = _global_appearance_text(config)
        if global_appearance:
            global_fragments.append(global_appearance)
    global_demeanour = _global_demeanour_text(config)
    if global_demeanour:
        global_fragments.append(global_demeanour)
    if not global_fragments:
        return clause
    return f"{clause}, {', '.join(global_fragments)}"


def _member_descriptor(member: CastMember, include_appearance: bool = True) -> str:
    """This member's own ``name, role[, appearance][, demeanour]`` fragment --
    the atomic unit :func:`_multi_character_clause` (and :func:`_present_clause`)
    list out, keeping each performer's appearance and demeanour attached to
    that performer.

    ``include_appearance=False`` (issue #46) drops the appearance fragment.
    Demeanour is never dropped (issue #74) -- see the module docstring for
    why it behaves like ``role`` rather than like ``appearance`` here.
    ``member.appearance`` and ``member.demeanour`` are both plain
    ``CastMember`` attributes, so this needs no ``RunConfig`` at all."""
    base = f"{member.name}, {member.role}"
    fragments = []
    if include_appearance:
        appearance = member.appearance.strip().rstrip(".") if member.appearance else ""
        if appearance:
            fragments.append(appearance)
    demeanour = member.demeanour
    if demeanour and demeanour.strip():
        fragments.append(demeanour.strip().rstrip("."))
    if not fragments:
        return base
    return f"{base}, {', '.join(fragments)}"


def _join_member_list(descriptors: tuple[str, ...]) -> str:
    """Natural-language list join with a serial (Oxford) comma even at two
    items -- each descriptor already contains internal commas (from the
    role), so a bare " and " between two of them reads as one continuous
    list of traits rather than a boundary between two people. The comma
    before "and" disambiguates the list boundary."""
    if len(descriptors) == 1:
        return descriptors[0]
    return ", ".join(descriptors[:-1]) + f", and {descriptors[-1]}"


def _global_appearance_text(config: RunConfig) -> str | None:
    """Whole-cast appearance direction, applied once regardless of how many
    characters are active on this chunk. ``None`` when nobody set it -- never
    a fabricated default."""
    if config.global_appearance and config.global_appearance.strip():
        return config.global_appearance.strip().rstrip(".")
    return None


def _global_demeanour_text(config: RunConfig) -> str | None:
    """Whole-cast demeanour direction, applied once regardless of how many
    characters are active on this chunk -- the demeanour counterpart of
    :func:`_global_appearance_text` (issue #74). ``None`` when nobody set
    it -- never a fabricated default."""
    if config.global_demeanour and config.global_demeanour.strip():
        return config.global_demeanour.strip().rstrip(".")
    return None


def _lyric_clause(
    text: str, character_count: int = 1, singers: tuple[CastMember, ...] = ()
) -> str:
    """The sole authority on whether anyone is singing (see module
    docstring). ``character_count`` only changes grammatical number --
    "the character is" vs "the characters are" -- never whether this clause
    says singing happened; that remains driven purely by whether ``text`` is
    non-empty.

    ``singers`` is passed only when somebody non-singing is also on screen
    (issue #59). "The character is actively singing" is unambiguous while the
    prompt describes one person and stops being so the moment it describes
    two, so in that case the singers are named outright. Left empty otherwise
    so the overwhelmingly common single-character prompt stays byte-identical.

    The instrumental branch deliberately does *not* name anyone: nobody is
    singing, so there is nothing to disambiguate."""
    stripped = text.strip() if text else ""
    plural = character_count > 1
    if not stripped:
        return _INSTRUMENTAL_CLAUSE_PLURAL if plural else _INSTRUMENTAL_CLAUSE_SINGULAR
    if singers:
        names = _join_member_list(tuple(member.name for member in singers))
        verb = "are" if plural else "is"
        return f"{names} {verb} actively singing the lyric: '{stripped}'"
    subject = "The characters are" if plural else "The character is"
    return f"{subject} actively singing the lyric: '{stripped}'"


def _join_sentences(parts: tuple[str, ...]) -> str:
    """Join sentence fragments with '. ', normalizing trailing periods so the
    result never doubles up (e.g. 'cinematic.. Wandering')."""
    cleaned = [p.strip().rstrip(".") for p in parts if p and p.strip()]
    return ". ".join(cleaned) + "."
