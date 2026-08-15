"""The authoring layer (issue #54): invent the concept, plan the shots, direct
the photography -- all before the GPU starts.

**Strictly outside the render path.** ``music_video_maker/authoring/`` is the
only place in this project that talks to an LLM or shells out to a subprocess
for inference. Everything under it produces a ``shot_plan.toml`` a human
reviews and commits; nothing here is ever imported by ``prompting.py`` or any
other module that runs during a render, and nothing outside this package may
import it -- enforced by ``tests/test_authoring_boundary.py``, not just
stated here (the same trick ``tests/test_repo_assets.py`` plays for #51).

Entry point is the separate ``mvm-author`` console script
(:mod:`music_video_maker.authoring.cli`), never a flag on
``music-video-maker`` itself -- see that module's docstring for why a second
binary is the point, not a compromise.

This package ships in phases (issue #54 design, section 12).

* **Phase 1** built the whole seam for one stage:
  :class:`~music_video_maker.authoring.driver.ModelDriver`,
  :mod:`~music_video_maker.authoring.session` staleness tracking, and
  :mod:`~music_video_maker.authoring.concept` (Stage 1).
* **Phase 2** adds :mod:`~music_video_maker.authoring.beats` (Stage 2 -- what
  happens in each chunk, structurally checked against
  ``docs/shot-writing-guide.md``'s three-beat rule) and
  :mod:`~music_video_maker.authoring.reanchor`, which maps those beats onto
  the timeline their own ``length_seconds`` produced. Its render-side twin is
  ``--prepare --from-plan``.

Photography and prose follow in later phases against the same seam.
"""

from __future__ import annotations
