"""The render/authoring import boundary, enforced mechanically (issue #54
design section 2) -- not just documented in a docstring.

Two directions, both checked by parsing every module's AST (never by
importing it -- a parse failure names the exact file and line without ever
executing untrusted-shaped code):

1. Nothing outside ``music_video_maker/authoring/`` may import
   ``music_video_maker.authoring`` (in any form) or shell out via
   ``subprocess`` -- the two things that would let a model call leak into
   the render path.
2. ``music_video_maker/authoring/`` may only reach *into* the render half
   through a small, named allowlist of modules -- never ``cli.py`` itself,
   which is exactly the seam ``mvm-author`` being a separate binary
   (``authoring/cli.py``'s own docstring) depends on staying one-way.

3. **Nothing anywhere in the package may import
   ``music_video_maker.castgen``** -- the issue #56 image generator, the only
   other module here that calls a model. It lives at the top level rather than
   in ``authoring/`` on purpose (it needs ``workflow_graph``, ``staging`` and
   ``custody``, and widening :data:`ALLOWED_AUTHORING_IMPORTS` by three
   render-side modules to host one authoring tool would blur exactly the
   boundary that list draws). So its boundary is enforced from the other
   side: it is a leaf. It is run by hand, once per character, like
   ``python -m music_video_maker.vramsample`` -- and the day something in the
   render path imports it, "does the render binary ever call a model?" stops
   being answerable by reading an import graph, which is the property all of
   this exists to protect.

Same trick ``tests/test_repo_assets.py`` plays for issue #51: the rule that
matters is the one a machine re-checks on every commit.

**Two considered deviations from the design's literal wording**, both
because the literal rule would fail against legitimate, already-tested,
pre-#54 code that has nothing to do with calling a model:

* ``subprocess`` is banned everywhere outside ``authoring/`` *except* the
  modules that already, correctly, shell out for non-model reasons (ffmpeg,
  host-sleep prevention, `tailscale ip -4`) -- see :data:`SUBPROCESS_ALLOWLIST`.
* ``authoring/`` is allowed to import ``logging_setup`` in addition to the
  design's named list (``config``, ``contracts``, ``shot_plan``,
  ``alignment``, ``slicing``, ``lyrics``) -- it is a side-effect-free
  stderr-logging helper, not a render-path concern, and duplicating it
  inside ``authoring/`` to avoid one more name on this list would be pure
  copy-paste for no safety this test would actually be buying.
* ``authoring/`` is also allowed ``alignment_quality``: it is model-free by
  construction (it is the module that must never be an inference call), and
  the already-allowed ``alignment`` imports it anyway, so the name on this
  list records a dependency that was always there rather than widening the
  boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "music_video_maker"
AUTHORING_ROOT = PACKAGE_ROOT / "authoring"

SUBPROCESS_ALLOWLIST = frozenset(
    {
        "assembly.py",  # Stage 5: ffmpeg concat/mux (issue #11) -- no model, predates #54
        "custody.py",  # host-sleep prevention around a render (issue #43) -- no model
        "continuity.py",  # ffprobe/ffmpeg frame extraction (issue #12) -- no model
        "alignment_quality.py",  # ffmpeg astats vocal-energy check (issue #71) -- no model
        "luminance.py",  # ffmpeg frame sampling for the darkness floor (issue #77) -- no model
        "scenecuts.py",  # ffmpeg scene-detection probe (issue #81) -- no model
        "webui.py",  # `tailscale ip -4` bind discovery + ffmpeg thumbnails (issue #36) -- no model
        # control.py (issue #36, 2026-10-04): spawns THIS project's own CLI
        # (`python -m music_video_maker.cli`) so a browser can start and stop
        # a render. The one entry on this list whose subprocess is neither
        # ffmpeg nor a probe -- and it is the strongest "no model" case here,
        # not the weakest: the thing it spawns is the render binary, which
        # pyproject.toml already guarantees calls no model at all. The point
        # of spawning rather than calling in-process is that the custody
        # pre-flight, the disk check, the render-envelope refusal and the
        # unconditional `POST /free` stay the CLI's own.
        "control.py",
        "dome.py",  # ffmpeg/ffprobe only: POST-render domemaster projection + checks -- no model
        # stereo.py (issue #68): ffmpeg probe/decode/encode for a POST-render
        # stereo pass, plus an optional shell-out to a monocular depth model.
        # Listed with the reason spelled out because it is the first entry
        # where "no model" is not the whole answer: a depth estimator is a
        # model, and what this rule protects is the RENDER path's determinism
        # and its no-metered-API guarantee. stereo.py is in neither -- it
        # reads finished chunk mp4s, writes new files, moves no
        # ChunkFingerprint and cannot cause a re-render -- and the depth call
        # is an injected callable that no test ever supplies.
        "stereo.py",
        # ffmpeg decode of a master for the #96 voicing calibration table -- no model.
        # It is a diagnostic an operator runs by hand; nothing imports it at render time.
        "calibrate_voicing.py",
        # `nvidia-smi` polling for the attended 277-frame VRAM proof (issue #98).
        # No model, and not in the render path at all: nothing imports it, it is
        # run by hand from its own terminal (`python -m ...`) beside a render.
        "vramsample.py",
    }
)

ALLOWED_AUTHORING_IMPORTS = frozenset(
    {
        "config",
        "contracts",
        "shot_plan",
        "alignment",
        "slicing",
        "lyrics",
        "logging_setup",  # see module docstring's "considered deviations"
        # alignment_quality is model-free (pure evaluation + ffmpeg astats) and is
        # already pulled in transitively by the allowed `alignment`. chunks.py needs
        # its suspect_segment_indices() so the authoring skeleton logs the same
        # segment-to-chunk mapping the render does (issues #96, #92).
        "alignment_quality",
    }
)


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _dotted_imports(path: Path) -> list[tuple[str, int]]:
    """Every fully-qualified name this file imports, or imports *from*, each
    paired with its 1-indexed source line.

    ``from music_video_maker.config import RunConfig`` yields both
    ``"music_video_maker.config.RunConfig"`` and ``"music_video_maker.config"``
    (the latter is what module-level checks below actually match against);
    ``from music_video_maker import authoring`` yields
    ``"music_video_maker.authoring"`` the same way, so the two spellings of
    "imports the authoring package" are indistinguishable to this checker,
    which is the point.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # relative "from . import x" -- unused in this codebase
            for alias in node.names:
                found.append((f"{node.module}.{alias.name}", node.lineno))
                found.append((node.module, node.lineno))
    return found


def test_no_module_outside_authoring_imports_the_authoring_package():
    violations = []
    for path in _python_files(PACKAGE_ROOT):
        if AUTHORING_ROOT in path.parents or path == AUTHORING_ROOT / "__init__.py":
            continue
        for name, lineno in _dotted_imports(path):
            if name == "music_video_maker.authoring" or name.startswith(
                "music_video_maker.authoring."
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {name!r}")

    assert not violations, (
        "The render path must never import the authoring package (issue #54 design "
        "section 2) -- that is what keeps 'does the render binary ever call a model?' "
        "answerable by reading pyproject.toml alone:\n  " + "\n  ".join(violations)
    )


def test_no_module_outside_authoring_shells_out_via_subprocess():
    violations = []
    for path in _python_files(PACKAGE_ROOT):
        if AUTHORING_ROOT in path.parents or path == AUTHORING_ROOT / "__init__.py":
            continue
        if path.name in SUBPROCESS_ALLOWLIST:
            continue
        for name, lineno in _dotted_imports(path):
            if name == "subprocess" or name.startswith("subprocess."):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {name!r}")

    assert not violations, (
        "A new subprocess import outside authoring/ and outside SUBPROCESS_ALLOWLIST "
        "-- if this is legitimate non-model use (another ffmpeg-shaped call), add the "
        "file to SUBPROCESS_ALLOWLIST with a one-line reason, the same way the three "
        "existing entries are justified. If it calls a model, it belongs in "
        "authoring/ instead:\n  " + "\n  ".join(violations)
    )


def test_nothing_in_the_package_imports_the_image_generator():
    violations = []
    for path in _python_files(PACKAGE_ROOT):
        if path.name == "castgen.py":
            continue
        for name, lineno in _dotted_imports(path):
            if name == "music_video_maker.castgen" or name.startswith(
                "music_video_maker.castgen."
            ):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {name!r}")

    assert not violations, (
        "music_video_maker.castgen calls an image model (issue #56), so it must stay a "
        "leaf nothing imports -- it is run by hand, once per character, at authoring "
        "time. If a render-path module needs something from it, that something belongs "
        "in a module neither of them calls a model from:\n  " + "\n  ".join(violations)
    )


def test_authoring_only_imports_the_allowed_render_side_modules():
    violations = []
    for path in _python_files(AUTHORING_ROOT):
        for name, lineno in _dotted_imports(path):
            if name == "music_video_maker" or not name.startswith("music_video_maker."):
                continue
            if name == "music_video_maker.authoring" or name.startswith(
                "music_video_maker.authoring."
            ):
                continue  # internal to the package -- always fine
            module = name.removeprefix("music_video_maker.").split(".", 1)[0]
            if module not in ALLOWED_AUTHORING_IMPORTS:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} imports "
                    f"'music_video_maker.{module}', not in {sorted(ALLOWED_AUTHORING_IMPORTS)}"
                )

    assert not violations, (
        "authoring/ may only reach into the render half through a small named "
        "allowlist (issue #54 design section 2) -- importing cli.py (or anything "
        "else not on the list) would blur the exact boundary this package exists "
        "to keep sharp:\n  " + "\n  ".join(violations)
    )
