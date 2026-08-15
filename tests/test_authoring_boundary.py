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

Same trick ``tests/test_repo_assets.py`` plays for issue #51: the rule that
matters is the one a machine re-checks on every commit.

**Two considered deviations from the design's literal wording**, both
because the literal rule would fail against legitimate, already-tested,
pre-#54 code that has nothing to do with calling a model:

* ``subprocess`` is banned everywhere outside ``authoring/`` *except* the
  three modules that already, correctly, shell out for non-model reasons
  (ffmpeg, host-sleep prevention) -- see :data:`SUBPROCESS_ALLOWLIST`.
* ``authoring/`` is allowed to import ``logging_setup`` in addition to the
  design's named list (``config``, ``contracts``, ``shot_plan``,
  ``alignment``, ``slicing``, ``lyrics``) -- it is a side-effect-free
  stderr-logging helper, not a render-path concern, and duplicating it
  inside ``authoring/`` to avoid one more name on this list would be pure
  copy-paste for no safety this test would actually be buying.
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
