"""Guards on what gets committed, because this repo is meant to be public (#51).

A rule in CLAUDE.md only works if someone reads it at the moment it matters.
Both open-source blockers found in the 2026-08-12 audit were introduced the
same day, in changes nobody thought of as risky: a 232 KB model file and three
PNGs added as regression fixtures. Neither author (me) paused to ask whether
they were redistributable or whose face was in them.

So the two mechanical questions are asked by a test instead:

* every committed binary is listed with its provenance and licence, and
* nothing depicting a real person arrives without that being a deliberate,
  recorded decision.

The judgement calls -- whether a licence is *acceptable*, whether a person has
*consented* -- are still human. This just makes them impossible to skip
silently.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

BINARY_SUFFIXES = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
        ".mp4", ".mov", ".webm", ".wav", ".mp3", ".flac", ".ogg",
        ".onnx", ".pt", ".pth", ".safetensors", ".ckpt", ".bin",
        ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".zip", ".tar", ".gz",
    }
)
"""Extensions that carry redistribution and likeness risk. Deliberately broad:
the cost of listing one extra asset is one line, the cost of missing one is
publishing something that was not ours to publish."""


class Asset:
    """One committed binary and why it is allowed to be here."""

    def __init__(
        self,
        source: str,
        licence: str,
        *,
        depicts_real_person: bool = False,
        consent: str | None = None,
    ) -> None:
        self.source = source
        self.licence = licence
        self.depicts_real_person = depicts_real_person
        self.consent = consent
        if depicts_real_person and not consent:
            raise ValueError(
                f"Asset(source={source!r}) depicts a real person but has no recorded "
                "consent -- that is exactly the silent-skip this manifest exists to prevent."
            )


ASSET_MANIFEST: dict[str, Asset] = {
    "models/face_detection_yunet_2023mar.onnx": Asset(
        source="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        licence=(
            "MIT (c) 2020 Shiqi Yu -- verified 2026-08-14 against the model directory's "
            "own LICENSE file (issue #51 #3); notice text carried alongside the weights "
            "at models/face_detection_yunet_2023mar.onnx.LICENSE"
        ),
    ),
    # Issue #47's regressions -- H3 output proving the seed-frame face gate
    # still refuses a faceless seed and a spurious detection. The two that
    # depict a person were originally Dianne (the lead cast member), replaced
    # 2026-08-14 with frames of Derek (the project's author) from the `storms`
    # render, because publishing synthetic imagery of an identifiable person
    # is their decision and Dianne had not been asked (#51 §2). The tests that
    # use these already skip when they are absent, so removing them costs
    # coverage, not a red suite.
    "tests/fixtures/seed_frames/seed_faceless_chunk21.png": Asset(
        source="rendered by this pipeline (storms, chunk 21 last frame)",
        licence="project-owned output",
        depicts_real_person=True,
        consent="Derek Ferguson, project author, 2026-08-14 -- his own likeness",
    ),
    "tests/fixtures/seed_frames/seed_frontal_chunk15.png": Asset(
        source="rendered by this pipeline (storms, chunk 15 first frame)",
        licence="project-owned output",
        depicts_real_person=True,
        consent="Derek Ferguson, project author, 2026-08-14 -- his own likeness",
    ),
    "tests/fixtures/seed_frames/seed_spurious_chunk10.png": Asset(
        source="rendered by this pipeline (storms, chunk 10 first frame)",
        licence="project-owned output",
        # No person in shot -- a pair of cymbals in a doorway, engraved with
        # Derek's name as a prop label. Kept for the spurious-detection
        # regression, not a likeness question.
    ),
    # README showcase still (issue #51 follow-up): the closing frame of storms
    # chunk 26, used as the clickable thumbnail linking to the YouTube demo
    # (youtu.be/ROjuCF_t1e4). Same source render and same subject as the
    # seed-frame fixtures above.
    "docs/images/storms-still.jpg": Asset(
        source="rendered by this pipeline (storms, chunk 26 last frame)",
        licence="project-owned output",
        depicts_real_person=True,
        consent="Derek Ferguson, project author, 2026-08-14 -- his own likeness",
    ),
}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available or not a git checkout")
    return [p for p in out.stdout.decode().split("\0") if p]


def test_every_committed_binary_is_declared_with_its_provenance():
    """A new binary asset fails this test until someone writes down where it
    came from and under what licence.

    That is the whole point: the failure lands on the person adding the file,
    at the moment they are adding it, rather than on whoever prepares the
    repository for release months later."""
    tracked = _tracked_files()
    binaries = {p for p in tracked if Path(p).suffix.lower() in BINARY_SUFFIXES}

    undeclared = sorted(binaries - set(ASSET_MANIFEST))
    assert not undeclared, (
        "Undeclared binary asset(s) committed:\n  "
        + "\n  ".join(undeclared)
        + "\n\nThis repo is intended to be open sourced (#51). Before committing a "
        "binary, confirm it is ours to redistribute and add it to ASSET_MANIFEST "
        "with its source and licence. If it depicts a real person, set "
        "depicts_real_person=True and read that test's message."
    )


def test_the_manifest_has_no_entries_for_files_that_are_gone():
    """A stale allowlist is a hole. If an asset is removed, its entry goes too,
    or the next file with that path is silently pre-approved."""
    tracked = set(_tracked_files())
    orphaned = sorted(path for path in ASSET_MANIFEST if path not in tracked)
    assert not orphaned, (
        "ASSET_MANIFEST lists file(s) that are no longer committed:\n  "
        + "\n  ".join(orphaned)
        + "\n\nRemove the entries so the allowlist cannot pre-approve a future file "
        "at the same path."
    )


def test_assets_depicting_real_people_are_accounted_for():
    """Not a prohibition -- a receipt.

    Three such fixtures exist today and are deliberate (#47's regressions,
    plus the README showcase still). This test exists so a *fourth* cannot
    arrive without someone saying so out loud, because publishing synthetic
    imagery of an identifiable person is that person's decision and nobody
    else's -- ``Asset.__init__`` already refuses one with no recorded
    ``consent``, and this is the matching check on the set as a whole."""
    depicting = sorted(p for p, a in ASSET_MANIFEST.items() if a.depicts_real_person)

    assert depicting == [
        "docs/images/storms-still.jpg",
        "tests/fixtures/seed_frames/seed_faceless_chunk21.png",
        "tests/fixtures/seed_frames/seed_frontal_chunk15.png",
    ], (
        "The set of committed assets depicting a real person has changed.\n\n"
        "Publishing synthetic imagery of an identifiable person is their decision "
        "(#51). If you are adding one: confirm consent, then update this list. If "
        "you are removing one: good -- update this list and check whether #51 can "
        "close."
    )


def test_dotenv_is_ignored():
    """The credential story is currently clean; this keeps it that way.

    Checked rather than assumed, because a .gitignore rewrite is exactly the
    sort of change that looks harmless."""
    result = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=REPO_ROOT, capture_output=True, timeout=30
    )
    assert result.returncode == 0, (
        ".env is no longer gitignored -- restore it before anything else. "
        "Telethon session files and API keys live there."
    )
