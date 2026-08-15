"""Small sha256 helpers shared across the authoring package (issue #54).

**Hashes, not content** (design section 8): provenance and session
staleness record what an input *was*, never what it *said*. A committed
shot plan should not drag the full lyric sheet or a doc's whole text into
git -- and per #51, anything committed ships.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_text(Path(path).read_text(encoding="utf-8"))


__all__ = ["sha256_file", "sha256_text"]
