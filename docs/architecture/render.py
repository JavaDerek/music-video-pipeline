"""Render the architecture boards to PNG.

    python docs/architecture/render.py            # every board
    python docs/architecture/render.py L3-1       # one, by file stem

The sources under canvas/ are in the format a Claude Design canvas takes (see
docs/ARCHITECTURE.md, "Presentation copies"). Each is a Design Component page:
an <x-dc> wrapper with a <helmet> for page styles, and a support.js the canvas
supplies. This script unwraps that -- helmet contents into <head>, the wrapper
and the component script dropped -- and screenshots the board's fixed-size
root. The boards are never edited here: change the source, re-run this.

Playwright is NOT a dependency of this package and must not become one for the
sake of six pictures. Install it wherever is convenient, outside the project's
own environment if you like:

    pip install playwright && playwright install chromium

BOARD_CHROMIUM, if set, is the path of a Chromium executable to use instead of
the one `playwright install` downloads.

Fonts come from Google Fonts at render time, so an offline render falls back
to system faces and looks slightly different. The PNGs are written at 2x for
crisp text; the width and height in canvas.json are the truth for each size.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANVAS = HERE / "canvas"
BOARDS = HERE / "boards"


def standalone(source: str) -> str:
    """Unwrap a Design Component page into a plain HTML document."""
    helmet = re.search(r"<helmet>([\s\S]*?)</helmet>", source)
    body = re.search(r"<x-dc>[\s\S]*?</helmet>([\s\S]*?)</x-dc>", source)
    title = re.search(r"<title>([\s\S]*?)</title>", source)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{title.group(1) if title else 'board'}</title>"
        f"{helmet.group(1) if helmet else ''}</head>"
        f"<body>{body.group(1) if body else ''}</body></html>"
    )


def main(argv: list[str]) -> int:
    from playwright.sync_api import sync_playwright

    only = argv[1] if len(argv) > 1 else None
    index = json.loads((CANVAS / "canvas.json").read_text(encoding="utf-8"))
    files = [f for f in index["order"] if not only or f.startswith(only)]
    BOARDS.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("BOARD_CHROMIUM") or None)
        try:
            for name in files:
                board = index["boards"][name]
                size = {"width": board["w"], "height": board["h"]}
                page = browser.new_page(viewport=size, device_scale_factor=2)
                page.set_content(
                    standalone((CANVAS / name).read_text(encoding="utf-8")),
                    wait_until="networkidle",
                )
                page.evaluate("() => document.fonts.ready")
                out = BOARDS / (name.removesuffix(".dc.html") + ".png")
                page.screenshot(path=str(out), clip={"x": 0, "y": 0, **size})
                sys.stdout.write(f"{name} -> {out} ({board['w']}x{board['h']} @2x)\n")
                page.close()
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
