# Architecture boards

Presentation copies of the seven C4 diagrams in [../ARCHITECTURE.md](../ARCHITECTURE.md), one board
per diagram. **That file is the source of truth**; these are pictures of it, and they rot at the
rates its "Presentation copies" section states: component boards first, the container board next,
the context board last. When a board and the Markdown disagree, the Markdown wins.

```
architecture/
├── canvas/      the board sources, in the format a Claude Design canvas takes
│   ├── canvas.json            the canvas index: board frames, notes, order
│   └── *.dc.html              one Design Component page per board
├── boards/      rendered PNGs at 2x, one per board
└── render.py    renders boards/ from canvas/ with Playwright's Chromium
```

| Board | Source | Rendered |
|---|---|---|
| Level 1 · System context | `canvas/Main.dc.html` | `boards/Main.png` |
| Level 2 · Containers | `canvas/L2-Containers.dc.html` | `boards/L2-Containers.png` |
| Level 3.1 · Stages 1–2: planning the timeline | `canvas/L3-1-Stages-1-2.dc.html` | `boards/L3-1-Stages-1-2.png` |
| Level 3.2 · Stages 3–5: the render loop and assembly | `canvas/L3-2-Render-Loop.dc.html` | `boards/L3-2-Render-Loop.png` |
| Level 3.3 · The authoring layer | `canvas/L3-3-Authoring.dc.html` | `boards/L3-3-Authoring.png` |
| Level 3.4 · The monitor and its control half | `canvas/L3-4-Monitor-Control.dc.html` | `boards/L3-4-Monitor-Control.png` |
| Level 3.5 · Leaf instruments | `canvas/L3-5-Instruments.dc.html` | `boards/L3-5-Instruments.png` |

## Regenerating

```bash
python docs/architecture/render.py          # all seven
python docs/architecture/render.py L3-1     # one, by file stem
```

Playwright is **not** a dependency of this package, in any extra, and should not become one for
seven pictures. Install it wherever is convenient (a throwaway virtualenv is fine):

```bash
pip install playwright && playwright install chromium
```

`BOARD_CHROMIUM=/path/to/chromium` points the script at a Chromium you already have instead.
Fonts are fetched from Google Fonts at render time; offline, system faces stand in.

## Editing

Edit a source under `canvas/`, re-render, and commit both. Each source is a self-contained page:
the `<helmet>` holds page styles, the fixed-size root `div` is the board, and the diagram is inline
SVG with hand-laid coordinates. `render.py` strips the canvas wrapper before screenshotting; nothing
else is needed. The sources have not been published to a canvas; if they ever are, record the link
here, because from then on a change here has to be published there too.

The format and the house style are run-dmcp's (`docs/architecture/` there), copied rather than
shared; the renderer is a Python port of its `render.mjs`, because nothing in this repository runs
Node. `render.py` is outside the package and outside `tests/`, so the coverage floor does not see
it; `ruff check .` does.
