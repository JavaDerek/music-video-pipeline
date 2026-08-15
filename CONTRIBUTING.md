# Contributing

## Setup

```bash
pip install -e ".[dev]"
```

That's enough to run the full test suite. The other extras (`align`, `faces`)
pull heavy dependencies (torch, OpenCV) that only matter on a machine actually
running alignment or the face-detection gate — see the README's
[Install](README.md#install) section for what each is for. Neither is needed to
build or test the code.

## Running tests

```bash
ruff check .
pytest -m "not integration"
```

**Tests run fully offline: no GPU, no network, no live ComfyUI, no real
`stable-ts`/OpenCV/Telethon.** A mock ComfyUI harness (`tests/harness/`) fakes
every HTTP/WebSocket call the orchestrator makes; forced alignment and face
detection are exercised through injected fakes, never the real models. If a
test you're adding needs something that isn't mocked, that's a sign the
interface under test needs a seam, not that the test needs a real service.

`-m "not integration"` deselects the two tests that *do* want a real binary
(`ffmpeg`, the `claude` CLI) — both self-skip anyway if that binary is
missing, but deselecting keeps CI's intent explicit. `pyproject.toml` enforces
80% branch coverage (`--cov-fail-under=80`); a PR that drops below it fails
CI, not just review.

CI (`.github/workflows/ci.yml`) runs exactly the two commands above, on the
core install plus `[dev]` only, across Python 3.10–3.13 — deliberately the
same as a fresh clone, so a green run there means a stranger can actually
reproduce it.

## Two invariants worth knowing before you touch the render path

- **Lyrics are immutable truth.** Forced alignment only, never ASR
  transcription of the vocals — see the README's "Non-negotiable invariants"
  for the full list, but this one shapes almost everything in `Stage 1-2`.
- **No LLM in the render path.** `prompting.py` is pure deterministic string
  composition; nothing under `music_video_maker/` (outside `authoring/`) ever
  calls a model. The one sanctioned exception is `music_video_maker/authoring/`
  (issue #54), which calls the `claude` CLI by hand, once per song, and writes
  a file a human reviews and commits — never at render time.
  `tests/test_authoring_boundary.py` enforces the boundary mechanically
  (parses every module's AST), so a PR that blurs this line fails a test, not
  just a review comment.

## Reading the issue history

Issues #1–#50+ are the actual work log and are cited throughout the README
and `CLAUDE.md` as evidence for specific design decisions. **Read the
comments, not just the body** — several issues were re-scoped or corrected
after real findings (a GPU render, a face-detection false positive, a
scoping conversation), and the comment is the part that's still accurate.
An issue body alone can be describing a plan that was later revised.

## Style

Match what's already there: sparse comments that explain a non-obvious *why*
(a measured number, a past incident, a subtle invariant), not what the code
already says by being well-named. `ruff check .` must pass; there's no
separate formatter config to fight.
