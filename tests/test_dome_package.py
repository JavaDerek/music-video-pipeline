"""The provenance guard on ``dome.py``'s render/verify path and the ``package``
subcommand. Offline: a render directory is laid out by hand, ffmpeg is never
run (the guard must refuse *before* the first command), and the ZIP is read
back with ``zipfile``."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from music_video_maker import dome
from music_video_maker.dome import (
    DISTRIBUTION_FILENAME,
    MANIFEST_FILENAME,
    VERIFICATION_FILENAME,
    DomeSpec,
    guard_provenance,
    package_render,
)
from music_video_maker.provenance import (
    PROVENANCE_FILENAME,
    FinalRenderGuardError,
    Provenance,
    ShotProvenance,
    load_provenance,
    save_provenance,
)


def _shot(shot_id: str, backend: str = "minimax_api", tier: str = "final") -> ShotProvenance:
    model = "MiniMax-H3" if backend == "minimax_api" else "h3-open-weights"
    return ShotProvenance(shot_id=shot_id, backend=backend, model=model, tier=tier,
                          resolution="2K" if tier == "final" else "480P")


def _provenance_file(path: Path, *shots: ShotProvenance, terms=None) -> Path:
    save_provenance(Provenance(shots=list(shots), terms=terms or {"url": "u"}), path)
    return path


def _render_dir(tmp_path: Path, *, ok: bool = True, stems: int = 7) -> Path:
    out = tmp_path / "render"
    (out / "audio").mkdir(parents=True)
    for name in (DISTRIBUTION_FILENAME, "preview_seat_view.mp4", "contact_sheet.png"):
        (out / name).write_bytes(b"media:" + name.encode())
    for name in list(dome.SURROUND_CHANNELS) + ["stereo"]:
        if stems <= 0:
            break
        (out / "audio" / f"{name}.wav").write_bytes(b"RIFF" + name.encode())
        stems -= 1
    (out / VERIFICATION_FILENAME).write_text(json.dumps({
        "ok": ok, "checks": [{"name": "frame_count", "ok": True, "detail": ""},
                             {"name": "sync", "ok": ok, "detail": ""}],
    }))
    (out / MANIFEST_FILENAME).write_text("{}")
    return out


# --------------------------------------------------------------------------- #
# guard_provenance
# --------------------------------------------------------------------------- #


def test_final_step_needs_a_provenance_file():
    with pytest.raises(FinalRenderGuardError, match="needs --provenance"):
        guard_provenance(None, tier="final", what="render")
    assert guard_provenance(None, tier="draft", what="render") is None
    with pytest.raises(ValueError):
        guard_provenance(None, tier="best", what="render")


def test_guard_merges_several_files_and_refuses_a_local_shot(tmp_path):
    good = _provenance_file(tmp_path / "a.json", _shot("a"))
    other = _provenance_file(tmp_path / "b.json", _shot("b"))
    merged = guard_provenance([good, other], tier="final", what="render")
    assert sorted(s.shot_id for s in merged.shots) == ["a", "b"]
    bad = _provenance_file(tmp_path / "c.json", _shot("c", backend="local_h3", tier="draft"))
    with pytest.raises(FinalRenderGuardError, match="local_h3"):
        guard_provenance([good, bad], tier="final", what="render")
    # a draft step tolerates a draft API shot but never a local one
    draft = _provenance_file(tmp_path / "d.json", _shot("d", tier="draft"))
    assert guard_provenance(draft, tier="draft", what="render") is not None
    with pytest.raises(FinalRenderGuardError):
        guard_provenance(bad, tier="draft", what="render")
    with pytest.raises(FinalRenderGuardError, match="draft'-tier"):
        guard_provenance(draft, tier="final", what="render")


# --------------------------------------------------------------------------- #
# render refuses before ffmpeg
# --------------------------------------------------------------------------- #


def _render_kwargs(tmp_path: Path, runner) -> dict:
    surround = tmp_path / "mix.flac"
    surround.write_bytes(b"fLaC")
    return dict(
        source=None, source_start=0.0, program_seconds=2.0, surround=surround,
        surround_offset_seconds=0.0, out_dir=tmp_path / "out",
        spec=DomeSpec(size=32, fps=10, leader_seconds=1.0, pop_seconds=0.5),
        window=None, credits_png=None, credits=None, retime="dup", runner=runner,
        base=tmp_path / "base.mkv",
    )


def test_final_render_refuses_a_local_shot_before_any_command(tmp_path):
    calls: list[list[str]] = []

    def runner(args):
        calls.append(list(args))
        raise AssertionError("ffmpeg must not run")

    bad = _provenance_file(tmp_path / "p.json", _shot("a", backend="local_draft", tier="draft"))
    with pytest.raises(FinalRenderGuardError, match="local_draft"):
        dome.render(**_render_kwargs(tmp_path, runner), provenance=[bad], tier="final")
    with pytest.raises(FinalRenderGuardError, match="needs --provenance"):
        dome.render(**_render_kwargs(tmp_path, runner), provenance=None, tier="final")
    assert calls == []


def test_render_parser_takes_provenance_and_tier(tmp_path):
    ns = dome.build_parser().parse_args([
        "render", "--base", "b.mkv", "--program-seconds", "2", "--surround", "s.flac",
        "--out-dir", "o", "--provenance", "a.json", "--provenance", "b.json", "--tier", "final",
    ])
    assert ns.provenance == [Path("a.json"), Path("b.json")] and ns.tier == "final"
    ns = dome.build_parser().parse_args(["verify", "--out-dir", "o", "--program-seconds", "2"])
    assert ns.provenance == [] and ns.tier == "draft"
    ns = dome.build_parser().parse_args(["package", "--out-dir", "o", "--zip", "z.zip",
                                         "--provenance", "p.json"])
    assert ns.tier == "final" and ns.extra == []


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #


def test_package_writes_the_deliverable_with_the_merged_provenance(tmp_path):
    out = _render_dir(tmp_path)
    one = _provenance_file(tmp_path / "one.json", _shot("chunk_0005"),
                           terms={"url": "https://platform.minimax.io/terms"})
    two = _provenance_file(tmp_path / "two.json", _shot("kropotkin_0"))
    readme = tmp_path / "LEEME.txt"
    readme.write_text("hola")
    extra = tmp_path / "viewing_2048.mp4"
    extra.write_bytes(b"view")
    zip_path = tmp_path / "pkg" / "Muestra.zip"

    result = package_render(out, zip_path, provenance=[one, two], tier="final",
                            readme=readme, extras=[extra])

    assert result == zip_path and zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(archive.namelist())
        infos = {i.filename: i for i in archive.infolist()}
    expected = {f"Muestra/{n}" for n in (DISTRIBUTION_FILENAME, "preview_seat_view.mp4",
                                          "contact_sheet.png", VERIFICATION_FILENAME,
                                          MANIFEST_FILENAME, PROVENANCE_FILENAME, "LEEME.txt",
                                          "viewing_2048.mp4")}
    expected |= {f"Muestra/audio/{n}.wav" for n in list(dome.SURROUND_CHANNELS) + ["stereo"]}
    assert set(names) == expected
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos.values())
    merged = load_provenance(out / PROVENANCE_FILENAME)
    assert sorted(s.shot_id for s in merged.shots) == ["chunk_0005", "kropotkin_0"]
    assert merged.terms["url"] == "https://platform.minimax.io/terms"


def test_package_refuses_a_failed_verification_or_a_local_shot(tmp_path):
    good = _provenance_file(tmp_path / "g.json", _shot("a"))
    failed = _render_dir(tmp_path / "f", ok=False)
    with pytest.raises(FinalRenderGuardError, match="failed check"):
        package_render(failed, tmp_path / "f.zip", provenance=[good])
    out = _render_dir(tmp_path)
    bad = _provenance_file(tmp_path / "b.json", _shot("b", backend="local_h3", tier="draft"))
    with pytest.raises(FinalRenderGuardError, match="local_h3"):
        package_render(out, tmp_path / "x.zip", provenance=[good, bad])
    assert not (tmp_path / "x.zip").exists()
    (out / VERIFICATION_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="verify the render first"):
        package_render(out, tmp_path / "x.zip", provenance=[good])


def test_package_refuses_missing_members(tmp_path):
    good = _provenance_file(tmp_path / "g.json", _shot("a"))
    out = _render_dir(tmp_path, stems=2)
    with pytest.raises(FileNotFoundError, match="stems"):
        package_render(out, tmp_path / "x.zip", provenance=[good])
    out = _render_dir(tmp_path / "second")
    (out / "contact_sheet.png").unlink()
    with pytest.raises(FileNotFoundError, match="contact_sheet.png"):
        package_render(out, tmp_path / "y.zip", provenance=[good])
    out = _render_dir(tmp_path / "third")
    with pytest.raises(FileNotFoundError, match="extra package member"):
        package_render(out, tmp_path / "z.zip", provenance=[good],
                       extras=[tmp_path / "nope.mp4"])


def test_package_cli_exit_codes(tmp_path, caplog):
    out = _render_dir(tmp_path)
    good = _provenance_file(tmp_path / "g.json", _shot("a"))
    bad = _provenance_file(tmp_path / "b.json", _shot("b", backend="local_h3", tier="draft"))
    assert dome.main(["package", "--out-dir", str(out), "--zip", str(tmp_path / "ok.zip"),
                      "--provenance", str(good)]) == 0
    assert dome.main(["package", "--out-dir", str(out), "--zip", str(tmp_path / "no.zip"),
                      "--provenance", str(bad)]) == 1
    assert "refused" in caplog.text
