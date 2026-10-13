"""Tests for the command line interface."""

from __future__ import annotations

import json

import pytest

from snipbg.cli import main
from snipbg.images import load_image


@pytest.fixture
def image_file(tmp_path, red_blue_png):
    path = tmp_path / "photo.png"
    path.write_bytes(red_blue_png)
    return path


def test_models_lists_the_registry(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "u2netp" in out and "MODEL" in out and "(default)" in out


def test_models_json(capsys):
    assert main(["models", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert {entry["name"] for entry in entries} >= {"u2net", "u2netp"}
    assert all(len(entry["sha256"]) == 64 for entry in entries)


def test_remove_writes_a_default_output_next_to_the_input(image_file, capsys):
    assert main(["remove", str(image_file), "-m", "u2netp"]) == 0
    produced = image_file.with_name("photo_nobg.png")
    assert produced.is_file()
    assert load_image(produced).mode == "RGBA"


def test_remove_honours_an_explicit_output(image_file, tmp_path):
    target = tmp_path / "out" / "cut.png"
    assert main(["remove", str(image_file), "-o", str(target), "-m", "u2netp"]) == 0
    assert target.is_file()


def test_remove_writes_into_a_directory_for_many_inputs(image_file, tmp_path, red_blue_png):
    second = tmp_path / "second.png"
    second.write_bytes(red_blue_png)
    outdir = tmp_path / "cutouts"

    assert main(["remove", str(image_file), str(second), "-o", str(outdir), "-m", "u2netp"]) == 0
    assert (outdir / "photo_nobg.png").is_file()
    assert (outdir / "second_nobg.png").is_file()


def test_remove_picks_the_format_from_the_output_suffix(image_file, tmp_path):
    target = tmp_path / "cut.jpg"
    assert main(["remove", str(image_file), "-o", str(target), "-m", "u2netp"]) == 0
    assert load_image(target).format == "JPEG"


def test_remove_only_mask(image_file, tmp_path):
    target = tmp_path / "mask.png"
    assert main(["remove", str(image_file), "-o", str(target), "--only-mask", "-m", "u2netp"]) == 0
    assert load_image(target).mode == "L"


def test_remove_with_a_background_colour(image_file, tmp_path):
    target = tmp_path / "flat.png"
    argv = ["remove", str(image_file), "-o", str(target), "--bg", "white", "-m", "u2netp"]
    assert main(argv) == 0
    assert load_image(target).mode == "RGB"


def test_remove_reports_a_missing_input_and_exits_nonzero(tmp_path, capsys):
    assert main(["remove", str(tmp_path / "ghost.png"), "-m", "u2netp"]) == 1
    assert "error:" in capsys.readouterr().err


def test_remove_continues_past_a_failure_in_a_batch(image_file, tmp_path, capsys):
    missing = tmp_path / "ghost.png"
    outdir = tmp_path / "out"
    assert main(["remove", str(missing), str(image_file), "-o", str(outdir), "-m", "u2netp"]) == 1
    assert (outdir / "photo_nobg.png").is_file(), "the good input must still be processed"


def test_remove_reads_stdin_and_writes_stdout(monkeypatch, capsysbinary, red_blue_png):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(red_blue_png)))
    assert main(["remove", "-", "-m", "u2netp"]) == 0
    assert load_image(capsysbinary.readouterr().out).size == (64, 32)


def test_an_unknown_model_is_reported(capsys):
    assert main(["remove", "whatever.png", "-m", "nope"]) == 1
    assert "unknown model" in capsys.readouterr().err


def test_download_check_reports_an_uncached_model(capsys):
    assert main(["download", "--check", "u2netp"]) == 0
    assert "not cached" in capsys.readouterr().out


def test_download_rejects_an_unknown_model(capsys):
    assert main(["download", "no-such-model"]) == 1
    assert "unknown model" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "snipbg" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
