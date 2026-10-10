"""Tests for input decoding and output encoding."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from bgremover.images import (
    encode_image,
    flatten_if_opaque,
    format_for_path,
    input_kind,
    load_image,
    to_numpy,
)


def test_input_kind_classifies_every_supported_type(tmp_path):
    assert input_kind(b"x") == "bytes"
    assert input_kind(bytearray(b"x")) == "bytes"
    assert input_kind(str(tmp_path / "a.png")) == "path"
    assert input_kind(tmp_path / "a.png") == "path"
    assert input_kind(Image.new("RGB", (1, 1))) == "pil"
    assert input_kind(np.zeros((1, 1, 3), np.uint8)) == "numpy"


def test_input_kind_rejects_nonsense():
    with pytest.raises(TypeError, match="unsupported image input"):
        input_kind(42)  # type: ignore[arg-type]


def test_load_image_from_bytes_and_path(tmp_path, red_blue_png):
    assert load_image(red_blue_png).size == (64, 32)

    path = tmp_path / "img.png"
    path.write_bytes(red_blue_png)
    assert load_image(path).size == (64, 32)
    assert load_image(str(path)).size == (64, 32)


def test_load_image_reports_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="no such image file"):
        load_image(tmp_path / "nope.png")


def test_load_image_reports_undecodable_bytes():
    with pytest.raises(ValueError, match="could not decode") as excinfo:
        load_image(b"definitely not an image")
    # The message reaches HTTP clients, so it must not carry a Python repr.
    assert "object at 0x" not in str(excinfo.value)


def test_load_image_from_numpy_variants():
    assert load_image(np.zeros((4, 5, 3), np.uint8)).size == (5, 4)
    assert load_image(np.zeros((4, 5, 4), np.uint8)).mode == "RGBA"
    assert load_image(np.zeros((4, 5), np.uint8)).mode == "L"
    assert load_image(np.zeros((4, 5, 1), np.uint8)).mode == "L"
    # Floats are read as 0..1 and scaled up.
    assert load_image(np.ones((2, 2, 3), np.float32)).getpixel((0, 0)) == (255, 255, 255)


def test_load_image_rejects_odd_arrays():
    with pytest.raises(ValueError, match="channels"):
        load_image(np.zeros((4, 4, 5), np.uint8))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        load_image(np.zeros((2, 2, 2, 2), np.uint8))


def test_load_image_applies_exif_rotation():
    # EXIF orientation 6 means "rotate 90 degrees": a 4x2 image becomes 2x4.
    base = Image.new("RGB", (4, 2), (255, 0, 0))
    exif = base.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    base.save(buffer, format="JPEG", exif=exif)

    assert load_image(buffer.getvalue()).size == (2, 4)


def test_encode_image_roundtrips_png_with_alpha():
    image = Image.new("RGBA", (4, 4), (1, 2, 3, 128))
    decoded = load_image(encode_image(image, "PNG"))
    assert decoded.mode == "RGBA"
    assert decoded.getpixel((0, 0)) == (1, 2, 3, 128)


def test_encode_image_flattens_alpha_for_jpeg():
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    decoded = load_image(encode_image(image, "JPEG", quality=95))
    assert decoded.mode == "RGB"
    # Fully transparent red over the white backing is white.
    assert decoded.getpixel((0, 0)) == (255, 255, 255)


def test_flatten_if_opaque_leaves_png_alone():
    image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    assert flatten_if_opaque(image, "PNG") is image


def test_format_for_path():
    assert format_for_path("a/b.JPG") == "JPEG"
    assert format_for_path("a/b.webp") == "WEBP"
    assert format_for_path("a/b.unknown") == "PNG"
    assert format_for_path("a/b.unknown", default="WEBP") == "WEBP"


def test_to_numpy_shape():
    assert to_numpy(Image.new("RGBA", (3, 2))).shape == (2, 3, 4)
