"""Getting images in and out: decoding inputs, encoding results."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

#: Anything :func:`load_image` accepts.
ImageInput = bytes | bytearray | memoryview | str | os.PathLike | Image.Image | np.ndarray

InputKind = Literal["bytes", "path", "pil", "numpy"]

#: Formats that cannot carry an alpha channel, so a cutout has to be flattened.
_OPAQUE_FORMATS = frozenset({"JPEG", "JPG", "BMP", "PCX", "PPM", "EPS"})

_SUFFIX_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".gif": "GIF",
}


def input_kind(data: ImageInput) -> InputKind:
    """Classify an input so the caller's result can match its type."""
    if isinstance(data, Image.Image):
        return "pil"
    if isinstance(data, np.ndarray):
        return "numpy"
    if isinstance(data, bytes | bytearray | memoryview):
        return "bytes"
    if isinstance(data, str | os.PathLike):
        return "path"
    raise TypeError(f"unsupported image input type: {type(data).__name__}")


def _from_numpy(array: np.ndarray) -> Image.Image:
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}")
    if array.dtype != np.uint8:
        working = array.astype(np.float32)
        # Floating point arrays are taken as 0..1 unless they clearly are not.
        if working.size and working.max() <= 1.0:
            working = working * 255.0
        array = np.clip(working, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(f"expected 1, 3 or 4 channels, got {array.shape[2]}")
    return Image.fromarray(array)


def load_image(data: ImageInput) -> Image.Image:
    """Decode any supported input into a PIL image with EXIF rotation applied.

    Raises:
        ValueError: if the bytes or file are not a readable image.
        TypeError: if the input type is not supported.
    """
    kind = input_kind(data)

    if kind == "pil":
        assert isinstance(data, Image.Image)
        image = data
    elif kind == "numpy":
        assert isinstance(data, np.ndarray)
        image = _from_numpy(data)
    else:
        source: BytesIO | Path
        if kind == "bytes":
            source = BytesIO(bytes(data))  # type: ignore[arg-type]
        else:
            source = Path(os.fspath(data))  # type: ignore[arg-type]
            if not source.is_file():
                raise ValueError(f"no such image file: {source}")
        try:
            image = Image.open(source)
            image.load()
        except OSError as exc:
            # PIL's message for a buffer embeds an object repr (and its memory
            # address), which would end up in an HTTP error body.
            if kind == "bytes":
                raise ValueError(
                    "could not decode an image from the given bytes: "
                    "the data is not a supported image format"
                ) from exc
            raise ValueError(f"could not decode an image from {source}: {exc}") from exc

    rotated = ImageOps.exif_transpose(image)
    if rotated is None:
        return image
    # exif_transpose hands back a copy, which does not carry .format.
    rotated.format = rotated.format or image.format
    return rotated


def format_for_path(path: str | os.PathLike, default: str = "PNG") -> str:
    """Pick an output format from a filename suffix."""
    return _SUFFIX_FORMATS.get(Path(os.fspath(path)).suffix.lower(), default)


def flatten_if_opaque(image: Image.Image, image_format: str, color: str = "white") -> Image.Image:
    """Composite away the alpha channel for formats that cannot store it."""
    if image_format.upper() not in _OPAQUE_FORMATS or image.mode not in ("RGBA", "LA", "P"):
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, color)
    return Image.alpha_composite(background, rgba).convert("RGB")


def encode_image(
    image: Image.Image,
    image_format: str = "PNG",
    *,
    quality: int | None = None,
) -> bytes:
    """Serialise an image to bytes.

    Alpha is flattened onto white for formats that cannot carry it, so asking
    for a JPEG never fails on an RGBA cutout.
    """
    image_format = image_format.upper()
    prepared = flatten_if_opaque(image, image_format)

    options: dict[str, object] = {}
    if quality is not None and image_format in ("JPEG", "WEBP"):
        options["quality"] = int(quality)
    if image_format == "PNG":
        options["optimize"] = True

    buffer = BytesIO()
    prepared.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def to_numpy(image: Image.Image) -> np.ndarray:
    """RGBA/RGB image as a uint8 array."""
    return np.asarray(image)
