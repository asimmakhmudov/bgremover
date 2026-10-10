"""Pure image math: tensor preparation, mask handling, compositing.

Nothing here touches ONNX, the network or the filesystem - it is all numpy and
PIL on values passed in, which keeps it trivially testable.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageColor, ImageFilter

#: Mask values are resized bilinearly: unlike LANCZOS it cannot overshoot and
#: ring, which on an alpha channel shows up as a halo.
_MASK_RESAMPLE = Image.Resampling.BILINEAR
_IMAGE_RESAMPLE = Image.Resampling.LANCZOS

Color = str | tuple[int, int, int] | tuple[int, int, int, int]


def preprocess(
    image: Image.Image,
    input_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    """Turn a PIL image into an NCHW float32 batch of one.

    The channel scaling matches the transform these checkpoints were trained
    with: pixels are divided by the image maximum rather than a flat 255, then
    standardised with ``mean``/``std``.
    """
    rgb = image.convert("RGB").resize(input_size, _IMAGE_RESAMPLE)
    array = np.asarray(rgb, dtype=np.float32)

    peak = float(array.max())
    array = array / peak if peak > 0 else np.zeros_like(array)

    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _as_2d(raw: np.ndarray) -> np.ndarray:
    """Reduce a model output of any common shape to a single 2-D map."""
    array = np.asarray(raw, dtype=np.float32)
    while array.ndim > 2:
        if array.shape[0] != 1 and array.ndim == 3:
            # (C, H, W) with several channels: the first one is the saliency map.
            array = array[0]
        else:
            array = array[0]
    if array.ndim != 2:
        raise ValueError(f"cannot read a 2-D mask out of shape {np.asarray(raw).shape}")
    return array


def postprocess(
    raw: np.ndarray,
    size: tuple[int, int],
    *,
    sigmoid: bool = False,
    normalize: bool | None = None,
) -> Image.Image:
    """Convert a raw model output into an 8-bit mask at ``size`` (w, h).

    ``sigmoid`` squashes graphs that emit logits. ``normalize`` rescales the map
    to use the full 0..255 range; it defaults to on for activated outputs (what
    the U²-Net and IS-Net exports expect) and off once a sigmoid has produced
    calibrated probabilities.
    """
    array = _as_2d(raw)

    if sigmoid:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -60.0, 60.0)))

    if normalize is None:
        normalize = not sigmoid
    if normalize:
        low = float(array.min())
        high = float(array.max())
        array = (array - low) / (high - low) if high > low else np.zeros_like(array)

    # A 2-D uint8 array already means mode "L"; passing mode= is deprecated.
    mask = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    if mask.size != size:
        mask = mask.resize(size, _MASK_RESAMPLE)
    return mask


def refine_mask(
    mask: Image.Image,
    *,
    foreground_threshold: int = 240,
    background_threshold: int = 15,
    erode_size: int = 0,
    blur_radius: float = 1.0,
) -> Image.Image:
    """Harden a mask's confident regions and smooth the band between them.

    Values at or above ``foreground_threshold`` become fully opaque and values
    at or below ``background_threshold`` fully transparent, so solid areas stop
    being faintly translucent. ``erode_size`` pulls the edge inwards by that
    many pixels to drop background fringing, and ``blur_radius`` feathers what
    is left.
    """
    if background_threshold >= foreground_threshold:
        raise ValueError("background_threshold must be below foreground_threshold")

    array = np.asarray(mask.convert("L"), dtype=np.float32)
    low = float(background_threshold)
    high = float(foreground_threshold)
    array = np.clip((array - low) / (high - low), 0.0, 1.0) * 255.0
    refined = Image.fromarray(array.round().astype(np.uint8))

    if erode_size > 0:
        # MinFilter needs an odd window; a size of n erodes by n // 2 pixels.
        window = erode_size * 2 + 1
        refined = refined.filter(ImageFilter.MinFilter(window))
    if blur_radius > 0:
        refined = refined.filter(ImageFilter.GaussianBlur(blur_radius))
    return refined


def cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Apply ``mask`` as the alpha channel of ``image``, returning RGBA."""
    rgba = image.convert("RGBA")
    alpha = mask.convert("L")
    if alpha.size != rgba.size:
        alpha = alpha.resize(rgba.size, _MASK_RESAMPLE)
    rgba.putalpha(alpha)
    return rgba


def apply_background(image: Image.Image, color: Color) -> Image.Image:
    """Composite an RGBA cutout onto a solid colour.

    A fully opaque colour yields an RGB image; a translucent one keeps RGBA.
    """
    rgba = image.convert("RGBA")
    fill = ImageColor.getrgb(color) if isinstance(color, str) else tuple(color)
    if len(fill) == 3:
        fill = (*fill, 255)
    background = Image.new("RGBA", rgba.size, fill)
    composed = Image.alpha_composite(background, rgba)
    return composed.convert("RGB") if fill[3] == 255 else composed


def trim(image: Image.Image, *, alpha_threshold: int = 0) -> Image.Image:
    """Crop away fully transparent borders.

    Pixels with alpha above ``alpha_threshold`` are kept. An image that is
    entirely transparent is returned unchanged.
    """
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    rows = np.where(alpha.max(axis=1) > alpha_threshold)[0]
    cols = np.where(alpha.max(axis=0) > alpha_threshold)[0]
    if rows.size == 0 or cols.size == 0:
        return image
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return image.crop(box)
