"""Tests for the pure image math."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bgremover.processing import (
    apply_background,
    cutout,
    postprocess,
    preprocess,
    refine_mask,
    trim,
)


def test_preprocess_shape_and_normalisation():
    image = Image.new("RGB", (10, 20), (255, 255, 255))
    tensor = preprocess(image, (320, 320), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    # White scales to 1.0, then (1 - 0.5) / 0.5 == 1.0.
    assert np.allclose(tensor, 1.0)


def test_preprocess_handles_an_all_black_image():
    tensor = preprocess(Image.new("RGB", (8, 8), (0, 0, 0)), (32, 32), (0.0, 0.0, 0.0), (1.0,) * 3)
    assert np.all(tensor == 0.0)


def test_preprocess_converts_non_rgb_modes():
    tensor = preprocess(Image.new("L", (8, 8), 128), (16, 16), (0.5,) * 3, (0.5,) * 3)
    assert tensor.shape == (1, 3, 16, 16)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (1, 8, 8), (8, 8)])
def test_postprocess_accepts_common_output_shapes(shape):
    raw = np.zeros(shape, dtype=np.float32)
    raw[..., 2:6, 2:6] = 1.0
    mask = postprocess(raw, (8, 8))
    assert mask.size == (8, 8)
    assert mask.mode == "L"
    assert mask.getpixel((4, 4)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_postprocess_normalises_a_low_contrast_output():
    raw = np.full((1, 1, 4, 4), 0.2, dtype=np.float32)
    raw[..., 0, 0] = 0.4
    mask = postprocess(raw, (4, 4))
    assert mask.getpixel((0, 0)) == 255


def test_postprocess_applies_sigmoid_without_overflow():
    raw = np.array([[[[-800.0, 800.0]]]], dtype=np.float32)
    mask = postprocess(raw, (2, 1), sigmoid=True)
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((1, 0)) == 255


def test_postprocess_of_a_flat_output_is_empty_not_noise():
    mask = postprocess(np.full((1, 1, 4, 4), 0.33, dtype=np.float32), (4, 4))
    assert mask.getextrema() == (0, 0)


def test_postprocess_resizes_to_the_requested_size():
    assert postprocess(np.zeros((1, 1, 320, 320), np.float32), (17, 9)).size == (17, 9)


def test_refine_mask_hardens_and_clears():
    mask = Image.new("L", (8, 8), 250)
    mask.paste(Image.new("L", (4, 8), 5), (0, 0))
    refined = refine_mask(mask, blur_radius=0)
    assert refined.getpixel((6, 4)) == 255
    assert refined.getpixel((1, 4)) == 0


def test_refine_mask_rejects_crossed_thresholds():
    with pytest.raises(ValueError, match="must be below"):
        refine_mask(Image.new("L", (4, 4)), foreground_threshold=10, background_threshold=200)


def test_refine_mask_erode_shrinks_the_edge():
    mask = Image.new("L", (16, 16), 0)
    mask.paste(Image.new("L", (8, 8), 255), (4, 4))
    eroded = refine_mask(mask, erode_size=2, blur_radius=0)
    assert np.asarray(eroded).sum() < np.asarray(mask).sum()


def test_cutout_uses_the_mask_as_alpha():
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    result = cutout(image, mask)
    assert result.mode == "RGBA"
    assert result.getpixel((1, 1)) == (10, 20, 30, 255)
    assert result.getpixel((0, 0))[3] == 0


def test_cutout_resizes_a_mismatched_mask():
    result = cutout(Image.new("RGB", (8, 8)), Image.new("L", (4, 4), 255))
    assert result.size == (8, 8)


def test_apply_background_flattens_to_rgb():
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    result = apply_background(image, "white")
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_apply_background_accepts_a_tuple_and_keeps_alpha():
    result = apply_background(Image.new("RGBA", (2, 2), (0, 0, 0, 0)), (1, 2, 3, 128))
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (1, 2, 3, 128)


def test_trim_crops_transparent_borders():
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (2, 3), (255, 0, 0, 255)), (4, 5))
    assert trim(image).size == (2, 3)


def test_trim_leaves_a_fully_transparent_image_alone():
    image = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    assert trim(image).size == (6, 6)
