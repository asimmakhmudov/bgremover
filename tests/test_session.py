"""Tests for Remover: laziness, caching, options and the return-type contract."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image

from snipbg import remove
from snipbg.images import load_image
from snipbg.models import get_spec
from snipbg.session import Remover, clear_removers, get_remover


def _alpha(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA").getchannel("A"))


def test_remover_is_lazy_until_used():
    remover = Remover("u2netp")
    assert remover.loaded is False
    remover.load()
    assert remover.loaded is True


def test_remover_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        Remover("nope")


def test_the_mask_follows_the_fake_session_rule(red_blue_image):
    """Red is foreground, blue is background."""
    mask = get_remover("u2netp").remove(red_blue_image, only_mask=True)
    array = np.asarray(mask)
    assert array[:, :16].mean() > 200, "the red half should be opaque"
    assert array[:, 48:].mean() < 55, "the blue half should be transparent"


def test_cutout_alpha_matches_the_mask(red_blue_image):
    result = get_remover("u2netp").remove(red_blue_image)
    assert result.mode == "RGBA"
    alpha = _alpha(result)
    assert alpha[:, :16].mean() > 200
    assert alpha[:, 48:].mean() < 55


@pytest.mark.parametrize(
    ("make_input", "expected"),
    [
        (lambda img, png, path: png, bytes),
        (lambda img, png, path: img, Image.Image),
        (lambda img, png, path: np.asarray(img), np.ndarray),
        (lambda img, png, path: str(path), Image.Image),
        (lambda img, png, path: path, Image.Image),
    ],
    ids=["bytes", "pil", "numpy", "str-path", "path-object"],
)
def test_return_type_follows_input_type(
    make_input, expected, red_blue_image, red_blue_png, tmp_path
):
    path = tmp_path / "input.png"
    path.write_bytes(red_blue_png)

    result = get_remover("u2netp").remove(make_input(red_blue_image, red_blue_png, path))
    assert isinstance(result, expected)


def test_bytes_in_bytes_out_is_a_decodable_png(red_blue_png):
    payload = get_remover("u2netp").remove(red_blue_png)
    assert isinstance(payload, bytes)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert load_image(payload).size == (64, 32)


def test_numpy_in_numpy_out_keeps_the_shape(red_blue_image):
    array = np.asarray(red_blue_image)
    result = get_remover("u2netp").remove(array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 64, 4), "a cutout gains an alpha channel"


def test_output_format_applies_to_bytes_input(red_blue_png):
    payload = get_remover("u2netp").remove(red_blue_png, output_format="JPEG", quality=80)
    assert load_image(payload).format == "JPEG"


def test_background_colour_replaces_transparency(red_blue_image):
    result = get_remover("u2netp").remove(red_blue_image, background="white")
    assert result.mode == "RGB"
    assert result.getpixel((63, 16)) == (255, 255, 255)


def test_trim_crops_to_the_subject(red_blue_image):
    result = get_remover("u2netp").remove(red_blue_image, trim_border=True)
    assert result.width < 64, "the transparent blue half should be cropped away"


def test_trim_still_crops_when_a_background_is_applied(red_blue_image):
    """Trim runs before compositing, so a background colour does not disable it."""
    result = get_remover("u2netp").remove(red_blue_image, trim_border=True, background="white")
    assert result.mode == "RGB"
    assert result.width < 64


def test_refine_hardens_the_edges(red_blue_image):
    """Refining pushes the confident regions to the extremes, keeping a soft edge."""
    remover = get_remover("u2netp")
    plain = np.asarray(remover.remove(red_blue_image, only_mask=True))
    refined = np.asarray(remover.remove(red_blue_image, only_mask=True, refine=True, blur_radius=0))

    def extreme(array):
        return ((array == 0) | (array == 255)).mean()

    assert extreme(refined) >= extreme(plain)
    assert refined[:, :16].mean() == 255, "the solid half becomes fully opaque"
    assert refined[:, 48:].mean() == 0, "the background becomes fully transparent"


def test_sigmoid_models_take_the_logit_path(red_blue_image):
    """The fake session emits logits for a sigmoid spec; the mask must still be right."""
    mask = get_remover("birefnet-general-lite").remove(red_blue_image, only_mask=True)
    array = np.asarray(mask)
    assert array[:, :16].mean() > 200
    assert array[:, 48:].mean() < 55


def test_get_remover_caches_per_configuration():
    clear_removers()
    assert get_remover("u2netp") is get_remover("u2netp")
    assert get_remover("u2netp") is not get_remover("silueta")
    assert get_remover("u2netp") is not get_remover("u2netp", num_threads=2)


def test_get_remover_accepts_a_spec_object():
    assert get_remover(get_spec("u2netp")).model_name == "u2netp"


def test_clear_removers_drops_the_cache():
    first = get_remover("u2netp")
    clear_removers()
    assert get_remover("u2netp") is not first


def test_the_session_is_built_once_under_concurrent_use():
    remover = Remover("u2netp")
    sessions = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        sessions.append(remover.session)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(sessions) == 8
    assert all(session is sessions[0] for session in sessions), "one session for all threads"


def test_module_level_remove_uses_the_shared_remover(red_blue_png):
    assert isinstance(remove(red_blue_png, "u2netp"), bytes)


def test_repr_mentions_the_model_and_state():
    remover = Remover("u2netp")
    assert "u2netp" in repr(remover) and "not loaded" in repr(remover)
    remover.load()
    assert "not loaded" not in repr(remover)
