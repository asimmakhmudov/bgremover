"""End-to-end checks against real weights.

These are the only tests that touch the network. They are deselected unless you
ask for them:

    pytest -q -m model
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from snipbg import get_remover
from snipbg.download import is_cached, model_path, verify_model
from snipbg.models import get_spec

pytestmark = pytest.mark.model

MODEL = "u2netp"  # the 4 MB one, so the download stays cheap


@pytest.fixture(scope="module")
def remover():
    return get_remover(MODEL)


@pytest.fixture(scope="module")
def photo() -> Image.Image:
    """A synthetic subject: a bright disc on a flat background."""
    image = Image.new("RGB", (320, 320), (18, 60, 140))
    pixels = image.load()
    for y in range(320):
        for x in range(320):
            if (x - 160) ** 2 + (y - 160) ** 2 < 90**2:
                pixels[x, y] = (240, 220, 90)
    return image


def test_the_download_matches_its_recorded_digest():
    """A wrong sha256 in the registry would make every install fail; check one."""
    spec = get_spec(MODEL)
    get_remover(MODEL).load()
    assert is_cached(spec)
    assert model_path(spec).stat().st_size == spec.size_bytes
    assert verify_model(spec) is True


def test_real_inference_produces_a_plausible_mask(remover, photo):
    mask = remover.remove(photo, only_mask=True)
    array = np.asarray(mask)

    assert mask.size == photo.size
    assert array.max() > 200, "the model found no foreground at all"
    assert array.min() < 50, "the model marked everything as foreground"

    inside = array[150:170, 150:170].mean()
    corner = array[:20, :20].mean()
    assert inside > corner, "the subject should be more opaque than the corner"


def test_real_cutout_keeps_the_input_size_and_gains_alpha(remover, photo):
    result = remover.remove(photo)
    assert result.mode == "RGBA"
    assert result.size == photo.size


def test_the_graph_output_matches_the_spec(remover, photo):
    """Guards the sigmoid flag: a wrong one silently ruins every mask."""
    from snipbg.processing import preprocess

    spec = get_spec(MODEL)
    session = remover.session
    tensor = preprocess(photo, spec.input_size, spec.mean, spec.std)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]

    looks_activated = float(raw.min()) >= -1e-3 and float(raw.max()) <= 1.0 + 1e-3
    assert looks_activated == (not spec.sigmoid), (
        f"{MODEL} emits values in [{raw.min():.3f}, {raw.max():.3f}] but "
        f"sigmoid={spec.sigmoid} in the registry"
    )
