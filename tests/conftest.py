"""Shared fixtures.

The ONNX runtime is replaced by :class:`FakeSession` so the suite runs in a
second with no network and no weights on disk. Its rule is simple and easy to
assert against: a pixel is foreground when it is more red than blue.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bgremover.models import ModelSpec, get_spec
from bgremover.session import Remover, clear_removers


class _FakeInput:
    name = "input.1"


class FakeSession:
    """Stand-in for ``onnxruntime.InferenceSession``.

    Undoes the normalisation applied by ``preprocess`` and marks pixels whose
    red channel beats their blue channel, which gives a predictable mask for a
    test image built out of red and blue halves.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.calls = 0

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _output_names, feeds):
        self.calls += 1
        tensor = next(iter(feeds.values()))
        assert tensor.ndim == 4 and tensor.shape[1] == 3, tensor.shape

        mean = np.asarray(self.spec.mean, dtype=np.float32)
        std = np.asarray(self.spec.std, dtype=np.float32)
        channels = tensor[0] * std[:, None, None] + mean[:, None, None]
        mask = (channels[0] > channels[2]).astype(np.float32)

        if self.spec.sigmoid:
            # Mimic a logit-emitting graph so the sigmoid path gets exercised.
            mask = mask * 20.0 - 10.0
        return [mask[None, None, :, :]]


@pytest.fixture(autouse=True)
def fake_onnx(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every Remover a FakeSession instead of a real ONNX one.

    Tests marked ``model`` opt out: they are the ones that exercise real
    weights, and they are deselected by default.
    """
    clear_removers()
    if request.node.get_closest_marker("model") is None:
        monkeypatch.setattr(Remover, "_build_session", lambda self: FakeSession(self.spec))
    yield
    clear_removers()


@pytest.fixture(autouse=True)
def isolated_cache(request: pytest.FixtureRequest, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests away from the user's real model cache.

    Marked ``model`` tests keep the real cache so a 4 MB download is not
    repeated on every run.
    """
    if request.node.get_closest_marker("model") is None:
        monkeypatch.setenv("BGREMOVER_HOME", str(tmp_path / "cache"))


@pytest.fixture
def red_blue_image() -> Image.Image:
    """64x32: red on the left half (foreground), blue on the right."""
    image = Image.new("RGB", (64, 32), (0, 0, 255))
    image.paste(Image.new("RGB", (32, 32), (255, 0, 0)), (0, 0))
    return image


@pytest.fixture
def red_blue_png(red_blue_image: Image.Image) -> bytes:
    from bgremover.images import encode_image

    return encode_image(red_blue_image, "PNG")


@pytest.fixture
def spec() -> ModelSpec:
    return get_spec("u2netp")
