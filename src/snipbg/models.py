"""Registry of pretrained segmentation models.

Only the standard library may be imported here: the Docker model stage installs
no third-party packages and runs this module together with :mod:`download`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Weight files live on rembg's release page. No rembg code is used; this is
#: only where the ONNX files are hosted. Point ``SNIPBG_MODEL_BASE_URL`` at a
#: mirror to download from somewhere else.
DEFAULT_MODEL_BASE_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"

#: Model used when the caller does not name one.
DEFAULT_MODEL = "u2net"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def model_base_url() -> str:
    """Base URL weights are downloaded from, honouring the env override."""
    return os.environ.get("SNIPBG_MODEL_BASE_URL", DEFAULT_MODEL_BASE_URL).rstrip("/")


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to fetch a model and run it.

    ``mean``/``std`` normalise the RGB input after it is scaled to ``0..1``.
    ``sigmoid`` is ``True`` for graphs that emit raw logits; U²-Net and IS-Net
    exports already apply the activation themselves.
    """

    name: str
    filename: str
    sha256: str
    size_bytes: int
    input_size: tuple[int, int]
    mean: tuple[float, float, float] = _IMAGENET_MEAN
    std: tuple[float, float, float] = _IMAGENET_STD
    sigmoid: bool = False
    description: str = ""
    license: str = ""

    @property
    def url(self) -> str:
        """Absolute download URL for this model's weights."""
        return f"{model_base_url()}/{self.filename}"

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


MODELS: dict[str, ModelSpec] = {
    "u2net": ModelSpec(
        name="u2net",
        filename="u2net.onnx",
        sha256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        size_bytes=175997641,
        input_size=(320, 320),
        description="U²-Net, general purpose. Good default.",
        license="Apache-2.0",
    ),
    "u2netp": ModelSpec(
        name="u2netp",
        filename="u2netp.onnx",
        sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        size_bytes=4574861,
        input_size=(320, 320),
        description="Small U²-Net. Fast, lower quality, 4 MB.",
        license="Apache-2.0",
    ),
    "u2net_human_seg": ModelSpec(
        name="u2net_human_seg",
        filename="u2net_human_seg.onnx",
        sha256="01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
        size_bytes=175997641,
        input_size=(320, 320),
        description="U²-Net fine-tuned for people.",
        license="Apache-2.0",
    ),
    "silueta": ModelSpec(
        name="silueta",
        filename="silueta.onnx",
        sha256="75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
        size_bytes=44173029,
        input_size=(320, 320),
        description="U²-Net pruned to 44 MB, near u2net quality.",
        license="Apache-2.0",
    ),
    "isnet-general-use": ModelSpec(
        name="isnet-general-use",
        filename="isnet-general-use.onnx",
        sha256="60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        size_bytes=178648008,
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        description="IS-Net (DIS). Sharper edges than u2net, slower.",
        license="Apache-2.0",
    ),
    "isnet-anime": ModelSpec(
        name="isnet-anime",
        filename="isnet-anime.onnx",
        sha256="f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
        size_bytes=176069933,
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        description="IS-Net tuned for anime and illustration.",
        license="Apache-2.0",
    ),
    "birefnet-general-lite": ModelSpec(
        name="birefnet-general-lite",
        filename="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        sha256="5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        size_bytes=224005088,
        input_size=(1024, 1024),
        sigmoid=True,
        description="BiRefNet tiny backbone, 224 MB. Needs ~8 GB RAM.",
        license="MIT",
    ),
    "birefnet-general": ModelSpec(
        name="birefnet-general",
        filename="BiRefNet-general-epoch_244.onnx",
        sha256="58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        size_bytes=972666916,
        input_size=(1024, 1024),
        sigmoid=True,
        description="BiRefNet full, best quality. Needs a GPU or 16 GB+ RAM.",
        license="MIT",
    ),
}


def list_models() -> list[ModelSpec]:
    """All known models, in registry order."""
    return list(MODELS.values())


def get_spec(name: str | None = None) -> ModelSpec:
    """Look up a model by name.

    Raises:
        ValueError: if the name is not in the registry.
    """
    key = name or DEFAULT_MODEL
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model {key!r}; available models: {known}") from None
