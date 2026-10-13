"""snipbg - remove image backgrounds with ONNX segmentation models.

    from snipbg import remove
    png_bytes = remove(open("photo.jpg", "rb").read())

Attributes are resolved lazily so that ``snipbg.models`` and
``snipbg.download`` stay importable without numpy, Pillow or onnxruntime
installed (the Docker model stage relies on this).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "DownloadError",
    "ModelSpec",
    "Remover",
    "__version__",
    "cache_dir",
    "ensure_model",
    "get_remover",
    "list_models",
    "remove",
]

if TYPE_CHECKING:
    from .download import DownloadError, cache_dir, ensure_model
    from .models import DEFAULT_MODEL, MODELS, ModelSpec, list_models
    from .session import Remover, get_remover, remove

_LAZY: dict[str, str] = {
    "DEFAULT_MODEL": "snipbg.models",
    "MODELS": "snipbg.models",
    "ModelSpec": "snipbg.models",
    "list_models": "snipbg.models",
    "DownloadError": "snipbg.download",
    "cache_dir": "snipbg.download",
    "ensure_model": "snipbg.download",
    "Remover": "snipbg.session",
    "get_remover": "snipbg.session",
    "remove": "snipbg.session",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
