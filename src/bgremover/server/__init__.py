"""Optional HTTP API. Install with ``pip install "bgremover-onnx[api]"``.

This subpackage is never imported by the core library.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
