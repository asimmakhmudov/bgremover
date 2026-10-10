"""The inference engine: a lazily loaded ONNX session with a friendly API."""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
from PIL import Image

from .download import ensure_model
from .images import ImageInput, encode_image, input_kind, load_image, to_numpy
from .models import DEFAULT_MODEL, ModelSpec, get_spec
from .processing import Color, apply_background, cutout, postprocess, preprocess, refine_mask, trim

_REMOVER_CACHE: dict[tuple[Any, ...], Remover] = {}
_CACHE_LOCK = threading.Lock()


def _default_providers() -> list[str] | None:
    """Execution providers from ``BGREMOVER_PROVIDERS``, else onnxruntime's own."""
    raw = os.environ.get("BGREMOVER_PROVIDERS", "").strip()
    if not raw:
        return None
    return [name.strip() for name in raw.split(",") if name.strip()]


def _default_threads() -> int | None:
    raw = os.environ.get("BGREMOVER_NUM_THREADS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


class Remover:
    """Removes backgrounds with one model.

    The ONNX session is built on first use and then reused, so constructing a
    ``Remover`` is cheap and does not touch the network. Instances are safe to
    share between threads: session creation is locked and onnxruntime's ``Run``
    is itself thread-safe.
    """

    def __init__(
        self,
        model: str | ModelSpec | None = None,
        *,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.spec = model if isinstance(model, ModelSpec) else get_spec(model)
        self.providers = providers if providers is not None else _default_providers()
        self.num_threads = num_threads if num_threads is not None else _default_threads()
        self._session: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.spec.name

    @property
    def loaded(self) -> bool:
        """True once the ONNX session exists."""
        return self._session is not None

    def _build_session(self) -> Any:
        import onnxruntime as ort

        path = ensure_model(self.spec)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        if self.num_threads:
            options.intra_op_num_threads = self.num_threads
        return ort.InferenceSession(str(path), sess_options=options, providers=self.providers)

    @property
    def session(self) -> Any:
        """The ONNX session, created (and the weights fetched) on first access."""
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = self._build_session()
        return self._session

    def load(self) -> None:
        """Download the weights and build the session now rather than later."""
        self.session  # noqa: B018 - the property does the work

    def predict_mask(self, image: Image.Image) -> Image.Image:
        """Run the model and return a soft mask the size of ``image``."""
        tensor = preprocess(image, self.spec.input_size, self.spec.mean, self.spec.std)
        session = self.session
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        return postprocess(outputs[0], image.size, sigmoid=self.spec.sigmoid)

    def remove(
        self,
        data: ImageInput,
        *,
        only_mask: bool = False,
        refine: bool = False,
        foreground_threshold: int = 240,
        background_threshold: int = 15,
        erode_size: int = 0,
        blur_radius: float = 1.0,
        background: Color | None = None,
        trim_border: bool = False,
        output_format: str = "PNG",
        quality: int | None = None,
    ) -> bytes | Image.Image | np.ndarray:
        """Remove the background from one image.

        The result type follows the input type: bytes in, bytes out; a PIL image
        in, a PIL image out; a numpy array in, a numpy array out; a path in, a
        PIL image out.

        Args:
            only_mask: return the grayscale mask instead of the cutout.
            refine: harden and feather the mask before cutting out.
            background: composite the cutout onto this colour instead of
                leaving it transparent.
            trim_border: crop fully transparent edges away.
            output_format: encoding used when the input was bytes.
            quality: JPEG/WEBP quality when the input was bytes.
        """
        kind = input_kind(data)
        image = load_image(data)

        mask = self.predict_mask(image)
        if refine:
            mask = refine_mask(
                mask,
                foreground_threshold=foreground_threshold,
                background_threshold=background_threshold,
                erode_size=erode_size,
                blur_radius=blur_radius,
            )

        if only_mask:
            result: Image.Image = mask
        else:
            result = cutout(image, mask)
            # Trim first: once a background colour is composited in there is no
            # transparent border left to crop against.
            if trim_border:
                result = trim(result)
            if background is not None:
                result = apply_background(result, background)

        if kind == "bytes":
            return encode_image(result, output_format, quality=quality)
        if kind == "numpy":
            return to_numpy(result)
        return result

    def __repr__(self) -> str:
        state = "loaded" if self.loaded else "not loaded"
        return f"<Remover model={self.spec.name!r} ({state})>"


def get_remover(
    model: str | ModelSpec | None = None,
    *,
    providers: list[str] | None = None,
    num_threads: int | None = None,
) -> Remover:
    """Return a shared :class:`Remover`, building one per distinct setup.

    Reusing it is what keeps the model from being loaded again on every call.
    """
    spec = model if isinstance(model, ModelSpec) else get_spec(model)
    key = (spec.name, tuple(providers) if providers else None, num_threads)
    with _CACHE_LOCK:
        remover = _REMOVER_CACHE.get(key)
        if remover is None:
            remover = Remover(spec, providers=providers, num_threads=num_threads)
            _REMOVER_CACHE[key] = remover
        return remover


def clear_removers() -> None:
    """Drop every cached :class:`Remover`, releasing their ONNX sessions."""
    with _CACHE_LOCK:
        _REMOVER_CACHE.clear()


def remove(
    data: ImageInput,
    model: str | ModelSpec | None = DEFAULT_MODEL,
    **kwargs: Any,
) -> bytes | Image.Image | np.ndarray:
    """Remove the background from one image using a shared model session.

    Keyword arguments are passed through to :meth:`Remover.remove`.
    """
    return get_remover(model).remove(data, **kwargs)
