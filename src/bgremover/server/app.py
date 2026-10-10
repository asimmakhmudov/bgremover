"""FastAPI application exposing background removal over HTTP.

The route names, the default port and the ``{"error": ...}`` error shape are
kept exactly as the original Node.js API had them, because existing clients
depend on them.
"""

from __future__ import annotations

import functools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import anyio.to_thread
from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..images import load_image
from ..models import MODELS
from ..session import get_remover
from .config import Settings, get_settings
from .fetch import FetchError, fetch_image

_READ_CHUNK = 64 * 1024

_FORMAT_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


class ApiError(Exception):
    """A client-visible failure carrying its HTTP status."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class UrlRequest(BaseModel):
    """Body of ``POST /remove-background-url``.

    ``imageUrl`` is the field the old Node.js API used; ``image_url`` is
    accepted too so newer clients can stay snake_case.
    """

    imageUrl: str | None = None  # noqa: N815 - the legacy client sends this name
    image_url: str | None = None
    model: str | None = None
    only_mask: bool = False
    refine: bool = False
    background: str | None = None
    trim: bool = False
    format: str | None = None
    quality: int | None = Field(default=None, ge=1, le=100)

    @property
    def url(self) -> str:
        target = self.imageUrl or self.image_url
        if not target or not target.strip():
            raise ApiError("No image URL provided")
        return target.strip()


def _resolve_model(name: str | None, settings: Settings) -> str:
    """Validate a requested model against the allow-list."""
    if not name:
        return settings.default_model
    candidate = name.strip()
    if candidate not in settings.allowed_models:
        allowed = ", ".join(settings.allowed_models)
        if candidate in MODELS:
            raise ApiError(f"Model '{candidate}' is not enabled on this server. Allowed: {allowed}")
        raise ApiError(f"Unknown model '{candidate}'. Allowed: {allowed}")
    return candidate


def _resolve_format(name: str | None) -> tuple[str, str]:
    """Return ``(format, media type)`` for a requested output format."""
    image_format = (name or "PNG").upper().replace("JPG", "JPEG")
    media_type = _FORMAT_MEDIA_TYPES.get(image_format)
    if media_type is None:
        allowed = ", ".join(sorted(_FORMAT_MEDIA_TYPES))
        raise ApiError(f"Unsupported output format '{image_format}'. Allowed: {allowed}")
    return image_format, media_type


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing anything over ``limit`` bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ApiError(
                f"File too large. Maximum size is {limit // (1024 * 1024)}MB.", status_code=413
            )
        chunks.append(chunk)
    if total == 0:
        raise ApiError("The uploaded file is empty")
    return b"".join(chunks)


def _run_removal(data: bytes, model: str, options: dict[str, Any]) -> bytes:
    """Blocking inference. Always called in a worker thread."""
    image = load_image(data)
    remover = get_remover(model)
    result = remover.remove(image, **options)
    from ..images import encode_image

    return encode_image(result, options.get("output_format", "PNG"), quality=options.get("quality"))


async def _process(
    request: Request,
    data: bytes,
    *,
    model: str,
    only_mask: bool,
    refine: bool,
    background: str | None,
    trim: bool,
    image_format: str,
    quality: int | None,
) -> bytes:
    """Validate the image and run inference in the bounded worker pool."""
    options = {
        "only_mask": only_mask,
        "refine": refine,
        "background": background,
        "trim_border": trim,
        "output_format": image_format,
        "quality": quality,
    }
    limiter: anyio.CapacityLimiter = request.app.state.limiter
    worker = functools.partial(_run_removal, data, model, options)
    try:
        return await anyio.to_thread.run_sync(worker, limiter=limiter)
    except ValueError as exc:
        # A bad upload, an unreadable image, or a bad option value.
        raise ApiError(str(exc)) from exc


def _disposition(image_format: str) -> dict[str, str]:
    suffix = "jpg" if image_format == "JPEG" else image_format.lower()
    return {"Content-Disposition": f'inline; filename="nobg.{suffix}"'}


router = APIRouter()


def _settings_of(request: Request) -> Settings:
    """The settings this app was built with."""
    return request.app.state.settings


@router.get("/")
async def index(request: Request) -> dict[str, Any]:
    """API description, in the shape the old Node.js API used."""
    settings = _settings_of(request)
    return {
        "name": "Background Remover API",
        "version": __version__,
        "engine": "bgremover",
        "models": list(settings.allowed_models),
        "default_model": settings.default_model,
        "endpoints": {
            "POST /remove-background": "Remove background from uploaded image file",
            "POST /remove-background-url": "Remove background from image URL",
            "GET /models": "List the models this server will serve",
            "GET /health": "Health check endpoint",
        },
        "usage": {
            "file_upload": 'Send image as multipart/form-data with field name "image"',
            "url_processing": 'Send JSON with "imageUrl" field containing image URL',
        },
    }


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "OK", "message": "Background Remover API is running"}


@router.get("/models")
async def list_served_models(request: Request) -> dict[str, Any]:
    """The models this server will accept, with their details."""
    settings = _settings_of(request)
    return {
        "default": settings.default_model,
        "models": [
            {
                "name": name,
                "description": MODELS[name].description,
                "input_size": list(MODELS[name].input_size),
                "size_bytes": MODELS[name].size_bytes,
                "license": MODELS[name].license,
            }
            for name in settings.allowed_models
        ],
    }


@router.post("/remove-background")
async def remove_background(
    request: Request,
    # B008 is the FastAPI idiom for declaring a required upload.
    image: UploadFile = File(..., description="The image to process"),  # noqa: B008
    model: str | None = Form(default=None),
    only_mask: bool = Form(default=False),
    refine: bool = Form(default=False),
    background: str | None = Form(default=None),
    trim: bool = Form(default=False),
    format: str | None = Form(default=None),
    quality: int | None = Form(default=None),
) -> Response:
    """Remove the background from an uploaded file."""
    settings = _settings_of(request)
    chosen = _resolve_model(model, settings)
    image_format, media_type = _resolve_format(format)
    data = await _read_upload(image, settings.max_upload_bytes)

    payload = await _process(
        request,
        data,
        model=chosen,
        only_mask=only_mask,
        refine=refine,
        background=background,
        trim=trim,
        image_format=image_format,
        quality=quality,
    )
    return Response(content=payload, media_type=media_type, headers=_disposition(image_format))


@router.post("/remove-background-url")
async def remove_background_url(request: Request, body: UrlRequest) -> Response:
    """Remove the background from an image named by URL."""
    settings = _settings_of(request)
    url = body.url
    chosen = _resolve_model(body.model, settings)
    image_format, media_type = _resolve_format(body.format)

    fetcher = functools.partial(
        fetch_image,
        url,
        max_bytes=settings.max_upload_bytes,
        timeout=settings.fetch_timeout,
        max_redirects=settings.max_redirects,
        allow_private=settings.allow_private_urls,
    )
    try:
        data = await anyio.to_thread.run_sync(fetcher)
    except FetchError as exc:
        raise ApiError(str(exc)) from exc

    payload = await _process(
        request,
        data,
        model=chosen,
        only_mask=body.only_mask,
        refine=body.refine,
        background=body.background,
        trim=body.trim,
        image_format=image_format,
        quality=body.quality,
    )
    return Response(content=payload, media_type=media_type, headers=_disposition(image_format))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    active = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One limiter for the whole process: inference is CPU bound, so letting
        # every request have a thread would just thrash.
        app.state.limiter = anyio.CapacityLimiter(active.max_concurrency)
        if active.preload:
            await anyio.to_thread.run_sync(get_remover(active.default_model).load)
        yield

    app = FastAPI(
        title="Background Remover API",
        description="Remove image backgrounds with ONNX segmentation models.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if errors and any(error.get("loc", ("", ""))[-1] == "image" for error in errors):
            message = "No image file provided"
        else:
            first = errors[0] if errors else {}
            location = ".".join(str(part) for part in first.get("loc", ())[1:])
            message = f"{location}: {first.get('msg', 'invalid request')}".lstrip(": ")
        return JSONResponse(status_code=400, content={"error": message})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    app.state.settings = active
    # Also set outside the lifespan so the app works when it is mounted or
    # exercised without startup events having run.
    app.state.limiter = anyio.CapacityLimiter(active.max_concurrency)
    app.include_router(router)
    return app


app = create_app()
