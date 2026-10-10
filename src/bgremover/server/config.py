"""Server settings, all of them environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from ..models import DEFAULT_MODEL, MODELS

#: Models the API will serve unless ``BGREMOVER_MODELS`` says otherwise.
#: Deliberately excludes anything above ~250 MB: those need far more RAM than a
#: small container has and would be downloaded on the first request.
DEFAULT_ALLOWED_MODELS = (
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
)

_TRUE = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in _TRUE if raw else default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Everything the API reads from the environment."""

    #: Hard cap on an uploaded or fetched image, in bytes.
    max_upload_bytes: int = 10 * 1024 * 1024
    #: Models clients may ask for.
    allowed_models: tuple[str, ...] = DEFAULT_ALLOWED_MODELS
    #: Model used when a request does not name one.
    default_model: str = DEFAULT_MODEL
    #: How many inferences may run at once.
    max_concurrency: int = 2
    #: Seconds allowed for fetching a remote image.
    fetch_timeout: float = 15.0
    #: Redirect hops allowed while fetching, each one re-checked.
    max_redirects: int = 3
    #: Let remote fetches reach private, loopback and link-local addresses.
    #: Only turn this on for a trusted internal network.
    allow_private_urls: bool = False
    #: CORS origins; ``*`` keeps the old Node API's wide-open behaviour.
    cors_origins: tuple[str, ...] = ("*",)
    #: Load the default model at startup instead of on the first request.
    preload: bool = False

    def validate(self) -> None:
        """Fail fast on a misconfigured allow-list."""
        unknown = [name for name in self.allowed_models if name not in MODELS]
        if unknown:
            raise ValueError(
                f"BGREMOVER_MODELS names unknown models: {', '.join(unknown)}. "
                f"Known models: {', '.join(sorted(MODELS))}"
            )
        if not self.allowed_models:
            raise ValueError("BGREMOVER_MODELS is empty; the API would serve nothing")
        if self.default_model not in self.allowed_models:
            raise ValueError(
                f"BGREMOVER_DEFAULT_MODEL {self.default_model!r} is not in the allowed list "
                f"({', '.join(self.allowed_models)})"
            )


def load_settings() -> Settings:
    """Build settings from the current environment."""
    settings = Settings(
        max_upload_bytes=_env_int("BGREMOVER_MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        allowed_models=_env_list("BGREMOVER_MODELS", DEFAULT_ALLOWED_MODELS),
        default_model=_env_str("BGREMOVER_DEFAULT_MODEL", DEFAULT_MODEL),
        max_concurrency=_env_int("BGREMOVER_MAX_CONCURRENCY", 2),
        fetch_timeout=float(_env_int("BGREMOVER_FETCH_TIMEOUT", 15)),
        max_redirects=_env_int("BGREMOVER_MAX_REDIRECTS", 3, minimum=0),
        allow_private_urls=_env_bool("BGREMOVER_ALLOW_PRIVATE_URLS", False),
        cors_origins=_env_list("BGREMOVER_CORS_ORIGINS", ("*",)),
        preload=_env_bool("BGREMOVER_PRELOAD", False),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the running process."""
    return load_settings()
