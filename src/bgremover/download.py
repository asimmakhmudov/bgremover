"""Model cache: where weights live, and how they get there.

Only the standard library may be imported here: the Docker model stage installs
no third-party packages and runs this module together with :mod:`models`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .models import ModelSpec, get_spec

_CHUNK = 1024 * 256
_USER_AGENT = "bgremover (+https://github.com/asimmakhmudov/bgremover)"

ProgressFn = Callable[[int, int], None]


class DownloadError(RuntimeError):
    """A model could not be downloaded, or arrived corrupted."""


def cache_dir() -> Path:
    """Directory holding downloaded weights.

    ``BGREMOVER_HOME`` wins; otherwise ``$XDG_CACHE_HOME/bgremover``, falling
    back to ``~/.cache/bgremover``.
    """
    env = os.environ.get("BGREMOVER_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "bgremover"


def model_path(spec: ModelSpec | str) -> Path:
    """Path the given model's weights are cached at (may not exist yet)."""
    if isinstance(spec, str):
        spec = get_spec(spec)
    return cache_dir() / spec.filename


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cached(spec: ModelSpec | str) -> bool:
    """True if the weights are already on disk with the expected size."""
    if isinstance(spec, str):
        spec = get_spec(spec)
    path = model_path(spec)
    if not path.is_file():
        return False
    if spec.size_bytes and path.stat().st_size != spec.size_bytes:
        return False
    return True


def _stderr_progress(done: int, total: int) -> None:
    if total <= 0:
        sys.stderr.write(f"\r  {done / 1048576:.1f} MB")
    else:
        pct = 100.0 * done / total
        sys.stderr.write(f"\r  {pct:5.1f}%  {done / 1048576:7.1f} / {total / 1048576:.1f} MB")
    sys.stderr.flush()


def download_model(
    spec: ModelSpec | str,
    *,
    progress: ProgressFn | bool = False,
    force: bool = False,
) -> Path:
    """Download a model's weights into the cache and return the path.

    The body is streamed to a temporary file next to the destination, hashed
    while it is written, and only renamed into place once the digest matches —
    so a cached path is never a half-written file.

    Raises:
        DownloadError: on a transport error or a digest mismatch.
    """
    if isinstance(spec, str):
        spec = get_spec(spec)

    dest = model_path(spec)
    if dest.is_file() and not force:
        return dest

    on_progress: ProgressFn | None
    if progress is True:
        on_progress = _stderr_progress
    elif progress is False:
        on_progress = None
    else:
        on_progress = progress

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed https base
            total = int(response.headers.get("Content-Length") or spec.size_bytes or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
    except OSError as exc:  # URLError and friends are OSError subclasses
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"could not download {spec.name} from {spec.url}: {exc}") from exc
    finally:
        if on_progress is _stderr_progress:
            sys.stderr.write("\n")

    actual = digest.hexdigest()
    if spec.sha256 and actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"{spec.name}: SHA-256 mismatch (expected {spec.sha256}, got {actual}). "
            "The download was discarded."
        )

    os.replace(tmp, dest)
    return dest


def ensure_model(spec: ModelSpec | str, *, progress: ProgressFn | bool = False) -> Path:
    """Return the cached path for a model, downloading it if it is missing."""
    if isinstance(spec, str):
        spec = get_spec(spec)
    if is_cached(spec):
        return model_path(spec)
    return download_model(spec, progress=progress, force=True)


def verify_model(spec: ModelSpec | str) -> bool:
    """Hash a cached file and compare it with the expected digest.

    Returns ``False`` if the file is missing or corrupt, ``True`` if it matches
    (or if the registry carries no digest for it).
    """
    if isinstance(spec, str):
        spec = get_spec(spec)
    path = model_path(spec)
    if not path.is_file():
        return False
    if not spec.sha256:
        return True
    return sha256_file(path) == spec.sha256


def remove_cached(spec: ModelSpec | str) -> bool:
    """Delete a cached model. True if a file was removed."""
    path = model_path(spec)
    if path.is_file():
        path.unlink()
        return True
    return False


def clear_cache() -> None:
    """Delete the whole model cache directory."""
    shutil.rmtree(cache_dir(), ignore_errors=True)
