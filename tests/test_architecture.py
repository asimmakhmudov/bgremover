"""Guards for the structural rules this package is built on.

These are cheap to check and expensive to notice by eye, so they live here
rather than only in CLAUDE.md.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src" / "snipbg"


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a file (absolute imports only)."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _core_modules() -> list[Path]:
    return sorted(p for p in SOURCE.glob("*.py"))


def test_no_rembg_anywhere():
    """Rule 1: this package must never depend on rembg."""
    for path in SOURCE.rglob("*.py"):
        text = path.read_text()
        assert "import rembg" not in text, path
        assert "from rembg" not in text, path

    pyproject = (SOURCE.parent.parent / "pyproject.toml").read_text()
    dependency_block = pyproject.split("[project.optional-dependencies]")[0]
    assert "rembg" not in dependency_block


def test_the_core_library_never_imports_the_server():
    """Rule 2: snipbg.server is optional and must stay that way."""
    for path in _core_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "server" in node.module:
                if path.name != "cli.py":
                    pytest.fail(f"{path.name} imports {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "snipbg.server" not in alias.name, path.name


def test_importing_the_core_library_does_not_pull_in_fastapi():
    code = textwrap.dedent(
        """
        import sys
        import snipbg
        from snipbg import Remover, remove
        from snipbg.session import get_remover
        loaded = [m for m in sys.modules if m.split(".")[0] in {"fastapi", "starlette", "uvicorn"}]
        assert not loaded, loaded
        print("clean")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@pytest.mark.parametrize("module", ["models.py", "download.py", "__init__.py"])
def test_the_docker_model_stage_files_stay_stdlib_only(module):
    """The Docker weights stage installs no third-party packages."""
    third_party = {"numpy", "PIL", "onnxruntime", "fastapi", "anyio", "pydantic", "starlette"}
    assert not (_imported_modules(SOURCE / module) & third_party), module


def test_processing_stays_pure():
    """Rule 3: no ONNX, no network, no file I/O in processing.py."""
    imported = _imported_modules(SOURCE / "processing.py")
    forbidden = {
        "onnxruntime",
        "urllib",
        "requests",
        "httpx",
        "socket",
        "os",
        "pathlib",
        "shutil",
        "tempfile",
    }
    assert not (imported & forbidden), imported & forbidden

    text = (SOURCE / "processing.py").read_text()
    for banned in ("open(", ".save(", "Image.open"):
        assert banned not in text, f"processing.py should not call {banned}"


def test_the_legacy_routes_still_exist():
    """Rule 6: old Node.js API clients depend on these."""
    from snipbg.server.app import create_app

    # The OpenAPI schema, not app.routes: newer FastAPI nests included routers.
    paths = set(create_app().openapi()["paths"])
    assert {"/remove-background", "/remove-background-url", "/health", "/"} <= paths


def test_the_server_keeps_its_safety_rails():
    """Rule 7: these protections must not be dropped."""
    from snipbg.server import app as app_module
    from snipbg.server import config, fetch

    assert "max_upload_bytes" in config.Settings.__dataclass_fields__
    assert "allowed_models" in config.Settings.__dataclass_fields__
    assert "max_concurrency" in config.Settings.__dataclass_fields__
    assert hasattr(fetch, "check_url") and hasattr(fetch, "resolve_and_check")
    # Rule 8: inference is handed to a worker thread under the limiter.
    source = Path(app_module.__file__).read_text()
    assert "anyio.to_thread.run_sync" in source
    assert "limiter=limiter" in source


def test_default_served_models_are_small_enough():
    """Rule 5: nothing above ~250 MB belongs in the server's default list."""
    from snipbg.models import MODELS
    from snipbg.server.config import DEFAULT_ALLOWED_MODELS

    for name in DEFAULT_ALLOWED_MODELS:
        assert MODELS[name].size_mb < 250, f"{name} is too large to serve by default"
