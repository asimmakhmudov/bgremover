"""Tests for the model registry."""

from __future__ import annotations

import pytest

from snipbg.models import (
    DEFAULT_MODEL,
    MODELS,
    ModelSpec,
    get_spec,
    list_models,
    model_base_url,
)


def test_default_model_is_registered():
    assert DEFAULT_MODEL in MODELS


def test_every_spec_is_complete():
    for name, spec in MODELS.items():
        assert spec.name == name, "the registry key must match the spec name"
        assert spec.filename.endswith(".onnx")
        assert len(spec.sha256) == 64, f"{name} needs a real sha256"
        assert int(spec.sha256, 16) >= 0, f"{name} sha256 is not hex"
        assert spec.size_bytes > 0
        assert spec.input_size[0] > 0 and spec.input_size[1] > 0
        assert len(spec.mean) == 3 and len(spec.std) == 3
        assert all(value > 0 for value in spec.std), "a zero std would divide by zero"
        assert spec.license, f"{name} must record its license"
        assert spec.description


def test_digests_and_filenames_are_unique():
    assert len({spec.sha256 for spec in MODELS.values()}) == len(MODELS)
    assert len({spec.filename for spec in MODELS.values()}) == len(MODELS)


def test_get_spec_defaults_and_rejects_unknown():
    assert get_spec().name == DEFAULT_MODEL
    assert get_spec("u2netp").name == "u2netp"
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("not-a-model")


def test_url_follows_the_base_url_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SNIPBG_MODEL_BASE_URL", "https://mirror.example/weights/")
    assert get_spec("u2netp").url == "https://mirror.example/weights/u2netp.onnx"


def test_list_models_matches_the_registry():
    assert [spec.name for spec in list_models()] == list(MODELS)


def test_size_mb_is_derived():
    spec = ModelSpec("x", "x.onnx", "0" * 64, 1024 * 1024, (32, 32))
    assert spec.size_mb == pytest.approx(1.0)


def test_model_base_url_has_no_trailing_slash(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SNIPBG_MODEL_BASE_URL", "https://example.com/x///")
    assert model_base_url() == "https://example.com/x"


def test_models_import_without_third_party_packages():
    """The Docker model stage runs this module with only the stdlib installed."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys, importlib.abc
        blocked = {"numpy", "PIL", "onnxruntime", "fastapi", "anyio", "pydantic"}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in blocked:
                    raise ImportError("blocked: " + name)
                return None
        sys.meta_path.insert(0, Blocker())
        from snipbg.download import ensure_model, cache_dir
        from snipbg.models import get_spec
        print(get_spec("u2netp").filename)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "u2netp.onnx" in result.stdout
