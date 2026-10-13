"""Tests for the cache and the downloader, served from a local HTTP server."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from snipbg import download as dl
from snipbg.models import ModelSpec

PAYLOAD = b"fake onnx weights" * 1000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's API
        if self.path == "/good.onnx":
            body = PAYLOAD
        elif self.path == "/tampered.onnx":
            body = PAYLOAD + b"!"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    # A short poll interval keeps shutdown() from costing half a second per test.
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def local_spec(server, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SNIPBG_MODEL_BASE_URL", server)
    return ModelSpec("local", "good.onnx", DIGEST, len(PAYLOAD), (32, 32))


def test_cache_dir_follows_snipbg_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SNIPBG_HOME", str(tmp_path / "models"))
    assert dl.cache_dir() == tmp_path / "models"


def test_cache_dir_falls_back_to_xdg(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SNIPBG_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert dl.cache_dir() == tmp_path / "snipbg"


def test_download_writes_and_verifies(local_spec):
    path = dl.download_model(local_spec)
    assert path.read_bytes() == PAYLOAD
    assert dl.verify_model(local_spec) is True
    assert dl.is_cached(local_spec) is True


def test_download_rejects_a_tampered_file_and_leaves_nothing_behind(local_spec):
    bad = ModelSpec("bad", "tampered.onnx", DIGEST, len(PAYLOAD), (32, 32))
    with pytest.raises(dl.DownloadError, match="SHA-256 mismatch"):
        dl.download_model(bad)

    assert not dl.model_path(bad).exists()
    assert not list(dl.cache_dir().glob("*.part")), "the partial file must be cleaned up"


def test_download_reports_a_missing_file(local_spec):
    missing = ModelSpec("missing", "absent.onnx", DIGEST, 10, (32, 32))
    with pytest.raises(dl.DownloadError, match="could not download"):
        dl.download_model(missing)


def test_ensure_model_is_a_no_op_once_cached(local_spec, monkeypatch: pytest.MonkeyPatch):
    first = dl.ensure_model(local_spec)

    def explode(*args, **kwargs):
        raise AssertionError("ensure_model downloaded an already cached file")

    monkeypatch.setattr(dl, "download_model", explode)
    assert dl.ensure_model(local_spec) == first


def test_is_cached_rejects_a_truncated_file(local_spec):
    path = dl.download_model(local_spec)
    path.write_bytes(PAYLOAD[:10])
    assert dl.is_cached(local_spec) is False


def test_verify_model_detects_corruption(local_spec):
    path = dl.download_model(local_spec)
    path.write_bytes(PAYLOAD.replace(b"fake", b"FAKE", 1))
    assert dl.verify_model(local_spec) is False


def test_verify_model_is_false_when_absent(local_spec):
    assert dl.verify_model(local_spec) is False


def test_progress_callback_reports_bytes(local_spec):
    seen: list[tuple[int, int]] = []
    dl.download_model(local_spec, progress=lambda done, total: seen.append((done, total)))
    assert seen
    assert seen[-1][0] == len(PAYLOAD)
    assert seen[-1][1] == len(PAYLOAD)


def test_remove_cached_and_clear_cache(local_spec):
    dl.download_model(local_spec)
    assert dl.remove_cached(local_spec) is True
    assert dl.remove_cached(local_spec) is False

    dl.download_model(local_spec)
    dl.clear_cache()
    assert not dl.cache_dir().exists()


def test_a_spec_without_a_digest_is_accepted(server, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SNIPBG_MODEL_BASE_URL", server)
    spec = ModelSpec("nodigest", "good.onnx", "", len(PAYLOAD), (32, 32))
    assert dl.download_model(spec).read_bytes() == PAYLOAD
    assert dl.verify_model(spec) is True
