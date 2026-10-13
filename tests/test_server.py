"""Tests for the HTTP API, including the compatibility surface of the old API."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from snipbg.images import load_image
from snipbg.server.app import create_app
from snipbg.server.config import Settings

pytestmark = pytest.mark.usefixtures("fake_onnx")

SETTINGS = Settings(
    max_upload_bytes=64 * 1024,
    allowed_models=("u2netp", "silueta"),
    default_model="u2netp",
    max_concurrency=2,
)


@pytest.fixture
def client():
    with TestClient(create_app(SETTINGS)) as test_client:
        yield test_client


@pytest.fixture
def upload(red_blue_png):
    return {"image": ("photo.png", red_blue_png, "image/png")}


# --- informational routes ------------------------------------------------


def test_index_keeps_the_old_shape(client):
    body = client.get("/").json()
    assert body["name"] == "Background Remover API"
    assert "POST /remove-background" in body["endpoints"]
    assert body["default_model"] == "u2netp"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_models_route_lists_only_allowed_models(client):
    body = client.get("/models").json()
    assert [entry["name"] for entry in body["models"]] == ["u2netp", "silueta"]
    assert body["default"] == "u2netp"


# --- file upload ---------------------------------------------------------


def test_remove_background_returns_a_png(client, upload):
    response = client.post("/remove-background", files=upload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = load_image(response.content)
    assert image.size == (64, 32)
    assert image.mode == "RGBA"


def test_remove_background_without_a_file_uses_the_old_error_shape(client):
    response = client.post("/remove-background")
    assert response.status_code == 400
    assert response.json() == {"error": "No image file provided"}


def test_remove_background_rejects_an_empty_file(client):
    response = client.post("/remove-background", files={"image": ("x.png", b"", "image/png")})
    assert response.status_code == 400
    assert "empty" in response.json()["error"]


def test_remove_background_rejects_a_non_image(client):
    files = {"image": ("x.png", b"this is not an image", "image/png")}
    response = client.post("/remove-background", files=files)
    assert response.status_code == 400
    message = response.json()["error"]
    assert "could not decode" in message
    assert "object at 0x" not in message, "internal reprs must not reach clients"


def test_trim_and_background_combine(client, upload):
    data = {"trim": "true", "background": "white"}
    response = client.post("/remove-background", files=upload, data=data)
    image = load_image(response.content)
    assert image.mode == "RGB"
    assert image.width < 64, "trim must still apply when a background is requested"


def test_upload_size_limit_is_enforced(client):
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (SETTINGS.max_upload_bytes + 1)
    response = client.post("/remove-background", files={"image": ("big.png", oversized)})
    assert response.status_code == 413
    assert "File too large" in response.json()["error"]


def test_options_are_honoured(client, upload):
    response = client.post("/remove-background", files=upload, data={"only_mask": "true"})
    assert load_image(response.content).mode == "L"

    response = client.post("/remove-background", files=upload, data={"background": "white"})
    assert load_image(response.content).mode == "RGB"

    response = client.post("/remove-background", files=upload, data={"format": "JPEG"})
    assert response.headers["content-type"] == "image/jpeg"
    assert load_image(response.content).format == "JPEG"


def test_trim_option_crops(client, upload):
    response = client.post("/remove-background", files=upload, data={"trim": "true"})
    assert load_image(response.content).width < 64


# --- the model allow-list ------------------------------------------------


def test_an_allowed_model_is_accepted(client, upload):
    assert (
        client.post("/remove-background", files=upload, data={"model": "silueta"}).status_code
        == 200
    )


def test_an_unknown_model_is_refused(client, upload):
    response = client.post("/remove-background", files=upload, data={"model": "nope"})
    assert response.status_code == 400
    assert "Unknown model" in response.json()["error"]


def test_a_known_but_disabled_model_is_refused(client, upload):
    """Big models must not be reachable just because they exist in the registry."""
    response = client.post("/remove-background", files=upload, data={"model": "birefnet-general"})
    assert response.status_code == 400
    assert "not enabled" in response.json()["error"]


def test_an_unsupported_format_is_refused(client, upload):
    response = client.post("/remove-background", files=upload, data={"format": "tiff"})
    assert response.status_code == 400
    assert "Unsupported output format" in response.json()["error"]


# --- URL route -----------------------------------------------------------


class _ImageHandler(BaseHTTPRequestHandler):
    payload = b""
    redirect_to: str | None = None

    def do_GET(self):  # noqa: N802
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", self.redirect_to or "/image.png")
            self.end_headers()
            return
        if self.path == "/huge":
            body = b"\x89PNG\r\n\x1a\n" + b"0" * 200_000
        elif self.path == "/image.png":
            body = self.payload
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def image_server(red_blue_png):
    _ImageHandler.payload = red_blue_png
    _ImageHandler.redirect_to = None
    httpd = HTTPServer(("127.0.0.1", 0), _ImageHandler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def loopback_client(monkeypatch: pytest.MonkeyPatch):
    """A client whose SSRF check treats loopback as public, leaving the rest real.

    This lets the local test server stand in for a public host while every other
    address class is still judged by the production logic.
    """
    from snipbg.server import fetch

    original = fetch._is_public

    def patched(ip):
        return True if ip.is_loopback else original(ip)

    monkeypatch.setattr(fetch, "_is_public", patched)
    with TestClient(create_app(SETTINGS)) as test_client:
        yield test_client


def test_url_route_processes_a_remote_image(loopback_client, image_server):
    response = loopback_client.post(
        "/remove-background-url", json={"imageUrl": f"{image_server}/image.png"}
    )
    assert response.status_code == 200
    assert load_image(response.content).size == (64, 32)


def test_url_route_accepts_snake_case_too(loopback_client, image_server):
    response = loopback_client.post(
        "/remove-background-url", json={"image_url": f"{image_server}/image.png"}
    )
    assert response.status_code == 200


def test_url_route_follows_a_safe_redirect(loopback_client, image_server):
    response = loopback_client.post(
        "/remove-background-url", json={"imageUrl": f"{image_server}/redirect"}
    )
    assert response.status_code == 200


def test_url_route_blocks_a_redirect_into_the_private_network(loopback_client, image_server):
    """The classic SSRF payload: a public URL that redirects to link-local metadata."""
    _ImageHandler.redirect_to = "http://169.254.169.254/latest/meta-data/"
    response = loopback_client.post(
        "/remove-background-url", json={"imageUrl": f"{image_server}/redirect"}
    )
    assert response.status_code == 400
    assert "non-public" in response.json()["error"]


def test_url_route_enforces_the_size_limit(loopback_client, image_server):
    response = loopback_client.post(
        "/remove-background-url", json={"imageUrl": f"{image_server}/huge"}
    )
    assert response.status_code == 400
    assert "larger than" in response.json()["error"]


def test_url_route_reports_a_missing_url(client):
    response = client.post("/remove-background-url", json={})
    assert response.status_code == 400
    assert response.json() == {"error": "No image URL provided"}


def test_url_route_reports_a_blank_url(client):
    response = client.post("/remove-background-url", json={"imageUrl": "   "})
    assert response.status_code == 400
    assert response.json() == {"error": "No image URL provided"}


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.png",
        "http://localhost/x.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/x.png",
        "file:///etc/passwd",
        "ftp://example.com/x.png",
        "not-a-url",
    ],
)
def test_url_route_blocks_ssrf_targets(client, url):
    response = client.post("/remove-background-url", json={"imageUrl": url})
    assert response.status_code == 400
    assert "error" in response.json()


def test_url_route_rejects_a_bad_quality_value(client):
    response = client.post(
        "/remove-background-url", json={"imageUrl": "https://example.com/a.png", "quality": 500}
    )
    assert response.status_code == 400


def test_cors_headers_are_present(client):
    response = client.get("/health", headers={"Origin": "https://example.com"})
    assert response.headers["access-control-allow-origin"] == "*"
