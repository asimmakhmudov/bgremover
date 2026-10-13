"""Fetching a remote image without letting callers probe the private network.

A URL handed to the API is attacker-controlled, so every hop is checked: the
scheme must be HTTP(S), the host must resolve only to public addresses, and the
body is capped while it streams. Redirects are followed by hand precisely so
that the destination of each hop goes through the same checks - a redirect to
``http://169.254.169.254/`` is the classic way past a naive check.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlsplit, urlunsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_CHUNK = 64 * 1024
_USER_AGENT = "snipbg (+https://github.com/asimmakhmudov/snipbg)"


class FetchError(ValueError):
    """A URL could not be fetched, or was not allowed to be fetched."""


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject anything that is not a routable public address."""
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return False
    if ip.is_reserved or ip.is_unspecified:
        return False
    # ::ffff:127.0.0.1 and friends: judge the embedded v4 address instead.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return _is_public(mapped)
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None:
        return _is_public(sixtofour)
    return True


def resolve_and_check(host: str, port: int, *, allow_private: bool = False) -> None:
    """Resolve a hostname and refuse it if any address is not public.

    Every resolved address is checked, not just the first: a host that answers
    with both a public and a private address must not slip through.

    Raises:
        FetchError: if resolution fails or any address is disallowed.
    """
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve host {host!r}: {exc}") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise FetchError(f"host {host!r} did not resolve to any address")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            raise FetchError(f"host {host!r} resolved to an unusable address") from None
        if not _is_public(ip):
            raise FetchError(
                f"refusing to fetch from {host!r}: it resolves to the non-public address {ip}"
            )


def check_url(url: str, *, allow_private: bool = False) -> str:
    """Validate one URL and return it normalised.

    Raises:
        FetchError: if the scheme, host or resolved addresses are not allowed.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise FetchError(f"unsupported URL scheme {parts.scheme or '(none)'!r}; use http or https")
    if not parts.hostname:
        raise FetchError("the URL has no host")

    try:
        port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    except ValueError:
        raise FetchError("the URL has an invalid port") from None

    resolve_and_check(parts.hostname, port, allow_private=allow_private)
    return urlunsplit(parts)


def fetch_image(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 15.0,
    max_redirects: int = 3,
    allow_private: bool = False,
) -> bytes:
    """Download an image, enforcing the checks above on every hop.

    This blocks, so call it from a worker thread.

    Raises:
        FetchError: on a disallowed URL, a transport error, or a body that
            exceeds ``max_bytes``.
    """
    current = check_url(url, allow_private=allow_private)
    seen = 0

    while True:
        request = urllib.request.Request(
            current,
            headers={"User-Agent": _USER_AGENT, "Accept": "image/*"},
            method="GET",
        )
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise FetchError("the server sent a redirect with no destination") from None
                seen += 1
                if seen > max_redirects:
                    raise FetchError(f"too many redirects (more than {max_redirects})") from None
                # Re-check the new target: this is the hop an SSRF payload uses.
                nxt = urljoin(current, location)
                current = check_url(nxt, allow_private=allow_private)
                continue
            raise FetchError(f"the server returned HTTP {exc.code} for the image URL") from None
        except OSError as exc:
            raise FetchError(f"could not fetch the image URL: {exc}") from None

        with response:
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError(
                    f"the image is larger than the {max_bytes // (1024 * 1024)} MB limit"
                )

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FetchError(
                        f"the image is larger than the {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)

        if not chunks:
            raise FetchError("the image URL returned an empty response")
        return b"".join(chunks)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn redirects into errors so the caller can re-check each destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None
