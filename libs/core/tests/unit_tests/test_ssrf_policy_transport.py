import importlib
import socket
import sys
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import langchain_core._httpx
import langchain_core._security
import langchain_core._security._transport
from langchain_core._security import (
    SSRFBlockedError,
    SSRFPolicy,
    validate_hostname,
    validate_resolved_ip,
    validate_url_sync,
)


@dataclass(frozen=True)
class _Backend:
    """The HTTPX backend under test and the transport module built on it."""

    httpx: ModuleType
    transport: ModuleType


def _reload_httpx_modules() -> None:
    importlib.reload(langchain_core._httpx)
    importlib.reload(langchain_core._security._transport)
    importlib.reload(langchain_core._security)


@pytest.fixture(params=["httpx2", "httpx"])
def backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_Backend]:
    """Run a test against each HTTPX backend the shim can select.

    `httpx2` is the default whenever it is installed. The `httpx` fallback is
    exercised by hiding `httpx2` from the import system and reloading the
    modules that bind the backend at import time.
    """
    if request.param == "httpx":
        monkeypatch.setitem(sys.modules, "httpx2", None)
        _reload_httpx_modules()
    try:
        selected = langchain_core._httpx.httpx
        assert selected.__name__ == request.param
        yield _Backend(httpx=selected, transport=langchain_core._security._transport)
    finally:
        if request.param == "httpx":
            monkeypatch.undo()
            _reload_httpx_modules()


def _fake_addrinfo(ip: str, port: int = 80) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


def _fake_addrinfo_v6(ip: str, port: int = 80) -> list[Any]:
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))]


def _ok_mock_transport(backend: _Backend) -> Any:
    def _ok_response(_request: httpx.Request) -> Any:
        return backend.httpx.Response(200, text="ok")

    return backend.httpx.MockTransport(_ok_response)


def test_validate_resolved_ip_blocks_nat64_embedded_private_ip() -> None:
    policy = SSRFPolicy()

    with pytest.raises(SSRFBlockedError, match="private IP range"):
        validate_resolved_ip("64:ff9b::c0a8:101", policy)


@pytest.mark.parametrize("block_private_ips", [True, False])
@pytest.mark.parametrize(
    "ip",
    [
        "fd00:ec2::254%eth0",
        "fd00:ec2::23%eth0",
        "fe80::a9fe:a9fe%eth0",
        "::1%eth0",
        "::169.254.169.254",
        "::169.254.170.2",
        "::100.100.100.200",
        "::127.0.0.1",
    ],
)
def test_scoped_and_compatible_ipv6_blocked(
    ip: str, *, block_private_ips: bool
) -> None:
    policy = SSRFPolicy(block_private_ips=block_private_ips)
    with pytest.raises(SSRFBlockedError):
        validate_resolved_ip(ip, policy)
    with pytest.raises(SSRFBlockedError):
        validate_url_sync(f"http://[{ip}]/", policy)


@pytest.mark.parametrize("ip", ["::8.8.8.8", "::192.168.1.1", "fd00::123%eth0", "::"])
def test_relaxed_policy_allows_non_metadata_ipv6(ip: str) -> None:
    validate_resolved_ip(ip, SSRFPolicy(block_private_ips=False))


@pytest.mark.parametrize("ip", ["::1", "::1%eth0", "::127.0.0.1"])
def test_relaxed_policy_can_allow_ipv6_localhost(ip: str) -> None:
    validate_resolved_ip(ip, SSRFPolicy(block_private_ips=False, block_localhost=False))


def test_validate_resolved_ip_blocks_cgnat() -> None:
    policy = SSRFPolicy()

    with pytest.raises(SSRFBlockedError, match="private IP range"):
        validate_resolved_ip("100.64.0.1", policy)


def test_validate_hostname_blocks_kubernetes_internal_dns() -> None:
    policy = SSRFPolicy()

    with pytest.raises(SSRFBlockedError, match="Kubernetes internal DNS"):
        validate_hostname("api.default.svc.cluster.local", policy)


def test_validate_url_sync_allows_explicit_allowed_host() -> None:
    policy = SSRFPolicy(allowed_hosts=frozenset({"metadata.google.internal"}))

    validate_url_sync("http://metadata.google.internal/path", policy)


def test_validate_url_sync_blocks_metadata_without_allowlist() -> None:
    policy = SSRFPolicy()

    with pytest.raises(SSRFBlockedError, match="cloud metadata endpoint"):
        validate_url_sync("http://metadata.google.internal/path", policy)


class _RecordingSyncTransport:
    """Inner transport stand-in that records the (pinned) requests it receives."""

    def __init__(self, backend: _Backend) -> None:
        self._httpx = backend.httpx
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> Any:
        self.requests.append(request)
        return self._httpx.Response(200, request=request, text="ok")

    def close(self) -> None:
        return None


class _RecordingAsyncTransport(_RecordingSyncTransport):
    async def handle_async_request(self, request: httpx.Request) -> Any:
        return self.handle_request(request)

    async def aclose(self) -> None:
        return None


def _assert_pinned(recorder: _RecordingSyncTransport, backend: _Backend) -> None:
    assert len(recorder.requests) == 1
    pinned_request = recorder.requests[0]
    assert isinstance(pinned_request, backend.httpx.Request)
    assert pinned_request.url.host == "93.184.216.34"
    assert pinned_request.headers["host"] == "example.com"
    # `str`, as the connection layer requires: `bytes` fails the TLS handshake
    # under `httpcore2` (sync and async) and under `httpcore` (async).
    assert pinned_request.extensions["sni_hostname"] == "example.com"


@pytest.mark.asyncio
async def test_ssrf_safe_transport_pins_ip_and_sets_sni(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeTransport()
    recorder = _RecordingAsyncTransport(backend)
    transport._inner = recorder

    addrinfo = [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 443),
        )
    ]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = backend.httpx.Request("GET", "https://example.com/resource")
        response = await transport.handle_async_request(request)

    assert response.status_code == 200
    _assert_pinned(recorder, backend)


@pytest.mark.asyncio
async def test_ssrf_safe_transport_blocks_private_resolution(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeTransport()

    addrinfo = [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            6,
            "",
            ("127.0.0.1", 443),
        )
    ]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = backend.httpx.Request("GET", "https://example.com/resource")
        with pytest.raises(SSRFBlockedError, match="private IP range"):
            await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_ssrf_safe_async_client_sets_redirect_defaults(backend: _Backend) -> None:
    client = backend.transport.ssrf_safe_async_client()
    try:
        assert client.follow_redirects is True
        assert client.max_redirects == 10
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Policy toggle: block_private_ips=False still blocks loopback/metadata/k8s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.1:8080/api",
        "http://172.16.0.1:3000/",
        "http://192.168.1.100/webhook",
    ],
)
def test_private_ip_allowed_when_block_disabled(url: str) -> None:
    policy = SSRFPolicy(block_private_ips=False)
    validate_url_sync(url, policy)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/",
        "http://127.0.0.2/",
        "http://[::1]:8080/",
    ],
)
def test_loopback_still_blocked_when_private_ips_allowed(url: str) -> None:
    policy = SSRFPolicy(block_private_ips=False)
    with pytest.raises(SSRFBlockedError):
        validate_url_sync(url, policy)


def test_docker_internal_blocked() -> None:
    policy = SSRFPolicy()
    with pytest.raises(SSRFBlockedError, match="localhost"):
        validate_url_sync("http://host.docker.internal:8080/", policy)


def test_metadata_still_blocked_when_private_ips_allowed() -> None:
    policy = SSRFPolicy(block_private_ips=False)
    with pytest.raises(SSRFBlockedError):
        validate_url_sync("http://metadata.google.internal/", policy)


def test_k8s_still_blocked_when_private_ips_allowed() -> None:
    policy = SSRFPolicy(block_private_ips=False)
    with pytest.raises(SSRFBlockedError):
        validate_url_sync("http://myservice.default.svc.cluster.local/", policy)


# ---------------------------------------------------------------------------
# Cloud metadata: link-local range and restored IPs blocked even with
# block_private_ips=False (regression test for dropped ranges/IPs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "169.254.169.254",
        "169.254.170.2",
        "169.254.170.23",  # AWS EKS Pod Identity Agent
        "100.100.100.200",
        "fd00:ec2::254",
        "fd00:ec2::23",  # AWS EKS Pod Identity Agent (IPv6)
        "fe80::a9fe:a9fe",  # OpenStack Nova metadata
    ],
)
def test_cloud_metadata_ips_blocked_when_private_ips_allowed(ip: str) -> None:
    policy = SSRFPolicy(block_private_ips=False)
    with pytest.raises(SSRFBlockedError, match="cloud metadata endpoint"):
        validate_resolved_ip(ip, policy)


@pytest.mark.parametrize(
    "ip",
    [
        "169.254.1.2",
        "169.254.255.254",
        "169.254.42.99",
    ],
)
def test_link_local_range_blocked_as_cloud_metadata_when_private_ips_allowed(
    ip: str,
) -> None:
    policy = SSRFPolicy(block_private_ips=False)
    with pytest.raises(SSRFBlockedError, match="cloud metadata endpoint"):
        validate_resolved_ip(ip, policy)


# ---------------------------------------------------------------------------
# Transport: redirect to private IP blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redirect_to_private_ip_blocked(
    monkeypatch: Any, backend: _Backend
) -> None:
    call_count = 0

    def _routing_addrinfo(*_args: Any, **_kwargs: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _fake_addrinfo("93.184.216.34")
        return _fake_addrinfo("127.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", _routing_addrinfo)

    def _redirect_responder(_request: httpx.Request) -> Any:
        return backend.httpx.Response(
            302,
            headers={"Location": "http://evil.com/pwned"},
        )

    transport = backend.transport.SSRFSafeTransport()
    transport._inner = backend.httpx.MockTransport(_redirect_responder)

    client = backend.httpx.AsyncClient(
        transport=transport,
        follow_redirects=True,
        max_redirects=5,
    )

    with pytest.raises(SSRFBlockedError):
        await client.get("http://safe.com/start")

    await client.aclose()


# ---------------------------------------------------------------------------
# Transport: IPv6-mapped IPv4, scheme rejection, DNS fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ipv6_mapped_ipv4_blocked(monkeypatch: Any, backend: _Backend) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_, **__: _fake_addrinfo_v6("::ffff:127.0.0.1"),
    )

    transport = backend.transport.SSRFSafeTransport()
    request = backend.httpx.Request("GET", "http://evil.com/")
    with pytest.raises(SSRFBlockedError):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_scheme_blocked(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeTransport()
    request = backend.httpx.Request("GET", "ftp://evil.com/file")
    with pytest.raises(SSRFBlockedError, match="scheme"):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_unresolvable_host_blocked(monkeypatch: Any, backend: _Backend) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_, **__: (_ for _ in ()).throw(
            socket.gaierror("Name or service not known")
        ),
    )

    transport = backend.transport.SSRFSafeTransport()
    request = backend.httpx.Request("GET", "http://nonexistent.invalid/")
    with pytest.raises(SSRFBlockedError, match="DNS resolution failed"):
        await transport.handle_async_request(request)


# ---------------------------------------------------------------------------
# Transport: allowed_hosts bypass and local env behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_host_bypass(backend: _Backend) -> None:
    policy = SSRFPolicy(allowed_hosts=frozenset({"special.host"}))
    transport = backend.transport.SSRFSafeTransport(policy=policy)
    transport._inner = _ok_mock_transport(backend)

    request = backend.httpx.Request("GET", "http://special.host/api")
    response = await transport.handle_async_request(request)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["local_dev", "local_test", "local_docker"])
async def test_localhost_allowed_in_local_env(
    monkeypatch: Any, env: str, backend: _Backend
) -> None:
    monkeypatch.setenv("LANGCHAIN_ENV", env)
    transport = backend.transport.SSRFSafeTransport()
    transport._inner = _ok_mock_transport(backend)

    request = backend.httpx.Request("GET", "http://localhost:8084/mcp")
    response = await transport.handle_async_request(request)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_localhost_blocked_in_production(
    monkeypatch: Any, backend: _Backend
) -> None:
    monkeypatch.setenv("LANGCHAIN_ENV", "production")
    transport = backend.transport.SSRFSafeTransport()

    request = backend.httpx.Request("GET", "http://localhost:8084/mcp")
    with pytest.raises(SSRFBlockedError):
        await transport.handle_async_request(request)


# ---------------------------------------------------------------------------
# Sync transport tests
# ---------------------------------------------------------------------------


def test_sync_transport_pins_ip_and_sets_sni(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeSyncTransport()
    recorder = _RecordingSyncTransport(backend)
    transport._inner = recorder

    addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = backend.httpx.Request("GET", "https://example.com/resource")
        response = transport.handle_request(request)

    assert response.status_code == 200
    _assert_pinned(recorder, backend)


def test_sync_transport_blocks_private_resolution(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeSyncTransport()

    addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = backend.httpx.Request("GET", "https://example.com/resource")
        with pytest.raises(SSRFBlockedError, match="private IP range"):
            transport.handle_request(request)


def test_sync_transport_redirect_to_private_blocked(
    monkeypatch: Any, backend: _Backend
) -> None:
    call_count = 0

    def _routing_addrinfo(*_args: Any, **__kwargs: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _fake_addrinfo("93.184.216.34")
        return _fake_addrinfo("127.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", _routing_addrinfo)

    def _redirect_responder(_request: httpx.Request) -> Any:
        return backend.httpx.Response(
            302,
            headers={"Location": "http://evil.com/pwned"},
        )

    transport = backend.transport.SSRFSafeSyncTransport()
    transport._inner = backend.httpx.MockTransport(_redirect_responder)

    client = backend.httpx.Client(
        transport=transport,
        follow_redirects=True,
        max_redirects=5,
    )

    with pytest.raises(SSRFBlockedError):
        client.get("http://safe.com/start")

    client.close()


def test_ssrf_safe_client_sets_redirect_defaults(backend: _Backend) -> None:
    client = backend.transport.ssrf_safe_client()
    try:
        assert client.follow_redirects is True
        assert client.max_redirects == 10
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Backend consistency: every HTTPX object comes from the selected backend
# ---------------------------------------------------------------------------


def test_sync_transport_built_on_selected_backend(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeSyncTransport(verify=False, retries=2)
    try:
        assert isinstance(transport, backend.httpx.BaseTransport)
        assert isinstance(transport._inner, backend.httpx.HTTPTransport)
        assert transport._inner._pool._retries == 2
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_async_transport_built_on_selected_backend(backend: _Backend) -> None:
    transport = backend.transport.SSRFSafeTransport(verify=False, retries=2)
    try:
        assert isinstance(transport, backend.httpx.AsyncBaseTransport)
        assert isinstance(transport._inner, backend.httpx.AsyncHTTPTransport)
        assert transport._inner._pool._retries == 2
    finally:
        await transport.aclose()


def test_ssrf_safe_client_built_on_selected_backend(backend: _Backend) -> None:
    client = backend.transport.ssrf_safe_client(timeout=5.0)
    try:
        assert isinstance(client, backend.httpx.Client)
        assert isinstance(client._transport, backend.transport.SSRFSafeSyncTransport)
        assert client.timeout == backend.httpx.Timeout(5.0)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_ssrf_safe_async_client_built_on_selected_backend(
    backend: _Backend,
) -> None:
    client = backend.transport.ssrf_safe_async_client(timeout=5.0)
    try:
        assert isinstance(client, backend.httpx.AsyncClient)
        assert isinstance(client._transport, backend.transport.SSRFSafeTransport)
        assert client.timeout == backend.httpx.Timeout(5.0)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Real connection layer: the pinned request must survive the TLS handshake
# setup of each backend (mocked inner transports cannot catch this).
# ---------------------------------------------------------------------------


@pytest.fixture
def hangup_socket() -> Iterator[str]:
    """Unix socket whose server accepts each connection and closes it at once.

    The client gets as far as starting TLS (which reads `sni_hostname`), then
    fails the handshake with a connection error. Unix sockets keep this
    offline under `--disable-socket --allow-unix-socket`.
    """
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("Unix sockets are not available on this platform")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = str(Path(tmp_dir) / "s")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen()
        stop = threading.Event()

        def _serve() -> None:
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                conn.close()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        try:
            yield path
        finally:
            stop.set()
            server.close()
            thread.join(timeout=5)


def test_sync_transport_starts_tls_with_pinned_request(
    backend: _Backend, hangup_socket: str
) -> None:
    transport = backend.transport.SSRFSafeSyncTransport(uds=hangup_socket)
    request = backend.httpx.Request("GET", "https://example.com/resource")
    try:
        with (
            patch(
                "langchain_core._security._transport.socket.getaddrinfo",
                return_value=_fake_addrinfo("93.184.216.34", 443),
            ),
            pytest.raises(backend.httpx.ConnectError),
        ):
            transport.handle_request(request)
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_async_transport_starts_tls_with_pinned_request(
    backend: _Backend, hangup_socket: str
) -> None:
    transport = backend.transport.SSRFSafeTransport(uds=hangup_socket)
    request = backend.httpx.Request("GET", "https://example.com/resource")
    try:
        with (
            patch(
                "langchain_core._security._transport.socket.getaddrinfo",
                return_value=_fake_addrinfo("93.184.216.34", 443),
            ),
            pytest.raises(backend.httpx.ConnectError),
        ):
            await transport.handle_async_request(request)
    finally:
        await transport.aclose()
