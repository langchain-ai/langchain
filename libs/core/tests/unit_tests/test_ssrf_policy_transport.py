import socket
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from langchain_core._security import (
    SSRFBlockedError,
    SSRFPolicy,
    SSRFSafeSyncTransport,
    SSRFSafeTransport,
    ssrf_safe_async_client,
    ssrf_safe_client,
    validate_hostname,
    validate_resolved_ip,
    validate_url_sync,
)


def _fake_addrinfo(ip: str, port: int = 80) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


def _fake_addrinfo_v6(ip: str, port: int = 80) -> list[Any]:
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0))]


def _ok_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="ok")


def test_validate_resolved_ip_blocks_nat64_embedded_private_ip() -> None:
    policy = SSRFPolicy()

    with pytest.raises(SSRFBlockedError, match="private IP range"):
        validate_resolved_ip("64:ff9b::c0a8:101", policy)


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


class _RecordingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_ssrf_safe_transport_pins_ip_and_sets_sni() -> None:
    transport = SSRFSafeTransport()
    recorder = _RecordingAsyncTransport()
    transport._inner = recorder  # type: ignore[assignment]

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
        request = httpx.Request("GET", "https://example.com/resource")
        response = await transport.handle_async_request(request)

    assert response.status_code == 200
    assert len(recorder.requests) == 1
    pinned_request = recorder.requests[0]
    assert pinned_request.url.host == "93.184.216.34"
    assert pinned_request.headers["host"] == "example.com"
    assert pinned_request.extensions["sni_hostname"] == b"example.com"


@pytest.mark.asyncio
async def test_ssrf_safe_transport_blocks_private_resolution() -> None:
    transport = SSRFSafeTransport()

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
        request = httpx.Request("GET", "https://example.com/resource")
        with pytest.raises(SSRFBlockedError, match="private IP range"):
            await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_ssrf_safe_async_client_sets_redirect_defaults() -> None:
    client = ssrf_safe_async_client()
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
async def test_redirect_to_private_ip_blocked(monkeypatch: Any) -> None:
    call_count = 0

    def _routing_addrinfo(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _fake_addrinfo("93.184.216.34")
        return _fake_addrinfo("127.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", _routing_addrinfo)

    def _redirect_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "http://evil.com/pwned"},
        )

    transport = SSRFSafeTransport()
    transport._inner = httpx.MockTransport(_redirect_responder)  # type: ignore[assignment]

    client = httpx.AsyncClient(
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
async def test_ipv6_mapped_ipv4_blocked(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: _fake_addrinfo_v6("::ffff:127.0.0.1"),
    )

    transport = SSRFSafeTransport()
    request = httpx.Request("GET", "http://evil.com/")
    with pytest.raises(SSRFBlockedError):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_scheme_blocked() -> None:
    transport = SSRFSafeTransport()
    request = httpx.Request("GET", "ftp://evil.com/file")
    with pytest.raises(SSRFBlockedError, match="scheme"):
        await transport.handle_async_request(request)


@pytest.mark.asyncio
async def test_unresolvable_host_blocked(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: (_ for _ in ()).throw(
            socket.gaierror("Name or service not known")
        ),
    )

    transport = SSRFSafeTransport()
    request = httpx.Request("GET", "http://nonexistent.invalid/")
    with pytest.raises(SSRFBlockedError, match="DNS resolution failed"):
        await transport.handle_async_request(request)


# ---------------------------------------------------------------------------
# Transport: allowed_hosts bypass and local env behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowed_host_bypass() -> None:
    policy = SSRFPolicy(allowed_hosts=frozenset({"special.host"}))
    transport = SSRFSafeTransport(policy=policy)
    transport._inner = httpx.MockTransport(_ok_response)  # type: ignore[assignment]

    request = httpx.Request("GET", "http://special.host/api")
    response = await transport.handle_async_request(request)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["local_dev", "local_test", "local_docker"])
async def test_localhost_allowed_in_local_env(monkeypatch: Any, env: str) -> None:
    monkeypatch.setenv("LANGCHAIN_ENV", env)
    transport = SSRFSafeTransport()
    transport._inner = httpx.MockTransport(_ok_response)  # type: ignore[assignment]

    request = httpx.Request("GET", "http://localhost:8084/mcp")
    response = await transport.handle_async_request(request)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_localhost_blocked_in_production(monkeypatch: Any) -> None:
    monkeypatch.setenv("LANGCHAIN_ENV", "production")
    transport = SSRFSafeTransport()

    request = httpx.Request("GET", "http://localhost:8084/mcp")
    with pytest.raises(SSRFBlockedError):
        await transport.handle_async_request(request)


# ---------------------------------------------------------------------------
# Sync transport tests
# ---------------------------------------------------------------------------


def test_sync_transport_pins_ip_and_sets_sni() -> None:
    transport = SSRFSafeSyncTransport()
    transport._inner = httpx.MockTransport(_ok_response)  # type: ignore[assignment]

    addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = httpx.Request("GET", "https://example.com/resource")
        response = transport.handle_request(request)

    assert response.status_code == 200


def test_sync_transport_blocks_private_resolution() -> None:
    transport = SSRFSafeSyncTransport()

    addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    with patch(
        "langchain_core._security._transport.socket.getaddrinfo",
        return_value=addrinfo,
    ):
        request = httpx.Request("GET", "https://example.com/resource")
        with pytest.raises(SSRFBlockedError, match="private IP range"):
            transport.handle_request(request)


def test_sync_transport_redirect_to_private_blocked(monkeypatch: Any) -> None:
    call_count = 0

    def _routing_addrinfo(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _fake_addrinfo("93.184.216.34")
        return _fake_addrinfo("127.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", _routing_addrinfo)

    def _redirect_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "http://evil.com/pwned"},
        )

    transport = SSRFSafeSyncTransport()
    transport._inner = httpx.MockTransport(_redirect_responder)  # type: ignore[assignment]

    client = httpx.Client(
        transport=transport,
        follow_redirects=True,
        max_redirects=5,
    )

    with pytest.raises(SSRFBlockedError):
        client.get("http://safe.com/start")

    client.close()


def test_ssrf_safe_client_sets_redirect_defaults() -> None:
    client = ssrf_safe_client()
    try:
        assert client.follow_redirects is True
        assert client.max_redirects == 10
    finally:
        client.close()
