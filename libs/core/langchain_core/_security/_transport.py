"""SSRF-safe httpx transport with DNS resolution and IP pinning."""

import asyncio
import socket

import httpx
import structlog

from lc_security.exceptions import SSRFBlockedError
from lc_security.policy import (
    SSRFPolicy,
    _effective_allowed_hosts,
    validate_resolved_ip,
    validate_url_sync,
)

logger = structlog.get_logger(__name__)

# Keys that AsyncHTTPTransport accepts (forwarded from factory kwargs).
_TRANSPORT_KWARGS = frozenset(
    {
        "verify",
        "cert",
        "trust_env",
        "http1",
        "http2",
        "limits",
        "retries",
    }
)


class SSRFSafeTransport(httpx.AsyncBaseTransport):
    """httpx async transport that validates DNS results against an SSRF policy.

    For every outgoing request the transport:
    1. Checks the URL scheme against ``policy.allowed_schemes``.
    2. Validates the hostname against blocked patterns.
    3. Resolves DNS and validates **all** returned IPs.
    4. Rewrites the request to connect to the first valid IP while
       preserving the original ``Host`` header and TLS SNI hostname.

    Redirects are re-validated on each hop because ``follow_redirects``
    is set on the *client*, causing ``handle_async_request`` to be called
    again for each redirect target.
    """

    def __init__(
        self,
        policy: SSRFPolicy = SSRFPolicy(),
        **transport_kwargs: object,
    ) -> None:
        self._policy = policy
        self._inner = httpx.AsyncHTTPTransport(**transport_kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Core request handler
    # ------------------------------------------------------------------ #

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        hostname = request.url.host or ""
        scheme = request.url.scheme.lower()

        # 1-3. Scheme, hostname, and pattern checks (reuse sync validator).
        try:
            validate_url_sync(str(request.url), self._policy)
        except SSRFBlockedError as exc:
            logger.warning("ssrf_blocked", hostname=hostname, reason=str(exc))
            raise

        # Allowed-hosts bypass — skip DNS/IP validation entirely.
        allowed = {h.lower() for h in _effective_allowed_hosts(self._policy)}
        if hostname.lower() in allowed:
            return await self._inner.handle_async_request(request)

        # 4. DNS resolution
        port = request.url.port or (443 if scheme == "https" else 80)
        try:
            addrinfo = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            logger.warning(
                "ssrf_blocked", hostname=hostname, reason="DNS resolution failed"
            )
            raise SSRFBlockedError("DNS resolution failed") from exc

        if not addrinfo:
            logger.warning(
                "ssrf_blocked",
                hostname=hostname,
                reason="DNS resolution returned no results",
            )
            raise SSRFBlockedError("DNS resolution returned no results")

        # 5. Validate ALL resolved IPs — any blocked means reject.
        for _family, _type, _proto, _canonname, sockaddr in addrinfo:
            ip_str = sockaddr[0]
            try:
                validate_resolved_ip(ip_str, self._policy)
            except SSRFBlockedError as exc:
                logger.warning("ssrf_blocked", hostname=hostname, reason=str(exc))
                raise

        # 6. Pin to first resolved IP.
        pinned_ip = addrinfo[0][4][0]

        # 7. Rewrite URL to use pinned IP, preserving Host header and SNI.
        pinned_url = request.url.copy_with(host=pinned_ip)

        # Build extensions dict, adding sni_hostname for HTTPS so TLS
        # certificate validation uses the original hostname.
        extensions = dict(request.extensions)
        if scheme == "https":
            extensions["sni_hostname"] = hostname.encode("ascii")

        pinned_request = httpx.Request(
            method=request.method,
            url=pinned_url,
            headers=request.headers,  # Host header already set to original
            content=request.content,
            extensions=extensions,
        )

        return await self._inner.handle_async_request(pinned_request)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------- #
# Factory
# ---------------------------------------------------------------------- #


def ssrf_safe_async_client(
    policy: SSRFPolicy = SSRFPolicy(),
    **kwargs: object,
) -> httpx.AsyncClient:
    """Create an ``httpx.AsyncClient`` with SSRF protection.

    Drop-in replacement for ``httpx.AsyncClient(...)`` — callers just swap
    the constructor call.  Transport-specific kwargs (``verify``, ``cert``,
    ``retries``, etc.) are forwarded to the inner ``AsyncHTTPTransport``;
    everything else goes to the ``AsyncClient``.
    """
    transport_kwargs: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}
    for key, value in kwargs.items():
        if key in _TRANSPORT_KWARGS:
            transport_kwargs[key] = value
        else:
            client_kwargs[key] = value

    transport = SSRFSafeTransport(policy=policy, **transport_kwargs)

    # Apply defaults only if not overridden by caller.
    client_kwargs.setdefault("follow_redirects", True)
    client_kwargs.setdefault("max_redirects", 10)

    return httpx.AsyncClient(
        transport=transport,
        **client_kwargs,  # type: ignore[arg-type]
    )
