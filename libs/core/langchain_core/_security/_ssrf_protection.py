"""SSRF Protection for validating URLs against Server-Side Request Forgery attacks.

This module provides utilities to validate user-provided URLs and prevent SSRF attacks
by blocking requests to:
- Private IP ranges (RFC 1918, loopback, link-local)
- Cloud metadata endpoints (AWS, GCP, Azure, etc.)
- Localhost addresses
- Invalid URL schemes

Usage:
    from lc_security.ssrf_protection import validate_safe_url, is_safe_url

    # Validate a URL (raises ValueError if unsafe)
    safe_url = validate_safe_url("https://example.com/webhook")

    # Check if URL is safe (returns bool)
    if is_safe_url("http://192.168.1.1"):
        # URL is safe
        pass

    # Allow private IPs for development/testing (still blocks cloud metadata)
    safe_url = validate_safe_url("http://localhost:8080", allow_private=True)
"""

import ipaddress
import os
import socket
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    BeforeValidator,
    HttpUrl,
)

# Private IP ranges (RFC 1918, RFC 4193, RFC 3927, loopback)
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (includes cloud metadata)
    ipaddress.ip_network("0.0.0.0/8"),  # Current network
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
]

# Cloud provider metadata endpoints
CLOUD_METADATA_IPS = [
    "169.254.169.254",  # AWS, GCP, Azure, DigitalOcean, Oracle Cloud
    "169.254.170.2",  # AWS ECS task metadata
    "100.100.100.200",  # Alibaba Cloud metadata
]

CLOUD_METADATA_HOSTNAMES = [
    "metadata.google.internal",  # GCP
    "metadata",  # Generic
    "instance-data",  # AWS EC2
]

# Localhost variations
LOCALHOST_NAMES = [
    "localhost",
    "localhost.localdomain",
]


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private range.

    Args:
        ip_str: IP address as a string (e.g., "192.168.1.1")

    Returns:
        True if IP is in a private range, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in range_ for range_ in PRIVATE_IP_RANGES)
    except ValueError:
        return False


def is_cloud_metadata(hostname: str, ip_str: str | None = None) -> bool:
    """Check if hostname or IP is a cloud metadata endpoint.

    Args:
        hostname: Hostname to check
        ip_str: Optional IP address to check

    Returns:
        True if hostname or IP is a known cloud metadata endpoint
    """
    # Check hostname
    if hostname.lower() in CLOUD_METADATA_HOSTNAMES:
        return True

    # Check IP
    if ip_str and ip_str in CLOUD_METADATA_IPS:  # noqa: SIM103
        return True

    return False


def is_localhost(hostname: str, ip_str: str | None = None) -> bool:
    """Check if hostname or IP is localhost.

    Args:
        hostname: Hostname to check
        ip_str: Optional IP address to check

    Returns:
        True if hostname or IP is localhost
    """
    # Check hostname
    if hostname.lower() in LOCALHOST_NAMES:
        return True

    # Check IP
    if ip_str:
        try:
            ip = ipaddress.ip_address(ip_str)
            # Check if loopback
            if ip.is_loopback:
                return True
            # Also check common localhost IPs
            if ip_str in ("127.0.0.1", "::1", "0.0.0.0"):  # noqa: S104
                return True
        except ValueError:
            pass

    return False


def validate_safe_url(
    url: str | AnyHttpUrl,
    *,
    allow_private: bool = False,
    allow_http: bool = True,
) -> str:
    """Validate a URL for SSRF protection.

    This function validates URLs to prevent Server-Side Request Forgery (SSRF) attacks
    by blocking requests to private networks and cloud metadata endpoints.

    Args:
        url: The URL to validate (string or Pydantic HttpUrl)
        allow_private: If True, allows private IPs and localhost (for development).
                      Cloud metadata endpoints are ALWAYS blocked.
        allow_http: If True, allows both HTTP and HTTPS. If False, only HTTPS.

    Returns:
        The validated URL as a string

    Raises:
        ValueError: If URL is invalid or potentially dangerous

    Examples:
        >>> validate_safe_url("https://hooks.slack.com/services/xxx")
        'https://hooks.slack.com/services/xxx'

        >>> validate_safe_url("http://127.0.0.1:8080")
        ValueError: Localhost URLs are not allowed

        >>> validate_safe_url("http://192.168.1.1")
        ValueError: URL resolves to private IP: 192.168.1.1

        >>> validate_safe_url("http://169.254.169.254/latest/meta-data/")
        ValueError: URL resolves to cloud metadata IP: 169.254.169.254

        >>> validate_safe_url("http://localhost:8080", allow_private=True)
        'http://localhost:8080'
    """
    url_str = str(url)
    parsed = urlparse(url_str)

    # Validate URL scheme
    if not allow_http and parsed.scheme != "https":
        msg = "Only HTTPS URLs are allowed"
        raise ValueError(msg)

    if parsed.scheme not in ("http", "https"):
        msg = f"Only HTTP/HTTPS URLs are allowed, got scheme: {parsed.scheme}"
        raise ValueError(msg)

    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        msg = "URL must have a valid hostname"
        raise ValueError(msg)

    # Special handling for test environments - allow test server hostnames
    # testserver is used by FastAPI/Starlette test clients and doesn't resolve via DNS
    # Only enabled when LANGCHAIN_ENV=local_test (set in conftest.py)
    if (
        os.environ.get("LANGCHAIN_ENV") == "local_test"
        and hostname.startswith("test")
        and "server" in hostname
    ):
        return url_str

    # ALWAYS block cloud metadata endpoints (even with allow_private=True)
    if is_cloud_metadata(hostname):
        msg = f"Cloud metadata endpoints are not allowed: {hostname}"
        raise ValueError(msg)

    # Check for localhost
    if is_localhost(hostname) and not allow_private:
        msg = f"Localhost URLs are not allowed: {hostname}"
        raise ValueError(msg)

    # Resolve hostname to IP addresses and validate each one.
    # Note: DNS resolution results are cached by the OS, so repeated calls are fast.
    try:
        # Get all IP addresses for this hostname
        addr_info = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            socket.AF_UNSPEC,  # Allow both IPv4 and IPv6
            socket.SOCK_STREAM,
        )

        for result in addr_info:
            ip_str: str = result[4][0]  # type: ignore[assignment]

            # ALWAYS block cloud metadata IPs
            if is_cloud_metadata(hostname, ip_str):
                msg = f"URL resolves to cloud metadata IP: {ip_str}"
                raise ValueError(msg)

            # Check for localhost IPs
            if is_localhost(hostname, ip_str) and not allow_private:
                msg = f"URL resolves to localhost IP: {ip_str}"
                raise ValueError(msg)

            # Check for private IPs
            if not allow_private and is_private_ip(ip_str):
                msg = f"URL resolves to private IP address: {ip_str}"
                raise ValueError(msg)

    except socket.gaierror as e:
        # DNS resolution failed - fail closed for security
        msg = f"Failed to resolve hostname '{hostname}': {e}"
        raise ValueError(msg) from e
    except OSError as e:
        # Other network errors - fail closed
        msg = f"Network error while validating URL: {e}"
        raise ValueError(msg) from e

    return url_str


def is_safe_url(
    url: str | AnyHttpUrl,
    *,
    allow_private: bool = False,
    allow_http: bool = True,
) -> bool:
    """Check if a URL is safe (non-throwing version of validate_safe_url).

    Args:
        url: The URL to check
        allow_private: If True, allows private IPs and localhost
        allow_http: If True, allows both HTTP and HTTPS

    Returns:
        True if URL is safe, False otherwise

    Examples:
        >>> is_safe_url("https://example.com")
        True

        >>> is_safe_url("http://127.0.0.1:8080")
        False

        >>> is_safe_url("http://localhost:8080", allow_private=True)
        True
    """
    try:
        validate_safe_url(url, allow_private=allow_private, allow_http=allow_http)
    except ValueError:
        return False
    else:
        return True


def _validate_url_ssrf_strict(v: Any) -> Any:
    """Validate URL for SSRF protection (strict mode)."""
    if isinstance(v, str):
        validate_safe_url(v, allow_private=False, allow_http=True)
    return v


def _validate_url_ssrf_https_only(v: Any) -> Any:
    """Validate URL for SSRF protection (HTTPS only, strict mode)."""
    if isinstance(v, str):
        validate_safe_url(v, allow_private=False, allow_http=False)
    return v


def _validate_url_ssrf_relaxed(v: Any) -> Any:
    """Validate URL for SSRF protection (relaxed mode - allows private IPs)."""
    if isinstance(v, str):
        validate_safe_url(v, allow_private=True, allow_http=True)
    return v


# Annotated types with SSRF protection
SSRFProtectedUrl = Annotated[HttpUrl, BeforeValidator(_validate_url_ssrf_strict)]
"""A Pydantic HttpUrl type with built-in SSRF protection.

This blocks private IPs, localhost, and cloud metadata endpoints.

Example:
    class WebhookSchema(BaseModel):
        url: SSRFProtectedUrl  # Automatically validated for SSRF
        headers: dict[str, str] | None = None
"""

SSRFProtectedUrlRelaxed = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_relaxed)
]
"""A Pydantic HttpUrl with relaxed SSRF protection (allows private IPs).

Use this for development/testing webhooks where localhost/private IPs are needed.
Cloud metadata endpoints are still blocked.

Example:
    class DevWebhookSchema(BaseModel):
        url: SSRFProtectedUrlRelaxed  # Allows localhost, blocks cloud metadata
"""

SSRFProtectedHttpsUrl = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_https_only)
]
"""A Pydantic HttpUrl with SSRF protection that only allows HTTPS.

This blocks private IPs, localhost, cloud metadata endpoints, and HTTP URLs.

Example:
    class SecureWebhookSchema(BaseModel):
        url: SSRFProtectedHttpsUrl  # Only HTTPS, blocks private IPs
"""

SSRFProtectedHttpsUrlStr = Annotated[
    str, BeforeValidator(_validate_url_ssrf_https_only)
]
"""A string type with SSRF protection that only allows HTTPS URLs.

Same as SSRFProtectedHttpsUrl but returns a string instead of HttpUrl.
Useful for FastAPI query parameters where you need a string URL.

Example:
    @router.get("/proxy")
    async def proxy_get(url: SSRFProtectedHttpsUrlStr):
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
"""
