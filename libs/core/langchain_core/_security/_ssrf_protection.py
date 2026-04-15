"""SSRF Protection - thin wrapper raising ValueError for internal callers.

Delegates all validation to `langchain_core._security._policy`.
"""

import os
import socket
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import (
    AnyHttpUrl,
    BeforeValidator,
    HttpUrl,
)

from langchain_core._security._exceptions import SSRFBlockedError
from langchain_core._security._policy import (
    SSRFPolicy,
)
from langchain_core._security._policy import (
    validate_resolved_ip as _validate_resolved_ip,
)
from langchain_core._security._policy import (
    validate_url_sync as _validate_url_sync,
)


def _policy_for(*, allow_private: bool, allow_http: bool) -> SSRFPolicy:
    """Build an `SSRFPolicy` from the legacy flag interface."""
    schemes = frozenset({"http", "https"}) if allow_http else frozenset({"https"})
    return SSRFPolicy(
        allowed_schemes=schemes,
        block_private_ips=not allow_private,
        block_localhost=not allow_private,
        block_cloud_metadata=True,
        block_k8s_internal=True,
    )


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
        url: The URL to validate (string or Pydantic HttpUrl).
        allow_private: If ``True``, allows private IPs and localhost (for development).
                      Cloud metadata endpoints are ALWAYS blocked.
        allow_http: If ``True``, allows both HTTP and HTTPS.  If ``False``, only HTTPS.

    Returns:
        The validated URL as a string.

    Raises:
        ValueError: If URL is invalid or potentially dangerous.
    """
    url_str = str(url)
    parsed = urlparse(url_str)
    hostname = parsed.hostname or ""

    # Test-environment bypass (preserved from original implementation)
    if (
        os.environ.get("LANGCHAIN_ENV") == "local_test"
        and hostname.startswith("test")
        and "server" in hostname
    ):
        return url_str

    policy = _policy_for(allow_private=allow_private, allow_http=allow_http)

    # Synchronous scheme + hostname checks
    try:
        _validate_url_sync(url_str, policy)
    except SSRFBlockedError as exc:
        raise ValueError(str(exc)) from exc

    # DNS resolution and IP validation
    try:
        addr_info = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )

        for result in addr_info:
            ip_str: str = result[4][0]  # type: ignore[assignment]
            try:
                _validate_resolved_ip(ip_str, policy)
            except SSRFBlockedError as exc:
                raise ValueError(str(exc)) from exc

    except socket.gaierror as e:
        msg = f"Failed to resolve hostname '{hostname}': {e}"
        raise ValueError(msg) from e
    except OSError as e:
        msg = f"Network error while validating URL: {e}"
        raise ValueError(msg) from e

    return url_str


def is_safe_url(
    url: str | AnyHttpUrl,
    *,
    allow_private: bool = False,
    allow_http: bool = True,
) -> bool:
    """Non-throwing version of `validate_safe_url`."""
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
SSRFProtectedUrlRelaxed = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_relaxed)
]
SSRFProtectedHttpsUrl = Annotated[
    HttpUrl, BeforeValidator(_validate_url_ssrf_https_only)
]
SSRFProtectedHttpsUrlStr = Annotated[
    str, BeforeValidator(_validate_url_ssrf_https_only)
]
