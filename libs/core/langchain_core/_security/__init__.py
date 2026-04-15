"""lc_security — SSRF protection and security utilities."""

from langchain_core._security._exceptions import SSRFBlockedError
from langchain_core._security._policy import (
    SSRFPolicy,
    validate_hostname,
    validate_resolved_ip,
    validate_url,
    validate_url_sync,
)
from langchain_core._security._transport import (
    SSRFSafeSyncTransport,
    SSRFSafeTransport,
    ssrf_safe_async_client,
    ssrf_safe_client,
)

__all__ = [
    "SSRFBlockedError",
    "SSRFPolicy",
    "SSRFSafeSyncTransport",
    "SSRFSafeTransport",
    "ssrf_safe_async_client",
    "ssrf_safe_client",
    "validate_hostname",
    "validate_resolved_ip",
    "validate_url",
    "validate_url_sync",
]
