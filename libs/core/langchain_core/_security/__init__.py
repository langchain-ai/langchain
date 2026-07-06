"""SSRF protection and security utilities.

This is an **internal** module (note the `_security` prefix). It is NOT part of
the public `langchain-core` API and may change or be removed at any time without
notice. External code should not import from or depend on anything in this
module. Any vulnerability reports should target the public APIs that use these
utilities, not this internal module directly.
"""

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
