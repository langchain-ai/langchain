"""Private helpers for limiting gateway requests to their original origin."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from pydantic import SecretStr

from langchain_core.utils._gateway import (
    _GATEWAY_RECONSTRUCTION_PROTECTION,
    _langsmith_gateway_active,
    _resolve_gateway_base_url,
)


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or not hostname:
        msg = "Gateway requests require an absolute HTTP URL."
        raise ValueError(msg)
    hostname = hostname.encode("idna").decode("ascii").lower()
    port = (
        parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
    )
    return scheme, hostname, port


def _protect_gateway_redirects(
    base_url: str | None, api_key: Any, provider_path: str
) -> bool:
    """Identify gateway credentials or endpoints while configuration is active."""
    if _langsmith_gateway_active() or _GATEWAY_RECONSTRUCTION_PROTECTION.get():
        return True
    gateway_url = _resolve_gateway_base_url(provider_path)
    if gateway_url is None:
        return False
    if base_url is not None and _origin(base_url) == _origin(gateway_url):
        return True
    key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    gateway_key = os.getenv("LANGSMITH_GATEWAY_API_KEY")
    return bool(gateway_key and isinstance(key, str) and key == gateway_key)


def _gateway_request_hooks(
    base_url: str | None, *, is_async: bool
) -> dict[str, list[Any]]:
    """Reject requests outside the gateway origin before the transport sees them."""
    if base_url is None:
        msg = "Gateway requests require an absolute HTTP URL."
        raise ValueError(msg)
    origin = _origin(base_url)

    def check(request: Any) -> None:
        if _origin(str(request.url)) != origin:
            msg = "Cross-origin redirects are not allowed for gateway requests."
            raise ValueError(msg)

    async def async_check(request: Any) -> None:
        check(request)

    return {"request": [async_check if is_async else check]}
