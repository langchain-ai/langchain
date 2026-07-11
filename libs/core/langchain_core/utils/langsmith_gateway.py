"""Authentication helpers for LangSmith Gateway clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator


class LangSmithGatewayOAuth(httpx.Auth):
    """Apply a LangSmith OAuth token to requests sent through Gateway.

    Gateway removes this credential before forwarding the request to the selected
    provider. The adapter therefore removes provider credential headers before
    setting the Gateway-specific bearer token.

    Args:
        token: LangSmith OAuth access token.
    """

    def __init__(self, token: str) -> None:
        """Initialize the auth adapter with a LangSmith OAuth access token."""
        self._token = token

    def _authorize(self, request: httpx.Request) -> httpx.Request:
        request.headers.pop("Authorization", None)
        request.headers.pop("X-Api-Key", None)
        request.headers["Authorization"] = f"Bearer {self._token}"
        return request

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Add the bearer token to a synchronous request."""
        yield self._authorize(request)

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Add the bearer token to an asynchronous request."""
        yield self._authorize(request)
