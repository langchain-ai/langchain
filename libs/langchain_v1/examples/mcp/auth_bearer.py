"""A server behind a static bearer token.

The simple end of MCP auth: the server never issues credentials, it only
verifies the token arriving in `Authorization: Bearer <token>`. No discovery,
no browser, no refresh — you provisioned the token out of band and the client
presents it.

    uv run examples/mcp/auth_bearer.py
"""

from __future__ import annotations

import asyncio

from _servers import free_port, serve_http
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from langchain.mcp import MCPAdapter

TOKEN = "demo-weather-token"  # noqa: S105


def guarded_server() -> FastMCP[None]:
    """A weather server that only answers callers presenting `TOKEN`."""
    # Demo only — `StaticTokenVerifier` holds tokens in plaintext memory. A real
    # deployment uses `JWTVerifier` against an IdP's JWKS, or OAuth (see
    # `auth_oauth.py`).
    mcp: FastMCP[None] = FastMCP(
        "weather",
        auth=StaticTokenVerifier(tokens={TOKEN: {"client_id": "demo", "scopes": ["read"]}}),
    )

    @mcp.tool
    def get_forecast(city: str) -> str:
        """Report the forecast for a city."""
        return f"{city}: 18C and clear."

    return mcp


async def main() -> None:
    """Call the guarded server with, and without, the token."""
    with serve_http(guarded_server(), free_port()) as url:
        # `auth` accepts a bearer token string, the literal "oauth", or any
        # `httpx2.Auth`. Config dicts take the same key.
        async with MCPAdapter(Client(url, auth=TOKEN)) as adapter:
            [forecast] = await adapter.get_tools()
            [block] = await forecast.ainvoke({"city": "Oslo"})
            print("with token:   ", block["text"])

        try:
            async with MCPAdapter(url) as adapter:
                await adapter.get_tools()
        except Exception as exc:
            print("without token:", type(exc).__name__)


if __name__ == "__main__":
    asyncio.run(main())
