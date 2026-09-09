"""A server behind a static bearer token.

The simple end of MCP auth: the server never issues credentials, it only
verifies the token arriving in `Authorization: Bearer <token>`. No discovery,
no browser, no refresh — you provisioned the token out of band and the client
presents it.

    uv run examples/mcp/auth_bearer.py
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.utilities.tests import run_server_in_process

from langchain.mcp import MCPAdapter

TOKEN = "demo-weather-token"  # noqa: S105


def run_guarded_server(host: str, port: int) -> None:
    """Serve a weather server that only answers callers presenting `TOKEN`."""
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

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")


async def main() -> None:
    """Call the guarded server with, and without, the token."""
    with run_server_in_process(run_guarded_server) as url:
        # `auth` takes a bearer token string, the literal "oauth", or any
        # `httpx2.Auth`. A config dict takes the same key per server.
        async with MCPAdapter(Client(f"{url}/mcp", auth=TOKEN)) as adapter:
            [forecast] = await adapter.list_tools()
            [block] = await forecast.ainvoke({"city": "Oslo"})
            print("with token:   ", block["text"])

        try:
            async with MCPAdapter(f"{url}/mcp") as adapter:
                await adapter.list_tools()
        except Exception as exc:
            print("without token:", type(exc).__name__)


if __name__ == "__main__":
    asyncio.run(main())
