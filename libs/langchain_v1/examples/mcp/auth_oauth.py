"""A server behind a full OAuth 2.1 flow, with no client credentials up front.

This one process plays two roles that are normally separate:

- Resource Server: serves `/mcp` and rejects unauthenticated calls with a 401
  pointing at its protected-resource metadata.
- Authorization Server: serves discovery, `/register`, `/authorize`, `/token`.

In production those split — the resource server is yours, the authorization
server is Auth0 / WorkOS / Okta. FastMCP ships providers for those, and the flow
the client runs is identical, which is the point of the spec.

Dynamic Client Registration is what makes this feel plug-and-play: the client
registers itself at runtime instead of you pre-provisioning a client ID.
`InMemoryOAuthProvider` auto-approves, so the browser tab opens and redirects
straight back; a real server would show login and consent there.

    uv run examples/mcp/auth_oauth.py

Opens a browser window to complete the flow.
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.utilities.tests import run_server_in_process
from mcp.server.auth.settings import ClientRegistrationOptions

from langchain.mcp import MCPAdapter


def run_oauth_server(host: str, port: int) -> None:
    """Serve a calendar server that mints its own tokens."""
    # `base_url` must match the URL clients actually reach: it is advertised in
    # discovery and used as the OAuth `resource`.
    auth = InMemoryOAuthProvider(
        base_url=f"http://127.0.0.1:{port}",
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=["calendar:read"],
            default_scopes=["calendar:read"],
        ),
        required_scopes=["calendar:read"],
    )
    mcp: FastMCP[None] = FastMCP("calendar", auth=auth)

    @mcp.tool
    def whoami() -> str:
        """Report the identity the server derived from the access token."""
        token = get_access_token()
        if token is None:
            return "unauthenticated"
        return f"client_id={token.client_id} scopes={token.scopes}"

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")


async def main() -> None:
    """Register, authorize, and call the guarded tool."""
    with run_server_in_process(run_oauth_server) as url:
        # "oauth" runs discovery, dynamic registration, the browser redirect,
        # and the token exchange. Tokens are held in memory, so each run
        # repeats the browser step; pass `OAuth(..., token_storage=...)` to
        # persist them.
        async with MCPAdapter(Client(f"{url}/mcp", auth="oauth")) as adapter:
            [whoami] = await adapter.list_tools()
            [block] = await whoami.ainvoke({})
            print("authenticated as:", block["text"])


if __name__ == "__main__":
    asyncio.run(main())
