"""Tiny MCP servers the examples share, so each example stays about one idea.

Nothing here is specific to LangChain — these are plain FastMCP servers. The
`run_*` functions exist because a server reached over stdio or HTTP has to be
started as its own process, and that entry point cannot be a lambda.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.dependencies import get_access_token
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    TextContent,
    ToolAnnotations,
)
from pydantic import SecretStr


def weather_server() -> FastMCP[None]:
    """A server with one tool that always succeeds."""
    mcp: FastMCP[None] = FastMCP("weather")

    @mcp.tool
    def get_forecast(city: str) -> str:
        """Report the forecast for a city."""
        return f"{city}: 18C and clear."

    return mcp


def calculator_server() -> FastMCP[None]:
    """A server whose tool reports failure for input it cannot handle."""
    mcp: FastMCP[None] = FastMCP("calculator")

    @mcp.tool
    def divide(numerator: float, denominator: float) -> str:
        """Divide two numbers."""
        if denominator == 0:
            # Raising inside a tool becomes an MCP error result (`isError=True`)
            # rather than a transport failure, which is what lets the agent see
            # it and retry. See `tool_errors.py`.
            msg = "Cannot divide by zero. Choose a non-zero denominator."
            raise ValueError(msg)
        return str(numerator / denominator)

    return mcp


def files_server() -> FastMCP[None]:
    """A server with a read-only tool and a destructive one.

    `delete_file` sets `destructiveHint=True`; the adapter surfaces that as
    `metadata["mcp"]["tool"]["annotations"]["destructive_hint"]`, which
    `destructive_interrupt.py` uses to gate the tool behind human approval.
    """
    mcp: FastMCP[None] = FastMCP("files")

    @mcp.tool
    def list_files() -> list[str]:
        """List the files in the workspace."""
        return ["report.md", "notes.txt"]

    @mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
    def delete_file(path: str) -> str:
        """Delete a file from the workspace."""
        return f"Deleted {path}."

    return mcp


def booking_server() -> FastMCP[None]:
    """A server whose tool cannot finish without an answer from a human.

    Uses the guard pattern: the tool checks whether the answer it needs has
    arrived and, if not, returns an `InputRequiredResult` describing the
    question instead of doing any work. Returning early is what makes the call
    safe to replay when the run resumes.
    """
    mcp: FastMCP[None] = FastMCP("booking")

    @mcp.tool
    async def book_table(party_size: int, ctx: Context) -> list[TextContent] | InputRequiredResult:
        """Book a restaurant table. Asks the user which date to book."""
        answers = ctx.input_responses
        if not answers or "date" not in answers:
            return InputRequiredResult(
                input_requests={
                    "date": ElicitRequest(
                        method="elicitation/create",
                        params=ElicitRequestFormParams(
                            mode="form",
                            message="What date would you like to book?",
                            requested_schema={
                                "type": "object",
                                "properties": {"date": {"type": "string", "format": "date"}},
                                "required": ["date"],
                            },
                        ),
                    )
                },
                request_state="awaiting-date",
            )

        answer = answers["date"]
        if not isinstance(answer, ElicitResult) or answer.action != "accept" or not answer.content:
            return [TextContent(type="text", text="No date given, so nothing was booked.")]
        date = answer.content["date"]
        return [TextContent(type="text", text=f"Booked a table for {party_size} on {date}.")]

    return mcp


def run_weather_stdio() -> None:
    """Entry point for a weather server spoken to over stdio."""
    weather_server().run()


def run_calculator_stdio() -> None:
    """Entry point for a calculator server spoken to over stdio."""
    calculator_server().run()


def run_weather_http(host: str, port: int) -> None:
    """Entry point for a weather server served over HTTP."""
    weather_server().run(
        transport="http", host=host, port=port, show_banner=False, log_level="warning"
    )


def run_calculator_http(host: str, port: int) -> None:
    """Entry point for a calculator server served over HTTP."""
    calculator_server().run(
        transport="http", host=host, port=port, show_banner=False, log_level="warning"
    )


ISSUER = "https://demo.issuer"
AUDIENCE = "mcp-fleet"


def _load_or_generate_keys() -> RSAKeyPair:
    """Return one keypair shared across processes, or a fresh one per process.

    A single process (the `run_server_in_process` examples) can generate its
    own keypair and pass its public half to the server. A multi-process demo
    cannot: the `langgraph dev` worker that mints tokens and the server process
    that verifies them are different interpreters. Point both at the same PEM
    file via `MCP_DEMO_KEYFILE` and they share one keypair; leave it unset and
    every process keeps its own, exactly as before.
    """
    keyfile = os.environ.get("MCP_DEMO_KEYFILE")
    if keyfile:
        data = json.loads(Path(keyfile).read_text())
        return RSAKeyPair(private_key=SecretStr(data["private_key"]), public_key=data["public_key"])
    return RSAKeyPair.generate()


_KEYS = _load_or_generate_keys()
"""Stands in for an identity provider, so the examples can mint real tokens."""

PUBLIC_KEY = _KEYS.public_key
"""Pass this to `run_guarded_server`; see why in its docstring."""


def token_for(user: str) -> str:
    """Mint an access token identifying `user`.

    The stand-in for whatever a deployment already has: an OAuth gateway that
    exchanges for a per-user token, or a provider from `fastmcp.client.auth` —
    `OAuth(mcp_url=..., token_storage=<per-user store>)` runs the full
    authorization-code flow per identity, which `auth_oauth.py` demonstrates.
    """
    return _KEYS.create_token(subject=user, issuer=ISSUER, audience=AUDIENCE)


def run_guarded_server(host: str, port: int, name: str, public_key: str) -> None:
    """Serve a token-guarded server whose one tool reports who called it.

    The key is a parameter rather than read from this module: the server runs
    in its own process, which re-imports this file and would otherwise mint a
    second, unrelated key pair.
    """
    mcp: FastMCP[None] = FastMCP(
        name,
        auth=JWTVerifier(public_key=public_key, issuer=ISSUER, audience=AUDIENCE),
        # Opt into client-side response caching (SEP-2549): the server stamps
        # each cacheable result with this TTL and a `private` scope, so a client
        # holding a `CacheConfig` can serve a repeat `tools/list` from its own
        # store instead of the wire. Without these hints the server sends
        # `ttlMs: 0` and nothing caches, which is why arbitrary third-party
        # servers cannot be relied on for this.
        cache_ttl=60,
        cache_scope="private",
    )

    @mcp.tool
    def whoami() -> str:
        """Report the identity the server derived from the access token."""
        token = get_access_token()
        return "unauthenticated" if token is None else str(token.claims.get("sub"))

    mcp.run(transport="http", host=host, port=port, show_banner=False, log_level="warning")
