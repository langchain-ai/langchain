"""Tiny MCP servers the examples share, so each example stays about one idea.

Nothing here is specific to LangChain — these are plain FastMCP servers.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING

import uvicorn
from fastmcp import Context, FastMCP
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    TextContent,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def free_port() -> int:
    """Pick an unused loopback port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def serve_http(mcp: FastMCP[None], port: int) -> Iterator[str]:
    """Serve `mcp` over streamable HTTP for the duration of the block.

    Yields the URL to point a client at. Runs uvicorn on a daemon thread so the
    example stays a single straightforward script.
    """
    server = uvicorn.Server(
        uvicorn.Config(mcp.http_app(), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


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
