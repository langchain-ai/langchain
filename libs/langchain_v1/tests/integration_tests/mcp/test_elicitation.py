"""Tests for MCP elicitation callback support."""

from mcp.client.session import ClientRequestContext as RequestContext
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ElicitRequestParams, ElicitResult
from pydantic import BaseModel

from langchain.mcp.callbacks import CallbackContext, Callbacks
from langchain.mcp.client import MultiServerMCPClient
from tests.integration_tests.mcp.utils import run_streamable_http


def _create_elicitation_server():
    class UserDetails(BaseModel):
        email: str
        age: int

    server = MCPServer()

    # Track how many times code before elicit runs (should be exactly once)
    server._pre_elicit_call_count = 0

    @server.tool()
    async def create_profile(name: str, ctx: Context) -> str:
        """Create a user profile with elicitation."""
        # This code should only run once, not be re-executed after elicitation
        server._pre_elicit_call_count += 1

        result = await ctx.elicit(
            message=f"Please provide details for {name}'s profile:",
            schema=UserDetails,
        )
        if result.action == "accept" and result.data:
            return (
                f"Created profile for {name}: "
                f"email={result.data.email}, age={result.data.age}, "
                f"pre_elicit_calls={server._pre_elicit_call_count}"
            )
        if result.action == "decline":
            return (
                f"User declined. Created minimal profile for {name}. "
                f"pre_elicit_calls={server._pre_elicit_call_count}"
            )
        return f"Profile creation cancelled. pre_elicit_calls={server._pre_elicit_call_count}"

    return server


async def test_elicitation_callback_accept(socket_enabled) -> None:
    """Test elicitation callback with user accepting and providing data."""
    elicitation_requests: list[tuple[RequestContext, ElicitRequestParams, CallbackContext]] = []

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        elicitation_requests.append((mcp_context, params, context))
        return ElicitResult(
            action="accept",
            content={"email": "alice@example.com", "age": 28},
        )

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "http",
                    "mode": "legacy",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "create_profile"

        # Call the tool
        result = await tools[0].ainvoke(
            {"args": {"name": "Alice"}, "id": "call_1", "type": "tool_call"}
        )

        # Verify elicitation callback was called
        assert len(elicitation_requests) == 1
        _, params, context = elicitation_requests[0]
        assert "Alice" in params.message
        assert context.server_name == "test"
        assert context.tool_name == "create_profile"

        # Verify result
        assert "alice@example.com" in str(result.content)
        assert "28" in str(result.content)

        # Verify code before ctx.elicit only ran once
        # (not re-executed after elicitation)
        assert "pre_elicit_calls=1" in str(result.content)


async def test_elicitation_callback_decline(socket_enabled) -> None:
    """Test elicitation callback with user declining."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="decline")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "http",
                    "mode": "legacy",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Bob"}, "id": "call_2", "type": "tool_call"}
        )

        assert "declined" in str(result.content).lower()
        # Verify code before ctx.elicit only ran once
        assert "pre_elicit_calls=1" in str(result.content)


async def test_elicitation_callback_cancel(socket_enabled) -> None:
    """Test elicitation callback with user cancelling."""

    async def on_elicitation(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        return ElicitResult(action="cancel")

    with run_streamable_http(_create_elicitation_server, 8184):
        client = MultiServerMCPClient(
            {
                "test": {
                    "url": "http://localhost:8184/mcp",
                    "transport": "http",
                    "mode": "legacy",
                }
            },
            callbacks=Callbacks(on_elicitation=on_elicitation),
        )

        tools = await client.get_tools()
        result = await tools[0].ainvoke(
            {"args": {"name": "Charlie"}, "id": "call_3", "type": "tool_call"}
        )

        assert "cancelled" in str(result.content).lower()
        # Verify code before ctx.elicit only ran once
        assert "pre_elicit_calls=1" in str(result.content)
