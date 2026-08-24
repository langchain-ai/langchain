import os
from pathlib import Path

import pytest
from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from mcp import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.shared._httpx_utils import create_mcp_http_client

from langchain.mcp.client import MultiServerMCPClient
from langchain.mcp.tools import load_mcp_tools
from tests.integration_tests.mcp.utils import IsLangChainID, run_streamable_http


async def test_multi_server_mcp_client(
    socket_enabled,
):
    """Test that MultiServerMCPClient can connect to multiple servers and load tools."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")
    time_server_path = os.path.join(current_dir, "servers/time_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python3",
                "args": [weather_server_path],
                "transport": "stdio",
            },
            "time": {
                "command": "python3",
                "args": [time_server_path],
                "transport": "stdio",
            },
        },
    )
    # Check that we have tools from both servers
    all_tools = await client.get_tools()

    # Should have 3 tools (add, multiply, get_weather)
    assert len(all_tools) == 4

    # Check that tools are BaseTool instances
    for tool in all_tools:
        assert isinstance(tool, BaseTool)

    # Verify tool names
    tool_names = {tool.name for tool in all_tools}
    assert tool_names == {"add", "multiply", "get_weather", "get_time"}

    # Check math server tools
    math_tools = await client.get_tools(server_name="math")
    assert len(math_tools) == 2
    math_tool_names = {tool.name for tool in math_tools}
    assert math_tool_names == {"add", "multiply"}

    # Check weather server tools
    weather_tools = await client.get_tools(server_name="weather")
    assert len(weather_tools) == 1
    assert weather_tools[0].name == "get_weather"

    # Check time server tools
    time_tools = await client.get_tools(server_name="time")
    assert len(time_tools) == 1
    assert time_tools[0].name == "get_time"

    # Test that we can call a math tool
    add_tool = next(tool for tool in all_tools if tool.name == "add")
    result = await add_tool.ainvoke({"a": 2, "b": 3})
    assert result == [{"type": "text", "text": "5", "id": IsLangChainID}]

    # Test that we can call a weather tool
    weather_tool = next(tool for tool in all_tools if tool.name == "get_weather")
    result = await weather_tool.ainvoke({"location": "London"})
    assert result == [{"type": "text", "text": "It's always sunny in London", "id": IsLangChainID}]

    # Test the multiply tool
    multiply_tool = next(tool for tool in all_tools if tool.name == "multiply")
    result = await multiply_tool.ainvoke({"a": 4, "b": 5})
    assert result == [{"type": "text", "text": "20", "id": IsLangChainID}]

    # Test that we can call a time tool
    time_tool = next(tool for tool in all_tools if tool.name == "get_time")
    result = await time_tool.ainvoke({"args": ""})
    assert result == [{"type": "text", "text": "5:20:00 PM EST", "id": IsLangChainID}]


async def test_multi_server_connect_methods(
    socket_enabled,
):
    """Test the different connect methods for MultiServerMCPClient."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    time_server_path = os.path.join(current_dir, "servers/time_server.py")

    # Initialize client without initial connections
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "time": {
                "command": "python3",
                "args": [time_server_path],
                "transport": "stdio",
            },
        },
    )
    tool_names = set()
    async with client.session("math") as session:
        tools = await load_mcp_tools(session)
        assert len(tools) == 2
        result = await tools[0].ainvoke({"a": 2, "b": 3})
        assert result == [{"type": "text", "text": "5", "id": IsLangChainID}]

        for tool in tools:
            tool_names.add(tool.name)

    async with client.session("time") as session:
        tools = await load_mcp_tools(session)
        assert len(tools) == 1
        result = await tools[0].ainvoke({"args": ""})
        assert result == [{"type": "text", "text": "5:20:00 PM EST", "id": IsLangChainID}]

        for tool in tools:
            tool_names.add(tool.name)

    assert tool_names == {"add", "multiply", "get_time"}


async def test_get_prompt():
    """Test retrieving prompts from MCP servers."""
    # Get the absolute path to the server scripts
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            }
        },
    )
    # Test getting a prompt from the math server
    messages = await client.get_prompt(
        "math",
        "configure_assistant",
        arguments={"skills": "math, addition, multiplication"},
    )

    # Check that we got an AIMessage back
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert "You are a helpful assistant" in messages[0].content
    assert "math, addition, multiplication" in messages[0].content


async def test_get_resources_from_all_servers():
    """Test that get_resources loads resources from all servers."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python3",
                "args": [weather_server_path],
                "transport": "stdio",
            },
        },
    )

    # Get all resources from all servers (no server_name specified)
    all_resources = await client.get_resources()

    # Should have resources from both servers
    assert len(all_resources) == 2
    assert all(isinstance(r, Blob) for r in all_resources)

    # Verify we have resources from both servers
    resource_uris = {str(r.metadata["uri"]) for r in all_resources}
    assert resource_uris == {"math://formulas", "weather://forecast"}

    # Verify resource content
    math_resource = next(r for r in all_resources if str(r.metadata["uri"]) == "math://formulas")
    weather_resource = next(
        r for r in all_resources if str(r.metadata["uri"]) == "weather://forecast"
    )
    assert math_resource.data == "E = mc^2"
    assert weather_resource.data == "Sunny with a chance of clouds"


async def test_get_resources_from_specific_server():
    """Test that get_resources loads resources from a specific server."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")
    weather_server_path = os.path.join(current_dir, "servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python3",
                "args": [math_server_path],
                "transport": "stdio",
            },
            "weather": {
                "command": "python3",
                "args": [weather_server_path],
                "transport": "stdio",
            },
        },
    )

    # Get resources from math server only
    math_resources = await client.get_resources(server_name="math")
    assert len(math_resources) == 1
    assert str(math_resources[0].metadata["uri"]) == "math://formulas"
    assert math_resources[0].data == "E = mc^2"

    # Get resources from weather server only
    weather_resources = await client.get_resources(server_name="weather")
    assert len(weather_resources) == 1
    assert str(weather_resources[0].metadata["uri"]) == "weather://forecast"
    assert weather_resources[0].data == "Sunny with a chance of clouds"


async def test_connections_accept_sdk_server_parameters():
    """The MCP SDK's own parameter models work anywhere a connection mapping does."""
    current_dir = Path(__file__).parent
    math_server_path = os.path.join(current_dir, "servers/math_server.py")

    client = MultiServerMCPClient(
        {
            "math": StdioServerParameters(
                command="python3",
                args=[math_server_path],
            ),
        },
    )

    tools = await client.get_tools()
    assert {"add", "multiply"} <= {tool.name for tool in tools}


async def test_connections_accept_a_transport_factory(socket_enabled):
    """A factory composes a server endpoint with a fully configured HTTP client.

    The SDK's own HTTP parameter models describe neither `auth` nor an `http_client`, so
    this is how the two layers are brought together. It must be a factory rather than a
    transport, because a session is opened per operation and transports are single-use.
    """

    def transport():
        return streamable_http_client(
            "http://localhost:8187/mcp",
            http_client=create_mcp_http_client(headers={"X-Test": "1"}),
        )

    with run_streamable_http(_create_time_server, 8187):
        client = MultiServerMCPClient({"time": transport})

        tools = {tool.name: tool for tool in await client.get_tools()}
        assert list(tools) == ["get_time"]

        # A second operation opens another session, which is why a factory is required.
        result = await tools["get_time"].ainvoke({"args": {}, "id": "1", "type": "tool_call"})
        assert "5:20" in str(result.content)


async def test_a_bare_transport_is_rejected(socket_enabled):
    """Passing a transport directly would work once and then fail, so it is refused."""
    client = MultiServerMCPClient(
        {"time": streamable_http_client("http://localhost:8187/mcp")},
    )
    with pytest.raises(TypeError, match="can only be entered once"):
        await client.get_tools()


def _create_time_server():
    server = MCPServer()

    @server.tool()
    def get_time() -> str:
        """Get current time"""
        return "5:20:00 PM EST"

    return server


async def test_context_manager_reuses_one_connection_per_server(tmp_path):
    """Entering the client holds connections open, so operations share one connection.

    Counted by subprocess spawns: a stdio server starts once per connection, so a held
    connection serves every operation from the same process.
    """
    log = tmp_path / "spawns.log"
    server = os.path.join(Path(__file__).parent, "servers/counting_server.py")
    config = {"time": {"command": "python3", "args": [server, str(log)], "transport": "stdio"}}
    call = {"args": {}, "id": "1", "type": "tool_call"}

    def spawns() -> int:
        count = len(log.read_text().split()) if log.exists() else 0
        log.unlink(missing_ok=True)
        return count

    async with MultiServerMCPClient(config) as client:
        tools = {tool.name: tool for tool in await client.get_tools()}
        await tools["get_time"].ainvoke(call)
        await tools["get_time"].ainvoke(call)
    held = spawns()

    # Not entering still works, but opens a connection per operation.
    unheld_client = MultiServerMCPClient(config)
    tools = {tool.name: tool for tool in await unheld_client.get_tools()}
    await tools["get_time"].ainvoke(call)
    await tools["get_time"].ainvoke(call)
    unheld = spawns()

    assert held == 1, f"a held connection should start the server once, got {held}"
    assert unheld > held, f"unheld should reconnect per operation, got {unheld}"
