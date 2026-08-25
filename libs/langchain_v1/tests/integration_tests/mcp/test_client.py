import os
from pathlib import Path

from langchain_core.documents.base import Blob
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langchain.mcp.client import MultiServerMCPClient
from langchain.mcp.tools import load_mcp_tools
from tests.integration_tests.mcp.utils import IsLangChainID


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
