import typing
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, ToolException
from mcp.server.mcpserver import MCPServer
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    ToolAnnotations,
)
from mcp.types import Tool as MCPTool

from langchain.mcp.client import MultiServerMCPClient
from langchain.mcp.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain.mcp.tools import (
    MCPToolArtifact,
    _convert_call_tool_result,
    convert_mcp_tool_to_langchain_tool,
    load_mcp_tools,
)
from tests.integration_tests.mcp.utils import IsLangChainID, run_streamable_http


def test_convert_empty_text_content():
    # Test with a single text content
    result = CallToolResult(content=[], isError=False)

    content, artifact = _convert_call_tool_result(result)

    assert content == []
    assert artifact is None


def test_convert_single_text_content():
    # Test with a single text content
    result = CallToolResult(content=[TextContent(type="text", text="test result")], isError=False)

    content, artifact = _convert_call_tool_result(result)

    assert content == [{"type": "text", "text": "test result", "id": IsLangChainID}]
    assert artifact is None


def test_convert_multiple_text_contents():
    # Test with multiple text contents
    result = CallToolResult(
        content=[
            TextContent(type="text", text="result 1"),
            TextContent(type="text", text="result 2"),
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {"type": "text", "text": "result 1", "id": IsLangChainID},
        {"type": "text", "text": "result 2", "id": IsLangChainID},
    ]
    assert artifact is None


def test_convert_with_non_text_content():
    # Test with non-text content (now converted to LangChain content blocks)
    image_content = ImageContent(type="image", mimeType="image/png", data="base64data")
    resource_content = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(uri="resource://test", mimeType="text/plain", text="hi"),
    )

    result = CallToolResult(
        content=[
            TextContent(type="text", text="text result"),
            image_content,
            resource_content,
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    # With mixed content, we get a list of LangChain content blocks
    assert content == [
        {"type": "text", "text": "text result", "id": IsLangChainID},
        {
            "type": "image",
            "base64": "base64data",
            "mime_type": "image/png",
            "id": IsLangChainID,
        },
        {
            "type": "text",
            "text": "hi",
            "id": IsLangChainID,
        },  # EmbeddedResource with text -> text block
    ]
    # No structuredContent in this result
    assert artifact is None


def test_convert_with_error():
    # Test with error
    result = CallToolResult(content=[TextContent(type="text", text="error message")], isError=True)

    with pytest.raises(ToolException) as exc_info:
        _convert_call_tool_result(result)

    assert str(exc_info.value) == "error message"


def test_convert_with_structured_content():
    """Test that structuredContent is returned as MCPToolArtifact."""
    result = CallToolResult(
        content=[TextContent(type="text", text="text result")],
        isError=False,
        structuredContent={"key": "value", "nested": {"data": 123}},
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [{"type": "text", "text": "text result", "id": IsLangChainID}]
    assert artifact == MCPToolArtifact(structured_content={"key": "value", "nested": {"data": 123}})


def test_convert_image_content():
    """Test ImageContent conversion to LangChain image block."""
    result = CallToolResult(
        content=[ImageContent(type="image", mimeType="image/jpeg", data="jpeg_base64_data")],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    # Single non-text content returns as a list of content blocks
    assert content == [
        {
            "type": "image",
            "base64": "jpeg_base64_data",
            "mime_type": "image/jpeg",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_resource_link():
    """Test ResourceLink conversion to LangChain file block for non-image types."""
    result = CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri="file:///path/to/document.pdf",
                name="document.pdf",
                mimeType="application/pdf",
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "file",
            "url": "file:///path/to/document.pdf",
            "mime_type": "application/pdf",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_resource_link_image():
    """Test ResourceLink with image mime type converts to image block with URL."""
    result = CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri="https://example.com/photo.png",
                name="photo.png",
                mimeType="image/png",
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "image",
            "url": "https://example.com/photo.png",
            "mime_type": "image/png",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_resource_link_image_jpeg():
    """Test ResourceLink with JPEG image mime type converts to image block."""
    result = CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri="file:///photos/vacation.jpg",
                name="vacation.jpg",
                mimeType="image/jpeg",
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "image",
            "url": "file:///photos/vacation.jpg",
            "mime_type": "image/jpeg",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_resource_link_text():
    """Test ResourceLink with text mime type converts to file block (can't inline)."""
    result = CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri="file:///docs/readme.txt",
                name="readme.txt",
                mimeType="text/plain",
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    # Text ResourceLinks become file blocks since we only have URL, not content
    assert content == [
        {
            "type": "file",
            "url": "file:///docs/readme.txt",
            "mime_type": "text/plain",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_resource_link_no_mime_type():
    """Test ResourceLink without mime type converts to file block."""
    result = CallToolResult(
        content=[
            ResourceLink(
                type="resource_link",
                uri="file:///data/unknown",
                name="unknown",
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "file",
            "url": "file:///data/unknown",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_embedded_resource_blob_image():
    """Test EmbeddedResource with blob image converts to image block."""
    result = CallToolResult(
        content=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri="resource://image",
                    blob="png_base64_data",
                    mimeType="image/png",
                ),
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "image",
            "base64": "png_base64_data",
            "mime_type": "image/png",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_embedded_resource_blob_file():
    """Test EmbeddedResource with non-image blob converts to file block."""
    result = CallToolResult(
        content=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri="resource://data",
                    blob="pdf_base64_data",
                    mimeType="application/pdf",
                ),
            )
        ],
        isError=False,
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {
            "type": "file",
            "base64": "pdf_base64_data",
            "mime_type": "application/pdf",
            "id": IsLangChainID,
        }
    ]
    assert artifact is None


def test_convert_audio_content_raises():
    """Test that AudioContent raises NotImplementedError."""
    result = CallToolResult(
        content=[AudioContent(type="audio", mimeType="audio/wav", data="audio_data")],
        isError=False,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        _convert_call_tool_result(result)

    assert "AudioContent conversion" in str(exc_info.value)
    assert "audio/wav" in str(exc_info.value)


def test_convert_mixed_content_with_structured_content():
    """Test mixed content with structuredContent returns both."""
    result = CallToolResult(
        content=[
            TextContent(type="text", text="Here's the analysis"),
            ImageContent(type="image", mimeType="image/png", data="chart_data"),
        ],
        isError=False,
        structuredContent={"analysis": {"score": 0.95, "confidence": "high"}},
    )

    content, artifact = _convert_call_tool_result(result)

    assert content == [
        {"type": "text", "text": "Here's the analysis", "id": IsLangChainID},
        {
            "type": "image",
            "base64": "chart_data",
            "mime_type": "image/png",
            "id": IsLangChainID,
        },
    ]
    assert artifact == MCPToolArtifact(
        structured_content={"analysis": {"score": 0.95, "confidence": "high"}}
    )


async def test_convert_mcp_tool_to_langchain_tool():
    tool_input_schema = {
        "properties": {
            "param1": {"title": "Param1", "type": "string"},
            "param2": {"title": "Param2", "type": "integer"},
        },
        "required": ["param1", "param2"],
        "title": "ToolSchema",
        "type": "object",
    }
    # Mock session and MCP tool
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="tool result")],
        isError=False,
    )

    mcp_tool = MCPTool(
        name="test_tool",
        description="Test tool description",
        inputSchema=tool_input_schema,
    )

    # Convert MCP tool to LangChain tool
    lc_tool = convert_mcp_tool_to_langchain_tool(session, mcp_tool)

    # Verify the converted tool
    assert lc_tool.name == "test_tool"
    assert lc_tool.description == "Test tool description"
    assert lc_tool.args_schema == tool_input_schema

    # Test calling the tool
    result = await lc_tool.ainvoke(
        {"args": {"param1": "test", "param2": 42}, "id": "1", "type": "tool_call"},
    )

    # Verify session.call_tool was called with correct arguments
    session.call_tool.assert_called_once_with(
        "test_tool", {"param1": "test", "param2": 42}, progress_callback=None
    )

    # Verify result
    assert result.name == "test_tool"
    assert result.tool_call_id == "1"
    assert result.content == [{"type": "text", "text": "tool result", "id": IsLangChainID}]


async def test_load_mcp_tools():
    tool_input_schema = {
        "properties": {
            "param1": {"title": "Param1", "type": "string"},
            "param2": {"title": "Param2", "type": "integer"},
        },
        "required": ["param1", "param2"],
        "title": "ToolSchema",
        "type": "object",
    }
    # Mock session and list_tools response
    session = AsyncMock()
    mcp_tools = [
        MCPTool(
            name="tool1",
            description="Tool 1 description",
            inputSchema=tool_input_schema,
        ),
        MCPTool(
            name="tool2",
            description="Tool 2 description",
            inputSchema=tool_input_schema,
        ),
    ]
    session.list_tools.return_value = MagicMock(tools=mcp_tools, next_cursor=None)

    # Mock call_tool to return different results for different tools
    async def mock_call_tool(tool_name, arguments, progress_callback=None):
        if tool_name == "tool1":
            return CallToolResult(
                content=[TextContent(type="text", text=f"tool1 result with {arguments}")],
                isError=False,
            )
        return CallToolResult(
            content=[TextContent(type="text", text=f"tool2 result with {arguments}")],
            isError=False,
        )

    session.call_tool.side_effect = mock_call_tool

    # Load MCP tools
    tools = await load_mcp_tools(session)

    # Verify the tools
    assert len(tools) == 2
    assert all(isinstance(tool, BaseTool) for tool in tools)
    assert tools[0].name == "tool1"
    assert tools[1].name == "tool2"

    # Test calling the first tool
    result1 = await tools[0].ainvoke(
        {"args": {"param1": "test1", "param2": 1}, "id": "1", "type": "tool_call"},
    )
    assert result1.name == "tool1"
    assert result1.tool_call_id == "1"
    assert result1.content == [
        {
            "type": "text",
            "text": "tool1 result with {'param1': 'test1', 'param2': 1}",
            "id": IsLangChainID,
        }
    ]

    # Test calling the second tool
    result2 = await tools[1].ainvoke(
        {"args": {"param1": "test2", "param2": 2}, "id": "2", "type": "tool_call"},
    )
    assert result2.name == "tool2"
    assert result2.tool_call_id == "2"
    assert result2.content == [
        {
            "type": "text",
            "text": "tool2 result with {'param1': 'test2', 'param2': 2}",
            "id": IsLangChainID,
        }
    ]


def _create_annotations_server():
    server = MCPServer()

    @server.tool(
        annotations=ToolAnnotations(title="Get Time", readOnlyHint=True, idempotentHint=False),
    )
    def get_time() -> str:
        """Get current time"""
        return "5:20:00 PM EST"

    return server


@pytest.mark.parametrize("transport", ["http", "streamable_http", "streamable-http"])
async def test_load_mcp_tools_with_http_variations(socket_enabled, transport) -> None:
    """Test load mcp tools with annotations."""
    with run_streamable_http(_create_annotations_server, 8181):
        # Initialize client without initial connections
        client = MultiServerMCPClient(
            {
                "time": {
                    "url": "http://localhost:8181/mcp",
                    "transport": transport,
                }
            },
        )
        # pass
        tools = await client.get_tools(server_name="time")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_time"


async def test_load_mcp_tools_with_annotations(socket_enabled) -> None:
    """Test load mcp tools with annotations."""
    with run_streamable_http(_create_annotations_server, 8181):
        # Initialize client without initial connections
        client = MultiServerMCPClient(
            {
                "time": {
                    "url": "http://localhost:8181/mcp",
                    "transport": "streamable_http",
                }
            },
        )
        # pass
        tools = await client.get_tools(server_name="time")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "get_time"
        assert tool.metadata == {
            "title": "Get Time",
            "readOnlyHint": True,
            "idempotentHint": False,
            "destructiveHint": None,
            "openWorldHint": None,
        }


def _create_status_server():
    server = MCPServer()

    @server.tool()
    def get_status() -> str:
        """Get server status"""
        return "Server is running"

    return server


# Tests for supplying an HTTP client


async def test_load_mcp_tools_with_supplied_http_client(socket_enabled) -> None:
    """A caller-supplied HTTP client is used instead of one built from the config."""
    http_client = httpx2.AsyncClient(
        timeout=httpx2.Timeout(30.0),
        limits=httpx2.Limits(max_keepalive_connections=5, max_connections=10),
    )

    with run_streamable_http(_create_status_server, 8182):
        client = MultiServerMCPClient(
            {
                "status": {
                    "url": "http://localhost:8182/mcp",
                    "transport": "streamable_http",
                    "http_client": http_client,
                },
            },
        )

        tools = await client.get_tools(server_name="status")
        assert len(tools) == 1
        assert tools[0].name == "get_status"

        result = await tools[0].ainvoke({"args": {}, "id": "1", "type": "tool_call"})
        assert result.content == [
            {"type": "text", "text": "Server is running", "id": IsLangChainID}
        ]


async def test_convert_mcp_tool_metadata_variants():
    """Verify metadata merging rules in convert_mcp_tool_to_langchain_tool."""
    tool_input_schema = {
        "properties": {},
        "required": [],
        "title": "EmptySchema",
        "type": "object",
    }

    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")], isError=False
    )

    mcp_tool_none = MCPTool(
        name="t_none",
        description="",
        inputSchema=tool_input_schema,
    )
    lc_tool_none = convert_mcp_tool_to_langchain_tool(session, mcp_tool_none)
    assert lc_tool_none.metadata is None

    mcp_tool_ann = MCPTool(
        name="t_ann",
        description="",
        inputSchema=tool_input_schema,
        annotations=ToolAnnotations(title="Title", readOnlyHint=True, idempotentHint=False),
    )
    lc_tool_ann = convert_mcp_tool_to_langchain_tool(session, mcp_tool_ann)
    assert lc_tool_ann.metadata == {
        "title": "Title",
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": None,
        "openWorldHint": None,
    }

    mcp_tool_meta = MCPTool(
        name="t_meta",
        description="",
        inputSchema=tool_input_schema,
        _meta={"source": "unit-test", "version": 1},
    )
    lc_tool_meta = convert_mcp_tool_to_langchain_tool(session, mcp_tool_meta)
    assert lc_tool_meta.metadata == {"_meta": {"source": "unit-test", "version": 1}}

    mcp_tool_both = MCPTool(
        name="t_both",
        description="",
        inputSchema=tool_input_schema,
        annotations=ToolAnnotations(title="Both"),
        _meta={"flag": True},
    )

    lc_tool_both = convert_mcp_tool_to_langchain_tool(session, mcp_tool_both)
    assert lc_tool_both.metadata == {
        "title": "Both",
        "readOnlyHint": None,
        "idempotentHint": None,
        "destructiveHint": None,
        "openWorldHint": None,
        "_meta": {"flag": True},
    }


def _create_increment_server():
    server = MCPServer()

    @server.tool()
    def increment(value: int) -> str:
        """Increment a counter"""
        return f"Incremented to {value + 1}"

    return server


try:
    import langchain.agents  # noqa: F401

    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False


class FixedGenericFakeChatModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[
            typing.Dict[str, Any] | type | Callable | BaseTool  # noqa: UP006
        ],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind-tools."""
        return self


@pytest.mark.skipif(not LANGCHAIN_INSTALLED, reason="langchain not installed")
async def test_mcp_tools_with_agent_and_command_interceptor(socket_enabled) -> None:
    """Test Command objects from interceptors work end-to-end with create_agent.

    This test verifies that:
    1. MCP tools can be used with create_agent
    2. Interceptors can return Command objects to short-circuit execution
    3. Commands can update custom agent state
    """
    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415
    from langgraph.types import Command  # noqa: PLC0415

    from langchain.agents import AgentState, create_agent  # noqa: PLC0415
    from langchain.tools import ToolRuntime  # noqa: PLC0415

    # Interceptor that returns Command to update state
    async def counter_interceptor(
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], typing.Awaitable[MCPToolCallResult]],
    ) -> Command:
        # Instead of calling the tool, return a Command that updates state
        tool_runtime: ToolRuntime = request.runtime
        assert tool_runtime.tool_call_id == "call_1"
        return Command(
            update={
                "counter": 42,
                "messages": [
                    ToolMessage(
                        content="Counter updated!",
                        tool_call_id=tool_runtime.tool_call_id,
                    ),
                    AIMessage(content="hello"),
                ],
            },
            goto="__end__",
        )

    with run_streamable_http(_create_increment_server, 8183):
        # Initialize client and connect to server
        client = MultiServerMCPClient(
            {
                "increment": {
                    "url": "http://localhost:8183/mcp",
                    "transport": "streamable_http",
                }
            },
            tool_interceptors=[counter_interceptor],
        )

        # Get tools from the server
        tools = await client.get_tools(server_name="increment")
        assert len(tools) == 1
        original_tool = tools[0]
        assert original_tool.name == "increment"

        # Custom state schema with counter field
        class CustomAgentState(AgentState):
            counter: typing.NotRequired[int]

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "increment",
                                "args": {"value": 1},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="The counter has been incremented.",
                    ),
                ]
            )
        )
        # Create agent with custom state
        agent = create_agent(
            model,
            tools,
            state_schema=CustomAgentState,
            checkpointer=MemorySaver(),
        )

        # Run agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Increment the counter")], "counter": 0},
            {"configurable": {"thread_id": "test_1"}},
        )

        # Verify Command updated the state
        assert result["counter"] == 42
        # Verify the Command's message was added
        assert any(
            isinstance(msg, ToolMessage) and msg.content == "Counter updated!"
            for msg in result["messages"]
        )


# Tests for tool_name_prefix functionality


def _create_weather_search_server():
    """Create a weather server with a search tool."""
    server = MCPServer()

    @server.tool()
    def search(query: str) -> str:
        """Search for weather information"""
        return f"Weather results for: {query}"

    return server


def _create_flights_search_server():
    """Create a flights server with a search tool."""
    server = MCPServer()

    @server.tool()
    def search(destination: str) -> str:
        """Search for flights"""
        return f"Flight results to: {destination}"

    return server


@pytest.mark.skipif(not LANGCHAIN_INSTALLED, reason="langchain not installed")
async def test_parallel_tool_invocation_across_multiple_servers(socket_enabled) -> None:
    """Test that two servers with identically named tools can be invoked in parallel.

    This test verifies that:
    1. Two MCP servers can each expose a tool with the same name (search)
    2. With tool_name_prefix=True, they get unique LangChain names
        (weather_search, flights_search)
    3. When an LLM calls both tools in parallel,
       each tool is routed to the correct server
    4. The correct results come back from each server
    """
    from langgraph.checkpoint.memory import MemorySaver  # noqa: PLC0415

    from langchain.agents import AgentState, create_agent  # noqa: PLC0415

    with (
        run_streamable_http(_create_weather_search_server, 8185),
        run_streamable_http(_create_flights_search_server, 8186),
    ):
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                },
                "flights": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=True,
        )
        tools = await client.get_tools()

        # Verify we have both prefixed tools
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"weather_search", "flights_search"}

        # Simulate an LLM calling both tools in parallel (common pattern for agents)
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "weather_search",
                                "args": {"query": "sunny in Paris"},
                                "id": "call_weather",
                                "type": "tool_call",
                            },
                            {
                                "name": "flights_search",
                                "args": {"destination": "Tokyo"},
                                "id": "call_flights",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="Here are your results."),
                ]
            )
        )

        agent = create_agent(
            model,
            tools,
            state_schema=AgentState,
            checkpointer=MemorySaver(),
        )

        # Run the agent - both tools should be called in parallel
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Search weather and flights")]},
            {"configurable": {"thread_id": "test_parallel"}},
        )

        # Verify both tools were called and returned correct results
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        assert len(tool_messages) == 2

        # Create a mapping of tool_call_id to content for easier assertion
        results_by_id = {msg.tool_call_id: msg.content for msg in tool_messages}

        # Verify the weather search was routed to the weather server
        assert results_by_id["call_weather"] == [
            {
                "type": "text",
                "text": "Weather results for: sunny in Paris",
                "id": IsLangChainID,
            }
        ]

        # Verify the flights search was routed to the flights server
        assert results_by_id["call_flights"] == [
            {
                "type": "text",
                "text": "Flight results to: Tokyo",
                "id": IsLangChainID,
            }
        ]


async def test_get_tools_with_name_conflict(socket_enabled) -> None:
    """Test fetching tools with name conflict using tool_name_prefix.

    This test verifies that:
    1. Without tool_name_prefix, both servers would have conflicting "search" tool names
    2. With tool_name_prefix=True, tools get unique names
        (weather_search, flights_search)
    """
    with (
        run_streamable_http(_create_weather_search_server, 8185),
        run_streamable_http(_create_flights_search_server, 8186),
    ):
        # First, verify that without prefix both tools would have the same name
        client_no_prefix = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                },
                "flights": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=False,
        )
        tools_no_prefix = await client_no_prefix.get_tools()
        # Both tools are named "search" without prefix
        assert all(t.name == "search" for t in tools_no_prefix)

        # Now test with prefix - tools should be disambiguated
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8185/mcp",
                    "transport": "streamable_http",
                },
                "flights": {
                    "url": "http://localhost:8186/mcp",
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=True,
        )
        tools = await client.get_tools()

        # Verify we have both prefixed tools with unique names
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"weather_search", "flights_search"}
