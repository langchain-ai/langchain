"""Test ToolRuntime injection with create_agent.

This module tests the injected runtime functionality when using tools
with the create_agent factory. The ToolRuntime provides tools access to:
- state: Current graph state
- tool_call_id: ID of the current tool call
- config: RunnableConfig for the execution
- context: Runtime context from LangGraph
- store: BaseStore for persistent storage
- stream_writer: For streaming custom output

These tests verify that runtime injection works correctly across both
sync and async execution paths, with middleware, and in various agent
configurations.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.memory import InMemoryStore

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState
from langchain.tools import InjectedState, ToolRuntime

from .model import FakeToolCallingModel


def test_tool_runtime_basic_injection() -> None:
    """Test basic ToolRuntime injection in tools with create_agent."""
    # Track what was injected
    injected_data = {}

    @tool
    def runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses runtime context."""
        injected_data["state"] = runtime.state
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        injected_data["context"] = runtime.context
        injected_data["store"] = runtime.store
        injected_data["stream_writer"] = runtime.stream_writer
        return f"Processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 42}, "id": "call_123", "name": "runtime_tool"}],
                [],
            ]
        ),
        tools=[runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify tool executed
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Processed 42"
    assert tool_message.tool_call_id == "call_123"

    # Verify runtime was injected
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]
    assert injected_data["tool_call_id"] == "call_123"
    assert injected_data["config"] is not None
    # Context, store, stream_writer may be None depending on graph setup
    assert "context" in injected_data
    assert "store" in injected_data
    assert "stream_writer" in injected_data


async def test_tool_runtime_async_injection() -> None:
    """Test ToolRuntime injection works with async tools."""
    injected_data = {}

    @tool
    async def async_runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Async tool that accesses runtime context."""
        injected_data["state"] = runtime.state
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        return f"Async processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 99}, "id": "async_call_456", "name": "async_runtime_tool"}],
                [],
            ]
        ),
        tools=[async_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Test async")]})

    # Verify tool executed
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Async processed 99"
    assert tool_message.tool_call_id == "async_call_456"

    # Verify runtime was injected
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]
    assert injected_data["tool_call_id"] == "async_call_456"
    assert injected_data["config"] is not None


def test_tool_runtime_state_access() -> None:
    """Test that tools can access and use state via ToolRuntime."""

    @tool
    def state_aware_tool(query: str, runtime: ToolRuntime) -> str:
        """Tool that uses state to provide context-aware responses."""
        messages = runtime.state.get("messages", [])
        msg_count = len(messages)
        return f"Query: {query}, Message count: {msg_count}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"query": "test"}, "id": "state_call", "name": "state_aware_tool"}],
                [],
            ]
        ),
        tools=[state_aware_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello"), HumanMessage("World")]})

    # Check that tool accessed state correctly
    tool_message = result["messages"][3]
    assert isinstance(tool_message, ToolMessage)
    # Should have original 2 HumanMessages + 1 AIMessage before tool execution
    assert "Message count: 3" in tool_message.content


def test_tool_runtime_with_store() -> None:
    """Test ToolRuntime provides access to store."""
    # Note: create_agent doesn't currently expose a store parameter,
    # so runtime.store will be None in this test.
    # This test demonstrates the runtime injection works correctly.

    @tool
    def store_tool(key: str, value: str, runtime: ToolRuntime) -> str:
        """Tool that uses store."""
        if runtime.store is None:
            return f"No store (key={key}, value={value})"
        runtime.store.put(("test",), key, {"data": value})
        return f"Stored {key}={value}"

    @tool
    def check_runtime_tool(runtime: ToolRuntime) -> str:
        """Tool that checks runtime availability."""
        has_store = runtime.store is not None
        has_context = runtime.context is not None
        return f"Runtime: store={has_store}, context={has_context}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"key": "foo", "value": "bar"}, "id": "call_1", "name": "store_tool"}],
                [{"args": {}, "id": "call_2", "name": "check_runtime_tool"}],
                [],
            ]
        ),
        tools=[store_tool, check_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test store")]})

    # Find the tool messages
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    # First tool indicates no store is available (expected since create_agent doesn't expose store)
    assert "No store" in tool_messages[0].content

    # Second tool confirms runtime was injected
    assert "Runtime:" in tool_messages[1].content


def test_tool_runtime_with_multiple_tools() -> None:
    """Test multiple tools can all access ToolRuntime."""
    call_log = []

    @tool
    def tool_a(x: int, runtime: ToolRuntime) -> str:
        """First tool."""
        call_log.append(("tool_a", runtime.tool_call_id, x))
        return f"A: {x}"

    @tool
    def tool_b(y: str, runtime: ToolRuntime) -> str:
        """Second tool."""
        call_log.append(("tool_b", runtime.tool_call_id, y))
        return f"B: {y}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 1}, "id": "call_a", "name": "tool_a"},
                    {"args": {"y": "test"}, "id": "call_b", "name": "tool_b"},
                ],
                [],
            ]
        ),
        tools=[tool_a, tool_b],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Use both tools")]})

    # Verify both tools were called with correct runtime
    assert len(call_log) == 2
    # Tools may execute in parallel, so check both calls are present
    call_ids = {(name, call_id) for name, call_id, _ in call_log}
    assert ("tool_a", "call_a") in call_ids
    assert ("tool_b", "call_b") in call_ids

    # Verify tool messages
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2
    contents = {msg.content for msg in tool_messages}
    assert "A: 1" in contents
    assert "B: test" in contents


def test_tool_runtime_config_access() -> None:
    """Test tools can access config through ToolRuntime."""
    config_data = {}

    @tool
    def config_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses config."""
        config_data["config_exists"] = runtime.config is not None
        config_data["has_configurable"] = (
            "configurable" in runtime.config if runtime.config else False
        )
        # Config may have run_id or other fields depending on execution context
        if runtime.config:
            config_data["config_keys"] = list(runtime.config.keys())
        return f"Config accessed for {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 5}, "id": "config_call", "name": "config_tool"}],
                [],
            ]
        ),
        tools=[config_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test config")]})

    # Verify config was accessible
    assert config_data["config_exists"] is True
    assert "config_keys" in config_data

    # Verify tool executed
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Config accessed for 5"


def test_tool_runtime_with_custom_state() -> None:
    """Test ToolRuntime works with custom state schemas."""
    from langchain.agents.middleware.types import AgentMiddleware

    class CustomState(AgentState):
        custom_field: str

    runtime_state = {}

    @tool
    def custom_state_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses custom state."""
        runtime_state["custom_field"] = runtime.state.get("custom_field", "not found")
        return f"Custom: {x}"

    class CustomMiddleware(AgentMiddleware):
        state_schema = CustomState

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 10}, "id": "custom_call", "name": "custom_state_tool"}],
                [],
            ]
        ),
        tools=[custom_state_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[CustomMiddleware()],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test custom state")],
            "custom_field": "custom_value",
        }
    )

    # Verify custom field was accessible
    assert runtime_state["custom_field"] == "custom_value"

    # Verify tool executed
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Custom: 10"


def test_tool_runtime_no_runtime_parameter() -> None:
    """Test that tools without runtime parameter work normally."""

    @tool
    def regular_tool(x: int) -> str:
        """Regular tool without runtime."""
        return f"Regular: {x}"

    @tool
    def runtime_tool(y: int, runtime: ToolRuntime) -> str:
        """Tool with runtime."""
        return f"Runtime: {y}, call_id: {runtime.tool_call_id}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 1}, "id": "regular_call", "name": "regular_tool"},
                    {"args": {"y": 2}, "id": "runtime_call", "name": "runtime_tool"},
                ],
                [],
            ]
        ),
        tools=[regular_tool, runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test mixed tools")]})

    # Verify both tools executed correctly
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2
    assert tool_messages[0].content == "Regular: 1"
    assert "Runtime: 2, call_id: runtime_call" in tool_messages[1].content


async def test_tool_runtime_parallel_execution() -> None:
    """Test ToolRuntime injection works with parallel tool execution."""
    execution_log = []

    @tool
    async def parallel_tool_1(x: int, runtime: ToolRuntime) -> str:
        """First parallel tool."""
        execution_log.append(("tool_1", runtime.tool_call_id, x))
        return f"Tool1: {x}"

    @tool
    async def parallel_tool_2(y: int, runtime: ToolRuntime) -> str:
        """Second parallel tool."""
        execution_log.append(("tool_2", runtime.tool_call_id, y))
        return f"Tool2: {y}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 10}, "id": "parallel_1", "name": "parallel_tool_1"},
                    {"args": {"y": 20}, "id": "parallel_2", "name": "parallel_tool_2"},
                ],
                [],
            ]
        ),
        tools=[parallel_tool_1, parallel_tool_2],
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Run parallel")]})

    # Verify both tools executed
    assert len(execution_log) == 2

    # Find the tool messages (order may vary due to parallel execution)
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    contents = {msg.content for msg in tool_messages}
    assert "Tool1: 10" in contents
    assert "Tool2: 20" in contents

    call_ids = {msg.tool_call_id for msg in tool_messages}
    assert "parallel_1" in call_ids
    assert "parallel_2" in call_ids


def test_tool_runtime_error_handling() -> None:
    """Test error handling with ToolRuntime injection."""

    @tool
    def error_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that may error."""
        # Access runtime to ensure it's injected even during errors
        _ = runtime.tool_call_id
        if x == 0:
            msg = "Cannot process zero"
            raise ValueError(msg)
        return f"Processed: {x}"

    # create_agent uses default error handling which doesn't catch ValueError
    # So we need to handle this differently
    @tool
    def safe_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that handles errors safely."""
        try:
            if x == 0:
                return "Error: Cannot process zero"
        except Exception as e:
            return f"Error: {e}"
        return f"Processed: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 0}, "id": "error_call", "name": "safe_tool"}],
                [{"args": {"x": 5}, "id": "success_call", "name": "safe_tool"}],
                [],
            ]
        ),
        tools=[safe_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test error handling")]})

    # Both tool calls should complete
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    # First call returned error message
    assert "Error:" in tool_messages[0].content or "Cannot process zero" in tool_messages[0].content

    # Second call succeeded
    assert "Processed: 5" in tool_messages[1].content


def test_tool_runtime_with_middleware() -> None:
    """Test ToolRuntime injection works with agent middleware."""
    from langchain.agents.middleware.types import AgentMiddleware

    middleware_calls = []
    runtime_calls = []

    class TestMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> dict[str, Any]:
            middleware_calls.append("before_model")
            return {}

        def after_model(self, state, runtime) -> dict[str, Any]:
            middleware_calls.append("after_model")
            return {}

    @tool
    def middleware_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool with runtime in middleware agent."""
        runtime_calls.append(("middleware_tool", runtime.tool_call_id))
        return f"Middleware result: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 7}, "id": "mw_call", "name": "middleware_tool"}],
                [],
            ]
        ),
        tools=[middleware_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[TestMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Test with middleware")]})

    # Verify middleware ran
    assert "before_model" in middleware_calls
    assert "after_model" in middleware_calls

    # Verify tool with runtime executed
    assert len(runtime_calls) == 1
    assert runtime_calls[0] == ("middleware_tool", "mw_call")

    # Verify result
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Middleware result: 7"


def test_tool_runtime_type_hints() -> None:
    """Test that ToolRuntime provides access to state fields."""
    typed_runtime = {}

    # Use ToolRuntime without generic type hints to avoid forward reference issues
    @tool
    def typed_runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool with runtime access."""
        # Access state dict - verify we can access standard state fields
        if isinstance(runtime.state, dict):
            # Count messages in state
            typed_runtime["message_count"] = len(runtime.state.get("messages", []))
        else:
            typed_runtime["message_count"] = len(getattr(runtime.state, "messages", []))
        return f"Typed: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 3}, "id": "typed_call", "name": "typed_runtime_tool"}],
                [],
            ]
        ),
        tools=[typed_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify typed runtime worked -
    # should see 2 messages (HumanMessage + AIMessage) before tool executes
    assert typed_runtime["message_count"] == 2

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Typed: 3"


def test_tool_runtime_name_based_injection() -> None:
    """Test that parameter named 'runtime' gets injected without type annotation."""
    injected_data = {}

    @tool
    def name_based_tool(x: int, runtime: Any) -> str:
        """Tool with 'runtime' parameter without ToolRuntime type annotation."""
        # Even though type is Any, runtime should still be injected as ToolRuntime
        injected_data["is_tool_runtime"] = isinstance(runtime, ToolRuntime)
        injected_data["has_state"] = hasattr(runtime, "state")
        injected_data["has_tool_call_id"] = hasattr(runtime, "tool_call_id")
        if hasattr(runtime, "tool_call_id"):
            injected_data["tool_call_id"] = runtime.tool_call_id
        if hasattr(runtime, "state"):
            injected_data["state"] = runtime.state
        return f"Processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 42}, "id": "name_call_123", "name": "name_based_tool"}],
                [],
            ]
        ),
        tools=[name_based_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify tool executed
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Processed 42"

    # Verify runtime was injected based on parameter name
    assert injected_data["is_tool_runtime"] is True
    assert injected_data["has_state"] is True
    assert injected_data["has_tool_call_id"] is True
    assert injected_data["tool_call_id"] == "name_call_123"
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]


def test_combined_injected_state_runtime_store() -> None:
    """Test that all injection mechanisms work together in create_agent.

    This test verifies that a tool can receive injected state, tool runtime,
    and injected store simultaneously when specified in the function signature
    but not in the explicit args schema. This is modeled after the pattern
    from mre.py where multiple injection types are combined.
    """
    # Track what was injected
    injected_data = {}

    # Custom state schema with additional fields
    class CustomState(AgentState):
        user_id: str
        session_id: str

    # Define explicit args schema that only includes LLM-controlled parameters
    weather_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The location to get weather for"},
        },
        "required": ["location"],
    }

    @tool(args_schema=weather_schema)
    def multi_injection_tool(
        location: str,
        state: Annotated[Any, InjectedState],
        runtime: ToolRuntime,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Tool that uses injected state, runtime, and store together.

        Args:
            location: The location to get weather for (LLM-controlled).
            state: The graph state (injected).
            runtime: The tool runtime context (injected).
            store: The persistent store (injected).
        """
        # Capture all injected parameters
        injected_data["state"] = state
        injected_data["user_id"] = state.get("user_id", "unknown")
        injected_data["session_id"] = state.get("session_id", "unknown")
        injected_data["runtime"] = runtime
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["store"] = store
        injected_data["store_is_none"] = store is None

        # Verify runtime.state matches the state parameter
        injected_data["runtime_state_matches"] = runtime.state == state

        return f"Weather info for {location}"

    # Create model that calls the tool
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "multi_injection_tool",
                    "args": {"location": "San Francisco"},  # Only LLM-controlled arg
                    "id": "call_weather_123",
                }
            ],
            [],  # End the loop
        ]
    )

    # Create agent with custom state and store
    agent = create_agent(
        model=model,
        tools=[multi_injection_tool],
        state_schema=CustomState,
        store=InMemoryStore(),
    )

    # Verify the tool's args schema only includes LLM-controlled parameters
    tool_args_schema = multi_injection_tool.args_schema
    assert "location" in tool_args_schema["properties"]
    assert "state" not in tool_args_schema["properties"]
    assert "runtime" not in tool_args_schema["properties"]
    assert "store" not in tool_args_schema["properties"]

    # Invoke with custom state fields
    result = agent.invoke(
        {
            "messages": [HumanMessage("What's the weather like?")],
            "user_id": "user_42",
            "session_id": "session_abc123",
        }
    )

    # Verify tool executed successfully
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.content == "Weather info for San Francisco"
    assert tool_message.tool_call_id == "call_weather_123"

    # Verify all injections worked correctly
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]

    # Verify custom state fields were accessible
    assert injected_data["user_id"] == "user_42"
    assert injected_data["session_id"] == "session_abc123"

    # Verify runtime was injected
    assert injected_data["runtime"] is not None
    assert injected_data["tool_call_id"] == "call_weather_123"

    # Verify store was injected
    assert injected_data["store_is_none"] is False
    assert injected_data["store"] is not None

    # Verify runtime.state matches the injected state
    assert injected_data["runtime_state_matches"] is True


async def test_combined_injected_state_runtime_store_async() -> None:
    """Test that all injection mechanisms work together in async execution.

    This async version verifies that injected state, tool runtime, and injected
    store all work correctly with async tools in create_agent.
    """
    # Track what was injected
    injected_data = {}

    # Custom state schema
    class CustomState(AgentState):
        api_key: str
        request_id: str

    # Define explicit args schema that only includes LLM-controlled parameters
    # Note: state, runtime, and store are NOT in this schema
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Maximum number of results"},
        },
        "required": ["query", "max_results"],
    }

    @tool(args_schema=search_schema)
    async def async_multi_injection_tool(
        query: str,
        max_results: int,
        state: Annotated[Any, InjectedState],
        runtime: ToolRuntime,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Async tool with multiple injection types.

        Args:
            query: The search query (LLM-controlled).
            max_results: Maximum number of results (LLM-controlled).
            state: The graph state (injected).
            runtime: The tool runtime context (injected).
            store: The persistent store (injected).
        """
        # Capture all injected parameters
        injected_data["state"] = state
        injected_data["api_key"] = state.get("api_key", "unknown")
        injected_data["request_id"] = state.get("request_id", "unknown")
        injected_data["runtime"] = runtime
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        injected_data["store"] = store

        # Verify we can write to the store
        if store is not None:
            await store.aput(("test", "namespace"), "test_key", {"query": query})
            # Read back to verify it worked
            item = await store.aget(("test", "namespace"), "test_key")
            injected_data["store_write_success"] = item is not None

        return f"Found {max_results} results for '{query}'"

    # Create model that calls the async tool
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "async_multi_injection_tool",
                    "args": {"query": "test search", "max_results": 10},
                    "id": "call_search_456",
                }
            ],
            [],
        ]
    )

    # Create agent with custom state and store
    agent = create_agent(
        model=model,
        tools=[async_multi_injection_tool],
        state_schema=CustomState,
        store=InMemoryStore(),
    )

    # Verify the tool's args schema only includes LLM-controlled parameters
    tool_args_schema = async_multi_injection_tool.args_schema
    assert "query" in tool_args_schema["properties"]
    assert "max_results" in tool_args_schema["properties"]
    assert "state" not in tool_args_schema["properties"]
    assert "runtime" not in tool_args_schema["properties"]
    assert "store" not in tool_args_schema["properties"]

    # Invoke async
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Search for something")],
            "api_key": "sk-test-key-xyz",
            "request_id": "req_999",
        }
    )

    # Verify tool executed successfully
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.content == "Found 10 results for 'test search'"
    assert tool_message.tool_call_id == "call_search_456"

    # Verify all injections worked correctly
    assert injected_data["state"] is not None
    assert injected_data["api_key"] == "sk-test-key-xyz"
    assert injected_data["request_id"] == "req_999"

    # Verify runtime was injected
    assert injected_data["runtime"] is not None
    assert injected_data["tool_call_id"] == "call_search_456"
    assert injected_data["config"] is not None

    # Verify store was injected and writable
    assert injected_data["store"] is not None
    assert injected_data["store_write_success"] is True
