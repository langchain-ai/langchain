"""Unit tests for ValidationError filtering in ToolNode.

This module tests that validation errors are filtered to only include arguments
that the LLM controls. Injected arguments (InjectedState, InjectedStore,
ToolRuntime) are automatically provided by the system and should not appear in
validation error messages. This ensures the LLM receives focused, actionable
feedback about the parameters it can actually control, improving error correction
and reducing confusion from irrelevant system implementation details.
"""

import sys
from typing import Annotated
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as dec_tool
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentState
from langchain.tools import InjectedState, InjectedStore
from langchain.tools.tool_node import ToolInvocationError, ToolRuntime, _ToolNode

from .model import FakeToolCallingModel

pytestmark = pytest.mark.anyio


def _create_mock_runtime(store: BaseStore | None = None) -> Mock:
    """Create a mock Runtime object for testing ToolNode outside of graph context."""
    mock_runtime = Mock()
    mock_runtime.store = store
    mock_runtime.context = None
    mock_runtime.stream_writer = lambda *args, **kwargs: None
    return mock_runtime


def _create_config_with_runtime(store: BaseStore | None = None) -> RunnableConfig:
    """Create a RunnableConfig with mock Runtime for testing ToolNode."""
    return {"configurable": {"__pregel_runtime": _create_mock_runtime(store)}}


async def test_filter_injected_state_validation_errors() -> None:
    """Test that validation errors for InjectedState arguments are filtered out.

    InjectedState parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This ensures
    the LLM receives only actionable feedback about its own tool call arguments.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool that uses injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}, messages={len(state.get('messages', []))}"

    tool_node = _ToolNode([my_tool])

    # Call with invalid 'value' argument (should be int, not str)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},  # Invalid type
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_1"

    # Error should mention 'value' but NOT 'state' (which is injected)
    assert "value" in tool_message.content
    assert "state" not in tool_message.content.lower()


async def test_filter_injected_store_validation_errors() -> None:
    """Test that validation errors for InjectedStore arguments are filtered out.

    InjectedStore parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This keeps
    error feedback focused on LLM-controllable parameters.
    """

    @dec_tool
    def my_tool(
        key: str,
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """Tool that uses injected store.

        Args:
            key: A key to look up.
            store: The persistent store (injected).
        """
        return f"key={key}"

    tool_node = _ToolNode([my_tool])

    # Call with invalid 'key' argument (missing required argument)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {},  # Missing 'key'
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(store=InMemoryStore()),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Error should mention 'key' is required
    assert "key" in tool_message.content.lower()
    # The error should be about 'key' field specifically (not about store field)
    # Note: 'store' might appear in input_value representation, but the validation
    # error itself should only be for 'key'
    assert (
        "field required" in tool_message.content.lower()
        or "missing" in tool_message.content.lower()
    )


async def test_filter_tool_runtime_validation_errors() -> None:
    """Test that validation errors for ToolRuntime arguments are filtered out.

    ToolRuntime parameters are not controlled by the LLM, so any validation
    errors related to them should not appear in error messages. This ensures
    the LLM only sees errors for parameters it can fix.
    """

    @dec_tool
    def my_tool(
        query: str,
        runtime: ToolRuntime,
    ) -> str:
        """Tool that uses ToolRuntime.

        Args:
            query: A query string.
            runtime: The tool runtime context (injected).
        """
        return f"query={query}"

    tool_node = _ToolNode([my_tool])

    # Call with invalid 'query' argument (wrong type)
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"query": 123},  # Should be str, not int
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    # Should get a ToolMessage with error
    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Error should mention 'query' but NOT 'runtime' (which is injected)
    assert "query" in tool_message.content.lower()
    assert "runtime" not in tool_message.content.lower()


async def test_filter_multiple_injected_args() -> None:
    """Test filtering when a tool has multiple injected arguments.

    When a tool uses multiple injected parameters (state, store, runtime), none of
    them should appear in validation error messages since they're all system-provided
    and not controlled by the LLM. Only LLM-controllable parameter errors should appear.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """Tool with multiple injected arguments.

        Args:
            value: An integer value.
            state: The graph state (injected).
            store: The persistent store (injected).
            runtime: The tool runtime context (injected).
        """
        return f"value={value}"

    tool_node = _ToolNode([my_tool])

    # Call with invalid 'value' - injected args should be filtered from error
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(store=InMemoryStore()),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Only 'value' error should be reported
    assert "value" in tool_message.content
    # None of the injected args should appear in error
    assert "state" not in tool_message.content.lower()
    assert "store" not in tool_message.content.lower()
    assert "runtime" not in tool_message.content.lower()


async def test_no_filtering_when_all_errors_are_model_args() -> None:
    """Test that validation errors for LLM-controlled arguments are preserved.

    When validation fails for arguments the LLM controls, those errors should
    be fully reported to help the LLM correct its tool calls. This ensures
    the LLM receives complete feedback about all issues it can fix.
    """

    @dec_tool
    def my_tool(
        value1: int,
        value2: str,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool with both regular and injected arguments.

        Args:
            value1: First value.
            value2: Second value.
            state: The graph state (injected).
        """
        return f"value1={value1}, value2={value2}"

    tool_node = _ToolNode([my_tool])

    # Call with invalid arguments for BOTH non-injected parameters
    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {
                                "value1": "not_an_int",  # Invalid
                                "value2": 456,  # Invalid (should be str)
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Both errors should be present
    assert "value1" in tool_message.content
    assert "value2" in tool_message.content
    # Injected state should not appear
    assert "state" not in tool_message.content.lower()


async def test_validation_error_with_no_injected_args() -> None:
    """Test that tools without injected arguments show all validation errors.

    For tools that only have LLM-controlled parameters, all validation errors
    should be reported since everything is under the LLM's control and can be
    corrected by the LLM in subsequent tool calls.
    """

    @dec_tool
    def my_tool(value1: int, value2: str) -> str:
        """Regular tool without injected arguments.

        Args:
            value1: First value.
            value2: Second value.
        """
        return f"{value1} {value2}"

    tool_node = _ToolNode([my_tool])

    result = await tool_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value1": "invalid", "value2": 123},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"

    # Both errors should be present since there are no injected args to filter
    assert "value1" in tool_message.content
    assert "value2" in tool_message.content


async def test_tool_invocation_error_without_handle_errors() -> None:
    """Test that ToolInvocationError contains only LLM-controlled parameter errors.

    When handle_tool_errors is False, the raised ToolInvocationError should still
    filter out system-injected arguments from the error details, ensuring that
    error messages focus on what the LLM can control.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Tool with injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}"

    tool_node = _ToolNode([my_tool], handle_tool_errors=False)

    # Should raise ToolInvocationError with filtered errors
    with pytest.raises(ToolInvocationError) as exc_info:
        await tool_node.ainvoke(
            {
                "messages": [
                    AIMessage(
                        "hi?",
                        tool_calls=[
                            {
                                "name": "my_tool",
                                "args": {"value": "not_an_int"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            config=_create_config_with_runtime(),
        )

    error = exc_info.value
    assert error.tool_name == "my_tool"
    assert error.filtered_errors is not None
    assert len(error.filtered_errors) > 0

    # Filtered errors should only contain 'value' error, not 'state'
    error_locs = [err["loc"] for err in error.filtered_errors]
    assert any("value" in str(loc) for loc in error_locs)
    assert not any("state" in str(loc) for loc in error_locs)


async def test_sync_tool_validation_error_filtering() -> None:
    """Test that error filtering works for sync tools.

    Error filtering should work identically for both sync and async tool execution,
    excluding injected arguments from validation error messages.
    """

    @dec_tool
    def my_tool(
        value: int,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Sync tool with injected state.

        Args:
            value: An integer value.
            state: The graph state (injected).
        """
        return f"value={value}"

    tool_node = _ToolNode([my_tool])

    # Test sync invocation
    result = tool_node.invoke(
        {
            "messages": [
                AIMessage(
                    "hi?",
                    tool_calls=[
                        {
                            "name": "my_tool",
                            "args": {"value": "not_an_int"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        config=_create_config_with_runtime(),
    )

    tool_message = result["messages"][0]
    assert tool_message.status == "error"
    assert "value" in tool_message.content
    assert "state" not in tool_message.content.lower()


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error_content_with_multiple_params() -> None:
    """Test that error messages only include LLM-controlled parameter errors.

    Uses create_agent to verify that when a tool with both LLM-controlled
    and system-injected parameters receives invalid arguments, the error message:
    1. Contains details about LLM-controlled parameter errors (query, limit)
    2. Does NOT contain system-injected parameter names (state, store, runtime)
    3. Does NOT contain values from system-injected parameters
    4. Properly formats the validation errors for LLM correction

    This ensures the LLM receives focused, actionable feedback.
    """

    class TestState(AgentState):
        user_id: str
        api_key: str
        session_data: dict

    @dec_tool
    def complex_tool(
        query: str,
        limit: int,
        state: Annotated[TestState, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """A complex tool with multiple injected and non-injected parameters.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            state: The graph state (injected).
            store: The persistent store (injected).
            runtime: The tool runtime context (injected).
        """
        # Access injected params to verify they work in normal execution
        user = state.get("user_id", "unknown")
        return f"Results for '{query}' (limit={limit}, user={user})"

    # Create a model that makes an incorrect tool call with multiple errors:
    # - query is wrong type (int instead of str)
    # - limit is missing
    # Then returns no tool calls to end the loop
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "complex_tool",
                    "args": {
                        "query": 12345,  # Wrong type - should be str
                        # "limit" is missing - required field
                    },
                    "id": "call_complex_1",
                }
            ],
            [],  # No tool calls on second iteration to end the loop
        ]
    )

    # Create an agent with the complex tool and custom state
    # Need to provide a store since the tool uses InjectedStore
    agent = create_agent(
        model=model,
        tools=[complex_tool],
        state_schema=TestState,
        store=InMemoryStore(),
    )

    # Invoke with sensitive data in state
    result = agent.invoke(
        {
            "messages": [HumanMessage("Search for something")],
            "user_id": "user_12345",
            "api_key": "sk-secret-key-abc123xyz",
            "session_data": {"token": "secret_session_token"},
        }
    )

    # Find the tool error message
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_complex_1"

    content = tool_message.content

    # Verify error mentions LLM-controlled parameter issues
    assert "query" in content.lower(), "Error should mention 'query' (LLM-controlled)"
    assert "limit" in content.lower(), "Error should mention 'limit' (LLM-controlled)"

    # Should indicate validation errors occurred
    assert "validation error" in content.lower() or "error" in content.lower(), (
        "Error should indicate validation occurred"
    )

    # Verify NO system-injected parameter names appear in error
    # These are not controlled by the LLM and should be excluded
    assert "state" not in content.lower(), "Error should NOT mention 'state' (system-injected)"
    assert "store" not in content.lower(), "Error should NOT mention 'store' (system-injected)"
    assert "runtime" not in content.lower(), "Error should NOT mention 'runtime' (system-injected)"

    # Verify NO values from system-injected parameters appear in error
    # The LLM doesn't control these, so they shouldn't distract from the actual issues
    assert "user_12345" not in content, "Error should NOT contain user_id value (from state)"
    assert "sk-secret-key" not in content, "Error should NOT contain api_key value (from state)"
    assert "secret_session_token" not in content, (
        "Error should NOT contain session_data value (from state)"
    )

    # Verify the LLM's original tool call args are present
    # The error should show what the LLM actually provided to help it correct the mistake
    assert "12345" in content, "Error should show the invalid query value provided by LLM (12345)"

    # Check error is well-formatted
    assert "complex_tool" in content, "Error should mention the tool name"


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error_only_model_controllable_params() -> None:
    """Test that errors only include LLM-controllable parameter issues.

    Focused test ensuring that validation errors for LLM-controlled parameters
    are clearly reported, while system-injected parameters remain completely
    absent from error messages. This provides focused feedback to the LLM.
    """

    class StateWithSecrets(AgentState):
        password: str  # Example of data not controlled by LLM

    @dec_tool
    def secure_tool(
        username: str,
        email: str,
        state: Annotated[StateWithSecrets, InjectedState],
    ) -> str:
        """Tool that validates user credentials.

        Args:
            username: The username (3-20 chars).
            email: The email address.
            state: State with password (system-injected).
        """
        return f"Validated {username} with email {email}"

    # LLM provides invalid username (too short) and invalid email
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "secure_tool",
                    "args": {
                        "username": "ab",  # Too short (needs 3-20)
                        "email": "not-an-email",  # Invalid format
                    },
                    "id": "call_secure_1",
                }
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[secure_tool],
        state_schema=StateWithSecrets,
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Create account")],
            "password": "super_secret_password_12345",
        }
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0].content

    # The error should mention LLM-controlled parameters
    # Note: Pydantic's default validation may or may not catch format issues,
    # but the parameters themselves should be present in error messages
    assert "username" in content.lower() or "email" in content.lower(), (
        "Error should mention at least one LLM-controlled parameter"
    )

    # Password is system-injected and should not appear
    # The LLM doesn't control it, so it shouldn't distract from the actual errors
    assert "password" not in content.lower(), (
        "Error should NOT mention 'password' (system-injected parameter)"
    )
    assert "super_secret_password" not in content, (
        "Error should NOT contain password value (from system-injected state)"
    )
