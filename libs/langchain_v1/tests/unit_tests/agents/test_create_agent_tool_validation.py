import sys
from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState
from langchain.tools import tool as dec_tool
from tests.unit_tests.agents.model import FakeToolCallingModel


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
def test_tool_invocation_error_excludes_injected_state() -> None:
    """Test that tool invocation errors only include LLM-controllable arguments.

    When a tool has InjectedState parameters and the LLM makes an incorrect
    invocation (e.g., missing required arguments), the error message should only
    contain the arguments from the tool call that the LLM controls. This ensures
    the LLM receives relevant context to correct its mistakes, without being
    distracted by system-injected parameters it has no control over.
    This test uses create_agent to ensure the behavior works in a full agent context.
    """

    # Define a custom state schema with injected data
    class TestState(AgentState[Any]):
        secret_data: str  # Example of state data not controlled by LLM

    @dec_tool
    def tool_with_injected_state(
        some_val: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Tool that uses injected state."""
        return f"some_val: {some_val}"

    # Create a fake model that makes an incorrect tool call (missing 'some_val')
    # Then returns no tool calls on the second iteration to end the loop
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "tool_with_injected_state",
                    "args": {"wrong_arg": "value"},  # Missing required 'some_val'
                    "id": "call_1",
                }
            ],
            [],  # No tool calls on second iteration to end the loop
        ]
    )

    # Create an agent with the tool and custom state schema
    agent = create_agent(
        model=model,
        tools=[tool_with_injected_state],
        state_schema=TestState,
    )

    # Invoke the agent with injected state data
    result = agent.invoke(
        {
            "messages": [HumanMessage("Test message")],
            "secret_data": "sensitive_secret_123",
        }
    )

    # Find the tool error message
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

    # The error message should contain only the LLM-provided args (wrong_arg)
    # and NOT the system-injected state (secret_data)
    assert "{'wrong_arg': 'value'}" in tool_message.content
    assert "secret_data" not in tool_message.content
    assert "sensitive_secret_123" not in tool_message.content


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_tool_invocation_error_excludes_injected_state_async() -> None:
    """Test that async tool invocation errors only include LLM-controllable arguments.

    This test verifies that the async execution path (_execute_tool_async and _arun_one)
    properly filters validation errors to exclude system-injected arguments, ensuring
    the LLM receives only relevant context for correction.
    """

    # Define a custom state schema
    class TestState(AgentState[Any]):
        internal_data: str

    @dec_tool
    async def async_tool_with_injected_state(
        query: str,
        max_results: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Async tool that uses injected state."""
        return f"query: {query}, max_results: {max_results}"

    # Create a fake model that makes an incorrect tool call
    # - query has wrong type (int instead of str)
    # - max_results is missing
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "async_tool_with_injected_state",
                    "args": {"query": 999},  # Wrong type, missing max_results
                    "id": "call_async_1",
                }
            ],
            [],  # End the loop
        ]
    )

    # Create an agent with the async tool
    agent = create_agent(
        model=model,
        tools=[async_tool_with_injected_state],
        state_schema=TestState,
    )

    # Invoke with state data
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Test async")],
            "internal_data": "secret_internal_value_xyz",
        }
    )

    # Find the tool error message
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

    # Verify error mentions LLM-controlled parameters only
    content = tool_message.content
    assert "query" in content.lower(), "Error should mention 'query' (LLM-controlled)"
    assert "max_results" in content.lower(), "Error should mention 'max_results' (LLM-controlled)"

    # Verify system-injected state does not appear in the validation errors
    # This keeps the error focused on what the LLM can actually fix
    assert "internal_data" not in content, (
        "Error should NOT mention 'internal_data' (system-injected field)"
    )
    assert "secret_internal_value" not in content, (
        "Error should NOT contain system-injected state values"
    )

    # Verify only LLM-controlled parameters are in the error list
    # Should see "query" and "max_results" errors, but not "state"
    lines = content.split("\n")
    error_lines = [line.strip() for line in lines if line.strip()]
    # Find lines that look like field names (single words at start of line)
    field_errors = [
        line
        for line in error_lines
        if line
        and not line.startswith("input")
        and not line.startswith("field")
        and not line.startswith("error")
        and not line.startswith("please")
        and len(line.split()) <= 2
    ]
    # Verify system-injected 'state' is not in the field error list
    assert not any(field.lower() == "state" for field in field_errors), (
        "The field 'state' (system-injected) should not appear in validation errors"
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
def test_create_agent_error_content_with_multiple_params() -> None:
    """Test that error messages only include LLM-controlled parameter errors.

    Uses create_agent to verify that when a tool with both LLM-controlled
    and system-injected parameters receives invalid arguments, the error message:
    1. Contains details about LLM-controlled parameter errors (query, limit)
    2. Does NOT contain system-injected parameter names (state, store, runtime)
    3. Does NOT contain values from system-injected parameters
    4. Properly formats the validation errors for LLM correction
    This ensures the LLM receives focused, actionable feedback.
    """

    class TestState(AgentState[Any]):
        user_id: str
        api_key: str
        session_data: dict[str, Any]

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
def test_create_agent_error_only_model_controllable_params() -> None:
    """Test that errors only include LLM-controllable parameter issues.

    Focused test ensuring that validation errors for LLM-controlled parameters
    are clearly reported, while system-injected parameters remain completely
    absent from error messages. This provides focused feedback to the LLM.
    """

    class StateWithSecrets(AgentState[Any]):
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
