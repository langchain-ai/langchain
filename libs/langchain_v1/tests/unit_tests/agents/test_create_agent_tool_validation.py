import sys
from typing import Annotated, Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState
from langchain.tools import tool as dec_tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class UserState(AgentState[Any]):
    user_id: str
    api_key: str
    session_data: dict[str, Any]


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

    class TestState(AgentState[Any]):
        secret_data: str

    @dec_tool
    def tool_with_injected_state(
        some_val: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Tool that uses injected state."""
        _ = (some_val, state)
        return "ok"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "tool_with_injected_state",
                    "args": {"wrong_arg": "value"},
                    "id": "call_1",
                }
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[tool_with_injected_state],
        state_schema=TestState,
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test message")],
            "secret_data": "sensitive_secret_123",
        }
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

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

    class TestState(AgentState[Any]):
        internal_data: str

    @dec_tool
    async def async_tool_with_injected_state(
        query: str,
        max_results: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Async tool that uses injected state."""
        _ = (query, max_results, state)
        return "ok"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "async_tool_with_injected_state",
                    "args": {"query": 999},
                    "id": "call_async_1",
                }
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[async_tool_with_injected_state],
        state_schema=TestState,
    )

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Test async")],
            "internal_data": "secret_internal_value_xyz",
        }
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

    content = tool_message.content
    assert "query" in content.lower(), "Error should mention 'query' (LLM-controlled)"
    assert "max_results" in content.lower(), "Error should mention 'max_results' (LLM-controlled)"
    assert "internal_data" not in content, (
        "Error should NOT mention 'internal_data' (system-injected field)"
    )
    assert "secret_internal_value" not in content, (
        "Error should NOT contain system-injected state values"
    )

    lines = content.split("\n")
    error_lines = [line.strip() for line in lines if line.strip()]
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
    assert not any(field.lower() == "state" for field in field_errors), (
        "The field 'state' (system-injected) should not appear in validation errors"
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
def test_create_agent_error() -> None:
    """Test that error messages only include LLM-controlled parameter errors.

    Uses create_agent to verify that when a tool with both LLM-controlled
    and system-injected parameters receives invalid arguments, the error message:
    1. Contains details about LLM-controlled parameter errors (query, limit)
    2. Does NOT contain system-injected parameter names (state, store, runtime)
    3. Does NOT contain values from system-injected parameters
    4. Properly formats the validation errors for LLM correction
    This ensures the LLM receives focused, actionable feedback.
    """

    @dec_tool
    def complex_tool(
        query: str,
        limit: int,
        state: Annotated[UserState, InjectedState],
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
        _ = (query, limit, state, store, runtime)
        return "ok"

    agent, payload = _build_complex_agent(complex_tool)
    _assert_agent_error(agent.invoke(payload))


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error_async() -> None:
    """Test that error messages only include LLM-controlled parameter errors.

    Uses create_agent to verify that when a tool with both LLM-controlled
    and system-injected parameters receives invalid arguments, the error message:
    1. Contains details about LLM-controlled parameter errors (query, limit)
    2. Does NOT contain system-injected parameter names (state, store, runtime)
    3. Does NOT contain values from system-injected parameters
    4. Properly formats the validation errors for LLM correction
    This ensures the LLM receives focused, actionable feedback.
    """

    @dec_tool
    async def complex_tool(
        query: str,
        limit: int,
        state: Annotated[UserState, InjectedState],
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
        _ = (query, limit, state, store, runtime)
        return "ok"

    agent, payload = _build_complex_agent(complex_tool)
    _assert_agent_error(await agent.ainvoke(payload))


def _build_complex_agent(tool: BaseTool) -> tuple[Any, dict[str, Any]]:
    """Build an agent and invocation payload shared by the complex-tool error tests.

    The model issues a single tool call with multiple errors:
    - `query` has the wrong type (int instead of str)
    - `wrong_arg` is not an expected parameter
    - `limit` is missing (required)
    Then returns no tool calls so the agent loop terminates.

    Args:
        tool: The complex tool (sync or async) under test.

    Returns:
        The compiled agent and the invocation payload (with sensitive state values).
    """
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "complex_tool",
                    "args": {
                        "query": 12345,  # Wrong type - should be str
                        "wrong_arg": "value",  # Not an expected parameter
                        # "limit" is missing - required field
                    },
                    "id": "call_complex_1",
                }
            ],
            [],  # No tool calls on second iteration to end the loop
        ]
    )

    # A store is required because the tool uses InjectedStore.
    agent = create_agent(
        model=model,
        tools=[tool],
        state_schema=UserState,
        store=InMemoryStore(),
    )

    payload = {
        "messages": [HumanMessage("Search for something")],
        "user_id": "user_12345",
        "api_key": "sk-secret-key-abc123xyz",
        "session_data": {"token": "secret_session_token"},
    }

    return agent, payload


def _assert_agent_error(result: dict[str, Any]) -> None:
    # Find the tool error message
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_complex_1"

    content = tool_message.content.lower()
    assert "with error:" in content, "Error should be formatted with an error section"
    _, _, error = content.partition("with error:")

    # Verify the error section mentions LLM-controlled parameter issues
    assert "query" in error, "Error should mention 'query' (LLM-controlled)"
    assert "limit" in error, "Error should mention 'limit' (LLM-controlled)"

    # The LLM's original (invalid) args are echoed back so it can self-correct
    assert "wrong_arg" in content, "Error should echo the LLM-provided 'wrong_arg'"
    assert "12345" in content, "Error should show the invalid query value provided by LLM (12345)"
    assert "complex_tool" in content, "Error should mention the tool name"

    # Verify NO system-injected parameter names appear in error
    # These are not controlled by the LLM and should be excluded
    assert "state" not in content, "Error should NOT mention 'state' (system-injected)"
    assert "store" not in content, "Error should NOT mention 'store' (system-injected)"
    assert "runtime" not in content, "Error should NOT mention 'runtime' (system-injected)"

    # Verify NO values from system-injected parameters appear in error
    # The LLM doesn't control these, so they shouldn't distract from the actual issues
    assert "user_12345" not in content, "Error should NOT contain user_id value (from state)"
    assert "sk-secret-key" not in content, "Error should NOT contain api_key value (from state)"
    assert "secret_session_token" not in content, (
        "Error should NOT contain session_data value (from state)"
    )


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
        password: str

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
        _ = state
        return f"Validated {username} with email {email}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "secure_tool",
                    "args": {
                        "username": "ab",
                        "email": "not-an-email",
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

    assert "username" in content.lower() or "email" in content.lower(), (
        "Error should mention at least one LLM-controlled parameter"
    )
    assert "password" not in content.lower(), (
        "Error should NOT mention 'password' (system-injected parameter)"
    )
    assert "super_secret_password" not in content, (
        "Error should NOT contain password value (from system-injected state)"
    )
