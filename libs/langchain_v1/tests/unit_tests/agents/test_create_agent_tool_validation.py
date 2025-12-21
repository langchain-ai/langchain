import sys
from collections.abc import Callable
from inspect import isawaitable
from typing import Annotated

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import StringConstraints

from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState
from langchain.tools import tool as dec_tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class UserState(AgentState):
    user_id: str
    api_key: Annotated[str, StringConstraints(min_length=50)]
    session_data: dict


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error() -> None:
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

    await _verify_agent_error(complex_tool, lambda agent, *args: agent.invoke(*args))


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

    async def invoke_async(agent, *args):
        return await agent.ainvoke(*args)

    await _verify_agent_error(complex_tool, invoke_async)


async def _verify_agent_error(tool: BaseTool, agent_func: Callable) -> None:
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
                        "wrong_arg": "value",  # Not expected
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
        tools=[tool],
        state_schema=UserState,
        store=InMemoryStore(),
    )

    result_ = agent_func(
        agent,
        {
            "messages": [HumanMessage("Search for something")],
            "user_id": "user_12345",
            "api_key": "sk-secret-key-abc123xyz",
            "session_data": {"token": "secret_session_token"},
        },
    )

    result = await result_ if isawaitable(result_) else result_

    # Find the tool error message
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_complex_1"

    content = tool_message.content.lower()
    error = content.split("with error:")[1]

    # Verify error mentions LLM-controlled parameter issues
    assert "query" in error, "Error should mention 'query' (LLM-controlled)"
    assert "limit" in error, "Error should mention 'limit' (LLM-controlled)"

    # Verify unexpected argument is mentioned
    assert "wrong_arg" in content, "Error should mention 'wrong_type'"

    # Should indicate validation errors occurred
    assert "error" in content, "Error should indicate validation occurred"

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

    # Verify the LLM's original tool call args are present
    # The error should show what the LLM actually provided to help it correct the mistake
    assert "12345" in content, "Error should show the invalid query value provided by LLM (12345)"

    # Check error is well-formatted
    assert "complex_tool" in content, "Error should mention the tool name"
