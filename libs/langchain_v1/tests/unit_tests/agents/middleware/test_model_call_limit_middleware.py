import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitMiddleware,
    ModelCallLimitExceededError,
)

from ..model import FakeToolCallingModel


@tool
def simple_tool(input: str) -> str:
    """A simple tool"""
    return input


def test_middleware_unit_functionality():
    """Test that the middleware works as expected in isolation."""
    # Test with end behavior
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1)

    # Mock runtime (not used in current implementation)
    runtime = None

    # Test when limits are not exceeded
    state = {"thread_model_call_count": 0, "run_model_call_count": 0}
    result = middleware.before_model(state, runtime)
    assert result is None

    # Test when thread limit is exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "thread limit (2/2)" in result["messages"][0].content

    # Test when run limit is exceeded
    state = {"thread_model_call_count": 1, "run_model_call_count": 1}
    result = middleware.before_model(state, runtime)
    assert result is not None
    assert result["jump_to"] == "end"
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "run limit (1/1)" in result["messages"][0].content

    # Test with error behavior
    middleware_exception = ModelCallLimitMiddleware(
        thread_limit=2, run_limit=1, exit_behavior="error"
    )

    # Test exception when thread limit exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "thread limit (2/2)" in str(exc_info.value)

    # Test exception when run limit exceeded
    state = {"thread_model_call_count": 1, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware_exception.before_model(state, runtime)

    assert "run limit (1/1)" in str(exc_info.value)


def test_thread_limit_with_create_agent():
    """Test that thread limits work correctly with create_agent."""
    model = FakeToolCallingModel()

    # Set thread limit to 1 (should be exceeded after 1 call)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[ModelCallLimitMiddleware(thread_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # First invocation should work - 1 model call, within thread limit
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    # Should complete successfully with 1 model call
    assert "messages" in result
    assert len(result["messages"]) == 2  # Human + AI messages

    # Second invocation in same thread should hit thread limit
    # The agent should jump to end after detecting the limit
    result2 = agent.invoke(
        {"messages": [HumanMessage("Hello again")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result2
    # The agent should have detected the limit and jumped to end with a limit exceeded message
    # So we should have: previous messages + new human message + limit exceeded AI message
    assert len(result2["messages"]) == 4  # Previous Human + AI + New Human + Limit AI
    assert isinstance(result2["messages"][0], HumanMessage)  # First human
    assert isinstance(result2["messages"][1], AIMessage)  # First AI response
    assert isinstance(result2["messages"][2], HumanMessage)  # Second human
    assert isinstance(result2["messages"][3], AIMessage)  # Limit exceeded message
    assert "thread limit" in result2["messages"][3].content


def test_run_limit_with_create_agent():
    """Test that run limits work correctly with create_agent."""
    # Create a model that will make 2 calls
    model = FakeToolCallingModel(
        tool_calls=[
            [{"name": "simple_tool", "args": {"input": "test"}, "id": "1"}],
            [],  # No tool calls on second call
        ]
    )

    # Set run limit to 1 (should be exceeded after 1 call)
    agent = create_agent(
        model=model,
        tools=[simple_tool],
        middleware=[ModelCallLimitMiddleware(run_limit=1)],
        checkpointer=InMemorySaver(),
    )

    # This should hit the run limit after the first model call
    result = agent.invoke(
        {"messages": [HumanMessage("Hello")]}, {"configurable": {"thread_id": "thread1"}}
    )

    assert "messages" in result
    # The agent should have made 1 model call then jumped to end with limit exceeded message
    # So we should have: Human + AI + Tool + Limit exceeded AI message
    assert len(result["messages"]) == 4  # Human + AI + Tool + Limit AI
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
    assert isinstance(result["messages"][3], AIMessage)  # Limit exceeded message
    assert "run limit" in result["messages"][3].content


def test_middleware_initialization_validation():
    """Test that middleware initialization validates parameters correctly."""
    # Test that at least one limit must be specified
    with pytest.raises(ValueError, match="At least one limit must be specified"):
        ModelCallLimitMiddleware()

    # Test invalid exit behavior
    with pytest.raises(ValueError, match="Invalid exit_behavior"):
        ModelCallLimitMiddleware(thread_limit=5, exit_behavior="invalid")

    # Test valid initialization
    middleware = ModelCallLimitMiddleware(thread_limit=5, run_limit=3)
    assert middleware.thread_limit == 5
    assert middleware.run_limit == 3
    assert middleware.exit_behavior == "end"

    # Test with only thread limit
    middleware = ModelCallLimitMiddleware(thread_limit=5)
    assert middleware.thread_limit == 5
    assert middleware.run_limit is None

    # Test with only run limit
    middleware = ModelCallLimitMiddleware(run_limit=3)
    assert middleware.thread_limit is None
    assert middleware.run_limit == 3


def test_exception_error_message():
    """Test that the exception provides clear error messages."""
    middleware = ModelCallLimitMiddleware(thread_limit=2, run_limit=1, exit_behavior="error")

    # Test thread limit exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 0}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg

    # Test run limit exceeded
    state = {"thread_model_call_count": 0, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "run limit (1/1)" in error_msg

    # Test both limits exceeded
    state = {"thread_model_call_count": 2, "run_model_call_count": 1}
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        middleware.before_model(state, None)

    error_msg = str(exc_info.value)
    assert "Model call limits exceeded" in error_msg
    assert "thread limit (2/2)" in error_msg
    assert "run limit (1/1)" in error_msg


def test_run_limit_resets_between_invocations() -> None:
    """Test that run_model_call_count resets between invocations, but thread_model_call_count accumulates."""

    # First: No tool calls per invocation, so model does not increment call counts internally
    middleware = ModelCallLimitMiddleware(thread_limit=3, run_limit=1, exit_behavior="error")
    model = FakeToolCallingModel(
        tool_calls=[[], [], [], []]
    )  # No tool calls, so only model call per run

    agent = create_agent(model=model, middleware=[middleware], checkpointer=InMemorySaver())

    thread_config = {"configurable": {"thread_id": "test_thread"}}
    agent.invoke({"messages": [HumanMessage("Hello")]}, thread_config)
    agent.invoke({"messages": [HumanMessage("Hello again")]}, thread_config)
    agent.invoke({"messages": [HumanMessage("Hello third")]}, thread_config)

    # Fourth run: should raise, thread_model_call_count == 3 (limit)
    with pytest.raises(ModelCallLimitExceededError) as exc_info:
        agent.invoke({"messages": [HumanMessage("Hello fourth")]}, thread_config)
    error_msg = str(exc_info.value)
    assert "thread limit (3/3)" in error_msg
