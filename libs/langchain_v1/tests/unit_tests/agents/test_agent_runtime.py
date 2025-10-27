"""Tests for wrap_model_call and awrap_model_call functionality."""

from dataclasses import dataclass

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse


@dataclass
class Context:
    """Test context for agent runtime."""

    user_id: str
    session_id: str


@pytest.fixture
def fake_chat_model():
    """Fixture providing a fake chat model for testing."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="test response")]))


def test_wrap_model_call_basic(fake_chat_model):
    """Test basic wrap_model_call functionality."""
    call_count = 0

    @wrap_model_call
    def count_calls(request: ModelRequest, handler):
        nonlocal call_count
        call_count += 1
        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[count_calls],
        name="TestAgent",
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})
    assert call_count == 1


def test_wrap_model_call_access_runtime(fake_chat_model):
    """Test accessing AgentRuntime via ModelRequest in wrap_model_call."""
    captured_agent_name = None
    captured_context = None

    @wrap_model_call
    def capture_from_request(request: ModelRequest, handler):
        nonlocal captured_agent_name, captured_context
        captured_agent_name = request.runtime.agent_name
        captured_context = request.runtime.context
        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_from_request],
        name="RuntimeAgent",
        context_schema=Context,
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        context=Context(user_id="user123", session_id="session456"),
    )

    assert captured_agent_name == "RuntimeAgent"
    assert captured_context.user_id == "user123"
    assert captured_context.session_id == "session456"


def test_wrap_model_call_modify_request(fake_chat_model):
    """Test modifying the model request in wrap_model_call."""
    modified_messages = []

    @wrap_model_call
    def modify_request(request: ModelRequest, handler):
        # Add a system prompt
        modified_request = request.override(system_prompt="You are a helpful assistant")
        modified_messages.append(modified_request.system_prompt)
        return handler(modified_request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[modify_request],
        name="ModifyAgent",
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})
    assert modified_messages[0] == "You are a helpful assistant"


def test_wrap_model_call_modify_response(fake_chat_model):
    """Test modifying the model response in wrap_model_call."""

    @wrap_model_call
    def modify_response(request: ModelRequest, handler):
        response = handler(request)
        # Modify the response content
        original_msg = response.result[0]
        modified_msg = AIMessage(
            content=f"[MODIFIED] {original_msg.content}",
            id=original_msg.id,
        )
        return ModelResponse(
            result=[modified_msg],
            structured_response=response.structured_response,
        )

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[modify_response],
        name="ModifyResponseAgent",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][-1].content == "[MODIFIED] test response"


def test_wrap_model_call_retry_logic(fake_chat_model):
    """Test retry logic in wrap_model_call."""
    attempt_count = 0
    model_call_count = 0

    @wrap_model_call
    def retry_on_error(request: ModelRequest, handler):
        nonlocal attempt_count, model_call_count
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            attempt_count += 1
            try:
                # Simulate failure on first two attempts
                model_call_count += 1
                if model_call_count < 3:
                    raise ValueError("Simulated failure")
                return handler(request)
            except ValueError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise

        raise last_error  # Should never reach here

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[retry_on_error],
        name="RetryAgent",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert attempt_count == 3
    # The model response should be from fake_chat_model
    assert result["messages"][-1].content == "test response"


def test_wrap_model_call_short_circuit(fake_chat_model):
    """Test short-circuiting model call in wrap_model_call."""
    handler_called = False

    @wrap_model_call
    def short_circuit(request: ModelRequest, handler):
        nonlocal handler_called
        # Check if we should short-circuit
        if len(request.messages) > 0 and "bypass" in request.messages[-1].content:
            # Return cached response without calling handler
            return AIMessage(content="Cached response")

        handler_called = True
        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[short_circuit],
        name="ShortCircuitAgent",
    )

    result = agent.invoke({"messages": [HumanMessage("bypass")]})
    assert not handler_called
    assert result["messages"][-1].content == "Cached response"


def test_wrap_model_call_multiple_middleware(fake_chat_model):
    """Test composing multiple wrap_model_call middleware."""
    execution_order = []

    @wrap_model_call(name="first")
    def first_middleware(request: ModelRequest, handler):
        execution_order.append("first_before")
        response = handler(request)
        execution_order.append("first_after")
        return response

    @wrap_model_call(name="second")
    def second_middleware(request: ModelRequest, handler):
        execution_order.append("second_before")
        response = handler(request)
        execution_order.append("second_after")
        return response

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[first_middleware, second_middleware],
        name="MultiWrapAgent",
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    # Middleware should compose as: first -> second -> model -> second -> first
    assert execution_order == [
        "first_before",
        "second_before",
        "second_after",
        "first_after",
    ]


async def test_awrap_model_call_basic(fake_chat_model):
    """Test basic awrap_model_call functionality."""
    call_count = 0

    @wrap_model_call
    async def count_calls_async(request: ModelRequest, handler):
        nonlocal call_count
        call_count += 1
        return await handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[count_calls_async],
        name="AsyncTestAgent",
    )

    await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert call_count == 1


async def test_awrap_model_call_access_runtime(fake_chat_model):
    """Test accessing AgentRuntime in async wrap_model_call."""
    captured_agent_name = None

    @wrap_model_call
    async def capture_async(request: ModelRequest, handler):
        nonlocal captured_agent_name
        captured_agent_name = request.runtime.agent_name
        return await handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_async],
        name="AsyncRuntimeAgent",
    )

    await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert captured_agent_name == "AsyncRuntimeAgent"


async def test_awrap_model_call_retry_logic(fake_chat_model):
    """Test async retry logic in awrap_model_call."""
    attempt_count = 0
    model_call_count = 0

    @wrap_model_call
    async def async_retry_on_error(request: ModelRequest, handler):
        nonlocal attempt_count, model_call_count
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            attempt_count += 1
            try:
                # Simulate failure on first two attempts
                model_call_count += 1
                if model_call_count < 3:
                    raise ValueError("Simulated async failure")
                return await handler(request)
            except ValueError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise

        raise last_error  # Should never reach here

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[async_retry_on_error],
        name="AsyncRetryAgent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert attempt_count == 3
    # The model response should be from fake_chat_model
    assert result["messages"][-1].content == "test response"


async def test_awrap_model_call_modify_response(fake_chat_model):
    """Test modifying response in async wrap_model_call."""

    @wrap_model_call
    async def async_modify_response(request: ModelRequest, handler):
        response = await handler(request)
        original_msg = response.result[0]
        modified_msg = AIMessage(
            content=f"[ASYNC MODIFIED] {original_msg.content}",
            id=original_msg.id,
        )
        return ModelResponse(
            result=[modified_msg],
            structured_response=response.structured_response,
        )

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[async_modify_response],
        name="AsyncModifyAgent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][-1].content == "[ASYNC MODIFIED] test response"
