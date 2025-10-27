"""Tests for AgentRuntime functionality in middleware."""

from dataclasses import dataclass

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentRuntime,
    before_agent,
    before_model,
    wrap_model_call,
)
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


def test_agent_runtime_structure():
    """Test that AgentRuntime has correct structure."""

    @dataclass
    class LocalContext:
        user_id: str

    context = LocalContext(user_id="test_user")
    agent_runtime = AgentRuntime(
        agent_name="TestAgent",
        context=context,
        store=None,
        stream_writer=None,
        previous=None,
    )

    assert agent_runtime.agent_name == "TestAgent"
    assert agent_runtime.context == context
    assert agent_runtime.context.user_id == "test_user"
    assert agent_runtime.store is None


def test_agent_runtime_in_before_agent_hook(fake_chat_model):
    """Test that AgentRuntime is correctly passed to before_agent hooks."""
    captured_agent_name = None
    captured_context = None

    @before_agent
    def capture_runtime(state, runtime: AgentRuntime):
        nonlocal captured_agent_name, captured_context
        captured_agent_name = runtime.agent_name
        captured_context = runtime.context
        return None

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_runtime],
        name="MyTestAgent",
        context_schema=Context,
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        context=Context(user_id="user123", session_id="session456"),
    )

    assert captured_agent_name == "MyTestAgent"
    assert captured_context.user_id == "user123"
    assert captured_context.session_id == "session456"


def test_agent_runtime_in_before_model_hook(fake_chat_model):
    """Test that AgentRuntime is correctly passed to before_model hooks."""
    captured_agent_name = None
    captured_context = None

    @before_model
    def capture_runtime(state, runtime: AgentRuntime):
        nonlocal captured_agent_name, captured_context
        captured_agent_name = runtime.agent_name
        captured_context = runtime.context
        return None

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_runtime],
        name="AnotherAgent",
        context_schema=Context,
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        context=Context(user_id="user456", session_id="session789"),
    )

    assert captured_agent_name == "AnotherAgent"
    assert captured_context.user_id == "user456"
    assert captured_context.session_id == "session789"


def test_agent_runtime_in_wrap_model_call(fake_chat_model):
    """Test that AgentRuntime is accessible via ModelRequest in wrap_model_call."""
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
        name="WrapAgent",
        context_schema=Context,
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        context=Context(user_id="user789", session_id="session123"),
    )

    assert captured_agent_name == "WrapAgent"
    assert captured_context.user_id == "user789"
    assert captured_context.session_id == "session123"


def test_agent_runtime_default_name(fake_chat_model):
    """Test that AgentRuntime uses default name when not specified."""
    captured_agent_name = None

    @before_agent
    def capture_name(state, runtime: AgentRuntime):
        nonlocal captured_agent_name
        captured_agent_name = runtime.agent_name
        return None

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[capture_name],
        # name not specified - should default to "LangGraph"
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert captured_agent_name == "LangGraph"


def test_agent_runtime_with_store(fake_chat_model):
    """Test that AgentRuntime provides access to store directly."""
    store = InMemoryStore()
    store.put(("test",), "key1", {"value": "test_data"})

    captured_store_value = None

    @before_agent
    def access_store(state, runtime: AgentRuntime):
        nonlocal captured_store_value
        if runtime.store:
            item = runtime.store.get(("test",), "key1")
            if item:
                captured_store_value = item.value
        return None

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[access_store],
        name="StoreAgent",
        store=store,
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert captured_store_value == {"value": "test_data"}


async def test_agent_runtime_async_hooks(fake_chat_model):
    """Test that AgentRuntime works with async middleware hooks."""
    captured_agent_name = None

    @before_agent
    async def async_capture(state, runtime: AgentRuntime):
        nonlocal captured_agent_name
        captured_agent_name = runtime.agent_name
        return None

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[async_capture],
        name="AsyncAgent",
    )

    await agent.ainvoke({"messages": [HumanMessage("Hello")]})

    assert captured_agent_name == "AsyncAgent"


def test_agent_runtime_multiple_middleware(fake_chat_model):
    """Test that all middleware receive correct AgentRuntime."""
    captured_names = []

    @before_agent(name="first")
    def first_middleware(state, runtime: AgentRuntime):
        captured_names.append(("first_before_agent", runtime.agent_name))
        return None

    @before_model(name="second")
    def second_middleware(state, runtime: AgentRuntime):
        captured_names.append(("second_before_model", runtime.agent_name))
        return None

    @wrap_model_call(name="third")
    def third_middleware(request: ModelRequest, handler):
        captured_names.append(("third_wrap_model", request.runtime.agent_name))
        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[first_middleware, second_middleware, third_middleware],
        name="MultiMiddlewareAgent",
    )

    agent.invoke({"messages": [HumanMessage("Hello")]})

    assert len(captured_names) == 3
    assert all(name == "MultiMiddlewareAgent" for _, name in captured_names)
    assert captured_names[0][0] == "first_before_agent"
    assert captured_names[1][0] == "second_before_model"
    assert captured_names[2][0] == "third_wrap_model"


def test_model_request_runtime_field_type():
    """Test that ModelRequest.runtime field has correct type."""
    from langchain.agents.middleware.types import ModelRequest

    # Check that the runtime field is AgentRuntime, not Runtime
    annotations = ModelRequest.__annotations__
    runtime_annotation = str(annotations["runtime"])

    # Should contain AgentRuntime
    assert "AgentRuntime" in runtime_annotation


def test_agent_runtime_access_pattern(fake_chat_model):
    """Test the recommended pattern for accessing agent name and context."""

    @wrap_model_call
    def middleware_using_runtime(request: ModelRequest, handler):
        # Access agent name directly
        agent_name = request.runtime.agent_name

        # Access runtime properties directly (flat structure)
        user_id = request.runtime.context.user_id if request.runtime.context else None
        store = request.runtime.store

        assert agent_name == "PatternTestAgent"
        assert user_id == "pattern_user"
        assert store is not None

        return handler(request)

    agent = create_agent(
        fake_chat_model,
        tools=[],
        middleware=[middleware_using_runtime],
        name="PatternTestAgent",
        context_schema=Context,
        store=InMemoryStore(),
    )

    agent.invoke(
        {"messages": [HumanMessage("Hello")]},
        context=Context(user_id="pattern_user", session_id="pattern_session"),
    )
