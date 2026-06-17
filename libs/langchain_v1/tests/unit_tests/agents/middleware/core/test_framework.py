import sys
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Generic

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
    ResponseT,
    after_agent,
    after_model,
    before_agent,
    before_model,
    hook_config,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import InjectedState
from tests.unit_tests.agents.messages import _AnyIdHumanMessage, _AnyIdToolMessage
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_create_agent_invoke(
    sync_checkpointer: BaseCheckpointSaver[str],
) -> None:
    calls = []

    class NoopSeven(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopEight.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(value: str) -> str:
        """A great tool."""
        calls.append("my_tool")
        return value.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"value": "yo"}, "id": "1", "name": "my_tool"},
                ],
                [],
            ]
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
        checkpointer=sync_checkpointer,
    )

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": ["hello"]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            AIMessage(
                content="You are a helpful assistant.-hello",
                additional_kwargs={},
                response_metadata={},
                id="0",
                tool_calls=[
                    {
                        "name": "my_tool",
                        "args": {"value": "yo"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(content="YO", name="my_tool", tool_call_id="1"),
            AIMessage(
                content="You are a helpful assistant.-hello-You are a helpful assistant.-hello-YO",
                additional_kwargs={},
                response_metadata={},
                id="1",
            ),
        ],
    }
    assert calls == [
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.wrap_model_call",
        "NoopEight.wrap_model_call",
        "NoopEight.after_model",
        "NoopSeven.after_model",
        "my_tool",
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.wrap_model_call",
        "NoopEight.wrap_model_call",
        "NoopEight.after_model",
        "NoopSeven.after_model",
    ]


def test_create_agent_jump(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver[str],
) -> None:
    calls = []

    class NoopSeven(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            calls.append("NoopEight.before_model")
            return {"jump_to": "end"}

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(value: str) -> str:
        """A great tool."""
        calls.append("my_tool")
        return value.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"value": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
        checkpointer=sync_checkpointer,
    )

    if isinstance(sync_checkpointer, InMemorySaver):
        assert agent_one.get_graph().draw_mermaid() == snapshot

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": []}, thread1) == {"messages": []}
    assert calls == ["NoopSeven.before_model", "NoopEight.before_model"]


def test_simple_agent_graph(snapshot: SnapshotAssertion) -> None:
    @tool
    def my_tool(input_string: str) -> str:
        """A great tool."""
        return input_string

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot


def test_agent_graph_with_jump_to_end_as_after_agent(snapshot: SnapshotAssertion) -> None:
    @tool
    def my_tool(input_string: str) -> str:
        """A great tool."""
        return input_string

    class NoopZero(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        @override
        def before_agent(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    class NoopOne(AgentMiddleware):
        @override
        def after_agent(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    class NoopTwo(AgentMiddleware):
        @override
        def after_agent(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopZero(), NoopOne(), NoopTwo()],
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot


def test_on_model_call() -> None:
    class ModifyMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            request.messages.append(HumanMessage("remember to be nice!"))
            return handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[ModifyMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert result["messages"][0].content == "Hello"
    assert result["messages"][1].content == "remember to be nice!"
    assert (
        result["messages"][2].content == "You are a helpful assistant.-Hello-remember to be nice!"
    )


def test_tools_to_model_edge_with_structured_and_regular_tool_calls() -> None:
    """Test tools to model edge with structured and regular tool calls.

    Test that when there are both structured and regular tool calls, we execute regular
    and jump to END.
    """

    class WeatherResponse(BaseModel):
        """Weather response."""

        temperature: float = Field(description="Temperature in fahrenheit")
        condition: str = Field(description="Weather condition")

    @tool
    def regular_tool(query: str) -> str:
        """A regular tool that returns a string."""
        return f"Regular tool result for: {query}"

    # Create a fake model that returns both structured and regular tool calls
    class FakeModelWithBothToolCalls(FakeToolCallingModel):
        def __init__(self) -> None:
            super().__init__()
            self.tool_calls = [
                [
                    ToolCall(
                        name="WeatherResponse",
                        args={"temperature": 72.0, "condition": "sunny"},
                        id="structured_call_1",
                    ),
                    ToolCall(
                        name="regular_tool", args={"query": "test query"}, id="regular_call_1"
                    ),
                ]
            ]

    # Create agent with both structured output and regular tools
    agent = create_agent(
        model=FakeModelWithBothToolCalls(),
        tools=[regular_tool],
        response_format=ToolStrategy(schema=WeatherResponse),
    )

    # Invoke the agent (already compiled)
    result = agent.invoke(
        {"messages": [HumanMessage("What's the weather and help me with a query?")]}
    )

    # Verify that we have the expected messages:
    # 1. Human message
    # 2. AI message with both tool calls
    # 3. Tool message from structured tool call
    # 4. Tool message from regular tool call

    messages = result["messages"]
    assert len(messages) >= 4

    # Check that we have the AI message with both tool calls
    ai_message = messages[1]
    assert isinstance(ai_message, AIMessage)
    assert len(ai_message.tool_calls) == 2

    # Check that we have a tool message from the regular tool
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1

    # The regular tool should have been executed
    regular_tool_message = next((m for m in tool_messages if m.name == "regular_tool"), None)
    assert regular_tool_message is not None
    assert "Regular tool result for: test query" in regular_tool_message.content

    # Verify that the structured response is available in the result
    assert "structured_response" in result
    assert result["structured_response"] is not None
    assert hasattr(result["structured_response"], "temperature")
    assert result["structured_response"].temperature == 72.0
    assert result["structured_response"].condition == "sunny"


def test_public_private_state_for_custom_middleware() -> None:
    """Test public and private state for custom middleware."""

    class CustomState(AgentState[Any]):
        omit_input: Annotated[str, OmitFromInput]
        omit_output: Annotated[str, OmitFromOutput]
        private_state: Annotated[str, PrivateStateAttr]

    class CustomMiddleware(AgentMiddleware[CustomState]):
        state_schema: type[CustomState] = CustomState

        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            assert "omit_input" not in state
            assert "omit_output" in state
            assert "private_state" not in state
            return {"omit_input": "test", "omit_output": "test", "private_state": "test"}

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    result = agent.invoke(
        {
            "messages": [HumanMessage("Hello")],
            "omit_input": "test in",
            "private_state": "test in",
            "omit_output": "test in",
        }
    )
    assert "omit_input" in result
    assert "omit_output" not in result
    assert "private_state" not in result


def test_runtime_injected_into_middleware() -> None:
    """Test that the runtime is injected into the middleware."""

    class CustomMiddleware(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            assert runtime is not None

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            assert request.runtime is not None
            return handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            assert runtime is not None

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    agent.invoke({"messages": [HumanMessage("Hello")]})


# test setup defined at this scope bc of pydantic issues inferring the namespace of
# custom state w/in a function


class CustomState(AgentState[ResponseT], Generic[ResponseT]):
    custom_state: str


@tool(description="Test the state")
def test_state_tool(
    state: Annotated[CustomState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Test tool that accesses injected state."""
    _ = tool_call_id
    assert "custom_state" in state
    return "success"


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState


agent = create_agent(
    model=FakeToolCallingModel(
        tool_calls=[
            [{"args": {}, "id": "test_call_1", "name": "test_state_tool"}],
            [],
        ]
    ),
    tools=[test_state_tool],
    system_prompt="You are a helpful assistant.",
    middleware=[CustomMiddleware()],
)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="pydantic 2.12 namespace management not working w/ 3.14"
)
def test_injected_state_in_middleware_agent() -> None:
    """Test that custom state is properly injected into tools when using middleware."""
    result = agent.invoke(
        {
            "custom_state": "I love pizza",
            "messages": [HumanMessage("Call the test state tool")],
        }
    )

    messages = result["messages"]
    assert len(messages) == 4  # Human message, AI message with tool call, tool message, AI message

    # Find the tool message
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1

    tool_message = tool_messages[0]
    assert tool_message.name == "test_state_tool"
    assert "success" in tool_message.content
    assert tool_message.tool_call_id == "test_call_1"


def test_jump_to_is_ephemeral() -> None:
    class MyMiddleware(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

    agent = create_agent(model=FakeToolCallingModel(), middleware=[MyMiddleware()])
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "jump_to" not in result


def test_create_agent_sync_invoke_with_only_async_middleware_raises_error() -> None:
    """Test that sync invoke with only async middleware works via run_in_executor."""

    class AsyncOnlyMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncOnlyMiddleware()],
    )

    with pytest.raises(NotImplementedError):
        agent.invoke({"messages": [HumanMessage("hello")]})


def test_create_agent_sync_invoke_with_mixed_middleware() -> None:
    """Test that sync invoke works with mixed sync/async middleware when sync versions exist."""
    calls = []

    class MixedMiddleware(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("MixedMiddleware.before_model")

        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("MixedMiddleware.abefore_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("MixedMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("MixedMiddleware.awrap_model_call")
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[MixedMiddleware()],
    )

    agent.invoke({"messages": [HumanMessage("hello")]})

    # In sync mode, only sync methods should be called
    assert calls == [
        "MixedMiddleware.before_model",
        "MixedMiddleware.wrap_model_call",
    ]


# =============================================================================
# Async Middleware Tests
# =============================================================================


async def test_create_agent_async_invoke() -> None:
    """Test async invoke with async middleware hooks."""
    calls = []

    class AsyncMiddleware(AgentMiddleware):
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("AsyncMiddleware.awrap_model_call")
            request.messages.append(HumanMessage("async middleware message"))
            return await handler(request)

        @override
        async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddleware.aafter_model")

    @tool
    def my_tool_async(value: str) -> str:
        """A great tool."""
        calls.append("my_tool_async")
        return value.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"value": "yo"}, "id": "1", "name": "my_tool_async"}],
                [],
            ]
        ),
        tools=[my_tool_async],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddleware()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})

    # Should have:
    # 1. Original hello message
    # 2. Async middleware message (first invoke)
    # 3. AI message with tool call
    # 4. Tool message
    # 5. Async middleware message (second invoke)
    # 6. Final AI message
    assert len(result["messages"]) == 6
    assert result["messages"][0].content == "hello"
    assert result["messages"][1].content == "async middleware message"
    assert calls == [
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
        "my_tool_async",
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
    ]


async def test_create_agent_async_invoke_multiple_middleware() -> None:
    """Test async invoke with multiple async middleware hooks."""
    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("AsyncMiddlewareOne.awrap_model_call")
            return await handler(request)

        @override
        async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddlewareOne.aafter_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddlewareTwo.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("AsyncMiddlewareTwo.awrap_model_call")
            return await handler(request)

        @override
        async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddlewareTwo.aafter_model")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    await agent.ainvoke({"messages": [HumanMessage("hello")]})

    assert calls == [
        "AsyncMiddlewareOne.abefore_model",
        "AsyncMiddlewareTwo.abefore_model",
        "AsyncMiddlewareOne.awrap_model_call",
        "AsyncMiddlewareTwo.awrap_model_call",
        "AsyncMiddlewareTwo.aafter_model",
        "AsyncMiddlewareOne.aafter_model",
    ]


async def test_create_agent_async_jump() -> None:
    """Test async invoke with async middleware using jump_to."""
    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            calls.append("AsyncMiddlewareTwo.abefore_model")
            return {"jump_to": "end"}

    @tool
    def my_tool_jump(value: str) -> str:
        """A great tool."""
        calls.append("my_tool_jump")
        return value.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool_jump", args={"value": "yo"})]],
        ),
        tools=[my_tool_jump],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    result = await agent.ainvoke({"messages": []})

    assert result == {"messages": []}
    assert calls == ["AsyncMiddlewareOne.abefore_model", "AsyncMiddlewareTwo.abefore_model"]


async def test_create_agent_mixed_sync_async_middleware_async_invoke() -> None:
    """Test async invoke with mixed sync and async middleware."""
    calls = []

    class MostlySyncMiddleware(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("MostlySyncMiddleware.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            calls.append("MostlySyncMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("MostlySyncMiddleware.awrap_model_call")
            return await handler(request)

        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("MostlySyncMiddleware.after_model")

    class AsyncMiddleware(AgentMiddleware):
        @override
        async def abefore_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("AsyncMiddleware.awrap_model_call")
            return await handler(request)

        @override
        async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            calls.append("AsyncMiddleware.aafter_model")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[MostlySyncMiddleware(), AsyncMiddleware()],
    )

    await agent.ainvoke({"messages": [HumanMessage("hello")]})

    # In async mode, both sync and async middleware should work
    # Note: Sync wrap_model_call is not called when running in async mode,
    # as the async version is preferred
    assert calls == [
        "MostlySyncMiddleware.before_model",
        "AsyncMiddleware.abefore_model",
        "MostlySyncMiddleware.awrap_model_call",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
        "MostlySyncMiddleware.after_model",
    ]


# =============================================================================
# Before/After Agent Hook Tests
# =============================================================================


class TestAgentMiddlewareHooks:
    """Test before_agent and after_agent middleware hooks."""

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_execution(self, *, is_async: bool, hook_type: str) -> None:
        """Test that agent hooks are called in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:
            if hook_type == "before":

                @before_agent
                async def log_hook(
                    state: AgentState[Any], *_args: Any, **_kwargs: Any
                ) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None

            else:

                @after_agent
                async def log_hook(
                    state: AgentState[Any], *_args: Any, **_kwargs: Any
                ) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None

        elif hook_type == "before":

            @before_agent
            def log_hook(
                state: AgentState[Any], *_args: Any, **_kwargs: Any
            ) -> dict[str, Any] | None:
                execution_log.append(f"{hook_type}_agent_called")
                execution_log.append(f"message_count: {len(state['messages'])}")
                return None

        else:

            @after_agent
            def log_hook(
                state: AgentState[Any], *_args: Any, **_kwargs: Any
            ) -> dict[str, Any] | None:
                execution_log.append(f"{hook_type}_agent_called")
                execution_log.append(f"message_count: {len(state['messages'])}")
                return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[log_hook])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Hi")]})
        else:
            agent.invoke({"messages": [HumanMessage("Hi")]})

        assert f"{hook_type}_agent_called" in execution_log
        assert any("message_count:" in log for log in execution_log)

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_with_class_inheritance(self, *, is_async: bool, hook_type: str) -> None:
        """Test agent hooks using class inheritance in both sync and async modes."""
        execution_log: list[str] = []

        class AsyncCustomMiddleware(AgentMiddleware):
            @override
            async def abefore_agent(
                self, state: AgentState[Any], runtime: Runtime
            ) -> dict[str, Any] | None:
                if hook_type == "before":
                    execution_log.append("hook_called")
                return None

            @override
            async def aafter_agent(
                self, state: AgentState[Any], runtime: Runtime
            ) -> dict[str, Any] | None:
                if hook_type == "after":
                    execution_log.append("hook_called")
                return None

        class CustomMiddleware(AgentMiddleware):
            @override
            def before_agent(
                self, state: AgentState[Any], runtime: Runtime
            ) -> dict[str, Any] | None:
                if hook_type == "before":
                    execution_log.append("hook_called")
                return None

            @override
            def after_agent(
                self, state: AgentState[Any], runtime: Runtime
            ) -> dict[str, Any] | None:
                if hook_type == "after":
                    execution_log.append("hook_called")
                return None

        middleware = AsyncCustomMiddleware() if is_async else CustomMiddleware()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[middleware])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]})

        assert "hook_called" in execution_log


class TestAgentHooksCombined:
    """Test before_agent and after_agent hooks working together."""

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_execution_order(self, *, is_async: bool) -> None:
        """Test that before_agent executes before after_agent in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:

            @before_agent
            async def log_before(*_args: Any, **_kwargs: Any) -> None:
                execution_log.append("before")

            @after_agent
            async def log_after(*_args: Any, **_kwargs: Any) -> None:
                execution_log.append("after")

        else:

            @before_agent
            def log_before(*_args: Any, **_kwargs: Any) -> None:
                execution_log.append("before")

            @after_agent
            def log_after(*_args: Any, **_kwargs: Any) -> None:
                execution_log.append("after")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[log_before, log_after])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["before", "after"]

    def test_state_passthrough(self) -> None:
        """Test that state modifications in before_agent are visible to after_agent."""

        @before_agent
        def modify_in_before(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"messages": [HumanMessage("Added by before_agent")]}

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[modify_in_before])
        result = agent.invoke({"messages": [HumanMessage("Original")]})

        message_contents = [msg.content for msg in result["messages"]]
        assert message_contents[1] == "Added by before_agent"

    def test_multiple_middleware_instances(self) -> None:
        """Test multiple before_agent and after_agent middleware instances."""
        execution_log = []

        @before_agent
        def before_one(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("before_1")

        @before_agent
        def before_two(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("before_2")

        @after_agent
        def after_one(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("after_1")

        @after_agent
        def after_two(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("after_2")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(
            model=model, tools=[], middleware=[before_one, before_two, after_one, after_two]
        )
        agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["before_1", "before_2", "after_2", "after_1"]

    def test_agent_hooks_run_once_with_multiple_model_calls(self) -> None:
        """Test that before_agent and after_agent run only once per thread.

        This test verifies that agent-level hooks (before_agent, after_agent) execute
        exactly once per agent invocation, regardless of how many tool calling loops occur.
        This is different from model-level hooks (before_model, after_model) which run
        on every model invocation within the tool calling loop.
        """
        execution_log = []

        @tool
        def sample_tool_agent(query: str) -> str:
            """A sample tool for testing."""
            return f"Result for: {query}"

        @before_agent
        def log_before_agent(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("before_agent")

        @before_model
        def log_before_model(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("before_model")

        @after_agent
        def log_after_agent(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("after_agent")

        @after_model
        def log_after_model(*_args: Any, **_kwargs: Any) -> None:
            execution_log.append("after_model")

        # Model will call a tool twice, then respond with final answer
        # This creates 3 model invocations total, but agent hooks should still run once
        model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "sample_tool_agent", "args": {"query": "first"}, "id": "1"}],
                [{"name": "sample_tool_agent", "args": {"query": "second"}, "id": "2"}],
                [],  # Third call returns no tool calls (final answer)
            ]
        )

        agent = create_agent(
            model=model,
            tools=[sample_tool_agent],
            middleware=[log_before_agent, log_before_model, log_after_model, log_after_agent],
        )

        agent.invoke(
            {"messages": [HumanMessage("Test")]}, config={"configurable": {"thread_id": "abc"}}
        )

        assert execution_log == [
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
        ]

        agent.invoke(
            {"messages": [HumanMessage("Test")]}, config={"configurable": {"thread_id": "abc"}}
        )

        assert execution_log == [
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
        ]
