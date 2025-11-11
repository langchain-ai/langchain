import sys
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from syrupy.assertion import SnapshotAssertion
from typing_extensions import Annotated

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    hook_config,
    ModelRequest,
    OmitFromInput,
    OmitFromOutput,
    PrivateStateAttr,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import InjectedState

from .messages import _AnyIdHumanMessage, _AnyIdToolMessage
from .model import FakeToolCallingModel


def test_create_agent_invoke(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopEight.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"input": "yo"}, "id": "1", "name": "my_tool"},
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
                        "args": {"input": "yo"},
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
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state, runtime):
            calls.append("NoopSeven.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopSeven.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state, runtime) -> dict[str, Any]:
            calls.append("NoopEight.before_model")
            return {"jump_to": "end"}

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("NoopEight.wrap_model_call")
            return handler(request)

        def after_model(self, state, runtime):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
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
        def before_agent(self, state, runtime) -> None:
            return None

    class NoopOne(AgentMiddleware):
        def after_agent(self, state, runtime) -> None:
            return None

    class NoopTwo(AgentMiddleware):
        def after_agent(self, state, runtime) -> None:
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
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
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


def test_tools_to_model_edge_with_structured_and_regular_tool_calls():
    """Test that when there are both structured and regular tool calls, we execute regular and jump to END."""

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
        def __init__(self):
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

    class CustomState(AgentState):
        omit_input: Annotated[str, OmitFromInput]
        omit_output: Annotated[str, OmitFromOutput]
        private_state: Annotated[str, PrivateStateAttr]

    class CustomMiddleware(AgentMiddleware[CustomState]):
        state_schema: type[CustomState] = CustomState

        def before_model(self, state: CustomState) -> dict[str, Any]:
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
    from langgraph.runtime import Runtime

    class CustomMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Runtime) -> None:
            assert runtime is not None
            return None

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            assert request.runtime is not None
            return handler(request)

        def after_model(self, state: AgentState, runtime: Runtime) -> None:
            assert runtime is not None
            return None

    middleware = CustomMiddleware()

    agent = create_agent(model=FakeToolCallingModel(), middleware=[CustomMiddleware()])
    agent.invoke({"messages": [HumanMessage("Hello")]})


# test setup defined at this scope bc of pydantic issues inferring the namespace of
# custom state w/in a function


class CustomState(AgentState):
    custom_state: str


@tool(description="Test the state")
def test_state_tool(
    state: Annotated[CustomState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Test tool that accesses injected state."""
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
        {"custom_state": "I love pizza", "messages": [HumanMessage("Call the test state tool")]}
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
        def before_model(self, state: AgentState) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

        def after_model(self, state: AgentState) -> dict[str, Any]:
            assert "jump_to" not in state
            return {"jump_to": "model"}

    agent = create_agent(model=FakeToolCallingModel(), middleware=[MyMiddleware()])
    result = agent.invoke({"messages": [HumanMessage("Hello")]})
    assert "jump_to" not in result


def test_create_agent_sync_invoke_with_only_async_middleware_raises_error() -> None:
    """Test that sync invoke with only async middleware works via run_in_executor."""
    from collections.abc import Awaitable

    class AsyncOnlyMiddleware(AgentMiddleware):
        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
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
    from collections.abc import Awaitable

    calls = []

    class MixedMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.before_model")

        async def abefore_model(self, state, runtime) -> None:
            calls.append("MixedMiddleware.abefore_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("MixedMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("MixedMiddleware.awrap_model_call")
            return await handler(request)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[MixedMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("hello")]})

    # In sync mode, only sync methods should be called
    assert calls == [
        "MixedMiddleware.before_model",
        "MixedMiddleware.wrap_model_call",
    ]
