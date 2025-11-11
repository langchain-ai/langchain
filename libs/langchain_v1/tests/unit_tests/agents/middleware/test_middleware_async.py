from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    hook_config,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

from .model import FakeToolCallingModel


async def test_create_agent_async_invoke() -> None:
    """Test async invoke with async middleware hooks."""
    calls = []

    class AsyncMiddleware(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddleware.awrap_model_call")
            request.messages.append(HumanMessage("async middleware message"))
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.aafter_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"input": "yo"}, "id": "1", "name": "my_tool"}],
                [],
            ]
        ),
        tools=[my_tool],
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
        "my_tool",
        "AsyncMiddleware.abefore_model",
        "AsyncMiddleware.awrap_model_call",
        "AsyncMiddleware.aafter_model",
    ]


async def test_create_agent_async_invoke_multiple_middleware() -> None:
    """Test async invoke with multiple async middleware hooks."""
    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddlewareOne.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.aafter_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareTwo.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddlewareTwo.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareTwo.aafter_model")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("hello")]})

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
    from typing import Any

    calls = []

    class AsyncMiddlewareOne(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddlewareOne.abefore_model")

    class AsyncMiddlewareTwo(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(self, state, runtime) -> dict[str, Any]:
            calls.append("AsyncMiddlewareTwo.abefore_model")
            return {"jump_to": "end"}

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[AsyncMiddlewareOne(), AsyncMiddlewareTwo()],
    )

    result = await agent.ainvoke({"messages": []})

    assert result == {"messages": []}
    assert calls == ["AsyncMiddlewareOne.abefore_model", "AsyncMiddlewareTwo.abefore_model"]


async def test_create_agent_mixed_sync_async_middleware() -> None:
    """Test async invoke with mixed sync and async middleware."""
    calls = []

    class MostlySyncMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> None:
            calls.append("MostlySyncMiddleware.before_model")

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], AIMessage],
        ) -> AIMessage:
            calls.append("MostlySyncMiddleware.wrap_model_call")
            return handler(request)

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            calls.append("MostlySyncMiddleware.awrap_model_call")
            return await handler(request)

        def after_model(self, state, runtime) -> None:
            calls.append("MostlySyncMiddleware.after_model")

    class AsyncMiddleware(AgentMiddleware):
        async def abefore_model(self, state, runtime) -> None:
            calls.append("AsyncMiddleware.abefore_model")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[AIMessage]],
        ) -> AIMessage:
            calls.append("AsyncMiddleware.awrap_model_call")
            return await handler(request)

        async def aafter_model(self, state, runtime) -> None:
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
