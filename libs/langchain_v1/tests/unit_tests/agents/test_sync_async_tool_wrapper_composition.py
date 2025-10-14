"""Tests for sync/async middleware composition with wrap_tool_call and awrap_tool_call.

These tests verify the desired behavior:
1. If middleware defines both sync and async -> use both on respective paths
2. If middleware defines only sync -> use on sync path, raise NotImplementedError on async path
3. If middleware defines only async -> use on async path, raise NotImplementedError on sync path
"""

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return f"Calculated: {expression}"


class TestSyncAsyncMiddlewareComposition:
    """Test sync/async middleware composition behavior."""

    def test_sync_only_middleware_works_on_sync_path(self) -> None:
        """Middleware with only sync wrap_tool_call works on sync path."""
        call_log = []

        class SyncOnlyMiddleware(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                call_log.append("sync_called")
                return handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[SyncOnlyMiddleware()],
            checkpointer=InMemorySaver(),
        )

        result = agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert "sync_called" in call_log
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "Results for: test" in tool_messages[0].content

    async def test_sync_only_middleware_raises_on_async_path(self) -> None:
        """Middleware with only sync wrap_tool_call raises NotImplementedError on async path."""

        class SyncOnlyMiddleware(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                return handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[SyncOnlyMiddleware()],
            checkpointer=InMemorySaver(),
        )

        # Should raise NotImplementedError because SyncOnlyMiddleware doesn't support async path
        with pytest.raises(NotImplementedError):
            await agent.ainvoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test"}},
            )

    async def test_async_only_middleware_works_on_async_path(self) -> None:
        """Middleware with only async awrap_tool_call works on async path."""
        call_log = []

        class AsyncOnlyMiddleware(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                call_log.append("async_called")
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[AsyncOnlyMiddleware()],
            checkpointer=InMemorySaver(),
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )

        assert "async_called" in call_log
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert "Results for: test" in tool_messages[0].content

    def test_async_only_middleware_raises_on_sync_path(self) -> None:
        """Middleware with only async awrap_tool_call raises NotImplementedError on sync path."""

        class AsyncOnlyMiddleware(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[AsyncOnlyMiddleware()],
            checkpointer=InMemorySaver(),
        )

        with pytest.raises(NotImplementedError):
            agent.invoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test"}},
            )

    def test_both_sync_and_async_middleware_uses_appropriate_path(self) -> None:
        """Middleware with both sync and async uses correct implementation per path."""
        call_log = []

        class BothSyncAsyncMiddleware(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                call_log.append("sync_called")
                return handler(request)

            async def awrap_tool_call(self, request, handler):
                call_log.append("async_called")
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[BothSyncAsyncMiddleware()],
            checkpointer=InMemorySaver(),
        )

        # Sync path
        call_log.clear()
        result_sync = agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test1"}},
        )
        assert "sync_called" in call_log
        assert "async_called" not in call_log

    async def test_both_sync_and_async_middleware_uses_appropriate_path_async(
        self,
    ) -> None:
        """Middleware with both sync and async uses correct implementation per path (async)."""
        call_log = []

        class BothSyncAsyncMiddleware(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                call_log.append("sync_called")
                return handler(request)

            async def awrap_tool_call(self, request, handler):
                call_log.append("async_called")
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[BothSyncAsyncMiddleware()],
            checkpointer=InMemorySaver(),
        )

        # Async path
        call_log.clear()
        result_async = await agent.ainvoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test2"}},
        )
        assert "async_called" in call_log
        assert "sync_called" not in call_log

    async def test_mixed_middleware_composition_async_path_fails_with_sync_only(
        self,
    ) -> None:
        """Multiple middleware on async path fails if any are sync-only."""

        class SyncOnlyMiddleware(AgentMiddleware):
            name = "SyncOnly"

            def wrap_tool_call(self, request, handler):
                return handler(request)

        class AsyncOnlyMiddleware(AgentMiddleware):
            name = "AsyncOnly"

            async def awrap_tool_call(self, request, handler):
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[
                SyncOnlyMiddleware(),
                AsyncOnlyMiddleware(),
            ],
            checkpointer=InMemorySaver(),
        )

        # Should raise NotImplementedError because SyncOnlyMiddleware can't run on async path
        with pytest.raises(NotImplementedError):
            await agent.ainvoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test"}},
            )

    def test_mixed_middleware_composition_sync_path_with_async_only_fails(self) -> None:
        """Multiple middleware on sync path fails if any are async-only."""

        class SyncOnlyMiddleware(AgentMiddleware):
            name = "SyncOnly"

            def wrap_tool_call(self, request, handler):
                return handler(request)

        class AsyncOnlyMiddleware(AgentMiddleware):
            name = "AsyncOnly"

            async def awrap_tool_call(self, request, handler):
                return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[
                SyncOnlyMiddleware(),
                AsyncOnlyMiddleware(),  # This will break sync path
            ],
            checkpointer=InMemorySaver(),
        )

        # Should raise NotImplementedError because AsyncOnlyMiddleware can't run on sync path
        with pytest.raises(NotImplementedError):
            agent.invoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test"}},
            )

    def test_decorator_sync_only_works_both_paths(self) -> None:
        """Decorator-created sync-only middleware works on both paths."""
        call_log = []

        @wrap_tool_call
        def my_wrapper(request: ToolCallRequest, handler):
            call_log.append("decorator_sync")
            return handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[my_wrapper],
            checkpointer=InMemorySaver(),
        )

        # Sync path
        call_log.clear()
        result = agent.invoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test1"}},
        )
        assert "decorator_sync" in call_log
        assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 1

    async def test_decorator_sync_only_raises_on_async_path(self) -> None:
        """Decorator-created sync-only middleware raises on async path."""
        call_log = []

        @wrap_tool_call
        def my_wrapper(request: ToolCallRequest, handler):
            call_log.append("decorator_sync")
            return handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[my_wrapper],
            checkpointer=InMemorySaver(),
        )

        # Should raise NotImplementedError because sync-only decorator doesn't support async path
        with pytest.raises(NotImplementedError):
            await agent.ainvoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test2"}},
            )

    async def test_decorator_async_only_works_async_path(self) -> None:
        """Decorator-created async-only middleware works on async path."""
        call_log = []

        @wrap_tool_call
        async def my_async_wrapper(request: ToolCallRequest, handler):
            call_log.append("decorator_async")
            return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[my_async_wrapper],
            checkpointer=InMemorySaver(),
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage("Search")]},
            {"configurable": {"thread_id": "test"}},
        )
        assert "decorator_async" in call_log
        assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 1

    def test_decorator_async_only_raises_on_sync_path(self) -> None:
        """Decorator-created async-only middleware raises on sync path."""

        @wrap_tool_call
        async def my_async_wrapper(request: ToolCallRequest, handler):
            return await handler(request)

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="search", args={"query": "test"}, id="1")],
                [],
            ]
        )

        agent = create_agent(
            model=model,
            tools=[search],
            middleware=[my_async_wrapper],
            checkpointer=InMemorySaver(),
        )

        with pytest.raises(NotImplementedError):
            agent.invoke(
                {"messages": [HumanMessage("Search")]},
                {"configurable": {"thread_id": "test"}},
            )
