"""Unit tests for on_agent_finish callback emission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from uuid import UUID
from langchain_core.agents import AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langgraph.typing import ContextT

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from tests.unit_tests.agents.model import FakeToolCallingModel


class TrackingHandler(BaseCallbackHandler):
    """Sync callback handler that records on_agent_finish calls."""

    def __init__(self) -> None:
        super().__init__()
        self.finishes: list[AgentFinish] = []
        self.run_ids: list[UUID] = []

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.finishes.append(finish)
        self.run_ids.append(run_id)


class AsyncTrackingHandler(AsyncCallbackHandler):
    """Async callback handler that records on_agent_finish calls."""

    def __init__(self) -> None:
        super().__init__()
        self.finishes: list[AgentFinish] = []
        self.run_ids: list[UUID] = []

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.finishes.append(finish)
        self.run_ids.append(run_id)


class IgnoreAgentHandler(BaseCallbackHandler):
    """Handler with ignore_agent=True. Should NOT receive on_agent_finish."""

    ignore_agent = True

    def __init__(self) -> None:
        super().__init__()
        self.finishes: list[AgentFinish] = []

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.finishes.append(finish)


def test_on_agent_finish_called_on_normal_completion() -> None:
    """Test on_agent_finish is called when agent completes without tools."""
    handler = TrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    assert len(handler.finishes) == 1
    assert isinstance(handler.finishes[0], AgentFinish)
    assert "output" in handler.finishes[0].return_values


def test_on_agent_finish_called_with_tools() -> None:
    """Test on_agent_finish is called after agent uses tools and responds."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    handler = TrackingHandler()
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="tc1")],
            [],  # No tools on second call = agent finishes
        ]
    )
    agent = create_agent(model=model, tools=[search])

    agent.invoke(
        {"messages": [HumanMessage("search for test")]},
        config={"callbacks": [handler]},
    )

    assert len(handler.finishes) == 1
    assert isinstance(handler.finishes[0], AgentFinish)


def test_on_agent_finish_called_on_jump_to_end() -> None:
    """Test on_agent_finish is called when middleware uses jump_to='end'.

    This is the core bug from issue #34165: when middleware returns
    jump_to='end', on_agent_finish should still fire.
    """

    class JumpToEndMiddleware(AgentMiddleware[AgentState[Any], ContextT]):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content="Blocked by middleware")],
            }

    handler = TrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    middleware = JumpToEndMiddleware()
    agent = create_agent(model=model, tools=[], middleware=[middleware])

    result = agent.invoke(
        {"messages": [HumanMessage("trigger")]},
        config={"callbacks": [handler]},
    )

    # Verify the agent was stopped by middleware
    messages = result["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    assert any("Blocked" in str(m.content) for m in ai_messages)

    # Verify on_agent_finish was called
    assert len(handler.finishes) == 1
    assert isinstance(handler.finishes[0], AgentFinish)
    assert handler.finishes[0].return_values["output"] == "Blocked by middleware"


@pytest.mark.asyncio
async def test_on_agent_finish_async() -> None:
    """Test async handler's on_agent_finish is called."""
    handler = AsyncTrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    await agent.ainvoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    assert len(handler.finishes) == 1
    assert isinstance(handler.finishes[0], AgentFinish)


@pytest.mark.asyncio
async def test_on_agent_finish_async_with_jump_to_end() -> None:
    """Test async handler's on_agent_finish is called on jump_to='end'."""

    class JumpToEndMiddleware(AgentMiddleware[AgentState[Any], ContextT]):
        @hook_config(can_jump_to=["end"])
        async def abefore_model(
            self, state: AgentState[Any], runtime: Any
        ) -> dict[str, Any] | None:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content="Blocked by middleware")],
            }

    handler = AsyncTrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    middleware = JumpToEndMiddleware()
    agent = create_agent(model=model, tools=[], middleware=[middleware])

    await agent.ainvoke(
        {"messages": [HumanMessage("trigger")]},
        config={"callbacks": [handler]},
    )

    assert len(handler.finishes) == 1
    assert handler.finishes[0].return_values["output"] == "Blocked by middleware"


def test_on_agent_finish_called_with_after_agent_middleware() -> None:
    """Test on_agent_finish fires after after_agent middleware completes."""
    call_order: list[str] = []

    class OrderTrackingMiddleware(AgentMiddleware[AgentState[Any], ContextT]):
        def after_agent(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
            call_order.append("after_agent")
            return None

    class OrderTrackingHandler(BaseCallbackHandler):
        def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            call_order.append("on_agent_finish")

    handler = OrderTrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    middleware = OrderTrackingMiddleware()
    agent = create_agent(model=model, tools=[], middleware=[middleware])

    agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    assert call_order == ["after_agent", "on_agent_finish"]


def test_on_agent_finish_content_matches_last_ai_message() -> None:
    """Test AgentFinish.return_values['output'] matches last AIMessage."""
    handler = TrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    result = agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [handler]},
    )

    # Find the last AIMessage
    messages = result["messages"]
    last_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    assert last_ai is not None
    assert handler.finishes[0].return_values["output"] == last_ai.content


def test_ignore_agent_handler_not_called() -> None:
    """Test handler with ignore_agent=True does not receive on_agent_finish."""
    ignore_handler = IgnoreAgentHandler()
    tracking_handler = TrackingHandler()
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    agent.invoke(
        {"messages": [HumanMessage("hello")]},
        config={"callbacks": [ignore_handler, tracking_handler]},
    )

    assert len(ignore_handler.finishes) == 0, "ignore_agent handler should not be called"
    assert len(tracking_handler.finishes) == 1, "Normal handler should be called"


def test_no_callbacks_no_error() -> None:
    """Test agent works fine when no callbacks are registered."""
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    # Should not raise
    result = agent.invoke({"messages": [HumanMessage("hello")]})
    assert result is not None
