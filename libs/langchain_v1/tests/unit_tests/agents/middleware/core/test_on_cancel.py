import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel

needs_cancel_handler = pytest.mark.skipif(
    not hasattr(StateGraph, "set_cancel_handler"),
    reason="installed langgraph has no `StateGraph.set_cancel_handler`",
)


@tool
async def slow(query: str) -> str:
    """Slow tool."""
    await asyncio.sleep(10)
    return query


def _model() -> FakeToolCallingModel:
    return FakeToolCallingModel(tool_calls=[[{"name": "slow", "args": {"query": "x"}, "id": "1"}]])


class AnswerOnCancel(AgentMiddleware):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    @property
    def name(self) -> str:
        return f"AnswerOnCancel{self.label}"

    def on_cancel(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        return {
            "messages": [
                ToolMessage(
                    f"cancelled ({self.label})",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status="error",
                )
                for tc in last.tool_calls
            ]
        }


class AsyncNoop(AgentMiddleware):
    calls = 0

    async def aon_cancel(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        AsyncNoop.calls += 1
        return None


async def _cancel_during_tool(agent: Any, config: RunnableConfig) -> None:
    run = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage("hi")]}, config))
    await asyncio.sleep(0.2)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run


@needs_cancel_handler
async def test_on_cancel_answers_dangling_tool_call() -> None:
    agent = create_agent(
        _model(), [slow], middleware=[AnswerOnCancel("a")], checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    await _cancel_during_tool(agent, config)

    messages = (await agent.aget_state(config)).values["messages"]
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage", "ToolMessage"]
    assert messages[-1].tool_call_id == "1"
    assert messages[-1].status == "error"


@needs_cancel_handler
async def test_next_run_continues_after_cancel() -> None:
    model = FakeToolCallingModel(
        tool_calls=[[{"name": "slow", "args": {"query": "x"}, "id": "1"}], []]
    )
    agent = create_agent(
        model, [slow], middleware=[AnswerOnCancel("a")], checkpointer=InMemorySaver()
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    await _cancel_during_tool(agent, config)
    result = await agent.ainvoke({"messages": [HumanMessage("again")]}, config)

    assert [type(m).__name__ for m in result["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "HumanMessage",
        "AIMessage",
    ]


@needs_cancel_handler
async def test_on_cancel_hooks_run_in_order_and_combine() -> None:
    AsyncNoop.calls = 0
    agent = create_agent(
        _model(),
        [slow],
        middleware=[AnswerOnCancel("a"), AsyncNoop(), AnswerOnCancel("b")],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    await _cancel_during_tool(agent, config)

    messages = (await agent.aget_state(config)).values["messages"]
    assert [m.content for m in messages[2:]] == ["cancelled (a)", "cancelled (b)"]
    assert AsyncNoop.calls == 1


async def test_on_cancel_warns_and_skips_on_unsupported_langgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(StateGraph, "set_cancel_handler", raising=False)

    with pytest.warns(UserWarning, match="not supported by the installed langgraph"):
        agent = create_agent(
            _model(), [slow], middleware=[AnswerOnCancel("a")], checkpointer=InMemorySaver()
        )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    await _cancel_during_tool(agent, config)

    messages = (await agent.aget_state(config)).values["messages"]
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage"]


async def test_no_cancel_handler_without_on_cancel() -> None:
    agent = create_agent(_model(), [slow], checkpointer=InMemorySaver())
    assert getattr(agent, "cancel_handler", None) is None
