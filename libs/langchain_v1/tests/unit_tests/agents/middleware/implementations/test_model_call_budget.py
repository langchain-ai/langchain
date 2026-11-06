import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import InputAgentState, ModelCallBudgetMiddleware
from langchain.agents.structured_output import ToolStrategy
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


CALL = {"id": "call", "name": "echo", "args": {"value": "evidence"}}


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("structured", [False, True])
def test_exhaustion(*, structured: bool) -> None:
    model = FakeToolCallingModel(tool_calls=[[CALL, {**CALL, "id": "other"}], []])
    agent = create_agent(
        model,
        tools=[echo],
        system_prompt=SystemMessage(content=[{"type": "text", "text": "Original instructions"}]),
        middleware=[ModelCallBudgetMiddleware(run_limit=1)],
        response_format=ToolStrategy(Answer) if structured else None,
    )
    result = agent.invoke({"messages": [HumanMessage("Investigate")]})
    assert model.index == 2
    assert result["model_budget_used"] == 2
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls
    assert "0/1 working turns used; 1 remaining" in result["messages"][1].text
    assert "1/1 working turns used; 0 remaining" in final.text
    assert "Original instructions" in final.text
    assert len([message for message in result["messages"] if message.type == "tool"]) == 2
    assert all(message.type != "system" for message in result["messages"])


@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("async_mode", [False, True])
async def test_early_final_and_reset(limit: int, *, async_mode: bool) -> None:
    model = FakeToolCallingModel()
    agent = create_agent(
        model,
        middleware=[ModelCallBudgetMiddleware(run_limit=limit)],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "budget"}}
    inputs: InputAgentState = {"messages": [HumanMessage("Answer")]}
    for _ in range(2):
        result = (
            await agent.ainvoke(inputs, config)
            if async_mode
            else await asyncio.to_thread(lambda: agent.invoke(inputs, config))
        )
        assert result["model_budget_used"] == 1
        assert f"0/{limit} working turns used" in result["messages"][-1].text
    assert model.index == 2


async def test_resume_keeps_budget() -> None:
    model = FakeToolCallingModel(tool_calls=[[CALL], []])
    agent = create_agent(
        model,
        tools=[echo],
        middleware=[ModelCallBudgetMiddleware(run_limit=1)],
        checkpointer=InMemorySaver(),
        interrupt_before=["tools"],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "resume"}}
    paused = await agent.ainvoke({"messages": [HumanMessage("Investigate")]}, config)
    assert paused["model_budget_used"] == 1
    result = await agent.ainvoke(None, config)
    assert result["model_budget_used"] == 2
    assert "1/1 working turns used; 0 remaining" in result["messages"][-1].text
    assert model.index == 2


@pytest.mark.parametrize("async_mode", [False, True])
async def test_final_tool_calls_rejected(*, async_mode: bool) -> None:
    agent = create_agent(
        FakeToolCallingModel(tool_calls=[[CALL]]),
        tools=[echo],
        middleware=[ModelCallBudgetMiddleware(run_limit=0)],
    )
    if async_mode:
        with pytest.raises(ValueError, match="must not contain tool calls"):
            await agent.ainvoke({"messages": [HumanMessage("Answer")]})
    else:
        with pytest.raises(ValueError, match="must not contain tool calls"):
            await asyncio.to_thread(lambda: agent.invoke({"messages": [HumanMessage("Answer")]}))


@pytest.mark.parametrize("limit", [-1, True, 1.5, "2", None])
def test_invalid_budget(limit: Any) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        ModelCallBudgetMiddleware(run_limit=limit)
