"""Integration tests for `HumanInTheLoopMiddleware` with `create_agent`.

These tests exercise the reject decision against real models to confirm that the
default rejection guidance actually discourages the model from retrying a rejected
tool call. The exact message wording is asserted by the unit tests; here we verify
the end-to-end behavior that the guidance is meant to produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from langchain.agents.middleware.types import _InputAgentState


def _get_model(provider: str) -> Any:
    """Get chat model for the specified provider."""
    if provider == "anthropic":
        return pytest.importorskip("langchain_anthropic").ChatAnthropic(
            model="claude-sonnet-4-5-20250929"
        )
    if provider == "openai":
        # Matches the model reported in the originating issue (langchain-ai/langchain#33787).
        return pytest.importorskip("langchain_openai").ChatOpenAI(model="gpt-5-nano")
    msg = f"Unknown provider: {provider}"
    raise ValueError(msg)


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"It is sunny in {location}."


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_hitl_reject_does_not_retry(provider: str) -> None:
    """A rejected tool call should not be retried after the default guidance.

    Because `interrupt_on` still applies to `get_weather`, any retry would re-trigger
    the interrupt. So a completed run with no new `__interrupt__` is a reliable signal
    that the model honored the guidance and did not retry.
    """
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        tools=[get_weather],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"get_weather": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "reject-test"}}

    interrupted = agent.invoke(
        {"messages": [HumanMessage("What is the weather in Paris?")]}, config
    )
    assert "__interrupt__" in interrupted, "Expected the tool call to trigger an interrupt"

    final = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)

    # The model must not retry: a retry would re-interrupt instead of completing.
    assert "__interrupt__" not in final, "Model retried the rejected tool call"

    # Exactly one rejection ToolMessage for the original (rejected) call.
    reject_messages = [
        msg
        for msg in final["messages"]
        if isinstance(msg, ToolMessage) and msg.name == "get_weather" and msg.status == "error"
    ]
    assert len(reject_messages) == 1, "Expected exactly one rejection ToolMessage"

    # The tool should have been called exactly once (the rejected call), never re-invoked.
    weather_tool_calls = [
        tool_call
        for msg in final["messages"]
        if isinstance(msg, AIMessage)
        for tool_call in msg.tool_calls
        if tool_call["name"] == "get_weather"
    ]
    assert len(weather_tool_calls) == 1, "Model issued more than one get_weather call"
