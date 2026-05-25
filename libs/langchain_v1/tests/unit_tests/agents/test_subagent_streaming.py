"""Regression tests for subagent stream event propagation.

Reproduces a bug where `create_agent` set ``ls_agent_type`` inside the
parent agent's ``configurable`` and, as a side effect, ``updates``,
``values``, and ``custom`` stream events from sub-agents invoked through
tools were dropped during ``stream(..., subgraphs=True)``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolCall
from langchain_core.tools import tool

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


def _make_subagent_caller_tool():
    """Build a subagent and a tool that invokes it."""
    subagent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        name="subagent",
    )

    @tool
    def call_subagent(query: str) -> str:
        """Delegate the query to a sub-agent."""
        result = subagent.invoke({"messages": [HumanMessage(query)]})
        return result["messages"][-1].text

    return call_subagent


def _make_parent_agent(call_subagent_tool) -> object:
    parent_tool_calls: list[list[ToolCall]] = [
        [{"args": {"query": "hi"}, "id": "call_1", "name": "call_subagent"}],
        [],
    ]
    return create_agent(
        model=FakeToolCallingModel(tool_calls=parent_tool_calls),
        tools=[call_subagent_tool],
        name="parent",
    )


def test_subagent_updates_emitted_when_streaming_with_subgraphs() -> None:
    """`updates` events from a tool-invoked sub-agent must be streamed.

    Without the fix, the parent agent's ``configurable`` overrode the
    streaming machinery's per-run state, suppressing ``updates`` events
    from any sub-graph invoked inside a tool.
    """
    call_subagent_tool = _make_subagent_caller_tool()
    parent = _make_parent_agent(call_subagent_tool)

    subagent_update_events = []
    for namespace, mode, _data in parent.stream(
        {"messages": [HumanMessage("hi")]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        if mode == "updates" and namespace:
            subagent_update_events.append(namespace)

    assert subagent_update_events, (
        "expected `updates` events from the sub-agent's subgraph namespace, but none were emitted"
    )


async def test_subagent_updates_emitted_when_astreaming_with_subgraphs() -> None:
    """Async counterpart of the sync regression test."""
    call_subagent_tool = _make_subagent_caller_tool()
    parent = _make_parent_agent(call_subagent_tool)

    subagent_update_events = []
    async for namespace, mode, _data in parent.astream(
        {"messages": [HumanMessage("hi")]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        if mode == "updates" and namespace:
            subagent_update_events.append(namespace)

    assert subagent_update_events, (
        "expected `updates` events from the sub-agent's subgraph namespace, but none were emitted"
    )
