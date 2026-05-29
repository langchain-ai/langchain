"""Tests for langchain.agents._subagent_transformer.

The transformer surfaces, on `run.subagents`, any nested run that carries an
`lc_agent_name` (set by `create_agent(name=...)`), tracked via langgraph's base
`_TasksLifecycleBase`. These tests drive a real supervisor `create_agent` that
dispatches a nested `create_agent` from a tool, giving true end-to-end coverage.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolCallTransformer

from langchain.agents import create_agent
from langchain.agents._subagent_transformer import SubagentTransformer
from tests.unit_tests.agents.model import FakeToolCallingModel


def _supervisor_model() -> FakeToolCallingModel:
    """Supervisor emits one tool call (id `call_w`) then a final message."""
    return FakeToolCallingModel(
        tool_calls=[
            [{"args": {"city": "SF"}, "id": "call_w", "name": "call_weather"}],
            [],
        ]
    )


def test_subagents_surfaces_named_subagent() -> None:
    """A nested named `create_agent` dispatched from a tool surfaces a handle."""
    weather_agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]), name="weather_agent")

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        result = weather_agent.invoke({"messages": [HumanMessage(f"weather in {city}")]})
        return result["messages"][-1].text

    supervisor = create_agent(
        model=_supervisor_model(),
        tools=[call_weather],
        name="supervisor",
    )

    run = supervisor.stream_events({"messages": [HumanMessage("weather?")]}, version="v3")

    handles = []
    for handle in run.subagents:
        handles.append(handle)
        # Drain the nested run so it completes.
        for _ in handle:
            pass

    assert len(handles) == 1
    assert handles[0].name == "weather_agent"
    assert handles[0].cause == {"type": "toolCall", "tool_call_id": "call_w"}


async def test_subagents_surfaces_named_subagent_async() -> None:
    """Async counterpart: the handle surfaces and reaches a terminal state.

    Drives the run through `astream_events`, so events flow via the async
    `aprocess`/`apush` lane and the child mini-mux is closed via `aclose`.
    """
    weather_agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]), name="weather_agent")

    @tool("call_weather")
    async def call_weather(city: str) -> str:
        """Call the weather agent."""
        result = await weather_agent.ainvoke({"messages": [HumanMessage(f"weather in {city}")]})
        return result["messages"][-1].text

    supervisor = create_agent(
        model=_supervisor_model(),
        tools=[call_weather],
        name="supervisor",
    )

    run = await supervisor.astream_events({"messages": [HumanMessage("weather?")]}, version="v3")

    handles = []
    async for handle in run.subagents:
        handles.append(handle)
        # Drain the nested run so it completes.
        async for _ in handle:
            pass

    assert len(handles) == 1
    assert handles[0].name == "weather_agent"
    assert handles[0].cause == {"type": "toolCall", "tool_call_id": "call_w"}
    assert handles[0]._seen_terminal is True
    assert handles[0].status == "completed"


def test_plain_tool_not_surfaced() -> None:
    """A tool that returns a string (spawns no nested run) surfaces nothing."""

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Return weather directly without invoking a subagent."""
        return f"Sunny in {city}"

    supervisor = create_agent(
        model=_supervisor_model(),
        tools=[call_weather],
        name="supervisor",
    )

    run = supervisor.stream_events({"messages": [HumanMessage("weather?")]}, version="v3")

    handles = list(run.subagents)
    # Drain the main run to completion.
    for _ in run:
        pass

    assert handles == []


def test_unnamed_inner_agent_surfaces_with_inherited_name() -> None:
    """An unnamed inner `create_agent` inherits the parent's name and surfaces.

    This is the accepted trade-off of detecting subagents by "carries an
    lc_agent_name": a nested run that didn't set its own name (an unnamed
    `create_agent`, or a plain `StateGraph`) inherits the parent's name and is
    surfaced under it. A caller can null `lc_agent_name` in the config it
    invokes such a graph with to exclude it.
    """
    inner_agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]))

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call an unnamed inner agent."""
        result = inner_agent.invoke({"messages": [HumanMessage(f"weather in {city}")]})
        return result["messages"][-1].text

    supervisor = create_agent(
        model=_supervisor_model(),
        tools=[call_weather],
        name="supervisor",
    )

    run = supervisor.stream_events({"messages": [HumanMessage("weather?")]}, version="v3")

    handles = []
    for handle in run.subagents:
        handles.append(handle)
        for _ in handle:
            pass

    assert len(handles) == 1
    # Inherited the parent's name (the trade-off).
    assert handles[0].name == "supervisor"


def test_same_name_nested_agent_surfaced() -> None:
    """A nested agent with the SAME name as its parent is still surfaced.

    A subagent that dispatches a same-named agent (or invokes itself) re-asserts
    its own `lc_agent_name`, so child lc == parent lc. The discriminator surfaces
    any nested run that carries a name, so the same-named run is reported instead
    of being folded into the parent (which the old `!= parent` rule did).
    """
    inner_agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]), name="weather_agent")

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call a same-named inner agent."""
        result = inner_agent.invoke({"messages": [HumanMessage(f"weather in {city}")]})
        return result["messages"][-1].text

    # The parent agent shares the inner agent's name.
    supervisor = create_agent(
        model=_supervisor_model(),
        tools=[call_weather],
        name="weather_agent",
    )

    run = supervisor.stream_events({"messages": [HumanMessage("weather?")]}, version="v3")

    handles = []
    for handle in run.subagents:
        handles.append(handle)
        for _ in handle:
            pass

    assert len(handles) == 1
    assert handles[0].name == "weather_agent"
    assert handles[0].cause == {"type": "toolCall", "tool_call_id": "call_w"}


def test_transformer_init_exposes_subagents_channel() -> None:
    """The transformer publishes a `subagents` channel from `init()`."""
    transformer = SubagentTransformer(scope=())
    state = transformer.init()
    assert "subagents" in state


def test_create_agent_registers_subagent_transformer() -> None:
    """`create_agent` registers `SubagentTransformer` in `stream_transformers`."""

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        return f"Sunny in {city}"

    agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]), tools=[call_weather])

    assert SubagentTransformer in agent.stream_transformers


def test_create_agent_subagent_transformer_precedes_user_transformers() -> None:
    """`ToolCallTransformer` precedes `SubagentTransformer` in registration order."""

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        return f"Sunny in {city}"

    agent = create_agent(model=FakeToolCallingModel(tool_calls=[[]]), tools=[call_weather])

    transformers = list(agent.stream_transformers)
    assert transformers.index(ToolCallTransformer) < transformers.index(SubagentTransformer)
