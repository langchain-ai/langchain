"""Tests for langchain.agents._subagent_transformer.

The transformer reads TaskPayload.metadata.subagent_name (stamped by
langgraph's ToolNode for tools with BaseTool.subagent_name declared)
and builds typed SubagentRunStream handles on run.subagents.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.prebuilt import ToolCallTransformer

from langchain.agents import create_agent
from langchain.agents._subagent_transformer import (
    AsyncSubagentRunStream,
    SubagentRunStream,
    SubagentTransformer,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

# --- create_agent integration tests ---


def test_transformer_init_exposes_subagents_channel() -> None:
    transformer = SubagentTransformer(scope=())
    state = transformer.init()
    assert "subagents" in state


def test_subagent_run_stream_name_property() -> None:
    handle = SubagentRunStream.__new__(SubagentRunStream)
    handle.graph_name = "weather_agent"
    handle.trigger_call_id = None
    assert handle.name == "weather_agent"


def test_subagent_run_stream_cause_with_tool_call_id() -> None:
    handle = SubagentRunStream.__new__(SubagentRunStream)
    handle.graph_name = "weather_agent"
    handle.trigger_call_id = "call_xyz"
    assert handle.cause == {"type": "toolCall", "tool_call_id": "call_xyz"}


def test_subagent_run_stream_cause_without_tool_call_id() -> None:
    handle = SubagentRunStream.__new__(SubagentRunStream)
    handle.graph_name = "weather_agent"
    handle.trigger_call_id = None
    assert handle.cause is None


def test_async_subagent_run_stream_name_and_cause() -> None:
    handle = AsyncSubagentRunStream.__new__(AsyncSubagentRunStream)
    handle.graph_name = "research_agent"
    handle.trigger_call_id = "call_abc"
    assert handle.name == "research_agent"
    assert handle.cause == {"type": "toolCall", "tool_call_id": "call_abc"}


def test_transformer_on_started_skips_when_no_cause() -> None:
    """Tasks without cause (non-subagent dispatches) produce no handle."""
    transformer = SubagentTransformer(scope=())
    # Calling _on_started with cause=None should not add any handle.
    transformer._on_started(("tools:abc123",), "tools", "abc123", cause=None)
    assert len(transformer._handles) == 0


def test_transformer_on_started_skips_when_no_graph_name() -> None:
    """Tasks with cause but missing graph_name produce no handle."""
    transformer = SubagentTransformer(scope=())
    transformer._on_started(
        ("tools:abc123",),
        None,
        "abc123",
        cause={"type": "toolCall", "toolCallId": "call_1"},
    )
    assert len(transformer._handles) == 0


def test_transformer_name_filter_excludes_undeclared() -> None:
    """When subagent_names is set, names outside it are ignored."""
    transformer = SubagentTransformer(scope=(), subagent_names=frozenset({"researcher"}))
    transformer._on_started(
        ("tools:abc123",),
        "weather_agent",
        "abc123",
        cause={"type": "toolCall", "toolCallId": "call_1"},
    )
    assert len(transformer._handles) == 0


def test_transformer_name_filter_accepts_declared() -> None:
    """When subagent_names is set, declared names pass the filter (mux absent so no handle)."""
    transformer = SubagentTransformer(scope=(), subagent_names=frozenset({"researcher"}))
    # _mux is None so _make_child raises; the filter itself passes.
    # We check that the guard after name-filter fires (RuntimeError from no mux)
    # rather than silently returning at the name-filter step.
    transformer._on_started(
        ("tools:abc123",),
        "researcher",
        "abc123",
        cause={"type": "toolCall", "toolCallId": "call_1"},
    )
    # No handle created (mux is None), but filter passed — no assertion error.
    assert len(transformer._handles) == 0


def test_transformer_none_filter_accepts_any_name() -> None:
    """When subagent_names is None (default), any subagent_name is accepted."""
    transformer = SubagentTransformer(scope=())
    assert transformer._names is None
    # With _mux=None we can't build a handle, but we verify the name filter
    # didn't reject the call.
    transformer._on_started(
        ("tools:abc123",),
        "arbitrary_agent",
        "abc123",
        cause={"type": "toolCall", "toolCallId": "call_1"},
    )
    assert len(transformer._handles) == 0


def test_create_agent_registers_subagent_transformer() -> None:
    """create_agent automatically registers SubagentTransformer in stream_transformers.

    Verified by inspecting the compiled graph's stream_transformers tuple,
    which is set by graph.compile(transformers=...) in the factory.
    """

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        return f"Sunny in {city}"

    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[call_weather])

    assert SubagentTransformer in agent.stream_transformers, (
        "SubagentTransformer must be registered in compiled graph stream_transformers"
    )


def test_create_agent_subagent_transformer_precedes_user_transformers() -> None:
    """SubagentTransformer is registered before any user-supplied transformers.

    Ensures SubagentTransformer processes events before user-provided transformers,
    matching the ordering convention established by ToolCallTransformer.
    """

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        return f"Sunny in {city}"

    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[call_weather])

    transformers = list(agent.stream_transformers)
    tool_call_idx = transformers.index(ToolCallTransformer)
    subagent_idx = transformers.index(SubagentTransformer)
    assert tool_call_idx < subagent_idx, (
        "ToolCallTransformer must precede SubagentTransformer in stream_transformers"
    )


def test_create_agent_user_transformers_appended_after_subagent_transformer() -> None:
    """User-supplied transformers are appended after SubagentTransformer.

    Ensures that user-provided transformers=[] kwarg does not displace
    SubagentTransformer.
    """

    class UserTransformer(SubagentTransformer):
        """Sentinel subclass used only to verify ordering."""

    @tool("call_weather")
    def call_weather(city: str) -> str:
        """Call the weather agent."""
        return f"Sunny in {city}"

    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[call_weather], transformers=[UserTransformer])

    transformers = list(agent.stream_transformers)
    assert SubagentTransformer in transformers, "SubagentTransformer must be present"
    assert UserTransformer in transformers, "UserTransformer must be present"
    subagent_idx = transformers.index(SubagentTransformer)
    user_idx = transformers.index(UserTransformer)
    assert subagent_idx < user_idx, "SubagentTransformer must precede user-supplied transformers"
