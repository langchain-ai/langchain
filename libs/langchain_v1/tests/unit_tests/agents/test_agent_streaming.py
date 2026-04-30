"""Unit tests for create_agent graphs streaming via `stream_events(version="v3")`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.stream import StreamChannel, StreamTransformer

from langchain.agents import create_agent
from langchain.tools import ToolRuntime  # noqa: TC001
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langgraph.prebuilt._tool_call_stream import ToolCallStream
    from langgraph.stream._types import ProtocolEvent


@tool
def echo(text: str) -> str:
    """Return the input unchanged."""
    return text


@tool
def streamer(text: str, runtime: ToolRuntime) -> str:
    """Stream two chunks, then return the full text."""
    for chunk in ("one", "two"):
        runtime.emit_output_delta(chunk)
    return text


@tool
async def astreamer(text: str, runtime: ToolRuntime) -> str:
    """Async: stream two chunks, then return the full text."""
    runtime.emit_output_delta(text)
    runtime.emit_output_delta(text + "!")
    return text


@tool
def boom() -> str:
    """Raise unconditionally."""
    msg = "nope"
    raise ValueError(msg)


def _single_tool_call_script(name: str, **args: Any) -> list[list[dict[str, Any]]]:
    """Script: one tool call on turn 0, finish on turn 1."""
    return [
        [{"name": name, "args": args, "id": "tc1"}],
        [],
    ]


class TestAgentStreamV2Sync:
    def test_stream_returns_agent_run_stream(self) -> None:
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("echo", text="x"))
        agent = create_agent(model, [echo])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})

        # Drain so the run closes cleanly.
        list(run.tool_calls)

    def test_tool_calls_populated_without_opt_in(self) -> None:
        """`ToolCallTransformer` is registered by default on the agent streamer."""
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("echo", text="x"))
        agent = create_agent(model, [echo])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})

        collected: list[ToolCallStream] = list(run.tool_calls)
        assert len(collected) == 1
        tc = collected[0]
        assert tc.tool_name == "echo"
        assert tc.tool_call_id == "tc1"
        assert tc.completed is True
        assert tc.error is None

    def test_tool_output_deltas_flow_through(self) -> None:
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("streamer", text="x"))
        agent = create_agent(model, [streamer])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})

        tool_calls: list[ToolCallStream] = []
        for tc in run.tool_calls:
            tool_calls.append(tc)
            assert list(tc.output_deltas) == ["one", "two"]
        assert len(tool_calls) == 1

    def test_no_tools_run_is_still_usable(self) -> None:
        """`.tool_calls` is empty when the model never calls a tool."""
        model = FakeToolCallingModel()  # no tool calls scripted
        agent = create_agent(model, [])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})
        assert list(run.tool_calls) == []
        assert run.output is not None

    def test_messages_projection_present(self) -> None:
        """`MessagesTransformer` is inherited from `GraphStreamer.builtin_factories`."""
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("echo", text="x"))
        agent = create_agent(model, [echo])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})
        # The native `messages` projection is bound as an instance attribute
        # by `BaseRunStream.__init__` whenever `MessagesTransformer` is
        # registered. Content population is covered by langgraph tests —
        # here we only assert the agent streamer inherits the built-in.
        assert "messages" in run._mux.extensions  # type: ignore[attr-defined]
        assert hasattr(run, "messages")
        # Drain so the run closes cleanly.
        for tc in run.tool_calls:
            list(tc.output_deltas)

    def test_caller_transformers_appended_not_replaced(self) -> None:
        """User-supplied transformers add to, rather than replace, the agent defaults."""

        class _Marker(StreamTransformer):
            required_stream_modes = ()

            def __init__(self, scope: tuple[str, ...] = ()) -> None:
                super().__init__(scope)
                self._log: StreamChannel[int] = StreamChannel()

            def init(self) -> dict[str, Any]:
                return {"marker": self._log}

            def process(self, event: ProtocolEvent) -> bool:
                del event
                return True

        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("echo", text="x"))
        agent = create_agent(model, [echo])

        run = agent.stream_v2(
            {"messages": [HumanMessage("hi")]},
            transformers=[_Marker],
        )
        # Both the agent default and the user transformer are registered.
        assert "tool_calls" in run._mux.extensions  # type: ignore[attr-defined]
        assert "marker" in run._mux.extensions  # type: ignore[attr-defined]
        list(run.tool_calls)

    def test_tool_error_sets_error_field(self) -> None:
        """Tool errors are surfaced on the `ToolCallStream.error` field.

        `create_agent`'s default tool-error handler re-raises, so the
        overall run fails — the assertion here is that the error is
        attached to the scoped `ToolCallStream` *before* the run raises.
        """
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("boom"))
        agent = create_agent(model, [boom])

        run = agent.stream_v2({"messages": [HumanMessage("hi")]})

        collected: list[ToolCallStream] = []

        def _drive() -> None:
            for tc in run.tool_calls:
                collected.append(tc)
                list(tc.output_deltas)

        with pytest.raises(ValueError, match="nope"):
            _drive()
        assert len(collected) == 1
        assert collected[0].error is not None
        assert "nope" in collected[0].error
        assert collected[0].completed is True


class TestAgentStreamV2Async:
    @pytest.mark.anyio
    async def test_astream_returns_async_agent_run_stream(self) -> None:
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("echo", text="x"))
        agent = create_agent(model, [echo])

        run = await agent.astream_v2({"messages": [HumanMessage("hi")]})
        async for tc in run.tool_calls:
            async for _ in tc.output_deltas:
                pass

    @pytest.mark.anyio
    async def test_async_tool_deltas_flow(self) -> None:
        model = FakeToolCallingModel(tool_calls=_single_tool_call_script("astreamer", text="hi"))
        agent = create_agent(model, [astreamer])

        run = await agent.astream_v2({"messages": [HumanMessage("hi")]})

        collected: list[ToolCallStream] = []
        async for tc in run.tool_calls:
            collected.append(tc)
            deltas = [d async for d in tc.output_deltas]
            assert deltas == ["hi", "hi!"]
        assert len(collected) == 1
        assert collected[0].completed is True
