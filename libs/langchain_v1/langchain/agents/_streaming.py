"""Streaming entry point for `create_agent` graphs.

`AgentStreamer` pre-registers `ToolCallTransformer` so every agent run
exposes `run.tool_calls` without the caller opting in.

Example:
    ```python
    from langchain.agents import AgentStreamer, create_agent

    agent = create_agent(model, tools)

    run = AgentStreamer(agent).stream({"messages": [...]})
    for tc in run.tool_calls:
        for delta in tc.output_deltas:
            print(delta, end="")
    print(run.output)
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.prebuilt import ToolCallTransformer
from langgraph.stream import GraphStreamer
from langgraph.stream.run_stream import AsyncGraphRunStream, GraphRunStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langgraph.stream._mux import StreamMux, TransformerFactory


class AgentRunStream(GraphRunStream):
    """Sync run stream for a `create_agent` graph.

    Native projections (`tool_calls`, `messages`, `values`) are bound as
    instance attributes by `BaseRunStream.__init__` whenever the
    matching transformer is registered — this subclass exists for
    `isinstance` checks and as an extension point for downstream
    streamers (e.g. a deepagents-layer `DeepAgentRunStream`).
    """


class AsyncAgentRunStream(AsyncGraphRunStream):
    """Async counterpart to `AgentRunStream`."""


class AgentStreamer(GraphStreamer):
    """`GraphStreamer` pre-configured for `create_agent` graphs.

    Extends `GraphStreamer.builtin_factories` with `ToolCallTransformer`
    so `run.tool_calls` is populated on every run without the caller
    passing `transformers=[ToolCallTransformer]`. Returns
    `AgentRunStream` / `AsyncAgentRunStream` for `isinstance` checks.

    Caller-supplied `transformers=[...]` on `stream()` / `astream()`
    are appended after the built-ins, matching `GraphStreamer`'s
    behavior — they add to, rather than replace, the agent defaults.
    """

    builtin_factories: ClassVar[tuple[TransformerFactory, ...]] = (
        *GraphStreamer.builtin_factories,
        ToolCallTransformer,
    )

    def _make_run_stream(
        self,
        graph_iter: Iterator[Any],
        mux: StreamMux,
    ) -> AgentRunStream:
        return AgentRunStream(graph_iter, mux)

    def _make_async_run_stream(
        self,
        graph_aiter: AsyncIterator[Any],
        mux: StreamMux,
    ) -> AsyncAgentRunStream:
        return AsyncAgentRunStream(graph_aiter, mux)
