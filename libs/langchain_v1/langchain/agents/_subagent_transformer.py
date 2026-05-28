"""Surface tools declaring `subagent_name` as typed `run.subagents` handles.

Generalizes deepagents' SubagentTransformer to work with any tool that uses
the `subagent_name=` parameter on `@tool` / `StructuredTool.from_function`.

Reads the resolved name from `TaskPayload.metadata['subagent_name']` (stamped
by langgraph's `ToolNode` at dispatch time). The base class
`_TasksLifecycleBase._handle_task_start` already extracts `subagent_name` from
metadata and populates `graph_name` and `cause` accordingly, so this
transformer only needs to act on `_on_started` calls where `cause` is non-None
(the discriminator for "this task was a subagent dispatch").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream.run_stream import (
    AsyncSubgraphRunStream,
    SubgraphRunStream,
)
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    LifecycleCause,
    SubgraphStatus,
    _TasksLifecycleBase,
)

if TYPE_CHECKING:
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent

logger = logging.getLogger(__name__)


class SubagentRunStream(SubgraphRunStream):
    """Typed sync handle for a declared subagent execution.

    Surfaces on `run.subagents` when the dispatching tool has
    `BaseTool.subagent_name` set (statically or via callable resolver).
    """

    @property
    def name(self) -> str | None:
        """Declared subagent name (alias of `graph_name`)."""
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        """Causation edge — the tool_call that triggered this subagent.

        Returns `{"type": "toolCall", "tool_call_id": ...}` when a
        trigger_call_id is set, else `None`.
        """
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "tool_call_id": self.trigger_call_id}


class AsyncSubagentRunStream(AsyncSubgraphRunStream):
    """Typed async handle for a declared subagent execution."""

    @property
    def name(self) -> str | None:
        """Declared subagent name (alias of `graph_name`)."""
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        """Causation edge — the tool_call that triggered this subagent.

        Returns `{"type": "toolCall", "tool_call_id": ...}` when a
        trigger_call_id is set, else `None`.
        """
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "tool_call_id": self.trigger_call_id}


class SubagentTransformer(_TasksLifecycleBase):
    """Promote declared subagents into typed handles on `run.subagents`.

    Watches `tasks` events at the configured scope. For each child task
    whose `TaskPayload.metadata` contains a `subagent_name` (stamped by
    `ToolNode` from `BaseTool.subagent_name`), builds a typed
    `SubagentRunStream` handle and pushes it onto the `subagents` log.

    The base class extracts `subagent_name` from metadata and passes it as
    `graph_name` to `_on_started`. It also builds `cause` only when
    `subagent_name` is present, so `cause is not None` is the discriminator
    for subagent tasks vs. ordinary nested-graph dispatches.

    Optionally filters to a declared set of subagent names via
    `subagent_names`; when `None` (default), all dispatched subagents surface.
    """

    _native: ClassVar[bool] = True

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        *,
        subagent_names: frozenset[str] | None = None,
    ) -> None:
        super().__init__(scope)
        self._names = subagent_names
        self._log: StreamChannel[SubagentRunStream | AsyncSubagentRunStream] = StreamChannel()
        self._handles: dict[tuple[str, ...], SubagentRunStream | AsyncSubagentRunStream] = {}
        self._mux: StreamMux | None = None

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,
        trigger_call_id: str | None,  # noqa: ARG002
        *,
        cause: LifecycleCause | None = None,
    ) -> None:
        # cause is only set by the base class when metadata.subagent_name was
        # present — it is the discriminator for "this is a subagent dispatch."
        if cause is None:
            return
        # graph_name carries the subagent_name extracted from metadata.
        if not isinstance(graph_name, str) or not graph_name:
            return
        if self._names is not None and graph_name not in self._names:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)  # noqa: SLF001
        except RuntimeError:
            logger.debug("SubagentTransformer: could not create child mux for %s", ns)
            return

        # Extract tool_call_id from cause for the SubagentRunStream.cause property.
        # cause is LifecycleCauseToolCall = {"type": "toolCall", "toolCallId": "..."}
        tool_call_id: str | None = None
        if isinstance(cause, dict):
            raw = cause.get("toolCallId")
            if isinstance(raw, str):
                tool_call_id = raw

        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=graph_name,
            trigger_call_id=tool_call_id,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or handle._seen_terminal:  # noqa: SLF001
            return
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True  # noqa: SLF001
        if handle._mux is None or handle._mux._events._closed:  # noqa: SLF001
            return
        if status == "failed":
            handle._mux.fail(RuntimeError(error or "Subagent failed"))  # noqa: SLF001
        else:
            handle._mux.close()  # noqa: SLF001

    def _handle_for_event(
        self, event: ProtocolEvent
    ) -> SubagentRunStream | AsyncSubagentRunStream | None:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:  # noqa: SLF001
            return None
        return handle

    def process(self, event: ProtocolEvent) -> bool:
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)  # noqa: SLF001
            handle._mux.push(event)  # noqa: SLF001
        return keep
