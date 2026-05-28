"""Surface tools declaring `subagent_name` as typed `run.subagents` handles.

Generalizes deepagents' SubagentTransformer to work with any tool that uses
the `subagent_name=` parameter on `@tool` / `StructuredTool.from_function`.

Reads the resolved name from `TaskPayload.metadata['subagent_name']`, which
langgraph's `ToolNode` derives from `BaseTool.subagent_name` via the
`pregel_dynamic_metadata` hook at task scheduling time. The metadata is
projected onto the parent-scope `tasks` event for the tool dispatch; this
transformer correlates that event to the child-scope `tasks` events emitted
by the nested run that the tool spawns.
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

    The Pregel scheduler emits a `tasks` event at the *parent* namespace
    when a node is scheduled, carrying the task's `id` and any metadata
    set on its config. ToolNode's `pregel_dynamic_metadata` hook stamps
    `subagent_name` and `tool_call_id` into that metadata when the
    dispatched tool declares `BaseTool.subagent_name`.

    Nested runs spawned inside the tool (e.g., a subagent's Pregel
    invocation) emit their own `tasks` events at a child namespace
    that shares the parent task's id (segment shape `"<node>:<id>"`).

    This transformer correlates the two: it stashes
    `task_id -> (subagent_name, tool_call_id)` from parent-scope events,
    and on the first matching child-scope event emits a typed handle on
    `run.subagents`.

    Optionally filters to a declared set of subagent names via
    `subagent_names`; when `None` (default), all dispatched subagents
    surface.
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
        # task_id -> {"subagent_name": ..., "tool_call_id": ...}, populated
        # from parent-scope `tasks` events and consumed on the first matching
        # child-scope task start.
        self._pending: dict[str, dict[str, str]] = {}

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _capture_pending_from_parent(self, event: ProtocolEvent) -> None:
        """Stash subagent metadata from a parent-scope task start.

        Pregel emits a `tasks` event at this transformer's scope when a child
        node is scheduled. The event's `data` carries the task `id` and the
        `metadata` projected from the task config (which `ToolNode.pregel_dynamic_metadata`
        populated with `subagent_name` and `tool_call_id`). Stash by id so the
        corresponding child-scope event can pop it.
        """
        ns = tuple(event["params"]["namespace"])
        if ns != self.scope:
            return
        data = event["params"]["data"]
        if not isinstance(data, dict) or "result" in data:
            return
        task_id = data.get("id")
        metadata = data.get("metadata") or {}
        if not isinstance(task_id, str) or not isinstance(metadata, dict):
            return
        subagent_name = metadata.get("subagent_name")
        if not isinstance(subagent_name, str) or not subagent_name:
            return
        tool_call_id = metadata.get("tool_call_id")
        self._pending[task_id] = {
            "subagent_name": subagent_name,
            "tool_call_id": tool_call_id if isinstance(tool_call_id, str) else "",
        }

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,  # noqa: ARG002
        trigger_call_id: str | None,
        *,
        cause: LifecycleCause | None = None,  # noqa: ARG002
    ) -> None:
        if trigger_call_id is None:
            return
        info = self._pending.pop(trigger_call_id, None)
        if info is None:
            return
        subagent_name = info["subagent_name"]
        if self._names is not None and subagent_name not in self._names:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)  # noqa: SLF001
        except RuntimeError:
            logger.debug("SubagentTransformer: could not create child mux for %s", ns)
            return

        tool_call_id = info["tool_call_id"] or None
        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=subagent_name,
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
        if event.get("method") == "tasks":
            self._capture_pending_from_parent(event)
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            handle._observe_event(event)  # noqa: SLF001
            handle._mux.push(event)  # noqa: SLF001
        return keep
