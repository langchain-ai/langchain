"""Surface nested named agents as typed `run.subagents` handles.

Detects subagents via the `lc_agent_name` transition that langgraph's base
`_TasksLifecycleBase` now computes. `create_agent(name=...)` binds
`lc_agent_name` into the run config; the base transformer records, per
namespace, the `lc_agent_name` seen on each task start (first-write-wins).

A subagent boundary is a nested run whose `lc_agent_name` is set *and* differs
from its parent namespace's `lc_agent_name`. Plain subgraphs inherit the
parent's name (so they compare equal and are excluded); unnamed agents have
`lc_agent_name == None` (also excluded). For genuine subagents the base also
recovers the originating tool call and passes it through as a `cause`
(`{"type": "toolCall", "tool_call_id": ...}`), joined from the parent task's
pending tool calls.

This transformer gates on that boundary and surfaces a typed handle on
`run.subagents`, then forwards child-scope events into the handle's mux so the
nested run can be consumed independently.
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
    SubgraphStatus,
    _TasksLifecycleBase,
)

if TYPE_CHECKING:
    from langchain_protocol.protocol import LifecycleCause
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent

logger = logging.getLogger(__name__)


class SubagentRunStream(SubgraphRunStream):
    """Typed sync handle for a nested named-agent execution.

    Surfaces on `run.subagents` when a nested run's `lc_agent_name` differs
    from its parent's (i.e., a `create_agent(name=...)` dispatched from a tool).
    """

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        cause: LifecycleCause | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._cause = cause

    @property
    def name(self) -> str | None:
        """Subagent name (the nested run's `lc_agent_name`)."""
        return self.graph_name

    @property
    def cause(self) -> LifecycleCause | None:
        """Causation edge — the tool call that triggered this subagent.

        Returns the `LifecycleCause` recovered by the base transformer (a
        `{"type": "toolCall", "tool_call_id": ...}` dict) when the originating
        tool call could be joined, else `None`.
        """
        return self._cause


class AsyncSubagentRunStream(AsyncSubgraphRunStream):
    """Typed async handle for a nested named-agent execution."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        cause: LifecycleCause | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self._cause = cause

    @property
    def name(self) -> str | None:
        """Subagent name (the nested run's `lc_agent_name`)."""
        return self.graph_name

    @property
    def cause(self) -> LifecycleCause | None:
        """Causation edge — the tool call that triggered this subagent.

        Returns the `LifecycleCause` recovered by the base transformer (a
        `{"type": "toolCall", "tool_call_id": ...}` dict) when the originating
        tool call could be joined, else `None`.
        """
        return self._cause


class SubagentTransformer(_TasksLifecycleBase):
    """Promote nested named agents into typed handles on `run.subagents`.

    The base `_TasksLifecycleBase` records each namespace's `lc_agent_name`
    (set by `create_agent(name=...)`) and, on every task start, fires
    `_on_started` with the resolved `graph_name` and a `cause` for genuine
    subagent boundaries. This transformer gates on that boundary using the
    inherited `_lc_by_ns` map: a nested run is a subagent when its
    `lc_agent_name` is set and differs from its parent's. Plain subgraphs
    (which inherit the parent's name) and unnamed agents (`None`) are excluded.

    On the first matching task start it builds a child mux and emits a typed
    handle on `run.subagents`, then forwards subsequent child-scope events into
    that handle so the nested run can be consumed independently.
    """

    _native: ClassVar[bool] = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
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
        trigger_call_id: str | None,
        *,
        cause: LifecycleCause | None = None,
    ) -> None:
        child_lc = self._lc_by_ns.get(ns)
        parent_lc = self._lc_by_ns.get(ns[:-1])
        # Surface only genuine subagents: a nested run whose lc_agent_name
        # is set and differs from its parent's. Plain subgraphs inherit the
        # parent's name (== parent); unnamed agents have None. Both excluded.
        if child_lc is None or child_lc == parent_lc:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)  # noqa: SLF001
        except RuntimeError:
            logger.debug("SubagentTransformer: could not create child mux for %s", ns)
            return

        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
            cause=cause,
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
