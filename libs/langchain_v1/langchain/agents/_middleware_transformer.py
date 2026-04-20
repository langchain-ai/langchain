"""Project graph `updates` events into typed `MiddlewareEvent` handles.

Exposed via `run.middleware` on `AgentStreamer` runs.

`create_agent` wires each middleware hook into the graph as a real
node named `"{middleware_name}.{phase}"`, where `phase` is one of
`before_agent` / `before_model` / `after_model` / `after_agent`.
When any of those nodes run, Pregel emits an `updates` event with
the node's state delta — we subscribe to `updates`, filter by the
phase-name suffix, and re-emit as `MiddlewareEvent` objects.

No cooperation from middleware implementations is required — the
node-name convention is the contract, and the transformer is a pure
consumer of the graph's existing `updates` stream mode.

`run.middleware` is an in-process-only projection: it surfaces
lifecycle information for local consumers and is not intended to
cross a wire boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from langgraph.stream._event_log import EventLog
from langgraph.stream._types import StreamTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent


MiddlewarePhase = Literal[
    "before_agent", "before_model", "after_model", "after_agent"
]
"""Lifecycle phase a middleware node represents."""


_PHASES: frozenset[str] = frozenset(
    ("before_agent", "before_model", "after_model", "after_agent")
)


@dataclass(frozen=True)
class MiddlewareEvent:
    """One lifecycle transition from a middleware node.

    Emitted once per node execution, carrying the phase, the name of
    the middleware that produced it, and the state delta returned by
    the hook.

    Attributes:
        phase: Which hook fired — `before_agent` / `before_model` /
            `after_model` / `after_agent`.
        middleware_name: The `name` attribute of the middleware class
            that produced this event (the part before the `.` in the
            graph node name).
        state_delta: The state diff the middleware hook returned.
            Empty dict if the hook didn't return an update.
        timestamp: Milliseconds since epoch, taken from the underlying
            protocol event.
    """

    phase: MiddlewarePhase
    middleware_name: str
    state_delta: dict[str, Any]
    timestamp: int


def _split_middleware_node(node_name: str) -> tuple[str, MiddlewarePhase] | None:
    """Split `"{name}.{phase}"` into `(name, phase)` if `phase` is valid.

    Returns `None` for any node name that doesn't match the convention,
    so non-middleware nodes are ignored.
    """
    name, sep, suffix = node_name.rpartition(".")
    if not sep or suffix not in _PHASES:
        return None
    return name, suffix  # type: ignore[return-value]


class MiddlewareTransformer(StreamTransformer):
    """Project `updates` events from middleware nodes into `MiddlewareEvent`s.

    Watches `updates` events at its mux scope, inspects the node name,
    and emits one `MiddlewareEvent` per middleware-node execution onto
    the `middleware` projection.

    Native transformer — the `middleware` projection is exposed as a
    direct attribute (`run.middleware`). Pre-registered by
    `AgentStreamer.builtin_factories` so it's on by default for any
    `create_agent` graph.

    `scope` (inherited from `StreamTransformer`) lets the transformer
    scope its filtering to its own mux — root mux sees root middleware,
    subgraph mini-muxes see their own middleware, via the same
    machinery `MessagesTransformer` uses.
    """

    _native: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: EventLog[MiddlewareEvent] = EventLog()

    def init(self) -> dict[str, Any]:
        return {"middleware": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        # Namespace filtering is handled by the mux via `scope_exact`.
        if event["method"] != "updates":
            return True

        params = event["params"]
        data = params.get("data")
        if not isinstance(data, dict):
            return True

        timestamp = int(params.get("timestamp", 0))
        for node_name, state_delta in data.items():
            split = _split_middleware_node(node_name)
            if split is None:
                continue
            name, phase = split
            self._log.push(
                MiddlewareEvent(
                    phase=phase,
                    middleware_name=name,
                    state_delta=state_delta if isinstance(state_delta, dict) else {},
                    timestamp=timestamp,
                )
            )
        # Pass through — wire consumers still see the raw `updates` event.
        return True
