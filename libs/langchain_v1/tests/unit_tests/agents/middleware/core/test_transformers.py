"""Tests for middleware-registered stream transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolCallTransformer
from langgraph.stream import StreamChannel, StreamTransformer

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent


class _MiddlewareMarker(StreamTransformer):
    """Marker transformer used to assert registration order."""

    required_stream_modes = ()

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[int] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"middleware_marker": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        del event
        return True


class _UserMarker(StreamTransformer):
    """Second marker to verify user-supplied transformers append last."""

    required_stream_modes = ()

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[int] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"user_marker": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        del event
        return True


def test_middleware_transformer_registered_on_compiled_graph() -> None:
    """A `transformers` factory declared on middleware is wired into the run mux."""

    class _Middleware(AgentMiddleware):
        transformers = (_MiddlewareMarker,)

    agent = create_agent(model=FakeToolCallingModel(), tools=[], middleware=[_Middleware()])

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    assert "middleware_marker" in run._mux.extensions  # type: ignore[attr-defined]
    # Drain to close the run cleanly.
    list(run.tool_calls)


def test_middleware_and_user_transformers_compose_in_order() -> None:
    """Order is: built-in `ToolCallTransformer` → middleware → user-supplied."""

    class _Middleware(AgentMiddleware):
        transformers = (_MiddlewareMarker,)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[_Middleware()],
        transformers=[_UserMarker],
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    transformers = run._mux._transformers  # type: ignore[attr-defined]
    tool_call_idx = next(
        i for i, t in enumerate(transformers) if isinstance(t, ToolCallTransformer)
    )
    middleware_idx = next(i for i, t in enumerate(transformers) if isinstance(t, _MiddlewareMarker))
    user_idx = next(i for i, t in enumerate(transformers) if isinstance(t, _UserMarker))

    assert tool_call_idx < middleware_idx < user_idx, (
        "transformers must register as: built-in, then middleware, then user-supplied"
    )

    list(run.tool_calls)


def test_transformers_from_multiple_middleware_preserve_middleware_order() -> None:
    """Transformers across middleware register in middleware-list order."""

    class _MarkerA(_MiddlewareMarker):
        def init(self) -> dict[str, Any]:
            return {"marker_a": self._log}

    class _MarkerB(_MiddlewareMarker):
        def init(self) -> dict[str, Any]:
            return {"marker_b": self._log}

    class _MwA(AgentMiddleware):
        transformers = (_MarkerA,)

    class _MwB(AgentMiddleware):
        transformers = (_MarkerB,)

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[_MwA(), _MwB()],
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    transformers = run._mux._transformers  # type: ignore[attr-defined]
    idx_a = next(i for i, t in enumerate(transformers) if isinstance(t, _MarkerA))
    idx_b = next(i for i, t in enumerate(transformers) if isinstance(t, _MarkerB))
    assert idx_a < idx_b

    list(run.tool_calls)


def test_middleware_without_transformers_does_not_affect_registry() -> None:
    """Middleware that omits `transformers` leaves the default registry intact."""

    class _Middleware(AgentMiddleware):
        pass

    agent = create_agent(model=FakeToolCallingModel(), tools=[], middleware=[_Middleware()])
    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    transformers = run._mux._transformers  # type: ignore[attr-defined]
    assert any(isinstance(t, ToolCallTransformer) for t in transformers)
    assert not any(isinstance(t, _MiddlewareMarker) for t in transformers)

    list(run.tool_calls)
