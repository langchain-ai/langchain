"""Unit tests for `MiddlewareTransformer`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from langchain.agents import (
    MiddlewareEvent,
    MiddlewareTransformer,
    create_agent,
)
from langchain.agents.middleware import AgentMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


# ---------------------------------------------------------------------------
# Unit-level: feed the transformer synthetic protocol events
# ---------------------------------------------------------------------------


def _updates_event(
    node_updates: dict[str, Any], *, namespace: list[str] | None = None
) -> dict[str, Any]:
    """Build a synthetic `updates` protocol event for testing `process`."""
    return {
        "type": "event",
        "method": "updates",
        "params": {
            "namespace": namespace or [],
            "timestamp": 1234,
            "data": node_updates,
        },
    }


def _fresh_transformer() -> MiddlewareTransformer:
    """Build a bound-and-subscribed transformer so `push` accumulates."""
    tr = MiddlewareTransformer()
    log = tr.init()["middleware"]
    log._bind(is_async=False)
    # Mark subscribed so `log.push` records items. Normally this happens via
    # `iter(log)`; for unit-level tests we want to inspect `_items` after
    # `process` without consuming the iterator mid-test.
    log._subscribed = True
    return tr


class TestMiddlewareTransformerProcess:
    """`MiddlewareTransformer.process` in isolation."""

    def test_emits_event_for_middleware_node(self) -> None:
        tr = _fresh_transformer()
        kept = tr.process(_updates_event({"MyMW.before_model": {"messages": ["hi"]}}))
        assert kept is True  # event passes through
        events = list(tr._log._items)
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, MiddlewareEvent)
        assert ev.phase == "before_model"
        assert ev.middleware_name == "MyMW"
        assert ev.state_delta == {"messages": ["hi"]}
        assert ev.timestamp == 1234

    def test_all_four_phases(self) -> None:
        tr = _fresh_transformer()
        for phase in (
            "before_agent",
            "before_model",
            "after_model",
            "after_agent",
        ):
            tr.process(_updates_event({f"MW.{phase}": {"k": phase}}))
        phases = [ev.phase for ev in tr._log._items]
        assert phases == [
            "before_agent",
            "before_model",
            "after_model",
            "after_agent",
        ]

    def test_ignores_non_middleware_nodes(self) -> None:
        tr = _fresh_transformer()
        tr.process(_updates_event({"model": {"messages": []}}))
        tr.process(_updates_event({"tools": {"messages": []}}))
        tr.process(_updates_event({"MW.weird_phase": {}}))  # not a real phase
        tr.process(_updates_event({"NoDot": {}}))
        assert list(tr._log._items) == []

    def test_ignores_non_updates_methods(self) -> None:
        tr = _fresh_transformer()
        kept = tr.process(
            {
                "type": "event",
                "method": "values",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": {"anything": "goes"},
                },
            }
        )
        assert kept is True
        assert list(tr._log._items) == []

    def test_multi_node_batch_in_single_event(self) -> None:
        tr = _fresh_transformer()
        tr.process(
            _updates_event(
                {
                    "A.before_model": {"x": 1},
                    "B.after_model": {"y": 2},
                    "regular_node": {"z": 3},
                }
            )
        )
        events = list(tr._log._items)
        assert len(events) == 2
        names_by_phase = {ev.phase: ev.middleware_name for ev in events}
        assert names_by_phase == {"before_model": "A", "after_model": "B"}

    def test_non_dict_state_delta_becomes_empty_dict(self) -> None:
        tr = _fresh_transformer()
        tr.process(_updates_event({"MW.before_agent": None}))
        events = list(tr._log._items)
        assert len(events) == 1
        assert events[0].state_delta == {}

    def test_dotted_middleware_name_uses_rpartition(self) -> None:
        # Middleware name with a `.` in it — rpartition splits on the last one.
        tr = _fresh_transformer()
        tr.process(_updates_event({"my.name.spaced.MW.before_model": {"ok": 1}}))
        events = list(tr._log._items)
        assert len(events) == 1
        assert events[0].middleware_name == "my.name.spaced.MW"
        assert events[0].phase == "before_model"


# ---------------------------------------------------------------------------
# Integration: create_agent auto-registers it; run.middleware works
# ---------------------------------------------------------------------------


class _MarkerMiddleware(AgentMiddleware):
    """Touches state in `before_model` so we can see an updates event fire."""

    name = "MarkerMW"

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return {"messages": []}


class TestCreateAgentRegistersMiddlewareTransformer:
    def test_middleware_projection_present(self) -> None:
        model = FakeToolCallingModel(tool_calls=[[], []])
        agent = create_agent(model, [], middleware=[_MarkerMiddleware()])
        run = agent.stream_v2({"messages": [HumanMessage("hi")]})
        assert "middleware" in run.extensions
        assert hasattr(run, "middleware")
        # Drive to completion.
        run.output  # noqa: B018

    def test_middleware_events_emitted_for_before_model_hook(self) -> None:
        model = FakeToolCallingModel(
            tool_calls=[[], []],
            # Have the model respond once and stop.
            responses=[AIMessage(content="done", id="a1", tool_calls=[])],
        )
        agent = create_agent(model, [], middleware=[_MarkerMiddleware()])
        run = agent.stream_v2({"messages": [HumanMessage("hi")]})
        events: list[MiddlewareEvent] = list(run.middleware)
        assert any(
            ev.phase == "before_model" and ev.middleware_name == "MarkerMW" for ev in events
        ), f"expected a MarkerMW.before_model event; got {events}"

    def test_no_events_when_no_middleware(self) -> None:
        model = FakeToolCallingModel(
            tool_calls=[[]],
            responses=[AIMessage(content="done", id="a1", tool_calls=[])],
        )
        agent = create_agent(model, [])
        run = agent.stream_v2({"messages": [HumanMessage("hi")]})
        events = list(run.middleware)
        assert events == []
