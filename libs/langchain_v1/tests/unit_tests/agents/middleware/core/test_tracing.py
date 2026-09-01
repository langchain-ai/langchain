"""End-to-end tests for `AgentMiddleware.trace_policy` and the global default."""

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tracers import BaseTracer, Run
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.factory import _wrap_trace_kwargs, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    TracePolicy,
    configure_trace_policy,
    omit_payload,
)
from langchain.agents.middleware.types import AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel


@pytest.fixture(autouse=True)
def _reset_global_trace_policy() -> Iterator[None]:
    """Keep the process-global default from leaking across tests."""
    configure_trace_policy(None)
    yield
    configure_trace_policy(None)


class _CaptureInputs(BaseTracer):
    """Tracer that records the inputs each run reports, keyed by run name.

    A real tracer (not a plain callback) so `TracePolicy` processors, which run only
    when a tracer is attached, actually fire.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inputs_by_name: dict[str, Any] = {}

    def _persist_run(self, run: Run) -> None:
        pass

    def _on_chain_start(self, run: Run) -> None:
        self.inputs_by_name.setdefault(run.name, run.inputs)


def _recorded_before_model_inputs(middleware: AgentMiddleware) -> Any:
    """Run the agent and return what the middleware's before_model node recorded."""
    agent = create_agent(model=FakeToolCallingModel(), middleware=[middleware])
    capture = _CaptureInputs()
    agent.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [capture]})
    return capture.inputs_by_name[f"{middleware.name}.before_model"]


class _NoopBeforeModel(AgentMiddleware):
    """Middleware with a before_model hook and no `trace_policy`."""

    @override
    def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
        return None


def test_tracing_records_hook_inputs_by_default() -> None:
    # no `trace_policy`, no global -> hooks trace normally; real state is recorded
    recorded = _recorded_before_model_inputs(_NoopBeforeModel())
    assert "messages" in recorded
    assert [m.content for m in recorded["messages"]] == ["hi"]


def test_tracing_omits_hook_inputs_with_omit_payload() -> None:
    class Scrubbed(AgentMiddleware):
        trace_policy = TracePolicy(process_inputs=omit_payload)

        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    # opt in via omit_payload -> the before_model node records an empty payload
    assert _recorded_before_model_inputs(Scrubbed()) == {}


def test_global_default_applies_when_middleware_unset() -> None:
    configure_trace_policy(TracePolicy(process_inputs=omit_payload))

    # middleware leaves trace_policy=None -> inherits the global default
    assert _recorded_before_model_inputs(_NoopBeforeModel()) == {}


def test_middleware_overrides_global_no_merge() -> None:
    configure_trace_policy(TracePolicy(process_inputs=omit_payload))

    class Override(AgentMiddleware):
        # sets only process_outputs; per override-not-merge this replaces the global
        # wholesale, so the global's input scrub does NOT apply -> inputs recorded raw
        trace_policy = TracePolicy(process_outputs=omit_payload)

        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    recorded = _recorded_before_model_inputs(Override())
    assert "messages" in recorded


def test_configure_after_create_agent_applies() -> None:
    mw = _NoopBeforeModel()
    agent = create_agent(model=FakeToolCallingModel(), middleware=[mw])
    # configure AFTER the graph is built; the resolver reads the global at trace time
    configure_trace_policy(TracePolicy(process_inputs=omit_payload))
    capture = _CaptureInputs()
    agent.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [capture]})
    assert capture.inputs_by_name[f"{mw.name}.before_model"] == {}


def test_configure_none_clears_global() -> None:
    configure_trace_policy(TracePolicy(process_inputs=omit_payload))
    configure_trace_policy(None)

    recorded = _recorded_before_model_inputs(_NoopBeforeModel())
    assert "messages" in recorded


def test_wrap_trace_kwargs_composes_scrub_baseline() -> None:
    seen: dict[str, Any] = {}

    def process_inputs(inp: Any) -> Any:
        seen["inputs"] = inp
        return inp

    class MW(AgentMiddleware):
        trace_policy = TracePolicy(process_inputs=process_inputs)

    process = _wrap_trace_kwargs(MW())["process_inputs"]
    result = process({"request": {"x": 1}, "handler": lambda: None})
    # the baseline (`_scrub_inputs`) strips `handler` before the policy callable runs
    assert "handler" not in seen["inputs"]
    assert seen["inputs"] == {"request": {"x": 1}}
    assert result == {"request": {"x": 1}}


def test_wrap_trace_kwargs_omit_drops_everything() -> None:
    class MW(AgentMiddleware):
        trace_policy = TracePolicy(process_inputs=omit_payload)

    process = _wrap_trace_kwargs(MW())["process_inputs"]
    assert process({"request": {"x": 1}, "handler": lambda: None}) == {}


def test_wrap_trace_kwargs_inherits_global() -> None:
    configure_trace_policy(TracePolicy(process_inputs=omit_payload))

    class MW(AgentMiddleware):
        pass

    # no middleware policy -> wrap hook inherits the global, still after the baseline
    process = _wrap_trace_kwargs(MW())["process_inputs"]
    assert process({"request": {"x": 1}, "handler": lambda: None}) == {}
