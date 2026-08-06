"""End-to-end tests for `AgentMiddleware.trace_policy` hook-input recording."""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.factory import _wrap_process_inputs, create_agent
from langchain.agents.middleware import AgentMiddleware, TracePolicy, omit_payload
from langchain.agents.middleware.types import AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel


class _CaptureInputs(BaseCallbackHandler):
    """Record the inputs each chain run reports on `on_chain_start`, by run name."""

    def __init__(self) -> None:
        self.inputs_by_name: dict[str, Any] = {}

    def on_chain_start(self, serialized: dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name is not None:
            self.inputs_by_name.setdefault(name, inputs)


def _recorded_before_model_inputs(middleware: AgentMiddleware) -> Any:
    """Run the agent and return what the middleware's before_model node recorded."""
    agent = create_agent(model=FakeToolCallingModel(), middleware=[middleware])
    capture = _CaptureInputs()
    agent.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [capture]})
    return capture.inputs_by_name[f"{middleware.name}.before_model"]


def test_tracing_records_hook_inputs_by_default() -> None:
    class Default(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    # no `trace_policy` -> hooks trace normally; the real state (incl. messages) is recorded
    recorded = _recorded_before_model_inputs(Default())
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


def test_wrap_process_inputs_composes_scrub_baseline() -> None:
    seen: dict[str, Any] = {}

    def process_inputs(inp: Any) -> Any:
        seen["inputs"] = inp
        return inp

    # the baseline (`_scrub_inputs`) strips `handler` before the policy callable runs
    composed = _wrap_process_inputs(TracePolicy(process_inputs=process_inputs))
    result = composed({"request": {"x": 1}, "handler": lambda: None})
    assert "handler" not in seen["inputs"]
    assert seen["inputs"] == {"request": {"x": 1}}
    assert result == {"request": {"x": 1}}


def test_wrap_process_inputs_omit_drops_everything() -> None:
    composed = _wrap_process_inputs(TracePolicy(process_inputs=omit_payload))
    assert composed({"request": {"x": 1}, "handler": lambda: None}) == {}
