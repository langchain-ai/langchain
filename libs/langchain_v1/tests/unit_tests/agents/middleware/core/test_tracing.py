"""End-to-end tests for `AgentMiddleware.tracing` hook-input recording."""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware import AgentMiddleware, TraceConfig
from langchain.agents.middleware.types import AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel


class _CaptureInputs(BaseCallbackHandler):
    """Record the inputs each chain run reports on `on_chain_start`, by run name."""

    def __init__(self) -> None:
        self.inputs_by_name: dict[str, Any] = {}

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: Any, **kwargs: Any
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name is not None:
            self.inputs_by_name.setdefault(name, inputs)


def _recorded_before_model_inputs(middleware: AgentMiddleware) -> Any:
    """Run the agent and return what the middleware's before_model node recorded."""
    agent = create_agent(model=FakeToolCallingModel(), middleware=[middleware])
    capture = _CaptureInputs()
    agent.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [capture]})
    return capture.inputs_by_name[f"{middleware.name}.before_model"]


def test_tracing_omits_hook_inputs_by_default() -> None:
    class Default(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    # no `tracing` config -> the before_model node records an empty payload
    assert _recorded_before_model_inputs(Default()) == {}


def test_tracing_records_hook_inputs_when_enabled() -> None:
    class Debug(AgentMiddleware):
        tracing: TraceConfig = {"inputs": True}  # noqa: RUF012

        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    # tracing={"inputs": True} -> the real state (incl. messages) is recorded
    recorded = _recorded_before_model_inputs(Debug())
    assert "messages" in recorded
    assert [m.content for m in recorded["messages"]] == ["hi"]
