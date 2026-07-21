"""End-to-end tests for `AgentMiddleware.tracing` (input recording + hidden spans)."""

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
    """Record the inputs and tags each chain run reports on `on_chain_start`, by name."""

    def __init__(self) -> None:
        self.inputs_by_name: dict[str, Any] = {}
        self.tags_by_name: dict[str, list[str]] = {}

    def on_chain_start(self, serialized: dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name is not None:
            self.inputs_by_name.setdefault(name, inputs)
            self.tags_by_name.setdefault(name, kwargs.get("tags") or [])


def _run(middleware: AgentMiddleware) -> _CaptureInputs:
    """Run a one-shot agent with the middleware and return the captured trace data."""
    agent = create_agent(model=FakeToolCallingModel(), middleware=[middleware])
    capture = _CaptureInputs()
    agent.invoke({"messages": [HumanMessage("hi")]}, {"callbacks": [capture]})
    return capture


def _recorded_before_model_inputs(middleware: AgentMiddleware) -> Any:
    """Run the agent and return what the middleware's before_model node recorded."""
    return _run(middleware).inputs_by_name[f"{middleware.name}.before_model"]


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


def test_tracing_hides_hook_spans_by_default() -> None:
    class Default(AgentMiddleware):
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    class Shown(AgentMiddleware):
        tracing: TraceConfig = {"hidden": False}  # noqa: RUF012

        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
            return None

    # hidden defaults to True: hook spans are tagged so LangSmith omits them from the tree
    assert "langsmith:hidden_middleware" in _run(Default()).tags_by_name["Default.before_model"]
    # tracing={"hidden": False} opts back in to showing the span
    assert "langsmith:hidden_middleware" not in _run(Shown()).tags_by_name["Shown.before_model"]
