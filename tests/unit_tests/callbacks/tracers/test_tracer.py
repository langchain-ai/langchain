"""Test Tracer classes."""

from langchain.callbacks.tracers.base import Tracer, SharedTracer, LLMRun, ChainRun, ToolRun

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class FakeTracer(Tracer, BaseModel):
    """Fake tracer that records LangChain execution."""

    compare_run: Union[LLMRun, ChainRun, ToolRun]

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        assert run == self.compare_run

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""

        parent_run.child_runs.append(child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""

        return None


def test_tracer() -> None:
    """Test tracer."""

    tracer = FakeTracer(compare_run=LLMRun())

    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult())

    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_end(outputs={})
    tracer.on_chain_error(error=Exception())

    tracer.on_tool_start(serialized={}, action=AgentAction())
    tracer.on_tool_end(output="")
    tracer.on_tool_error(error=Exception())

