"""Test Tracer classes."""

from langchain.callbacks.tracers.base import Tracer, SharedTracer, LLMRun, ChainRun, ToolRun

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from langchain.schema import AgentAction, LLMResult
from freezegun import freeze_time
from datetime import datetime


class FakeTracer(Tracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self, compare_run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Initialize the tracer."""

        super().__init__()
        self.compare_run = compare_run

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


@freeze_time("2023-01-01")
def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""

    tracer = FakeTracer(compare_run=LLMRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        prompts=[],
        response=LLMResult([[]])
    ))

    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test traceron a Chain run."""

    tracer = FakeTracer(compare_run=ChainRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        inputs={},
        outputs={},
    ))

    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_end(outputs={})


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""

    tracer = FakeTracer(compare_run=ToolRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        tool_input="test",
        output="test",
        action="action"
    ))

    tracer.on_tool_start(serialized={}, action=AgentAction(tool="action", tool_input="test", log=""))
    tracer.on_tool_end("test")


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""

    tracer = FakeTracer(compare_run=ChainRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        inputs={},
        outputs={},
        child_runs=[
            ToolRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                serialized={},
                tool_input="test",
                output="test",
                action="action",
                child_runs=[
                    LLMRun(
                        id=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=3,
                        serialized={},
                        prompts=[],
                        response=LLMResult([[]])
                    )
                ]
            ),
            LLMRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                serialized={},
                prompts=[],
                response=LLMResult([[]])
            )
        ]
    ))

    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_tool_start(serialized={}, action=AgentAction(tool="action", tool_input="test", log=""))
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    tracer.on_tool_end("test")
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    tracer.on_chain_end(outputs={})
