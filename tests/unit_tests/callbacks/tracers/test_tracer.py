"""Test Tracer classes."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from freezegun import freeze_time
from pydantic import BaseModel

from langchain.callbacks.tracers.base import (
    BaseTracer,
    ChainRun,
    LLMRun,
    SharedTracer,
    ToolRun,
    Tracer,
)
from langchain.schema import AgentAction, LLMResult


@freeze_time("2023-01-01")
def _get_compare_run() -> Union[LLMRun, ChainRun, ToolRun]:
    return ChainRun(
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
                        response=LLMResult([[]]),
                    )
                ],
            ),
            LLMRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                serialized={},
                prompts=[],
                response=LLMResult([[]]),
            ),
        ],
    )


def _perform_nested_run(tracer: BaseTracer):
    """Perform a nested run."""

    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_tool_start(
        serialized={}, action=AgentAction(tool="action", tool_input="test", log="")
    )
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    tracer.on_tool_end("test")
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    tracer.on_chain_end(outputs={})


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


class FakeSharedTracer(SharedTracer):
    """Fake shared tracer that records LangChain execution."""

    runs = []

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

        with self._lock:
            self.runs.append(run)

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

    tracer = FakeTracer(
        compare_run=LLMRun(
            id=None,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            extra={},
            execution_order=1,
            serialized={},
            prompts=[],
            response=LLMResult([[]]),
        )
    )

    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test traceron a Chain run."""

    tracer = FakeTracer(
        compare_run=ChainRun(
            id=None,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            extra={},
            execution_order=1,
            serialized={},
            inputs={},
            outputs={},
        )
    )

    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_end(outputs={})


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""

    tracer = FakeTracer(
        compare_run=ToolRun(
            id=None,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            extra={},
            execution_order=1,
            serialized={},
            tool_input="test",
            output="test",
            action="action",
        )
    )

    tracer.on_tool_start(
        serialized={}, action=AgentAction(tool="action", tool_input="test", log="")
    )
    tracer.on_tool_end("test")


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""

    tracer = FakeTracer(compare_run=_get_compare_run())
    _perform_nested_run(tracer)


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    _perform_nested_run(tracer)
    assert tracer.runs == [_get_compare_run()]


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run_multithreaded() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    threads = []
    num_threads = 10
    for _ in range(num_threads):
        thread = threading.Thread(target=_perform_nested_run, args=(tracer,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    assert tracer.runs == [_get_compare_run()] * num_threads
