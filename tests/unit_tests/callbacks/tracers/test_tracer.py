"""Test Tracer classes."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pytest
from freezegun import freeze_time
from pydantic import BaseModel

from langchain.callbacks.tracers.base import (
    BaseTracer,
    ChainRun,
    LLMRun,
    SharedTracer,
    ToolRun,
    Tracer,
    TracerException,
    TracerSession,
)
from langchain.schema import AgentAction, LLMResult

TEST_SESSION_ID = "test_session_id"


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
        session_id=TEST_SESSION_ID,
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
                session_id=TEST_SESSION_ID,
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
                        session_id=TEST_SESSION_ID,
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
                session_id=TEST_SESSION_ID,
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

    def _persist_session(self, session: TracerSession) -> None:
        """Persist a tracing session."""

        session.id = TEST_SESSION_ID


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

    def _persist_session(self, session: TracerSession) -> None:
        """Persist a tracing session."""

        session.id = TEST_SESSION_ID

    def remove_runs(self):
        """Remove all runs."""

        with self._lock:
            self.runs = []


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
            session_id=TEST_SESSION_ID,
        )
    )

    session = tracer.new_session()
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    assert session.child_runs == [tracer.compare_run]


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_session() -> None:
    """Test tracer on an LLM run without a session."""

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
            session_id=TEST_SESSION_ID,
        )
    )

    with pytest.raises(TracerException):
        tracer.on_llm_start(serialized={}, prompts=[])


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""

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
            session_id=TEST_SESSION_ID,
        )
    )

    session = tracer.new_session()
    with pytest.raises(TracerException):
        tracer.on_llm_end(response=LLMResult([[]]))
    assert session.child_runs == []


@freeze_time("2023-01-01")
def test_tracer_multiple_llm_runs() -> None:
    """Test the tracer with multiple runs."""

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
            session_id=TEST_SESSION_ID,
        )
    )

    num_runs = 10
    session = tracer.new_session()
    for _ in range(num_runs):
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_end(response=LLMResult([[]]))
    assert len(session.child_runs) == num_runs
    assert all(run == tracer.compare_run for run in session.child_runs)


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""

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
            session_id=TEST_SESSION_ID,
        )
    )

    session = tracer.new_session()
    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_end(outputs={})
    assert session.child_runs == [tracer.compare_run]


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
            session_id=TEST_SESSION_ID,
        )
    )

    session = tracer.new_session()
    tracer.on_tool_start(
        serialized={}, action=AgentAction(tool="action", tool_input="test", log="")
    )
    tracer.on_tool_end("test")
    assert session.child_runs == [tracer.compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""

    tracer = FakeTracer(compare_run=_get_compare_run())
    session = tracer.new_session()
    _perform_nested_run(tracer)
    assert session.child_runs == [tracer.compare_run]


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    session = tracer.new_session()
    tracer.remove_runs()
    _perform_nested_run(tracer)
    assert tracer.runs == [_get_compare_run()]
    assert session.child_runs == tracer.runs


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run_multithreaded() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    session = tracer.new_session()
    tracer.remove_runs()
    threads = []
    num_threads = 10
    for _ in range(num_threads):
        thread = threading.Thread(target=_perform_nested_run, args=(tracer,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    assert tracer.runs == [_get_compare_run()] * num_threads
    assert session.child_runs == tracer.runs
