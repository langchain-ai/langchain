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

    def __init__(self) -> None:
        """Initialize the tracer."""

        super().__init__()
        self.runs = []

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""

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

    def load_session(self, session_id: Union[int, str]) -> TracerSession:
        """Load a tracing session."""

        return TracerSession(id=session_id, start_time=datetime.utcnow())


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

    def load_session(self, session_id: Union[int, str]) -> TracerSession:
        """Load a tracing session."""

        return TracerSession(id=session_id, start_time=datetime.utcnow())


@freeze_time("2023-01-01")
def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""

    compare_run = LLMRun(
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
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult([[]]))
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_session() -> None:
    """Test tracer on an LLM run without a session."""

    tracer = FakeTracer()

    with pytest.raises(TracerException):
        tracer.on_llm_start(serialized={}, prompts=[])


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""

    tracer = FakeTracer()

    tracer.new_session()
    with pytest.raises(TracerException):
        tracer.on_llm_end(response=LLMResult([[]]))


@freeze_time("2023-01-01")
def test_tracer_multiple_llm_runs() -> None:
    """Test the tracer with multiple runs."""

    compare_run = LLMRun(
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
    tracer = FakeTracer()

    tracer.new_session()
    num_runs = 10
    for _ in range(num_runs):
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_end(response=LLMResult([[]]))

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""

    compare_run = ChainRun(
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
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_end(outputs={})
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""

    compare_run = ToolRun(
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
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_tool_start(
        serialized={}, action=AgentAction(tool="action", tool_input="test", log="")
    )
    tracer.on_tool_end("test")
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""

    tracer = FakeTracer()
    tracer.new_session()
    _perform_nested_run(tracer)
    assert tracer.runs == [_get_compare_run()]


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    tracer.new_session()
    tracer.remove_runs()
    _perform_nested_run(tracer)
    assert tracer.runs == [_get_compare_run()]


@freeze_time("2023-01-01")
def test_shared_tracer_nested_run_multithreaded() -> None:
    """Test shared tracer on a nested run."""

    tracer = FakeSharedTracer()
    tracer.remove_runs()
    tracer.new_session()
    threads = []
    num_threads = 10
    for _ in range(num_threads):
        thread = threading.Thread(target=_perform_nested_run, args=(tracer,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    assert tracer.runs == [_get_compare_run()] * num_threads
