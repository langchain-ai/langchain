"""Test Tracer classes."""
from __future__ import annotations

import threading
from datetime import datetime
from typing import List, Optional, Union

import pytest
from freezegun import freeze_time

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
from langchain.callbacks.tracers.schemas import TracerSessionCreate
from langchain.schema import LLMResult

TEST_SESSION_ID = 2023


@freeze_time("2023-01-01")
def _get_compare_run() -> Union[LLMRun, ChainRun, ToolRun]:
    return ChainRun(
        id=None,
        error=None,
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
                action="{}",
                session_id=TEST_SESSION_ID,
                error=None,
                child_runs=[
                    LLMRun(
                        id=None,
                        error=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=3,
                        serialized={},
                        prompts=[],
                        response=LLMResult(generations=[[]]),
                        session_id=TEST_SESSION_ID,
                    )
                ],
            ),
            LLMRun(
                id=None,
                error=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                serialized={},
                prompts=[],
                response=LLMResult(generations=[[]]),
                session_id=TEST_SESSION_ID,
            ),
        ],
    )


def _perform_nested_run(tracer: BaseTracer) -> None:
    """Perform a nested run."""
    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_tool_start(serialized={}, input_str="test")
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult(generations=[[]]))
    tracer.on_tool_end("test")
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult(generations=[[]]))
    tracer.on_chain_end(outputs={})


def _add_child_run(
    parent_run: Union[ChainRun, ToolRun],
    child_run: Union[LLMRun, ChainRun, ToolRun],
) -> None:
    """Add child run to a chain run or tool run."""
    parent_run.child_runs.append(child_run)


def _generate_id() -> Optional[Union[int, str]]:
    """Generate an id for a run."""
    return None


def load_session(session_name: str) -> TracerSession:
    """Load a tracing session."""
    return TracerSession(id=1, name=session_name, start_time=datetime.utcnow())


def _persist_session(session: TracerSessionCreate) -> TracerSession:
    """Persist a tracing session."""
    return TracerSession(id=TEST_SESSION_ID, **session.dict())


def load_default_session() -> TracerSession:
    """Load a tracing session."""
    return TracerSession(id=1, name="default", start_time=datetime.utcnow())


class FakeTracer(Tracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Union[LLMRun, ChainRun, ToolRun]] = []

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        self.runs.append(run)

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""
        _add_child_run(parent_run, child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return _generate_id()

    def _persist_session(self, session: TracerSessionCreate) -> TracerSession:
        """Persist a tracing session."""
        return _persist_session(session)

    def load_session(self, session_name: str) -> TracerSession:
        """Load a tracing session."""
        return load_session(session_name)

    def load_default_session(self) -> TracerSession:
        """Load a tracing session."""
        return load_default_session()


class FakeSharedTracer(SharedTracer):
    """Fake shared tracer that records LangChain execution."""

    runs: List[Union[LLMRun, ChainRun, ToolRun]] = []

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        with self._lock:
            self.runs.append(run)

    def remove_runs(self) -> None:
        """Remove all runs."""
        with self._lock:
            self.runs = []

    def _add_child_run(
        self,
        parent_run: Union[ChainRun, ToolRun],
        child_run: Union[LLMRun, ChainRun, ToolRun],
    ) -> None:
        """Add child run to a chain run or tool run."""
        _add_child_run(parent_run, child_run)

    def _generate_id(self) -> Optional[Union[int, str]]:
        """Generate an id for a run."""
        return _generate_id()

    def _persist_session(self, session: TracerSessionCreate) -> TracerSession:
        """Persist a tracing session."""
        return _persist_session(session)

    def load_session(self, session_name: str) -> TracerSession:
        """Load a tracing session."""
        return load_session(session_name)

    def load_default_session(self) -> TracerSession:
        """Load a tracing session."""
        return load_default_session()


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
        response=LLMResult(generations=[[]]),
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_end(response=LLMResult(generations=[[]]))
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
        tracer.on_llm_end(response=LLMResult(generations=[[]]))


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
        response=LLMResult(generations=[[]]),
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    num_runs = 10
    for _ in range(num_runs):
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_end(response=LLMResult(generations=[[]]))

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
        error=None,
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
        action="{}",
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_tool_start(serialized={}, input_str="test")
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
def test_tracer_llm_run_on_error() -> None:
    """Test tracer on an LLM run with an error."""
    exception = Exception("test")

    compare_run = LLMRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        prompts=[],
        response=None,
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_llm_start(serialized={}, prompts=[])
    tracer.on_llm_error(exception)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")

    compare_run = ChainRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        inputs={},
        outputs=None,
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_chain_start(serialized={}, inputs={})
    tracer.on_chain_error(exception)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")

    compare_run = ToolRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        tool_input="test",
        output=None,
        action="{}",
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_tool_start(serialized={}, input_str="test")
    tracer.on_tool_error(exception)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_runs_on_error() -> None:
    """Test tracer on a nested run with an error."""
    exception = Exception("test")

    tracer = FakeTracer()
    tracer.new_session()

    for _ in range(3):
        tracer.on_chain_start(serialized={}, inputs={})
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_end(response=LLMResult(generations=[[]]))
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_end(response=LLMResult(generations=[[]]))
        tracer.on_tool_start(serialized={}, input_str="test")
        tracer.on_llm_start(serialized={}, prompts=[])
        tracer.on_llm_error(exception)
        tracer.on_tool_error(exception)
        tracer.on_chain_error(exception)

    compare_run = ChainRun(
        id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        session_id=TEST_SESSION_ID,
        error=repr(exception),
        inputs={},
        outputs=None,
        child_runs=[
            LLMRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                serialized={},
                session_id=TEST_SESSION_ID,
                error=None,
                prompts=[],
                response=LLMResult(generations=[[]], llm_output=None),
            ),
            LLMRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=3,
                serialized={},
                session_id=TEST_SESSION_ID,
                error=None,
                prompts=[],
                response=LLMResult(generations=[[]], llm_output=None),
            ),
            ToolRun(
                id=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                serialized={},
                session_id=TEST_SESSION_ID,
                error=repr(exception),
                tool_input="test",
                output=None,
                action="{}",
                child_runs=[
                    LLMRun(
                        id=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=5,
                        serialized={},
                        session_id=TEST_SESSION_ID,
                        error=repr(exception),
                        prompts=[],
                        response=None,
                    )
                ],
                child_llm_runs=[],
                child_chain_runs=[],
                child_tool_runs=[],
            ),
        ],
        child_llm_runs=[],
        child_chain_runs=[],
        child_tool_runs=[],
    )

    assert tracer.runs == [compare_run] * 3


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
