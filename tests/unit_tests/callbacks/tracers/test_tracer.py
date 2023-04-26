"""Test Tracer classes."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union
from uuid import uuid4

import pytest
from freezegun import freeze_time

from langchain.callbacks.tracers.base import (
    BaseTracer,
    ChainRun,
    LLMRun,
    ToolRun,
    TracerException,
    TracerSession,
)
from langchain.callbacks.tracers.schemas import TracerSessionCreate
from langchain.schema import LLMResult

TEST_SESSION_ID = 2023


@freeze_time("2023-01-01")
def _get_compare_run() -> Union[LLMRun, ChainRun, ToolRun]:
    return ChainRun(
        uuid="chain_uuid",
        error=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        inputs={},
        outputs={},
        session_id=TEST_SESSION_ID,
        child_chain_runs=[],
        child_tool_runs=[
            ToolRun(
                uuid="tool_uuid",
                parent_uuid="chain_uuid",
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
                child_chain_runs=[],
                child_tool_runs=[],
                child_llm_runs=[
                    LLMRun(
                        uuid="llm_uuid1",
                        parent_uuid="tool_uuid",
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
        ],
        child_llm_runs=[
            LLMRun(
                uuid="llm_uuid2",
                parent_uuid="chain_uuid",
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
    chain_uuid = "chain_uuid"
    tool_uuid = "tool_uuid"
    llm_uuid1 = "llm_uuid1"
    llm_uuid2 = "llm_uuid2"

    tracer.on_chain_start(serialized={}, inputs={}, run_id=chain_uuid)
    tracer.on_tool_start(
        serialized={}, input_str="test", run_id=tool_uuid, parent_run_id=chain_uuid
    )
    tracer.on_llm_start(
        serialized={}, prompts=[], run_id=llm_uuid1, parent_run_id=tool_uuid
    )
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
    tracer.on_tool_end("test", run_id=tool_uuid)
    tracer.on_llm_start(
        serialized={}, prompts=[], run_id=llm_uuid2, parent_run_id=chain_uuid
    )
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
    tracer.on_chain_end(outputs={}, run_id=chain_uuid)


def load_session(session_name: str) -> TracerSession:
    """Load a tracing session."""
    return TracerSession(id=1, name=session_name, start_time=datetime.utcnow())


def _persist_session(session: TracerSessionCreate) -> TracerSession:
    """Persist a tracing session."""
    return TracerSession(id=TEST_SESSION_ID, **session.dict())


def load_default_session() -> TracerSession:
    """Load a tracing session."""
    return TracerSession(id=1, name="default", start_time=datetime.utcnow())


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Union[LLMRun, ChainRun, ToolRun]] = []

    def _persist_run(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        self.runs.append(run)

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
    uuid = str(uuid4())
    compare_run = LLMRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_llm_start(serialized={}, prompts=[], run_id=uuid)
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""
    tracer = FakeTracer()

    tracer.new_session()
    with pytest.raises(TracerException):
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=str(uuid4()))


@freeze_time("2023-01-01")
def test_tracer_multiple_llm_runs() -> None:
    """Test the tracer with multiple runs."""
    uuid = str(uuid4())
    compare_run = LLMRun(
        uuid=uuid,
        parent_uuid=None,
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
        tracer.on_llm_start(serialized={}, prompts=[], run_id=uuid)
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""
    uuid = str(uuid4())
    compare_run = ChainRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_chain_start(serialized={}, inputs={}, run_id=uuid)
    tracer.on_chain_end(outputs={}, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""
    uuid = str(uuid4())
    compare_run = ToolRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_tool_start(serialized={}, input_str="test", run_id=uuid)
    tracer.on_tool_end("test", run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""
    tracer = FakeTracer()
    tracer.new_session()
    [_perform_nested_run(tracer) for _ in range(10)]
    assert tracer.runs == [_get_compare_run()] * 10


@freeze_time("2023-01-01")
def test_tracer_llm_run_on_error() -> None:
    """Test tracer on an LLM run with an error."""
    exception = Exception("test")
    uuid = str(uuid4())

    compare_run = LLMRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_llm_start(serialized={}, prompts=[], run_id=uuid)
    tracer.on_llm_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")
    uuid = str(uuid4())

    compare_run = ChainRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_chain_start(serialized={}, inputs={}, run_id=uuid)
    tracer.on_chain_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")
    uuid = str(uuid4())

    compare_run = ToolRun(
        uuid=uuid,
        parent_uuid=None,
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
    tracer.on_tool_start(serialized={}, input_str="test", run_id=uuid)
    tracer.on_tool_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_runs_on_error() -> None:
    """Test tracer on a nested run with an error."""
    exception = Exception("test")

    tracer = FakeTracer()
    tracer.new_session()
    chain_uuid = "chain_uuid"
    tool_uuid = "tool_uuid"
    llm_uuid1 = "llm_uuid1"
    llm_uuid2 = "llm_uuid2"
    llm_uuid3 = "llm_uuid3"

    for _ in range(3):
        tracer.on_chain_start(serialized={}, inputs={}, run_id=chain_uuid)
        tracer.on_llm_start(
            serialized={}, prompts=[], run_id=llm_uuid1, parent_run_id=chain_uuid
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        tracer.on_llm_start(
            serialized={}, prompts=[], run_id=llm_uuid2, parent_run_id=chain_uuid
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        tracer.on_tool_start(
            serialized={}, input_str="test", run_id=tool_uuid, parent_run_id=chain_uuid
        )
        tracer.on_llm_start(
            serialized={}, prompts=[], run_id=llm_uuid3, parent_run_id=tool_uuid
        )
        tracer.on_llm_error(exception, run_id=llm_uuid3)
        tracer.on_tool_error(exception, run_id=tool_uuid)
        tracer.on_chain_error(exception, run_id=chain_uuid)

    compare_run = ChainRun(
        uuid=chain_uuid,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        serialized={},
        session_id=TEST_SESSION_ID,
        error=repr(exception),
        inputs={},
        outputs=None,
        child_llm_runs=[
            LLMRun(
                uuid=llm_uuid1,
                parent_uuid=chain_uuid,
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
                uuid=llm_uuid2,
                parent_uuid=chain_uuid,
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
        ],
        child_chain_runs=[],
        child_tool_runs=[
            ToolRun(
                uuid=tool_uuid,
                parent_uuid=chain_uuid,
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
                child_llm_runs=[
                    LLMRun(
                        uuid=llm_uuid3,
                        parent_uuid=tool_uuid,
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
                child_chain_runs=[],
                child_tool_runs=[],
            ),
        ],
    )

    assert tracer.runs == [compare_run] * 3
