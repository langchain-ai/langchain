"""Test Tracer classes."""
from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

import pytest
from freezegun import freeze_time

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.base import BaseTracer, TracerException
from langchain.callbacks.tracers.schemas import Run
from langchain.schema import LLMResult


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.runs.append(run)


@freeze_time("2023-01-01")
def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""
    uuid = uuid4()
    compare_run = Run(
        id=uuid,
        parent_run_id=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "llm"},
        inputs={"prompts": []},
        outputs=LLMResult(generations=[[]]),
        error=None,
        run_type="llm",
    )
    tracer = FakeTracer()

    tracer.on_llm_start(serialized={"name": "llm"}, prompts=[], run_id=uuid)
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chat_model_run() -> None:
    """Test tracer on a Chat Model run."""
    uuid = uuid4()
    compare_run = Run(
        id=str(uuid),
        name="chat_model",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "chat_model"},
        inputs=dict(prompts=[""]),
        outputs=LLMResult(generations=[[]]),
        error=None,
        run_type="llm",
    )
    tracer = FakeTracer()
    manager = CallbackManager(handlers=[tracer])
    run_manager = manager.on_chat_model_start(
        serialized={"name": "chat_model"}, messages=[[]], run_id=uuid
    )
    run_manager.on_llm_end(response=LLMResult(generations=[[]]))
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""
    tracer = FakeTracer()

    with pytest.raises(TracerException):
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid4())


@freeze_time("2023-01-01")
def test_tracer_multiple_llm_runs() -> None:
    """Test the tracer with multiple runs."""
    uuid = uuid4()
    compare_run = Run(
        id=uuid,
        name="llm",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "llm"},
        inputs=dict(prompts=[]),
        outputs=LLMResult(generations=[[]]),
        error=None,
        run_type="llm",
    )
    tracer = FakeTracer()

    num_runs = 10
    for _ in range(num_runs):
        tracer.on_llm_start(serialized={"name": "llm"}, prompts=[], run_id=uuid)
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""
    uuid = uuid4()
    compare_run = Run(
        id=str(uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        error=None,
        run_type="chain",
    )
    tracer = FakeTracer()

    tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    tracer.on_chain_end(outputs={}, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""
    uuid = uuid4()
    compare_run = Run(
        id=str(uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "tool"},
        inputs={"input": "test"},
        outputs={"output": "test"},
        error=None,
        run_type="tool",
    )
    tracer = FakeTracer()
    tracer.on_tool_start(serialized={"name": "tool"}, input_str="test", run_id=uuid)
    tracer.on_tool_end("test", run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""
    tracer = FakeTracer()

    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    for _ in range(10):
        tracer.on_chain_start(
            serialized={"name": "chain"}, inputs={}, run_id=chain_uuid
        )
        tracer.on_tool_start(
            serialized={"name": "tool"},
            input_str="test",
            run_id=tool_uuid,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_start(
            serialized={"name": "llm"},
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=tool_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        tracer.on_tool_end("test", run_id=tool_uuid)
        tracer.on_llm_start(
            serialized={"name": "llm"},
            prompts=[],
            run_id=llm_uuid2,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        tracer.on_chain_end(outputs={}, run_id=chain_uuid)

    compare_run = Run(
        id=str(chain_uuid),
        error=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=4,
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        run_type="chain",
        child_runs=[
            Run(
                id=tool_uuid,
                parent_run_id=chain_uuid,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                child_execution_order=3,
                serialized={"name": "tool"},
                inputs=dict(input="test"),
                outputs=dict(output="test"),
                error=None,
                run_type="tool",
                child_runs=[
                    Run(
                        id=str(llm_uuid1),
                        parent_run_id=str(tool_uuid),
                        error=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=3,
                        child_execution_order=3,
                        serialized={"name": "llm"},
                        inputs=dict(prompts=[]),
                        outputs=LLMResult(generations=[[]]),
                        run_type="llm",
                    )
                ],
            ),
            Run(
                id=str(llm_uuid2),
                parent_run_id=str(chain_uuid),
                error=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=4,
                serialized={"name": "llm"},
                inputs=dict(prompts=[]),
                outputs=LLMResult(generations=[[]]),
                run_type="llm",
            ),
        ],
    )
    assert tracer.runs[0] == compare_run
    assert tracer.runs == [compare_run] * 10


@freeze_time("2023-01-01")
def test_tracer_llm_run_on_error() -> None:
    """Test tracer on an LLM run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "llm"},
        inputs=dict(prompts=[]),
        outputs=None,
        error=repr(exception),
        run_type="llm",
    )
    tracer = FakeTracer()

    tracer.on_llm_start(serialized={"name": "llm"}, prompts=[], run_id=uuid)
    tracer.on_llm_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "chain"},
        inputs={},
        outputs=None,
        error=repr(exception),
        run_type="chain",
    )
    tracer = FakeTracer()

    tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    tracer.on_chain_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "tool"},
        inputs=dict(input="test"),
        outputs=None,
        action="{'name': 'tool'}",
        error=repr(exception),
        run_type="tool",
    )
    tracer = FakeTracer()

    tracer.on_tool_start(serialized={"name": "tool"}, input_str="test", run_id=uuid)
    tracer.on_tool_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_runs_on_error() -> None:
    """Test tracer on a nested run with an error."""
    exception = Exception("test")

    tracer = FakeTracer()
    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    llm_uuid3 = uuid4()

    for _ in range(3):
        tracer.on_chain_start(
            serialized={"name": "chain"}, inputs={}, run_id=chain_uuid
        )
        tracer.on_llm_start(
            serialized={"name": "llm"},
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        tracer.on_llm_start(
            serialized={"name": "llm"},
            prompts=[],
            run_id=llm_uuid2,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        tracer.on_tool_start(
            serialized={"name": "tool"},
            input_str="test",
            run_id=tool_uuid,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_start(
            serialized={"name": "llm"},
            prompts=[],
            run_id=llm_uuid3,
            parent_run_id=tool_uuid,
        )
        tracer.on_llm_error(exception, run_id=llm_uuid3)
        tracer.on_tool_error(exception, run_id=tool_uuid)
        tracer.on_chain_error(exception, run_id=chain_uuid)

    compare_run = Run(
        id=str(chain_uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=5,
        serialized={"name": "chain"},
        error=repr(exception),
        inputs={},
        outputs=None,
        run_type="chain",
        child_runs=[
            Run(
                id=str(llm_uuid1),
                parent_run_id=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                child_execution_order=2,
                serialized={"name": "llm"},
                error=None,
                inputs=dict(prompts=[]),
                outputs=LLMResult(generations=[[]], llm_output=None),
                run_type="llm",
            ),
            Run(
                id=str(llm_uuid2),
                parent_run_id=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=3,
                child_execution_order=3,
                serialized={"name": "llm"},
                error=None,
                inputs=dict(prompts=[]),
                outputs=LLMResult(generations=[[]], llm_output=None),
                run_type="llm",
            ),
            Run(
                id=str(tool_uuid),
                parent_run_id=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=5,
                serialized={"name": "tool"},
                error=repr(exception),
                inputs=dict(input="test"),
                outputs=None,
                action="{'name': 'tool'}",
                child_runs=[
                    Run(
                        id=str(llm_uuid3),
                        parent_run_id=str(tool_uuid),
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=5,
                        child_execution_order=5,
                        serialized={"name": "llm"},
                        error=repr(exception),
                        inputs=dict(prompts=[]),
                        outputs=None,
                        run_type="llm",
                    )
                ],
                run_type="tool",
            ),
        ],
    )
    assert tracer.runs == [compare_run] * 3
