"""Test Tracer classes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import langsmith
import pytest
from freezegun import freeze_time
from langsmith import Client, traceable

from langchain_core.callbacks import CallbackManager
from langchain_core.exceptions import TracerException
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import chain as as_runnable
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

SERIALIZED = {"id": ["llm"]}
SERIALIZED_CHAT = {"id": ["chat_model"]}


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.runs.append(run)


def _compare_run_with_error(run: Any, expected_run: Any) -> None:
    if run.child_runs:
        assert len(expected_run.child_runs) == len(run.child_runs)
        for received, expected in zip(
            run.child_runs, expected_run.child_runs, strict=False
        ):
            _compare_run_with_error(received, expected)
    if hasattr(run, "model_dump"):
        received = run.model_dump(exclude={"child_runs"})
    else:
        received = run.dict(exclude={"child_runs"})  # type: ignore[deprecated]
    received_err = received.pop("error")
    if hasattr(expected_run, "model_dump"):
        expected = expected_run.model_dump(exclude={"child_runs"})
    else:
        expected = expected_run.dict(exclude={"child_runs"})  # type: ignore[deprecated]
    expected_err = expected.pop("error")

    assert received == expected
    if expected_err is not None:
        assert received_err is not None
        assert expected_err in received_err
    else:
        assert received_err is None


@freeze_time("2023-01-01")
def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""
    uuid = uuid4()
    compare_run = Run(
        id=uuid,
        parent_run_id=None,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized=SERIALIZED,
        inputs={"prompts": []},
        outputs=LLMResult(generations=[[]]).model_dump(),
        error=None,
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeTracer()

    tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chat_model_run() -> None:
    """Test tracer on a Chat Model run."""
    tracer = FakeTracer()
    manager = CallbackManager(handlers=[tracer])
    run_managers = manager.on_chat_model_start(
        serialized=SERIALIZED_CHAT, messages=[[HumanMessage(content="")]]
    )
    compare_run = Run(
        id=str(run_managers[0].run_id),
        name="chat_model",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized=SERIALIZED_CHAT,
        inputs={"prompts": ["Human: "]},
        outputs=LLMResult(generations=[[]]).model_dump(),
        error=None,
        run_type="llm",
        trace_id=run_managers[0].run_id,
        dotted_order=f"20230101T000000000000Z{run_managers[0].run_id}",
    )
    for run_manager in run_managers:
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
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized=SERIALIZED,
        inputs={"prompts": []},
        outputs=LLMResult(generations=[[]]).model_dump(),
        error=None,
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeTracer()

    num_runs = 10
    for _ in range(num_runs):
        tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""
    uuid = uuid4()
    compare_run = Run(
        id=str(uuid),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        error=None,
        run_type="chain",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
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
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "tool"},
        inputs={"input": "test"},
        outputs={"output": "test"},
        error=None,
        run_type="tool",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
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
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=tool_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        tracer.on_tool_end("test", run_id=tool_uuid)
        tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid2,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        tracer.on_chain_end(outputs={}, run_id=chain_uuid)

    compare_run = Run(
        id=str(chain_uuid),
        error=None,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "end", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        run_type="chain",
        trace_id=chain_uuid,
        dotted_order=f"20230101T000000000000Z{chain_uuid}",
        child_runs=[
            Run(
                id=tool_uuid,
                parent_run_id=chain_uuid,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                events=[
                    {"name": "start", "time": datetime.now(timezone.utc)},
                    {"name": "end", "time": datetime.now(timezone.utc)},
                ],
                extra={},
                serialized={"name": "tool"},
                inputs={"input": "test"},
                outputs={"output": "test"},
                error=None,
                run_type="tool",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{tool_uuid}",
                child_runs=[
                    Run(
                        id=str(llm_uuid1),
                        parent_run_id=str(tool_uuid),
                        error=None,
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                        events=[
                            {"name": "start", "time": datetime.now(timezone.utc)},
                            {"name": "end", "time": datetime.now(timezone.utc)},
                        ],
                        extra={},
                        serialized=SERIALIZED,
                        inputs={"prompts": []},
                        outputs=LLMResult(generations=[[]]).model_dump(),
                        run_type="llm",
                        trace_id=chain_uuid,
                        dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{tool_uuid}.20230101T000000000000Z{llm_uuid1}",
                    )
                ],
            ),
            Run(
                id=str(llm_uuid2),
                parent_run_id=str(chain_uuid),
                error=None,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                events=[
                    {"name": "start", "time": datetime.now(timezone.utc)},
                    {"name": "end", "time": datetime.now(timezone.utc)},
                ],
                extra={},
                serialized=SERIALIZED,
                inputs={"prompts": []},
                outputs=LLMResult(generations=[[]]).model_dump(),
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid2}",
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
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "error", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized=SERIALIZED,
        inputs={"prompts": []},
        outputs=None,
        error=repr(exception),
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeTracer()

    tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    tracer.on_llm_error(exception, run_id=uuid)
    assert len(tracer.runs) == 1
    _compare_run_with_error(tracer.runs[0], compare_run)


@freeze_time("2023-01-01")
def test_tracer_llm_run_on_error_callback() -> None:
    """Test tracer on an LLM run with an error and a callback."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "error", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized=SERIALIZED,
        inputs={"prompts": []},
        outputs=None,
        error=repr(exception),
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )

    class FakeTracerWithLlmErrorCallback(FakeTracer):
        error_run = None

        def _on_llm_error(self, run: Run) -> None:
            self.error_run = run

    tracer = FakeTracerWithLlmErrorCallback()
    tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    tracer.on_llm_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.error_run, compare_run)


@freeze_time("2023-01-01")
def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "error", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "chain"},
        inputs={},
        outputs=None,
        error=repr(exception),
        run_type="chain",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeTracer()

    tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    tracer.on_chain_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.runs[0], compare_run)


@freeze_time("2023-01-01")
def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(
        id=str(uuid),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "error", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "tool"},
        inputs={"input": "test"},
        outputs=None,
        error=repr(exception),
        run_type="tool",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeTracer()

    tracer.on_tool_start(serialized={"name": "tool"}, input_str="test", run_id=uuid)
    tracer.on_tool_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.runs[0], compare_run)


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
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=chain_uuid,
        )
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        tracer.on_llm_start(
            serialized=SERIALIZED,
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
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid3,
            parent_run_id=tool_uuid,
        )
        tracer.on_llm_error(exception, run_id=llm_uuid3)
        tracer.on_tool_error(exception, run_id=tool_uuid)
        tracer.on_chain_error(exception, run_id=chain_uuid)

    compare_run = Run(
        id=str(chain_uuid),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        events=[
            {"name": "start", "time": datetime.now(timezone.utc)},
            {"name": "error", "time": datetime.now(timezone.utc)},
        ],
        extra={},
        serialized={"name": "chain"},
        error=repr(exception),
        inputs={},
        outputs=None,
        run_type="chain",
        trace_id=chain_uuid,
        dotted_order=f"20230101T000000000000Z{chain_uuid}",
        child_runs=[
            Run(
                id=str(llm_uuid1),
                parent_run_id=str(chain_uuid),
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                events=[
                    {"name": "start", "time": datetime.now(timezone.utc)},
                    {"name": "end", "time": datetime.now(timezone.utc)},
                ],
                extra={},
                serialized=SERIALIZED,
                error=None,
                inputs={"prompts": []},
                outputs=LLMResult(generations=[[]], llm_output=None).model_dump(),
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid1}",
            ),
            Run(
                id=str(llm_uuid2),
                parent_run_id=str(chain_uuid),
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                events=[
                    {"name": "start", "time": datetime.now(timezone.utc)},
                    {"name": "end", "time": datetime.now(timezone.utc)},
                ],
                extra={},
                serialized=SERIALIZED,
                error=None,
                inputs={"prompts": []},
                outputs=LLMResult(generations=[[]], llm_output=None).model_dump(),
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid2}",
            ),
            Run(
                id=str(tool_uuid),
                parent_run_id=str(chain_uuid),
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                events=[
                    {"name": "start", "time": datetime.now(timezone.utc)},
                    {"name": "error", "time": datetime.now(timezone.utc)},
                ],
                extra={},
                serialized={"name": "tool"},
                error=repr(exception),
                inputs={"input": "test"},
                outputs=None,
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{tool_uuid}",
                child_runs=[
                    Run(
                        id=str(llm_uuid3),
                        parent_run_id=str(tool_uuid),
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                        events=[
                            {"name": "start", "time": datetime.now(timezone.utc)},
                            {"name": "error", "time": datetime.now(timezone.utc)},
                        ],
                        extra={},
                        serialized=SERIALIZED,
                        error=repr(exception),
                        inputs={"prompts": []},
                        outputs=None,
                        run_type="llm",
                        trace_id=chain_uuid,
                        dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{tool_uuid}.20230101T000000000000Z{llm_uuid3}",
                    )
                ],
                run_type="tool",
            ),
        ],
    )
    assert len(tracer.runs) == 3
    for run in tracer.runs:
        _compare_run_with_error(run, compare_run)


def _get_mock_client() -> Client:
    mock_session = MagicMock()
    return Client(session=mock_session, api_key="test")


def test_traceable_to_tracing() -> None:
    has_children = False

    def _collect_run(run: Any) -> None:
        nonlocal has_children
        has_children = bool(run.child_runs)

    @as_runnable
    def foo(x: int) -> int:
        return x + 1

    @traceable
    def some_parent(a: int, b: int) -> int:
        return foo.invoke(a) + foo.invoke(b)

    mock_client_ = _get_mock_client()
    with langsmith.run_helpers.tracing_context(enabled=True):
        result = some_parent(
            1, 2, langsmith_extra={"client": mock_client_, "on_end": _collect_run}
        )
    assert result == 5
    assert has_children, "Child run not collected"
