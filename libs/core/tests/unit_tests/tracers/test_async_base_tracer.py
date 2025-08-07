"""Test Tracer classes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from freezegun import freeze_time

from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.exceptions import TracerException
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.base import AsyncBaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.v1.messages import HumanMessage as HumanMessageV1
from langchain_core.v1.messages import MessageV1

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

SERIALIZED = {"id": ["llm"]}
SERIALIZED_CHAT = {"id": ["chat_model"]}


class FakeAsyncTracer(AsyncBaseTracer):
    """Fake tracer to test async based tracers."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: list[Run] = []

    async def _persist_run(self, run: Run) -> None:
        self.runs.append(run)


def _compare_run_with_error(run: Any, expected_run: Any) -> None:
    if run.child_runs:
        assert len(expected_run.child_runs) == len(run.child_runs)
        for received, expected in zip(run.child_runs, expected_run.child_runs):
            _compare_run_with_error(received, expected)
    received = run.dict(exclude={"child_runs"})
    received_err = received.pop("error")
    expected = expected_run.dict(exclude={"child_runs"})
    expected_err = expected.pop("error")

    assert received == expected
    if expected_err is not None:
        assert received_err is not None
        assert expected_err in received_err
    else:
        assert received_err is None


@freeze_time("2023-01-01")
async def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""
    uuid = uuid4()
    compare_run = Run(  # type: ignore[call-arg]
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
        outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
        error=None,
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeAsyncTracer()

    await tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
async def test_tracer_chat_model_run() -> None:
    """Test tracer on a Chat Model run."""
    tracer = FakeAsyncTracer()
    manager = AsyncCallbackManager(handlers=[tracer])
    messages: list[list[BaseMessage]] = [[HumanMessage(content="")]]
    run_managers = await manager.on_chat_model_start(
        serialized=SERIALIZED_CHAT, messages=messages
    )
    compare_run = Run(
        id=str(run_managers[0].run_id),  # type: ignore[arg-type]
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
        outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
        error=None,
        run_type="llm",
        trace_id=run_managers[0].run_id,
        dotted_order=f"20230101T000000000000Z{run_managers[0].run_id}",
    )
    for run_manager in run_managers:
        await run_manager.on_llm_end(response=LLMResult(generations=[[]]))
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
async def test_tracer_chat_model_run_v1() -> None:
    """Test tracer on a Chat Model run."""
    tracer = FakeAsyncTracer()
    manager = AsyncCallbackManager(handlers=[tracer])
    messages: list[MessageV1] = [HumanMessageV1("")]
    run_managers = await manager.on_chat_model_start(
        serialized=SERIALIZED_CHAT, messages=messages
    )
    compare_run = Run(
        id=str(run_managers[0].run_id),  # type: ignore[arg-type]
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
        outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
        error=None,
        run_type="llm",
        trace_id=run_managers[0].run_id,
        dotted_order=f"20230101T000000000000Z{run_managers[0].run_id}",
    )
    for run_manager in run_managers:
        await run_manager.on_llm_end(response=LLMResult(generations=[[]]))
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
async def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""
    tracer = FakeAsyncTracer()

    with pytest.raises(TracerException):
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid4())


@freeze_time("2023-01-01")
async def test_tracer_multiple_llm_runs() -> None:
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
        outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
        error=None,
        run_type="llm",
        trace_id=uuid,
        dotted_order=f"20230101T000000000000Z{uuid}",
    )
    tracer = FakeAsyncTracer()

    num_runs = 10
    for _ in range(num_runs):
        await tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
async def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""
    uuid = uuid4()
    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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
    tracer = FakeAsyncTracer()

    await tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    await tracer.on_chain_end(outputs={}, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
async def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""
    uuid = uuid4()
    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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
    tracer = FakeAsyncTracer()
    await tracer.on_tool_start(
        serialized={"name": "tool"}, input_str="test", run_id=uuid
    )
    await tracer.on_tool_end("test", run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
async def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""
    tracer = FakeAsyncTracer()

    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    for _ in range(10):
        await tracer.on_chain_start(
            serialized={"name": "chain"}, inputs={}, run_id=chain_uuid
        )
        await tracer.on_tool_start(
            serialized={"name": "tool"},
            input_str="test",
            run_id=tool_uuid,
            parent_run_id=chain_uuid,
        )
        await tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=tool_uuid,
        )
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        await tracer.on_tool_end("test", run_id=tool_uuid)
        await tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid2,
            parent_run_id=chain_uuid,
        )
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        await tracer.on_chain_end(outputs={}, run_id=chain_uuid)

    compare_run = Run(  # type: ignore[call-arg]
        id=str(chain_uuid),  # type: ignore[arg-type]
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
            Run(  # type: ignore[call-arg]
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
                    Run(  # type: ignore[call-arg]
                        id=str(llm_uuid1),  # type: ignore[arg-type]
                        parent_run_id=str(tool_uuid),  # type: ignore[arg-type]
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
                        outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
                        run_type="llm",
                        trace_id=chain_uuid,
                        dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{tool_uuid}.20230101T000000000000Z{llm_uuid1}",
                    )
                ],
            ),
            Run(  # type: ignore[call-arg]
                id=str(llm_uuid2),  # type: ignore[arg-type]
                parent_run_id=str(chain_uuid),  # type: ignore[arg-type]
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
                outputs=LLMResult(generations=[[]]),  # type: ignore[arg-type]
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid2}",
            ),
        ],
    )
    assert tracer.runs[0] == compare_run
    assert tracer.runs == [compare_run] * 10


@freeze_time("2023-01-01")
async def test_tracer_llm_run_on_error() -> None:
    """Test tracer on an LLM run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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
    tracer = FakeAsyncTracer()

    await tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    await tracer.on_llm_error(exception, run_id=uuid)
    assert len(tracer.runs) == 1
    _compare_run_with_error(tracer.runs[0], compare_run)


@freeze_time("2023-01-01")
async def test_tracer_llm_run_on_error_callback() -> None:
    """Test tracer on an LLM run with an error and a callback."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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

    class FakeTracerWithLlmErrorCallback(FakeAsyncTracer):
        error_run = None

        async def _on_llm_error(self, run: Run) -> None:
            self.error_run = run

    tracer = FakeTracerWithLlmErrorCallback()
    await tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    await tracer.on_llm_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.error_run, compare_run)


@freeze_time("2023-01-01")
async def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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
    tracer = FakeAsyncTracer()

    await tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    await tracer.on_chain_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.runs[0], compare_run)


@freeze_time("2023-01-01")
async def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = Run(  # type: ignore[call-arg]
        id=str(uuid),  # type: ignore[arg-type]
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
    tracer = FakeAsyncTracer()

    await tracer.on_tool_start(
        serialized={"name": "tool"}, input_str="test", run_id=uuid
    )
    await tracer.on_tool_error(exception, run_id=uuid)
    _compare_run_with_error(tracer.runs[0], compare_run)


@freeze_time("2023-01-01")
async def test_tracer_nested_runs_on_error() -> None:
    """Test tracer on a nested run with an error."""
    exception = Exception("test")

    tracer = FakeAsyncTracer()
    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    llm_uuid3 = uuid4()

    for _ in range(3):
        await tracer.on_chain_start(
            serialized={"name": "chain"}, inputs={}, run_id=chain_uuid
        )
        await tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid1,
            parent_run_id=chain_uuid,
        )
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid1)
        await tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid2,
            parent_run_id=chain_uuid,
        )
        await tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=llm_uuid2)
        await tracer.on_tool_start(
            serialized={"name": "tool"},
            input_str="test",
            run_id=tool_uuid,
            parent_run_id=chain_uuid,
        )
        await tracer.on_llm_start(
            serialized=SERIALIZED,
            prompts=[],
            run_id=llm_uuid3,
            parent_run_id=tool_uuid,
        )
        await tracer.on_llm_error(exception, run_id=llm_uuid3)
        await tracer.on_tool_error(exception, run_id=tool_uuid)
        await tracer.on_chain_error(exception, run_id=chain_uuid)

    compare_run = Run(  # type: ignore[call-arg]
        id=str(chain_uuid),  # type: ignore[arg-type]
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
            Run(  # type: ignore[call-arg]
                id=str(llm_uuid1),  # type: ignore[arg-type]
                parent_run_id=str(chain_uuid),  # type: ignore[arg-type]
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
                outputs=LLMResult(generations=[[]], llm_output=None),  # type: ignore[arg-type]
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid1}",
            ),
            Run(  # type: ignore[call-arg]
                id=str(llm_uuid2),  # type: ignore[arg-type]
                parent_run_id=str(chain_uuid),  # type: ignore[arg-type]
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
                outputs=LLMResult(generations=[[]], llm_output=None),  # type: ignore[arg-type]
                run_type="llm",
                trace_id=chain_uuid,
                dotted_order=f"20230101T000000000000Z{chain_uuid}.20230101T000000000000Z{llm_uuid2}",
            ),
            Run(  # type: ignore[call-arg]
                id=str(tool_uuid),  # type: ignore[arg-type]
                parent_run_id=str(chain_uuid),  # type: ignore[arg-type]
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
                    Run(  # type: ignore[call-arg]
                        id=str(llm_uuid3),  # type: ignore[arg-type]
                        parent_run_id=str(tool_uuid),  # type: ignore[arg-type]
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
