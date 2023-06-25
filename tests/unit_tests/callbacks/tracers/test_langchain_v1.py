"""Test Tracer classes."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Union
from uuid import uuid4

import pytest
from freezegun import freeze_time

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.base import BaseTracer, TracerException
from langchain.callbacks.tracers.langchain_v1 import (
    ChainRun,
    LangChainTracerV1,
    LLMRun,
    ToolRun,
    TracerSessionV1,
)
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum, TracerSessionV1Base
from langchain.schema import HumanMessage, LLMResult

TEST_SESSION_ID = 2023

SERIALIZED = {"id": ["llm"]}
SERIALIZED_CHAT = {"id": ["chat_model"]}


def load_session(session_name: str) -> TracerSessionV1:
    """Load a tracing session."""
    return TracerSessionV1(
        id=TEST_SESSION_ID, name=session_name, start_time=datetime.utcnow()
    )


def new_session(name: Optional[str] = None) -> TracerSessionV1:
    """Create a new tracing session."""
    return TracerSessionV1(
        id=TEST_SESSION_ID, name=name or "default", start_time=datetime.utcnow()
    )


def _persist_session(session: TracerSessionV1Base) -> TracerSessionV1:
    """Persist a tracing session."""
    return TracerSessionV1(**{**session.dict(), "id": TEST_SESSION_ID})


def load_default_session() -> TracerSessionV1:
    """Load a tracing session."""
    return TracerSessionV1(
        id=TEST_SESSION_ID, name="default", start_time=datetime.utcnow()
    )


@pytest.fixture
def lang_chain_tracer_v1(monkeypatch: pytest.MonkeyPatch) -> LangChainTracerV1:
    monkeypatch.setenv("LANGCHAIN_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://test-endpoint.com")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "foo")
    tracer = LangChainTracerV1()
    return tracer


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Union[LLMRun, ChainRun, ToolRun]] = []

    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        """Persist a run."""
        if isinstance(run, Run):
            with pytest.MonkeyPatch().context() as m:
                m.setenv("LANGCHAIN_TENANT_ID", "test-tenant-id")
                m.setenv("LANGCHAIN_ENDPOINT", "http://test-endpoint.com")
                m.setenv("LANGCHAIN_API_KEY", "foo")
                tracer = LangChainTracerV1()
                tracer.load_default_session = load_default_session  # type: ignore
                run = tracer._convert_to_v1_run(run)
        self.runs.append(run)

    def _persist_session(self, session: TracerSessionV1Base) -> TracerSessionV1:
        """Persist a tracing session."""
        return _persist_session(session)

    def new_session(self, name: Optional[str] = None) -> TracerSessionV1:
        """Create a new tracing session."""
        return new_session(name)

    def load_session(self, session_name: str) -> TracerSessionV1:
        """Load a tracing session."""
        return load_session(session_name)

    def load_default_session(self) -> TracerSessionV1:
        """Load a tracing session."""
        return load_default_session()


@freeze_time("2023-01-01")
def test_tracer_llm_run() -> None:
    """Test tracer on an LLM run."""
    uuid = uuid4()
    compare_run = LLMRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized=SERIALIZED,
        prompts=[],
        response=LLMResult(generations=[[]]),
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chat_model_run() -> None:
    """Test tracer on a Chat Model run."""
    tracer = FakeTracer()

    tracer.new_session()
    manager = CallbackManager(handlers=[tracer])
    run_managers = manager.on_chat_model_start(
        serialized=SERIALIZED_CHAT, messages=[[HumanMessage(content="")]]
    )
    compare_run = LLMRun(
        uuid=str(run_managers[0].run_id),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized=SERIALIZED_CHAT,
        prompts=["Human: "],
        response=LLMResult(generations=[[]]),
        session_id=TEST_SESSION_ID,
        error=None,
    )
    for run_manager in run_managers:
        run_manager.on_llm_end(response=LLMResult(generations=[[]]))
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_llm_run_errors_no_start() -> None:
    """Test tracer on an LLM run without a start."""
    tracer = FakeTracer()

    tracer.new_session()
    with pytest.raises(TracerException):
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid4())


@freeze_time("2023-01-01")
def test_tracer_multiple_llm_runs() -> None:
    """Test the tracer with multiple runs."""
    uuid = uuid4()
    compare_run = LLMRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized=SERIALIZED,
        prompts=[],
        response=LLMResult(generations=[[]]),
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    num_runs = 10
    for _ in range(num_runs):
        tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
        tracer.on_llm_end(response=LLMResult(generations=[[]]), run_id=uuid)

    assert tracer.runs == [compare_run] * num_runs


@freeze_time("2023-01-01")
def test_tracer_chain_run() -> None:
    """Test tracer on a Chain run."""
    uuid = uuid4()
    compare_run = ChainRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    tracer.on_chain_end(outputs={}, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run() -> None:
    """Test tracer on a Tool run."""
    uuid = uuid4()
    compare_run = ToolRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "tool"},
        tool_input="test",
        output="test",
        action="{'name': 'tool'}",
        session_id=TEST_SESSION_ID,
        error=None,
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_tool_start(serialized={"name": "tool"}, input_str="test", run_id=uuid)
    tracer.on_tool_end("test", run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_run() -> None:
    """Test tracer on a nested run."""
    tracer = FakeTracer()
    tracer.new_session()

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

    compare_run = ChainRun(
        uuid=str(chain_uuid),
        error=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=4,
        serialized={"name": "chain"},
        inputs={},
        outputs={},
        session_id=TEST_SESSION_ID,
        child_chain_runs=[],
        child_tool_runs=[
            ToolRun(
                uuid=str(tool_uuid),
                parent_uuid=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                child_execution_order=3,
                serialized={"name": "tool"},
                tool_input="test",
                output="test",
                action="{'name': 'tool'}",
                session_id=TEST_SESSION_ID,
                error=None,
                child_chain_runs=[],
                child_tool_runs=[],
                child_llm_runs=[
                    LLMRun(
                        uuid=str(llm_uuid1),
                        parent_uuid=str(tool_uuid),
                        error=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=3,
                        child_execution_order=3,
                        serialized=SERIALIZED,
                        prompts=[],
                        response=LLMResult(generations=[[]]),
                        session_id=TEST_SESSION_ID,
                    )
                ],
            ),
        ],
        child_llm_runs=[
            LLMRun(
                uuid=str(llm_uuid2),
                parent_uuid=str(chain_uuid),
                error=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=4,
                serialized=SERIALIZED,
                prompts=[],
                response=LLMResult(generations=[[]]),
                session_id=TEST_SESSION_ID,
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

    compare_run = LLMRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized=SERIALIZED,
        prompts=[],
        response=None,
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_llm_start(serialized=SERIALIZED, prompts=[], run_id=uuid)
    tracer.on_llm_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_chain_run_on_error() -> None:
    """Test tracer on a Chain run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = ChainRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "chain"},
        inputs={},
        outputs=None,
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_chain_start(serialized={"name": "chain"}, inputs={}, run_id=uuid)
    tracer.on_chain_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_tool_run_on_error() -> None:
    """Test tracer on a Tool run with an error."""
    exception = Exception("test")
    uuid = uuid4()

    compare_run = ToolRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
        serialized={"name": "tool"},
        tool_input="test",
        output=None,
        action="{'name': 'tool'}",
        session_id=TEST_SESSION_ID,
        error=repr(exception),
    )
    tracer = FakeTracer()

    tracer.new_session()
    tracer.on_tool_start(serialized={"name": "tool"}, input_str="test", run_id=uuid)
    tracer.on_tool_error(exception, run_id=uuid)
    assert tracer.runs == [compare_run]


@freeze_time("2023-01-01")
def test_tracer_nested_runs_on_error() -> None:
    """Test tracer on a nested run with an error."""
    exception = Exception("test")

    tracer = FakeTracer()
    tracer.new_session()
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

    compare_run = ChainRun(
        uuid=str(chain_uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=5,
        serialized={"name": "chain"},
        session_id=TEST_SESSION_ID,
        error=repr(exception),
        inputs={},
        outputs=None,
        child_llm_runs=[
            LLMRun(
                uuid=str(llm_uuid1),
                parent_uuid=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=2,
                child_execution_order=2,
                serialized=SERIALIZED,
                session_id=TEST_SESSION_ID,
                error=None,
                prompts=[],
                response=LLMResult(generations=[[]], llm_output=None),
            ),
            LLMRun(
                uuid=str(llm_uuid2),
                parent_uuid=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=3,
                child_execution_order=3,
                serialized=SERIALIZED,
                session_id=TEST_SESSION_ID,
                error=None,
                prompts=[],
                response=LLMResult(generations=[[]], llm_output=None),
            ),
        ],
        child_chain_runs=[],
        child_tool_runs=[
            ToolRun(
                uuid=str(tool_uuid),
                parent_uuid=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=5,
                serialized={"name": "tool"},
                session_id=TEST_SESSION_ID,
                error=repr(exception),
                tool_input="test",
                output=None,
                action="{'name': 'tool'}",
                child_llm_runs=[
                    LLMRun(
                        uuid=str(llm_uuid3),
                        parent_uuid=str(tool_uuid),
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=5,
                        child_execution_order=5,
                        serialized=SERIALIZED,
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


@pytest.fixture
def sample_tracer_session_v1() -> TracerSessionV1:
    return TracerSessionV1(id=2, name="Sample session")


@freeze_time("2023-01-01")
def test_convert_run(
    lang_chain_tracer_v1: LangChainTracerV1,
    sample_tracer_session_v1: TracerSessionV1,
) -> None:
    """Test converting a run to a V1 run."""
    llm_run = Run(
        id="57a08cc4-73d2-4236-8370-549099d07fad",
        name="llm_run",
        execution_order=1,
        child_execution_order=1,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        session_id=TEST_SESSION_ID,
        inputs={"prompts": []},
        outputs=LLMResult(generations=[[]]).dict(),
        serialized={},
        extra={},
        run_type=RunTypeEnum.llm,
    )
    chain_run = Run(
        id="57a08cc4-73d2-4236-8371-549099d07fad",
        name="chain_run",
        execution_order=1,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        child_execution_order=1,
        serialized={},
        inputs={},
        outputs={},
        child_runs=[llm_run],
        extra={},
        run_type=RunTypeEnum.chain,
    )

    tool_run = Run(
        id="57a08cc4-73d2-4236-8372-549099d07fad",
        name="tool_run",
        execution_order=1,
        child_execution_order=1,
        inputs={"input": "test"},
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        outputs=None,
        serialized={},
        child_runs=[],
        extra={},
        run_type=RunTypeEnum.tool,
    )

    expected_llm_run = LLMRun(
        uuid="57a08cc4-73d2-4236-8370-549099d07fad",
        name="llm_run",
        execution_order=1,
        child_execution_order=1,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        session_id=2,
        prompts=[],
        response=LLMResult(generations=[[]]),
        serialized={},
        extra={},
    )

    expected_chain_run = ChainRun(
        uuid="57a08cc4-73d2-4236-8371-549099d07fad",
        name="chain_run",
        execution_order=1,
        child_execution_order=1,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        session_id=2,
        serialized={},
        inputs={},
        outputs={},
        child_llm_runs=[expected_llm_run],
        child_chain_runs=[],
        child_tool_runs=[],
        extra={},
    )
    expected_tool_run = ToolRun(
        uuid="57a08cc4-73d2-4236-8372-549099d07fad",
        name="tool_run",
        execution_order=1,
        child_execution_order=1,
        session_id=2,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        tool_input="test",
        action="{}",
        serialized={},
        child_llm_runs=[],
        child_chain_runs=[],
        child_tool_runs=[],
        extra={},
    )
    lang_chain_tracer_v1.session = sample_tracer_session_v1
    converted_llm_run = lang_chain_tracer_v1._convert_to_v1_run(llm_run)
    converted_chain_run = lang_chain_tracer_v1._convert_to_v1_run(chain_run)
    converted_tool_run = lang_chain_tracer_v1._convert_to_v1_run(tool_run)

    assert isinstance(converted_llm_run, LLMRun)
    assert isinstance(converted_chain_run, ChainRun)
    assert isinstance(converted_tool_run, ToolRun)
    assert converted_llm_run == expected_llm_run
    assert converted_tool_run == expected_tool_run
    assert converted_chain_run == expected_chain_run
