"""Test Tracer classes."""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Tuple, Union
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

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
from langchain.callbacks.tracers.langchain import LangChainTracerV2
from langchain.callbacks.tracers.schemas import (
    RunCreate,
    TracerSessionBase,
    TracerSessionV2,
    TracerSessionV2Create,
)
from langchain.schema import LLMResult

TEST_SESSION_ID = 2023


def load_session(session_name: str) -> TracerSession:
    """Load a tracing session."""
    return TracerSession(id=1, name=session_name, start_time=datetime.utcnow())


def _persist_session(session: TracerSessionBase) -> TracerSession:
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

    def _persist_session(self, session: TracerSessionBase) -> TracerSession:
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
    uuid = uuid4()
    compare_run = LLMRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
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
    uuid = uuid4()
    compare_run = ChainRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
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
    uuid = uuid4()
    compare_run = ToolRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
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

    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    for _ in range(10):
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

    compare_run = ChainRun(
        uuid=str(chain_uuid),
        error=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=4,
        serialized={},
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
                        uuid=str(llm_uuid1),
                        parent_uuid=str(tool_uuid),
                        error=None,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=3,
                        child_execution_order=3,
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
                uuid=str(llm_uuid2),
                parent_uuid=str(chain_uuid),
                error=None,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=4,
                serialized={},
                prompts=[],
                response=LLMResult(generations=[[]]),
                session_id=TEST_SESSION_ID,
            ),
        ],
    )
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
    uuid = uuid4()

    compare_run = ChainRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
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
    uuid = uuid4()

    compare_run = ToolRun(
        uuid=str(uuid),
        parent_uuid=None,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=1,
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
    chain_uuid = uuid4()
    tool_uuid = uuid4()
    llm_uuid1 = uuid4()
    llm_uuid2 = uuid4()
    llm_uuid3 = uuid4()

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
        uuid=str(chain_uuid),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        extra={},
        execution_order=1,
        child_execution_order=5,
        serialized={},
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
                serialized={},
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
                uuid=str(tool_uuid),
                parent_uuid=str(chain_uuid),
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                extra={},
                execution_order=4,
                child_execution_order=5,
                serialized={},
                session_id=TEST_SESSION_ID,
                error=repr(exception),
                tool_input="test",
                output=None,
                action="{}",
                child_llm_runs=[
                    LLMRun(
                        uuid=str(llm_uuid3),
                        parent_uuid=str(tool_uuid),
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        extra={},
                        execution_order=5,
                        child_execution_order=5,
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


_SESSION_ID = UUID("4fbf7c55-2727-4711-8964-d821ed4d4e2a")
_TENANT_ID = UUID("57a08cc4-73d2-4236-8378-549099d07fad")


@pytest.fixture
def lang_chain_tracer_v2(monkeypatch: pytest.MonkeyPatch) -> LangChainTracerV2:
    monkeypatch.setenv("LANGCHAIN_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://test-endpoint.com")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "foo")
    tracer = LangChainTracerV2()
    return tracer


# Mock a sample TracerSessionV2 object
@pytest.fixture
def sample_tracer_session_v2() -> TracerSessionV2:
    return TracerSessionV2(id=_SESSION_ID, name="Sample session", tenant_id=_TENANT_ID)


# Mock a sample LLMRun, ChainRun, and ToolRun objects
@pytest.fixture
def sample_runs() -> Tuple[LLMRun, ChainRun, ToolRun]:
    llm_run = LLMRun(
        uuid="57a08cc4-73d2-4236-8370-549099d07fad",
        name="llm_run",
        execution_order=1,
        child_execution_order=1,
        session_id=1,
        prompts=[],
        response=LLMResult(generations=[[]]),
        serialized={},
        extra={},
    )
    chain_run = ChainRun(
        uuid="57a08cc4-73d2-4236-8371-549099d07fad",
        name="chain_run",
        execution_order=1,
        child_execution_order=1,
        session_id=1,
        serialized={},
        inputs={},
        outputs=None,
        child_llm_runs=[llm_run],
        child_chain_runs=[],
        child_tool_runs=[],
        extra={},
    )
    tool_run = ToolRun(
        uuid="57a08cc4-73d2-4236-8372-549099d07fad",
        name="tool_run",
        execution_order=1,
        child_execution_order=1,
        session_id=1,
        tool_input="test",
        action="{}",
        serialized={},
        child_llm_runs=[],
        child_chain_runs=[],
        child_tool_runs=[],
        extra={},
    )
    return llm_run, chain_run, tool_run


def test_get_default_query_params(lang_chain_tracer_v2: LangChainTracerV2) -> None:
    expected = {"tenant_id": "test-tenant-id"}
    result = lang_chain_tracer_v2._get_default_query_params()
    assert result == expected


@patch("langchain.callbacks.tracers.langchain.requests.get")
def test_load_session(
    mock_requests_get: Mock,
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
) -> None:
    """Test that load_session method returns a TracerSessionV2 object."""
    mock_requests_get.return_value.json.return_value = [sample_tracer_session_v2.dict()]
    result = lang_chain_tracer_v2.load_session("test-session-name")
    mock_requests_get.assert_called_with(
        "http://test-endpoint.com/sessions",
        headers={"Content-Type": "application/json", "x-api-key": "foo"},
        params={"tenant_id": "test-tenant-id", "name": "test-session-name"},
    )
    assert result == sample_tracer_session_v2


def test_convert_run(
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
    sample_runs: Tuple[LLMRun, ChainRun, ToolRun],
) -> None:
    llm_run, chain_run, tool_run = sample_runs
    lang_chain_tracer_v2.session = sample_tracer_session_v2
    converted_llm_run = lang_chain_tracer_v2._convert_run(llm_run)
    converted_chain_run = lang_chain_tracer_v2._convert_run(chain_run)
    converted_tool_run = lang_chain_tracer_v2._convert_run(tool_run)

    assert isinstance(converted_llm_run, RunCreate)
    assert isinstance(converted_chain_run, RunCreate)
    assert isinstance(converted_tool_run, RunCreate)


def test_persist_run(
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
    sample_runs: Tuple[LLMRun, ChainRun, ToolRun],
) -> None:
    """Test that persist_run method calls requests.post once per method call."""
    with patch("langchain.callbacks.tracers.langchain.requests.post") as post, patch(
        "langchain.callbacks.tracers.langchain.requests.get"
    ) as get:
        post.return_value.raise_for_status.return_value = None
        lang_chain_tracer_v2.session = sample_tracer_session_v2
        llm_run, chain_run, tool_run = sample_runs
        lang_chain_tracer_v2._persist_run(llm_run)
        lang_chain_tracer_v2._persist_run(chain_run)
        lang_chain_tracer_v2._persist_run(tool_run)

        assert post.call_count == 3
        assert get.call_count == 0


def test_persist_run_with_example_id(
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
    sample_runs: Tuple[LLMRun, ChainRun, ToolRun],
) -> None:
    """Test the example ID is assigned only to the parent run and not the children."""
    example_id = uuid4()
    llm_run, chain_run, tool_run = sample_runs
    chain_run.child_tool_runs = [tool_run]
    tool_run.child_llm_runs = [llm_run]
    with patch("langchain.callbacks.tracers.langchain.requests.post") as post, patch(
        "langchain.callbacks.tracers.langchain.requests.get"
    ) as get:
        post.return_value.raise_for_status.return_value = None
        lang_chain_tracer_v2.session = sample_tracer_session_v2
        lang_chain_tracer_v2.example_id = example_id
        lang_chain_tracer_v2._persist_run(chain_run)

        assert post.call_count == 1
        assert get.call_count == 0
        posted_data = json.loads(post.call_args[1]["data"])
        assert posted_data["id"] == chain_run.uuid
        assert posted_data["reference_example_id"] == str(example_id)

        def assert_child_run_no_example_id(run: dict) -> None:
            assert not run.get("reference_example_id")
            for child_run in run.get("child_runs", []):
                assert_child_run_no_example_id(child_run)

        for child_run in posted_data["child_runs"]:
            assert_child_run_no_example_id(child_run)


def test_get_session_create(lang_chain_tracer_v2: LangChainTracerV2) -> None:
    """Test creating the 'SessionCreate' object."""
    lang_chain_tracer_v2.tenant_id = str(_TENANT_ID)
    session_create = lang_chain_tracer_v2._get_session_create(name="test")
    assert isinstance(session_create, TracerSessionV2Create)
    assert session_create.name == "test"
    assert session_create.tenant_id == _TENANT_ID


@patch("langchain.callbacks.tracers.langchain.requests.post")
def test_persist_session(
    mock_requests_post: Mock,
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
) -> None:
    """Test persist_session returns a TracerSessionV2 with the updated ID."""
    session_create = TracerSessionV2Create(**sample_tracer_session_v2.dict())
    new_id = str(uuid4())
    mock_requests_post.return_value.json.return_value = {"id": new_id}
    result = lang_chain_tracer_v2._persist_session(session_create)
    assert isinstance(result, TracerSessionV2)
    res = sample_tracer_session_v2.dict()
    res["id"] = UUID(new_id)
    assert result.dict() == res


@patch("langchain.callbacks.tracers.langchain.LangChainTracerV2.load_session")
def test_load_default_session(
    mock_load_session: Mock,
    lang_chain_tracer_v2: LangChainTracerV2,
    sample_tracer_session_v2: TracerSessionV2,
) -> None:
    """Test load_default_session attempts to load with the default name."""
    mock_load_session.return_value = sample_tracer_session_v2
    result = lang_chain_tracer_v2.load_default_session()
    assert result == sample_tracer_session_v2
    mock_load_session.assert_called_with("default")
