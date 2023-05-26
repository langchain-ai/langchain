"""Test Tracer classes."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Tuple
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from freezegun import freeze_time

from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.schemas import Run, RunTypeEnum, TracerSession
from langchain.schema import LLMResult

_SESSION_ID = UUID("4fbf7c55-2727-4711-8964-d821ed4d4e2a")
_TENANT_ID = UUID("57a08cc4-73d2-4236-8378-549099d07fad")


@pytest.fixture
def lang_chain_tracer_v2(monkeypatch: pytest.MonkeyPatch) -> LangChainTracer:
    monkeypatch.setenv("LANGCHAIN_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://test-endpoint.com")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "foo")
    tracer = LangChainTracer()
    return tracer


# Mock a sample TracerSession object
@pytest.fixture
def sample_tracer_session_v2() -> TracerSession:
    return TracerSession(id=_SESSION_ID, name="Sample session", tenant_id=_TENANT_ID)


@freeze_time("2023-01-01")
@pytest.fixture
def sample_runs() -> Tuple[Run, Run, Run]:
    llm_run = Run(
        id="57a08cc4-73d2-4236-8370-549099d07fad",
        name="llm_run",
        execution_order=1,
        child_execution_order=1,
        parent_run_id="57a08cc4-73d2-4236-8371-549099d07fad",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        session_id=1,
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
    return llm_run, chain_run, tool_run


def test_persist_run(
    lang_chain_tracer_v2: LangChainTracer,
    sample_tracer_session_v2: TracerSession,
    sample_runs: Tuple[Run, Run, Run],
) -> None:
    """Test that persist_run method calls requests.post once per method call."""
    with patch("langchain.callbacks.tracers.langchain.requests.post") as post, patch(
        "langchain.callbacks.tracers.langchain.requests.get"
    ) as get:
        post.return_value.raise_for_status.return_value = None
        lang_chain_tracer_v2.session = sample_tracer_session_v2
        for run in sample_runs:
            lang_chain_tracer_v2.run_map[str(run.id)] = run
        for run in sample_runs:
            lang_chain_tracer_v2._end_trace(run)

        assert post.call_count == 3
        assert get.call_count == 0


def test_persist_run_with_example_id(
    lang_chain_tracer_v2: LangChainTracer,
    sample_tracer_session_v2: TracerSession,
    sample_runs: Tuple[Run, Run, Run],
) -> None:
    """Test the example ID is assigned only to the parent run and not the children."""
    example_id = uuid4()
    llm_run, chain_run, tool_run = sample_runs
    chain_run.child_runs = [tool_run]
    tool_run.child_runs = [llm_run]
    with patch("langchain.callbacks.tracers.langchain.requests.post") as post, patch(
        "langchain.callbacks.tracers.langchain.requests.get"
    ) as get:
        post.return_value.raise_for_status.return_value = None
        lang_chain_tracer_v2.session = sample_tracer_session_v2
        lang_chain_tracer_v2.example_id = example_id
        lang_chain_tracer_v2._persist_run(chain_run)

        assert post.call_count == 3
        assert get.call_count == 0
        posted_data = [
            json.loads(call_args[1]["data"]) for call_args in post.call_args_list
        ]
        assert posted_data[0]["id"] == str(chain_run.id)
        assert posted_data[0]["reference_example_id"] == str(example_id)
        assert posted_data[1]["id"] == str(tool_run.id)
        assert not posted_data[1].get("reference_example_id")
        assert posted_data[2]["id"] == str(llm_run.id)
        assert not posted_data[2].get("reference_example_id")
