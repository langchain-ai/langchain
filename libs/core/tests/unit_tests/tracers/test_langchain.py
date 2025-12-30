import threading
import time
import unittest.mock
import uuid
from typing import Any
from uuid import UUID

import pytest
from langsmith import Client
from langsmith.run_trees import RunTree
from langsmith.utils import get_env_var, get_tracer_project

from langchain_core.outputs import LLMResult
from langchain_core.tracers.langchain import (
    LangChainTracer,
    _get_usage_metadata_from_generations,
)
from langchain_core.tracers.schemas import Run


def test_example_id_assignment_threadsafe() -> None:
    """Test that example assigned at callback start/end is honored."""
    example_ids = {}

    def mock_create_run(**kwargs: Any) -> Any:
        example_ids[kwargs.get("id")] = kwargs.get("reference_example_id")
        return unittest.mock.MagicMock()

    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    client.create_run = mock_create_run
    tracer = LangChainTracer(client=client)
    old_persist_run_single = tracer._persist_run_single

    def new_persist_run_single(run: Run) -> None:
        time.sleep(0.01)
        old_persist_run_single(run)

    with unittest.mock.patch.object(
        tracer, "_persist_run_single", new=new_persist_run_single
    ):
        run_id_1 = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
        run_id_2 = UUID("f1f9fa53-8b2f-4742-bdbc-38215f7bd1e1")
        run_id_3 = UUID("f1f9fa53-8b2f-4742-bdbc-38215f7cd1e1")
        example_id_1 = UUID("57e42c57-8c79-4d9f-8765-bf6cd3a98055")
        tracer.example_id = example_id_1
        tracer.on_llm_start({"name": "example_1"}, ["foo"], run_id=run_id_1)
        tracer.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id_1)
        example_id_2 = UUID("4f31216e-7c26-4027-a5fd-0bbf9ace17dc")
        tracer.example_id = example_id_2
        tracer.on_llm_start({"name": "example_2"}, ["foo"], run_id=run_id_2)
        tracer.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id_2)
        tracer.example_id = None
        tracer.on_chain_start(
            {"name": "no_examples"}, {"inputs": (i for i in range(10))}, run_id=run_id_3
        )
        tracer.on_chain_error(ValueError("Foo bar"), run_id=run_id_3)
        expected_example_ids = {
            run_id_1: example_id_1,
            run_id_2: example_id_2,
            run_id_3: None,
        }
        tracer.wait_for_futures()
        assert example_ids == expected_example_ids


def test_tracer_with_run_tree_parent() -> None:
    mock_session = unittest.mock.MagicMock()
    client = Client(session=mock_session, api_key="test")
    parent = RunTree(name="parent", inputs={"input": "foo"}, ls_client=client)
    run_id = uuid.uuid4()
    tracer = LangChainTracer(client=client)
    tracer.order_map[parent.id] = (parent.trace_id, parent.dotted_order)
    tracer.run_map[str(parent.id)] = parent
    tracer.on_chain_start(
        {"name": "child"}, {"input": "bar"}, run_id=run_id, parent_run_id=parent.id
    )
    tracer.on_chain_end({}, run_id=run_id)
    assert parent.child_runs
    assert len(parent.child_runs) == 1
    assert parent.child_runs[0].id == run_id
    assert parent.child_runs[0].trace_id == parent.id
    assert parent.child_runs[0].parent_run_id == parent.id


def test_log_lock() -> None:
    """Test that example assigned at callback start/end is honored."""
    client = unittest.mock.MagicMock(spec=Client)
    tracer = LangChainTracer(client=client)

    with unittest.mock.patch.object(tracer, "_persist_run_single", new=lambda _: _):
        run_id_1 = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
        lock = threading.Lock()
        tracer.on_chain_start({"name": "example_1"}, {"input": lock}, run_id=run_id_1)
        tracer.on_chain_end({}, run_id=run_id_1)
        tracer.wait_for_futures()


@pytest.mark.parametrize(
    ("envvars", "expected_project_name"),
    [
        (
            {},
            "default",
        ),
        (
            {"LANGCHAIN_SESSION": "old_timey_session"},
            "old_timey_session",
        ),
        (
            {
                "LANGCHAIN_SESSION": "old_timey_session",
                "LANGCHAIN_PROJECT": "modern_session",
            },
            "modern_session",
        ),
    ],
    ids=[
        "default to 'default' when no project provided",
        "use session_name for legacy tracers",
        "use LANGCHAIN_PROJECT over SESSION_NAME",
    ],
)
def test_correct_get_tracer_project(
    envvars: dict[str, str], expected_project_name: str
) -> None:
    if hasattr(get_env_var, "cache_clear"):
        get_env_var.cache_clear()
    if hasattr(get_tracer_project, "cache_clear"):
        get_tracer_project.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        for k, v in envvars.items():
            mp.setenv(k, v)

        client = unittest.mock.MagicMock(spec=Client)
        tracer = LangChainTracer(client=client)
        projects = []

        def mock_create_run(**kwargs: Any) -> Any:
            projects.append(kwargs.get("session_name"))
            return unittest.mock.MagicMock()

        client.create_run = mock_create_run

        tracer.on_llm_start(
            {"name": "example_1"},
            ["foo"],
            run_id=UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a"),
        )
        tracer.wait_for_futures()
        assert projects == [expected_project_name]


@pytest.mark.parametrize(
    ("generations", "expected"),
    [
        # Returns usage_metadata when present
        (
            [
                [
                    {
                        "text": "Hello!",
                        "message": {
                            "content": "Hello!",
                            "usage_metadata": {
                                "input_tokens": 10,
                                "output_tokens": 20,
                                "total_tokens": 30,
                            },
                        },
                    }
                ]
            ],
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        ),
        # Returns None when no usage_metadata
        ([[{"text": "Hello!", "message": {"content": "Hello!"}}]], None),
        # Returns None when no message
        ([[{"text": "Hello!"}]], None),
        # Returns None for empty generations
        ([], None),
        ([[]], None),
        # Aggregates usage_metadata across multiple generations
        (
            [
                [
                    {
                        "text": "First",
                        "message": {
                            "content": "First",
                            "usage_metadata": {
                                "input_tokens": 5,
                                "output_tokens": 10,
                                "total_tokens": 15,
                            },
                        },
                    },
                    {
                        "text": "Second",
                        "message": {
                            "content": "Second",
                            "usage_metadata": {
                                "input_tokens": 50,
                                "output_tokens": 100,
                                "total_tokens": 150,
                            },
                        },
                    },
                ]
            ],
            {"input_tokens": 55, "output_tokens": 110, "total_tokens": 165},
        ),
        # Finds usage_metadata across multiple batches
        (
            [
                [{"text": "No message here"}],
                [
                    {
                        "text": "Has message",
                        "message": {
                            "content": "Has message",
                            "usage_metadata": {
                                "input_tokens": 10,
                                "output_tokens": 20,
                                "total_tokens": 30,
                            },
                        },
                    }
                ],
            ],
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        ),
    ],
    ids=[
        "returns_usage_metadata_when_present",
        "returns_none_when_no_usage_metadata",
        "returns_none_when_no_message",
        "returns_none_for_empty_list",
        "returns_none_for_empty_batch",
        "aggregates_across_multiple_generations",
        "finds_across_multiple_batches",
    ],
)
def test_get_usage_metadata_from_generations(
    generations: list[list[dict[str, Any]]], expected: dict[str, Any] | None
) -> None:
    """Test `_get_usage_metadata_from_generations` utility function."""
    result = _get_usage_metadata_from_generations(generations)
    assert result == expected


def test_on_llm_end_stores_usage_metadata_in_run_extra() -> None:
    """Test that `usage_metadata` is stored in `run.extra.metadata` on llm end."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_llm_start({"name": "test_llm"}, ["foo"], run_id=run_id)

    run = tracer.run_map[str(run_id)]
    usage_metadata = {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}
    run.outputs = {
        "generations": [
            [
                {
                    "text": "Hello!",
                    "message": {"content": "Hello!", "usage_metadata": usage_metadata},
                }
            ]
        ]
    }

    captured_run = None

    def capture_run(r: Run) -> None:
        nonlocal captured_run
        captured_run = r

    with unittest.mock.patch.object(tracer, "_update_run_single", capture_run):
        tracer._on_llm_end(run)

    assert captured_run is not None
    assert "metadata" in captured_run.extra
    assert captured_run.extra["metadata"]["usage_metadata"] == usage_metadata


def test_on_llm_end_no_usage_metadata_when_not_present() -> None:
    """Test that no `usage_metadata` is added when not present in outputs."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_llm_start({"name": "test_llm"}, ["foo"], run_id=run_id)

    run = tracer.run_map[str(run_id)]
    run.outputs = {
        "generations": [[{"text": "Hello!", "message": {"content": "Hello!"}}]]
    }

    captured_run = None

    def capture_run(r: Run) -> None:
        nonlocal captured_run
        captured_run = r

    with unittest.mock.patch.object(tracer, "_update_run_single", capture_run):
        tracer._on_llm_end(run)

    assert captured_run is not None
    extra_metadata = captured_run.extra.get("metadata", {})
    assert "usage_metadata" not in extra_metadata


def test_on_llm_end_preserves_existing_metadata() -> None:
    """Test that existing metadata is preserved when adding `usage_metadata`."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_llm_start(
        {"name": "test_llm"},
        ["foo"],
        run_id=run_id,
        metadata={"existing_key": "existing_value"},
    )

    run = tracer.run_map[str(run_id)]
    usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    run.outputs = {
        "generations": [
            [
                {
                    "text": "Hello!",
                    "message": {"content": "Hello!", "usage_metadata": usage_metadata},
                }
            ]
        ]
    }

    captured_run = None

    def capture_run(r: Run) -> None:
        nonlocal captured_run
        captured_run = r

    with unittest.mock.patch.object(tracer, "_update_run_single", capture_run):
        tracer._on_llm_end(run)

    assert captured_run is not None
    assert "metadata" in captured_run.extra
    assert captured_run.extra["metadata"]["usage_metadata"] == usage_metadata
    assert captured_run.extra["metadata"]["existing_key"] == "existing_value"


def test_on_chain_start_skips_persist_when_defers_inputs() -> None:
    """Test that `_on_chain_start` skips persist when `defers_inputs` is set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    # Pass defers_inputs=True to signal deferred inputs
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": ""},
        run_id=run_id,
        defers_inputs=True,
    )

    run = tracer.run_map[str(run_id)]

    persist_called = False

    def mock_persist() -> None:
        nonlocal persist_called
        persist_called = True

    with unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist):
        tracer._on_chain_start(run)

    # Should NOT call persist when defers_inputs is set
    assert not persist_called


def test_on_chain_start_persists_when_not_defers_inputs() -> None:
    """Test that `_on_chain_start` persists when `defers_inputs` is not set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    # Normal chain start without defers_inputs
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": "hello"},
        run_id=run_id,
    )

    run = tracer.run_map[str(run_id)]

    persist_called = False

    def mock_persist(_: Any) -> None:
        nonlocal persist_called
        persist_called = True

    with unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist):
        tracer._on_chain_start(run)

    # Should call persist when defers_inputs is not set
    assert persist_called


def test_on_chain_end_persists_when_defers_inputs() -> None:
    """Test that `_on_chain_end` calls persist (POST) when `defers_inputs` is set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": ""},
        run_id=run_id,
        defers_inputs=True,
    )

    run = tracer.run_map[str(run_id)]
    run.outputs = {"output": "result"}
    run.inputs = {"input": "realized input"}

    persist_called = False
    update_called = False

    def mock_persist(_: Any) -> None:
        nonlocal persist_called
        persist_called = True

    def mock_update(_: Any) -> None:
        nonlocal update_called
        update_called = True

    with (
        unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist),
        unittest.mock.patch.object(tracer, "_update_run_single", mock_update),
    ):
        tracer._on_chain_end(run)

    # Should call persist (POST), not update (PATCH) for deferred inputs
    assert persist_called
    assert not update_called


def test_on_chain_end_updates_when_not_defers_inputs() -> None:
    """Tests `_on_chain_end` calls update (PATCH) when `defers_inputs` is not set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": "hello"},
        run_id=run_id,
    )

    run = tracer.run_map[str(run_id)]
    run.outputs = {"output": "result"}

    persist_called = False
    update_called = False

    def mock_persist(_: Any) -> None:
        nonlocal persist_called
        persist_called = True

    def mock_update(_: Any) -> None:
        nonlocal update_called
        update_called = True

    with (
        unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist),
        unittest.mock.patch.object(tracer, "_update_run_single", mock_update),
    ):
        tracer._on_chain_end(run)

    # Should call update (PATCH), not persist (POST) for normal inputs
    assert not persist_called
    assert update_called


def test_on_chain_error_persists_when_defers_inputs() -> None:
    """Test that `_on_chain_error` calls persist (POST) when `defers_inputs` is set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": ""},
        run_id=run_id,
        defers_inputs=True,
    )

    run = tracer.run_map[str(run_id)]
    run.error = "Test error"
    run.inputs = {"input": "realized input"}

    persist_called = False
    update_called = False

    def mock_persist(_: Any) -> None:
        nonlocal persist_called
        persist_called = True

    def mock_update(_: Any) -> None:
        nonlocal update_called
        update_called = True

    with (
        unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist),
        unittest.mock.patch.object(tracer, "_update_run_single", mock_update),
    ):
        tracer._on_chain_error(run)

    # Should call persist (POST), not update (PATCH) for deferred inputs
    assert persist_called
    assert not update_called


def test_on_chain_error_updates_when_not_defers_inputs() -> None:
    """Tests `_on_chain_error` calls update (PATCH) when `defers_inputs` is not set."""
    client = unittest.mock.MagicMock(spec=Client)
    client.tracing_queue = None
    tracer = LangChainTracer(client=client)

    run_id = UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a")
    tracer.on_chain_start(
        {"name": "test_chain"},
        {"input": "hello"},
        run_id=run_id,
    )

    run = tracer.run_map[str(run_id)]
    run.error = "Test error"

    persist_called = False
    update_called = False

    def mock_persist(_: Any) -> None:
        nonlocal persist_called
        persist_called = True

    def mock_update(_: Any) -> None:
        nonlocal update_called
        update_called = True

    with (
        unittest.mock.patch.object(tracer, "_persist_run_single", mock_persist),
        unittest.mock.patch.object(tracer, "_update_run_single", mock_update),
    ):
        tracer._on_chain_error(run)

    # Should call update (PATCH), not persist (POST) for normal inputs
    assert not persist_called
    assert update_called
