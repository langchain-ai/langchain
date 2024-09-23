import threading
import time
import unittest
import unittest.mock
import uuid
from typing import Any
from uuid import UUID

import pytest
from langsmith import Client
from langsmith.run_trees import RunTree
from langsmith.utils import get_env_var, get_tracer_project

from langchain_core.outputs import LLMResult
from langchain_core.tracers.langchain import LangChainTracer
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
    parent = RunTree(name="parent", inputs={"input": "foo"}, _client=client)
    run_id = uuid.uuid4()
    tracer = LangChainTracer(client=client)
    tracer.order_map[parent.id] = (parent.trace_id, parent.dotted_order)
    tracer.run_map[str(parent.id)] = parent  # type: ignore
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


class LangChainProjectNameTest(unittest.TestCase):
    """
    Test that the project name is set correctly for runs.
    """

    class SetProperTracerProjectTestCase:
        def __init__(
            self, test_name: str, envvars: dict[str, str], expected_project_name: str
        ):
            self.test_name = test_name
            self.envvars = envvars
            self.expected_project_name = expected_project_name

    def test_correct_get_tracer_project(self) -> None:
        cases = [
            self.SetProperTracerProjectTestCase(
                test_name="default to 'default' when no project provided",
                envvars={},
                expected_project_name="default",
            ),
            self.SetProperTracerProjectTestCase(
                test_name="use session_name for legacy tracers",
                envvars={"LANGCHAIN_SESSION": "old_timey_session"},
                expected_project_name="old_timey_session",
            ),
            self.SetProperTracerProjectTestCase(
                test_name="use LANGCHAIN_PROJECT over SESSION_NAME",
                envvars={
                    "LANGCHAIN_SESSION": "old_timey_session",
                    "LANGCHAIN_PROJECT": "modern_session",
                },
                expected_project_name="modern_session",
            ),
        ]

        for case in cases:
            get_env_var.cache_clear()
            get_tracer_project.cache_clear()
            with self.subTest(msg=case.test_name):
                with pytest.MonkeyPatch.context() as mp:
                    for k, v in case.envvars.items():
                        mp.setenv(k, v)

                    client = unittest.mock.MagicMock(spec=Client)
                    tracer = LangChainTracer(client=client)
                    projects = []

                    def mock_create_run(**kwargs: Any) -> Any:
                        projects.append(kwargs.get("project_name"))  # noqa: B023
                        return unittest.mock.MagicMock()

                    client.create_run = mock_create_run

                    tracer.on_llm_start(
                        {"name": "example_1"},
                        ["foo"],
                        run_id=UUID("9d878ab3-e5ca-4218-aef6-44cbdc90160a"),
                    )
                    tracer.wait_for_futures()
                    assert projects == [case.expected_project_name]
