import threading
import time
import unittest.mock
from typing import Any
from uuid import UUID

from langsmith import Client

from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.schemas import Run
from langchain.schema.output import LLMResult


def test_example_id_assignment_threadsafe() -> None:
    """Test that example assigned at callback start/end is honored."""
    example_ids = {}

    def mock_create_run(**kwargs: Any) -> Any:
        example_ids[kwargs.get("id")] = kwargs.get("reference_example_id")
        return unittest.mock.MagicMock()

    client = unittest.mock.MagicMock(spec=Client)
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
        example_id_1 = UUID("57e42c57-8c79-4d9f-8765-bf6cd3a98055")
        tracer.example_id = example_id_1
        tracer.on_llm_start({"name": "example_1"}, ["foo"], run_id=run_id_1)
        tracer.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id_1)
        example_id_2 = UUID("4f31216e-7c26-4027-a5fd-0bbf9ace17dc")
        tracer.example_id = example_id_2
        tracer.on_llm_start({"name": "example_2"}, ["foo"], run_id=run_id_2)
        tracer.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=run_id_2)
        tracer.example_id = None
        expected_example_ids = {
            run_id_1: example_id_1,
            run_id_2: example_id_2,
        }
        tracer.wait_for_futures()
        assert example_ids == expected_example_ids


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
