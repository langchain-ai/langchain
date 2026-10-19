import asyncio
import gc
import weakref
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langsmith import Client, get_tracing_context, traceable, tracing_context
from langsmith.evaluation._arunner import _aforward
from langsmith.evaluation._runner import _forward
from langsmith.schemas import Example

from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import LangChainTracer

pytestmark = pytest.mark.skipif(
    "retain_child_runs" not in get_tracing_context(),
    reason="Requires the LangSmith SDK retention context",
)


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(spec=Client)
    client.tracing_queue = None
    return client


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_evaluation_preserves_nested_langchain_steps(
    client: MagicMock, *, asynchronous: bool
) -> None:
    tracer = LangChainTracer(client=client)

    @RunnableLambda
    def inner(inputs: dict[str, int]) -> dict[str, int]:
        return {"answer": inputs["value"] + 1}

    @RunnableLambda
    def outer(inputs: dict[str, int]) -> dict[str, int]:
        return inner.invoke(inputs)

    @traceable
    def target(inputs: dict[str, int]) -> dict[str, int]:
        return outer.invoke(inputs, config={"callbacks": [tracer]})

    @traceable
    async def async_target(inputs: dict[str, int]) -> dict[str, int]:
        return await outer.ainvoke(inputs, config={"callbacks": [tracer]})

    example = Example(id=uuid4(), dataset_id=uuid4(), inputs={"value": 1})
    with tracing_context(client=client, retain_child_runs=False):
        if asynchronous:
            result = await _aforward(async_target, example, "retention", {}, client)
        else:
            result = _forward(
                target, example, "retention", {}, client, upload_results=False
            )
        assert get_tracing_context()["retain_child_runs"] is False
    root = result["run"]
    assert root.outputs == {"answer": 2}
    assert root.child_runs is not None
    outer_run = root.child_runs[0]
    assert outer_run.name == "outer"
    assert outer_run.child_runs is not None
    inner_run = outer_run.child_runs[0]
    assert inner_run.name == "inner"
    assert inner_run.outputs == {"answer": 2}
    assert tracer.run_map == {}


async def test_reused_tracer_isolates_observers_from_ordinary_tracing(
    client: MagicMock,
) -> None:
    tracer = LangChainTracer(client=client)

    @traceable
    async def target() -> bool:
        root_id, child_id = uuid4(), uuid4()
        root = tracer.on_chain_start({}, {}, run_id=root_id)
        child = tracer.on_chain_start({}, {}, run_id=child_id, parent_run_id=root_id)
        child_ref = weakref.ref(child)
        tracer.on_chain_end({}, run_id=child_id)
        del child
        client.reset_mock()
        await asyncio.sleep(0)
        gc.collect()
        retained = child_ref() is not None
        assert bool(root.child_runs) == retained
        tracer.on_chain_end({}, run_id=root_id)
        return retained

    with tracing_context(enabled="local", client=client, retain_child_runs=False):
        observed, ordinary = await asyncio.gather(
            target(langsmith_extra={"on_end": lambda _run: None}), target()
        )
        assert observed is True
        assert ordinary is False
        assert get_tracing_context()["retain_child_runs"] is False
    assert tracer.run_map == {}
    assert tracer.retain_child_runs is None


def test_context_opt_in_preserves_direct_inspection(client: MagicMock) -> None:
    tracer = LangChainTracer(client=client)
    with tracing_context(retain_child_runs=True, client=client, enabled="local"):
        root_id, child_id = uuid4(), uuid4()
        root = tracer.on_chain_start({}, {}, run_id=root_id)
        tracer.on_chain_start({}, {}, run_id=child_id, parent_run_id=root_id)
        tracer.on_chain_end({}, run_id=child_id)
        tracer.on_chain_end({}, run_id=root_id)
    assert [child.id for child in root.child_runs] == [child_id]
