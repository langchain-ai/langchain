"""Test the server and client together."""
import asyncio
from asyncio import AbstractEventLoop
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableLambda
from pytest_mock import MockerFixture

from langserve.client import RemoteRunnable
from langserve.server import add_routes


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture()
def app(event_loop: AbstractEventLoop) -> FastAPI:
    """A simple server that wraps a Runnable and exposes it as an API."""

    async def add_one_or_passthrough(
        x: Union[int, HumanMessage]
    ) -> Union[int, HumanMessage]:
        """Add one to int or passthrough."""
        if isinstance(x, int):
            return x + 1
        else:
            return x

    runnable_lambda = RunnableLambda(func=add_one_or_passthrough)
    app = FastAPI()
    try:
        add_routes(app, runnable_lambda)
        yield app
    finally:
        del app


@pytest.fixture()
def client(app: FastAPI) -> RemoteRunnable:
    """Create a FastAPI app that exposes the Runnable as an API."""
    remote_runnable_client = RemoteRunnable(url="http://localhost:9999")
    sync_client = TestClient(app=app)
    remote_runnable_client.sync_client = sync_client
    yield remote_runnable_client
    sync_client.close()


@asynccontextmanager
async def get_async_client(
    server: FastAPI, path: Optional[str] = None
) -> RemoteRunnable:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    remote_runnable_client = RemoteRunnable(url=url)
    async_client = AsyncClient(app=server, base_url=url)
    remote_runnable_client.async_client = async_client
    try:
        yield remote_runnable_client
    finally:
        await async_client.aclose()


@pytest_asyncio.fixture()
async def async_client(app: FastAPI) -> RemoteRunnable:
    """Create a FastAPI app that exposes the Runnable as an API."""
    async with get_async_client(app) as client:
        yield client


def test_server(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    sync_client = TestClient(app=app)

    # Test invoke
    response = sync_client.post("/invoke", json={"input": 1})
    assert response.json() == {"output": 2}

    # Test batch
    response = sync_client.post("/batch", json={"inputs": [1]})
    assert response.json() == {
        "output": [2],
    }

    # TODO(Team): Fix test. Issue with eventloops right now when using sync client
    ## Test stream
    # response = sync_client.post("/stream", json={"input": 1})
    # assert response.text == "event: data\r\ndata: 2\r\n\r\nevent: end\r\n\r\n"


@pytest.mark.asyncio
async def test_server_async(app: FastAPI) -> None:
    """Test the server directly via HTTP requests."""
    async_client = AsyncClient(app=app, base_url="http://localhost:9999")

    # Test invoke
    response = await async_client.post("/invoke", json={"input": 1})
    assert response.json() == {"output": 2}

    # Test batch
    response = await async_client.post("/batch", json={"inputs": [1]})
    assert response.json() == {
        "output": [2],
    }

    # Test stream
    response = await async_client.post("/stream", json={"input": 1})
    assert response.text == "event: data\r\ndata: 2\r\n\r\nevent: end\r\n\r\n"


def test_invoke(client: RemoteRunnable) -> None:
    """Test sync invoke."""
    assert client.invoke(1) == 2
    assert client.invoke(HumanMessage(content="hello")) == HumanMessage(content="hello")
    # Test invocation with config
    assert client.invoke(1, config={"tags": ["test"]}) == 2


def test_batch(client: RemoteRunnable) -> None:
    """Test sync batch."""
    assert client.batch([]) == []
    assert client.batch([1, 2, 3]) == [2, 3, 4]
    assert client.batch([HumanMessage(content="hello")]) == [
        HumanMessage(content="hello")
    ]


@pytest.mark.asyncio
async def test_ainvoke(async_client: RemoteRunnable) -> None:
    """Test async invoke."""
    assert await async_client.ainvoke(1) == 2
    assert await async_client.ainvoke(HumanMessage(content="hello")) == HumanMessage(
        content="hello"
    )


@pytest.mark.asyncio
async def test_abatch(async_client: RemoteRunnable) -> None:
    """Test async batch."""
    assert await async_client.abatch([]) == []
    assert await async_client.abatch([1, 2, 3]) == [2, 3, 4]
    assert await async_client.abatch([HumanMessage(content="hello")]) == [
        HumanMessage(content="hello")
    ]


# TODO(Team): Determine how to test
# Some issue with event loops
# def test_stream(client: RemoteRunnable) -> None:
#     """Test stream."""
#     assert list(client.stream(1)) == [2]


@pytest.mark.asyncio
async def test_astream(async_client: RemoteRunnable) -> None:
    """Test async stream."""
    outputs = []

    async for chunk in async_client.astream(1):
        outputs.append(chunk)

    assert outputs == [2]

    outputs = []
    data = HumanMessage(content="hello")

    async for chunk in async_client.astream(data):
        outputs.append(chunk)

    assert outputs == [data]


@pytest.mark.asyncio
async def test_astream_log(async_client: RemoteRunnable) -> None:
    """Test async stream."""
    outputs = []

    async for chunk in async_client.astream_log(1):
        outputs.append(chunk)

    assert len(outputs) == 3

    op = outputs[0].ops[0]
    uuid = op["value"]["id"]
    assert op == {
        "op": "replace",
        "path": "",
        "value": {
            "final_output": {"output": 2},
            "id": uuid,
            "logs": [],
            "streamed_output": [],
        },
    }


def test_invoke_as_part_of_sequence(client: RemoteRunnable) -> None:
    """Test as part of sequence."""
    runnable = client | RunnableLambda(func=lambda x: x + 1)
    # without config
    assert runnable.invoke(1) == 3
    # with config
    assert runnable.invoke(1, config={"tags": ["test"]}) == 3
    # without config
    assert runnable.batch([1, 2]) == [3, 4]
    # with config
    assert runnable.batch([1, 2], config={"tags": ["test"]}) == [3, 4]
    # TODO(Team): Determine how to test some issues with event loops for testing
    #   set up
    # without config
    # assert list(runnable.stream([1, 2])) == [3, 4]
    # # with config
    # assert list(runnable.stream([1, 2], config={"tags": ["test"]})) == [3, 4]


@pytest.mark.asyncio
async def test_invoke_as_part_of_sequence_async(async_client: RemoteRunnable) -> None:
    """Test as part of a sequence.

    This helps to verify that config is handled properly (e.g., callbacks are not
    passed to the server, but other config is)
    """
    runnable = async_client | RunnableLambda(
        func=lambda x: x + 1 if isinstance(x, int) else x
    ).with_config({"run_name": "hello"})
    # without config
    assert await runnable.ainvoke(1) == 3
    # with config
    assert await runnable.ainvoke(1, config={"tags": ["test"]}) == 3
    # without config
    assert await runnable.abatch([1, 2]) == [3, 4]
    # with config
    assert await runnable.abatch([1, 2], config={"tags": ["test"]}) == [3, 4]

    # Verify can pass many configs to batch
    configs = [{"tags": ["test"]}, {"tags": ["test2"]}]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    # Verify can ValueError on mismatched configs  number
    with pytest.raises(ValueError):
        assert await runnable.abatch([1, 2], config=[configs[0]]) == [3, 4]

    configs = [{"tags": ["test"]}, {"tags": ["test2"]}]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    configs = [
        {"tags": ["test"]},
        {"tags": ["test2"], "other": "test"},
    ]
    assert await runnable.abatch([1, 2], config=configs) == [3, 4]

    # Without config
    assert [x async for x in runnable.astream(1)] == [3]

    # With Config
    assert [x async for x in runnable.astream(1, config={"tags": ["test"]})] == [3]

    # With config and LC input data
    assert [
        x
        async for x in runnable.astream(
            HumanMessage(content="hello"), config={"tags": ["test"]}
        )
    ] == [HumanMessage(content="hello")]

    log_patches = [x async for x in runnable.astream_log(1)]
    for log_patch in log_patches:
        assert isinstance(log_patch, RunLogPatch)
    # Only check the first entry (not validating implementation here)
    first_op = log_patches[0].ops[0]
    assert first_op["op"] == "replace"
    assert first_op["path"] == ""

    # Validate with HumanMessage
    log_patches = [x async for x in runnable.astream_log(HumanMessage(content="hello"))]
    for log_patch in log_patches:
        assert isinstance(log_patch, RunLogPatch)
    # Only check the first entry (not validating implementation here)
    first_op = log_patches[0].ops[0]
    assert first_op == {
        "op": "replace",
        "path": "",
        "value": {
            "final_output": None,
            "id": first_op["value"]["id"],
            "logs": [],
            "streamed_output": [],
        },
    }


@pytest.mark.asyncio
async def test_multiple_runnables(event_loop: AbstractEventLoop) -> None:
    """Test serving multiple runnables."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    async def mul_2(x: int) -> int:
        """Add one to simulate a valid function"""
        return x * 2

    app = FastAPI()
    add_routes(app, RunnableLambda(add_one), path="/add_one")
    add_routes(
        app,
        RunnableLambda(mul_2),
        input_type=int,
        path="/mul_2",
    )

    async with get_async_client(app, path="/add_one") as runnable:
        async with get_async_client(app, path="/mul_2") as runnable2:
            assert await runnable.ainvoke(1) == 2
            assert await runnable2.ainvoke(4) == 8

            composite_runnable = runnable | runnable2
            assert await composite_runnable.ainvoke(3) == 8

            # Invoke runnable (remote add_one), local add_one, remote mul_2
            composite_runnable_2 = runnable | add_one | runnable2
            assert await composite_runnable_2.ainvoke(3) == 10


@pytest.mark.asyncio
async def test_input_validation(
    event_loop: AbstractEventLoop, mocker: MockerFixture
) -> None:
    """Test client side and server side exceptions."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    server_runnable = RunnableLambda(func=add_one, afunc=add_one)
    server_runnable2 = RunnableLambda(func=add_one, afunc=add_one)

    app = FastAPI()
    add_routes(
        app,
        server_runnable,
        input_type=int,
        path="/add_one",
    )

    add_routes(
        app,
        server_runnable2,
        input_type=int,
        path="/add_one_config",
        config_keys=["tags", "run_name"],
    )

    async with get_async_client(app, path="/add_one") as runnable:
        # Verify that can be invoked with valid input
        assert await runnable.ainvoke(1) == 2
        # Verify that the following substring is present in the error message
        with pytest.raises(httpx.HTTPError):
            await runnable.ainvoke("hello")

        with pytest.raises(httpx.HTTPError):
            await runnable.abatch(["hello"])

    config = {"tags": ["test"]}

    invoke_spy_1 = mocker.spy(server_runnable, "ainvoke")
    # Verify config is handled correctly
    async with get_async_client(app, path="/add_one") as runnable1:
        # Verify that can be invoked with valid input
        # Config ignored for runnable1
        assert await runnable1.ainvoke(1, config=config) == 2
        assert invoke_spy_1.call_args[1]["config"] == {}

    invoke_spy_2 = mocker.spy(server_runnable2, "ainvoke")
    async with get_async_client(app, path="/add_one_config") as runnable2:
        # Config accepted for runnable2
        assert await runnable2.ainvoke(1, config=config) == 2
        # Config ignored
        assert invoke_spy_2.call_args[1]["config"] == config


@pytest.mark.asyncio
async def test_input_validation_with_lc_types(event_loop: AbstractEventLoop) -> None:
    """Test client side and server side exceptions."""

    app = FastAPI()
    # Test with langchain objects
    add_routes(
        app, RunnablePassthrough(), input_type=List[HumanMessage], config_keys=["tags"]
    )
    # Invoke request
    async with get_async_client(app) as passthrough_runnable:
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke("Hello")
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke(["hello"])
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke(HumanMessage(content="h"))
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.ainvoke([SystemMessage(content="hello")])

        # Valid
        result = await passthrough_runnable.ainvoke([HumanMessage(content="hello")])
        assert isinstance(result, list)
        assert isinstance(result[0], HumanMessage)

    # Batch request
    async with get_async_client(app) as passthrough_runnable:
        # invalid
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch("Hello")
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch(["hello"])
        with pytest.raises(httpx.HTTPError):
            await passthrough_runnable.abatch([[SystemMessage(content="hello")]])

        # valid
        result = await passthrough_runnable.abatch([[HumanMessage(content="hello")]])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], HumanMessage)


def test_client_close() -> None:
    """Test that the client can be automatically."""
    runnable = RemoteRunnable(url="/dev/null", timeout=1)
    sync_client = runnable.sync_client
    async_client = runnable.async_client
    assert async_client.is_closed is False
    assert sync_client.is_closed is False
    del runnable
    assert sync_client.is_closed is True
    assert async_client.is_closed is True


@pytest.mark.asyncio
async def test_async_client_close() -> None:
    """Test that the client can be automatically."""
    runnable = RemoteRunnable(url="/dev/null", timeout=1)
    sync_client = runnable.sync_client
    async_client = runnable.async_client
    assert async_client.is_closed is False
    assert sync_client.is_closed is False
    del runnable
    assert sync_client.is_closed is True
    assert async_client.is_closed is True


@pytest.mark.asyncio
async def test_openapi_docs_with_identical_runnables(
    event_loop: AbstractEventLoop, mocker: MockerFixture
) -> None:
    """Test client side and server side exceptions."""

    async def add_one(x: int) -> int:
        """Add one to simulate a valid function"""
        return x + 1

    server_runnable = RunnableLambda(func=add_one)
    server_runnable2 = RunnableLambda(func=add_one)

    app = FastAPI()
    add_routes(
        app,
        server_runnable,
        path="/1",
    )
    # Add another route that uses the same schema (inferred from runnable input schema)
    add_routes(
        app,
        server_runnable2,
        path="/2",
        config_keys=["tags", "run_name"],
    )

    async with AsyncClient(app=app, base_url="http://localhost:9999") as async_client:
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200
