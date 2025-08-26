import os
from collections.abc import AsyncIterator, Iterator
from uuid import UUID

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pytest_mock import MockerFixture

from .conftest_checkpointer import (
    _checkpointer_memory,
    _checkpointer_postgres,
    _checkpointer_postgres_aio,
    _checkpointer_postgres_aio_pipe,
    _checkpointer_postgres_aio_pool,
    _checkpointer_postgres_pipe,
    _checkpointer_postgres_pool,
    _checkpointer_sqlite,
    _checkpointer_sqlite_aio,
)
from .conftest_store import (
    _store_memory,
    _store_postgres,
    _store_postgres_aio,
    _store_postgres_aio_pipe,
    _store_postgres_aio_pool,
    _store_postgres_pipe,
    _store_postgres_pool,
)

# Global variables for checkpointer and store configurations
FAST_MODE = os.getenv("LANGGRAPH_TEST_FAST", "true").lower() in ("true", "1", "yes")

SYNC_CHECKPOINTER_PARAMS = (
    ["memory"]
    if FAST_MODE
    else [
        "memory",
        "sqlite",
        "postgres",
        "postgres_pipe",
        "postgres_pool",
    ]
)

ASYNC_CHECKPOINTER_PARAMS = (
    ["memory"]
    if FAST_MODE
    else [
        "memory",
        "sqlite_aio",
        "postgres_aio",
        "postgres_aio_pipe",
        "postgres_aio_pool",
    ]
)

SYNC_STORE_PARAMS = (
    ["in_memory"]
    if FAST_MODE
    else [
        "in_memory",
        "postgres",
        "postgres_pipe",
        "postgres_pool",
    ]
)

ASYNC_STORE_PARAMS = (
    ["in_memory"]
    if FAST_MODE
    else [
        "in_memory",
        "postgres_aio",
        "postgres_aio_pipe",
        "postgres_aio_pool",
    ]
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)


# checkpointer fixtures


@pytest.fixture(
    params=SYNC_STORE_PARAMS,
)
def sync_store(request: pytest.FixtureRequest) -> Iterator[BaseStore]:
    store_name = request.param
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        with _store_memory() as store:
            yield store
    elif store_name == "postgres":
        with _store_postgres() as store:
            yield store
    elif store_name == "postgres_pipe":
        with _store_postgres_pipe() as store:
            yield store
    elif store_name == "postgres_pool":
        with _store_postgres_pool() as store:
            yield store
    else:
        msg = f"Unknown store {store_name}"
        raise NotImplementedError(msg)


@pytest.fixture(
    params=ASYNC_STORE_PARAMS,
)
async def async_store(request: pytest.FixtureRequest) -> AsyncIterator[BaseStore]:
    store_name = request.param
    if store_name is None:
        yield None
    elif store_name == "in_memory":
        with _store_memory() as store:
            yield store
    elif store_name == "postgres_aio":
        async with _store_postgres_aio() as store:
            yield store
    elif store_name == "postgres_aio_pipe":
        async with _store_postgres_aio_pipe() as store:
            yield store
    elif store_name == "postgres_aio_pool":
        async with _store_postgres_aio_pool() as store:
            yield store
    else:
        msg = f"Unknown store {store_name}"
        raise NotImplementedError(msg)


@pytest.fixture(
    params=SYNC_CHECKPOINTER_PARAMS,
)
def sync_checkpointer(
    request: pytest.FixtureRequest,
) -> Iterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "memory":
        with _checkpointer_memory() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "sqlite":
        with _checkpointer_sqlite() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres":
        with _checkpointer_postgres() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_pipe":
        with _checkpointer_postgres_pipe() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_pool":
        with _checkpointer_postgres_pool() as checkpointer:
            yield checkpointer
    else:
        msg = f"Unknown checkpointer: {checkpointer_name}"
        raise NotImplementedError(msg)


@pytest.fixture(
    params=ASYNC_CHECKPOINTER_PARAMS,
)
async def async_checkpointer(
    request: pytest.FixtureRequest,
) -> AsyncIterator[BaseCheckpointSaver]:
    checkpointer_name = request.param
    if checkpointer_name == "memory":
        with _checkpointer_memory() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "sqlite_aio":
        async with _checkpointer_sqlite_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio":
        async with _checkpointer_postgres_aio() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pipe":
        async with _checkpointer_postgres_aio_pipe() as checkpointer:
            yield checkpointer
    elif checkpointer_name == "postgres_aio_pool":
        async with _checkpointer_postgres_aio_pool() as checkpointer:
            yield checkpointer
    else:
        msg = f"Unknown checkpointer: {checkpointer_name}"
        raise NotImplementedError(msg)
