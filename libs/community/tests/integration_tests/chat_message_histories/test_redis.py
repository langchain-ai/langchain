from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pytest_mock import MockerFixture

from langchain_community.chat_message_histories.redis import RedisChatMessageHistory

if TYPE_CHECKING:
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis

# Using a non-standard port to avoid conflicts with potentially local running
# redis instances
# You can spin up a local redis using docker compose
# cd [repository-root]/docker
# docker-compose up redis
REDIS_TEST_URL = "redis://localhost:6020"


@pytest.fixture()
def sync_redis_client() -> "Redis":
    from redis import Redis

    return Redis.from_url(REDIS_TEST_URL)


@pytest.fixture()
def async_redis_client() -> "AsyncRedis":
    from redis.asyncio import Redis as AsyncRedis

    return AsyncRedis.from_url(REDIS_TEST_URL)


@pytest.fixture()
def mocked_utilities_get_client(
    mocker: MockerFixture, sync_redis_client: "Redis"
) -> Mock:
    return mocker.patch(
        "langchain_community.chat_message_histories.redis.get_client",
        return_value=sync_redis_client,
    )


@pytest.mark.asyncio
async def test_init_with_sync_client(
    sync_redis_client: "Redis", mocked_utilities_get_client: Mock
) -> None:
    session_id = "test_session_id"

    sync_memory = RedisChatMessageHistory(session_id, redis_client=sync_redis_client)

    assert sync_memory.redis_client is not None
    mocked_utilities_get_client.assert_not_called()

    with pytest.raises(RuntimeError):
        _ = sync_memory.async_redis_client

    for method, args in {
        sync_memory.aget_messages: (),
        sync_memory.aadd_messages: ([],),
        sync_memory.aclear: (),
    }.items():
        with pytest.raises(RuntimeError):
            await method(*args)


@pytest.mark.asyncio
async def test_init_with_default_client(mocked_utilities_get_client: Mock) -> None:
    session_id = "test_session_id"

    sync_default_memory = RedisChatMessageHistory(session_id)

    assert sync_default_memory.redis_client is not None
    mocked_utilities_get_client.assert_called_once()

    with pytest.raises(RuntimeError):
        _ = sync_default_memory.async_redis_client


def test_init_with_async_client(
    async_redis_client: "AsyncRedis", mocked_utilities_get_client: Mock
) -> None:
    session_id = "test_session_id"
    async_memory = RedisChatMessageHistory(
        session_id, async_redis_client=async_redis_client
    )

    assert async_memory.async_redis_client is not None
    mocked_utilities_get_client.assert_not_called()

    with pytest.raises(RuntimeError):
        _ = async_memory.redis_client


@pytest.mark.asyncio
async def test_only_async_memory_usage(async_redis_client: "AsyncRedis") -> None:
    session_id = "test_session_id"
    memory = RedisChatMessageHistory(session_id, async_redis_client=async_redis_client)
    history = [
        HumanMessage("1"),
        HumanMessage("2"),
        AIMessage("3"),
    ]

    await memory.aadd_messages(history)
    assert await memory.aget_messages() == history

    await memory.aclear()
    assert await memory.aget_messages() == []

    with pytest.raises(RuntimeError):
        _ = memory.messages

    for method, args in {
        memory.add_message: (history[0],),
        memory.add_messages: (history,),
        memory.clear: (),
    }.items():
        with pytest.raises(RuntimeError):
            method(*args)


@pytest.mark.asyncio
async def test_only_sync_memory_usage(sync_redis_client: "Redis") -> None:
    session_id = "test_session_id"
    memory = RedisChatMessageHistory(session_id, redis_client=sync_redis_client)
    history = [
        HumanMessage("1"),
        HumanMessage("2"),
        AIMessage("3"),
    ]

    memory.add_message(history[0])
    memory.add_messages(history[1:])

    assert memory.messages == history
    memory.clear()

    assert memory.messages == []

    for method, args in {
        memory.aget_messages: (),
        memory.aadd_messages: (history,),
        memory.aclear: (),
    }.items():
        with pytest.raises(RuntimeError):
            await method(*args)


@pytest.mark.asyncio
async def test_memory_usage(
    sync_redis_client: "Redis", async_redis_client: "AsyncRedis"
) -> None:
    session_id = "test_session_id"
    memory = RedisChatMessageHistory(
        session_id,
        redis_client=sync_redis_client,
        async_redis_client=async_redis_client,
    )
    history = [
        HumanMessage("1"),
        HumanMessage("2"),
        AIMessage("3"),
    ]

    memory.add_messages(history)
    assert await memory.aget_messages() == history

    memory.clear()
    assert await memory.aget_messages() == []

    await memory.aadd_messages(history)
    assert memory.messages == history

    await memory.aclear()
    assert memory.messages == []
