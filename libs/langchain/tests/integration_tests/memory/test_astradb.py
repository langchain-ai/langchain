import os
from typing import AsyncIterable, Iterable

import pytest
from langchain_community.chat_message_histories.astradb import (
    AstraDBChatMessageHistory,
)
from langchain_community.utilities.astradb import SetupMode
from langchain_core.messages import AIMessage, HumanMessage

from langchain.memory import ConversationBufferMemory


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


@pytest.fixture(scope="function")
def history1() -> Iterable[AstraDBChatMessageHistory]:
    history1 = AstraDBChatMessageHistory(
        session_id="session-test-1",
        collection_name="langchain_cmh_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield history1
    history1.collection.astra_db.delete_collection("langchain_cmh_test")


@pytest.fixture(scope="function")
def history2() -> Iterable[AstraDBChatMessageHistory]:
    history2 = AstraDBChatMessageHistory(
        session_id="session-test-2",
        collection_name="langchain_cmh_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield history2
    history2.collection.astra_db.delete_collection("langchain_cmh_test")


@pytest.fixture
async def async_history1() -> AsyncIterable[AstraDBChatMessageHistory]:
    history1 = AstraDBChatMessageHistory(
        session_id="async-session-test-1",
        collection_name="langchain_cmh_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        setup_mode=SetupMode.ASYNC,
    )
    yield history1
    await history1.async_collection.astra_db.delete_collection("langchain_cmh_test")


@pytest.fixture(scope="function")
async def async_history2() -> AsyncIterable[AstraDBChatMessageHistory]:
    history2 = AstraDBChatMessageHistory(
        session_id="async-session-test-2",
        collection_name="langchain_cmh_test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        setup_mode=SetupMode.ASYNC,
    )
    yield history2
    await history2.async_collection.astra_db.delete_collection("langchain_cmh_test")


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
def test_memory_with_message_store(history1: AstraDBChatMessageHistory) -> None:
    """Test the memory with a message store."""
    memory = ConversationBufferMemory(
        memory_key="baz",
        chat_memory=history1,
        return_messages=True,
    )

    assert memory.chat_memory.messages == []

    # add some messages
    memory.chat_memory.add_messages(
        [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
    )

    messages = memory.chat_memory.messages
    expected = [
        AIMessage(content="This is me, the AI"),
        HumanMessage(content="This is me, the human"),
    ]
    assert messages == expected

    # clear the store
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
async def test_memory_with_message_store_async(
    async_history1: AstraDBChatMessageHistory,
) -> None:
    """Test the memory with a message store."""
    memory = ConversationBufferMemory(
        memory_key="baz",
        chat_memory=async_history1,
        return_messages=True,
    )

    assert await memory.chat_memory.aget_messages() == []

    # add some messages
    await memory.chat_memory.aadd_messages(
        [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
    )

    messages = await memory.chat_memory.aget_messages()
    expected = [
        AIMessage(content="This is me, the AI"),
        HumanMessage(content="This is me, the human"),
    ]
    assert messages == expected

    # clear the store
    await memory.chat_memory.aclear()

    assert await memory.chat_memory.aget_messages() == []


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
def test_memory_separate_session_ids(
    history1: AstraDBChatMessageHistory, history2: AstraDBChatMessageHistory
) -> None:
    """Test that separate session IDs do not share entries."""
    memory1 = ConversationBufferMemory(
        memory_key="mk1",
        chat_memory=history1,
        return_messages=True,
    )
    memory2 = ConversationBufferMemory(
        memory_key="mk2",
        chat_memory=history2,
        return_messages=True,
    )

    memory1.chat_memory.add_messages([AIMessage(content="Just saying.")])

    assert memory2.chat_memory.messages == []

    memory2.chat_memory.clear()

    assert memory1.chat_memory.messages != []

    memory1.chat_memory.clear()

    assert memory1.chat_memory.messages == []


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
async def test_memory_separate_session_ids_async(
    async_history1: AstraDBChatMessageHistory, async_history2: AstraDBChatMessageHistory
) -> None:
    """Test that separate session IDs do not share entries."""
    memory1 = ConversationBufferMemory(
        memory_key="mk1",
        chat_memory=async_history1,
        return_messages=True,
    )
    memory2 = ConversationBufferMemory(
        memory_key="mk2",
        chat_memory=async_history2,
        return_messages=True,
    )

    await memory1.chat_memory.aadd_messages([AIMessage(content="Just saying.")])

    assert await memory2.chat_memory.aget_messages() == []

    await memory2.chat_memory.aclear()

    assert await memory1.chat_memory.aget_messages() != []

    await memory1.chat_memory.aclear()

    assert await memory1.chat_memory.aget_messages() == []
