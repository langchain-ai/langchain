import os
import time
from typing import Optional

from cassandra.cluster import Cluster

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.cassandra import (
    CassandraChatMessageHistory,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
)


def _chat_message_history(
    session_id: str = "test-session",
    drop: bool = True,
    ttl_seconds: Optional[int] = None,
) -> CassandraChatMessageHistory:
    keyspace = "cmh_test_keyspace"
    table_name = "cmh_test_table"
    # get db connection
    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = os.environ["CONTACT_POINTS"].split(",")
        cluster = Cluster(contact_points)
    else:
        cluster = Cluster()
    #
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")
    #
    return CassandraChatMessageHistory(
        session_id=session_id,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
        **({} if ttl_seconds is None else {"ttl_seconds": ttl_seconds}),
    )


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup cassandra as a message store
    message_history = _chat_message_history()
    memory = ConversationBufferMemory(
        memory_key="baz",
        chat_memory=message_history,
        return_messages=True,
    )

    assert memory.chat_memory.messages == []

    # add some messages
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    messages = memory.chat_memory.messages
    expected = [
        AIMessage(content="This is me, the AI"),
        HumanMessage(content="This is me, the human"),
    ]
    assert messages == expected

    # clear the store
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []


def test_memory_separate_session_ids() -> None:
    """Test that separate session IDs do not share entries."""
    message_history1 = _chat_message_history(session_id="test-session1")
    memory1 = ConversationBufferMemory(
        memory_key="mk1",
        chat_memory=message_history1,
        return_messages=True,
    )
    message_history2 = _chat_message_history(session_id="test-session2")
    memory2 = ConversationBufferMemory(
        memory_key="mk2",
        chat_memory=message_history2,
        return_messages=True,
    )

    memory1.chat_memory.add_ai_message("Just saying.")

    assert memory2.chat_memory.messages == []

    memory1.chat_memory.clear()
    memory2.chat_memory.clear()


def test_memory_ttl() -> None:
    """Test time-to-live feature of the memory."""
    message_history = _chat_message_history(ttl_seconds=5)
    memory = ConversationBufferMemory(
        memory_key="baz",
        chat_memory=message_history,
        return_messages=True,
    )
    #
    assert memory.chat_memory.messages == []
    memory.chat_memory.add_ai_message("Nothing special here.")
    time.sleep(2)
    assert memory.chat_memory.messages != []
    time.sleep(5)
    assert memory.chat_memory.messages == []
