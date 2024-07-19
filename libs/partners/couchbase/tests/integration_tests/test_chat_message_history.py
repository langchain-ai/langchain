"""Test Couchbase Chat Message History functionality"""

import os
import time
from datetime import timedelta
from typing import Any

import pytest
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from langchain_couchbase.chat_message_histories import CouchbaseChatMessageHistory

CONNECTION_STRING = os.getenv("COUCHBASE_CONNECTION_STRING", "")
BUCKET_NAME = os.getenv("COUCHBASE_BUCKET_NAME", "")
SCOPE_NAME = os.getenv("COUCHBASE_SCOPE_NAME", "")
MESSAGE_HISTORY_COLLECTION_NAME = os.getenv(
    "COUCHBASE_CHAT_HISTORY_COLLECTION_NAME", ""
)
USERNAME = os.getenv("COUCHBASE_USERNAME", "")
PASSWORD = os.getenv("COUCHBASE_PASSWORD", "")
SLEEP_DURATION = 0.2


def set_all_env_vars() -> bool:
    """Check if all environment variables are set"""
    return all(
        [
            CONNECTION_STRING,
            BUCKET_NAME,
            SCOPE_NAME,
            MESSAGE_HISTORY_COLLECTION_NAME,
            USERNAME,
            PASSWORD,
        ]
    )


def get_cluster() -> Any:
    """Get a couchbase cluster object"""
    auth = PasswordAuthenticator(USERNAME, PASSWORD)
    options = ClusterOptions(auth)
    connect_string = CONNECTION_STRING
    cluster = Cluster(connect_string, options)

    # Wait until the cluster is ready for use.
    cluster.wait_until_ready(timedelta(seconds=5))

    return cluster


@pytest.fixture()
def cluster() -> Any:
    """Get a couchbase cluster object"""
    return get_cluster()


@pytest.mark.skipif(
    not set_all_env_vars(), reason="Missing Couchbase environment variables"
)
class TestCouchbaseCache:
    def test_memory_with_message_store(self, cluster: Any) -> None:
        """Test chat message history with a message store"""

        message_history = CouchbaseChatMessageHistory(
            cluster=cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=MESSAGE_HISTORY_COLLECTION_NAME,
            session_id="test-session",
        )

        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )

        # clear the memory
        memory.chat_memory.clear()

        # wait for the messages to be cleared
        time.sleep(SLEEP_DURATION)
        assert memory.chat_memory.messages == []

        # add some messages
        ai_message = AIMessage(content="Hello, how are you doing ?")
        user_message = HumanMessage(content="I'm good, how are you?")
        memory.chat_memory.add_messages([ai_message, user_message])

        # wait until the messages can be retrieved
        time.sleep(SLEEP_DURATION)

        # check that the messages are in the memory
        messages = memory.chat_memory.messages
        assert len(messages) == 2
        for message in messages:
            assert message in [ai_message, user_message]

        # clear the memory
        memory.chat_memory.clear()
        time.sleep(SLEEP_DURATION)
        assert memory.chat_memory.messages == []

    def test_memory_with_separate_sessions(self, cluster: Any) -> None:
        """Test the chat message history with multiple sessions"""

        message_history_a = CouchbaseChatMessageHistory(
            cluster=cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=MESSAGE_HISTORY_COLLECTION_NAME,
            session_id="test-session-a",
        )

        message_history_b = CouchbaseChatMessageHistory(
            cluster=cluster,
            bucket_name=BUCKET_NAME,
            scope_name=SCOPE_NAME,
            collection_name=MESSAGE_HISTORY_COLLECTION_NAME,
            session_id="test-session-b",
        )

        memory_a = ConversationBufferMemory(
            memory_key="a", chat_memory=message_history_a, return_messages=True
        )
        memory_b = ConversationBufferMemory(
            memory_key="b", chat_memory=message_history_b, return_messages=True
        )

        # clear the memory
        memory_a.chat_memory.clear()
        memory_b.chat_memory.clear()

        # add some messages
        ai_message = AIMessage(content="Hello, how are you doing ?")
        user_message = HumanMessage(content="I'm good, how are you?")
        memory_a.chat_memory.add_ai_message(ai_message)
        memory_b.chat_memory.add_user_message(user_message)

        # wait until the messages can be retrieved
        time.sleep(SLEEP_DURATION)

        # check that the messages are in the memory
        messages_a = memory_a.chat_memory.messages
        messages_b = memory_b.chat_memory.messages
        assert len(messages_a) == 1
        assert len(messages_b) == 1
        assert messages_a[0] == ai_message
        assert messages_b[0] == user_message

        # clear the memory
        memory_a.chat_memory.clear()
        memory_b.chat_memory.clear()
