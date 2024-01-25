import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_message_histories import TiDBChatMessageHistory

try:
    CONNECTION_STRING = os.getenv("TEST_TiDB_CHAT_URL", "")

    if CONNECTION_STRING == "":
        raise OSError("TEST_TiDB_URL environment variable is not set")

    tidb_available = True
except (OSError, ImportError):
    tidb_available = False


@pytest.mark.skipif(not tidb_available, reason="tidb is not available")
def test_add_messages() -> None:
    """Basic testing: adding messages to the TiDBChatMessageHistory."""
    message_store = TiDBChatMessageHistory("23334", CONNECTION_STRING)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")

    # create another message store to check if the messages are stored correctly
    message_store_another = TiDBChatMessageHistory("46666", CONNECTION_STRING)
    message_store_another.clear()
    assert len(message_store_another.messages) == 0
    message_store_another.add_user_message("Hello! Bot!")
    message_store_another.add_ai_message("Hi there!")
    message_store_another.add_user_message("How's this pr going?")

    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
    assert isinstance(message_store.messages[0], HumanMessage)
    assert isinstance(message_store.messages[1], AIMessage)
    assert message_store.messages[0].content == "Hello! Language Chain!"
    assert message_store.messages[1].content == "Hi Guys!"

    assert len(message_store_another.messages) == 3
    assert isinstance(message_store_another.messages[0], HumanMessage)
    assert isinstance(message_store_another.messages[1], AIMessage)
    assert isinstance(message_store_another.messages[2], HumanMessage)
    assert message_store_another.messages[0].content == "Hello! Bot!"
    assert message_store_another.messages[1].content == "Hi there!"
    assert message_store_another.messages[2].content == "How's this pr going?"

    # Now clear the first history
    message_store.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 3
    message_store_another.clear()
    assert len(message_store.messages) == 0
    assert len(message_store_another.messages) == 0


def test_tidb_recent_chat_message():
    """Test the TiDBChatMessageHistory with earliest_time parameter."""
    import time
    from datetime import datetime

    # prepare some messages
    message_store = TiDBChatMessageHistory("2333", CONNECTION_STRING)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")
    assert len(message_store.messages) == 2
    assert isinstance(message_store.messages[0], HumanMessage)
    assert isinstance(message_store.messages[1], AIMessage)
    assert message_store.messages[0].content == "Hello! Language Chain!"
    assert message_store.messages[1].content == "Hi Guys!"

    # now we add some recent messages to the database
    earliest_time = datetime.utcnow()
    time.sleep(1)

    message_store.add_user_message("How's this pr going?")
    message_store.add_ai_message("It's almost done!")
    assert len(message_store.messages) == 4
    assert isinstance(message_store.messages[2], HumanMessage)
    assert isinstance(message_store.messages[3], AIMessage)
    assert message_store.messages[2].content == "How's this pr going?"
    assert message_store.messages[3].content == "It's almost done!"

    # now we create another message store with earliest_time parameter
    message_store_another = TiDBChatMessageHistory(
        "2333", CONNECTION_STRING, earliest_time=earliest_time
    )
    assert len(message_store_another.messages) == 2
    assert isinstance(message_store_another.messages[0], HumanMessage)
    assert isinstance(message_store_another.messages[1], AIMessage)
    assert message_store_another.messages[0].content == "How's this pr going?"
    assert message_store_another.messages[1].content == "It's almost done!"

    # now we clear the message store
    message_store.clear()
    assert len(message_store.messages) == 0
