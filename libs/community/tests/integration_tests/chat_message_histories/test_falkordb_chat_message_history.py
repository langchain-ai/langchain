"""
Integration tests for FalkorDB Chat History/Memory functionality.

Note:
These tests are conducted using a local FalkorDB instance but can also
be run against a Cloud FalkorDB instance. Ensure that appropriate host,port
cusername, and password configurations are set up
before running the tests.

Test Cases:
1. test_add_messages: Test basic functionality of adding and retrieving
   chat messages from FalkorDB.
2. test_add_messages_graph_object: Test chat message functionality
   when passing the FalkorDB driver through a graph object.

"""

from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_message_histories.falkordb import (
    FalkorDBChatMessageHistory,
)
from langchain_community.graphs import FalkorDBGraph


def test_add_messages() -> None:
    """Basic testing: add messages to the FalkorDBChatMessageHistory."""
    message_store = FalkorDBChatMessageHistory("500daysofSadiya")
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")

    # create another message store to check if the messages are stored correctly
    message_store_another = FalkorDBChatMessageHistory("Shebrokemyheart")
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


def test_add_messages_graph_object() -> None:
    """Basic testing: Passing driver through graph object."""
    graph = FalkorDBGraph("NeverGonnaLetYouDownNevergonnagiveyouup")
    message_store = FalkorDBChatMessageHistory(
        "Gonnahavetoteachmehowtoloveyouagain", graph=graph
    )
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")
    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
