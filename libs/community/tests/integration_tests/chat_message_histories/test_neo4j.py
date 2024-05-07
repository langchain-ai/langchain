import os

from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph


def test_add_messages() -> None:
    """Basic testing: adding messages to the Neo4jChatMessageHistory."""
    assert os.environ.get("NEO4J_URI") is not None
    assert os.environ.get("NEO4J_USERNAME") is not None
    assert os.environ.get("NEO4J_PASSWORD") is not None
    message_store = Neo4jChatMessageHistory("23334")
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")

    # create another message store to check if the messages are stored correctly
    message_store_another = Neo4jChatMessageHistory("46666")
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
    assert os.environ.get("NEO4J_URI") is not None
    assert os.environ.get("NEO4J_USERNAME") is not None
    assert os.environ.get("NEO4J_PASSWORD") is not None
    graph = Neo4jGraph()
    # rewrite env for testing
    os.environ["NEO4J_USERNAME"] = "foo"
    message_store = Neo4jChatMessageHistory("23334", graph=graph)
    message_store.clear()
    assert len(message_store.messages) == 0
    message_store.add_user_message("Hello! Language Chain!")
    message_store.add_ai_message("Hi Guys!")
    # Now check if the messages are stored in the database correctly
    assert len(message_store.messages) == 2
