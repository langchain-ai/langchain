import json
import os

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import CosmosDBChatMessageHistory
from langchain.schema import _message_to_dict

# Replace these with your Azure Cosmos DB endpoint and key
endpoint = os.environ["COSMOS_DB_ENDPOINT"]
credential = os.environ["COSMOS_DB_KEY"]


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup Azure Cosmos DB as a message store
    message_history = CosmosDBChatMessageHistory(
        cosmos_endpoint=endpoint,
        cosmos_database="chat_history",
        cosmos_container="messages",
        credential=credential,
        session_id="my-test-session",
        user_id="my-test-user",
        ttl=10,
    )
    message_history.prepare_cosmos()
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # add some messages
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # get the message history from the memory store and turn it into a json
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # remove the record from Azure Cosmos DB, so the next test run won't pick it up
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []
