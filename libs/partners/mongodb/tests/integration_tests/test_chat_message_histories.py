import json
import os

from langchain.memory import ConversationBufferMemory  # type: ignore[import-not-found]
from langchain_core.messages import message_to_dict

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

DATABASE = "langchain_test_db"
COLLECTION = "langchain_test_chat"

# Replace these with your mongodb connection string
connection_string = os.environ.get("MONGODB_ATLAS_URI", "")


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup MongoDB as a message store
    message_history = MongoDBChatMessageHistory(
        connection_string=connection_string,
        session_id="test-session",
        database_name=DATABASE,
        collection_name=COLLECTION,
    )
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # add some messages
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # get the message history from the memory store and turn it into a json
    messages = memory.chat_memory.messages
    messages_json = json.dumps([message_to_dict(msg) for msg in messages])

    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # remove the record from MongoDB, so the next test run won't pick it up
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []
