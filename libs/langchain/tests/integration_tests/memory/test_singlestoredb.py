import json

from langchain.memory import ConversationBufferMemory, SingleStoreDBChatMessageHistory
from langchain.schema.messages import _message_to_dict

# Replace these with your mongodb connection string
TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup SingleStoreDB as a message store
    message_history = SingleStoreDBChatMessageHistory(
        session_id="test-session",
        host=TEST_SINGLESTOREDB_URL,
    )
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

    # remove the record from SingleStoreDB, so the next test run won't pick it up
    memory.chat_memory.clear()

    assert memory.chat_memory.messages == []
