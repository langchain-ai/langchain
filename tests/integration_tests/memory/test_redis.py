import json

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.schema import _message_to_dict


def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup Redis as a message store
    message_history = RedisChatMessageHistory(
        url="redis://localhost:6379/0", ttl=10, session_id="my-test-session"
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

    # remove the record from Redis, so the next test run won't pick it up
    memory.chat_memory.clear()
