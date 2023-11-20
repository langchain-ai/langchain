import json

import pytest

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.upstash_redis import (
    UpstashRedisChatMessageHistory,
)
from langchain.schema.messages import _message_to_dict

URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"


@pytest.mark.requires("upstash_redis")
def test_memory_with_message_store() -> None:
    """Test the memory with a message store."""
    # setup Upstash Redis as a message store
    message_history = UpstashRedisChatMessageHistory(
        url=URL, token=TOKEN, ttl=10, session_id="my-test-session"
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
