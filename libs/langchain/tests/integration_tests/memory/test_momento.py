"""Test Momento chat message history functionality.

To run tests, set the environment variable MOMENTO_AUTH_TOKEN to a valid
Momento auth token. This can be obtained by signing up for a free
Momento account at https://gomomento.com/.
"""
import json
import uuid
from datetime import timedelta
from typing import Iterator

import pytest

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import MomentoChatMessageHistory
from langchain.schema.messages import _message_to_dict


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.fixture(scope="function")
def message_history() -> Iterator[MomentoChatMessageHistory]:
    from momento import CacheClient, Configurations, CredentialProvider

    cache_name = f"langchain-test-cache-{random_string()}"
    client = CacheClient(
        Configurations.Laptop.v1(),
        CredentialProvider.from_environment_variable("MOMENTO_API_KEY"),
        default_ttl=timedelta(seconds=30),
    )
    try:
        chat_message_history = MomentoChatMessageHistory(
            session_id="my-test-session",
            cache_client=client,
            cache_name=cache_name,
        )
        yield chat_message_history
    finally:
        client.delete_cache(cache_name)


def test_memory_empty_on_new_session(
    message_history: MomentoChatMessageHistory,
) -> None:
    memory = ConversationBufferMemory(
        memory_key="foo", chat_memory=message_history, return_messages=True
    )
    assert memory.chat_memory.messages == []


def test_memory_with_message_store(message_history: MomentoChatMessageHistory) -> None:
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # Add some messages to the memory store
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # Verify that the messages are in the store
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Verify clearing the store
    memory.chat_memory.clear()
    assert memory.chat_memory.messages == []
