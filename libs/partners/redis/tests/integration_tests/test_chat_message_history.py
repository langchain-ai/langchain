import asyncio
import time
from typing import Generator, List, Type, Union, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from redis import Redis
from redis.commands.search.query import Query
from ulid import ULID

from langchain_redis import RedisChatMessageHistory


@pytest.fixture
def redis_url() -> str:
    return "redis://localhost:6379"


@pytest.fixture
def redis_client(redis_url: str) -> Redis:
    return Redis.from_url(redis_url)


@pytest.fixture
def chat_history(redis_url: str) -> Generator[RedisChatMessageHistory, None, None]:
    session_id = f"test_session_{str(ULID())}"
    history = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history.clear()
    try:
        yield history
    finally:
        history.clear()


def test_add_and_retrieve_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, AI!"))
    chat_history.add_message(AIMessage(content="Hello, human!"))

    messages = chat_history.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello, AI!"
    assert messages[1].content == "Hello, human!"


def test_clear_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Test message"))
    assert len(chat_history.messages) == 1

    chat_history.clear()
    assert len(chat_history.messages) == 0


def test_add_multiple_messages(chat_history: RedisChatMessageHistory) -> None:
    messages = [
        HumanMessage(content="Message 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Message 2"),
    ]
    for message in messages:
        chat_history.add_message(message)

    assert len(chat_history.messages) == 3
    assert [msg.content for msg in chat_history.messages] == [
        "Message 1",
        "Response 1",
        "Message 2",
    ]


def test_search_messages(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))
    chat_history.add_message(AIMessage(content="I'm doing well, thank you!"))
    chat_history.add_message(HumanMessage(content="What's the weather like today?"))

    results = chat_history.search_messages("weather")

    assert len(results) == 1
    assert "weather" in results[0]["content"]

    # Test retrieving all messages
    all_messages = chat_history.messages
    assert len(all_messages) == 3
    assert all_messages[0].content == "Hello, how are you?"
    assert all_messages[1].content == "I'm doing well, thank you!"
    assert all_messages[2].content == "What's the weather like today?"


def test_length(chat_history: RedisChatMessageHistory) -> None:
    messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
    for message in messages:
        chat_history.add_message(message)

    assert len(chat_history) == 5


def test_ttl(redis_url: str, redis_client: Redis) -> None:
    session_id = f"ttl_test_{str(ULID())}"
    chat_history = RedisChatMessageHistory(
        session_id=session_id, redis_url=redis_url, ttl=1
    )
    chat_history.add_message(HumanMessage(content="This message will expire"))

    # Check that the message was added
    assert len(chat_history.messages) == 1

    # Find the key for the added message
    query = Query(f"@session_id:{{{chat_history.id}}}")
    results = chat_history.redis_client.ft(chat_history.index_name).search(query)
    assert len(results.docs) == 1
    message_key = results.docs[0].id

    # Check TTL on the message key
    ttl_result = redis_client.ttl(message_key)
    if asyncio.iscoroutine(ttl_result):
        ttl = asyncio.get_event_loop().run_until_complete(ttl_result)
    else:
        ttl = ttl_result
    assert ttl > 0

    time.sleep(2)

    # Verify that the message has expired
    assert len(chat_history.messages) == 0

    # Verify that the key no longer exists
    assert redis_client.exists(message_key) == 0


def test_multiple_sessions(redis_url: str) -> None:
    session1 = f"ttl_test_{str(ULID())}"
    session2 = f"ttl_test_{str(ULID())}"
    history1 = RedisChatMessageHistory(session_id=session1, redis_url=redis_url)
    history2 = RedisChatMessageHistory(session_id=session2, redis_url=redis_url)

    history1.add_message(HumanMessage(content="Message for session 1"))
    history2.add_message(HumanMessage(content="Message for session 2"))

    assert len(history1.messages) == 1
    assert len(history2.messages) == 1
    assert history1.messages[0].content != history2.messages[0].content

    history1.clear()
    history2.clear()


def test_index_creation(redis_client: Redis, redis_url: str) -> None:
    session_id = f"index_test_{str(ULID())}"
    RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    index_info = redis_client.ft("idx:chat_history").info()
    assert index_info is not None
    assert index_info["index_name"] == "idx:chat_history"


@pytest.mark.parametrize("message_type", [HumanMessage, AIMessage, SystemMessage])
def test_different_message_types(
    chat_history: RedisChatMessageHistory,
    message_type: Type[Union[HumanMessage, AIMessage, SystemMessage]],
) -> None:
    message = message_type(content="Test content")
    chat_history.add_message(message)

    retrieved = chat_history.messages[-1]
    assert isinstance(retrieved, BaseMessage)
    assert isinstance(retrieved, message_type)
    assert retrieved.content == "Test content"

    # Use type casting to satisfy mypy
    typed_retrieved = cast(Union[HumanMessage, AIMessage, SystemMessage], retrieved)
    assert typed_retrieved.content == "Test content"


def test_large_number_of_messages(chat_history: RedisChatMessageHistory) -> None:
    large_number = 1000
    messages: List[BaseMessage] = [
        HumanMessage(content=f"Message {i}") for i in range(large_number)
    ]
    for message in messages:
        chat_history.add_message(message)

    retrieved_messages = chat_history.messages

    for i, message in enumerate(retrieved_messages):
        message_content = message.content
        expected_content = f"Message {i}"
        assert (
            message_content == expected_content
        ), f"Message at index {i} has content '{message_content}', \
            expected '{expected_content}'"

    assert (
        retrieved_messages[-1].content == f"Message {large_number - 1}"
    ), f"Last message content is '{retrieved_messages[-1].content}', \
        expected 'Message {large_number - 1}'"


def test_empty_messages(chat_history: RedisChatMessageHistory) -> None:
    assert len(chat_history.messages) == 0


def test_json_structure(
    redis_client: Redis, chat_history: RedisChatMessageHistory
) -> None:
    chat_history.add_message(HumanMessage(content="Test message"))

    # Find the key for the added message
    query = Query(f"@session_id:{{{chat_history.session_id}}}")
    results = chat_history.redis_client.ft(chat_history.index_name).search(query)
    assert len(results.docs) == 1, "Expected one message in the chat history"

    # Get the key of the first (and only) message
    message_key = results.docs[0].id

    # Retrieve the JSON data for this key
    json_data = redis_client.json().get(message_key)

    # Assert the structure of the JSON data
    assert "session_id" in json_data, "session_id should be present in the JSON data"
    assert "type" in json_data, "type should be present in the JSON data"
    assert "data" in json_data, "data should be present in the JSON data"
    assert "timestamp" in json_data, "timestamp should be present in the JSON data"

    # Check the content of the data field
    assert "content" in json_data["data"], "content should be present in the data field"
    assert (
        json_data["data"]["content"] == "Test message"
    ), "Content should match the added message"
    assert (
        json_data["data"]["type"] == "human"
    ), "Type should be 'human' for a HumanMessage"

    # Check the type at the root level
    assert (
        json_data["type"] == "human"
    ), "Type at root level should be 'human' for a HumanMessage"

    # Check the session_id
    assert (
        json_data["session_id"] == chat_history.session_id
    ), "session_id should match the chat history session_id"


def test_search_non_existent_message(chat_history: RedisChatMessageHistory) -> None:
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))
    results = chat_history.search_messages("nonexistent")
    assert len(results) == 0


def test_add_message_to_existing_session(redis_url: str) -> None:
    session_id = f"existing_session_{str(ULID())}"
    history1 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history1.add_message(HumanMessage(content="First message"))

    history2 = RedisChatMessageHistory(session_id=session_id, redis_url=redis_url)
    history2.add_message(HumanMessage(content="Second message"))

    assert len(history1.messages) == 2
    assert len(history2.messages) == 2
