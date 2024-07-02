from typing import Any

import redis
from langchain_core.messages import AIMessage, HumanMessage
from mockredis import mock_redis_client

from langchain_community.chat_message_histories import RedisChatMessageHistory


def add_messages(redis_history: RedisChatMessageHistory) -> None:
    redis_history.add_message(HumanMessage(content="Who are you?"))
    redis_history.add_message(AIMessage(content="I am an ai assistant."))
    redis_history.add_message(HumanMessage(content="Tell me a story."))
    redis_history.add_message(AIMessage(content="Yes,I will tell you a story."))
    redis_history.add_message(
        HumanMessage(content="Tell me a story about dog and cat.")
    )
    redis_history.add_message(
        AIMessage(content="I will tell you a story about dog and cat.")
    )


def test_add_messages_with_default(mocker: Any) -> None:
    mocker.patch("redis.Redis", mock_redis_client)

    session_id = "xxx-yyy-zzzz-ddd"

    redis_history = RedisChatMessageHistory(session_id)
    redis_history.redis_client = redis.Redis()

    add_messages(redis_history)

    ret = redis_history.messages
    assert len(ret) == 6


def test_add_messages_with_limit(mocker: Any) -> None:
    mocker.patch("redis.Redis", mock_redis_client)
    session_id = "zzz-yyy-dddd-xxx"

    redis_history = RedisChatMessageHistory(session_id, last_k_pair_msgs=1)
    redis_history.redis_client = redis.Redis()

    add_messages(redis_history)

    ret = redis_history.messages
    assert len(ret) == 2
    assert type(ret[0]) == HumanMessage
    assert ret[0].content == "Tell me a story about dog and cat."


def test_add_messages_with_negative(mocker: Any) -> None:
    mocker.patch("redis.Redis", mock_redis_client)
    session_id = "rrr-uuu-ddd-zzz"

    redis_history = RedisChatMessageHistory(session_id, last_k_pair_msgs=-1)
    redis_history.redis_client = redis.Redis()

    add_messages(redis_history)

    ret = redis_history.messages
    assert len(ret) == 6
    assert ret[0].content == "Who are you?"


def test_clear_messages(mocker: Any) -> None:
    mocker.patch("redis.Redis", mock_redis_client)
    session_id = "eeee-ffff-gggg"
    redis_history = RedisChatMessageHistory(session_id)
    redis_history.redis_client = redis.Redis()

    add_messages(redis_history)

    ret = redis_history.messages

    ret = redis_history.messages
    assert len(ret) == 6

    redis_history.clear()
    ret = redis_history.messages
    assert len(ret) == 0


def test_multiple_session(mocker: Any) -> None:
    mocker.patch("redis.Redis", mock_redis_client)

    # first session
    session_id_1 = "1-session-multiple"
    redis_history_1 = RedisChatMessageHistory(session_id_1, last_k_pair_msgs=1)
    redis_history_1.redis_client = redis.Redis()
    add_messages(redis_history_1)
    ret1 = redis_history_1.messages
    assert len(ret1) == 2
    assert ret1[0].content == "Tell me a story about dog and cat."

    # second sesson
    session_id_2 = "2-session-multiple"
    redis_history_2 = RedisChatMessageHistory(session_id_2, last_k_pair_msgs=2)
    redis_history_2.redis_client = redis.Redis()
    add_messages(redis_history_2)
    ret2 = redis_history_2.messages
    assert len(ret2) == 4
    assert ret2[0].content == "Tell me a story."
