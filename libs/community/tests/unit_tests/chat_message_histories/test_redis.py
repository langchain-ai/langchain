from unittest.mock import MagicMock, Mock, PropertyMock

import pytest

from langchain_community.chat_message_histories import (
    RedisChatMessageHistoryWithTokenLimit,
)


@pytest.fixture
def mock_redis(mocker):
    mocker.patch.dict("sys.modules", redis=MagicMock())
    mock_redis = Mock()
    mocker.patch(
        "langchain_community.chat_message_histories.redis.get_client",
        return_value=mock_redis,
    )
    return mock_redis


@pytest.fixture
def mock_super_messages(mocker):
    mocker.patch(
        "langchain_community.chat_message_histories.redis.RedisChatMessageHistory.messages",
        new_callable=PropertyMock,
        return_value=["Message1", "Message2"],
    )


@pytest.fixture
def mock_llm_tokens_below_max_size():
    mock_llm = Mock()
    mock_llm.get_num_tokens_from_messages = MagicMock(return_value=1000)
    return mock_llm


@pytest.fixture
def mock_llm_tokens_over_max_size():
    mock_llm = Mock()
    mock_llm.get_num_tokens_from_messages.side_effect = [3000, 999]
    return mock_llm


def test_history_larger_than_max_token_limit(
    mock_super_messages, mock_redis, mock_llm_tokens_below_max_size
) -> None:
    history = RedisChatMessageHistoryWithTokenLimit(
        session_id="test_session_id", llm=mock_llm_tokens_below_max_size
    )
    messages = history.messages
    mock_llm_tokens_below_max_size.get_num_tokens_from_messages.assert_called_once()
    mock_redis.rpop.assert_not_called()
    assert messages == ["Message1", "Message2"]


def test_history_smaller_than_max_token_limit(
    mock_super_messages, mock_redis, mock_llm_tokens_over_max_size
) -> None:
    history = RedisChatMessageHistoryWithTokenLimit(
        session_id="test_session_id", llm=mock_llm_tokens_over_max_size
    )
    messages = history.messages
    assert mock_llm_tokens_over_max_size.get_num_tokens_from_messages.call_count == 2
    mock_redis.rpop.assert_called_once()
    assert messages == ["Message2"]
