"""Test chat model integration."""
import json
import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic.v1 import SecretStr

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_naver import ChatNaver
from langchain_naver.chat_models import (
    _convert_naver_chat_message_to_message,
    _convert_message_to_naver_chat_message
)

os.environ["NCP_CLOVASTUDIO_API_KEY"] = "test_api_key"
os.environ["NCP_APIGW_API_KEY"] = "test_gw_key"


def test_initialization_api_key() -> None:
    """Test chat model initialization."""
    chat_model = ChatNaver(clovastudio_api_key="foo", apigw_api_key="bar")
    assert cast(SecretStr, chat_model.ncp_clovastudio_api_key).get_secret_value() == "foo"
    assert cast(SecretStr, chat_model.ncp_apigw_api_key).get_secret_value() == "bar"


def test_initialization_model_name() -> None:
    llm = ChatNaver(model="HCX-DASH-001")
    assert llm.model_name == "HCX-DASH-001"
    llm = ChatNaver(model_name="HCX-DASH-001")
    assert llm.model_name == "HCX-DASH-001"


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message


def test_convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_naver_chat_message_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_naver_chat_message(expected_output) == message



@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "HCX-003",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bab",
                    "name": "KimSolar",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_naver_invoke(mock_completion: dict) -> None:
    llm = ChatNaver()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bab")
        assert res.content == "Bab"
    assert completed


async def test_naver_ainvoke(mock_completion: dict) -> None:
    llm = ChatNaver()
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("bab")
        assert res.content == "Bab"
    assert completed


def test_naver_invoke_name(mock_completion: dict) -> None:
    llm = ChatNaver()

    mock_client = MagicMock()
    mock_client.create.return_value = mock_completion

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        messages = [
            HumanMessage(content="Foo", name="Zorba"),
        ]
        res = llm.invoke(messages)
        call_args, call_kwargs = mock_client.create.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Zorba"

        # check return type has name
        assert res.content == "Bab"
        assert res.name == "KimSolar"
