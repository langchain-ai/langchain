"""Test OpenAI Chat API wrapper."""
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain.adapters.openai import convert_dict_to_message
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    llm = ChatOpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = ChatOpenAI(model_name="foo")
    assert llm.model_name == "foo"


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar Baz",
                },
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.requires("openai")
def test_openai_predict(mock_completion: dict) -> None:
    llm = ChatOpenAI()
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
        res = llm.predict("bar")
        assert res == "Bar Baz"
    assert completed


@pytest.mark.requires("openai")
async def test_openai_apredict(mock_completion: dict) -> None:
    llm = ChatOpenAI()
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
        res = llm.predict("bar")
        assert res == "Bar Baz"
    assert completed
