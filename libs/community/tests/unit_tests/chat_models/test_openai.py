"""Test OpenAI Chat API wrapper."""

import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.adapters.openai import convert_dict_to_message
from langchain_community.chat_models.openai import ChatOpenAI


@pytest.mark.requires("openai")
def test_openai_model_param() -> None:
    test_cases: List[dict] = [
        {"model_name": "foo", "openai_api_key": "foo"},
        {"model": "foo", "openai_api_key": "foo"},
        {"model_name": "foo", "api_key": "foo"},
        {"model_name": "foo", "openai_api_key": "foo", "max_retries": 2},
    ]

    for case in test_cases:
        llm = ChatOpenAI(**case)
        assert llm.model_name == "foo", "Model name should be 'foo'"
        assert llm.openai_api_key == "foo", "API key should be 'foo'"
        assert hasattr(llm, "max_retries"), "max_retries attribute should exist"
        assert llm.max_retries == 2, "max_retries default should be set to 2"


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
    llm = ChatOpenAI(openai_api_key="foo")  # type: ignore[call-arg]
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
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"
    assert completed


@pytest.mark.requires("openai")
async def test_openai_apredict(mock_completion: dict) -> None:
    llm = ChatOpenAI(openai_api_key="foo")  # type: ignore[call-arg]
    mock_client = MagicMock()
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
        res = await llm.apredict("bar")
        assert res == "Bar Baz"
    assert completed
