"""Test Groq Chat API wrapper."""

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langchain_core.load as lc_load
import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)

from langchain_groq.chat_models import ChatGroq, _convert_dict_to_message

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "fake-key"


def test_groq_model_param() -> None:
    llm = ChatGroq(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = ChatGroq(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
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
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="foo", response_metadata={"model_provider": "groq"}
    )
    assert result == expected_output


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
        "function": {
            "arguments": '{"name":"Sally","hair_color":"green"}',
            "name": "GenerateUsername",
        },
        "type": "function",
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                type="tool_call",
            )
        ],
        response_metadata={"model_provider": "groq"},
    )
    assert result == expected_output

    # Test malformed tool call
    raw_tool_calls = [
        {
            "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
            "function": {
                "arguments": "oops",
                "name": "GenerateUsername",
            },
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"name":"Sally","hair_color":"green"}',
                "name": "GenerateUsername",
            },
            "type": "function",
        },
    ]
    message = {"role": "assistant", "content": None, "tool_calls": raw_tool_calls}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
                type="invalid_tool_call",
            ),
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_abc123",
                type="tool_call",
            ),
        ],
        response_metadata={"model_provider": "groq"},
    )
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "test-model",
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


def test_groq_invoke(mock_completion: dict) -> None:
    llm = ChatGroq(model="foo")
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
        assert type(res) is AIMessage
    assert completed


async def test_groq_ainvoke(mock_completion: dict) -> None:
    llm = ChatGroq(model="foo")
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
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


def test_chat_groq_extra_kwargs() -> None:
    """Test extra kwargs to chat groq."""
    # Check that foo is saved in extra_kwargs.
    with pytest.warns(UserWarning) as record:
        llm = ChatGroq(model="foo", foo=3, max_tokens=10)  # type: ignore[call-arg]
        assert llm.max_tokens == 10
        assert llm.model_kwargs == {"foo": 3}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if extra_kwargs are provided, they are added to it.
    with pytest.warns(UserWarning) as record:
        llm = ChatGroq(model="foo", foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
        assert llm.model_kwargs == {"foo": 3, "bar": 2}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatGroq(model="foo", foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatGroq(model="foo", model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatGroq(model="foo", model_kwargs={"model": "test-model"})


def test_chat_groq_invalid_streaming_params() -> None:
    """Test that an error is raised if streaming is invoked with n>1."""
    with pytest.raises(ValueError):
        ChatGroq(
            model="foo",
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_chat_groq_secret() -> None:
    """Test that secret is not printed."""
    secret = "secretKey"  # noqa: S105
    not_secret = "safe"  # noqa: S105
    llm = ChatGroq(model="foo", api_key=secret, model_kwargs={"not_secret": not_secret})  # type: ignore[call-arg, arg-type]
    stringified = str(llm)
    assert not_secret in stringified
    assert secret not in stringified


@pytest.mark.filterwarnings("ignore:The function `loads` is in beta")
def test_groq_serialization() -> None:
    """Test that ChatGroq can be successfully serialized and deserialized."""
    api_key1 = "top secret"
    api_key2 = "topest secret"
    llm = ChatGroq(model="foo", api_key=api_key1, temperature=0.5)  # type: ignore[call-arg, arg-type]
    dump = lc_load.dumps(llm)
    llm2 = lc_load.loads(
        dump,
        valid_namespaces=["langchain_groq"],
        secrets_map={"GROQ_API_KEY": api_key2},
    )

    assert type(llm2) is ChatGroq

    # Ensure api key wasn't dumped and instead was read from secret map.
    assert llm.groq_api_key is not None
    assert llm.groq_api_key.get_secret_value() not in dump
    assert llm2.groq_api_key is not None
    assert llm2.groq_api_key.get_secret_value() == api_key2

    # Ensure a non-secret field was preserved
    assert llm.temperature == llm2.temperature

    # Ensure a None was preserved
    assert llm.groq_api_base == llm2.groq_api_base
