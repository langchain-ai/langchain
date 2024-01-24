from typing import cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
)
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.chat_models.baichuan import (
    ChatBaichuan,
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"role": "user", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"role": "assistant", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    with pytest.raises(TypeError) as e:
        _convert_message_to_dict(message)
    assert "Got unknown type" in str(e)


def test__convert_message_to_dict_function() -> None:
    message = FunctionMessage(name="foo", content="bar")
    with pytest.raises(TypeError) as e:
        _convert_message_to_dict(message)
    assert "Got unknown type" in str(e)


def test__convert_dict_to_message_human() -> None:
    message_dict = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message_dict = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_other_role() -> None:
    message_dict = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = ChatMessage(role="system", content="foo")
    assert result == expected_output


def test__convert_delta_to_message_assistant() -> None:
    delta = {"role": "assistant", "content": "foo"}
    result = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    expected_output = AIMessageChunk(content="foo")
    assert result == expected_output


def test__convert_delta_to_message_human() -> None:
    delta = {"role": "user", "content": "foo"}
    result = _convert_delta_to_message_chunk(delta, HumanMessageChunk)
    expected_output = HumanMessageChunk(content="foo")
    assert result == expected_output


def test_baichuan_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("BAICHUAN_API_KEY", "test-api-key")

    chat = ChatBaichuan()
    print(chat.baichuan_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_baichuan_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = ChatBaichuan(baichuan_api_key="test-api-key")
    print(chat.baichuan_api_key, end="")
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = ChatBaichuan(
        baichuan_api_key="test-api-key",
        baichuan_secret_key="test-secret-key",  # For backward compatibility
    )
    assert cast(SecretStr, chat.baichuan_api_key).get_secret_value() == "test-api-key"
    assert (
        cast(SecretStr, chat.baichuan_secret_key).get_secret_value()
        == "test-secret-key"
    )
