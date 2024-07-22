from typing import cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
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


def test_initialization() -> None:
    """Test chat model initialization."""

    for model in [
        ChatBaichuan(model="Baichuan2-Turbo-192K", api_key="test-api-key", timeout=40),  # type: ignore[arg-type, call-arg]
        ChatBaichuan(  # type: ignore[call-arg]
            model="Baichuan2-Turbo-192K",
            baichuan_api_key="test-api-key",
            request_timeout=40,
        ),
    ]:
        assert model.model == "Baichuan2-Turbo-192K"
        assert isinstance(model.baichuan_api_key, SecretStr)
        assert model.request_timeout == 40
        assert model.temperature == 0.3


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
    result = _convert_message_to_dict(message)
    expected_output = {"role": "system", "content": "foo"}
    assert result == expected_output


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
    expected_output = SystemMessage(content="foo")
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

    chat = ChatBaichuan()  # type: ignore[call-arg]
    print(chat.baichuan_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_baichuan_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = ChatBaichuan(baichuan_api_key="test-api-key")  # type: ignore[call-arg]
    print(chat.baichuan_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = ChatBaichuan(  # type: ignore[call-arg]
        baichuan_api_key="test-api-key",
        baichuan_secret_key="test-secret-key",  # type: ignore[arg-type] # For backward compatibility
    )
    assert cast(SecretStr, chat.baichuan_api_key).get_secret_value() == "test-api-key"
    assert (
        cast(SecretStr, chat.baichuan_secret_key).get_secret_value()
        == "test-secret-key"
    )


def test_chat_baichuan_with_base_url() -> None:
    chat = ChatBaichuan(  # type: ignore[call-arg]
        api_key="your-api-key",  # type: ignore[arg-type]
        base_url="https://exmaple.com",  # type: ignore[arg-type]
    )
    assert chat.baichuan_api_base == "https://exmaple.com"
