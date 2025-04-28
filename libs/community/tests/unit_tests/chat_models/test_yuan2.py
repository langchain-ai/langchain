"""Test ChatYuan2 wrapper."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_models.yuan2 import (
    ChatYuan2,
    _convert_dict_to_message,
    _convert_message_to_dict,
)


@pytest.mark.requires("openai")
def test_yuan2_model_param() -> None:
    chat = ChatYuan2(model="foo")
    assert chat.model_name == "foo"
    chat = ChatYuan2(model_name="foo")  # type: ignore[call-arg]
    assert chat.model_name == "foo"


@pytest.mark.requires("openai")
def test_yuan2_timeout_param() -> None:
    chat = ChatYuan2(request_timeout=5)  # type: ignore[call-arg]
    assert chat.request_timeout == 5
    chat = ChatYuan2(timeout=10)
    assert chat.request_timeout == 10


@pytest.mark.requires("openai")
def test_yuan2_stop_sequences_param() -> None:
    chat = ChatYuan2(stop=["<eod>"])  # type: ignore[call-arg]
    assert chat.stop == ["<eod>"]
    chat = ChatYuan2(stop_sequences=["<eod>"])
    assert chat.stop == ["<eod>"]


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


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "hello"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="hello")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "hello"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="hello")
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "hello"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="hello")
    assert result == expected_output
