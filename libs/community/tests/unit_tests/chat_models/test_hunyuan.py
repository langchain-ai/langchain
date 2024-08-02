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

from langchain_community.chat_models.hunyuan import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"Role": "user", "Content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"Role": "assistant", "Content": "foo"}
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
    message_dict = {"Role": "user", "Content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message_dict = {"Role": "assistant", "Content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_other_role() -> None:
    message_dict = {"Role": "system", "Content": "foo"}
    result = _convert_dict_to_message(message_dict)
    expected_output = ChatMessage(role="system", content="foo")
    assert result == expected_output


def test__convert_delta_to_message_assistant() -> None:
    delta = {"Role": "assistant", "Content": "foo"}
    result = _convert_delta_to_message_chunk(delta, AIMessageChunk)
    expected_output = AIMessageChunk(content="foo")
    assert result == expected_output


def test__convert_delta_to_message_human() -> None:
    delta = {"Role": "user", "Content": "foo"}
    result = _convert_delta_to_message_chunk(delta, HumanMessageChunk)
    expected_output = HumanMessageChunk(content="foo")
    assert result == expected_output
