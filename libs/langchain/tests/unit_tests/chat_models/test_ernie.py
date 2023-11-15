import pytest

from langchain.chat_models.ernie import _convert_message_to_dict
from langchain.schema.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


def test__convert_dict_to_message_human() -> None:
    message = HumanMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"role": "user", "content": "foo"}
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = AIMessage(content="foo")
    result = _convert_message_to_dict(message)
    expected_output = {"role": "assistant", "content": "foo"}
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = SystemMessage(content="foo")
    with pytest.raises(ValueError) as e:
        _convert_message_to_dict(message)
    assert "Got unknown type" in str(e)


def test__convert_dict_to_message_function() -> None:
    message = FunctionMessage(name="foo", content="bar")
    with pytest.raises(ValueError) as e:
        _convert_message_to_dict(message)
    assert "Got unknown type" in str(e)
