import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_models.lindorm_chat import (
    convert_message_to_dict
)


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"Role": "user", "Content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"Role": "assistant", "Content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"Role": "system", "Content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_function() -> None:
    message = FunctionMessage(name="foo", content="bar")
    with pytest.raises(TypeError) as e:
        convert_message_to_dict(message)
    assert "Got unknown type" in str(e)
