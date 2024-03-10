import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_models.llama_edge import (
    LlamaEdgeChatService,
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
    expected_output = ChatMessage(role="system", content="foo")
    assert result == expected_output


def test_wasm_chat_without_service_url() -> None:
    chat = LlamaEdgeChatService()

    # create message sequence
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    with pytest.raises(ValueError) as e:
        chat(messages)

    assert "Error code: 503" in str(e)
    assert "reason: The IP address or port of the chat service is incorrect." in str(e)
