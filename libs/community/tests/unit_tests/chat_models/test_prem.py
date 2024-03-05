"""Test PremChat model"""

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture
from langchain_community.chat_models import ChatPrem
from langchain_community.chat_models.prem import _messages_to_prompt_dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@pytest.mark.requires("premai")
def test_api_key_is_string() -> None:
    llm = ChatPrem(premai_api_key="secret-api-key", project_id=8)
    assert isinstance(llm.premai_api_key, SecretStr)


@pytest.mark.requires("premai")
def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatPrem(premai_api_key="secret-api-key", project_id=8)
    print(llm.premai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_messages_to_prompt_dict_with_valid_messages() -> None:
    system_message, result = _messages_to_prompt_dict(
        [
            SystemMessage(content="System Prompt"),
            HumanMessage(content="User message #1"),
            AIMessage(content="AI message #1"),
            HumanMessage(content="User message #2"),
            AIMessage(content="AI message #2"),
        ]
    )
    expected = [
        {"role": "user", "content": "User message #1"},
        {"role": "assistant", "content": "AI message #1"},
        {"role": "user", "content": "User message #2"},
        {"role": "assistant", "content": "AI message #2"},
    ]

    assert system_message == "System Prompt"
    assert result == expected


def test_premchat_raises_with_parameter_n() -> None:
    """Since param: n is not supported right now, we need to take that edge case under consideration"""

    messages = [[HumanMessage(content="hello")]]
    with pytest.raises(NotImplementedError) as error_msg:
        ChatPrem(premai_api_key="fake", project_id=8).generate(messages=messages, n=2)

    assert "parameter: n is not supported for now." in str(error_msg)


def test_premchat_raises_with_parameter_stop() -> None:
    """Since param: stop is not supported right now, we need to take that edge case under consideration"""

    messages = [[HumanMessage(content="hello")]]
    with pytest.raises(NotImplementedError) as error_msg:
        ChatPrem(premai_api_key="fake", project_id=8).generate(
            messages=messages, stop=["stop"]
        )

    assert "Parameter: stop has no support yet" in str(error_msg)
