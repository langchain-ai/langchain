"""Test Google PaLM Chat API wrapper."""

import pytest

from langchain.chat_models.google_palm import (
    ChatGooglePalm,
    ChatGooglePalmError,
    _messages_to_prompt_dict,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


@pytest.mark.requires("google.generativeai")
def test_messages_to_prompt_dict_with_valid_messages() -> None:
    result = _messages_to_prompt_dict(
        [
            SystemMessage(content="Prompt"),
            HumanMessage(example=True, content="Human example #1"),
            AIMessage(example=True, content="AI example #1"),
            HumanMessage(example=True, content="Human example #2"),
            AIMessage(example=True, content="AI example #2"),
            HumanMessage(content="Real human message"),
            AIMessage(content="Real AI message"),
        ]
    )
    expected = {
        "context": "Prompt",
        "examples": [
            {"author": "human", "content": "Human example #1"},
            {"author": "ai", "content": "AI example #1"},
            {"author": "human", "content": "Human example #2"},
            {"author": "ai", "content": "AI example #2"},
        ],
        "messages": [
            {"author": "human", "content": "Real human message"},
            {"author": "ai", "content": "Real AI message"},
        ],
    }

    assert result == expected


@pytest.mark.requires("google.generativeai")
def test_messages_to_prompt_dict_raises_with_misplaced_system_message() -> None:
    with pytest.raises(ChatGooglePalmError) as e:
        _messages_to_prompt_dict(
            [
                HumanMessage(content="Real human message"),
                SystemMessage(content="Prompt"),
            ]
        )
    assert "System message must be first" in str(e)


@pytest.mark.requires("google.generativeai")
def test_messages_to_prompt_dict_raises_with_misordered_examples() -> None:
    with pytest.raises(ChatGooglePalmError) as e:
        _messages_to_prompt_dict(
            [
                AIMessage(example=True, content="AI example #1"),
                HumanMessage(example=True, content="Human example #1"),
            ]
        )
    assert "AI example message must be immediately preceded" in str(e)


@pytest.mark.requires("google.generativeai")
def test_messages_to_prompt_dict_raises_with_mismatched_examples() -> None:
    with pytest.raises(ChatGooglePalmError) as e:
        _messages_to_prompt_dict(
            [
                HumanMessage(example=True, content="Human example #1"),
                AIMessage(example=False, content="AI example #1"),
            ]
        )
    assert "Human example message must be immediately followed" in str(e)


@pytest.mark.requires("google.generativeai")
def test_messages_to_prompt_dict_raises_with_example_after_real() -> None:
    with pytest.raises(ChatGooglePalmError) as e:
        _messages_to_prompt_dict(
            [
                HumanMessage(example=False, content="Real message"),
                HumanMessage(example=True, content="Human example #1"),
                AIMessage(example=True, content="AI example #1"),
            ]
        )
    assert "Message examples must come before other" in str(e)


@pytest.mark.requires("google.generativeai")
def test_chat_google_raises_with_invalid_temperature() -> None:
    with pytest.raises(ValueError) as e:
        ChatGooglePalm(google_api_key="fake", temperature=2.0)
    assert "must be in the range" in str(e)


@pytest.mark.requires("google.generativeai")
def test_chat_google_raises_with_invalid_top_p() -> None:
    with pytest.raises(ValueError) as e:
        ChatGooglePalm(google_api_key="fake", top_p=2.0)
    assert "must be in the range" in str(e)


@pytest.mark.requires("google.generativeai")
def test_chat_google_raises_with_invalid_top_k() -> None:
    with pytest.raises(ValueError) as e:
        ChatGooglePalm(google_api_key="fake", top_k=-5)
    assert "must be positive" in str(e)
