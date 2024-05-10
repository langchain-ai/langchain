"""Test EdenAI Chat API wrapper."""

from typing import List

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.edenai import (
    ChatEdenAI,
    _format_edenai_messages,
    _message_role,
)


def test_edenai_initialization() -> None:
    chat = ChatEdenAI(
        api_key="dummy_api_key",
        model="gpt-4",
        temperature=0.5,
        max_tokens=100,
    )
    if chat.edenai_api_key is not None:
        assert chat.edenai_api_key.get_secret_value() == "dummy_api_key"
    else:
        raise AssertionError("edenai_api_key is None")
    assert chat.model == "gpt-4"
    assert chat.temperature == 0.5
    assert chat.max_tokens == 100


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [
                SystemMessage(content="Translate the text from English to French"),
                HumanMessage(content="Hello how are you today?"),
            ],
            {
                "text": "Hello how are you today?",
                "previous_history": [],
                "chatbot_global_action": "Translate the text from English to French",
            },
        )
    ],
)
def test_edenai_messages_formatting(messages: List[BaseMessage], expected: str) -> None:
    result = _format_edenai_messages(messages)
    assert result == expected


@pytest.mark.parametrize(
    ("role", "role_response"),
    [("ai", "assistant"), ("human", "user"), ("chat", "user")],
)
def test_edenai_message_role(role: str, role_response) -> None:  # type: ignore[no-untyped-def]
    role = _message_role(role)
    assert role == role_response
