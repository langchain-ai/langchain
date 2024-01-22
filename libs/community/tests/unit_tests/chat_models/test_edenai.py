"""Test EdenAI Chat API wrapper."""
from typing import List

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.edenai import (
    _format_edenai_messages,
    _message_role,
)


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
def test_edenai_message_role(role: str, role_response) -> None:
    role = _message_role(role)
    assert role == role_response
