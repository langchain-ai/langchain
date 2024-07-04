"""Test EdenAI Chat API wrapper."""

from typing import List

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.dappier import _format_dappier_messages


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [
                SystemMessage(
                    content="You are a chat model with real time search tools"
                ),
                HumanMessage(content="Hello how are you today?"),
            ],
            [
                {
                    "role": "system",
                    "content": "You are a chat model with real time search tools",
                },
                {"role": "user", "content": "Hello how are you today?"},
            ],
        )
    ],
)
def test_dappier_messages_formatting(
    messages: List[BaseMessage], expected: str
) -> None:
    result = _format_dappier_messages(messages)
    assert result == expected
