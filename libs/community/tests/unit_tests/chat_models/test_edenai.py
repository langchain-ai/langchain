"""Test EdenAI Chat API wrapper."""

from typing import List

import pytest
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_community.chat_models.edenai import (
    _extract_edenai_tool_results_from_messages,
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
                "tool_results": [],
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


def test_extract_edenai_tool_results_mixed_messages() -> None:
    fake_other_msg = BaseMessage(content="content", type="other message")
    messages = [
        fake_other_msg,
        ToolMessage(tool_call_id="id1", content="result1"),
        fake_other_msg,
        ToolMessage(tool_call_id="id2", content="result2"),
        ToolMessage(tool_call_id="id3", content="result3"),
    ]
    expected_tool_results = [
        {"id": "id2", "result": "result2"},
        {"id": "id3", "result": "result3"},
    ]
    expected_other_messages = [
        fake_other_msg,
        ToolMessage(tool_call_id="id1", content="result1"),
        fake_other_msg,
    ]
    tool_results, other_messages = _extract_edenai_tool_results_from_messages(messages)
    assert tool_results == expected_tool_results
    assert other_messages == expected_other_messages
