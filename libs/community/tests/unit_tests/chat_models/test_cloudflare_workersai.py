"""Test CloudflareWorkersAI Chat API wrapper."""

from typing import Any, Dict, List, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_community.chat_models.cloudflare_workersai import (
    ChatCloudflareWorkersAI,
    _convert_messages_to_cloudflare_messages,
)


class TestChatCloudflareWorkersAI(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCloudflareWorkersAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "account_id": "my_account_id",
            "api_token": "my_api_token",
            "model": "@hf/nousresearch/hermes-2-pro-mistral-7b",
        }


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # Test case with a single HumanMessage
        (
            [HumanMessage(content="Hello, AI!")],
            [{"role": "user", "content": "Hello, AI!"}],
        ),
        # Test case with SystemMessage, HumanMessage, and AIMessage without tool calls
        (
            [
                SystemMessage(content="System initialized."),
                HumanMessage(content="Hello, AI!"),
                AIMessage(content="Response from AI"),
            ],
            [
                {"role": "system", "content": "System initialized."},
                {"role": "user", "content": "Hello, AI!"},
                {"role": "assistant", "content": "Response from AI"},
            ],
        ),
        # Test case with ToolMessage and tool_call_id
        (
            [
                ToolMessage(
                    content="Tool message content", tool_call_id="tool_call_123"
                ),
            ],
            [
                {
                    "role": "tool",
                    "content": "Tool message content",
                    "tool_call_id": "tool_call_123",
                }
            ],
        ),
    ],
)
def test_convert_messages_to_cloudflare_format(
    messages: List[BaseMessage], expected: List[Dict[str, Any]]
) -> None:
    result = _convert_messages_to_cloudflare_messages(messages)
    assert result == expected
