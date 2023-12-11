"""Test Anthropic Chat API wrapper."""
from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_community.chat_models import BedrockChat
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        ([HumanMessage(content="Hello")], "[INST] Hello [/INST]"),
        (
            [HumanMessage(content="Hello"), AIMessage(content="Answer:")],
            "[INST] Hello [/INST]\nAnswer:",
        ),
        (
            [
                SystemMessage(content="You're an assistant"),
                HumanMessage(content="Hello"),
                AIMessage(content="Answer:"),
            ],
            "<<SYS>> You're an assistant <</SYS>>\n[INST] Hello [/INST]\nAnswer:",
        ),
    ],
)
def test_formatting(messages: List[BaseMessage], expected: str) -> None:
    result = convert_messages_to_prompt_llama(messages)
    assert result == expected


def test_anthropic_bedrock() -> None:
    client = MagicMock()
    respbody = MagicMock(
        read=MagicMock(
            return_value=MagicMock(
                decode=MagicMock(return_value=b'{"completion":"Hi back"}')
            )
        )
    )
    client.invoke_model.return_value = {"body": respbody}
    model = BedrockChat(model_id="anthropic.claude-v2", client=client)

    # should not throw an error
    model.invoke("hello there")
