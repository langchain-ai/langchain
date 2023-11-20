"""Test Anthropic Chat API wrapper."""
from typing import List

import pytest

from langchain.chat_models.meta import convert_messages_to_prompt_llama
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage


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
