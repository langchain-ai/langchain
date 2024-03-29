from typing import Any

import pytest
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.memory.chat_memory import BaseChatMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.parametrize(
    "memory, output_obj",
    [
        (ConversationBufferMemory(memory_key="baz"), "bar"),
        (ConversationBufferMemory(memory_key="baz"), {"bar": "qux"}),
        (ConversationBufferMemory(memory_key="baz"), ["bar", "qux"]),
        (ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"), "bar"),
        (ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"), {"bar": "qux"}),
        (ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"), ["bar", "qux"]),
        (ConversationBufferWindowMemory(memory_key="baz"), "bar"),
        (ConversationBufferWindowMemory(memory_key="baz"), {"bar": "qux"}),
        (ConversationBufferWindowMemory(memory_key="baz"), ["bar", "qux"]),
    ],
)
def test_memory_save_context(memory: BaseChatMemory, output_obj: Any) -> None:
    # all message content must be strings
    memory.save_context({"input": "foo"}, {"output": output_obj})
    assert len(memory.chat_memory.messages) == 2
    assert memory.chat_memory.messages[0] == HumanMessage(content="foo")
    assert memory.chat_memory.messages[1] == AIMessage(content=str(output_obj))
