"""Test for Serialization of memory"""

from typing import List

import pytest
import json

from langchain.memory import CombinedMemory, ConversationBufferMemory, ChatMessageHistory
from langchain.schema import ChatMessage, HumanMessage, AIMessage, SystemMessage


@pytest.fixture()
def example_memory() -> ConversationBufferMemory:
    messages  = [
        SystemMessage(content="This is a system message"),
        AIMessage(content="This is an AI message"),
        HumanMessage(content="This is a human message"),
        ChatMessage(role="foo", content="This is a chat message")
    ]
    history = ChatMessageHistory(messages=messages)
    memory = ConversationBufferMemory(chat_memory=history)
    return memory
    


def test_serialization(example_memory: ConversationBufferMemory) -> None:
    """Test serialization and deserialization of ConversationBufferMemory"""
    serialized_memory = example_memory.to_dict()
    assert type(serialized_memory)==dict
    assert "chat_memory" in serialized_memory
    assert "messages" in serialized_memory["chat_memory"]
        

