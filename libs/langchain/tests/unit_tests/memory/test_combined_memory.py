"""Test for CombinedMemory class"""
# from langchain.prompts import PromptTemplate
from typing import List

import pytest

from langchain.memory import CombinedMemory, ConversationBufferMemory


@pytest.fixture()
def example_memory() -> List[ConversationBufferMemory]:
    example_1 = ConversationBufferMemory(memory_key="foo")
    example_2 = ConversationBufferMemory(memory_key="bar")
    example_3 = ConversationBufferMemory(memory_key="bar")
    return [example_1, example_2, example_3]


def test_basic_functionality(example_memory: List[ConversationBufferMemory]) -> None:
    """Test basic functionality of methods exposed by class"""
    combined_memory = CombinedMemory(memories=[example_memory[0], example_memory[1]])
    assert combined_memory.memory_variables == ["foo", "bar"]
    assert combined_memory.load_memory_variables({}) == {"foo": "", "bar": ""}
    combined_memory.save_context(
        {"input": "Hello there"}, {"output": "Hello, how can I help you?"}
    )
    assert combined_memory.load_memory_variables({}) == {
        "foo": "Human: Hello there\nAI: Hello, how can I help you?",
        "bar": "Human: Hello there\nAI: Hello, how can I help you?",
    }
    combined_memory.clear()
    assert combined_memory.load_memory_variables({}) == {"foo": "", "bar": ""}


def test_repeated_memory_var(example_memory: List[ConversationBufferMemory]) -> None:
    """Test raising error when repeated memory variables found"""
    with pytest.raises(ValueError):
        CombinedMemory(memories=[example_memory[1], example_memory[2]])
