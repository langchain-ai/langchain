import pytest

from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.memory import ReadOnlySharedMemory, SimpleMemory
from langchain.schema import BaseMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_simple_memory() -> None:
    """Test SimpleMemory."""
    memory = SimpleMemory(memories={"baz": "foo"})

    output = memory.load_memory_variables({})

    assert output == {"baz": "foo"}
    assert ["baz"] == memory.memory_variables


@pytest.mark.parametrize(
    "memory",
    [
        ConversationBufferMemory(memory_key="baz"),
        ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"),
        ConversationBufferWindowMemory(memory_key="baz"),
    ],
)
def test_readonly_memory(memory: BaseMemory) -> None:
    read_only_memory = ReadOnlySharedMemory(memory=memory)
    memory.save_context({"input": "bar"}, {"output": "foo"})

    assert read_only_memory.load_memory_variables({}) == memory.load_memory_variables(
        {}
    )
