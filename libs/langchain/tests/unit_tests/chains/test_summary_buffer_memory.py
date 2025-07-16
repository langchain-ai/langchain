"""Test memory functionality."""

from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_summary_buffer_memory_no_buffer_yet() -> None:
    """Test ConversationSummaryBufferMemory when no inputs put in buffer yet."""
    memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
    output = memory.load_memory_variables({})
    assert output == {"baz": ""}


async def test_summary_buffer_memory_no_buffer_yet_async() -> None:
    """Test ConversationSummaryBufferMemory when no inputs put in buffer yet."""
    memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
    output = await memory.aload_memory_variables({})
    assert output == {"baz": ""}


def test_summary_buffer_memory_buffer_only() -> None:
    """Test ConversationSummaryBufferMemory when only buffer."""
    memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
    memory.save_context({"input": "bar"}, {"output": "foo"})
    assert memory.buffer == "Human: bar\nAI: foo"
    output = memory.load_memory_variables({})
    assert output == {"baz": "Human: bar\nAI: foo"}


async def test_summary_buffer_memory_buffer_only_async() -> None:
    """Test ConversationSummaryBufferMemory when only buffer."""
    memory = ConversationSummaryBufferMemory(llm=FakeLLM(), memory_key="baz")
    await memory.asave_context({"input": "bar"}, {"output": "foo"})
    assert memory.buffer == "Human: bar\nAI: foo"
    output = await memory.aload_memory_variables({})
    assert output == {"baz": "Human: bar\nAI: foo"}


def test_summary_buffer_memory_summary() -> None:
    """Test ConversationSummaryBufferMemory when only buffer."""
    llm = FakeLLM(queries={0: "summary"}, sequential_responses=True)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="baz",
        max_token_limit=5,
    )
    memory.save_context({"input": "bar"}, {"output": "foo"})
    memory.save_context({"input": "bar1"}, {"output": "foo1"})
    assert memory.buffer == "System: summary\nHuman: bar1\nAI: foo1"
    output = memory.load_memory_variables({})
    assert output == {"baz": "System: summary\nHuman: bar1\nAI: foo1"}


async def test_summary_buffer_memory_summary_async() -> None:
    """Test ConversationSummaryBufferMemory when only buffer."""
    llm = FakeLLM(queries={0: "summary"}, sequential_responses=True)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="baz",
        max_token_limit=5,
    )
    await memory.asave_context({"input": "bar"}, {"output": "foo"})
    await memory.asave_context({"input": "bar1"}, {"output": "foo1"})
    assert memory.buffer == "System: summary\nHuman: bar1\nAI: foo1"
    output = await memory.aload_memory_variables({})
    assert output == {"baz": "System: summary\nHuman: bar1\nAI: foo1"}
