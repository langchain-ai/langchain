"""Test TokenBufferMemory."""
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_token_buffer_memory() -> None:
    """Test ConversationTokenBufferMemory with a max_token_limit of 6."""
    memory = ConversationTokenBufferMemory(
        llm=FakeLLM(), memory_key="baz", max_token_limit=6
    )
    memory.save_context({"input": "bar"}, {"output": "foo"})
    output = memory.load_memory_variables({})
    # Check the initial conversation.
    assert output == {"baz": "Human: bar\nAI: foo"}
    memory.save_context({"input": "bar1"}, {"output": "foo1"})
    # Check the pruned conversation after exceeding the token limit.
    output = memory.load_memory_variables({})
    assert output == {"baz": "AI: foo\nHuman: bar1\nAI: foo1"}
