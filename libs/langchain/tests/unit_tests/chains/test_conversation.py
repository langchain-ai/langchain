"""Test conversation chain and memory."""
import pytest

from langchain.chains.conversation.base import ConversationChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.summary import ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_memory_ai_prefix() -> None:
    """Test that ai_prefix in the memory component works."""
    memory = ConversationBufferMemory(memory_key="foo", ai_prefix="Assistant")
    memory.save_context({"input": "bar"}, {"output": "foo"})
    assert memory.buffer == "Human: bar\nAssistant: foo"


def test_memory_human_prefix() -> None:
    """Test that human_prefix in the memory component works."""
    memory = ConversationBufferMemory(memory_key="foo", human_prefix="Friend")
    memory.save_context({"input": "bar"}, {"output": "foo"})
    assert memory.buffer == "Friend: bar\nAI: foo"


def test_conversation_chain_works() -> None:
    """Test that conversation chain works in basic setting."""
    llm = FakeLLM()
    prompt = PromptTemplate(input_variables=["foo", "bar"], template="{foo} {bar}")
    memory = ConversationBufferMemory(memory_key="foo")
    chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, input_key="bar")
    chain.run("foo")


def test_conversation_chain_errors_bad_prompt() -> None:
    """Test that conversation chain raise error with bad prompt."""
    llm = FakeLLM()
    prompt = PromptTemplate(input_variables=[], template="nothing here")
    with pytest.raises(ValueError):
        ConversationChain(llm=llm, prompt=prompt)


def test_conversation_chain_errors_bad_variable() -> None:
    """Test that conversation chain raise error with bad variable."""
    llm = FakeLLM()
    prompt = PromptTemplate(input_variables=["foo"], template="{foo}")
    memory = ConversationBufferMemory(memory_key="foo")
    with pytest.raises(ValueError):
        ConversationChain(llm=llm, prompt=prompt, memory=memory, input_key="foo")


@pytest.mark.parametrize(
    "memory",
    [
        ConversationBufferMemory(memory_key="baz"),
        ConversationBufferWindowMemory(memory_key="baz"),
        ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"),
    ],
)
def test_conversation_memory(memory: BaseMemory) -> None:
    """Test basic conversation memory functionality."""
    # This is a good input because the input is not the same as baz.
    good_inputs = {"foo": "bar", "baz": "foo"}
    # This is a good output because these is one variable.
    good_outputs = {"bar": "foo"}
    memory.save_context(good_inputs, good_outputs)
    # This is a bad input because there are two variables that aren't the same as baz.
    bad_inputs = {"foo": "bar", "foo1": "bar"}
    with pytest.raises(ValueError):
        memory.save_context(bad_inputs, good_outputs)
    # This is a bad input because the only variable is the same as baz.
    bad_inputs = {"baz": "bar"}
    with pytest.raises(ValueError):
        memory.save_context(bad_inputs, good_outputs)
    # This is a bad output because it is empty.
    with pytest.raises(ValueError):
        memory.save_context(good_inputs, {})
    # This is a bad output because there are two keys.
    bad_outputs = {"foo": "bar", "foo1": "bar"}
    with pytest.raises(ValueError):
        memory.save_context(good_inputs, bad_outputs)


@pytest.mark.parametrize(
    "memory",
    [
        ConversationBufferMemory(memory_key="baz"),
        ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"),
        ConversationBufferWindowMemory(memory_key="baz"),
    ],
)
def test_clearing_conversation_memory(memory: BaseMemory) -> None:
    """Test clearing the conversation memory."""
    # This is a good input because the input is not the same as baz.
    good_inputs = {"foo": "bar", "baz": "foo"}
    # This is a good output because there is one variable.
    good_outputs = {"bar": "foo"}
    memory.save_context(good_inputs, good_outputs)

    memory.clear()
    assert memory.load_memory_variables({}) == {"baz": ""}
