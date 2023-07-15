import pytest

from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.chains import LLMChain
from langchain.memory import ReadOnlySharedMemory, SimpleMemory
from langchain.schema import BaseMemory
from langchain.prompts import PromptTemplate
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_chain_with_memory_and_multiple_prompt_params() -> None:
    template = """
    {resource}

    {history}
    Human: {human_input}
    AI: 
    """

    prompt = PromptTemplate(input_variables=["resource", "history", "human_input"], template=template)
    memory = ConversationBufferMemory(memory_key="history")
    llm_chain = LLMChain(llm=FakeLLM(), prompt=prompt, verbose=True, memory=memory)
    resource="lorem ipsum"
    print(llm_chain.predict(human_input="Bar?", resource=resource) + "\n")
    print(llm_chain.predict(human_input="Bazz?", resource=resource) + "\n")

def test_chain_with_memory_and_multiple_prompt_params_and_input_key() -> None:
    template = """
    {resource}

    {history}
    Human: {human_input}
    AI: 
    """

    prompt = PromptTemplate(input_variables=["resource", "history", "human_input"], template=template)
    memory = ConversationBufferMemory(memory_key="history", input_key="human_input")
    llm_chain = LLMChain(llm=FakeLLM(), prompt=prompt, verbose=True, memory=memory)
    resource="lorem ipsum"
    print(llm_chain.predict(human_input="Bar?", resource=resource) + "\n")
    print(llm_chain.predict(human_input="Bazz?", resource=resource) + "\n")


def test_chain_with_multiple_prompt_params() -> None:
    template = """
    {resource}

    Human: {human_input}
    AI: 
    """

    prompt = PromptTemplate(input_variables=["resource", "human_input"], template=template)
    llm_chain = LLMChain(llm=FakeLLM(), prompt=prompt, verbose=True)
    resource="lorem ipsum"
    print(llm_chain.predict(human_input="Bar?", resource=resource) + "\n")
    print(llm_chain.predict(human_input="Bazz?", resource=resource) + "\n")

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
