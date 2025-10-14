"""Test conversation chain and memory."""

import re
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.memory import BaseMemory
from langchain_core.prompts.prompt import PromptTemplate
from typing_extensions import override

from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_classic.memory.summary import ConversationSummaryMemory
from tests.unit_tests.llms.fake_llm import FakeLLM


class DummyLLM(LLM):
    last_prompt: str = ""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "dummy"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.last_prompt = prompt
        return "dummy"


def test_memory_ai_prefix() -> None:
    """Test that ai_prefix in the memory component works."""
    memory = ConversationBufferMemory(memory_key="foo", ai_prefix="Assistant")
    memory.save_context({"input": "bar"}, {"output": "foo"})
    assert memory.load_memory_variables({}) == {"foo": "Human: bar\nAssistant: foo"}


def test_memory_human_prefix() -> None:
    """Test that human_prefix in the memory component works."""
    memory = ConversationBufferMemory(memory_key="foo", human_prefix="Friend")
    memory.save_context({"input": "bar"}, {"output": "foo"})
    assert memory.load_memory_variables({}) == {"foo": "Friend: bar\nAI: foo"}


async def test_memory_async() -> None:
    memory = ConversationBufferMemory(memory_key="foo", ai_prefix="Assistant")
    await memory.asave_context({"input": "bar"}, {"output": "foo"})
    assert await memory.aload_memory_variables({}) == {
        "foo": "Human: bar\nAssistant: foo",
    }


async def test_conversation_chain_works() -> None:
    """Test that conversation chain works in basic setting."""
    llm = DummyLLM()
    prompt = PromptTemplate(input_variables=["foo", "bar"], template="{foo} {bar}")
    memory = ConversationBufferMemory(memory_key="foo")
    chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, input_key="bar")
    chain.run("aaa")
    assert llm.last_prompt == " aaa"
    chain.run("bbb")
    assert llm.last_prompt == "Human: aaa\nAI: dummy bbb"
    await chain.arun("ccc")
    assert llm.last_prompt == "Human: aaa\nAI: dummy\nHuman: bbb\nAI: dummy ccc"


def test_conversation_chain_errors_bad_prompt() -> None:
    """Test that conversation chain raise error with bad prompt."""
    llm = FakeLLM()
    prompt = PromptTemplate(input_variables=[], template="nothing here")
    with pytest.raises(
        ValueError, match="Value error, Got unexpected prompt input variables"
    ):
        ConversationChain(llm=llm, prompt=prompt)


def test_conversation_chain_errors_bad_variable() -> None:
    """Test that conversation chain raise error with bad variable."""
    llm = FakeLLM()
    prompt = PromptTemplate(input_variables=["foo"], template="{foo}")
    memory = ConversationBufferMemory(memory_key="foo")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Value error, The input key foo was also found in the memory keys (['foo'])"
        ),
    ):
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
    with pytest.raises(ValueError, match="One input key expected"):
        memory.save_context(bad_inputs, good_outputs)
    # This is a bad input because the only variable is the same as baz.
    bad_inputs = {"baz": "bar"}
    with pytest.raises(ValueError, match=re.escape("One input key expected got []")):
        memory.save_context(bad_inputs, good_outputs)
    # This is a bad output because it is empty.
    with pytest.raises(ValueError, match="Got multiple output keys"):
        memory.save_context(good_inputs, {})
    # This is a bad output because there are two keys.
    bad_outputs = {"foo": "bar", "foo1": "bar"}
    with pytest.raises(ValueError, match="Got multiple output keys"):
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


@pytest.mark.parametrize(
    "memory",
    [
        ConversationBufferMemory(memory_key="baz"),
        ConversationSummaryMemory(llm=FakeLLM(), memory_key="baz"),
        ConversationBufferWindowMemory(memory_key="baz"),
    ],
)
async def test_clearing_conversation_memory_async(memory: BaseMemory) -> None:
    """Test clearing the conversation memory."""
    # This is a good input because the input is not the same as baz.
    good_inputs = {"foo": "bar", "baz": "foo"}
    # This is a good output because there is one variable.
    good_outputs = {"bar": "foo"}
    await memory.asave_context(good_inputs, good_outputs)

    await memory.aclear()
    assert await memory.aload_memory_variables({}) == {"baz": ""}
