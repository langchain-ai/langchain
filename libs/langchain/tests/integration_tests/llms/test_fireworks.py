"""Test Fireworks AI API Wrapper."""
import sys
from typing import Generator

import pytest

from langchain.chains import LLMChain
from langchain.llms.fireworks import Fireworks
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import LLMResult

if sys.version_info < (3, 9):
    pytest.skip("fireworks-ai requires Python > 3.8", allow_module_level=True)


@pytest.fixture
def llm() -> Fireworks:
    return Fireworks(model_kwargs={"temperature": 0, "max_tokens": 512})


@pytest.mark.scheduled
def test_fireworks_call(llm: Fireworks) -> None:
    """Test valid call to fireworks."""
    output = llm("How is the weather in New York today?")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_fireworks_in_chain() -> None:
    """Tests fireworks AI in a Langchain chain"""
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat = Fireworks()
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    output = chain.run("football helmets")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_fireworks_model_param() -> None:
    """Tests model parameters for Fireworks"""
    llm = Fireworks(model="foo")
    assert llm.model == "foo"


@pytest.mark.scheduled
def test_fireworks_invoke(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    output = llm.invoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_ainvoke(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    output = await llm.ainvoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","


@pytest.mark.scheduled
def test_fireworks_batch(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    llm = Fireworks()
    output = llm.batch(
        [
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
        ],
        stop=[","],
    )
    for token in output:
        assert isinstance(token, str)
        assert token[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_abatch(llm: Fireworks) -> None:
    """Tests completion with invoke"""
    output = await llm.abatch(
        [
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
            "How is the weather in New York today?",
        ],
        stop=[","],
    )
    for token in output:
        assert isinstance(token, str)
        assert token[-1] == ","


@pytest.mark.scheduled
def test_fireworks_multiple_prompts(
    llm: Fireworks,
) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


@pytest.mark.scheduled
def test_fireworks_streaming(llm: Fireworks) -> None:
    """Test stream completion."""
    generator = llm.stream("Who's the best quarterback in the NFL?")
    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@pytest.mark.scheduled
def test_fireworks_streaming_stop_words(llm: Fireworks) -> None:
    """Test stream completion with stop words."""
    generator = llm.stream("Who's the best quarterback in the NFL?", stop=[","])
    assert isinstance(generator, Generator)

    last_token = ""
    for token in generator:
        last_token = token
        assert isinstance(token, str)
    assert last_token[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_streaming_async(llm: Fireworks) -> None:
    """Test stream completion."""

    last_token = ""
    async for token in llm.astream(
        "Who's the best quarterback in the NFL?", stop=[","]
    ):
        last_token = token
        assert isinstance(token, str)
    assert last_token[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_async_agenerate(llm: Fireworks) -> None:
    """Test async."""
    output = await llm.agenerate(["What is the best city to live in California?"])
    assert isinstance(output, LLMResult)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_fireworks_multiple_prompts_async_agenerate(llm: Fireworks) -> None:
    output = await llm.agenerate(
        ["How is the weather in New York today?", "I'm pickle rick"]
    )
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2
