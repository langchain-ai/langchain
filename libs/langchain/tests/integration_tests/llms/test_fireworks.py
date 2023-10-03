"""Test Fireworks AI API Wrapper."""
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


def test_fireworks_call() -> None:
    """Test valid call to fireworks."""
    llm = Fireworks()
    output = llm("Who's the best quarterback in the NFL?")
    assert isinstance(output, str)


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


def test_fireworks_model_param() -> None:
    """Tests model parameters for Fireworks"""
    llm = Fireworks(model="foo")
    assert llm.model == "foo"


def test_fireworks_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = Fireworks()
    output = llm.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


def test_fireworks_streaming() -> None:
    """Test stream completion."""
    llm = Fireworks()
    generator = llm.stream("Who's the best quarterback in the NFL?")
    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@pytest.mark.asyncio
async def test_fireworks_streaming_async() -> None:
    """Test stream completion."""
    llm = Fireworks()

    async for token in llm.astream("Who's the best quarterback in the NFL?"):
        assert isinstance(token, str)


@pytest.mark.asyncio
async def test_fireworks_async_agenerate() -> None:
    """Test async."""
    llm = Fireworks()
    output = await llm.agenerate(["What is the best city to live in California?"])
    assert isinstance(output, LLMResult)


@pytest.mark.asyncio
async def test_fireworks_multiple_prompts_async_agenerate() -> None:
    llm = Fireworks()
    output = await llm.agenerate(
        ["How is the weather in New York today?", "I'm pickle rick"]
    )
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2
