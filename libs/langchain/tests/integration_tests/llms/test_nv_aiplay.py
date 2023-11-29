"""Test AI Playground API Wrapper. Inspired by Fireworks"""

from typing import Generator

import pytest

from langchain.chains import LLMChain
from langchain.llms.nv_aiplay import LlamaLLM, MistralLLM, NemotronQA, SteerLM
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import LLMResult


@pytest.fixture
def llm() -> LlamaLLM:
    return LlamaLLM(model_kwargs={"temperature": 0, "max_tokens": 512})


@pytest.mark.scheduled
def test_aiplay_call(llm: LlamaLLM) -> None:
    """Test valid call to aiplay."""
    output = llm("How is the weather in New York today?")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_aiplay_in_chain() -> None:
    """Tests aiplay AI in a Langchain chain"""
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat = LlamaLLM()
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    output = chain.run("football helmets")
    assert isinstance(output, str)


@pytest.mark.scheduled
def test_aiplay_model_param() -> None:
    """Tests model parameters for LlamaLLM"""
    llm = LlamaLLM(model_name="mistral")
    assert llm.model_name == "mistral"
    llm = LlamaLLM(model="mistral")
    assert llm.model_name == "mistral"


@pytest.mark.scheduled
def test_aiplay_invoke(llm: LlamaLLM) -> None:
    """Tests completion with invoke"""
    output = llm.invoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_ainvoke(llm: LlamaLLM) -> None:
    """Tests completion with invoke"""
    output = await llm.ainvoke("How is the weather in New York today?", stop=[","])
    assert isinstance(output, str)
    assert output[-1] == ","


@pytest.mark.scheduled
def test_aiplay_batch(llm: LlamaLLM) -> None:
    """Tests completion with invoke"""
    llm = MistralLLM()
    output = llm.batch(
        [
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
        ],
        stop=[","],
    )
    for token in output:
        assert isinstance(token, str)
        assert token[-1] == ","


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch(llm: LlamaLLM) -> None:
    """Tests completion with invoke"""
    output = await llm.abatch(
        [
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
            "Give me some example usages of commas",
        ],
        stop=[","],
    )
    for token in output:
        assert isinstance(token, str)
        assert token[-1] == ","


@pytest.mark.scheduled
def test_aiplay_multiple_prompts(
    llm: LlamaLLM,
) -> None:
    """Test completion with multiple prompts."""
    output = llm.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


@pytest.mark.scheduled
def test_aiplay_streaming(llm: LlamaLLM) -> None:
    """Test stream completion."""
    generator = llm.stream("Who's the best quarterback in the NFL?")
    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


@pytest.mark.scheduled
def test_aiplay_streaming_stop_words(llm: LlamaLLM) -> None:
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
async def test_aiplay_streaming_async(llm: LlamaLLM) -> None:
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
async def test_aiplay_async_agenerate(llm: LlamaLLM) -> None:
    """Test async."""
    output = await llm.agenerate(["What is the best city to live in California?"])
    assert isinstance(output, LLMResult)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_multiple_prompts_async_agenerate(llm: LlamaLLM) -> None:
    output = await llm.agenerate(
        ["How is the weather in New York today?", "I'm pickle rick"]
    )
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


@pytest.mark.scheduled
def test_aiplay_steerlm(llm: LlamaLLM) -> None:
    """Test completion with multiple prompts."""
    llm2 = SteerLM()
    output = llm2.generate(["How is the weather in New York today?", "I'm pickle rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


@pytest.mark.scheduled
def test_aiplay_nemotron_qa(llm: LlamaLLM) -> None:
    """Test completion with multiple prompts."""
    llm2 = NemotronQA()
    output = llm2.generate(
        [
            """
        ///ROLE USER: How is the weather in New York today?
        ///ROLE CONTEXT: There is extreme heat in New York today, and some extreme 
        cold in the south. The east is underwater. There is panic everywhere.
    """
        ]
    )
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 1
