"""Test Friendli API."""

import pytest
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.llm_result import LLMResult

from langchain_community.llms.friendli import Friendli


@pytest.fixture
def friendli_llm() -> Friendli:
    """Friendli LLM."""
    return Friendli(temperature=0, max_tokens=10)


def test_friendli_invoke(friendli_llm: Friendli) -> None:
    """Test invoke."""
    output = friendli_llm.invoke("Say hello world.")
    assert isinstance(output, str)


async def test_friendli_ainvoke(friendli_llm: Friendli) -> None:
    """Test async invoke."""
    output = await friendli_llm.ainvoke("Say hello world.")
    assert isinstance(output, str)


def test_friendli_batch(friendli_llm: Friendli) -> None:
    """Test batch."""
    outputs = friendli_llm.batch(["Say hello world.", "Say bye world."])
    for output in outputs:
        assert isinstance(output, str)


async def test_friendli_abatch(friendli_llm: Friendli) -> None:
    """Test async batch."""
    outputs = await friendli_llm.abatch(["Say hello world.", "Say bye world."])
    for output in outputs:
        assert isinstance(output, str)


def test_friendli_generate(friendli_llm: Friendli) -> None:
    """Test generate."""
    result = friendli_llm.generate(["Say hello world.", "Say bye world."])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info


async def test_friendli_agenerate(friendli_llm: Friendli) -> None:
    """Test async generate."""
    result = await friendli_llm.agenerate(["Say hello world.", "Say bye world."])
    assert isinstance(result, LLMResult)
    generations = result.generations
    assert len(generations) == 2
    for generation in generations:
        gen_ = generation[0]
        assert isinstance(gen_, Generation)
        text = gen_.text
        assert isinstance(text, str)
        generation_info = gen_.generation_info
        if generation_info is not None:
            assert "token" in generation_info


def test_friendli_stream(friendli_llm: Friendli) -> None:
    """Test stream."""
    stream = friendli_llm.stream("Say hello world.")
    for chunk in stream:
        assert isinstance(chunk, str)


async def test_friendli_astream(friendli_llm: Friendli) -> None:
    """Test async stream."""
    stream = friendli_llm.astream("Say hello world.")
    async for chunk in stream:
        assert isinstance(chunk, str)
