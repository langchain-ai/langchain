from typing import Generator

import pytest

from langchain.llms import ErnieBot
from langchain.schema import LLMResult


def test_call() -> None:
    """Test valid call to qianfan."""
    llm = ErnieBot()
    output = llm("write a joke")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to qianfan."""
    llm = ErnieBot()
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_stream() -> None:
    """Test valid call to qianfan."""
    llm = ErnieBot()
    output = llm.stream("write a joke")
    assert isinstance(output, Generator)


@pytest.mark.asyncio
async def test_aio_stream() -> None:
    llm = ErnieBot(streaming=True)

    async for token in llm.astream("hi qianfan."):
        assert isinstance(token, str)
