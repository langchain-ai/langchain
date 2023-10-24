"""Test Baidu Qianfan LLM Endpoint."""
from typing import Generator

import pytest

from langchain.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain.schema import LLMResult


def test_call() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm("write a joke")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_stream() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()
    output = llm.stream("write a joke")
    assert isinstance(output, Generator)


@pytest.mark.asyncio
async def test_qianfan_aio() -> None:
    llm = QianfanLLMEndpoint(streaming=True)

    async for token in llm.astream("hi qianfan."):
        assert isinstance(token, str)
