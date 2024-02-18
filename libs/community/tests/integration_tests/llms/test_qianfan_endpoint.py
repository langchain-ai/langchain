"""Test Baidu Qianfan LLM Endpoint."""
from typing import Generator

from langchain_core.outputs import LLMResult

from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint


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


async def test_qianfan_aio() -> None:
    llm = QianfanLLMEndpoint(streaming=True)

    async for token in llm.astream("hi qianfan."):
        assert isinstance(token, str)


def test_rate_limit() -> None:
    llm = QianfanLLMEndpoint(model="ERNIE-Bot", init_kwargs={"query_per_second": 2})
    assert llm.client._client._rate_limiter._sync_limiter._query_per_second == 2
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
