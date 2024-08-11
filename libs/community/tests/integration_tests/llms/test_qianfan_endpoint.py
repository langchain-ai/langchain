"""Test Baidu Qianfan LLM Endpoint."""

from typing import Generator, cast

from langchain_core.outputs import LLMResult
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint


def test_call() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()  # type: ignore[call-arg]
    output = llm.invoke("write a joke")
    assert isinstance(output, str)


def test_generate() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()  # type: ignore[call-arg]
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_generate_stream() -> None:
    """Test valid call to qianfan."""
    llm = QianfanLLMEndpoint()  # type: ignore[call-arg]
    output = llm.stream("write a joke")
    assert isinstance(output, Generator)


async def test_qianfan_aio() -> None:
    llm = QianfanLLMEndpoint(streaming=True)  # type: ignore[call-arg]

    async for token in llm.astream("hi qianfan."):
        assert isinstance(token, str)


def test_rate_limit() -> None:
    llm = QianfanLLMEndpoint(model="ERNIE-Bot", init_kwargs={"query_per_second": 2})  # type: ignore[call-arg]
    assert llm.client._client._rate_limiter._sync_limiter._query_per_second == 2
    output = llm.generate(["write a joke"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_qianfan_with_param_alias() -> None:
    """Test with qianfan llm parameter alias."""
    llm = QianfanLLMEndpoint(  # type: ignore[call-arg]
        api_key="your-api-key",  # type: ignore[arg-type]
        secret_key="your-secret-key",  # type: ignore[arg-type]
        timeout=50,
    )  # type: ignore[call-arg]
    assert cast(SecretStr, llm.qianfan_ak).get_secret_value() == "your-api-key"
    assert cast(SecretStr, llm.qianfan_sk).get_secret_value() == "your-secret-key"
    assert llm.request_timeout == 50
