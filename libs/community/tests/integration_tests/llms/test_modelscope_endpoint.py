"""Test ModelscopeEndpoint API wrapper."""

from typing import AsyncIterator, Iterator

from langchain_community.llms.modelscope_endpoint import ModelScopeEndpoint


def test_modelscope_call() -> None:
    """Test valid call to Modelscope."""
    llm = ModelScopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_modelscope_streaming() -> None:
    """Test streaming call to Modelscope."""
    llm = ModelScopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
    generator = llm.stream("write a quick sort in python")
    stream_results_string = ""
    assert isinstance(generator, Iterator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1


async def test_modelscope_call_async() -> None:
    llm = ModelScopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
    output = await llm.ainvoke("write a quick sort in python")
    assert isinstance(output, str)


async def test_modelscope_streaming_async() -> None:
    llm = ModelScopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
    generator = llm.astream("write a quick sort in python")
    stream_results_string = ""
    assert isinstance(generator, AsyncIterator)

    async for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1
