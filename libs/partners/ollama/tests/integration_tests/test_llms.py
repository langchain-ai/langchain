"""Test OllamaLLM llm."""

import os

import pytest
from langchain_core.outputs import GenerationChunk
from langchain_core.runnables import RunnableConfig

from langchain_ollama.llms import OllamaLLM

MODEL_NAME = os.environ.get("OLLAMA_TEST_MODEL", "llama3.1")
REASONING_MODEL_NAME = os.environ.get("OLLAMA_REASONING_TEST_MODEL", "deepseek-r1:1.5b")
SAMPLE = "What is 3^3?"


def test_invoke() -> None:
    """Test sync invoke returning a string."""
    llm = OllamaLLM(model=MODEL_NAME)
    result = llm.invoke("I'm Pickle Rick", config=RunnableConfig(tags=["foo"]))
    assert isinstance(result, str)


async def test_ainvoke() -> None:
    """Test async invoke returning a string."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.ainvoke("I'm Pickle Rick", config=RunnableConfig(tags=["foo"]))
    assert isinstance(result, str)


def test_batch() -> None:
    """Test batch sync token generation from `OllamaLLM`."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch() -> None:
    """Test batch async token generation from `OllamaLLM`."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


def test_batch_tags() -> None:
    """Test batch sync token generation with tags."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = llm.batch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch async token generation with tags."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_stream_text_tokens() -> None:
    """Test streaming raw string tokens from `OllamaLLM`."""
    llm = OllamaLLM(model=MODEL_NAME)

    for token in llm.stream("Hi."):
        assert isinstance(token, str)


async def test_astream_text_tokens() -> None:
    """Test async streaming raw string tokens from `OllamaLLM`."""
    llm = OllamaLLM(model=MODEL_NAME)

    async for token in llm.astream("Hi."):
        assert isinstance(token, str)


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test__stream_no_reasoning(model: str) -> None:
    """Test low-level chunk streaming of a simple prompt with `reasoning=False`."""
    llm = OllamaLLM(model=model, num_ctx=2**12)

    result_chunk = None
    for chunk in llm._stream(SAMPLE):
        assert isinstance(chunk, GenerationChunk)
        if result_chunk is None:
            result_chunk = chunk
        else:
            result_chunk += chunk

    # The final result must be a GenerationChunk with visible content
    assert isinstance(result_chunk, GenerationChunk)
    assert result_chunk.text
    assert result_chunk.generation_info
    assert not result_chunk.generation_info.get("reasoning_content")


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test__astream_no_reasoning(model: str) -> None:
    """Test low-level async chunk streaming with `reasoning=False`."""
    llm = OllamaLLM(model=model, num_ctx=2**12)

    result_chunk = None
    async for chunk in llm._astream(SAMPLE):
        assert isinstance(chunk, GenerationChunk)
        if result_chunk is None:
            result_chunk = chunk
        else:
            result_chunk += chunk

    # The final result must be a GenerationChunk with visible content
    assert isinstance(result_chunk, GenerationChunk)
    assert result_chunk.text
    assert result_chunk.generation_info
    assert not result_chunk.generation_info.get("reasoning_content")


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test__stream_with_reasoning(model: str) -> None:
    """Test low-level chunk streaming with `reasoning=True`."""
    llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)

    result_chunk = None
    for chunk in llm._stream(SAMPLE):
        assert isinstance(chunk, GenerationChunk)
        if result_chunk is None:
            result_chunk = chunk
        else:
            result_chunk += chunk

    assert isinstance(result_chunk, GenerationChunk)
    assert result_chunk.text

    # Should have extracted reasoning into generation_info
    assert result_chunk.generation_info
    reasoning_content = result_chunk.generation_info.get("reasoning_content")
    assert reasoning_content
    assert len(reasoning_content) > 0
    # And neither the visible nor the hidden portion contains <think> tags
    assert "<think>" not in result_chunk.text
    assert "</think>" not in result_chunk.text
    assert "<think>" not in reasoning_content
    assert "</think>" not in reasoning_content


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test__astream_with_reasoning(model: str) -> None:
    """Test low-level async chunk streaming with `reasoning=True`."""
    llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)

    result_chunk = None
    async for chunk in llm._astream(SAMPLE):
        assert isinstance(chunk, GenerationChunk)
        if result_chunk is None:
            result_chunk = chunk
        else:
            result_chunk += chunk

    assert isinstance(result_chunk, GenerationChunk)
    assert result_chunk.text

    # Should have extracted reasoning into generation_info
    assert result_chunk.generation_info
    reasoning_content = result_chunk.generation_info.get("reasoning_content")
    assert reasoning_content
    assert len(reasoning_content) > 0
    # And neither the visible nor the hidden portion contains <think> tags
    assert "<think>" not in result_chunk.text
    assert "</think>" not in result_chunk.text
    assert "<think>" not in reasoning_content
    assert "</think>" not in reasoning_content
