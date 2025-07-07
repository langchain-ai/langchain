"""Test OllamaLLM llm."""

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.runnables import RunnableConfig

from langchain_ollama.llms import OllamaLLM

MODEL_NAME = "llama3.1"

SAMPLE = "What is 3^3?"


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OllamaLLM(model=MODEL_NAME)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reasoning=False`"""
    llm = OllamaLLM(model=model, num_ctx=2**12)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    for chunk in llm.stream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_stream(model: str) -> None:
    """Test streaming with `reasoning=True`"""
    llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    for chunk in llm.stream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OllamaLLM(model=MODEL_NAME)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reasoning=False`"""
    llm = OllamaLLM(model=model, num_ctx=2**12)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    async for chunk in llm.astream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_reasoning_astream(model: str) -> None:
    """Test async streaming with `reasoning=True`"""
    llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    async for chunk in llm.astream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


async def test_abatch() -> None:
    """Test streaming tokens from OllamaLLM."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from OllamaLLM."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token, str)


def test_batch() -> None:
    """Test batch tokens from OllamaLLM."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from OllamaLLM."""
    llm = OllamaLLM(model=MODEL_NAME)

    result = await llm.ainvoke("I'm Pickle Rick", config=RunnableConfig(tags=["foo"]))
    assert isinstance(result, str)


# TODO
# @pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
# async def test_ainvoke_no_reasoning(model: str) -> None:
#     """Test using async invoke with `reasoning=False`"""
#     llm = OllamaLLM(model=model, num_ctx=2**12)
#     message = SAMPLE
#     result = await llm.ainvoke(message)
#     assert result.content
#     assert "reasoning_content" not in result.additional_kwargs

#     # Sanity check the old behavior isn't present
#     assert "<think>" not in result.content and "</think>" not in result.content


# @pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
# async def test_reasoning_ainvoke(model: str) -> None:
#     """Test invoke with `reasoning=True`"""
#     llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)
#     message = SAMPLE
#     result = await llm.ainvoke(message)
#     assert result.content
#     assert "reasoning_content" in result.additional_kwargs
#     assert len(result.additional_kwargs["reasoning_content"]) > 0

#     # Sanity check the old behavior isn't present
#     assert "<think>" not in result.content and "</think>" not in result.content
#     assert "<think>" not in result.additional_kwargs["reasoning_content"]
#     assert "</think>" not in result.additional_kwargs["reasoning_content"]


def test_invoke() -> None:
    """Test invoke tokens from OllamaLLM."""
    llm = OllamaLLM(model=MODEL_NAME)
    result = llm.invoke("I'm Pickle Rick", config=RunnableConfig(tags=["foo"]))
    assert isinstance(result, str)


# TODO
# @pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
# def test_invoke_no_reasoning(model: str) -> None:
#     """Test using invoke with `reasoning=False`"""
#     llm = OllamaLLM(model=model, num_ctx=2**12)
#     message = SAMPLE
#     result = llm.invoke(message)
#     assert result.content
#     assert "reasoning_content" not in result.additional_kwargs

#     # Sanity check the old behavior isn't present
#     assert "<think>" not in result.content and "</think>" not in result.content


# @pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
# def test_reasoning_invoke(model: str) -> None:
#     """Test invoke with `reasoning=True`"""
#     llm = OllamaLLM(model=model, num_ctx=2**12, reasoning=True)
#     message = SAMPLE
#     result = llm.invoke(message)
#     assert result.content
#     assert "reasoning_content" in result.additional_kwargs
#     assert len(result.additional_kwargs["reasoning_content"]) > 0

#     # Sanity check the old behavior isn't present
#     assert "<think>" not in result.content and "</think>" not in result.content
#     assert "<think>" not in result.additional_kwargs["reasoning_content"]
#     assert "</think>" not in result.additional_kwargs["reasoning_content"]
