"""Ollama integration tests for reasoning chat models."""

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, HumanMessage

from langchain_ollama import ChatOllama

SAMPLE = "What is 3^3?"

REASONING_MODEL_NAME = "deepseek-r1:1.5b"


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_stream_no_reasoning(model: str, use_async: bool) -> None:
    """Test streaming with `reasoning=False`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    if use_async:
        async for chunk in llm.astream(messages):
            assert isinstance(chunk, BaseMessageChunk)
            if result is None:
                result = chunk
                continue
            result += chunk
    else:
        for chunk in llm.stream(messages):
            assert isinstance(chunk, BaseMessageChunk)
            if result is None:
                result = chunk
                continue
            result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "<think>" not in result.content
    assert "</think>" not in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_stream_reasoning_none(model: str, use_async: bool) -> None:
    """Test streaming with `reasoning=None`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    if use_async:
        async for chunk in llm.astream(messages):
            assert isinstance(chunk, BaseMessageChunk)
            if result is None:
                result = chunk
                continue
            result += chunk
    else:
        for chunk in llm.stream(messages):
            assert isinstance(chunk, BaseMessageChunk)
            if result is None:
                result = chunk
                continue
            result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    # reasoning_content is only captured when reasoning=True
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_reasoning_stream(model: str, use_async: bool) -> None:
    """Test streaming with `reasoning=True`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    messages = [
        {
            "role": "user",
            "content": SAMPLE,
        }
    ]
    result = None
    if use_async:
        async for chunk in llm.astream(messages):
            assert isinstance(chunk, BaseMessageChunk)
            if result is None:
                result = chunk
                continue
            result += chunk
    else:
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
    assert "<think>" not in result.content
    assert "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]

    content_blocks = result.content_blocks
    assert content_blocks is not None
    assert len(content_blocks) > 0
    reasoning_blocks = [
        block for block in content_blocks if block.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) > 0


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_invoke_no_reasoning(model: str, use_async: bool) -> None:
    """Test invoke with `reasoning=False`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    message = HumanMessage(content=SAMPLE)
    if use_async:
        result = await llm.ainvoke([message])
    else:
        result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content
    assert "</think>" not in result.content


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_invoke_reasoning_none(model: str, use_async: bool) -> None:
    """Test invoke with `reasoning=None`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    if use_async:
        result = await llm.ainvoke([message])
    else:
        result = llm.invoke([message])
    assert result.content
    # reasoning_content is only captured when reasoning=True
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
@pytest.mark.parametrize("use_async", [False, True])
async def test_reasoning_invoke(model: str, use_async: bool) -> None:
    """Test invoke with `reasoning=True`."""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    message = HumanMessage(content=SAMPLE)
    if use_async:
        result = await llm.ainvoke([message])
    else:
        result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" not in result.content
    assert "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]

    content_blocks = result.content_blocks
    assert content_blocks is not None
    assert len(content_blocks) > 0
    reasoning_blocks = [
        block for block in content_blocks if block.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) > 0


@pytest.mark.parametrize("model", [REASONING_MODEL_NAME])
def test_reasoning_modes_behavior(model: str) -> None:
    """Test the behavior differences between reasoning modes.

    This test documents how the Ollama API and LangChain handle reasoning content
    for DeepSeek R1 models across different reasoning settings.

    Current Ollama API behavior:
    - Ollama automatically separates reasoning content into a 'thinking' field
    - No <think> tags are present in responses
    - `think=False` prevents the 'thinking' field from being included
    - `think=None` includes the 'thinking' field (model default)
    - `think=True` explicitly requests the 'thinking' field

    LangChain behavior:
    - `reasoning=False`: Does not capture reasoning content
    - `reasoning=None`: Does not capture reasoning content (model default behavior)
    - `reasoning=True`: Captures reasoning in `additional_kwargs['reasoning_content']`
    """
    message = HumanMessage(content=SAMPLE)

    # Test with reasoning=None (model default - no reasoning captured)
    llm_default = ChatOllama(model=model, reasoning=None, num_ctx=2**12)
    result_default = llm_default.invoke([message])
    assert result_default.content
    assert "<think>" not in result_default.content
    assert "</think>" not in result_default.content
    assert "reasoning_content" not in result_default.additional_kwargs

    # Test with reasoning=False (explicit disable - no reasoning captured)
    llm_disabled = ChatOllama(model=model, reasoning=False, num_ctx=2**12)
    result_disabled = llm_disabled.invoke([message])
    assert result_disabled.content
    assert "<think>" not in result_disabled.content
    assert "</think>" not in result_disabled.content
    assert "reasoning_content" not in result_disabled.additional_kwargs

    # Test with reasoning=True (reasoning captured separately)
    llm_enabled = ChatOllama(model=model, reasoning=True, num_ctx=2**12)
    result_enabled = llm_enabled.invoke([message])
    assert result_enabled.content
    assert "<think>" not in result_enabled.content
    assert "</think>" not in result_enabled.content
    assert "reasoning_content" in result_enabled.additional_kwargs
    assert len(result_enabled.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" not in result_enabled.additional_kwargs["reasoning_content"]
    assert "</think>" not in result_enabled.additional_kwargs["reasoning_content"]
