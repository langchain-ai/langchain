"""Ollama integration tests for reasoning chat models."""

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, HumanMessage

from langchain_ollama import ChatOllama

SAMPLE = "What is 3^3?"


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
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
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
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
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_reasoning_none(model: str) -> None:
    """Test streaming with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
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
    assert "<think>" in result.content and "</think>" in result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.additional_kwargs.get("reasoning_content", "")
    assert "</think>" not in result.additional_kwargs.get("reasoning_content", "")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_astream_reasoning_none(model: str) -> None:
    """Test async streaming with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
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
    assert "<think>" in result.content and "</think>" in result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.additional_kwargs.get("reasoning_content", "")
    assert "</think>" not in result.additional_kwargs.get("reasoning_content", "")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_stream(model: str) -> None:
    """Test streaming with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
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
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_reasoning_astream(model: str) -> None:
    """Test async streaming with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
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
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_invoke_no_reasoning(model: str) -> None:
    """Test using invoke with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_no_reasoning(model: str) -> None:
    """Test using async invoke with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_invoke_reasoning_none(model: str) -> None:
    """Test using invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs.get("reasoning_content", "")
    assert "</think>" not in result.additional_kwargs.get("reasoning_content", "")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_reasoning_none(model: str) -> None:
    """Test using async invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs.get("reasoning_content", "")
    assert "</think>" not in result.additional_kwargs.get("reasoning_content", "")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_invoke(model: str) -> None:
    """Test invoke with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_reasoning_ainvoke(model: str) -> None:
    """Test invoke with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_think_tag_stripping_necessity(model: str) -> None:
    """Test that demonstrates why ``_strip_think_tags`` is necessary.

    DeepSeek R1 models include reasoning/thinking as their default behavior.
    When ``reasoning=False`` is set, the user explicitly wants no reasoning content,
    but Ollama cannot disable thinking at the API level for these models.
    Therefore, post-processing is required to strip the ``<think>`` tags.

    This test documents the specific behavior that necessitates the
    ``_strip_think_tags`` function in the chat_models.py implementation.
    """
    # Test with reasoning=None (default behavior - should include think tags)
    llm_default = ChatOllama(model=model, reasoning=None, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)

    result_default = llm_default.invoke([message])

    # With reasoning=None, the model's default behavior includes <think> tags
    # This demonstrates why we need the stripping logic
    assert "<think>" in result_default.content
    assert "</think>" in result_default.content
    assert "reasoning_content" not in result_default.additional_kwargs

    # Test with reasoning=False (explicit disable - should NOT include think tags)
    llm_disabled = ChatOllama(model=model, reasoning=False, num_ctx=2**12)

    result_disabled = llm_disabled.invoke([message])

    # With reasoning=False, think tags should be stripped from content
    # This verifies that _strip_think_tags is working correctly
    assert "<think>" not in result_disabled.content
    assert "</think>" not in result_disabled.content
    assert "reasoning_content" not in result_disabled.additional_kwargs

    # Verify the difference: same model, different reasoning settings
    # Default includes tags, disabled strips them
    assert result_default.content != result_disabled.content
