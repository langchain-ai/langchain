"""Ollama specific chat model integration tests for reasoning models."""

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, HumanMessage
from pydantic import ValidationError

from langchain_ollama import ChatOllama

SAMPLE = "What is 3^3?"


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_no_reasoning(model: str) -> None:
    """Test deepseek model without parsing."""
    llm = ChatOllama(model=model, num_ctx=2**12)
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


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True"""
    llm = ChatOllama(model=model, num_ctx=2**12, extract_reasoning=True)
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
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" in result.additional_kwargs["reasoning_content"]
    assert "</think>" in result.additional_kwargs["reasoning_content"]
    clean_content = (
        result.additional_kwargs["reasoning_content"]
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    assert len(clean_content) > 0


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_tuple(model: str) -> None:
    """Test deepseek model with reasoning with tuple=..."""
    llm = ChatOllama(
        model=model, num_ctx=2**12, extract_reasoning=("<think>", "</think>")
    )
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
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" in result.additional_kwargs["reasoning_content"]
    assert "</think>" in result.additional_kwargs["reasoning_content"]
    clean_content = (
        result.additional_kwargs["reasoning_content"]
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    assert len(clean_content) > 0


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_no_reasoning(model: str) -> None:
    """Test deepseek model without parsing using invoke."""
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "<think>" in result.content and "</think>" in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True using invoke"""
    llm = ChatOllama(model=model, num_ctx=2**12, extract_reasoning=True)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" in result.additional_kwargs["reasoning_content"]
    assert "</think>" in result.additional_kwargs["reasoning_content"]
    clean_content = (
        result.additional_kwargs["reasoning_content"]
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    assert len(clean_content) > 0


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_tuple(model: str) -> None:
    """Test deepseek model with reasoning with tuple=... using invoke"""
    llm = ChatOllama(
        model=model, num_ctx=2**12, extract_reasoning=("<think>", "</think>")
    )
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "<think>" in result.additional_kwargs["reasoning_content"]
    assert "</think>" in result.additional_kwargs["reasoning_content"]
    clean_content = (
        result.additional_kwargs["reasoning_content"]
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    assert len(clean_content) > 0


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_invalid(model: str) -> None:
    """Test deepseek model with reasoning raises ValidationError"""
    with pytest.raises(ValidationError):
        _ = ChatOllama(model=model, extract_reasoning={"invalid": "data"})  # type: ignore[arg-type]
