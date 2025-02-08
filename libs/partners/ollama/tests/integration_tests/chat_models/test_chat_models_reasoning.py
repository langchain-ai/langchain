"""Ollama specific chat model integration tests for reasoning models."""

import pytest
from langchain_core.messages import BaseMessageChunk, HumanMessage
from pydantic import ValidationError

from langchain_ollama import ChatOllama


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_no_reasoning(model: str) -> None:
    """Test deepseek model without parsing."""
    model = ChatOllama(model=model, num_ctx=2**14, extract_reasoning=False)
    messages = [
        {
            "role": "user",
            "content": "What is 2+2?",
        }
    ]
    result = None
    for chunk in model.stream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert result.content
    assert "<think>" in result.content and "</think>" in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True"""
    model = ChatOllama(model=model, num_ctx=2**14, extract_reasoning=True)
    messages = [
        {
            "role": "user",
            "content": "What is 2+2?",
        }
    ]
    result = None
    for chunk in model.stream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "4" in result.content
    assert (
        "<think>" in result.additional_kwargs["reasoning_content"]
        and "</think>" in result.additional_kwargs["reasoning_content"]
    )


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_tuple(model: str) -> None:
    """Test deepseek model with reasoning with tuple=..."""
    model = ChatOllama(
        model=model, num_ctx=2**14, extract_reasoning=("<think>", "</think>")
    )
    messages = [
        {
            "role": "user",
            "content": "What is 2+2?",
        }
    ]
    result = None
    for chunk in model.stream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "4" in result.content
    assert (
        "<think>" in result.additional_kwargs["reasoning_content"]
        and "</think>" in result.additional_kwargs["reasoning_content"]
    )


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_no_reasoning(model: str) -> None:
    """Test deepseek model without parsing using invoke."""
    model = ChatOllama(model=model, num_ctx=2**14, extract_reasoning=False)
    message = HumanMessage(content="What is 2+2?")
    result = model.invoke([message])
    assert result.content
    assert "<think>" in result.content and "</think>" in result.content
    assert "reasoning_content" not in result.additional_kwargs


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True using invoke"""
    model = ChatOllama(model=model, num_ctx=2**14, extract_reasoning=True)
    message = HumanMessage(content="What is 2+2?")
    result = model.invoke([message])
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "4" in result.content
    assert (
        "<think>" in result.additional_kwargs["reasoning_content"]
        and "</think>" in result.additional_kwargs["reasoning_content"]
    )


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_tuple(model: str) -> None:
    """Test deepseek model with reasoning with tuple=... using invoke"""
    model = ChatOllama(
        model=model, num_ctx=2**14, extract_reasoning=("<think>", "</think>")
    )
    message = HumanMessage(content="What is 2+2?")
    result = model.invoke([message])
    assert result.content
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0
    assert "4" in result.content
    assert (
        "<think>" in result.additional_kwargs["reasoning_content"]
        and "</think>" in result.additional_kwargs["reasoning_content"]
    )


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_invalid(model: str) -> None:
    """Test deepseek model with reasoning raises ValidationError"""
    with pytest.raises(ValidationError):
        model = ChatOllama(
            model=model, extract_reasoning={"invalid": "data"}
        )
