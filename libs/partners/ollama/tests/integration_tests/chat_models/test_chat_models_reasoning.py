"""Ollama specific chat model integration tests for reasoning models."""

import pytest
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

SAMPLE = "What is 3^3?"


class MathAnswer(BaseModel):
    """A mathematical expression and its numerical answer."""

    expression: str = Field(description="The mathematical expression to evaluate.")
    answer: int = Field(description="The numerical answer to the expression.")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reasoning=False`"""
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
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12)
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
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


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
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


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
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


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
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_no_reasoning(model: str) -> None:
    """Test using async invoke with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_invoke_reasoning_none(model: str) -> None:
    """Test using invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_reasoning_none(model: str) -> None:
    """Test using async invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs
    assert "<think>" in result.content and "</think>" in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


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
