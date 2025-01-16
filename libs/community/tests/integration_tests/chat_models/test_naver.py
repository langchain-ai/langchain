"""Test ChatClovaX chat model."""

import pytest
from httpx_sse import SSEError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)

from langchain_community.chat_models import ChatClovaX


def test_stream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    for token in llm.stream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata


async def test_astream() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    async for token in llm.astream("I'm Clova"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata


async def test_abatch() -> None:
    """Test streaming tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = await llm.abatch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = await llm.abatch(["I'm Clova", "I'm not Clova"], config={"tags": ["foo"]})
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata


def test_batch() -> None:
    """Test batch tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = llm.batch(["I'm Clova", "I'm not Clova"])
    for token in result:
        assert isinstance(token, AIMessage)
        assert isinstance(token.content, str)
        if token.response_metadata:
            assert "input_length" in token.response_metadata
            assert "output_length" in token.response_metadata
            assert "stop_reason" in token.response_metadata
            assert "ai_filter" in token.response_metadata


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = await llm.ainvoke("I'm Clova", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert "input_length" in result.response_metadata
        assert "output_length" in result.response_metadata
        assert "stop_reason" in result.response_metadata
        assert "ai_filter" in result.response_metadata


def test_invoke() -> None:
    """Test invoke tokens from ChatClovaX."""
    llm = ChatClovaX(include_ai_filters=True)

    result = llm.invoke("I'm Clova", config=dict(tags=["foo"]))
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    if result.response_metadata:
        assert "input_length" in result.response_metadata
        assert "output_length" in result.response_metadata
        assert "stop_reason" in result.response_metadata
        assert "ai_filter" in result.response_metadata


def test_stream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(SSEError):
        for _ in llm.stream(prompt * 1000):
            pass


async def test_astream_error_event() -> None:
    """Test streaming error event from ChatClovaX."""
    llm = ChatClovaX()
    prompt = "What is the best way to reduce my carbon footprint?"

    with pytest.raises(SSEError):
        async for _ in llm.astream(prompt * 1000):
            pass
