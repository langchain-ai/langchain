"""Ollama specific chat model integration tests for reasoning models."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ConnectError
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

SAMPLE = "What is 3^3?"


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_no_reasoning(model: str) -> None:
    """Test deepseek model without `think`."""
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

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_stream_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True"""
    llm = ChatOllama(model=model, num_ctx=2**12, reason=True)
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


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_no_reasoning(model: str) -> None:
    """Test deepseek model without parsing using invoke."""
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_deepseek_messages_invoke_bool(model: str) -> None:
    """Test deepseek model with reasoning bool=True using invoke"""
    llm = ChatOllama(model=model, num_ctx=2**12, reason=True)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_deepseek_messages_ainvoke_bool(model: str) -> None:
    """Test async invoke with reasoning enabled."""
    llm = ChatOllama(model=model, reason=True)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_deepseek_messages_astream_bool(model: str) -> None:
    """Test async stream with reasoning enabled."""
    llm = ChatOllama(model=model, reason=True)
    messages = [{"role": "user", "content": SAMPLE}]
    result = None
    async for chunk in llm.astream(messages):
        assert isinstance(chunk, BaseMessageChunk)
        if result is None:
            result = chunk
        else:
            result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0


class MathAnswer(BaseModel):
    """A mathematical expression and its numerical answer."""

    expression: str = Field(description="The mathematical expression to evaluate.")
    answer: int = Field(description="The numerical answer to the expression.")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_with_structured_output(model: str) -> None:
    """Test structured output with json_schema method."""
    llm = ChatOllama(model=model, temperature=0)
    structured_llm = llm.with_structured_output(MathAnswer, method="json_schema")
    result = structured_llm.invoke(f"Calculate {SAMPLE}")
    assert isinstance(result, MathAnswer)
    assert "3^3" in result.expression or "3**3" in result.expression
    assert result.answer == 27


@patch("langchain_ollama.chat_models.Client.list")
def test_init_connection_error(mock_list: MagicMock) -> None:
    """Test that a ValueError wrapping ConnectError is raised on connection failure."""
    mock_list.side_effect = ConnectError("Test connection error")
    with pytest.raises(ValueError) as excinfo:
        ChatOllama(model="any-model")
    assert "Connection to Ollama failed" in str(excinfo.value)
    assert "Test connection error" in str(excinfo.value)


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_complex_message_history(model: str) -> None:
    """Test the model with a more complex message history."""
    llm = ChatOllama(model=model)
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers math problems."
        ),
        HumanMessage(content="What color is the sun?"),
        AIMessage(content="Yellow"),
        HumanMessage(content="What color is the moon?"),
    ]
    result = llm.invoke(messages)
    assert isinstance(result, AIMessage)
    assert result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_abatch(model: str) -> None:
    """Test the async batch method."""
    llm = ChatOllama(model=model)

    prompt_strings = ["What is 2+2?", "What is 3+3?"]
    messages: list[LanguageModelInput] = [
        [HumanMessage(content=prompt)] for prompt in prompt_strings
    ]
    results = await llm.abatch(messages)

    assert len(results) == 2
    assert isinstance(results[0], AIMessage)
    assert isinstance(results[1], AIMessage)
    assert results[0].content is not None and "4" in results[0].content
    assert results[1].content is not None and "6" in results[1].content
