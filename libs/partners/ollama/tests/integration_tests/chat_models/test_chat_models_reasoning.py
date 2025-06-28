"""Ollama specific chat model integration tests for reasoning models."""

import pytest
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


class MathAnswer(BaseModel):
    """A mathematical expression and its numerical answer."""

    expression: str = Field(description="The mathematical expression to evaluate.")
    answer: int = Field(description="The numerical answer to the expression.")


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reason=False`"""
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
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reason=False`"""
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

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_stream(model: str) -> None:
    """Test streaming with `reason=True`"""
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
async def test_reasoning_astream(model: str) -> None:
    """Test async streaming with `reason=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reason=True)
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


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_invoke_no_reasoning(model: str) -> None:
    """Test using invoke with `reason=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_no_reasoning(model: str) -> None:
    """Test using async invoke with `reason=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_invoke(model: str) -> None:
    """Test invoke with `reason=True`"""
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
async def test_reasoning_ainvoke(model: str) -> None:
    """Test invoke with `reason=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reason=True)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" in result.additional_kwargs
    assert len(result.additional_kwargs["reasoning_content"]) > 0

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content
    assert "<think>" not in result.additional_kwargs["reasoning_content"]
    assert "</think>" not in result.additional_kwargs["reasoning_content"]


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_structured_output(model: str) -> None:
    """Test structured output with json_schema method"""
    llm = ChatOllama(model=model, temperature=0)
    structured_llm = llm.with_structured_output(MathAnswer, method="json_schema")
    result = structured_llm.invoke(f"Calculate {SAMPLE}")
    assert isinstance(result, MathAnswer)
    assert "3^3" in result.expression or "3**3" in result.expression
    assert result.answer == 27


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_complex_message_history(model: str) -> None:
    """Test the model with a more complex message history"""
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
@pytest.mark.xfail(reason="Model sometimes doesn't return expected content")
def test_batch(model: str) -> None:
    """Test batching"""
    llm = ChatOllama(model=model)

    prompt_strings = ["What is 2+2?", "What is 3+3?"]
    messages: list[LanguageModelInput] = [
        [HumanMessage(content=prompt)] for prompt in prompt_strings
    ]
    results = llm.batch(messages)

    assert len(results) == 2
    assert isinstance(results[0], AIMessage)
    assert isinstance(results[1], AIMessage)
    assert results[0].content is not None and "4" in results[0].content
    assert results[1].content is not None and "6" in results[1].content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
@pytest.mark.xfail(reason="Model sometimes doesn't return expected content")
async def test_abatch(model: str) -> None:
    """Test async batching"""
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
