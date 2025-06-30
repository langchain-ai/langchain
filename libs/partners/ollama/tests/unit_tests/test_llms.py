"""Test Ollama Chat API wrapper."""

import pytest
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.base import BaseMessageChunk

from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import HumanMessage

SAMPLE = "What is 3^3?"

MODEL_NAME = "llama3.1"


def test_initialization() -> None:
    """Test integration initialization."""
    OllamaLLM(model=MODEL_NAME)


def test_model_params() -> None:
    # Test standard tracing params
    llm = OllamaLLM(model=MODEL_NAME)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
    }

    llm = OllamaLLM(model=MODEL_NAME, num_predict=3)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "ollama",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_max_tokens": 3,
    }


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reason=False`"""
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
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reason=False`"""
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
def test_reasoning_stream(model: str) -> None:
    """Test streaming with `reason=True`"""
    llm = OllamaLLM(model=model, num_ctx=2**12, reason=True)
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
    llm = OllamaLLM(model=model, num_ctx=2**12, reason=True)
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
    llm = OllamaLLM(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
async def test_ainvoke_no_reasoning(model: str) -> None:
    """Test using async invoke with `reason=False`"""
    llm = OllamaLLM(model=model, num_ctx=2**12)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content
    assert "reasoning_content" not in result.additional_kwargs

    # Sanity check the old behavior isn't present
    assert "<think>" not in result.content and "</think>" not in result.content


@pytest.mark.parametrize(("model"), [("deepseek-r1:1.5b")])
def test_reasoning_invoke(model: str) -> None:
    """Test invoke with `reason=True`"""
    llm = OllamaLLM(model=model, num_ctx=2**12, reason=True)
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
    llm = OllamaLLM(model=model, num_ctx=2**12, reason=True)
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
