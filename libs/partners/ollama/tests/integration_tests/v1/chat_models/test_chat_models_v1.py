"""Ollama-specific v1 chat model integration tests.

Standard tests are handled in `test_chat_models_v1_standard.py`.

"""

from __future__ import annotations

from typing import Annotated, Optional

import pytest
from langchain_core.messages.content_blocks import is_reasoning_block
from langchain_core.v1.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_ollama.v1.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"
REASONING_MODEL_NAME = "deepseek-r1:1.5b"

SAMPLE = "What is 3^3?"


@pytest.mark.parametrize(("method"), [("function_calling"), ("json_schema")])
def test_structured_output(method: str) -> None:
    """Test to verify structured output via tool calling and `format` parameter."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, temperature=0)
    query = "Tell me a joke about cats."

    # Pydantic
    if method == "function_calling":
        structured_llm = llm.with_structured_output(Joke, method="function_calling")
        result = structured_llm.invoke(query)
        assert isinstance(result, Joke)

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, Joke)

    # JSON Schema
    if method == "json_schema":
        structured_llm = llm.with_structured_output(
            Joke.model_json_schema(), method="json_schema"
        )
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}

        # Typed Dict
        class JokeSchema(TypedDict):
            """Joke to tell user."""

            setup: Annotated[str, "question to set up a joke"]
            punchline: Annotated[str, "answer to resolve the joke"]

        structured_llm = llm.with_structured_output(JokeSchema, method="json_schema")
        result = structured_llm.invoke(query)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"setup", "punchline"}

        for chunk in structured_llm.stream(query):
            assert isinstance(chunk, dict)
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == {"setup", "punchline"}


@pytest.mark.parametrize(("model"), [(DEFAULT_MODEL_NAME)])
def test_structured_output_deeply_nested(model: str) -> None:
    """Test to verify structured output with a nested objects."""
    llm = ChatOllama(model=model, temperature=0)

    class Person(BaseModel):
        """Information about a person."""

        name: Optional[str] = Field(default=None, description="The name of the person")
        hair_color: Optional[str] = Field(
            default=None, description="The color of the person's hair if known"
        )
        height_in_meters: Optional[str] = Field(
            default=None, description="Height measured in meters"
        )

    class Data(BaseModel):
        """Extracted data about people."""

        people: list[Person]

    chat = llm.with_structured_output(Data)
    text = (
        "Alan Smith is 6 feet tall and has blond hair."
        "Alan Poe is 3 feet tall and has grey hair."
    )
    result = chat.invoke(text)
    assert isinstance(result, Data)

    for chunk in chat.stream(text):
        assert isinstance(chunk, Data)


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_stream_no_reasoning(model: str) -> None:
    """Test streaming with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    result = None
    for chunk in llm.stream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" not in content_types, (
        f"Expected no reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_astream_no_reasoning(model: str) -> None:
    """Test async streaming with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    result = None
    async for chunk in llm.astream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" not in content_types, (
        f"Expected no reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_stream_reasoning_none(model: str) -> None:
    """Test streaming with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    result = None
    for chunk in llm.stream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    assert "<think>" in result.text and "</think>" in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_astream_reasoning_none(model: str) -> None:
    """Test async streaming with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    result = None
    async for chunk in llm.astream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    assert "<think>" in result.text and "</think>" in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_reasoning_stream(model: str) -> None:
    """Test streaming with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    result = None
    for chunk in llm.stream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" in content_types, (
        f"Expected reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text

    # Assert non-empty reasoning content in ReasoningContentBlock
    reasoning_blocks = [block for block in result.content if is_reasoning_block(block)]
    for block in reasoning_blocks:
        assert block.get("reasoning"), "Expected non-empty reasoning content"
        assert len(block.get("reasoning", "")) > 0, (
            "Expected reasoning content to be non-empty"
        )


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_reasoning_astream(model: str) -> None:
    """Test async streaming with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    result = None
    async for chunk in llm.astream(SAMPLE):
        assert isinstance(chunk, AIMessageChunk)
        if result is None:
            result = chunk
            continue
        result += chunk
    assert isinstance(result, AIMessageChunk)
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" in content_types, (
        f"Expected reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text

    # Assert non-empty reasoning content in ReasoningContentBlock
    reasoning_blocks = [block for block in result.content if is_reasoning_block(block)]
    for block in reasoning_blocks:
        assert block.get("reasoning"), "Expected non-empty reasoning content"
        assert len(block.get("reasoning", "")) > 0, (
            "Expected reasoning content to be non-empty"
        )


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_invoke_no_reasoning(model: str) -> None:
    """Test using invoke with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    message = HumanMessage(SAMPLE)
    result = llm.invoke([message])
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" not in content_types, (
        f"Expected no reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_ainvoke_no_reasoning(model: str) -> None:
    """Test using async invoke with `reasoning=False`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=False)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" not in content_types, (
        f"Expected no reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_invoke_reasoning_none(model: str) -> None:
    """Test using invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content

    assert "<think>" in result.text and "</think>" in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_ainvoke_reasoning_none(model: str) -> None:
    """Test using async invoke with `reasoning=None`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=None)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content

    assert "<think>" in result.text and "</think>" in result.text


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
def test_reasoning_invoke(model: str) -> None:
    """Test invoke with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    message = HumanMessage(content=SAMPLE)
    result = llm.invoke([message])
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" in content_types, (
        f"Expected reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text

    # Assert non-empty reasoning content in ReasoningContentBlock
    reasoning_blocks = [block for block in result.content if is_reasoning_block(block)]
    for block in reasoning_blocks:
        assert block.get("reasoning"), "Expected non-empty reasoning content"
        assert len(block.get("reasoning", "")) > 0, (
            "Expected reasoning content to be non-empty"
        )


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
async def test_reasoning_ainvoke(model: str) -> None:
    """Test invoke with `reasoning=True`"""
    llm = ChatOllama(model=model, num_ctx=2**12, reasoning=True)
    message = HumanMessage(content=SAMPLE)
    result = await llm.ainvoke([message])
    assert result.content

    content_types = set()
    for content_block in result.content:
        type_ = content_block.get("type")
        if type_:
            content_types.add(type_)

    assert "reasoning" in content_types, (
        f"Expected reasoning content, got types: {content_types}"
    )
    assert "non_standard" not in content_types, (
        f"Expected no non-standard content, got types: {content_types}"
    )
    assert "<think>" not in result.text and "</think>" not in result.text

    # Assert non-empty reasoning content in ReasoningContentBlock
    reasoning_blocks = [block for block in result.content if is_reasoning_block(block)]
    for block in reasoning_blocks:
        assert block.get("reasoning"), "Expected non-empty reasoning content"
        assert len(block.get("reasoning", "")) > 0, (
            "Expected reasoning content to be non-empty"
        )


@pytest.mark.parametrize(("model"), [(REASONING_MODEL_NAME)])
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
    assert "<think>" in result_default.text
    assert "</think>" in result_default.text

    # Test with reasoning=False (explicit disable - should NOT include think tags)
    llm_disabled = ChatOllama(model=model, reasoning=False, num_ctx=2**12)

    result_disabled = llm_disabled.invoke([message])

    # With reasoning=False, think tags should be stripped from content
    # This verifies that _strip_think_tags is working correctly
    assert "<think>" not in result_disabled.text
    assert "</think>" not in result_disabled.text

    # Verify the difference: same model, different reasoning settings
    # Default includes tags, disabled strips them
    assert result_default.content != result_disabled.content
