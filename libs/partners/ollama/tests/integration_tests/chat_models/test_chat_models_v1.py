"""Integration tests for ChatOllama v1 format support."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.utils.utils import LC_AUTO_PREFIX
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"
REASONING_MODEL = "deepseek-r1:1.5b"


def test_v1_basic_output_format() -> None:
    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)

    result = llm.invoke([HumanMessage("Say 'Hello, World!'")])

    assert isinstance(result.content, list)
    assert len(result.content) >= 1

    text_block = result.content[0]
    assert isinstance(text_block, dict)
    assert text_block["type"] == "text"
    assert "Hello, World!" in text_block["text"]
    assert "id" in text_block
    assert text_block["id"].startswith(LC_AUTO_PREFIX)


def test_v1_reasoning_output() -> None:
    """Test v1 format with reasoning enabled."""
    # TODO: xfail this in case reasoning isn't returned by the model
    llm = ChatOllama(
        model=REASONING_MODEL, output_version="v1", reasoning=True, temperature=0
    )

    result = llm.invoke(
        [HumanMessage("How many 'r' in 'strawberry'? Explain your reasoning.")]
    )

    assert isinstance(result.content, list)

    reasoning_blocks = [
        b
        for b in result.content
        if (isinstance(b, dict) and b.get("type") == "reasoning")
    ]
    text_blocks = [
        b for b in result.content if (isinstance(b, dict) and b.get("type") == "text")
    ]

    assert len(reasoning_blocks) >= 1
    assert len(text_blocks) >= 1
    for block in result.content:
        assert isinstance(block, dict)
        assert "id" in block
        assert block.get("id", "").startswith(LC_AUTO_PREFIX)


def test_v1_streaming() -> None:
    """Test v1 format with streaming."""
    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)

    chunks = list(llm.stream([HumanMessage("Count to 3")]))

    for chunk in chunks:
        assert isinstance(chunk.content, list)
        if chunk.content:  # Skip empty chunks
            for block in chunk.content:
                assert isinstance(block, dict)
                assert "type" in block


@pytest.mark.asyncio
async def test_v1_async() -> None:
    """Test v1 format with async invocation."""
    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)

    result = await llm.ainvoke([HumanMessage("Say hello")])

    assert isinstance(result.content, list)
    assert len(result.content) >= 1
    text_block = result.content[0]
    assert isinstance(text_block, dict)
    assert text_block["type"] == "text"
    assert "id" in text_block
    assert text_block["id"].startswith(LC_AUTO_PREFIX)


@pytest.mark.asyncio
async def test_v1_async_streaming() -> None:
    """Test v1 format with async streaming."""
    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)

    chunks = [chunk async for chunk in llm.astream([HumanMessage("Tell me a joke")])]

    for chunk in chunks:
        assert isinstance(chunk.content, list)
        if chunk.content:  # Skip empty chunks
            for block in chunk.content:
                assert isinstance(block, dict)
                assert "type" in block


def test_v1_structured_output() -> None:
    """Test v1 format with structured output (tool calling)."""

    class Person(BaseModel):
        """Information about a person."""

        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)
    structured_llm = llm.with_structured_output(Person, method="function_calling")

    result = structured_llm.invoke("John is 25 years old")

    assert isinstance(result, Person)
    assert result.name
    assert result.age


def test_v1_structured_output_streaming() -> None:
    """Test v1 format with structured output streaming."""

    class Color(BaseModel):
        """A color description."""

        name: str = Field(description="The color name")

    llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)
    structured_llm = llm.with_structured_output(Color, method="function_calling")

    chunks = list(structured_llm.stream("The sky is blue"))

    for chunk in chunks:
        assert isinstance(chunk, Color)


def test_v1_vs_v0_comparison() -> None:
    """Test that v0 and v1 formats produce different content structures."""

    message = [HumanMessage("What is your name?")]

    # v0 format
    llm_v0 = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v0", temperature=0)
    result_v0 = llm_v0.invoke(message)

    # v1 format
    llm_v1 = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1", temperature=0)
    result_v1 = llm_v1.invoke(message)

    # v0 should have string content
    assert isinstance(result_v0.content, str)

    # v1 should have list content
    assert isinstance(result_v1.content, list)
    assert len(result_v1.content) >= 1
    for block in result_v1.content:
        if isinstance(block, dict):
            assert "id" in block
            assert block["id"].startswith(LC_AUTO_PREFIX)


def test_v1_content_blocks_property() -> None:
    """Test that content_blocks property works with v1 format."""
    llm = ChatOllama(
        model=REASONING_MODEL, output_version="v1", reasoning=True, temperature=0
    )

    result = llm.invoke([HumanMessage("Explain why 1+1=2")])

    # content_blocks should return the same as .content w/ v1
    assert result.content_blocks == result.content
    assert isinstance(result.content_blocks, list)
    for block in result.content_blocks:
        assert isinstance(block, dict)
        assert "type" in block
        assert "id" in block
