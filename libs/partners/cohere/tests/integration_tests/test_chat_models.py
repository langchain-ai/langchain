"""Test ChatCohere chat model."""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_cohere import ChatCohere


def test_stream() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_abatch() -> None:
    """Test streaming tokens from ChatCohere."""
    llm = ChatCohere()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatCohere."""
    llm = ChatCohere()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatCohere."""
    llm = ChatCohere()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatCohere."""
    llm = ChatCohere()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatCohere."""
    llm = ChatCohere()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_invoke_tool_calls() -> None:
    llm = ChatCohere(temperature=0)

    class Person(BaseModel):
        name: str = Field(type=str, description="The name of the person")
        age: int = Field(type=int, description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    result = tool_llm.invoke("Erick, 27 years old")

    assert isinstance(result, AIMessage)
    additional_kwargs = result.additional_kwargs
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }


def test_streaming_tool_call() -> None:
    llm = ChatCohere(temperature=0)

    class Person(BaseModel):
        name: str = Field(type=str, description="The name of the person")
        age: int = Field(type=int, description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it calls the tool
    strm = tool_llm.stream("Erick, 27 years old")

    additional_kwargs = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    assert "tool_calls" in additional_kwargs
    assert len(additional_kwargs["tool_calls"]) == 1
    assert additional_kwargs["tool_calls"][0]["function"]["name"] == "Person"
    assert json.loads(additional_kwargs["tool_calls"][0]["function"]["arguments"]) == {
        "name": "Erick",
        "age": 27,
    }


@pytest.mark.xfail(
    reason="Cohere models return empty output when a tool is passed in but not called."
)
def test_streaming_tool_call_no_tool_calls() -> None:
    llm = ChatCohere(temperature=0)

    class Person(BaseModel):
        name: str = Field(type=str, description="The name of the person")
        age: int = Field(type=int, description="The age of the person")

    tool_llm = llm.bind_tools([Person])

    # where it doesn't call the tool
    strm = tool_llm.stream("What is 2+2?")
    acc: Any = None
    for chunk in strm:
        assert isinstance(chunk, AIMessageChunk)
        acc = chunk if acc is None else acc + chunk
    assert acc.content != ""
    assert "tool_calls" not in acc.additional_kwargs
