"""Test ChatFireworks API wrapper

You will need FIREWORKS_API_KEY set in your environment to run these tests.
"""

import json
from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.pydantic_v1 import BaseModel

from langchain_fireworks import ChatFireworks


def test_chat_fireworks_call() -> None:
    """Test valid call to fireworks."""
    llm = ChatFireworks(  # type: ignore[call-arg]
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    resp = llm.invoke("Hello!")
    assert isinstance(resp, AIMessage)

    assert len(resp.content) > 0


def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = ChatFireworks(  # type: ignore[call-arg]
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"
    assert isinstance(resp.tool_calls, list)
    assert len(resp.tool_calls) == 1
    tool_call = resp.tool_calls[0]
    assert tool_call["name"] == "MyTool"
    assert tool_call["args"] == {"age": 27, "name": "Erick"}


def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""

    llm = ChatFireworks(  # type: ignore[call-arg]
        model="accounts/fireworks/models/firefunction-v1", temperature=0
    )

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice=True)

    resp = with_tool.invoke("Who was the 27 year old named Erick?")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


def test_stream() -> None:
    """Test streaming tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_astream() -> None:
    """Test streaming tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
    if chunks_with_token_counts != 1:
        raise AssertionError(
            "Expected exactly one chunk with token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )


async def test_abatch() -> None:
    """Test abatch tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatFireworks."""
    llm = ChatFireworks()  # type: ignore[call-arg]

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
