"""Unit tests for ToolMessage and ToolMessageChunk."""

import pytest

from langchain_core.messages import (
    AIMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.tool import add_tool_message_chunks


def test_tool_message_chunk_add_pairwise() -> None:
    first = ToolMessageChunk(
        content="a",
        tool_call_id="call_1",
        artifact={"k": 1},
        status="error",
        additional_kwargs={"foo": "bar"},
        response_metadata={"meta1": 1},
    )
    second = ToolMessageChunk(
        content="b",
        tool_call_id="call_1",
        artifact={"k2": 2},
        status="error",
        additional_kwargs={"baz": "qux"},
        response_metadata={"meta2": 2},
    )

    combined = first + second
    assert isinstance(combined, ToolMessageChunk)
    assert combined.content == "ab"
    assert combined.tool_call_id == "call_1"
    assert combined.status == "error"
    assert combined.artifact == {"k": 1, "k2": 2}
    assert combined.additional_kwargs == {"foo": "bar", "baz": "qux"}
    assert combined.response_metadata == {"meta1": 1, "meta2": 2}


def test_tool_message_chunk_add_sequence() -> None:
    first = ToolMessageChunk(
        content="a",
        tool_call_id="call_1",
        artifact={"k": 1},
        status="error",
    )
    second = ToolMessageChunk(
        content="b",
        tool_call_id="call_1",
        artifact={"k2": 2},
    )
    third = ToolMessageChunk(
        content="c",
        tool_call_id="call_1",
    )

    combined = first + [second, third]
    assert isinstance(combined, ToolMessageChunk)
    assert combined.content == "abc"
    assert combined.tool_call_id == "call_1"
    assert combined.status == "error"
    assert combined.artifact == {"k": 1, "k2": 2}


def test_tool_message_chunk_add_cross_type() -> None:
    tool_chunk = ToolMessageChunk(
        content="a",
        tool_call_id="call_1",
        artifact={"k": 1},
        status="error",
    )
    ai_chunk = AIMessageChunk(content="z")

    # Tool + AI
    res_tool_ai = tool_chunk + ai_chunk
    assert isinstance(res_tool_ai, ToolMessageChunk)
    assert res_tool_ai.content == "az"
    assert res_tool_ai.tool_call_id == "call_1"
    assert res_tool_ai.status == "error"
    assert res_tool_ai.artifact == {"k": 1}

    # AI + Tool (mirror)
    res_ai_tool = ai_chunk + tool_chunk
    assert isinstance(res_ai_tool, AIMessageChunk)
    assert res_ai_tool.content == "za"

    # Tool + [AI]
    res_tool_ai_seq = tool_chunk + [ai_chunk]
    assert isinstance(res_tool_ai_seq, ToolMessageChunk)
    assert res_tool_ai_seq.content == "az"
    assert res_tool_ai_seq.tool_call_id == "call_1"


def test_tool_message_chunk_mismatched_id_raises() -> None:
    first = ToolMessageChunk(content="a", tool_call_id="call_1")
    second = ToolMessageChunk(content="b", tool_call_id="call_2")

    with pytest.raises(ValueError, match="Cannot concatenate ToolMessageChunks"):
        _ = first + second

    with pytest.raises(ValueError, match="Cannot concatenate ToolMessageChunks"):
        _ = first + [second]


def test_add_tool_message_chunks_helper() -> None:
    c1 = ToolMessageChunk(content="a", tool_call_id="call_1")
    c2 = ToolMessageChunk(content="b", tool_call_id="call_1")
    c3 = ToolMessageChunk(content="c", tool_call_id="call_1")

    combined = add_tool_message_chunks(c1, c2, c3)
    assert isinstance(combined, ToolMessageChunk)
    assert combined.content == "abc"
    assert combined.tool_call_id == "call_1"
