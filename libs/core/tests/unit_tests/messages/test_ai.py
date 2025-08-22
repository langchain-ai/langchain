from typing import Union, cast

import pytest

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
    add_ai_message_chunks,
    add_usage,
    subtract_usage,
)
from langchain_core.messages.tool import invalid_tool_call as create_invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk


def test_serdes_message() -> None:
    msg = AIMessage(
        content=[{"text": "blah", "type": "text"}],
        tool_calls=[create_tool_call(name="foo", args={"bar": 1}, id="baz")],
        invalid_tool_calls=[
            create_invalid_tool_call(name="foobad", args="blah", id="booz", error="bad")
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "type": "ai",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": "bad",
                    "type": "invalid_tool_call",
                }
            ],
        },
    }
    actual = dumpd(msg)
    assert actual == expected
    assert load(actual) == msg


def test_serdes_message_chunk() -> None:
    chunk = AIMessageChunk(
        content=[{"text": "blah", "type": "text"}],
        tool_call_chunks=[
            create_tool_call_chunk(name="foo", args='{"bar": 1}', id="baz", index=0),
            create_tool_call_chunk(
                name="foobad",
                args="blah",
                id="booz",
                index=1,
            ),
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessageChunk"],
        "kwargs": {
            "type": "AIMessageChunk",
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [
                {"name": "foo", "args": {"bar": 1}, "id": "baz", "type": "tool_call"}
            ],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            "tool_call_chunks": [
                {
                    "name": "foo",
                    "args": '{"bar": 1}',
                    "id": "baz",
                    "index": 0,
                    "type": "tool_call_chunk",
                },
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "index": 1,
                    "type": "tool_call_chunk",
                },
            ],
        },
    }
    actual = dumpd(chunk)
    assert actual == expected
    assert load(actual) == chunk


def test_add_usage_both_none() -> None:
    result = add_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = add_usage(usage, None)
    assert result == usage


def test_add_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = add_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=15, output_tokens=30, total_tokens=45)


def test_add_usage_with_details() -> None:
    usage1 = UsageMetadata(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        input_token_details=InputTokenDetails(audio=5),
        output_token_details=OutputTokenDetails(reasoning=10),
    )
    usage2 = UsageMetadata(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_token_details=InputTokenDetails(audio=3),
        output_token_details=OutputTokenDetails(reasoning=5),
    )
    result = add_usage(usage1, usage2)
    assert result["input_token_details"]["audio"] == 8
    assert result["output_token_details"]["reasoning"] == 15


def test_subtract_usage_both_none() -> None:
    result = subtract_usage(None, None)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_subtract_usage_one_none() -> None:
    usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage, None)
    assert result == usage


def test_subtract_usage_both_present() -> None:
    usage1 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    usage2 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)


def test_subtract_usage_with_negative_result() -> None:
    usage1 = UsageMetadata(input_tokens=5, output_tokens=10, total_tokens=15)
    usage2 = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
    result = subtract_usage(usage1, usage2)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_add_ai_message_chunks_usage() -> None:
    chunks = [
        AIMessageChunk(content="", usage_metadata=None),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2, output_tokens=3, total_tokens=5
            ),
        ),
        AIMessageChunk(
            content="",
            usage_metadata=UsageMetadata(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                input_token_details=InputTokenDetails(audio=1, cache_read=1),
                output_token_details=OutputTokenDetails(audio=1, reasoning=2),
            ),
        ),
    ]
    combined = add_ai_message_chunks(*chunks)
    assert combined == AIMessageChunk(
        content="",
        usage_metadata=UsageMetadata(
            input_tokens=4,
            output_tokens=6,
            total_tokens=10,
            input_token_details=InputTokenDetails(audio=1, cache_read=1),
            output_token_details=OutputTokenDetails(audio=1, reasoning=2),
        ),
    )


def test_content_blocks() -> None:
    message = AIMessage(
        "",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}}
    ]
    assert message.content == ""

    message = AIMessage(
        "foo",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"}
        ],
    )
    assert len(message.content_blocks) == 2
    assert message.content_blocks[0]["type"] == "text"
    assert message.content_blocks[1]["type"] == "tool_call"
    assert message.content_blocks == [
        {"type": "text", "text": "foo"},
        {"type": "tool_call", "id": "abc_123", "name": "foo", "args": {"a": "b"}},
    ]
    assert message.content == "foo"

    # With standard blocks
    standard_content: list[types.ContentBlock] = [
        {"type": "reasoning", "reasoning": "foo"},
        {"type": "text", "text": "bar"},
        {
            "type": "text",
            "text": "baz",
            "annotations": [{"type": "citation", "url": "http://example.com"}],
        },
        {
            "type": "image",
            "url": "http://example.com/image.png",
            "extras": {"foo": "bar"},
        },
        {
            "type": "non_standard",
            "value": {"custom_key": "custom_value", "another_key": 123},
        },
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
    ]
    missing_tool_call: types.ToolCall = {
        "type": "tool_call",
        "name": "bar",
        "args": {"c": "d"},
        "id": "abc_234",
    }
    message = AIMessage(
        content_blocks=standard_content,
        tool_calls=[
            {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
            missing_tool_call,
        ],
    )
    assert message.content_blocks == [*standard_content, missing_tool_call]

    # Check we auto-populate tool_calls
    standard_content = [
        {"type": "text", "text": "foo"},
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"a": "b"},
            "id": "abc_123",
        },
        missing_tool_call,
    ]
    message = AIMessage(content_blocks=standard_content)
    assert message.tool_calls == [
        {"type": "tool_call", "name": "foo", "args": {"a": "b"}, "id": "abc_123"},
        missing_tool_call,
    ]

    # Chunks
    message = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": "",
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    assert len(message.content_blocks) == 1
    assert message.content_blocks[0]["type"] == "tool_call_chunk"
    assert message.content_blocks == [
        {
            "type": "tool_call_chunk",
            "name": "foo",
            "args": "",
            "id": "abc_123",
            "index": 0,
        }
    ]
    assert message.content == ""

    # Test we parse tool call chunks into tool calls for v1 content
    chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "foo",
                "args": '{"foo": "b',
                "id": "abc_123",
                "index": 0,
            }
        ],
    )
    chunk_1.content = cast("Union[str, list[Union[str, dict]]]", chunk_1.content_blocks)
    chunk_1.response_metadata["output_version"] = "v1"

    chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": 'ar"}',
                "id": "abc_123",
                "index": 0,
            }
        ],
        chunk_position="last",
    )
    chunk_2.content = cast("Union[str, list[Union[str, dict]]]", chunk_2.content_blocks)

    chunk = chunk_1 + chunk_2
    assert chunk.content == [
        {
            "type": "tool_call",
            "name": "foo",
            "args": {"foo": "bar"},
            "id": "abc_123",
        }
    ]

    # Non-standard
    standard_content_1: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar "}}
    ]
    standard_content_2: list[types.ContentBlock] = [
        {"type": "non_standard", "index": 0, "value": {"foo": "baz"}}
    ]
    chunk_1 = AIMessageChunk(
        content=cast("Union[str, list[Union[str, dict]]]", standard_content_1)
    )
    chunk_2 = AIMessageChunk(
        content=cast("Union[str, list[Union[str, dict]]]", standard_content_2)
    )
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {"type": "non_standard", "index": 0, "value": {"foo": "bar baz"}},
    ]

    # Test non-standard + non-standard
    chunk_1 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "non_standard_tool", "foo": "bar"},
            }
        ]
    )
    chunk_2 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "a"},
            }
        ]
    )
    chunk_3 = AIMessageChunk(
        content=[
            {
                "type": "non_standard",
                "index": 0,
                "value": {"type": "input_json_delta", "partial_json": "b"},
            }
        ]
    )
    merged_chunk = chunk_1 + chunk_2 + chunk_3
    assert merged_chunk.content == [
        {
            "type": "non_standard",
            "index": 0,
            "value": {"type": "non_standard_tool", "foo": "bar", "partial_json": "ab"},
        }
    ]

    # Test standard + non-standard with same index
    standard_content_1 = [
        {"type": "web_search_call", "id": "ws_123", "query": "web query", "index": 0}
    ]
    standard_content_2 = [{"type": "non_standard", "value": {"foo": "bar"}, "index": 0}]
    chunk_1 = AIMessageChunk(
        content=cast("Union[str, list[Union[str, dict]]]", standard_content_1)
    )
    chunk_2 = AIMessageChunk(
        content=cast("Union[str, list[Union[str, dict]]]", standard_content_2)
    )
    merged_chunk = chunk_1 + chunk_2
    assert merged_chunk.content == [
        {
            "type": "web_search_call",
            "id": "ws_123",
            "query": "web query",
            "index": 0,
            "extras": {"foo": "bar"},
        }
    ]


def test_provider_warns() -> None:
    # Test that major providers warn if content block standardization is not yet
    # implemented.
    # This test should be removed when all major providers support content block
    # standardization.
    message = AIMessage("Hello.", response_metadata={"model_provider": "groq"})
    with pytest.warns(match="not yet fully supported for Groq"):
        content_blocks = message.content_blocks

    assert content_blocks == [{"type": "text", "text": "Hello."}]
