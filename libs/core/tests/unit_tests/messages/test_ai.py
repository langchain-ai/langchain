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
    assert load(actual, allowed_objects=[AIMessage]) == msg


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
    assert load(actual, allowed_objects=[AIMessageChunk]) == chunk


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


def test_init_tool_calls() -> None:
    # Test we add "type" key on init
    msg = AIMessage("", tool_calls=[{"name": "foo", "args": {"a": "b"}, "id": "abc"}])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "tool_call"

    # Test we can assign without adding type key
    msg.tool_calls = [{"name": "bar", "args": {"c": "d"}, "id": "def"}]


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
        content="The answer is 42.", additional_kwargs={"other_field": "some value"}
    )
    content_blocks = message.content_blocks
    assert len(content_blocks) == 1
    assert content_blocks[0]["type"] == "text"


def test_aimessagechunk_fragmented_tool_call_chunks() -> None:
    """Test that fragmented tool call chunks are properly merged (issue #35514).

    Some providers (e.g., OpenRouter with deepseek/deepseek-v3.2) fragment tool
    call arguments across multiple SSE chunks. This test ensures that
    AIMessageChunk properly concatenates args from all chunks before parsing,
    preventing tools from being executed with empty or partial arguments.
    """
    # Simulate fragmented tool call chunks as described in issue #35514
    chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "my_tool",
                "args": "",  # First chunk arrives with empty args
                "id": "call_34db6a0a9d314d71a8a74b3d",
                "index": 0,
            }
        ],
    )

    chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",  # Name is empty in subsequent chunks
                "args": '{"url": "http://mywebsite.com"}',
                "id": "call_34db6a0a9d314d71a8a74b3d",
                "index": 0,
            }
        ],
    )

    # When chunks are merged, the tool call should have complete args
    merged = chunk_1 + chunk_2

    # Should have exactly one tool call with complete arguments
    assert len(merged.tool_calls) == 1
    assert merged.tool_calls[0]["name"] == "my_tool"
    assert merged.tool_calls[0]["args"] == {"url": "http://mywebsite.com"}
    assert merged.tool_calls[0]["id"] == "call_34db6a0a9d314d71a8a74b3d"

    # Test with multiple fragmented arguments (as shown in the issue)
    chunk_3 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "my_tool",
                "args": "",  # Empty args in first chunk
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    chunk_4 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": "{",  # Opening brace in second chunk
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    chunk_5 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": '"url": "http://mywebsite.com"',  # Key-value pair
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    chunk_6 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": "}",  # Closing brace in final chunk
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    merged_fragments = chunk_3 + chunk_4 + chunk_5 + chunk_6

    # Should properly concatenate all args fragments
    assert len(merged_fragments.tool_calls) == 1
    assert merged_fragments.tool_calls[0]["name"] == "my_tool"
    assert merged_fragments.tool_calls[0]["args"] == {
        "url": "http://mywebsite.com"
    }

    # Test with multiple tool calls in a single chunk
    multi_call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "tool_1",
                "args": '{"arg1": "val1"}',
                "id": "call_1",
                "index": 0,
            },
            {
                "type": "tool_call_chunk",
                "name": "tool_2",
                "args": '{"arg2": "val2"}',
                "id": "call_2",
                "index": 1,
            },
        ],
    )

    assert len(multi_call_chunk.tool_calls) == 2
    assert multi_call_chunk.tool_calls[0]["name"] == "tool_1"
    assert multi_call_chunk.tool_calls[0]["args"] == {"arg1": "val1"}
    assert multi_call_chunk.tool_calls[1]["name"] == "tool_2"
    assert multi_call_chunk.tool_calls[1]["args"] == {"arg2": "val2"}


def test_fragmented_tool_call_chunks() -> None:
    """Test that fragmented tool call arguments are properly accumulated.

    This tests the fix for issue #35514 where SSE fragmentation causes
    tool calls to be executed with empty args because each chunk is parsed
    individually instead of accumulating args first.

    Example from the issue:
    - Chunk 1: name="my_tool", args="", id="call_123"
    - Chunk 2: name="", args='{"url": "http://mywebsite.com"}', id=""

    The fix ensures args are accumulated by id before parsing.
    """
    # Simulate fragmented tool call chunks from OpenRouter/deepseek
    chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "my_tool",
                "args": "",
                "id": "call_123",
                "index": 0,
            }
        ],
    )

    # First chunk alone should NOT produce a valid tool call with empty args
    # because args="" parses to {} which would fail tool validation
    assert len(chunk_1.tool_calls) == 1
    # The tool call should be created but with empty args initially
    assert chunk_1.tool_calls[0]["name"] == "my_tool"
    assert chunk_1.tool_calls[0]["id"] == "call_123"

    chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": '{"url": "http://mywebsite.com"}',
                "id": "call_123",
                "index": 0,
            }
        ],
    )

    # When merged, args should be accumulated
    merged = chunk_1 + chunk_2
    assert len(merged.tool_calls) == 1
    assert merged.tool_calls[0]["name"] == "my_tool"
    assert merged.tool_calls[0]["id"] == "call_123"
    # The args should be properly accumulated and parsed
    assert merged.tool_calls[0]["args"] == {"url": "http://mywebsite.com"}

    # Test with multiple fragments for the same tool call
    chunk_3 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "my_tool",
                "args": "",
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    chunk_4 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": '{"url": "http://mysite.com", ',
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    chunk_5 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": '"timeout": 30}',
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    # Test with empty args in middle chunk
    chunk_6 = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "",
                "args": "",
                "id": "call_xyz",
                "index": 0,
            }
        ],
    )

    merged_fragments = chunk_3 + chunk_4 + chunk_5 + chunk_6

    # Should properly concatenate all args fragments
    assert len(merged_fragments.tool_calls) == 1
    assert merged_fragments.tool_calls[0]["name"] == "my_tool"
    assert merged_fragments.tool_calls[0]["args"] == {
        "url": "http://mysite.com",
        "timeout": 30
    }
