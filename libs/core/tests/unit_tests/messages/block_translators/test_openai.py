from typing import Optional

import pytest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from tests.unit_tests.language_models.chat_models.test_base import (
    _content_blocks_equal_ignore_id,
)


def test_convert_to_v1_from_responses() -> None:
    message = AIMessage(
        [
            {"type": "reasoning", "id": "abc123", "summary": []},
            {
                "type": "reasoning",
                "id": "abc234",
                "summary": [
                    {"type": "summary_text", "text": "foo bar"},
                    {"type": "summary_text", "text": "baz"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
            {
                "type": "function_call",
                "call_id": "call_234",
                "name": "get_weather_2",
                "arguments": '{"location": "New York"}',
                "id": "fc_123",
            },
            {"type": "text", "text": "Hello "},
            {
                "type": "text",
                "text": "world",
                "annotations": [
                    {"type": "url_citation", "url": "https://example.com"},
                    {
                        "type": "file_citation",
                        "filename": "my doc",
                        "index": 1,
                        "file_id": "file_123",
                    },
                    {"bar": "baz"},
                ],
            },
            {"type": "image_generation_call", "id": "ig_123", "result": "..."},
            {
                "type": "file_search_call",
                "id": "fs_123",
                "queries": ["query for file search"],
                "results": [{"file_id": "file-123"}],
                "status": "completed",
            },
            {"type": "something_else", "foo": "bar"},
        ],
        tool_calls=[
            {
                "type": "tool_call",
                "id": "call_123",
                "name": "get_weather",
                "args": {"location": "San Francisco"},
            },
            {
                "type": "tool_call",
                "id": "call_234",
                "name": "get_weather_2",
                "args": {"location": "New York"},
            },
        ],
        response_metadata={"model_provider": "openai"},
    )
    expected_content: list[types.ContentBlock] = [
        {"type": "reasoning", "id": "abc123"},
        {"type": "reasoning", "id": "abc234", "reasoning": "foo bar"},
        {"type": "reasoning", "id": "abc234", "reasoning": "baz"},
        {
            "type": "tool_call",
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "San Francisco"},
        },
        {
            "type": "tool_call",
            "id": "call_234",
            "name": "get_weather_2",
            "args": {"location": "New York"},
            "extras": {"item_id": "fc_123"},
        },
        {"type": "text", "text": "Hello "},
        {
            "type": "text",
            "text": "world",
            "annotations": [
                {"type": "citation", "url": "https://example.com"},
                {
                    "type": "citation",
                    "title": "my doc",
                    "extras": {"file_id": "file_123", "index": 1},
                },
                {"type": "non_standard_annotation", "value": {"bar": "baz"}},
            ],
        },
        {"type": "image", "base64": "...", "id": "ig_123"},
        {
            "type": "server_tool_call",
            "name": "file_search",
            "id": "fs_123",
            "args": {"queries": ["query for file search"]},
        },
        {
            "type": "server_tool_result",
            "tool_call_id": "fs_123",
            "output": [{"file_id": "file-123"}],
            "status": "success",
        },
        {
            "type": "non_standard",
            "value": {"type": "something_else", "foo": "bar"},
        },
    ]
    assert message.content_blocks == expected_content

    # Check no mutation
    assert message.content != expected_content


def test_convert_to_v1_from_responses_chunk() -> None:
    chunks = [
        AIMessageChunk(
            content=[{"type": "reasoning", "id": "abc123", "summary": [], "index": 0}],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "summary": [
                        {"type": "summary_text", "text": "foo ", "index": 0},
                    ],
                    "index": 1,
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "summary": [
                        {"type": "summary_text", "text": "bar", "index": 0},
                    ],
                    "index": 1,
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "summary": [
                        {"type": "summary_text", "text": "baz", "index": 1},
                    ],
                    "index": 1,
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
    ]
    expected_chunks = [
        AIMessageChunk(
            content=[{"type": "reasoning", "id": "abc123", "index": "lc_rs_305f30"}],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "reasoning": "foo ",
                    "index": "lc_rs_315f30",
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "reasoning": "bar",
                    "index": "lc_rs_315f30",
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "reasoning",
                    "id": "abc234",
                    "reasoning": "baz",
                    "index": "lc_rs_315f31",
                }
            ],
            response_metadata={"model_provider": "openai"},
        ),
    ]
    for chunk, expected in zip(chunks, expected_chunks):
        assert chunk.content_blocks == expected.content_blocks

    full: Optional[AIMessageChunk] = None
    for chunk in chunks:
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    expected_content = [
        {"type": "reasoning", "id": "abc123", "summary": [], "index": 0},
        {
            "type": "reasoning",
            "id": "abc234",
            "summary": [
                {"type": "summary_text", "text": "foo bar", "index": 0},
                {"type": "summary_text", "text": "baz", "index": 1},
            ],
            "index": 1,
        },
    ]
    assert full.content == expected_content

    expected_content_blocks = [
        {"type": "reasoning", "id": "abc123", "index": "lc_rs_305f30"},
        {
            "type": "reasoning",
            "id": "abc234",
            "reasoning": "foo bar",
            "index": "lc_rs_315f30",
        },
        {
            "type": "reasoning",
            "id": "abc234",
            "reasoning": "baz",
            "index": "lc_rs_315f31",
        },
    ]
    assert full.content_blocks == expected_content_blocks


def test_convert_to_v1_from_openai_input() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."},
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": "<base64 string>",
                },
            },
            {
                "type": "file",
                "file": {
                    "filename": "draconomicon.pdf",
                    "file_data": "data:application/pdf;base64,<base64 string>",
                },
            },
            {
                "type": "file",
                "file": {"file_id": "<file id>"},
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "Hello"},
        {
            "type": "image",
            "url": "https://example.com/image.png",
        },
        {
            "type": "image",
            "base64": "/9j/4AAQSkZJRg...",
            "mime_type": "image/jpeg",
        },
        {
            "type": "audio",
            "base64": "<base64 string>",
            "mime_type": "audio/wav",
        },
        {
            "type": "file",
            "base64": "<base64 string>",
            "mime_type": "application/pdf",
            "extras": {"filename": "draconomicon.pdf"},
        },
        {"type": "file", "file_id": "<file id>"},
    ]

    assert _content_blocks_equal_ignore_id(message.content_blocks, expected)


def test_compat_responses_v03() -> None:
    # Check compatibility with v0.3 legacy message format
    message_v03 = AIMessage(
        content=[
            {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
        ],
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [
                    {"type": "summary_text", "text": "summary 1"},
                    {"type": "summary_text", "text": "summary 2"},
                ],
            },
            "tool_outputs": [
                {
                    "type": "web_search_call",
                    "id": "websearch_123",
                    "status": "completed",
                }
            ],
            "refusal": "I cannot assist with that.",
            "__openai_function_call_ids__": {"call_abc": "fc_abc"},
        },
        tool_calls=[
            {"type": "tool_call", "name": "my_tool", "args": {"x": 3}, "id": "call_abc"}
        ],
        response_metadata={"id": "resp_123", "model_provider": "openai"},
        id="msg_123",
    )

    expected_content: list[types.ContentBlock] = [
        {"type": "reasoning", "id": "rs_123", "reasoning": "summary 1"},
        {"type": "reasoning", "id": "rs_123", "reasoning": "summary 2"},
        {
            "type": "text",
            "text": "Hello, world!",
            "annotations": [
                {"type": "non_standard_annotation", "value": {"type": "foo"}}
            ],
            "id": "msg_123",
        },
        {
            "type": "non_standard",
            "value": {"type": "refusal", "refusal": "I cannot assist with that."},
        },
        {
            "type": "tool_call",
            "name": "my_tool",
            "args": {"x": 3},
            "id": "call_abc",
            "extras": {"item_id": "fc_abc"},
        },
        {
            "type": "server_tool_call",
            "name": "web_search",
            "args": {},
            "id": "websearch_123",
        },
        {
            "type": "server_tool_result",
            "tool_call_id": "websearch_123",
            "status": "success",
        },
    ]
    assert message_v03.content_blocks == expected_content

    # Test chunks
    ## Tool calls
    chunk_1 = AIMessageChunk(
        content=[],
        additional_kwargs={"__openai_function_call_ids__": {"call_abc": "fc_abc"}},
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": "my_tool",
                "args": "",
                "id": "call_abc",
                "index": 0,
            }
        ],
        response_metadata={"model_provider": "openai"},
    )
    expected_content = [
        {
            "type": "tool_call_chunk",
            "name": "my_tool",
            "args": "",
            "id": "call_abc",
            "index": 0,
            "extras": {"item_id": "fc_abc"},
        }
    ]
    assert chunk_1.content_blocks == expected_content

    chunk_2 = AIMessageChunk(
        content=[],
        additional_kwargs={"__openai_function_call_ids__": {}},
        tool_call_chunks=[
            {
                "type": "tool_call_chunk",
                "name": None,
                "args": "{",
                "id": None,
                "index": 0,
            }
        ],
    )
    expected_content = [
        {"type": "tool_call_chunk", "name": None, "args": "{", "id": None, "index": 0}
    ]

    chunk = chunk_1 + chunk_2
    expected_content = [
        {
            "type": "tool_call_chunk",
            "name": "my_tool",
            "args": "{",
            "id": "call_abc",
            "index": 0,
            "extras": {"item_id": "fc_abc"},
        }
    ]
    assert chunk.content_blocks == expected_content

    ## Reasoning
    chunk_1 = AIMessageChunk(
        content=[],
        additional_kwargs={
            "reasoning": {"id": "rs_abc", "summary": [], "type": "reasoning"}
        },
        response_metadata={"model_provider": "openai"},
    )
    expected_content = [{"type": "reasoning", "id": "rs_abc"}]
    assert chunk_1.content_blocks == expected_content

    chunk_2 = AIMessageChunk(
        content=[],
        additional_kwargs={
            "reasoning": {
                "summary": [
                    {"index": 0, "type": "summary_text", "text": "reasoning text"}
                ]
            }
        },
        response_metadata={"model_provider": "openai"},
    )
    expected_content = [{"type": "reasoning", "reasoning": "reasoning text"}]
    assert chunk_2.content_blocks == expected_content

    chunk = chunk_1 + chunk_2
    expected_content = [
        {"type": "reasoning", "reasoning": "reasoning text", "id": "rs_abc"}
    ]
    assert chunk.content_blocks == expected_content


def test_convert_to_openai_data_block() -> None:
    # Chat completions
    ## Image / url
    block = {
        "type": "image",
        "url": "https://example.com/test.png",
    }
    expected = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/test.png"},
    }
    result = convert_to_openai_data_block(block)
    assert result == expected

    ## Image / base64
    block = {
        "type": "image",
        "base64": "<base64 string>",
        "mime_type": "image/png",
    }
    expected = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,<base64 string>"},
    }
    result = convert_to_openai_data_block(block)
    assert result == expected

    ## File / url
    block = {
        "type": "file",
        "url": "https://example.com/test.pdf",
    }
    with pytest.raises(ValueError, match="does not support"):
        result = convert_to_openai_data_block(block)

    ## File / base64
    block = {
        "type": "file",
        "base64": "<base64 string>",
        "mime_type": "application/pdf",
        "filename": "test.pdf",
    }
    expected = {
        "type": "file",
        "file": {
            "file_data": "data:application/pdf;base64,<base64 string>",
            "filename": "test.pdf",
        },
    }
    result = convert_to_openai_data_block(block)
    assert result == expected

    ## File / file ID
    block = {
        "type": "file",
        "file_id": "file-abc123",
    }
    expected = {"type": "file", "file": {"file_id": "file-abc123"}}
    result = convert_to_openai_data_block(block)
    assert result == expected

    ## Audio / base64
    block = {
        "type": "audio",
        "base64": "<base64 string>",
        "mime_type": "audio/wav",
    }
    expected = {
        "type": "input_audio",
        "input_audio": {"data": "<base64 string>", "format": "wav"},
    }
    result = convert_to_openai_data_block(block)
    assert result == expected

    # Responses
    ## Image / url
    block = {
        "type": "image",
        "url": "https://example.com/test.png",
    }
    expected = {"type": "input_image", "image_url": "https://example.com/test.png"}
    result = convert_to_openai_data_block(block, api="responses")
    assert result == expected

    ## Image / base64
    block = {
        "type": "image",
        "base64": "<base64 string>",
        "mime_type": "image/png",
    }
    expected = {
        "type": "input_image",
        "image_url": "data:image/png;base64,<base64 string>",
    }
    result = convert_to_openai_data_block(block, api="responses")
    assert result == expected

    ## File / url
    block = {
        "type": "file",
        "url": "https://example.com/test.pdf",
    }
    expected = {"type": "input_file", "file_url": "https://example.com/test.pdf"}

    ## File / base64
    block = {
        "type": "file",
        "base64": "<base64 string>",
        "mime_type": "application/pdf",
        "filename": "test.pdf",
    }
    expected = {
        "type": "input_file",
        "file_data": "data:application/pdf;base64,<base64 string>",
        "filename": "test.pdf",
    }
    result = convert_to_openai_data_block(block, api="responses")
    assert result == expected

    ## File / file ID
    block = {
        "type": "file",
        "file_id": "file-abc123",
    }
    expected = {"type": "input_file", "file_id": "file-abc123"}
    result = convert_to_openai_data_block(block, api="responses")
    assert result == expected
