from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import content as types
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
