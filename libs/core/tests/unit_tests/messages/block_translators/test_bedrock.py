from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import content as types


def test_convert_to_v1_from_bedrock() -> None:
    message = AIMessage(
        [
            {"type": "thinking", "thinking": "foo", "signature": "foo_signature"},
            {"type": "text", "text": "Let's call a tool."},
            {
                "type": "tool_use",
                "id": "abc_123",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            },
            {
                "type": "text",
                "text": "It's sunny.",
                "citations": [
                    {
                        "type": "search_result_location",
                        "cited_text": "The weather is sunny.",
                        "source": "source_123",
                        "title": "Document Title",
                        "search_result_index": 1,
                        "start_block_index": 0,
                        "end_block_index": 2,
                    },
                    {"bar": "baz"},
                ],
            },
            {"type": "something_else", "foo": "bar"},
        ],
        response_metadata={
            "model_provider": "bedrock",
            "model_name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        },
    )
    expected_content: list[types.ContentBlock] = [
        {
            "type": "reasoning",
            "reasoning": "foo",
            "extras": {"signature": "foo_signature"},
        },
        {"type": "text", "text": "Let's call a tool."},
        {
            "type": "tool_call",
            "id": "abc_123",
            "name": "get_weather",
            "args": {"location": "San Francisco"},
        },
        {
            "type": "text",
            "text": "It's sunny.",
            "annotations": [
                {
                    "type": "citation",
                    "title": "Document Title",
                    "cited_text": "The weather is sunny.",
                    "extras": {
                        "source": "source_123",
                        "search_result_index": 1,
                        "start_block_index": 0,
                        "end_block_index": 2,
                    },
                },
                {"type": "non_standard_annotation", "value": {"bar": "baz"}},
            ],
        },
        {
            "type": "non_standard",
            "value": {"type": "something_else", "foo": "bar"},
        },
    ]
    assert message.content_blocks == expected_content

    # Check no mutation
    assert message.content != expected_content

    # Test with a non-Anthropic message
    message = AIMessage(
        [
            {"type": "text", "text": "Let's call a tool."},
            {"type": "something_else", "foo": "bar"},
        ],
        tool_calls=[
            {
                "type": "tool_call",
                "id": "abc_123",
                "name": "get_weather",
                "args": {"location": "San Francisco"},
            }
        ],
        response_metadata={"model_provider": "bedrock"},
    )
    expected_content = [
        {"type": "text", "text": "Let's call a tool."},
        {
            "type": "non_standard",
            "value": {"type": "something_else", "foo": "bar"},
        },
        {
            "type": "tool_call",
            "id": "abc_123",
            "name": "get_weather",
            "args": {"location": "San Francisco"},
        },
    ]
    assert message.content_blocks == expected_content


def test_convert_to_v1_from_bedrock_chunk() -> None:
    chunks = [
        AIMessageChunk(
            content=[{"text": "Looking ", "type": "text", "index": 0}],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[{"text": "now.", "type": "text", "index": 0}],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[
                {
                    "type": "tool_use",
                    "name": "get_weather",
                    "input": {},
                    "id": "toolu_abc123",
                    "index": 1,
                }
            ],
            tool_call_chunks=[
                {
                    "type": "tool_call_chunk",
                    "name": "get_weather",
                    "args": "",
                    "id": "toolu_abc123",
                    "index": 1,
                }
            ],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[{"type": "input_json_delta", "partial_json": "", "index": 1}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "",
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[
                {"type": "input_json_delta", "partial_json": '{"loca', "index": 1}
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"loca',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[
                {"type": "input_json_delta", "partial_json": 'tion": "San ', "index": 1}
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'tion": "San ',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
            response_metadata={"model_provider": "bedrock"},
        ),
        AIMessageChunk(
            content=[
                {"type": "input_json_delta", "partial_json": 'Francisco"}', "index": 1}
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'Francisco"}',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
            response_metadata={"model_provider": "bedrock"},
        ),
    ]
    expected_contents: list[types.ContentBlock] = [
        {"type": "text", "text": "Looking ", "index": 0},
        {"type": "text", "text": "now.", "index": 0},
        {
            "type": "tool_call_chunk",
            "name": "get_weather",
            "args": "",
            "id": "toolu_abc123",
            "index": 1,
        },
        {"name": None, "args": "", "id": None, "index": 1, "type": "tool_call_chunk"},
        {
            "name": None,
            "args": '{"loca',
            "id": None,
            "index": 1,
            "type": "tool_call_chunk",
        },
        {
            "name": None,
            "args": 'tion": "San ',
            "id": None,
            "index": 1,
            "type": "tool_call_chunk",
        },
        {
            "name": None,
            "args": 'Francisco"}',
            "id": None,
            "index": 1,
            "type": "tool_call_chunk",
        },
    ]
    for chunk, expected in zip(chunks, expected_contents):
        assert chunk.content_blocks == [expected]

    full: Optional[AIMessageChunk] = None
    for chunk in chunks:
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    expected_content = [
        {"type": "text", "text": "Looking now.", "index": 0},
        {
            "type": "tool_use",
            "name": "get_weather",
            "partial_json": '{"location": "San Francisco"}',
            "input": {},
            "id": "toolu_abc123",
            "index": 1,
        },
    ]
    assert full.content == expected_content

    expected_content_blocks = [
        {"type": "text", "text": "Looking now.", "index": 0},
        {
            "type": "tool_call_chunk",
            "name": "get_weather",
            "args": '{"location": "San Francisco"}',
            "id": "toolu_abc123",
            "index": 1,
        },
    ]
    assert full.content_blocks == expected_content_blocks


def test_convert_to_v1_from_bedrock_input() -> None:
    message = HumanMessage(
        [
            {"type": "text", "text": "foo"},
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "data": "<base64 data>",
                    "media_type": "application/pdf",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "<document url>",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "The grass is green"},
                        {"type": "text", "text": "The sky is blue"},
                    ],
                },
                "citations": {"enabled": True},
            },
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "data": "<plain text data>",
                    "media_type": "text/plain",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "<base64 image data>",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "<image url>",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": "<image file id>",
                },
            },
            {
                "type": "document",
                "source": {"type": "file", "file_id": "<pdf file id>"},
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "foo"},
        {
            "type": "file",
            "base64": "<base64 data>",
            "mime_type": "application/pdf",
        },
        {
            "type": "file",
            "url": "<document url>",
        },
        {
            "type": "non_standard",
            "value": {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "The grass is green"},
                        {"type": "text", "text": "The sky is blue"},
                    ],
                },
                "citations": {"enabled": True},
            },
        },
        {
            "type": "text-plain",
            "text": "<plain text data>",
            "mime_type": "text/plain",
        },
        {
            "type": "image",
            "base64": "<base64 image data>",
            "mime_type": "image/jpeg",
        },
        {
            "type": "image",
            "url": "<image url>",
        },
        {
            "type": "image",
            "id": "<image file id>",
        },
        {
            "type": "file",
            "id": "<pdf file id>",
        },
    ]

    assert message.content_blocks == expected
