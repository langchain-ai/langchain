from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import content as types


def test_convert_to_v1_from_bedrock_converse() -> None:
    message = AIMessage(
        [
            {
                "type": "reasoning_content",
                "reasoning_content": {"text": "foo", "signature": "foo_signature"},
            },
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
                        "title": "Document Title",
                        "source_content": [{"text": "The weather is sunny."}],
                        "location": {
                            "document_char": {
                                "document_index": 0,
                                "start": 58,
                                "end": 96,
                            }
                        },
                    },
                    {
                        "title": "Document Title",
                        "source_content": [{"text": "The weather is sunny."}],
                        "location": {
                            "document_page": {"document_index": 0, "start": 1, "end": 2}
                        },
                    },
                    {
                        "title": "Document Title",
                        "source_content": [{"text": "The weather is sunny."}],
                        "location": {
                            "document_chunk": {
                                "document_index": 0,
                                "start": 1,
                                "end": 2,
                            }
                        },
                    },
                    {"bar": "baz"},
                ],
            },
            {"type": "something_else", "foo": "bar"},
        ],
        response_metadata={"model_provider": "bedrock_converse"},
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
                        "location": {
                            "document_char": {
                                "document_index": 0,
                                "start": 58,
                                "end": 96,
                            }
                        },
                    },
                },
                {
                    "type": "citation",
                    "title": "Document Title",
                    "cited_text": "The weather is sunny.",
                    "extras": {
                        "location": {
                            "document_page": {"document_index": 0, "start": 1, "end": 2}
                        },
                    },
                },
                {
                    "type": "citation",
                    "title": "Document Title",
                    "cited_text": "The weather is sunny.",
                    "extras": {
                        "location": {
                            "document_chunk": {
                                "document_index": 0,
                                "start": 1,
                                "end": 2,
                            }
                        }
                    },
                },
                {"type": "citation", "extras": {"bar": "baz"}},
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


def test_convert_to_v1_from_converse_chunk() -> None:
    chunks = [
        AIMessageChunk(
            content=[{"text": "Looking ", "type": "text", "index": 0}],
            response_metadata={"model_provider": "bedrock_converse"},
        ),
        AIMessageChunk(
            content=[{"text": "now.", "type": "text", "index": 0}],
            response_metadata={"model_provider": "bedrock_converse"},
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
            response_metadata={"model_provider": "bedrock_converse"},
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
            response_metadata={"model_provider": "bedrock_converse"},
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
            response_metadata={"model_provider": "bedrock_converse"},
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
            response_metadata={"model_provider": "bedrock_converse"},
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
            response_metadata={"model_provider": "bedrock_converse"},
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
    for chunk, expected in zip(chunks, expected_contents, strict=False):
        assert chunk.content_blocks == [expected]

    full: AIMessageChunk | None = None
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


def test_convert_to_v1_from_converse_input() -> None:
    message = HumanMessage(
        [
            {"text": "foo"},
            {
                "document": {
                    "format": "txt",
                    "name": "doc_name_1",
                    "source": {"text": "doc_text_1"},
                    "context": "doc_context_1",
                    "citations": {"enabled": True},
                },
            },
            {
                "document": {
                    "format": "pdf",
                    "name": "doc_name_2",
                    "source": {"bytes": b"doc_text_2"},
                },
            },
            {
                "document": {
                    "format": "txt",
                    "name": "doc_name_3",
                    "source": {"content": [{"text": "doc_text"}, {"text": "_3"}]},
                    "context": "doc_context_3",
                },
            },
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": b"image_bytes"},
                }
            },
            {
                "document": {
                    "format": "pdf",
                    "name": "doc_name_4",
                    "source": {
                        "s3Location": {"uri": "s3://bla", "bucketOwner": "owner"}
                    },
                },
            },
        ]
    )

    expected: list[types.ContentBlock] = [
        {"type": "text", "text": "foo"},
        {
            "type": "text-plain",
            "mime_type": "text/plain",
            "text": "doc_text_1",
            "extras": {
                "name": "doc_name_1",
                "context": "doc_context_1",
                "citations": {"enabled": True},
            },
        },
        {
            "type": "file",
            "mime_type": "application/pdf",
            "base64": "ZG9jX3RleHRfMg==",
            "extras": {"name": "doc_name_2"},
        },
        {
            "type": "non_standard",
            "value": {
                "document": {
                    "format": "txt",
                    "name": "doc_name_3",
                    "source": {"content": [{"text": "doc_text"}, {"text": "_3"}]},
                    "context": "doc_context_3",
                },
            },
        },
        {
            "type": "image",
            "base64": "aW1hZ2VfYnl0ZXM=",
            "mime_type": "image/jpeg",
        },
        {
            "type": "non_standard",
            "value": {
                "document": {
                    "format": "pdf",
                    "name": "doc_name_4",
                    "source": {
                        "s3Location": {"uri": "s3://bla", "bucketOwner": "owner"}
                    },
                },
            },
        },
    ]

    assert message.content_blocks == expected
