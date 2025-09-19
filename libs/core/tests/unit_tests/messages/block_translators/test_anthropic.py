from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.messages import content as types


def test_convert_to_v1_from_anthropic() -> None:
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
            {
                "type": "server_tool_use",
                "name": "web_search",
                "input": {"query": "web search query"},
                "id": "srvtoolu_abc123",
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_abc123",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "Page Title 1",
                        "url": "<page url 1>",
                        "page_age": "January 1, 2025",
                        "encrypted_content": "<encrypted content 1>",
                    },
                    {
                        "type": "web_search_result",
                        "title": "Page Title 2",
                        "url": "<page url 2>",
                        "page_age": "January 2, 2025",
                        "encrypted_content": "<encrypted content 2>",
                    },
                ],
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_def456",
                "name": "code_execution",
                "input": {"code": "import numpy as np..."},
            },
            {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_def456",
                "content": {
                    "type": "code_execution_result",
                    "stdout": "Mean: 5.5\nStandard deviation...",
                    "stderr": "",
                    "return_code": 0,
                },
            },
            {"type": "something_else", "foo": "bar"},
        ],
        response_metadata={"model_provider": "anthropic"},
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
            "type": "server_tool_call",
            "name": "web_search",
            "id": "srvtoolu_abc123",
            "args": {"query": "web search query"},
        },
        {
            "type": "server_tool_result",
            "tool_call_id": "srvtoolu_abc123",
            "output": [
                {
                    "type": "web_search_result",
                    "title": "Page Title 1",
                    "url": "<page url 1>",
                    "page_age": "January 1, 2025",
                    "encrypted_content": "<encrypted content 1>",
                },
                {
                    "type": "web_search_result",
                    "title": "Page Title 2",
                    "url": "<page url 2>",
                    "page_age": "January 2, 2025",
                    "encrypted_content": "<encrypted content 2>",
                },
            ],
            "status": "success",
            "extras": {"block_type": "web_search_tool_result"},
        },
        {
            "type": "server_tool_call",
            "name": "code_interpreter",
            "id": "srvtoolu_def456",
            "args": {"code": "import numpy as np..."},
        },
        {
            "type": "server_tool_result",
            "tool_call_id": "srvtoolu_def456",
            "output": {
                "type": "code_execution_result",
                "return_code": 0,
                "stdout": "Mean: 5.5\nStandard deviation...",
                "stderr": "",
            },
            "status": "success",
            "extras": {"block_type": "code_execution_tool_result"},
        },
        {
            "type": "non_standard",
            "value": {"type": "something_else", "foo": "bar"},
        },
    ]
    assert message.content_blocks == expected_content

    # Check no mutation
    assert message.content != expected_content

    message = AIMessage("Hello", response_metadata={"model_provider": "anthropic"})
    expected_content = [{"type": "text", "text": "Hello"}]
    assert message.content_blocks == expected_content
    assert message.content != expected_content  # check no mutation


def test_convert_to_v1_from_anthropic_chunk() -> None:
    chunks = [
        AIMessageChunk(
            content=[{"text": "Looking ", "type": "text", "index": 0}],
            response_metadata={"model_provider": "anthropic"},
        ),
        AIMessageChunk(
            content=[{"text": "now.", "type": "text", "index": 0}],
            response_metadata={"model_provider": "anthropic"},
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
            response_metadata={"model_provider": "anthropic"},
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
            response_metadata={"model_provider": "anthropic"},
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
            response_metadata={"model_provider": "anthropic"},
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
            response_metadata={"model_provider": "anthropic"},
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
            response_metadata={"model_provider": "anthropic"},
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


def test_convert_to_v1_from_anthropic_input() -> None:
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
