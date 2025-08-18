from typing import Optional

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content_blocks as types


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
            "type": "non_standard",
            "value": {"type": "something_else", "foo": "bar"},
        },
    ]
    assert message.content_blocks == expected_content

    # Check no mutation
    assert message.content != expected_content


def test_convert_to_v1_from_anthropic_chunk() -> None:
    chunks = [
        AIMessageChunk(content=[{"text": "Looking ", "type": "text", "index": 0}]),
        AIMessageChunk(content=[{"text": "now.", "type": "text", "index": 0}]),
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
        full = chunk if full is None else full + chunk  # type: ignore[assignment]
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
