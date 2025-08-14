from langchain_core.messages import AIMessage
from langchain_core.messages import content_blocks as types
from langchain_core.messages.block_translators.openai import translate_content


def test_convert_to_v1_from_responses() -> None:
    message = AIMessage(
        [
            {"type": "reasoning", "id": "abc123", "summary": []},
            {
                "type": "reasoning",
                "id": "abc234",
                "summary": [
                    {"type": "summary_text", "text": "foo "},
                    {"type": "summary_text", "text": "bar"},
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
    )
    expected_content: list[types.ContentBlock] = [
        {"type": "reasoning", "id": "abc123"},
        {"type": "reasoning", "id": "abc234", "reasoning": "foo "},
        {"type": "reasoning", "id": "abc234", "reasoning": "bar"},
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
    result = translate_content(message)
    assert result == expected_content
