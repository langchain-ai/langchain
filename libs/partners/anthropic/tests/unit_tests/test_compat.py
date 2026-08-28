"""Tests for Anthropic content compatibility helpers."""

from langchain_core.messages import content as types

from langchain_anthropic._compat import _convert_from_v1_to_anthropic


def test_convert_from_v1_filters_invalid_tool_calls() -> None:
    """Test that invalid tool calls are not sent to Anthropic."""
    content: list[types.ContentBlock] = [
        {"type": "text", "text": "Let me check."},
        {
            "type": "invalid_tool_call",
            "id": "toolu_invalid",
            "name": "get_weather",
            "args": '{"city":',
            "error": "Invalid JSON",
        },
        {
            "type": "tool_call",
            "id": "toolu_valid",
            "name": "get_weather",
            "args": {"city": "San Francisco"},
        },
    ]

    result = _convert_from_v1_to_anthropic(content, [], "anthropic")

    assert result == [
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_valid",
            "name": "get_weather",
            "input": {"city": "San Francisco"},
        },
    ]
