"""Tests for Anthropic content compatibility helpers."""

from langchain_core.messages import content as types

from langchain_anthropic._compat import _convert_from_v1_to_anthropic


def test_convert_from_v1_restores_anthropic_invalid_tool_calls() -> None:
    """Test that Anthropic invalid tool calls retain their tool-use identity."""
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
            "id": "toolu_invalid",
            "name": "get_weather",
            "input": {},
        },
        {
            "type": "tool_use",
            "id": "toolu_valid",
            "name": "get_weather",
            "input": {"city": "San Francisco"},
        },
    ]


def test_convert_from_v1_preserves_toolset_name() -> None:
    """Preserve provider metadata on complete and streamed tool calls."""
    content: list[types.ContentBlock] = [
        {
            "type": "tool_call",
            "id": "toolu_1",
            "name": "screenshot",
            "args": {},
            "extras": {"toolset_name": "computer"},
        },
        {
            "type": "tool_call_chunk",
            "id": "toolu_2",
            "name": "left_click",
            "args": '{"coordinate": [10, 20]}',
            "extras": {"toolset_name": "computer"},
        },
    ]

    result = _convert_from_v1_to_anthropic(content, [], "anthropic")

    assert result == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "screenshot",
            "input": {},
            "toolset_name": "computer",
        },
        {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "left_click",
            "input": {"coordinate": [10, 20]},
            "toolset_name": "computer",
        },
    ]


def test_convert_from_v1_filters_non_anthropic_invalid_tool_calls() -> None:
    """Test that invalid calls from other providers are not promoted."""
    content: list[types.ContentBlock] = [
        {
            "type": "invalid_tool_call",
            "id": "call_invalid",
            "name": "get_weather",
            "args": '{"city":',
            "error": "Invalid JSON",
        }
    ]

    assert _convert_from_v1_to_anthropic(content, [], "openai") == []


def test_convert_from_v1_filters_unidentified_invalid_tool_calls() -> None:
    """Test that incomplete blocks are not promoted into tool calls."""
    content: list[types.ContentBlock] = [
        {
            "type": "invalid_tool_call",
            "id": None,
            "name": "get_weather",
            "args": '{"city":',
            "error": "Invalid JSON",
        }
    ]

    assert _convert_from_v1_to_anthropic(content, [], "anthropic") == []
