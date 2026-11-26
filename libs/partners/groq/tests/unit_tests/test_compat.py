"""Tests for Groq content compatibility helpers."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import content as types

from langchain_groq._compat import _convert_from_v1_to_groq
from langchain_groq.chat_models import _convert_message_to_dict


@pytest.mark.parametrize("model_provider", ["groq", "other", None])
def test_convert_from_v1_filters_invalid_tool_calls(
    model_provider: str | None,
) -> None:
    """Test that invalid tool calls are filtered out in v1 conversion."""
    content: list[types.ContentBlock] = [
        {"type": "text", "text": "hello"},
        {
            "type": "invalid_tool_call",
            "name": "get_weather",
            "args": "{bad json",
            "id": "call_1",
            "error": "Malformed args.",
        },
    ]

    new_content, additional_kwargs = _convert_from_v1_to_groq(content, model_provider)

    assert new_content == "hello"
    assert additional_kwargs == {}


@pytest.mark.parametrize("model_provider", ["groq", "other", None])
def test_convert_from_v1_filters_invalid_tool_calls_multiple_blocks(
    model_provider: str | None,
) -> None:
    """Test that multiple text blocks are preserved without invalid tool calls."""
    content: list[types.ContentBlock] = [
        {"type": "text", "text": "hello"},
        {
            "type": "invalid_tool_call",
            "name": "get_weather",
            "args": "{bad json",
            "id": "call_1",
            "error": "Malformed args.",
        },
        {"type": "text", "text": "world"},
    ]

    new_content, additional_kwargs = _convert_from_v1_to_groq(content, model_provider)

    assert new_content == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]
    assert additional_kwargs == {}


@pytest.mark.parametrize("model_provider", ["groq", "other", None])
def test_convert_from_v1_only_invalid_tool_calls(
    model_provider: str | None,
) -> None:
    """Test that content with only invalid tool calls results in empty list."""
    content: list[types.ContentBlock] = [
        {
            "type": "invalid_tool_call",
            "name": "get_weather",
            "args": "{bad json",
            "id": "call_1",
            "error": "Malformed args.",
        }
    ]

    new_content, additional_kwargs = _convert_from_v1_to_groq(content, model_provider)

    assert new_content == []
    assert additional_kwargs == {}


def test_convert_message_to_dict_with_v1_invalid_tool_calls() -> None:
    """Test that AIMessage with v1 invalid tool calls converts cleanly."""
    message = AIMessage(
        content=[
            {"type": "text", "text": "Let me check."},
            {
                "type": "invalid_tool_call",
                "name": "get_weather",
                "args": "{bad json",
                "id": "call_1",
                "error": "Malformed args.",
            },
        ],
        invalid_tool_calls=[
            {
                "name": "get_weather",
                "args": "{bad json",
                "id": "call_1",
                "error": "Malformed args.",
                "type": "invalid_tool_call",
            }
        ],
        response_metadata={"output_version": "v1", "model_provider": "groq"},
    )

    converted = _convert_message_to_dict(message)

    assert converted == {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{bad json",
                },
            }
        ],
    }
