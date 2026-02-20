"""Unit tests for ChatFireworks."""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_fireworks.chat_models import (
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)


def test_convert_dict_to_message_with_reasoning_content() -> None:
    """Test that reasoning_content is correctly extracted from API response."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Let me think about this step by step...",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" in message.additional_kwargs
    expected_reasoning = "Let me think about this step by step..."
    assert message.additional_kwargs["reasoning_content"] == expected_reasoning


def test_convert_dict_to_message_without_reasoning_content() -> None:
    """Test that messages without reasoning_content work correctly."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" not in message.additional_kwargs


# -- Streaming inbound: _convert_chunk_to_message_chunk --


def test_chunk_captures_reasoning_content() -> None:
    """Test that reasoning_content is captured from streaming deltas."""
    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Let me reason about this...",
                }
            }
        ],
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.additional_kwargs["reasoning_content"] == (
        "Let me reason about this..."
    )


def test_chunk_without_reasoning_content() -> None:
    """Test that chunks without reasoning_content still work."""
    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "Hello!",
                }
            }
        ],
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello!"
    assert "reasoning_content" not in result.additional_kwargs


def test_chunk_reasoning_content_with_tool_calls() -> None:
    """Test that reasoning_content coexists with tool calls in streaming."""
    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I need to subtract these numbers.",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_123",
                            "function": {
                                "name": "subtract",
                                "arguments": '{"a": 5, "b": 3}',
                            },
                        }
                    ],
                }
            }
        ],
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.additional_kwargs["reasoning_content"] == (
        "I need to subtract these numbers."
    )
    assert "tool_calls" in result.additional_kwargs
    assert len(result.tool_call_chunks) == 1
    assert result.tool_call_chunks[0]["name"] == "subtract"


# -- Outbound: _convert_message_to_dict --


def test_message_to_dict_forwards_reasoning_content() -> None:
    """Test that reasoning_content is forwarded when converting to dict."""
    message = AIMessage(
        content="The answer is 42.",
        additional_kwargs={"reasoning_content": "Step by step reasoning..."},
    )

    result = _convert_message_to_dict(message)

    assert result["role"] == "assistant"
    assert result["content"] == "The answer is 42."
    assert result["reasoning_content"] == "Step by step reasoning..."


def test_message_to_dict_without_reasoning_content() -> None:
    """Test that messages without reasoning_content don't add spurious keys."""
    message = AIMessage(content="Hello!")

    result = _convert_message_to_dict(message)

    assert result["role"] == "assistant"
    assert result["content"] == "Hello!"
    assert "reasoning_content" not in result


def test_message_to_dict_reasoning_content_with_tool_calls() -> None:
    """Test that reasoning_content and tool_calls coexist in outbound dict."""
    message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "I should call the tool."},
        tool_calls=[{"name": "subtract", "id": "call_123", "args": {"a": 5, "b": 3}}],
    )

    result = _convert_message_to_dict(message)

    assert result["role"] == "assistant"
    assert result["reasoning_content"] == "I should call the tool."
    assert "tool_calls" in result
    assert result["content"] is None  # content is None when tool_calls present


# -- Round-trip: API → message → API dict --


def test_reasoning_content_round_trip() -> None:
    """Test that reasoning_content survives a full round-trip."""
    # Simulate non-streaming API response
    api_response = {
        "role": "assistant",
        "content": "The result is 135700070955.",
        "reasoning_content": "I need to add 123456314532 and 12243756423.",
    }

    # Inbound: API dict → LangChain message
    message = _convert_dict_to_message(api_response)
    assert message.additional_kwargs["reasoning_content"] == (
        "I need to add 123456314532 and 12243756423."
    )

    # Outbound: LangChain message → API dict (for next turn)
    outbound = _convert_message_to_dict(message)
    assert outbound["reasoning_content"] == (
        "I need to add 123456314532 and 12243756423."
    )
    assert outbound["content"] == "The result is 135700070955."


def test_streaming_reasoning_content_round_trip() -> None:
    """Test that reasoning_content from streaming chunks round-trips."""
    # Simulate streaming chunks
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Let me think",
                    }
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "",
                        "reasoning_content": " about this step by step.",
                    }
                }
            ],
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "The answer is 42.",
                    }
                }
            ],
        },
    ]

    # Collect streaming chunks
    all_chunks = []
    for raw_chunk in chunks:
        msg_chunk = _convert_chunk_to_message_chunk(raw_chunk, AIMessageChunk)
        all_chunks.append(msg_chunk)

    # Verify reasoning_content captured in first two chunks
    assert all_chunks[0].additional_kwargs["reasoning_content"] == "Let me think"
    assert (
        all_chunks[1].additional_kwargs["reasoning_content"]
        == " about this step by step."
    )
    assert "reasoning_content" not in all_chunks[2].additional_kwargs

    # Aggregate chunks (simulating what LangChain does)
    full = all_chunks[0]
    for c in all_chunks[1:]:
        full = full + c

    # The aggregated message should have the concatenated reasoning_content
    assert isinstance(full, AIMessageChunk)
    assert full.content == "The answer is 42."

    # Outbound: convert the aggregated message to dict for the next API call
    outbound = _convert_message_to_dict(full)
    assert outbound["role"] == "assistant"
    assert "reasoning_content" in outbound


def test_backward_compat_no_reasoning_content() -> None:
    """Test that existing tool-calling flows without reasoning still work."""
    # Standard tool call without reasoning
    api_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }

    message = _convert_dict_to_message(api_response)
    assert isinstance(message, AIMessage)
    assert "reasoning_content" not in message.additional_kwargs
    assert len(message.tool_calls) == 1

    outbound = _convert_message_to_dict(message)
    assert "reasoning_content" not in outbound
    assert "tool_calls" in outbound
    assert outbound["content"] is None
