# SPDX-License-Identifier: MIT
"""Unit tests for langchain_openai.chat_models.reasoning_parser."""

import pytest

from langchain_openai.chat_models.reasoning_parser import (
    extract_reasoning_content,
    extract_reasoning_delta,
)


@pytest.mark.parametrize(
    ("model_name", "response_dict", "expected"),
    [
        # Standard Qwen with reasoning_content
        (
            "qwen3-chat",
            {
                "choices": [
                    {"message": {"content": "hi", "reasoning_content": "I am thinking"}}
                ]
            },
            "I am thinking",
        ),
        # Qwen with alternative field names
        (
            "qwen2.5-instruct",
            {"choices": [{"message": {"content": "hi", "think": "Another thought"}}]},
            "Another thought",
        ),
        (
            "qwen-1.8",
            {
                "choices": [
                    {"message": {"content": "hi", "thought": "Internal reasoning"}}
                ]
            },
            "Internal reasoning",
        ),
        # Non-Qwen model → should not extract anything
        (
            "gpt-4-turbo",
            {
                "choices": [
                    {"message": {"content": "hi", "reasoning_content": "ignore me"}}
                ]
            },
            None,
        ),
        # Invalid structure: no choices
        ("qwen3-chat", {"message": {"content": "hi"}}, None),
        # Invalid structure: message is not dict
        ("qwen3-chat", {"choices": [{"message": "not a dict"}]}, None),
        # Empty / malformed response
        ("qwen3-chat", {}, None),
    ],
)
def test_extract_reasoning_content(
    model_name: str, response_dict: dict, expected: str | None
) -> None:
    """Ensure reasoning extraction works correctly for various inputs."""
    result = extract_reasoning_content(model_name, response_dict)
    assert result == expected


@pytest.mark.parametrize(
    ("model_name", "delta_dict", "expected"),
    [
        # Qwen stream delta with reasoning_content
        (
            "qwen3-chat",
            {"reasoning_content": "Streaming reasoning"},
            "Streaming reasoning",
        ),
        # Alternative field key
        ("qwen3-chat", {"think": "Stream thinking..."}, "Stream thinking..."),
        # Unsupported model → None
        ("gpt-4o", {"reasoning_content": "should ignore"}, None),
        # Malformed inputs
        ("qwen3-chat", {}, None),
        ("qwen3-chat", None, None),
    ],
)
def test_extract_reasoning_delta(
    model_name: str, delta_dict: dict | None, expected: str | None
) -> None:
    """Ensure streaming delta reasoning extraction functions robustly."""
    result = extract_reasoning_delta(model_name, delta_dict or {})
    assert result == expected
