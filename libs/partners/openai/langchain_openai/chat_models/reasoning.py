"""Reasoning content extraction and normalization utilities.

This module provides provider-agnostic utilities for extracting and preserving
reasoning/thinking data from OpenAI-compatible API responses. It handles various
field names and formats used by different providers (vLLM, DeepSeek, OpenRouter, etc.)
while maintaining a consistent interface for the rest of LangChain.

The reasoning data is treated as first-class metadata that must be preserved
throughout the message lifecycle, from raw API responses to final AIMessage objects.
"""

from __future__ import annotations

from typing import Any, Mapping

# Standardized field name for reasoning content in LangChain messages
REASONING_CONTENT_KEY = "reasoning_content"

# Provider-specific field names that map to reasoning content
# These are checked in order of preference
REASONING_FIELD_NAMES = (
    "reasoning_content",  # vLLM, DeepSeek, XAI, Groq
    "reasoning",  # OpenRouter, some vLLM configurations
    "thinking",  # Some providers use this
)


def extract_reasoning_from_dict(
    data: Mapping[str, Any], *, default: str | None = None
) -> str | None:
    """Extract reasoning content from a dictionary.

    Checks multiple field names to support various OpenAI-compatible providers.
    Returns the first non-None value found, or the default if none found.

    Args:
        data: Dictionary that may contain reasoning fields.
        default: Default value to return if no reasoning is found.

    Returns:
        Reasoning content as a string, or None/default if not found.

    Examples:
        >>> extract_reasoning_from_dict({"reasoning_content": "Let me think..."})
        'Let me think...'
        >>> extract_reasoning_from_dict({"reasoning": "Step by step..."})
        'Step by step...'
        >>> extract_reasoning_from_dict({"content": "Hello"})
        None
    """
    for field_name in REASONING_FIELD_NAMES:
        if value := data.get(field_name):
            if isinstance(value, str):
                return value
            # Handle non-string values (shouldn't happen, but be defensive)
            if value is not None:
                return str(value)
    return default


def extract_reasoning_from_openai_message(
    message: Any, *, default: str | None = None
) -> str | None:
    """Extract reasoning content from an OpenAI SDK message object.

    Handles both direct attributes and model_extra (used by OpenRouter and others).

    Args:
        message: OpenAI SDK message object (e.g., ChatCompletionMessage).
        default: Default value to return if no reasoning is found.

    Returns:
        Reasoning content as a string, or None/default if not found.

    Examples:
        >>> # Direct attribute
        >>> msg = MagicMock()
        >>> msg.reasoning_content = "Let me think..."
        >>> extract_reasoning_from_openai_message(msg)
        'Let me think...'
        >>> # Via model_extra
        >>> msg.model_extra = {"reasoning": "Step by step..."}
        >>> extract_reasoning_from_openai_message(msg)
        'Step by step...'
    """
    # Check direct attributes first
    for field_name in REASONING_FIELD_NAMES:
        if hasattr(message, field_name):
            value = getattr(message, field_name, None)
            if value is not None:
                if isinstance(value, str):
                    return value
                # Handle non-string values
                if value is not None:
                    return str(value)

    # Check model_extra (used by OpenRouter and some providers)
    if hasattr(message, "model_extra") and isinstance(message.model_extra, dict):
        for field_name in REASONING_FIELD_NAMES:
            if value := message.model_extra.get(field_name):
                if isinstance(value, str):
                    return value
                if value is not None:
                    return str(value)

    return default


def normalize_reasoning_for_additional_kwargs(
    additional_kwargs: dict[str, Any], data: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize reasoning content into additional_kwargs.

    Extracts reasoning from the data dictionary and adds it to additional_kwargs
    under the standardized REASONING_CONTENT_KEY. Does not overwrite existing
    reasoning_content if already present.

    Args:
        additional_kwargs: Dictionary to update with reasoning content.
        data: Source dictionary that may contain reasoning fields.

    Returns:
        Updated additional_kwargs dictionary (mutated in place, also returned).

    Examples:
        >>> kwargs = {}
        >>> normalize_reasoning_for_additional_kwargs(kwargs, {"reasoning_content": "..."})
        {'reasoning_content': '...'}
        >>> # Does not overwrite existing
        >>> kwargs = {"reasoning_content": "existing"}
        >>> normalize_reasoning_for_additional_kwargs(kwargs, {"reasoning": "new"})
        {'reasoning_content': 'existing'}
    """
    if REASONING_CONTENT_KEY not in additional_kwargs:
        if reasoning := extract_reasoning_from_dict(data):
            additional_kwargs[REASONING_CONTENT_KEY] = reasoning
    return additional_kwargs


def should_preserve_reasoning_in_streaming() -> bool:
    """Check if reasoning content should be preserved during streaming.

    This function exists to allow future configuration of reasoning handling
    behavior. Currently always returns True.

    Returns:
        True if reasoning should be preserved, False otherwise.
    """
    return True
