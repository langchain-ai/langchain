"""Shared types and utilities for context size management.

This module centralizes type definitions and helper functions used across
context editing and summarization middleware implementations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

from langchain_core.messages import BaseMessage

# Type aliases for context size specifications
ContextFraction: TypeAlias = tuple[Literal["fraction"], float]
"""Fractional threshold (0.0 to 1.0) of max model input tokens.

Example: `("fraction", 0.8)` means trigger at 80% of max input tokens.
"""

ContextTokens: TypeAlias = tuple[Literal["tokens"], int]
"""Absolute token count threshold.

Example: `("tokens", 100_000)` means trigger at exactly 100,000 tokens.
"""

ContextMessages: TypeAlias = tuple[Literal["messages"], int]
"""Message count threshold.

Example: `("messages", 50)` means trigger at exactly 50 messages.
"""

ContextSize: TypeAlias = ContextFraction | ContextTokens | ContextMessages
"""Union type for context size specifications.

Provides type-safe representation of context size thresholds using one of:
- `("fraction", float)`: Fractional threshold (0.0 to 1.0) of max model input tokens
- `("tokens", int)`: Absolute token count threshold
- `("messages", int)`: Message count threshold
"""

# Token counter callable type
TokenCounter: TypeAlias = Callable[
    [Sequence[BaseMessage]],
    int,
]
"""Callable that counts tokens in a sequence of messages."""

# Constants
DEFAULT_TOOL_PLACEHOLDER = "[cleared]"
"""Default placeholder text for cleared tool outputs."""

DEFAULT_MESSAGES_TO_KEEP = 20
"""Default number of messages to keep after summarization."""

DEFAULT_TRIM_TOKEN_LIMIT = 4000
"""Default token limit when trimming messages for summarization."""

DEFAULT_FALLBACK_MESSAGE_COUNT = 15
"""Default fallback message count when trimming fails."""

SEARCH_RANGE_FOR_TOOL_PAIRS = 5
"""Range to search for AI/Tool message pairs when determining safe cutoff points."""


def coerce_to_context_size(
    value: int | ContextSize, *, kind: Literal["trigger", "keep"], param_name: str
) -> ContextSize:
    """Coerce integer values to ContextSize tuples for backwards compatibility.

    Args:
        value: Integer or ContextSize tuple.
        kind: Whether this is for a trigger or keep parameter.
        param_name: Name of the parameter for deprecation warnings.

    Returns:
        ContextSize tuple.
    """
    if isinstance(value, int):
        # trigger uses tokens, keep uses messages (backwards compat with old API)
        if kind == "trigger":
            warnings.warn(
                f"{param_name}={value} (int) is deprecated. "
                f"Use {param_name}=('tokens', {value}) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return ("tokens", value)
        warnings.warn(
            f"{param_name}={value} (int) is deprecated. "
            f"Use {param_name}=('messages', {value}) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return ("messages", value)
    return value


def validate_context_size(
    context: ContextSize,
    parameter_name: str,
    *,
    allow_zero_for_keep: bool = False,
) -> ContextSize:
    """Validate context configuration tuples.

    Args:
        context: The ContextSize tuple to validate.
        parameter_name: Name of the parameter being validated (for error messages).
        allow_zero_for_keep: Whether to allow zero values for "keep" parameters.

    Returns:
        The validated ContextSize tuple.

    Raises:
        ValueError: If the context configuration is invalid.
    """
    kind, value = context
    if kind == "fraction":
        if not 0 < value <= 1:
            msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
            raise ValueError(msg)
    elif kind in {"tokens", "messages"}:
        # For keep parameters, 0 is valid (means keep nothing) when allow_zero_for_keep is True
        # For trigger parameters, must be >= 1
        # For summarization keep (allow_zero_for_keep=False), must be > 0
        if parameter_name == "keep" and allow_zero_for_keep:
            # Context editing allows 0 for keep (means keep nothing)
            min_value = 0
            if value < min_value:
                msg = f"{parameter_name} thresholds must be >= {min_value}, got {value}."
                raise ValueError(msg)
        elif parameter_name == "trigger":
            # Trigger must be >= 1 for both context editing and summarization
            min_value = 1
            if value < min_value:
                msg = f"{parameter_name} thresholds must be >= {min_value}, got {value}."
                raise ValueError(msg)
        # Summarization keep must be > 0
        elif value <= 0:
            msg = f"{parameter_name} thresholds must be >= 1, got {value}."
            raise ValueError(msg)
    else:
        msg = f"Unsupported context size type {kind} for {parameter_name}."
        raise ValueError(msg)
    return context


__all__ = [
    "DEFAULT_FALLBACK_MESSAGE_COUNT",
    "DEFAULT_MESSAGES_TO_KEEP",
    "DEFAULT_TOOL_PLACEHOLDER",
    "DEFAULT_TRIM_TOKEN_LIMIT",
    "SEARCH_RANGE_FOR_TOOL_PAIRS",
    "ContextFraction",
    "ContextMessages",
    "ContextSize",
    "ContextTokens",
    "TokenCounter",
    "coerce_to_context_size",
    "validate_context_size",
]
