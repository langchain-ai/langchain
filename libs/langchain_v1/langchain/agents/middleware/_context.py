"""Shared context size types for middleware that manages message history."""

from typing import Literal

ContextFraction = tuple[Literal["fraction"], float]
"""Tuple specifying context size as a fraction of the model's context window."""

ContextTokens = tuple[Literal["tokens"], int]
"""Tuple specifying context size as a number of tokens."""

ContextMessages = tuple[Literal["messages"], int]
"""Tuple specifying context size as a number of messages."""

ContextSize = ContextFraction | ContextTokens | ContextMessages
"""Context size tuple to specify how much history to preserve or trigger conditions."""

ContextCondition = ContextSize | list[ContextSize | list[ContextSize]]
"""Recursive type to support nested AND/OR conditions.

Top-level list = OR logic, nested list = AND logic.
"""


__all__ = [
    "ContextCondition",
    "ContextFraction",
    "ContextMessages",
    "ContextSize",
    "ContextTokens",
]
