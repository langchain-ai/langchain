"""Serializable components for structured output functionality."""

from __future__ import annotations

from typing import Any


class SerializableParsingErrorHandler:
    """Serializable replacement for `lambda _: None` in parsing error scenarios.

    This class provides the same functionality as the lambda function but can be
    pickled for thread persistence in LangGraph Studio.
    """

    def __call__(self, _: Any) -> None:
        """Return None for any input, mimicking lambda _: None behavior."""
        return

    def __repr__(self) -> str:
        """Provide a clear representation for debugging."""
        return "SerializableParsingErrorHandler()"

    def __eq__(self, other: object) -> bool:
        """Support equality comparison for testing."""
        return isinstance(other, SerializableParsingErrorHandler)

    def __hash__(self) -> int:
        """Support hashing for set/dict operations."""
        return hash(self.__class__.__name__)

    def __getstate__(self) -> dict[str, Any]:
        """Control pickle serialization for production robustness."""
        # This class has no state to preserve, return empty dict
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Control pickle deserialization for production robustness."""
        # This class has no state to restore, nothing to do


class SerializableNoneAssigner:
    """Serializable replacement for `lambda _: None` in RunnablePassthrough.assign().

    Used in the fallback chain when parsing fails.
    """

    def __call__(self, _: Any) -> None:
        """Return None for any input, used as fallback parsed value."""
        return

    def __repr__(self) -> str:
        """Provide a clear representation for debugging."""
        return "SerializableNoneAssigner()"

    def __eq__(self, other: object) -> bool:
        """Support equality comparison for testing."""
        return isinstance(other, SerializableNoneAssigner)

    def __hash__(self) -> int:
        """Support hashing for set/dict operations."""
        return hash(self.__class__.__name__)

    def __getstate__(self) -> dict[str, Any]:
        """Control pickle serialization for production robustness."""
        # This class has no state to preserve, return empty dict
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Control pickle deserialization for production robustness."""
        # This class has no state to restore, nothing to do


def get_serializable_error_handler() -> SerializableParsingErrorHandler:
    """Get a serializable error handler instance.

    This is a convenience function that can be used directly in place of
    lambda _: None in RunnablePassthrough.assign() calls.

    Returns:
        SerializableParsingErrorHandler instance.
    """
    return SerializableParsingErrorHandler()


def get_serializable_none_assigner() -> SerializableNoneAssigner:
    """Get a serializable None assigner instance.

    This is a convenience function that can be used directly in place of
    lambda _: None in parsed value assignments.

    Returns:
        SerializableNoneAssigner instance.
    """
    return SerializableNoneAssigner()
