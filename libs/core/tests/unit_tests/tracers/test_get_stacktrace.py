"""Test _get_stacktrace method in tracers core."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from langchain_core.tracers.core import _TracerCore

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run


class MockTracerCore(_TracerCore):
    """Mock tracer core for testing _get_stacktrace.

    Provides a minimal implementation of the abstract _TracerCore class
    for unit testing the _get_stacktrace static method.
    """

    def __init__(self) -> None:
        """Initialize the mock tracer."""
        super().__init__()

    def _persist_run(self, run: Run) -> None:
        """No-op implementation required by abstract base class."""


def _raise_value_error(message: str) -> None:
    """Raise a ValueError with the given message.

    Args:
        message: The error message.

    Raises:
        ValueError: Always raised with the provided message.
    """
    raise ValueError(message)


def _raise_chained_exception() -> None:
    """Raise a RuntimeError chained from a ValueError.

    Raises:
        RuntimeError: Chained from an inner ValueError.
    """
    try:
        _raise_value_error("Inner error")
    except ValueError as inner:
        msg = "Outer error"
        raise RuntimeError(msg) from inner


def test_get_stacktrace_returns_formatted_traceback() -> None:
    """Test that _get_stacktrace returns error repr and formatted traceback."""
    try:
        _raise_value_error("Test error message")
    except ValueError as e:
        result = _TracerCore._get_stacktrace(e)

    # Should contain the repr of the error
    assert "ValueError('Test error message')" in result
    # Should contain traceback information
    assert "Traceback" in result
    assert "_raise_value_error" in result


def test_get_stacktrace_handles_exception_without_traceback() -> None:
    """Test _get_stacktrace with an exception created without raising."""
    error = ValueError("Error without traceback")
    result = _TracerCore._get_stacktrace(error)

    # Should still contain the repr
    assert "ValueError('Error without traceback')" in result


def test_get_stacktrace_falls_back_to_repr_on_format_failure() -> None:
    """Test that _get_stacktrace returns repr(error) when formatting fails."""
    error = ValueError("Test error")
    expected_repr = repr(error)

    with patch(
        "langchain_core.tracers.core.traceback.format_exception",
        side_effect=Exception("format failed"),
    ):
        result = _TracerCore._get_stacktrace(error)

    # Should return exactly the repr when format_exception fails
    assert result == expected_repr


def test_get_stacktrace_with_chained_exception() -> None:
    """Test that _get_stacktrace captures exception chain information."""
    try:
        _raise_chained_exception()
    except RuntimeError as e:
        result = _TracerCore._get_stacktrace(e)

    # Should contain the outer exception
    assert "RuntimeError('Outer error')" in result
    # Should contain traceback
    assert "Traceback" in result
    # Should capture the chained inner exception
    assert "ValueError" in result
    assert "Inner error" in result


def test_get_stacktrace_with_custom_exception_class() -> None:
    """Test that _get_stacktrace handles custom exception classes."""

    class CustomApplicationError(Exception):
        """Custom exception for testing."""

    def _raise_custom() -> None:
        msg = "Custom error occurred"
        raise CustomApplicationError(msg)

    try:
        _raise_custom()
    except CustomApplicationError as e:
        result = _TracerCore._get_stacktrace(e)

    assert "CustomApplicationError" in result
    assert "Custom error occurred" in result
    assert "Traceback" in result
