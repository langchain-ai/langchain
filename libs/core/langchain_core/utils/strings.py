"""String utilities."""

from collections.abc import Iterable
from typing import Any


def stringify_value(val: Any) -> str:
    """Stringify a value.

    Args:
        val: The value to stringify.

    Returns:
        The stringified value.
    """
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return "\n" + stringify_dict(val)
    if isinstance(val, list):
        return "\n".join(stringify_value(v) for v in val)
    return str(val)


def stringify_dict(data: dict) -> str:
    """Stringify a dictionary.

    Args:
        data: The dictionary to stringify.

    Returns:
        The stringified dictionary.
    """
    return "".join(f"{key}: {stringify_value(value)}\n" for key, value in data.items())


def comma_list(items: Iterable[Any]) -> str:
    """Convert an iterable to a comma-separated string.

    Args:
        items: The iterable to convert.

    Returns:
        The comma-separated string.
    """
    return ", ".join(str(item) for item in items)


def sanitize_for_postgres(text: str, replacement: str = "") -> str:
    r"""Sanitize text by removing NUL bytes that are incompatible with PostgreSQL.

    PostgreSQL text fields cannot contain `NUL (0x00)` bytes, which can cause
    `psycopg.DataError` when inserting documents. This function removes or replaces
    such characters to ensure compatibility.

    Args:
        text: The text to sanitize.
        replacement: String to replace `NUL` bytes with.

    Returns:
        The sanitized text with `NUL` bytes removed or replaced.

    Example:
        >>> sanitize_for_postgres("Hello\\x00world")
        'Helloworld'
        >>> sanitize_for_postgres("Hello\\x00world", " ")
        'Hello world'
    """
    return text.replace("\x00", replacement)
