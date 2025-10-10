"""String utilities."""

import re
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
    text = ""
    for key, value in data.items():
        text += key + ": " + stringify_value(value) + "\n"
    return text


def comma_list(items: list[Any]) -> str:
    """Convert a list to a comma-separated string.

    Args:
        items: The list to convert.

    Returns:
        The comma-separated string.
    """
    return ", ".join(str(item) for item in items)


def truncate_with_ellipsis(text: str, max_length: int, *, ellipsis: str = "...") -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: The text to truncate.
        max_length: Maximum length of the resulting string.
        ellipsis: String to append when truncating. Defaults to "...".

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(ellipsis)] + ellipsis


def normalize_whitespace(text: str, *, preserve_newlines: bool = True) -> str:
    """Normalize whitespace in text.

    Args:
        text: The text to normalize.
        preserve_newlines: Whether to preserve newline characters. Defaults to True.

    Returns:
        Text with normalized whitespace.
    """
    if preserve_newlines:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +", " ", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def sanitize_for_postgres(text: str, replacement: str = "") -> str:
    r"""Sanitize text by removing NUL bytes that are incompatible with PostgreSQL.

    PostgreSQL text fields cannot contain NUL (0x00) bytes, which can cause
    psycopg.DataError when inserting documents. This function removes or replaces
    such characters to ensure compatibility.

    Args:
        text: The text to sanitize.
        replacement: String to replace NUL bytes with. Defaults to empty string.

    Returns:
        The sanitized text with NUL bytes removed or replaced.

    Example:
        >>> sanitize_for_postgres("Hello\\x00world")
        'Helloworld'
        >>> sanitize_for_postgres("Hello\\x00world", " ")
        'Hello world'
    """
    return text.replace("\x00", replacement)
