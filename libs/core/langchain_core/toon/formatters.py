"""Primitive encoding and formatting utilities."""

from __future__ import annotations

import re
from typing import Literal

from .constants import (
    BACKSLASH,
    COMMA,
    DEFAULT_DELIMITER,
    DOUBLE_QUOTE,
    FALSE_LITERAL,
    LIST_ITEM_MARKER,
    NULL_LITERAL,
    TRUE_LITERAL,
    DelimiterType,
    JsonPrimitive,
)


def encode_primitive(value: JsonPrimitive, delimiter: DelimiterType = COMMA) -> str:
    """Encode a primitive value to TOON format.

    Args:
        value: Primitive value to encode (None, bool, int, float, or str).
        delimiter: Delimiter character used for array values.

    Returns:
        TOON-formatted string representation of the value.
    """
    if value is None:
        return NULL_LITERAL

    if isinstance(value, bool):
        return TRUE_LITERAL if value else FALSE_LITERAL

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        return _encode_number(value)

    return encode_string_literal(str(value), delimiter)


def _encode_number(value: float) -> str:
    """Encode a floating point number.

    Args:
        value: Float value to encode.

    Returns:
        String representation with trailing zeros removed.
    """
    if value.is_integer():
        return str(int(value))

    text = format(value, ".15f")
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def encode_string_literal(value: str, delimiter: DelimiterType = COMMA) -> str:
    """Encode a string literal, quoting if necessary.

    Strings are quoted if they contain structural characters, delimiters,
    whitespace, or could be confused with literals or numbers.

    Args:
        value: String to encode.
        delimiter: Delimiter character to check for.

    Returns:
        Quoted or unquoted string representation.
    """
    if is_safe_unquoted(value, delimiter):
        return value
    return f"{DOUBLE_QUOTE}{escape_string(value)}{DOUBLE_QUOTE}"


def escape_string(value: str) -> str:
    """Escape special characters in a string.

    Args:
        value: String to escape.

    Returns:
        Escaped string with backslash escapes for special characters.
    """
    return (
        value.replace(BACKSLASH, f"{BACKSLASH}{BACKSLASH}")
        .replace(DOUBLE_QUOTE, f"{BACKSLASH}{DOUBLE_QUOTE}")
        .replace("\n", f"{BACKSLASH}n")
        .replace("\r", f"{BACKSLASH}r")
        .replace("\t", f"{BACKSLASH}t")
    )


def is_safe_unquoted(value: str, delimiter: DelimiterType = COMMA) -> bool:
    """Check if a string can be safely represented without quotes.

    Args:
        value: String to check.
        delimiter: Delimiter character to check for.

    Returns:
        `True` if the string can be unquoted, `False` otherwise.
    """
    if not value:
        return False

    if value != value.strip():
        return False

    # Check for any whitespace (space, tab, newline, etc.)
    if any(ch in value for ch in (" ", "\n", "\r", "\t")):
        return False

    if value in {TRUE_LITERAL, FALSE_LITERAL, NULL_LITERAL}:
        return False

    if _is_numeric_like(value):
        return False

    if ":" in value:
        return False

    if '"' in value or "\\" in value:
        return False

    if any(ch in value for ch in "[]{}"):
        return False

    if delimiter in value:
        return False

    return not value.startswith(LIST_ITEM_MARKER)


_NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)
_LEADING_ZERO_PATTERN = re.compile(r"0\d+")


def _is_numeric_like(value: str) -> bool:
    """Check if a string looks like a number.

    Args:
        value: String to check.

    Returns:
        `True` if the string resembles a numeric value.
    """
    return bool(
        _NUMERIC_PATTERN.fullmatch(value) or _LEADING_ZERO_PATTERN.fullmatch(value)
    )


def encode_key(key: str) -> str:
    r"""Encode an object key, quoting if necessary.

    Keys are quoted if they don't match the pattern [A-Za-z_][\\w.]*.

    Args:
        key: Object key to encode.

    Returns:
        Quoted or unquoted key representation.
    """
    if _is_valid_unquoted_key(key):
        return key
    return f"{DOUBLE_QUOTE}{escape_string(key)}{DOUBLE_QUOTE}"


_KEY_PATTERN = re.compile(r"[A-Za-z_][\w.]*")


def _is_valid_unquoted_key(key: str) -> bool:
    """Check if a key can be represented without quotes.

    Args:
        key: Key string to check.

    Returns:
        `True` if the key can be unquoted.
    """
    return bool(_KEY_PATTERN.fullmatch(key))


def join_encoded_values(
    values: list[JsonPrimitive], delimiter: DelimiterType = COMMA
) -> str:
    """Join multiple primitive values with the specified delimiter.

    Args:
        values: List of primitive values to encode and join.
        delimiter: Delimiter character to use between values.

    Returns:
        Joined string of encoded values.
    """
    return delimiter.join(encode_primitive(value, delimiter) for value in values)


def format_header(
    length: int,
    *,
    key: str | None = None,
    fields: list[str] | None = None,
    delimiter: DelimiterType = COMMA,
    length_marker: Literal["#", False] = False,
) -> str:
    """Format an array header with optional key, length marker, and field names.

    Args:
        length: Number of items in the array.
        key: Optional key name for the array.
        fields: Optional list of field names for tabular data.
        delimiter: Delimiter character (only shown if non-default).
        length_marker: If `'#'`, prefix the length with `'#'`.

    Returns:
        Formatted header string like `key[N]:` or `key[N]{field1,field2}:`.
    """
    active_delimiter = delimiter or DEFAULT_DELIMITER
    header = ""

    if key:
        header += encode_key(key)

    marker = length_marker or ""
    delimiter_suffix = "" if active_delimiter == DEFAULT_DELIMITER else active_delimiter
    header += f"[{marker}{length}{delimiter_suffix}]"

    if fields:
        encoded_fields = (encode_key(field) for field in fields)
        header += f"{{{delimiter.join(encoded_fields)}}}"

    header += ":"
    return header
