"""Value normalization utilities for converting Python objects to JSON-like values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from math import copysign, isfinite
from typing import Any

from .constants import JsonArray, JsonValue


def normalize_value(value: Any) -> JsonValue:
    """Normalize arbitrary input into a JSON-compatible structure.

    This function converts various Python types into a JSON-like representation
    suitable for TOON encoding. It handles primitives, collections, dataclasses,
    datetime objects, and delegates LangChain-specific types to specialized handlers.

    Args:
        value: Arbitrary Python object to normalize.

    Returns:
        A JSON-compatible value (primitive, dict, or list).
    """
    if value is None:
        return None

    if isinstance(value, (str, bool)):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float):
            if not isfinite(value):
                return None
            # Normalize -0.0 to 0
            if value == 0.0 and copysign(1.0, value) == -1.0:
                return 0
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # Import here to avoid circular dependencies
    from .langchain_support import normalize_langchain_value

    langchain_result = normalize_langchain_value(value)
    if langchain_result is not None:
        return langchain_result

    if is_dataclass(value) and not isinstance(value, type):
        return normalize_value(asdict(value))

    if isinstance(value, (set, frozenset)):
        return [normalize_value(v) for v in value]

    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized[str(key)] = normalize_value(item)
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_value(item) for item in value]

    return None


def is_json_primitive(value: Any) -> bool:
    """Check if a value is a JSON primitive type.

    Args:
        value: Value to check.

    Returns:
        `True` if the value is None, str, int, float, or bool.
    """
    return value is None or isinstance(value, (str, int, float, bool))


def is_json_array(value: Any) -> bool:
    """Check if a value is a JSON array (list-like).

    Args:
        value: Value to check.

    Returns:
        `True` if the value is a sequence but not a string or bytes.
    """
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def is_json_object(value: Any) -> bool:
    """Check if a value is a JSON object (dict-like).

    Args:
        value: Value to check.

    Returns:
        `True` if the value is a mapping.
    """
    return isinstance(value, Mapping)


def is_array_of_primitives(value: JsonArray) -> bool:
    """Check if an array contains only primitive values.

    Args:
        value: Array to check.

    Returns:
        `True` if all elements are primitives.
    """
    return all(is_json_primitive(item) for item in value)


def is_array_of_arrays(value: JsonArray) -> bool:
    """Check if an array contains only arrays.

    Args:
        value: Array to check.

    Returns:
        `True` if all elements are arrays.
    """
    return all(is_json_array(item) for item in value)


def is_array_of_objects(value: JsonArray) -> bool:
    """Check if an array contains only objects.

    Args:
        value: Array to check.

    Returns:
        `True` if all elements are objects.
    """
    return all(is_json_object(item) for item in value)
