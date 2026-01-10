"""Validation utilities for LangChain serialization.

Provides escape-based protection against injection attacks in serialized objects. The
approach uses an allowlist design: only dicts explicitly produced by
`Serializable.to_json()` are treated as LC objects during deserialization.

## How escaping works

During serialization, plain dicts (user data) that contain an `'lc'` key are wrapped:

```python
{"lc": 1, ...}  # user data that looks like LC object
# becomes:
{"__lc_escaped__": {"lc": 1, ...}}
```

During deserialization, escaped dicts are unwrapped and returned as plain dicts,
NOT instantiated as LC objects.
"""

from typing import Any

from langchain_core.load.serializable import (
    Serializable,
    to_json_not_implemented,
)

_LC_ESCAPED_KEY = "__lc_escaped__"
"""Sentinel key used to mark escaped user dicts during serialization.

When a plain dict contains 'lc' key (which could be confused with LC objects),
we wrap it as {"__lc_escaped__": {...original...}}.
"""


def _needs_escaping(obj: dict[str, Any]) -> bool:
    """Check if a dict needs escaping to prevent confusion with LC objects.

    A dict needs escaping if:

    1. It has an `'lc'` key (could be confused with LC serialization format)
    2. It has only the escape key (would be mistaken for an escaped dict)
    """
    return "lc" in obj or (len(obj) == 1 and _LC_ESCAPED_KEY in obj)


def _escape_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Wrap a dict in the escape marker.

    Example:
        ```python
        {"key": "value"}  # becomes {"__lc_escaped__": {"key": "value"}}
        ```
    """
    return {_LC_ESCAPED_KEY: obj}


def _is_escaped_dict(obj: dict[str, Any]) -> bool:
    """Check if a dict is an escaped user dict.

    Example:
        ```python
        {"__lc_escaped__": {...}}  # is an escaped dict
        ```
    """
    return len(obj) == 1 and _LC_ESCAPED_KEY in obj


def _serialize_value(obj: Any) -> Any:
    """Serialize a value with escaping of user dicts.

    Called recursively on kwarg values to escape any plain dicts that could be confused
    with LC objects.

    Args:
        obj: The value to serialize.

    Returns:
        The serialized value with user dicts escaped as needed.
    """
    if isinstance(obj, Serializable):
        # This is an LC object - serialize it properly (not escaped)
        return _serialize_lc_object(obj)
    if isinstance(obj, dict):
        if not all(isinstance(k, (str, int, float, bool, type(None))) for k in obj):
            # if keys are not json serializable
            return to_json_not_implemented(obj)
        # Check if dict needs escaping BEFORE recursing into values.
        # If it needs escaping, wrap it as-is - the contents are user data that
        # will be returned as-is during deserialization (no instantiation).
        # This prevents re-escaping of already-escaped nested content.
        if _needs_escaping(obj):
            return _escape_dict(obj)
        # Safe dict (no 'lc' key) - recurse into values
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Non-JSON-serializable object (datetime, custom objects, etc.)
    return to_json_not_implemented(obj)


def _is_lc_secret(obj: Any) -> bool:
    """Check if an object is a LangChain secret marker."""
    expected_num_keys = 3
    return (
        isinstance(obj, dict)
        and obj.get("lc") == 1
        and obj.get("type") == "secret"
        and "id" in obj
        and len(obj) == expected_num_keys
    )


def _serialize_lc_object(obj: Any) -> dict[str, Any]:
    """Serialize a `Serializable` object with escaping of user data in kwargs.

    Args:
        obj: The `Serializable` object to serialize.

    Returns:
        The serialized dict with user data in kwargs escaped as needed.

    Note:
        Kwargs values are processed with `_serialize_value` to escape user data (like
        metadata) that contains `'lc'` keys. Secret fields (from `lc_secrets`) are
        skipped because `to_json()` replaces their values with secret markers.
    """
    if not isinstance(obj, Serializable):
        msg = f"Expected Serializable, got {type(obj)}"
        raise TypeError(msg)

    serialized: dict[str, Any] = dict(obj.to_json())

    # Process kwargs to escape user data that could be confused with LC objects
    # Skip secret fields - to_json() already converted them to secret markers
    if serialized.get("type") == "constructor" and "kwargs" in serialized:
        serialized["kwargs"] = {
            k: v if _is_lc_secret(v) else _serialize_value(v)
            for k, v in serialized["kwargs"].items()
        }

    return serialized


def _unescape_value(obj: Any) -> Any:
    """Unescape a value, processing escape markers in dict values and lists.

    When an escaped dict is encountered (`{"__lc_escaped__": ...}`), it's
    unwrapped and the contents are returned AS-IS (no further processing).
    The contents represent user data that should not be modified.

    For regular dicts and lists, we recurse to find any nested escape markers.

    Args:
        obj: The value to unescape.

    Returns:
        The unescaped value.
    """
    if isinstance(obj, dict):
        if _is_escaped_dict(obj):
            # Unwrap and return the user data as-is (no further unescaping).
            # The contents are user data that may contain more escape keys,
            # but those are part of the user's actual data.
            return obj[_LC_ESCAPED_KEY]

        # Regular dict - recurse into values to find nested escape markers
        return {k: _unescape_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unescape_value(item) for item in obj]
    return obj
