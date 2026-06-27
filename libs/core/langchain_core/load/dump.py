"""Serialize LangChain objects to JSON.

Provides `dumps` (to JSON string) and `dumpd` (to dict) for serializing
`Serializable` objects.

## Escaping

During serialization, plain dicts (user data) that contain an `'lc'` key are escaped
by wrapping them: `{"__lc_escaped__": {...original...}}`. This prevents injection
attacks where malicious data could trick the deserializer into instantiating
arbitrary classes. The escape marker is removed during deserialization.

This is an allowlist approach: only dicts explicitly produced by
`Serializable.to_json()` are treated as LC objects; everything else is escaped if it
could be confused with the LC format.
"""

import json
from typing import Any

from pydantic import BaseModel

from langchain_core.load._validation import _serialize_value
from langchain_core.load.serializable import Serializable, to_json_not_implemented
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def default(obj: Any) -> Any:
    """Return a default value for an object.

    Args:
        obj: The object to serialize to json if it is a Serializable object.

    Returns:
        A JSON serializable object or a SerializedNotImplemented object.
    """
    if isinstance(obj, Serializable):
        return obj.to_json()
    return to_json_not_implemented(obj)


def _dump_pydantic_models(obj: Any) -> Any:
    """Convert nested Pydantic models to dicts for JSON serialization.

    Recursively replaces any Pydantic `BaseModel` instance with its dict
    representation so that the object can be serialized using `json.dumps`.
    This handles models appearing in arbitrary positions (e.g., inside
    `additional_kwargs`, tool calls, or custom Serializable subclasses).

    Args:
        obj: The object to process.

    Returns:
        A copy of the object with all nested Pydantic models converted to
        dicts, or the original object unchanged if no conversion was needed.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _dump_pydantic_models(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump_pydantic_models(v) for v in obj]
    if hasattr(obj, 'model_fields'):
        # LangChain Serializable, pydantic.v1, etc.
        try:
            if hasattr(obj, 'model_dump'):
                return _dump_pydantic_models(obj.model_dump())
        except Exception:
            pass
    return obj


def dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str:
    """Return a JSON string representation of an object.

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: The object to dump.
        pretty: Whether to pretty print the json.

            If `True`, the json will be indented by either 2 spaces or the amount
            provided in the `indent` kwarg.
        **kwargs: Additional arguments to pass to `json.dumps`

    Returns:
        A JSON string representation of the object.

    Raises:
        ValueError: If `default` is passed as a kwarg.
    """
    if "default" in kwargs:
        msg = "`default` should not be passed to dumps"
        raise ValueError(msg)

    obj = _dump_pydantic_models(obj)
    serialized = _serialize_value(obj)

    if pretty:
        indent = kwargs.pop("indent", 2)
        return json.dumps(serialized, indent=indent, **kwargs)
    return json.dumps(serialized, **kwargs)


def dumpd(obj: Any) -> Any:
    """Return a dict representation of an object.

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: The object to dump.

    Returns:
        Dictionary that can be serialized to json using `json.dumps`.
    """
    obj = _dump_pydantic_models(obj)
    return _serialize_value(obj)
