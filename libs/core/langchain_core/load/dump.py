"""Dump objects to json."""

import json
from typing import Any

from pydantic import BaseModel

from langchain_core.load.serializable import Serializable, to_json_not_implemented
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def default(obj: Any) -> Any:
    """Return a default value for an object.

    Args:
        obj: The object to serialize to json if it is a Serializable object.

    Returns:
        A json serializable object or a SerializedNotImplemented object.
    """
    if isinstance(obj, Serializable):
        return obj.to_json()
    return to_json_not_implemented(obj)


def _dump_pydantic_models(obj: Any) -> Any:
    if (
        isinstance(obj, ChatGeneration)
        and isinstance(obj.message, AIMessage)
        and (parsed := obj.message.additional_kwargs.get("parsed"))
        and isinstance(parsed, BaseModel)
    ):
        obj_copy = obj.model_copy(deep=True)
        obj_copy.message.additional_kwargs["parsed"] = parsed.model_dump()
        return obj_copy
    return obj


def dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str:
    """Return a json string representation of an object.

    Args:
        obj: The object to dump.
        pretty: Whether to pretty print the json. If true, the json will be
            indented with 2 spaces (if no indent is provided as part of kwargs).
            Default is False.
        kwargs: Additional arguments to pass to json.dumps

    Returns:
        A json string representation of the object.

    Raises:
        ValueError: If `default` is passed as a kwarg.
    """
    if "default" in kwargs:
        msg = "`default` should not be passed to dumps"
        raise ValueError(msg)
    try:
        obj = _dump_pydantic_models(obj)
        if pretty:
            indent = kwargs.pop("indent", 2)
            return json.dumps(obj, default=default, indent=indent, **kwargs)
        return json.dumps(obj, default=default, **kwargs)
    except TypeError:
        if pretty:
            indent = kwargs.pop("indent", 2)
            return json.dumps(to_json_not_implemented(obj), indent=indent, **kwargs)
        return json.dumps(to_json_not_implemented(obj), **kwargs)


def dumpd(obj: Any) -> Any:
    """Return a dict representation of an object.

    .. note::
        Unfortunately this function is not as efficient as it could be because it first
        dumps the object to a json string and then loads it back into a dictionary.

    Args:
        obj: The object to dump.

    Returns:
        dictionary that can be serialized to json using json.dumps
    """
    return json.loads(dumps(obj))
