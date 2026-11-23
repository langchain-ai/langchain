"""Normalize TypeSafe state containing LangChain messages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import JsonValue

if TYPE_CHECKING:
    from langchain_typesafe.types import State


def _serialize_state_value(value: object) -> JsonValue:
    if isinstance(value, BaseMessage):
        return _serialize_state_value(convert_to_openai_messages(value))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            message = "TypeSafe state object keys must be strings."
            raise TypeError(message)
        return {key: _serialize_state_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_state_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    message = f"Unsupported TypeSafe state value: {type(value).__name__}."
    raise TypeError(message)


def serialize_state(state: State) -> JsonValue:
    """Recursively convert LangChain messages inside TypeSafe state to JSON.

    Args:
        state: Native TypeSafe state containing zero or more LangChain messages.

    Returns:
        A string, object, or array suitable for the TypeSafe `state` field. Every
        message or message sequence is replaced with role/content JSON while its
        surrounding object and array structure is preserved.

    Raises:
        TypeError: If the root is a JSON scalar other than a string, or if any nested
            value cannot be represented as JSON or LangChain messages.
    """
    if state is None or isinstance(state, (int, float, bool)):
        message = (
            "TypeSafe state must be a string, object, array, BaseMessage, or sequence "
            "of BaseMessage objects."
        )
        raise TypeError(message)
    return _serialize_state_value(state)


__all__ = ["serialize_state"]
