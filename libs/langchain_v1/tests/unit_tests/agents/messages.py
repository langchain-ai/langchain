"""Redefined messages as a work-around for pydantic issue with AnyStr.

The code below creates version of pydantic models
that will work in unit tests with AnyStr as id field
Please note that the `id` field is assigned AFTER the model is created
to workaround an issue with pydantic ignoring the __eq__ method on
subclassed strings.
"""

from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from tests.any_str import AnyStr


def _AnyIdHumanMessage(**kwargs: Any) -> HumanMessage:
    """Create a human message with an any id field."""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdToolMessage(**kwargs: Any) -> ToolMessage:
    """Create a tool message with an any id field."""
    message = ToolMessage(**kwargs)
    message.id = AnyStr()
    return message
