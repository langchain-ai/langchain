from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.utils import mustache
from pydantic import BaseModel

from .core import Invoker, Prompty, SimpleModel


def _convert_messages_to_dicts(data: Any) -> Any:
    """Recursively convert BaseMessage objects to dicts for mustache compatibility.

    The mustache implementation only supports dict, list, and tuple traversal
    for security reasons. This converts any BaseMessage objects to dicts.
    """
    if isinstance(data, BaseMessage):
        return data.model_dump()
    elif isinstance(data, dict):
        return {k: _convert_messages_to_dicts(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_messages_to_dicts(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_convert_messages_to_dicts(item) for item in data)
    return data


class MustacheRenderer(Invoker):
    """Render a mustache template."""

    def __init__(self, prompty: Prompty) -> None:
        self.prompty = prompty

    def invoke(self, data: BaseModel) -> BaseModel:
        if not isinstance(data, SimpleModel):
            raise ValueError("Expected data to be an instance of SimpleModel")
        # Convert any BaseMessage objects to dicts for mustache compatibility
        converted_data = _convert_messages_to_dicts(data.item)
        generated = mustache.render(self.prompty.content, converted_data)
        return SimpleModel[str](item=generated)
