"""Serialization module for Well Known LangChain objects.

Specialized JSON serialization for well known LangChain objects that
can be expected to be frequently transmitted between chains.
"""
import json
from typing import Union, Any, Dict

from pydantic import BaseModel
from pydantic import ValidationError

from langchain.schema.messages import (
    HumanMessage,
    SystemMessage,
    ChatMessage,
    FunctionMessage,
    AIMessage,
    HumanMessageChunk,
    SystemMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    AIMessageChunk,
)

WellKnownTypes = (
    HumanMessage,
    SystemMessage,
    ChatMessage,
    FunctionMessage,
    AIMessage,
    HumanMessageChunk,
    SystemMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    AIMessageChunk,
)


class WellKnownLCObject(BaseModel):
    """A well known LangChain object."""

    __root__: Union[
        HumanMessage,
        SystemMessage,
        ChatMessage,
        FunctionMessage,
        AIMessage,
        HumanMessageChunk,
        SystemMessageChunk,
        ChatMessageChunk,
        FunctionMessageChunk,
        AIMessageChunk,
    ]


# Custom JSON Encoder
class _LangChainEncoder(json.JSONEncoder):
    """Custom JSON Encoder that can encode pydantic objects as well."""

    def default(self, obj) -> Any:
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)


# Custom JSON Decoder
class _LangChainDecoder(json.JSONDecoder):
    """Custom JSON Decoder that handles well known LangChain objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the LangChainDecoder."""
        super().__init__(object_hook=self.decoder, *args, **kwargs)

    def decoder(self, value) -> Any:
        """Decode the value."""
        if isinstance(value, dict):
            try:
                obj = WellKnownLCObject.parse_obj(value)
                return obj.dict()["__root__"]
            except ValidationError:
                return value
        return value


# PUBLIC API


def simple_dumpd(obj: Any) -> Any:
    """Convert the given object to a JSON serializable object."""
    return json.loads(json.dumps(obj, cls=_LangChainEncoder))


def dumps(obj: Any) -> str:
    """Dump the given object as a JSON string."""
    return json.dumps(obj, cls=_LangChainEncoder)


def loads(s: str) -> Any:
    """Load the given JSON string."""
    return json.loads(s, cls=_LangChainDecoder)
