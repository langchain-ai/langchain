import json
from typing import Union, Any

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
class LangChainEncoder(json.JSONEncoder):
    """Custom JSON Encoder that handles well known LangChain objects."""

    def default(self, obj):
        if isinstance(obj, WellKnownTypes):
            return obj.dict()
        return super().default(obj)


# Custom JSON Decoder
class LangChainDecoder(json.JSONDecoder):
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
