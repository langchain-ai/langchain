from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


class AnyStr(str):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)


# The code below creates version of pydantic models
# that will work in unit tests with AnyStr as id field
# Please note that the `id` field is assigned AFTER the model is created
# to workaround an issue with pydantic ignoring the __eq__ method on
# subclassed strings.


def _any_id_document(**kwargs: Any) -> Document:
    """Create a document with an id field."""
    message = Document(**kwargs)
    message.id = AnyStr()
    return message


def _any_id_ai_message(**kwargs: Any) -> AIMessage:
    """Create ai message with an any id field."""
    message = AIMessage(**kwargs)
    message.id = AnyStr()
    return message


def _any_id_ai_message_chunk(**kwargs: Any) -> AIMessageChunk:
    """Create ai message with an any id field."""
    message = AIMessageChunk(**kwargs)
    message.id = AnyStr()
    return message


def _any_id_human_message(**kwargs: Any) -> HumanMessage:
    """Create a human with an any id field."""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message
