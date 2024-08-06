from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)


def _AnyIdDocument(**kwargs) -> Document:
    """Create a document with an id field."""
    message = Document(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdAIMessage(**kwargs) -> AIMessage:
    """Create ai message with an any id field."""
    message = AIMessage(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdAIMessageChunk(**kwargs) -> AIMessageChunk:
    """Create ai message with an any id field."""
    message = AIMessageChunk(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdHumanMessage(**kwargs) -> HumanMessage:
    """Create a human with an any id field."""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message
