from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage


class AnyStr(str):
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str)

    __hash__ = str.__hash__


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
    # Set default additional_kwargs to include output_version if not provided
    if "additional_kwargs" not in kwargs:
        kwargs["additional_kwargs"] = {"output_version": "v0"}
    elif (
        isinstance(kwargs["additional_kwargs"], dict)
        and "output_version" not in kwargs["additional_kwargs"]
    ):
        kwargs["additional_kwargs"] = {
            **kwargs["additional_kwargs"],
            "output_version": "v0",
        }

    message = AIMessage(**kwargs)
    message.id = AnyStr()
    return message


def _any_id_ai_message_chunk(**kwargs: Any) -> AIMessageChunk:
    """Create ai message with an any id field."""
    # Only exclude output_version from last chunks that have empty content
    # (synthetic chunks)
    is_empty_last_chunk = (
        kwargs.get("chunk_position") == "last"
        and not kwargs.get("content")
        and "additional_kwargs" not in kwargs
    )

    # Set default additional_kwargs to include output_version if not provided and not
    # an empty last chunk
    if not is_empty_last_chunk:
        if "additional_kwargs" not in kwargs:
            kwargs["additional_kwargs"] = {"output_version": "v0"}
        elif (
            isinstance(kwargs["additional_kwargs"], dict)
            and "output_version" not in kwargs["additional_kwargs"]
        ):
            kwargs["additional_kwargs"] = {
                **kwargs["additional_kwargs"],
                "output_version": "v0",
            }

    message = AIMessageChunk(**kwargs)
    message.id = AnyStr()
    return message


def _any_id_human_message(**kwargs: Any) -> HumanMessage:
    """Create a human with an any id field."""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message
