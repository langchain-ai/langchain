"""Refusal handling shared by structured output parsers."""

from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import BaseMessage


def raise_if_refused(message: BaseMessage) -> None:
    """Raise `ModelRefusalError` if the model refused instead of answering."""
    for block in message.content_blocks:
        if block["type"] == "refusal":
            raise ModelRefusalError.from_refusal(block, raw_message=message)
