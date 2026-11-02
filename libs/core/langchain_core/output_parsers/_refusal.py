"""Refusal handling shared by structured output parsers."""

from langchain_core.exceptions import ModelRefusalError
from langchain_core.messages import BaseMessage
from langchain_core.messages.refusal import get_refusal


def raise_if_refused(message: BaseMessage) -> None:
    """Raise `ModelRefusalError` if the model refused instead of answering."""
    if refusal := get_refusal(message):
        raise ModelRefusalError.from_refusal(refusal, raw_message=message)
