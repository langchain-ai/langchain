"""**Chat Sessions** are a collection of messages and function calls.

"""
from typing import Sequence, TypedDict

from langchain_core.messages import BaseMessage


class ChatSession(TypedDict, total=False):
    """Chat Session represents a single
    conversation, channel, or other group of messages."""

    messages: Sequence[BaseMessage]
    """The LangChain chat messages loaded from the source."""
    functions: Sequence[dict]
    """The function calling specs for the messages."""
