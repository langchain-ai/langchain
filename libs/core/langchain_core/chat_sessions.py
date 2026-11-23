"""**Chat Sessions** are a collection of messages and function calls."""

from collections.abc import Sequence
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class ChatSession(TypedDict, total=False):
    """Chat Session.

    Chat Session represents a single conversation, channel, or other group of messages.
    """

    messages: Sequence[BaseMessage]
    """A sequence of the LangChain chat messages loaded from the source."""

    functions: Sequence[dict[str, Any]]
    """A sequence of the function calling specs for the messages."""
