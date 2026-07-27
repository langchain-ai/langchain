"""Abstract memory interface for conversation history storage.

Defines a small protocol (async-friendly) that memory backends must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseMemory(ABC):
    """Abstract base class for memory backends.

    Implementations must be async-compatible to allow future remote stores.
    """

    @abstractmethod
    async def add_message(self, message: ChatMessage) -> None:
        """Add a message to history."""

    @abstractmethod
    async def get_history(self) -> List[ChatMessage]:
        """Return the conversation history as a list of ChatMessage."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear stored history."""
