"""In-memory memory backend for MVP.

Simple, thread-safe-ish implementation using an in-process list. Not durable,
intended for MVP and tests. Interface matched to BaseMemory.
"""
from __future__ import annotations

from typing import List
import asyncio

from content_growth_agent.core.base_memory import BaseMemory, ChatMessage


class InMemoryMemory(BaseMemory):
    """A simple in-memory list-backed conversation history.

    This is NOT suitable for production but is perfect for an MVP. It stores
    messages in process and supports async interface.
    """

    def __init__(self) -> None:
        self._messages: List[ChatMessage] = []
        self._lock = asyncio.Lock()

    async def add_message(self, message: ChatMessage) -> None:
        async with self._lock:
            self._messages.append(message)

    async def get_history(self) -> List[ChatMessage]:
        async with self._lock:
            # return a shallow copy to avoid external mutation
            return list(self._messages)

    async def clear(self) -> None:
        async with self._lock:
            self._messages.clear()
