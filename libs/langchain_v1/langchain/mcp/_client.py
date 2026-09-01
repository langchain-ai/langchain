"""Private FastMCP client helpers used by the MCP adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastmcp.client.group import ClientGroup
from typing_extensions import Self

if TYPE_CHECKING:
    from types import TracebackType


class _ReentrantClientGroup(ClientGroup):
    """A `ClientGroup` whose context manager supports nested and concurrent users."""

    def __init__(self, group: ClientGroup) -> None:
        """Rebuild the group coordinator around the caller's existing clients."""
        super().__init__(group.clients)
        self._depth = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Acquire a shared connection to this group."""
        async with self._lock:
            if self._depth == 0:
                await super().__aenter__()
            self._depth += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release a shared connection to this group."""
        async with self._lock:
            self._depth -= 1
            if self._depth == 0:
                await super().__aexit__(exc_type, exc_value, traceback)
