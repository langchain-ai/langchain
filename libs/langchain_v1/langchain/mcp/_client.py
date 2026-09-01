"""Private FastMCP client helpers used by the MCP adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastmcp.client.group import ClientGroup
from typing_extensions import Self

from langchain.mcp.elicitation import _declare_elicitation_capability

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


def _group_declaring_elicitation(group: ClientGroup) -> ClientGroup:
    """Rebuild a group with clients that advertise elicitation support.

    FastMCP declares the capability per client, so a group must declare it on
    every member. Cloning rather than calling `set_elicitation_callback` on the
    originals keeps the caller's own clients untouched.
    """
    clients = {}
    for name, member in group.clients.items():
        clone = member.new()
        clone.set_elicitation_callback(_declare_elicitation_capability)
        clients[name] = clone
    return ClientGroup(clients)
