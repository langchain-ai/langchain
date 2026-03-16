"""TTL-based in-memory cache for non-serializable middleware state."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

__all__ = ["MiddlewareSessionCache"]

T = TypeVar("T")


class MiddlewareSessionCache(Generic[T]):
    """TTL-based in-memory cache for per-session non-serializable middleware state.

    Middleware instances are shared across all graph nodes, making them a natural
    home for non-serializable per-session state (e.g., `asyncio.Task` references,
    live index objects, `BaseTool` references) that cannot pass through LangGraph's
    `JsonPlusSerializer` at checkpoint boundaries.

    This utility provides per-session isolation keyed by `thread_id`, idle-TTL
    eviction with refresh-on-read semantics, an optional eviction callback for
    resource cleanup, and thread-safety via an internal lock.

    Example:
        ```python
        from dataclasses import dataclass, field
        import asyncio
        from langchain.agents.middleware import MiddlewareSessionCache

        @dataclass
        class _SessionState:
            tasks: set[asyncio.Task] = field(default_factory=set)

        class MyMiddleware:
            def __init__(self) -> None:
                self._cache: MiddlewareSessionCache[_SessionState] = (
                    MiddlewareSessionCache(
                        idle_ttl=3600.0,
                        on_evict=lambda s: [t.cancel() for t in s.tasks],
                    )
                )

            async def abefore_agent(self, state, config):
                thread_id = config["configurable"]["thread_id"]
                if self._cache.get(thread_id) is None:
                    self._cache.put(thread_id, _SessionState())
        ```

    Args:
        idle_ttl: Seconds of inactivity after which an entry is eligible for
            eviction. Reads refresh the timer so active sessions never expire.
        on_evict: Optional callback invoked with the evicted value whenever an
            entry is removed via `sweep_expired` or `pop`. Use this to cancel
            tasks, close connections, or flush indexes.
    """

    def __init__(
        self,
        *,
        idle_ttl: float = 3600.0,
        on_evict: Callable[[T], None] | None = None,
    ) -> None:
        self._idle_ttl = idle_ttl
        self._on_evict = on_evict
        self._entries: dict[str, tuple[T, float]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> T | None:
        """Return the cached value for `session_id`, refreshing its TTL.

        Args:
            session_id: The session identifier (typically `thread_id`).

        Returns:
            The cached value, or `None` if no entry exists for `session_id`.
        """
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            value, _ = entry
            self._entries[session_id] = (value, time.monotonic())
            return value

    def put(self, session_id: str, value: T) -> None:
        """Store `value` under `session_id`, setting its last-accessed time to now.

        Args:
            session_id: The session identifier (typically `thread_id`).
            value: The non-serializable object to cache.
        """
        with self._lock:
            self._entries[session_id] = (value, time.monotonic())

    def pop(self, session_id: str) -> T | None:
        """Remove and return the entry for `session_id`, invoking `on_evict` if set.

        Args:
            session_id: The session identifier to remove.

        Returns:
            The removed value, or `None` if no entry existed.
        """
        with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is None:
            return None
        value, _ = entry
        if self._on_evict is not None:
            self._on_evict(value)
        return value

    def sweep_expired(self) -> list[T]:
        """Evict all entries whose idle time exceeds `idle_ttl`.

        Invokes `on_evict` for each removed entry (if set). Intended to be
        called periodically, e.g., from `aafter_agent` or a background task.

        Returns:
            List of evicted values (may be empty).
        """
        now = time.monotonic()
        with self._lock:
            expired_ids = [
                sid
                for sid, (_, last_accessed) in self._entries.items()
                if now - last_accessed > self._idle_ttl
            ]
            evicted = [self._entries.pop(sid)[0] for sid in expired_ids]

        if self._on_evict is not None:
            for value in evicted:
                self._on_evict(value)
        return evicted

    def __len__(self) -> int:
        """Return the number of currently cached entries."""
        with self._lock:
            return len(self._entries)

    def __contains__(self, session_id: object) -> bool:
        """Return `True` if `session_id` has a cached entry."""
        with self._lock:
            return session_id in self._entries
