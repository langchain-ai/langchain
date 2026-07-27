"""Memory factory: instantiate memory backend according to settings.

For MVP this returns an in-memory backend. Redis/Postgres backends can be
added later and selected via config without changing business logic.
"""
from __future__ import annotations

from typing import Any

from content_growth_agent.config.settings import settings
from content_growth_agent.core.memory_backends.in_memory import InMemoryMemory


def get_memory(**override: Any):
    """Return a memory backend instance according to configuration.

    Args:
        override: Optional overrides (e.g., memory_type='redis')
    """
    memory_type = override.get("memory_type", settings.memory_type.value)

    if memory_type == "in_memory":
        return InMemoryMemory()

    raise NotImplementedError(f"Memory backend '{memory_type}' not implemented yet")
