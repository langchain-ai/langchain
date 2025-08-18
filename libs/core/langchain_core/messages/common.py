"""Common constants and utilities for LangChain messages."""

from typing import Optional
from uuid import uuid4

LC_AUTO_PREFIX = "lc_"
"""LangChain auto-generated ID prefix for messages and content blocks."""

LC_ID_PREFIX = f"{LC_AUTO_PREFIX}run-"
"""Internal tracing/callback system identifier.

Used for:
- Tracing. Every LangChain operation (LLM call, chain execution, tool use, etc.)
  gets a unique run_id (UUID)
- Enables tracking parent-child relationships between operations
"""


def ensure_id(id_val: Optional[str]) -> str:
    """Ensure the ID is a valid string, generating a new UUID if not provided.

    Auto-generated UUIDs are prefixed by ``'lc_'`` to indicate they are
    LangChain-generated IDs.

    Args:
        id_val: Optional string ID value to validate.

    Returns:
        A string ID, either the validated provided value or a newly generated UUID4.
    """
    return id_val or str(f"{LC_AUTO_PREFIX}{uuid4()}")
