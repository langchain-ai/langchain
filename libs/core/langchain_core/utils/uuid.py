"""UUID utility functions.

This module exports a uuid7 function to generate monotonic, time-ordered UUIDs
for tracing and similar operations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.utils._internal._uuid import uuid7
try:
    # Exported after version 0.4.43.
    # When available, re-use to ensure monotonicity between IDs
    from langsmith import uuid7  # type: ignore[no-redef]
except ImportError:
    from langchain_core.utils._internal._uuid import uuid7


__all__ = ["uuid7"]
