"""Utilities for getting information about the runtime environment."""

import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def get_runtime_environment() -> dict:
    """Get information about the LangChain runtime environment.

    Returns:
        A dictionary with information about the runtime environment.
    """
    # Lazy import to avoid circular imports
    from langchain_core import __version__

    return {
        "library_version": __version__,
        "library": "langchain-core",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
    }
