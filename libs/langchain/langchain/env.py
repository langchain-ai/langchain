import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def get_runtime_environment() -> dict:
    """Get information about the LangChain runtime environment."""
    # Lazy import to avoid circular imports
    from langchain import __version__

    return {
        "library_version": __version__,
        "library": "langchain",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
    }
