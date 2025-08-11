"""Main entrypoint into package."""

from importlib import metadata
from typing import Any

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Get an attribute from the package."""
    if name == "verbose":
        from langchain.globals import _verbose

        return _verbose
    if name == "debug":
        from langchain.globals import _debug

        return _debug
    if name == "llm_cache":
        from langchain.globals import _llm_cache

        return _llm_cache
    msg = f"Could not find: {name}"
    raise AttributeError(msg)
