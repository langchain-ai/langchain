"""Main entrypoint into LangChain."""

from typing import Any

__version__ = "1.0.0a1"


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Get an attribute from the package.

    TODO: will be removed in a future alpha version.
    """
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
