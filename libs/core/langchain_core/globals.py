"""Global values and configuration that apply to all of LangChain."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache


# DO NOT USE THESE VALUES DIRECTLY!
# Use them only via `get_<X>()` and `set_<X>()` below,
# or else your code may behave unexpectedly with other uses of these global settings:
# https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
_verbose: bool = False
_debug: bool = False
_llm_cache: Optional["BaseCache"] = None


def set_verbose(value: bool) -> None:  # noqa: FBT001
    """Set a new value for the `verbose` global setting.

    Args:
        value: The new value for the `verbose` global setting.
    """
    global _verbose  # noqa: PLW0603
    _verbose = value


def get_verbose() -> bool:
    """Get the value of the `verbose` global setting.

    Returns:
        The value of the `verbose` global setting.
    """
    return _verbose


def set_debug(value: bool) -> None:  # noqa: FBT001
    """Set a new value for the `debug` global setting.

    Args:
        value: The new value for the `debug` global setting.
    """
    global _debug  # noqa: PLW0603
    _debug = value


def get_debug() -> bool:
    """Get the value of the `debug` global setting.

    Returns:
        The value of the `debug` global setting.
    """
    return _debug


def set_llm_cache(value: Optional["BaseCache"]) -> None:
    """Set a new LLM cache, overwriting the previous value, if any.

    Args:
        value: The new LLM cache to use. If `None`, the LLM cache is disabled.
    """
    global _llm_cache  # noqa: PLW0603
    _llm_cache = value


def get_llm_cache() -> Optional["BaseCache"]:
    """Get the value of the `llm_cache` global setting.

    Returns:
        The value of the `llm_cache` global setting.
    """
    return _llm_cache
