"""Global values and configuration that apply to all of LangChain."""

import warnings
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


def set_verbose(
    value: bool,  # noqa: FBT001
) -> None:
    """Set a new value for the `verbose` global setting."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Importing verbose from langchain root module is no longer supported"
            ),
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.verbose` is no longer supported, and once all users
        # have migrated to using `set_verbose()` here.
        langchain.verbose = value

    global _verbose  # noqa: PLW0603
    _verbose = value


def get_verbose() -> bool:
    """Get the value of the `verbose` global setting."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Importing verbose from langchain root module is no longer supported"
            ),
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.verbose` is no longer supported, and once all users
        # have migrated to using `set_verbose()` here.
        #
        # In the meantime, the `verbose` setting is considered True if either the old
        # or the new value are True. This accommodates users who haven't migrated
        # to using `set_verbose()` yet. Those users are getting deprecation warnings
        # directing them to use `set_verbose()` when they import `langchain.verbose`.
        old_verbose = langchain.verbose

    return _verbose or old_verbose


def set_debug(
    value: bool,  # noqa: FBT001
) -> None:
    """Set a new value for the `debug` global setting."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Importing debug from langchain root module is no longer supported",
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.debug` is no longer supported, and once all users
        # have migrated to using `set_debug()` here.
        langchain.debug = value

    global _debug  # noqa: PLW0603
    _debug = value


def get_debug() -> bool:
    """Get the value of the `debug` global setting."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Importing debug from langchain root module is no longer supported",
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.debug` is no longer supported, and once all users
        # have migrated to using `set_debug()` here.
        #
        # In the meantime, the `debug` setting is considered True if either the old
        # or the new value are True. This accommodates users who haven't migrated
        # to using `set_debug()` yet. Those users are getting deprecation warnings
        # directing them to use `set_debug()` when they import `langchain.debug`.
        old_debug = langchain.debug

    return _debug or old_debug


def set_llm_cache(value: Optional["BaseCache"]) -> None:
    """Set a new LLM cache, overwriting the previous value, if any."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Importing llm_cache from langchain root module is no longer supported"
            ),
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.llm_cache` is no longer supported, and
        # once all users have migrated to using `set_llm_cache()` here.
        langchain.llm_cache = value

    global _llm_cache  # noqa: PLW0603
    _llm_cache = value


def get_llm_cache() -> "BaseCache":
    """Get the value of the `llm_cache` global setting."""
    import langchain

    # We're about to run some deprecated code, don't report warnings from it.
    # The user called the correct (non-deprecated) code path and shouldn't get warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Importing llm_cache from langchain root module is no longer supported"
            ),
        )
        # N.B.: This is a workaround for an unfortunate quirk of Python's
        #       module-level `__getattr__()` implementation:
        # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
        #
        # Remove it once `langchain.llm_cache` is no longer supported, and
        # once all users have migrated to using `set_llm_cache()` here.
        #
        # In the meantime, the `llm_cache` setting returns whichever of
        # its two backing sources is truthy (not `None` and non-empty),
        # or the old value if both are falsy. This accommodates users
        # who haven't migrated to using `set_llm_cache()` yet.
        # Those users are getting deprecation warnings directing them
        # to use `set_llm_cache()` when they import `langchain.llm_cache`.
        old_llm_cache = langchain.llm_cache

    return _llm_cache or old_llm_cache
