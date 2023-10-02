"""Global values and configuration that apply to all of LangChain."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain.schema import BaseCache


verbose: bool = False
debug: bool = False
llm_cache: Optional["BaseCache"] = None


def set_verbose(value: bool) -> None:
    """Set a new value for the `verbose` global setting."""
    import langchain

    # N.B.: This is a workaround for an unfortunate quirk of Python's
    #       module-level `__getattr__()` implementation:
    # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
    #
    # Remove it once `langchain.verbose` is no longer supported, and once all users
    # have migrated to using `set_verbose()` here.
    langchain.verbose = value

    global verbose
    verbose = value


def set_debug(value: bool) -> None:
    """Set a new value for the `debug` global setting."""
    import langchain

    # N.B.: This is a workaround for an unfortunate quirk of Python's
    #       module-level `__getattr__()` implementation:
    # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
    #
    # Remove it once `langchain.debug` is no longer supported, and once all users
    # have migrated to using `set_debug()` here.
    langchain.debug = value

    global debug
    debug = value


def set_llm_cache(value: "BaseCache") -> None:
    """Set a new LLM cache, overwriting the previous value, if any."""
    import langchain

    # N.B.: This is a workaround for an unfortunate quirk of Python's
    #       module-level `__getattr__()` implementation:
    # https://github.com/langchain-ai/langchain/pull/11311#issuecomment-1743780004
    #
    # Remove it once `langchain.llm_cache` is no longer supported, and once all users
    # have migrated to using `set_llm_cache()` here.
    langchain.llm_cache = value

    global llm_cache
    llm_cache = value
