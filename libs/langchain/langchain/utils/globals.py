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

    langchain.verbose = value

    global verbose
    verbose = value


def set_debug(value: bool) -> None:
    """Set a new value for the `debug` global setting."""
    import langchain

    langchain.debug = value

    global debug
    debug = value


def set_llm_cache(value: "BaseCache") -> None:
    """Set a new LLM cache, overwriting the previous value, if any."""
    import langchain

    langchain.llm_cache = value

    global llm_cache
    llm_cache = value
