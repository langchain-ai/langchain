"""Global values and configuration that apply to all of LangChain."""
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from langchain.schema import BaseCache


verbose: bool = False
debug: bool = False
llm_cache: Optional["BaseCache"] = None


def set_verbose(value: bool) -> None:
    """Set a new value for the `verbose` global setting."""
    global verbose
    verbose = value

g
def set_debug(value: bool) -> None:
    """Set a new value for the `debug` global setting."""
    global debug
    debug = value


def replace_llm_cache(value: "BaseCache") -> None:
    """Set a new LLM cache, overwriting the previous value, if any."""
    global llm_cache
    llm_cache = value
