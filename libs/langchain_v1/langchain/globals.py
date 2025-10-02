"""Global values and configuration that apply to all of LangChain.

TODO: will be removed in a future alpha version.
"""

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
