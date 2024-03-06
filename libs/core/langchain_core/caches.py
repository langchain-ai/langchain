"""
.. warning::
  Beta Feature!

**Cache** provides an optional caching layer for LLMs.

Cache is useful for two reasons:

- It can save you money by reducing the number of API calls you make to the LLM
  provider if you're often requesting the same completion multiple times.
- It can speed up your application by reducing the number of API calls you make
  to the LLM provider.

Cache directly competes with Memory. See documentation for Pros and Cons.

**Class hierarchy:**

.. code-block::

    BaseCache --> <name>Cache  # Examples: InMemoryCache, RedisCache, GPTCache
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from langchain_core.outputs import Generation
from langchain_core.runnables import run_in_executor

RETURN_VAL_TYPE = Sequence[Generation]


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments."""

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return await run_in_executor(None, self.lookup, prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Update cache based on prompt and llm_string."""
        return await run_in_executor(None, self.update, prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        """Clear cache that can take additional keyword arguments."""
        return await run_in_executor(None, self.clear, **kwargs)
