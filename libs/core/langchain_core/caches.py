from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from langchain_core.outputs import Generation

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
