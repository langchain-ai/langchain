from typing import Any, Dict, Optional, Sequence, Tuple

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.outputs import Generation

DEPRECATED_IMPORTS = [
    "FullLLMCache",
    "SQLAlchemyCache",
    "SQLiteCache",
    "UpstashRedisCache",
    "RedisCache",
    "RedisSemanticCache",
    "GPTCache",
    "MomentoCache",
    "CassandraCache",
    "CassandraSemanticCache",
    "FullMd5LLMCache",
    "SQLAlchemyMd5Cache",
    "AstraDBCache",
    "AstraDBSemanticCache",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.cache import {name}`"
        )

    raise AttributeError()


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[Tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._cache = {}

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self.lookup(prompt, llm_string)

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Update cache based on prompt and llm_string."""
        self.update(prompt, llm_string, return_val)

    async def aclear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self.clear()
