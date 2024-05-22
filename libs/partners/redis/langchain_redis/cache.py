"""Redis cache implementation for LangChain."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, List, Optional, Union

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from pydantic.v1 import Field
from redis import Redis
from redis.commands.json.path import Path
from redisvl.extensions.llmcache import (  # type: ignore[import]
    SemanticCache as RedisVLSemanticCache,
)
from redisvl.utils.vectorize import BaseVectorizer  # type: ignore[import]


class EmbeddingsVectorizer(BaseVectorizer):
    embeddings: Embeddings = Field(...)
    model: str = Field(default="custom_embeddings")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, embeddings: Embeddings):
        dims = len(embeddings.embed_query("test"))
        super().__init__(model="custom_embeddings", dims=dims, embeddings=embeddings)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return np.array(self.embeddings.embed_query(texts), dtype=np.float32)
        return np.array(self.embeddings.embed_documents(texts), dtype=np.float32)

    def embed(self, text: str) -> List[float]:
        return self.encode(text).tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts).tolist()

    async def aembed(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed, text)

    async def aembed_many(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_many, texts)


class RedisCache(BaseCache):
    """Redis cache implementation."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: Optional[int] = None,
        prefix: Optional[str] = "redis",
        redis: Optional[Redis] = None,
    ):
        self.redis = redis or Redis.from_url(redis_url)
        self.ttl = ttl
        self.prefix = prefix

    def _key(self, prompt: str, llm_string: str) -> str:
        """Create a key for the cache."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        llm_string_hash = hashlib.md5(llm_string.encode()).hexdigest()
        return f"{self.prefix}:{prompt_hash}:{llm_string_hash}"

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        key = self._key(prompt, llm_string)
        result = self.redis.json().get(key)
        if result:
            return [loads(json.dumps(result))]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        key = self._key(prompt, llm_string)
        json_value = json.loads(dumps(return_val[0]))
        self.redis.json().set(key, Path.root_path(), json_value)
        if self.ttl is not None:
            self.redis.expire(key, self.ttl)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        cursor = 0
        while True:
            pipe = self.redis.pipeline()
            cursor, keys = self.redis.scan(cursor, match=f"{self.prefix}:*", count=100)  # type: ignore[misc]
            if keys:
                pipe.delete(*keys)
                pipe.execute()

            if cursor == 0:
                break


class RedisSemanticCache(BaseCache):
    def __init__(
        self,
        embeddings: Embeddings,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.2,
        ttl: Optional[int] = None,
        name: Optional[str] = "llmcache",
        prefix: Optional[str] = "llmcache",
        redis: Optional[Redis] = None,
    ):
        self.redis = redis or Redis.from_url(redis_url)
        self.embeddings = embeddings
        self.prefix = prefix
        vectorizer = EmbeddingsVectorizer(embeddings=self.embeddings)

        self.cache = RedisVLSemanticCache(
            vectorizer=vectorizer,
            redis_url=redis_url,
            distance_threshold=distance_threshold,
            ttl=ttl,
            name=name,
            prefix=prefix,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        vector = self.cache._vectorize_prompt(prompt)
        results = self.cache.check(vector=vector)

        if results:
            for result in results:
                if result.get("metadata", {}).get("llm_string") == llm_string:
                    try:
                        return [
                            loads(gen_str)
                            for gen_str in json.loads(result.get("response"))
                        ]
                    except (json.JSONDecodeError, TypeError):
                        return None
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        serialized_response = json.dumps([dumps(gen) for gen in return_val])
        vector = self.cache._vectorize_prompt(prompt)

        self.cache.store(
            prompt=prompt,
            response=serialized_response,
            vector=vector,
            metadata={"llm_string": llm_string},
        )

    def clear(self, **kwargs: Any) -> None:
        self.cache.clear()

    def _key(self, prompt: str, llm_string: str) -> str:
        """Create a key for the cache."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        llm_string_hash = hashlib.md5(llm_string.encode()).hexdigest()
        return f"{self.prefix}:{prompt_hash}:{llm_string_hash}"

    def name(self) -> str:
        return self.cache.index.name
