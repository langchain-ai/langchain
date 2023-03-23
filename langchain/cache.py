"""Beta Feature: base interface for cache."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.schema import Generation

RETURN_VAL_TYPE = List[Generation]


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""


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


Base = declarative_base()


class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_llm_cache"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)
    response = Column(String)


class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine, cache_schema: Any = FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            generations = [Generation(text=row[0]) for row in session.execute(stmt)]
            if len(generations) > 0:
                return generations
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Look up based on prompt and llm_string."""
        for i, generation in enumerate(return_val):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, response=generation.text, idx=i
            )
            with Session(self.engine) as session, session.begin():
                session.merge(item)


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)


class RedisCache(BaseCache):
    """Cache that uses Redis as a backend."""

    def __init__(self, redis_: Any):
        """Initialize by passing in Redis instance."""
        try:
            from redis import Redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Redis object.")
        self.redis = redis_

    def _key(self, prompt: str, llm_string: str, idx: int) -> str:
        """Compute key from prompt, llm_string, and idx."""
        return str(hash(prompt + llm_string)) + "_" + str(idx)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        idx = 0
        generations = []
        while self.redis.get(self._key(prompt, llm_string, idx)):
            result = self.redis.get(self._key(prompt, llm_string, idx))
            if not result:
                break
            elif isinstance(result, bytes):
                result = result.decode()
            generations.append(Generation(text=result))
            idx += 1
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for i, generation in enumerate(return_val):
            self.redis.set(self._key(prompt, llm_string, i), generation.text)


class BaseEmbeddingsCache(ABC):
    """Base interface for embeddings cache."""

    @abstractmethod
    def lookup(self, text: str) -> Optional[List[float]]:
        """Look up based on text."""

    @abstractmethod
    def update(
        self, text: str, embeddings: List[float]
    ) -> None:
        """Update cache based on text."""


class InMemoryEmbeddingsCache(BaseEmbeddingsCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[int, Optional[List[float]]] = {}

    def lookup(self, text: str) -> Optional[List[float]]:
        """Look up based on text hash."""
        h = hash(text.encode())
        return self._cache.get(h, None)

    def update(self, text: str, embeddings: List[float]) -> None:
        """Update cache with embeddings from text."""
        h = hash(text.encode())
        self._cache[h] = embeddings


class RedisEmbeddingsCache(BaseEmbeddingsCache):
    """Cache that uses Redis as a backend."""

    index_name: str

    def __init__(self, redis_: Any, index_name: Optional[str] = None):
        """Initialize by passing in Redis instance."""
        try:
            from redis import Redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Redis object.")
        self.redis = redis_
        if not index_name:
            self.index_name = uuid.uuid4().hex
        else:
            self.index_name = index_name
        self.prefix = "text"  # prefix for the text keys

    def lookup(self, text: str) -> Optional[List[float]]:
        """Look up based on text hash."""
        res = self.redis.hget(f'{self.prefix}:{hash(text)}', "embeddings_vector")
        if res:
            try:
                return np.frombuffer(res, dtype=np.float32).tolist()
            except ValueError:
                return None
        return None

    def update(self, text: str, embeddings: List[float]) -> None:
        """Initialize by passing in Redis instance."""
        try:
            from redis.commands.search.field import VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        """Update cache based on text."""
        dim = len(embeddings)
        # Constants
        vector_number = 1  # initial number of vectors
        distance_metric = (
            "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
        )
        content_embedding = VectorField(
            "embeddings_vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": distance_metric,
                "INITIAL_CAP": vector_number,
            },
        )
        fields = [content_embedding]

        # Check if index exists
        try:
            self.redis.ft(self.index_name).info()
            print(f'Index {self.index_name} already exists.')
        except:  # noqa
            # Create Redis Index
            self.redis.ft(self.index_name).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH),
            )

        key = f"{self.prefix}:{hash(text)}"
        self.redis.hset(
            key,
            mapping={
                "embeddings_vector": np.array(
                    embeddings, dtype=np.float32
                ).tobytes(),
            },
        )
