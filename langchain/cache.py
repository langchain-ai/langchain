"""Beta Feature: base interface for cache."""
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class GPTCache(BaseCache):
    """Cache that uses GPTCache as a backend."""

    def __init__(self, init_func: Callable[[Any], None]):
        """Initialize by passing in the `init` GPTCache func

        Args:
            init_func (Callable[[Any], None]): init `GPTCache` function

        Example:
        .. code-block:: python

            import gptcache
            from gptcache.processor.pre import get_prompt
            from gptcache.manager.factory import get_data_manager

            # Avoid multiple caches using the same file,
            causing different llm model caches to affect each other
            i = 0
            file_prefix = "data_map"

            def init_gptcache_map(cache_obj: gptcache.Cache):
                nonlocal i
                cache_path = f'{file_prefix}_{i}.txt'
                cache_obj.init(
                    pre_embedding_func=get_prompt,
                    data_manager=get_data_manager(data_path=cache_path),
                )
                i += 1

            langchain.llm_cache = GPTCache(init_gptcache_map)

        """
        try:
            import gptcache  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import gptcache python package. "
                "Please install it with `pip install gptcache`."
            )
        self.init_gptcache_func: Callable[[Any], None] = init_func
        self.gptcache_dict: Dict[str, Any] = {}

    @staticmethod
    def _update_cache_callback_none(*_: Any, **__: Any) -> None:
        """When updating cached data, do nothing.

        Because currently only cached queries are processed."""
        return None

    @staticmethod
    def _llm_handle_none(*_: Any, **__: Any) -> None:
        """Do nothing on a cache miss"""
        return None

    @staticmethod
    def _cache_data_converter(data: str) -> RETURN_VAL_TYPE:
        """Convert the `data` in the cache to the `RETURN_VAL_TYPE` data format."""
        return [Generation(**generation_dict) for generation_dict in json.loads(data)]

    def _get_gptcache(self, llm_string: str) -> Any:
        """Get a cache object.

        When the corresponding llm model cache does not exist, it will be created."""
        from gptcache import Cache

        _gptcache = self.gptcache_dict.get(llm_string, None)
        if _gptcache is None:
            _gptcache = Cache()
            self.init_gptcache_func(_gptcache)
            self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the cache data.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then retrieve the data from the cache based on the `prompt`.
        """
        from gptcache.adapter.adapter import adapt

        _gptcache = self.gptcache_dict.get(llm_string)
        if _gptcache is None:
            return None
        res = adapt(
            GPTCache._llm_handle_none,
            GPTCache._cache_data_converter,
            GPTCache._update_cache_callback_none,
            cache_obj=_gptcache,
            prompt=prompt,
        )
        return res

    @staticmethod
    def _update_cache_callback(
        llm_data: RETURN_VAL_TYPE, update_cache_func: Callable[[Any], None]
    ) -> None:
        """Save the `llm_data` to cache storage"""
        handled_data = json.dumps([generation.dict() for generation in llm_data])
        update_cache_func(handled_data)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then store the `prompt` and `return_val` in the cache object.
        """
        from gptcache.adapter.adapter import adapt

        _gptcache = self._get_gptcache(llm_string)

        def llm_handle(*_: Any, **__: Any) -> RETURN_VAL_TYPE:
            return return_val

        return adapt(
            llm_handle,
            GPTCache._cache_data_converter,
            GPTCache._update_cache_callback,
            cache_obj=_gptcache,
            cache_skip=True,
            prompt=prompt,
        )
