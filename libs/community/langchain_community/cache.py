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

import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

from langchain_community.utilities.cassandra import SetupMode as CassandraSetupMode
from langchain_community.vectorstores.azure_cosmos_db import (
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_community.vectorstores.utils import DistanceStrategy

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core._api.deprecation import deprecated, warn_deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env

from langchain_community.utilities.astradb import (
    SetupMode as AstraSetupMode,
)
from langchain_community.utilities.astradb import (
    _AstraDBCollectionEnvironment,
)
from langchain_community.vectorstores import (
    AzureCosmosDBNoSqlVectorSearch,
    AzureCosmosDBVectorSearch,
)
from langchain_community.vectorstores import (
    OpenSearchVectorSearch as OpenSearchVectorStore,
)
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
from langchain_community.vectorstores.singlestoredb import SingleStoreDB

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    import momento
    import pymemcache
    from astrapy.db import AstraDB, AsyncAstraDB
    from azure.cosmos.cosmos_client import CosmosClient
    from cassandra.cluster import Session as CassandraSession


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: Json representing a list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    return json.dumps([generation.dict() for generation in generations])


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """
    Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This function (+ its counterpart `_loads_generations`) rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """
    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            f"Legacy 'Generation' cached blob encountered: '{generations_str}'"
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            f"Malformed/unparsable cached blob encountered: '{generations_str}'"
        )
        return None


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

    def __init__(self, engine: Engine, cache_schema: Type[FullLLMCache] = FullLLMCache):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)  # type: ignore
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                try:
                    return [loads(row[0]) for row in rows]
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    return [Generation(text=row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        items = [
            self.cache_schema(prompt=prompt, llm=llm_string, response=dumps(gen), idx=i)
            for i, gen in enumerate(return_val)
        ]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        with Session(self.engine) as session:
            session.query(self.cache_schema).delete()
            session.commit()


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)


class UpstashRedisCache(BaseCache):
    """Cache that uses Upstash Redis as a backend."""

    def __init__(self, redis_: Any, *, ttl: Optional[int] = None):
        """
        Initialize an instance of UpstashRedisCache.

        This method initializes an object with Upstash Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of an Upstash Redis
        client class, allowing the object to interact with Upstash Redis
        server for caching purposes.

        Parameters:
            redis_: An instance of Upstash Redis client class
                (e.g., Redis) used for caching.
                This allows the object to communicate with
                Redis server for caching operations on.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from upstash_redis import Redis
        except ImportError:
            raise ImportError(
                "Could not import upstash_redis python package. "
                "Please install it with `pip install upstash_redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Upstash Redis object.")
        self.redis = redis_
        self.ttl = ttl

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        # Read from a HASH
        results = self.redis.hgetall(self._key(prompt, llm_string))
        if results:
            for _, text in results.items():
                generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "UpstashRedisCache supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
            if isinstance(gen, ChatGeneration):
                warnings.warn(
                    "NOTE: Generation has not been cached. UpstashRedisCache does not"
                    " support caching ChatModel outputs."
                )
                return
        # Write to a HASH
        key = self._key(prompt, llm_string)

        mapping = {
            str(idx): generation.text for idx, generation in enumerate(return_val)
        }
        self.redis.hset(key=key, values=mapping)

        if self.ttl is not None:
            self.redis.expire(key, self.ttl)

    def clear(self, **kwargs: Any) -> None:
        """
        Clear cache. If `asynchronous` is True, flush asynchronously.
        This flushes the *whole* db.
        """
        asynchronous = kwargs.get("asynchronous", False)
        if asynchronous:
            asynchronous = "ASYNC"
        else:
            asynchronous = "SYNC"
        self.redis.flushdb(flush_type=asynchronous)


class _RedisCacheBase(BaseCache, ABC):
    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    @staticmethod
    def _ensure_generation_type(return_val: RETURN_VAL_TYPE) -> None:
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "RedisCache only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )

    @staticmethod
    def _get_generations(
        results: dict[str | bytes, str | bytes],
    ) -> Optional[List[Generation]]:
        generations = []
        if results:
            for _, text in results.items():
                try:
                    generations.append(loads(cast(str, text)))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.append(Generation(text=text))  # type: ignore[arg-type]
        return generations if generations else None

    @staticmethod
    def _configure_pipeline_for_update(
        key: str, pipe: Any, return_val: RETURN_VAL_TYPE, ttl: Optional[int] = None
    ) -> None:
        pipe.hset(
            key,
            mapping={
                str(idx): dumps(generation) for idx, generation in enumerate(return_val)
            },
        )
        if ttl is not None:
            pipe.expire(key, ttl)


class RedisCache(_RedisCacheBase):
    """
    Cache that uses Redis as a backend. Allows to use a sync `redis.Redis` client.
    """

    def __init__(self, redis_: Any, *, ttl: Optional[int] = None):
        """
        Initialize an instance of RedisCache.

        This method initializes an object with Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of a Redis
        client class (`redis.Redis`), allowing the object
        to interact with a Redis server for caching purposes.

        Parameters:
            redis_ (Any): An instance of a Redis client class
                (`redis.Redis`) to be used for caching.
                This allows the object to communicate with a
                Redis server for caching operations.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from redis import Redis
        except ImportError:
            raise ImportError(
                "Could not import `redis` python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass a valid `redis.Redis` client.")
        self.redis = redis_
        self.ttl = ttl

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        # Read from a Redis HASH
        try:
            results = self.redis.hgetall(self._key(prompt, llm_string))
            return self._get_generations(results)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Redis lookup failed: {e}")
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._ensure_generation_type(return_val)
        key = self._key(prompt, llm_string)
        try:
            with self.redis.pipeline() as pipe:
                self._configure_pipeline_for_update(key, pipe, return_val, self.ttl)
                pipe.execute()
        except Exception as e:
            logger.error(f"Redis update failed: {e}")

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        try:
            asynchronous = kwargs.get("asynchronous", False)
            self.redis.flushdb(asynchronous=asynchronous, **kwargs)
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")


class AsyncRedisCache(_RedisCacheBase):
    """
    Cache that uses Redis as a backend. Allows to use an
    async `redis.asyncio.Redis` client.
    """

    def __init__(self, redis_: Any, *, ttl: Optional[int] = None):
        """
        Initialize an instance of AsyncRedisCache.

        This method initializes an object with Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of a Redis
        client class (`redis.asyncio.Redis`), allowing the object
        to interact with a Redis server for caching purposes.

        Parameters:
            redis_ (Any): An instance of a Redis client class
                (`redis.asyncio.Redis`) to be used for caching.
                This allows the object to communicate with a
                Redis server for caching operations.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from redis.asyncio import Redis
        except ImportError:
            raise ImportError(
                "Could not import `redis.asyncio` python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass a valid `redis.asyncio.Redis` client.")
        self.redis = redis_
        self.ttl = ttl

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        raise NotImplementedError(
            "This async Redis cache does not implement `lookup()` method. "
            "Consider using the async `alookup()` version."
        )

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string. Async version."""
        try:
            results = await self.redis.hgetall(self._key(prompt, llm_string))
            return self._get_generations(results)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Redis async lookup failed: {e}")
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        raise NotImplementedError(
            "This async Redis cache does not implement `update()` method. "
            "Consider using the async `aupdate()` version."
        )

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Update cache based on prompt and llm_string. Async version."""
        self._ensure_generation_type(return_val)
        key = self._key(prompt, llm_string)
        try:
            async with self.redis.pipeline() as pipe:
                self._configure_pipeline_for_update(key, pipe, return_val, self.ttl)
                await pipe.execute()  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Redis async update failed: {e}")

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        raise NotImplementedError(
            "This async Redis cache does not implement `clear()` method. "
            "Consider using the async `aclear()` version."
        )

    async def aclear(self, **kwargs: Any) -> None:
        """
        Clear cache. If `asynchronous` is True, flush asynchronously.
        Async version.
        """
        try:
            asynchronous = kwargs.get("asynchronous", False)
            await self.redis.flushdb(asynchronous=asynchronous, **kwargs)
        except Exception as e:
            logger.error(f"Redis async clear failed: {e}")


class RedisSemanticCache(BaseCache):
    """Cache that uses Redis as a vector-store backend."""

    # TODO - implement a TTL policy in Redis

    DEFAULT_SCHEMA = {
        "content_key": "prompt",
        "text": [
            {"name": "prompt"},
        ],
        "extra": [{"name": "return_val"}, {"name": "llm_string"}],
    }

    def __init__(
        self, redis_url: str, embedding: Embeddings, score_threshold: float = 0.2
    ):
        """Initialize by passing in the `init` GPTCache func

        Args:
            redis_url (str): URL to connect to Redis.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):

        Example:

        .. code-block:: python

            from langchain_community.globals import set_llm_cache

            from langchain_community.cache import RedisSemanticCache
            from langchain_community.embeddings import OpenAIEmbeddings

            set_llm_cache(RedisSemanticCache(
                redis_url="redis://localhost:6379",
                embedding=OpenAIEmbeddings()
            ))

        """
        self._cache_dict: Dict[str, RedisVectorstore] = {}
        self.redis_url = redis_url
        self.embedding = embedding
        self.score_threshold = score_threshold

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> RedisVectorstore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        try:
            self._cache_dict[index_name] = RedisVectorstore.from_existing_index(
                embedding=self.embedding,
                index_name=index_name,
                redis_url=self.redis_url,
                schema=cast(Dict, self.DEFAULT_SCHEMA),
            )
        except ValueError:
            redis = RedisVectorstore(
                embedding=self.embedding,
                index_name=index_name,
                redis_url=self.redis_url,
                index_schema=cast(Dict, self.DEFAULT_SCHEMA),
            )
            _embedding = self.embedding.embed_query(text="test")
            redis._create_index_if_not_exist(dim=len(_embedding))
            self._cache_dict[index_name] = redis

        return self._cache_dict[index_name]

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop_index(
                index_name=index_name, delete_documents=True, redis_url=self.redis_url
            )
            del self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
            distance_threshold=self.score_threshold,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "RedisSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)

        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])


class GPTCache(BaseCache):
    """Cache that uses GPTCache as a backend."""

    def __init__(
        self,
        init_func: Union[
            Callable[[Any, str], None], Callable[[Any], None], None
        ] = None,
    ):
        """Initialize by passing in init function (default: `None`).

        Args:
            init_func (Optional[Callable[[Any], None]]): init `GPTCache` function
            (default: `None`)

        Example:
        .. code-block:: python

            # Initialize GPTCache with a custom init function
            import gptcache
            from gptcache.processor.pre import get_prompt
            from gptcache.manager.factory import get_data_manager
            from langchain_community.globals import set_llm_cache

            # Avoid multiple caches using the same file,
            causing different llm model caches to affect each other

            def init_gptcache(cache_obj: gptcache.Cache, llm str):
                cache_obj.init(
                    pre_embedding_func=get_prompt,
                    data_manager=manager_factory(
                        manager="map",
                        data_dir=f"map_cache_{llm}"
                    ),
                )

            set_llm_cache(GPTCache(init_gptcache))

        """
        try:
            import gptcache  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import gptcache python package. "
                "Please install it with `pip install gptcache`."
            )

        self.init_gptcache_func: Union[
            Callable[[Any, str], None], Callable[[Any], None], None
        ] = init_func
        self.gptcache_dict: Dict[str, Any] = {}

    def _new_gptcache(self, llm_string: str) -> Any:
        """New gptcache object"""
        from gptcache import Cache
        from gptcache.manager.factory import get_data_manager
        from gptcache.processor.pre import get_prompt

        _gptcache = Cache()
        if self.init_gptcache_func is not None:
            sig = inspect.signature(self.init_gptcache_func)
            if len(sig.parameters) == 2:
                self.init_gptcache_func(_gptcache, llm_string)  # type: ignore[call-arg]
            else:
                self.init_gptcache_func(_gptcache)  # type: ignore[call-arg]
        else:
            _gptcache.init(
                pre_embedding_func=get_prompt,
                data_manager=get_data_manager(data_path=llm_string),
            )

        self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def _get_gptcache(self, llm_string: str) -> Any:
        """Get a cache object.

        When the corresponding llm model cache does not exist, it will be created."""
        _gptcache = self.gptcache_dict.get(llm_string, None)
        if not _gptcache:
            _gptcache = self._new_gptcache(llm_string)
        return _gptcache

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the cache data.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then retrieve the data from the cache based on the `prompt`.
        """
        from gptcache.adapter.api import get

        _gptcache = self._get_gptcache(llm_string)

        res = get(prompt, cache_obj=_gptcache)
        return _loads_generations(res) if res is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then store the `prompt` and `return_val` in the cache object.
        """
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "GPTCache only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
        from gptcache.adapter.api import put

        _gptcache = self._get_gptcache(llm_string)
        handled_data = _dumps_generations(return_val)
        put(prompt, handled_data, cache_obj=_gptcache)
        return None

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        from gptcache import Cache

        for gptcache_instance in self.gptcache_dict.values():
            gptcache_instance = cast(Cache, gptcache_instance)
            gptcache_instance.flush()

        self.gptcache_dict.clear()


def _ensure_cache_exists(cache_client: momento.CacheClient, cache_name: str) -> None:
    """Create cache if it doesn't exist.

    Raises:
        SdkException: Momento service or network error
        Exception: Unexpected response
    """
    from momento.responses import CreateCache

    create_cache_response = cache_client.create_cache(cache_name)
    if isinstance(create_cache_response, CreateCache.Success) or isinstance(
        create_cache_response, CreateCache.CacheAlreadyExists
    ):
        return None
    elif isinstance(create_cache_response, CreateCache.Error):
        raise create_cache_response.inner_exception
    else:
        raise Exception(f"Unexpected response cache creation: {create_cache_response}")


def _validate_ttl(ttl: Optional[timedelta]) -> None:
    if ttl is not None and ttl <= timedelta(seconds=0):
        raise ValueError(f"ttl must be positive but was {ttl}.")


class MomentoCache(BaseCache):
    """Cache that uses Momento as a backend. See https://gomomento.com/"""

    def __init__(
        self,
        cache_client: momento.CacheClient,
        cache_name: str,
        *,
        ttl: Optional[timedelta] = None,
        ensure_cache_exists: bool = True,
    ):
        """Instantiate a prompt cache using Momento as a backend.

        Note: to instantiate the cache client passed to MomentoCache,
        you must have a Momento account. See https://gomomento.com/.

        Args:
            cache_client (CacheClient): The Momento cache client.
            cache_name (str): The name of the cache to use to store the data.
            ttl (Optional[timedelta], optional): The time to live for the cache items.
                Defaults to None, ie use the client default TTL.
            ensure_cache_exists (bool, optional): Create the cache if it doesn't
                exist. Defaults to True.

        Raises:
            ImportError: Momento python package is not installed.
            TypeError: cache_client is not of type momento.CacheClientObject
            ValueError: ttl is non-null and non-negative
        """
        try:
            from momento import CacheClient
        except ImportError:
            raise ImportError(
                "Could not import momento python package. "
                "Please install it with `pip install momento`."
            )
        if not isinstance(cache_client, CacheClient):
            raise TypeError("cache_client must be a momento.CacheClient object.")
        _validate_ttl(ttl)
        if ensure_cache_exists:
            _ensure_cache_exists(cache_client, cache_name)

        self.cache_client = cache_client
        self.cache_name = cache_name
        self.ttl = ttl

    @classmethod
    def from_client_params(
        cls,
        cache_name: str,
        ttl: timedelta,
        *,
        configuration: Optional[momento.config.Configuration] = None,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,  # for backwards compatibility
        **kwargs: Any,
    ) -> MomentoCache:
        """Construct cache from CacheClient parameters."""
        try:
            from momento import CacheClient, Configurations, CredentialProvider
        except ImportError:
            raise ImportError(
                "Could not import momento python package. "
                "Please install it with `pip install momento`."
            )
        if configuration is None:
            configuration = Configurations.Laptop.v1()

        # Try checking `MOMENTO_AUTH_TOKEN` first for backwards compatibility
        try:
            api_key = auth_token or get_from_env("auth_token", "MOMENTO_AUTH_TOKEN")
        except ValueError:
            api_key = api_key or get_from_env("api_key", "MOMENTO_API_KEY")
        credentials = CredentialProvider.from_string(api_key)
        cache_client = CacheClient(configuration, credentials, default_ttl=ttl)
        return cls(cache_client, cache_name, ttl=ttl, **kwargs)

    def __key(self, prompt: str, llm_string: str) -> str:
        """Compute cache key from prompt and associated model and settings.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model version and settings.

        Returns:
            str: The cache key.
        """
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Lookup llm generations in cache by prompt and associated model and settings.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model version and settings.

        Raises:
            SdkException: Momento service or network error

        Returns:
            Optional[RETURN_VAL_TYPE]: A list of language model generations.
        """
        from momento.responses import CacheGet

        generations: RETURN_VAL_TYPE = []

        get_response = self.cache_client.get(
            self.cache_name, self.__key(prompt, llm_string)
        )
        if isinstance(get_response, CacheGet.Hit):
            value = get_response.value_string
            generations = _load_generations_from_json(value)
        elif isinstance(get_response, CacheGet.Miss):
            pass
        elif isinstance(get_response, CacheGet.Error):
            raise get_response.inner_exception
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Store llm generations in cache.

        Args:
            prompt (str): The prompt run through the language model.
            llm_string (str): The language model string.
            return_val (RETURN_VAL_TYPE): A list of language model generations.

        Raises:
            SdkException: Momento service or network error
            Exception: Unexpected response
        """
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "Momento only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
        key = self.__key(prompt, llm_string)
        value = _dump_generations_to_json(return_val)
        set_response = self.cache_client.set(self.cache_name, key, value, self.ttl)
        from momento.responses import CacheSet

        if isinstance(set_response, CacheSet.Success):
            pass
        elif isinstance(set_response, CacheSet.Error):
            raise set_response.inner_exception
        else:
            raise Exception(f"Unexpected response: {set_response}")

    def clear(self, **kwargs: Any) -> None:
        """Clear the cache.

        Raises:
            SdkException: Momento service or network error
        """
        from momento.responses import CacheFlush

        flush_response = self.cache_client.flush_cache(self.cache_name)
        if isinstance(flush_response, CacheFlush.Success):
            pass
        elif isinstance(flush_response, CacheFlush.Error):
            raise flush_response.inner_exception


CASSANDRA_CACHE_DEFAULT_TABLE_NAME = "langchain_llm_cache"
CASSANDRA_CACHE_DEFAULT_TTL_SECONDS = None


class CassandraCache(BaseCache):
    """
    Cache that uses Cassandra / Astra DB as a backend.

    Example:

        .. code-block:: python

            import cassio

            from langchain_community.cache import CassandraCache
            from langchain_core.globals import set_llm_cache

            cassio.init(auto=True)  # Requires env. variables, see CassIO docs

            set_llm_cache(CassandraCache())

    It uses a single Cassandra table.
    The lookup keys (which get to form the primary key) are:
        - prompt, a string
        - llm_string, a deterministic str representation of the model parameters.
          (needed to prevent same-prompt-different-model collisions)

    Args:
        session: an open Cassandra session.
            Leave unspecified to use the global cassio init (see below)
        keyspace: the keyspace to use for storing the cache.
            Leave unspecified to use the global cassio init (see below)
        table_name: name of the Cassandra table to use as cache
        ttl_seconds: time-to-live for cache entries
            (default: None, i.e. forever)
        setup_mode: a value in langchain_community.utilities.cassandra.SetupMode.
            Choose between SYNC, ASYNC and OFF - the latter if the Cassandra
            table is guaranteed to exist already, for a faster initialization.

    Note:
        The session and keyspace parameters, when left out (or passed as None),
        fall back to the globally-available cassio settings if any are available.
        In other words, if a previously-run 'cassio.init(...)' has been
        executed previously anywhere in the code, Cassandra-based objects
        need not specify the connection parameters at all.
    """

    def __init__(
        self,
        session: Optional[CassandraSession] = None,
        keyspace: Optional[str] = None,
        table_name: str = CASSANDRA_CACHE_DEFAULT_TABLE_NAME,
        ttl_seconds: Optional[int] = CASSANDRA_CACHE_DEFAULT_TTL_SECONDS,
        skip_provisioning: bool = False,
        setup_mode: CassandraSetupMode = CassandraSetupMode.SYNC,
    ):
        if skip_provisioning:
            warn_deprecated(
                "0.0.33",
                name="skip_provisioning",
                alternative=(
                    "setup_mode=langchain_community.utilities.cassandra.SetupMode.OFF"
                ),
                pending=True,
            )
        try:
            from cassio.table import ElasticCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install -U cassio`."
            )

        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds

        kwargs = {}
        if setup_mode == CassandraSetupMode.ASYNC:
            kwargs["async_setup"] = True

        self.kv_cache = ElasticCassandraTable(
            session=self.session,
            keyspace=self.keyspace,
            table=self.table_name,
            keys=["llm_string", "prompt"],
            primary_key_type=["TEXT", "TEXT"],
            ttl_seconds=self.ttl_seconds,
            skip_provisioning=skip_provisioning or setup_mode == CassandraSetupMode.OFF,
            **kwargs,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        item = self.kv_cache.get(
            llm_string=_hash(llm_string),
            prompt=_hash(prompt),
        )
        if item is not None:
            return _loads_generations(item["body_blob"])
        else:
            return None

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        item = await self.kv_cache.aget(
            llm_string=_hash(llm_string),
            prompt=_hash(prompt),
        )
        if item is not None:
            return _loads_generations(item["body_blob"])
        else:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        blob = _dumps_generations(return_val)
        self.kv_cache.put(
            llm_string=_hash(llm_string),
            prompt=_hash(prompt),
            body_blob=blob,
        )

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        blob = _dumps_generations(return_val)
        await self.kv_cache.aput(
            llm_string=_hash(llm_string),
            prompt=_hash(prompt),
            body_blob=blob,
        )

    def delete_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> None:
        """
        A wrapper around `delete` with the LLM being passed.
        In case the llm.invoke(prompt) calls have a `stop` param, you should
        pass it here
        """
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.delete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        return self.kv_cache.delete(
            llm_string=_hash(llm_string),
            prompt=_hash(prompt),
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. This is for all LLMs at once."""
        self.kv_cache.clear()

    async def aclear(self, **kwargs: Any) -> None:
        """Clear cache. This is for all LLMs at once."""
        await self.kv_cache.aclear()


# This constant is in fact a similarity - the 'distance' name is kept for compatibility:
CASSANDRA_SEMANTIC_CACHE_DEFAULT_DISTANCE_METRIC = "dot"
CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD = 0.85
CASSANDRA_SEMANTIC_CACHE_DEFAULT_TABLE_NAME = "langchain_llm_semantic_cache"
CASSANDRA_SEMANTIC_CACHE_DEFAULT_TTL_SECONDS = None
CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16


class CassandraSemanticCache(BaseCache):
    """
    Cache that uses Cassandra as a vector-store backend for semantic
    (i.e. similarity-based) lookup.

    Example:

        .. code-block:: python

            import cassio

            from langchain_community.cache import CassandraSemanticCache
            from langchain_core.globals import set_llm_cache

            cassio.init(auto=True)  # Requires env. variables, see CassIO docs

            my_embedding = ...

            set_llm_cache(CassandraSemanticCache(
                embedding=my_embedding,
                table_name="my_semantic_cache",
            ))

    It uses a single (vector) Cassandra table and stores, in principle,
    cached values from several LLMs, so the LLM's llm_string is part
    of the rows' primary keys.

    One can choose a similarity measure (default: "dot" for dot-product).
    Choosing another one ("cos", "l2") almost certainly requires threshold tuning.
    (which may be in order nevertheless, even if sticking to "dot").

    Args:
        session: an open Cassandra session.
            Leave unspecified to use the global cassio init (see below)
        keyspace: the keyspace to use for storing the cache.
            Leave unspecified to use the global cassio init (see below)
        embedding: Embedding provider for semantic
            encoding and search.
        table_name: name of the Cassandra (vector) table
            to use as cache. There is a default for "simple" usage, but
            remember to explicitly specify different tables if several embedding
            models coexist in your app (they cannot share one cache table).
        distance_metric: an alias for the 'similarity_measure' parameter (see below).
            As the "distance" terminology is misleading, please prefer
            'similarity_measure' for clarity.
        score_threshold: numeric value to use as
            cutoff for the similarity searches
        ttl_seconds: time-to-live for cache entries
            (default: None, i.e. forever)
        similarity_measure: which measure to adopt for similarity searches.
            Note: this parameter is aliased by 'distance_metric' - however,
            it is suggested to use the "similarity" terminology since this value
            is in fact a similarity (i.e. higher means closer).
            Note that at most one of the two parameters 'distance_metric'
            and 'similarity_measure' can be provided.
        setup_mode: a value in langchain_community.utilities.cassandra.SetupMode.
            Choose between SYNC, ASYNC and OFF - the latter if the Cassandra
            table is guaranteed to exist already, for a faster initialization.

    Note:
        The session and keyspace parameters, when left out (or passed as None),
        fall back to the globally-available cassio settings if any are available.
        In other words, if a previously-run 'cassio.init(...)' has been
        executed previously anywhere in the code, Cassandra-based objects
        need not specify the connection parameters at all.
    """

    def __init__(
        self,
        session: Optional[CassandraSession] = None,
        keyspace: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        table_name: str = CASSANDRA_SEMANTIC_CACHE_DEFAULT_TABLE_NAME,
        distance_metric: Optional[str] = None,
        score_threshold: float = CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD,
        ttl_seconds: Optional[int] = CASSANDRA_SEMANTIC_CACHE_DEFAULT_TTL_SECONDS,
        skip_provisioning: bool = False,
        similarity_measure: str = CASSANDRA_SEMANTIC_CACHE_DEFAULT_DISTANCE_METRIC,
        setup_mode: CassandraSetupMode = CassandraSetupMode.SYNC,
    ):
        if skip_provisioning:
            warn_deprecated(
                "0.0.33",
                name="skip_provisioning",
                alternative=(
                    "setup_mode=langchain_community.utilities.cassandra.SetupMode.OFF"
                ),
                pending=True,
            )
        try:
            from cassio.table import MetadataVectorCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install -U cassio`."
            )

        if not embedding:
            raise ValueError("Missing required parameter 'embedding'.")

        # detect if legacy 'distance_metric' parameter used
        if distance_metric is not None:
            # if passed, takes precedence over 'similarity_measure', but we warn:
            warn_deprecated(
                "0.0.33",
                name="distance_metric",
                alternative="similarity_measure",
                pending=True,
            )
            similarity_measure = distance_metric

        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding
        self.table_name = table_name
        self.similarity_measure = similarity_measure
        self.score_threshold = score_threshold
        self.ttl_seconds = ttl_seconds

        # The contract for this class has separate lookup and update:
        # in order to spare some embedding calculations we cache them between
        # the two calls.
        # Note: each instance of this class has its own `_get_embedding` with
        # its own lru.
        @lru_cache(maxsize=CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text: str) -> List[float]:
            return self.embedding.embed_query(text=text)

        self._get_embedding = _cache_embedding

        @_async_lru_cache(maxsize=CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        async def _acache_embedding(text: str) -> List[float]:
            return await self.embedding.aembed_query(text=text)

        self._aget_embedding = _acache_embedding

        embedding_dimension: Union[int, Awaitable[int], None] = None
        if setup_mode == CassandraSetupMode.ASYNC:
            embedding_dimension = self._aget_embedding_dimension()
        elif setup_mode == CassandraSetupMode.SYNC:
            embedding_dimension = self._get_embedding_dimension()

        kwargs = {}
        if setup_mode == CassandraSetupMode.ASYNC:
            kwargs["async_setup"] = True

        self.table = MetadataVectorCassandraTable(
            session=self.session,
            keyspace=self.keyspace,
            table=self.table_name,
            primary_key_type=["TEXT"],
            vector_dimension=embedding_dimension,
            ttl_seconds=self.ttl_seconds,
            metadata_indexing=("allow", {"_llm_string_hash"}),
            skip_provisioning=skip_provisioning or setup_mode == CassandraSetupMode.OFF,
            **kwargs,
        )

    def _get_embedding_dimension(self) -> int:
        return len(self._get_embedding(text="This is a sample sentence."))

    async def _aget_embedding_dimension(self) -> int:
        return len(await self._aget_embedding(text="This is a sample sentence."))

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        embedding_vector = self._get_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)
        body = _dumps_generations(return_val)
        metadata = {
            "_prompt": prompt,
            "_llm_string_hash": llm_string_hash,
        }
        row_id = f"{_hash(prompt)}-{llm_string_hash}"

        self.table.put(
            body_blob=body,
            vector=embedding_vector,
            row_id=row_id,
            metadata=metadata,
        )

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        embedding_vector = await self._aget_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)
        body = _dumps_generations(return_val)
        metadata = {
            "_prompt": prompt,
            "_llm_string_hash": llm_string_hash,
        }
        row_id = f"{_hash(prompt)}-{llm_string_hash}"

        await self.table.aput(
            body_blob=body,
            vector=embedding_vector,
            row_id=row_id,
            metadata=metadata,
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        hit_with_id = self.lookup_with_id(prompt, llm_string)
        if hit_with_id is not None:
            return hit_with_id[1]
        else:
            return None

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        hit_with_id = await self.alookup_with_id(prompt, llm_string)
        if hit_with_id is not None:
            return hit_with_id[1]
        else:
            return None

    def lookup_with_id(
        self, prompt: str, llm_string: str
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry)
        """
        prompt_embedding: List[float] = self._get_embedding(text=prompt)
        hits = list(
            self.table.metric_ann_search(
                vector=prompt_embedding,
                metadata={"_llm_string_hash": _hash(llm_string)},
                n=1,
                metric=self.similarity_measure,
                metric_threshold=self.score_threshold,
            )
        )
        if hits:
            hit = hits[0]
            generations = _loads_generations(hit["body_blob"])
            if generations is not None:
                # this protects against malformed cached items:
                return (
                    hit["row_id"],
                    generations,
                )
            else:
                return None
        else:
            return None

    async def alookup_with_id(
        self, prompt: str, llm_string: str
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry)
        """
        prompt_embedding: List[float] = await self._aget_embedding(text=prompt)
        hits = list(
            await self.table.ametric_ann_search(
                vector=prompt_embedding,
                metadata={"_llm_string_hash": _hash(llm_string)},
                n=1,
                metric=self.similarity_measure,
                metric_threshold=self.score_threshold,
            )
        )
        if hits:
            hit = hits[0]
            generations = _loads_generations(hit["body_blob"])
            if generations is not None:
                # this protects against malformed cached items:
                return (
                    hit["row_id"],
                    generations,
                )
            else:
                return None
        else:
            return None

    def lookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.lookup_with_id(prompt, llm_string=llm_string)

    async def alookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = (
            await aget_prompts(
                {**llm.dict(), **{"stop": stop}},
                [],
            )
        )[1]
        return await self.alookup_with_id(prompt, llm_string=llm_string)

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        self.table.delete(row_id=document_id)

    async def adelete_by_document_id(self, document_id: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        await self.table.adelete(row_id=document_id)

    def clear(self, **kwargs: Any) -> None:
        """Clear the *whole* semantic cache."""
        self.table.clear()

    async def aclear(self, **kwargs: Any) -> None:
        """Clear the *whole* semantic cache."""
        await self.table.aclear()


class FullMd5LLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_md5_llm_cache"
    id = Column(String, primary_key=True)
    prompt_md5 = Column(String, index=True)
    llm = Column(String, index=True)
    idx = Column(Integer, index=True)
    prompt = Column(String)
    response = Column(String)


class SQLAlchemyMd5Cache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(
        self, engine: Engine, cache_schema: Type[FullMd5LLMCache] = FullMd5LLMCache
    ):
        """Initialize by creating all tables."""
        self.engine = engine
        self.cache_schema = cache_schema
        self.cache_schema.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        rows = self._search_rows(prompt, llm_string)
        if rows:
            return [loads(row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        with Session(self.engine) as session, session.begin():
            self._delete_previous(session, prompt, llm_string)
            prompt_md5 = self.get_md5(prompt)
            items = [
                self.cache_schema(
                    id=str(uuid.uuid1()),
                    prompt=prompt,
                    prompt_md5=prompt_md5,
                    llm=llm_string,
                    response=dumps(gen),
                    idx=i,
                )
                for i, gen in enumerate(return_val)
            ]
            for item in items:
                session.merge(item)

    def _delete_previous(self, session: Session, prompt: str, llm_string: str) -> None:
        stmt = (
            delete(self.cache_schema)
            .where(self.cache_schema.prompt_md5 == self.get_md5(prompt))  # type: ignore
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.prompt == prompt)
        )
        session.execute(stmt)

    def _search_rows(self, prompt: str, llm_string: str) -> Sequence[Row]:
        prompt_pd5 = self.get_md5(prompt)
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt_md5 == prompt_pd5)  # type: ignore
            .where(self.cache_schema.llm == llm_string)
            .where(self.cache_schema.prompt == prompt)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            return session.execute(stmt).fetchall()

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        with Session(self.engine) as session:
            session.execute(self.cache_schema.delete())

    @staticmethod
    def get_md5(input_string: str) -> str:
        return hashlib.md5(input_string.encode()).hexdigest()


ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_cache"


@deprecated(
    since="0.0.28",
    removal="1.0",
    alternative_import="langchain_astradb.AstraDBCache",
)
class AstraDBCache(BaseCache):
    @staticmethod
    def _make_id(prompt: str, llm_string: str) -> str:
        return f"{_hash(prompt)}#{_hash(llm_string)}"

    def __init__(
        self,
        *,
        collection_name: str = ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        pre_delete_collection: bool = False,
        setup_mode: AstraSetupMode = AstraSetupMode.SYNC,
    ):
        """
        Cache that uses Astra DB as a backend.

        It uses a single collection as a kv store
        The lookup keys, combined in the _id of the documents, are:
            - prompt, a string
            - llm_string, a deterministic str representation of the model parameters.
              (needed to prevent same-prompt-different-model collisions)

        Args:
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as `https://<DB-ID>-us-east1.apps.astra.datastax.com`.
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = self.collection.find_one(
            filter={
                "_id": doc_id,
            },
            projection={
                "body_blob": 1,
            },
        )["data"]["document"]
        return _loads_generations(item["body_blob"]) if item is not None else None

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = (
            await self.async_collection.find_one(
                filter={
                    "_id": doc_id,
                },
                projection={
                    "body_blob": 1,
                },
            )
        )["data"]["document"]
        return _loads_generations(item["body_blob"]) if item is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        self.collection.upsert(
            {
                "_id": doc_id,
                "body_blob": blob,
            },
        )

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        await self.async_collection.upsert(
            {
                "_id": doc_id,
                "body_blob": blob,
            },
        )

    def delete_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> None:
        """
        A wrapper around `delete` with the LLM being passed.
        In case the llm.invoke(prompt) calls have a `stop` param, you should
        pass it here
        """
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.delete(prompt, llm_string=llm_string)

    async def adelete_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> None:
        """
        A wrapper around `adelete` with the LLM being passed.
        In case the llm.invoke(prompt) calls have a `stop` param, you should
        pass it here
        """
        llm_string = (
            await aget_prompts(
                {**llm.dict(), **{"stop": stop}},
                [],
            )
        )[1]
        return await self.adelete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        self.collection.delete_one(doc_id)

    async def adelete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        await self.async_collection.delete_one(doc_id)

    def clear(self, **kwargs: Any) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.clear()

    async def aclear(self, **kwargs: Any) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.clear()


ASTRA_DB_SEMANTIC_CACHE_DEFAULT_THRESHOLD = 0.85
ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_semantic_cache"
ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16


_unset = ["unset"]


class _CachedAwaitable:
    """Caches the result of an awaitable so it can be awaited multiple times"""

    def __init__(self, awaitable: Awaitable[Any]):
        self.awaitable = awaitable
        self.result = _unset

    def __await__(self) -> Generator:
        if self.result is _unset:
            self.result = yield from self.awaitable.__await__()
        return self.result  # type: ignore


def _reawaitable(func: Callable) -> Callable:
    """Makes an async function result awaitable multiple times"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _CachedAwaitable:
        return _CachedAwaitable(func(*args, **kwargs))

    return wrapper


def _async_lru_cache(maxsize: int = 128, typed: bool = False) -> Callable:
    """Least-recently-used async cache decorator.
    Equivalent to functools.lru_cache for async functions"""

    def decorating_function(user_function: Callable) -> Callable:
        return lru_cache(maxsize, typed)(_reawaitable(user_function))

    return decorating_function


@deprecated(
    since="0.0.28",
    removal="1.0",
    alternative_import="langchain_astradb.AstraDBSemanticCache",
)
class AstraDBSemanticCache(BaseCache):
    def __init__(
        self,
        *,
        collection_name: str = ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        setup_mode: AstraSetupMode = AstraSetupMode.SYNC,
        pre_delete_collection: bool = False,
        embedding: Embeddings,
        metric: Optional[str] = None,
        similarity_threshold: float = ASTRA_DB_SEMANTIC_CACHE_DEFAULT_THRESHOLD,
    ):
        """
        Cache that uses Astra DB as a vector-store backend for semantic
        (i.e. similarity-based) lookup.

        It uses a single (vector) collection and can store
        cached values from several LLMs, so the LLM's 'llm_string' is stored
        in the document metadata.

        You can choose the preferred similarity (or use the API default).
        The default score threshold is tuned to the default metric.
        Tune it carefully yourself if switching to another distance metric.

        Args:
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as `https://<DB-ID>-us-east1.apps.astra.datastax.com`.
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
            embedding: Embedding provider for semantic encoding and search.
            metric: the function to use for evaluating similarity of text embeddings.
                Defaults to 'cosine' (alternatives: 'euclidean', 'dot_product')
            similarity_threshold: the minimum similarity for accepting a
                (semantic-search) match.
        """
        self.embedding = embedding
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name

        # The contract for this class has separate lookup and update:
        # in order to spare some embedding calculations we cache them between
        # the two calls.
        # Note: each instance of this class has its own `_get_embedding` with
        # its own lru.
        @lru_cache(maxsize=ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text: str) -> List[float]:
            return self.embedding.embed_query(text=text)

        self._get_embedding = _cache_embedding

        @_async_lru_cache(maxsize=ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        async def _acache_embedding(text: str) -> List[float]:
            return await self.embedding.aembed_query(text=text)

        self._aget_embedding = _acache_embedding

        embedding_dimension: Union[int, Awaitable[int], None] = None
        if setup_mode == AstraSetupMode.ASYNC:
            embedding_dimension = self._aget_embedding_dimension()
        elif setup_mode == AstraSetupMode.SYNC:
            embedding_dimension = self._get_embedding_dimension()

        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            embedding_dimension=embedding_dimension,
            metric=metric,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    def _get_embedding_dimension(self) -> int:
        return len(self._get_embedding(text="This is a sample sentence."))

    async def _aget_embedding_dimension(self) -> int:
        return len(await self._aget_embedding(text="This is a sample sentence."))

    @staticmethod
    def _make_id(prompt: str, llm_string: str) -> str:
        return f"{_hash(prompt)}#{_hash(llm_string)}"

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        llm_string_hash = _hash(llm_string)
        embedding_vector = self._get_embedding(text=prompt)
        body = _dumps_generations(return_val)
        #
        self.collection.upsert(
            {
                "_id": doc_id,
                "body_blob": body,
                "llm_string_hash": llm_string_hash,
                "$vector": embedding_vector,
            }
        )

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        llm_string_hash = _hash(llm_string)
        embedding_vector = await self._aget_embedding(text=prompt)
        body = _dumps_generations(return_val)
        #
        await self.async_collection.upsert(
            {
                "_id": doc_id,
                "body_blob": body,
                "llm_string_hash": llm_string_hash,
                "$vector": embedding_vector,
            }
        )

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        hit_with_id = self.lookup_with_id(prompt, llm_string)
        if hit_with_id is not None:
            return hit_with_id[1]
        else:
            return None

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        hit_with_id = await self.alookup_with_id(prompt, llm_string)
        if hit_with_id is not None:
            return hit_with_id[1]
        else:
            return None

    def lookup_with_id(
        self, prompt: str, llm_string: str
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry) for the top hit
        """
        self.astra_env.ensure_db_setup()
        prompt_embedding: List[float] = self._get_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)

        hit = self.collection.vector_find_one(
            vector=prompt_embedding,
            filter={
                "llm_string_hash": llm_string_hash,
            },
            fields=["body_blob", "_id"],
            include_similarity=True,
        )

        if hit is None or hit["$similarity"] < self.similarity_threshold:
            return None
        else:
            generations = _loads_generations(hit["body_blob"])
            if generations is not None:
                # this protects against malformed cached items:
                return hit["_id"], generations
            else:
                return None

    async def alookup_with_id(
        self, prompt: str, llm_string: str
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry) for the top hit
        """
        await self.astra_env.aensure_db_setup()
        prompt_embedding: List[float] = await self._aget_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)

        hit = await self.async_collection.vector_find_one(
            vector=prompt_embedding,
            filter={
                "llm_string_hash": llm_string_hash,
            },
            fields=["body_blob", "_id"],
            include_similarity=True,
        )

        if hit is None or hit["$similarity"] < self.similarity_threshold:
            return None
        else:
            generations = _loads_generations(hit["body_blob"])
            if generations is not None:
                # this protects against malformed cached items:
                return hit["_id"], generations
            else:
                return None

    def lookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = get_prompts(
            {**llm.dict(), **{"stop": stop}},
            [],
        )[1]
        return self.lookup_with_id(prompt, llm_string=llm_string)

    async def alookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: Optional[List[str]] = None
    ) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = (
            await aget_prompts(
                {**llm.dict(), **{"stop": stop}},
                [],
            )
        )[1]
        return await self.alookup_with_id(prompt, llm_string=llm_string)

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        self.astra_env.ensure_db_setup()
        self.collection.delete_one(document_id)

    async def adelete_by_document_id(self, document_id: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_one(document_id)

    def clear(self, **kwargs: Any) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.clear()

    async def aclear(self, **kwargs: Any) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.clear()


class AzureCosmosDBSemanticCache(BaseCache):
    """Cache that uses Cosmos DB Mongo vCore vector-store backend"""

    DEFAULT_DATABASE_NAME = "CosmosMongoVCoreCacheDB"
    DEFAULT_COLLECTION_NAME = "CosmosMongoVCoreCacheColl"

    def __init__(
        self,
        cosmosdb_connection_string: str,
        database_name: str,
        collection_name: str,
        embedding: Embeddings,
        *,
        cosmosdb_client: Optional[Any] = None,
        num_lists: int = 100,
        similarity: CosmosDBSimilarityType = CosmosDBSimilarityType.COS,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        dimensions: int = 1536,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 40,
        score_threshold: Optional[float] = None,
        application_name: str = "LangChain-CDBMongoVCore-SemanticCache-Python",
    ):
        """
        Args:
            cosmosdb_connection_string: Cosmos DB Mongo vCore connection string
            cosmosdb_client: Cosmos DB Mongo vCore client
            embedding (Embedding): Embedding provider for semantic encoding and search.
            database_name: Database name for the CosmosDBMongoVCoreSemanticCache
            collection_name: Collection name for the CosmosDBMongoVCoreSemanticCache
            num_lists: This integer is the number of clusters that the
                inverted file (IVF) index uses to group the vector data.
                We recommend that numLists is set to documentCount/1000
                for up to 1 million documents and to sqrt(documentCount)
                for more than 1 million documents.
                Using a numLists value of 1 is akin to performing
                brute-force search, which has limited performance
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000
            similarity: Similarity metric to use with the IVF index.

                Possible options are:
                    - CosmosDBSimilarityType.COS (cosine distance),
                    - CosmosDBSimilarityType.L2 (Euclidean distance), and
                    - CosmosDBSimilarityType.IP (inner product).
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw: available as a preview feature only,
                                   to enable visit https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/preview-features
            m: The max number of connections per layer (16 by default, minimum
               value is 2, maximum value is 100). Higher m is suitable for datasets
               with high dimensionality and/or high accuracy requirements.
            ef_construction: the size of the dynamic candidate list for constructing
                            the graph (64 by default, minimum value is 4, maximum
                            value is 1000). Higher ef_construction will result in
                            better index quality and higher accuracy, but it will
                            also increase the time required to build the index.
                            ef_construction has to be at least 2 * m
            ef_search: The size of the dynamic candidate list for search
                       (40 by default). A higher value provides better
                       recall at the cost of speed.
            score_threshold: Maximum score used to filter the vector search documents.
            application_name: Application name for the client for tracking and logging
        """

        self._validate_enum_value(similarity, CosmosDBSimilarityType)
        self._validate_enum_value(kind, CosmosDBVectorSearchType)

        if not cosmosdb_connection_string:
            raise ValueError(" CosmosDB connection string can be empty.")

        self.cosmosdb_connection_string = cosmosdb_connection_string
        self.cosmosdb_client = cosmosdb_client
        self.embedding = embedding
        self.database_name = database_name or self.DEFAULT_DATABASE_NAME
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.num_lists = num_lists
        self.dimensions = dimensions
        self.similarity = similarity
        self.kind = kind
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.score_threshold = score_threshold
        self._cache_dict: Dict[str, AzureCosmosDBVectorSearch] = {}
        self.application_name = application_name

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> AzureCosmosDBVectorSearch:
        index_name = self._index_name(llm_string)

        namespace = self.database_name + "." + self.collection_name

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        if self.cosmosdb_client:
            collection = self.cosmosdb_client[self.database_name][self.collection_name]
            self._cache_dict[index_name] = AzureCosmosDBVectorSearch(
                collection=collection,
                embedding=self.embedding,
                index_name=index_name,
            )
        else:
            self._cache_dict[index_name] = (
                AzureCosmosDBVectorSearch.from_connection_string(
                    connection_string=self.cosmosdb_connection_string,
                    namespace=namespace,
                    embedding=self.embedding,
                    index_name=index_name,
                    application_name=self.application_name,
                )
            )

        # create index for the vectorstore
        vectorstore = self._cache_dict[index_name]
        if not vectorstore.index_exists():
            vectorstore.create_index(
                self.num_lists,
                self.dimensions,
                self.similarity,
                self.kind,
                self.m,
                self.ef_construction,
            )

        return vectorstore

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
            kind=self.kind,
            ef_search=self.ef_search,
            score_threshold=self.score_threshold,  # type: ignore[arg-type]
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "CosmosDBMongoVCoreSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )

        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].get_collection().delete_many({})

    @staticmethod
    def _validate_enum_value(value: Any, enum_type: Type[Enum]) -> None:
        if not isinstance(value, enum_type):
            raise ValueError(f"Invalid enum value: {value}. Expected {enum_type}.")


class AzureCosmosDBNoSqlSemanticCache(BaseCache):
    """Cache that uses Cosmos DB NoSQL backend"""

    def __init__(
        self,
        embedding: Embeddings,
        cosmos_client: CosmosClient,
        database_name: str = "CosmosNoSqlCacheDB",
        container_name: str = "CosmosNoSqlCacheContainer",
        *,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
    ):
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.container_name = container_name
        self.embedding = embedding
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_container_properties = cosmos_container_properties
        self.cosmos_database_properties = cosmos_database_properties
        self._cache_dict: Dict[str, AzureCosmosDBNoSqlVectorSearch] = {}

    def _cache_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> AzureCosmosDBNoSqlVectorSearch:
        cache_name = self._cache_name(llm_string)

        # return vectorstore client for the specific llm string
        if cache_name in self._cache_dict:
            return self._cache_dict[cache_name]

        # create new vectorstore client to create the cache
        if self.cosmos_client:
            self._cache_dict[cache_name] = AzureCosmosDBNoSqlVectorSearch(
                cosmos_client=self.cosmos_client,
                embedding=self.embedding,
                vector_embedding_policy=self.vector_embedding_policy,
                indexing_policy=self.indexing_policy,
                cosmos_container_properties=self.cosmos_container_properties,
                cosmos_database_properties=self.cosmos_database_properties,
                database_name=self.database_name,
                container_name=self.container_name,
            )

        return self._cache_dict[cache_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )

                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "CosmosDBNoSqlSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        cache_name = self._cache_name(llm_string=kwargs["llm-string"])
        if cache_name in self._cache_dict:
            container = self._cache_dict["cache_name"].get_container()
            for item in container.read_all_items():
                container.delete_item(item)


class OpenSearchSemanticCache(BaseCache):
    """Cache that uses OpenSearch vector store backend"""

    def __init__(
        self,
        opensearch_url: str,
        embedding: Embeddings,
        score_threshold: float = 0.2,
        **kwargs: Any,
    ):
        """
        Args:
            opensearch_url (str): URL to connect to OpenSearch.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):
        Example:
        .. code-block:: python
            import langchain
            from langchain.cache import OpenSearchSemanticCache
            from langchain.embeddings import OpenAIEmbeddings
            langchain.llm_cache = OpenSearchSemanticCache(
                opensearch_url="http//localhost:9200",
                embedding=OpenAIEmbeddings()
            )
        """
        self._cache_dict: Dict[str, OpenSearchVectorStore] = {}
        self.opensearch_url = opensearch_url
        self.embedding = embedding
        self.score_threshold = score_threshold
        self.connection_kwargs = kwargs

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache_{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> OpenSearchVectorStore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        self._cache_dict[index_name] = OpenSearchVectorStore(
            opensearch_url=self.opensearch_url,
            index_name=index_name,
            embedding_function=self.embedding,
            **self.connection_kwargs,
        )

        # create index for the vectorstore
        vectorstore = self._cache_dict[index_name]
        if not vectorstore.index_exists():
            _embedding = self.embedding.embed_query(text="test")
            vectorstore.create_index(len(_embedding), index_name)
        return vectorstore

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
            score_threshold=self.score_threshold,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )

                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "OpenSearchSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].delete_index(index_name=index_name)
            del self._cache_dict[index_name]


class SingleStoreDBSemanticCache(BaseCache):
    """Cache that uses SingleStore DB as a backend"""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        cache_table_prefix: str = "cache_",
        search_threshold: float = 0.2,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:
            embedding (Embeddings): A text embedding model.
            cache_table_prefix (str, optional): Prefix for the cache table name.
                Defaults to "cache_".
            search_threshold (float, optional): The minimum similarity score for
                a search result to be considered a match. Defaults to 0.2.

            Following arguments pertrain to the SingleStoreDB vector store:

            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to DOT_PRODUCT.
                Available options are:
                - DOT_PRODUCT: Computes the scalar product of two vectors.
                    This is the default behavior
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This metric is not
                    compatible with the WEIGHTED_SUM search strategy.

            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".
            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".
            vector_field (str, optional): Specifies the field to store the vector.
                Defaults to "vector".
            id_field (str, optional): Specifies the field to store the id.
                Defaults to "id".

            use_vector_index (bool, optional): Toggles the use of a vector index.
                Works only with SingleStoreDB 8.5 or later. Defaults to False.
                If set to True, vector_size parameter is required to be set to
                a proper value.

            vector_index_name (str, optional): Specifies the name of the vector index.
                Defaults to empty. Will be ignored if use_vector_index is set to False.

            vector_index_options (dict, optional): Specifies the options for
                the vector index. Defaults to {}.
                Will be ignored if use_vector_index is set to False. The options are:
                index_type (str, optional): Specifies the type of the index.
                    Defaults to IVF_PQFS.
                For more options, please refer to the SingleStoreDB documentation:
                https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/

            vector_size (int, optional): Specifies the size of the vector.
                Defaults to 1536. Required if use_vector_index is set to True.
                Should be set to the same value as the size of the vectors
                stored in the vector_field.

            Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.
            local_infile (bool, optional): Allows local file uploads.
            charset (str, optional): Specifies the character set for string values.
            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.
            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.
            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.
            ssl_cipher (str, optional): Sets the SSL cipher list.
            ssl_disabled (bool, optional): Disables SSL usage.
            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.
            ssl_verify_identity (bool, optional): Verifies the server's identity.
            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.
            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.
            autocommit (bool, optional): Enables autocommits.
            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.
            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.

        Examples:
            Basic Usage:

            .. code-block:: python

                import langchain
                from langchain.cache import SingleStoreDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = SingleStoreDBSemanticCache(
                    embedding=OpenAIEmbeddings(),
                    host="https://user:password@127.0.0.1:3306/database"
                )

            Advanced Usage:

            .. code-block:: python

                import langchain
                from langchain.cache import SingleStoreDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = = SingleStoreDBSemanticCache(
                    embeddings=OpenAIEmbeddings(),
                    use_vector_index=True,
                    host="127.0.0.1",
                    port=3306,
                    user="user",
                    password="password",
                    database="db",
                    table_name="my_custom_table",
                    pool_size=10,
                    timeout=60,
                )
        """

        self._cache_dict: Dict[str, SingleStoreDB] = {}
        self.embedding = embedding
        self.cache_table_prefix = cache_table_prefix
        self.search_threshold = search_threshold

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"{self.cache_table_prefix}{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> SingleStoreDB:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name not in self._cache_dict:
            self._cache_dict[index_name] = SingleStoreDB(
                embedding=self.embedding,
                table_name=index_name,
                **self.connection_kwargs,
            )
        return self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search_with_score(
            query=prompt,
            k=1,
        )
        if results:
            for document_score in results:
                if (
                    document_score[1] > self.search_threshold
                    and llm_cache.distance_strategy == DistanceStrategy.DOT_PRODUCT
                ) or (
                    document_score[1] < self.search_threshold
                    and llm_cache.distance_strategy
                    == DistanceStrategy.EUCLIDEAN_DISTANCE
                ):
                    generations.extend(loads(document_score[0].metadata["return_val"]))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "SingleStoreDBSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop()
            del self._cache_dict[index_name]


class MemcachedCache(BaseCache):
    """Cache that uses Memcached backend through pymemcache client lib"""

    def __init__(self, client_: Any):
        """
        Initialize an instance of MemcachedCache.

        Args:
            client_ (str): An instance of any of pymemcache's Clients
                (Client, PooledClient, HashClient)
        Example:
        .. code-block:: python
            ifrom langchain.globals import set_llm_cache
            from langchain_openai import OpenAI

            from langchain_community.cache import MemcachedCache
            from pymemcache.client.base import Client

            llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)
            set_llm_cache(MemcachedCache(Client('localhost')))

            # The first time, it is not yet in cache, so it should take longer
            llm.invoke("Which city is the most crowded city in the USA?")

            # The second time it is, so it goes faster
            llm.invoke("Which city is the most crowded city in the USA?")
        """

        try:
            from pymemcache.client import (
                Client,
                HashClient,
                PooledClient,
                RetryingClient,
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import pymemcache python package. "
                "Please install it with `pip install -U pymemcache`."
            )

        if not (
            isinstance(client_, Client)
            or isinstance(client_, PooledClient)
            or isinstance(client_, HashClient)
            or isinstance(client_, RetryingClient)
        ):
            raise ValueError("Please pass a valid pymemcached client")

        self.client = client_

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        key = _hash(prompt + llm_string)
        try:
            result = self.client.get(key)
        except pymemcache.MemcacheError:
            return None

        return _loads_generations(result) if result is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        key = _hash(prompt + llm_string)

        # Validate input is made of standard LLM generations
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "Memcached only supports caching of normal LLM generations, "
                    + f"got {type(gen)}"
                )

        # Deserialize return_val into string and update cache
        value = _dumps_generations(return_val)
        self.client.set(key, value)

    def clear(self, **kwargs: Any) -> None:
        """
        Clear the entire cache. Takes optional kwargs:

        delay: optional int, the number of seconds to wait before flushing,
                or zero to flush immediately (the default). NON-BLOCKING, returns
                immediately.
        noreply: optional bool, True to not wait for the reply (defaults to
                client.default_noreply).
        """
        delay = kwargs.get("delay", 0)
        noreply = kwargs.get("noreply", None)

        self.client.flush_all(delay, noreply)
