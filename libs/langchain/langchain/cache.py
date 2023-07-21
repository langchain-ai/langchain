"""Beta Feature: base interface for cache."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

from langchain.utils import get_from_env

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.embeddings.base import Embeddings
from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import ChatGeneration, Generation
from langchain.vectorstores.redis import Redis as RedisVectorstore

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    import momento

RETURN_VAL_TYPE = Sequence[Generation]


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: Json representing a list of generations.
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
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


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


class RedisCache(BaseCache):
    """Cache that uses Redis as a backend."""

    # TODO - implement a TTL policy in Redis

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

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        # Read from a Redis HASH
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
                    "RedisCache only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
            if isinstance(gen, ChatGeneration):
                warnings.warn(
                    "NOTE: Generation has not been cached. RedisCache does not"
                    " support caching ChatModel outputs."
                )
                return
        # Write to a Redis HASH
        key = self._key(prompt, llm_string)
        self.redis.hset(
            key,
            mapping={
                str(idx): generation.text for idx, generation in enumerate(return_val)
            },
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        asynchronous = kwargs.get("asynchronous", False)
        self.redis.flushdb(asynchronous=asynchronous, **kwargs)


class RedisSemanticCache(BaseCache):
    """Cache that uses Redis as a vector-store backend."""

    # TODO - implement a TTL policy in Redis

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

            import langchain

            from langchain.cache import RedisSemanticCache
            from langchain.embeddings import OpenAIEmbeddings

            langchain.llm_cache = RedisSemanticCache(
                redis_url="redis://localhost:6379",
                embedding=OpenAIEmbeddings()
            )

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
            )
        except ValueError:
            redis = RedisVectorstore(
                embedding_function=self.embedding.embed_query,
                index_name=index_name,
                redis_url=self.redis_url,
            )
            _embedding = self.embedding.embed_query(text="test")
            redis._create_index(dim=len(_embedding))
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
        generations = []
        # Read from a Hash
        results = llm_cache.similarity_search_limit_score(
            query=prompt,
            k=1,
            score_threshold=self.score_threshold,
        )
        if results:
            for document in results:
                for text in document.metadata["return_val"]:
                    generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "RedisSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
            if isinstance(gen, ChatGeneration):
                warnings.warn(
                    "NOTE: Generation has not been cached. RedisSentimentCache does not"
                    " support caching ChatModel outputs."
                )
                return
        llm_cache = self._get_llm_cache(llm_string)
        # Write to vectorstore
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": [generation.text for generation in return_val],
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

            langchain.llm_cache = GPTCache(init_gptcache)

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

        return self.gptcache_dict.get(llm_string, self._new_gptcache(llm_string))

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up the cache data.
        First, retrieve the corresponding cache object using the `llm_string` parameter,
        and then retrieve the data from the cache based on the `prompt`.
        """
        from gptcache.adapter.api import get

        _gptcache = self.gptcache_dict.get(llm_string, None)
        if _gptcache is None:
            return None
        res = get(prompt, cache_obj=_gptcache)
        if res:
            return [
                Generation(**generation_dict) for generation_dict in json.loads(res)
            ]
        return None

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
        handled_data = json.dumps([generation.dict() for generation in return_val])
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
        auth_token: Optional[str] = None,
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
        auth_token = auth_token or get_from_env("auth_token", "MOMENTO_AUTH_TOKEN")
        credentials = CredentialProvider.from_string(auth_token)
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
