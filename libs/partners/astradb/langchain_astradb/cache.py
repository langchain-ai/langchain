import hashlib
import json
from functools import lru_cache, wraps
from typing import Any, Awaitable, Callable, Generator, List, Optional, Tuple, Union

from astrapy.db import AstraDB, AsyncAstraDB, logger
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from langchain_core.language_models.llms import aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_astradb.utils.astradb import SetupMode, _AstraDBCollectionEnvironment

ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_cache"
ASTRA_DB_SEMANTIC_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_semantic_cache"
ASTRA_DB_SEMANTIC_CACHE_DEFAULT_THRESHOLD = 0.85
ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


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
        setup_mode: SetupMode = SetupMode.SYNC,
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
        In case the llm(prompt) calls have a `stop` param, you should pass it here
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
        In case the llm(prompt) calls have a `stop` param, you should pass it here
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


_unset = ["unset"]


class _CachedAwaitable:
    """Caches the result of an awaitable so it can be awaited multiple times"""

    def __init__(self, awaitable: Awaitable[Any]):
        self.awaitable = awaitable
        self.result = _unset

    def __await__(self) -> Generator:
        if self.result is _unset:
            self.result = yield from self.awaitable.__await__()
        return self.result


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


class AstraDBSemanticCache(BaseCache):
    def __init__(
        self,
        *,
        collection_name: str = ASTRA_DB_SEMANTIC_CACHE_DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
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
        if setup_mode == SetupMode.ASYNC:
            embedding_dimension = self._aget_embedding_dimension()
        elif setup_mode == SetupMode.SYNC:
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
