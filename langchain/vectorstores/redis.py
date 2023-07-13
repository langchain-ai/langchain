"""Wrapper around Redis vector database."""

from __future__ import annotations

import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import numpy as np
from pydantic import root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.client import Redis as RedisType
    from redis.commands.search.query import Query


# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20400},
    {"name": "searchlight", "ver": 20400},
]

# distance mmetrics
REDIS_DISTANCE_METRICS = Literal["COSINE", "IP", "L2"]


def _check_redis_module_exist(client: RedisType, required_modules: List[dict]) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in required_modules:
        if module["name"] in installed_modules and int(
            installed_modules[module["name"]][b"ver"]
        ) >= int(module["ver"]):
            return
    # otherwise raise error
    error_message = (
        "Redis cannot be used as a vector database without RediSearch >=2.4"
        "Please head to https://redis.io/docs/stack/search/quick_start/"
        "to know more about installing the RediSearch module within Redis Stack."
    )
    logging.error(error_message)
    raise ValueError(error_message)


def _check_index_exists(client: RedisType, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except:  # noqa: E722
        logger.info("Index does not exist")
        return False
    logger.info("Index already exists")
    return True


def _redis_key(prefix: str) -> str:
    """Redis key schema for a given prefix."""
    return f"{prefix}:{uuid.uuid4().hex}"


def _redis_prefix(index_name: str) -> str:
    """Redis key prefix for a given index."""
    return f"doc:{index_name}"


def _default_relevance_score(val: float) -> float:
    return 1 - val


class Redis(VectorStore):
    """Wrapper around Redis vector database.

    To use, you should have the ``redis`` python package installed.

    Example:
        .. code-block:: python

            from langchain.vectorstores import Redis
            from langchain.embeddings import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vectorstore = Redis(
                redis_url="redis://username:password@localhost:6379"
                index_name="my-index",
                embedding_function=embeddings.embed_query,
            )
    """

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding_function: Callable,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        distance_metric: REDIS_DISTANCE_METRICS = "COSINE",
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis>=4.1.0`."
            )

        self.embedding_function = embedding_function
        self.index_name = index_name
        try:
            # connect to redis from url
            redis_client = redis.from_url(redis_url, **kwargs)
            # check if redis has redisearch module installed
            _check_redis_module_exist(redis_client, REDIS_REQUIRED_MODULES)
        except ValueError as e:
            raise ValueError(f"Redis failed to connect: {e}")

        self.client = redis_client
        self.content_key = content_key
        self.metadata_key = metadata_key
        self.vector_key = vector_key
        self.distance_metric = distance_metric
        self.relevance_score_fn = relevance_score_fn

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.relevance_score_fn:
            return self.relevance_score_fn

        if self.distance_metric == "COSINE":
            return self._cosine_relevance_score_fn
        elif self.distance_metric == "IP":
            return self._max_inner_product_relevance_score_fn
        elif self.distance_metric == "L2":
            return self._euclidean_relevance_score_fn
        else:
            return _default_relevance_score

    def _create_index(self, dim: int = 1536) -> None:
        try:
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Check if index exists
        if not _check_index_exists(self.client, self.index_name):
            # Define schema
            schema = (
                TextField(name=self.content_key),
                TextField(name=self.metadata_key),
                VectorField(
                    self.vector_key,
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dim,
                        "DISTANCE_METRIC": self.distance_metric,
                    },
                ),
            )
            prefix = _redis_prefix(self.index_name)

            # Create Redis Index
            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.
            keys (List[str]) or ids (List[str]): Identifiers of entries.
                Defaults to None.
            batch_size (int, optional): Batch size to use for writes. Defaults to 1000.

        Returns:
            List[str]: List of ids added to the vectorstore
        """
        ids = []
        prefix = _redis_prefix(self.index_name)

        # Get keys or ids from kwargs
        # Other vectorstores use ids
        keys_or_ids = kwargs.get("keys", kwargs.get("ids"))

        # Write data to redis
        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided values by default or fallback
            key = keys_or_ids[i] if keys_or_ids else _redis_key(prefix)
            metadata = metadatas[i] if metadatas else {}
            embedding = embeddings[i] if embeddings else self.embedding_function(text)
            pipeline.hset(
                key,
                mapping={
                    self.content_key: text,
                    self.vector_key: np.array(embedding, dtype=np.float32).tobytes(),
                    self.metadata_key: json.dumps(metadata),
                },
            )
            ids.append(key)

            # Write batch
            if i % batch_size == 0:
                pipeline.execute()

        # Cleanup final batch
        pipeline.execute()
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_limit_score(
        self, query: str, k: int = 4, score_threshold: float = 0.2, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_threshold (float): The minimum matching score required for a document
            to be considered a match. Defaults to 0.2.
            Because the similarity calculation algorithm is based on cosine similarity,
            the smaller the angle, the higher the similarity.

        Returns:
            List[Document]: A list of documents that are most similar to the query text,
            including the match score for each document.

        Note:
            If there are no documents that satisfy the score_threshold value,
            an empty list is returned.

        """
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, score in docs_and_scores if score < score_threshold]

    def _prepare_query(self, k: int) -> Query:
        try:
            from redis.commands.search.query import Query
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        # Prepare the Query
        hybrid_fields = "*"
        base_query = (
            f"{hybrid_fields}=>[KNN {k} @{self.vector_key} $vector AS vector_score]"
        )
        return_fields = [self.metadata_key, self.content_key, "vector_score"]
        return (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        # Creates embedding vector from user query
        embedding = self.embedding_function(query)

        # Creates Redis query
        redis_query = self._prepare_query(k)

        params_dict: Mapping[str, str] = {
            "vector": np.array(embedding)  # type: ignore
            .astype(dtype=np.float32)
            .tobytes()
        }

        # Perform vector search
        results = self.client.ft(self.index_name).search(redis_query, params_dict)

        # Prepare document results
        docs = [
            (
                Document(
                    page_content=result.content, metadata=json.loads(result.metadata)
                ),
                float(result.vector_score),
            )
            for result in results.docs
        ]

        return docs

    @classmethod
    def from_texts_return_keys(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        distance_metric: REDIS_DISTANCE_METRICS = "COSINE",
        **kwargs: Any,
    ) -> Tuple[Redis, List[str]]:
        """Create a Redis vectorstore from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in Redis.
            3. Adds the documents to the newly created Redis index.
            4. Returns the keys of the newly created documents.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch, keys = RediSearch.from_texts_return_keys(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if "redis_url" in kwargs:
            kwargs.pop("redis_url")

        # Name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex

        # Create instance
        instance = cls(
            redis_url,
            index_name,
            embedding.embed_query,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            distance_metric=distance_metric,
            **kwargs,
        )

        # Create embeddings over documents
        embeddings = embedding.embed_documents(texts)

        # Create the search index
        instance._create_index(dim=len(embeddings[0]))

        # Add data to Redis
        keys = instance.add_texts(texts, metadatas, embeddings)
        return instance, keys

    @classmethod
    def from_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Create a Redis vectorstore from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in Redis.
            3. Adds the documents to the newly created Redis index.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain.vectorstores import Redis
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = RediSearch.from_texts(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        instance, _ = cls.from_texts_return_keys(
            texts,
            embedding,
            metadatas=metadatas,
            index_name=index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            **kwargs,
        )
        return instance

    @staticmethod
    def delete(
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a Redis entry.

        Args:
            ids: List of ids (keys) to delete.

        Returns:
            bool: Whether or not the deletions were successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if ids is None:
            raise ValueError("'ids' (keys)() were not provided.")

        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = redis.from_url(url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            client.delete(*ids)
            logger.info("Entries deleted")
            return True
        except:  # noqa: E722
            # ids does not exist
            return False

    @staticmethod
    def drop_index(
        index_name: str,
        delete_documents: bool,
        **kwargs: Any,
    ) -> bool:
        """
        Drop a Redis search index.

        Args:
            index_name (str): Name of the index to drop.
            delete_documents (bool): Whether to drop the associated documents.

        Returns:
            bool: Whether or not the drop was successful.
        """
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = redis.from_url(url=redis_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Your redis connected error: {e}")
        # Check if index exists
        try:
            client.ft(index_name).dropindex(delete_documents)
            logger.info("Drop index")
            return True
        except:  # noqa: E722
            # Index not exist
            return False

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
        vector_key: str = "content_vector",
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index."""
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            import redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            if "redis_url" in kwargs:
                kwargs.pop("redis_url")
            client = redis.from_url(url=redis_url, **kwargs)
            # check if redis has redisearch module installed
            _check_redis_module_exist(client, REDIS_REQUIRED_MODULES)
            # ensure that the index already exists
            assert _check_index_exists(
                client, index_name
            ), f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return cls(
            redis_url,
            index_name,
            embedding.embed_query,
            content_key=content_key,
            metadata_key=metadata_key,
            vector_key=vector_key,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> RedisVectorStoreRetriever:
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs)


class RedisVectorStoreRetriever(VectorStoreRetriever):
    vectorstore: Redis
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "similarity_limit"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, k=self.k)
        elif self.search_type == "similarity_limit":
            docs = self.vectorstore.similarity_search_limit_score(
                query, k=self.k, score_threshold=self.score_threshold
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError("RedisVectorStoreRetriever does not support async")

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)
