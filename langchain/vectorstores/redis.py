"""Wrapper around Redis vector database."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type

import numpy as np
from pydantic import BaseModel, root_validator
from redis.asyncio import Redis as RedisType

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()


# required modules
REDIS_REQUIRED_MODULES = [
    {
        "name": "search",
        "ver": 20400
    },
]


class Redis(VectorStore):
    def __init__(
        self,
        client: RedisType,
        index_name: str,
        embedding_function: Callable,
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        self.loop = kwargs.pop('loop', asyncio.get_event_loop())
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.client = client

    @staticmethod
    def _redis_key(prefix: str) -> str:
        """Redis key schema for a given prefix."""
        return f"{prefix}:{uuid.uuid4().hex}"

    @staticmethod
    def _redis_prefix(index_name: str) -> str:
        """Redis key prefix for a given index."""
        return f"doc:{index_name}"

    @staticmethod
    async def _check_redis_module_exist(client: RedisType, modules: List[dict]) -> None:
        """Check if the correct Redis modules are installed."""
        installed_modules = await client.module_list()
        installed_modules = {
            module[b"name"].decode("utf-8"): module for module in installed_modules
        }
        for module in modules:
            if module["name"] not in installed_modules or int(
                installed_modules[module["name"]][b"ver"]
            ) < int(module["ver"]):
                error_message = (
                    "You must add the RediSearch (>= 2.4) module from Redis Stack. "
                    "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
                )
                logging.error(error_message)
                raise ValueError(error_message)

    @staticmethod
    async def _check_redis_index_exists(client: RedisType, index_name: str) -> bool:
        """Check if Redis index exists."""
        try:
            await client.ft(index_name).info()
        except:  # noqa: E722
            logger.info("Index does not exist")
            return False
        logger.info("Index already exists")
        return True

    async def _load_redis_client(
        self,
        index_name: str,
        check_module: bool = True,
        check_index: bool = False,
        **kwargs
    ):
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        try:
            # We need to first remove redis_url from kwargs,
            # otherwise passing it to Redis will result in an error.
            kwargs.pop("redis_url")
            client = redis.from_url(url=redis_url, **kwargs)

            # check if redis has redisearch module installed
            # FIXME: DvirDu: We should always check for this
            if check_module:
                await self._check_redis_module_exist(client, REDIS_REQUIRED_MODULES)

            # ensure that the index already exists
            if check_index:
                exists = await self._check_redis_index_exists(client, index_name)
                assert exists, f"Index {index_name} does not exist"
        except Exception as e:
            raise ValueError(f"Redis failed to connect: {e}")

        return client

    async def _setup_redis_index(
        self,
        index_name: str,
        dim: int
    ):
        """"""
        try:
            from redis.commands.search.field import TextField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        # Prefix for the document keys
        prefix = self._redis_prefix(index_name)
        # Check if index exists
        if not await self._check_redis_index_exists(self.client, index_name):
            # Create embeddings for the first document to get the dimension
            # TODO: This is a hack to get the dimension of the embeddings, we should
            # find a better way to do this.
            distance_metric = (
                "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
            )
            schema = (
                TextField(name="content"),
                TextField(name="metadata"),
                VectorField(
                    "content_vector",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dim,
                        "DISTANCE_METRIC": distance_metric,
                    },
                ),
            )
            # Create Redis Index
            await self.client.ft(index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    async def _query_redis(self, embedding: List[float], k: int) -> List[Document]:
        """"""
        try:
            from redis.commands.search.query import Query
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )

        # Setup Redis query
        return_fields = ["metadata", "content", "vector_score"]
        vector_field = "content_vector"
        hybrid_fields = "*"
        base_query = (
            f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
        )
        redis_query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict: Mapping[str, str] = {
            "vector": np.array(embedding)  # type: ignore
            .astype(dtype=np.float32)
            .tobytes()
        }

        # Perform vector search
        results = (
            await self.client
              .ft(self.index_name)
              .search(redis_query, params_dict)
        )

        # Aggregate result documents
        return [
            (
                Document(
                    page_content=result.content,
                    metadata=json.loads(result.metadata)
                ),
                float(result.vector_score),
            )
            for result in results.docs
        ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        return self.loop.run_until_complete(
            self.aadd_texts(texts, metadatas, embeddings, **kwargs)
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[float]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add texts data to an existing index."""
        prefix = self._redis_prefix(self.index_name)
        keys = kwargs.get("keys")
        # Init list of keys
        ids: List[str] = []
        # Write data to redis in a pipeline
        pipeline = await self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided key otherwise use default key
            key = keys[i] if keys else self._redis_key(prefix)
            metadata = metadatas[i] if metadatas else {}
            embedding = embeddings[i] if embeddings else self.embedding_function(text)
            await pipeline.hset(
                key,
                mapping={
                    "content": text,
                    "content_vector": np.array(
                        embedding, dtype=np.float32
                    ).tobytes(),
                    "metadata": json.dumps(metadata),
                }
            )
            ids.append(key)
        await pipeline.execute()
        return ids

    async def asimilarity_search_with_score(
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
        # Perform search in Redis
        docs = await self._query_redis(embedding, k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        return self.loop.run_until_complete(
            self.asimilarity_search_by_vector(embedding, k, **kwargs)
        )

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector."""
        docs = await self._query_redis(embedding, k)
        return docs

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
        return self.loop.run_until_complete(
            self.asimilarity_search(query, k, **kwargs)
        )

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.
        """
        docs_and_scores = await self.asimilarity_search_with_score(query, k=k)
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
        return self.loop.run_until_complete(
            self.asimilarity_search_limit_score(query, k, score_threshold, **kwargs)
        )

    async def asimilarity_search_limit_score(
        self, query: str, k: int = 4, score_threshold: float = 0.2, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text within the
        score_threshold range.
        """
        docs_and_scores = await self.asimilarity_search_with_score(query, k=k)
        return [doc for doc, score in docs_and_scores if score < score_threshold]

    @classmethod
    def from_texts(
        cls: Type[Redis],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Redis:
        """Construct RediSearch wrapper from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the RediSearch instance.
            3. Adds the documents to the newly created RediSearch index.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain import RediSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redisearch = RediSearch.from_texts(
                    texts,
                    embeddings,
                    redis_url="redis://username:password@localhost:6379"
                )
        """
        loop = asyncio.get_event_loop()
        instance = loop.run_until_complete(
            cls.afrom_texts(texts, embedding, metadatas, index_name, **kwargs)
        )
        return instance

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any
    ) -> Redis:
        # Check index name
        if not index_name:
            index_name = uuid.uuid4().hex
        # Setup Redis client
        client = await cls._load_redis_client(cls, index_name=index_name, **kwargs)
        instance = cls(client, index_name, embedding.embed_query)
        embeddings = embedding.embed_documents(texts)
        dim = len(embeddings[0])
        await instance._setup_redis_index(index_name, dim)
        await instance.aadd_texts(texts, metadatas, embeddings, **kwargs)
        return instance

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
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index."""
        loop = asyncio.get_event_loop()
        instance = loop.run_until_complete(
            cls.afrom_existing_index(embedding, index_name, **kwargs)
        )
        return instance

    @classmethod
    async def afrom_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str,
        **kwargs: Any,
    ) -> Redis:
        """Connect to an existing Redis index."""
        client = await cls._load_redis_client(
            cls, index_name=index_name, check_index=True, **kwargs
        )
        return cls(client, index_name, embedding.embed_query)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RedisVectorStoreRetriever(vectorstore=self, **kwargs)


class RedisVectorStoreRetriever(BaseRetriever, BaseModel):
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

    def get_relevant_documents(self, query: str) -> List[Document]:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(
            self.aget_relevant_documents(query)
        )

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(query, k=self.k)
        elif self.search_type == "similarity_limit":
            docs = await self.vectorstore.asimilarity_search_limit_score(
                query, k=self.k, score_threshold=self.score_threshold
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs
