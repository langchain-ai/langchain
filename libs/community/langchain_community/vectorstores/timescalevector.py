"""VectorStore wrapper around a Postgres-TimescaleVector database."""
from __future__ import annotations

import enum
import logging
import uuid
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import DistanceStrategy

if TYPE_CHECKING:
    from timescale_vector import Predicates


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

ADA_TOKEN_COUNT = 1536

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain_store"


class TimescaleVector(VectorStore):
    """Timescale Postgres vector store

    To use, you should have the ``timescale_vector`` python package installed.

    Args:
        service_url: Service url on timescale cloud.
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain_store)
            This will become the table name used for the collection.
        distance_strategy: The distance strategy to use. (default: COSINE)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import TimescaleVector
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            SERVICE_URL = "postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = TimescaleVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                service_url=SERVICE_URL,
            )
    """  # noqa: E501

    def __init__(
        self,
        service_url: str,
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        num_dimensions: int = ADA_TOKEN_COUNT,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        time_partition_interval: Optional[timedelta] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from timescale_vector import client
        except ImportError:
            raise ImportError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )

        self.service_url = service_url
        self.embedding = embedding
        self.collection_name = collection_name
        self.num_dimensions = num_dimensions
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self._time_partition_interval = time_partition_interval
        self.sync_client = client.Sync(
            self.service_url,
            self.collection_name,
            self.num_dimensions,
            self._distance_strategy.value.lower(),
            time_partition_interval=self._time_partition_interval,
            **kwargs,
        )
        self.async_client = client.Async(
            self.service_url,
            self.collection_name,
            self.num_dimensions,
            self._distance_strategy.value.lower(),
            time_partition_interval=self._time_partition_interval,
            **kwargs,
        )
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        self.sync_client.create_tables()
        if self.pre_delete_collection:
            self.sync_client.delete_all()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def drop_tables(self) -> None:
        self.sync_client.drop_table()

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        service_url: Optional[str] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        num_dimensions = len(embeddings[0])

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        if service_url is None:
            service_url = cls.get_service_url(kwargs)

        store = cls(
            service_url=service_url,
            num_dimensions=num_dimensions,
            collection_name=collection_name,
            embedding=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    @classmethod
    async def __afrom(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        service_url: Optional[str] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        num_dimensions = len(embeddings[0])

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        if service_url is None:
            service_url = cls.get_service_url(kwargs)

        store = cls(
            service_url=service_url,
            num_dimensions=num_dimensions,
            collection_name=collection_name,
            embedding=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        await store.aadd_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        records = list(zip(ids, metadatas, texts, embeddings))
        self.sync_client.upsert(records)

        return ids

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        records = list(zip(ids, metadatas, texts, embeddings))
        await self.async_client.upsert(records)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))
        return await self.aadd_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def _embed_query(self, query: str) -> Optional[List[float]]:
        # an empty query should not be embedded
        if query is None or query == "" or query.isspace():
            return None
        else:
            return self.embedding.embed_query(query)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with TimescaleVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self._embed_query(query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            predicates=predicates,
            **kwargs,
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with TimescaleVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self._embed_query(query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            predicates=predicates,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self._embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            predicates=predicates,
            **kwargs,
        )
        return docs

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """

        embedding = self._embed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            predicates=predicates,
            **kwargs,
        )

    def date_to_range_filter(self, **kwargs: Any) -> Any:
        constructor_args = {
            key: kwargs[key]
            for key in [
                "start_date",
                "end_date",
                "time_delta",
                "start_inclusive",
                "end_inclusive",
            ]
            if key in kwargs
        }
        if not constructor_args or len(constructor_args) == 0:
            return None

        try:
            from timescale_vector import client
        except ImportError:
            raise ImportError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )
        return client.UUIDTimeRange(**constructor_args)

    def similarity_search_with_score_by_vector(
        self,
        embedding: Optional[List[float]],
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        try:
            from timescale_vector import client
        except ImportError:
            raise ImportError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )

        results = self.sync_client.search(
            embedding,
            limit=k,
            filter=filter,
            predicates=predicates,
            uuid_time_filter=self.date_to_range_filter(**kwargs),
        )

        docs = [
            (
                Document(
                    page_content=result[client.SEARCH_RESULT_CONTENTS_IDX],
                    metadata=result[client.SEARCH_RESULT_METADATA_IDX],
                ),
                result[client.SEARCH_RESULT_DISTANCE_IDX],
            )
            for result in results
        ]
        return docs

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: Optional[List[float]],
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        try:
            from timescale_vector import client
        except ImportError:
            raise ImportError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )

        results = await self.async_client.search(
            embedding,
            limit=k,
            filter=filter,
            predicates=predicates,
            uuid_time_filter=self.date_to_range_filter(**kwargs),
        )

        docs = [
            (
                Document(
                    page_content=result[client.SEARCH_RESULT_CONTENTS_IDX],
                    metadata=result[client.SEARCH_RESULT_METADATA_IDX],
                ),
                result[client.SEARCH_RESULT_DISTANCE_IDX],
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: Optional[List[float]],
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, predicates=predicates, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: Optional[List[float]],
        k: int = 4,
        filter: Optional[Union[dict, list]] = None,
        predicates: Optional[Predicates] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, predicates=predicates, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[TimescaleVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the TIMESCALE_SERVICE_URL environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls: Type[TimescaleVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the TIMESCALE_SERVICE_URL environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return await cls.__afrom(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        """Construct TimescaleVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the TIMESCALE_SERVICE_URL environment variable.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import TimescaleVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                tvs = TimescaleVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        """Construct TimescaleVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the TIMESCALE_SERVICE_URL environment variable.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import TimescaleVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                tvs = TimescaleVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return await cls.__afrom(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[TimescaleVector],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> TimescaleVector:
        """
        Get instance of an existing TimescaleVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        service_url = cls.get_service_url(kwargs)

        store = cls(
            service_url=service_url,
            collection_name=collection_name,
            embedding=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @classmethod
    def get_service_url(cls, kwargs: Dict[str, Any]) -> str:
        service_url: str = get_from_dict_or_env(
            data=kwargs,
            key="service_url",
            env_key="TIMESCALE_SERVICE_URL",
        )

        if not service_url:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the TIMESCALE_SERVICE_URL environment variable."
            )

        return service_url

    @classmethod
    def service_url_from_db_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to TimescaleVector constructor."
            )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")

        self.sync_client.delete_by_ids(ids)
        return True

    # todo should this be part of delete|()?
    def delete_by_metadata(
        self, filter: Union[Dict[str, str], List[Dict[str, str]]], **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        self.sync_client.delete_by_metadata(filter)
        return True

    class IndexType(str, enum.Enum):
        """Enumerator for the supported Index types"""

        TIMESCALE_VECTOR = "tsv"
        PGVECTOR_IVFFLAT = "ivfflat"
        PGVECTOR_HNSW = "hnsw"

    DEFAULT_INDEX_TYPE = IndexType.TIMESCALE_VECTOR

    def create_index(
        self, index_type: Union[IndexType, str] = DEFAULT_INDEX_TYPE, **kwargs: Any
    ) -> None:
        try:
            from timescale_vector import client
        except ImportError:
            raise ImportError(
                "Could not import timescale_vector python package. "
                "Please install it with `pip install timescale-vector`."
            )

        index_type = (
            index_type.value if isinstance(index_type, self.IndexType) else index_type
        )
        if index_type == self.IndexType.PGVECTOR_IVFFLAT.value:
            self.sync_client.create_embedding_index(client.IvfflatIndex(**kwargs))

        if index_type == self.IndexType.PGVECTOR_HNSW.value:
            self.sync_client.create_embedding_index(client.HNSWIndex(**kwargs))

        if index_type == self.IndexType.TIMESCALE_VECTOR.value:
            self.sync_client.create_embedding_index(
                client.TimescaleVectorIndex(**kwargs)
            )

    def drop_index(self) -> None:
        self.sync_client.drop_embedding_index()
