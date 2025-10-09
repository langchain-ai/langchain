from __future__ import annotations

import functools
import os
import uuid
import warnings
from collections.abc import Callable
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal

from langchain_qdrant._utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable, Sequence

    DictFilter = dict[str, str | int | bool | dict | list]
    MetadataFilter = DictFilter | models.Filter


class QdrantException(Exception):  # noqa: N818
    """`Qdrant` related exceptions."""


def sync_call_fallback(method: Callable) -> Callable:
    """Call the synchronous method if the async method is not implemented.

    This decorator should only be used for methods that are defined as async in the
    class.

    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            # If the async method is not implemented, call the synchronous method
            # by removing the first letter from the method name. For example,
            # if the async method is called `aadd_texts`, the synchronous method
            # will be called `aad_texts`.
            return await run_in_executor(
                None, getattr(self, method.__name__[1:]), *args, **kwargs
            )

    return wrapper


@deprecated(since="0.1.2", alternative="QdrantVectorStore", removal="0.5.0")
class Qdrant(VectorStore):
    """`Qdrant` vector store.

    Example:
        .. code-block:: python

            from qdrant_client import QdrantClient
            from langchain_qdrant import Qdrant

            client = QdrantClient()
            collection_name = "MyCollection"
            qdrant = Qdrant(client, collection_name, embedding_function)

    """

    CONTENT_KEY: str = "page_content"
    METADATA_KEY: str = "metadata"
    VECTOR_NAME: str | None = None

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embeddings: Embeddings | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: str | None = VECTOR_NAME,
        async_client: Any | None = None,
        embedding_function: Callable | None = None,  # deprecated
    ) -> None:
        """Initialize with necessary components."""
        if not isinstance(client, QdrantClient):
            msg = (
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )
            raise TypeError(msg)

        if async_client is not None and not isinstance(async_client, AsyncQdrantClient):
            msg = (
                f"async_client should be an instance of qdrant_client.AsyncQdrantClient"
                f"got {type(async_client)}"
            )
            raise ValueError(msg)

        if embeddings is None and embedding_function is None:
            msg = "`embeddings` value can't be None. Pass `embeddings` instance."
            raise ValueError(msg)

        if embeddings is not None and embedding_function is not None:
            msg = (
                "Both `embeddings` and `embedding_function` are passed. "
                "Use `embeddings` only."
            )
            raise ValueError(msg)

        self._embeddings = embeddings
        self._embeddings_function = embedding_function
        self.client: QdrantClient = client
        self.async_client: AsyncQdrantClient | None = async_client
        self.collection_name = collection_name
        self.content_payload_key = content_payload_key or self.CONTENT_KEY
        self.metadata_payload_key = metadata_payload_key or self.METADATA_KEY
        self.vector_name = vector_name or self.VECTOR_NAME

        if embedding_function is not None:
            warnings.warn(
                "Using `embedding_function` is deprecated. "
                "Pass `Embeddings` instance to `embeddings` instead.",
                stacklevel=2,
            )

        if not isinstance(embeddings, Embeddings):
            warnings.warn(
                "`embeddings` should be an instance of `Embeddings`."
                "Using `embeddings` as `embedding_function` which is deprecated",
                stacklevel=2,
            )
            self._embeddings_function = embeddings
            self._embeddings = None

        self.distance_strategy = distance_strategy.upper()

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: `64`
            **kwargs: Additional keyword arguments.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        added_ids = []
        for batch_ids, points in self._generate_rest_batches(
            texts, metadatas, ids, batch_size
        ):
            self.client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    @sync_call_fallback
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: `64`
            **kwargs: Additional keyword arguments.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)

        added_ids = []
        async for batch_ids, points in self._agenerate_rest_batches(
            texts, metadatas, ids, batch_size
        ):
            await self.async_client.upsert(
                collection_name=self.collection_name, points=points, **kwargs
            )
            added_ids.extend(batch_ids)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of Documents most similar to the query.

        """
        results = self.similarity_search_with_score(
            query,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query.

        """
        results = await self.asimilarity_search_with_score(query, k, filter, **kwargs)
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of documents most similar to the query text and distance for each.

        """
        return self.similarity_search_with_score_by_vector(
            self._embed_query(query),
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    @sync_call_fallback
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            List of documents most similar to the query text and distance for each.

        """
        query_embedding = await self._aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            query_embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of Documents most similar to the query.

        """
        results = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            List of Documents most similar to the query.

        """
        results = await self.asimilarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            search_params=search_params,
            offset=offset,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of documents most similar to the query text and distance for each.

        """
        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
                stacklevel=2,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter

        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, embedding)  # type: ignore[assignment]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # LangChain does not expect vectors to be returned
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return [
            (
                self._document_from_scored_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    @sync_call_fallback
    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        offset: int = 0,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            List of documents most similar to the query text and distance for each.

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)
        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
                stacklevel=2,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter

        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, embedding)  # type: ignore[assignment]

        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # LangChain does not expect vectors to be returned
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return [
            (
                self._document_from_scored_point(
                    result,
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                result.score,
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of Documents selected by maximal marginal relevance.

        """
        query_embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    @sync_call_fallback
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            List of Documents selected by maximal marginal relevance.

        """
        query_embedding = await self._aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of Documents selected by maximal marginal relevance.

        """
        results = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    @sync_call_fallback
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to
                AsyncQdrantClient.Search().

        Returns:
            List of Documents selected by maximal marginal relevance and distance for
            each.

        """
        results = await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            search_params=search_params,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return list(map(itemgetter(0), results))

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas
            **kwargs:
                Any other named arguments to pass through to QdrantClient.search()

        Returns:
            List of Documents selected by maximal marginal relevance and distance for
            each.

        """
        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, query_vector)  # type: ignore[assignment]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            search_params=search_params,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        embeddings = [
            result.vector.get(self.vector_name)  # type: ignore[index, union-attr]
            if self.vector_name is not None
            else result.vector
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point(
                    results[i],
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                results[i].score,
            )
            for i in mmr_selected
        ]

    @sync_call_fallback
    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: MetadataFilter | None = None,  # noqa: A002
        search_params: models.SearchParams | None = None,
        score_threshold: float | None = None,
        consistency: models.ReadConsistency | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to `None`.
            search_params: Additional search params.
            score_threshold: Define a minimal score threshold for the result.
            consistency: Read consistency of the search.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by maximal marginal relevance and distance for
            each.

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)
        query_vector = embedding
        if self.vector_name is not None:
            query_vector = (self.vector_name, query_vector)  # type: ignore[assignment]

        results = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter,
            search_params=search_params,
            limit=fetch_k,
            with_payload=True,
            with_vectors=True,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        embeddings = [
            result.vector.get(self.vector_name)  # type: ignore[index, union-attr]
            if self.vector_name is not None
            else result.vector
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            (
                self._document_from_scored_point(
                    results[i],
                    self.collection_name,
                    self.content_payload_key,
                    self.metadata_payload_key,
                ),
                results[i].score,
            )
            for i in mmr_selected
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            True if deletion is successful, False otherwise.

        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return result.status == models.UpdateStatus.COMPLETED

    @sync_call_fallback
    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            True if deletion is successful, False otherwise.

        """
        if self.async_client is None or isinstance(
            self.async_client._client, AsyncQdrantLocal
        ):
            msg = "QdrantLocal cannot interoperate with sync and async clients"
            raise NotImplementedError(msg)

        result = await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

        return result.status == models.UpdateStatus.COMPLETED

    @classmethod
    def from_texts(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        """Construct Qdrant wrapper from a list of texts.

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If ':memory:' - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "scheme | None, host, port | None,
                prefix | None".
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key:
                    API key for authentication in Qdrant Cloud. Default: None
                    Can also be set via environment variable `QDRANT_API_KEY`.
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config:
                Params for quantization, if None - quantization will be disabled
            init_from:
                Use data stored in another collection to initialize this collection
            on_disk:
                If true - vectors will be stored on disk, reducing memory usage.
            force_recreate:
                Force recreating the collection
            **kwargs:
                Additional arguments passed directly into REST client initialization

        This is a user-friendly interface that:
        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
           (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_qdrant import Qdrant
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                qdrant = Qdrant.from_texts(texts, embeddings, "localhost")

        """
        qdrant = cls.construct_instance(
            texts,
            embedding,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        qdrant.add_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def from_existing_collection(
        cls: type[Qdrant],
        embedding: Embeddings,
        path: str | None = None,
        collection_name: str | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        distance_strategy: str = "COSINE",
        vector_name: str | None = VECTOR_NAME,
        **kwargs: Any,
    ) -> Qdrant:
        """Get instance of an existing Qdrant collection.

        This method will return the instance of the store without inserting any new
        embeddings.
        """
        if collection_name is None:
            msg = "Must specify collection_name. Received None."
            raise ValueError(msg)

        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        return cls(
            client=client,
            async_client=async_client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_strategy,
            vector_name=vector_name,
        )

    @classmethod
    @sync_call_fallback
    async def afrom_texts(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        batch_size: int = 64,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        """Construct Qdrant wrapper from a list of texts.

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If ':memory:' - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "scheme | None, host, port | None,
                prefix | None".
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key:
                    API key for authentication in Qdrant Cloud. Default: None
                    Can also be set via environment variable `QDRANT_API_KEY`.
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            vector_name:
                Name of the vector to be used internally in Qdrant.
                Default: None
            batch_size:
                How many vectors upload per-request.
                Default: 64
            shard_number: Number of shards in collection. Default is 1, minimum is 1.
            replication_factor:
                Replication factor for collection. Default is 1, minimum is 1.
                Defines how many copies of each shard will be created.
                Have effect only in distributed mode.
            write_consistency_factor:
                Write consistency factor for collection. Default is 1, minimum is 1.
                Defines how many replicas should apply the operation for us to consider
                it successful. Increasing this number will make the collection more
                resilient to inconsistencies, but will also make it fail if not enough
                replicas are available.
                Does not have any performance impact.
                Have effect only in distributed mode.
            on_disk_payload:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - remain in RAM.
            hnsw_config: Params for HNSW index
            optimizers_config: Params for optimizer
            wal_config: Params for Write-Ahead-Log
            quantization_config:
                Params for quantization, if None - quantization will be disabled
            init_from:
                Use data stored in another collection to initialize this collection
            on_disk:
                If true - point`s payload will not be stored in memory.
                It will be read from the disk every time it is requested.
                This setting saves RAM by (slightly) increasing the response time.
                Note: those payload values that are involved in filtering and are
                indexed - remain in RAM.
            force_recreate:
                Force recreating the collection
            **kwargs:
                Additional arguments passed directly into REST client initialization

        This is a user-friendly interface that:
        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
           (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_qdrant import Qdrant
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                qdrant = await Qdrant.afrom_texts(texts, embeddings, "localhost")

        """
        qdrant = await cls.aconstruct_instance(
            texts,
            embedding,
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            collection_name,
            distance_func,
            content_payload_key,
            metadata_payload_key,
            vector_name,
            shard_number,
            replication_factor,
            write_consistency_factor,
            on_disk_payload,
            hnsw_config,
            optimizers_config,
            wal_config,
            quantization_config,
            init_from,
            on_disk,
            force_recreate,
            **kwargs,
        )
        await qdrant.aadd_texts(texts, metadatas, ids, batch_size)
        return qdrant

    @classmethod
    def construct_instance(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        # Just do a single quick embedding to get vector size
        partial_embeddings = embedding.embed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )
        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False

        if collection_exists:
            # Get the vector configuration of the existing collection and vector, if it
            # was specified. If the old configuration does not match the current one,
            # an exception is raised.
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    msg = (
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                    raise QdrantException(msg)
                current_vector_config = current_vector_config.get(vector_name)  # type: ignore[assignment]
            elif isinstance(current_vector_config, dict) and vector_name is None:
                msg = (
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                msg = (
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )
                raise QdrantException(msg)
            if not isinstance(current_vector_config, models.VectorParams):
                msg = (
                    "Expected current_vector_config to be an instance of "
                    f"models.VectorParams, but got {type(current_vector_config)}"
                )
                raise ValueError(msg)
            # Check if the vector configuration has the same dimensionality.
            if current_vector_config.size != vector_size:
                msg = (
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                msg = (
                    f"Existing Qdrant collection is configured for "
                    f"{current_distance_func} similarity, but requested "
                    f"{distance_func}. Please set `distance_func` parameter to "
                    f"`{current_distance_func}` if you want to reuse it. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
        else:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance_func],
                on_disk=on_disk,
            )

            # If vector name was provided, we're going to use the named vectors feature
            # with just a single vector.
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        return cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )

    @classmethod
    async def aconstruct_instance(
        cls: type[Qdrant],
        texts: list[str],
        embedding: Embeddings,
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        vector_name: str | None = VECTOR_NAME,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,  # noqa: FBT001
        hnsw_config: models.HnswConfigDiff | None = None,
        optimizers_config: models.OptimizersConfigDiff | None = None,
        wal_config: models.WalConfigDiff | None = None,
        quantization_config: models.QuantizationConfig | None = None,
        init_from: models.InitFrom | None = None,
        on_disk: bool | None = None,  # noqa: FBT001
        force_recreate: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> Qdrant:
        # Just do a single quick embedding to get vector size
        partial_embeddings = await embedding.aembed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])
        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()
        client, async_client = cls._generate_clients(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        collection_exists = client.collection_exists(collection_name)

        if collection_exists and force_recreate:
            client.delete_collection(collection_name)
            collection_exists = False

        if collection_exists:
            # Get the vector configuration of the existing collection and vector, if it
            # was specified. If the old configuration does not match the current one,
            # an exception is raised.
            collection_info = client.get_collection(collection_name=collection_name)
            current_vector_config = collection_info.config.params.vectors
            if isinstance(current_vector_config, dict) and vector_name is not None:
                if vector_name not in current_vector_config:
                    msg = (
                        f"Existing Qdrant collection {collection_name} does not "
                        f"contain vector named {vector_name}. Did you mean one of the "
                        f"existing vectors: {', '.join(current_vector_config.keys())}? "
                        f"If you want to recreate the collection, set `force_recreate` "
                        f"parameter to `True`."
                    )
                    raise QdrantException(msg)
                current_vector_config = current_vector_config.get(vector_name)  # type: ignore[assignment]
            elif isinstance(current_vector_config, dict) and vector_name is None:
                msg = (
                    f"Existing Qdrant collection {collection_name} uses named vectors. "
                    f"If you want to reuse it, please set `vector_name` to any of the "
                    f"existing named vectors: "
                    f"{', '.join(current_vector_config.keys())}."
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)
            elif (
                not isinstance(current_vector_config, dict) and vector_name is not None
            ):
                msg = (
                    f"Existing Qdrant collection {collection_name} doesn't use named "
                    f"vectors. If you want to reuse it, please set `vector_name` to "
                    f"`None`. If you want to recreate the collection, set "
                    f"`force_recreate` parameter to `True`."
                )
                raise QdrantException(msg)
            if not isinstance(current_vector_config, models.VectorParams):
                msg = (
                    "Expected current_vector_config to be an instance of "
                    f"models.VectorParams, but got {type(current_vector_config)}"
                )
                raise ValueError(msg)

            # Check if the vector configuration has the same dimensionality.
            if current_vector_config.size != vector_size:
                msg = (
                    f"Existing Qdrant collection is configured for vectors with "
                    f"{current_vector_config.size} "
                    f"dimensions. Selected embeddings are {vector_size}-dimensional. "
                    f"If you want to recreate the collection, set `force_recreate` "
                    f"parameter to `True`."
                )
                raise QdrantException(msg)

            current_distance_func = (
                current_vector_config.distance.name.upper()  # type: ignore[union-attr]
            )
            if current_distance_func != distance_func:
                msg = (
                    f"Existing Qdrant collection is configured for "
                    f"{current_vector_config.distance} "  # type: ignore[union-attr]
                    f"similarity. Please set `distance_func` parameter to "
                    f"`{distance_func}` if you want to reuse it. If you want to "
                    f"recreate the collection, set `force_recreate` parameter to "
                    f"`True`."
                )
                raise QdrantException(msg)
        else:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance_func],
                on_disk=on_disk,
            )

            # If vector name was provided, we're going to use the named vectors feature
            # with just a single vector.
            if vector_name is not None:
                vectors_config = {  # type: ignore[assignment]
                    vector_name: vectors_config,
                }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                wal_config=wal_config,
                quantization_config=quantization_config,
                init_from=init_from,
                timeout=timeout,  # type: ignore[arg-type]
            )
        return cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
            distance_strategy=distance_func,
            vector_name=vector_name,
            async_client=async_client,
        )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return (distance + 1.0) / 2.0

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Your 'correct' relevance function may differ depending on a few things.

        For example:
        - The distance / similarity metric used by the VectorStore
        - The scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - Embedding dimensionality
        - etc.
        """
        if self.distance_strategy == "COSINE":
            return self._cosine_relevance_score_fn
        if self.distance_strategy == "DOT":
            return self._max_inner_product_relevance_score_fn
        if self.distance_strategy == "EUCLID":
            return self._euclidean_relevance_score_fn
        msg = (
            "Unknown distance strategy, must be cosine, max_inner_product, or euclidean"
        )
        raise ValueError(msg)

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)

        """
        return self.similarity_search_with_score(query, k, **kwargs)

    @sync_call_fallback
    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)

        """
        return await self.asimilarity_search_with_score(query, k, **kwargs)

    @classmethod
    def _build_payloads(
        cls,
        texts: Iterable[str],
        metadatas: list[dict] | None,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> list[dict]:
        payloads = []
        for i, text in enumerate(texts):
            if text is None:
                msg = (
                    "At least one of the texts is None. Please remove it before "
                    "calling .from_texts or .add_texts on Qdrant instance."
                )
                raise ValueError(msg)
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append(
                {
                    content_payload_key: text,
                    metadata_payload_key: metadata,
                }
            )

        return payloads

    @classmethod
    def _document_from_scored_point(
        cls,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        return Document(
            page_content=scored_point.payload.get(content_payload_key, ""),
            metadata=metadata,
        )

    def _build_condition(self, key: str, value: Any) -> list[models.FieldCondition]:
        out = []

        if isinstance(value, dict):
            for _key, _value in value.items():
                out.extend(self._build_condition(f"{key}.{_key}", _value))
        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    out.extend(self._build_condition(f"{key}[]", _value))
                else:
                    out.extend(self._build_condition(f"{key}", _value))
        else:
            out.append(
                models.FieldCondition(
                    key=f"{self.metadata_payload_key}.{key}",
                    match=models.MatchValue(value=value),
                )
            )

        return out

    def _qdrant_filter_from_dict(
        self, filter_: DictFilter | None
    ) -> models.Filter | None:
        if not filter_:
            return None

        return models.Filter(
            must=[
                condition
                for key, value in filter_.items()  # type: ignore[union-attr]
                for condition in self._build_condition(key, value)
            ]
        )

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            query: Query text.

        Returns:
            List of floats representing the query embedding.

        """
        if self.embeddings is not None:
            embedding = self.embeddings.embed_query(query)
        elif self._embeddings_function is not None:
            embedding = self._embeddings_function(query)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    async def _aembed_query(self, query: str) -> list[float]:
        """Embed query text asynchronously.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            query: Query text.

        Returns:
            List of floats representing the query embedding.

        """
        if self.embeddings is not None:
            embedding = await self.embeddings.aembed_query(query)
        elif self._embeddings_function is not None:
            embedding = self._embeddings_function(query)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    def _embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed search texts.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            List of floats representing the texts embedding.

        """
        if self.embeddings is not None:
            embeddings = self.embeddings.embed_documents(list(texts))
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        elif self._embeddings_function is not None:
            embeddings = []
            for text in texts:
                embedding = self._embeddings_function(text)
                if hasattr(embeddings, "tolist"):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)

        return embeddings

    async def _aembed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed search texts.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            List of floats representing the texts embedding.

        """
        if self.embeddings is not None:
            embeddings = await self.embeddings.aembed_documents(list(texts))
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        elif self._embeddings_function is not None:
            embeddings = []
            for text in texts:
                embedding = self._embeddings_function(text)
                if hasattr(embeddings, "tolist"):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
        else:
            msg = "Neither of embeddings or embedding_function is set"
            raise ValueError(msg)

        return embeddings

    def _generate_rest_batches(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> Generator[tuple[list[str], list[models.PointStruct]], None, None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Generate the embeddings for all the texts in a batch
            batch_embeddings = self._embed_texts(batch_texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector  # type: ignore[arg-type]
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                    strict=False,
                )
            ]

            yield batch_ids, points

    async def _agenerate_rest_batches(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: Sequence[str] | None = None,
        batch_size: int = 64,
    ) -> AsyncGenerator[tuple[list[str], list[models.PointStruct]], None]:
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Generate the embeddings for all the texts in a batch
            batch_embeddings = await self._aembed_texts(batch_texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector  # type: ignore[arg-type]
                    if self.vector_name is None
                    else {self.vector_name: vector},
                    payload=payload,
                )
                for point_id, vector, payload in zip(
                    batch_ids,
                    batch_embeddings,
                    self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                    strict=False,
                )
            ]

            yield batch_ids, points

    @staticmethod
    def _generate_clients(
        location: str | None = None,
        url: str | None = None,
        port: int | None = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,  # noqa: FBT001, FBT002
        https: bool | None = None,  # noqa: FBT001
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        **kwargs: Any,
    ) -> tuple[QdrantClient, AsyncQdrantClient | None]:
        if api_key is None:
            api_key = os.getenv("QDRANT_API_KEY")

        sync_client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        if location == ":memory:" or path is not None:
            # Local Qdrant cannot co-exist with Sync and Async clients
            # We fallback to sync operations in this case
            async_client = None
        else:
            async_client = AsyncQdrantClient(
                location=location,
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                path=path,
                **kwargs,
            )

        return sync_client, async_client
