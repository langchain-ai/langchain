from __future__ import annotations

import asyncio
import logging
import threading
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

if TYPE_CHECKING:
    from aerospike_vector.types import HostPort
    from aerospike_vector.vectordb_client import VectorDbClient

logger = logging.getLogger(__name__)


def _import_aerospike() -> Any:  # TODO: Replace this with Any
    try:
        from aerospike_vector import vectordb_client
    except ImportError as e:
        raise ImportError(
            "Could not import aerospike_vector python package. "
            "Please install it with `pip install aerospike_vector`."
        ) from e
    return vectordb_client


# def _import_aerospike_admin() -> vectordb_admin:  # TODO: Replace this with Any
#     try:
#         from aerospike_vector import vectordb_admin
#     except ImportError as e:
#         raise ImportError(
#             "Could not import aerospike_vector python package. "
#             "Please install it with `pip install aerospike_vector`."
#         ) from e
#     return vectordb_admin


class Aerospike(VectorStore):
    """`Aerospike` vector store.

    To use, you should have the ``aerospike_vector`` python package installed.
    """

    def __init__(
        self,
        client: Any,
        embedding: Union[Embeddings, Callable],
        text_key: str,
        vector_key: str,
        index_name: str,
        namespace: str,
        set_name: Optional[str] = None,
        distance_strategy: Optional[DistanceStrategy] = DistanceStrategy.COSINE,
    ):
        """Initialize with Aerospike client.

        Args:
            client: Aerospike client.
            embedding: Embeddings object or Callable (deprecated) to embed text.
            text_key: Key to use for text in metadata.
            vector_key: Key to use for vector in metadata. This should match the
            key used during index creation.
            index_name: Name of the index previously created in Aerospike. This
            should match the index name used during index creation.
            namespace: Namespace to use for storing vectors. This should match
            the key used during index creation.
            set_name: Default set name to use for storing vectors.
            distance_strategy: Distance strategy to use for similarity search.
        """

        aerospike = _import_aerospike()

        if not isinstance(embedding, Embeddings):
            warnings.warn(
                "Passing in `embedding` as a Callable is deprecated. Please pass in an"
                " Embeddings object instead."
            )

        if not isinstance(
            client, aerospike.VectorDbClient
        ):  # TODO: Add "or aerospike.AsycnClient"
            raise ValueError(
                f"client should be an instance of aerospike_vector.Client, "
                f"got {type(client)}"
            )

        # TODO: if isinstance(client, aerospike.AsyncClient):
        # try:
        #     # self._loop = asyncio.get_event_loop()
        #     # self._event_loop_thread = threading.Thread(target=self._loop.run_forever(), daemon=True)
        #     # self._event_loop_thread.start()  # TODO: Hopefully this is temporary
        # except RuntimeError:
        #     self._loop = None
        #     self._event_loop_thread = None
        #     raise RuntimeError(
        #         "Aerospike is either uninitialized or initialized in a different thread. Please initialize Aerospike in the same thread."
        #     )

        self._client = client
        self._embedding = embedding
        self._text_key = text_key
        self._vector_key = vector_key
        self._index_name = index_name
        self._namespace = namespace
        self._set_name = set_name
        self.distance_strategy = distance_strategy

    # def __del__(self):
    #     if self._loop:
    #         self._loop.stop()


    def run_async(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def run_async_gather(self, coroutines):
        return [self.run_async(co).result() for co in coroutines]

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        if isinstance(self._embedding, Embeddings):
            return self._embedding
        return None

    def _embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed search docs."""
        if isinstance(self._embedding, Embeddings):
            return self._embedding.embed_documents(list(texts))
        return [self._embedding(t) for t in texts]

    def _embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        if isinstance(self._embedding, Embeddings):
            return self._embedding.embed_query(text)
        return self._embedding(text)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        set_name: Optional[
            str
        ] = None,  # TODO: Should we allow namespaces to be passed in? They are much less flexible than pinecones.
        # batch_size: int = 32, TODO: When client has batch insert.
        embedding_chunk_size: int = 1000,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.


        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            set_name: Optional aerospike set name to add the texts to.
            batch_size: Batch size to use when adding the texts to the vectorstore.
            embedding_chunk_size: Chunk size to use when embedding the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        if set_name is None:
            set_name = self._set_name

        if set_name is None and self._namespace is not None:
            raise ValueError(
                f"Namespace must be provided if set_name is provided. namespace: {self._namespace}, set_name: {set_name}"
            )

        # TODO: Should we check that the index already exists before inserting?

        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        for i in range(0, len(texts), embedding_chunk_size):
            chunk_texts = texts[i : i + embedding_chunk_size]
            chunk_ids = ids[i : i + embedding_chunk_size]
            chunk_metadatas = metadatas[i : i + embedding_chunk_size]
            embeddings = self._embed_documents(chunk_texts)
            for metadata, embedding, text in zip(
                chunk_metadatas, embeddings, chunk_texts
            ):
                metadata[self._vector_key] = embedding
                metadata[self._text_key] = text

            coroutines = [None] * len(chunk_ids)
            for idx, id, metadata in zip(
                range(len(chunk_ids)), chunk_ids, chunk_metadatas
            ):
                coroutines[idx] = self._client.put(
                    namespace=self._namespace,
                    key=id,
                    set_name=set_name,
                    record_data=metadata,
                )

            await asyncio.gather(*coroutines)

        return ids

    def add_texts(
        self,
        *args,
        **kwargs: Any,
    ) -> List[str]:
        """TODO"""
        return self.run_async(self.aadd_texts(*args, **kwargs)).result()

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        # TODO: Client does not have a delete method yet.
        raise NotImplementedError("delete method must be implemented by subclass.")

    def delete(self, *args, **kwargs: Any) -> Optional[bool]:
        """TODO"""
        return self.run_async(self.adelete(*args, **kwargs)).result()

    def similarity_search_with_threshold(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        score_threshold = kwargs.pop("score_threshold", None)
        result = self.vector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return aerospike documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and associated scores.
        """

        return self.similarity_search_by_vector_with_score(
            self._embed_query(query), k=k
        )

    async def asimilarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return aerospike documents most similar to embedding, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and associated scores.

        """

        docs = []
        results = await self._client.vector_search(
            index_name=self._index_name,
            namespace=self._namespace,
            query=embedding,
            limit=k,
        )

        for result in results:
            metadata = result.bins
            if self._text_key in metadata and self._vector_key in metadata:
                text = metadata.pop(self._text_key)
                metadata.pop(self._vector_key)
                score = result.distance
                docs.append((Document(page_content=text, metadata=metadata), score))
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )

        return docs

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        *,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return aerospike documents most similar to embedding, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and associated scores.

        """

        return self.run_async(
            self.asimilarity_search_by_vector_with_score(embedding, k=k)
        ).result()

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
        return [
            doc
            for doc, _ in self.similarity_search_by_vector_with_score(embedding, k=k)
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return aerospike documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Aerospike's relevance_fn assume embeddings are normalized to unit norm.
        """

        if self.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product "
                "(dot product), or euclidean"
            )

    @staticmethod
    def _cosine_relevance_score_fn(score: float) -> float:
        """Aerospike returns cosine similarity scores between [-1,1]"""
        return (score + 1) / 2

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
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
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        results = self._client.vector_search(
            index_name=self._index_name,
            namespace=self._namespace,
            query=embedding,
            limit=fetch_k,
            **kwargs,
        )

        filtered_results = []
        for result in results:
            if self._vector_key not in result.bins or self._text_key not in result.bins:
                logger.warning(
                    f"Found document with no `{self._vector_key}` key. Skipping."
                )

            filtered_results.append(result)

        results = filtered_results
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [result.bins[self._vector_key] for result in results],
            k=k,
            lambda_mult=lambda_mult,
        )

        for i in mmr_selected:
            metadata = results[i].bins
            metadata.pop(self._vector_key)
            metadata.pop(self._text_key)

        selected = [results[i].bins["metadata"] for i in mmr_selected]
        return [
            Document(page_content=metadata.pop((self._text_key)), metadata=metadata)
            for metadata in selected
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter, namespace, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        seeds: List[HostPort],
        texts: List[str],
        embedding: Embeddings,
        index_name: str,
        namespace: str,
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        embeddings_chunk_size: int = 1000,
        client_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> Aerospike:
        """
        This is a user friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Aerospike index

        This is intended to be a quick way to get started.

        The `pool_threads` affects the speed of the upsert operations.

        Example:
            .. code-block:: python

                from langchain_community import AerospikeVectorStore TODO check this
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                index_name = "my-index"
                namespace = "my-namespace"
                vectorstore = Aerospike(
                    index_name=index_name,
                    embedding=embedding,
                    namespace=namespace,
                )
        """
        """
                client: vectordb_client.VectorDbClient,  # TODO: Replace this with any
        embedding: Union[Embeddings, Callable],
        text_key: str,
        vector_key: str,
        index_name: str,
        namespace: str,
        set_name: Optional[str] = None,
        distance_strategy: Optional[DistanceStrategy] =
        """
        aeorspike_client = _import_aerospike()
        client = aeorspike_client.VectorDbClient(seeds, **(client_kwargs or {}))
        aerospike = cls(
            client,
            embedding,
            index_name,
            namespace,
            **kwargs,
        )

        aerospike.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            namespace=namespace,
            embedding_chunk_size=embeddings_chunk_size,
        )
        return aerospike
