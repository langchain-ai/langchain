from __future__ import annotations

import logging
import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

if TYPE_CHECKING:
    from aerospike_vector_search import Client
    from aerospike_vector_search.types import Neighbor, VectorDistanceMetric

logger = logging.getLogger(__name__)


def _import_aerospike() -> Any:
    try:
        from aerospike_vector_search import Client
    except ImportError as e:
        raise ImportError(
            "Could not import aerospike_vector_search python package. "
            "Please install it with `pip install aerospike_vector`."
        ) from e
    return Client


AVST = TypeVar("AVST", bound="Aerospike")


class Aerospike(VectorStore):
    """`Aerospike` vector store.

    To use, you should have the ``aerospike_vector_search`` python package installed.
    """

    def __init__(
        self,
        client: Client,
        embedding: Union[Embeddings, Callable],
        namespace: str,
        index_name: Optional[str] = None,
        vector_key: str = "_vector",
        text_key: str = "_text",
        id_key: str = "_id",
        set_name: Optional[str] = None,
        distance_strategy: Optional[
            Union[DistanceStrategy, VectorDistanceMetric]
        ] = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        """Initialize with Aerospike client.

        Args:
            client: Aerospike client.
            embedding: Embeddings object or Callable (deprecated) to embed text.
            namespace: Namespace to use for storing vectors. This should match
            index_name: Name of the index previously created in Aerospike. This
            vector_key: Key to use for vector in metadata. This should match the
                key used during index creation.
            text_key: Key to use for text in metadata.
            id_key: Key to use for id in metadata.
            set_name: Default set name to use for storing vectors.
            distance_strategy: Distance strategy to use for similarity search
                This should match the distance strategy used during index creation.
        """

        aerospike = _import_aerospike()

        if not isinstance(embedding, Embeddings):
            warnings.warn(
                "Passing in `embedding` as a Callable is deprecated. Please pass in an"
                " Embeddings object instead."
            )

        if not isinstance(client, aerospike):
            raise ValueError(
                f"client should be an instance of aerospike_vector_search.Client, "
                f"got {type(client)}"
            )

        self._client = client
        self._embedding = embedding
        self._text_key = text_key
        self._vector_key = vector_key
        self._id_key = id_key
        self._index_name = index_name
        self._namespace = namespace
        self._set_name = set_name
        self._distance_strategy = self.convert_distance_strategy(distance_strategy)

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

    @staticmethod
    def convert_distance_strategy(
        distance_strategy: Union[VectorDistanceMetric, DistanceStrategy],
    ) -> DistanceStrategy:
        """
        Convert Aerospikes distance strategy to langchains DistanceStrategy
        enum. This is a convenience method to allow users to pass in the same
        distance metric used to create the index.
        """
        from aerospike_vector_search.types import VectorDistanceMetric

        if isinstance(distance_strategy, DistanceStrategy):
            return distance_strategy

        if distance_strategy == VectorDistanceMetric.COSINE:
            return DistanceStrategy.COSINE

        if distance_strategy == VectorDistanceMetric.DOT_PRODUCT:
            return DistanceStrategy.DOT_PRODUCT

        if distance_strategy == VectorDistanceMetric.SQUARED_EUCLIDEAN:
            return DistanceStrategy.EUCLIDEAN_DISTANCE

        raise ValueError(
            "Unknown distance strategy, must be cosine, dot_product" ", or euclidean"
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        set_name: Optional[str] = None,
        embedding_chunk_size: int = 1000,
        index_name: Optional[str] = None,
        wait_for_index: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.


        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.
            ids: Optional list of ids to associate with the texts.
            set_name: Optional aerospike set name to add the texts to.
            batch_size: Batch size to use when adding the texts to the vectorstore.
            embedding_chunk_size: Chunk size to use when embedding the texts.
            index_name: Optional aerospike index name used for waiting for index
                completion. If not provided, the default index_name will be used.
            wait_for_index: If True, wait for the all the texts to be indexed
                before returning. Requires index_name to be provided. Defaults
                to True.
            kwargs: Additional keyword arguments to pass to the client upsert call.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        if set_name is None:
            set_name = self._set_name

        if index_name is None:
            index_name = self._index_name

        if wait_for_index and index_name is None:
            raise ValueError("if wait_for_index is True, index_name must be provided")

        texts = list(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        # We need to shallow copy so that we can add the vector and text keys
        if metadatas:
            metadatas = [m.copy() for m in metadatas]
        else:
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

            for id, metadata in zip(chunk_ids, chunk_metadatas):
                metadata[self._id_key] = id
                self._client.upsert(
                    namespace=self._namespace,
                    key=id,
                    set_name=set_name,
                    record_data=metadata,
                    **kwargs,
                )

        if wait_for_index:
            self._client.wait_for_index_completion(
                namespace=self._namespace,
                name=index_name,
            )

        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        set_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments to pass to client delete call.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        from aerospike_vector_search import AVSServerError

        if ids:
            for id in ids:
                try:
                    self._client.delete(
                        namespace=self._namespace,
                        key=id,
                        set_name=set_name,
                        **kwargs,
                    )
                except AVSServerError:
                    return False

        return True

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return aerospike documents most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_keys: List of metadata keys to return with the documents.
                If None, all metadata keys will be returned. Defaults to None.
            index_name: Name of the index to search. Overrides the default
                index_name.
            kwargs: Additional keyword arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query and associated scores.
        """

        return self.similarity_search_by_vector_with_score(
            self._embed_query(query),
            k=k,
            metadata_keys=metadata_keys,
            index_name=index_name,
            **kwargs,
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return aerospike documents most similar to embedding, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_keys: List of metadata keys to return with the documents.
                If None, all metadata keys will be returned. Defaults to None.
            index_name: Name of the index to search. Overrides the default
                index_name.
            kwargs: Additional keyword arguments to pass to the client
                vector_search method.

        Returns:
            List of Documents most similar to the query and associated scores.

        """

        docs = []

        if metadata_keys and self._text_key not in metadata_keys:
            metadata_keys = [self._text_key] + metadata_keys

        if index_name is None:
            index_name = self._index_name

        if index_name is None:
            raise ValueError("index_name must be provided")

        results: list[Neighbor] = self._client.vector_search(
            index_name=index_name,
            namespace=self._namespace,
            query=embedding,
            limit=k,
            field_names=metadata_keys,
            **kwargs,
        )

        for result in results:
            metadata = result.fields

            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = result.distance
                docs.append((Document(page_content=text, metadata=metadata), score))
            else:
                logger.warning(
                    f"Found document with no `{self._text_key}` key. Skipping."
                )
                continue

        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_keys: List of metadata keys to return with the documents.
                If None, all metadata keys will be returned. Defaults to None.
            index_name: Name of the index to search. Overrides the default
                index_name.
            kwargs: Additional keyword arguments to pass to the search method.


        Returns:
            List of Documents most similar to the query vector.
        """
        return [
            doc
            for doc, _ in self.similarity_search_by_vector_with_score(
                embedding,
                k=k,
                metadata_keys=metadata_keys,
                index_name=index_name,
                **kwargs,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return aerospike documents most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            metadata_keys: List of metadata keys to return with the documents.
                If None, all metadata keys will be returned. Defaults to None.
            index_name: Optional name of the index to search. Overrides the
                default index_name.

        Returns:
            List of Documents most similar to the query and score for each
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, metadata_keys=metadata_keys, index_name=index_name, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        0 is dissimilar, 1 is similar.

        Aerospike's relevance_fn assume euclidean and dot product embeddings are
        normalized to unit norm.
        """
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.DOT_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, dot_product"
                ", or euclidean"
            )

    @staticmethod
    def _cosine_relevance_score_fn(score: float) -> float:
        """Aerospike returns cosine distance scores between [0,2]

        0 is dissimilar, 1 is similar.
        """
        return 1 - (score / 2)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree of
                diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            metadata_keys: List of metadata keys to return with the documents.
                If None, all metadata keys will be returned. Defaults to None.
            index_name: Optional name of the index to search. Overrides the
                default index_name.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        if metadata_keys and self._vector_key not in metadata_keys:
            metadata_keys = [self._vector_key] + metadata_keys

        docs = self.similarity_search_by_vector(
            embedding,
            k=fetch_k,
            metadata_keys=metadata_keys,
            index_name=index_name,
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [doc.metadata[self._vector_key] for doc in docs],
            k=k,
            lambda_mult=lambda_mult,
        )

        if metadata_keys and self._vector_key in metadata_keys:
            for i in mmr_selected:
                docs[i].metadata.pop(self._vector_key)

        return [docs[i] for i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        metadata_keys: Optional[List[str]] = None,
        index_name: Optional[str] = None,
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
            index_name: Name of the index to search.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult,
            metadata_keys=metadata_keys,
            index_name=index_name,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Client = None,
        namespace: str = "test",
        index_name: Optional[str] = None,
        ids: Optional[List[str]] = None,
        embeddings_chunk_size: int = 1000,
        client_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> Aerospike:
        """
        This is a user friendly interface that:
            1. Embeds text.
            2. Converts the texts into documents.
            3. Adds the documents to a provided Aerospike index

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Aerospike
                from langchain_openai import OpenAIEmbeddings
                from aerospike_vector_search import Client, HostPort

                client = Client(seeds=HostPort(host="localhost", port=5000))
                aerospike = Aerospike.from_texts(
                    ["foo", "bar", "baz"],
                    embedder,
                    client,
                    "namespace",
                    index_name="index",
                    vector_key="vector",
                    distance_strategy=MODEL_DISTANCE_CALC,
                )
        """
        aerospike = cls(
            client,
            embedding,
            namespace,
            **kwargs,
        )

        aerospike.add_texts(
            texts,
            metadatas=metadatas,
            ids=ids,
            index_name=index_name,
            embedding_chunk_size=embeddings_chunk_size,
            **(client_kwargs or {}),
        )
        return aerospike
