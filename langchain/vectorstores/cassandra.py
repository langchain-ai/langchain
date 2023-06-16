"""Wrapper around Cassandra vector-store capabilities, based on cassIO."""
from __future__ import annotations

import hashlib
import typing
from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar

import numpy as np

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

CVST = TypeVar("CVST", bound="Cassandra")

# a positive number of seconds to expire entries, or None for no expiration.
CASSANDRA_VECTORSTORE_DEFAULT_TTL_SECONDS = None


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


class Cassandra(VectorStore):
    """Wrapper around Cassandra embeddings platform.

    There is no notion of a default table name, since each embedding
    function implies its own vector dimension, which is part of the schema.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Cassandra
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                session = ...
                keyspace = 'my_keyspace'
                vectorstore = Cassandra(embeddings, session, keyspace, 'my_doc_archive')
    """

    _embedding_dimension: int | None

    def _getEmbeddingDimension(self) -> int:
        if self._embedding_dimension is None:
            self._embedding_dimension = len(
                self.embedding.embed_query("This is a sample sentence.")
            )
        return self._embedding_dimension

    def __init__(
        self,
        embedding: Embeddings,
        session: Session,
        keyspace: str,
        table_name: str,
        ttl_seconds: int | None = CASSANDRA_VECTORSTORE_DEFAULT_TTL_SECONDS,
    ) -> None:
        try:
            from cassio.vector import VectorTable
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import cassio python package. "
                "Please install it with `pip install cassio`."
            )
        """Create a vector table."""
        self.embedding = embedding
        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        #
        self._embedding_dimension = None
        #
        self.table = VectorTable(
            session=session,
            keyspace=keyspace,
            table=table_name,
            embedding_dimension=self._getEmbeddingDimension(),
            auto_id=False,  # the `add_texts` contract admits user-provided ids
        )

    def delete_collection(self) -> None:
        """
        Just an alias for `clear`
        (to better align with other VectorStore implementations).
        """
        self.clear()

    def clear(self) -> None:
        """Empty the collection."""
        self.table.clear()

    def delete_by_document_id(self, document_id: str) -> None:
        return self.table.delete(document_id)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        _texts = list(texts)  # lest it be a generator or something
        if ids is None:
            # unless otherwise specified, we have deterministic IDs:
            # re-inserting an existing document will not create a duplicate.
            # (and effectively update the metadata)
            ids = [_hash(text) for text in _texts]
        if metadatas is None:
            metadatas = [{} for _ in _texts]
        #
        ttl_seconds = kwargs.get("ttl_seconds", self.ttl_seconds)
        #
        embedding_vectors = self.embedding.embed_documents(_texts)
        for text, embedding_vector, text_id, metadata in zip(
            _texts, embedding_vectors, ids, metadatas
        ):
            self.table.put(
                document=text,
                embedding_vector=embedding_vector,
                document_id=text_id,
                metadata=metadata,
                ttl_seconds=ttl_seconds,
            )
        #
        return ids

    # id-returning search facilities
    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        No support for `filter` query (on metadata) along with vector search.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        hits = self.table.search(
            embedding_vector=embedding,
            top_k=k,
            metric="cos",
            metric_threshold=None,
        )
        # We stick to 'cos' distance as it can be normalized on a 0-1 axis
        # (1=most relevant), as required by this class' contract.
        return [
            (
                Document(
                    page_content=hit["document"],
                    metadata=hit["metadata"],
                ),
                0.5 + 0.5 * hit["distance"],
                hit["document_id"],
            )
            for hit in hits
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
        )

    # id-unaware search facilities
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        No support for `filter` query (on metadata) along with vector search.

        Args:
            embedding (str): Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
        Returns:
            List of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, docId) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        #
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
        )

    # Even though this is a `_`-method,
    # it is apparently used by VectorSearch parent class
    # in an exposed method (`similarity_search_with_relevance_scores`).
    # So we implement it (hmm).
    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(
            query,
            k,
            **kwargs,
        )

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
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        prefetchHits = self.table.search(
            embedding_vector=embedding,
            top_k=fetch_k,
            metric="cos",
            metric_threshold=None,
        )
        # let the mmr utility pick the *indices* in the above array
        mmrChosenIndices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [pfHit["embedding_vector"] for pfHit in prefetchHits],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmrHits = [
            pfHit
            for pfIndex, pfHit in enumerate(prefetchHits)
            if pfIndex in mmrChosenIndices
        ]
        return [
            Document(
                page_content=hit["document"],
                metadata=hit["metadata"],
            )
            for hit in mmrHits
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Optional.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
        )

    @classmethod
    def from_texts(
        cls: Type[CVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from raw texts.

        No support for specifying text IDs

        Returns:
            a Cassandra vectorstore.
        """
        session: Session = kwargs["session"]
        keyspace: str = kwargs["keyspace"]
        table_name: str = kwargs["table_name"]
        cassandraStore = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
        )
        cassandraStore.add_texts(texts=texts, metadatas=metadatas)
        return cassandraStore

    @classmethod
    def from_documents(
        cls: Type[CVST],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from a document list.

        No support for specifying text IDs

        Returns:
            a Cassandra vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        session: Session = kwargs["session"]
        keyspace: str = kwargs["keyspace"]
        table_name: str = kwargs["table_name"]
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
        )
