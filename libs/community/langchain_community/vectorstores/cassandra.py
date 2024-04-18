from __future__ import annotations

import typing
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

if typing.TYPE_CHECKING:
    from cassandra.cluster import Session

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

CVST = TypeVar("CVST", bound="Cassandra")

_NOT_SET = object()


class Cassandra(VectorStore):
    """Wrapper around Apache Cassandra(R) for vector-store workloads.

    To use it, you need a recent installation of the `cassio` library
    and a Cassandra cluster / Astra DB instance supporting vector capabilities.

    Visit the cassio.org website for extensive quickstarts and code examples.

    Example:
        .. code-block:: python

                from langchain_community.vectorstores import Cassandra
                from langchain_community.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                session = ...             # create your Cassandra session object
                keyspace = 'my_keyspace'  # the keyspace should exist already
                table_name = 'my_vector_store'
                vectorstore = Cassandra(embeddings, session, keyspace, table_name)

    Args:
        embedding: Embedding function to use.
        session: Cassandra driver session.
        keyspace: Cassandra key space.
        table_name: Cassandra table.
        ttl_seconds: Optional time-to-live for the added texts.
        body_index_options: Optional options used to create the body index.
            Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
    """

    _embedding_dimension: Union[int, None]

    def _get_embedding_dimension(self) -> int:
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
        ttl_seconds: Optional[int] = None,
        *,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
    ) -> None:
        try:
            from cassio.table import MetadataVectorCassandraTable
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
        kwargs = {}
        if body_index_options is not None:
            kwargs["body_index_options"] = body_index_options

        self.table = MetadataVectorCassandraTable(
            session=session,
            keyspace=keyspace,
            table=table_name,
            vector_dimension=self._get_embedding_dimension(),
            metadata_indexing="all",
            primary_key_type="TEXT",
            **kwargs,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The underlying VectorTable already returns a "score proper",
        i.e. one in [0, 1] where higher means more *similar*,
        so here the final score transformation is not reversing the interval:
        """
        return lambda score: score

    def delete_collection(self) -> None:
        """
        Just an alias for `clear`
        (to better align with other VectorStore implementations).
        """
        self.clear()

    def clear(self) -> None:
        """Empty the table."""
        self.table.clear()

    def delete_by_document_id(self, document_id: str) -> None:
        return self.table.delete(row_id=document_id)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.


        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        for document_id in ids:
            self.delete_by_document_id(document_id)
        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        ttl_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Texts to add to the vectorstore.
            metadatas: Optional list of metadatas.
            ids: Optional list of IDs.
            batch_size: Number of concurrent requests to send to the server.
            ttl_seconds: Optional time-to-live for the added texts.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        _texts = list(texts)  # lest it be a generator or something
        if ids is None:
            ids = [uuid.uuid4().hex for _ in _texts]
        if metadatas is None:
            metadatas = [{} for _ in _texts]
        #
        ttl_seconds = ttl_seconds or self.ttl_seconds
        #
        embedding_vectors = self.embedding.embed_documents(_texts)
        #
        for i in range(0, len(_texts), batch_size):
            batch_texts = _texts[i : i + batch_size]
            batch_embedding_vectors = embedding_vectors[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            futures = [
                self.table.put_async(
                    row_id=text_id,
                    body_blob=text,
                    vector=embedding_vector,
                    metadata=metadata or {},
                    ttl_seconds=ttl_seconds,
                )
                for text, embedding_vector, text_id, metadata in zip(
                    batch_texts, batch_embedding_vectors, batch_ids, batch_metadatas
                )
            ]
            for future in futures:
                future.result()
        return ids

    # id-returning search facilities
    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            body_search: Document textual search terms to apply.
                Only supported by Astra DB at the moment.
        Returns:
            List of (Document, score, id), the most similar to the query vector.
        """
        kwargs: Dict[str, Any] = {}
        if filter is not None:
            kwargs["metadata"] = filter
        if body_search is not None:
            kwargs["body_search"] = body_search
        #
        hits = self.table.metric_ann_search(
            vector=embedding,
            n=k,
            metric="cos",
            **kwargs,
        )
        # We stick to 'cos' distance as it can be normalized on a 0-1 axis
        # (1=most relevant), as required by this class' contract.
        return [
            (
                Document(
                    page_content=hit["body_blob"],
                    metadata=hit["metadata"],
                ),
                0.5 + 0.5 * hit["distance"],
                hit["row_id"],
            )
            for hit in hits
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[Document, float, str]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
            body_search=body_search,
        )

    # id-unaware search facilities
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            body_search: Document textual search terms to apply.
                Only supported by Astra DB at the moment.
        Returns:
            List of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, docId) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                body_search=body_search,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k,
            filter=filter,
            body_search=body_search,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
                body_search=body_search,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[Document, float]]:
        embedding_vector = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding_vector,
            k,
            filter=filter,
            body_search=body_search,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter: Filter on the metadata to apply.
            body_search: Document textual search terms to apply.
                Only supported by Astra DB at the moment.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        _kwargs: Dict[str, Any] = {}
        if filter is not None:
            _kwargs["metadata"] = filter
        if body_search is not None:
            _kwargs["body_search"] = body_search

        prefetch_hits = list(
            self.table.metric_ann_search(
                vector=embedding,
                n=fetch_k,
                metric="cos",
                **_kwargs,
            )
        )
        # let the mmr utility pick the *indices* in the above array
        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [pf_hit["vector"] for pf_hit in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_hits = [
            pf_hit
            for pf_index, pf_hit in enumerate(prefetch_hits)
            if pf_index in mmr_chosen_indices
        ]
        return [
            Document(
                page_content=hit["body_blob"],
                metadata=hit["metadata"],
            )
            for hit in mmr_hits
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        body_search: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter: Filter on the metadata to apply.
            body_search: Document textual search terms to apply.
                Only supported by Astra DB at the moment.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            body_search=body_search,
        )

    @classmethod
    def from_texts(
        cls: Type[CVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        session: Session = _NOT_SET,
        keyspace: str = "",
        table_name: str = "",
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from raw texts.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            session: Cassandra driver session (required).
            keyspace: Cassandra key space (required).
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the texts.
            batch_size: Number of concurrent requests to send to the server.
                Defaults to 16.
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]

        Returns:
            a Cassandra vectorstore.
        """
        if session is _NOT_SET:
            raise ValueError("session parameter is required")
        if not keyspace:
            raise ValueError("keyspace parameter is required")
        if not table_name:
            raise ValueError("table_name parameter is required")
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
        )
        store.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, batch_size=batch_size
        )
        return store

    @classmethod
    def from_documents(
        cls: Type[CVST],
        documents: List[Document],
        embedding: Embeddings,
        *,
        session: Session = _NOT_SET,
        keyspace: str = "",
        table_name: str = "",
        ids: Optional[List[str]] = None,
        batch_size: int = 16,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        **kwargs: Any,
    ) -> CVST:
        """Create a Cassandra vectorstore from a document list.

        Args:
            documents: Documents to add to the vectorstore.
            embedding: Embedding function to use.
            session: Cassandra driver session (required).
            keyspace: Cassandra key space (required).
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the documents.
            batch_size: Number of concurrent requests to send to the server.
                Defaults to 16.
            ttl_seconds: Optional time-to-live for the added documents.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]

        Returns:
            a Cassandra vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ids=ids,
            batch_size=batch_size,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            **kwargs,
        )
