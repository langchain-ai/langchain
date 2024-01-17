from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class PGVecto_rs(VectorStore):
    """VectorStore backed by pgvecto_rs."""

    _store = None
    _embedding: Embeddings

    def __init__(
        self,
        embedding: Embeddings,
        dimension: int,
        db_url: str,
        collection_name: str,
        new_table: bool = False,
    ) -> None:
        """Initialize a PGVecto_rs vectorstore.

        Args:
            embedding: Embeddings to use.
            dimension: Dimension of the embeddings.
            db_url: Database URL.
            collection_name: Name of the collection.
            new_table: Whether to create a new table or connect to an existing one.
            If true, the table will be dropped if exists, then recreated.
            Defaults to False.
        """
        try:
            from pgvecto_rs.sdk import PGVectoRs
        except ImportError as e:
            raise ImportError(
                "Unable to import pgvector_rs.sdk , please install with "
                '`pip install "pgvector_rs[sdk]"`.'
            ) from e
        self._store = PGVectoRs(
            db_url=db_url,
            collection_name=collection_name,
            dimension=dimension,
            recreate=new_table,
        )
        self._embedding = embedding

    # ================ Create interface =================
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        db_url: str = "",
        collection_name: str = str(uuid.uuid4().hex),
        **kwargs: Any,
    ) -> PGVecto_rs:
        """Return VectorStore initialized from texts and optional metadatas."""
        sample_embedding = embedding.embed_query("Hello pgvecto_rs!")
        dimension = len(sample_embedding)
        if db_url is None:
            raise ValueError("db_url must be provided")
        _self: PGVecto_rs = cls(
            embedding=embedding,
            dimension=dimension,
            db_url=db_url,
            collection_name=collection_name,
        )
        _self.add_texts(texts, metadatas, **kwargs)
        return _self

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        db_url: str = "",
        collection_name: str = str(uuid.uuid4().hex),
        **kwargs: Any,
    ) -> PGVecto_rs:
        """Return VectorStore initialized from documents."""
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        return cls.from_texts(
            texts, embedding, metadatas, db_url, collection_name, **kwargs
        )

    @classmethod
    def from_collection_name(
        cls,
        embedding: Embeddings,
        db_url: str,
        collection_name: str,
    ) -> PGVecto_rs:
        """Create new empty vectorstore with collection_name.
        Or connect to an existing vectorstore in database if exists.
        Arguments should be the same as when the vectorstore was created."""
        sample_embedding = embedding.embed_query("Hello pgvecto_rs!")
        return cls(
            embedding=embedding,
            dimension=len(sample_embedding),
            db_url=db_url,
            collection_name=collection_name,
        )

    # ================ Insert interface =================

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids of the added texts.

        """
        from pgvecto_rs.sdk import Record

        embeddings = self._embedding.embed_documents(list(texts))
        records = [
            Record.from_text(text, embedding, meta)
            for text, embedding, meta in zip(texts, embeddings, metadatas or [])
        ]
        self._store.insert(records)
        return [str(record.id) for record in records]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.

        Returns:
            List of ids of the added documents.
        """
        return self.add_texts(
            [document.page_content for document in documents],
            [document.metadata for document in documents],
            **kwargs,
        )

    # ================ Query interface =================
    def similarity_search_with_score_by_vector(
        self,
        query_vector: List[float],
        k: int = 4,
        distance_func: Literal[
            "sqrt_euclid", "neg_dot_prod", "ned_cos"
        ] = "sqrt_euclid",
        filter: Union[None, Dict[str, Any], Any] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query vector, with its score."""

        from pgvecto_rs.sdk.filters import meta_contains

        distance_func_map = {
            "sqrt_euclid": "<->",
            "neg_dot_prod": "<#>",
            "ned_cos": "<=>",
        }
        if filter is None:
            real_filter = None
        elif isinstance(filter, dict):
            real_filter = meta_contains(filter)
        else:
            real_filter = filter
        results = self._store.search(
            query_vector,
            distance_func_map[distance_func],
            k,
            filter=real_filter,
        )

        return [
            (
                Document(
                    page_content=res[0].text,
                    metadata=res[0].meta,
                ),
                res[1],
            )
            for res in results
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        distance_func: Literal[
            "sqrt_euclid", "neg_dot_prod", "ned_cos"
        ] = "sqrt_euclid",
        filter: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score_by_vector(
                embedding, k, distance_func, **kwargs
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        distance_func: Literal[
            "sqrt_euclid", "neg_dot_prod", "ned_cos"
        ] = "sqrt_euclid",
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            query_vector, k, distance_func, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        distance_func: Literal[
            "sqrt_euclid", "neg_dot_prod", "ned_cos"
        ] = "sqrt_euclid",
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        query_vector = self._embedding.embed_query(query)
        return [
            doc
            for doc, _score in self.similarity_search_with_score_by_vector(
                query_vector, k, distance_func, **kwargs
            )
        ]
