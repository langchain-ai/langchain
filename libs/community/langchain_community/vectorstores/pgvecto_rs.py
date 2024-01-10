from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Literal, Optional, Tuple, Type

import numpy as np
import sqlalchemy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.orm.session import Session


class _ORMBase(DeclarativeBase):
    __tablename__: str
    id: Mapped[uuid.UUID]
    text: Mapped[str]
    meta: Mapped[dict]
    embedding: Mapped[np.ndarray]


class PGVecto_rs(VectorStore):
    """VectorStore backed by pgvecto_rs."""

    _engine: sqlalchemy.engine.Engine
    _table: Type[_ORMBase]
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
              Defaults to False.
        """
        try:
            from pgvecto_rs.sqlalchemy import Vector
        except ImportError as e:
            raise ImportError(
                "Unable to import pgvector_rs, please install with "
                "`pip install pgvector_rs`."
            ) from e

        class _Table(_ORMBase):
            __tablename__ = f"collection_{collection_name}"
            id: Mapped[uuid.UUID] = mapped_column(
                postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
            )
            text: Mapped[str] = mapped_column(sqlalchemy.String)
            meta: Mapped[dict] = mapped_column(postgresql.JSONB)
            embedding: Mapped[np.ndarray] = mapped_column(Vector(dimension))

        self._engine = sqlalchemy.create_engine(db_url)
        self._table = _Table
        self._table.__table__.create(self._engine, checkfirst=not new_table)  # type: ignore
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
            new_table=True,
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
        embeddings = self._embedding.embed_documents(list(texts))
        with Session(self._engine) as _session:
            results: List[str] = []
            for text, embedding, metadata in zip(
                texts, embeddings, metadatas or [dict()] * len(list(texts))
            ):
                t = insert(self._table).values(
                    text=text, meta=metadata, embedding=embedding
                )
                id = _session.execute(t).inserted_primary_key[0]  # type: ignore
                results.append(str(id))
            _session.commit()
            return results

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
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query vector, with its score."""
        with Session(self._engine) as _session:
            real_distance_func = (
                self._table.embedding.squared_euclidean_distance
                if distance_func == "sqrt_euclid"
                else self._table.embedding.negative_dot_product_distance
                if distance_func == "neg_dot_prod"
                else self._table.embedding.negative_cosine_distance
                if distance_func == "ned_cos"
                else None
            )
            if real_distance_func is None:
                raise ValueError("Invalid distance function")

            t = (
                select(self._table, real_distance_func(query_vector).label("score"))
                .order_by("score")
                .limit(k)  # type: ignore
            )
            return [
                (Document(page_content=row[0].text, metadata=row[0].meta), row[1])
                for row in _session.execute(t)
            ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        distance_func: Literal[
            "sqrt_euclid", "neg_dot_prod", "ned_cos"
        ] = "sqrt_euclid",
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, score in self.similarity_search_with_score_by_vector(
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
            for doc, score in self.similarity_search_with_score_by_vector(
                query_vector, k, distance_func, **kwargs
            )
        ]
