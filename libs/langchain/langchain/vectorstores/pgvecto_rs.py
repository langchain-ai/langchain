import uuid
import numpy as np
from typing import Any, Iterable, List, Optional

import sqlalchemy
from sqlalchemy import insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.orm.session import Session
from pgvecto_rs.sqlalchemy import Vector

from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.schema import Document


class _ORMBase(DeclarativeBase):
    pass


class PGVecto_rs(VectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        dimension: int,
        db_url: str,
        collection_name: str,
        new_table: bool = False,
    ) -> None:
        def define_table():
            class _Table(_ORMBase):
                __tablename__ = f"collection_{collection_name}"
                id: Mapped[uuid.UUID] = mapped_column(
                    postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
                )
                text: Mapped[str] = mapped_column(sqlalchemy.String)
                meta: Mapped[dict] = mapped_column(postgresql.JSONB)
                embedding: Mapped[np.ndarray] = mapped_column(Vector(dimension))

            return _Table

        self._engine = sqlalchemy.create_engine(db_url)
        self._table = define_table()
        self._table.__table__.create(self._engine, checkfirst=not new_table)
        self._embedding = embedding
        print("Initialized PGVecto_rs with collection name: " + collection_name)

    @classmethod
    def from_texts(
        cls: "PGVecto_rs",
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        db_url: str = "",
        collection_name: str = str(uuid.uuid4().hex),
        **kwargs: Any,
    ) -> "PGVecto_rs":
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
        cls: "PGVecto_rs",
        documents: List[Document],
        embedding: Embeddings,
        db_url: str = "",
        collection_name: str = str(uuid.uuid4().hex),
        **kwargs: Any,
    ) -> "PGVecto_rs":
        """Return VectorStore initialized from documents."""
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        return cls.from_texts(
            texts, embedding, metadatas, db_url, collection_name, **kwargs
        )

    @classmethod
    def from_collection_name(
        cls: "PGVecto_rs",
        embedding: Embeddings,
        db_url: str,
        collection_name: str,
    ) -> "PGVecto_rs":
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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        """
        embeddings = self._embedding.embed_documents(texts)
        with Session(self._engine) as _session:
            for text, embedding, metadata in zip(
                texts, embeddings, metadatas or [dict()] * len(texts)
            ):
                t = insert(self._table).values(
                    text=text, meta=metadata, embedding=embedding
                )
                _session.execute(t)
            _session.commit()

    def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
        """
        return self.add_texts(
            [document.page_content for document in documents],
            [document.metadata for document in documents],
            **kwargs,
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        with Session(self._engine) as _session:
            query_embedding = self._embedding.embed_documents([query])[0]
            t = (
                select(self._table)
                .order_by(
                    self._table.embedding.squared_euclidean_distance(query_embedding)
                )
                .limit(k)
            )
            return [
                Document(page_content=row[0].text, metadata=row[0].meta)
                for row in _session.execute(t)
            ]
