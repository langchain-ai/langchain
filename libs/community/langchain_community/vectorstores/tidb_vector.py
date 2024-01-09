import contextlib
import enum
import logging
import numpy as np
import uuid
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Tuple

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy.orm import Session, declarative_base

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger()


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DEFAULT_COLLECTION_NAME = "langchain_tidb_vector"

Base = declarative_base()  # type: Any
_classes: Any = None


def _create_vector_model(table_name: str):
    global _classes
    if _classes is not None:
        return _classes

    class SQLVectorModel(Base):
        __tablename__ = table_name
        id = sqlalchemy.Column(
            sqlalchemy.BINARY(16),
            primary_key=True,
            default=lambda: uuid.uuid4().bytes,
        )
        embedding = sqlalchemy.Column(Vector(None))
        document = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)

    return SQLVectorModel


class TiDBVec(VectorStore):
    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = "DEFAULT_COLLECTION_NAME",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self._embedding_function = embedding_function
        self._distance_strategy = distance_strategy
        self._engine_args = engine_args or {}
        self._bind = self._create_engine()
        self._table_model = _create_vector_model(collection_name)
        self.create_table_if_not_exists()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding_function

    def create_table_if_not_exists(self) -> None:
        with Session(self._bind) as session, session.begin():
            Base.metadata.create_all(session.get_bind())
            # wait for tidb support vector index

    def drop_table(self) -> None:
        with Session(self._bind) as session, session.begin():
            Base.metadata.drop_all(session.get_bind())

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(url=self.connection_string, **self._engine_args)

    def __del__(self) -> None:
        if isinstance(self._bind, sqlalchemy.engine.Connection):
            self._bind.close()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._bind)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        connection_string: str,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        *,
        engine_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        embeddings = embedding.embed_documents(list(texts))

        tidb_vs = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            **kwargs,
        )

        tidb_vs.add_texts(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return tidb_vs

    @classmethod
    def from_existing_collection(
        cls,
        embedding: Embeddings,
        connection_string: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        return cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            **kwargs,
        )
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self._embedding_function.embed_documents(list(texts))
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._bind) as session:
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embeded_doc = self._table_model(
                    id=id,
                    embedding=embedding,
                    document=text,
                    meta=metadata,
                )
                session.add(embeded_doc)
            session.commit()

        return ids
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        with Session(self._bind) as session:
            if ids is not None:
                stmt = sqlalchemy.delete(self._table_model).where(
                    self._table_model.id.in_(ids)
                )

                session.execute(stmt)
            session.commit()

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        result = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in result]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_vector = self._embedding_function.embed_query(query)
        relevant_docs = self._query_collection(query_vector, k, filter)
        return [
            (Document(
                page_content=doc._table_model.document,
                metadata=doc._table_model.mata,
            ), doc.distance)
            for doc in relevant_docs
        ]

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._table_model.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self._table_model.embedding.cosion_distance
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        with Session(self._bind) as session:
            results: List[Any] = (
                session.query(
                    self._table_model,
                    self.distance_strategy(embedding).label("distance"),  # type: ignore
                )
                .order_by(sqlalchemy.asc("distance"))
                .limit(k)
                .all()
            )
        return results
        
