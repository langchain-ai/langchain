import contextlib
import datetime
import uuid
import warnings
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import ColumnExpressionArgument, ForeignKey, Row, delete, select, text
from sqlalchemy.dialects.postgresql import DATE, JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

FilterType = Optional[Dict[str, Union[str, bool, Dict[str, List[Union[str, bool]]]]]]


class Base(DeclarativeBase):
    pass


class CollectionStore(Base):
    __tablename__ = "langchain_pg_collection_store"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        nullable=False,
        unique=True,
        index=True,
    )
    cmetadata: Mapped[Dict[Any, Any]] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DATE,
        nullable=False,
        default=datetime.datetime.utcnow,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DATE,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )
    embeddings = relationship(
        "EmbeddingStore",
        back_populates="collection",
        cascade="all",
    )


class EmbeddingStore(Base):
    __tablename__ = "langchain_pg_embedding_store"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            f"{CollectionStore.__tablename__}.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )
    embedding: Mapped[Vector] = mapped_column(
        Vector,
        nullable=False,
    )
    document: Mapped[str] = mapped_column(
        nullable=False,
    )
    cmetadata: Mapped[Dict[Any, Any]] = mapped_column(
        JSONB,
        nullable=True,
    )
    custom_id: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DATE,
        nullable=False,
        default=datetime.datetime.utcnow,
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DATE,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    collection = relationship(
        "CollectionStore",
        back_populates="embeddings",
    )


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

DEFAULT_COLLECTION_NAME = "langchain"


class PGVectorAsync(VectorStore):
    """`Postgres`/`PGVector` vector store.

    To use this vector store, you need to install the `pgvector`
    extension in your database.

    Args:
        embeddings (Embeddings): Any embedding function implementing
          `langchain.embeddings.base.Embeddings` interface.
        collection_name (str, optional): Name of the collection.
          Defaults to "langchain".
        collection_metadata (Optional[Dict[Any, Any]], optional):
          Metadata for the collection. Defaults to None.
        distance_strategy (DistanceStrategy, optional): Distance
          strategy to use. Defaults to DistanceStrategy.COSINE.
        engine (Optional[AsyncEngine], optional): Async engine
          to use. Defaults to None.
        db_url (Optional[str], optional): Database URL to use.
          Defaults to None.
        engine_kwargs (Optional[Dict[str, Any]], optional): Keyword
          arguments for the engine. Defaults to None.
        relevance_score_fn (Optional[Callable[[float], float]], optional):
            Relevance score function to use. Defaults to None.

    Note:
        Schema is not created automatically. You need to call
          `create_schema` method to create the schema.
        In this way, you can integrate the schema creation with
          your database migrations.

    Raises:
        ValueError: If both `engine` and `db_url` are provided or
          if none of them are provided.

    Example:
        .. code-block:: python

            from langchain.vectorstores import PGVectorAsync
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "postgresql+asyncpg://hwc@localhost:5432/test3"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = await PGVector.afrom_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                db_url=DATABASE_URL,
            )
    """

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        if engine is not None and db_url is None:
            _engine = engine

        elif engine is None and db_url is not None:
            _engine = create_async_engine(db_url, **(engine_kwargs or {}))
        else:
            raise ValueError(
                "Exactly one of engine, session or db_url must be provided."
            )

        self.engine = _engine
        self.session_factory = async_sessionmaker(
            bind=_engine,
            expire_on_commit=False,
        )
        self.embedding_function = embeddings
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata or {}
        self.EmbeddingStore = EmbeddingStore
        self.CollectionStore = CollectionStore
        self._distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn

    async def create_schema(self) -> None:
        """Create the schema for the vector store."""
        async with self.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)

    async def drop_schema(self) -> None:
        """Drop the schema for the vector store."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def get_collection(self) -> Optional[CollectionStore]:
        """Get a collection."""
        async with self._make_session() as session:
            query = select(CollectionStore).filter(
                CollectionStore.name == self.collection_name
            )
            collection = (await session.execute(query)).scalars().first()
            return collection

    async def create_collection(self) -> CollectionStore:
        """Create a collection if it does not exist."""
        async with self._make_session() as session:
            query = select(CollectionStore).filter(
                CollectionStore.name == self.collection_name
            )
            collection = (await session.execute(query)).scalars().first()

            if collection is None:
                collection = CollectionStore(
                    name=self.collection_name,
                    cmetadata=self.collection_metadata,
                )
                session.add(collection)
                await session.commit()

            return collection

    async def delete_collection(self) -> None:
        """Delete a collection."""
        async with self._make_session() as session:
            await session.execute(
                delete(CollectionStore).filter(
                    CollectionStore.name == self.collection_name
                )
            )
            await session.commit()

    async def delete_collection_embeddings(self) -> None:
        """Delete all embeddings in a collection."""
        async with self._make_session() as session:
            await session.execute(
                delete(EmbeddingStore).filter(
                    EmbeddingStore.collection_id == self.collection_name
                )
            )
            await session.commit()

    @contextlib.asynccontextmanager
    async def _make_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a new session. and close it when done."""
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by custom id.

        Args:
            ids (Optional[List[str]], optional): List of custom ids
              to delete. Defaults to None.
        """
        if ids is None:
            return

        async with self._make_session() as session:
            await session.execute(
                delete(EmbeddingStore).filter(EmbeddingStore.custom_id.in_(ids))
            )
            await session.commit()

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vector store.

        Args:
            texts (Iterable[str]): List of texts to add.
            embeddings (List[List[float]]): List of embeddings to add.
            metadatas (Optional[List[Dict[Any, Any]]], optional): List
              of metadata to add. Defaults to None.
            ids (Optional[List[str]], optional): List of custom ids to add.
              Defaults to None.

        Returns:
            List[str]: List of custom ids of the added embeddings.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        collection = await self.get_collection()

        if collection is None:
            collection = await self.create_collection()

        async with self._make_session() as session:
            for document, embedding, metadata, id in zip(
                texts,
                embeddings,
                metadatas,
                ids,
            ):
                embedding = EmbeddingStore(
                    collection_id=collection.id,
                    embedding=embedding,  # type: ignore
                    document=document,
                    cmetadata=metadata,
                    custom_id=id,
                )
                session.add(embedding)

            await session.commit()

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: List[Dict[Any, Any]] | None = None,
        ids: List[str] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts (Iterable[str]): List of texts to add.
            metadatas (List[Dict[Any, Any]], optional): List of metadata to add.
              Defaults to None.
            ids (List[str], optional): List of custom ids to add. Defaults to None.

        Returns:
            List[str]: List of custom ids of the added embeddings.
        """

        embeddigns = await self.embeddings.aembed_documents(list(texts))
        return await self.aadd_embeddings(
            texts=texts,
            embeddings=embeddigns,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: List[str] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents (List[Document]): List of documents to add.
            ids (List[str], optional): List of custom ids to add. Defaults to None.

        Returns:
            List[str]: List of custom ids of the added embeddings.
        """

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]  # type: ignore

        return await self.aadd_texts(
            texts=texts,
            metadatas=metadatas,  # type: ignore
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def __from(
        cls,
        texts: Iterable[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: List[Dict[Any, Any]] | None = None,
        ids: List[str] | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        vectorstore = cls(
            embeddings=embedding,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            engine=engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            relevance_score_fn=relevance_score_fn,
        )

        if pre_delete_collection:
            await vectorstore.delete_collection()

        await vectorstore.aadd_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

        return vectorstore

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: List[Dict[Any, Any]] | None = None,
        ids: List[str] | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        """Create a vector store from a list of texts.

        Args:
            texts (List[str]): List of texts to add.
            embedding (Embeddings): Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            metadatas (List[Dict[Any, Any]], optional): List of metadata to add.
              Defaults to None.
            ids (List[str], optional): List of custom ids to add. Defaults to None.
            collection_name (str, optional): Name of the collection. Defaults to
              "langchain".
            collection_metadata (Optional[Dict[Any, Any]], optional): Metadata for
              the collection. Defaults to None.
            distance_strategy (DistanceStrategy, optional): Distance strategy to use.
              Defaults to DistanceStrategy.COSINE.
            engine (Optional[AsyncEngine], optional): Async engine to use.
              Defaults to None.
            db_url (Optional[str], optional): Database URL to use. Defaults to None.
            engine_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for
              the engine. Defaults to None.
            pre_delete_collection (bool, optional): Whether to delete the collection
              before creating it. Defaults to False.
            relevance_score_fn (Optional[Callable[[float], float]], optional):
                Relevance score function to use. Defaults to None.

        Returns:
            PGVectorAsync: PGVectorAsync instance.
        """
        embeddings = await embedding.aembed_documents(texts)
        return await cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            engine=engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            pre_delete_collection=pre_delete_collection,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: List[str] | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        """Create a vector store from a list of documents.

        Args:
            documents (List[Document]): List of documents to add.
            embedding (Embeddings): Any embedding function implementing
              `langchain.embeddings.base.Embeddings` interface.
            ids (List[str], optional): List of custom ids to add. Defaults to None.
            collection_name (str, optional): Name of the collection. Defaults to
              "langchain".
            collection_metadata (Optional[Dict[Any, Any]], optional): Metadata for
              the collection. Defaults to None.
            distance_strategy (DistanceStrategy, optional): Distance strategy to use.
              Defaults to DistanceStrategy.COSINE.
            engine (Optional[AsyncEngine], optional): Async engine to use.
              Defaults to None.
            db_url (Optional[str], optional): Database URL to use. Defaults to None.
            engine_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for
              the engine. Defaults to None.
            pre_delete_collection (bool, optional): Whether to delete the collection
              before creating it. Defaults to False.
            relevance_score_fn (Optional[Callable[[float], float]], optional):
                Relevance score function to use. Defaults to None.


        Returns:
            PGVectorAsync: PGVectorAsync instance.
        """
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]  # type: ignore

        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,  # type: ignore
            ids=ids,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            engine=engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            pre_delete_collection=pre_delete_collection,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )

    @classmethod
    async def afrom_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        """Create a vector store from a list of text embeddings.

        Args:
            text_embeddings (List[Tuple[str, List[float]]]):
              List of text embeddings to add.
            embedding (Embeddings): Any embedding function
              implementing `langchain.embeddings.base.Embeddings`
                interface.
            metadatas (Optional[List[Dict[Any, Any]]], optional):
              List of metadata to add. Defaults to None.
            ids (Optional[List[str]], optional): List of custom ids
              to add. Defaults to None.
            collection_name (str, optional): Name of the collection.
              Defaults to "langchain".
            collection_metadata (Optional[Dict[Any, Any]], optional):
              Metadata for the collection. Defaults to None.
            distance_strategy (DistanceStrategy, optional):
              Distance strategy to use. Defaults to DistanceStrategy.COSINE.
            engine (Optional[AsyncEngine], optional): Async engine to use.
              Defaults to None.
            db_url (Optional[str], optional): Database URL to use.
              Defaults to None.
            engine_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments
              for the engine. Defaults to None.
            pre_delete_collection (bool, optional): Whether to delete the collection
              before creating it. Defaults to False.
            relevance_score_fn (Optional[Callable[[float], float]], optional):
                Relevance score function to use. Defaults to None.

        Returns:
            PGVectorAsync: PGVectorAsync instance.
        """

        texts = [text_content for text_content, _ in text_embeddings]
        embeddings = [embedding for _, embedding in text_embeddings]

        return await cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            engine=engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            pre_delete_collection=pre_delete_collection,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )

    @classmethod
    async def afrom_existing_index(
        cls,
        embedding: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[Dict[Any, Any]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        engine: Optional[AsyncEngine] = None,
        db_url: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        """Create a vector store from an existing index.

        Args:
            embedding (Embeddings): Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            collection_name (str, optional): Name of the collection.
              Defaults to "langchain".
            collection_metadata (Optional[Dict[Any, Any]], optional): Metadata for
              the collection. Defaults to None.
            distance_strategy (DistanceStrategy, optional): Distance strategy to use.
              Defaults to DistanceStrategy.COSINE.
            engine (Optional[AsyncEngine], optional): Async engine to use.
              Defaults to None.
            db_url (Optional[str], optional): Database URL to use. Defaults to None.
            engine_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for
              the engine. Defaults to None.
            pre_delete_collection (bool, optional): Whether to delete the collection
              before creating it. Defaults to False.
            relevance_score_fn (Optional[Callable[[float], float]], optional):
                Relevance score function to use. Defaults to None.

        Returns:
            PGVectorAsync: PGVectorAsync instance.
        """
        vectorstore = cls(
            embeddings=embedding,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
            distance_strategy=distance_strategy,
            engine=engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            relevance_score_fn=relevance_score_fn,
        )

        if pre_delete_collection:
            await vectorstore.delete_collection()

        return vectorstore

    @property
    def distance_strategy(self) -> Any:
        """Return the distance strategy
        implementation provided by `pgvector` extension.

        Returns:
            Any: Distance strategy implementation.
        """

        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: List[Dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError(
            "This method is not implemented for async vector stores."
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError(
            "This method is not implemented for async vector stores."
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: List[Dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> "PGVectorAsync":
        raise NotImplementedError(
            "This method is not implemented for async vector stores."
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query.

        Args:
            query (str): Query text.
            k (int, optional): Number of results to return.
              Defaults to 4.
            filter: Filter to apply to the metadata.
              Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query.
        """
        embedding = await self.embeddings.aembed_query(query)
        return await self.asimilarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to the query with scores.

        Args:
            query (str): Query text.
            k (int, optional): Number of results to return. Defaults to 4.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar
              to the query with scores.
        """

        embedding = await self.embeddings.aembed_query(query)
        results = await self.asimilaryty_serarch_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        return results

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the embedding.

        Args:
            embedding (List[float]): Embedding to use for the search.
            k (int, optional): Number of results to return. Defaults to 4.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the embedding.
        """
        docs_and_scores = await self.asimilaryty_serarch_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

        return self._docs_from_docs_and_scores(docs_and_scores)

    async def asimilaryty_serarch_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: FilterType = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to the embedding with scores.

        Args:
            embedding (List[float]): Embedding to use for the search.
            k (int, optional): Number of results to return. Defaults to 4.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents
              most similar to the embedding with scores.
        """

        results = await self._query_collection(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        return self._results_to_docs_and_scores(results)

    def _results_to_docs_and_scores(
        self, results: Sequence[Row[Tuple[EmbeddingStore, float]]]
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.distance,
            )
            for result in results
        ]
        return docs

    def _docs_from_docs_and_scores(
        self, docs_and_scores: List[Tuple[Document, float]]
    ) -> List[Document]:
        """Return docs from docs and scores."""
        return [doc for doc, _ in docs_and_scores]

    async def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: FilterType = None,
    ) -> Sequence[Row[Tuple[EmbeddingStore, float]]]:
        """Query the collection.

        Args:
            embedding (List[float]): Embedding to use for the search.
            k (int, optional): Number of results to return. Defaults to 4.
            filter: Filter to apply to the metadata. Defaults to None.

            Returns:
                Sequence[Row[Tuple[EmbeddingStore, float]]]: Results of the query.

            Raises:
                ValueError: If the collection does not exist.
                ValueError: If the filter is not valid.
        """

        async with self._make_session() as session:
            collection = await self.get_collection()
            if collection is None:
                raise ValueError("Collection does not exist.")

            filter_by = self.EmbeddingStore.collection_id == collection.id

            if filter is not None:
                filter_clauses: list[ColumnExpressionArgument[bool]] = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = (
                            self.EmbeddingStore.cmetadata[key]
                            .as_string()
                            .in_(value_case_insensitive[IN])
                        )
                        filter_clauses.append(filter_by_metadata)
                    else:
                        if isinstance(value, bool):
                            filter_by_metadata = (
                                self.EmbeddingStore.cmetadata[key].as_boolean() == value
                            )
                        elif isinstance(value, str):
                            filter_by_metadata = (
                                self.EmbeddingStore.cmetadata[key].as_string() == value
                            )
                        else:
                            raise ValueError(
                                f"Got unexpected value for filter: {value}. "
                                "Nested dictionary filters are only supported "
                                "for the 'in' operator."
                            )

                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            query = (
                select(  # type: ignore
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),
                )
                .filter(filter_by)
                .order_by(sqlalchemy.asc(text("distance")))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.id,
                )
                .limit(k)
            )

            results = await session.execute(query)
            return results.all()

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance algorithm.

        Maximal marginal relevance optimizes for similarity to
        query and dissimilarity to already selected documents.

        Args:
            query (str): Query text.
            k (int, optional): Number of Documents to return.
              Defaults to 4.
            fetch_k (int, optional): Number of Documents to fetch
              and pass to MMR algorithm. Defaults to 20.
            lambda_mult (float, optional): Lambda multiplier.
              Number between and 1 that determines the degree
              of diversity among the results with 0 corresponding to
              maximum diversity and 1 to minimum diversity.
              Defaults to 0.5.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Document]: List of documents selected using the
              maximal marginal relevance algorithm.
        """
        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal
        relevance algorithm with scores.

        Maximal marginal relevance optimizes for similarity
        to query and dissimilarity to already selected documents.

        Args:
            query (str): Query text.
            k (int, optional): Number of Documents to return.
              Defaults to 4.
            fetch_k (int, optional): Number of Documents to
              fetch and pass to MMR algorithm.
              Defaults to 20.
            lambda_mult (float, optional): Lambda multiplier.
              Number between and 1 that determines the degree
              of diversity among the results with 0 corresponding to
              maximum diversity and 1 to minimum diversity.
              Defaults to 0.5.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents selected
            using the maximal marginal relevance algorithm with scores.
        """

        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance algorithm.

        Maximal marginal relevance optimizes for similarity
        to query and dissimilarity to already selected documents.

        Args:
            embedding (List[float]): Embedding to use for the search.
            k (int, optional): Number of Documents to return. Defaults to 4.
            fetch_k (int, optional): Number of Documents to fetch and
              pass to MMR algorithm. Defaults to 20.
            lambda_mult (float, optional): Lambda multiplier.
              Number between and 1 that determines the degree of diversity
              among the results with 0 corresponding to
              maximum diversity and 1 to minimum diversity.
              Defaults to 0.5.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Document]: List of documents selected using
            the maximal marginal relevance algorithm.
        """

        results = await self._query_collection(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )

        embedding_list: List[Vector] = [
            result.EmbeddingStore.embedding for result in results
        ]

        mmr_selected = maximal_marginal_relevance(
            query_embedding=np.array(embedding, dtype=np.float32),
            embedding_list=embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [doc for i, (doc, _) in enumerate(candidates) if i in mmr_selected]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: FilterType = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal
        relevance algorithm with scores.

        Maximal marginal relevance optimizes for similarity
        to query and dissimilarity to already selected documents.

        Args:
            embedding (List[float]): Embedding to use for the search.
            k (int, optional): Number of Documents to return.
              Defaults to 4.
            fetch_k (int, optional): Number of Documents to fetch and
              pass to MMR algorithm. Defaults to 20.
            lambda_mult (float, optional): Lambda multiplier.
              Number between and 1 that determines the degree of diversity
              among the results with 0 corresponding to
              maximum diversity and 1 to minimum diversity.
              Defaults to 0.5.
            filter: Filter to apply to the metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents selected
            using the maximal marginal relevance algorithm with scores.
        """

        results = await self._query_collection(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )

        embedding_list: List[Vector] = [
            result.EmbeddingStore.embedding for result in results
        ]

        mmr_selected = maximal_marginal_relevance(
            query_embedding=np.array(embedding, dtype=np.float32),
            embedding_list=embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

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
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: FilterType = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to the query with relevance scores.

        Args:
            query (str): Query text.
            k (int, optional): Number of results to return. Defaults to 4.
            filter: Filter to apply to the metadata. Defaults to None.
            score_threshold (Optional[float], optional): Minimum score
                to return. Defaults to None.

        Warning:
            Relevance scores must be between 0 and 1.
            No results will be returned if the score threshold is too high.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar
                to the query with relevance scores.
        """

        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

        docs_and_similarities = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]

        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    f"Score threshold of {score_threshold} resulted in no results."
                )

        return docs_and_similarities
