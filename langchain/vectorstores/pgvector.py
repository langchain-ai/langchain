"""VectorStore wrapper around a Postgres/PGVector database."""
from __future__ import annotations

import enum
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import sqlalchemy
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session, declarative_base

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    from langchain.vectorstores._pgvector_data_models import CollectionStore


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class PGVector(VectorStore):
    """VectorStore implementation using Postgres and pgvector.

    To use, you should have the ``pgvector`` python package installed.

    Args:
        connection_string: Postgres connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: COSINE)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain.vectorstores import PGVector
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "postgresql+psycopg2://hwc@localhost:5432/test3"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = PGVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )


    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        self._conn = self.connect()
        # self.create_vector_extension()
        from langchain.vectorstores._pgvector_data_models import (
            CollectionStore,
            EmbeddingStore,
        )

        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        self.create_tables_if_not_exists()
        self.create_collection()

    def connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        return conn

    def create_vector_extension(self) -> None:
        try:
            with Session(self._conn) as session:
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
                session.execute(statement)
                session.commit()
        except Exception as e:
            self.logger.exception(e)

    def create_tables_if_not_exists(self) -> None:
        with self._conn.begin():
            Base.metadata.create_all(self._conn)

    def drop_tables(self) -> None:
        with self._conn.begin():
            Base.metadata.drop_all(self._conn)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with Session(self._conn) as session:
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()

    def get_collection(self, session: Session) -> Optional["CollectionStore"]:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        connection_string: Optional[str] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        if connection_string is None:
            connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = self.EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                    collection_id=collection.uuid,
                )
                session.add(embedding_store)
            session.commit()

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with PGVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == "l2":
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == "cosine":
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == "inner":
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of `l2`, `cosine`, `inner`."
            )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        with Session(self._conn) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = self.EmbeddingStore.collection_id == collection.uuid

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = self.EmbeddingStore.cmetadata[
                            key
                        ].astext.in_(value_case_insensitive[IN])
                        filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = self.EmbeddingStore.cmetadata[
                            key
                        ].astext == str(value)
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),  # type: ignore
                )
                .filter(filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        docs = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[PGVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """Construct PGVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from langchain import PGVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = PGVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[PGVector],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """
        Get intsance of an existing PGVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="PGVECTOR_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the PGVECTOR_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[PGVector],
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
        cls,
        driver: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"

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
        if self.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self.distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
