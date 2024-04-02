from __future__ import annotations

import contextlib
import enum
import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import sqlalchemy
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from sqlalchemy.sql import quoted_name

from langchain_community.vectorstores.utils import maximal_marginal_relevance

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class BaseEmbeddingStore:
    """Base class for the Lantern embedding store."""


def get_embedding_store(
    distance_strategy: DistanceStrategy, collection_name: str
) -> Any:
    """Get the embedding store class."""

    embedding_type = None

    if distance_strategy == DistanceStrategy.HAMMING:
        embedding_type = sqlalchemy.INTEGER  # type: ignore
    else:
        embedding_type = sqlalchemy.REAL  # type: ignore

    DynamicBase = declarative_base(class_registry=dict())  # type: Any

    class EmbeddingStore(DynamicBase, BaseEmbeddingStore):
        __tablename__ = collection_name
        uuid = sqlalchemy.Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
        )
        __table_args__ = {"extend_existing": True}
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata = sqlalchemy.Column(JSON, nullable=True)
        # custom_id : any user defined id
        custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        embedding = sqlalchemy.Column(sqlalchemy.ARRAY(embedding_type))  # type: ignore

    return EmbeddingStore


class QueryResult:
    """Result from a query."""

    EmbeddingStore: BaseEmbeddingStore
    distance: float


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2sq"
    COSINE = "cosine"
    HAMMING = "hamming"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


class Lantern(VectorStore):
    """`Postgres` with the `lantern` extension as a vector store.

    lantern uses sequential scan by default. but you can create a HNSW index
    using the create_hnsw_index method.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is the name of the table in which embedding data will be stored
            The table will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
        - `EUCLIDEAN` is the euclidean distance.
        - `COSINE` is the cosine distance.
        - `HAMMING` is the hamming distance.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
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
        self.EmbeddingStore = get_embedding_store(
            self.distance_strategy, collection_name
        )
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        self._conn = self.connect()
        self.create_hnsw_extension()
        self.create_collection()

    @property
    def distance_strategy(self) -> DistanceStrategy:
        if isinstance(self._distance_strategy, DistanceStrategy):
            return self._distance_strategy

        if self._distance_strategy == DistanceStrategy.EUCLIDEAN.value:
            return DistanceStrategy.EUCLIDEAN
        elif self._distance_strategy == DistanceStrategy.COSINE.value:
            return DistanceStrategy.COSINE
        elif self._distance_strategy == DistanceStrategy.HAMMING.value:
            return DistanceStrategy.HAMMING
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

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

    def connect(self) -> sqlalchemy.engine.Connection:
        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        return conn

    @property
    def distance_function(self) -> Any:
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "l2sq_dist"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            return "cos_dist"
        elif self.distance_strategy == DistanceStrategy.HAMMING:
            return "hamming_dist"

    def create_hnsw_extension(self) -> None:
        try:
            with Session(self._conn) as session:
                statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS lantern")
                session.execute(statement)
                session.commit()
        except Exception as e:
            self.logger.exception(e)

    def create_tables_if_not_exists(self) -> None:
        try:
            self.create_collection()
        except ProgrammingError:
            pass

    def drop_table(self) -> None:
        try:
            self.EmbeddingStore.__table__.drop(self._conn.engine)
        except ProgrammingError:
            pass

    def drop_tables(self) -> None:
        self.drop_table()

    def _hamming_relevance_score_fn(self, distance: float) -> float:
        return distance

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
        elif self.distance_strategy == DistanceStrategy.HAMMING:
            return self._hamming_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to Lantern constructor."
            )

    def _get_op_class(self) -> str:
        if self.distance_strategy == DistanceStrategy.COSINE:
            return "dist_cos_ops"
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "dist_l2sq_ops"
        elif self.distance_strategy == DistanceStrategy.HAMMING:
            return "dist_hamming_ops"
        else:
            raise ValueError(
                "No supported operator class"
                f" for distance_strategy of {self._distance_strategy}."
            )

    def _get_operator(self) -> str:
        if self.distance_strategy == DistanceStrategy.COSINE:
            return "<=>"
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return "<->"
        elif self.distance_strategy == DistanceStrategy.HAMMING:
            return "<+>"
        else:
            raise ValueError(
                "No supported operator"
                f" for distance_strategy of {self._distance_strategy}."
            )

    def _typed_arg_for_distance(
        self, embedding: List[Union[float, int]]
    ) -> List[Union[float, int]]:
        if self.distance_strategy == DistanceStrategy.HAMMING:
            return list(map(lambda x: int(x), embedding))
        return embedding

    @property
    def _index_name(self) -> str:
        return f"langchain_{self.collection_name}_idx"

    def create_hnsw_index(
        self,
        dims: int = ADA_TOKEN_COUNT,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        **_kwargs: Any,
    ) -> None:
        """Create HNSW index on collection.

        Optional Keyword Args for HNSW Index:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            ef: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 64

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 64

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

            dims: Dimensions of the vectors in collection. default: 1536
        """
        create_index_query = sqlalchemy.text(
            "CREATE INDEX IF NOT EXISTS {} "
            "ON {} USING hnsw (embedding {}) "
            "WITH ("
            "dim = :dim, "
            "m = :m, "
            "ef_construction = :ef_construction, "
            "ef = :ef"
            ");".format(
                quoted_name(self._index_name, True),
                quoted_name(self.collection_name, True),
                self._get_op_class(),
            )
        )

        with Session(self._conn) as session:
            # Create the HNSW index
            session.execute(
                create_index_query,
                {
                    "dim": dims,
                    "m": m,
                    "ef_construction": ef_construction,
                    "ef": ef_search,
                },
            )
            session.commit()
        self.logger.info("HNSW extension and index created successfully.")

    def drop_index(self) -> None:
        with Session(self._conn) as session:
            # Drop the HNSW index
            session.execute(
                sqlalchemy.text(
                    "DROP INDEX IF EXISTS {}".format(
                        quoted_name(self._index_name, True)
                    )
                )
            )
            session.commit()

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
            self.drop_table()

        with self._conn.begin():
            try:
                self.EmbeddingStore.__table__.create(self._conn.engine)
            except ProgrammingError as e:
                # Duplicate table
                if e.code == "f405":
                    pass
                else:
                    raise e

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        self.drop_table()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._conn)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
        """
        with Session(self._conn) as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )
                stmt = delete(self.EmbeddingStore).where(
                    self.EmbeddingStore.custom_id.in_(ids)
                )
                session.execute(stmt)
            session.commit()

    @classmethod
    def _initialize_from_embeddings(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Lantern:
        """
        Order of elements for lists `ids`, `embeddings`, `texts`, `metadatas`
        should match, so each row will be associated with correct values.

        Postgres connection string is required
        "Either pass it as `connection_string` parameter
        or set the LANTERN_CONNECTION_STRING environment variable.

        - `texts` texts to insert into collection.
        - `embeddings` an Embeddings to insert into collection
        - `embedding` is :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        - `metadatas` row metadata to insert into collection.
        - `ids` row ids to insert into collection.
        - `collection_name` is the name of the collection to use. (default: langchain)
            - NOTE: This is the name of the table in which embedding data will be stored
                The table will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
        - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
            - `EUCLIDEAN` is the euclidean distance.
            - `COSINE` is the cosine distance.
            - `HAMMING` is the hamming distance.
        - `pre_delete_collection` if True, will delete the collection if it exists.
            (default: False)
            - Useful for testing.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        connection_string = cls.__get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
            distance_strategy=distance_strategy,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        store.create_hnsw_index(**kwargs)

        return store

    def add_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        **kwargs: Any,
    ) -> None:
        with Session(self._conn) as session:
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = self.EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                )
                session.add(embedding_store)
            session.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._conn) as session:
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = self.EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                )
                session.add(embedding_store)
            session.commit()

        return ids

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
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

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
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
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_collection(embedding=embedding, k=k, filter=filter)

        return self._results_to_docs_and_scores(results)

    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Any]:
        with Session(self._conn) as session:
            set_enable_seqscan_stmt = sqlalchemy.text("SET enable_seqscan = off")
            set_init_k = sqlalchemy.text("SET hnsw.init_k = :k")
            session.execute(set_enable_seqscan_stmt)
            session.execute(set_init_k, {"k": k})

            filter_by = None
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

                filter_by = sqlalchemy.and_(*filter_clauses)

            embedding = self._typed_arg_for_distance(embedding)
            query = session.query(
                self.EmbeddingStore,
                getattr(func, self.distance_function)(
                    self.EmbeddingStore.embedding, embedding
                ).label("distance"),
            )  # Specify the columns you need here, e.g., EmbeddingStore.embedding

            if filter_by is not None:
                query = query.filter(filter_by)

            results: List[QueryResult] = (
                query.order_by(
                    self.EmbeddingStore.embedding.op(self._get_operator())(embedding)
                )  # Using PostgreSQL specific operator with the correct column name
                .limit(k)
                .all()
            )

        return results

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)

    @classmethod
    def from_texts(
        cls: Type[Lantern],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Lantern:
        """
        Initialize Lantern vectorstore from list of texts.
        The embeddings will be generated using `embedding` class provided.

        Order of elements for lists `ids`, `texts`, `metadatas` should match,
        so each row will be associated with correct values.

        Postgres connection string is required
        "Either pass it as `connection_string` parameter
        or set the LANTERN_CONNECTION_STRING environment variable.

        - `connection_string` is fully populated connection string for postgres database
        - `texts` texts to insert into collection.
        - `embedding` is :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        - `metadatas` row metadata to insert into collection.
        - `collection_name` is the name of the collection to use. (default: langchain)
            - NOTE: This is the name of the table in which embedding data will be stored
                The table will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
        - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
            - `EUCLIDEAN` is the euclidean distance.
            - `COSINE` is the cosine distance.
            - `HAMMING` is the hamming distance.
        - `ids` row ids to insert into collection.
        - `pre_delete_collection` if True, will delete the collection if it exists.
            (default: False)
            - Useful for testing.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls._initialize_from_embeddings(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            pre_delete_collection=pre_delete_collection,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        **kwargs: Any,
    ) -> Lantern:
        """Construct Lantern wrapper from raw documents and pre-
        generated embeddings.

        Postgres connection string is required
        "Either pass it as `connection_string` parameter
        or set the LANTERN_CONNECTION_STRING environment variable.

        Order of elements for lists `ids`, `text_embeddings`, `metadatas` should match,
        so each row will be associated with correct values.

        - `connection_string` is fully populated connection string for postgres database
        - `text_embeddings` is array with tuples (text, embedding)
                to insert into collection.
        - `embedding` is :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        - `metadatas` row metadata to insert into collection.
        - `collection_name` is the name of the collection to use. (default: langchain)
            - NOTE: This is the name of the table in which embedding data will be stored
                The table will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
        - `ids` row ids to insert into collection.
        - `pre_delete_collection` if True, will delete the collection if it exists.
            (default: False)
            - Useful for testing.
        - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
            - `EUCLIDEAN` is the euclidean distance.
            - `COSINE` is the cosine distance.
            - `HAMMING` is the hamming distance.
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls._initialize_from_embeddings(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            pre_delete_collection=pre_delete_collection,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Lantern],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        **kwargs: Any,
    ) -> Lantern:
        """
        Get instance of an existing Lantern store.This method will
        return the instance of the store without inserting any new
        embeddings

        Postgres connection string is required
        "Either pass it as `connection_string` parameter
        or set the LANTERN_CONNECTION_STRING environment variable.

        - `connection_string` is a postgres connection string.
        - `embedding` is :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        - `collection_name` is the name of the collection to use. (default: langchain)
            - NOTE: This is the name of the table in which embedding data will be stored
                The table will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
        - `ids` row ids to insert into collection.
        - `pre_delete_collection` if True, will delete the collection if it exists.
            (default: False)
            - Useful for testing.
        - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
            - `EUCLIDEAN` is the euclidean distance.
            - `COSINE` is the cosine distance.
            - `HAMMING` is the hamming distance.
        """
        connection_string = cls.__get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            pre_delete_collection=pre_delete_collection,
            distance_strategy=distance_strategy,
        )

        return store

    @classmethod
    def __get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="LANTERN_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as `connection_string` parameter"
                "or set the LANTERN_CONNECTION_STRING variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[Lantern],
        documents: List[Document],
        embedding: Embeddings,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Lantern:
        """
        Initialize a vector store with a set of documents.

        Postgres connection string is required
        "Either pass it as `connection_string` parameter
        or set the LANTERN_CONNECTION_STRING environment variable.

        - `connection_string` is a postgres connection string.
        - `documents` is list of :class:`Document` to initialize the vector store with
        - `embedding` is :class:`Embeddings` that will be used for
                embedding the text sent. If none is sent, then the
                multilingual Tensorflow Universal Sentence Encoder will be used.
        - `collection_name` is the name of the collection to use. (default: langchain)
            - NOTE: This is the name of the table in which embedding data will be stored
                The table will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
        - `distance_strategy` is the distance strategy to use. (default: EUCLIDEAN)
            - `EUCLIDEAN` is the euclidean distance.
            - `COSINE` is the cosine distance.
            - `HAMMING` is the hamming distance.
        - `ids` row ids to insert into collection.
        - `pre_delete_collection` if True, will delete the collection if it exists.
            (default: False)
            - Useful for testing.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.__get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)
        embedding_list = [result.EmbeddingStore.embedding for result in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

        return _results_to_docs(docs_and_scores)
