from __future__ import annotations

import contextlib
import enum
import json
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
)

import numpy as np
import sqlalchemy
from langchain_core._api import deprecated, warn_deprecated
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Session, relationship

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

try:
    from sqlalchemy import SQLColumnExpression
except ImportError:
    # for sqlalchemy < 2
    SQLColumnExpression = Any  # type: ignore

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class BaseModel(Base):
    """Base model for the SQL stores."""

    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


_classes: Any = None

COMPARISONS_TO_NATIVE = {
    "$eq": "==",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)


def _get_embedding_collection_store(
    vector_dimension: Optional[int] = None, *, use_jsonb: bool = True
) -> Any:
    global _classes
    if _classes is not None:
        return _classes

    from pgvector.sqlalchemy import Vector

    class CollectionStore(BaseModel):
        """Collection store."""

        __tablename__ = "langchain_pg_collection"

        name = sqlalchemy.Column(sqlalchemy.String)
        cmetadata = sqlalchemy.Column(JSON)

        embeddings = relationship(
            "EmbeddingStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            return session.query(cls).filter(cls.name == name).first()  # type: ignore

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

    if use_jsonb:
        # TODO(PRIOR TO LANDING): Create a gin index on the cmetadata field
        class EmbeddingStore(BaseModel):
            """Embedding store."""

            __tablename__ = "langchain_pg_embedding"

            collection_id = sqlalchemy.Column(
                UUID(as_uuid=True),
                sqlalchemy.ForeignKey(
                    f"{CollectionStore.__tablename__}.uuid",
                    ondelete="CASCADE",
                ),
            )
            collection = relationship(CollectionStore, back_populates="embeddings")

            embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
            document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            cmetadata = sqlalchemy.Column(JSONB, nullable=True)

            # custom_id : any user defined id
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

            __table_args__ = (
                sqlalchemy.Index(
                    "ix_cmetadata_gin",
                    "cmetadata",
                    postgresql_using="gin",
                    postgresql_ops={"cmetadata": "jsonb_path_ops"},
                ),
            )
    else:
        # For backwards comaptibilty with older versions of pgvector
        # This should be removed in the future (remove during migration)
        class EmbeddingStore(BaseModel):  # type: ignore[no-redef]
            """Embedding store."""

            __tablename__ = "langchain_pg_embedding"

            collection_id = sqlalchemy.Column(
                UUID(as_uuid=True),
                sqlalchemy.ForeignKey(
                    f"{CollectionStore.__tablename__}.uuid",
                    ondelete="CASCADE",
                ),
            )
            collection = relationship(CollectionStore, back_populates="embeddings")

            embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
            document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            cmetadata = sqlalchemy.Column(JSON, nullable=True)

            # custom_id : any user defined id
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

    _classes = (EmbeddingStore, CollectionStore)

    return _classes


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


@deprecated(
    since="0.0.31",
    message=(
        "This class is pending deprecation and may be removed in a future version. "
        "You can swap to using the `PGVector` "
        "implementation in `langchain_postgres`. "
        "Please read the guidelines in the doc-string of this class "
        "to follow prior to migrating as there are some differences "
        "between the implementations. "
        "See <https://github.com/langchain-ai/langchain-postgres> for details about "
        "the new implementation."
    ),
    alternative="from langchain_postgres import PGVector;",
    pending=True,
)
class PGVector(VectorStore):
    """`Postgres`/`PGVector` vector store.

    **DEPRECATED**: This class is pending deprecation and will likely receive
        no updates. An improved version of this class is available in
        `langchain_postgres` as `PGVector`. Please use that class instead.

        When migrating please keep in mind that:
            * The new implementation works with psycopg3, not with psycopg2
              (This implementation does not work with psycopg3).
            * Filtering syntax has changed to use $ prefixed operators for JSONB
              metadata fields. (New implementation only uses JSONB field for metadata)
            * The new implementation made some schema changes to address issues
              with the existing implementation. So you will need to re-create
              your tables and re-index your data or else carry out a manual
              migration.

    To use, you should have the ``pgvector`` python package installed.

    Args:
        connection_string: Postgres connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        embedding_length: The length of the embedding vector. (default: None)
            NOTE: This is not mandatory. Defining it will prevent vectors of
            any other size to be added to the embeddings table but, without it,
            the embeddings can't be indexed.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: COSINE)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.
        engine_args: SQLAlchemy's create engine arguments.
        use_jsonb: Use JSONB instead of JSON for metadata. (default: True)
            Strongly discouraged from using JSON as it's not as efficient
            for querying.
            It's provided here for backwards compatibility with older versions,
            and will be removed in the future.
        create_extension: If True, will create the vector extension if it doesn't exist.
            disabling creation is useful when using ReadOnly Databases.

    Example:

       .. code-block:: python

           from langchain_community.vectorstores import PGVector
           from langchain_community.embeddings.openai import OpenAIEmbeddings
           CONNECTION_STRING = "postgresql+psycopg2://hwc@localhost:5432/test3"
           COLLECTION_NAME = "state_of_the_union_test"
           embeddings = OpenAIEmbeddings()
           vectorestore = PGVector.from_documents(
               embedding=embeddings,
               documents=docs,
               collection_name=COLLECTION_NAME,
               connection_string=CONNECTION_STRING,
               use_jsonb=True,

    """  # noqa: E501

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        embedding_length: Optional[int] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        *,
        connection: Optional[sqlalchemy.engine.Connection] = None,
        engine_args: Optional[dict[str, Any]] = None,
        use_jsonb: bool = False,
        create_extension: bool = True,
    ) -> None:
        """Initialize the PGVector store."""
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self._embedding_length = embedding_length
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.engine_args = engine_args or {}
        self._bind = connection if connection else self._create_engine()
        self.use_jsonb = use_jsonb
        self.create_extension = create_extension

        if not use_jsonb:
            # Replace with a deprecation warning.
            warn_deprecated(
                "0.0.29",
                pending=True,
                message=(
                    "Please use JSONB instead of JSON for metadata. "
                    "This change will allow for more efficient querying that "
                    "involves filtering based on metadata. "
                    "Please note that filtering operators have been changed "
                    "when using JSONB metadata to be prefixed with a $ sign "
                    "to avoid name collisions with columns. "
                    "If you're using an existing database, you will need to create a "
                    "db migration for your metadata column to be JSONB and update your "
                    "queries to use the new operators. "
                ),
                alternative=(
                    "Instantiate with use_jsonb=True to use JSONB instead "
                    "of JSON for metadata."
                ),
            )
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """Initialize the store."""
        if self.create_extension:
            self.create_vector_extension()

        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self._embedding_length, use_jsonb=self.use_jsonb
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        self.create_tables_if_not_exists()
        self.create_collection()

    def __del__(self) -> None:
        if isinstance(self._bind, sqlalchemy.engine.Connection):
            self._bind.close()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine(url=self.connection_string, **self.engine_args)

    def create_vector_extension(self) -> None:
        try:
            with Session(self._bind) as session:  # type: ignore[arg-type]
                # The advisor lock fixes issue arising from concurrent
                # creation of the vector extension.
                # https://github.com/langchain-ai/langchain/issues/12933
                # For more information see:
                # https://www.postgresql.org/docs/16/explicit-locking.html#ADVISORY-LOCKS
                statement = sqlalchemy.text(
                    "BEGIN;"
                    "SELECT pg_advisory_xact_lock(1573678846307946496);"
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                    "COMMIT;"
                )
                session.execute(statement)
                session.commit()
        except Exception as e:
            raise Exception(f"Failed to create vector extension: {e}") from e

    def create_tables_if_not_exists(self) -> None:
        with Session(self._bind) as session, session.begin():  # type: ignore[arg-type]
            Base.metadata.create_all(session.get_bind())

    def drop_tables(self) -> None:
        with Session(self._bind) as session, session.begin():  # type: ignore[arg-type]
            Base.metadata.drop_all(session.get_bind())

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with Session(self._bind) as session:  # type: ignore[arg-type]
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        with Session(self._bind) as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session, bind to _conn string."""
        yield Session(self._bind)  # type: ignore[arg-type]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
            collection_only: Only delete ids in the collection.
        """
        with Session(self._bind) as session:  # type: ignore[arg-type]
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )

                stmt = delete(self.EmbeddingStore)

                if collection_only:
                    collection = self.get_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return

                    stmt = stmt.where(
                        self.EmbeddingStore.collection_id == collection.uuid
                    )

                stmt = stmt.where(self.EmbeddingStore.custom_id.in_(ids))
                session.execute(stmt)
            session.commit()

    def get_collection(self, session: Session) -> Any:
        return self.CollectionStore.get_by_name(session, self.collection_name)

    @classmethod
    def _from(
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
        *,
        use_jsonb: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

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
            use_jsonb=use_jsonb,
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
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        with Session(self._bind) as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            documents = []
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                embedding_store = self.EmbeddingStore(
                    embedding=embedding,
                    document=text,
                    cmetadata=metadata,
                    custom_id=id,
                    collection_id=collection.uuid,
                )
                documents.append(embedding_store)
            session.bulk_save_objects(documents)
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
            List of Documents most similar to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    @property
    def distance_strategy(self) -> Any:
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

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results = self._query_collection(embedding=embedding, k=k, filter=filter)

        return self._results_to_docs_and_scores(results)

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

    def _handle_field_filter(
        self,
        field: str,
        value: Any,
    ) -> SQLColumnExpression:
        """Create a filter for a specific field.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sqlalchemy expression
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )

        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: "
                f"{field}"
            )

        # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            # This is a filter specification
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            # Verify that that operator is an operator
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:  # Then we assume an equality operator
            operator = "$eq"
            filter_value = value

        if operator in COMPARISONS_TO_NATIVE:
            # Then we implement an equality filter
            # native is trusted input
            native = COMPARISONS_TO_NATIVE[operator]
            return func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                f"$.{field} {native} $value",
                json.dumps({"value": filter_value}),
            )
        elif operator == "$between":
            # Use AND with two comparisons
            low, high = filter_value

            lower_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                f"$.{field} >= $value",
                json.dumps({"value": low}),
            )
            upper_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                f"$.{field} <= $value",
                json.dumps({"value": high}),
            )
            return sqlalchemy.and_(lower_bound, upper_bound)
        elif operator in {"$in", "$nin", "$like", "$ilike"}:
            # We'll do force coercion to text
            if operator in {"$in", "$nin"}:
                for val in filter_value:
                    if not isinstance(val, (str, int, float)):
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

            queried_field = self.EmbeddingStore.cmetadata[field].astext

            if operator in {"$in"}:
                return queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$nin"}:
                return queried_field.not_in([str(val) for val in filter_value])
            elif operator in {"$like"}:
                return queried_field.like(filter_value)
            elif operator in {"$ilike"}:
                return queried_field.ilike(filter_value)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _create_filter_clause_deprecated(self, key, value):  # type: ignore[no-untyped-def]
        """Deprecated functionality.

        This is for backwards compatibility with the JSON based schema for metadata.
        It uses incorrect operator syntax (operators are not prefixed with $).

        This implementation is not efficient, and has bugs associated with
        the way that it handles numeric filter clauses.
        """
        IN, NIN, BETWEEN, GT, LT, NE = "in", "nin", "between", "gt", "lt", "ne"
        EQ, LIKE, CONTAINS, OR, AND = "eq", "like", "contains", "or", "and"

        value_case_insensitive = {k.lower(): v for k, v in value.items()}
        if IN in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.in_(
                value_case_insensitive[IN]
            )
        elif NIN in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.not_in(
                value_case_insensitive[NIN]
            )
        elif BETWEEN in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.between(
                str(value_case_insensitive[BETWEEN][0]),
                str(value_case_insensitive[BETWEEN][1]),
            )
        elif GT in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext > str(
                value_case_insensitive[GT]
            )
        elif LT in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext < str(
                value_case_insensitive[LT]
            )
        elif NE in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext != str(
                value_case_insensitive[NE]
            )
        elif EQ in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext == str(
                value_case_insensitive[EQ]
            )
        elif LIKE in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.like(
                value_case_insensitive[LIKE]
            )
        elif CONTAINS in map(str.lower, value):
            filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext.contains(
                value_case_insensitive[CONTAINS]
            )
        elif OR in map(str.lower, value):
            or_clauses = [
                self._create_filter_clause_deprecated(key, sub_value)
                for sub_value in value_case_insensitive[OR]
            ]
            filter_by_metadata = sqlalchemy.or_(*or_clauses)
        elif AND in map(str.lower, value):
            and_clauses = [
                self._create_filter_clause_deprecated(key, sub_value)
                for sub_value in value_case_insensitive[AND]
            ]
            filter_by_metadata = sqlalchemy.and_(*and_clauses)

        else:
            filter_by_metadata = None

        return filter_by_metadata

    def _create_filter_clause_json_deprecated(
        self, filter: Any
    ) -> List[SQLColumnExpression]:
        """Convert filters from IR to SQL clauses.

        **DEPRECATED** This functionality will be deprecated in the future.

        It implements translation of filters for a schema that uses JSON
        for metadata rather than the JSONB field which is more efficient
        for querying.
        """
        filter_clauses = []
        for key, value in filter.items():
            if isinstance(value, dict):
                filter_by_metadata = self._create_filter_clause_deprecated(key, value)

                if filter_by_metadata is not None:
                    filter_clauses.append(filter_by_metadata)
            else:
                filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext == str(
                    value
                )
                filter_clauses.append(filter_by_metadata)
        return filter_clauses

    def _create_filter_clause(self, filters: Any) -> Any:
        """Convert LangChain IR filter representation to matching SQLAlchemy clauses.

        At the top level, we still don't know if we're working with a field
        or an operator for the keys. After we've determined that we can
        call the appropriate logic to handle filter creation.

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            SQLAlchemy clause to apply to the query.
        """
        if isinstance(filters, dict):
            if len(filters) == 1:
                # The only operators allowed at the top level are $AND and $OR
                # First check if an operator or a field
                key, value = list(filters.items())[0]
                if key.startswith("$"):
                    # Then it's an operator
                    if key.lower() not in ["$and", "$or"]:
                        raise ValueError(
                            f"Invalid filter condition. Expected $and or $or "
                            f"but got: {key}"
                        )
                else:
                    # Then it's a field
                    return self._handle_field_filter(key, filters[key])

                # Here we handle the $and and $or operators
                if not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )
                if key.lower() == "$and":
                    and_ = [self._create_filter_clause(el) for el in value]
                    if len(and_) > 1:
                        return sqlalchemy.and_(*and_)
                    elif len(and_) == 1:
                        return and_[0]
                    else:
                        raise ValueError(
                            "Invalid filter condition. Expected a dictionary "
                            "but got an empty dictionary"
                        )
                elif key.lower() == "$or":
                    or_ = [self._create_filter_clause(el) for el in value]
                    if len(or_) > 1:
                        return sqlalchemy.or_(*or_)
                    elif len(or_) == 1:
                        return or_[0]
                    else:
                        raise ValueError(
                            "Invalid filter condition. Expected a dictionary "
                            "but got an empty dictionary"
                        )
                else:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and or $or "
                        f"but got: {key}"
                    )
            elif len(filters) > 1:
                # Then all keys have to be fields (they cannot be operators)
                for key in filters.keys():
                    if key.startswith("$"):
                        raise ValueError(
                            f"Invalid filter condition. Expected a field but got: {key}"
                        )
                # These should all be fields and combined using an $and operator
                and_ = [self._handle_field_filter(k, v) for k, v in filters.items()]
                if len(and_) > 1:
                    return sqlalchemy.and_(*and_)
                elif len(and_) == 1:
                    return and_[0]
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            else:
                raise ValueError("Got an empty dictionary for filters.")
        else:
            raise ValueError(
                f"Invalid type: Expected a dictionary but got type: {type(filters)}"
            )

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        with Session(self._bind) as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                if self.use_jsonb:
                    filter_clauses = self._create_filter_clause(filter)
                    if filter_clauses is not None:
                        filter_by.append(filter_clauses)
                else:
                    # Old way of doing things
                    filter_clauses = self._create_filter_clause_json_deprecated(filter)
                    filter_by.extend(filter_clauses)

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),  # type: ignore
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
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
        return _results_to_docs(docs_and_scores)

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
        *,
        use_jsonb: bool = False,
        **kwargs: Any,
    ) -> PGVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls._from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            use_jsonb=use_jsonb,
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

                from langchain_community.vectorstores import PGVector
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = PGVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls._from(
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
        Get instance of an existing PGVector store.This method will
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
        *,
        use_jsonb: bool = False,
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
            use_jsonb=use_jsonb,
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
        results = self._query_collection(embedding=embedding, k=fetch_k, filter=filter)

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

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
