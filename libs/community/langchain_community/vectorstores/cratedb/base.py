from __future__ import annotations

import enum
from typing import (
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
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.pgvector import PGVector
from sqlalchemy.orm import sessionmaker

from langchain_community.vectorstores.cratedb.model import ModelFactory


class DistanceStrategy(str, enum.Enum):
    """
    Enumerator of the Distance strategies.

    Note that CrateDB and Lucene currently only implement
    similarity based on the Euclidean distance.

    > Today, when creating a FLOAT_VECTOR, it uses the default
    > EUCLIDEAN_HNSW (L2) similarity.
    >
    > -- https://github.com/crate/crate/issues/15768
    """

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


class CrateDBVectorStore(PGVector):
    """`CrateDB` vector store.

    To use it, please install the Python package `sqlalchemy-cratedb`.

        uv pip install --upgrade sqlalchemy-cratedb

    Args:
        connection_string: Database connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: EUCLIDEAN)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.

    Example:
        .. code-block:: python

            from langchain.vectorstores import CrateDBVectorStore
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "crate://crate@localhost:4200/test3"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = CrateDBVectorStore.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )


    """

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """

        self._engine = self._bind
        self.Session = sessionmaker(bind=self._engine)  # type: ignore[call-overload]

        # Patch dialect to invoke `REFRESH TABLE` after each DML operation.
        from sqlalchemy_cratedb.support import refresh_after_dml

        refresh_after_dml(self._engine)

        # Need to defer initialization, because dimension size
        # can only be figured out at runtime.
        self.BaseModel = None
        self.CollectionStore = None  # type: ignore[assignment]
        self.EmbeddingStore = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """
        Work around premature session close.

        sqlalchemy.orm.exc.DetachedInstanceError: Parent instance <CollectionStore at 0x1212ca3d0> is not bound
        to a Session; lazy load operation of attribute 'embeddings' cannot proceed.
        -- https://docs.sqlalchemy.org/en/20/errors.html#error-bhk3

        TODO: Review!
        """  # noqa: E501
        pass

    def _init_models(self, embedding: List[float]) -> None:
        """
        Create SQLAlchemy models at runtime, when not established yet.
        """

        # TODO: Use a better way to run this only once.
        if self.CollectionStore is not None and self.EmbeddingStore is not None:
            return

        size = len(embedding)
        self._init_models_with_dimensionality(size=size)

    def _init_models_with_dimensionality(self, size: int) -> None:
        mf = ModelFactory(dimensions=size)
        self.BaseModel, self.CollectionStore, self.EmbeddingStore = (
            mf.BaseModel,  # type: ignore[assignment]
            mf.CollectionStore,
            mf.EmbeddingStore,
        )

    def get_collection(self, session: sqlalchemy.orm.Session) -> Any:
        if self.CollectionStore is None:
            raise RuntimeError(
                "Collection can't be accessed without specifying "
                "dimension size of embedding vectors"
            )
        return self.CollectionStore.get_by_name(session, self.collection_name)

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
        from sqlalchemy_cratedb.support import refresh_table

        if not embeddings:
            return []
        self._init_models(embeddings[0])

        # When the user requested to delete the collection before running subsequent
        # operations on it, run the deletion gracefully if the table does not exist
        # yet.
        if self.pre_delete_collection:
            try:
                self.delete_collection()
            except sqlalchemy.exc.ProgrammingError as ex:
                if "RelationUnknown" not in str(ex):
                    raise

        # Tables need to be created at runtime, because the `EmbeddingStore.embedding`
        # field, a `FloatVector`, needs to be initialized with a dimensionality
        # parameter, which is only obtained at runtime.
        self.create_tables_if_not_exists()
        self.create_collection()

        # After setting up the table/collection at runtime, add embeddings.
        embedding_ids = super().add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        refresh_table(self.Session(), self.EmbeddingStore)
        return embedding_ids

    def create_tables_if_not_exists(self) -> None:
        """
        Need to overwrite because this `Base` is different from parent's `Base`.
        """
        if self.BaseModel is None:
            raise RuntimeError("Storage models not initialized")
        self.BaseModel.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """
        Need to overwrite because this `Base` is different from parent's `Base`.
        """
        mf = ModelFactory()
        mf.Base.metadata.drop_all(self._engine)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors by ids or uuids.

        Remark: Specialized for CrateDB to synchronize data.

        Args:
            ids: List of ids to delete.

        Remark: Patch for CrateDB needs to overwrite this, in order to
                add a "REFRESH TABLE" statement afterwards. The other
                patch, listening to `after_delete` events seems not be
                able to catch it.
        """
        from sqlalchemy_cratedb.support import refresh_table

        super().delete(ids=ids, collection_only=collection_only, **kwargs)

        # CrateDB: Synchronize data because `on_flush` does not catch it.
        with self.Session() as session:
            refresh_table(session, self.EmbeddingStore)

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.euclidean_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            raise NotImplementedError("Cosine similarity not implemented yet")
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            raise NotImplementedError("Dot-product similarity not implemented yet")
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        docs = [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.similarity if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    def _query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self._init_models(embedding)
        with self.Session() as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            return self._query_collection_multi(
                collections=[collection], embedding=embedding, k=k, filter=filter
            )

    def _query_collection_multi(
        self,
        collections: List[Any],
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        self._init_models(embedding)

        collection_names = [coll.name for coll in collections]
        collection_uuids = [coll.uuid for coll in collections]
        self.logger.info(f"Querying collections: {collection_names}")

        with self.Session() as session:
            filter_by = self.EmbeddingStore.collection_id.in_(collection_uuids)

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    IN = "in"
                    if isinstance(value, dict) and IN in map(str.lower, value):
                        value_case_insensitive = {
                            k.lower(): v for k, v in value.items()
                        }
                        filter_by_metadata = self.EmbeddingStore.cmetadata[key].in_(
                            value_case_insensitive[IN]
                        )
                        filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = self.EmbeddingStore.cmetadata[key] == str(
                            value
                        )  # type: ignore[assignment]
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)  # type: ignore[assignment]

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(  # type: ignore[attr-defined]
                    self.EmbeddingStore,
                    # TODO: Original pgvector code uses `self.distance_strategy`.
                    #       CrateDB currently only supports EUCLIDEAN.
                    #       self.distance_strategy(embedding).label("distance")
                    sqlalchemy.func.vector_similarity(
                        self.EmbeddingStore.embedding,
                        # TODO: Just reference the `embedding` symbol here, don't
                        #       serialize its value prematurely.
                        #       https://github.com/crate/crate/issues/16912
                        #
                        # Until that got fixed, marshal the arguments to
                        # `vector_similarity()` manually, in order to work around
                        # this edge case bug. We don't need to use JSON marshalling,
                        # because Python's string representation of a list is just
                        # right.
                        sqlalchemy.text(str(embedding)),
                    ).label("similarity"),
                )
                .filter(filter_by)
                # CrateDB applies `KNN_MATCH` within the `WHERE` clause.
                .filter(
                    sqlalchemy.func.knn_match(
                        self.EmbeddingStore.embedding, embedding, k
                    )
                )
                .order_by(sqlalchemy.desc("similarity"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
            )
        return results

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[CrateDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> CrateDBVectorStore:
        """
        Return VectorStore initialized from texts and embeddings.
        Database connection string is required.

        Either pass it as a parameter, or set the CRATEDB_CONNECTION_STRING
        environment variable.

        Remark: Needs to be overwritten, because CrateDB uses a different
                DEFAULT_DISTANCE_STRATEGY.
        """
        return super().from_texts(  # type: ignore[return-value]
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,  # type: ignore[arg-type]
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="CRATEDB_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Database connection string is required."
                "Either pass it as a parameter, or set the "
                "CRATEDB_CONNECTION_STRING environment variable."
            )

        return connection_string

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
        return str(
            sqlalchemy.URL.create(
                drivername=driver,
                host=host,
                port=port,
                username=user,
                password=password,
                query={"schema": database},
            )
        )

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
                "No supported normalization function for distance_strategy of "
                f"{self._distance_strategy}. Consider providing relevance_score_fn to "
                "CrateDBVectorStore constructor."
            )

    @staticmethod
    def _euclidean_relevance_score_fn(similarity: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the Euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)

        # CrateDB uses the `vector_similarity()` SQL function in this context,
        # which already returns a normalized value.
        # https://cratedb.com/docs/crate/reference/en/latest/general/builtins/scalar-functions.html#vector-similarity-float-vector-float-vector
        return similarity
