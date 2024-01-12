import contextlib
import enum
import logging
import uuid
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

import sqlalchemy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy.orm import Session, declarative_base
from tidb_vector.sqlalchemy import VectorType

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
    """Create a vector model class."""

    global _classes
    if _classes is not None:
        return _classes

    class SQLVectorModel(Base):
        """
        embedding: The column to store the vector.
        document: The column to store the text document.
        meta: The column to store the metadata of the document.
            It can be used to filter the document when performing search
            e.g. {"title": "The title of the document", "custom_id": "123"}
        """

        __tablename__ = table_name
        id = sqlalchemy.Column(
            sqlalchemy.String(36), primary_key=True, default=lambda: str(uuid.uuid4())
        )
        embedding = sqlalchemy.Column(VectorType())
        document = sqlalchemy.Column(sqlalchemy.Text, nullable=True)
        meta = sqlalchemy.Column(sqlalchemy.JSON, nullable=True)

    _classes = SQLVectorModel
    return _classes


class TiDBVector(VectorStore):
    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a TiDBVector.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            embedding_function: The embedding function used to generate embeddings.
            collection_name (str): The name of the collection,
                defaults to "langchain_tidb_vector".
            distance_strategy: The strategy used for similarity search,
                defaults to "cosine", valid values: "l2", "cosine".
            engine_args (Optional[Dict]): Additional arguments for the database engine,
                defaults to None.
            pre_delete_collection: Delete the collection before creating a new one,
                defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Examples:
            .. code-block:: python

            from langchain_community.vectorstores.tidb_vector import TiDBVector
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            CONNECTION_STRING = "mysql+pymysql://root@34.212.137.91:4000/test"

            vs = TiDBVector.from_texts(
                embedding=embeddings,
                texts = [..., ...],
                connection_string=CONNECTION_STRING,
                distance_strategy="l2",
                collection_name="tidb_vector_langchain",
            )

            query = "What did the president say about Ketanji Brown Jackson"
            docs = db.similarity_search_with_score(query)

        """

        super().__init__(**kwargs)
        self.connection_string = connection_string
        self._embedding_function = embedding_function
        self._distance_strategy = distance_strategy
        self._engine_args = engine_args or {}
        self._pre_delete_collection = pre_delete_collection
        self._bind = self._create_engine()
        self._table_model = _create_vector_model(collection_name)
        _ = self.distance_strategy  # check if distance strategy is valid
        self.create_table_if_not_exists()

    @property
    def embeddings(self) -> Embeddings:
        """Return the function used to generate embeddings."""
        return self._embedding_function

    def create_table_if_not_exists(self) -> None:
        """
        If the `self._pre_delete_collection` flag is set,
        the existing table will be dropped before creating a new one.
        """
        if self._pre_delete_collection:
            self.drop_table()
        with Session(self._bind) as session, session.begin():
            Base.metadata.create_all(session.get_bind())
            # wait for tidb support vector index

    def drop_table(self) -> None:
        """Drops the table if it exists."""
        with Session(self._bind) as session, session.begin():
            Base.metadata.drop_all(session.get_bind())

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        """Create a sqlalchemy engine."""
        return sqlalchemy.create_engine(url=self.connection_string, **self._engine_args)

    def __del__(self) -> None:
        """Close the connection when the object is deleted."""
        if isinstance(self._bind, sqlalchemy.engine.Connection):
            self._bind.close()

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a context manager for the session."""
        yield Session(self._bind)

    @property
    def distance_strategy(self) -> Any:
        """
        Returns the distance function based on the current distance strategy value.
        """
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._table_model.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self._table_model.embedding.cosine_distance
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

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
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a VectorStore from a list of texts.

        Args:
            texts (List[str]): The list of texts to be added to the TiDB Vector.
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            metadatas: The list of metadata dictionaries corresponding to each text,
                defaults to None.
            collection_name (str, optional): The name of the collection,
                defaults to "langchain_tidb_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed strategies: "l2", "cosine".
            ids (Optional[List[str]]): The list of IDs corresponding to each text,
                defaults to None.
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            pre_delete_collection: Delete existing collection before adding the texts,
                defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            VectorStore: The created VectorStore.

        """
        embeddings = embedding.embed_documents(list(texts))

        tidbvs = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        tidbvs.add_texts(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return tidbvs

    @classmethod
    def from_existing_collection(
        cls,
        embedding: Embeddings,
        connection_string: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a VectorStore instance from an existing collection in the TiDB database.

        Args:
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            collection_name (str, optional): The name of the collection,
                defaults to "langchain_tidb_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed strategies: "l2", "cosine".
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            VectorStore: The VectorStore instance.

        Raises:
            NoSuchTableError: If the specified table does not exist in the TiDB.
        """

        engine = sqlalchemy.create_engine(connection_string, **(engine_args or {}))

        try:
            # check if the table exists
            table_query = sqlalchemy.sql.text(
                "SELECT 1 FROM information_schema.tables WHERE table_name = :table_name"
            )
            with engine.connect() as connection:
                if (
                    connection.execute(
                        table_query,
                        {"table_name": collection_name},
                    ).fetchone()
                    is None
                ):
                    raise sqlalchemy.exc.NoSuchTableError(
                        f"The table '{collection_name}' does not exist in the database."
                    )

            return cls(
                connection_string=connection_string,
                collection_name=collection_name,
                embedding_function=embedding,
                distance_strategy=distance_strategy,
                engine_args=engine_args,
                **kwargs,
            )
        finally:
            # Close the engine after quering the tale
            engine.dispose()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to TiDB Vector.

        Args:
            texts (Iterable[str]): The texts to be added.
            metadatas (Optional[List[dict]]): The metadata associated with each text,
                Defaults to None.
            ids (Optional[List[str]]): The IDs to be assigned to each text,
                Defaults to None, will be generated if not provided.

        Returns:
            List[str]: The IDs assigned to the added texts.
        """

        embeddings = self._embedding_function.embed_documents(list(texts))
        if ids is None:
            ids = [uuid.uuid4() for _ in texts]
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
        """
        Delete vectors from the TiDB vector.

        Args:
            ids (Optional[List[str]]): A list of vector IDs to delete.
            **kwargs: Additional keyword arguments.
        """
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
        """
        Perform a similarity search using the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to retrieve. Defaults to 4.
            filter (dict, optional): A filter to apply to the search results.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of Document objects representing the search results.
        """
        result = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in result]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with score based on the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to return. Defaults to 5.
            filter (dict, optional): A filter to apply to the search results.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of tuples containing relevant documents and their similarity scores.
        """
        query_vector = self._embedding_function.embed_query(query)
        relevant_docs = self._query_collection(query_vector, k, filter)
        return [
            (
                Document(
                    page_content=doc.document,
                    metadata=doc.meta,
                ),
                doc.distance,
            )
            for doc in relevant_docs
        ]

    def _query_collection(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query from collection."""

        filter_by = self._build_filter_clause(filter)
        with Session(self._bind) as session:
            results: List[Any] = (
                session.query(
                    self._table_model.meta,
                    self._table_model.document,
                    self.distance_strategy(query_embedding).label("distance"),
                )
                .filter(filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .limit(k)
                .all()
            )
        return results

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select the relevance score function based on the distance strategy.
        """
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )

    def _build_filter_clause(self, filter: Dict[str, str]) -> Any:
        """
        Builds the filter clause for querying the database based on the provided filter.

        Args:
            filter (Dict[str, str]): The filter conditions to apply.

        Returns:
            Any: The filter clause to be used in the database query.
        """

        filter_by = sqlalchemy.true()
        if filter is not None:
            filter_clauses = []

            for key, value in filter.items():
                if key.lower() == "$and":
                    and_clauses = [
                        self._build_filter_clause(condition) for condition in value
                    ]
                    filter_by_metadata = sqlalchemy.and_(*and_clauses)
                    filter_clauses.append(filter_by_metadata)
                elif key.lower() == "$or":
                    or_clauses = [
                        self._build_filter_clause(condition) for condition in value
                    ]
                    filter_by_metadata = sqlalchemy.or_(*or_clauses)
                    filter_clauses.append(filter_by_metadata)
                elif isinstance(value, dict):
                    filter_by_metadata = self._create_filter_clause(key, value)

                    if filter_by_metadata is not None:
                        filter_clauses.append(filter_by_metadata)
                else:
                    filter_by_metadata = (
                        sqlalchemy.func.json_extract(self._table_model.meta, f"$.{key}")
                        == value
                    )
                    filter_clauses.append(filter_by_metadata)

            filter_by = sqlalchemy.and_(filter_by, *filter_clauses)
        return filter_by

    def _create_filter_clause(self, key, value):
        """
        Create a filter clause based on the provided key-value pair.

        Args:
            key (str): How to filter the value
            value (dict): The value to filter with.

        Returns:
            sqlalchemy.sql.elements.BinaryExpression: The filter clause.

        Raises:
            None

        """

        IN, NIN, GT, GTE, LT, LTE = "$in", "$nin", "$gt", "$gte", "$lt", "$lte"
        EQ, NE, OR, AND = "$eq", "$ne", "$or", "$and"

        json_key = sqlalchemy.func.json_extract(self._table_model.meta, f"$.{key}")
        value_case_insensitive = {k.lower(): v for k, v in value.items()}

        if IN in map(str.lower, value):
            filter_by_metadata = json_key.in_(value_case_insensitive[IN])
        elif NIN in map(str.lower, value):
            filter_by_metadata = ~json_key.in_(value_case_insensitive[NIN])
        elif GT in map(str.lower, value):
            filter_by_metadata = json_key > value_case_insensitive[GT]
        elif GTE in map(str.lower, value):
            filter_by_metadata = json_key >= value_case_insensitive[GTE]
        elif LT in map(str.lower, value):
            filter_by_metadata = json_key < value_case_insensitive[LT]
        elif LTE in map(str.lower, value):
            filter_by_metadata = json_key <= value_case_insensitive[LTE]
        elif NE in map(str.lower, value):
            filter_by_metadata = json_key != value_case_insensitive[NE]
        elif EQ in map(str.lower, value):
            filter_by_metadata = json_key == value_case_insensitive[EQ]
        elif OR in map(str.lower, value):
            or_clauses = [
                self._build_filter_clause(sub_value)
                for sub_value in value_case_insensitive[OR]
            ]
            filter_by_metadata = sqlalchemy.or_(or_clauses)
        elif AND in map(str.lower, value):
            and_clauses = [
                self._build_filter_clause(sub_value)
                for sub_value in value_case_insensitive[AND]
            ]
            filter_by_metadata = sqlalchemy.and_(and_clauses)
        else:
            logger.warning(
                f"Unsupported filter operator: {value}. Consider using "
                "one of $in, $nin, $gt, $gte, $lt, $lte, $eq, $ne, $or, $and."
            )
            filter_by_metadata = None

        return filter_by_metadata
