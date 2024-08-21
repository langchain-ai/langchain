from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from sqlalchemy import (
    CheckConstraint,
    Column,
    Uuid,
    asc,
    bindparam,
    create_engine,
    label,
    text,
)
from sqlalchemy.dialects.mssql import JSON, NVARCHAR, VARBINARY, VARCHAR
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

import json
import logging
import uuid


class DistanceStrategy(str, Enum):
    """Enumerator of the distance strategies for calculating distances
    between vectors.
    """

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT = "dot"


# String Constants
#
DISTANCE = "distance"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_STRATEGY = "distancestrategy"
EMBEDDING = "embedding"
EMBEDDING_LENGTH = "embedding_length"
EMBEDDING_VALUES = "embeddingvalues"
EMPTY_IDS_ERROR_MESSAGE = "Empty list of ids provided"
INVALID_IDS_ERROR_MESSAGE = "Invalid list of ids provided"
INVALID_INPUT_ERROR_MESSAGE = "Input is not valid."

# Query Constants
#
EMBEDDING_LENGTH_CONSTRAINT = f"ISVECTOR(embeddings, :{EMBEDDING_LENGTH}) = 1"
JSON_TO_ARRAY_QUERY = f"select JSON_ARRAY_TO_VECTOR (:{EMBEDDING_VALUES})"
VECTOR_DISTANCE_QUERY = f"""
VECTOR_DISTANCE(:{DISTANCE_STRATEGY}, JSON_ARRAY_TO_VECTOR(:{EMBEDDING}), embeddings)"""


class SQLServer_VectorStore(VectorStore):
    """SQL Server Vector Store.

    This class provides a vector store interface for adding texts and performing
        similarity searches on the texts in SQL Server.

    """

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_string: str,
        db_schema: Optional[str] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        embedding_function: Embeddings,
        embedding_length: int,
        table_name: str,
    ) -> None:
        """Initialize the SQL Server vector store.

        Args:
            connection: Optional SQLServer connection.
            connection_string: SQLServer connection string.
            db_schema: The schema in which the vector store will be created.
                This schema must exist and the user must have permissions to the schema.
            distance_strategy: The distance strategy to use for comparing embeddings.
                Default value is COSINE. Available options are:
                - COSINE
                - DOT
                - EUCLIDEAN
            embedding_function: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            embedding_length: The length (dimension) of the vectors to be stored in the
                table.
                Note that only vectors of same size can be added to the vector store.
            table_name: The name of the table to use for storing embeddings.

        """

        self.connection_string = connection_string
        self._distance_strategy = distance_strategy
        self.embedding_function = embedding_function
        self._embedding_length = embedding_length
        self.schema = db_schema
        self.table_name = table_name
        self._bind: Union[Connection, Engine] = (
            connection if connection else self._create_engine()
        )
        self._embedding_store = self._get_embedding_store(table_name, db_schema)
        self._create_table_if_not_exists()

    def _create_engine(self) -> Engine:
        return create_engine(url=self.connection_string)

    def _create_table_if_not_exists(self) -> None:
        logging.info(f"Creating table {self.table_name}.")
        try:
            with Session(self._bind) as session:
                self._embedding_store.__table__.create(
                    session.get_bind(), checkfirst=True
                )
                session.commit()
        except ProgrammingError as e:
            logging.error(f"Create table {self.table_name} failed.")
            raise Exception(e.__cause__) from None

    def _get_embedding_store(self, name: str, schema: Optional[str]) -> Any:
        DynamicBase = declarative_base(class_registry=dict())  # type: Any

        class EmbeddingStore(DynamicBase):
            """This is the base model for SQL vector store."""

            __tablename__ = name
            __table_args__ = {"schema": schema}
            id = Column(Uuid, primary_key=True, default=uuid.uuid4)
            custom_id = Column(VARCHAR, nullable=True)  # column for user defined ids.
            content_metadata = Column(JSON, nullable=True)
            content = Column(NVARCHAR, nullable=False)  # defaults to NVARCHAR(MAX)

            # Add check constraint to embeddings column
            # this will ensure only vectors of the same size
            # are allowed in the vector store.
            embeddings = Column(
                VARBINARY(8000),
                CheckConstraint(
                    text(EMBEDDING_LENGTH_CONSTRAINT).bindparams(
                        bindparam(EMBEDDING_LENGTH, self._embedding_length)
                    )
                ),
                nullable=False,
            )

        return EmbeddingStore

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @property
    def distance_strategy(self) -> str:
        # Value of distance strategy passed in should be one of the supported values.
        if isinstance(self._distance_strategy, DistanceStrategy):
            return self._distance_strategy.value

        # Match string value with appropriate enum value, if supported.
        distance_strategy_lower = str.lower(self._distance_strategy)

        if distance_strategy_lower == DistanceStrategy.EUCLIDEAN.value:
            return DistanceStrategy.EUCLIDEAN.value
        elif distance_strategy_lower == DistanceStrategy.COSINE.value:
            return DistanceStrategy.COSINE.value
        elif distance_strategy_lower == DistanceStrategy.DOT.value:
            return DistanceStrategy.DOT.value
        else:
            raise ValueError(f"{self._distance_strategy} is not supported.")

    @distance_strategy.setter
    def distance_strategy(self, value: DistanceStrategy):
        self._distance_strategy = value

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        return super().from_texts(texts, embedding, metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to given query.

        Args:
            query: Text to look up the most similar embedding to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of Documents most similar to the query provided.
        """
        embedded_query = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedded_query, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to the embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of Documents most similar to the embedding provided.
        """
        similar_docs_with_scores = self.similarity_search_by_vector_with_score(
            embedding, k, **kwargs
        )
        return self._docs_from_result(similar_docs_with_scores)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance and
            return docs most similar to the embedding vector.

        Args:
            query: Text to look up the most similar embedding to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of tuple of Document and an accompanying score in order of
            similarity to the query provided.
            Note that, a smaller score implies greater similarity.
        """
        embedded_query = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedded_query, k, **kwargs)

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance, given an embedding
            and return docs most similar to the embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Values for filtering on metadata during similarity search.

        Returns:
            List of tuple of Document and an accompanying score in order of
            similarity to the embedding provided.
            Note that, a smaller score implies greater similarity.
        """
        similar_docs = self._search_store(embedding, k, **kwargs)
        docs_and_scores = self._docs_and_scores_from_result(similar_docs)
        return docs_and_scores

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Compute the embeddings for the input texts and store embeddings
            in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """

        # Embed the texts passed in.
        embedded_texts = self.embedding_function.embed_documents(list(texts))

        # Insert the embedded texts in the vector store table.
        return self._insert_embeddings(texts, embedded_texts, metadatas, ids)

    def drop(self) -> None:
        """Drops every table created during initialization of vector store."""
        logging.info(f"Dropping vector store: {self.table_name}")
        try:
            with Session(bind=self._bind) as session:
                # Drop the table associated with the session bind.
                self._embedding_store.__table__.drop(session.get_bind())
                session.commit()

            logging.info(f"Vector store `{self.table_name}` dropped successfully.")
        except ProgrammingError as e:
            logging.error(f"Unable to drop vector store.\n {e.__cause__}.")

    def _search_store(
        self, embedding: List[float], k: int, filter: Optional[dict] = None
    ) -> List[Any]:
        try:
            with Session(self._bind) as session:
                results = (
                    session.query(
                        self._embedding_store,
                        label(
                            DISTANCE,
                            text(VECTOR_DISTANCE_QUERY).bindparams(
                                bindparam(
                                    DISTANCE_STRATEGY,
                                    self.distance_strategy,
                                    literal_execute=True,
                                ),
                                bindparam(
                                    EMBEDDING,
                                    json.dumps(embedding),
                                    literal_execute=True,
                                ),
                            ),
                        ),
                    )
                    .filter()
                    .order_by(asc(text(DISTANCE)))
                    .limit(k)
                    .all()
                )
        except ProgrammingError as e:
            logging.error(f"An error has occurred during the search.\n {e.__cause__}")
            raise Exception(e.__cause__) from None

        return results

    def _create_filter_clause(self, filter: dict) -> None:
        """TODO: parse filter and create a sql clause."""

    def _docs_from_result(
        self, results: List[Tuple[Document, float]]
    ) -> List[Document]:
        """Formats the input into a result of type List[Document]."""
        docs = [doc for doc, _ in results if doc is not None]
        return docs

    def _docs_and_scores_from_result(
        self, results: List[Any]
    ) -> List[Tuple[Document, float]]:
        """Formats the input into a result of type Tuple[Document, float].
        If an invalid input is given, it does not attempt to format the value
        and instead logs an error."""

        docs_and_scores = []

        for result in results:
            if (
                result is not None
                and result.EmbeddingStore is not None
                and result.distance is not None
            ):
                docs_and_scores.append(
                    (
                        Document(
                            page_content=result.EmbeddingStore.content,
                            metadata=result.EmbeddingStore.content_metadata,
                        ),
                        result.distance,
                    )
                )
            else:
                logging.error(INVALID_INPUT_ERROR_MESSAGE)

        return docs_and_scores

    def _insert_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert the embeddings and the texts in the vectorstore.

        Args:
            texts: Iterable of strings to add into the vectorstore.
            embeddings: List of list of embeddings.
            metadatas: List of metadatas (python dicts) associated with the input texts.
            ids: List of IDs for the input texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """

        if metadatas is None:
            metadatas = [{} for _ in texts]

        try:
            if ids is None:
                # Get IDs from metadata if available.
                ids = [metadata.get("id", uuid.uuid4()) for metadata in metadatas]

            with Session(self._bind) as session:
                documents = []
                for idx, query in enumerate(texts):
                    # For a query, if there is no corresponding ID,
                    # we generate a uuid and add it to the list of IDs to be returned.
                    if idx < len(ids):
                        id = ids[idx]
                    else:
                        ids.append(str(uuid.uuid4()))
                        id = ids[-1]
                    embedding = embeddings[idx]
                    metadata = metadatas[idx] if idx < len(metadatas) else {}

                    # Construct text, embedding, metadata as EmbeddingStore model
                    # to be inserted into the table.
                    sqlquery = text(JSON_TO_ARRAY_QUERY).bindparams(
                        bindparam(
                            EMBEDDING_VALUES,
                            json.dumps(embedding),
                            # render the value of the parameter into SQL statement
                            # at statement execution time
                            literal_execute=True,
                        )
                    )
                    result = session.scalar(sqlquery)
                    embedding_store = self._embedding_store(
                        custom_id=id,
                        content_metadata=metadata,
                        content=query,
                        embeddings=result,
                    )
                    documents.append(embedding_store)
                session.bulk_save_objects(documents)
                session.commit()
        except DBAPIError as e:
            logging.error(f"Add text failed:\n {e.__cause__}\n")
            raise Exception(e.__cause__) from None
        except AttributeError:
            logging.error("Metadata must be a list of dictionaries.")
            raise
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete embeddings in the vectorstore by the ids.

        Args:
            ids: List of IDs to delete.
            kwargs: vectorstore specific parameters.

        Returns:
            Optional[bool]
        """

        if ids is None or len(ids) == 0:
            logging.info(EMPTY_IDS_ERROR_MESSAGE)
            return False

        result = self._delete_texts_by_ids(ids)
        if result == 0:
            logging.info(INVALID_IDS_ERROR_MESSAGE)
            return False

        logging.info(result, " rows affected.")
        return True

    def _delete_texts_by_ids(self, ids: Optional[List[str]] = None) -> int:
        try:
            with Session(bind=self._bind) as session:
                result = (
                    session.query(self._embedding_store)
                    .filter(self._embedding_store.custom_id.in_(ids))
                    .delete()
                )
                session.commit()
        except DBAPIError as e:
            logging.error(e.__cause__)
        return result
