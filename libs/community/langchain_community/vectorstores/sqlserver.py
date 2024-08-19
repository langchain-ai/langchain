from typing import Any, Iterable, List, Optional, Type, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from sqlalchemy import Column, Uuid, bindparam, create_engine, text
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

EMPTY_IDS_ERROR_MESSAGE = "Empty list of ids provided"
INVALID_IDS_ERROR_MESSAGE = "Invalid list of ids provided"


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
        embedding_function: Embeddings,
        table_name: str,
    ) -> None:
        """Initialize the SQL Server vector store.

        Args:
            connection: Optional SQLServer connection.
            connection_string: SQLServer connection string.
            db_schema: The schema in which the vector store will be created.
                This schema must exist and the user must have permissions to the schema.
            embedding_function: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            table_name: The name of the table to use for storing embeddings.

        """

        self.connection_string = connection_string
        self.embedding_function = embedding_function
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
            query_metadata = Column(JSON, nullable=True)
            query = Column(NVARCHAR, nullable=False)  # defaults to NVARCHAR(MAX)
            embeddings = Column(VARBINARY, nullable=False)

        return EmbeddingStore

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

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
        # placeholder
        return []

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
            kwargs: vectorstore specific parameters.

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
            kwargs: vectorstore specific parameters.

        Returns:
            List of IDs generated from adding the texts into the vectorstore.
        """

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if ids is None:
            # Get IDs from metadata if available.
            ids = [metadata.get("id", uuid.uuid4()) for metadata in metadatas]

        try:
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
                    metadata = metadatas[idx] if idx < len(metadatas) else None

                    # Construct text, embedding, metadata as EmbeddingStore model
                    # to be inserted into the table.
                    sqlquery = text(
                        "select JSON_ARRAY_TO_VECTOR (:embeddingvalues)"
                    ).bindparams(
                        bindparam(
                            "embeddingvalues",
                            json.dumps(embedding),
                            # render the value of the parameter into SQL statement
                            # at statement execution time
                            literal_execute=True,
                        )
                    )
                    result = session.scalar(sqlquery)
                    embedding_store = self._embedding_store(
                        custom_id=id,
                        query_metadata=metadata,
                        query=query,
                        embeddings=result,
                    )
                    documents.append(embedding_store)
                session.bulk_save_objects(documents)
                session.commit()
        except DBAPIError as e:
            logging.error(f"Add text failed:\n {e.__cause__}.")
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
