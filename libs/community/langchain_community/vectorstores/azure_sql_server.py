from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from sqlalchemy import Column, Uuid, bindparam, create_engine, text
from sqlalchemy.dialects.mssql import JSON, NVARCHAR, VARBINARY, VARCHAR
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

import json
import logging
import uuid

Base = declarative_base()  # type: Any

_embedding_store: Any = None


class SQLServer_VectorStore(VectorStore):
    """SQL Server Vector Store.

    This class provides a vector store interface for adding texts and performing
        similarity searches on the texts in SQL Server.

    Args:
        connection: Optional SQLServer connection.
        connection_string: SQLServer connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        table_name: The name of the table to use for storing embeddings.

    """

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_string: str,
        embedding_function: Embeddings,
        table_name: str,
    ) -> None:
        """Initialize the SQL Server vector store."""

        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.table_name = table_name
        self._bind = connection if connection else self._create_engine()
        self.EmbeddingStore = self._get_embedding_store(table_name)
        self._create_table_if_not_exists()

    def _create_engine(self) -> Engine:
        return create_engine(url=self.connection_string, echo=True)

    def _create_table_if_not_exists(self) -> None:
        logging.info("Creating table %s", json.dumps(self.table_name))
        with Session(bind=self._bind) as session:
            Base.metadata.create_all(session.get_bind())

    def _get_embedding_store(self, name: str) -> Any:
        global _embedding_store
        if _embedding_store is not None:
            return _embedding_store

        class EmbeddingStore(Base):
            """This is the base model for SQL vector store."""

            __tablename__ = name
            id = Column(Uuid, primary_key=True, default=uuid.uuid4)
            custom_id = Column(VARCHAR, nullable=True)  # column for user defined ids.
            query_metadata = Column(JSON, nullable=True)
            query = Column(NVARCHAR("max"), nullable=False)
            embeddings = Column(VARBINARY, nullable=False)

        _embedding_store = EmbeddingStore
        return _embedding_store

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @classmethod
    def from_texts(
        cls: type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: List[dict] | None = None,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        return super().from_texts(texts, embedding, metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return super().similarity_search(query, k, **kwargs)

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
            ids = [metadata.pop("id", uuid.uuid4()) for metadata in metadatas]

        try:
            with Session(bind=self._bind) as session:
                documents = []
                for idx, query in enumerate(texts):
                    # For a query, if there is no corresponding ID, we generate a uuid and add it to the list of IDs to be returned.
                    id = (
                        ids[idx]
                        if idx < len(ids)
                        else ids.append(uuid.uuid4()) or ids[-1]
                    )
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
                            literal_execute=True,  # render the value of the parameter into SQL statement at statement execution time
                        )
                    )
                    result = session.scalar(sqlquery)
                    embedding_store = self.EmbeddingStore(
                        custom_id=id,
                        query_metadata=metadata,
                        query=query,
                        embeddings=result,
                    )
                    documents.append(embedding_store)
                session.bulk_save_objects(documents)
                session.commit()
            return ids
        except DBAPIError as e:
            logging.error(e.__cause__)
