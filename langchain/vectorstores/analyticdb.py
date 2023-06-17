"""VectorStore wrapper around a Postgres/PGVector database."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from sqlalchemy import REAL, Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT
from sqlalchemy.engine import Row

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

_LANGCHAIN_DEFAULT_EMBEDDING_DIM = 1536
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain_document"

Base = declarative_base()  # type: Any


class AnalyticDB(VectorStore):
    """
    VectorStore implementation using AnalyticDB.
    AnalyticDB is a distributed full PostgresSQL syntax cloud-native database.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `collection_name` is the name of the collection to use. (default: langchain)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.

    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        embedding_dimension: int = _LANGCHAIN_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        self.engine = create_engine(self.connection_string)
        self.create_collection()

    def create_table_if_not_exists(self) -> None:
        # Define the dynamic table
        Table(
            self.collection_name,
            Base.metadata,
            Column("id", TEXT, primary_key=True, default=uuid.uuid4),
            Column("embedding", ARRAY(REAL)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )
        with self.engine.connect() as conn:
            # Create the table
            Base.metadata.create_all(conn)

            # Check if the index exists
            index_name = f"{self.collection_name}_embedding_idx"
            index_query = text(
                f"""
                SELECT 1
                FROM pg_indexes
                WHERE indexname = '{index_name}';
            """
            )
            result = conn.execute(index_query).scalar()

            # Create the index if it doesn't exist
            if not result:
                index_statement = text(
                    f"""
                    CREATE INDEX {index_name}
                    ON {self.collection_name} USING ann(embedding)
                    WITH (
                        "dim" = {self.embedding_dimension},
                        "hnsw_m" = 100
                    );
                """
                )
                conn.execute(index_statement)
            conn.commit()

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.create_table_if_not_exists()

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name};")
        with self.engine.connect() as conn:
            conn.execute(drop_statement)
            conn.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
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
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        # Define the table schema
        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", TEXT, primary_key=True),
            Column("embedding", ARRAY(REAL)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )

        chunks_table_data = []
        with self.engine.connect() as conn:
            for document, metadata, chunk_id, embedding in zip(
                texts, metadatas, ids, embeddings
            ):
                chunks_table_data.append(
                    {
                        "id": chunk_id,
                        "embedding": embedding,
                        "document": document,
                        "metadata": metadata,
                    }
                )

                # Execute the batch insert when the batch size is reached
                if len(chunks_table_data) == batch_size:
                    conn.execute(insert(chunks_table).values(chunks_table_data))
                    # Clear the chunks_table_data list for the next batch
                    chunks_table_data.clear()

            # Insert any remaining records that didn't make up a full batch
            if chunks_table_data:
                conn.execute(insert(chunks_table).values(chunks_table_data))

            # Commit the transaction only once after all records have been inserted
            conn.commit()

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AnalyticDB with distance.

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

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        return self.similarity_search_with_score(query, k, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        # Add the filter if provided
        filter_condition = ""
        if filter is not None:
            conditions = [
                f"metadata->>{key!r} = {value!r}" for key, value in filter.items()
            ]
            filter_condition = f"WHERE {' AND '.join(conditions)}"

        # Define the base query
        sql_query = f"""
            SELECT *, l2_distance(embedding, :embedding) as distance
            FROM {self.collection_name}
            {filter_condition}
            ORDER BY embedding <-> :embedding
            LIMIT :k
        """

        # Set up the query parameters
        params = {"embedding": embedding, "k": k}

        # Execute the query and fetch the results
        with self.engine.connect() as conn:
            results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()

        documents_with_scores = [
            (
                Document(
                    page_content=result.document,
                    metadata=result.metadata,
                ),
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return documents_with_scores

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
        cls: Type[AnalyticDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        embedding_dimension: int = _LANGCHAIN_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> AnalyticDB:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        Either pass it as a parameter
        or set the PGVECTOR_CONNECTION_STRING environment variable.
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            embedding_dimension=embedding_dimension,
            pre_delete_collection=pre_delete_collection,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="PG_CONNECTION_STRING",
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
        cls: Type[AnalyticDB],
        documents: List[Document],
        embedding: Embeddings,
        embedding_dimension: int = _LANGCHAIN_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> AnalyticDB:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        Either pass it as a parameter
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
            embedding_dimension=embedding_dimension,
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
