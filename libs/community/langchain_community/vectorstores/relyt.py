from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from sqlalchemy import Column, String, Table, create_engine, insert, text
from sqlalchemy.dialects.postgresql import JSON, TEXT

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

_LANGCHAIN_DEFAULT_EMBEDDING_DIM = 1536
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain_document"

Base = declarative_base()  # type: Any


class Relyt(VectorStore):
    """`Relyt` (distributed PostgreSQL) vector store.

    Relyt is a distributed full postgresql syntax cloud-native database.
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
        engine_args: Optional[dict] = None,
    ) -> None:
        """Initialize a PGVecto_rs vectorstore.

        Args:
            embedding: Embeddings to use.
            dimension: Dimension of the embeddings.
            db_url: Database URL.
            collection_name: Name of the collection.
            new_table: Whether to create a new table or connect to an existing one.
            If true, the table will be dropped if exists, then recreated.
            Defaults to False.
        """
        try:
            from pgvecto_rs.sdk import PGVectoRs

            PGVectoRs(
                db_url=connection_string,
                collection_name=collection_name,
                dimension=embedding_dimension,
                recreate=pre_delete_collection,
            )
        except ImportError as e:
            raise ImportError(
                "Unable to import pgvector_rs.sdk , please install with "
                '`pip install "pgvecto_rs[sdk]"`.'
            ) from e

        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__(engine_args)

    def __post_init__(
        self,
        engine_args: Optional[dict] = None,
    ) -> None:
        """
        Initialize the store.
        """

        _engine_args = engine_args or {}

        if (
            "pool_recycle" not in _engine_args
        ):  # Check if pool_recycle is not in _engine_args
            _engine_args["pool_recycle"] = (
                3600  # Set pool_recycle to 3600s if not present
            )

        self.engine = create_engine(self.connection_string, **_engine_args)
        self.create_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._euclidean_relevance_score_fn

    def create_table_if_not_exists(self) -> None:
        # Define the dynamic table
        """
        Table(
            self.collection_name,
            Base.metadata,
            Column("id", TEXT, primary_key=True, default=uuid.uuid4),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )
        """
        with self.engine.connect() as conn:
            with conn.begin():
                # create vectors
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vectors"))
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))

                # Create the table
                # Base.metadata.create_all(conn)
                table_name = f"{self.collection_name}"
                table_query = text(
                    f"""
                    SELECT 1
                    FROM pg_class
                    WHERE relname = '{table_name}';
                """
                )
                result = conn.execute(table_query).scalar()
                if not result:
                    table_statement = text(
                        f"""
                            CREATE TABLE {table_name} (
                                id TEXT PRIMARY KEY DEFAULT uuid_generate_v4(),
                                embedding vector({self.embedding_dimension}),
                                document TEXT,
                                metadata JSON
                            ) USING heap;
                        """
                    )
                    conn.execute(table_statement)

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
                        ON {self.collection_name}
                        USING vectors (embedding vector_l2_ops)
                        WITH (options = $$
                        optimizing.optimizing_threads = 30
                        segment.max_growing_segment_size = 600
                        segment.max_sealed_segment_size = 30000000
                        [indexing.hnsw]
                        m=30
                        ef_construction=500
                        $$);
                    """
                    )
                    conn.execute(index_statement)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.create_table_if_not_exists()

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name};")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)

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
        from pgvecto_rs.sqlalchemy import Vector

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
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )

        chunks_table_data = []
        with self.engine.connect() as conn:
            with conn.begin():
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

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        # Add the filter if provided
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        filter_condition = ""
        if filter is not None:
            conditions = [
                f"metadata->>{key!r} = {value!r}" for key, value in filter.items()
            ]
            filter_condition = f"WHERE {' AND '.join(conditions)}"

        # Define the base query
        sql_query = f"""
            set vectors.enable_search_growing = on;
            set vectors.enable_search_write = on;
            SELECT document, metadata, embedding <-> :embedding as distance
            FROM {self.collection_name}
            {filter_condition}
            ORDER BY embedding <-> :embedding
            LIMIT :k
        """

        # Set up the query parameters
        embedding_str = ", ".join(format(x) for x in embedding)
        embedding_str = "[" + embedding_str + "]"
        params = {"embedding": embedding_str, "k": k}

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

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        from pgvecto_rs.sqlalchemy import Vector

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # Define the table schema
        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", TEXT, primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )

        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = chunks_table.c.id.in_(ids)
                    conn.execute(chunks_table.delete().where(delete_condition))
                    return True
        except Exception as e:
            print("Delete operation failed:", str(e))  # noqa: T201
            return False

    @classmethod
    def from_texts(
        cls: Type[Relyt],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        embedding_dimension: int = _LANGCHAIN_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> Relyt:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            embedding_dimension=embedding_dimension,
            pre_delete_collection=pre_delete_collection,
            engine_args=engine_args,
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
                "or set the PG_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[Relyt],
        documents: List[Document],
        embedding: Embeddings,
        embedding_dimension: int = _LANGCHAIN_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> Relyt:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
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
            engine_args=engine_args,
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
