from __future__ import annotations

import uuid
import pandas as pd
import json
from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class DuckDB(VectorStore):
    """`DuckDB` vector store.

    This integration requires the `duckdb` Python package.
    You can install it with `pip install duckdb`.

    Args:
        connection: Optional DuckDB connection. If not provided, a new connection will be created.
        embedding: The embedding function or model to use for generating embeddings.
        vector_key: The column name for storing vectors. Defaults to `embedding`.
        id_key: The column name for storing unique identifiers. Defaults to `id`.
        text_key: The column name for storing text. Defaults to `text`.
        table_name: The name of the table to use for storing embeddings. Defaults to `embeddings`.

    Example:
        .. code-block:: python

            import duckdb
            conn = duckdb.connect(database=':memory:')
            embedding_function = ... # Define or import your embedding function here
            vector_store = DuckDB(conn, embedding_function)
            vector_store.add_texts(['text1', 'text2'])
            result = vector_store.similarity_search('text1')
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        vector_key: Optional[str] = "embedding",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
    ):
        """Initialize with DuckDB connection and setup for vector storage."""
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "Could not import duckdb package. "
                "Please install it with `pip install duckdb`."
            )
        self.duckdb = duckdb
        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
        self._table_name = table_name

        if self._embedding is None:
            raise ValueError("An embedding function or model must be provided.")

        self._connection = connection or self.duckdb.connect(database=':memory:')
        self._ensure_table()
        self._table = self._connection.table(self._table_name)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Returns the embedding object used by the vector store."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database using Pandas DataFrame

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            # Serialize metadata if present, else default to None
            metadata = json.dumps(metadatas[idx]) if metadatas and idx < len(metadatas) else None
            doc = {
                self._id_key: ids[idx],
                self._text_key: text,
                self._vector_key: embedding,
                "metadata": metadata,
            }
            docs.append(doc)
        df = pd.DataFrame(docs)
        print(df['embedding'])
        # self._table.insert(df)
        self._connection.sql(f"INSERT INTO {self._table_name} SELECT * FROM df")
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Performs a similarity search for a given query string.

        Args:
            query: The query string to search for.
            k: The number of similar texts to return.

        Returns:
            A list of Documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)  # type: ignore
        list_cosine_similarity = self.duckdb.FunctionExpression('list_cosine_similarity', self.duckdb.ColumnExpression(self._vector_key), self.duckdb.ConstantExpression(embedding))
        docs = (self._table.
            select(*[self.duckdb.StarExpression(),list_cosine_similarity.alias("similarity")]).
            order("similarity desc").
            limit(k).
            select(self.duckdb.StarExpression(exclude=["similarity", self._vector_key])).
            fetchdf()
        )
        return [
            Document(
                page_content=docs[self._text_key][idx],
                metadata=json.loads(docs["metadata"][idx]) if docs["metadata"][idx] else {},
            )
            for idx in range(len(docs))
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection: Any = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "embeddings",
        **kwargs: Any,
    ) -> DuckDB:
        """Creates an instance of DuckDB and populates it with texts and their embeddings.

        Args:
            texts: List of strings to add to the vector store.
            embedding: The embedding function or model to use for generating embeddings.
            metadatas: Optional list of metadata dictionaries associated with the texts.
            connection: DuckDB connection. If not provided, a new connection will be created.
            vector_key: The column name for storing vectors. Defaults to "vector".
            id_key: The column name for storing unique identifiers. Defaults to "id".
            text_key: The column name for storing text. Defaults to "text".
            table_name: The name of the table to use for storing embeddings. Defaults to "embeddings".
        
        Returns:
            An instance of DuckDB with the provided texts and their embeddings added.
        """
        # Create an instance of DuckDB
        instance = DuckDB(
            connection,
            embedding,
            vector_key,
            id_key,
            text_key,
            table_name,
        )
        # Add texts and their embeddings to the DuckDB vector store
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance

    def _ensure_table(self):
        """Ensures the table for storing embeddings exists."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            {self._id_key} VARCHAR PRIMARY KEY,
            {self._text_key} VARCHAR,
            {self._vector_key} FLOAT[],
            metadata VARCHAR
        )
        """
        self._connection.execute(create_table_sql)

