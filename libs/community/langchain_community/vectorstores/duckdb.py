from __future__ import annotations

import uuid
import pandas as pd
import json
from typing import Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

class DuckDB(VectorStore):
    """`DuckDB` vector store.

    This integration requires the `duckdb` Python package.
    You can install it with `pip install duckdb`.

    *Security Notice*:  This class enables direct interactions with the file system 
    through DuckDB's functionalities, such as reading from and writing to local 
    files. This capability poses security considerations when integrating this class 
    into applications, particularly those accessible by third-party users or systems.

    By **default**, DuckDB can interact with files across the entire file system, 
    which includes abilities to read, write, and list files and directories.

    To mitigate potential security risks, consider implementing the 
    following measures:
    - Limit access to particular directories using `home_directory`.
    - Consider setting `enable_external_access` to `false` in the connection
    - Use filesystem permissions to restrict access and permissions to only
        the files and directories required by the agent.
    - Limit the tools available to the agent to only the file operations
        necessary for the agent's intended use.
    - Sandbox the agent by running it in a container.

    See https://python.langchain.com/docs/security for more information.

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
        *,
        connection: Optional[Any] = None,
        embedding: Embeddings,
        vector_key: str = "embedding",
        id_key: str = "id",
        text_key: str = "text",
        table_name: str = "vectorstore",
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

        self._connection = connection or self.duckdb.connect(database=':memory:',config={'enable_external_access': 'false'})
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
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database using Pandas DataFrame

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional parameters including optional 'ids' to associate with the texts.

        Returns:
            List of ids of the added texts.
        """

        # Extract ids from kwargs or generate new ones if not provided
        ids = kwargs.pop('ids', [str(uuid.uuid4()) for _ in texts])

        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            # Serialize metadata if present, else default to None
            metadata = json.dumps(metadatas[idx]) if metadatas and idx < len(metadatas) else None
            self._connection.execute(f"INSERT INTO {self._table_name} VALUES (?,?,?,?)",[ids[idx],text,embedding,metadata])
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
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """Creates an instance of DuckDB and populates it with texts and their embeddings.

        Args:
            texts: List of strings to add to the vector store.
            embedding: The embedding function or model to use for generating embeddings.
            metadatas: Optional list of metadata dictionaries associated with the texts.
            **kwargs: Additional keyword arguments including:
                - connection: DuckDB connection. If not provided, a new connection will be created.
                - vector_key: The column name for storing vectors. Defaults to "vector".
                - id_key: The column name for storing unique identifiers. Defaults to "id".
                - text_key: The column name for storing text. Defaults to "text".
                - table_name: The name of the table to use for storing embeddings. Defaults to "embeddings".
            
        Returns:
            An instance of DuckDB with the provided texts and their embeddings added.
        """

        # Extract kwargs for DuckDB instance creation
        connection = kwargs.get('connection', None)
        vector_key = kwargs.get('vector_key', "vector")
        id_key = kwargs.get('id_key', "id")
        text_key = kwargs.get('text_key', "text")
        table_name = kwargs.get('table_name', "embeddings")
        
        # Create an instance of DuckDB
        instance = DuckDB(
            connection = connection,
            embedding = embedding,
            vector_key = vector_key,
            id_key = id_key,
            text_key = text_key,
            table_name = table_name,
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

