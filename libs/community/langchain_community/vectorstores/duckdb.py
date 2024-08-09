# mypy: disable-error-code=func-returns-value
from __future__ import annotations

import json
import logging
import uuid
import warnings
from typing import Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_KEY = "embedding"
DEFAULT_ID_KEY = "id"
DEFAULT_TEXT_KEY = "text"
DEFAULT_TABLE_NAME = "embeddings"
SIMILARITY_ALIAS = "similarity_score"


class DuckDB(VectorStore):
    """`DuckDB` vector store.

    This class provides a vector store interface for adding texts and performing
    similarity searches using DuckDB.

    For more information about DuckDB, see: https://duckdb.org/

    This integration requires the `duckdb` Python package.
    You can install it with `pip install duckdb`.

    *Security Notice*: The default DuckDB configuration is not secure.

        By **default**, DuckDB can interact with files across the entire file system,
        which includes abilities to read, write, and list files and directories.
        It can also access some python variables present in the global namespace.

        When using this DuckDB vectorstore, we suggest that you initialize the
        DuckDB connection with a secure configuration.

        For example, you can set `enable_external_access` to `false` in the connection
        configuration to disable external access to the DuckDB connection.

        You can view the DuckDB configuration options here:

        https://duckdb.org/docs/configuration/overview.html

        Please review other relevant security considerations in the DuckDB
        documentation. (e.g., "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false")

        See https://python.langchain.com/docs/security for more information.

    Args:
        connection: Optional DuckDB connection
        embedding: The embedding function or model to use for generating embeddings.
        vector_key: The column name for storing vectors. Defaults to `embedding`.
        id_key: The column name for storing unique identifiers. Defaults to `id`.
        text_key: The column name for storing text. Defaults to `text`.
        table_name: The name of the table to use for storing embeddings. Defaults to
          `embeddings`.

    Example:
        .. code-block:: python

            import duckdb
            conn = duckdb.connect(database=':memory:',
                config={
                    # Sample configuration to restrict some DuckDB capabilities
                    # List is not exhaustive. Please review DuckDB documentation.
                        "enable_external_access": "false",
                        "autoinstall_known_extensions": "false",
                        "autoload_known_extensions": "false"
                    }
            )
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
        vector_key: str = DEFAULT_VECTOR_KEY,
        id_key: str = DEFAULT_ID_KEY,
        text_key: str = DEFAULT_TEXT_KEY,
        table_name: str = DEFAULT_TABLE_NAME,
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

        if connection is None:
            warnings.warn(
                "No DuckDB connection provided. A new connection will be created."
                "This connection is running in memory and no data will be persisted."
                "To persist data, specify `connection=duckdb.connect(...)` when using "
                "the API. Please review the documentation of the vectorstore for "
                "security recommendations on configuring the connection."
            )

        self._connection = connection or self.duckdb.connect(
            database=":memory:", config={"enable_external_access": "false"}
        )
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
            kwargs: Additional parameters including optional 'ids' to associate
              with the texts.

        Returns:
            List of ids of the added texts.
        """
        have_pandas = False
        try:
            import pandas as pd

            have_pandas = True
        except ImportError:
            logger.info(
                "Unable to import pandas. "
                "Install it with `pip install -U pandas` "
                "to improve performance of add_texts()."
            )

        # Extract ids from kwargs or generate new ones if not provided
        ids = kwargs.pop("ids", [str(uuid.uuid4()) for _ in texts])

        # Embed texts and create documents
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))
        data = []
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            # Serialize metadata if present, else default to None
            metadata = (
                json.dumps(metadatas[idx])
                if metadatas and idx < len(metadatas)
                else None
            )
            if have_pandas:
                data.append(
                    {
                        self._id_key: ids[idx],
                        self._text_key: text,
                        self._vector_key: embedding,
                        "metadata": metadata,
                    }
                )
            else:
                self._connection.execute(
                    f"INSERT INTO {self._table_name} VALUES (?,?,?,?)",
                    [ids[idx], text, embedding, metadata],
                )

        if have_pandas:
            # noinspection PyUnusedLocal
            df = pd.DataFrame.from_dict(data)  # noqa: F841
            self._connection.register("df", df)
            self._connection.execute(
                f"INSERT INTO {self._table_name} SELECT * FROM df",
            )
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Performs a similarity search for a given query string.

        Args:
            query: The query string to search for.
            k: The number of similar texts to return.

        Returns:
            A list of Documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)  # type: ignore
        list_cosine_similarity = self.duckdb.FunctionExpression(
            "list_cosine_similarity",
            self.duckdb.ColumnExpression(self._vector_key),
            self.duckdb.ConstantExpression(embedding),
        )
        docs = (
            self._table.select(
                *[
                    self.duckdb.StarExpression(exclude=[]),
                    list_cosine_similarity.alias(SIMILARITY_ALIAS),
                ]
            )
            .order(f"{SIMILARITY_ALIAS} desc")
            .limit(k)
            .fetchdf()
        )
        return [
            Document(
                page_content=docs[self._text_key][idx],
                metadata={
                    **json.loads(docs["metadata"][idx]),
                    # using underscore prefix to avoid conflicts with user metadata keys
                    f"_{SIMILARITY_ALIAS}": docs[SIMILARITY_ALIAS][idx],
                }
                if docs["metadata"][idx]
                else {},
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
    ) -> DuckDB:
        """Creates an instance of DuckDB and populates it with texts and
          their embeddings.

        Args:
            texts: List of strings to add to the vector store.
            embedding: The embedding function or model to use for generating embeddings.
            metadatas: Optional list of metadata dictionaries associated with the texts.
            kwargs: Additional keyword arguments including:
                - connection: DuckDB connection. If not provided, a new connection will
                  be created.
                - vector_key: The column name for storing vectors. Default "vector".
                - id_key: The column name for storing unique identifiers. Default "id".
                - text_key: The column name for storing text. Defaults to "text".
                - table_name: The name of the table to use for storing embeddings.
                    Defaults to "embeddings".

        Returns:
            An instance of DuckDB with the provided texts and their embeddings added.
        """

        # Extract kwargs for DuckDB instance creation
        connection = kwargs.get("connection", None)
        vector_key = kwargs.get("vector_key", DEFAULT_VECTOR_KEY)
        id_key = kwargs.get("id_key", DEFAULT_ID_KEY)
        text_key = kwargs.get("text_key", DEFAULT_TEXT_KEY)
        table_name = kwargs.get("table_name", DEFAULT_TABLE_NAME)

        # Create an instance of DuckDB
        instance = DuckDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            table_name=table_name,
        )
        # Add texts and their embeddings to the DuckDB vector store
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance

    def _ensure_table(self) -> None:
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
