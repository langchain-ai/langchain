"""Wrapper around LanceDB vector database"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class LanceDB(VectorStore):
    """Wrapper around LanceDB vector database.

    To use, you should have ``lancedb`` python package installed.

    Example:
        .. code-block:: python

            # TODO: add example here
    """

    def __init__(self, connection: Any, embedding_function: Callable, table: str):
        """Initialize with Lance DB connection"""
        try:
            import lancedb
        except ImportError:
            raise ValueError(
                "Could not import lancedb python package. "
                "Please install it with `pip install lancedb`."
            )
        if not isinstance(connection, lancedb.LanceDBConnection):
            raise ValueError(
                f"connection should be an instance of lancedb.LanceDBConnection, ",
                f"got {type(connection)}",
            )
        if not table in connection.table_names:
            raise ValueError(f"table {table} does not exist in the database connection")
        self._connection = connection
        self._embedding_function = embedding_function
        self._table = table

    def add_texts(
        self,
        texts: Iterable[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embeddings and add it to the database

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        client = self._connection.open_table(self._table)
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for idx, text in enumerate(texts):
            embedding = self._embedding_function(text)
            metadata = metadatas[idx] if metadatas else {}
            docs.append({"vector": embedding, "id": ids[idx], **metadata})

        client.add(docs)
