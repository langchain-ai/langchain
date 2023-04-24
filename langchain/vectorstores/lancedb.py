"""Wrapper around LanceDB vector database"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, List, Optional, Type

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class LanceDB(VectorStore):
    """Wrapper around LanceDB vector database.

    To use, you should have ``lancedb`` python package installed.

    Example:
        .. code-block:: python

            db = lancedb.connect('./lancedb')
            table = db.open_table('my_table')
            vectorstore = LanceDB(table, embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self,
        connection: Any,
        embedding_function: Callable,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
    ):
        """Initialize with Lance DB connection"""
        try:
            import lancedb
        except ImportError:
            raise ValueError(
                "Could not import lancedb python package. "
                "Please install it with `pip install lancedb`."
            )
        if not isinstance(connection, lancedb.db.LanceTable):
            raise ValueError(
                f"connection should be an instance of lancedb.db.LanceTable, ",
                f"got {type(connection)}",
            )
        self._connection = connection
        self._embedding_function = embedding_function
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key

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
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        for idx, text in enumerate(texts):
            embedding = self._embedding_function(text)
            metadata = metadatas[idx] if metadatas else {}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    **metadata,
                }
            )

        self._connection.add(docs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.

        Returns:
            List of documents most similar to the query.
        """
        embedding = self._embedding_function(query)
        docs = self._connection.search(embedding).limit(k).to_df()
        return [
            Document(
                page_content=doc[self._text_key],
                metadata=doc[docs.columns != self._text_key],
            )
            for doc in docs
        ]

    @classmethod
    def from_texts(
        cls: Type[LanceDB],
        connection: Any,
        embedding_function: Callable,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        **kwargs: Any,
    ) -> LanceDB:
        instance = LanceDB(connection, embedding_function, vector_key, id_key, text_key)
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance
