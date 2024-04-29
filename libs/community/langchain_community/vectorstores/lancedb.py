from __future__ import annotations

import os
import uuid
import warnings
from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def import_lancedb() -> Any:
    """Import lancedb package."""

    try:
        import lancedb
    except ImportError as e:
        raise ImportError(
            "Could not import pinecone lancedb package. "
            "Please install it with `pip install lancedb`."
        ) from e
    return lancedb


class LanceDB(VectorStore):
    """`LanceDB` vector store.

    To use, you should have ``lancedb`` python package installed.
    You can install it with ``pip install lancedb``.

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        api_key: API key to use for LanceDB cloud database.
        region: Region to use for LanceDB cloud database.
        mode: Mode to use for adding data to the table. Defaults to ``overwrite``.



    Example:
        .. code-block:: python
            vectorstore = LanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        uri: Optional[str] = "/tmp/lancedb",
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "overwrite",
    ):
        """Initialize with Lance DB vectorstore"""
        lancedb = import_lancedb()
        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
        self._table_name = table_name
        self.api_key = api_key or os.getenv("LANCE_API_KEY") if api_key != "" else None
        self.region = region
        self.mode = mode

        if isinstance(uri, str) and self.api_key is None:
            if uri.startswith("db://"):
                raise ValueError("API key is required for LanceDB cloud.")

        if self._embedding is None:
            raise ValueError("embedding object should be provided")

        if isinstance(connection, lancedb.db.LanceDBConnection):
            self._connection = connection
        elif isinstance(connection, (str, lancedb.db.LanceTable)):
            raise ValueError(
                "`connection` has to be a lancedb.db.LanceDBConnection object.\
                `lancedb.db.LanceTable` is deprecated."
            )
        else:
            if self.api_key is None:
                self._connection = lancedb.connect(uri)
            else:
                if isinstance(uri, str):
                    if uri.startswith("db://"):
                        self._connection = lancedb.connect(
                            uri, api_key=self.api_key, region=self.region
                        )
                    else:
                        self._connection = lancedb.connect(uri)
                        warnings.warn(
                            "api key provided with local uri.\
                            The data will be stored locally"
                        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate w    ith the texts.

        Returns:
            List of ids of the added texts.
        """
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))  # type: ignore
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    "metadata": metadata,
                }
            )

        if self._table_name in self._connection.table_names():
            tbl = self._connection.open_table(self._table_name)
            if self.api_key is None:
                tbl.add(docs, mode=self.mode)
            else:
                tbl.add(docs)
        else:
            self._connection.create_table(self._table_name, data=docs)
        return ids

    def get_table(self, name: Optional[str] = None) -> Any:
        if name is not None:
            try:
                self._connection.open_table(name)
            except Exception:
                raise ValueError(f"Table {name} not found in the database")
        else:
            return self._connection.open_table(self._table_name)

    def create_index(
        self,
        col_name: Optional[str] = None,
        vector_col: Optional[str] = None,
        num_partitions: Optional[int] = 256,
        num_sub_vectors: Optional[int] = 96,
        index_cache_size: Optional[int] = None,
    ) -> None:
        """
        Create a scalar(for non-vector cols) or a vector index on a table.
        Make sure your vector column has enough data before creating an index on it.

        Args:
            vector_col: Provide if you want to create index on a vector column.
            col_name: Provide if you want to create index on a non-vector column.
            metric: Provide the metric to use for vector index. Defaults to 'L2'
                    choice of metrics: 'L2', 'dot', 'cosine'

        Returns:
            None
        """
        if vector_col:
            self._connection.create_index(
                vector_column_name=vector_col,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_cache_size=index_cache_size,
            )
        elif col_name:
            self._connection.create_scalar_index(col_name)
        else:
            raise ValueError("Provide either vector_col or col_name")

    def similarity_search(
        self, query: str, k: int = 4, name: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.

        Returns:
            List of documents most similar to the query.
        """
        embedding = self._embedding.embed_query(query)  # type: ignore
        tbl = self.get_table(name)
        docs = (
            tbl.search(embedding, vector_column_name=self._vector_key)
            .limit(k)
            .to_arrow()
        )
        columns = docs.schema.names
        return [
            Document(
                page_content=docs[self._text_key][idx].as_py(),
                metadata={
                    col: docs[col][idx].as_py()
                    for col in columns
                    if col != self._text_key
                },
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
        **kwargs: Any,
    ) -> LanceDB:
        instance = LanceDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
        )
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        filter: Optional[str] = None,
        drop_columns: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Allows deleting rows by filtering, by ids or drop columns from the table.

        Args:
            filter: Provide a string SQL expression -  "{col} {operation} {value}".
            ids: Provide list of ids to delete from the table.
            drop_columns: Provide list of columns to drop from the table.
            delete_all: If True, delete all rows from the table.
        """
        tbl = self.get_table(name)
        if filter:
            tbl.delete(filter)
        elif ids:
            tbl.delete("id in ('{}')".format(",".join(ids)))
        elif drop_columns:
            if self.api_key is not None:
                raise NotImplementedError(
                    "Column operations currently not supported in LanceDB Cloud."
                )
            else:
                tbl.drop_columns(drop_columns)
        elif delete_all:
            tbl.delete("true")
        else:
            raise ValueError("Provide either filter, ids, drop_columns or delete_all")
