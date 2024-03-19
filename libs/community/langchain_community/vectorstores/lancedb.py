from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


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
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
    ):
        """Initialize with Lance DB vectorstore"""
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "Could not import lancedb python package. "
                "Please install it with `pip install lancedb`."
            )
        self.lancedb = lancedb
        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
        self._table_name = table_name

        if self._embedding is None:
            raise ValueError("embedding should be provided")

        if connection is not None:
            if not isinstance(connection, lancedb.db.LanceTable):
                raise ValueError(
                    "connection should be an instance of lancedb.db.LanceTable, ",
                    f"got {type(connection)}",
                )
            self._connection = connection
        else:
            self._connection = None

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
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))  # type: ignore
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    'metadata': metadata,
                }
            )
        if self._connection is None:
            self._connection = self._init_table(docs)
            self._connection.add(docs)
        else:
            self._connection.add(docs)
        return ids
    
    def create_index(self,
                     col_name: Optional[str]= None,
                     vector_col: Optional[str]= None,
                     num_partitions: Optional[int]= 256,
                     num_sub_vectors: Optional[int]= 96,
                     index_cache_size: Optional[int]= None
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
            self._connection.create_index(vector_column_name=vector_col,
                                          num_partitions=num_partitions,
                                          num_sub_vectors=num_sub_vectors,
                                          index_cache_size=index_cache_size)
        elif col_name:
            self._connection.create_scalar_index(col_name)
        else:
            raise ValueError("Provide either vector_col or col_name")
        
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
        embedding = self._embedding.embed_query(query)  # type: ignore
        docs = (
            self._connection.search(embedding, vector_column_name=self._vector_key)
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
            connection,
            embedding,
            vector_key,
            id_key,
            text_key,
        )
        instance.add_texts(texts, metadatas=metadatas, **kwargs)

        return instance

    def _init_table(self,data) -> Any:
        db = self.lancedb.connect("/tmp/lancedb")
        tbl = db.create_table(self._table_name, data=data, mode="overwrite")
        return tbl

    def delete(self,
               filter: Optional[str] = None,
               ids: Optional[List[str]] = None,
               drop_columns: Optional[List[str]] = None,
               delete_all: Optional[bool] = False,
               ) -> Any:
        """
        Allows deleting rows by filtering, by ids or drop columns from the table.

        Args:
            filter: Provide a string SQL expression -  "{col} {operation} {value}".
            ids: Provide list of ids to delete from the table.
            drop_columns: Provide list of columns to drop from the table.
            delete_all: If True, delete all rows from the table.
        """
        if filter:
            self._connection.delete(filter)
        elif ids:
            self._connection.delete("id in ('{}')".format(",".join(ids)))
        elif drop_columns:
            self._connection.drop_columns(drop_columns)
        elif delete_all:
            self._connection.delete("true")
        else: 
            raise ValueError("Provide either filter, ids, drop_columns or delete_all")
        