"""Test HANA functionality."""
from __future__ import annotations

import json
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.vectorstores.utils import DistanceStrategy

if TYPE_CHECKING:
    from hdbcli import dbapi

HANA_DISTANCE_FUNCTION: dict = {
    DistanceStrategy.COSINE: ("COSINE_SIMILARITY", "DESC"),
    DistanceStrategy.EUCLIDEAN_DISTANCE: ("L2DISTANCE", "ASC"),
}

class HanaDB(VectorStore):
    """`HANA DB` vector store.

    The prerequisite for using this class is the installation of the ``hdbcli``
    Python package.

    The HanaDB vectorstore can be created by providing an embedding function and
    the relevant parameters for the database connection, connection pool, and
    optionally, the names of the table and the fields to use.
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        table_name: str = "EMBEDDINGS",
        content_field: str = "DOC_TEXT",
        content_field_length: int = 2048,
        metadata_field: str = "DOC_META",
        metadata_field_length: int = 2048,
        vector_field: str = "DOC_VECTOR",
        vector_field_length: int = -1, # -1 means dynamic length
        **kwargs: Any,
    ):
        try:
            from hdbcli import dbapi
        except ImportError:
            raise ImportError(
                "Could not import hdbcli python package. "
                "Please install it with `pip install hdbcli`."
            )
        
        valid_distance = False
        for key in HANA_DISTANCE_FUNCTION.keys():
            if key is distance_strategy:
                valid_distance = True
        if not valid_distance:
            raise ValueError("Unsupported distance_strategy: {}".format(distance_strategy))
        
        self.connection = connection
        self.embedding = embedding
        self.distance_strategy = distance_strategy
        self.table_name = self._sanitize_input(table_name)
        self.content_field = self._sanitize_input(content_field)
        self.content_field_length = HanaDB._sanitize_int(content_field_length)
        self.metadata_field = self._sanitize_input(metadata_field)
        self.metadata_field_length = HanaDB._sanitize_int(metadata_field_length)
        self.vector_field = self._sanitize_input(vector_field)
        self.vector_field_length = HanaDB._sanitize_int(vector_field_length)

        # Check if the table exists, and eventually create it
        if not self._table_exists(self.table_name):
            sql_str = f"CREATE TABLE {self.table_name}({self.content_field} NVARCHAR({self.content_field_length}), {self.metadata_field} NVARCHAR({self.metadata_field_length}), {self.vector_field} REAL_VECTOR"
            if self.vector_field_length == -1:
                sql_str += f");"
            else:
                sql_str += f"({self.vector_field_length}));"

            try:
                cur = self.connection.cursor()
                cur.execute(sql_str)
            finally:
                cur.close()

        # Check if the needed columns exists
        self._check_column(self.table_name, self.content_field, "NVARCHAR", self.content_field_length)
        self._check_column(self.table_name, self.metadata_field, "NVARCHAR", self.metadata_field_length)
        self._check_column(self.table_name, self.vector_field, "REAL_VECTOR", self.vector_field_length)

    def _table_exists(self, table_name) -> bool:
        sql_str = "SELECT COUNT(*) FROM TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?"
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                print(rows)
                if rows[0][0] == 1:
                    return True
        finally:
            cur.close()
        return False


    def _check_column(self, table_name, column_name, column_type, column_length):
        sql_str = "SELECT DATA_TYPE_NAME, LENGTH FROM TABLE_COLUMNS WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ? AND COLUMN_NAME = ?"
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name, column_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                if len(rows) == 0:
                    raise AttributeError(f"Column {column_name} does not exist")
                # Check data type
                if rows[0][0] != column_type:
                    raise AttributeError(f"Column {column_name} has the wrong type: {rows[0][0]}")
                if rows[0][1] != column_length:
                    raise AttributeError(f"Column {column_name} has the wrong length: {rows[0][1]}")
            else: 
                raise AttributeError(f"Column {column_name} does not exist")
        finally:
            cur.close()


    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    def _sanitize_int(input_int: any) -> int:
        value = int(str(input_int))
        if value < -1:
            raise ValueError(f"Value ({value}) must not be smaller than -1")
        return int(str(input_int))

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: empty list
        """
        cur = self.connection.cursor()
        try:
            # Write data to singlestore db
            for i, text in enumerate(texts):
                # Use provided values by default or fallback
                metadata = metadatas[i] if metadatas else {}
                embedding = (
                    embeddings[i]
                    if embeddings
                    else self.embedding.embed_documents([text])[0]
                )
                sql_str = "INSERT INTO {} (DOC_TEXT, DOC_META, DOC_VECTOR) VALUES (?, ?, TO_REAL_VECTOR (ARRAY({})));".format(
                        self.table_name,
                        "{}".format(",".join(map(str, embedding)))
                    )
                # print(sql_str)
                cur.execute(
                    sql_str,
                    (
                        text,
                        json.dumps(metadata)
                    ),
                )
            self.connection.commit()
        finally:
            cur.close()
        return []

    @classmethod
    def from_texts(
        cls: Type[HanaDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ):
        """Create a HANA vectorstore from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new table for the embeddings in HANA.
            3. Adds the documents to the newly created table.
        This is intended to be a quick way to get started.
        """

        instance = cls(
            texts,
            metadatas,
            embedding,
            **kwargs,
        )
        instance.add_texts(texts, metadatas, embedding.embed_documents(texts), **kwargs)
        return instance

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        # Creates embedding vector from user query
        embedding = self.embedding.embed_query(query)
        result = []
        sql_str = "SELECT TOP {} * , {} (DOC_VECTOR, TO_REAL_VECTOR (ARRAY({}))) AS CS FROM {}".format(
            k, HANA_DISTANCE_FUNCTION[self.distance_strategy][0], "{}".format(",".join(map(str, embedding))), self.table_name)
        order_str = " order by CS {}".format(
            HANA_DISTANCE_FUNCTION[self.distance_strategy][1])
        where_str = " "
        if filter:
            for i, key in enumerate(filter.keys()):
                if i == 0:
                    where_str = where_str + " WHERE "
                else:
                    where_str = where_str + " AND "

                where_str = where_str + " JSON_QUERY({}, '$.{}') = '{}'".format(self.metadata_field, key, filter[key])
        sql_str = sql_str + where_str
        sql_str = sql_str + order_str
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str)
            if cur.has_result_set():
                rows = cur.fetchall()
                for row in rows:
                    doc = Document(page_content=row[0])
                    result.append((doc, row[-1]))
        finally:
            cur.close()
        return result

