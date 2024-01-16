"""Test HANA functionality."""
from __future__ import annotations

import json
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

if TYPE_CHECKING:
    from hdbcli import dbapi

HANA_DISTANCE_FUNCTION: dict = {
    DistanceStrategy.COSINE: ("COSINE_SIMILARITY", "DESC"),
    DistanceStrategy.EUCLIDEAN_DISTANCE: ("L2DISTANCE", "ASC"),
}


default_distance_strategy = DistanceStrategy.COSINE
default_table_name: str = "EMBEDDINGS"
default_content_field: str = "DOC_TEXT"
default_content_field_length: int = 2048
default_metadata_field: str = "DOC_META"
default_metadata_field_length: int = 2048
default_vector_field: str = "DOC_VECTOR"
default_vector_field_length: int = -1  # -1 means dynamic length


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
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_field: str = default_content_field,
        content_field_length: int = default_content_field_length,
        metadata_field: str = default_metadata_field,
        metadata_field_length: int = default_metadata_field_length,
        vector_field: str = default_vector_field,
        vector_field_length: int = default_vector_field_length,  # -1 means dynamic length
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
            raise ValueError(
                "Unsupported distance_strategy: {}".format(distance_strategy)
            )

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
        self._check_column(
            self.table_name, self.content_field, "NVARCHAR", self.content_field_length
        )
        self._check_column(
            self.table_name, self.metadata_field, "NVARCHAR", self.metadata_field_length
        )
        self._check_column(
            self.table_name, self.vector_field, "REAL_VECTOR", self.vector_field_length
        )

    def _table_exists(self, table_name) -> bool:
        sql_str = "SELECT COUNT(*) FROM TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?"
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name))
            if cur.has_result_set():
                rows = cur.fetchall()
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
                    raise AttributeError(
                        f"Column {column_name} has the wrong type: {rows[0][0]}"
                    )
                if rows[0][1] != column_length:
                    raise AttributeError(
                        f"Column {column_name} has the wrong length: {rows[0][1]}"
                    )
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
        # Create all embeddings of the texts beforehand to improve performance
        if embeddings == None:
            embeddings = self.embedding.embed_documents(list(texts))

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
                sql_str = f"INSERT INTO {self.table_name} ({self.content_field}, {self.metadata_field}, {self.vector_field}) VALUES (?, ?, TO_REAL_VECTOR (?));"
                cur.execute(
                    sql_str,
                    (
                        text,
                        json.dumps(metadata),
                        "[{}]".format(",".join(map(str, embedding))),
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
        connection: dbapi.Connection = None,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_field: str = default_content_field,
        content_field_length: int = default_content_field_length,
        metadata_field: str = default_metadata_field,
        metadata_field_length: int = default_metadata_field_length,
        vector_field: str = default_vector_field,
        vector_field_length: int = default_vector_field_length,  # -1 means dynamic length
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
            connection=connection,
            texts=texts,
            metadatas=metadatas,
            embedding=embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_field=content_field,
            content_field_length=content_field_length,
            metadata_field=metadata_field,
            metadata_field_length=metadata_field_length,
            vector_field=vector_field,
            vector_field_length=vector_field_length,  # -1 means dynamic length
            **kwargs,
        )
        instance.add_texts(texts, metadatas, **kwargs)
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
        """Return docs most similar to query. Expects the embeddings in the database to be normalized.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query. Expects the embeddings in the database to be normalized.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        result = []
        sql_str = f"SELECT TOP {k} {self.content_field}, {self.metadata_field} , {HANA_DISTANCE_FUNCTION[self.distance_strategy][0]} ({self.vector_field}, TO_REAL_VECTOR (ARRAY({'{}'.format(','.join(map(str, embedding)))}))) AS CS FROM {self.table_name}"
        order_str = f" order by CS {HANA_DISTANCE_FUNCTION[self.distance_strategy][1]}"
        where_str = self.create_where_by_filter(filter)
        sql_str = sql_str + where_str
        sql_str = sql_str + order_str
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str)
            if cur.has_result_set():
                rows = cur.fetchall()
                for row in rows:
                    js = json.loads(row[1])
                    doc = Document(page_content=row[0], metadata=js)
                    result.append((doc, row[-1]))
        finally:
            cur.close()
        return result

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, filter: Optional[dict] = None
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def create_where_by_filter(self, filter):
        where_str = ""
        if filter:
            for i, key in enumerate(filter.keys()):
                if i == 0:
                    where_str = where_str + " WHERE "
                else:
                    where_str = where_str + " AND "

                where_str = (
                    where_str
                    + f" JSON_QUERY({self.metadata_field}, '$.{key}') = '{filter[key]}'"
                )
        return where_str

    def delete(
        self, ids: Optional[List[str]] = None, filter: Optional[dict] = None
    ) -> Optional[bool]:
        """Delete by filter with metadata values

        Args:
            ids: Deletion with ids is not supported! A ValueError will be raised.
            filter: A dictionary of metadata fields and values to filter by.
                    An empty filter ({}) will delete all entries in the table.

        Returns:
            Optional[bool]: True, if deletion is technically successful.
                            Deletion of zero entries, due to non-matching filters is considered successs.
        """

        if ids != None:
            raise ValueError("Deletion via ids is not supported")

        if filter == None:
            raise ValueError("Parameter 'filter' is required when calling 'delete'")

        where_str = self.create_where_by_filter(filter)
        sql_str = f"DELETE FROM {self.table_name} {where_str}"

        try:
            cur = self.connection.cursor()
            cur.execute(sql_str)
        finally:
            cur.close()

        return True

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    @classmethod
    def _parse_float_array_from_string(
        cls: Type[HanaDB], array_as_string: str
    ) -> List[float]:
        array_wo_brackets = array_as_string[1:-1]
        return [float(x) for x in array_wo_brackets.split(",")]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs = []
        embeddings = []
        sql_str = f"SELECT TOP {k} {self.content_field}, {self.metadata_field}, TO_NVARCHAR({self.vector_field}), {HANA_DISTANCE_FUNCTION[self.distance_strategy][0]} ({self.vector_field}, TO_REAL_VECTOR (ARRAY({'{}'.format(','.join(map(str, embedding)))}))) AS CS FROM {self.table_name}"
        order_str = f" order by CS {HANA_DISTANCE_FUNCTION[self.distance_strategy][1]}"
        where_str = self.create_where_by_filter(filter)
        sql_str = sql_str + where_str
        sql_str = sql_str + order_str
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str)
            if cur.has_result_set():
                rows = cur.fetchall()
                for row in rows:
                    js = json.loads(row[1])
                    doc = Document(page_content=row[0], metadata=js)
                    docs.append((doc, row[-1]))
                    embeddings.append(HanaDB._parse_float_array_from_string(row[2]))
        finally:
            cur.close()

        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        return [docs[i][0] for i in mmr_doc_indexes]

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        return distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        if self.distance_strategy == DistanceStrategy.COSINE:
            return HanaDB._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return HanaDB._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unsupported distance_strategy: {}".format(self.distance_strategy)
            )
