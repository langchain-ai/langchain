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
        metadata_field: str = "DOC_META",
        vector_field: str = "DOC_VECTOR",
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
        self.metadata_field = self._sanitize_input(metadata_field)
        self.vector_field = self._sanitize_input(vector_field)

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

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

