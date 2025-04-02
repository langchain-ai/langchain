from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger()


class YDBSearchStrategy(str, enum.Enum):
    """Enumerator of the search strategies."""

    INNER_PRODUCT_SIMILARITY = "InnerProductSimilarity"
    COSINE_SIMILARITY = "CosineSimilarity"
    COSINE_DISTANCE = "CosineDistance"
    MANHATTAN_DISTANCE = "ManhattanDistance"
    EUCLIDEAN_DISTANCE = "EuclideanDistance"

    def __str__(self) -> str:
        return self.value


DEFAULT_SEARCH_STRATEGY = YDBSearchStrategy.COSINE_SIMILARITY


def _get_default_column_map_dict() -> Dict[str, str]:
    return {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }


@dataclass
class YDBSettings:
    """`YDB` client configuration.

    Attribute:
        host (str) : An URL to connect to YDB. Defaults to 'localhost'.
        port (int) : URL port to connect with GRPC. Defaults to 2136.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        secure (bool) : Connect to server over secure connection. Defaults to False.
        database (str) : Database name to find the table. Defaults to '/local'.
        table (str) : Table name to operate on. Defaults to 'langchain_store'.
        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
        strategy (str) : Strategy to perform search,
                         supported are ('InnerProductSimilarity', 'CosineSimilarity',
                         'CosineDistance', 'ManhattanDistance',
                         'EuclideanDistance'). Defaults to 'CosineSimilarity'.
                         Enum `YDBSearchStrategy` contains all of them.
        drop_existing_table (bool) : Flag to drop existing table while init.
                                     Defaults to False.
    """

    host: str = "localhost"
    port: int = 2136

    username: Optional[str] = None
    password: Optional[str] = None

    secure: bool = False

    database: str = "/local"
    table: str = "langchain_store"

    column_map: Dict[str, str] = field(default_factory=_get_default_column_map_dict)

    strategy: str = DEFAULT_SEARCH_STRATEGY

    drop_existing_table: bool = False


class YDB(VectorStore):
    """`YDB` vector store.

    To use, you should have the ``ydb-dbapi`` python package installed.
    """

    def __init__(
        self,
        embedding: Embeddings,
        config: Optional[YDBSettings] = None,
        **kwargs: Any,
    ) -> None:
        """YDB Wrapper to LangChain

        Args:
            embedding (Embeddings): embedding function to use
            config (YDBSettings): Configuration to YDB DBAPI
            kwargs (any): Other keyword arguments will pass into ydb-dbapi
        """
        try:
            import ydb
            import ydb_dbapi

            self._ydb_lib = ydb
        except ImportError:
            raise ImportError(
                "Could not import ydb-dbapi python package. "
                "Please install it with `pip install ydb-dbapi`."
            )

        try:
            from tqdm import tqdm

            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x

        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = YDBSettings()

        assert self.config
        assert self.config.host and self.config.port
        assert self.config.database and self.config.table
        assert self.config.column_map and self.config.strategy

        self.sort_order = (
            "DESC" if self.config.strategy.endswith("Similarity") else "ASC"
        )

        self.embedding_function = embedding

        # Create a connection to ydb
        self.connection = ydb_dbapi.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            username=self.config.username,
            password=self.config.password,
            protocol="grpcs" if self.config.secure else "grpc",
            **kwargs,
        )

        if self.config.drop_existing_table:
            self.drop()

        self._execute_query(self._prepare_scheme_query(), ddl=True)

        self._insert_query = self._prepare_insert_query()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self.embedding_function

    def _execute_query(
        self,
        query: str,
        params: Optional[dict] = None,
        ddl: bool = False,
    ) -> List:
        with self.connection.cursor() as cursor:
            if ddl:
                cursor.execute_scheme(query, params)
            else:
                cursor.execute(query, params)

            if cursor.description is None:
                return []

            columns = [col[0] for col in cursor.description]

            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _prepare_scheme_query(self) -> str:
        """Create table schema
        :param dim: dimension of embeddings
        :param index_params: parameters used for index

        This function returns a `CREATE TABLE` statement based on the value of
        `self.config.index_type`.
        If an index type is specified that index will be created, otherwise
        no index will be created.
        In the case of there being no index, a linear scan will be performed
        when the embedding field is queried.
        """
        return f"""
        CREATE TABLE IF NOT EXISTS `{self.config.table}` (
            {self.config.column_map["id"]} Utf8,
            {self.config.column_map["document"]} Utf8,
            {self.config.column_map["embedding"]} String,
            {self.config.column_map["metadata"]} Json,
            PRIMARY KEY ({self.config.column_map["id"]})
        );"""

    def _escape_str(self, text: str) -> str:
        escape = "\\"
        chars_to_escape = ["\\", '"', "'"]
        return "".join(
            [ch if ch not in chars_to_escape else escape + ch for ch in text]
        )

    def _prepare_insert_query(self) -> str:
        return f"""
        DECLARE $id AS Utf8;
        DECLARE $document as Utf8;
        DECLARE $embedding as List<Float>;
        DECLARE $metadata as Json;

        UPSERT INTO `{self.config.table}`
        (
        {self.config.column_map["id"]},
        {self.config.column_map["document"]},
        {self.config.column_map["embedding"]},
        {self.config.column_map["metadata"]}
        )
        VALUES
        (
        $id,
        $document,
        Untag(Knn::ToBinaryStringFloat($embedding), "FloatVector"),
        $metadata
        );
        """

    def _prepare_search_query(
        self,
        k: int,
        filter: Optional[dict],
    ) -> str:
        where_statement = ""
        if filter:
            where_statement = "WHERE "
            metadata_col = self.config.column_map["metadata"]
            stmts = []
            for key, value in filter.items():
                stmts.append(f'JSON_VALUE({metadata_col}, "$.{key}") = "{value}"')

            where_statement = f"WHERE {' AND '.join(stmts)}"

        strategy = self.config.strategy
        embedding_col = self.config.column_map["embedding"]

        return f"""
        DECLARE $embedding as List<Float>;

        $TargetEmbedding = Knn::ToBinaryStringFloat($embedding);

        SELECT
            {self.config.column_map["id"]} as id,
            {self.config.column_map["document"]} as document,
            {self.config.column_map["metadata"]} as metadata,
        Knn::{strategy}({embedding_col}, $TargetEmbedding) as score
        FROM {self.config.table} {where_statement}
        ORDER BY score
        {self.sort_order}
        LIMIT {k};
        """

    def _prepare_delete_query(self, ids: Optional[list[str]]) -> str:
        query = f"DELETE FROM {self.config.table}"
        if ids:
            query += f" WHERE {self.config.column_map['id']} IN {str(ids)}"
        return query

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            **kwargs: vectorstore specific parameters.
                One of the kwargs should be `ids` which is a list of ids
                associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts_ = texts if isinstance(texts, (list, tuple)) else list(texts)

        if ids is None:
            ids = [sha1(t.encode("utf-8")).hexdigest() for t in texts_]

        if metadatas and len(metadatas) != len(texts_):
            msg = (
                "The number of metadatas must match the number of texts."
                f"Got {len(metadatas)} metadatas and {len(texts_)} texts."
            )
            raise ValueError(msg)

        metadatas = metadatas if metadatas else [{} for _ in range(len(texts_))]

        ydb = self._ydb_lib

        for id, text, metadata in self.pgbar(
            zip(ids, texts, metadatas),
            desc="Inserting data...",
            total=len(ids),
        ):
            embedding = self.embedding_function.embed_query(text)
            self._execute_query(
                self._insert_query,
                {
                    "$id": id,
                    "$document": text,
                    "$embedding": (embedding, ydb.ListType(ydb.PrimitiveType.Float)),
                    "$metadata": (json.dumps(metadata), ydb.PrimitiveType.Json),
                },
            )

        return ids

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        config: Optional[YDBSettings] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> YDB:
        """Return YDB VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            ids: Optional list of IDs associated with the texts.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        vs = cls(embedding, config, **kwargs)
        vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vs

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        query = self._prepare_delete_query(ids)
        self._execute_query(query)
        return True

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        ydb = self._ydb_lib
        query = self._prepare_search_query(k, filter=filter)
        res = self._execute_query(
            query,
            params={"$embedding": (embedding, ydb.ListType(ydb.PrimitiveType.Float))},
        )
        return [
            Document(
                page_content=row["document"],
                metadata=json.loads(row["metadata"]),
            )
            for row in res
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            *args: Arguments to pass to the search method.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        ydb = self._ydb_lib
        embedding = self.embedding_function.embed_query(query)
        query = self._prepare_search_query(k, filter=filter)
        res = self._execute_query(
            query,
            params={"$embedding": (embedding, ydb.ListType(ydb.PrimitiveType.Float))},
        )
        return [
            (
                Document(
                    page_content=row[self.config.column_map["document"]],
                    metadata=json.loads(row["metadata"]),
                ),
                row["score"],
            )
            for row in res
        ]

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self._execute_query(
            f"DROP TABLE IF EXISTS `{self.config.table}`",
            ddl=True,
        )
