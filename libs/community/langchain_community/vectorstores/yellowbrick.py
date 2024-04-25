from __future__ import annotations

import csv
import enum
import json
import logging
import random
import time
import uuid
import warnings
from contextlib import contextmanager
from io import StringIO
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.document import Document

logger = logging.getLogger(__name__)


class Yellowbrick(VectorStore):
    """Yellowbrick as a vector database.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import Yellowbrick
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """

    class IndexType(str, enum.Enum):
        """Enumerator for the supported Index types within Yellowbrick."""

        NONE = "none"
        LSH = "lsh"

    class IndexParams:
        """Parameters for configuring a Yellowbrick index."""

        def __init__(
            self,
            index_type: Optional["Yellowbrick.IndexType"] = None,
            params: Optional[Dict[str, Any]] = None,
        ):
            if index_type is None:
                index_type = Yellowbrick.IndexType.NONE
            self.index_type = index_type
            self.params = params or {}

        def get_param(self, key: str, default: Any = None) -> Any:
            return self.params.get(key, default)

    def __init__(
        self,
        embedding: Embeddings,
        connection_info: Union[str, Any],
        table: str,
        *,
        idle_threshold_seconds: Optional[int] = 300,
        seed: Optional[float] = 0.42,
        drop: Optional[bool] = False,
    ) -> None:
        """Initialize with yellowbrick client.
        Args:
            embedding: Embedding operator
            connection_info: Format 'postgres://username:password@host:port/database'
            or connection object
            table: Table used to store / retrieve embeddings from
        """

        import psycopg2
        from psycopg2 import extras

        extras.register_uuid()

        if not isinstance(embedding, Embeddings):
            warnings.warn("embeddings input must be Embeddings object.")

        self.LSH_INDEX_TABLE: str = "_lsh_index"
        self.LSH_HYPERPLANE_TABLE: str = "_lsh_hyperplane"
        self.CONTENT_TABLE: str = "_content"

        if isinstance(connection_info, str):
            self.connection_string = connection_info
            self._connection = psycopg2.connect(self.connection_string)
        elif isinstance(connection_info, psycopg2.extensions.connection):
            self.connection = connection_info
        else:
            raise ValueError(
                """connection_info must be either a connection string 
                or a psycopg2 connection object"""
            )

        self._table = table
        self._embedding = embedding
        self._max_embedding_len = None
        self.idle_threshold_seconds = idle_threshold_seconds
        self.last_used_time = time.time()
        self._check_database_utf8()

        if drop:
            self.drop(self._table)
            self.drop(self._table + self.CONTENT_TABLE)
            self._drop_lsh_index_tables()

        self._create_table()

        if seed is not None:
            random.seed(seed)
        self._seed = seed

    def __del__(self) -> None:
        if self._connection:
            self._connection.close()

    @contextmanager
    def _get_cursor(self) -> Any:
        cursor = self._get_connection().cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _get_connection(self) -> Any:
        import psycopg2
        from psycopg2 import Error, OperationalError

        current_time = time.time()
        if self.idle_threshold_seconds is None:
            self.idle_threshold_seconds = 300
        if self._connection.closed:
            if self.connection_string:
                self._connection = psycopg2.connect(self.connection_string)
                self.last_used_time = current_time
            else:
                self._connection = None
        elif (current_time - self.last_used_time) > self.idle_threshold_seconds:
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("SELECT 1")
            except (OperationalError, Error):
                if self.connection_string:
                    self._connection = psycopg2.connect(self.connection_string)
                else:
                    self._connection = None
            self.last_used_time = current_time

        return self._connection

    def _create_table(self) -> None:
        """
        Helper function: create table if not exists
        """
        from psycopg2 import sql

        with self._get_cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t} (
                    doc_id UUID NOT NULL,
                    text VARCHAR(60000) NOT NULL,
                    metadata VARCHAR(1024) NOT NULL,
                    CONSTRAINT {c} PRIMARY KEY (doc_id))
                    DISTRIBUTE ON (doc_id) SORT ON (doc_id)
                """
                ).format(
                    t=sql.Identifier(self._table + self.CONTENT_TABLE),
                    c=sql.Identifier(self._table + self.CONTENT_TABLE + "_pk_doc_id"),
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t1} (
                    doc_id UUID NOT NULL,
                    embedding_id SMALLINT NOT NULL,
                    embedding FLOAT NOT NULL,
                    CONSTRAINT {c1} PRIMARY KEY (doc_id, embedding_id),
                    CONSTRAINT {c2} FOREIGN KEY (doc_id) REFERENCES {t2}(doc_id))
                    DISTRIBUTE ON (doc_id) SORT ON (doc_id)
                """
                ).format(
                    t1=sql.Identifier(self._table),
                    t2=sql.Identifier(self._table + self.CONTENT_TABLE),
                    c1=sql.Identifier(
                        self._table + self.CONTENT_TABLE + "_pk_doc_id_embedding_id"
                    ),
                    c2=sql.Identifier(self._table + self.CONTENT_TABLE + "_fk_doc_id"),
                )
            )
            self._get_connection().commit()

    def drop(self, table: str) -> None:
        """
        Helper function: Drop data
        """
        from psycopg2 import sql

        with self._get_cursor() as cursor:
            cursor.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table))
            )
            self._get_connection().commit()

    def _check_database_utf8(self) -> bool:
        """
        Helper function: Test the database is UTF-8 encoded
        """
        with self._get_cursor() as cursor:
            query = """
                SELECT pg_encoding_to_char(encoding)
                FROM pg_database
                WHERE datname = current_database();
            """
            cursor.execute(query)
            encoding = cursor.fetchone()[0]

        if encoding.lower() == "utf8" or encoding.lower() == "utf-8":
            return True
        else:
            raise Exception("Database encoding is not UTF-8")

        return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        batch_size = 10000

        texts = list(texts)
        embeddings = self._embedding.embed_documents(list(texts))
        results = []
        if not metadatas:
            metadatas = [{} for _ in texts]

        index_params = kwargs.get("index_params", Yellowbrick.IndexParams())

        with self._get_cursor() as cursor:
            content_io = StringIO()
            embeddings_io = StringIO()
            content_writer = csv.writer(
                content_io, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            embeddings_writer = csv.writer(
                embeddings_io, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            current_batch_size = 0

            for i, text in enumerate(texts):
                doc_uuid = str(uuid.uuid4())
                results.append(doc_uuid)

                content_writer.writerow([doc_uuid, text, json.dumps(metadatas[i])])

                for embedding_id, embedding in enumerate(embeddings[i]):
                    embeddings_writer.writerow([doc_uuid, embedding_id, embedding])

                current_batch_size += 1

                if current_batch_size >= batch_size:
                    self._copy_to_db(cursor, content_io, embeddings_io, self._table)

                    content_io.seek(0)
                    content_io.truncate(0)
                    embeddings_io.seek(0)
                    embeddings_io.truncate(0)
                    current_batch_size = 0

            if current_batch_size > 0:
                self._copy_to_db(cursor, content_io, embeddings_io, self._table)

            self._get_connection().commit()

        if index_params.index_type == Yellowbrick.IndexType.LSH:
            self.update_index(index_params, uuid.UUID(doc_uuid))

        return results

    def _copy_to_db(
        self,
        cursor: Any,
        content_io: StringIO,
        embeddings_io: StringIO,
        table_name: str,
    ) -> None:
        content_io.seek(0)
        embeddings_io.seek(0)
        cursor.copy_expert(
            f"COPY {table_name + self.CONTENT_TABLE} (doc_id, text, metadata) \
                FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')",
            content_io,
        )
        cursor.copy_expert(
            f"COPY {table_name} (doc_id, embedding_id, embedding) \
                FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')",
            embeddings_io,
        )

    @classmethod
    def from_texts(
        cls: Type[Yellowbrick],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_info: Union[str, Any] = None,
        table: str = "langchain",
        drop: Optional[bool] = False,
        **kwargs: Any,
    ) -> Yellowbrick:
        """Add texts to the vectorstore index.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            connection_info: URI to Yellowbrick instance or a connection object
            embedding: Embedding function
            table: table to store embeddings
            kwargs: vectorstore specific parameters
        """
        vss = cls(
            embedding=embedding,
            connection_info=connection_info,
            table=table,
            drop=drop,
        )
        vss.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return vss

    def _generate_vector_uuid(self, vector: List[float]) -> uuid.UUID:
        import hashlib

        vector_str = ",".join(map(str, vector))
        hash_object = hashlib.sha1(vector_str.encode())
        hash_digest = hash_object.digest()
        vector_uuid = uuid.UUID(bytes=hash_digest[:16])
        return vector_uuid

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Yellowbrick with vector

        Args:
            embedding (List[float]): query embedding
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document, float]: List of Documents and scores
        """
        from psycopg2 import sql
        from psycopg2.extras import execute_values

        index_params = kwargs.get("index_params", Yellowbrick.IndexParams())

        with self._get_cursor() as cursor:
            tmp_embeddings_table = "tmp_" + self._table
            tmp_doc_id = self._generate_vector_uuid(embedding)
            create_table_query = sql.SQL(
                """ 
                CREATE TEMPORARY TABLE {} (
                doc_id UUID,
                embedding_id SMALLINT,
                embedding FLOAT)
                DISTRIBUTE REPLICATE
            """
            ).format(sql.Identifier(tmp_embeddings_table))
            cursor.execute(create_table_query)

            data_input = [
                (str(tmp_doc_id), embedding_id, embedding_value)
                for embedding_id, embedding_value in enumerate(embedding)
            ]

            insert_query = sql.SQL(
                "INSERT INTO {table} (doc_id, embedding_id, embedding) VALUES %s"
            ).format(table=sql.Identifier(tmp_embeddings_table))
            execute_values(cursor, insert_query, data_input)
            self._get_connection().commit()

            if index_params.index_type == Yellowbrick.IndexType.LSH:
                input_hash_table = self._table + "_tmp_hash"
                self._generate_lsh_hashes(
                    embedding_table=tmp_embeddings_table,
                    target_hash_table=input_hash_table,
                )
                sql_query = sql.SQL(
                    """
                    WITH index_docs AS (
                    SELECT
                        t1.doc_id,
                        SUM(ABS(t1.hash-t2.hash)) as hamming_distance
                    FROM
                        {lsh_index} t1
                    INNER JOIN
                        {input_hash_table} t2
                    ON t1.hash_index = t2.hash_index
                    GROUP BY t1.doc_id
                    HAVING hamming_distance <= {hamming_distance}
                    )
                    SELECT
                        text,
                        metadata,
                       SUM(v1.embedding * v2.embedding) /
                        (SQRT(SUM(v1.embedding * v1.embedding)) *
                       SQRT(SUM(v2.embedding * v2.embedding))) AS score
                    FROM
                        {v1} v1
                    INNER JOIN
                        {v2} v2
                    ON v1.embedding_id = v2.embedding_id
                    INNER JOIN
                        {v3} v3
                    ON v2.doc_id = v3.doc_id
                    INNER JOIN
                        index_docs v4
                    ON v2.doc_id = v4.doc_id
                    GROUP BY v3.doc_id, v3.text, v3.metadata
                    ORDER BY score DESC
                    LIMIT %s
                """
                ).format(
                    v1=sql.Identifier(tmp_embeddings_table),
                    v2=sql.Identifier(self._table),
                    v3=sql.Identifier(self._table + self.CONTENT_TABLE),
                    lsh_index=sql.Identifier(self._table + self.LSH_INDEX_TABLE),
                    input_hash_table=sql.Identifier(input_hash_table),
                    hamming_distance=sql.Literal(
                        index_params.get_param("hamming_distance", 0)
                    ),
                )
                cursor.execute(
                    sql_query,
                    (k,),
                )
                self.drop(input_hash_table)
            else:
                sql_query = sql.SQL(
                    """
                    SELECT 
                        text,
                        metadata,
                        score
                    FROM
                        (SELECT
                            v2.doc_id doc_id,
                            SUM(v1.embedding * v2.embedding) /
                            (SQRT(SUM(v1.embedding * v1.embedding)) *
                            SQRT(SUM(v2.embedding * v2.embedding))) AS score
                        FROM
                            {v1} v1
                        INNER JOIN
                            {v2} v2
                        ON v1.embedding_id = v2.embedding_id
                        GROUP BY v2.doc_id
                        ORDER BY score DESC LIMIT %s
                        ) v4
                    INNER JOIN
                        {v3} v3
                    ON v4.doc_id = v3.doc_id
                    ORDER BY score DESC
                """
                ).format(
                    v1=sql.Identifier(tmp_embeddings_table),
                    v2=sql.Identifier(self._table),
                    v3=sql.Identifier(self._table + self.CONTENT_TABLE),
                )
                cursor.execute(sql_query, (k,))

            self.drop(tmp_embeddings_table)

            results = cursor.fetchall()

            documents: List[Tuple[Document, float]] = []
            for result in results:
                metadata = json.loads(result[1]) or {}
                doc = Document(page_content=result[0], metadata=metadata)
                documents.append((doc, result[2]))

        return documents

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with Yellowbrick

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of Documents
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Yellowbrick

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with Yellowbrick by vectors

        Args:
            embedding (List[float]): query embedding
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of documents
        """
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _ in documents]

    def _generate_lsh_hashes(
        self,
        embedding_table: str,
        target_hash_table: Optional[str] = None,
        doc_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Generate hashes for vectors"""
        from psycopg2 import sql

        if doc_id:
            condition = sql.SQL("WHERE e.doc_id = {doc_id}").format(
                doc_id=sql.Literal(str(doc_id))
            )
            group_by = sql.SQL("GROUP BY 1, 2")
        else:
            condition = sql.SQL("")
            group_by = (
                sql.SQL("GROUP BY 1") if target_hash_table else sql.SQL("GROUP BY 1, 2")
            )

        if target_hash_table:
            table_name = sql.Identifier(target_hash_table)
            query_prefix = sql.SQL("CREATE TEMPORARY TABLE {table_name} AS").format(
                table_name=table_name
            )
        else:
            table_name = sql.Identifier(self._table + self.LSH_INDEX_TABLE)
            query_prefix = sql.SQL("INSERT INTO {table_name}").format(
                table_name=table_name
            )

        input_query = query_prefix + sql.SQL(
            """
            SELECT
                {select_columns}
                h.id as hash_index,
                CASE WHEN
                    SUM(e.embedding * h.hyperplane) > 0
                THEN
                    1
                ELSE
                    0
                END as hash
            FROM {embedding_table} e
            INNER JOIN {hyperplanes} h ON e.embedding_id = h.hyperplane_id
            {condition}
            {group_by}
            """
        ).format(
            select_columns=sql.SQL("e.doc_id,")
            if not target_hash_table or doc_id
            else sql.SQL(""),
            embedding_table=sql.Identifier(embedding_table),
            hyperplanes=sql.Identifier(self._table + self.LSH_HYPERPLANE_TABLE),
            condition=condition,
            group_by=group_by,
        )

        with self._get_cursor() as cursor:
            cursor.execute(input_query)
            self._get_connection().commit()

    def _populate_hyperplanes(self, num_hyperplanes: int) -> None:
        """Generate random hyperplanes and store in Yellowbrick"""
        from psycopg2 import sql

        with self._get_cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {};").format(
                    sql.Identifier(self._table + self.LSH_HYPERPLANE_TABLE)
                )
            )
            if cursor.fetchone()[0] > 0:
                return

            cursor.execute(
                sql.SQL("SELECT MAX(embedding_id) FROM {};").format(
                    sql.Identifier(self._table)
                )
            )
            num_dimensions = cursor.fetchone()[0]
            num_dimensions += 1

            insert_query = sql.SQL(
                """
                WITH parameters AS (
                    SELECT {num_hyperplanes} AS num_hyperplanes,
                        {dims_per_hyperplane} AS dims_per_hyperplane
                ),
                seed AS (
                    SELECT setseed({seed_value})
                )
                INSERT INTO {hyperplanes_table} (id, hyperplane_id, hyperplane)
                    SELECT id, hyperplane_id, (random() * 2 - 1) AS hyperplane
                    FROM
                    (SELECT range-1 id FROM sys.rowgenerator
                        WHERE range BETWEEN 1 AND
                        (SELECT num_hyperplanes FROM parameters) AND
                        worker_lid = 0 AND thread_id = 0) a,
                    (SELECT range-1 hyperplane_id FROM sys.rowgenerator
                        WHERE range BETWEEN 1 AND
                        (SELECT dims_per_hyperplane FROM parameters) AND
                        worker_lid = 0 AND thread_id = 0) b
            """
            ).format(
                num_hyperplanes=sql.Literal(num_hyperplanes),
                dims_per_hyperplane=sql.Literal(num_dimensions),
                hyperplanes_table=sql.Identifier(
                    self._table + self.LSH_HYPERPLANE_TABLE
                ),
                seed_value=sql.Literal(self._seed),
            )
            cursor.execute(insert_query)
            self._get_connection().commit()

    def _create_lsh_index_tables(self) -> None:
        """Create LSH index and hyperplane tables"""
        from psycopg2 import sql

        with self._get_cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t1} (
                    doc_id UUID NOT NULL,
                    hash_index SMALLINT NOT NULL,
                    hash SMALLINT NOT NULL,
                    CONSTRAINT {c1} PRIMARY KEY (doc_id, hash_index),
                    CONSTRAINT {c2} FOREIGN KEY (doc_id) REFERENCES {t2}(doc_id))
                    DISTRIBUTE ON (doc_id) SORT ON (doc_id)
                """
                ).format(
                    t1=sql.Identifier(self._table + self.LSH_INDEX_TABLE),
                    t2=sql.Identifier(self._table + self.CONTENT_TABLE),
                    c1=sql.Identifier(
                        self._table + self.LSH_INDEX_TABLE + "_pk_doc_id"
                    ),
                    c2=sql.Identifier(
                        self._table + self.LSH_INDEX_TABLE + "_fk_doc_id"
                    ),
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {t} (
                    id SMALLINT NOT NULL,
                    hyperplane_id SMALLINT NOT NULL,
                    hyperplane FLOAT NOT NULL,
                    CONSTRAINT {c} PRIMARY KEY (id, hyperplane_id))
                    DISTRIBUTE REPLICATE SORT ON (id)
                """
                ).format(
                    t=sql.Identifier(self._table + self.LSH_HYPERPLANE_TABLE),
                    c=sql.Identifier(
                        self._table + self.LSH_HYPERPLANE_TABLE + "_pk_id_hp_id"
                    ),
                )
            )
            self._get_connection().commit()

    def _drop_lsh_index_tables(self) -> None:
        """Drop LSH index tables"""
        self.drop(self._table + self.LSH_INDEX_TABLE)
        self.drop(self._table + self.LSH_HYPERPLANE_TABLE)

    def create_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Create index from existing vectors"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            self._drop_lsh_index_tables()
            self._create_lsh_index_tables()
            self._populate_hyperplanes(index_params.get_param("num_hyperplanes", 128))
            self._generate_lsh_hashes(embedding_table=self._table)

    def drop_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Drop an index"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            self._drop_lsh_index_tables()

    def update_index(
        self, index_params: Yellowbrick.IndexParams, doc_id: uuid.UUID
    ) -> None:
        """Update an index with a new or modified embedding in the embeddings table"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            self._generate_lsh_hashes(embedding_table=self._table, doc_id=doc_id)
