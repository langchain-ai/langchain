from __future__ import annotations

import atexit
import csv
import enum
import json
import logging
import uuid
from contextlib import contextmanager
from io import StringIO
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.document import Document

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection
    from psycopg2.extensions import cursor as PgCursor


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
        connection_string: str,
        table: str,
        *,
        schema: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        drop: bool = False,
    ) -> None:
        """Initialize with yellowbrick client.
        Args:
            embedding: Embedding operator
            connection_string: Format 'postgres://username:password@host:port/database'
            table: Table used to store / retrieve embeddings from
        """
        from psycopg2 import extras

        extras.register_uuid()

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.ERROR)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if not isinstance(embedding, Embeddings):
            self.logger.error("embeddings input must be Embeddings object.")
            return

        self.LSH_INDEX_TABLE: str = "_lsh_index"
        self.LSH_HYPERPLANE_TABLE: str = "_lsh_hyperplane"
        self.CONTENT_TABLE: str = "_content"

        self.connection_string = connection_string
        self.connection = Yellowbrick.DatabaseConnection(connection_string, self.logger)
        atexit.register(self.connection.close_connection)

        self._schema = schema
        self._table = table
        self._embedding = embedding
        self._max_embedding_len = None
        self._check_database_utf8()

        with self.connection.get_cursor() as cursor:
            if drop:
                self.drop(table=self._table, schema=self._schema, cursor=cursor)
                self.drop(
                    table=self._table + self.CONTENT_TABLE,
                    schema=self._schema,
                    cursor=cursor,
                )
                self._drop_lsh_index_tables(cursor)

            self._create_schema(cursor)
            self._create_table(cursor)

    class DatabaseConnection:
        _instance = None
        _connection_string: str
        _connection: Optional["PgConnection"] = None
        _logger: logging.Logger

        def __new__(
            cls, connection_string: str, logger: logging.Logger
        ) -> "Yellowbrick.DatabaseConnection":
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._connection_string = connection_string
                cls._instance._logger = logger
            return cls._instance

        def close_connection(self) -> None:
            if self._connection and not self._connection.closed:
                self._connection.close()
                self._connection = None

        def get_connection(self) -> "PgConnection":
            import psycopg2

            if not self._connection or self._connection.closed:
                self._connection = psycopg2.connect(self._connection_string)
                self._connection.autocommit = False

            return self._connection

        @contextmanager
        def get_managed_connection(self) -> Generator["PgConnection", None, None]:
            from psycopg2 import DatabaseError

            conn = self.get_connection()
            try:
                yield conn
            except DatabaseError as e:
                conn.rollback()
                self._logger.error(
                    "Database error occurred, rolling back transaction.", exc_info=True
                )
                raise RuntimeError("Database transaction failed.") from e
            else:
                conn.commit()

        @contextmanager
        def get_cursor(self) -> Generator["PgCursor", None, None]:
            with self.get_managed_connection() as conn:
                cursor = conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()

    def _create_schema(self, cursor: "PgCursor") -> None:
        """
        Helper function: create schema if not exists
        """
        from psycopg2 import sql

        if self._schema:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE SCHEMA IF NOT EXISTS {s}
                """
                ).format(
                    s=sql.Identifier(self._schema),
                )
            )

    def _create_table(self, cursor: "PgCursor") -> None:
        """
        Helper function: create table if not exists
        """
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        t = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c = sql.Identifier(self._table + self.CONTENT_TABLE + "_pk_doc_id")
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
                t=t,
                c=c,
            )
        )

        schema_prefix = (self._schema,) if self._schema else ()
        t1 = sql.Identifier(*schema_prefix, self._table)
        t2 = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c1 = sql.Identifier(
            self._table + self.CONTENT_TABLE + "_pk_doc_id_embedding_id"
        )
        c2 = sql.Identifier(self._table + self.CONTENT_TABLE + "_fk_doc_id")
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
                t1=t1,
                t2=t2,
                c1=c1,
                c2=c2,
            )
        )

    def drop(
        self,
        table: str,
        schema: Optional[str] = None,
        cursor: Optional["PgCursor"] = None,
    ) -> None:
        """
        Helper function: Drop data. If a cursor is provided, use it;
        otherwise, obtain a new cursor for the operation.
        """
        if cursor is None:
            with self.connection.get_cursor() as cursor:
                self._drop_table(cursor, table, schema=schema)
        else:
            self._drop_table(cursor, table, schema=schema)

    def _drop_table(
        self,
        cursor: "PgCursor",
        table: str,
        schema: Optional[str] = None,
    ) -> None:
        """
        Executes the drop table command using the given cursor.
        """
        from psycopg2 import sql

        if schema:
            table_name = sql.Identifier(schema, table)
        else:
            table_name = sql.Identifier(table)

        drop_table_query = sql.SQL(
            """
        DROP TABLE IF EXISTS {} CASCADE
        """
        ).format(table_name)
        cursor.execute(drop_table_query)

    def _check_database_utf8(self) -> bool:
        """
        Helper function: Test the database is UTF-8 encoded
        """
        with self.connection.get_cursor() as cursor:
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

        index_params = kwargs.get("index_params") or Yellowbrick.IndexParams()

        with self.connection.get_cursor() as cursor:
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
                    self._copy_to_db(cursor, content_io, embeddings_io)

                    content_io.seek(0)
                    content_io.truncate(0)
                    embeddings_io.seek(0)
                    embeddings_io.truncate(0)
                    current_batch_size = 0

            if current_batch_size > 0:
                self._copy_to_db(cursor, content_io, embeddings_io)

        if index_params.index_type == Yellowbrick.IndexType.LSH:
            self._update_index(index_params, uuid.UUID(doc_uuid))

        return results

    def _copy_to_db(
        self, cursor: "PgCursor", content_io: StringIO, embeddings_io: StringIO
    ) -> None:
        content_io.seek(0)
        embeddings_io.seek(0)

        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        table = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        content_copy_query = sql.SQL(
            """
            COPY {table} (doc_id, text, metadata) FROM 
            STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')
        """
        ).format(table=table)
        cursor.copy_expert(content_copy_query, content_io)

        schema_prefix = (self._schema,) if self._schema else ()
        table = sql.Identifier(*schema_prefix, self._table)
        embeddings_copy_query = sql.SQL(
            """
            COPY {table} (doc_id, embedding_id, embedding) FROM 
            STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')
        """
        ).format(table=table)
        cursor.copy_expert(embeddings_copy_query, embeddings_io)

    @classmethod
    def from_texts(
        cls: Type[Yellowbrick],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_string: str = "",
        table: str = "langchain",
        schema: str = "public",
        drop: bool = False,
        **kwargs: Any,
    ) -> Yellowbrick:
        """Add texts to the vectorstore index.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            connection_string: URI to Yellowbrick instance
            embedding: Embedding function
            table: table to store embeddings
            kwargs: vectorstore specific parameters
        """
        vss = cls(
            embedding=embedding,
            connection_string=connection_string,
            table=table,
            schema=schema,
            drop=drop,
        )
        vss.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return vss

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by uuids.

        Args:
            ids: List of ids to delete, where each id is a uuid string.
        """
        from psycopg2 import sql

        if delete_all:
            where_sql = sql.SQL(
                """
                WHERE 1=1
            """
            )
        elif ids is not None:
            uuids = tuple(sql.Literal(id) for id in ids)
            ids_formatted = sql.SQL(", ").join(uuids)
            where_sql = sql.SQL(
                """
                WHERE doc_id IN ({ids})
            """
            ).format(
                ids=ids_formatted,
            )
        else:
            raise ValueError("Either ids or delete_all must be provided.")

        schema_prefix = (self._schema,) if self._schema else ()
        with self.connection.get_cursor() as cursor:
            table_identifier = sql.Identifier(
                *schema_prefix, self._table + self.CONTENT_TABLE
            )
            query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                table=table_identifier, where_sql=where_sql
            )
            cursor.execute(query)

            table_identifier = sql.Identifier(*schema_prefix, self._table)
            query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                table=table_identifier, where_sql=where_sql
            )
            cursor.execute(query)

            if self._table_exists(
                cursor, self._table + self.LSH_INDEX_TABLE, *schema_prefix
            ):
                table_identifier = sql.Identifier(
                    *schema_prefix, self._table + self.LSH_INDEX_TABLE
                )
                query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                    table=table_identifier, where_sql=where_sql
                )
                cursor.execute(query)

        return None

    def _table_exists(
        self, cursor: "PgCursor", table_name: str, schema: str = "public"
    ) -> bool:
        """
        Checks if a table exists in the given schema
        """
        from psycopg2 import sql

        schema = sql.Literal(schema)
        table_name = sql.Literal(table_name)
        cursor.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM sys.table t INNER JOIN sys.schema s ON t.schema_id = s.schema_id
                WHERE s.name = {schema} AND t.name = {table_name}
            """
            ).format(
                schema=schema,
                table_name=table_name,
            )
        )
        return cursor.fetchone()[0] > 0

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

        index_params = kwargs.get("index_params") or Yellowbrick.IndexParams()

        with self.connection.get_cursor() as cursor:
            tmp_embeddings_table = "tmp_" + self._table
            tmp_doc_id = self._generate_vector_uuid(embedding)
            create_table_query = sql.SQL(
                """ 
                CREATE TEMPORARY TABLE {} (
                doc_id UUID,
                embedding_id SMALLINT,
                embedding FLOAT)
                ON COMMIT DROP
                DISTRIBUTE REPLICATE
            """
            ).format(sql.Identifier(tmp_embeddings_table))
            cursor.execute(create_table_query)
            data_input = [
                (str(tmp_doc_id), embedding_id, embedding_value)
                for embedding_id, embedding_value in enumerate(embedding)
            ]
            insert_query = sql.SQL(
                "INSERT INTO {} (doc_id, embedding_id, embedding) VALUES %s"
            ).format(sql.Identifier(tmp_embeddings_table))
            execute_values(cursor, insert_query, data_input)

            v1 = sql.Identifier(tmp_embeddings_table)
            schema_prefix = (self._schema,) if self._schema else ()
            v2 = sql.Identifier(*schema_prefix, self._table)
            v3 = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
            if index_params.index_type == Yellowbrick.IndexType.LSH:
                tmp_hash_table = self._table + "_tmp_hash"
                self._generate_tmp_lsh_hashes(
                    cursor,
                    tmp_embeddings_table,
                    tmp_hash_table,
                )

                schema_prefix = (self._schema,) if self._schema else ()
                lsh_index = sql.Identifier(
                    *schema_prefix, self._table + self.LSH_INDEX_TABLE
                )
                input_hash_table = sql.Identifier(tmp_hash_table)
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
                    v1=v1,
                    v2=v2,
                    v3=v3,
                    lsh_index=lsh_index,
                    input_hash_table=input_hash_table,
                    hamming_distance=sql.Literal(
                        index_params.get_param("hamming_distance", 0)
                    ),
                )
                cursor.execute(
                    sql_query,
                    (k,),
                )
                results = cursor.fetchall()
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
                    v1=v1,
                    v2=v2,
                    v3=v3,
                )
                cursor.execute(sql_query, (k,))
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

    def _update_lsh_hashes(
        self,
        cursor: "PgCursor",
        doc_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Add hashes to LSH index"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        lsh_hyperplane_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        lsh_index_table_id = sql.Identifier(
            *schema_prefix, self._table + self.LSH_INDEX_TABLE
        )
        embedding_table_id = sql.Identifier(*schema_prefix, self._table)
        query_prefix_id = sql.SQL("INSERT INTO {}").format(lsh_index_table_id)
        condition = (
            sql.SQL("WHERE e.doc_id = {doc_id}").format(doc_id=sql.Literal(str(doc_id)))
            if doc_id
            else sql.SQL("")
        )
        group_by = sql.SQL("GROUP BY 1, 2")

        input_query = sql.SQL(
            """
            {query_prefix}
            SELECT
                e.doc_id as doc_id,
                h.id as hash_index,
                CASE WHEN SUM(e.embedding * h.hyperplane) > 0 THEN 1 ELSE 0 END as hash
            FROM {embedding_table} e
            INNER JOIN {hyperplanes} h ON e.embedding_id = h.hyperplane_id
            {condition}
            {group_by}
        """
        ).format(
            query_prefix=query_prefix_id,
            embedding_table=embedding_table_id,
            hyperplanes=lsh_hyperplane_table,
            condition=condition,
            group_by=group_by,
        )
        cursor.execute(input_query)

    def _generate_tmp_lsh_hashes(
        self, cursor: "PgCursor", tmp_embedding_table: str, tmp_hash_table: str
    ) -> None:
        """Generate temp LSH"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        lsh_hyperplane_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        tmp_embedding_table_id = sql.Identifier(tmp_embedding_table)
        tmp_hash_table_id = sql.Identifier(tmp_hash_table)
        query_prefix = sql.SQL("CREATE TEMPORARY TABLE {} ON COMMIT DROP AS").format(
            tmp_hash_table_id
        )
        group_by = sql.SQL("GROUP BY 1")

        input_query = sql.SQL(
            """
            {query_prefix}
            SELECT
                h.id as hash_index,
                CASE WHEN SUM(e.embedding * h.hyperplane) > 0 THEN 1 ELSE 0 END as hash
            FROM {embedding_table} e
            INNER JOIN {hyperplanes} h ON e.embedding_id = h.hyperplane_id
            {group_by}
            DISTRIBUTE REPLICATE
        """
        ).format(
            query_prefix=query_prefix,
            embedding_table=tmp_embedding_table_id,
            hyperplanes=lsh_hyperplane_table,
            group_by=group_by,
        )
        cursor.execute(input_query)

    def _populate_hyperplanes(self, cursor: "PgCursor", num_hyperplanes: int) -> None:
        """Generate random hyperplanes and store in Yellowbrick"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        hyperplanes_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {t}").format(t=hyperplanes_table))
        if cursor.fetchone()[0] > 0:
            return

        t = sql.Identifier(*schema_prefix, self._table)
        cursor.execute(sql.SQL("SELECT MAX(embedding_id) FROM {t}").format(t=t))
        num_dimensions = cursor.fetchone()[0]
        num_dimensions += 1

        insert_query = sql.SQL(
            """
            WITH parameters AS (
                SELECT {num_hyperplanes} AS num_hyperplanes,
                    {dims_per_hyperplane} AS dims_per_hyperplane
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
            hyperplanes_table=hyperplanes_table,
        )
        cursor.execute(insert_query)

    def _create_lsh_index_tables(self, cursor: "PgCursor") -> None:
        """Create LSH index and hyperplane tables"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        t1 = sql.Identifier(*schema_prefix, self._table + self.LSH_INDEX_TABLE)
        t2 = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c1 = sql.Identifier(self._table + self.LSH_INDEX_TABLE + "_pk_doc_id")
        c2 = sql.Identifier(self._table + self.LSH_INDEX_TABLE + "_fk_doc_id")
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
                t1=t1,
                t2=t2,
                c1=c1,
                c2=c2,
            )
        )

        schema_prefix = (self._schema,) if self._schema else ()
        t = sql.Identifier(*schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE)
        c = sql.Identifier(self._table + self.LSH_HYPERPLANE_TABLE + "_pk_id_hp_id")
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
                t=t,
                c=c,
            )
        )

    def _drop_lsh_index_tables(self, cursor: "PgCursor") -> None:
        """Drop LSH index tables"""
        self.drop(
            schema=self._schema, table=self._table + self.LSH_INDEX_TABLE, cursor=cursor
        )
        self.drop(
            schema=self._schema,
            table=self._table + self.LSH_HYPERPLANE_TABLE,
            cursor=cursor,
        )

    def create_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Create index from existing vectors"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._drop_lsh_index_tables(cursor)
                self._create_lsh_index_tables(cursor)
                self._populate_hyperplanes(
                    cursor, index_params.get_param("num_hyperplanes", 128)
                )
                self._update_lsh_hashes(cursor)

    def drop_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Drop an index"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._drop_lsh_index_tables(cursor)

    def _update_index(
        self, index_params: Yellowbrick.IndexParams, doc_id: uuid.UUID
    ) -> None:
        """Update an index with a new or modified embedding in the embeddings table"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._update_lsh_hashes(cursor, doc_id)

    def migrate_schema_v1_to_v2(self) -> None:
        from psycopg2 import sql

        try:
            with self.connection.get_cursor() as cursor:
                schema_prefix = (self._schema,) if self._schema else ()
                embeddings = sql.Identifier(*schema_prefix, self._table)
                old_embeddings = sql.Identifier(*schema_prefix, self._table + "_v1")
                content = sql.Identifier(
                    *schema_prefix, self._table + self.CONTENT_TABLE
                )
                alter_table_query = sql.SQL("ALTER TABLE {t1} RENAME TO {t2}").format(
                    t1=embeddings,
                    t2=old_embeddings,
                )
                cursor.execute(alter_table_query)

                self._create_table(cursor)

                insert_query = sql.SQL(
                    """
                    INSERT INTO {t1} (doc_id, embedding_id, embedding) 
                    SELECT id, embedding_id, embedding FROM {t2}
                """
                ).format(
                    t1=embeddings,
                    t2=old_embeddings,
                )
                cursor.execute(insert_query)

                insert_content_query = sql.SQL(
                    """
                    INSERT INTO {t1} (doc_id, text, metadata) 
                    SELECT DISTINCT id, text, metadata FROM {t2}
                """
                ).format(t1=content, t2=old_embeddings)
                cursor.execute(insert_content_query)
        except Exception as e:
            raise RuntimeError(f"Failed to migrate schema: {e}") from e
