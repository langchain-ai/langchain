from __future__ import annotations

import json
import re
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy.pool import QueuePool

from langchain_community.vectorstores.utils import DistanceStrategy

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.DOT_PRODUCT

ORDERING_DIRECTIVE: dict = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "",
    DistanceStrategy.DOT_PRODUCT: "DESC",
}


class SingleStoreDB(VectorStore):
    """`SingleStore DB` vector store.

    The prerequisite for using this class is the installation of the ``singlestoredb``
    Python package.

    The SingleStoreDB vectorstore can be created by providing an embedding function and
    the relevant parameters for the database connection, connection pool, and
    optionally, the names of the table and the fields to use.
    """

    def _get_connection(self: SingleStoreDB) -> Any:
        try:
            import singlestoredb as s2
        except ImportError:
            raise ImportError(
                "Could not import singlestoredb python package. "
                "Please install it with `pip install singlestoredb`."
            )
        return s2.connect(**self.connection_kwargs)

    def __init__(
        self,
        embedding: Embeddings,
        *,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        table_name: str = "embeddings",
        content_field: str = "content",
        metadata_field: str = "metadata",
        vector_field: str = "vector",
        use_vector_index: bool = False,
        vector_index_name: str = "",
        vector_index_options: Optional[dict] = None,
        vector_size: int = 1536,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:
            embedding (Embeddings): A text embedding model.

            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to DOT_PRODUCT.
                Available options are:
                - DOT_PRODUCT: Computes the scalar product of two vectors.
                    This is the default behavior
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships.

            table_name (str, optional): Specifies the name of the table in use.
                Defaults to "embeddings".
            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".
            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".
            vector_field (str, optional): Specifies the field to store the vector.
                Defaults to "vector".

            use_vector_index (bool, optional): Toggles the use of a vector index.
                Works only with SingleStoreDB 8.5 or later. Defaults to False.
                If set to True, vector_size parameter is required to be set to
                a proper value.

            vector_index_name (str, optional): Specifies the name of the vector index.
                Defaults to empty. Will be ignored if use_vector_index is set to False.

            vector_index_options (dict, optional): Specifies the options for
                the vector index. Defaults to {}.
                Will be ignored if use_vector_index is set to False. The options are:
                index_type (str, optional): Specifies the type of the index.
                    Defaults to IVF_PQFS.
                For more options, please refer to the SingleStoreDB documentation:
                https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/

            vector_size (int, optional): Specifies the size of the vector.
                Defaults to 1536. Required if use_vector_index is set to True.
                Should be set to the same value as the size of the vectors
                stored in the vector_field.

            Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.
            local_infile (bool, optional): Allows local file uploads.
            charset (str, optional): Specifies the character set for string values.
            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.
            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.
            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.
            ssl_cipher (str, optional): Sets the SSL cipher list.
            ssl_disabled (bool, optional): Disables SSL usage.
            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.
            ssl_verify_identity (bool, optional): Verifies the server's identity.
            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.
            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.
            autocommit (bool, optional): Enables autocommits.
            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.
            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.

        Examples:
            Basic Usage:

            .. code-block:: python

                from langchain_community.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import SingleStoreDB

                vectorstore = SingleStoreDB(
                    OpenAIEmbeddings(),
                    host="https://user:password@127.0.0.1:3306/database"
                )

            Advanced Usage:

            .. code-block:: python

                from langchain_community.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import SingleStoreDB

                vectorstore = SingleStoreDB(
                    OpenAIEmbeddings(),
                    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
                    host="127.0.0.1",
                    port=3306,
                    user="user",
                    password="password",
                    database="db",
                    table_name="my_custom_table",
                    pool_size=10,
                    timeout=60,
                )

            Using environment variables:

            .. code-block:: python

                from langchain_community.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import SingleStoreDB

                os.environ['SINGLESTOREDB_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                vectorstore = SingleStoreDB(OpenAIEmbeddings())

            Using vector index:

            .. code-block:: python

                from langchain_community.embeddings import OpenAIEmbeddings
                from langchain_community.vectorstores import SingleStoreDB

                os.environ['SINGLESTOREDB_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                vectorstore = SingleStoreDB(
                    OpenAIEmbeddings(),
                    use_vector_index=True,
                )
        """

        self.embedding = embedding
        self.distance_strategy = distance_strategy
        self.table_name = self._sanitize_input(table_name)
        self.content_field = self._sanitize_input(content_field)
        self.metadata_field = self._sanitize_input(metadata_field)
        self.vector_field = self._sanitize_input(vector_field)

        self.use_vector_index = bool(use_vector_index)
        self.vector_index_name = self._sanitize_input(vector_index_name)
        self.vector_index_options = dict(vector_index_options or {})
        self.vector_index_options["metric_type"] = self.distance_strategy
        self.vector_size = int(vector_size)

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

        # Add program name and version to connection attributes.
        if "conn_attrs" not in self.connection_kwargs:
            self.connection_kwargs["conn_attrs"] = dict()

        self.connection_kwargs["conn_attrs"]["_connector_name"] = "langchain python sdk"
        self.connection_kwargs["conn_attrs"]["_connector_version"] = "1.0.2"

        # Create connection pool.
        self.connection_pool = QueuePool(
            self._get_connection,
            max_overflow=max_overflow,
            pool_size=pool_size,
            timeout=timeout,
        )
        self._create_table()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._max_inner_product_relevance_score_fn

    def _create_table(self: SingleStoreDB) -> None:
        """Create table if it doesn't exist."""
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                if self.use_vector_index:
                    index_options = ""
                    if self.vector_index_options and len(self.vector_index_options) > 0:
                        index_options = "INDEX_OPTIONS '{}'".format(
                            json.dumps(self.vector_index_options)
                        )
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        ({} TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
                        {} VECTOR({}, F32) NOT NULL, {} JSON,
                        VECTOR INDEX {} ({}) {});""".format(
                            self.table_name,
                            self.content_field,
                            self.vector_field,
                            self.vector_size,
                            self.metadata_field,
                            self.vector_index_name,
                            self.vector_field,
                            index_options,
                        ),
                    )
                else:
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        ({} TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
                        {} BLOB, {} JSON);""".format(
                            self.table_name,
                            self.content_field,
                            self.vector_field,
                            self.metadata_field,
                        ),
                    )
            finally:
                cur.close()
        finally:
            conn.close()

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run images through the embeddings and add to the vectorstore.

        Args:
            uris List[str]: File path to images.
                Each URI will be added to the vectorstore as document content.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: empty list
        """
        # Set embeddings
        if (
            embeddings is None
            and self.embedding is not None
            and hasattr(self.embedding, "embed_image")
        ):
            embeddings = self.embedding.embed_image(uris=uris)
        return self.add_texts(uris, metadatas, embeddings, **kwargs)

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
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
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
                    cur.execute(
                        "INSERT INTO {} VALUES (%s, JSON_ARRAY_PACK(%s), %s)".format(
                            self.table_name
                        ),
                        (
                            text,
                            "[{}]".format(",".join(map(str, embedding))),
                            json.dumps(metadata),
                        ),
                    )
                if self.use_vector_index:
                    cur.execute("OPTIMIZE TABLE {} FLUSH;".format(self.table_name))
            finally:
                cur.close()
        finally:
            conn.close()
        return []

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Uses cosine similarity.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (dict): A dictionary of metadata fields and values to filter by.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        Examples:
            .. code-block:: python
                from langchain_community.vectorstores import SingleStoreDB
                from langchain_community.embeddings import OpenAIEmbeddings
                s2 = SingleStoreDB.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database"
                )
                s2.similarity_search("query text", 1,
                    {"metadata_field": "metadata_value"})
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query. Uses cosine similarity.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        # Creates embedding vector from user query
        embedding = self.embedding.embed_query(query)
        conn = self.connection_pool.connect()
        result = []
        where_clause: str = ""
        where_clause_values: List[Any] = []
        if filter:
            where_clause = "WHERE "
            arguments = []

            def build_where_clause(
                where_clause_values: List[Any],
                sub_filter: dict,
                prefix_args: Optional[List[str]] = None,
            ) -> None:
                prefix_args = prefix_args or []
                for key in sub_filter.keys():
                    if isinstance(sub_filter[key], dict):
                        build_where_clause(
                            where_clause_values, sub_filter[key], prefix_args + [key]
                        )
                    else:
                        arguments.append(
                            "JSON_EXTRACT_JSON({}, {}) = %s".format(
                                self.metadata_field,
                                ", ".join(["%s"] * (len(prefix_args) + 1)),
                            )
                        )
                        where_clause_values += prefix_args + [key]
                        where_clause_values.append(json.dumps(sub_filter[key]))

            build_where_clause(where_clause_values, filter)
            where_clause += " AND ".join(arguments)

        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """SELECT {}, {}, {}({}, JSON_ARRAY_PACK(%s)) as __score
                    FROM {} {} ORDER BY __score {} LIMIT %s""".format(
                        self.content_field,
                        self.metadata_field,
                        self.distance_strategy.name
                        if isinstance(self.distance_strategy, DistanceStrategy)
                        else self.distance_strategy,
                        self.vector_field,
                        self.table_name,
                        where_clause,
                        ORDERING_DIRECTIVE[self.distance_strategy],
                    ),
                    ("[{}]".format(",".join(map(str, embedding))),)
                    + tuple(where_clause_values)
                    + (k,),
                )

                for row in cur.fetchall():
                    doc = Document(page_content=row[0], metadata=row[1])
                    result.append((doc, float(row[2])))
            finally:
                cur.close()
        finally:
            conn.close()
        return result

    @classmethod
    def from_texts(
        cls: Type[SingleStoreDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        table_name: str = "embeddings",
        content_field: str = "content",
        metadata_field: str = "metadata",
        vector_field: str = "vector",
        use_vector_index: bool = False,
        vector_index_name: str = "",
        vector_index_options: Optional[dict] = None,
        vector_size: int = 1536,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ) -> SingleStoreDB:
        """Create a SingleStoreDB vectorstore from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new table for the embeddings in SingleStoreDB.
            3. Adds the documents to the newly created table.
        This is intended to be a quick way to get started.
        Example:
            .. code-block:: python
                from langchain_community.vectorstores import SingleStoreDB
                from langchain_community.embeddings import OpenAIEmbeddings
                s2 = SingleStoreDB.from_texts(
                    texts,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database"
                )
        """

        instance = cls(
            embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_field=content_field,
            metadata_field=metadata_field,
            vector_field=vector_field,
            pool_size=pool_size,
            max_overflow=max_overflow,
            timeout=timeout,
            use_vector_index=use_vector_index,
            vector_index_name=vector_index_name,
            vector_index_options=vector_index_options,
            vector_size=vector_size,
            **kwargs,
        )
        instance.add_texts(texts, metadatas, embedding.embed_documents(texts), **kwargs)
        return instance


# SingleStoreDBRetriever is not needed, but we keep it for backwards compatibility
SingleStoreDBRetriever = VectorStoreRetriever
