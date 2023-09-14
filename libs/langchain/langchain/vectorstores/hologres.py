from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_TABLE_NAME = "langchain_pg_embedding"


class HologresWrapper:
    """`Hologres API` wrapper."""

    def __init__(self, connection_string: str, ndims: int, table_name: str) -> None:
        """Initialize the wrapper.

        Args:
            connection_string: Hologres connection string.
            ndims: Number of dimensions of the embedding output.
            table_name: Name of the table to store embeddings and data.
        """

        import psycopg2

        self.table_name = table_name
        self.conn = psycopg2.connect(connection_string)
        self.cursor = self.conn.cursor()
        self.conn.autocommit = False
        self.ndims = ndims

    def create_vector_extension(self) -> None:
        self.cursor.execute("create extension if not exists proxima")
        self.conn.commit()

    def create_table(self, drop_if_exist: bool = True) -> None:
        if drop_if_exist:
            self.cursor.execute(f"drop table if exists {self.table_name}")
        self.conn.commit()

        self.cursor.execute(
            f"""create table if not exists {self.table_name} (
id text,
embedding float4[] check(array_ndims(embedding) = 1 and \
array_length(embedding, 1) = {self.ndims}),
metadata json,
document text);"""
        )
        self.cursor.execute(
            f"call set_table_property('{self.table_name}'"
            + """, 'proxima_vectors', 
'{"embedding":{"algorithm":"Graph",
"distance_method":"SquaredEuclidean",
"build_params":{"min_flush_proxima_row_count" : 1,
"min_compaction_proxima_row_count" : 1, 
"max_total_size_to_merge_mb" : 2000}}}');"""
        )
        self.conn.commit()

    def get_by_id(self, id: str) -> List[Tuple]:
        statement = (
            f"select id, embedding, metadata, "
            f"document from {self.table_name} where id = %s;"
        )
        self.cursor.execute(
            statement,
            (id),
        )
        self.conn.commit()
        return self.cursor.fetchall()

    def insert(
        self,
        embedding: List[float],
        metadata: dict,
        document: str,
        id: Optional[str] = None,
    ) -> None:
        self.cursor.execute(
            f'insert into "{self.table_name}" '
            f"values (%s, array{json.dumps(embedding)}::float4[], %s, %s)",
            (id if id is not None else "null", json.dumps(metadata), document),
        )
        self.conn.commit()

    def query_nearest_neighbours(
        self, embedding: List[float], k: int, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, str, float]]:
        params = []
        filter_clause = ""
        if filter is not None:
            conjuncts = []
            for key, val in filter.items():
                conjuncts.append("metadata->>%s=%s")
                params.append(key)
                params.append(val)
            filter_clause = "where " + " and ".join(conjuncts)

        sql = (
            f"select document, metadata::text, "
            f"pm_approx_squared_euclidean_distance(array{json.dumps(embedding)}"
            f"::float4[], embedding) as distance from"
            f" {self.table_name} {filter_clause} order by distance asc limit {k};"
        )
        self.cursor.execute(sql, tuple(params))
        self.conn.commit()
        return self.cursor.fetchall()


class Hologres(VectorStore):
    """`Hologres API` vector store.

    - `connection_string` is a hologres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `ndims` is the number of dimensions of the embedding output.
    - `table_name` is the name of the table to store embeddings and data.
        (default: langchain_pg_embedding)
        - NOTE: The table will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `pre_delete_table` if True, will delete the table if it exists.
        (default: False)
        - Useful for testing.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.connection_string = connection_string
        self.ndims = ndims
        self.table_name = table_name
        self.embedding_function = embedding_function
        self.pre_delete_table = pre_delete_table
        self.logger = logger or logging.getLogger(__name__)
        self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """
        self.storage = HologresWrapper(
            self.connection_string, self.ndims, self.table_name
        )
        self.create_vector_extension()
        self.create_table()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def create_vector_extension(self) -> None:
        try:
            self.storage.create_vector_extension()
        except Exception as e:
            self.logger.exception(e)
            raise e

    def create_table(self) -> None:
        self.storage.create_table(self.pre_delete_table)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding_function: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding_function,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: List[str],
        **kwargs: Any,
    ) -> None:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        try:
            for text, metadata, embedding, id in zip(texts, metadatas, embeddings, ids):
                self.storage.insert(embedding, metadata, text, id)
        except Exception as e:
            self.logger.exception(e)
            self.storage.conn.commit()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        self.add_embeddings(texts, embeddings, metadatas, ids, **kwargs)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Hologres with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results: List[Tuple[str, str, float]] = self.storage.query_nearest_neighbours(
            embedding, k, filter
        )

        docs = [
            (
                Document(
                    page_content=result[0],
                    metadata=json.loads(result[1]),
                ),
                result[2],
            )
            for result in results
        ]
        return docs

    @classmethod
    def from_texts(
        cls: Type[Hologres],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """Construct Hologres wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from langchain import Hologres
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = Hologres.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            pre_delete_table=pre_delete_table,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Hologres],
        embedding: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        pre_delete_table: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Get intsance of an existing Hologres store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            ndims=ndims,
            table_name=table_name,
            embedding_function=embedding,
            pre_delete_table=pre_delete_table,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="HOLOGRES_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the HOLOGRES_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[Hologres],
        documents: List[Document],
        embedding: Embeddings,
        ndims: int = ADA_TOKEN_COUNT,
        table_name: str = _LANGCHAIN_DEFAULT_TABLE_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Hologres:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            ndims=ndims,
            table_name=table_name,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return (
            f"dbname={database} user={user} password={password} host={host} port={port}"
        )
