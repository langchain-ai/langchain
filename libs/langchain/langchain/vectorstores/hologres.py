from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain.utils import get_from_dict_or_env

ADA_TOKEN_COUNT = 1536
_LANGCHAIN_DEFAULT_TABLE_NAME = "langchain_pg_embedding"


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
        from hologres_vector import HologresVector

        self.storage = HologresVector(
            self.connection_string,
            ndims=self.ndims,
            table_name=self.table_name,
            table_schema={"document": "text"},
            pre_delete_table=self.pre_delete_table,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

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
            schema_datas = [{"document": t} for t in texts]
            self.storage.upsert_vectors(embeddings, ids, metadatas, schema_datas)
        except Exception as e:
            self.logger.exception(e)

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
        results: List[dict[str, Any]] = self.storage.search(
            embedding, k=k, select_columns=["document"], metadata_filters=filter
        )

        docs = [
            (
                Document(
                    page_content=result["document"],
                    metadata=result["metadata"],
                ),
                result["distance"],
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
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        HologresVector.connection_string_from_db_params
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
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        HologresVector.connection_string_from_db_params

        Example:
            .. code-block:: python

                from langchain.vectorstores import Hologres
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
        Get instance of an existing Hologres store.This method will
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
                "Hologres connection string is required"
                "Either pass it as a parameter"
                "or set the HOLOGRES_CONNECTION_STRING environment variable."
                "Create the connection string by calling"
                "HologresVector.connection_string_from_db_params"
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
        Hologres connection string is required
        "Either pass it as a parameter
        or set the HOLOGRES_CONNECTION_STRING environment variable.
        Create the connection string by calling
        HologresVector.connection_string_from_db_params
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
