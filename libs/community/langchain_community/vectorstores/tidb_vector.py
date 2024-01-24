import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_DISTANCE_STRATEGY = "cosine"  # or "l2"
DEFAULT_TiDB_VECTOR_STORE_NAME = "langchain_tidb_vector"


class TiDBVector(VectorStore):
    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        tidb_vectorstore: Optional[Any] = None,
        vectorstore_name: str = DEFAULT_TiDB_VECTOR_STORE_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        drop_existing_vectorstore: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a TiDB Vector Store.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            embedding_function: The embedding function used to generate embeddings.
            tidb_vectorstore (Optional): The existing TiDB Vector Store to use.
            vectorstore_name (str, optional): The name of the TiDB Vector Store,
                defaults to "langchain_tidb_vector".
            distance_strategy: The strategy used for similarity search,
                defaults to "cosine", valid values: "l2", "cosine".
            engine_args (Optional[Dict]): Additional arguments for the database engine,
                defaults to None.
            drop_existing_vectorstore: Drop the existing TiDB Vector Store
                before initializing, defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Examples:
            .. code-block:: python

            from langchain_community.vectorstores.tidb_vector import TiDBVector
            from langchain_openai import OpenAIEmbeddings

            embeddingFunc = OpenAIEmbeddings()
            CONNECTION_STRING = "mysql+pymysql://root@34.212.137.91:4000/test"

            vs = TiDBVector.from_texts(
                embedding=embeddingFunc,
                texts = [..., ...],
                connection_string=CONNECTION_STRING,
                distance_strategy="l2",
                vectorstore_name="tidb_vector_langchain",
            )

            query = "What did the president say about Ketanji Brown Jackson"
            docs = db.similarity_search_with_score(query)

        """

        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._embedding_function = embedding_function
        self._distance_strategy = distance_strategy

        try:
            from tidb_vector.integrations import VectorStore as TiDBVectorStore
        except ImportError:
            raise ImportError(
                "Could not import tidbvec python package. "
                "Please install it with `pip install tidbvec`."
            )

        self._tidb = tidb_vectorstore or TiDBVectorStore(
            connection_string=connection_string,
            table_name=vectorstore_name,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            drop_existing_table=drop_existing_vectorstore,
            **kwargs,
        )

    @property
    def embeddings(self) -> Embeddings:
        """Return the function used to generate embeddings."""
        return self._embedding_function

    @property
    def distance_strategy(self) -> Any:
        """
        Returns the current distance strategy.
        """
        return self._distance_strategy

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        connection_string: str,
        metadatas: Optional[List[dict]] = None,
        vectorstore_name: str = DEFAULT_TiDB_VECTOR_STORE_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        drop_existing_vectorstore: bool = False,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a VectorStore from a list of texts.

        Args:
            texts (List[str]): The list of texts to be added to the TiDB Vector.
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            metadatas: The list of metadata dictionaries corresponding to each text,
                defaults to None.
            vectorstore_name (str, optional): The name of the TiDB Vector Store,
                defaults to "langchain_tidb_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed strategies: "l2", "cosine".
            ids (Optional[List[str]]): The list of IDs corresponding to each text,
                defaults to None.
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            drop_existing_vectorstore: Drop the existing TiDB Vector Store
                before creating a new one, defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            VectorStore: The created TiDB Vector Store.

        """
        embeddings = embedding.embed_documents(list(texts))

        vs = cls(
            connection_string=connection_string,
            vectorstore_name=vectorstore_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            drop_existing_vectorstore=drop_existing_vectorstore,
            **kwargs,
        )

        vs._tidb.insert(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return vs

    @classmethod
    def from_existing_vectorstore(
        cls,
        embedding: Embeddings,
        connection_string: str,
        vectorstore_name: str,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a VectorStore instance from an existing TiDB Vector Store in TiDB.

        Args:
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            vectorstore_name (str): The name of the TiDB Vector Store,
                defaults to "langchain_tidb_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed strategies: "l2", "cosine".
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            VectorStore: The VectorStore instance.

        Raises:
            NoSuchTableError: If the specified table does not exist in the TiDB.
        """

        try:
            from tidb_vector.integrations import VectorStore as TiDBVectorStore
        except ImportError:
            raise ImportError(
                "Could not import tidbvec python package. "
                "Please install it with `pip install tidbvec`."
            )

        tidb_vectorstore = TiDBVectorStore.get_vectorstore(
            connection_string=connection_string,
            table_name=vectorstore_name,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            **kwargs,
        )

        return cls(
            connection_string=connection_string,
            tidb_vectorstore=tidb_vectorstore,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
        )

    def drop_vectorstore(self) -> None:
        """
        Drop the Vector Store from the TiDB database.
        """
        self._tidb.drop_table()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to TiDB Vector Store.

        Args:
            texts (Iterable[str]): The texts to be added.
            metadatas (Optional[List[dict]]): The metadata associated with each text,
                Defaults to None.
            ids (Optional[List[str]]): The IDs to be assigned to each text,
                Defaults to None, will be generated if not provided.

        Returns:
            List[str]: The IDs assigned to the added texts.
        """

        embeddings = self._embedding_function.embed_documents(list(texts))
        if ids is None:
            ids = [uuid.uuid4() for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]

        return self._tidb.insert(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete vector data from the TiDB Vector Store.

        Args:
            ids (Optional[List[str]]): A list of vector IDs to delete.
            **kwargs: Additional keyword arguments.
        """

        self._tidb.delete(ids=ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform a similarity search using the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to retrieve. Defaults to 4.
            filter (dict, optional): A filter to apply to the search results.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of Document objects representing the search results.
        """
        result = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in result]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with score based on the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of results to return. Defaults to 5.
            filter (dict, optional): A filter to apply to the search results.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of tuples containing relevant documents and their similarity scores.
        """
        query_vector = self._embedding_function.embed_query(query)
        relevant_docs = self._tidb.query(
            query_vector=query_vector, k=k, filter=filter, **kwargs
        )
        return [
            (
                Document(
                    page_content=doc.document,
                    metadata=doc.metadata,
                ),
                doc.distance,
            )
            for doc in relevant_docs
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select the relevance score function based on the distance strategy.
        """
        if self._distance_strategy == "cosine":
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == "l2":
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to PGVector constructor."
            )
