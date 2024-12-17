import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_DISTANCE_STRATEGY = "cosine"  # or "l2"
DEFAULT_TiDB_VECTOR_TABLE_NAME = "langchain_vector"


class TiDBVectorStore(VectorStore):
    """TiDB Vector Store."""

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        table_name: str = DEFAULT_TiDB_VECTOR_TABLE_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        drop_existing_table: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a TiDB Vector Store in Langchain with a flexible
        and standardized table structure for storing vector data
        which remains fixed regardless of the dynamic table name setting.

        The vector table schema includes:
        - 'id': a UUID for each entry.
        - 'embedding': stores vector data in a VectorType column.
        - 'document': a Text column for the original data or additional information.
        - 'meta': a JSON column for flexible metadata storage.
        - 'create_time' and 'update_time': timestamp columns for tracking data changes.

        This table structure caters to general use cases and
        complex scenarios where the table serves as a semantic layer for advanced
        data integration and analysis, leveraging SQL for join queries.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            embedding_function: The embedding function used to generate embeddings.
            table_name (str, optional): The name of the table that will be used to
                store vector data. If you do not provide a table name,
                a default table named `langchain_vector` will be created automatically.
            distance_strategy: The strategy used for similarity search,
                defaults to "cosine", valid values: "l2", "cosine".
            engine_args (Optional[Dict]): Additional arguments for the database engine,
                defaults to None.
            drop_existing_table: Drop the existing TiDB table before initializing,
                defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Examples:
            .. code-block:: python

            from langchain_community.vectorstores import TiDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            embeddingFunc = OpenAIEmbeddings()
            CONNECTION_STRING = "mysql+pymysql://root@34.212.137.91:4000/test"

            vs = TiDBVector.from_texts(
                embedding=embeddingFunc,
                texts = [..., ...],
                connection_string=CONNECTION_STRING,
                distance_strategy="l2",
                table_name="tidb_vector_langchain",
            )

            query = "What did the president say about Ketanji Brown Jackson"
            docs = db.similarity_search_with_score(query)

        """

        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._embedding_function = embedding_function
        self._distance_strategy = distance_strategy
        self._vector_dimension = self._get_dimension()

        try:
            from tidb_vector.integrations import TiDBVectorClient
        except ImportError:
            raise ImportError(
                "Could not import tidbvec python package. "
                "Please install it with `pip install tidb-vector`."
            )

        self._tidb = TiDBVectorClient(
            connection_string=connection_string,
            table_name=table_name,
            distance_strategy=distance_strategy,
            vector_dimension=self._vector_dimension,
            engine_args=engine_args,
            drop_existing_table=drop_existing_table,
            **kwargs,
        )

    @property
    def embeddings(self) -> Embeddings:
        """Return the function used to generate embeddings."""
        return self._embedding_function

    @property
    def tidb_vector_client(self) -> Any:
        """Return the TiDB Vector Client."""
        return self._tidb

    @property
    def distance_strategy(self) -> Any:
        """
        Returns the current distance strategy.
        """
        return self._distance_strategy

    def _get_dimension(self) -> int:
        """
        Get the dimension of the vector using embedding functions.
        """
        return len(self._embedding_function.embed_query("test embedding length"))

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "TiDBVectorStore":
        """
        Create a VectorStore from a list of texts.

        Args:
            texts (List[str]): The list of texts to be added to the TiDB Vector.
            embedding (Embeddings): The function to use for generating embeddings.
            metadatas: The list of metadata dictionaries corresponding to each text,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
                connection_string (str): The connection string for the TiDB database,
                    format: "mysql+pymysql://root@34.212.137.91:4000/test".
                table_name (str, optional): The name of table used to store vector data,
                    defaults to "langchain_vector".
                distance_strategy: The distance strategy used for similarity search,
                    defaults to "cosine", allowed: "l2", "cosine".
                ids (Optional[List[str]]): The list of IDs corresponding to each text,
                    defaults to None.
                engine_args: Additional arguments for the underlying database engine,
                    defaults to None.
                drop_existing_table: Drop the existing TiDB table before initializing,
                    defaults to False.

        Returns:
            VectorStore: The created TiDB Vector Store.

        """

        # Extract arguments from kwargs with default values
        connection_string = kwargs.pop("connection_string", None)
        if connection_string is None:
            raise ValueError("please provide your tidb connection_url")
        table_name = kwargs.pop("table_name", "langchain_vector")
        distance_strategy = kwargs.pop("distance_strategy", "cosine")
        ids = kwargs.pop("ids", None)
        engine_args = kwargs.pop("engine_args", None)
        drop_existing_table = kwargs.pop("drop_existing_table", False)

        embeddings = embedding.embed_documents(list(texts))

        vs = cls(
            connection_string=connection_string,
            table_name=table_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            drop_existing_table=drop_existing_table,
            **kwargs,
        )

        vs._tidb.insert(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return vs

    @classmethod
    def from_existing_vector_table(
        cls,
        embedding: Embeddings,
        connection_string: str,
        table_name: str,
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
            table_name (str, optional): The name of table used to store vector data,
                defaults to "langchain_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed: "l2", "cosine".
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            **kwargs (Any): Additional keyword arguments.
        Returns:
            VectorStore: The VectorStore instance.

        Raises:
            NoSuchTableError: If the specified table does not exist in the TiDB.
        """

        try:
            from tidb_vector.integrations import check_table_existence
        except ImportError:
            raise ImportError(
                "Could not import tidbvec python package. "
                "Please install it with `pip install tidb-vector`."
            )

        if check_table_existence(connection_string, table_name):
            return cls(
                connection_string=connection_string,
                table_name=table_name,
                embedding_function=embedding,
                distance_strategy=distance_strategy,
                engine_args=engine_args,
                **kwargs,
            )
        else:
            raise ValueError(f"Table {table_name} does not exist in the TiDB database.")

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
            ids = [str(uuid.uuid4()) for _ in texts]
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
            kwargs: Additional keyword arguments.
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
            kwargs: Additional keyword arguments.

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
            kwargs: Additional keyword arguments.

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
