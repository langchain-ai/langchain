import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_DISTANCE_STRATEGY = "cosine"  # or "l2"
DEFAULT_COLLECTION_NAME = "langchain_tidb_vector"


def get_or_create_collection(
    connection_string: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    pre_delete_collection: bool = False,
    **kwargs: Any,
):
    try:
        from tidb_vector.integrations.vectorstore import TiDBCollection
    except ImportError:
        raise ImportError(
            "Could not import tidbvec python package. "
            "Please install it with `pip install tidbvec`."
        )
    return TiDBCollection.get_collection(
        connection_string=connection_string,
        collection_name=collection_name,
        distance_strategy=distance_strategy,
        engine_args=engine_args,
        pre_delete_collection=pre_delete_collection,
        **kwargs,
    )


class TiDBVector(VectorStore):
    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        collection: Optional[Any] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a TiDBVector.

        Args:
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            embedding_function: The embedding function used to generate embeddings.
            collection (Optional[TiDBVectorStore]): The existing collection to use.
            collection_name (str): The name of the collection,
                defaults to "langchain_tidb_vector".
            distance_strategy: The strategy used for similarity search,
                defaults to "cosine", valid values: "l2", "cosine".
            engine_args (Optional[Dict]): Additional arguments for the database engine,
                defaults to None.
            pre_delete_collection: Delete the collection before creating a new one,
                defaults to False.
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
                collection_name="tidb_vector_langchain",
            )

            query = "What did the president say about Ketanji Brown Jackson"
            docs = db.similarity_search_with_score(query)

        """

        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._embedding_function = embedding_function
        self._distance_strategy = distance_strategy
        self._tidb = collection or get_or_create_collection(
            connection_string=connection_string,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            pre_delete_collection=pre_delete_collection,
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
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        pre_delete_collection: bool = False,
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
            collection_name (str, optional): The name of the collection,
                defaults to "langchain_tidb_vector".
            distance_strategy: The distance strategy used for similarity search,
                defaults to "cosine", allowed strategies: "l2", "cosine".
            ids (Optional[List[str]]): The list of IDs corresponding to each text,
                defaults to None.
            engine_args: Additional arguments for the underlying database engine,
                defaults to None.
            pre_delete_collection: Delete existing collection before adding the texts,
                defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            VectorStore: The created VectorStore.

        """
        embeddings = embedding.embed_documents(list(texts))

        vs = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        vs._tidb.insert(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return vs

    @classmethod
    def from_existing_collection(
        cls,
        embedding: Embeddings,
        connection_string: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_strategy: str = DEFAULT_DISTANCE_STRATEGY,
        *,
        engine_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a VectorStore instance from an existing collection in the TiDB database.

        Args:
            embedding (Embeddings): The function to use for generating embeddings.
            connection_string (str): The connection string for the TiDB database,
                format: "mysql+pymysql://root@34.212.137.91:4000/test".
            collection_name (str, optional): The name of the collection,
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

        tidb_collection = get_or_create_collection(
            connection_string=connection_string,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            engine_args=engine_args,
            **kwargs,
        )

        return cls(
            connection_string=connection_string,
            collection=tidb_collection,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
        )

    def drop_collection(self) -> None:
        """
        Drop the collection from the TiDB database.
        """
        self._tidb.drop_collection()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to TiDB Vector.

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
        Delete vectors from the TiDB vector.

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
