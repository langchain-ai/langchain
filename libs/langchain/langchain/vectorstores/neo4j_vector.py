from __future__ import annotations

import enum
import logging
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE


class Neo4jVector(VectorStore):
    """`Neo4j` vector index.

    To use, you should have the ``neo4j`` python package installed.

    Args:
        url: Neo4j connection url
        username: Neo4j username.
        password: Neo4j password
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        distance_strategy: The distance strategy to use. (default: COSINE)

    Example:
        .. code-block:: python

            from langchain.vectorstores.neo4j_vector import Neo4jVector
            from langchain.embeddings.openai import OpenAIEmbeddings

            url="bolt://localhost:7687"
            username="neo4j"
            password="pleaseletmein"
            embeddings = OpenAIEmbeddings()
            vectorestore = Neo4jVector.from_documents(
                embedding=embeddings,
                documents=docs,
                url=url
                username=username,
                password=password,
            )


    """

    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        embedding_function: Embeddings,
        database: str = "neo4j",
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

        # Verify if the version support vector index
        self.verify_version()

        self.embedding_function = embedding_function
        self._distance_strategy = distance_strategy.value
        self.index_name = index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.logger = logger or logging.getLogger(__name__)

        # Calculate embedding dimension
        self.embedding_dimension = len(embedding_function.embed_query("test"))

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        This method sends a Cypher query to the connected Neo4j database
        and returns the results as a list of dictionaries.

        Args:
            query (str): The Cypher query to execute.
            params (dict, optional): Dictionary of query parameters. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the query results.

        Raises:
            ValueError: If the generated Cypher statement is not valid.
        """
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def verify_version(self) -> None:
        """
        Check if the connected Neo4j database version supports vector indexing.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.11.0) that is known to support vector
        indexing. Raises a ValueError if the connected Neo4j version is
        not supported.
        """
        version = self.query("CALL dbms.components()")[0]["versions"][0]
        version_tuple = tuple(map(int, version.split(".")))
        target_version = (5, 11, 0)

        if version_tuple < target_version:
            raise ValueError(
                "Version index is only supported in Neo4j version 5.11 or greater"
            )

    def retrieve_existing_index(self) -> Optional[int]:
        """
        Check if the vector index exists in the Neo4j database
        and returns its embedding dimension.

        This method queries the Neo4j database for existing indexes
        and attempts to retrieve the dimension of the vector index
        with the specified name. If the index exists, its dimension is returned.
        If the index doesn't exist, `None` is returned.

        Returns:
            int or None: The embedding dimension of the existing index if found.
        """

        index_information = self.query(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "WHERE type = 'VECTOR' AND name = $index_name "
            "RETURN name, labelsOrTypes, properties, options ",
            {"index_name": self.index_name},
        )
        try:
            self.node_label = index_information[0]["labelsOrTypes"][0]
            self.embedding_node_property = index_information[0]["properties"][0]
            embedding_dimension = index_information[0]["options"]["indexConfig"][
                "vector.dimensions"
            ]

            return embedding_dimension
        except IndexError:
            return None

    def create_new_index(self) -> None:
        """
        This method constructs a Cypher query and executes it
        to create a new vector index in Neo4j.
        """
        index_query = (
            "CALL db.index.vector.createNodeIndex("
            "$index_name,"
            "$node_label,"
            "$embedding_node_property,"
            "toInteger($embedding_dimension),"
            "$similarity_metric )"
        )

        parameters = {
            "index_name": self.index_name,
            "node_label": self.node_label,
            "embedding_node_property": self.embedding_node_property,
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": self._distance_strategy,
        }
        self.query(index_query, parameters)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        username: str,
        password: str,
        url: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        store = cls(
            username=username,
            password=password,
            url=url,
            embedding_function=embedding,
            **kwargs,
        )

        # Check if the index already exists
        embedding_dimension = store.retrieve_existing_index()

        # If the index doesn't exist yet
        if not embedding_dimension:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        import_query = (
            "UNWIND $data AS row "
            f"CREATE (c:{self.node_label}) "
            f"SET c.{self.embedding_node_property} = row.embedding "
            "SET c.id = row.id "
            f"SET c.{self.text_node_property} = row.text "
            "SET c += row.metadata "
        )

        parameters = {
            "data": [
                {"text": text, "metadata": metadata, "embedding": embedding, "id": id}
                for text, metadata, embedding, id in zip(
                    texts, metadatas, embeddings, ids
                )
            ]
        }

        self.query(import_query, parameters)

        return ids

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
        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Neo4jVector.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
        )

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(embedding=embedding, k=k)
        return docs

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == "euclidean":
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == "cosine":
            return self.EmbeddingStore.embedding.cosine_distance
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of `l2`, `cosine`, `inner`."
            )

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the Neo4j database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the Neo4j database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """

        read_query = (
            "CALL db.index.vector.queryNodes($index, $k, $embedding) "
            "YIELD node, score "
            f"RETURN node.{self.text_node_property} AS text, score, "
            f"node {{.*, {self.text_node_property}: Null, "
            f"{self.embedding_node_property}: Null }} AS metadata"
        )

        parameters = {"index": self.index_name, "k": k, "embedding": embedding}

        results = self.query(read_query, parameters)

        docs = [
            (
                Document(
                    page_content=result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls: Type[Neo4jVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from texts and embeddings.
        Neo4j credentials are required in the form of
        `url`, `username`, and `password` parameters.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        url: str,
        username: str,
        password: str,
        metadatas: Optional[List[dict]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> Neo4jVector:
        """Construct Neo4jVector wrapper from raw documents and pre-
        generated embeddings.

        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of
        `url`, `username`, and `password` parameters.

        Example:
            .. code-block:: python

                from langchain.vectorstores.neo4j_vector import Neo4jVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                vectorstore = Neo4jVector.from_embeddings(
                    text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            url=url,
            username=username,
            password=password,
            metadatas=metadatas,
            ids=ids,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        index_name: str,
        username: str,
        password: str,
        url: str,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Get instance of an existing Neo4j vector index.This method will
        return the instance of the store without inserting any new
        embeddings.
        Neo4j credentials are required in the form of
        `url`, `username`, and `password` parameters along with
        the `index_name` definition.
        """

        store = cls(
            username=username,
            password=password,
            url=url,
            embedding_function=embedding,
            index_name=index_name,
            distance_strategy=distance_strategy,
            **kwargs,
        )

        embedding_dimension = store.retrieve_existing_index()

        if not embedding_dimension:
            raise ValueError(
                "The specified vector index name does not exist. "
                "Make sure to check if you spelled it correctly"
            )

        # Check if embedding function and vector index dimensions match
        if not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )

        return store

    @classmethod
    def from_documents(
        cls: Type[Neo4jVector],
        documents: List[Document],
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Return Neo4jVector initialized from documents and embeddings.
        Neo4j credentials are required in the form of
        `url`, `username`, and `password` parameters.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
