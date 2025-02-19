from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from packaging import version

try:
    from arango.database import StandardDatabase
    from arango.exceptions import ArangoServerError

    ARANGO_INSTALLED = True
except ImportError:
    print("ArangoDB not installed, please install with `pip install python-arango`.")
    ARANGO_INSTALLED = False


from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "l2",
    DistanceStrategy.COSINE: "cosine",
}


class SearchType(str, Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    # HYBRID = "hybrid" # TODO


DEFAULT_SEARCH_TYPE = SearchType.VECTOR


class ArangoVector(VectorStore):
    """ArangoDB vector index.

    To use this, you should have the `python-arango` python package installed.

    Args:
        embedding: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        embedding_dimension: The dimension of the to-be-inserted embedding vectors.
        database: The python-arango database instance.
        collection_name: The name of the collection to use. (default: "documents")
        search_type: The type of search to be performed, currently only 'vector' is supported.
        embedding_field: The field name storing the embedding vector. (default: "embedding")
        text_field: The field name storing the text. (default: "text")
        index_name: The name of the vector index to use. (default: "vector_index")
        distance_strategy: The distance strategy to use. (default: "COSINE")
        num_centroids: The number of centroids for the vector index. (default: 1)
        relevance_score_fn: A function to normalize the relevance score. If not provided,
            the default normalization function for the distance strategy will be used.

    Example:
        .. code-block:: python

            from arango import ArangoClient
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            from langchain_community.vectorstores.arangodb_vector import ArangoVector

            db = ArangoClient("http://localhost:8529").db("test", username="root", password="openSesame")

            embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=dimension)

            vector_store = ArangoVector.from_texts(
                texts=["hello world", "hello langchain", "hello arangodb"],
                embedding=embedding,
                database=db,
                collection_name="Documents"
            )

            print(vector_store.similarity_search("arangodb", k=1))
    """

    def __init__(
        self,
        embedding: Embeddings,
        embedding_dimension: int,
        database: "StandardDatabase",
        collection_name: str = "documents",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        embedding_field: str = "embedding",
        text_field: str = "text",
        index_name: str = "vector_index",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        if not ARANGO_INSTALLED:
            m = "ArangoDB not installed, please install with `pip install python-arango`."
            raise ImportError(m)

        if search_type not in [SearchType.VECTOR]:
            raise ValueError("search_type must be 'vector'")

        if distance_strategy not in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        ]:
            m = "distance_strategy must be 'COSINE' or 'EUCLIDEAN_DISTANCE'"
            raise ValueError(m)

        self.embedding = embedding
        self.embedding_dimension = int(embedding_dimension)
        self.db = database
        self.async_db = self.db.begin_async_execution(return_result=False)
        self.search_type = search_type
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.index_name = index_name
        self._distance_strategy = distance_strategy
        self.num_centroids = num_centroids
        self.override_relevance_score_fn = relevance_score_fn

        if not self.db.has_collection(collection_name):
            self.db.create_collection(collection_name)

        self.collection = self.db.collection(self.collection_name)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def retrieve_vector_index(self) -> dict[str, Any] | None:
        """Retrieve the vector index from the collection."""
        indexes = self.collection.indexes()
        for index in indexes:
            if index["name"] == self.index_name:
                return index

        return None

    def create_vector_index(self) -> None:
        """Create the vector index on the collection."""
        self.collection.add_index(
            {
                "name": self.index_name,
                "type": "vector",
                "fields": [self.embedding_field],
                "params": {
                    "metric": DISTANCE_MAPPING[self._distance_strategy],
                    "dimension": self.embedding_dimension,
                    "nLists": self.num_centroids,
                },
            }
        )

    def delete_vector_index(self) -> None:
        """Delete the vector index from the collection."""
        index = self.retrieve_vector_index()

        if index is not None:
            self.collection.delete_index(index["id"])

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
        use_async_db: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore."""
        if ids is None:
            try:
                import farmhash
            except ImportError:
                m = "Farmhash not installed, please install with `pip install cityhash`.  Alternatively, provide ids."
                raise ImportError(m)

            ids = [str(farmhash.Fingerprint64(text.encode("utf-8"))) for text in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        if len(ids) != len(texts) != len(embeddings) != len(metadatas):
            m = "Length of ids, texts, embeddings and metadatas must be the same."
            raise ValueError(m)

        db = self.async_db if use_async_db else self.db
        collection = db.collection(self.collection_name)

        data = []
        for _key, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            data.append(
                {
                    **metadata,
                    "_key": _key,
                    self.text_field: text,
                    self.embedding_field: embedding,
                }
            )

            if len(data) == batch_size:
                collection.import_bulk(data, on_duplicate="update", **kwargs)
                data = []

        collection.import_bulk(data, on_duplicate="update", **kwargs)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding.embed_documents(list(texts))

        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with ArangoDB.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate search. Defaults to True. If False,
                exact search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = embedding or self.embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate search. Defaults to True. If False,
                exact search will be used.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            return_fields=return_fields,
            use_approx=use_approx,
            **kwargs,
        )

        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate search. Defaults to True. If False,
                exact search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = embedding or self.embedding.embed_query(query)
        result = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            query=query,
            return_fields=return_fields,
            use_approx=use_approx,
            **kwargs,
        )
        return result

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set. To
                return all fields, use return_all_fields=True.
            use_approx: Whether to use approximate search. Defaults to True. If False,
                exact search will be used.
        Returns:
            List of Documents most similar to the query vector.
        """
        if self._distance_strategy == DistanceStrategy.COSINE:
            score_func = "APPROX_NEAR_COSINE" if use_approx else "COSINE_SIMILARITY"
            sort_order = "DESC"
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            score_func = "APPROX_NEAR_L2" if use_approx else "L2_DISTANCE"
            sort_order = "ASC"
        else:
            raise ValueError(f"Unsupported metric: {self._distance_strategy}")

        if use_approx:
            if version.parse(self.db.version()) < version.parse("3.12.4"):
                raise ValueError("Approximate Nearest Neighbor search requires ArangoDB >= 3.12.4, consider setting use_approx=False.")

            if not self.retrieve_vector_index():
                self.create_vector_index()

        return_fields.update({"_key", self.text_field})
        return_fields_list = list(return_fields)

        aql = f"""
            FOR doc IN @@collection
                LET score = {score_func}(doc.{self.embedding_field}, @query_embedding)
                SORT score {sort_order}
                LIMIT {k}
                LET data = KEEP(doc, {return_fields_list})
                RETURN {{data, score}}
        """

        bind_vars = {
            "@collection": self.collection_name,
            "query_embedding": embedding,
        }

        cursor = self.db.aql.execute(aql, bind_vars=bind_vars)

        score: float
        data: dict[str, Any]
        result: dict[str, Any]
        results = []

        for result in cursor:
            data, score = result["data"], result["score"]

            _key = data.pop("_key")
            page_content = data.pop(self.text_field)

            doc = Document(page_content=page_content, id=_key, metadata=data)
            results.append((doc, score))

        return results

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that can be used to delete vectors.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if not ids:
            return False

        for result in self.collection.delete_many(ids, **kwargs):
            if isinstance(result, ArangoServerError):
                print(result)
                return False

        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to get.

        Returns:
            List of Documents with the given ids.
        """
        docs = []
        doc: dict[str, Any]

        for doc in self.collection.get_many(ids):
            _key = doc.pop("_key")
            page_content = doc.pop(self.text_field)

            docs.append(Document(page_content=page_content, id=_key, metadata=doc))

        return docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        return_fields: set[str] = set(),
        use_approx: bool = True,
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            return_fields: Fields to return in the result. For example,
                {"foo", "bar"} will return the "foo" and "bar" fields of the document,
                in addition to the _key & text field. Defaults to an empty set.
            use_approx: Whether to use approximate search. Defaults to True. If False,
                exact search will be used.
            embedding: Optional embedding to use for the query. If not provided,
                the query will be embedded using the embedding function provided
                in the constructor.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return_fields.add(self.embedding_field)

        # Embed the query
        query_embedding = embedding or self.embedding.embed_query(query)

        # Fetch the initial documents
        docs_with_scores = self.similarity_search_by_vector_with_score(
            embedding=query_embedding,
            k=fetch_k,
            return_fields=return_fields,
            use_approx=use_approx,
            **kwargs,
        )

        # Get the embeddings for the fetched documents
        embeddings = [doc.metadata[self.embedding_field] for doc, _ in docs_with_scores]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        selected_docs = [docs_with_scores[i][0] for i in selected_indices]

        return selected_docs

    @classmethod
    def from_texts(
        cls: Type[ArangoVector],
        texts: List[str],
        embedding: Embeddings,
        database: "StandardDatabase",
        collection_name: str = "documents",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        embedding_field: str = "embedding",
        text_field: str = "text",
        index_name: str = "vector_index",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        num_centroids: int = 1,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        overwrite_index: bool = False,
        **kwargs: Any,
    ) -> ArangoVector:
        """
        Return ArangoDBVector initialized from texts, embeddings and a database.
        """
        embeddings = embedding.embed_documents(list(texts))

        embedding_dimension = len(embeddings[0])

        store = cls(
            embedding,
            embedding_dimension=embedding_dimension,
            database=database,
            collection_name=collection_name,
            search_type=search_type,
            embedding_field=embedding_field,
            text_field=text_field,
            index_name=index_name,
            distance_strategy=distance_strategy,
            num_centroids=num_centroids,
            **kwargs,
        )

        if overwrite_index:
            store.delete_vector_index()

        store.add_embeddings(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

        return store

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy in [
            DistanceStrategy.COSINE,
            DistanceStrategy.EUCLIDEAN_DISTANCE,
        ]:
            return lambda x: x
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to ArangoVector constructor."
            )
