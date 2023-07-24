"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations

import uuid
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


def _default_text_mapping(dim: int) -> Dict:
    return {
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": dim},
        }
    }


def _default_script_query(query_vector: List[float], filter: Optional[dict]) -> Dict:
    if filter:
        ((key, value),) = filter.items()
        filter = {"match": {f"metadata.{key}.keyword": f"{value}"}}
    else:
        filter = {"match_all": {}}
    return {
        "script_score": {
            "query": filter,
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector},
            },
        }
    }


# ElasticVectorSearch is a concrete implementation of the abstract base class
# VectorStore, which defines a common interface for all vector database
# implementations. By inheriting from the ABC class, ElasticVectorSearch can be
# defined as an abstract base class itself, allowing the creation of subclasses with
# their own specific implementations. If you plan to subclass ElasticVectorSearch,
# you can inherit from it and define your own implementation of the necessary methods
# and attributes.
class ElasticVectorSearch(VectorStore, ABC):
    """Wrapper around Elasticsearch as a vector database.

    To connect to an Elasticsearch instance that does not require
    login credentials, pass the Elasticsearch URL and index name along with the
    embedding object to the constructor.

    Example:
        .. code-block:: python

            from langchain import ElasticVectorSearch
            from langchain.embeddings import OpenAIEmbeddings

            embedding = OpenAIEmbeddings()
            elastic_vector_search = ElasticVectorSearch(
                elasticsearch_url="http://localhost:9200",
                index_name="test_index",
                embedding=embedding
            )


    To connect to an Elasticsearch instance that requires login credentials,
    including Elastic Cloud, use the Elasticsearch URL format
    https://username:password@es_host:9243. For example, to connect to Elastic
    Cloud, create the Elasticsearch URL with the required authentication details and
    pass it to the ElasticVectorSearch constructor as the named parameter
    elasticsearch_url.

    You can obtain your Elastic Cloud URL and login credentials by logging in to the
    Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
    navigating to the "Deployments" page.

    To obtain your Elastic Cloud password for the default "elastic" user:

    1. Log in to the Elastic Cloud console at https://cloud.elastic.co
    2. Go to "Security" > "Users"
    3. Locate the "elastic" user and click "Edit"
    4. Click "Reset password"
    5. Follow the prompts to reset the password

    The format for Elastic Cloud URLs is
    https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.

    Example:
        .. code-block:: python

            from langchain import ElasticVectorSearch
            from langchain.embeddings import OpenAIEmbeddings

            embedding = OpenAIEmbeddings()

            elastic_host = "cluster_id.region_id.gcp.cloud.es.io"
            elasticsearch_url = f"https://username:password@{elastic_host}:9243"
            elastic_vector_search = ElasticVectorSearch(
                elasticsearch_url=elasticsearch_url,
                index_name="test_index",
                embedding=embedding
            )

    Args:
        elasticsearch_url (str): The URL for the Elasticsearch instance.
        index_name (str): The name of the Elasticsearch index for the embeddings.
        embedding (Embeddings): An object that provides the ability to embed text.
                It should be an instance of a class that subclasses the Embeddings
                abstract base class, such as OpenAIEmbeddings()

    Raises:
        ValueError: If the elasticsearch python package is not installed.
    """

    def __init__(
        self,
        elasticsearch_url: str,
        index_name: str,
        embedding: Embeddings,
        *,
        ssl_verify: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with necessary components."""
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        self.embedding = embedding
        self.index_name = index_name
        _ssl_verify = ssl_verify or {}
        try:
            self.client = elasticsearch.Elasticsearch(elasticsearch_url, **_ssl_verify)
        except ValueError as e:
            raise ValueError(
                f"Your elasticsearch client string is mis-formatted. Got error: {e} "
            )

    @property
    def embeddings(self) -> Embeddings:
        return self.embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        refresh_indices: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        try:
            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding.embed_documents(list(texts))
        dim = len(embeddings[0])
        mapping = _default_text_mapping(dim)

        # check to see if the index already exists
        try:
            self.client.indices.get(index=self.index_name)
        except NotFoundError:
            # TODO would be nice to create index before embedding,
            # just to save expensive steps for last
            self.create_index(self.client, self.index_name, mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "vector": embeddings[i],
                "text": text,
                "metadata": metadata,
                "_id": ids[i],
            }
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        documents = [d[0] for d in docs_and_scores]
        return documents

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(query)
        script_query = _default_script_query(embedding, filter)
        response = self.client_search(
            self.client, self.index_name, script_query, size=k
        )
        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"]["text"],
                    metadata=hit["_source"]["metadata"],
                ),
                hit["_score"],
            )
            for hit in hits
        ]
        return docs_and_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        refresh_indices: bool = True,
        **kwargs: Any,
    ) -> ElasticVectorSearch:
        """Construct ElasticVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Elasticsearch instance.
            3. Adds the documents to the newly created Elasticsearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import ElasticVectorSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                elastic_vector_search = ElasticVectorSearch.from_texts(
                    texts,
                    embeddings,
                    elasticsearch_url="http://localhost:9200"
                )
        """
        elasticsearch_url = get_from_dict_or_env(
            kwargs, "elasticsearch_url", "ELASTICSEARCH_URL"
        )
        index_name = index_name or uuid.uuid4().hex
        vectorsearch = cls(elasticsearch_url, index_name, embedding, **kwargs)
        vectorsearch.add_texts(
            texts, metadatas=metadatas, ids=ids, refresh_indices=refresh_indices
        )
        return vectorsearch

    def create_index(self, client: Any, index_name: str, mapping: Dict) -> None:
        version_num = client.info()["version"]["number"][0]
        version_num = int(version_num)
        if version_num >= 8:
            client.indices.create(index=index_name, mappings=mapping)
        else:
            client.indices.create(index=index_name, body={"mappings": mapping})

    def client_search(
        self, client: Any, index_name: str, script_query: Dict, size: int
    ) -> Any:
        version_num = client.info()["version"]["number"][0]
        version_num = int(version_num)
        if version_num >= 8:
            response = client.search(index=index_name, query=script_query, size=size)
        else:
            response = client.search(
                index=index_name, body={"query": script_query, "size": size}
            )
        return response

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """

        if ids is None:
            raise ValueError("No ids provided to delete.")

        # TODO: Check if this can be done in bulk
        for id in ids:
            self.client.delete(index=self.index_name, id=id)


class ElasticKnnSearch(ElasticVectorSearch):
    """
    A class for performing k-Nearest Neighbors (k-NN) search on an Elasticsearch index.
    The class is designed for a text search scenario where documents are text strings
    and their embeddings are vector representations of those strings.
    """

    def __init__(
        self,
        index_name: str,
        embedding: Embeddings,
        es_connection: Optional["Elasticsearch"] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
        vector_query_field: Optional[str] = "vector",
        query_field: Optional[str] = "text",
    ):
        """
        Initializes an instance of the ElasticKnnSearch class and sets up the
            Elasticsearch client.

        Args:
            index_name: The name of the Elasticsearch index.
            embedding: An instance of the Embeddings class, used to generate vector
                representations of text strings.
            es_connection: An existing Elasticsearch connection.
            es_cloud_id: The Cloud ID of the Elasticsearch instance. Required if
                creating a new connection.
            es_user: The username for the Elasticsearch instance. Required if
                creating a new connection.
            es_password: The password for the Elasticsearch instance. Required if
                creating a new connection.
        """
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        self.embedding = embedding
        self.index_name = index_name
        self.query_field = query_field
        self.vector_query_field = vector_query_field

        # If a pre-existing Elasticsearch connection is provided, use it.
        if es_connection is not None:
            self.client = es_connection
        else:
            # If credentials for a new Elasticsearch connection are provided,
            # create a new connection.
            if es_cloud_id and es_user and es_password:
                self.client = elasticsearch.Elasticsearch(
                    cloud_id=es_cloud_id, basic_auth=(es_user, es_password)
                )
            else:
                raise ValueError(
                    """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
                )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        refresh_indices: bool = True,
        es_connection: Optional["Elasticsearch"] = None,
        es_cloud_id: Optional[str] = None,
        es_user: Optional[str] = None,
        es_password: Optional[str] = None,
        **kwargs: Any,
    ) -> ElasticKnnSearch:
        """Construct ElasticKnnSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Elasticsearch instance.
            3. Adds the documents to the newly created Elasticsearch index.

        This is intended to be a quick way to get started.
        """
        index_name = index_name or uuid.uuid4().hex
        vectorsearch = cls(
            index_name,
            embedding,
            es_connection=es_connection,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_password=es_password,
            **kwargs,
        )
        vectorsearch.add_texts(
            texts, metadatas=metadatas, refresh_indices=refresh_indices, ids=ids
        )
        return vectorsearch

    @staticmethod
    def _default_knn_mapping(dims: int) -> Dict:
        """Generates a default index mapping for kNN search."""
        return {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "dot_product",
                },
            }
        }

    def _default_knn_query(
        self,
        query_vector: Optional[List[float]] = None,
        query: Optional[str] = None,
        model_id: Optional[str] = None,
        k: Optional[int] = 10,
        num_candidates: Optional[int] = 10,
    ) -> Dict:
        knn: Dict = {
            "field": self.vector_query_field,
            "k": k,
            "num_candidates": num_candidates,
        }

        # Case 1: `query_vector` is provided, but not `model_id` -> use query_vector
        if query_vector and not model_id:
            knn["query_vector"] = query_vector

        # Case 2: `query` and `model_id` are provided, -> use query_vector_builder
        elif query and model_id:
            knn["query_vector_builder"] = {
                "text_embedding": {
                    "model_id": model_id,  # use 'model_id' argument
                    "model_text": query,  # use 'query' argument
                }
            }

        else:
            raise ValueError(
                "Either `query_vector` or `model_id` must be provided, but not both."
            )

        return knn

    def knn_search(
        self,
        query: Optional[str] = None,
        k: Optional[int] = 10,
        query_vector: Optional[List[float]] = None,
        model_id: Optional[str] = None,
        size: Optional[int] = 10,
        source: Optional[bool] = True,
        fields: Optional[
            Union[List[Mapping[str, Any]], Tuple[Mapping[str, Any], ...], None]
        ] = None,
    ) -> Dict:
        """
        Performs a k-nearest neighbor (k-NN) search on the Elasticsearch index.

        The search can be conducted using either a raw query vector or a model ID.
        The method first generates
        the body of the search query, which can be interpreted by Elasticsearch.
        It then performs the k-NN
        search on the Elasticsearch index and returns the results.

        Args:
            query: The query or queries to be used for the search. Required if
                `query_vector` is not provided.
            k: The number of nearest neighbors to return. Defaults to 10.
            query_vector: The query vector to be used for the search. Required if
                `query` is not provided.
            model_id: The ID of the model to use for generating the query vector, if
                `query` is provided.
            size: The number of search hits to return. Defaults to 10.
            source: Whether to include the source of each hit in the results.
            fields: The fields to include in the source of each hit. If None, all
                fields are included.
            vector_query_field: Field name to use in knn search if not default 'vector'

        Returns:
            The search results.

        Raises:
            ValueError: If neither `query_vector` nor `model_id` is provided, or if
                both are provided.
        """

        knn_query_body = self._default_knn_query(
            query_vector=query_vector, query=query, model_id=model_id, k=k
        )

        # Perform the kNN search on the Elasticsearch index and return the results.
        res = self.client.search(
            index=self.index_name,
            knn=knn_query_body,
            size=size,
            source=source,
            fields=fields,
        )
        return dict(res)

    def knn_hybrid_search(
        self,
        query: Optional[str] = None,
        k: Optional[int] = 10,
        query_vector: Optional[List[float]] = None,
        model_id: Optional[str] = None,
        size: Optional[int] = 10,
        source: Optional[bool] = True,
        knn_boost: Optional[float] = 0.9,
        query_boost: Optional[float] = 0.1,
        fields: Optional[
            Union[List[Mapping[str, Any]], Tuple[Mapping[str, Any], ...], None]
        ] = None,
    ) -> Dict[Any, Any]:
        """Performs a hybrid k-nearest neighbor (k-NN) and text-based search on the
            Elasticsearch index.

        The search can be conducted using either a raw query vector or a model ID.
        The method first generates
        the body of the k-NN search query and the text-based query, which can be
        interpreted by Elasticsearch.
        It then performs the hybrid search on the Elasticsearch index and returns the
        results.

        Args:
            query: The query or queries to be used for the search. Required if
                `query_vector` is not provided.
            k: The number of nearest neighbors to return. Defaults to 10.
            query_vector: The query vector to be used for the search. Required if
                `query` is not provided.
            model_id: The ID of the model to use for generating the query vector, if
                `query` is provided.
            size: The number of search hits to return. Defaults to 10.
            source: Whether to include the source of each hit in the results.
            knn_boost: The boost factor for the k-NN part of the search.
            query_boost: The boost factor for the text-based part of the search.
            fields
                The fields to include in the source of each hit. If None, all fields are
                included. Defaults to None.
            vector_query_field: Field name to use in knn search if not default 'vector'
            query_field: Field name to use in search if not default 'text'

        Returns:
            The search results.

        Raises:
            ValueError: If neither `query_vector` nor `model_id` is provided, or if
                both are provided.
        """

        knn_query_body = self._default_knn_query(
            query_vector=query_vector, query=query, model_id=model_id, k=k
        )

        # Modify the knn_query_body to add a "boost" parameter
        knn_query_body["boost"] = knn_boost

        # Generate the body of the standard Elasticsearch query
        match_query_body = {
            "match": {self.query_field: {"query": query, "boost": query_boost}}
        }

        # Perform the hybrid search on the Elasticsearch index and return the results.
        res = self.client.search(
            index=self.index_name,
            query=match_query_body,
            knn=knn_query_body,
            fields=fields,
            size=size,
            source=source,
        )
        return dict(res)
