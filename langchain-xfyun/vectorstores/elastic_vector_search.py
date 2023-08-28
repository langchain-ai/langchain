"""[DEPRECATED] Please use ElasticsearchStore instead."""
from __future__ import annotations

import uuid
import warnings
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

from langchain._api import deprecated
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


@deprecated("0.0.265", alternative="ElasticsearchStore class.", pending=True)
class ElasticVectorSearch(VectorStore):
    """[DEPRECATED] `Elasticsearch` vector store.

    To connect to an `Elasticsearch` instance that does not require
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
        warnings.warn(
            "ElasticVectorSearch will be removed in a future release. See"
            "Elasticsearch integration docs on how to upgrade."
        )

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
        return self.embedding

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
        if "elasticsearch_url" in kwargs:
            del kwargs["elasticsearch_url"]
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


class ElasticKnnSearch(VectorStore):
    """[DEPRECATED] `Elasticsearch` with k-nearest neighbor search
    (`k-NN`) vector store.

    It creates an Elasticsearch index of text data that
    can be searched using k-NN search. The text data is transformed into
    vector embeddings using a provided embedding model, and these embeddings
    are stored in the Elasticsearch index.

    Attributes:
        index_name (str): The name of the Elasticsearch index.
        embedding (Embeddings): The embedding model to use for transforming text data
            into vector embeddings.
        es_connection (Elasticsearch, optional): An existing Elasticsearch connection.
        es_cloud_id (str, optional): The Cloud ID of your Elasticsearch Service
            deployment.
        es_user (str, optional): The username for your Elasticsearch Service deployment.
        es_password (str, optional): The password for your Elasticsearch Service
            deployment.
        vector_query_field (str, optional): The name of the field in the Elasticsearch
            index that contains the vector embeddings.
        query_field (str, optional): The name of the field in the Elasticsearch index
            that contains the original text data.

    Usage:
        >>> from embeddings import Embeddings
        >>> embedding = Embeddings.load('glove')
        >>> es_search = ElasticKnnSearch('my_index', embedding)
        >>> es_search.add_texts(['Hello world!', 'Another text'])
        >>> results = es_search.knn_search('Hello')
        [(Document(page_content='Hello world!', metadata={}), 0.9)]
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
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        warnings.warn(
            "ElasticKnnSearch will be removed in a future release."
            "Use ElasticsearchStore instead. See Elasticsearch "
            "integration docs on how to upgrade."
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

    @staticmethod
    def _default_knn_mapping(
        dims: int, similarity: Optional[str] = "dot_product"
    ) -> Dict:
        return {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": similarity,
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

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Pass through to `knn_search`
        """
        results = self.knn_search(query=query, k=k, **kwargs)
        return [doc for doc, score in results]

    def similarity_search_with_score(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Pass through to `knn_search including score`"""
        return self.knn_search(query=query, k=k, **kwargs)

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
        page_content: Optional[str] = "text",
    ) -> List[Tuple[Document, float]]:
        """
        Perform a k-NN search on the Elasticsearch index.

        Args:
            query (str, optional): The query text to search for.
            k (int, optional): The number of nearest neighbors to return.
            query_vector (List[float], optional): The query vector to search for.
            model_id (str, optional): The ID of the model to use for transforming the
                query text into a vector.
            size (int, optional): The number of search results to return.
            source (bool, optional): Whether to return the source of the search results.
            fields (List[Mapping[str, Any]], optional): The fields to return in the
                search results.
            page_content (str, optional): The name of the field that contains the page
                content.

        Returns:
            A list of tuples, where each tuple contains a Document object and a score.
        """

        # if not source and (fields == None or page_content not in fields):
        if not source and (
            fields is None or not any(page_content in field for field in fields)
        ):
            raise ValueError("If source=False `page_content` field must be in `fields`")

        knn_query_body = self._default_knn_query(
            query_vector=query_vector, query=query, model_id=model_id, k=k
        )

        # Perform the kNN search on the Elasticsearch index and return the results.
        response = self.client.search(
            index=self.index_name,
            knn=knn_query_body,
            size=size,
            source=source,
            fields=fields,
        )

        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"][page_content]
                    if source
                    else hit["fields"][page_content][0],
                    metadata=hit["fields"] if fields else {},
                ),
                hit["_score"],
            )
            for hit in hits
        ]

        return docs_and_scores

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
        page_content: Optional[str] = "text",
    ) -> List[Tuple[Document, float]]:
        """
        Perform a hybrid k-NN and text search on the Elasticsearch index.

        Args:
            query (str, optional): The query text to search for.
            k (int, optional): The number of nearest neighbors to return.
            query_vector (List[float], optional): The query vector to search for.
            model_id (str, optional): The ID of the model to use for transforming the
                query text into a vector.
            size (int, optional): The number of search results to return.
            source (bool, optional): Whether to return the source of the search results.
            knn_boost (float, optional): The boost value to apply to the k-NN search
                results.
            query_boost (float, optional): The boost value to apply to the text search
                results.
            fields (List[Mapping[str, Any]], optional): The fields to return in the
                search results.
            page_content (str, optional): The name of the field that contains the page
                content.

        Returns:
            A list of tuples, where each tuple contains a Document object and a score.
        """

        # if not source and (fields == None or page_content not in fields):
        if not source and (
            fields is None or not any(page_content in field for field in fields)
        ):
            raise ValueError("If source=False `page_content` field must be in `fields`")

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
        response = self.client.search(
            index=self.index_name,
            query=match_query_body,
            knn=knn_query_body,
            fields=fields,
            size=size,
            source=source,
        )

        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            (
                Document(
                    page_content=hit["_source"][page_content]
                    if source
                    else hit["fields"][page_content][0],
                    metadata=hit["fields"] if fields else {},
                ),
                hit["_score"],
            )
            for hit in hits
        ]

        return docs_and_scores

    def create_knn_index(self, mapping: Dict) -> None:
        """
        Create a new k-NN index in Elasticsearch.

        Args:
            mapping (Dict): The mapping to use for the new index.

        Returns:
            None
        """

        self.client.indices.create(index=self.index_name, mappings=mapping)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        model_id: Optional[str] = None,
        refresh_indices: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add a list of texts to the Elasticsearch index.

        Args:
            texts (Iterable[str]): The texts to add to the index.
            metadatas (List[Dict[Any, Any]], optional): A list of metadata dictionaries
                to associate with the texts.
            model_id (str, optional): The ID of the model to use for transforming the
                texts into vectors.
            refresh_indices (bool, optional): Whether to refresh the Elasticsearch
                indices after adding the texts.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A list of IDs for the added texts.
        """

        # Check if the index exists.
        if not self.client.indices.exists(index=self.index_name):
            dims = kwargs.get("dims")

            if dims is None:
                raise ValueError("ElasticKnnSearch requires 'dims' parameter")

            similarity = kwargs.get("similarity")
            optional_args = {}

            if similarity is not None:
                optional_args["similarity"] = similarity

            mapping = self._default_knn_mapping(dims=dims, **optional_args)
            self.create_knn_index(mapping)

        embeddings = self.embedding.embed_documents(list(texts))

        # body = []
        body: List[Mapping[str, Any]] = []
        for text, vector in zip(texts, embeddings):
            body.extend(
                [
                    {"index": {"_index": self.index_name}},
                    {"text": text, "vector": vector},
                ]
            )

        responses = self.client.bulk(operations=body)

        ids = [
            item["index"]["_id"]
            for item in responses["items"]
            if item["index"]["result"] == "created"
        ]

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)

        return ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> ElasticKnnSearch:
        """
        Create a new ElasticKnnSearch instance and add a list of texts to the
            Elasticsearch index.

        Args:
            texts (List[str]): The texts to add to the index.
            embedding (Embeddings): The embedding model to use for transforming the
                texts into vectors.
            metadatas (List[Dict[Any, Any]], optional): A list of metadata dictionaries
                to associate with the texts.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A new ElasticKnnSearch instance.
        """

        index_name = kwargs.get("index_name", str(uuid.uuid4()))
        es_connection = kwargs.get("es_connection")
        es_cloud_id = kwargs.get("es_cloud_id")
        es_user = kwargs.get("es_user")
        es_password = kwargs.get("es_password")
        vector_query_field = kwargs.get("vector_query_field", "vector")
        query_field = kwargs.get("query_field", "text")
        model_id = kwargs.get("model_id")
        dims = kwargs.get("dims")

        if dims is None:
            raise ValueError("ElasticKnnSearch requires 'dims' parameter")

        optional_args = {}

        if vector_query_field is not None:
            optional_args["vector_query_field"] = vector_query_field

        if query_field is not None:
            optional_args["query_field"] = query_field

        knnvectorsearch = cls(
            index_name=index_name,
            embedding=embedding,
            es_connection=es_connection,
            es_cloud_id=es_cloud_id,
            es_user=es_user,
            es_password=es_password,
            **optional_args,
        )
        # Encode the provided texts and add them to the newly created index.
        knnvectorsearch.add_texts(texts, model_id=model_id, dims=dims, **optional_args)

        return knnvectorsearch
