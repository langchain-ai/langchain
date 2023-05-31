"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations

import uuid
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_env
from langchain.vectorstores.base import VectorStore


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

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        refresh_indices: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
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
        ids = []
        embeddings = self.embedding.embed_documents(list(texts))
        dim = len(embeddings[0])
        mapping = _default_text_mapping(dim)

        # check to see if the index already exists
        try:
            self.client.indices.get(index=self.index_name)
        except NotFoundError:
            # TODO would be nice to create index before embedding,
            # just to save expensive steps for last
            self.client.indices.create(index=self.index_name, mappings=mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "vector": embeddings[i],
                "text": text,
                "metadata": metadata,
                "_id": _id,
            }
            ids.append(_id)
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
        response = self.client.search(index=self.index_name, query=script_query, size=k)
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
        elasticsearch_url: Optional[str] = None,
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
        elasticsearch_url = elasticsearch_url or get_from_env(
            "elasticsearch_url", "ELASTICSEARCH_URL"
        )
        index_name = index_name or uuid.uuid4().hex
        vectorsearch = cls(elasticsearch_url, index_name, embedding, **kwargs)
        vectorsearch.add_texts(
            texts, metadatas=metadatas, refresh_indices=refresh_indices
        )
        return vectorsearch


class ElasticKnnSearch(ElasticVectorSearch):
    """
        TODO docstrings
    """

    def __init__(
        self, 
        index_name: str,
        embedding: Embeddings,
        es_connection: Optional["Elasticsearch"] = None,
        es_cloud_id: Optional[str] = None, 
        es_user: Optional[str] = None, 
        es_password: Optional[str] = None, 
#        *args, **kwargs
    ):
        """Initialize an instance of ElasticKnnSearch."""

        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        self.embedding = embedding
        self.index_name = index_name

        # If a pre-existing Elasticsearch connection is provided, use it.
        if es_connection is not None:
            self.client = es_connection
        else:
            # If credentials for a new Elasticsearch connection are provided, create a new connection.
            if es_cloud_id and es_user and es_password:
                self.client = elasticsearch.Elasticsearch(
                    cloud_id=es_cloud_id, 
                    basic_auth=(es_user, es_password)
                )
            else:
                raise ValueError(
                    """Either provide a pre-existing Elasticsearch connection, \
                or valid credentials for creating a new connection."""
                    )
                
    
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
                    "similarity": "dot_product"
                }
            }
        }

    @staticmethod
    def _default_knn_query(query_vector: Optional[List[float]] = None,
                           query: Optional[str] = None,
                           model_id: Optional[str] = None,
                           field: str = 'vector',
                           k: int = 10,
                           num_candidates: int = 10
                          ) -> Dict:
        knn = {
            "field": field,
            "k": k, 
            "num_candidates": num_candidates,
        }
        
        # Case 1: `query_vector` is provided, but not `query` and `model_id`
        if query_vector and not (query and model_id):
            knn["query_vector"] = query_vector
        
        # Case 2: `query` and `model_id` are provided, but not `query_vector`
        elif query and model_id and not query_vector:
            knn["query_vector_builder"] = {
                "text_embedding": { 
                    "model_id": model_id,  # use 'model_id' argument
                    "model_text": query  # use 'query' argument
                }
            }
        
        else:
            raise ValueError("Either `query_vector` or both `query` and `model_id` must be provided, but not both.")
        
        return knn

    def create_index(self):
        """Creates an Elasticsearch index."""
        self.client.indices.create(
            index=self.index_name,
            body={"mappings": self.mapping},
            ignore=400
        )
    
    
    def add_texts(self, texts: List[str], model_id: Optional[str] = None) -> None:
        """Adds the provided texts to the Elasticsearch index."""

        try:
#            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        # Check if the index exists.
        if not self.client.indices.exists(index=self.index_name):
            # If the index does not exist, raise an exception.
            raise Exception(f"The index '{self.index_name}' does not exist. If you want to create a new index while encoding texts, call 'from_texts' instead.")
    
        # Assign the encoding function from the instance's 'embedding' attribute to 'emb_func'
        emb_func = self.embedding.embed_documents
    
        # Generate embeddings for the input texts.
        # If 'model_id' is provided, use it as an argument to 'emb_func'.
        # Otherwise, call 'emb_func' with 'texts' as the only argument.
        embeddings = emb_func(texts) if not model_id else emb_func(texts, model_id=model_id)
    
        # Create a list of dictionaries, each containing a text and its corresponding embedding.
        # 'zip(texts, embeddings)' is used to iterate over 'texts' and 'embeddings' in parallel.
        body = [
            {
                '_op_type': 'index',
                '_index': self.index_name,
                "text": text,
                "vector": vector
            }
            for text, vector in zip(texts, embeddings)
        ]
    
        # Add the list of text-embedding pairs to the Elasticsearch index.
        bulk(self.client, body)

    
    def from_texts(
            self, texts: List[str],
            dims: int,
            model_id: Optional[str] = None
        ) -> None:
        """
        Creates an Elasticsearch index and adds the provided texts (encoded as embeddings)
        to the created index.
    
        :param texts: A list of text strings to be added to the index.
        :param model_id: An optional id specifying the model to use for encoding the texts.
                         If not provided, the default encoding method is used.
        """
        # Create the mapping using the provided dimensions.
        self.mapping = self._default_knn_mapping(dims=dims)
    
        # Create a new Elasticsearch index.
        self.create_index()
    
        # Encode the provided texts and add them to the newly created index.
        self.add_texts(texts, model_id=model_id)

    def knn_search(self, 
                   query: Union[str, List[str]], 
                   k: int = 10, 
                   query_vector: Optional[List[float]] = None, 
                   model_id: Optional[str] = None,
                   size: int = 10
                  ) -> Dict:
        """
        Performs a k-nearest neighbors (kNN) search on the Elasticsearch index using the
        provided query.
    
        :param query: The query text (or list of texts) to be used for the kNN search.
                      The query is transformed into an embedding using the same model as 
                      the texts in the index.
        :param k: The number of nearest neighbors to return.
        :param query_vector: An optional query vector.
        :param model_id: An optional id specifying the model to use for encoding the query.
                         If not provided, the default encoding method is used.
    
        :return: A dictionary containing the search results.
        """
        # Generate the body of the search query by calling the '_default_knn_query' method.
        # This method generates a search query that Elasticsearch can interpret.
        knn_query_body = self._default_knn_query(
            query_vector=query_vector, 
            query=query, 
            model_id=model_id, 
            k=k        )
    
        # Perform the kNN search on the Elasticsearch index and return the results.
        return self.client.search(
            index=self.index_name, 
            knn=knn_query_body,
            size=size
        )
    
    
    def knn_hybrid_search(self, 
                    query: Union[str, List[str]], 
                    k: int = 10, 
                    query_vector: Optional[List[float]] = None, 
                    model_id: Optional[str] = None,
                    size: int = 10
                    ) -> Dict:
        """
        Performs a hybrid search that combines a k-nearest neighbors (kNN) search 
        with a standard Elasticsearch query.
    
        :param query: The query text (or list of texts) to be used for the search.
                      The query is transformed into an embedding using the same model as 
                      the texts in the index.
        :param k: The number of nearest neighbors to return.
        :param query_vector: An optional query vector.
        :param model_id: An optional id specifying the model to use for encoding the query.
                         If not provided, the default encoding method is used.
    
        :return: A dictionary containing the search results.
        """
        # Generate the body of the kNN search query by calling the '_default_knn_query' method.
        knn_query_body = self._default_knn_query(
            query_vector=query_vector, 
            query=query, 
            model_id=model_id, 
            k=k)
    
        # Modify the knn_query_body to add a "boost" parameter
        knn_query_body["boost"] = 0.9
    
        # Generate the body of the standard Elasticsearch query
        match_query_body = {
                "match": {
                    "text": {
                        "query": query,
                        "boost": 0.1
                    }
                }
            }
    

        # Perform the hybrid search on the Elasticsearch index and return the results.
        return self.client.search(
                            index=self.index_name,
                            query=match_query_body,
                            knn=knn_query_body,
#                            fields=fields,
                            size=size,
#                            source=False
                          )

# TODO source variable and fields variable