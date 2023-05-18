"""Wrapper around Elasticsearch vector database."""
from __future__ import annotations

import uuid
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
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

    def __init__(self, elasticsearch_url: str, index_name: str, embedding: Embeddings):
        """Initialize with necessary components."""
        try:
            import elasticsearch
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        self.embedding = embedding
        self.index_name = index_name
        try:
            es_client = elasticsearch.Elasticsearch(elasticsearch_url)  # noqa
        except ValueError as e:
            raise ValueError(
                f"Your elasticsearch client string is misformatted. Got error: {e} "
            )
        self.client = es_client

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
            raise ValueError(
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
        try:
            import elasticsearch
            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        try:
            client = elasticsearch.Elasticsearch(elasticsearch_url)
        except ValueError as e:
            raise ValueError(
                "Your elasticsearch client string is misformatted. " f"Got error: {e} "
            )
        index_name = kwargs.get("index_name", uuid.uuid4().hex)
        embeddings = embedding.embed_documents(texts)
        dim = len(embeddings[0])
        mapping = _default_text_mapping(dim)

        # check to see if the index already exists
        try:
            client.indices.get(index=index_name)
        except NotFoundError:
            # TODO would be nice to create index before embedding,
            # just to save expensive steps for last
            client.indices.create(index=index_name, mappings=mapping)

        requests = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            request = {
                "_op_type": "index",
                "_index": index_name,
                "vector": embeddings[i],
                "text": text,
                "metadata": metadata,
            }
            requests.append(request)
        bulk(client, requests)
        client.indices.refresh(index=index_name)
        return cls(elasticsearch_url, index_name, embedding)
