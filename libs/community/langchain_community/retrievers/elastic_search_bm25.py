"""Wrapper around Elasticsearch vector database."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ElasticSearchBM25Retriever(BaseRetriever):
    """`Elasticsearch` retriever that uses `BM25`.

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
    """

    client: Any
    """Elasticsearch client."""
    index_name: str
    """Name of the index to use in Elasticsearch."""
    k: int = 4
    """Number of documents to return."""

    @classmethod
    def create(
        cls,
        elasticsearch_url: str,
        index_name: str,
        delete_if_exists: bool = False,
        k1: float = 2.0,
        b: float = 0.75,
        analyzer_type: str = "standard",
        es_params: dict = {},
    ) -> ElasticSearchBM25Retriever:
        """
        Create a ElasticSearchBM25Retriever from a list of texts.

        Args:
            elasticsearch_url: URL of the Elasticsearch instance to connect to.
            index_name: Name of the index to use in Elasticsearch.
            k1: BM25 parameter k1.
            b: BM25 parameter b.
            analyzer_type: Index analyzer (default is standard).
            es_params: Parameters to pass to the Elasticsearch client.

        Returns:

        """
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url, **es_params)

        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": f"{analyzer_type}"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",  # Use the custom BM25 similarity
                }
            }
        }

        # Create the index with the specified settings and mappings
        if delete_if_exists:
            es.indices.delete(index=index_name)
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(client=es, index_name=index_name)

    def add_texts(
        self,
        texts: Iterable[str],
        metadata: Optional[List[dict]] = None,
        refresh_indices: bool = True,
    ) -> List[str]:
        """Add texts to the index.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        metadata = metadata or [{}] * len(list(texts))
        for i, text in enumerate(texts):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "_id": _id,
                "metadata": metadata[i],
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def add_documents(
        self,
        documents: List[Document],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Add documents to the index.

        Args:
            texts: Iterable of Document to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """

        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        return self.add_texts(texts, metadata)

    def build_query_body(self, query: str) -> Dict:
        """Build query body for the search API"""

        return {"query": {"match": {"content": query}}}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_dict = self.build_query_body(query)
        res = self.client.search(
            index=self.index_name, body=query_dict, source=["content", "metadata"]
        )

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(
                Document(
                    metadata=r["_source"]["metadata"],
                    page_content=r["_source"]["content"],
                )
            )
        return docs[: self.k]
