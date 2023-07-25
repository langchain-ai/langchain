"""Wrapper around Elasticsearch vector database."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


class ElasticSearchBM25Retriever(BaseRetriever):
    """Retriever for the Elasticsearch using BM25 as a retrieval method.

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

    @classmethod
    def create(
        cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75
    ) -> ElasticSearchBM25Retriever:
        """
        Create a ElasticSearchBM25Retriever from a list of texts.

        Args:
            elasticsearch_url: URL of the Elasticsearch instance to connect to.
            index_name: Name of the index to use in Elasticsearch.
            k1: BM25 parameter k1.
            b: BM25 parameter b.

        Returns:

        """
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url)

        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
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
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(client=es, index_name=index_name)

    def add_texts(
        self,
        texts: Iterable[str],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ValueError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        for i, text in enumerate(texts):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_dict = {"query": {"match": {"content": query}}}
        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(Document(page_content=r["_source"]["content"]))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
