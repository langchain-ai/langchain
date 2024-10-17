from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


class OpenSearchEmbeddings(Embeddings):
    def __init__(
        self,
        client: OpenSearch,
        model_id: str,
    ):
        self.client = client
        self.model_id = model_id

    @classmethod
    def from_connection(
        cls,
        opensearch_connection: OpenSearch,
        model_id: str,
    ) -> OpenSearchEmbeddings:
        """
        Class method to create an OpenSearchEmbeddings object
        from an OpenSearch connection.

        Args:
            opensearch_connection (OpenSearch): The OpenSearch connection.
            model_id (str): The ML model ID for generating embeddings.

        Returns:
            OpenSearchEmbeddings: An instance of the OpenSearchEmbedding class.
        """
        return cls(opensearch_connection, model_id)

    def _embedding_func(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method that sends a request to OpenSearch's text
        embedding endpoint and retrieves embeddings for the provided texts.

        Args:
            texts (List[str]): A list of strings to be embedded.

        Returns:
            List[List[float]]: A list of embeddings,
            where each embedding is a list of floats.
        """
        endpoint = f"/_plugins/_ml/_predict/text_embedding/{self.model_id}"
        body = {
            "text_docs": texts,
            "return_number": True,
            "target_response": ["sentence_embedding"],
        }

        response = self.client.transport.perform_request(
            method="POST",
            url=endpoint,
            body=json.dumps(body),
        )
        embeddings = [
            item["output"][0]["data"] for item in response["inference_results"]
        ]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document.
        """
        return self._embedding_func(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query.

        Args:
            text (str): The text query to embed.

        Returns:
            List[float]: The embedding for the query.
        """
        return self._embedding_func([text])[0]
