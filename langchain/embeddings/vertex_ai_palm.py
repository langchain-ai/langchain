"""Wrapper around OpenAI embedding models."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


class GoogleCloudVertexAIPalmEmbeddings(BaseModel, Embeddings):
    """Wrapper around Google Vertex AI's PaLM embedding models.

    To use you must have the google-cloud-aiplatform Python package installed and
    either:

    1. Have credentials configured for your environment -
        (gcloud, workload identity, etc...)
    2. Store the path to a service account JSON file as
        the GOOGLE_APPLICATION_CREDENTIALS environment variable

        Example:
            .. code-block:: python

                from langchain.embeddings import GoogleCloudVertexAIPalmEmbeddings
                embeddings = GoogleCloudVertexAIPalmEmbeddings()
                text = "This is a test query."
                query_result = embeddings.embed_query(text)
    """

    client: Any  #: :meta private:
    google_application_credentials: Optional[str]
    model_name: str = "textembedding-gecko@001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate auth and that the python package exists in environment."""
        import google.auth

        credentials, project_id = google.auth.default()
        try:
            from vertexai.preview.language_models import TextEmbeddingModel

        except ImportError:
            raise ImportError(
                "Could not import vertexai python package."
                "Try running `pip install google-cloud-aiplatform>=1.25.0`"
            )

        values["client"] = TextEmbeddingModel.from_pretrained(values["model_name"])

        return values

    def _embedding_func(self, texts: List[str]) -> List[float]:
        """Call out to Google Vertex AI PaLM Embedding Model."""
        # handle large input text
        embedding_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(self.client.get_embeddings)
        result = embedding_with_retry(texts)
        return result

    def embed_documents(
        self, texts: List[str], chunk_size: int = 5
    ) -> List[List[float]]:
        """Call out to Google Vertex AI's PaLM embedding
            endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will default to 5
                (which is currently the Vertex AI limit)

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: As of my last conversation with folks supporting the Vertex AI PaLM
        #       product, Google does not offer a tokenizer that matches
        #       the embedding model. This means we don't have a good way to
        #       approximate tokens prior to sending a request to embed a document.
        #       If Google releases a tokenizer, then we should copy the length-
        #       safe embedding function from the OpenAI embedding class.
        all_embeddings = []
        for i in range(0, len(texts), chunk_size):
            batch = texts[i : i + chunk_size]  # type:ignore
            embeddings = self._embedding_func(batch)
            list_embeddings = [emb.values for emb in embeddings]  # type:ignore
            all_embeddings.extend(list_embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Google Vertex AI's PaLM
            embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = self._embedding_func([text])
        return embedding[0].values  # type:ignore
