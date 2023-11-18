""" This file is for LLMRails Embedding """
import logging
import os
from typing import List, Optional

import requests

from langchain.pydantic_v1 import BaseModel, Extra
from langchain.schema.embeddings import Embeddings


class LLMRailsEmbeddings(BaseModel, Embeddings):
    """LLMRails embedding models.

    To use, you should have the  environment
    variable ``LLM_RAILS_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Model can be one of ["embedding-english-v1","embedding-multi-v1"]

    Example:
        .. code-block:: python

            from langchain.embeddings import LLMRailsEmbeddings
            cohere = LLMRailsEmbeddings(
                model="embedding-english-v1", api_key="my-api-key"
            )
    """

    model: str = "embedding-english-v1"
    """Model name to use."""

    api_key: Optional[str] = None
    """LLMRails API key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        api_key = self.api_key or os.environ.get("LLM_RAILS_API_KEY")
        if api_key is None:
            logging.warning("Can't find LLMRails credentials in environment.")
            raise ValueError("LLM_RAILS_API_KEY is not set")

        response = requests.post(
            "https://api.llmrails.com/v1/embeddings",
            headers={"X-API-KEY": api_key},
            json={"input": texts, "model": self.model},
            timeout=60,
        )
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Call out to Cohere's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
