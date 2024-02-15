"""Wrapper around Bookend AI embedding models."""

import json
from typing import Any, List

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field

API_URL = "https://api.bookend.ai/"
DEFAULT_TASK = "embeddings"
PATH = "/models/predict"


class BookendEmbeddings(BaseModel, Embeddings):
    """Bookend AI sentence_transformers embedding models.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import BookendEmbeddings

            bookend = BookendEmbeddings(
                domain={domain}
                api_token={api_token}
                model_id={model_id}
            )
            bookend.embed_documents([
                "Please put on these earmuffs because I can't you hear.",
                "Baby wipes are made of chocolate stardust.",
            ])
            bookend.embed_query(
                "She only paints with bold colors; she does not like pastels."
            )
    """

    domain: str
    """Request for a domain at https://bookend.ai/ to use this embeddings module."""
    api_token: str
    """Request for an API token at https://bookend.ai/ to use this embeddings module."""
    model_id: str
    """Embeddings model ID to use."""
    auth_header: dict = Field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.auth_header = {"Authorization": "Basic {}".format(self.api_token)}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a Bookend deployed embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        result = []
        headers = self.auth_header
        headers["Content-Type"] = "application/json; charset=utf-8"
        params = {
            "model_id": self.model_id,
            "task": DEFAULT_TASK,
        }

        for text in texts:
            data = json.dumps(
                {"text": text, "question": None, "context": None, "instruction": None}
            )
            r = requests.request(
                "POST",
                API_URL + self.domain + PATH,
                headers=headers,
                params=params,
                data=data,
            )
            result.append(r.json()[0]["data"])

        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a Bookend deployed embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
