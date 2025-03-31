from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field, model_validator

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env

class SiliconFlowEmbeddings(BaseModel, Embeddings):
    """
    SiliconFlow embedding model integration.

    Setup:
        To use, you should set the environment variable ``SILICONFLOW_API_KEY`` with your API key.

        .. code-block:: bash

            export SILICONFLOW_API_KEY="your-api-key"

    Key init args:
        model: str
            Name of the embedding model to use. Default is "BAAI/bge-m3".
        api_key: str
            Automatically inferred from env var `SILICONFLOW_API_KEY` if not provided.
        encoding_format: str
            The encoding format for embeddings. Default is "float". Available options: "float", "base64".

    Instantiate:
        .. code-block:: python

            from your_module import SiliconFlowEmbeddings

            embed = SiliconFlowEmbeddings(
                model="BAAI/bge-m3",
                # api_key="...",  # Optional, will use env var if not provided
            )

    Embed single text:
        .. code-block:: python

            input_text = "Silicon flow embedding online: fast, affordable, and high-quality embedding services."
            embed.embed_query(input_text)

        .. code-block:: python

            [0.003832892, -0.049372625, 0.035413884, ...]

    Embed multiple texts:
        .. code-block:: python

            input_texts = ["This is a test query1.", "This is a test query2."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            [
                [0.0083934665, 0.037985895, -0.06684559, ...],
                [-0.02713102, -0.005470169, 0.032321047, ...]
            ]
    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = Field(default="BAAI/bge-m3")
    """Model name"""
    api_key: str
    """Automatically inferred from env var `SILICONFLOW_API_KEY` if not provided."""
    encoding_format: str = Field(default="float")
    """The encoding format for embeddings."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that API key exists in environment or provided explicitly."""
        values["api_key"] = get_from_dict_or_env(
            values, "api_key", "SILICONFLOW_API_KEY"
        )
        return values

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to send embedding requests to SiliconFlow API.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embeddings for each document in the input list.
        """
        url = "https://api.siliconflow.cn/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": self.encoding_format
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Error in embedding request: {response.text}")

        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single text.

        Args:
            text: A text to embed.

        Returns:
            The embedding of the input text as a list of floats.
        """
        return self._embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embeddings for each document in the input list.
        """
        return self._embed(texts)
