import json
import os
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
import requests
import asyncio
import aiohttp

DEFAULT_MODEL = "Cohere-embed-v3-multilingual"

class AzureMLEndpointEmbeddings(BaseModel, Embeddings):
    """Azure ML embedding endpoint for embeddings.

    To use, set up Azure ML API endpoint and provide the endpoint URL and API key.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import AzureMLEndpointEmbeddings
            azure_ml = AzureMLEndpointEmbeddings(
                embed_url="Endpoint URL from Azure ML Serverless API",
                api_key="your-api-key"
            )
    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    embed_url: Optional[str] = None
    """Azure ML endpoint URL to use for embedding."""
    api_key: Optional[str] = Field(
        default_factory=from_env("AZURE_ML_API_KEY", default=None)
    )
    """API Key to use for authentication."""

    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the embedding API."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists in environment."""
        if not self.api_key:
            self.api_key = os.getenv("AZURE_ML_API_KEY")

        if not self.api_key:
            raise ValueError("API Key must be provided or set in the environment.")

        if not self.embed_url:
            raise ValueError("Azure ML endpoint URL must be provided.")

        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call Azure ML embedding endpoint to embed documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        data = {
            "input": texts
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.embed_url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            response_data = response.json()
            embeddings = [item['embedding'] for item in response_data['data']]
            return embeddings
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async Call to Azure ML embedding endpoint to embed documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        data = {
            "input": texts
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.embed_url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    embeddings = [item['embedding'] for item in response_data['data']]
                    return embeddings
                else:
                    response_text = await response.text()
                    raise Exception(f"Error: {response.status}, {response_text}")

    def embed_query(self, text: str) -> List[float]:
        """Call Azure ML embedding endpoint to embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> List[float]:
        """Async Call to Azure ML embedding endpoint to embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = (await self.aembed_documents([text]))[0]
        return response
