"""This file is for LLMRails Embedding"""

from typing import Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init


class LLMRailsEmbeddings(BaseModel, Embeddings):
    """LLMRails embedding models.

    To use, you should have the  environment
    variable ``LLM_RAILS_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Model can be one of ["embedding-english-v1","embedding-multi-v1"]

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LLMRailsEmbeddings
            cohere = LLMRailsEmbeddings(
                model="embedding-english-v1", api_key="my-api-key"
            )
    """

    model: str = "embedding-english-v1"
    """Model name to use."""

    api_key: Optional[SecretStr] = None
    """LLMRails API key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "api_key", "LLM_RAILS_API_KEY")
        )
        values["api_key"] = api_key
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Cohere's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        response = requests.post(
            "https://api.llmrails.com/v1/embeddings",
            headers={"X-API-KEY": self.api_key.get_secret_value()},  # type: ignore[union-attr]
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
