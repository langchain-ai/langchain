import os
from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str
from openai import OpenAI  # type: ignore


class FireworksEmbeddings(BaseModel, Embeddings):
    """FireworksEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_fireworks import FireworksEmbeddings

            model = FireworksEmbeddings(
                model='nomic-ai/nomic-embed-text-v1.5'
            )
    """

    _client: OpenAI = Field(default=None)
    fireworks_api_key: SecretStr = convert_to_secret_str("")
    model: str = "nomic-ai/nomic-embed-text-v1.5"

    @root_validator()
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment variables."""
        fireworks_api_key = convert_to_secret_str(
            values.get("fireworks_api_key") or os.getenv("FIREWORKS_API_KEY") or ""
        )
        values["fireworks_api_key"] = fireworks_api_key

        # note this sets it globally for module
        # there isn't currently a way to pass it into client
        api_key = fireworks_api_key.get_secret_value()
        values["_client"] = OpenAI(
            api_key=api_key, base_url="https://api.fireworks.ai/inference/v1"
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [
            i.embedding
            for i in self._client.embeddings.create(input=texts, model=self.model).data
        ]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
