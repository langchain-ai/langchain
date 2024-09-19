import os
from typing import Dict, List, Optional, Set

import requests
from langchain_core.embeddings import Embeddings
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    model_validator,
)
from regex import fullmatch


class GithubEmbeddings(BaseModel, Embeddings):
    model_name: Optional[str] = None
    github_endpoint_url: str = "https://models.inference.ai.azure.com/embeddings"
    api_version: str = "2024-04-01-preview"
    github_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("GITHUB_TOKEN", ""))
    )

    SUPPORTED_MODELS: Set[str] = {
        "cohere-embed-v3-english",
        "cohere-embed-v3-multilingual",
        "text-embedding-3-large",
        "text-embedding-3-small",
    }

    @model_validator(mode="after")
    def validate_environment(self) -> Dict:
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model name. Supported models are: {self.SUPPORTED_MODELS}"
            )

        if not self.github_api_key.get_secret_value():
            raise ValueError("Github API key is required.")

        if not fullmatch(r"^\d{4}-\d{2}-\d{2}(-preview)?$", self.api_version):
            raise ValueError(
                "Invalid API version. Must be in the format YYYY-MM-DD or YYYY-MM-DD-preview"
            )

        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.github_api_key.get_secret_value()}",
        }
        data = {
            "model": "cohere-embed-v3-english",
            "input": texts,
            "encoding_format": "float",
        }
        response = requests.post(
            f"{self.github_endpoint_url}?api-version={self.api_version}",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
