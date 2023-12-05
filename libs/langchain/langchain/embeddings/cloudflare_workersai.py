from typing import Any, Dict, List

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra

DEFAULT_MODEL_NAME = "@cf/baai/bge-base-en-v1.5"


class CloudflareWorkersAIEmbeddings(BaseModel, Embeddings):
    """Cloudflare Workers AI embedding model.

    To use, you need to provide an API token and
    account ID to access Cloudflare Workers AI.

    Example:
        .. code-block:: python

            from langchain.embeddings import CloudflareWorkersAIEmbeddings

            account_id = "my_account_id"
            api_token = "my_secret_api_token"
            model_name =  "@cf/baai/bge-small-en-v1.5"

            cf = CloudflareWorkersAIEmbeddings(
                account_id=account_id,
                api_token=api_token,
                model_name=model_name
            )
    """

    api_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    account_id: str
    api_token: str
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 50
    strip_new_lines: bool = True
    headers: Dict[str, str] = {"Authorization": "Bearer "}

    def __init__(self, **kwargs: Any):
        """Initialize the Cloudflare Workers AI client."""
        super().__init__(**kwargs)

        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using Cloudflare Workers AI.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if self.strip_new_lines:
            texts = [text.replace("\n", " ") for text in texts]

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embeddings = []

        for batch in batches:
            response = requests.post(
                f"{self.api_base_url}/{self.account_id}/ai/run/{self.model_name}",
                headers=self.headers,
                json={"text": batch},
            )
            embeddings.extend(response.json()["result"]["data"])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using Cloudflare Workers AI.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ") if self.strip_new_lines else text
        response = requests.post(
            f"{self.api_base_url}/{self.account_id}/ai/run/{self.model_name}",
            headers=self.headers,
            json={"text": [text]},
        )
        return response.json()["result"]["data"][0]
