import json
from typing import List

import requests

from langchain.schema.embeddings import Embeddings
from langchain.utils.iter import batch_iterate

DEFAULT_BATCH_SIZE = 6


class VoyageEmbeddings(Embeddings):
    """Voyage AI text embedding model wrapper."""

    def __init__(
        self, url: str, model: str, *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """Initialize model.

        Args:
            url: Model URL.
            model: Model name.
            batch_size: Batch size for embedding documents.
        """
        self.url = url
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for batch in batch_iterate(self.batch_size, texts):
            data = json.dumps({"input": batch, "model": self.model})
            response = requests.post(
                self.url, headers={"Content-Type": "application/json"}, data=data
            )
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"Received status code {response.status_code} and response "
                    f"{response.text}"
                )
            response_data = response.json()["data"]
            embeddings.extend([x["embedding"] for x in response_data])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
