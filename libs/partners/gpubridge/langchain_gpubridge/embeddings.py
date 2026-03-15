"""GPU-Bridge Embeddings integration for LangChain."""

import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr

GPUBRIDGE_API_URL = "https://api.gpubridge.xyz/run"
GPUBRIDGE_BASE_URL = "https://api.gpubridge.xyz"


class GPUBridgeEmbeddings(BaseModel, Embeddings):
    """GPU-Bridge text embeddings.

    GPU-Bridge exposes high-throughput embedding inference at ~$0.00002/call.
    Supports multilingual and code embeddings.

    Setup:
        Install with: pip install langchain-gpubridge

        .. code-block:: python

            from langchain_gpubridge import GPUBridgeEmbeddings

            embeddings = GPUBridgeEmbeddings(api_key="gpub_...")

    Key init args:
        api_key: GPU-Bridge API key.
        service: Embedding service. Default ``embedding-l4``.
        batch_size: Number of texts per API call. Default 32.
        base_url: GPU-Bridge API base URL.
    """

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="GPU-Bridge API key (starts with gpub_).",
    )
    service: str = Field(
        default="embedding-l4",
        description="GPU-Bridge embedding service.",
    )
    batch_size: int = Field(
        default=32,
        description="Number of texts to embed per request.",
    )
    base_url: str = Field(
        default=GPUBRIDGE_API_URL,
        description="GPU-Bridge API endpoint.",
    )

    class Config:
        arbitrary_types_allowed = True

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def _poll_job(self, status_url: str) -> Dict[str, Any]:
        """Poll a GPU-Bridge async job until completion."""
        for _ in range(30):
            time.sleep(1)
            response = requests.get(
                f"{GPUBRIDGE_BASE_URL}{status_url}",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "completed":
                return data
            if data.get("status") == "failed":
                raise ValueError(f"GPU-Bridge job failed: {data}")
        raise TimeoutError("GPU-Bridge embedding job timed out")

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text, handling async jobs."""
        payload: Dict[str, Any] = {
            "service": self.service,
            "input": {"texts": [text]},
        }
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self._get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(f"GPU-Bridge error: {data['error']}")

        # Handle async job
        if data.get("status") == "pending" and "status_url" in data:
            data = self._poll_job(data["status_url"])

        output = data.get("output", {})
        # API returns 'embedding' (singular) for single text
        return output.get("embedding", output.get("embeddings", [[]])[0] if output.get("embeddings") else [])

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        return [self._embed_single(text) for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self._embed([text])
        return result[0]
