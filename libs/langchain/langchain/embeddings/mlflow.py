from __future__ import annotations

from typing import Any, Iterator, List
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MlflowEmbeddings(Embeddings, BaseModel):
    """Wrapper around embeddings LLMs in MLflow.

    To use, you should have the `mlflow[genai]` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/server.html.

    Example:
        .. code-block:: python

            from langchain.embeddings import MlflowEmbeddings

            embeddings = MlflowEmbeddings(
                target_uri="http://localhost:5000",
                endpoint="embeddings",
            )
    """

    endpoint: str
    """The endpoint to use."""
    target_uri: str
    """The target URI to use."""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate_uri()
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                f"Please run `pip install mlflow{self._mlflow_extras}` to install "
                "required dependencies."
            ) from e

    @property
    def _mlflow_extras(self) -> str:
        return "[genai]"

    def _validate_uri(self) -> None:
        if self.target_uri == "databricks":
            return
        allowed = ["http", "https", "databricks"]
        if urlparse(self.target_uri).scheme not in allowed:
            raise ValueError(
                f"Invalid target URI: {self.target_uri}. "
                f"The scheme must be one of {allowed}."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for txt in _chunk(texts, 20):
            resp = self._client.predict(endpoint=self.endpoint, inputs={"input": txt})
            embeddings.extend(r["embedding"] for r in resp["data"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
