from __future__ import annotations

from typing import Any, Iterator, List

from langchain_core.pydantic_v1 import BaseModel, PrivateAttr

from langchain.embeddings.base import Embeddings


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class DatabricksEmbeddings(Embeddings, BaseModel):
    """Wrapper around embeddings LLMs in Databricks.

    To use, you should have the ``mlflow`` python package installed.
    For more information, see https://mlflow.org/docs/latest/llms/deployments/databricks.html.

    Example:
        .. code-block:: python

            from langchain.embeddings import DatabricksEmbeddings

            embeddings = DatabricksEmbeddings(
                target_uri="databricks",
                endpoint="embeddings",
            )
    """

    endpoint: str
    """The endpoint to use."""
    target_uri: str
    """The target URI to use. For example, ``"""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        try:
            from mlflow.deployments import get_deploy_client

            self._client = get_deploy_client(self.target_uri)
        except ImportError as e:
            raise ImportError(
                "Failed to create the client. "
                "Please install mlflow with `pip install mlflow`."
            ) from e

    def _query(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for txt in _chunk(texts, 20):
            resp = self._client.predict(endpoint=self.endpoint, inputs={"input": txt})
            embeddings.extend(r["embedding"] for r in resp["data"])
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]
