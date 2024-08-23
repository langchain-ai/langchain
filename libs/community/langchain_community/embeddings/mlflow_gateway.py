from __future__ import annotations

import warnings
from typing import Any, Iterator, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MlflowAIGatewayEmbeddings(Embeddings, BaseModel):
    """MLflow AI Gateway embeddings.

    To use, you should have the ``mlflow[gateway]`` python package installed.
    For more information, see https://mlflow.org/docs/latest/gateway/index.html.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import MlflowAIGatewayEmbeddings

            embeddings = MlflowAIGatewayEmbeddings(
                gateway_uri="<your-mlflow-ai-gateway-uri>",
                route="<your-mlflow-ai-gateway-embeddings-route>"
            )
    """

    route: str
    """The route to use for the MLflow AI Gateway API."""
    gateway_uri: Optional[str] = None
    """The URI for the MLflow AI Gateway API."""

    def __init__(self, **kwargs: Any):
        warnings.warn(
            "`MlflowAIGatewayEmbeddings` is deprecated. Use `MlflowEmbeddings` or "
            "`DatabricksEmbeddings` instead.",
            DeprecationWarning,
        )
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        super().__init__(**kwargs)
        if self.gateway_uri:
            mlflow.gateway.set_gateway_uri(self.gateway_uri)

    def _query(self, texts: List[str]) -> List[List[float]]:
        try:
            import mlflow.gateway
        except ImportError as e:
            raise ImportError(
                "Could not import `mlflow.gateway` module. "
                "Please install it with `pip install mlflow[gateway]`."
            ) from e

        embeddings = []
        for txt in _chunk(texts, 20):
            resp = mlflow.gateway.query(self.route, data={"text": txt})
            # response is List[List[float]]
            if isinstance(resp["embeddings"][0], List):
                embeddings.extend(resp["embeddings"])
            # response is List[float]
            else:
                embeddings.append(resp["embeddings"])
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]
