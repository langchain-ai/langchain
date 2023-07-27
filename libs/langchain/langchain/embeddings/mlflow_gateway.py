from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence

from pydantic import BaseModel

from langchain.callbacks.manager import CallbackManagerForEmbeddingsRun
from langchain.embeddings.base import Embeddings


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class MlflowAIGatewayEmbeddings(Embeddings, BaseModel):
    """
    Wrapper around embeddings LLMs in the MLflow AI Gateway.

    To use, you should have the ``mlflow[gateway]`` python package installed.
    For more information, see https://mlflow.org/docs/latest/gateway/index.html.

    Example:
        .. code-block:: python

            from langchain.embeddings import MlflowAIGatewayEmbeddings

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
            embeddings.append(resp["embeddings"])
        return embeddings

    def _embed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[CallbackManagerForEmbeddingsRun],
    ) -> List[List[float]]:
        return self._query(texts)

    def _embed_query(
        self,
        text: str,
        *,
        run_manager: CallbackManagerForEmbeddingsRun,
    ) -> List[float]:
        return self._query([text])[0]
