from __future__ import annotations

from typing import Any, Iterator, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(texts), size):
        yield texts[i : i + size]


class JavelinAIGatewayEmbeddings(Embeddings, BaseModel):
    """Javelin AI Gateway embeddings.

    To use, you should have the ``javelin_sdk`` python package installed.
    For more information, see https://docs.getjavelin.io

    Example:
        .. code-block:: python

            from langchain_community.embeddings import JavelinAIGatewayEmbeddings

            embeddings = JavelinAIGatewayEmbeddings(
                gateway_uri="<javelin-ai-gateway-uri>",
                route="<your-javelin-gateway-embeddings-route>"
            )
    """

    client: Any
    """javelin client."""

    route: str
    """The route to use for the Javelin AI Gateway API."""

    gateway_uri: Optional[str] = None
    """The URI for the Javelin AI Gateway API."""

    javelin_api_key: Optional[str] = None
    """The API key for the Javelin AI Gateway API."""

    def __init__(self, **kwargs: Any):
        try:
            from javelin_sdk import (
                JavelinClient,
                UnauthorizedError,
            )
        except ImportError:
            raise ImportError(
                "Could not import javelin_sdk python package. "
                "Please install it with `pip install javelin_sdk`."
            )

        super().__init__(**kwargs)
        if self.gateway_uri:
            try:
                self.client = JavelinClient(
                    base_url=self.gateway_uri, api_key=self.javelin_api_key
                )
            except UnauthorizedError as e:
                raise ValueError("Javelin: Incorrect API Key.") from e

    def _query(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for txt in _chunk(texts, 20):
            try:
                resp = self.client.query_route(self.route, query_body={"input": txt})
                resp_dict = resp.dict()

                embeddings_chunk = resp_dict.get("llm_response", {}).get("data", [])
                for item in embeddings_chunk:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
            except ValueError as e:
                print("Failed to query route: " + str(e))  # noqa: T201

        return embeddings

    async def _aquery(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for txt in _chunk(texts, 20):
            try:
                resp = await self.client.aquery_route(
                    self.route, query_body={"input": txt}
                )
                resp_dict = resp.dict()

                embeddings_chunk = resp_dict.get("llm_response", {}).get("data", [])
                for item in embeddings_chunk:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
            except ValueError as e:
                print("Failed to query route: " + str(e))  # noqa: T201

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._aquery(texts)

    async def aembed_query(self, text: str) -> List[float]:
        result = await self._aquery([text])
        return result[0]
