"""OpenRouter embeddings."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self


class OpenRouterEmbeddings(BaseModel, Embeddings):
    """OpenRouter embedding models via the OpenAI-compatible embeddings API."""

    model: str = Field(default="openai/text-embedding-3-small")
    openrouter_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    openrouter_api_base: str = Field(
        default_factory=from_env(
            "OPENROUTER_API_BASE",
            default="https://openrouter.ai/api/v1",
        ),
        alias="base_url",
    )
    request_timeout: float | None = Field(default=60.0, alias="timeout")

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if not (self.openrouter_api_key and self.openrouter_api_key.get_secret_value()):
            msg = "OPENROUTER_API_KEY must be set."
            raise ValueError(msg)
        return self

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.openrouter_api_key.get_secret_value()}",  # type: ignore[union-attr]
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        with httpx.Client(
            base_url=self.openrouter_api_base,
            headers=self._headers(),
            timeout=self.request_timeout,
        ) as client:
            response = client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        async with httpx.AsyncClient(
            base_url=self.openrouter_api_base,
            headers=self._headers(),
            timeout=self.request_timeout,
        ) as client:
            response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]
