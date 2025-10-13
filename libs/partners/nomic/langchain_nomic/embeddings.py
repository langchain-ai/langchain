"""Nomic partner integration for LangChain."""

from __future__ import annotations

import os
from typing import Literal, overload

import nomic  # type: ignore[import]
from langchain_core.embeddings import Embeddings
from nomic import embed


class NomicEmbeddings(Embeddings):
    """NomicEmbeddings embedding model.

    Example:
        ```python
        from langchain_nomic import NomicEmbeddings

        model = NomicEmbeddings()
        ```
    """

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: Literal["remote"] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: Literal["local", "dynamic"],
        device: str | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = ...,
        dimensionality: int | None = ...,
        inference_mode: str,
        device: str | None = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: str | None = None,
        dimensionality: int | None = None,
        inference_mode: str = "remote",
        device: str | None = None,
        vision_model: str | None = None,
    ):
        """Initialize NomicEmbeddings model.

        Args:
            model: model name
            nomic_api_key: optionally, set the Nomic API key. Uses the `NOMIC_API_KEY`
                environment variable by default.
            dimensionality: The embedding dimension, for use with Matryoshka-capable
                models. Defaults to full-size.
            inference_mode: How to generate embeddings. One of `'remote'`, `'local'`
                (Embed4All), or `'dynamic'` (automatic). Defaults to `'remote'`.
            device: The device to use for local embeddings. Choices include
                `'cpu'`, `'gpu'`, `'nvidia'`, `'amd'`, or a specific device
                name. See the docstring for `GPT4All.__init__` for more info.
                Typically defaults to `'cpu'`. Do not use on macOS.
            vision_model: The vision model to use for image embeddings.

        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        if _api_key:
            nomic.login(_api_key)
        self.model = model
        self.dimensionality = dimensionality
        self.inference_mode = inference_mode
        self.device = device
        self.vision_model = vision_model

    def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: list of texts to embed
            task_type: the task type to use when embedding. One of `'search_query'`,
                `'search_document'`, `'classification'`, `'clustering'`

        """
        output = embed.text(
            texts=texts,
            model=self.model,
            task_type=task_type,
            dimensionality=self.dimensionality,
            inference_mode=self.inference_mode,
            device=self.device,
        )
        return output["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: list of texts to embed as documents

        """
        return self.embed(
            texts=texts,
            task_type="search_document",
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: query text

        """
        return self.embed(
            texts=[text],
            task_type="search_query",
        )[0]

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        """Embed images.

        Args:
            uris: list of image URIs to embed
        """
        return embed.image(
            images=uris,
            model=self.vision_model,
        )["embeddings"]
