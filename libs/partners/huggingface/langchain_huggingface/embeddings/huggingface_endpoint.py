from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
VALID_TASKS = ("feature-extraction",)


class HuggingFaceEndpointEmbeddings(BaseModel, Embeddings):
    """HuggingFaceHub embedding models.

    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            model = "sentence-transformers/all-mpnet-base-v2"
            hf = HuggingFaceEndpointEmbeddings(
                model=model,
                task="feature-extraction",
                huggingfacehub_api_token="my-api-key",
            )

    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: Optional[str] = None
    """Model name to use."""
    provider: Optional[str] = None
    """Name of the provider to use for inference with the model specified in
        ``repo_id``. e.g. "sambanova". if not specified, defaults to HF Inference API.
        available providers can be found in the [huggingface_hub documentation](https://huggingface.co/docs/huggingface_hub/guides/inference#supported-providers-and-tasks)."""
    repo_id: Optional[str] = None
    """Huggingfacehub repository id, for backward compatibility."""
    task: Optional[str] = "feature-extraction"
    """Task to call the model with."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""

    huggingfacehub_api_token: Optional[str] = Field(
        default_factory=from_env("HUGGINGFACEHUB_API_TOKEN", default=None)
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = self.huggingfacehub_api_token or os.getenv(
            "HF_TOKEN"
        )

        try:
            from huggingface_hub import (  # type: ignore[import]
                AsyncInferenceClient,
                InferenceClient,
            )

            if self.model:
                self.repo_id = self.model
            elif self.repo_id:
                self.model = self.repo_id
            else:
                self.model = DEFAULT_MODEL
                self.repo_id = DEFAULT_MODEL

            client = InferenceClient(
                model=self.model,
                token=huggingfacehub_api_token,
                provider=self.provider,  # type: ignore[arg-type]
            )

            async_client = AsyncInferenceClient(
                model=self.model,
                token=huggingfacehub_api_token,
                provider=self.provider,  # type: ignore[arg-type]
            )

            if self.task not in VALID_TASKS:
                msg = (
                    f"Got invalid task {self.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
                raise ValueError(msg)
            self.client = client
            self.async_client = async_client

        except ImportError as e:
            msg = (
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
            raise ImportError(msg) from e
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        responses = self.client.feature_extraction(text=texts, **_model_kwargs)
        return responses.tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = await self.async_client.feature_extraction(
            text=texts, **_model_kwargs
        )
        return responses.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> list[float]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        return (await self.aembed_documents([text]))[0]
