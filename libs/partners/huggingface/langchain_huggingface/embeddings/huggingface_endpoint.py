import json
import os
from typing import Any, List, Optional

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
            )

            async_client = AsyncInferenceClient(
                model=self.model,
                token=huggingfacehub_api_token,
            )

            if self.task not in VALID_TASKS:
                raise ValueError(
                    f"Got invalid task {self.task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
            self.client = client
            self.async_client = async_client

        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
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
        responses = self.client.post(
            json={"inputs": texts, **_model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        _model_kwargs = self.model_kwargs or {}
        responses = await self.async_client.post(
            json={"inputs": texts, "parameters": _model_kwargs}, task=self.task
        )
        return json.loads(responses.decode())

    def embed_query(self, text: str) -> List[float]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = self.embed_documents([text])[0]
        return response

    async def aembed_query(self, text: str) -> List[float]:
        """Async Call to HuggingFaceHub's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        response = (await self.aembed_documents([text]))[0]
        return response
