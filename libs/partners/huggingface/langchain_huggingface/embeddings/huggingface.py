from __future__ import annotations

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace transformer embedding models using Inference API.

    This implementation uses the HuggingFace Inference API for generating embeddings,
    removing the need for heavy dependencies like torch, sentence-transformers, or
    pillow.

    Example:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            encode_kwargs = {"normalize_embeddings": False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs=encode_kwargs
            )

    """

    model_name: str = Field(default=DEFAULT_MODEL_NAME, alias="model")
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. (Not used with Inference API)"""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the encode method."""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the encode method for queries."""
    multi_process: bool = False
    """Not supported with Inference API."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the HuggingFace Inference API client."""
        super().__init__(**kwargs)

        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            msg = (
                "Could not import huggingface_hub. "
                "Please install it with `pip install huggingface-hub`."
            )
            raise ImportError(msg) from exc

        self._client = InferenceClient()

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed texts using HuggingFace Inference API.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the
                encode method.

        Returns:
            List of embeddings, one for each text.

        """
        # Clean texts
        texts = [x.replace("\n", " ") for x in texts]

        try:
            # Process texts one by one as the API expects individual strings
            embeddings = []
            for text in texts:
                response = self._client.feature_extraction(text, model=self.model_name)
                list_response: list[float] = response.tolist()
                embeddings.append(list_response)
            return embeddings
        except Exception as e:
            msg = f"Failed to get embeddings from HuggingFace API: {e}"
            raise RuntimeError(msg) from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using HuggingFace Inference API.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using HuggingFace Inference API.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed([text], embed_kwargs)[0]
