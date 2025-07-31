from __future__ import annotations

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class TransformersEmbeddings(BaseModel, Embeddings):
    """HuggingFace transformers embedding models.

    This replaces HuggingFaceEmbeddings by using transformers directly
    instead of sentence-transformers, avoiding the pillow dependency.

    To use, you should have the ``transformers`` and ``torch`` python packages
    installed.

    Example:
        .. code-block:: python

            from langchain_huggingface import TransformersEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            embeddings = TransformersEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    """

    model_name: str = Field(default=DEFAULT_MODEL_NAME, alias="model")
    """Model name to use."""

    cache_dir: Optional[str] = None
    """Path to store models.

    Can be also set by ``TRANSFORMERS_CACHE`` environment variable.
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the transformers model."""

    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the encode method."""

    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when encoding queries."""

    normalize_embeddings: bool = True
    """Whether to normalize embeddings to unit length."""

    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the transformers embedding model."""
        super().__init__(**kwargs)

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            msg = (
                "Could not import transformers or torch python packages. "
                "Please install them with `pip install transformers torch`."
            )
            raise ImportError(msg) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, **self.model_kwargs
        )

        self._model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, **self.model_kwargs
        )

        # Set model to evaluation mode
        self._model.eval()

        # Import torch for tensor operations
        self._torch = torch

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[
            0
        ]  # First element contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return self._torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / self._torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed a list of texts using the transformers model.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Additional keyword arguments for encoding.

        Returns:
            List of embeddings, one for each text.

        """
        # Clean texts
        texts = [x.replace("\n", " ") for x in texts]

        # Tokenize texts
        encoded_input = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        # Generate embeddings
        with self._torch.no_grad():
            model_output = self._model(**encoded_input)

        # Apply mean pooling
        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings if requested
        if self.normalize_embeddings or encode_kwargs.get(
            "normalize_embeddings", False
        ):
            sentence_embeddings = self._torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )

        return sentence_embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a transformers model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a transformers model.

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
