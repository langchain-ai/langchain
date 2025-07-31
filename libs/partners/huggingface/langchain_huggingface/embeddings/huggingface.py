from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_MIN_OPTIMUM_VERSION = "1.22"


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace transformer embedding models.

    To use, you should have the ``transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

    """

    model_name: str = Field(default=DEFAULT_MODEL_NAME, alias="model")
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by TRANSFORMERS_CACHE environment variable."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the transformer model, such as `device`,
    `revision`, `trust_remote_code`, or `token`."""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents,
    such as `batch_size`, `normalize_embeddings`, and more."""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query,
    such as `batch_size`, `normalize_embeddings`, and more."""
    multi_process: bool = False
    """Run encode() on multiple GPUs. Note: This feature is not supported with
    transformers and will be ignored with a warning."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the transformer model."""
        super().__init__(**kwargs)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            msg = (
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
            raise ImportError(msg) from exc

        # Extract device from model_kwargs
        self.device = self.model_kwargs.get("device", "cpu")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Remove device from model_kwargs as it's not a valid argument for
        # from_pretrained
        model_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "device"}

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_folder, **model_kwargs
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.cache_folder, **model_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        # Warn about multi-process not being supported
        if self.multi_process:
            import warnings

            warnings.warn(
                "Multi-process encoding is not supported with the transformers "
                "implementation. This parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def _mean_pooling(
        self, model_output: Any, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to model output."""
        # Extract token embeddings from model output
        token_embeddings = model_output[0]

        # Expand attention mask for broadcasting
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Apply mean pooling
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,  # noqa: FBT001
        normalize_embeddings: bool = False,  # noqa: FBT001, FBT002
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: The list of texts to embed.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show progress bar.
            normalize_embeddings: Whether to normalize embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            Array of embeddings.

        """
        if show_progress_bar is None:
            show_progress_bar = self.show_progress

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize texts
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                **kwargs,
            )

            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Apply mean pooling
            embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            # Normalize if requested
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        return np.vstack(all_embeddings)

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed a text using the HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the
                encode method.

        Returns:
            List of embeddings, one for each text.

        """
        texts = [x.replace("\n", " ") for x in texts]
        embeddings = self.encode(
            texts,
            show_progress_bar=self.show_progress,
            **encode_kwargs,
        )
        return embeddings.tolist()  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        return self._embed(texts, self.encode_kwargs)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a HuggingFace transformer model.

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
