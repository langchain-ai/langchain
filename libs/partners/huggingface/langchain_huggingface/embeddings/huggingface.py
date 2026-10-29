from __future__ import annotations

import warnings
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field


def _get_accelerator_devices() -> list[str] | None:
    """Return a list of available accelerator device strings for multi-process pools.

    Uses ``torch.accelerator`` (PyTorch ≥ 2.6) when available for a fully
    device-agnostic result.  Falls back to checking CUDA, then XPU individually
    so that Intel XPU machines running PyTorch 2.5 are handled correctly.

    Returns ``None`` when no accelerator is found so that
    ``start_multi_process_pool`` keeps its own default behaviour (CPU workers).
    """
    try:
        import torch  # type: ignore[import]
    except ImportError:
        return None

    if hasattr(torch, "accelerator"):
        count = torch.accelerator.device_count()
        if count > 0:
            device_type = torch.accelerator.current_accelerator().type
            return [f"{device_type}:{i}" for i in range(count)]
    else:
        cuda_count = torch.cuda.device_count()
        if cuda_count > 0:
            return [f"cuda:{i}" for i in range(cuda_count)]
        if hasattr(torch, "xpu") and torch.xpu.device_count() > 0:
            return [f"xpu:{i}" for i in range(torch.xpu.device_count())]

    return None


class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the `sentence_transformers` python package installed.

    Example:
        ```python
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        ```
    """

    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="model"
    )
    """Model name to use."""
    cache_folder: str | None = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            msg = (
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
            raise ImportError(msg) from exc

        if self.model_kwargs.get("backend", "torch") == "ipex":
            msg = (
                "`backend='ipex'` is no longer supported; the default `torch` "
                "backend will be used instead. Intel GPU acceleration is "
                "natively supported in PyTorch 2.5 and later."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            self.model_kwargs["backend"] = "torch"

        model_cls = sentence_transformers.SentenceTransformer

        self._client = model_cls(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def _embed(
        self, texts: list[str], encode_kwargs: dict[str, Any]
    ) -> list[list[float]]:
        """Embed a text using the HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the
                `encode` method for the documents of the SentenceTransformer
                encode method.

        Returns:
            List of embeddings, one for each text.

        """
        texts = [x.replace("\n", " ") for x in texts]
        if self.multi_process:
            # Build a device list that covers CUDA, XPU, MPS, and NPU so that
            # non-CUDA accelerators are used instead of falling back to CPU
            # workers (the sentence-transformers default only checks CUDA/NPU).
            devices = _get_accelerator_devices()
            pool = self._client.start_multi_process_pool(
                **({"target_devices": devices} if devices is not None else {})
            )
            try:
                # Use encode(pool=...) rather than the deprecated
                # encode_multi_process(), and forward encode_kwargs and
                # show_progress so callers get consistent behaviour in both
                # single- and multi-process modes.
                embeddings = self._client.encode(
                    texts,
                    pool=pool,
                    show_progress_bar=self.show_progress,
                    **encode_kwargs,
                )
            finally:
                # Always stop the pool, even if encoding raises, to prevent
                # orphaned worker processes.
                pool.stop_workers()
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )

        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)

        return embeddings.tolist()  # type: ignore[return-type]

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
