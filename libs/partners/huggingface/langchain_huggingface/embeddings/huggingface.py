from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from langchain_huggingface.utils.import_utils import (
    IMPORT_ERROR,
    is_ipex_available,
    is_optimum_intel_available,
    is_optimum_intel_version,
)

_MIN_OPTIMUM_VERSION = "1.22"


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
    batch_size: int = 32
    """Batch size passed to `encode()`. Larger values improve GPU/MPS throughput
    at the cost of memory. If `batch_size` is also present in `encode_kwargs` or
    `query_encode_kwargs`, those values take precedence over this field."""

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
            if not is_optimum_intel_available() or not is_ipex_available():
                msg = f"Backend: ipex {IMPORT_ERROR.format('optimum[ipex]')}"
                raise ImportError(msg)

            if is_optimum_intel_version("<", _MIN_OPTIMUM_VERSION):
                msg = (
                    f"Backend: ipex requires optimum-intel>="
                    f"{_MIN_OPTIMUM_VERSION}. You can install it with pip: "
                    "`pip install --upgrade --upgrade-strategy eager "
                    "`optimum[ipex]`."
                )
                raise ImportError(msg)

            from optimum.intel import IPEXSentenceTransformer  # type: ignore[import]

            model_cls = IPEXSentenceTransformer

        else:
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
        """Embed a list of texts using the HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.
            encode_kwargs: Keyword arguments to pass when calling the `encode`
                method of the SentenceTransformer model. Values here take
                precedence over `batch_size` and the `convert_to_tensor` default.

        Returns:
            List of embeddings, one for each text.

        """
        import sentence_transformers  # type: ignore[import]

        texts = [x.replace("\n", " ") for x in texts]

        # Default to convert_to_tensor=True so all micro-batch outputs stay on
        # the model's device (MPS/CUDA) and are torch.cat'd there. This avoids
        # one device→CPU memory transfer per micro-batch; instead there is
        # exactly one transfer at the very end. encode_kwargs can override this.
        effective_encode_kwargs = {
            "batch_size": self.batch_size,
            "convert_to_tensor": True,
            **encode_kwargs,
        }

        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                texts,
                show_progress_bar=self.show_progress,
                **effective_encode_kwargs,
            )

        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)

        # torch.Tensor path (convert_to_tensor=True default or user override):
        # single device→CPU transfer, then numpy's C-optimized tolist().
        if hasattr(embeddings, "cpu"):
            return embeddings.cpu().numpy().tolist()

        # numpy array path (user set convert_to_tensor=False in encode_kwargs):
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
