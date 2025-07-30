# libs/langchain/langchain/embeddings/gemini15.py
# Copyright (c) 2025 LangChain contributors

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env

# --------------------------------------------------------------------------- #
# Optional dependency: google-generativeai
# --------------------------------------------------------------------------- #
try:
    import google.generativeai as genai  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover
    msg = (
        "Package `google-generativeai` is required to use `Gemini15Embeddings`.\n"
        "Install it with:\n\n    pip install google-generativeai\n"
    )
    raise ImportError(msg) from e


class Gemini15Embeddings(Embeddings):
    """Lightweight wrapper around the *Gemini 1.5 Preview* embeddings endpoint."""

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "models/embedding-001",
        **genai_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        api_key
            Google AI Studio API key.  If omitted, the value of the
            ``GOOGLE_API_KEY`` environment variable is used.
        model
            Gemini model identifier to call for embeddings.
        **genai_kwargs
            Extra keyword arguments forwarded to ``genai.configure``.
        """
        self.api_key: str = get_from_dict_or_env(
            {"api_key": api_key}, "api_key", "GOOGLE_API_KEY"
        )
        genai.configure(api_key=self.api_key, **genai_kwargs)
        self._client = genai.GenerativeModel(model)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of *documents*."""
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single *query*."""
        return self._embed_batch([text])[0]

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Gemini and return one vector per input text."""
        # NOTE: GenerativeModel.embed_content returns a dict like
        #       {"embedding": [[...], [...], ...]}
        response = self._client.embed_content(input=texts)  # type: ignore[attr-defined]
        return response["embedding"]
