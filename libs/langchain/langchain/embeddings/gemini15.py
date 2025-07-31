# libs/langchain/langchain/embeddings/gemini15.py
# Copyright (c) 2025 LangChain contributors

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env

# ------------------------------------------------------------------------------ #
# Optional dependency: google-generativeai
# ------------------------------------------------------------------------------ #
try:
    import google.generativeai as genai
except ImportError:
    from typing import Any  # تأكد أن Any متوفّر

    class _MissingGenAI:
        """Placeholder for google.generativeai when it's not installed."""

        def __getattr__(self, _name: str) -> Any:
            msg = (
                "Package 'google-generativeai' is required to use "
                "`Gemini15Embeddings`.\n"
                "Install it with:\n\n    pip install google-generativeai\n"
            )
            raise ImportError(msg)

    genai = _MissingGenAI()


class Gemini15Embeddings(Embeddings):
    """Lightweight wrapper around the *Gemini 1.5 Preview* embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "models/embedding-081",
        **genai_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        api_key
            Google AI Studio API key. If omitted, uses the 'GOOGLE_API_KEY'
            environment variable.
        model
            Gemini model identifier to call for embeddings.
        **genai_kwargs
            Extra keyword arguments forwarded to ``genai.configure``.
        """
        # احصل على المفتاح من المتعامل أو من البيئة
        self.api_key = get_from_dict_or_env(
            {"api_key": api_key}, "api_key", "GOOGLE_API_KEY"
        )
        # اضبط إعدادات المكتبة
        genai.configure(api_key=self.api_key, **genai_kwargs)
        # اسم العميل الداخلي
        self._client = genai.GenerativeModel(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of *documents*."""
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single *query*."""
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Gemini and return one vector per input text."""
        response = self._client.embed_content(input=texts)
        return response["embedding"]
