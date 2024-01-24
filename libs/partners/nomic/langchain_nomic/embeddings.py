import os
from typing import List, Optional

import nomic
from langchain_core.embeddings import Embeddings


class NomicEmbeddings(Embeddings):
    """NomicEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_nomic import NomicEmbeddings

            model = NomicEmbeddings()
    """

    def __init__(self, *, model: str, nomic_api_key: Optional[str] = None):
        """Initialize NomicEmbeddings model.

        Args:
            model: model name
        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        nomic.login(api_key=_api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        output = nomic.embed.text(
            texts=texts,
            model=self.model,
        )
        return output["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]
