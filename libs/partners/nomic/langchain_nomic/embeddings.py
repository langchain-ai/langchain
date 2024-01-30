import os
from typing import List, Optional

import nomic  # type: ignore
from nomic import embed
from langchain_core.embeddings import Embeddings


class NomicEmbeddings(Embeddings):
    """NomicEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_nomic import NomicEmbeddings

            model = NomicEmbeddings()
    """

    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: Optional[str] = None,
        task_type: Optional[str] = None
    ):
        """Initialize NomicEmbeddings model.

        Args:
            model: model name
            task_type: optionally, lock the model to a specific task type. By default
                the model uses "search_query" when calling `embed_query` and
                "search_document" for `embed_documents`.
            nomic_api_key: optionally, set the Nomic API key. Uses the NOMIC_API_KEY
                environment variable by default.
        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        nomic.login(_api_key)
        self.model = model
        self.task_type = task_type

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        output = embed.text(
            texts=texts,
            model=self.model,
            task_type=self.task_type
            if self.task_type is not None
            else "search_document",
        )
        return output["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        output = embed.text(
            texts=[text],
            model=self.model,
            task_type=self.task_type if self.task_type is not None else "search_query",
        )
        return output["embeddings"][0]
