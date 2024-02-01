import os
from typing import List, Optional

import nomic  # type: ignore
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
    ):
        """Initialize NomicEmbeddings model.

        Args:
            model: model name
            nomic_api_key: optionally, set the Nomic API key. Uses the NOMIC_API_KEY
                environment variable by default.
        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        if _api_key:
            nomic.login(_api_key)
        self.model = model

    def embed(self, texts: List[str], *, task_type: str) -> List[List[float]]:
        """Embed texts.

        Args:
            texts: list of texts to embed
            task_type: the task type to use when embedding. One of `search_query`,
                `search_document`, `classification`, `clustering`
        """
        # TODO: do this via nomic.embed when fixed in nomic sdk
        from nomic import embed

        output = embed.text(texts=texts, model=self.model, task_type=task_type)
        return output["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: list of texts to embed as documents
        """
        return self.embed(
            texts=texts,
            task_type="search_document",
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: query text
        """
        return self.embed(
            texts=[text],
            task_type="search_query",
        )[0]
