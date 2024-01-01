from typing import List

from langchain_core.embeddings import Embeddings


class AI21Embeddings(Embeddings):
    """AI21Embeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_ai21 import AI21Embeddings

            model = AI21Embeddings()
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        raise NotImplementedError

    # only keep aembed_documents and aembed_query if they're implemented!
    # delete them otherwise to use the base class' default
    # implementation, which calls the sync version in an executor
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
