from typing import List, Sequence

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForEmbeddingRun,
    CallbackManagerForEmbeddingRun,
)
from langchain_core.embeddings import Embeddings


class __ModuleName__Embeddings(Embeddings):
    """__ModuleName__Embeddings embedding model.

    Example:
        .. code-block:: python

            from __module_name__ import __ModuleName__Embeddings

            model = __ModuleName__Embeddings()
    """

    def _embed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[CallbackManagerForEmbeddingRun],
    ) -> List[List[float]]:
        """Embed search docs."""
        raise NotImplementedError

    def _embed_query(
        self,
        text: str,
        *,
        run_manager: CallbackManagerForEmbeddingRun,
    ) -> List[float]:
        """Embed query text."""
        raise NotImplementedError

    # only keep aembed_documents and aembed_query if they're implemented!
    # delete them otherwise to use the base class' default
    # implementation, which calls the sync version in an executor
    async def _aembed_documents(
        self,
        texts: List[str],
        *,
        run_managers: Sequence[AsyncCallbackManagerForEmbeddingRun],
    ) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def _aembed_query(
        self,
        text: str,
        *,
        run_manager: AsyncCallbackManagerForEmbeddingRun,
    ) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
