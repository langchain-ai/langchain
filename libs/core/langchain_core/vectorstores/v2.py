from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence, Union

from langchain_core.indexes.base import Index
from langchain_core.retrievers.v2 import RetrievalResponse, RetrieverV2


class VectorStoreV2(Index, RetrieverV2[Union[str, Sequence[float]]]):
    @abstractmethod
    def _retrieve(
        self,
        query: Union[str, Sequence[float]],
        *,
        method: str = "similarity",
        metric: str = "cosine_similarity",
        **kwargs: Any,
    ) -> RetrievalResponse:
        """Retrieve documents by query"""

    async def _aretrieve(
        self,
        query: Union[str, Sequence[float]],
        *,
        method: str = "similarity",
        metric: str = "cosine_similarity",
        **kwargs: Any,
    ) -> RetrievalResponse:
        """Retrieve documents by query"""
        return await super()._aretrieve(query, method=method, metric=metric, **kwargs)
