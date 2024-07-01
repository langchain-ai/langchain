from abc import ABC, abstractmethod
from typing import Dict, List

from scipy.sparse import csr_array  # type: ignore


class BaseSparseEmbedding(ABC):
    """Interface for Sparse embedding models.

    You can inherit from it and implement your custom sparse embedding model.
    """

    @abstractmethod
    def embed_query(self, query: str) -> Dict[int, float]:
        """Embed query text."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        """Embed search docs."""


class BM25SparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on BM25.

    This class uses the BM25 model in Milvus model to implement sparse vector embedding.
    This model requires pymilvus[model] to be installed.
    `pip install pymilvus[model]`
    For more information please refer to:
    https://milvus.io/docs/embed-with-bm25.md
    """

    def __init__(self, corpus: List[str], language: str = "en"):
        from pymilvus.model.sparse import BM25EmbeddingFunction  # type: ignore
        from pymilvus.model.sparse.bm25.tokenizers import (  # type: ignore
            build_default_analyzer,
        )

        self.analyzer = build_default_analyzer(language=language)
        self.bm25_ef = BM25EmbeddingFunction(self.analyzer, num_workers=1)
        self.bm25_ef.fit(corpus)

    def embed_query(self, text: str) -> Dict[int, float]:
        return self._sparse_to_dict(self.bm25_ef.encode_queries([text]))

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        sparse_arrays = self.bm25_ef.encode_documents(texts)
        return [self._sparse_to_dict(sparse_array) for sparse_array in sparse_arrays]

    def _sparse_to_dict(self, sparse_array: csr_array) -> Dict[int, float]:
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        result_dict = {}
        for col_index, value in zip(col_indices, non_zero_values):
            result_dict[col_index] = value
        return result_dict
