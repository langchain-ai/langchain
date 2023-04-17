""""""
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain.document_filter.base import PipelineFilter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

SIMILARITY_FN_TYPE = Callable[[List[List[float]], List[List[float]]], List[List[float]]]


class EmbeddingRedundantDocumentFilter(PipelineFilter):
    """Filter that drops redundant documents."""

    embeddings: Embeddings
    """"""
    similarity_fn: SIMILARITY_FN_TYPE = cosine_similarity
    """"""
    similarity_threshold: float = 0.95
    """"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _filter_embeddings(self, embedded_docs: List[List[float]]) -> List[int]:
        similarity = np.tril(self.similarity_fn(embedded_docs, embedded_docs), k=-1)
        redundant = np.where(similarity > self.similarity_threshold)
        redundant_stacked = np.column_stack(redundant)
        redundant_sorted = np.argsort(similarity[redundant])[::-1]
        included_idxs = set(range(len(embedded_docs)))
        for first_idx, second_idx in redundant_stacked[redundant_sorted]:
            if first_idx in included_idxs and second_idx in included_idxs:
                included_idxs.remove(second_idx)
        return list(sorted(included_idxs))

    def filter_pipeline(
        self, docs: List[Document], query: str, **kwargs: Any
    ) -> Tuple[List[Document], Dict]:
        if "embedded_docs" in kwargs:
            embedded_docs = kwargs["embedded_docs"]
        else:
            embedded_docs = self.embeddings.embed_documents(
                [d.page_content for d in docs]
            )
        included_idxs = self._filter_embeddings(embedded_docs)
        docs = [docs[i] for i in sorted(included_idxs)]
        extra_info = {"embedded_docs": [embedded_docs[i] for i in included_idxs]}
        return docs, extra_info
