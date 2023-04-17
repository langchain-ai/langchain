""""""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import root_validator
from sklearn.metrics.pairwise import cosine_similarity

from langchain.document_filter.base import PipelineFilter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

SIMILARITY_FN_TYPE = Callable[[List[List[float]], List[List[float]]], List[List[float]]]


class EmbeddingRelevancyDocumentFilter(PipelineFilter):
    embeddings: Embeddings
    """"""
    similarity_fn: SIMILARITY_FN_TYPE = cosine_similarity
    """"""
    top_k: Optional[int] = 20
    """"""
    similarity_threshold: Optional[float] = None
    """"""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_params(cls, values: Dict) -> Dict:
        """Validate similarity parameters."""
        if values["top_k"] is None and values["similarity_threshold"] is None:
            raise ValueError("Must specify one of `top_k` or `similarity_threshold`.")
        return values

    def filter_pipeline(
        self, docs: List[Document], query: str, **kwargs: Any
    ) -> Tuple[List[Document], Dict]:
        """"""
        if self.top_k is not None and len(docs) <= self.top_k:
            return docs, kwargs
        if "embedded_docs" in kwargs:
            embedded_docs = kwargs["embedded_docs"]
        else:
            embedded_docs = self.embeddings.embed_documents(
                [d.page_content for d in docs]
            )
        if "embedded_query" in kwargs:
            embedded_query = kwargs["embedded_query"]
        else:
            embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_docs)[0]
        included_idxs = np.arange(len(embedded_docs))
        if self.top_k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.top_k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        docs = [docs[i] for i in included_idxs]
        embedded_docs = [embedded_docs[i] for i in included_idxs]
        extra_dict = {"embedded_docs": embedded_docs, "embedded_query": embedded_query}
        return docs, extra_dict
