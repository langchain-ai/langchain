"""DocumentFilter that uses embeddings to drop documents unrelated to the query."""
from typing import Callable, Dict, List, Optional

import numpy as np
from pydantic import root_validator

from langchain.embeddings.base import Embeddings
from langchain.math_utils import cosine_similarity
from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    _RetrievedDocument,
)


class EmbeddingRelevancyDocumentFilter(BaseDocumentFilter):
    embeddings: Embeddings
    """Embeddings to use for embedding document contents and queries."""
    similarity_fn: Callable = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    k: Optional[int] = 20
    """The number of relevant documents to return. Can be set to None, in which case
    `similarity_threshold` must be specified. Defaults to 20."""
    similarity_threshold: Optional[float]
    """Threshold for determining when two documents are similar enough
    to be considered redundant. Defaults to None, must be specified if `k` is set
    to None."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_params(cls, values: Dict) -> Dict:
        """Validate similarity parameters."""
        if values["k"] is None and values["similarity_threshold"] is None:
            raise ValueError("Must specify one of `k` or `similarity_threshold`.")
        return values

    def _get_embedded_docs(self, docs: List[_RetrievedDocument]) -> List[List[float]]:
        if len(docs) and "embedded_doc" in docs[0].query_metadata:
            embedded_docs = [doc.query_metadata["embedded_doc"] for doc in docs]
        else:
            embedded_docs = self.embeddings.embed_documents(
                [d.page_content for d in docs]
            )
            for doc, embedding in zip(docs, embedded_docs):
                doc.query_metadata["embedded_doc"] = embedding
        return embedded_docs

    def filter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Filter documents based on similarity of their embeddings to the query."""
        embedded_docs = self._get_embedded_docs(docs)
        embedded_query = self.embeddings.embed_query(query)
        similarity = self.similarity_fn([embedded_query], embedded_docs)[0]
        included_idxs = np.arange(len(embedded_docs))
        if self.k is not None:
            included_idxs = np.argsort(similarity)[::-1][: self.k]
        if self.similarity_threshold is not None:
            similar_enough = np.where(
                similarity[included_idxs] > self.similarity_threshold
            )
            included_idxs = included_idxs[similar_enough]
        docs = [docs[i] for i in included_idxs]
        return docs

    async def afilter(
        self, docs: List[_RetrievedDocument], query: str
    ) -> List[_RetrievedDocument]:
        """Filter down documents."""
        raise NotImplementedError
