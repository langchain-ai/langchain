"""DocumentFilter that uses embeddings to drop redundant documents."""
from typing import Callable, List

import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.math_utils import cosine_similarity
from langchain.retrievers.document_filters.base import (
    BaseDocumentFilter,
    RetrievedDocument,
)


class EmbeddingRedundantDocumentFilter(BaseDocumentFilter):
    """Filter that drops redundant documents."""

    embeddings: Embeddings
    """Embeddings to use for embedding document contents."""
    similarity_fn: Callable = cosine_similarity
    """Similarity function for comparing documents. Function expected to take as input
    two matrices (List[List[float]]) and return a matrix of scores where higher values
    indicate greater similarity."""
    similarity_threshold: float = 0.95
    """Threshold for determining when two documents are similar enough
    to be considered redundant."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _filter_embeddings(self, embedded_docs: List[List[float]]) -> List[int]:
        """Filter redundant documents based on the similarity of their embeddings."""
        similarity = np.tril(self.similarity_fn(embedded_docs, embedded_docs), k=-1)
        redundant = np.where(similarity > self.similarity_threshold)
        redundant_stacked = np.column_stack(redundant)
        redundant_sorted = np.argsort(similarity[redundant])[::-1]
        included_idxs = set(range(len(embedded_docs)))
        for first_idx, second_idx in redundant_stacked[redundant_sorted]:
            if first_idx in included_idxs and second_idx in included_idxs:
                # Default to dropping the second document of any highly similar pair.
                included_idxs.remove(second_idx)
        return list(sorted(included_idxs))

    def _get_embedded_docs(self, docs: List[RetrievedDocument]) -> List[List[float]]:
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
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        """Filter down documents."""
        embedded_docs = self._get_embedded_docs(docs)
        included_idxs = self._filter_embeddings(embedded_docs)
        docs = [docs[i] for i in sorted(included_idxs)]
        return docs

    async def afilter(
        self, docs: List[RetrievedDocument], query: str
    ) -> List[RetrievedDocument]:
        raise NotImplementedError
